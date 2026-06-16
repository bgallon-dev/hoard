# M1 design — from-scratch OLMoE forward pass (all experts resident)

Goal: feed reference token IDs, greedy-decode, match llama.cpp text token-for-token.
Principle: **correctness over speed**. No KV cache, no streaming yet — those are M2/M3.

## Files
- `src/engine/model.h` / `model.cpp` — load GGUF → ggml tensors in ONE Vulkan buffer
  (all 195 tensors resident; ~4.2 GB fits in 8 GB). Holds hparams + tensor handles by name.
- `src/engine/forward.cpp` — `build_graph(model, token_ids[T])` → ggml_cgraph producing
  logits[vocab, T]. The whole OLMoE forward, op-for-op per notes/olmoe_spec.md.
- `src/engine/detok.cpp` — token id → text via GGUF `tokenizer.ggml.tokens` + GPT2 byte
  decode (Ġ→space, Ċ→newline, byte table). Detok only (no tokenizer needed — we feed IDs).
- `src/run_m1.cpp` — load, greedy loop, compare vs reference; per-prompt PASS/FAIL.

## Loading (all-resident)
1. `gguf_init_from_file(no_alloc=true, &ctx)` → tensor shapes/types, no data.
2. `ggml_backend_alloc_ctx_tensors(ctx, vk)` → one Vulkan buffer holds all tensors.
3. per tensor: `fread` quantized bytes at `data_offset+gguf_get_tensor_offset` →
   `ggml_backend_tensor_set(t, buf, 0, nbytes)`. (M2 will divert experts to host RAM.)

## Forward pass — no KV cache (recompute full prefix each step)
For T tokens build a fresh graph each step. ~45 tokens total → O(T^2) is trivially cheap,
and it removes a whole class of cache bugs while proving the math. (KV cache arrives in M2.)

helper: `rmsnorm(x,w) = ggml_mul(ggml_rms_norm(x, 1e-5), w)`

```
inp = ggml_get_rows(tok_embd, ids)              # [2048, T]
pos = 0..T-1 (i32);  mask = causal [T,T] f32 (0 on/below diag, -inf above)
for il in 0..15:
  cur = rmsnorm(inp, attn_norm[il])
  Q = mul_mat(wq[il],cur); K = mul_mat(wk[il],cur); V = mul_mat(wv[il],cur)   # [2048,T]
  Q = rmsnorm(Q, attn_q_norm[il]); K = rmsnorm(K, attn_k_norm[il])           # QK-norm over 2048
  Q=reshape(Q,128,16,T); K=reshape(K,128,16,T); V=reshape(V,128,16,T)
  Q = rope_ext(Q,pos,NULL,128, NEOX(2), 4096, 10000,1, 0,1,32,1)
  K = rope_ext(K,pos,NULL,128, NEOX(2), 4096, 10000,1, 0,1,32,1)
  # attention (explicit, transparent path):
  Qp=permute(Q,0,2,1,3); Kp=permute(K,0,2,1,3)                 # [128,T,16]
  S = mul_mat(Kp,Qp)                                            # [T_kv,T_q,16]
  S = soft_max_ext(S, mask, 1/sqrt(128), 0)
  Vp = cont(permute(V,1,2,0,3))                                 # [T,128,16]
  O = mul_mat(Vp,S)                                             # [128,T,16]
  O = reshape(cont(permute(O,0,2,1,3)), 2048, T)
  O = mul_mat(wo[il], O)
  ffn_inp = add(O, inp)                                         # residual 1
  # MoE FFN (mirrors llama.cpp build_moe_ffn, norm_w=false, gating=softmax):
  c = rmsnorm(ffn_inp, ffn_norm[il])
  rl = mul_mat(gate_inp[il], c)                                 # [64,T] router logits
  pr = soft_max(rl)                                             # softmax over ALL 64
  sel = top_k(pr, 8)                                            # [8,T] indices
  w  = get_rows(reshape(pr,1,64,T), sel)                        # [1,8,T]  (NO renorm)
  cx = reshape(c, 2048,1,T)
  up = mul_mat_id(up_exps[il], cx, sel)                         # [1024,8,T]
  ga = silu(mul_mat_id(gate_exps[il], cx, sel))                # [1024,8,T]
  ex = mul_mat_id(down_exps[il], mul(up,ga), sel)              # [2048,8,T]
  ex = mul(ex, w)                                               # weight each expert
  moe = sum_{e=0..7} view(ex,:,e,:)                             # [2048,T]
  inp = add(moe, ffn_inp)                                       # residual 2
cur = rmsnorm(inp, output_norm)
logits = mul_mat(output, cur)                                   # [50304, T]
next = argmax(logits[:, T-1])                                   # greedy on CPU
```

Attention choice: explicit `soft_max_ext` path (transparent, lets us inspect scores,
matches classic llama.cpp decode). `ggml_flash_attn_ext` is a 1-call alternative (fewer
ops, Vulkan-supported) — fallback if the explicit path misbehaves.

## Comparison harness
Capture reference greedy text for ~4 diverse prompts via `llama-completion --verbose-prompt`
(factual / code / list / story). Store {prompt token IDs, reference text}. Engine feeds the
same IDs, generates same N, detokenizes, compares **exact text**. PASS = all match.

## Debug strategy on divergence (the curriculum)
- Garbage output → structural bug (wrong transpose/tensor/shape). Coherent-but-wrong → subtle
  (eps, rope mode, renorm, accumulation order).
- First generated token differs → forward-pass bug; bisect via per-layer hidden-state norms
  (print ours; sanity-check magnitudes) and unit-check single ops on tiny inputs.
- First token matches, later diverge → numeric drift; suspect accumulation/precision.
- Prime suspects, in order: QK-norm placement, router no-renorm, softmax-before-topk, NEOX
  rope, eps. (the olmoe_spec.md gotcha list.)
```
