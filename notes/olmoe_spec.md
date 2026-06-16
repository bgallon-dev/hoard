# OLMoE-1B-7B-0924-Instruct — exact forward-pass contract (M1 reference)

Source of truth: HF `config.json` (constants) + llama.cpp `src/models/olmoe.cpp`
(graph) + `src/llama-model.cpp` (rope type). Anything marked **[runtime-confirm]**
is verified against the `llama-cli` load log on the actual GGUF.

## Constants (HF config.json)
| field | value |
|---|---|
| hidden_size (n_embd) | 2048 |
| n_layer | 16 |
| n_head / n_head_kv | 16 / 16  → **MHA, not GQA** |
| head_dim (n_embd_head, n_rot) | 128 |
| intermediate_size (per-expert n_ff) | 1024 |
| n_expert / n_expert_used | 64 / 8 (top-8) |
| vocab_size | 50304 |
| rope_theta (freq_base) | 10000.0 |
| rope_scaling | null (freq_scale=1) |
| rms_norm_eps | **1e-5** (GGUF: 9.99999975e-06) ✓confirmed |
| hidden_act | silu (SwiGLU) |
| attention_bias | false (no bias on q/k/v/o) |
| clip_qkv | null |
| norm_topk_prob | **false** → top-8 weights NOT renormalized |
| tie_word_embeddings | false → separate lm_head |
| expert_weights_scale | **absent in GGUF → no scaling** ✓confirmed |
| tensor precisions | gate/up exps=q4_K, down exps=q6_K, attn_v & lm_head=q6_K, norms & router=f32 ✓ |
| byte split | resident(non-expert)=**311 MB**, experts=**3901 MB** (~4.08 MB/expert) ✓ |
| rope_type | **NEOX** (LLM_ARCH_OLMOE in llama_model::rope_type) |

## Per-layer op sequence (exact, from olmoe.cpp `build_arch_graph`)
```
inpSA = inpL
cur   = RMSNorm(inpL, attn_norm)                 # pre-attn, eps=f_norm_rms_eps
# --- attention ---
Qcur  = wq @ cur                                  # no bias  (2048->2048)
Kcur  = wk @ cur                                  # no bias  (2048->2048)
Vcur  = wv @ cur                                  # no bias  (2048->2048)
Qcur  = RMSNorm(Qcur, attn_q_norm)   # QK-NORM over FULL 2048 dim, weight {2048}
Kcur  = RMSNorm(Kcur, attn_k_norm)   # BEFORE reshape, BEFORE RoPE
Qcur  = reshape(Qcur, [128,16,T]); Kcur=reshape([128,16,T]); Vcur=reshape([128,16,T])
Qcur  = rope_neox(Qcur, pos, n_rot=128, base=10000, scale=1)
Kcur  = rope_neox(Kcur, pos, n_rot=128, base=10000, scale=1)
cur   = attn(Q,K,V, scale=1/sqrt(128), causal) ; cur = wo @ cur   # no bias
ffn_inp = cur + inpSA                              # residual #1
# --- MoE FFN ---
cur   = RMSNorm(ffn_inp, ffn_norm)
logits_e = ffn_gate_inp @ cur                      # router: 2048->64, no bias
probs    = softmax(logits_e)        over ALL 64    # gating=SOFTMAX
idx      = topk(probs, 8)                          # select top-8
w_i      = probs[idx]                               # NORM_W=false: NOT renormalized
for each selected expert e in idx:
    h_e = silu(ffn_gate_exps[e] @ cur) * (ffn_up_exps[e] @ cur)   # SwiGLU, n_ff=1024
    y_e = ffn_down_exps[e] @ h_e                                   # 1024->2048
cur   = sum_e( w_i[e] * y_e )       # (* expert_weights_scale [runtime-confirm, exp 1.0])
cur   = cur + ffn_inp                              # residual #2
inpL  = cur
# --- final ---
cur    = RMSNorm(inpL, output_norm)
logits = output @ cur                              # lm_head, separate weight
```

## Token-exact gotcha ranking (most likely silent divergences)
1. **QK-norm**: RMSNorm on Q and K over the full 2048 projection, BEFORE reshape & BEFORE
   RoPE. Omitting it or applying per-head/after-RoPE silently diverges. weights:
   `attn_q_norm`/`attn_k_norm`, shape {2048}.
2. **Router not renormalized** (`norm_topk_prob=false`): w_i are softmax-over-64 values at
   the top-8 positions; they do NOT sum to 1. Do not re-softmax over the 8.
3. **Softmax BEFORE top-k** (gating=SOFTMAX): softmax over all 64 logits, THEN pick top-8.
   Not top-8-then-softmax.
4. **RoPE = NEOX** style (pairs offset by n_rot/2), not the interleaved "normal" variant.
5. **RMS eps** value (1e-5 expected) — wrong eps drifts slowly, diverges after many tokens.
6. **Residual points**: add inpSA after attention (pre-FFN-norm), add ffn_inp after MoE.
7. **No biases** anywhere; **no shared experts** (pure top-8 of 64).
8. **SwiGLU order**: `silu(gate(x)) * up(x)` then `down(...)`. silu on the gate branch only.

## GGUF metadata keys to read back (confirm against this spec)
- `olmoe.block_count` (16), `olmoe.embedding_length` (2048)
- `olmoe.attention.head_count` (16), `olmoe.attention.head_count_kv` (16)
- `olmoe.feed_forward_length` (1024), `olmoe.expert_count` (64), `olmoe.expert_used_count` (8)
- `olmoe.attention.layer_norm_rms_epsilon`, `olmoe.rope.freq_base` (10000)
- `olmoe.expert_weights_scale` (likely absent → default)
- tensor names per layer: `blk.N.attn_norm`, `blk.N.attn_q`/`_k`/`_v`/`_output`,
  `blk.N.attn_q_norm`, `blk.N.attn_k_norm`, `blk.N.ffn_norm`, `blk.N.ffn_gate_inp`,
  `blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, `blk.N.ffn_down_exps`;
  global `token_embd`, `output_norm`, `output`.

## Per-token expert traffic (why this model is good for the cache demo)
16 layers × 8 experts = **128 expert FFN invocations per token**. Each expert =
3 matrices (gate 2048×1024, up 2048×1024, down 1024×2048). The naive streamer re-copies
all 128 per token; the residency cache exploits hot-expert skew (MoE-ERAS finding) to
keep the working set resident. This is the M3 mechanism in miniature.
