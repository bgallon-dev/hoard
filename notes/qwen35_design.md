# Milestone 5: Qwen3.5/3.6 hybrid linear-attention support (qwen35moe)

The frontier moved: Qwen3 (`qwen3moe`, standard transformer) -> Qwen3.5+ (`qwen35moe`/`qwen3next`,
HYBRID linear attention). Goal: run **Qwen3.5-35B-A3B** (the A3B sweet spot) on the streaming engine.

## Architecture (Qwen3.5-35B-A3B, model_type qwen3_5_moe / GGUF arch `qwen35moe`)
- 40 layers, **hybrid**: `is_recr(il) = (il+1) % full_attention_interval != 0`, interval=4
  -> 30 **linear-attention (Gated Delta Net)** layers + 10 **full-attention** layers (il = 3,7,...,39).
- MoE: **256 experts, top-8**, moe_intermediate **512** (smaller than Qwen3's 768) + a **shared expert**
  (intermediate 512, sigmoid-gated, always-on) — NEW vs qwen3moe. softmax routing, norm_topk_prob=true.
- hidden 2048, vocab **248320**, rope_theta (multi-section / IMRoPE).
- Full attention: GQA **16 query / 2 KV heads**, **head_dim 256**, fused Q+gate proj (wq -> [head*2]),
  Q/K RMS-norm per head, multi-section RoPE (ggml_rope_multi), sigmoid gate on attn output, out proj.
- Linear attention (GDN): conv_kernel 4, key_head_dim 128 (16 heads), value_head_dim 128 (32 heads).
  qkvz proj -> ssm_conv (causal conv1d, silu) -> l2_norm(q,k) -> delta-net recurrence (state) ->
  gated RMS-norm(out, z) -> out proj. Decay: alpha=softplus(ssm_alpha.x + ssm_dt)*ssm_a; beta=sigmoid(...).
- MTP/NextN: 1 extra block (n_layer_nextn) for speculative draft — NOT executed in main forward. SKIP it.

## Reference (in reference/llama.cpp-src/src/models/)
- `qwen35moe.cpp` — top-level graph (build_layer_attn / build_layer_attn_linear / build_layer_ffn). PORT THIS.
- `delta-net-base.cpp` — `build_delta_net` (unfused chunked delta-rule scan) + `build_recurrent_attn`
  (state read/write) + `ggml_gated_delta_net` (fused; may not be on Vulkan -> use unfused path).

## Feasibility (Vulkan op gate — checked)
ssm_conv 14 / ssm_scan 6 / l2_norm 8 / rwkv 10 / cumsum 8 refs in ggml-vulkan.cpp -> PRESENT.
GAPS: SOFTPLUS (0) — but acts on tiny [n_v_heads, n_tok] tensor -> compute on host. ROPE_MULTI (0) —
verify whether it's GGML_OP_ROPE + sections mode (Vulkan has ROPE) or needs a fallback.

## Build phases (incremental, each validated vs llama.cpp reference like every prior milestone)
- P0 hparams + scaffold: parse SSM dims (ssm_d_conv/inner/state/dt_rank/n_group), is_recr[] per layer,
  shared-expert dims, head_dim 256, rope_sections, vocab; per-layer dispatch linear-vs-full.
- P1 full-attention layers (10): fused QG -> Q/K-norm -> multi-section RoPE -> GQA 16/2 -> sigmoid-gate -> o.
- P2 recurrent state infra: per-linear-layer conv_state + ssm_state buffers, persistent across tokens.
- P3 GDN linear layers (30): qkvz -> ssm_conv -> l2_norm -> delta-net recurrence -> gated norm -> o. (HARD)
- P4 MoE + shared experts: reuse streaming MoE (256/8) + gated shared-expert path.
- P5 integrate + validate token-for-token vs llama.cpp; then wire the 3-tier expert streaming.

## Per-expert size (streamable): gate/up [2048,512] + down [512,2048] ~1.7 MB/expert; 256x40 ~17 GB experts.
Same cliff regime as Qwen3-30B (A3B, ~3B active) -> expect ~similar tok/s once streaming wired.

## Qwen3-Next-80B-A3B port (qwen3next arch) — TODO when model downloads (~45 GB)
Reuses the qwen35moe GDN engine. Differences to handle (gated on arch=="qwen3next"):
- qkvz: new GGUF uses attn_qkv + attn_gate (SAME as qwen35moe). legacy uses fused ssm_in (split). Confirm via gguf_dump.
- ssm_ba (LLM_TENSOR_SSM_BETA_ALPHA, fused) -> split into beta[0:num_v_heads] + alpha[num_v_heads:] (vs qwen35moe separate ssm_beta/ssm_alpha).
- full-attn RoPE: ggml_rope_ext (NEOX) NOT rope_multi/IMRoPE -> simpler, ip=[p] (1 pos), like qwen3moe.
- hparams (generic read): 48 layers, 512 experts / 10 active, hidden 2048, head_dim 256, GQA 16/2.
- F<B holds (10*48*~1.7MB = ~0.8 GB/token active < VRAM budget) -> persistent slot cache works, NO slot redesign needed.
Same: ssm_conv, ssm_a/dt/norm/out, gated norm, MoE 512/10 (stream), gated shared expert, recurrent state infra.

## Qwen3-Next-80B-A3B PORT -- DONE, validated 16/16 (see notes/STATUS.md "M6")
Confirmed via gguf_dump: NEW format (attn_qkv+attn_gate, fused ssm_ba [2048,64], rope.dimension_count=64 NEOX,
NO rope.dimension_sections, NO full_attention_interval key -> default 4). Implemented in run_qwen35.cpp gated on
is_qwen3next (arch=="qwen3next"): ssm_ba de-interleave, ggml_rope_ext NEOX (1 pos), plain chat turn (Instruct).
*** THE NON-OBVIOUS BUG ***: qwen3next ALWAYS repeat-interleaves q/k Hk(16)->Hv(32) before ggml_gated_delta_net;
qwen35moe does NOT (fused path uses the op's internal broadcast). So gate the repeat on arch==qwen3next ONLY --
NOT on Hv!=Hk (that repeats for qwen35moe too and breaks it 252->15/256). Without it: 80B logits ~0.5x compressed
(GDN 0.844x/layer compounding), 10/16. With it: 16/16. Also: BF16 ffn_gate_inp_shexp -> upconvert to f32 (RX6600
bf16:0; turned out bit-identical but kept). Found by per-layer cosine/norm diff vs CPU ref_dump (src/ref_dump.cpp).
