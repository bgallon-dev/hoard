# Streaming MoE engine — architecture (M1 → M3)

## Weight inventory (OLMoE-1B-7B Q4_K_M, ~4.0 GB on disk)
Two classes, split by whether they stream:

**Resident (always in VRAM, small):**
- token_embd (2048×50304), output (50304×2048), output_norm
- per layer ×16: attn_norm, wq/wk/wv/wo (2048×2048 each), attn_q_norm/attn_k_norm
  (2048), ffn_norm, **ffn_gate_inp** (router, 2048×64)
- Estimate ≈ 0.4–0.5 B params ⇒ **~300–400 MB** quantized. Always resident.

**Streamed (the bulk):**
- per layer ×16, per expert ×64: ffn_gate_exps, ffn_up_exps, ffn_down_exps
- gate 2048×1024 + up 2048×1024 + down 1024×2048 = 6.29 M params/expert
- Q4_K ≈ 3.5 MB/expert × 1024 experts ≈ **~3.6 GB**. This is what we page.

Per token: 16 layers × 8 used = **128 expert loads** if naive (re-copy every token).
This is the "144 redundant copies/token" pathology, scaled to OLMoE.

## Execution model per milestone

**M1 (coherence, all-resident):** load everything into VRAM (4 GB fits in 8 GB).
Build the full OLMoE graph per `notes/olmoe_spec.md`. Use ggml's fused
`ggml_mul_mat_id` for the MoE FFN — identical op to llama.cpp's `build_moe_ffn`,
so numerics match the reference closely. Goal: greedy tokens == reference.
Feed **token IDs directly** (dumped from llama.cpp) so the tokenizer is out of
scope for M1.

**M2 (streaming):** expert weights live in host RAM (mmap of the GGUF, the
expert tensor regions). Non-expert weights stay VRAM-resident. The MoE FFN
switches from the fused all-64 `mul_mat_id` to a **per-expert streamed loop**
over the 8 selected experts:
```
for e in top8(layer):
    slot = ensure_resident(layer, e)      # copy H2D if not already in a slot
    g = mul_mat(gate_exps[slot], x); u = mul_mat(up_exps[slot], x)
    h = silu(g) * u
    y = mul_mat(down_exps[slot], h)
    acc += w_e * y                         # replicate llama.cpp accum order
```
VRAM budget = resident (~0.4 GB) + N expert slots × 3.5 MB. Peak VRAM is the
M2 proof number. Must keep M1 token output (replicate accumulation order so the
weighted sum doesn't drift the greedy argmax).

**M3 (cache):** `ensure_resident` is the cache.
- *Naive baseline:* every token, copy all 8 experts of each layer into slots,
  evict after. Copies/token = 128. Measure tok/s.
- *Residency cache:* persistent slot pool keyed by (layer, expert). On a token's
  top-8, slots already holding those experts are **hits** (no copy). Misses copy
  in, evicting by policy (LRU, or LFU/hot-set per MoE-ERAS skew). Measure tok/s
  AND hit rate. Speedup ≈ explained by (1 − miss_rate × copy_cost_fraction).
- Two reported numbers: **speedup ratio** and **hit rate**.

## ggml integration
- GGUF read via ggml `gguf.h` (`gguf_init_from_file`) → metadata + tensor descriptors
  + data offsets. Confirm hparams vs `olmoe_spec.md`.
- Backends: `ggml-cpu` (host tensors) + `ggml-vulkan` (device). Use a
  `ggml_backend_sched` or manual: allocate device buffers, `ggml_backend_tensor_set`
  to copy host→device (this IS the "stream in" primitive; instrument it to count
  bytes + copies — the core M3 metric).
- Slot pool = pre-allocated device buffers sized to one expert's 3 tensors;
  paging = `ggml_backend_tensor_set` into a free/evicted slot, then point the
  per-expert matmul at that slot's tensor.
- KV cache: own it explicitly (decode is one token at a time; MHA 16 heads ×128).

## Why this model is ideal for the demo
top-8 of 64 (high traffic, 128 loads/token) + known hot/cold skew (MoE-ERAS) ⇒
cache hit rate should land in the 70–85% band the goal predicts, and the naive
baseline is genuinely wasteful, so the speedup is real and explainable.

## Open items to confirm at runtime
- rms eps (exp 1e-5), expert_weights_scale (exp 1.0), rope NEOX — from llama-cli log
- exact `build_moe_ffn` accumulation/normalization order (read ggml source) so the
  streamed loop matches the fused path bit-for-greedy.
