# The tool-shaped edge — cold-storage large-model reviewer (saved plan, not built)

This is the **utility path**, distinct from the M1–M4 understanding build. Goal: turn the
engine from "a thing I understand" into "a thing I use" — run a 60B+ MoE (the batch code
reviewer actually wanted) on the 8 GB RX 6600 / 16 GB RAM box by adding a storage tier below
RAM. The M1–M4 understanding transfers; the surface area is new and open-ended.

## Why a new tier is the crux
M2/M3 stream experts RAM→VRAM. A 60B+ MoE's experts (tens of GB) exceed 16 GB RAM, so the
backing store becomes **SSD→RAM→VRAM** (FluxMoE's "expert storage hierarchy"). The new
bottleneck is SSD bandwidth (~GB/s, ~10x slower than the PCIe RAM→VRAM path), so the whole
game shifts to **hiding disk latency**, the way M3's cache hid PCIe latency.

## What carries over vs what's new
Carries over (already built + validated):
- per-layer streaming, the residency cache (LRU + per-layer slot pool), the KV-cache decode,
  the host-orchestrated router→stream→FFN split, the metrics harness.
Genuinely new:
1. **3-tier hierarchy**: experts on SSD (mmap'd GGUF) → page into a RAM cache → stream into
   VRAM slots. Two cache levels (RAM-resident set, VRAM-resident set), each LRU/hot-set.
2. **Prefetch to hide SSD latency**: SSD is too slow to fetch on-demand on the critical path.
   Need predictive prefetch — the SP-MoE/MoE-Infinity idea: use early-layer gating (or a draft)
   to predict later-layer experts and start the SSD read ahead of need. This is where M4's
   residency-aware draft becomes a *prefetch oracle*, not just a speculation draft.
3. **A different model** (e.g. Qwen3-30B-A3B / a 60B+ MoE): re-derive its exact forward-pass
   contract (GQA, different norm/rope/expert-count, shared experts) against a llama.cpp
   reference, same dense-oracle method as M1a. Likely needs GQA + shared-expert support in the
   engine (OLMoE had neither).
4. **Different correctness bar**: utility, not token-exact. Greedy-match a reference for a
   sanity window, then optimize for usable tok/s on real review prompts.

## Rough milestones (when/if pursued)
- T1: load a 60B+ MoE with experts mmap'd from SSD; non-expert resident in VRAM; correct output
  on a short prompt (vs llama.cpp). Measure: does it run at all, and at what tok/s.
- T2: RAM cache layer (hot experts in RAM, cold on SSD); measure SSD-read fraction + tok/s.
- T3: predictive prefetch (early-gate or draft) to hide SSD latency; measure prefetch hit rate
  + tok/s gain. (This is where M4's draft is reused as a prefetcher.)
- T4: end-to-end batch code review on a real repo; measure wall-clock per file + quality.

## Hardware reality check
16 GB RAM is the binding constraint: a 60B MoE in Q4 is ~30–35 GB → most experts live on SSD,
not RAM. So the RAM cache is small relative to the model and prefetch quality dominates. NVMe
~3–7 GB/s read; with ~64 experts/layer touched per token, SSD bandwidth, not PCIe, sets the
ceiling — exactly why prefetch (not just caching) is the central problem at this scale.

## Motivation note (so the reach is deliberate)
This door *opens* the utility path; it does not *finish* the research seam (that's the M4
integrated loop — see notes/STATUS.md M4). Different kind of value: tool, not understanding.
