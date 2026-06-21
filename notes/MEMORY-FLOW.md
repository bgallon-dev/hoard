# Memory-Flow Analysis — Streaming MoE Inference (Qwen3-Next-80B-A3B on 8 GB VRAM / 16 GB RAM / NVMe)

A layer-by-layer map of *where every byte moves* during decode. All numbers are measured on this
machine (RX 6600, 8 GB; 16 GB DDR; ~3.5 GB/s NVMe) via the engine and the **validated** offline cache
simulator (98.8 % per-access tier agreement vs the real engine). Token-rate: ~5.8 tok/s at the
operating point (K=48 slots/layer, ram_cap=2048 experts).

## 0. The model, in one paragraph
48 transformer layers, `n_embd`=2048, MoE FFN every layer (512 experts, **top-10 routed + 1 shared**,
expert FFN dim `n_ff`=512, **~1.87 MB/expert** of Q4_K data — gate/up `[2048,512]` + down `[512,2048]`). Attention is **hybrid**: 36 layers use a Gated-Delta-Net (GDN) *linear/recurrent*
attention, 12 layers (every 4th) use *full* softmax attention with a KV cache. Active params/token ≈
3 B (hence "A3B"); total ≈ 80 B, of which **~50 GB of expert weights live on NVMe** and stream in on
demand. The whole optimization problem is moving those expert bytes through the memory hierarchy fast
enough.

## 1. The four memory channels (per decode token)

| channel | direction | traffic / token | bandwidth | share of decode |
|---|---|---|---|---|
| **A. Expert weights** | NVMe → RAM(LRU) → VRAM slots | **262 MB read** + **283 MB staged H2D** | NVMe 3.1 GB/s; PCIe ~16 GB/s | fetch **64 %** (disk 48 % + H2D 16 %) |
| **B. Attention state** | in-place per layer, carried token→token | GDN: 0 grow (79 MB fixed); KV: +48 KB | on-VRAM | part of graphA 14 % |
| **C. Residual stream** | hidden state, layer L → L+1 | **8 KB** (2048 f32) | on-VRAM | negligible |
| **D. Compute scratch** | transient per graph | galloc buffer (~MBs) | on-VRAM | graphA+graphB |

The headline: **channel A dwarfs everything.** The residual stream that "flows between layers" in the
textbook sense is 8 KB/token; the *expert* bytes that flow between the storage tiers are **262 MB/token**
— a 33,000× ratio. In a streaming MoE engine, "memory flow between layers" *is* expert streaming.

A second non-obvious point: the **PCIe/H2D channel carries more traffic than NVMe** (283 vs 262 MB),
because every RAM-tier hit *also* gets staged into a VRAM slot. NVMe is the slower pipe (3.1 vs
~16 GB/s) so it dominates wall-time, but the busier *bus* is PCIe.

```
   per layer, per token (strict serial order):

   ┌── graphA (attn + router) ──┐      ┌── fetch (the bottleneck) ──┐   ┌─ graphB ─┐
   │ residual_in ─► GDN/KV attn │      │ classify 10 experts:        │   │ FFN over │
   │            ─► post-norm     │ ──►  │   VRAM hit 68% (resident)   │──►│ 10 slots │──► residual_out
   │            ─► router top-10 │      │   RAM  hit  2% (PCIe stage) │   │ +shared  │      │
   └────────────────────────────┘      │   NVMe cold 30% (disk→RAM→  │   │ +Σ w·e   │      ▼
        knows experts only HERE         │              →VRAM)         │   └──────────┘   next layer
                                        └─────────────────────────────┘
```

## 2. Channel A — expert weights (the dominant flow), MEASURED

### 2.1 The three-tier hierarchy
- **VRAM slot pool** — `K=48` slots **per layer** (48×48 = 2304 slots), LRU per layer. The landing
  zone for streamed experts. **4388 MB** — the single largest VRAM consumer (see §5).
- **RAM tier** — a global LRU of `ram_cap=2048` expert byte-buffers (~3.8 GB) holding recently-read
  experts (the `RT` structure; aligned `VirtualAlloc` buffers that `NO_BUFFERING` reads land in directly).
- **NVMe** — the remaining ~50 GB of expert weights, read device-direct.

Each expert = three sub-tensors read separately (they live in different GGUF tensors): gate / up / down,
**~1.87 MB of data** total, landing in 4 KB-aligned host buffers of 584 / 584 / 848 KB (the alignment
slack is why the buffers sum to ~2.0 MB while the streamed data is ~1.87 MB). Per token the router
selects 10/layer × 48 = **480 expert-accesses**, of which ~140 miss to NVMe.

### 2.2 The per-layer fetch pipeline (strict, serial)
`graphA(L)` runs attention + router → **only now are layer L's 10 experts known** → Phase A classifies
each (VRAM/RAM/NVMe) and builds read-jobs for the cold ones → Phase B issues them through the
**persistent 8-worker device-direct pool** (one barrier; QD≈8) → Phase C stages each into a VRAM slot
(single-threaded Vulkan H2D) → `graphB(L)` computes the FFN. Then `graphA(L+1)`. **No expert can be
prefetched across this barrier** because the next layer's experts are a data-dependent function of this
layer's output — the architectural reason predictive prefetch is dead (measured +0.0 %).

### 2.3 Where the bytes move, layer by layer (general prompt, per token)
The flow is **highly non-uniform** — a ~5× spread and a periodic sawtooth (minima at L12/24/36, a
~12-layer cycle), with **early layers routing most diffusely** (more cold reads, larger per-layer
working sets) and **full-attention layers ~21 % hotter** than linear ones:

| layer band | cold reads/tok | NVMe-in/tok | note |
|---|---|---|---|
| L0 (hottest) | 6.5 | 12.1 MB | broadest routing (workset 434/512) |
| L12/24/36 (minima) | ~1.3 | ~2.4 MB | most concentrated routing |
| early L0–L3 | 19.2 | 36 MB | early > late |
| late L44–L47 | 13.3 | 25 MB | |
| **full-attn (12 layers)** | **3.35 / layer** | | vs linear **2.77 / layer** |
| **TOTAL / token** | **140** | **262 MB** | + **283 MB** staged H2D |

(Coding prompt is heavier: 160 cold → 299 MB NVMe + 321 MB H2D — bigger working set, more NVMe-bound.)

Implication: cache budget is worth most at the *early/diffuse* layers; the concentrated mid-stack layers
(L12/24/36) are nearly free. This per-layer structure is the actionable part of the map.

## 3. Channel B — attention state (the hybrid), per layer type

This is the memory that is *carried between tokens within a layer*, and it's where the hybrid earns its
keep. Two regimes:

- **36 GDN linear layers — FIXED recurrent state, 0 growth with context.** Per layer: an SSM state
  tensor `[Sv,Sv,Hv]` = [128,128,32] f32 = **2.0 MB**, plus a conv state `[d_conv−1, conv_dim]` =
  [3, …]. Total **79 MB** for all 36 layers, **constant** regardless of sequence length — the recurrence
  folds history into a fixed-size state, updated in place each token.
- **12 full-attention layers — GROWING KV cache.** Per layer per token: K and V each a slab
  `[head_dim=256, n_head_kv=2]` F32 = 2048 B, so K+V = **4 KB/token/layer**. Across 12 layers =
  **48 KB/token**. At ctx=41 → 2.0 MB (measured ✓ — this is the check that caught the head_dim error);
  at 32 K context → **~1.6 GB**. (Written to the `p`-th slab via `ggml_cpy`, then the full `[0..p]`
  history is gathered each step — `run_qwen35.cpp:478-485`.)

**Why the hybrid reaches ~4× the context at the same VRAM:** a pure 48-layer transformer would grow KV
at 192 KB/token (48 × 4 KB); this hybrid grows at 48 KB/token (12 × 4 KB) — **one quarter the rate** —
while the 36 linear layers add nothing. So the KV frontier (where KV + slots + model cross 8 GB) sits
~4× further out. The recurrent state replaces unbounded KV with a bounded 79 MB.

## 4. Channel C — the residual stream + the serialization

Between layers, only the **hidden state flows**: `n_embd`=2048 floats = **8 KB/token**, layer L's
`residual_out` → layer L+1's `residual_in`. It actually lives in a host `std::vector<float> x`
(`run_qwen35.cpp:416`) and **round-trips device→host→device every layer** — graphA reads it in, the
final weighted sum `x[d]=Σ_j w_j·e_j[d] + shared[d] + ffn_inp[d]` is computed on the **host** (`:660-662`),
then graphB/graphA of the next layer copy it back to VRAM. So the cross-layer channel is 8 KB bouncing
over PCIe twice per layer — trivial bandwidth, but it's the serialization point.

That 8 KB is the entire inter-layer data dependency, and it's what forces the strict
`graphA→fetch→graphB→graphA` chain: layer L+1's router needs L's output, so the expert identities — and
therefore the 262 MB of reads — cannot be known or prefetched ahead of the barrier. **The 8 KB residual
is the tiny causal thread that gates the 262 MB flood.** (The KV slab and GDN/conv state never cross the
boundary — they're written *in place* inside each layer, persistent in VRAM keyed by layer.)

## 5. The footprint budget (static VRAM, K=48, short context) — MEASURED

```
  static VRAM = 6161 MB
    ├─ expert slot pool (K=48 × 48 layers × ~1.9 MB) ... 4388 MB   <-- LARGEST: the streaming landing zone
    ├─ non-expert model weights (699 tensors)        ... 1692 MB   (attn/GDN/norms/router/shared/embd/output)
    ├─ GDN recurrent state (36 layers, fixed)        ...   79 MB
    ├─ KV cache (12 layers, grows ~48 KB/tok)        ...    2 MB   (at ctx=41; ~1.6 GB at 32K)
    └─ compute scratch (galloc)                      ...  ~MBs
  host RAM = RT tier (2048 experts × ~1.9 MB ≈ 3.8 GB) + load scratch + runtime
  NVMe     = ~50 GB expert weights (streamed)
```

The slot pool (4.4 GB) — **2.6× the resident model weights** — is the dominant VRAM consumer, and it
scales linearly with K (each +1 K = +48 experts ≈ +90 MB). On an 8 GB card the static 6.16 GB plus a
growing KV crosses the limit and **WDDM over-commits into shared system RAM over PCIe** (measured:
graceful, ~16 GB/s, faster than NVMe) — a soft frontier, not a cliff. This is why K and context trade
against VRAM, and why the *slot pool*, not the model, sets the VRAM frontier.

## 6. Where the flow stalls (bottleneck) — MEASURED

Per-token decode time: **graphA 14 % | fetch 64 % [disk 48 % + H2D 16 %] | graphB 19 % | head 1 %.**
Fetch dominates, and within it the NVMe read is the single largest cost. Two structural ceilings:
- **The per-layer barrier.** 48 small (~9-read) bursts/token with compute gaps; the drive can't sustain
  its peak across them. Measured: in isolation the read pool hits 3.5 GB/s; real decode pulls 3.11 — the
  gap is the barrier, the autoregressive floor.
- **Amdahl.** ~30 % of decode is fixed (attn/router/FFN/head). Even driving cold reads to *zero* caps
  the speedup at ~2.17×; realistic cuts give single-digit-percent gains.

This session's **drive unlock** (persistent I/O pool + unbuffered weight load to keep the OS page cache
clean → `NO_BUFFERING` reads stay on their fast path) took NVMe 1.98 → **3.11 GB/s** and decode
4.28 → **5.78 tok/s (+35 %)**, lossless. The drive is now effectively maxed for this access pattern.

## 7. What the flow's structure implies for optimization (measured verdicts)

- **The working set never saturates** (42–50 % of the model touched by 256 tokens, still climbing) —
  diffuse routing means no fixed cache captures it; the cold tail is irreducibly broad.
- **Prefetch (temporal): dead, +0.0 %** — the predictable experts are already cached; the cold reads are
  the unpredictable novel tail (channel A's diffuseness + the §2.2 barrier).
- **Clustering: dead** — per-layer selected sets are fresh each token (Jaccard 0.30), and the drive is
  read-size-insensitive (3.5 GB/s at 512 KB == 8 MB), so contiguous layout buys ~0.
- **Session/domain tiering: the lone lossless survivor, ~+3 % tok/s.** Usage is separable into ~5
  clusters (code/math/creative+poetic/history/science/legal); a domain-detected warm pin cuts cold reads
  ~+5.6 % (held-out), but a *global* pin **hurts −5 %** (wrong experts starve LRU) — domain-adaptivity
  is required.
- **More RAM tier: dead on this box.** The simulator's miss-ratio surface says ram 2048→4096 cuts cold
  reads −33 %, but on 16 GB it triggers memory pressure that **collapses NVMe bandwidth 3.1→1.44 GB/s**
  (net zero tok/s), and ram=6144 OOMs. ram_cap≈2048 is the practical ceiling; the dominant lever needs
  *physical* RAM the box doesn't have.

## 8. Bottom line
The memory flow is a 262 MB/token expert flood, gated by an 8 KB residual thread, landing in a 4.4 GB
VRAM slot pool, bottlenecked on a now-maxed 3.1 GB/s NVMe pipe through a per-layer barrier — with a
hybrid attention design that keeps the *other* growing channel (KV) at one-quarter rate so context, not
attention memory, is rarely the binding constraint. The diffuse, non-saturating expert tail is the wall;
every lossless software lever bends on it, leaving only modest domain-aware tiering and (unavailable
here) more physical RAM.

---
*Instruments (all in `bench/`): `profiler.py` (+validated simulator), `sysmap.py` (miss-ratio surface),
`per_layer_flow.py` (this §2.3 table), `corpus_map.py`/`cluster_pin.py` (domain separability),
`domain_gate.py`. Traces: `results/gatedump_80b.txt` (general), `results/gatedump_coding.txt`,
`results/corpus_profile.txt` (10 domains). Engine instrumentation: `GATEDUMP`, `PROFILE`, `DRYRUN`,
`IOSELFTEST`.*
