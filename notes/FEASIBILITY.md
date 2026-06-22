# A predictive feasibility law for streaming-MoE-on-constrained-VRAM

**Claim.** The measured constants in this repo collapse into one closed-form predictor that, given a
model's *a-priori* physical parameters plus the host's storage/PCIe bandwidths, outputs decode tok/s
**and** a regime label. It survives a genuine out-of-sample test: with the expert byte-size computed
from the GGUF, the cache-miss curve **predicted** from a routing model fit on a *different* model, and
the compute floor transferred — i.e. **nothing measured on the target except for scoring** — it
predicts an 80B's throughput curve to **~8 %** (routing-isolated; 3 % composite with disclosed bias
cancellation), and correctly labels a never-run 235B as infeasible.

Reproduce (reads committed CSVs/traces; writes figures):
`py bench/feasibility.py` · `py bench/spill.py` · `py bench/spill_graphb.py` · `py bench/routing_model.py [trace]` · `py bench/close_loop.py`

---

## 1. The law

Per-token decode latency is a fixed compute floor plus two bandwidth-bound streaming terms:

```
t_tok =  t_fixed  +  A·m_nvme / B_nvme  +  A·(1 − h_vram) / B_pcie
tok/s =  1 / t_tok

A         = n_used · L · b_e        per-token "all-accesses" expert byte volume   (MB)
m_nvme(K) = NVMe-miss fraction      experts caught by neither VRAM nor RAM tier
h_vram(K) = VRAM-hit fraction       experts already resident → no H2D restage
B_nvme    = sustained drive bandwidth (GB/s);  B_pcie ≈ 16 GB/s (PCIe4 x8 H2D)
t_fixed   = compute + graph-launch + router + head — the Amdahl floor
```

### What is definitional vs what is falsifiable (read this before trusting the fit)
A skeptic should know which parts are physics and which are bookkeeping:

- **Bookkeeping (true by construction):** `A = n_used·L·b_e` is just the per-token byte volume. The
  bandwidth terms `A·m/B` are `time = bytes / rate`. These hold or the byte-counting is wrong; they are
  not predictions.
- **The falsifiable core is three assumptions:** (a) `t_fixed` is **constant across K and context** —
  compute/launch/router cost doesn't vary with cache state; (b) `B_nvme`, `B_pcie` are **stable across
  operating points**; (c) **additive separability** — the floor and the two bandwidth terms sum with no
  cross-terms (no contention between NVMe reads, H2D stages, and compute).
- **(a) is tested,** not assumed: the per-K spread of the fitted `t_fixed` is **±12 %** (35B [51,65] ms,
  80B [77,93] ms) — non-trivial but bounded, and that spread *is* the evidence for the assumption.
- **(b) is false under memory pressure** — and we don't hide it: the RAM-collapse (§3c) is exactly a
  `B_nvme` regime transition. The law is **piecewise-constant-BW**; the collapse is a labeled boundary,
  not a refutation.
- **(c) is justified by the architecture, not asserted.** The per-layer barrier — layer L+1's router
  needs layer L's output — forces `graphA → fetch → graphB` strictly serial and **non-overlapping in
  wall-clock** ([`run_qwen35.cpp` fetch phases](../src/run_qwen35.cpp), `notes/MEMORY-FLOW.md` §2.2). With
  no overlap there are no cross-terms to absorb, so additivity is a defended modeling choice.

The cache-coverage curve `m_nvme(K)`, `h_vram(K)` is the law's one non-trivial input. §3b shows it is
itself **predictable** from a low-dimensional routing model — so the composition "architecture → flow →
wall-time" runs end-to-end with nothing measured on the target.

### Why the K-sweep, not the storage-throttle sweep, is the right instrument
Regressing `1/tok_s` on `1/B` across the throttled storage sweep **fails**: R²≈0.97 but residuals are
one-signed and the real-drive point is over-predicted by 240–380 ms, because `THROTTLE_MBPS` injects
per-batch *sleep* latency that is not pure `bytes/BW`. The K-sweep holds bandwidth fixed and varies only
cache fractions, isolating the terms. (Throttle-swept bandwidth curves are contaminated and should not
be used to extract a compute floor — itself a result.)

---

## 2. Coefficients — `b_e` is a-priori, not a fudge factor

The expert byte-size `b_e` is **not** "measured once"; it is the summed byte-size of the expert tensors
in the GGUF — a closed form of `(n_embd, n_ff, n_expert, per-tensor quant)`, computable with **zero
engine runs** (`PYTHONPATH=reference/llama.cpp-src/gguf-py`, sum `blk.0.ffn_*_exps`). Across 5 models
spanning 3 families, 3 expert-FFN widths, and 2 quants it tracks both the config and the
trace-measured value:

| model | family / arch | n_ff | quant (gate·up / down) | **a-priori b_e** | trace b_e |
|---|---|---|---|---|---|
| OLMoE-1B-7B | OLMoE (MHA) | 1024 | Q4_K / Q6_K | 3.89 MB | ~4.08 |
| OLMoE-1B-7B | OLMoE (MHA) | 1024 | Q3_K / Q5_K | 3.09 MB | — |
| Qwen3-30B-A3B | `qwen3moe` | 768 | Q4_K / Q6_K | 2.92 MB | ~2.9 |
| Qwen3.5-35B-A3B | `qwen35moe` | 512 | Q4_K / Q5_K | 1.81 MB | 1.90 |
| Qwen3-Next-80B-A3B | `qwen3next` | 512 | Q4_K / Q6_K | 1.95 MB | 1.91 |

`b_e` ranges **1.81→3.89 MB** and is in every case the GGUF expert-tensor byte count, matching the trace
value to **2–5 %** (the gap is 4 KB read-alignment slack). The 35B/80B near-equality (1.90≈1.91) is *not*
a confound — they happen to share both n_ff=512 and n_embd=2048; OLMoE (n_ff=1024) and the 30B (n_ff=768)
show the formula travels. So `A = n_used·L·b_e` is **fully a-priori** (config × closed-form `b_e`).

Per-model latency coefficients (the only fitted quantities), from the K-sweep decomposition:

| model | n_used | L | A = n_used·L·b_e | B_nvme | t_fixed | t_fixed/layer |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 8 | 40 | 608 MB | 2.08 GB/s | 57.8 ms | 1.44 ms |
| Qwen3-Next-80B-A3B | 10 | 48 | 917 MB | 2.11 GB/s | 83.7 ms | 1.74 ms |

(Latency uses the trace `b_e`=1.90/1.91 — the actually-streamed bytes incl. alignment; the a-priori
GGUF `b_e` differs <2 % and is what §3b's prediction uses.)

---

## 3. Validation

### 3a. In-sample — the law tracks each full K-sweep with **one** fitted constant
| | naive | K16 | K24 | K32 | K48 | K64 | **MAPE** |
|---|---|---|---|---|---|---|---|
| 35B measured | 5.47 | 6.41 | 6.72 | 7.11 | 7.01 | 7.52 | |
| 35B law | 5.70 | 6.33 | 6.54 | 6.76 | 7.11 | 7.73 | **2.9 %** |
| 80B measured | 3.36 | 3.81 | 3.98 | 4.12 | 4.24 | 4.52 | |
| 80B law | 3.47 | 3.77 | 3.88 | 4.02 | 4.29 | 4.59 | **2.0 %** |

### 3b. Out-of-sample — predict the 80B end-to-end with **nothing measured on the 80B**

This replaces an earlier, weaker test (which fed the 80B its *own* measured miss curve — almost no
degrees of freedom). The miss curve is instead **predicted** from a routing model.

**The routing model.** `m_nvme(K)` is *not* a function of the routing frequency distribution alone. Drawing
i.i.d. from the exact per-layer frequency table misses the real LRU curve by **56 %** (80B); α-only
(uniform-fresh) by **98 %**. It takes **both** the marginal skew **and** the temporal persistence
(lag-1 expert reuse α) to reproduce it. A "sticky" generator — keep a fraction α of the previous token's
experts, draw the rest from the skew — matches the real per-K miss curve to **2.1 %** (80B) / **4.4 %**
(35B) (`bench/routing_model.py`). *Mechanism:* a hot, temporally-sticky head (α≈0.4, cacheable) plus a
diffuse cold tail — which is why LRU works but prefetch is dead, and it corrects the repo's earlier
"fresh sets each token" framing.

**The parameters are family-stable**, which is what licenses transfer:

| | Zipf s | norm. entropy | mean α |
|---|---|---|---|
| 35B (fit) | 0.999 | 0.778 | 0.391 |
| 80B (measured) | 1.041 | 0.738 | 0.428 |

**The closed loop** (`bench/close_loop.py`): fit `(s, α-profile)` on the **35B** trace, generate
synthetic routing for the 80B's *config* (E=512, u=10, L=48), run it through the validated LRU sim →
predicted `m_nvme(K)`; combine with a-priori `b_e`=1.95, transferred `t_fixed/layer`=1.44, box BW:

| K | 80B tok/s measured | predicted (zero 80B data) | err |
|---|---|---|---|
| 16 | 3.81 | 3.77 | −1.2 % |
| 24 | 3.98 | 3.82 | −3.9 % |
| 32 | 4.12 | 3.90 | −5.4 % |
| 48 | 4.24 | 4.14 | −2.4 % |
| 64 | 4.52 | 4.42 | −2.3 % |

**Honest decomposition (so 3.0 % is not a falsely-clean headline):** the predicted miss curve is **~15 %
pessimistic** (transferred skew is slightly too diffuse → over-predicts cold reads), and the transferred
`t_fixed/layer`=1.44 is **~17 % low** vs the 80B's true 1.74. These biases **partly cancel**. Isolating
the routing error (predicted-`m` + the 80B's *true* floor) gives **8.4 %** — that is the honest
routing-prediction cost; the 3.0 % composite benefits from ~half of it cancelling the floor
under-transfer. The defensible claim: **flow predicted to ~15 %, end-to-end throughput to ~8 %, with
zero target measurements**, across a 2.3× param / 256→512-expert jump.

### 3c. The law explains the RAM-collapse the m-only simulator got wrong
The simulator (models `m` at fixed BW) said RAM tier 2048→4096 cuts cold reads −33 % → a win. On the
16 GB box that pressure collapses NVMe 3.11→1.44 GB/s. The law couples **both**:

| ram | m_nvme | B_nvme | predicted tok/s |
|---|---|---|---|
| 2048 | 0.30 | 3.11 | 5.25 |
| 4096 | 0.20 (−33 %) | 1.44 | **4.36** |

The law predicts more RAM is **net-negative** (bandwidth loss > miss reduction) — matching the hardware
finding, and pinpointing *why* a miss-only model mispredicts. (This is also the falsifiable-core point
(b): `B_nvme` is not globally constant; the collapse is a labeled regime transition.)

---

## 4. The feasibility ratio F/B — the regime gate (now de-circularized)

`F/B = A / B_vram`. The earlier `B_vram`≈6.5 GB was calibrated to reproduce the 30B's 0.17 — circular,
and anchored to the weakest (cross-engine) point. The spill plane (§6) gives `B_vram`
**independently**: the max expert-slot bytes that fit 8 GB = `8192 − base(non-expert resident)`, read
straight off `static_vram(K, ctx)`:

| model | base (resident) | B_vram = 8192−base | A | **F/B** |
|---|---|---|---|---|
| Qwen3.5-35B | 2609 MB | 5583 MB | 608 | **0.11** |
| Qwen3-Next-80B | 1771 MB | 6421 MB | 917 | **0.14** |
| Qwen3-235B-A22B (predicted) | ~est | ~6 GB | 8742 | **~1.4** |

- **F/B ≪ 1** — one token's experts fit in VRAM → cross-token caching is *possible*; steady-state `m`
  floors at the cold tail.
- **F/B ≳ 1** — one token's experts approach/exceed the whole budget → `m` pinned near 1, tok/s collapses
  to `1/(t_fixed + A/B_nvme)`.

**The boundary is order-of-magnitude, not razor-sharp.** `B_vram` carries ~±15 % (what counts as
"budget" is itself soft — §6 shows VRAM spills gracefully into system RAM), so F/B is good to ~±15 %.
The qualitative verdict is robust to this: the three A3B models sit at F/B≈0.1 (deep in the feasible
regime); the 235B at F/B≈1.4 is infeasible **by a wide margin** regardless of the exact `B_vram` — and
more fundamentally because its multi-token working set (~100+ GB) dwarfs *all* host fast memory
(8 GB VRAM + 16 GB RAM), so it streams from NVMe at `m`≈1. Predicted ~0.2 tok/s = unusable-crawl,
consistent with the repo's earlier 235B projections (0.12–0.28 tok/s).

*(The 30B's documented 0.17 used the old anchor; the consistent-basis value needs its static plane,
not yet measured. Earlier notes quoted the 235B at F/B=2.8 from a coarser expert-size estimate — same
verdict, reconciled to ~1.4 here.)*

---

## 5. Inversion — the on-prem sizing question

The law inverts to answer "can model X run at usable speed on hardware Y?" before touching X:

| model | Amdahl ceiling (B_nvme=∞) | drive for 8 tok/s | drive for 10 tok/s |
|---|---|---|---|
| Qwen3.5-35B | 17.3 tok/s | 2.2 GB/s ✓ | 3.9 GB/s ✓ (real Gen4) |
| Qwen3-Next-80B | 11.9 tok/s | 9.9 GB/s (Gen5/RAID) | unreachable (above ceiling at K=64) |

**Faster storage helps the 35B reach 10 tok/s but cannot lift the 80B past ~8** — its compute floor
(`t_fixed`=83.7 ms → 11.9 tok/s ceiling) dominates once streaming is cached down. A falsifiable
hardware-purchasing statement F/B alone cannot make.

---

## 6. The fourth tier — WDDM shared-RAM spill (`bench/spill.py`)

The law's `B_vram` treats the 8 GB card as a hard budget. The hardware has a **fourth tier** the engine
can't see: on Windows/WDDM, when the slot pool or KV cache overflows dedicated VRAM, the OS
**transparently backs the overflow with system RAM over PCIe**. A nominal "VRAM hit" on a spilled page
is really a PCIe read (~16 GB/s) — still **~8× faster than the NVMe (~2 GB/s) it replaces**. So the 8 GB
limit is a *bandwidth tax, not a capacity wall*. Tier order: **true-VRAM → shared-RAM-over-PCIe →
RAM-tier → NVMe**.

**The frontier is exactly predictable.** The `DRYRUN`/`MAXKV` probe (`ctx_cliff.csv`) fits a *perfectly*
linear plane (max residual 0 MB):

```
static_vram(K, ctx) = base + s_K·K + s_ctx·ctx
  80B:  1771 MB + 91.4·K + 0.0492·ctx     35B:  2609 MB + 76.0·K + 0.0410·ctx   (MB)
```

`static_vram = 8192` gives the spill onset for any config (80B: ctx≈41K at K=48, ≈11.6K at K=64). This
also supplies the independent `B_vram` used in §4.

**Crossing it is graceful on both axes (measured):**
- **K axis** (`k_spill.csv`): grow the slot pool 6.1→10.5 GB (K 48→96, **2.3 GB into shared RAM**),
  tok/s keeps *rising* 4.51→5.20.
- **Context axis** (`ctx_spill.csv` K=64 to 24K vs non-spill `ctx_long` control): decode decays
  *smoothly* across the 12K frontier, **no discontinuity**; NVMe-miss stays flat (~16–20 %).

**The spill tax is real and PCIe-rate — and the aggregate `tok/s` hid it** (`bench/spill_graphb.py`,
which required adding an absolute graphB ms/tok timer to the engine). An aggregate-decode null is *not*
"no tax": isolating `graphB` — the phase that reads expert operands out of the slots — shows it is **flat
at 34.4 ms/tok for K=16…48** (peak 3.2–6.1 GB, no spill), then **jumps sharply at the ~7 GB effective
VRAM limit** (the 8 GB card minus WDDM/driver reserve) to **64 ms at K=96** (2.3 GB spilled): a measured
**+30 ms PCIe tax**, **`B_spill` ≈ 8 GB/s** (range 6–11), ~4× the 2.1 GB/s NVMe. It is invisible in
aggregate `tok/s` only because the **same** extra slots cut NVMe-miss time by a near-equal **−30 ms**
(disk 82→52 ms) — the two swings cancel, so `tok/s` stays flat (5.68→5.94). **Spill is graceful not
because it is free, but because each spilled slot trades a 2 GB/s NVMe read for an ~8 GB/s PCIe read — a
~4× win.** *(This corrects an earlier inference from the aggregate null that the tax was ≈0 and that WDDM
preferentially keeps hot pages resident; the direct measurement shows a real, roughly geometric tax —
the measurement the user insisted on, overturning the inferred null.)*

**Procurement reframe.** `B_vram_eff = dedicated VRAM + system-RAM overcommit`, overflow charged at
~PCIe (≈ free vs NVMe), hard ceiling = physical system RAM. So to push **K or context past 8 GB** you
want **system-RAM headroom**, not a bigger card; to go **faster** you want a quicker NVMe up to the
Amdahl ceiling. On this class of box the answer is **cheap system RAM + a fast NVMe, not a bigger
GPU** — the opposite of the naive instinct. *(Windows-specific; see boundary 7.)*

---

## 7. Honest boundaries

1. **Validated on diffuse-routing A3B-class hybrids** (Qwen3-Next-80B, Qwen3.5-35B) + one `qwen3moe`
   transformer, on **one storage profile** (Samsung 970 EVO Plus, ~2.1 GB/s in-engine).
2. **Not validated for MLA attention** (DeepSeek-V2/V3 — not implemented) nor for **concentrated-routing**
   MoE; both could change the `m(K)` curve and the routing model.
3. **`t_fixed` transfer is a 2-point claim** (35B→80B, same engine). `b_e` is now a-priori across 5
   models, and `m(K)` is predicted via the routing model, but the compute floor still rests on two
   same-engine points; a third *same-engine* hybrid (a Qwen3-Next/3.5 variant) would settle it — none is
   on disk and 46 GB free won't hold one. The floor is the small (~30 %) term, but this is the weakest
   remaining link.
4. **The routing-model transfer is validated on one pair.** `(s, α)` are family-stable 35B↔80B (±5–9 %);
   that they generalize to a *different* family (OLMoE) or a concentrated router is untested. The routing
   model itself (skew+persistence) is the falsifiable claim; the cross-model parameter stability is the
   weaker, single-pair one.
5. **`B_nvme` is not globally constant** (§3c): the law is piecewise-constant-BW with the RAM-collapse as
   a labeled transition. `B_pcie`=16 GB/s is nominal; the H2D term is small (8–16 %) so the fit is
   insensitive to it.
6. **The spill tier (§6) is now directly measured:** `B_spill` ≈ 8 GB/s (range 6–11) from the graphB
   timer; the value's spread reflects the effective-VRAM-limit estimate (~7 GB) and the slot hit-fraction.
   The static frontier is exact. Caveat: the byte-accounting that turns the graphB tax into `B_spill`
   assumes graphB reads spilled slots roughly in proportion to their capacity share — plausible given the
   measured tax tracks the spill fraction, but not independently verified.
7. **The spill *gracefulness* and its procurement conclusion are Windows/WDDM-specific.** On a Linux
   allocator that hard-fails instead of paging to host RAM, the overflow OOMs and "buy RAM not VRAM"
   **reverses**. For a non-Windows deployment this must be re-tested or the conclusion scoped to
   Windows-workstation targets.
