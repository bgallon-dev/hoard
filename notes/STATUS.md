# Project status

**Goal:** from-scratch streaming MoE inference engine on AMD/Vulkan. 4 milestones:
M1 coherence (token-exact vs llama.cpp), M2 streaming (peak VRAM << model size),
M3 residency cache (speedup + hit rate), M4 (stretch) residency-aware speculation.
Success = M3 with M1 intact, every layer understood. See the user's goal statement.

## Hardware (confirmed)
- AMD Radeon RX 6600, RDNA2, **8 GB** GDDR6, PCIe 4.0 x8 (~16 GB/s = streaming ceiling).
  Vulkan caps: fp16 yes, bf16 no, int-dot yes, no matrix cores, 32KB shared, warp 32.
- 16 GB system RAM (~12 usable), i7-12700K. No ROCm → Vulkan only. Intel iGPU = Vulkan1 (avoid).

## Decisions
- Model: **OLMoE-1B-7B-0924-Instruct**, Q4_K_M (`models/olmoe-...q4_k_m.gguf`, ~4.2 GB).
- Scope: hand-build forward pass + router + streaming + cache; **GEMM/dequant via ggml ops**.
- Reference: prebuilt `llama-completion.exe` (build b9660), Vulkan. Use `llama-completion`
  NOT `llama-cli` (cli ignores `-no-cnv` and hangs in conversation mode).

## Toolchain (IMPORTANT — Option C)
Source build via cmake FAILED (0xC0000139 DLL-entrypoint; MSYS2 fork flakiness from Git Bash).
Working path: **ucrt64 g++ links the prebuilt ggml DLLs directly** (C ABI). Drive from
**PowerShell** (native; MSYS2 shells are flaky). Build with `scripts\build.ps1`.
- headers: `reference/llama.cpp-src/ggml/include` (b9660 clone)
- DLLs: `reference/llama-vulkan/ggml*.dll` (copied into `build/`)
- run exes from `build/` so `ggml_backend_load_all()` finds backend DLLs.

## Done (Phase 0)
- Reference oracle runs, coherent text via Vulkan (`reference/ref_complete.log`).
- `notes/olmoe_spec.md` = exact forward-pass contract, ALL values confirmed vs GGUF.
- `src/hello_ggml.c` (backend enum), `src/gguf_dump.c` (metadata, `notes/gguf_dump.txt`).
- Byte split: **311 MB resident** / **3901 MB experts** (48 tensors, ~4.08 MB/expert).

## M1 reference contract
Prompt tokens (15, instruct-templated): 29,93,4537,49651,187,510,5347,273,6181,310,29,93,515,5567,49651
Greedy continuation (temp0): "Paris. Paris is one of the most popular tourist destinations
in the world and is known for its iconic landmarks such as the Eiff[el]..."
My engine feeds identical token IDs → must reproduce these greedy tokens.

## M1a DONE — dense oracle (per-stage construction harness)
Per the design amendment: build the forward pass against a DENSE oracle (validate every
stage) not a sparse one (only final tokens). QK-norm and no-renorm-router are coherent-then-
drift bugs invisible at token level — catch them at their own layer.
- `src/ref_dump.cpp` instruments prebuilt llama.dll via `cb_eval` (FA disabled). Dumped 38
  golden tensors to `notes/ref_acts/` for the France prompt: attn_norm-0, Qcur-0 (#0 raw,
  #1 post-RoPE), Qcur_normed-0, Kcur*, Vcur*, ffn_inp/norm/moe_out-0, l_out-0..15,
  result_norm, result_output. Stats in `notes/ref_acts/_meta.txt`.
- **QK-norm axis CONFIRMED from data**: Qcur_normed-0 is [2048,15] at norm time → rms reduces
  over full 2048 (not per-128-head). Spec reading verified against the reference tensor.
- `tools/fbin.c` (build/fbin.exe): `top <bin> [k]` and `diff <a> <b>` (max/mean abs err, rms,
  argmax match). The workhorse for stage validation.
- **First-token target**: argmax(result_output) = **token 187 ('\n'), logit 28.9** (next 190@8.7).
  My forward pass, given the 15 prompt IDs, must reproduce this at position 15.
- Attention internals captured (de-mask the residual): kq-0 (scores), kq_soft_max-0 (weights,
  first=1.0 causal), kqv-0 (weighted-V), **kqv_out-0** (post-wo, PRE-residual attn output).
  Validate attention directly vs kqv_out-0, not only residual-masked ffn_inp-0.

## M1b validation discipline (per design feedback)
- Read per-layer l_out error as a SLOPE not a threshold: a bug = step change in error at its
  stage; honest FP/dequant divergence = gradual accumulation. Gate on argmax-match + relative
  error loosening with depth.
- MoE dequant is NOT my code: Option C → ggml's kernels dequant q4_K/q6_K on bytes I load
  verbatim (same kernels as reference) ⇒ dequant correct by construction. Residual MoE risk is
  ASSEMBLY (mul_mat_id operand layout, ids tensor, router no-renorm weight gather, 8-way
  weighted sum) — caught by ffn_moe_out-0. Expect MoE to take the most iterations.

## M1b DONE — forward pass built + validated (src/run_m1.cpp)
Full OLMoE forward over ggml ops, all experts resident, no KV cache (recompute prefix).
Loader fix: **_fseeki64** (32-bit fseek overflows past 2GB → late tensors fail to load).
Validated stage-by-stage vs notes/ref_acts/:
- embd, attn_norm, Q/K/V, QK-norm, RoPE: **bit-exact (0.0)**.
- ffn_norm, ffn_moe_out, l_out-0..14: smooth FP-accumulation ramp 1e-4→6e-3 (no bug spike).
- result_output argmax = **187, logit 28.9121 vs ref 28.9063** ✓ first-token target hit.
- **kqv_out anomaly RESOLVED**: ref's cb'd kqv_out dump is unreliable (ref ffn_inp-kqv_out ≠
  ref embd by 6.7e-3) — a transient node aliased in the Vulkan sched, read stale by cb_eval.
  My attention is provably correct: my (ffn_inp - kqv_out) == my embd to 3.7e-9, and ffn_inp
  matches ref. LESSON: validate against residual-stream tensors (embd/ffn_inp/l_out/result),
  which are kept & reliable; transient internals (kqv_out) can be stale in cb_eval dumps.

## Next: M1c — multi-token greedy + diverse prompts
Extend run_m1 to greedy-generate N tokens (append argmax, re-run growing prefix), compare
full token sequence to reference continuation. Capture refs for diverse prompts (factual /
code / list / story) via llama-completion (FA disabled). PASS = all sequences match.

## M2 + M3 DONE (src/run_moe_stream.cpp) — streaming MoE decode
Architecture: non-expert weights resident in VRAM; experts in HOST RAM, streamed into a
bounded per-layer VRAM slot pool. Per-layer split: graph A (attn w/ KV cache + router) ->
read top-8 -> stream experts -> graph B (per-expert FFN over slots) -> host weighted sum.
Modes: naive (8 slots, recopy/token) | cache (K slots, per-layer LRU).

**M2 (streaming):** naive output == resident KV-decode 24/24 (mechanism correct).
peak VRAM = **807.5 MB** running the **4212 MB** model = **5.2x reduction**.
Bug found+fixed: Q4_K_M uses PER-LAYER mixed quant (down_exps q6_K on some layers, q4_K on
others) — must track expert ggml_type per layer, not globally.

**M3 (cache):** naive 13.6 tok/s (0% hit) vs cache: K16=21.5(50%), K32=30.1(75%), K48=41.6
tok/s (**89.6% hit, 3.05x speedup**). Cache is LOSSLESS (cache==naive tokens, verified).
Speedup tracks hit rate: copies 13696 (naive) -> 1422 (K48), ~10x fewer. VRAM/speed knob = K.

KV-cache decode note: greedy output FP-drifts from the recompute reference after ~18-24 tokens
on near-tied numeric continuations (benign; M1 proof is on recompute 4/4). naive==cache proves
the streaming/cache mechanism is exact.

## SUCCESS (M1-M3): coherent ref-matched output; 5.2x less VRAM; 3.05x cache speedup @ 89.6% hit.

## M4 (stretch, in progress): residency-aware speculation
src/run_spec.cpp: draft = same model, TOP-1 routing restricted to RESIDENT (cached) experts
=> 0 streaming copies, ~3.1x faster than top-8 target. Per-step acceptance (does residency-
biased top-1 draft argmax == top-8 target argmax, given target's verified context):
  K=16: 64.1% accept, draft 3.1x, projected joint 1.30x vs cache-alone target
  K=32: 65.6% accept, projected 1.25x
  K=48: 67.2% accept, projected 1.10x
Acceptance is the explanatory number; joint tok/s is the standard greedy spec-decode
projection from MEASURED acceptance + draft/target speeds (a fully-integrated batched-verify
+ cache-rollback loop would realize it; that's the remaining engineering).

## ALL 4 MILESTONES COMPLETE
M1 coherence: 4/4 token-for-token vs llama.cpp.
M2 streaming: 807.5 MB peak VRAM vs 4212 MB model = 5.2x reduction, output intact.
M3 cache: 3.05x speedup @ 89.6% hit (vs naive re-copy-every-token), lossless.
M4 speculation: 65.6% draft acceptance, projected 1.1-1.3x joint speedup over cache-alone.

## M4 VALIDATION (src/run_specloop.cpp) — projection -> MEASUREMENT
Built the integrated greedy spec-decode loop: residency-biased top-1 draft (k tokens) +
ONE batched verify over the k tokens (top-8, expert loads amortized via mul_mat_id over a
per-layer slot POOL [in,out,K]) + accept-longest-prefix + correction + KV rollback (position
counter; verify K/V overwrite draft K/V).

MEASURED (story prompt, K=48):
  k=4: target-alone 51.9 tok/s vs spec 25.1 tok/s = 0.48x (acceptance 33.3%, 2.29 acc/round)
  k=2: target-alone 52.4 tok/s vs spec 33.7 tok/s = 0.64x (acceptance 81.6%, 2.58 acc/round)
=> The 1.25x PROJECTION did NOT hold; measured spec is SLOWER on this engine. Why:
  1. Autoregressive draft acceptance compounds down (65.6% per-position -> ~2.3-2.6 acc/round).
  2. Engine per-token overhead dominates (34 graph launches + host round-trips/token), so the
     top-1 draft is not cheap enough in wall-clock to amortize the extra draft+verify+correct.
  3. Batched verify must stream the expert UNION (up to ~32 experts/layer for k=4) > a single
     token's 8 — more streaming, not less.
Spec output is lossless w.r.t. its batched-verify model (coherent), FP-diverges from the
single-token baseline (same benign greedy FP sensitivity as M2/M3).

HONEST CONCLUSION: residency-aware speculation does NOT beat cache-alone AS BUILT. It would
need a genuinely cheap draft (separate small / few-layer model) and a verify whose union stays
cache-resident — matching the papers' finding that MoE-offload speculation is non-trivial
(SpecMoEOff). The validation's value is exactly this: the projection was optimistic, the
measurement falsified it, and the mechanism is now understood.

## Chat interface (src/chat.cpp) — text in / text out
Interactive REPL over the streaming engine. Tokenize/detokenize via llama.dll (vocab_only load,
same path as ref_gen.cpp); the per-token forward pass (graphs E/A/B + head, KV cache, LRU slot
pool) is copied verbatim from run_moe_stream.cpp so behaviour is identical. Build with
`scripts\build.ps1 src\chat.cpp chat -WithLlama` (new -WithLlama switch links llama.dll + adds
the include path), run from build\.
  chat <model> [--raw] [--ctx N] [--max N] [-K N] [--naive] [--show-prompt]
- Default = instruct: per turn feeds `<|user|>\n{msg}<|assistant|>` (+ BOS 50279 at conv start).
  Verified the templated IDs reproduce the M1 framing tokens exactly:
  `50279 | 29 93 4537 49651 187 | <content> | 29 93 515 5567 49651`. Model emits the leading
  `\n` itself (M1 first-token=187), which the REPL trims for display.
- Multi-turn: KV cache + position persist across turns; "France?" then "Germany?" -> Berlin
  (context carried). Stops on EOG (50279); feeds the eos into KV as the turn boundary.
- --raw: token-exact vs ref_gen — "The capital of France is" -> ids `510 5347 273 6181 310`,
  continues " Paris." Sanitizes a leading UTF-8 BOM / trailing CR from stdin first.
- Per-turn metrics line: tokens, decode tok/s, cache hit %, ctx fill. Default cache K=32 ->
  ~76-80% hit (matches M3's K32=75%); ~25-32 tok/s decode on the RX 6600.

## 3-TIER CACHE (VRAM->RAM->NVMe) built + gating measurement (src/run_moe_stream.cpp)
Experts now stay ON DISK (file offsets); bounded RAM LRU tier above; VRAM slot cache on top.
`ram_experts` arg caps the RAM tier (simulates too-big-for-RAM by holding < all experts).
Bit-identical output verified (disk path == all-resident) — M2 check repeated for disk.

GATING measurement (OLMoE, 100 tok, the three-way access split):
- VRAM K=48 (holds 75% experts): VRAM 89.6%; of misses RAM caught 33%->0% as RAM shrinks.
  Large VRAM -> residual cold tail has poor reuse -> RAM tier nearly useless.
- VRAM K=16 (holds 25%, the realistic too-big ratio): VRAM 49.6%; of misses RAM caught
  86%(RAM=100%) / 54%(50%) / 13%(25%) / 0%(12%). NVMe fall-through 7% -> 50% of total.
KEY FINDING: NVMe viability hinges on RAM holding the WORKING SET (~955/1024 experts touched),
not the whole model. RAM >= working set -> green (NVMe rare 7%). RAM < working set -> red
(NVMe up to 50%). Transition is steep, governed by routing skew. AND the RAM tier matters
MOST when VRAM is small (the regime that matters) — the tiers are not independent.
CAVEATS: (1) OLMoE's skew, not a 30B's (shape generalizes, numbers don't). (2) disk reads hit
OS page cache (OLMoE fits RAM) -> split is exact but tok/s != real NVMe latency. Real latency
needs model > RAM => Qwen3-30B-A3B (engine-compatible: GQA+QK-norm+no-shared) is the confirmation.

## QWEN3-30B-A3B Q4_K_M: real too-big-for-RAM regime (17.28 GB, experts ~16-18.8 GB)
Engine generalized + VALIDATED on Qwen3 (qwen3moe: 48L, GQA 32/4, head_dim 128 -> n_embd_attn
4096!=2048, per-head QK-norm, 128 experts/top-8, expert_ff 768, ~2.9 MB/expert).

CORRECTNESS GATE (the curriculum bug): Qwen3 needs norm_topk_prob=TRUE (renormalize top-8
router weights to sum 1); OLMoE=FALSE. llama.cpp HARDCODES this per-arch in build_moe_ffn
(8th arg norm_w: qwen3moe `true`, olmoe `false`) — it is NOT a GGUF key. My engine used OLMoE's
raw-softmax weights -> Qwen3 experts under-weighted by 1/sum(top8). Signature: logits
systematically ~3-7 low + one gross token error (pos7 picked 4180 vs ref 15072). Fix: divide
wsel by its sum when ARCH!=olmoe. After fix, teacher-forced logits match CPU reference within
fp at every position (6/8 argmax exact; 2 disagreements are sub-0.06-logit dead-heats). OLMoE
stays 8/8 token-for-token (GPU-vs-GPU exact). Reference: CPU-only (17 GB won't fit 8 GB VRAM;
must remove ggml-vulkan.dll from run dir — llama auto-grabs Vulkan0 + AMD ~947 MB single-alloc
cap otherwise). Also added: GGUF eos_token_id read (Qwen3 151645), vocab-only tok.exe.

WORKING SET = GREEN/CLIFF VERDICT (native top-8, /no_think short-output regime, ~6 GB free RAM):
  GENERAL (256 tok):     4094 distinct experts = 11.7 GB (67%); spills 6 GB RAM by tok 16 (6.4 GB)
  CODE REVIEW (256 tok): 4705 distinct experts = 13.5 GB (77%)  <- MORE diffuse than general
  Saturation: steep to ~64 tok (general: 16->6.4, 32->8.1, 64->10.2 GB) then plateaus 67-77%.
VERDICT: CLIFF. Working set 2.2x over free RAM, spills within 16 tokens; no short regime fits.
Fine-grained 8-of-128 does NOT concentrate routing — code review is the WORST (most diffuse) case,
falsifying the specialization prior. Routing is diffuse like OLMoE (91%/100tok anchor held).
ARCH RESULT: bounded RAM tier caught 0.3% (working set >> tier -> reuse distance > capacity ->
3-tier degenerates to VRAM 66% + NVMe 34%). RAM tier only helps if it holds a big WS fraction.
NEXT (now warranted, gated on cliff): real COLD-cache NVMe latency (current 4 tok/s is
page-cache-assisted; ~35% of 17 GB file cached) to quantify usable-slow vs unusable-crawl.

## QWEN3-30B NVMe LATENCY (the usable-slow vs crawl number) — Samsung 970 EVO Plus NVMe
Instrumented H.read() timing (disk_ns/bytes), isolated from compute. cache K=24, ram_cap=512.
  WARM (128 tok): 34537 reads, 99.5 GB @ 2.00 GB/s, avg 1.118 ms / 2.88 MB expert-read.
  COLD (64 tok, after evicting 12 GB cache): 26609 reads, 76.6 GB @ 2.00 GB/s, avg 0.982 ms.
COLD ~= WARM: page cache is IRRELEVANT because working set (11-13 GB) >> cacheable RAM — nothing
useful to cache, so the warm number was already the real operating point (not optimistic).
End-to-end ~4.1-4.6 tok/s. Decode is ~54% NVMe streaming / ~46% compute+overhead. Engine streams
~340-777 MB per token from disk (VRAM tier catches 69%, NVMe 31%, RAM 0.3%).
VERDICT: USABLE-SLOW, not crawl. Fast NVMe (2 GB/s, ~1 ms/2.88 MB read) keeps the cliff tolerable
— a 256-tok review ~= 60 s. On SATA SSD (~0.5 GB/s) this would be ~1 tok/s (crawl); HDD unusable.
This IS the "NVMe-backed inference" regime Qwen3-30B-A3B was designed for. Lever to go faster:
raise K (more VRAM slots -> higher VRAM hit -> fewer disk reads); RAM tier is a dead end here.
Tooling lesson: build from PowerShell g++, NOT the Bash tool (Bash silently drops the linked exe).

## QWEN3-30B RESIDENCY OPTIMIZATION (staying on the sweet spot — two free levers, no download)
EXP A (VRAM lever, K sweep, 64 tok, ram=512): K=16/32/48 -> VRAM hit 56/77/86%, NVMe 42/23/14%,
  tok/s 3.05/4.17/4.73, peak VRAM 3.2/5.4/7.6 GB. K is the speed dial; K~48 maxes the 8 GB.
  Diminishing returns (compute becomes the floor as disk shrinks). RAM tier ~0% at every K.
EXP B (RAM lever, ram_cap sweep, 32 tok, K=16): ram 256/1024/2048 (0.7/3/5.9 GB) -> RAM hit
  0/10.7/27.5%, NVMe 43/33/16%. The "dead" tier was STARVED, not dead: value ~= tier/working_set.
  The 0.3% earlier was ram=512 (1.5 GB) vs 13 GB WS. VRAM & RAM are SUBSTITUTABLE disk-read
  reducers: K=16+5.9GB-RAM (84.3% cached) ~= K=48 alone (85.7%) — trade cheap RAM for scarce VRAM.
COMBINED OPTIMUM (K=48 + ram=2048): tok/s=5.20 (vs 4.0 original = +30%), VRAM 85.7% / RAM 2.7% /
  NVMe 11.5%; peak VRAM 7.6 GB, RAM tier 6.3 GB. 256-tok review ~49 s (was ~64 s).
IRREDUCIBLE FLOOR: residual 11.5% NVMe = the COLD TAIL (experts touched once/gen -> never
  cacheable -> must read >=1x). True cold random 2.9 MB read = 2.51 ms = ~1 GB/s (vs 2 GB/s blended).
  This is the fundamental streaming floor for diffuse-routing MoE; caching cannot remove it.
COMPLETE RESIDENCY MODEL: 3 tiers/3 speeds (VRAM instant > RAM ~0.3 ms memcpy > NVMe ~2.5 ms cold).
  Minimize NVMe by maximizing (VRAM slots + RAM tier) coverage of the working set. On 8 GB VRAM +
  ~8 GB free RAM, combined cache ~13.5 GB ~= Qwen3-30B WS, so NVMe bottoms at the cold-tail ~11%.

## I/O-FORK CEILING — two cheap measurements gating the parallel-read build (project discipline)
GATE 1 (drive bandwidth vs queue depth, src/iobench.cpp, cold FILE_FLAG_NO_BUFFERING, model file):
  864 KB scattered (real expert sub-read): QD1 1.77 / QD2 3.31 / QD4 3.52 GB/s -> SATURATES AT QD4 (~2x).
  2.9 MB scattered: QD1 2.84 / QD4 3.55 GB/s (1.25x; big reads already use device at QD1). Ceiling ~3.5 GB/s.
  KEY: my engine cold = ~1.0 GB/s (buffered fread, QD1) vs device-direct QD1 1.77-2.84 GB/s. Engine I/O is
  at ~1/3.5 of the drive. Total achievable ~3.5x = ~1.8x device-direct (free, QD1) + ~2x QD4 parallelism.
  IMPACT: at K=48, decode is ~67% cold-tail disk time -> 3.5x I/O ~= 30B 5.2->~8-10 tok/s; 235B 0.24->~0.84.
  Build target is SIMPLE: 4 in-flight reads + bypass the buffered/page-cache path (NOT 24-deep, NOT exotic).
GATE 2 (prefetch ceiling = cross-token routing locality, TRACE=1, review 128 tok):
  avg consecutive-token top-8 overlap = 3.37/8 = 42.2% (predict t+1 from t). L0=9% (input, volatile) -> L31=60%.
  ECONOMICS VERDICT: prefetch is NOT a co-equal win. A wrong prefetch (58%) is free ONLY if it uses otherwise-
  idle I/O; once QD4 saturates the drive there is no idle bandwidth, so mispredicts steal from on-demand reads.
  Net-positive only if strictly idle-gated to the layer-boundary bubble (graph A compute) = ~few %. Secondary.
CONCLUSION: gate passed for parallel+direct I/O (measured ~3.5x, half is a trivial buffered->direct fix, helps
  the 30B we run AND the cold-tail floor proven irreducible-by-caching). Prefetch demoted to marginal/idle-gated.

## BUILD: device-direct + QD-parallel expert reads (src/run_moe_stream.cpp) — measured, NOT the projection
DirectIO: 8 NO_BUFFERING handles + aligned scratch + buffered EOF fallback; per-layer fetch refactored to
3 phases (classify+reserve RAM slots -> parallel device-direct reads via run_jobs -> serial RAM->VRAM stage).
CORRECTNESS preserved: OLMoE 8/8 bit-exact through the new path; Qwen3 review coherent.
MEASURED gain (review prompt): bandwidth 1.0 -> 2.3 GB/s (2.3x). tok/s K=48 ram=512: 4.73 -> 5.54 (+17%);
combined optimum K=48 ram=2048: 5.20 -> 5.95. Big-batch naive (QD8, 235B-relevant): 2.30 GB/s.
PROJECTION OVERSHOT (3.5x / 8-10 tok/s). Three measured reasons:
  (1) realized 2.3 not 3.5 GB/s: the scratch->RT-buffer memcpy is INSIDE the read path (iobench had no copy);
  (2) disk is only ~40-57% of decode, not 67%: serial H2D VRAM staging + per-layer graph construction is a
      non-disk floor I/O can't touch; (3) 2.2x on a ~50% fraction = +17%, not 2x.
LESSON: the iobench gate was right about the DEVICE (3.5 GB/s achievable) but measured a sub-component, not the
full read->memcpy->stage->compute path. A gated projection can still overshoot when the gate isn't the whole path.
NEXT (optional): read directly into 4096-aligned RT buffers (kill the memcpy) -> ~3.2 GB/s -> 30B ~6, 235B ~0.4
tok/s (vs ~0.28 now, ~0.12 old buffered). Marginal for 30B (non-disk floor); matters most for streaming-bound 235B.

## DECODE BREAKDOWN (gate before optimizing the non-disk floor) — K=48 ram=2048, 64 tok, 6.18 tok/s
  disk 37% | graphB(expert FFN)+sum 33% | graphA(attn+router) 17% | H2D-stage 12% | embed/head ~1%
CORRECTS the assumption: H2D (the assumed "non-disk floor") is only 12%; the real non-disk cost is graphB
33% = Vulkan DISPATCH OVERHEAD (24 tiny matmuls/layer: 8 experts x gate/up/down) + per-layer graph rebuild,
NOT compute (single-token expert FFN is ~38M MACs, trivial). Disk STILL the biggest at 37%.
LEVERS (each ~1/3): (1) disk 37% -> memcpy-free aligned reads (cheap) -> ~26%; (2) graphB 33% -> ggml_mul_mat_id
(one batched expert op, needs slot pool as one indexable tensor) OR graph-structure reuse across layers (medium).
H2D 12% not worth it. Measure-before-optimize redirected the target away from the assumed H2D floor.

## MEMCPY-FREE READS (read device-direct into aligned RT buffers) — hypothesis FALSIFIED
RamTier buffers now 4096-aligned (lazy VirtualAlloc), reads land DIRECTLY in them (head offset tracked,
to_vram offsets by head); removed scratch + memcpy. Correctness preserved (OLMoE 8/8). tok/s 6.18->6.42 (+4%).
BUT bandwidth UNCHANGED: 2.15 -> 2.22 GB/s. The scratch->RT memcpy was NOT the cap (my hypothesis was wrong).
REAL CAP (structural): the per-layer BARRIER (each layer needs all its experts before it can compute) makes I/O
bursty -- short read bursts (3-24 reads) between compute, so the NVMe queue ramps up/down and never reaches the
sustained high-QD throughput iobench's long-lived back-to-back threads hit. Big-batch naive (QD8) also = 2.3, so
it's not spawn overhead -- the engine CANNOT issue reads ahead of the dependency. Only speculative prefetch
(42% accurate, marginal economics) could sustain QD. So ~2.3 GB/s is the engine's structural ceiling; iobench's
3.5 (sustained) is unreachable here. THIRD microbenchmark-vs-reality gap: iobench measured ideal sustained reads;
the engine's real pattern is bursty per-layer. I/O journey total: 5.20 -> 6.42 tok/s (+23%); disk now structurally
capped. Remaining lever is graphB (34%, non-disk: 24 tiny matmuls/layer + graph rebuild) -> mul_mat_id / graph reuse.

## CHAT INTERFACE (the artifact) — src/run_moe_stream.cpp `chat` mode
Refactored the per-token forward into a `generate(seq,T,maxnew,on_token)` lambda (params SHADOW seq/T/ngen so
the 178-line loop body was untouched; one-shot and chat both call it). Added a `chat` REPL: links llama.dll for
vocab (vocab_only load), applies the Qwen3 chat template (+/no_think), tokenizes, streams detokenized tokens live
to stdout, suppresses <think></think> (ids 151667/151668), multi-turn history, multi-line input (blank line sends,
/quit exits, EOF sends pending). Run:  run_moe_stream.exe <model> chat [K] [ram_experts]   (e.g. chat 48 2048).
Build now links llama:  g++ ... -I reference/llama.cpp-src/include ... llama.dll.
VALIDATED: one-shot OLMoE still 8/8 bit-exact (refactor safe). Chat on Qwen3-30B: correct streamed code review at
~6 tok/s, caught both planted bugs (i<=n OOB, malloc(n) vs n*sizeof(int)) + missing NULL check. The engine is now
a usable private local 30B-class streaming code reviewer on the 8 GB RX 6600.
