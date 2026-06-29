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

## MILESTONE 5 COMPLETE: Qwen3.5-35B-A3B hybrid linear-attention (src/run_qwen35.cpp)
The frontier arch: qwen35moe = 40 layers (30 Gated Delta Net LINEAR + 10 gated FULL attention) + 256/8 MoE
+ shared expert. Forked run_moe_stream; reused 3-tier streaming + device-direct I/O + chat REPL.
P1 full-attn: fused Q+gate proj (attn_q double-width, view-split), per-head q/k RMS-norm (head_dim 256),
  IMRoPE (ggml_rope_multi, mode 40, sections [11,11,10,0], 4 pos/token), GQA 16/2, sigmoid gate on attn out.
P2 recurrent state: per-linear-layer GDN state [128,128,32] + conv window [3,8192], persistent VRAM, zeroed/gen.
P3 GDN: attn_qkv/attn_gate proj -> ssm_conv (causal conv1d + silu, rolling conv state) -> l2_norm q/k ->
  decay g=softplus(ssm_alpha.x+ssm_dt)*ssm_a, beta=sigmoid(ssm_beta.x) -> FUSED ggml_gated_delta_net(q,k,v,g,b,
  state,K=1) [on Vulkan!] -> gated RMS-norm(out,z) -> ssm_out. Single-token recurrence (no chunked scan needed).
P4: routed 256/8 (streamed, same cache) + gated shared expert ffn_*_shexp (sigmoid(gate_inp_shexp)*down(silu*up)).
P5 VALIDATED: token-for-token vs llama.cpp oracle = 6/6 EXACT ("The capital of France is" -> 11751 13 198 760
  6511 314). Runs ~5 tok/s streaming experts from NVMe. The engine now runs the CURRENT frontier architecture.
BUGS FOUND (curriculum): (1) ip (positions) unused in linear layers -> galloc didn't allocate -> tensor_set
  crash; guard set only for full-attn. (2) IMRoPE needs 4 positions/token (mrope sections), not 1.
KEY: my per-position engine makes GDN a single-step recurrence -> the fused ggml_gated_delta_net op (Vulkan)
  does it directly; no need to port the chunked delta-rule scan. All ops (ssm_conv/l2_norm/softplus/gdn/imrope) on Vulkan.

## M5 chat robustness fix
Chat (long generation) hit ggml.c:1751 view-out-of-bounds: full-attn KV cache was sized max_kv=T+ngen+2
(=1026 for chat) but a long <think> trace ran past it. Fix: max_kv=4096 for chat + only allocate KV for the
10 full-attn layers (qwen35: linear layers use recurrent state, not KV -> saves VRAM) + clean stop at max_kv-1.
One-shot stays 6/6 exact. Chat now completes (e.g. 189 tok @ 4 tok/s, no crash). Open cosmetic: qwen35 <think>
tokens differ from Qwen3's (151667/8) so the REPL's think-suppression doesn't catch them (different vocab).

## CHAT WIRED to Qwen3.5-35B (chat35.ps1 launcher)
chat35.ps1 (project root): runs run_qwen35.exe on the Qwen3.5 model, K=48 + 2048-expert RAM tier defaults
(.\chat35.ps1 [K] [Ram] to override). Runs from build/ so DLLs load. NOTE: don't set ErrorActionPreference=Stop
(engine stderr would abort it). Fixes for usability: (1) <think> suppression uses qwen3.5 ids 248068/248069
(not qwen3's 151667/8). (2) Qwen3.5 ignores the /no_think soft switch and is a heavy thinker -> for "what is a
hash map" it spent 1024 tokens thinking and hit the cap before answering (suppressed -> empty output). FIX:
disable thinking by pre-filling an empty <think></think> in the assistant prompt (enable_thinking=false). Now:
concise visible answers, ~7.6 tok/s at K=48 (peak VRAM fits 8 GB; chat KV is only 10 full-attn layers x max_kv 4096).

## RECURRENCE DRIFT: 256-position teacher-forced oracle (closes the GDN footnote)
Setup: llama.cpp CPU oracle generated a 256-token NON-degenerate answer (stack-vs-queue explanation, not a
repetition loop) -> oref256.{txt,log}. Teacher-forced the engine on the same 23-prompt+256-gen=279 id seq
(TFLEN=23, K=48/RAM=2048), dumped top-5 logits/pos, diffed vs oracle. (Build harness: seq256.csv, eng256.err.)
RESULT: top-1 agreement 252/256 = 98.4%. ALL 4 misses are model near-ties: at pos 50/99/217/225 the oracle's
OWN top1-vs-top2 gap was 0.012/0.041/0.079/0.007 logits -> a sub-0.1 CPU-vs-Vulkan FP diff flips a coin-flip
argmax (re-running the oracle with other threading flips them too). Not drift.
DRIFT (the real question - does per-step error compound?): NO. Mean |delta logit| by 32-pos block:
 0-31:0.115  32-63:0.104  64-95:0.131  96-127:0.115  128-159:0.154  160-191:0.167  192-223:0.196  224-255:0.178.
Grows ~0.11 -> ~0.20 then PLATEAUS and turns back down = sublinear, bounded. Signature of a CONTRACTIVE
decay-gated recurrence: GDN gate alpha=softplus(.)*ssm_a < 1 forgets old error faster than new error accrues,
so FP noise saturates into a ~0.2-logit band instead of running away. Distribution: mean 0.145, median 0.102,
p90 0.284, p99 1.275, max 1.365 (vs logits of magnitude 20-36 -> ~0.3% of signal). HONEST BOUNDARY: token-exact
validated to 256 state updates; the SATURATION (not the count) is the evidence the recurrence is numerically
stable and is what extrapolates past 256 - NOT proven token-exact to the 262K-context ceiling.
(Re-validated post-qwen3next-fix below: still 252/256, mean 0.145 - bit-identical, the arch-gated GDN fix does not touch qwen35moe.)

## M6: Qwen3-Next-80B-A3B (qwen3next arch) - same engine, validated 16/16
Ported run_qwen35.cpp to also run qwen3next (80B total / ~3B active, 48 layers: 36 GDN-linear + 12 full-attn,
512 experts/top-10 + shared expert, head_dim 256). One binary, dispatched on the GGUF arch string.
DELTAS vs qwen35moe: (a) fused ssm_ba [2*Hv] split into beta/alpha by de-interleave (per k-head [vpg b][vpg a]);
(b) full-attn rope = ggml_rope_ext NEOX n_rot=64 (NOT IMRoPE/rope_multi), 1 pos/token; (c) full_attn_interval key
absent -> default 4 (same (il+1)%4 rule); (d) standard Qwen vocab 151936, eos 151645, Instruct (no <think>).
TWO BUGS found via per-layer activation diff vs a CPU ref_dump (the decisive tool - dumps llama.cpp's per-layer
l_out / attn_residual / ffn_shexp stats + .bin vectors; compared cosine + norm ratio engine-vs-ref per layer):
 BUG 1 (the real one): ggml_gated_delta_net q/k head broadcast. qwen3next.cpp ALWAYS repeat-interleaves q/k
   from Hk=16 to Hv=32 heads ([g0,g0,g1,g1,...]) BEFORE the fused GDN; qwen35moe.cpp does NOT (its fused path
   relies on the op's internal broadcast - `repeat only if !fused`). Without the repeat the 80B GDN was cosine
   0.994 but only 0.844x magnitude -> compounded over 36 GDN layers into a uniform ~0.5x logit COMPRESSION and a
   collapsed massive-activation (the shared expert's -68 attention-sink dim came out ~1/3). MUST gate the repeat
   on arch==qwen3next, NOT on Hv!=Hk: gating on Hv!=Hk also repeats for qwen35moe and CRASHED it to 15/256.
 BUG 2 (latent, harmless here): ffn_gate_inp_shexp is BF16 in the 80B (F32 in the 35B) and the RX 6600 reports
   bf16:0 -> upconvert bf16->f32 on load for all non-expert tensors. Turned out bit-identical (Vulkan handled it)
   but kept as a correctness guard.
LOCALIZATION TRAIL (how, not just what): per-layer |x| ratio engine/ref was ~0.9 through L5 then dropped to ~0.47
at L6 and stayed -> L6 = first big massive activation. Split L6 -> MoE-driven (routed 0.26 vs shared 19.8) ->
shared expert -> ungated FFN 0.44x + gate 0.74x -> input fx cosine 0.955, att+res cosine 0.97 mag 0.70 -> GDN
attn cosine 0.994 mag 0.844 -> the q/k repeat made it 1.000/1.001 exactly.
VALIDATED: 80B teacher-forced 16/16 top-1 vs CPU oracle, |dL1| ~0.05-0.96 (FP-level, same as 35B). Chat works:
"capital of France + landmark" -> "Paris ... Eiffel Tower" @ 4.29 tok/s, 80B on 8 GB. Launcher: chatnext.ps1.
This is the most powerful model the engine runs at usable speed (A3B keeps it fast; bigger-active would crawl).

## PREFILL FEASIBILITY (the coding-harness gate) — measured + adversarially audited (4/4 claims high-conf)
Question: can the engine host an agentic coding harness? Decode (6.3-7.6 tok/s) was never the issue; PREFILL
(time-to-first-token) is. Instrumented run_qwen35.cpp with two ENV-GATED hooks (normal runs byte-identical):
`SYNTH_LEN=N` fills the prompt with N xorshift pseudo-random ids (diffuse routing); a `PREFILL:` line in the
one-shot summary reports TTFT = clock from before pos 0 (tg0, after reset_state) to first emitted token (tfirst).
MEASURED (Qwen3.5-35B qwen35moe, K=48, ram=2048/10240 experts):
  256 tok -> 47.4 s (185 ms/tok) ; 2048 -> 279 s (136) ; 4096 -> 595 s (145, RISING).
SUPER-LINEAR (not linear): secant ms/tok 129 (256->2048) -> 154 (2048->4096). Mechanism: the 10 FULL-attention
layers concat K/V to pos p then matmul over p+1 keys = O(T^2) integrated over prefill; plus working set outgrows
the fixed 2048-expert RAM tier -> rising cold-tail NVMe. Extrapolation is ORDER-OF-MAGNITUDE ONLY (3-pt quadratic
has 0 dof, +-28% swing from +-5% single-point noise, curvature contaminated by ~18 s fixed startup in the 256-pt):
qualitatively 8k ~tens of min, 16-24k ~hour+, and likely an UNDER-estimate (24k KV pressure + cold tail unmodeled).
PER-TURN RE-PREFILL (confirmed, run_qwen35.cpp:445): chat re-prefills the ENTIRE conversation every turn
(reset_state zeros GDN/conv state + loop from p=0 over full hist) -> cost O(N^2) over N turns, NOT pay-once.
THE DECOMPOSITION (why this flips the strategy) — 595 s @ 4096 tok breaks down as:
  per-position graph dispatch (40 layers x ~28 tiny matmuls x 4096 pos) = 237 s (40%)  [REMOVABLE by batching]
  redundant expert re-streaming (evict from 48-slot pool + re-read later)  = 350 s (59%)  [REMOVABLE by batching]
  irreducible NVMe floor (read each of 19 GB experts ONCE)                 =   8 s (1%)   [hardware floor]
Read the WHOLE 19 GB expert set once = 9.1 s @ 2.08 GB/s (5.4 s @ device 3.5). So ~99% of the prefill wall is an
ARTIFACT of reusing the single-token decode loop for prefill, NOT a hardware limit. b_e=1.90 MB/expert, t_fixed=57.8 ms/pos.
BATCHED-PREFILL ESTIMATE: collapse 4096 per-position graphs into one graph/layer (GEMV->GEMM) + stream each touched
expert once -> ~21 s @ 4096 (12 s compute @ ~2 TFLOP/s Q4/Vulkan + 8 s I/O floor) = 20-30x; helps ALL context sizes
(small contexts are dispatch-bound, so batching the dispatch is the whole win; ~9-10x even at 256 tok). [ESTIMATE, not built.]
VERDICT: a real file-reading/many-turn coding harness is GATED on the ENGINE, not the GPU. Two levers, both feasible
without new hardware: (1) BATCHED PREFILL (20-30x on first-context ingestion — the big one); (2) INCREMENTAL RESUME
across turns (skip reset_state, resume at a 'state-valid up to R' watermark; gdn_state/conv_state/kv.* already persist,
allocated once) kills the O(N^2). Today's engine only supports a narrow short-context few-turn async helper. The Python
agent loop on serve.py stays trivial; it's gated on these. AUDIT (4 parallel skeptics): measurement valid; per-turn
re-prefill confirmed; synth-prompt proxy fair-to-conservative vs real diffuse code routing; batched-prefill floor confirmed.

## BATCHED PREFILL BUILT (the lever) — token-exact, 7.7x measured (src/run_qwen35.cpp prefill_batched, ENV BATCH_PREFILL)
Built the batched-prefill path: process the prompt in chunks of C tokens with per-BATCH graphs (not per-position),
streaming each routed expert ONCE per (layer,chunk) instead of once per position. ENV: BATCH_PREFILL=1, PREFILL_CHUNK=C,
PP_ORACLE=1 (compare final-pos argmax to the per-position path). Carries KV/GDN/conv state across chunks (persistent) -> token-exact.
DE-RISK FIRST (src/batchgdn_test.cpp, no model): ggml_gated_delta_net AND ggml_ssm_conv with n_tokens=T are BIT-EXACT
(max|d|=0.0) vs T sequential single-step calls on Vulkan, T up to 16. So batched GDN is a SINGLE gated_delta_net(...,K=1)
call with the token dim lifted 1->C (the CPU ref runs the same per-token scan internally); Sv=128 is Vulkan-supported.
This killed the top risk (the engine had only ever called these ops with n_tokens=1). mul_mat_id over a slot pool was
already proven by run_specloop. Conv-state carry = last d_conv-1 cols of the [d_conv-1+C, cd] window (also bit-exact).
IMPLEMENTATION: embed[n_embd,C] -> per layer { graph A batched (full-attn: q/k-norm + IMRoPE(qwen35,4*C section-major pos)
/NEOX(qwen3next) + GQA + a HOST-BUILT CAUSAL MASK[nkv,C] in soft_max_ext (the one batched-only piece; per-pos passed nullptr) ;
OR GDN: ssm_conv over [d_conv-1+C,cd] + single gated_delta_net(n_tokens=C) + arch-gated q/k repeat-interleave) -> router
softmax[n_expert,C] -> host top-n_used + wnorm + expert UNION -> stream union ONCE into a GRAPH-LOCAL typed pool [n_embd,n_ff,U]
(re-typed per layer, U=chunk union, backed by the existing RAM/NVMe tiers) -> graph B: mul_mat_id(GP/UP/DP, cx, idT) + gated
shared expert + weighted sum + residual } -> head on last position. The per-position decode path is UNTOUCHED (stays the oracle).
VALIDATED token-exact (final-pos argmax == per-position oracle): qwen35moe C=1/2/4/8/16/64/128, multi-chunk carry (T=64 C=16==C=64,
T=256 C=128), and qwen3next-80B C=1/4. Both arches' GDN/conv/rope/ssm_ba/repeat batch correctly.
MEASURED (Qwen3.5-35B, K=48, ram=2048; per-position baselines from the PREFILL FEASIBILITY section above):
  T=2048: per-pos 278.9s ; batched C=256/512/1024/2048 = 125.2/75.7/51.0/36.7 s = 2.2/3.7/5.5/7.6x (single chunk best).
  T=4096: per-pos 595s   ; batched C=2048/4096 = 84.9/77.3 s = 7.0/7.7x ; 18.9 ms/tok vs 145 (per-pos) = 7.7x per-token.
  Qwen3-Next-80B: batched T=1024 C=1024 = 39.3 s (38.4 ms/tok) ~ 5-6x.
Speedup grows with C (fewer chunks -> less cross-chunk re-stream, more GEMM amortization); single-chunk is best when it fits 8 GB.
HONEST GAP: measured 7.7x < the 20-30x PROJECTION (~20s @4096) -- same projection-overshoots-measurement pattern as the I/O-fork
ceiling. The residual ~57s is overhead the projection ignored: per-(layer,chunk) graph construction + gallocr, RAM->VRAM tensor_set
re-streaming into a FRESH graph-local pool every chunk (NO cross-chunk/cross-layer VRAM residency reuse -- the deferred persistent-
typed-pool optimization), the first-pass NVMe read of the ~19 GB working set, and the 10 full-attn layers' O(T^2) score matrices.
Those are the next levers (persistent VRAM pool w/ LRU across chunks is the biggest).
FULLY WIRED INTO generate() (decode-resume DONE): generate() takes a Cbatch arg; when >0 it calls prefill_batched to build KV/GDN/conv
state for 0..T-1, samples the first token from the final-position logits, then runs the EXISTING per-position decode loop from p=T on that
state (no reset). Used by BOTH one-shot and the chat REPL when BATCH_PREFILL=1. The per-position path (Cbatch=0) is untouched = the oracle.
VALIDATED token-exact END-TO-END (not just first token): VALIDATE_PREFILL runs per-position vs batched+decode over N gen tokens and compares
the full sequence -> MATCH (qwen35moe); real greedy chat output ("three prime numbers..." / "capital of Japan") is BYTE-IDENTICAL batched vs
per-position on both 35B and 80B. Launchers chat35.ps1 / chatnext.ps1 now default to batched prefill (PREFILL_CHUNK=256; -Chunk 0 disables).
Tooling: BATCH_PREFILL/PREFILL_CHUNK/VALIDATE_PREFILL/VNGEN env-gated; normal runs unaffected. New file src/batchgdn_test.cpp (primitive smoke test).
REMAINING (optional polish, not blocking use): persistent VRAM pool to close toward ~20s; incremental resume ACROSS chat turns (only prefill new
tokens -> kills the O(N^2) multi-turn re-prefill; today each turn batched-prefills the whole conversation, but fast).

## BATCHED PREFILL OPTIMIZATION ROUND (measure-first; PTIME=1 phase timers)
MEASURED the batched-prefill breakdown (T=4096 C=4096, PTIME) to optimize the REAL bottleneck, not the assumed one:
  graphA 38s (compute 29s [full-attn 20s + GDN 10s] + construct 8s) | graphB 35s (compute 28s) | nvme 10s | poolstage 2.6s.
=> COMPUTE-BOUND (~58s of GPU compute = the RX 6600 floor), NOT streaming-bound. This REJECTED the planned "persistent VRAM expert pool"
  optimization: it targets poolstage (2.6s) + nvme re-stream, which are tiny -- not worth the mixed-quant-type complexity. Measure-first paid off again.
OPTIMIZATIONS DONE (token-exact preserved; both numerically inert -> verified single-chunk still == per-position, coherent multi-chunk == per-position):
  (1) ggml_init context 512MB->16MB for graphA/graphB: the per-(layer,chunk) 512MB metadata-arena malloc/free was churning OS page tables
      (~80 graphs/chunk). Cut graphA construct 18s->8s. The metadata needs <1MB; 512MB was 500x oversized.
  (2) Mask + position HOIST: the causal mask [nkv,m] and IMRoPE position buffer are identical for all 10 full-attn layers in a chunk -> build ONCE
      per chunk instead of 10x (was rebuilding a 67MB mask + 16.7M-iteration fill per full-attn layer).
  Net single-chunk T=4096: ~86s (compute floor ~58s + construct ~13s + nvme ~10s) = ~7x; the compute floor is the hard GPU limit (q4_K/Vulkan
  on RX 6600 ~1 TFLOP/s effective). Further single-chunk gains need kernel work (flash-attn for the 20s full-attn O(T^2), faster mul_mat_id) -- out of scope.
INCREMENTAL RESUME ACROSS TURNS (the big multi-turn win) -- DONE: prefill_batched takes resume_from; chat tracks g_state_pos = positions with
  valid KV/GDN/conv state, carried across turns. Each turn prefills ONLY the new tokens (resume_from..Tp), not the whole conversation -> O(N) per turn
  instead of O(N^2). Demonstrated: turns prefilled 23(cold)/20/19/19 new tokens over growing 23/44/64/85 ctx; answers Paris/Berlin/Rome/Madrid correct.
  VALIDATED token-exact: incremental == full-re-prefill (NO_INCREMENTAL=1) byte-identical (Paris/Berlin/Rome). Watermark reset on <<<RESET>>>/LOAD.
CORRECTNESS NOTE (multi-chunk): batched multi-chunk == per-position on COHERENT prompts (verified PREFILL_CHUNK=8 over a real prompt = byte-identical).
  On DEGENERATE SYNTH random-token prompts the chunked-GDN-scan FP reassociation (4x64-with-carry vs one 256-scan) flips near-tied argmaxes into an
  attractor -- the same benign greedy near-tie sensitivity already documented for CPU-vs-Vulkan. Default chat PREFILL_CHUNK=256: prompts <=256 are
  single-chunk (bit-exact); larger use multi-chunk (FP-close, coherent-validated). New env PTIME=1 (phase timers), NO_INCREMENTAL=1 (force full re-prefill).

## OPTIMIZATION-IDEAS TEST ROUND (after a web-research pass; measure-first, all 4 candidate angles tested -- 3 rejected/marginal)
Refined PTIME breakdown @T=4096 C=4096: graphA 43s (compute 29s [full-attn 19s + GDN 10s] + H<->D-xfer 7.5s + build/alloc 6.8s) | graphB compute 31s
  | nvme 13s | poolstage ~3s. => ENGINE IS COMPUTE-BOUND (~59s q4_K matmul on the RX 6600). Reducible overhead ~20s = H<->D residual round-trip + cold-tail nvme.
(1) CHUNKWISE-PARALLEL GDN -- REJECTED by isolation benchmark (src/batchgdn_test.cpp time_gdn): the fused ggml_gated_delta_net op is ALREADY parallel
   (0.0017 ms/token, plateaus; ~225 ms TOTAL for 30 GDN layers @4096). The "10s GDN" in PTIME is the surrounding DENSE MATMULS (attn_qkv/attn_gate/ssm_out
   projections + conv + router), NOT the recurrence. A chunked-matmul delta rule (WY/UT transform) would optimize the wrong thing. Clean negative result.
(2) SPECULATIVE PREFETCH to hide nvme -- nvme (13s) is the irreducible COLD TAIL: RAM tier is already 11.4 GB of 15.8 GB system RAM, so a bigger tier can't
   reduce it. Hideable behind compute (we're compute-bound) but needs a speculative expert predictor (PreScope LLaPor / Mixtral-offload gating-similarity, ~90-99%
   acc) + a THREAD-SAFE RAM tier -- a real sub-project, bounded ~13s. DEFERRED with the path documented.
(3) PERSISTENT SCRATCH BUFFER (arena) -- IMPLEMENTED (token-exact): one 32 MB host metadata arena reused for every per-(layer) ggml_init instead of malloc/free.
   But malloc was NOT the bottleneck (the earlier 512MB->16MB fix already killed the big-malloc churn): construct = H<->D-xfer 7.5s + build/alloc 6.8s. Marginal; kept as hygiene.
(4) PINNED / page-locked host memory -- REJECTED for this setup: the 11.4 GB RAM tier exceeds VirtualLock working-set limits, AND ggml-Vulkan stages H<->D through
   its OWN internal buffer (pinning the user pointer never reaches the DMA path). Not actionable without modifying ggml-vulkan.
ON-GPU RESIDUAL -- DONE (the one clean lever): keep the residual stream in PERSISTENT VRAM across layers instead of round-tripping it through host.
   3 persistent VRAM tensors per prefill_batched call (Xg=residual, fxg=router/expert input, fig=post-attn residual; [n_embd,Cm], ~96 MB @ Cm=4096, freed at end).
   graphA reads Xg (ix, no H2D) + cpy's fx->fxg, ffn_inp->fig (no D2H); only the 4 MB router probs leaves the GPU. graphB reads fxg (ifx) + fig, does the residual
   add moe+shared+ffn_inp ON-GPU and cpy's the new residual into Xg (no D2H). head reads Xg's last column. RESULT: H<->D-xfer 7.5s -> 0.4s (-7.1s), the host-orchestrated
   round-trip is gone. VALIDATED token-exact: single-chunk batched==per-position; coherent multi-chunk (PREFILL_CHUNK=8) byte-identical; multi-turn incremental correct.
   ADVERSARIAL REVIEW (3 lenses) = CORRECT: cpy-into-persistent-view is the canonical ggml pattern (== KV-cache writes); the shared gallocr provably never touches the
   rbuf tensors (t->buffer!=NULL excludes them); residual add order == oracle; Cm=min(C,Tp-start) >= every chunk's m (no overrun); works for both archs (qwen3next also
   has a shared expert + is_qwen35==true). Added an rbuf NULL-check (clean fail on VRAM-OOM vs segfault). Net single-chunk ~80s (compute floor ~58s + nvme ~10-14s noisy + build/alloc ~7s).
   REMAINING overhead is the cold-tail nvme (10-14s, needs speculative prefetch + thread-safe tier) and ggml build/alloc (~7s); the 28s mul_mat_id + 19s full-attn are the
   hard q4_K/Vulkan compute floor (only faster kernels / flash-attn move them -- out of scope for a hand-built ggml engine). New diagnostic: time_gdn in batchgdn_test.

## POST-OPTIMIZATION PREFILL BENCHMARK (BENCH=K env: K in-process reps, rep0 cold / rest warm-RAM-tier; noise-averaged, supersedes noisy single runs)
Qwen3.5-35B, K=48, RAM=2048, single-chunk (PREFILL_CHUNK=4096). WARM-mean TTFT (3 reps):
  T=512 -> 17.8s (34.7 ms/tok, 28.8 tok/s) | T=1024 -> 24.1s (23.5, 42.5) | T=2048 -> 34.0s (16.6, 60.2) | T=4096 -> 71.7s (17.5, 57.1).
HEADLINE: steady-state ~16-17 ms/tok (~58-60 tok/s prefill) at T>=2048; ~8.3x vs per-position (T=2048 279s->34s, T=4096 595s->71.7s), token-exact.
Per-token cost falls then plateaus (fixed ~7s build/alloc amortizes); slight uptick @4096 = full-attn O(T^2). COLD ~= WARM at T>=1024 (cold 72.8 vs warm 71.7 @4096):
once the model file is OS-cached + RAM tier holds the working set, repeated prefills don't degrade -- cold-tail nvme is small in practice. Small prompts pay the fixed
graph-construction tax (34.7 ms/tok @512). The remaining floor is pure q4_K/Vulkan compute (mul_mat_id + full-attn). New env: BENCH=K (in-process prefill bench).
