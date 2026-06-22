# hoard

**A from-scratch streaming Mixture-of-Experts inference engine that runs 30B–80B models on an 8 GB GPU — by keeping the experts on NVMe and streaming only the few that each token needs.**

Hand-written forward pass, router, KV cache, and a three-tier residency cache (VRAM → RAM → NVMe), built on an AMD RX 6600 (Vulkan, no ROCm). Validated token-for-token against llama.cpp.

## The headline

| | |
|---|---|
| Model | Qwen3-30B-A3B (30.5B total, ~3.3B active/token), Q4_K_M — **17.3 GB on disk** |
| GPU | AMD Radeon RX 6600, **8 GB VRAM**, Vulkan |
| Peak VRAM | **4.4 GB** — the model is ~3.9× the resident footprint and still runs |
| Throughput | **~6 tok/s**, streaming cold experts from NVMe |
| Correctness | OLMoE: **8/8 token-for-token** vs llama.cpp · Qwen3: logits match the reference within fp |

The result is a usable artifact: a private, local, 30B-class **code reviewer** on a ~$200 GPU.

The same engine now streams the larger **hybrid-attention** models too — Qwen3.5-35B-A3B and **Qwen3-Next-80B-A3B** (~45 GB on disk → 6.1 GB VRAM, ~5.8 tok/s), token-validated against llama.cpp. Their 36 linear (gated-delta-net) + 12 full-attention layer mix keeps the KV cache at ¼ rate, so context is rarely the binding constraint. See [`notes/RESULTS-cmp.md`](notes/RESULTS-cmp.md) and the per-layer [`notes/MEMORY-FLOW.md`](notes/MEMORY-FLOW.md).

## How it works

A Mixture-of-Experts model only activates a handful of experts per token (Qwen3: 8 of 128 per layer). hoard keeps the ~1 GB of always-needed weights (attention, router, embeddings) resident in VRAM, and **streams only the active experts** on demand through a three-tier cache:

- **VRAM** — a per-layer LRU pool of `K` expert slots; hot experts stay resident across tokens
- **RAM** — a bounded host-memory tier
- **NVMe** — the full expert set lives on disk; cold experts are fetched with device-direct parallel reads (`FILE_FLAG_NO_BUFFERING`, queue-depth parallelism)

Each layer runs as: **graph A** (attention + KV cache + router → top-8 selection) → **stream** the 8 selected experts into VRAM slots → **graph B** (expert FFN). All matmuls/dequant run on Vulkan via ggml ops; the orchestration, cache, router weighting, and I/O are hand-written.

## What it measured (the interesting part)

This was as much a measurement project as an engineering one. Highlights (full trail in [`notes/STATUS.md`](notes/STATUS.md)):

- **One ratio predicts feasibility — now a validated, out-of-sample law** ([`notes/FEASIBILITY.md`](notes/FEASIBILITY.md), `bench/feasibility.py`). Per-token expert footprint `F = n_used × n_layers × bytes_per_expert` vs the VRAM expert budget `B`. `F/B ≪ 1` → caching works (Qwen3-30B: 0.17, ~86 % VRAM hit); `F/B > 1` → a single token's experts don't even fit, so cross-token caching is *structurally* impossible (Qwen3-235B). Turned into a closed-form tok/s predictor `t_tok = t_fixed + A·m_nvme/B_nvme + A·(1−h_vram)/B_pcie`: it reproduces each model's K-sweep at **~2 %**, and — with the expert byte-size taken from the GGUF, the miss curve **predicted** from a routing model fit on the 35B, and the compute floor transferred (**nothing measured on the 80B except for scoring**) — predicts the **80B's** throughput to **~8 %** across a 2.3× param jump it never saw, while correctly labelling the never-run 235B *unusable-crawl*. The miss curve itself turned out to be predictable only with **both** routing skew **and** temporal expert-reuse (memoryless models miss by 56 %+). It inverts to the on-prem sizing question ("what drive does model X need for Y tok/s?") and exposes a hard **Amdahl ceiling** — faster storage lifts the 35B to 10 tok/s but cannot push the 80B past ~8.
- **MoE routing is diffuse, not concentrated.** Over 256 generated tokens a focused code-review prompt touches **77 % of all experts (13.5 GB)** — *more* than open-ended text. That exceeds 16 GB of RAM, putting the model on the "cliff" where NVMe latency decides usable-slow vs unusable-crawl. It lands on usable-slow (~6 tok/s) only because the NVMe sustains ~2 GB/s.
- **The streaming I/O "ceiling" was mostly a bug, not physics.** Realized bandwidth long sat at ~2.0 GB/s, blamed on the per-layer barrier. Bisection with a standalone pool harness ([`src/pooltest*.cpp`](src/pooltest2.cpp)) found the real causes on the 80B engine: per-call I/O-thread spawning collapsed effective queue depth, and — the big one — loading weights through *buffered* reads pollutes the OS page cache, which silently routes later `FILE_FLAG_NO_BUFFERING` reads through the cache manager and **halves** their throughput (3.5 → 1.9 GB/s). A persistent worker pool + an unbuffered weight loader lifted real-decode bandwidth to **3.11 GB/s** and throughput **+35 %** (4.28 → 5.78 tok/s on the 80B), losslessly. The per-layer barrier is real but small — it caps decode at ~3.0 GB/s vs the drive's 3.5 GB/s isolated max, *not* 2.3.
- **The memory flow, mapped end-to-end** ([`notes/MEMORY-FLOW.md`](notes/MEMORY-FLOW.md)). Decode moves **262 MB/token** of expert weights NVMe→RAM→VRAM — and *more* (283 MB) over PCIe, since RAM-tier hits also stage to VRAM — all gated by an 8 KB residual thread between layers (a 33,000× mismatch). A cache simulator validated at 98.8 % against the engine, plus a 10-domain expert-activation corpus, settled the optimization space: prefetch and clustering are **dead** (the cold tail is diffuse and unpredictable), domain-aware tiering is the lone lossless survivor (~+3 %), and a bigger RAM tier — which the simulator predicted as a win — is **hardware-falsified** (memory pressure collapses NVMe bandwidth and then OOMs; the 16 GB box is already at its `ram_cap` ceiling).
- **The 8 GB VRAM limit is a bandwidth tax, not a wall — there's a hidden 4th cache tier** ([`notes/FEASIBILITY.md` §6](notes/FEASIBILITY.md), `bench/spill.py`). On Windows/WDDM, overflowing dedicated VRAM — a bigger slot pool *or* a longer context — spills transparently into system RAM over PCIe (~16 GB/s), still **~8× the NVMe** it stands in for. So K and context degrade *gracefully* past 8 GB rather than off a cliff: the slot pool runs to **10.5 GB still rising** (4.51→5.20 tok/s), and a 24K-context run crosses its 12K spill point with **no discontinuity**. A direct graphB-timing sweep shows the spill tax is **real and PCIe-rate** — graphB jumps from 34→64 ms/tok once slots spill past the ~7 GB effective limit, a measured **+30 ms tax paid at PCIe class** — but it's **invisible in aggregate `tok/s` because the same extra slots cut NVMe-miss time by a near-equal −30 ms** (the two cancel). So spill is graceful not because it's free, but because each spilled slot trades a slow NVMe read for a much faster PCIe read. *(The tax and direction are measured; the bandwidth value `B_spill`~8 GB/s is a bounded estimate — see [FEASIBILITY.md §6](notes/FEASIBILITY.md). This corrects an earlier inference that the tax was ≈0.)* The frontier is exactly predictable — `static_vram = 1771 + 91.4·K + 0.049·ctx` MB. Procurement consequence: on an 8 GB box you want **cheap system RAM + a fast NVMe, not a bigger GPU**.
- **The curriculum of bugs.** Per-layer mixed quantization, `_fseeki64` for >2 GB offsets, an architecture-specific router bug (`norm_topk_prob`: Qwen3 renormalizes the top weights, OLMoE doesn't, and llama.cpp hardcodes this per-arch rather than storing it in the GGUF), and the gated-delta-net q/k repeat-interleave that must key on the architecture, not the head counts. See `notes/STATUS.md` and `notes/RESULTS-cmp.md`.

## Requirements

- Windows, an AMD GPU with Vulkan (developed on RX 6600 8 GB), 16 GB RAM, an NVMe SSD
- [MSYS2 UCRT64](https://www.msys2.org/) `g++` — the build links prebuilt DLLs; it does **not** use cmake or the MSYS2 shell
- Prebuilt **ggml + llama Vulkan DLLs** and the matching headers from [llama.cpp](https://github.com/ggml-org/llama.cpp) (developed against build **b9660**)
- A GGUF model, e.g. [Qwen3-30B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) (Q4_K_M) or [OLMoE-1B-7B-0924-Instruct-GGUF](https://huggingface.co/) (for validation)

These large dependencies are **not** in the repo — fetch them into:

```
reference/
  llama.cpp-src/      # clone of llama.cpp (provides ggml + llama headers)
  llama-vulkan/       # prebuilt Vulkan release DLLs: ggml*.dll, llama.dll, ...
models/
  qwen3-30b-a3b-q4_k_m.gguf
```

## Build

From the repo root, in **PowerShell** (not the MSYS2 shell — see [`scripts/build.ps1`](scripts/build.ps1)):

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -O2 -std=c++17 `
  -I reference\llama.cpp-src\ggml\include -I reference\llama.cpp-src\include `
  src\run_moe_stream.cpp `
  reference\llama-vulkan\ggml.dll reference\llama-vulkan\ggml-base.dll reference\llama-vulkan\llama.dll `
  -o build\run_moe_stream.exe
Copy-Item reference\llama-vulkan\*.dll build\    # runtime DLLs next to the exe so backends load
```

## Run

**Interactive streaming chat / code reviewer:**

```powershell
cd build
.\run_moe_stream.exe ..\models\qwen3-30b-a3b-q4_k_m.gguf chat 48 2048
```

`48` = VRAM expert slots per layer (`K`); `2048` = RAM-tier size in experts. Type at the `user>` prompt — multi-line is supported (paste code, blank line to send, `/quit` to exit).

**One-shot / benchmark mode** (feeds token IDs; prints the three-tier access split, NVMe bandwidth, and a decode-phase breakdown):

```powershell
.\run_moe_stream.exe ..\models\qwen3-30b-a3b-q4_k_m.gguf "785,6722,315,9625,374" 32 cache 48 2048
```

## Repo layout

| path | what |
|---|---|
| `src/run_moe_stream.cpp` | **the original engine** (30B) — streaming forward, 3-tier cache, device-direct NVMe I/O, chat REPL |
| `src/run_qwen35.cpp` | **the hybrid engine** (Qwen3.5-35B, Qwen3-Next-80B) — adds gated-delta-net linear attention, a persistent I/O worker pool, an unbuffered weight loader, and `GATEDUMP`/`PROFILE` activation-trace instrumentation |
| `src/run_m1.cpp` | reference recompute used for token-for-token validation |
| `src/ref_gen.cpp` · `src/tok.cpp` · `src/gguf_dump.c` | reference generator, vocab-only tokenizer, GGUF metadata dumper |
| `src/iobench.cpp` · `src/pooltest*.cpp` · `src/evict.c` | NVMe queue-depth/block-size benchmark, standalone I/O-pool harnesses (isolated the page-cache throughput bug), OS page-cache eviction |
| `src/run_spec*.cpp` | residency-aware speculative decoding (measured *slower* — see STATUS) |
| `tools/fbin.c` | f32 tensor comparator |
| `notes/STATUS.md` · `notes/RESULTS-cmp.md` · `notes/MEMORY-FLOW.md` | the measured trail; 35B-vs-80B streaming sweep + VRAM frontier; per-layer memory-flow analysis |
| `scripts/build.ps1` | build driver |

## Status & limitations

- **Platform-specific**: Windows + AMD/Vulkan + NVMe; the I/O path uses Win32 (`CreateFile`/`ReadFile`).
- **Architectures**: validated on OLMoE (MHA), Qwen3-MoE (GQA + per-head QK-norm + `norm_topk_prob`), and the hybrid gated-delta-net families Qwen3.5-35B-A3B and Qwen3-Next-80B-A3B (36 linear + 12 full-attention layers). MLA (DeepSeek-V2/V3) is not implemented.
- **Speed**: ~5.8–6 tok/s — this buys *reach* (running models far larger than VRAM), not throughput; the Amdahl ceiling (~30 % fixed overhead) caps any lossless I/O win at ~2.2×, and the diffuse, non-saturating expert working set is the wall.

## References

Background reading that informed the residency-cache design (cited by arXiv id):
MoE-ERAS (expert residency) · 2508.21706 · 2510.10302 · 2511.14102 · 2604.02715.

## License

[MIT](LICENSE).
