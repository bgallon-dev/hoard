# hoard

**A from-scratch streaming Mixture-of-Experts inference engine that runs a 30B-parameter model on an 8 GB GPU — by keeping the experts on NVMe and streaming only the few that each token needs.**

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

## How it works

A Mixture-of-Experts model only activates a handful of experts per token (Qwen3: 8 of 128 per layer). hoard keeps the ~1 GB of always-needed weights (attention, router, embeddings) resident in VRAM, and **streams only the active experts** on demand through a three-tier cache:

- **VRAM** — a per-layer LRU pool of `K` expert slots; hot experts stay resident across tokens
- **RAM** — a bounded host-memory tier
- **NVMe** — the full expert set lives on disk; cold experts are fetched with device-direct parallel reads (`FILE_FLAG_NO_BUFFERING`, queue-depth parallelism)

Each layer runs as: **graph A** (attention + KV cache + router → top-8 selection) → **stream** the 8 selected experts into VRAM slots → **graph B** (expert FFN). All matmuls/dequant run on Vulkan via ggml ops; the orchestration, cache, router weighting, and I/O are hand-written.

## What it measured (the interesting part)

This was as much a measurement project as an engineering one. Highlights (full trail in [`notes/STATUS.md`](notes/STATUS.md)):

- **One ratio predicts feasibility.** Per-token expert footprint `F = n_used × n_layers × bytes_per_expert` vs the VRAM expert budget `B`. `F/B ≪ 1` → caching works (Qwen3-30B: 0.17, ~86 % VRAM hit). `F/B > 1` → a single token's experts don't even fit, so cross-token caching is *structurally* impossible (Qwen3-235B: 2.8).
- **MoE routing is diffuse, not concentrated.** Over 256 generated tokens a focused code-review prompt touches **77 % of all experts (13.5 GB)** — *more* than open-ended text. That exceeds 16 GB of RAM, putting the model on the "cliff" where NVMe latency decides usable-slow vs unusable-crawl. It lands on usable-slow (~6 tok/s) only because the NVMe sustains ~2 GB/s.
- **The streaming I/O has a structural ceiling.** The per-layer barrier (a layer needs all its experts before it can compute) makes reads bursty, capping realized bandwidth at ~2.3 GB/s — short of the drive's ~3.5 GB/s sustained ceiling — without speculative prefetch.
- **The curriculum of bugs.** Per-layer mixed quantization, `_fseeki64` for >2 GB offsets, and an architecture-specific router bug (`norm_topk_prob`: Qwen3 renormalizes the top-8 weights, OLMoE doesn't, and llama.cpp hardcodes this per-arch rather than storing it in the GGUF). See `notes/STATUS.md`.

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
| `src/run_moe_stream.cpp` | **the engine** — streaming forward, 3-tier cache, device-direct NVMe I/O, chat REPL |
| `src/run_m1.cpp` | reference recompute used for token-for-token validation |
| `src/ref_gen.cpp` · `src/tok.cpp` · `src/gguf_dump.c` | reference generator, vocab-only tokenizer, GGUF metadata dumper |
| `src/iobench.cpp` · `src/evict.c` | NVMe queue-depth benchmark, OS page-cache eviction |
| `src/run_spec*.cpp` | residency-aware speculative decoding (measured *slower* — see STATUS) |
| `tools/fbin.c` | f32 tensor comparator |
| `notes/STATUS.md` | the full measured trail (every number, every bug) |
| `scripts/build.ps1` | build driver |

## Status & limitations

- **Platform-specific**: Windows + AMD/Vulkan + NVMe; the I/O path uses Win32 (`CreateFile`/`ReadFile`).
- **Architectures**: validated on OLMoE (MHA) and Qwen3-MoE (GQA + per-head QK-norm + `norm_topk_prob`). MLA (DeepSeek-V2/V3) is not implemented.
- **Speed**: ~6 tok/s — this buys *reach* (running models far larger than VRAM), not throughput.

## References

Background reading that informed the residency-cache design (cited by arXiv id):
MoE-ERAS (expert residency) · 2508.21706 · 2510.10302 · 2511.14102 · 2604.02715.

## License

[MIT](LICENSE).
