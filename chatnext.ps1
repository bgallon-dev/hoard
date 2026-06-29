# Launch the Qwen3-Next-80B-A3B streaming chat.
# Hybrid linear-attention MoE (qwen3next) running on the 8 GB RX 6600 by streaming experts from NVMe.
# 80B total / ~3B active: same engine as Qwen3.5-35B (run_qwen35.exe dispatches on the GGUF arch).
#   usage:  .\chatnext.ps1            (defaults K=48 VRAM slots, 2048-expert RAM tier, batched prefill chunk=256)
#           .\chatnext.ps1 32 1024    (override K and RAM-tier size)
#           .\chatnext.ps1 -Chunk 0   (disable batched prefill -> per-position prompt processing)
param([int]$K = 48, [int]$Ram = 2048, [int]$Chunk = 256)
# NOTE: do NOT set ErrorActionPreference=Stop — the engine writes progress to stderr,
# which PowerShell would treat as a terminating error and kill the chat.
$root  = $PSScriptRoot
$model = Join-Path $root "models\qwen3next-80b-a3b-q4_k_m.gguf"
$exe   = Join-Path $root "build\run_qwen35.exe"
if (-not (Test-Path $model)) { Write-Error "model not found: $model"; exit 1 }
if (-not (Test-Path $exe))   { Write-Error "engine not built: $exe (run scripts\build.ps1 src\run_qwen35.cpp run_qwen35 -WithLlama)"; exit 1 }
Set-Location (Join-Path $root "build")   # run from build\ so ggml/llama DLLs load
if ($Chunk -gt 0) { $env:BATCH_PREFILL = "1"; $env:PREFILL_CHUNK = "$Chunk" }   # batched prefill: token-exact, faster TTFT
& $exe $model chat $K $Ram
