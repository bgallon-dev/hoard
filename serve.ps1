# Launch the persistent browser-chat server for the streaming-MoE engine.
# Loads a model once (run_qwen35.exe SERVE mode) and serves a web UI at http://localhost:<port>.
# Every model file that exists under models\ is offered for in-UI switching.
#   .\serve.ps1                 # default 80b, port 8080
#   .\serve.ps1 35b 8080 48 2048
param([string]$Model = "80b", [int]$Port = 8080, [int]$K = 48, [int]$Ram = 2048)
$root = $PSScriptRoot
$defs = @(
  @{ n = "80b"; p = "models\qwen3next-80b-a3b-q4_k_m.gguf" }
  @{ n = "35b"; p = "models\qwen3.5-35b-a3b-q4_k_m.gguf" }
)
$exe = Join-Path $root "build\run_qwen35.exe"
if (-not (Test-Path $exe)) { Write-Error "engine not built: $exe"; exit 1 }
$parts = @(); $avail = @()
foreach ($e in $defs) { $path = Join-Path $root $e.p; if (Test-Path $path) { $parts += "$($e.n)=$path"; $avail += $e.n } }
if (-not $parts) { Write-Error "no model files found under models\"; exit 1 }
$models = $parts -join ";"
if ($avail -notcontains $Model) { $Model = $avail[0] }
Write-Host "Starting server ($Model) on http://localhost:$Port  (models: $($avail -join ', ')); first load ~15-20s"
py "$root\server\serve.py" --models $models --model $Model --exe $exe --builddir (Join-Path $root "build") `
   --www (Join-Path $root "server") --k $K --ram $Ram --port $Port
