# Launch the persistent browser-chat server for the streaming-MoE engine.
# Loads the model once (run_qwen35.exe SERVE mode) and serves a web UI at http://localhost:<port>.
#   .\serve.ps1                 # 80B, port 8080
#   .\serve.ps1 35b 8080 48 2048
param([string]$Model = "80b", [int]$Port = 8080, [int]$K = 48, [int]$Ram = 2048)
$root = $PSScriptRoot
$map = @{
  "80b" = @("models\qwen3next-80b-a3b-q4_k_m.gguf", "Qwen3-Next-80B-A3B")
  "35b" = @("models\qwen3.5-35b-a3b-q4_k_m.gguf",  "Qwen3.5-35B-A3B")
}
if (-not $map.ContainsKey($Model)) { Write-Error "unknown model '$Model' (use: $($map.Keys -join ', '))"; exit 1 }
$mpath = Join-Path $root $map[$Model][0]; $name = $map[$Model][1]
$exe = Join-Path $root "build\run_qwen35.exe"
if (-not (Test-Path $mpath)) { Write-Error "model not found: $mpath"; exit 1 }
if (-not (Test-Path $exe))   { Write-Error "engine not built: $exe"; exit 1 }
Write-Host "Starting $name server on http://localhost:$Port  (loads once; first load ~15-20s)"
py "$root\server\serve.py" --model $mpath --exe $exe --builddir (Join-Path $root "build") `
   --www (Join-Path $root "server") --name $name --k $K --ram $Ram --port $Port
