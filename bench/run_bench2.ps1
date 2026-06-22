# Generalized streaming-MoE benchmark driver (35B qwen35moe OR 80B qwen3next).
# One binary run_qwen35.exe dispatches on the GGUF arch. Writes CSVs to bench/results/<Tag>/.
#   powershell -File bench\run_bench2.ps1 -Tag q80 -Model models\qwen3next-80b-a3b-q4_k_m.gguf -PromptIds bench\general_80b_ids.txt -DomainIds bench\domain_80b_ids.txt -Which fast
# Which: fast (K+RAM+storage), wl (domain vs general working set), ctx (one long 16K gen), all
param(
  [Parameter(Mandatory=$true)][string]$Tag,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$PromptIds,
  [string]$DomainIds = "",
  [int]$CtxTokens = 16384,
  [string]$Which = "fast"
)
$root = Split-Path -Parent $PSScriptRoot
$exe  = "$root\build\run_qwen35.exe"
function Resolve-P([string]$p){ if ([System.IO.Path]::IsPathRooted($p)) { return $p } else { return "$root\$p" } }
$model = Resolve-P $Model
$res   = "$root\bench\results\$Tag"
New-Item -ItemType Directory -Force $res | Out-Null
$ids   = (Get-Content (Resolve-P $PromptIds)).Trim()

function Run-Engine {
  param([int]$Ngen,[string]$Mode,[int]$K,[int]$Ram,[int]$ThrottleMbps=0,[string]$Csv="",[string]$IdsArg=$ids)
  if ($ThrottleMbps -gt 0) { $env:THROTTLE_MBPS = "$ThrottleMbps" } else { Remove-Item Env:\THROTTLE_MBPS -ErrorAction SilentlyContinue }
  if ($Csv -ne "") { $env:CSV = $Csv } else { Remove-Item Env:\CSV -ErrorAction SilentlyContinue }
  $out = & $exe $model $IdsArg $Ngen $Mode $K $Ram 2>$null | Out-String
  Remove-Item Env:\THROTTLE_MBPS -ErrorAction SilentlyContinue
  Remove-Item Env:\CSV -ErrorAction SilentlyContinue
  $m = @{ toks=0; vram=0; vpct=0; rpct=0; npct=0; gbps=0 }
  if ($out -match 'tok/s=([\d.]+)')                         { $m.toks = [double]$Matches[1] }
  if ($out -match 'peak_VRAM=([\d.]+) MB')                  { $m.vram = [double]$Matches[1] }
  if ($out -match 'VRAM=([\d.]+)%\s+RAM=([\d.]+)%\s+NVMe=([\d.]+)%') { $m.vpct=[double]$Matches[1]; $m.rpct=[double]$Matches[2]; $m.npct=[double]$Matches[3] }
  if ($out -match 'NVMe:.*= ([\d.]+) GB/s')                 { $m.gbps = [double]$Matches[1] }
  return $m
}

function Sweep-K {
  $f = "$res\k_sweep.csv"; "K,tok_s,vram_pct,ram_pct,nvme_pct,peak_vram_mb" | Out-File $f -Encoding utf8
  $m = Run-Engine -Ngen 64 -Mode "naive" -K 0 -Ram 2048   # K=0 -> engine sizes naive pool to n_used (8 for 35B, 10 for 80B)
  "0,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
  Write-Host "[$Tag] naive: $($m.toks) tok/s"
  foreach ($K in 16,24,32,48,64) {
    $m = Run-Engine -Ngen 64 -Mode "cache" -K $K -Ram 2048
    "$K,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
    $flag = if ($m.toks -eq 0) { ' <-- FAILED (likely OOM)' } else { '' }
    Write-Host "[$Tag] K=$K : $($m.toks) tok/s, VRAM hit $($m.vpct)%, peak $($m.vram) MB$flag"
  }
}

function Sweep-Ram {
  # ram_cap in #experts; ~1.95 MB/expert -> 512~1GB 1024~2GB 2048~4GB 3072~6GB. 8/12GB infeasible (15.8GB phys RAM).
  $f = "$res\ram_sweep.csv"; "ram_cap,gb_approx,tok_s,vram_pct,ram_pct,nvme_pct" | Out-File $f -Encoding utf8
  foreach ($ram in 256,512,1024,2048,3072) {
    $gb = [math]::Round($ram*1.95/1024,1)
    $m = Run-Engine -Ngen 48 -Mode "cache" -K 24 -Ram $ram
    "$ram,$gb,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct)" | Out-File $f -Append -Encoding utf8
    Write-Host "[$Tag] ram=$ram (~$gb GB) : $($m.toks) tok/s, RAM hit $($m.rpct)%, NVMe $($m.npct)%"
  }
}

function Sweep-Storage {
  # measured only: throttle DOWN from the real ~2 GB/s drive (faster-NVMe is unmeasurable on this hardware)
  $f = "$res\storage_sweep.csv"; "throttle_mbps,tok_s,nvme_pct,disk_gbps" | Out-File $f -Encoding utf8
  foreach ($mbps in 100,300,600,1200) {
    $m = Run-Engine -Ngen 32 -Mode "cache" -K 24 -Ram 1024 -ThrottleMbps $mbps
    "$mbps,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
    Write-Host "[$Tag] throttle=$mbps : $($m.toks) tok/s"
  }
  $m = Run-Engine -Ngen 32 -Mode "cache" -K 24 -Ram 1024
  "0,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
  Write-Host "[$Tag] unthrottled (real NVMe $($m.gbps) GB/s) : $($m.toks) tok/s"
}

function Sweep-WL {
  Run-Engine -Ngen 200 -Mode "cache" -K 24 -Ram 2048 -Csv "$res\wl_general.csv" | Out-Null
  Write-Host "[$Tag] wl_general written"
  if ($DomainIds -ne "") {
    $did = (Get-Content (Resolve-P $DomainIds)).Trim()
    Run-Engine -Ngen 200 -Mode "cache" -K 24 -Ram 2048 -Csv "$res\wl_domain.csv" -IdsArg $did | Out-Null
    Write-Host "[$Tag] wl_domain written"
  }
}

function Sweep-Ctx {
  $g = [math]::Max(1, $CtxTokens - ($ids -split ',').Count)
  Write-Host "[$Tag] ctx run: ngen=$g (target ctx ~$CtxTokens, NOEOS) -- the slow one"
  $env:NOEOS = "1"   # keep generating to ngen even past EOS (we're measuring decode-vs-context, not quality)
  Run-Engine -Ngen $g -Mode "cache" -K 32 -Ram 2048 -Csv "$res\ctx_long.csv" | Out-Null
  Remove-Item Env:\NOEOS -ErrorAction SilentlyContinue
  Write-Host "[$Tag] ctx_long written"
}

switch ($Which) {
  "fast" { Sweep-K; Sweep-Ram; Sweep-Storage }
  "wl"   { Sweep-WL }
  "ctx"  { Sweep-Ctx }
  "all"  { Sweep-K; Sweep-Ram; Sweep-Storage; Sweep-WL; Sweep-Ctx }
}
Write-Host "[$Tag] done: $Which"
