# Benchmark driver for the streaming-MoE engine on QWEN3.5-35B-A3B (run_qwen35.exe).
# The 21 GB model > 16 GB RAM, so experts genuinely stream from NVMe (no page-cache shortcut):
# every metric here is the real too-big-for-RAM regime. Writes CSVs to bench/results/qwen/.
# Run from project root:  powershell -File bench\run_bench_qwen.ps1 -Which all
param([string]$Which = "all")

$root  = Split-Path -Parent $PSScriptRoot
$exe   = "$root\build\run_qwen35.exe"
$model = "$root\models\qwen3.5-35b-a3b-q4_k_m.gguf"
$res   = "$root\bench\results\qwen"
New-Item -ItemType Directory -Force $res | Out-Null

# fixed chat-templated prompt ids (38 tok), shared by all timing sweeps
$ids = (Get-Content "$root\bench\prompt_qwen_ids.txt").Trim()

function Run-Engine {
    param([int]$Ngen,[string]$Mode,[int]$K,[int]$Ram,[int]$ThrottleMbps=0,[string]$Csv="")
    if ($ThrottleMbps -gt 0) { $env:THROTTLE_MBPS = "$ThrottleMbps" } else { Remove-Item Env:\THROTTLE_MBPS -ErrorAction SilentlyContinue }
    if ($Csv -ne "") { $env:CSV = $Csv } else { Remove-Item Env:\CSV -ErrorAction SilentlyContinue }
    $out = & $exe $model $ids $Ngen $Mode $K $Ram 2>$null | Out-String
    Remove-Item Env:\THROTTLE_MBPS -ErrorAction SilentlyContinue
    Remove-Item Env:\CSV -ErrorAction SilentlyContinue
    $m = @{}
    if ($out -match 'tok/s=([\d.]+)')                         { $m.toks = [double]$Matches[1] }
    if ($out -match 'peak_VRAM=([\d.]+) MB')                  { $m.vram = [double]$Matches[1] }
    if ($out -match 'VRAM=([\d.]+)%\s+RAM=([\d.]+)%\s+NVMe=([\d.]+)%') { $m.vpct=[double]$Matches[1]; $m.rpct=[double]$Matches[2]; $m.npct=[double]$Matches[3] }
    if ($out -match 'NVMe:.*= ([\d.]+) GB/s')                 { $m.gbps = [double]$Matches[1] } else { $m.gbps = 0 }
    return $m
}

function Sweep-K {
    $f = "$res\k_sweep.csv"; "K,tok_s,vram_pct,ram_pct,nvme_pct,peak_vram_mb" | Out-File $f -Encoding utf8
    $m = Run-Engine -Ngen 64 -Mode "naive" -K 8 -Ram 2048
    "0,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
    Write-Host "naive: $($m.toks) tok/s, NVMe $($m.npct)%"
    foreach ($K in 16,24,32,40,48) {
        $m = Run-Engine -Ngen 64 -Mode "cache" -K $K -Ram 2048
        "$K,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
        Write-Host "K=$K : $($m.toks) tok/s, VRAM hit $($m.vpct)%, NVMe $($m.npct)%, peak $($m.vram) MB"
    }
}

function Sweep-Vram {
    $f = "$res\vram_vs_ctx.csv"; "ngen,ctx,peak_vram_mb,tok_s" | Out-File $f -Encoding utf8
    $plen = ($ids -split ',').Count
    foreach ($ngen in 16,64,128,256,512) {
        $m = Run-Engine -Ngen $ngen -Mode "cache" -K 32 -Ram 2048
        $ctx = $plen + $ngen
        "$ngen,$ctx,$($m.vram),$($m.toks)" | Out-File $f -Append -Encoding utf8
        Write-Host "ngen=$ngen ctx=$ctx : peak $($m.vram) MB"
    }
}

function Sweep-Ram {
    # RAM-tier value under REAL NVMe latency (model > RAM, so reads are real disk, not page cache)
    $f = "$res\ram_sweep.csv"; "ram_cap,tok_s,vram_pct,ram_pct,nvme_pct" | Out-File $f -Encoding utf8
    foreach ($ram in 256,512,1024,2048,3072) {
        $m = Run-Engine -Ngen 48 -Mode "cache" -K 24 -Ram $ram
        "$ram,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct)" | Out-File $f -Append -Encoding utf8
        Write-Host "ram=$ram : $($m.toks) tok/s, RAM hit $($m.rpct)%, NVMe $($m.npct)%"
    }
}

function Sweep-Storage {
    # storage sensitivity: sweep simulated drive bandwidth below the real NVMe; unthrottled = real ~2 GB/s drive
    $f = "$res\storage_sweep.csv"; "throttle_mbps,tok_s,nvme_pct,disk_gbps" | Out-File $f -Encoding utf8
    foreach ($mbps in 100,300,600,1200) {
        $m = Run-Engine -Ngen 32 -Mode "cache" -K 24 -Ram 1024 -ThrottleMbps $mbps
        "$mbps,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
        Write-Host "throttle=$mbps MB/s : $($m.toks) tok/s, NVMe $($m.npct)%"
    }
    $m = Run-Engine -Ngen 32 -Mode "cache" -K 24 -Ram 1024
    "0,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
    Write-Host "unthrottled (real NVMe $($m.gbps) GB/s) : $($m.toks) tok/s"
}

function Sweep-TS {
    Run-Engine -Ngen 200 -Mode "cache" -K 24 -Ram 2048 -Csv "$res\ts_main.csv"  | Out-Null
    Write-Host "ts_main written (K=24, RAM 2048)"
    Run-Engine -Ngen 200 -Mode "cache" -K 48 -Ram 3072 -Csv "$res\ts_large.csv" | Out-Null
    Write-Host "ts_large written (K=48, RAM 3072)"
}

switch ($Which) {
    "k"       { Sweep-K }
    "vram"    { Sweep-Vram }
    "ram"     { Sweep-Ram }
    "storage" { Sweep-Storage }
    "ts"      { Sweep-TS }
    "all"     { Sweep-K; Sweep-Vram; Sweep-Ram; Sweep-Storage; Sweep-TS }
}
Write-Host "done: $Which"
