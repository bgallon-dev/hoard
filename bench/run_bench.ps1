# Benchmark driver for the streaming-MoE engine (OLMoE-1B-7B). Produces reproducible CSVs in
# bench/results/ for the eight "legibility" graphs. Each sweep is selectable via -Which.
# Run from project root:  powershell -File bench\run_bench.ps1 -Which all
param([string]$Which = "all")

$root  = Split-Path -Parent $PSScriptRoot
$exe   = "$root\build\run_moe_stream.exe"
$model = "$root\models\olmoe-1b-7b-0924-instruct-q4_k_m.gguf"
$res   = "$root\bench\results"
New-Item -ItemType Directory -Force $res | Out-Null

# fixed prompt ids (BOS 50279 + the instruct-templated CPU-pipeline question), shared by all timing sweeps
$ids = "50279," + (Get-Content "$root\bench\prompt_ids.txt").Trim()

# --- run the engine once; return parsed metrics from stdout ---
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
    if ($out -match 'access split \((\d+) requests\)')        { $m.reqs = [int]$Matches[1] }
    return $m
}

function Sweep-K {
    $f = "$res\k_sweep.csv"; "K,tok_s,vram_pct,ram_pct,nvme_pct,peak_vram_mb" | Out-File $f -Encoding utf8
    # naive baseline (8 slots, recopy every token) for the speedup reference
    $m = Run-Engine -Ngen 128 -Mode "naive" -K 8 -Ram 1024
    "0,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
    Write-Host "naive: $($m.toks) tok/s"
    foreach ($K in 8,12,16,24,32,48,64) {
        $m = Run-Engine -Ngen 128 -Mode "cache" -K $K -Ram 1024
        "$K,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct),$($m.vram)" | Out-File $f -Append -Encoding utf8
        Write-Host "K=$K : $($m.toks) tok/s, VRAM hit $($m.vpct)%, peak $($m.vram) MB"
    }
}

function Sweep-Vram {
    $f = "$res\vram_vs_ctx.csv"; "ngen,ctx,peak_vram_mb,tok_s" | Out-File $f -Encoding utf8
    $plen = ($ids -split ',').Count
    foreach ($ngen in 16,32,64,128,256,512) {
        $m = Run-Engine -Ngen $ngen -Mode "cache" -K 32 -Ram 1024
        $ctx = $plen + $ngen
        "$ngen,$ctx,$($m.vram),$($m.toks)" | Out-File $f -Append -Encoding utf8
        Write-Host "ngen=$ngen ctx=$ctx : peak $($m.vram) MB"
    }
}

function Sweep-Ram {
    # RAM tier value: hold K (VRAM) fixed small, grow the RAM tier. Under a fixed NVMe-class throttle
    # so the access-split shift shows up in tok/s; the split itself is exact regardless of throttle.
    $f = "$res\ram_sweep.csv"; "ram_cap,tok_s,vram_pct,ram_pct,nvme_pct" | Out-File $f -Encoding utf8
    foreach ($ram in 64,128,256,512,768,1024) {
        $m = Run-Engine -Ngen 96 -Mode "cache" -K 16 -Ram $ram -ThrottleMbps 2000
        "$ram,$($m.toks),$($m.vpct),$($m.rpct),$($m.npct)" | Out-File $f -Append -Encoding utf8
        Write-Host "ram=$ram : $($m.toks) tok/s, RAM hit $($m.rpct)%, NVMe $($m.npct)%"
    }
}

function Sweep-Storage {
    # storage sensitivity: fixed cache config, sweep simulated drive bandwidth (HDD..NVMe..RAM).
    $f = "$res\storage_sweep.csv"; "throttle_mbps,tok_s,nvme_pct,disk_gbps" | Out-File $f -Encoding utf8
    foreach ($mbps in 100,300,600,1200,2400) {
        $m = Run-Engine -Ngen 48 -Mode "cache" -K 16 -Ram 256 -ThrottleMbps $mbps
        "$mbps,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
        Write-Host "throttle=$mbps MB/s : $($m.toks) tok/s, NVMe $($m.npct)%"
    }
    # unthrottled (page-cache / RAM-speed) reference point, recorded as 0 = "no limit"
    $m = Run-Engine -Ngen 48 -Mode "cache" -K 16 -Ram 256
    "0,$($m.toks),$($m.npct),$($m.gbps)" | Out-File $f -Append -Encoding utf8
    Write-Host "unthrottled : $($m.toks) tok/s"
}

function Sweep-TS {
    # time-series for hit-rate-over-time, latency percentiles, expert reuse. Two configs:
    #   small  = realistic too-big ratio (K=16, RAM 256) -> hierarchy visibly working
    #   large  = well-provisioned (K=48, RAM 1024)       -> warms to a high plateau
    Run-Engine -Ngen 200 -Mode "cache" -K 16 -Ram 256  -Csv "$res\ts_small.csv" | Out-Null
    Write-Host "ts_small written"
    Run-Engine -Ngen 200 -Mode "cache" -K 48 -Ram 1024 -Csv "$res\ts_large.csv" | Out-Null
    Write-Host "ts_large written"
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
