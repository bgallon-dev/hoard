# Map the VRAM cliff: for each K, grow MAXKV (target context) until the engine OOMs at allocation.
# DRYRUN reports static VRAM (model + KV + slot pool + GDN state) and exits before generation;
# an OOM at the KV/slot alloc crashes with no DRYRUN line. Early-stop per K on first OOM.
#   powershell -File bench\cliff_sweep.ps1
$root = Split-Path -Parent $PSScriptRoot
$exe  = "$root\build\run_qwen35.exe"
$models = @(
  @{ tag="q80"; path="$root\models\qwen3next-80b-a3b-q4_k_m.gguf"; ids="$root\bench\general_80b_ids.txt" },
  @{ tag="q35"; path="$root\models\qwen3.5-35b-a3b-q4_k_m.gguf";  ids="$root\bench\prompt_qwen_ids.txt" }
)
$Ks      = 16,24,32,48,64
$MaxKVs  = 4096,8192,16384,32768,49152,65536,81920,98304,131072
foreach ($m in $models) {
  $ids = (Get-Content $m.ids).Trim()
  $out = "$root\bench\results\$($m.tag)\ctx_cliff.csv"
  "K,maxkv,static_vram_mb" | Out-File $out -Encoding utf8
  foreach ($K in $Ks) {
    foreach ($MK in $MaxKVs) {
      $env:DRYRUN="1"; $env:MAXKV="$MK"; $env:CSV=""; $env:THROTTLE_MBPS=""
      $o = & $exe $m.path $ids 4 cache $K 2048 2>$null | Out-String
      $env:DRYRUN=""; $env:MAXKV=""
      if ($o -match 'static_vram_mb=([\d.]+)') {
        $v = $Matches[1]; "$K,$MK,$v" | Out-File $out -Append -Encoding utf8
        Write-Host "[$($m.tag)] K=$K maxkv=$MK -> $v MB"
      } else {
        "$K,$MK,OOM" | Out-File $out -Append -Encoding utf8
        Write-Host "[$($m.tag)] K=$K maxkv=$MK -> OOM (cliff for K=$K is between prev and $MK)"
        break   # larger contexts also OOM at this K
      }
    }
  }
}
Write-Host "CLIFF SWEEP DONE"
