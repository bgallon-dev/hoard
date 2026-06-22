#!/bin/bash
# Map the VRAM cliff: for each K, increase MAXKV (context) until the engine OOMs at allocation.
# DRYRUN reports static VRAM and exits; OOM = no DRYRUN line. Early-stop per K on first OOM.
exe="build/run_qwen35.exe"
probe() { # $1=model $2=K $3=maxkv  -> echoes "static_mb" or "OOM"
  DRYRUN=1 MAXKV=$3 CSV= THROTTLE_MBPS= "$exe" "$1" "$4" 4 cache "$2" 2048 2>/dev/null | grep -oP 'static_vram_mb=\K[0-9.]+' || echo OOM
}
run_model() { # $1=model $2=tag $3=ids
  out="bench/results/$2/ctx_cliff.csv"; echo "K,maxkv,static_vram_mb" > "$out"
  for K in 16 24 32 48 64; do
    for MK in 4096 8192 16384 32768 49152 65536 81920 98304 131072; do
      v=$(probe "$1" "$K" "$MK" "$3")
      echo "$K,$MK,$v" >> "$out"
      echo "[$2] K=$K maxkv=$MK -> $v"
      [ "$v" = "OOM" ] && break   # everything larger also OOMs
    done
  done
}
run_model "models/qwen3next-80b-a3b-q4_k_m.gguf" q80 "$(cat bench/general_80b_ids.txt)"
run_model "models/qwen3.5-35b-a3b-q4_k_m.gguf"  q35 "$(cat bench/prompt_qwen_ids.txt)"
echo "CLIFF SWEEP DONE"
