"""
spill_graphb.py -- the targeted measurement that converts the spill NULL into a measured tax.

§6 of FEASIBILITY.md originally inferred "spill is ~free" from an aggregate-decode NULL (the tax sat
under the ~2% end-to-end noise). That inference was WRONG, and a direct measurement shows why: the
per-token graphB (expert-FFN) timer -- the phase that actually reads expert operands out of the VRAM
slots -- is isolated by `tB` in the engine (added to the DECODE BREAKDOWN as absolute ms/tok).

Sweeping K on the 80B (bench/results/q80/spill_graphb.csv):
  graphB is FLAT at ~34.4 ms for K=16..48 (peak 3.2..6.1 GB, no spill), then JUMPS at K=64
  (peak 7.5 GB) and climbs to 64 ms at K=96 (2.3 GB nominal spill). A sharp THRESHOLD at the
  effective VRAM limit (~7 GB; the 8 GB card minus WDDM/driver reserve), i.e. the spill -- a real,
  PCIe-rate tax, NOT the below-noise null the aggregate suggested.

Why the aggregate hid it: the SAME extra slots cut NVMe-miss time by a near-equal amount, so the two
~30 ms swings cancel and tok/s stays flat/rising. So spill is "graceful" not because it is free, but
because each spilled slot trades a ~2 GB/s NVMe read for a ~9 GB/s PCIe read -- still a ~4x win.
Reproduce: py bench/spill_graphb.py
"""
import csv, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "bench/results/q80/spill_graphb.csv"), encoding="utf-8-sig")))
K     = np.array([float(r["K"]) for r in rows])
peak  = np.array([float(r["peak_vram_mb"]) for r in rows])
gB    = np.array([float(r["graphB_ms"]) for r in rows])
disk  = np.array([float(r["disk_ms"]) if r["disk_ms"] else np.nan for r in rows])

base = gB[peak < 6500].mean()        # graphB with no spill (K=16..48, dead flat)
V_eff = 7000.0                        # effective VRAM limit: graphB is flat to peak 6.1GB, jumps by 7.5GB
print(f"graphB no-spill baseline (K<=48, peak<6.5GB) = {base:.1f} ms/tok  (flat: {gB[peak<6500]})")
print(f"effective VRAM limit ~{V_eff/1024:.1f} GB (8 GB card minus WDDM/driver reserve)\n")
print(f"  {'K':>4} {'peak GB':>8} {'spill GB':>9} {'graphB ms':>10} {'tax ms':>7} {'B_spill GB/s':>13}")

# law inputs for the byte accounting (A and h_vram from feasibility / sysmap)
A = 917.0                             # MB/tok expert volume
h_vram = {64: 0.73, 72: 0.75, 80: 0.76, 96: 0.78}   # VRAM-hit fraction (sysmap, extrapolated past 64)
bspill = []
for i in range(len(K)):
    sp_gb = max(0.0, peak[i]-V_eff)/1024
    tax = gB[i] - base
    if peak[i] > V_eff and K[i] in h_vram:
        slot_pool = 91.4*K[i]                       # MB (s_K from the static plane)
        frac = (peak[i]-V_eff)/slot_pool            # spilled fraction of the slot pool
        bytes_pcie = A*h_vram[K[i]]*frac            # expert bytes graphB reads over PCIe per token
        bs = bytes_pcie/tax if tax > 0 else float('nan')   # MB/ms = GB/s
        bspill.append(bs); bsstr = f"{bs:>10.1f}"
    else:
        bsstr = f"{'(no spill)':>13}"
    print(f"  {K[i]:>4.0f} {peak[i]/1024:>8.1f} {sp_gb:>9.1f} {gB[i]:>10.1f} {tax:>+7.1f} {bsstr}")

print(f"\n  => B_spill = {np.nanmean(bspill):.1f} GB/s (range {np.nanmin(bspill):.1f}-{np.nanmax(bspill):.1f}); "
      f"consistent with realized PCIe4 x8, ~{np.nanmean(bspill)/2.1:.0f}x the 2.1 GB/s NVMe it replaces.")

# the compensation that hid the tax in aggregate decode
m = ~np.isnan(disk)
print("\nwhy the aggregate looked free -- graphB tax is paid back by disk savings (K=48->96):")
i48 = int(np.where(K == 48)[0][0]); i96 = int(np.where(K == 96)[0][0])
print(f"  graphB: {gB[i48]:.0f} -> {gB[i96]:.0f} ms  ({gB[i96]-gB[i48]:+.0f} ms, the spill tax)")
print(f"  disk  : {disk[i48]:.0f} -> {disk[i96]:.0f} ms  ({disk[i96]-disk[i48]:+.0f} ms, more slots = fewer NVMe misses)")
print(f"  net   : {(gB[i96]-gB[i48])+(disk[i96]-disk[i48]):+.0f} ms -> tok/s ~flat ({rows[i48]['tok_s']} -> {rows[i96]['tok_s']})")

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(peak/1024, gB, "o-", color="tab:red", label="graphB ms/tok")
ax[0].axhline(base, ls=":", color="gray", label=f"no-spill baseline {base:.0f} ms")
ax[0].axvline(V_eff/1024, ls="--", color="tab:orange", label=f"~{V_eff/1024:.0f} GB effective VRAM limit")
ax[0].set(xlabel="peak VRAM (GB)", ylabel="graphB ms/token",
          title="graphB tax IS measurable: flat, then PCIe-rate\nrise once slots spill past the VRAM limit")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
mm = ~np.isnan(disk)
ax[1].plot(K[mm], gB[mm], "o-", color="tab:red", label="graphB (spill tax)")
ax[1].plot(K[mm], disk[mm], "o-", color="tab:blue", label="disk (NVMe miss)")
ax[1].plot(K[mm], gB[mm]+disk[mm], "o--", color="tab:green", label="graphB+disk (~flat)")
ax[1].set(xlabel="K (VRAM expert slots)", ylabel="ms/token",
          title="why aggregate looked free:\nspill tax ~cancels disk savings")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
fig.tight_layout(); out = os.path.join(ROOT, "bench/figs/spill_graphb.png"); fig.savefig(out, dpi=120)
print(f"\nfigure -> {out}")
