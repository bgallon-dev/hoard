"""
spill.py -- characterize the WDDM shared-RAM-over-PCIe spill as a first-class 4th cache tier,
and fold it into the feasibility law (notes/FEASIBILITY.md, bench/feasibility.py).

The 3-tier model is  VRAM-slot -> RAM-tier -> NVMe.  On Windows/WDDM the hardware has a FOURTH
tier the engine cannot see: when the slot pool (or KV cache) overflows dedicated VRAM, WDDM
transparently backs the overflow with SYSTEM RAM over PCIe. A nominal "VRAM hit" on a spilled
page is really a PCIe read (~16 GB/s) -- still ~8x faster than the NVMe (~2 GB/s) it replaces.
So the 8 GB limit is a BANDWIDTH TAX, not a CAPACITY WALL.

This script:
  1. fits the static-VRAM plane from the DRYRUN/MAXKV probe (ctx_cliff) -> the EXACT spill frontier
  2. shows spill is graceful on both axes (K: k_spill; context: ctx_spill vs the ctx_long control)
  3. extracts an effective spill bandwidth B_spill and folds a 4th term into the feasibility law
  4. reframes procurement: VRAM vs PCIe vs system-RAM.
Reproduce: py bench/spill.py   (reads committed CSVs; writes bench/figs/spill.png)
"""
import csv, os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VRAM_CARD = 8192.0     # MB dedicated VRAM on the RX 6600
B_PCIE = 16.0          # GB/s PCIe4 x8 nominal

def csvrows(*p):
    return list(csv.DictReader(open(os.path.join(ROOT, *p), encoding="utf-8-sig")))

# ---------------------------------------------------------------------------------------------
# 1. STATIC-VRAM PLANE + FRONTIER  (ctx_cliff: K, maxkv, static_vram_mb)
# ---------------------------------------------------------------------------------------------
def fit_plane(tag):
    r = csvrows("bench", "results", tag, "ctx_cliff.csv")
    K   = np.array([float(x["K"]) for x in r])
    kv  = np.array([float(x["maxkv"]) for x in r])
    v   = np.array([float(x["static_vram_mb"]) for x in r])
    # static_vram = base + sK*K + sctx*maxkv   (linear plane)
    A = np.vstack([np.ones_like(K), K, kv]).T
    (base, sK, sctx), *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - (base + sK*K + sctx*kv)
    return base, sK, sctx, np.max(np.abs(resid))

print("="*80)
print("1. STATIC-VRAM PLANE  static_vram = base + s_K*K + s_ctx*ctx   (frontier = where it hits 8 GB)")
print("="*80)
planes = {}
for name, tag in (("Qwen3-Next-80B","q80"), ("Qwen3.5-35B","q35")):
    base, sK, sctx, mxr = fit_plane(tag)
    planes[tag] = (base, sK, sctx)
    print(f"\n  {name}:  base={base:.0f} MB   s_K={sK:.1f} MB/slot   s_ctx={sctx*1000:.1f} KB/ctx-tok"
          f"   (max resid {mxr:.0f} MB)")
    print(f"    spill frontier ctx*(K) where model+slots+KV crosses 8 GB:")
    for K in (16, 32, 48, 64):
        ctx_star = (VRAM_CARD - base - sK*K) / sctx
        print(f"      K={K:>2}: slots={base+sK*K:6.0f} MB resident -> KV headroom to 8GB = {ctx_star:7.0f} ctx tokens")

# ---------------------------------------------------------------------------------------------
# 2. SPILL IS GRACEFUL  (K axis: k_spill ; context axis: ctx_spill vs non-spill control ctx_long)
# ---------------------------------------------------------------------------------------------
print("\n" + "="*80)
print("2a. K-AXIS SPILL (k_spill): grow the slot pool PAST 8 GB -> tok/s keeps RISING")
print("="*80)
ks = csvrows("bench","results","q80","k_spill.csv")
print(f"  {'K':>4} {'peak_VRAM':>10} {'spilled':>8} {'tok/s':>7}")
for x in ks:
    pv = float(x["peak_vram_mb"]); sp = max(0.0, pv-VRAM_CARD)
    print(f"  {float(x['K']):>4.0f} {pv:>9.0f}MB {sp:>7.0f}MB {float(x['tok_s']):>7.2f}")

def binned(tag, fname, plen, nbins=12):
    """per-token trace -> per-context-band: mean dt_ms and steady NVMe-miss fraction."""
    r = csvrows("bench","results",tag,fname)
    tok  = np.array([float(x["tok"]) for x in r])
    dt   = np.array([float(x["dt_ms"]) for x in r])
    nv   = np.array([float(x["nvme_fall"]) for x in r])
    rq   = np.array([float(x["reqs"]) for x in r])
    ctx  = tok + plen
    # per-token deltas for miss fraction
    dnv = np.diff(nv); drq = np.diff(rq); miss = dnv/drq
    edges = np.linspace(ctx.min(), ctx.max(), nbins+1)
    out = []
    for i in range(nbins):
        m = (ctx[1:] >= edges[i]) & (ctx[1:] < edges[i+1])
        if m.sum() < 3: continue
        out.append((edges[i], edges[i+1], np.median(dt[1:][m]), np.mean(miss[m])))
    return out

print("\n" + "="*80)
print("2b. CONTEXT-AXIS SPILL: ctx_spill K=64 (spills ~12K) vs ctx_long K=32 (no spill to 16K)")
print("="*80)
print("    [median decode ms/token and steady NVMe-miss% by 2K context band]")
for tag, fname, K in (("q80","ctx_spill.csv",64), ("q80","ctx_long.csv",32)):
    b = binned(tag, fname, 38, nbins=12)
    print(f"\n  {fname} (K={K}):")
    print(f"    {'ctx band':>14} {'dt ms':>7} {'tok/s':>7} {'nvme%':>6}")
    for lo,hi,dt,miss in b:
        print(f"    {lo:>6.0f}-{hi:<6.0f} {dt:>7.1f} {1000/dt:>7.2f} {miss*100:>6.1f}")

# ---------------------------------------------------------------------------------------------
# 3. EXTRACT B_spill and fold into the law
#    Re-anchor the feasibility law to the k_spill session on its two NON-spilled points (K=48,64),
#    extrapolate the NVMe-miss curve past K=64, predict K=72/80/96 WITHOUT a spill penalty,
#    and read the shortfall as the spilled-slot PCIe tax.
# ---------------------------------------------------------------------------------------------
print("\n" + "="*80)
print("3. B_spill: predict spill points with NO penalty -> shortfall measures the tax (it is ~0)")
print("="*80)
# 80B physical inputs (from feasibility.py)
A = 10*48*1.91     # MB/tok all-access expert volume
Bn = 2.11          # GB/s NVMe (k_sweep session); k_spill ran a touch faster -> absorbed by recalibration
# fit nvme-miss(K) = floor + (n0-floor)*exp(-K/tau) from k_sweep K>=16
kk = csvrows("bench","results","q80","k_sweep.csv")
Kk = np.array([float(x["K"]) for x in kk]); nvk = np.array([float(x["nvme_pct"]) for x in kk])/100
vrk = np.array([float(x["vram_pct"]) for x in kk])/100
m = Kk >= 16
# simple log-space-ish fit by least squares on a fixed floor grid (robust for 5 pts)
best = None
for floor in np.linspace(0.10, 0.27, 35):
    for tau in np.linspace(20, 200, 60):
        pred = floor + (nvk[m][0]-floor)*np.exp(-(Kk[m]-16)/tau)
        e = np.sum((pred-nvk[m])**2)
        if best is None or e < best[0]: best = (e, floor, tau)
_, floor, tau = best
nvme_of = lambda K: floor + (nvk[m][0]-floor)*np.exp(-(K-16)/tau)
vram_of = lambda K: np.interp(K, Kk, vrk)     # vram-hit, linearly interp/extrapolate
print(f"  fitted NVMe-miss(K) = {floor:.3f} + {nvk[m][0]-floor:.3f}*exp(-(K-16)/{tau:.0f})   (floor={floor*100:.1f}%)")

# recalibrate t_fixed to the k_spill session using its two non-spilled points (K=48,64)
ksd = {float(x["K"]): float(x["tok_s"]) for x in ks}
ksd_pv = {float(x["K"]): float(x["peak_vram_mb"]) for x in ks}
def law3(K, tf, h2d_bw=B_PCIE):
    return tf + A*nvme_of(K)/Bn + A*(1-vram_of(K))/h2d_bw
tf_cal = np.mean([1000/ksd[K] - A*nvme_of(K)/Bn - A*(1-vram_of(K))/B_PCIE for K in (48,64)])
print(f"  recalibrated t_fixed (k_spill session, from K=48/64 no-spill) = {tf_cal:.1f} ms")
print(f"\n  {'K':>4} {'spilled':>8} {'meas tok/s':>10} {'3-tier pred':>11} {'shortfall ms':>12}")
spill_pts = []
for x in ks:
    K = float(x["K"]); pv=float(x["peak_vram_mb"]); sp=max(0.0,pv-VRAM_CARD); meas=float(x["tok_s"])
    t3 = law3(K, tf_cal); short = 1000/meas - t3
    print(f"  {K:>4.0f} {sp:>7.0f}MB {meas:>10.2f} {1000/t3:>11.2f} {short:>+11.1f}")
    if sp > 50: spill_pts.append((K, sp, short))
# What B_spill would the naive (uniformly-hot slots) model predict vs what we measure?
print("\n  the spill tax is BELOW this data's ~2% (+/-5 ms) noise floor. Why -- the WDDM-LRU mechanism:")
K96 = 96.0; sp96 = max(0.0, ksd_pv[K96]-VRAM_CARD)
slot_pool96 = planes['q80'][1]*K96
cap_overflow = sp96/slot_pool96                       # fraction of slot pool that lives in shared RAM
h96 = float(vram_of(K96))                              # slot-hit (VRAM) fraction at K=96
naive_tax = A*h96*cap_overflow/B_PCIE                  # ms, IF spilled pages were as hot as resident ones
print(f"    at K=96: capacity overflow = {cap_overflow*100:.0f}% of the slot pool ({sp96:.0f} MB in shared RAM)")
print(f"    naive 'uniformly-hot' model (B_spill=PCIe {B_PCIE:.0f} GB/s) would predict +{naive_tax:.1f} ms/tok tax")
print(f"    measured shortfall ~0 ms in AGGREGATE decode -- but that null was MISLEADING:")
print(f"  *** SUPERSEDED by the direct graphB-timing sweep (bench/spill_graphb.py): the spill tax IS real")
print(f"      (~+30 ms/tok at K=96, B_spill ~8 GB/s ~= PCIe). It cancels in aggregate decode because the")
print(f"      SAME extra slots cut NVMe-miss time by ~30 ms (disk 82->52 ms). 'Aggregate null' != 'no tax'.")
print(f"      Spill is graceful because each spilled slot trades a 2 GB/s NVMe read for an ~8 GB/s PCIe read")
print(f"      (~4x win), NOT because it is free. Run spill_graphb.py for the measured story.")

# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
# (a) k-axis spill
Kx = np.array([float(x["K"]) for x in ks]); ty=np.array([float(x["tok_s"]) for x in ks])
pv = np.array([float(x["peak_vram_mb"]) for x in ks])
ax[0].plot(Kx, ty, "o-", color="tab:red")
for xi,yi,p in zip(Kx,ty,pv):
    ax[0].annotate(f"{p/1024:.1f}GB", (xi,yi), fontsize=7, xytext=(0,6), textcoords="offset points", ha="center")
ax[0].axvspan(Kx[pv>VRAM_CARD].min() if (pv>VRAM_CARD).any() else Kx.max(), Kx.max(), color="tab:orange", alpha=.12)
ax[0].set(xlabel="K (VRAM expert slots)", ylabel="tok/s", title="K-axis: slot pool spills past 8 GB\n(orange = spilled, tok/s still rises)")
ax[0].grid(alpha=.3)
# (b) context-axis: spill vs control
for tag,fn,K,c in (("q80","ctx_spill.csv",64,"tab:red"),("q80","ctx_long.csv",32,"tab:blue")):
    b = binned(tag, fn, 38, 12); mid=[(lo+hi)/2 for lo,hi,_,_ in b]; ts=[1000/dt for _,_,dt,_ in b]
    ax[1].plot(mid, ts, "o-", color=c, label=f"{fn} K={K}")
base,sK,sctx = planes['q80']; ctx_star=(VRAM_CARD-base-sK*64)/sctx
ax[1].axvline(ctx_star, color="tab:orange", ls="--", label=f"K=64 spill onset ~{ctx_star/1000:.0f}K")
ax[1].set(xlabel="context (tokens)", ylabel="tok/s", title="Context-axis: smooth decay across spill\n(no discontinuity at the frontier)")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
# (c) the frontier map
base,sK,sctx = planes['q80']
Kg = np.linspace(8,96,50); ctxg=(VRAM_CARD-base-sK*Kg)/sctx
ax[2].plot(Kg, np.clip(ctxg,0,None)/1000, "-", color="tab:green")
ax[2].fill_between(Kg, 0, np.clip(ctxg,0,None)/1000, color="tab:green", alpha=.12)
ax[2].set(xlabel="K (VRAM expert slots)", ylabel="context before spill (K tokens)",
          title="80B spill frontier (green = fits 8 GB)\nbeyond = shared-RAM tier, graceful")
ax[2].grid(alpha=.3)
fig.tight_layout(); out=os.path.join(ROOT,"bench","figs","spill.png"); fig.savefig(out, dpi=120)
print(f"\nfigure -> {out}")
