"""
feasibility.py  --  a predictive feasibility law for streaming-MoE-on-constrained-VRAM.

WHAT THIS IS
------------
Turns the pile of measured constants in this repo into a closed-form predictor that takes
a model's *a-priori* physical parameters plus the host's storage/PCIe bandwidths and outputs
(a) predicted decode tok/s and (b) a regime label {cache-effective | usable-slow | unusable-crawl}.

THE LAW
-------
Per-token decode latency decomposes into a fixed compute floor plus two bandwidth-bound
streaming terms:

    t_tok = t_fixed  +  A * m_nvme / B_nvme  +  A * (1 - h_vram) / B_pcie
    tok/s = 1 / t_tok

    A         = n_used * L * b_e         per-token "all-accesses" expert byte volume  (MB)
    m_nvme(K) = NVMe-miss fraction       (experts not caught by VRAM or RAM tiers)
    h_vram(K) = VRAM-hit fraction        (experts already resident -> no H2D restage)
    B_nvme    = sustained drive bandwidth (GB/s) ; B_pcie = H2D stage bandwidth (~16 GB/s PCIe4 x8)
    t_fixed   = compute + graph-launch + router + head, the Amdahl floor (the ONE fitted constant)

The cache-coverage fractions m_nvme(K), h_vram(K) are an INPUT here: measured cheaply (one K-sweep)
or produced by the validated offline cache simulator (98.8% per-access agreement, see notes/MEMORY-FLOW.md).
This law converts memory-flow -> wall-time; the simulator predicts the flow.

THE FEASIBILITY RATIO (the regime gate)
---------------------------------------
    F/B = A / B_vram          A = one token's expert footprint ; B_vram = VRAM expert-slot budget
  F/B < 1  : a single token's experts fit in VRAM -> cross-token caching is POSSIBLE; the steady-state
             miss m floors at the cold tail (working-set / B), giving cache-effective or usable-slow.
  F/B > 1  : one token's experts exceed the whole VRAM budget -> caching is STRUCTURALLY impossible,
             m is pinned at ~1, and tok/s collapses to 1/(t_fixed + A/B_nvme): unusable-crawl.

VALIDATION (see __main__):
  - in-sample : predicts each model's full K-sweep at ~2-3% MAPE with one fitted t_fixed.
  - OUT-OF-SAMPLE : coefficients calibrated on the 35B predict the *80B* tok/s curve to within ~9%,
                    using only the 80B's a-priori params + routing-miss curve (no 80B timing peeked).
  - cross-arch : the 30B (different arch, earlier engine build) -- structure holds, t_fixed/layer ~2x.
  - falsified regime : F/B=2.8 (Qwen3-235B) -> predicted ~0.1-0.3 tok/s, unusable-crawl.

HONEST BOUNDARY: validated on diffuse-routing A3B-class hybrids (Qwen3-Next / Qwen3.5) and one
qwen3moe transformer, on ONE storage profile (Samsung 970 EVO Plus, ~2.1 GB/s in-engine). It is NOT
validated for MLA attention (DeepSeek -- not implemented here) nor for concentrated-routing MoE.
"""
import csv, os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_PCIE = 16.0          # GB/s, PCIe4 x8 H2D stage (small correction term)
B_VRAM = 6.505 * 1024  # MB, the 8 GB card's usable expert-slot budget (calibrates 30B F/B=0.17, README)

def csvrows(*parts):
    return list(csv.DictReader(open(os.path.join(ROOT, *parts), encoding="utf-8-sig")))

def derive_be(tag):
    """bytes/expert (MB), empirical: steady-state d(disk_mb)/d(nvme_accesses) over tok 100->200."""
    r = csvrows("bench", "results", tag, "wl_general.csv")
    a, b = r[99], r[199]
    return (float(b["disk_mb"]) - float(a["disk_mb"])) / (float(b["nvme_fall"]) - float(a["nvme_fall"]))

def predict_tps(t_fixed, A, m_nvme, h_vram, B_nvme):
    """The law. t_fixed ms, A MB, fractions in [0,1], B_nvme GB/s -> tok/s. (MB/(GB/s)=ms.)"""
    t = t_fixed + A * m_nvme / B_nvme + A * (1 - h_vram) / B_PCIE
    return 1000.0 / t

# ---- the two clean same-engine CSV models -------------------------------------------------
CSV_MODELS = {
    "Qwen3.5-35B-A3B":   dict(tag="q35", n_used=8,  L=40, n_exp=256, B_nvme=2.08),
    "Qwen3-Next-80B-A3B":dict(tag="q80", n_used=10, L=48, n_exp=512, B_nvme=2.11),
}

def load_model(name, M):
    be = derive_be(M["tag"]); A = M["n_used"] * M["L"] * be
    ks = csvrows("bench", "results", M["tag"], "k_sweep.csv")
    d = dict(name=name, be=be, A=A, **M,
             K=np.array([float(x["K"]) for x in ks]),
             tps=np.array([float(x["tok_s"]) for x in ks]),
             vram=np.array([float(x["vram_pct"]) for x in ks]) / 100,
             nvme=np.array([float(x["nvme_pct"]) for x in ks]) / 100)
    # fit the single free constant t_fixed by averaging the residual over all K
    tfix = np.mean(1000.0 / d["tps"] - d["A"] * d["nvme"] / d["B_nvme"] - d["A"] * (1 - d["vram"]) / B_PCIE)
    d["t_fixed"] = tfix
    d["tps_pred"] = predict_tps(tfix, d["A"], d["nvme"], d["vram"], d["B_nvme"])
    d["mape"] = np.mean(np.abs(d["tps_pred"] - d["tps"]) / d["tps"]) * 100
    return d

def regime(tps, fb):
    if fb >= 1.0:        return "UNUSABLE-CRAWL (F/B>1: caching structurally impossible)"
    if tps >= 5.0:       return "CACHE-EFFECTIVE"
    if tps >= 1.0:       return "usable-slow"
    return "UNUSABLE-CRAWL"

if __name__ == "__main__":
    M35 = load_model("Qwen3.5-35B-A3B", CSV_MODELS["Qwen3.5-35B-A3B"])
    M80 = load_model("Qwen3-Next-80B-A3B", CSV_MODELS["Qwen3-Next-80B-A3B"])

    print("=" * 78)
    print("1. IN-SAMPLE: the law predicts each model's full K-sweep (one fitted t_fixed)")
    print("=" * 78)
    for d in (M35, M80):
        fb = d["A"] / B_VRAM
        print(f"\n  {d['name']}:  b_e={d['be']:.2f} MB  A={d['A']:.0f} MB/tok  F/B={fb:.3f}"
              f"  t_fixed={d['t_fixed']:.1f} ms ({d['t_fixed']/d['L']:.2f} ms/layer)  MAPE={d['mape']:.1f}%")
        print(f"    {'K':>4} {'meas':>7} {'pred':>7} {'err%':>6}")
        for i in range(len(d["K"])):
            e = (d["tps_pred"][i] - d["tps"][i]) / d["tps"][i] * 100
            print(f"    {d['K'][i]:>4.0f} {d['tps'][i]:>7.2f} {d['tps_pred'][i]:>7.2f} {e:>+6.1f}")

    print("\n" + "=" * 78)
    print("2. t_fixed-TRANSFER (calibrate floor+b_e on 35B; uses 80B's MEASURED miss curve)")
    print("   NOTE: this isolates the compute-floor transfer only. The FULL out-of-sample test --")
    print("   with the miss curve itself PREDICTED from a routing model -- is in bench/close_loop.py.")
    print("=" * 78)
    # transferable coefficients taken from the 35B alone:
    tfix_per_layer = M35["t_fixed"] / M35["L"]      # ms/layer compute floor
    be_xfer        = M35["be"]                       # MB/expert (same quant, same n_embd=2048)
    t_fixed_80_pred = tfix_per_layer * M80["L"]
    A_80_pred       = M80["n_used"] * M80["L"] * be_xfer
    tps_80_oos = predict_tps(t_fixed_80_pred, A_80_pred, M80["nvme"], M80["vram"], M80["B_nvme"])
    oos_err = (tps_80_oos - M80["tps"]) / M80["tps"] * 100
    print(f"  transfer from 35B:  t_fixed/layer={tfix_per_layer:.2f} ms  b_e={be_xfer:.2f} MB")
    print(f"  applied to 80B a-priori inputs: L=48, n_used=10  ->  t_fixed={t_fixed_80_pred:.1f} ms, A={A_80_pred:.0f} MB")
    print(f"    {'K':>4} {'meas':>7} {'pred_OOS':>9} {'err%':>6}")
    for i in range(len(M80["K"])):
        print(f"    {M80['K'][i]:>4.0f} {M80['tps'][i]:>7.2f} {tps_80_oos[i]:>9.2f} {oos_err[i]:>+6.1f}")
    print(f"  OUT-OF-SAMPLE MAPE = {np.mean(np.abs(oos_err)):.1f}%   (mean signed {np.mean(oos_err):+.1f}% -- "
          f"80B's larger router + 2 extra active experts/layer add ~{(M80['t_fixed']-t_fixed_80_pred)/M80['L']:.2f} ms/layer the 35B can't see)")

    print("\n" + "=" * 78)
    print("3. RAM-COLLAPSE: the law explains why 'more RAM tier' hardware-falsified (80B, K=48)")
    print("=" * 78)
    # notes/MEMORY-FLOW.md §7: the simulator said ram 2048->4096 cuts cold reads -33% (m 0.30->0.20),
    # but on a 16 GB box that memory pressure collapses NVMe bandwidth 3.11 -> 1.44 GB/s.
    # The simulator modelled only m (fixed BW) and predicted a WIN. The law couples m AND B_nvme:
    A80 = M80["A"]; tf80 = M80["t_fixed"]; h2d = A80 * (1 - 0.684) / B_PCIE
    for ram, m, bw in [("2048 (current)", 0.30, 3.11), ("4096 (sim says +33% fewer cold)", 0.20, 1.44)]:
        tps = 1000.0 / (tf80 + A80 * m / bw + h2d)
        print(f"  ram={ram:<34} m_nvme={m:.2f}  B_nvme={bw:.2f} GB/s  ->  {tps:.2f} tok/s")
    print("  verdict: law predicts MORE ram is net-NEGATIVE here (BW collapse > miss reduction) -- matches")
    print("           the hardware finding the m-only simulator missed.")

    print("\n" + "=" * 78)
    print("4. REGIME MAP via F/B  (B_vram = 6.5 GB expert budget on the 8 GB card)")
    print("=" * 78)
    # 30B: cross-arch/engine anchor from notes/STATUS.md prose (qwen3moe, earlier build, ram=512)
    A_30 = 8 * 48 * 2.88
    # 235B: from PUBLISHED specs (non-circular). Qwen3-235B-A22B: 94 L, 128 exp, top-8, n_embd 4096,
    # moe_ff 1536 -> expert = gate/up[4096,1536]x2 (q4_K) + down[1536,4096] (q6_K) ~= 12.2 MB.
    be_235 = (4096 * 1536 * 2 * 0.5625 + 1536 * 4096 * 0.8125) / 1024**2
    A_235  = 8 * 94 * be_235
    rows = [
        ("Qwen3-30B-A3B (qwen3moe)", A_30, 4.73, "measured K=48 ram=512, earlier engine"),
        (M35["name"],  M35["A"], M35["tps"].max(), "measured K=64"),
        (M80["name"],  M80["A"], M80["tps"].max(), "measured K=64"),
        ("Qwen3-235B-A22B (predicted)", A_235, None, f"never run; b_e~{be_235:.1f}MB, m pinned=1"),
    ]
    print(f"  {'model':<34} {'A MB/tok':>9} {'F/B':>6} {'tok/s':>7}  regime")
    for nm, A, tps, note in rows:
        fb = A / B_VRAM
        if tps is None:  # F/B>1: can't hold one token's experts -> m=1, no VRAM reuse
            tps = predict_tps(1.74 * 94, A, 1.0, 0.0, 2.11)   # t_fixed ~1.74 ms/layer x 94 layers
        print(f"  {nm:<34} {A:>9.0f} {fb:>6.2f} {tps:>7.2f}  {regime(tps, fb):<22}  ({note})")
    print(f"  [note] consistent-basis 235B F/B={A_235/B_VRAM:.2f} (>1, infeasible). Earlier notes quoted a")
    print(f"         rougher 2.8 from a coarser expert-size estimate -- same verdict (see notes/FEASIBILITY.md).")

    print("\n" + "=" * 78)
    print("5. INVERSE (the on-prem sizing question): drive bandwidth needed for a target tok/s")
    print("=" * 78)
    for d in (M35, M80):
        # at best cache (K=64), what B_nvme hits a target? invert: t_nvme = 1000/target - t_fixed - t_h2d
        m = d["nvme"][-1]; hv = d["vram"][-1]; A = d["A"]; tf = d["t_fixed"]
        amdahl = 1000.0 / tf       # absolute ceiling at infinite bandwidth (streaming -> 0)
        print(f"  {d['name']:<22} Amdahl ceiling (B_nvme=inf) = {amdahl:.1f} tok/s")
        for tgt in (8.0, 10.0):
            budget = 1000.0 / tgt - tf - A * (1 - hv) / B_PCIE
            if budget <= 0:
                ok = f"unreachable (above the {amdahl:.0f} tok/s Amdahl ceiling)"
            else:
                need = A * m / budget
                ok = f"{need:.1f} GB/s" + ("  (>> any real drive)" if need > 50 else "")
            print(f"      target {tgt:>4.0f} tok/s @ K=64 -> needs B_nvme = {ok}")

    # ---- figure -------------------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for d, c in ((M35, "tab:blue"), (M80, "tab:red")):
        ax[0].plot(d["K"], d["tps"], "o", color=c, label=f"{d['name']} measured")
        ax[0].plot(d["K"], d["tps_pred"], "-", color=c, alpha=.6, label=f"{d['name']} law (MAPE {d['mape']:.1f}%)")
    ax[0].set(xlabel="VRAM expert slots K", ylabel="decode tok/s",
              title="In-sample: law vs measured (one fitted t_fixed)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].plot(M80["K"], M80["tps"], "o", color="tab:red", label="80B measured")
    ax[1].plot(M80["K"], tps_80_oos, "--", color="tab:green",
               label=f"80B predicted from 35B coeffs (MAPE {np.mean(np.abs(oos_err)):.1f}%)")
    ax[1].set(xlabel="VRAM expert slots K", ylabel="decode tok/s",
              title="Out-of-sample: 80B predicted from 35B alone")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(ROOT, "bench", "figs", "feasibility.png")
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")
