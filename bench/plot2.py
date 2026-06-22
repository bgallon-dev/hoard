#!/usr/bin/env python3
# 35B (qwen35moe) vs 80B (qwen3next) comparison graphs for the streaming-MoE engine.
# Reads bench/results/q35/ and bench/results/q80/ (from run_bench2.ps1). Run: py bench/plot2.py
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "bench", "figs")
os.makedirs(FIGS, exist_ok=True)

# per-model constants
M = {
    "q35": dict(label="Qwen3.5-35B-A3B (40L, 256/8, 10240 exp, 21 GB)", color="C0",
                experts=256*40, mb=1.86),
    "q80": dict(label="Qwen3-Next-80B-A3B (48L, 512/10, 24576 exp, 45 GB)", color="C3",
                experts=512*48, mb=1.95),
}

def load(tag, name):
    p = os.path.join(ROOT, "bench", "results", tag, name)
    if not os.path.exists(p): return None
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows: return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def save(fig, name, sub):
    fig.suptitle(sub, y=0.005, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = os.path.join(FIGS, name); fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))

def gA():  # tok/s vs K + peak VRAM vs K (does more VRAM still help? where's the 8GB ceiling per model?)
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); ax2 = ax.twinx()
    title_bits = []
    for tag in ("q35", "q80"):
        d = load(tag, "k_sweep.csv")
        if d is None: continue
        naive = d["tok_s"][d["K"] == 0]
        m = (d["K"] > 0) & (d["tok_s"] > 0)   # drop OOM-failed (tok_s=0) points
        K, toks, vram = d["K"][m], d["tok_s"][m], d["peak_vram_mb"][m]
        c = M[tag]["color"]
        ax.plot(K, toks, "o-", color=c, lw=2, label=M[tag]["label"])
        ax2.plot(K, vram/1000, "s:", color=c, alpha=0.45)
        if len(naive): title_bits.append(f"{tag}: K-best {toks.max()/naive[0]:.2f}x naive")
        # mark the highest K that fit
        ax.annotate(f"K{int(K[-1])}", (K[-1], toks[-1]), textcoords="offset points", xytext=(4,0), fontsize=7, color=c)
    ax.axhline(0, color="none")
    ax2.axhline(8, ls=":", color="0.5"); ax2.annotate("8 GB VRAM limit", (16, 8), fontsize=7, va="bottom", color="0.4")
    ax.set_xlabel("K = VRAM expert slots per layer"); ax.set_ylabel("decode tok/s")
    ax2.set_ylabel("peak VRAM (GB), dotted")
    ax.set_title("A - Speed vs VRAM slots K (does the hierarchy scale upward?)\n" + " ; ".join(title_bits), fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    save(fig, "cmp_A_tok_vs_k.png", "solid = tok/s (left) ; dotted = peak VRAM (right)")

def gB():  # tok/s vs RAM tier (does a bigger RAM buffer reduce disk misses?)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag in ("q35", "q80"):
        d = load(tag, "ram_sweep.csv")
        if d is None: continue
        gb = d["ram_cap"] * M[tag]["mb"] / 1000
        mm = d["tok_s"] > 0
        ax.plot(gb[mm], d["tok_s"][mm], "o-", color=M[tag]["color"], lw=2, label=M[tag]["label"])
    ax.set_xlabel("RAM tier size (GB of warm experts)"); ax.set_ylabel("decode tok/s")
    ax.set_title("B - RAM tier value (15.8 GB phys RAM caps tier ~6 GB; 8/12 GB infeasible)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "cmp_B_tok_vs_ram.png", "bounded RAM tier under real NVMe streaming")

def gC():  # tok/s vs storage bandwidth (measured down-throttle only)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag in ("q35", "q80"):
        d = load(tag, "storage_sweep.csv")
        if d is None: continue
        thr, toks, gbps = d["throttle_mbps"], d["tok_s"], d["disk_gbps"]
        real = gbps[thr == 0]; real_mbps = real[0]*1000 if len(real) and real[0] > 0 else 2060.0
        x = np.where(thr == 0, real_mbps, thr); o = np.argsort(x)
        ax.plot(x[o]/1000, toks[o], "o-", color=M[tag]["color"], lw=2, label=M[tag]["label"])
    ax.set_xlabel("effective storage bandwidth (GB/s) - measured down to HDD-class")
    ax.set_ylabel("decode tok/s")
    ax.set_title("C - Storage sensitivity (drive maxes ~2 GB/s; faster unmeasurable here)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "cmp_C_tok_vs_storage.png", "throttle only goes DOWN from the real ~2 GB/s NVMe")

def _roll(a, w=64):
    if len(a) < w: return a
    return np.convolve(a, np.ones(w)/w, mode="same")

def gD():  # tok/s vs context length (the context/prefill tradeoff) from one long generation
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag in ("q35", "q80"):
        d = load(tag, "ctx_long.csv")
        if d is None: continue
        tok, dt = d["tok"], d["dt_ms"]
        mm = dt > 0
        tps = 1000.0/dt[mm]
        ax.plot(tok[mm], _roll(tps), "-", color=M[tag]["color"], lw=1.5, label=M[tag]["label"])
    ax.set_xlabel("context length (generated token index)"); ax.set_ylabel("decode tok/s (64-tok rolling)")
    ax.set_title("D - Speed vs context length to ~16K (KV grows only in full-attn layers)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "cmp_D_tok_vs_ctx.png", "hybrid: GDN layers are O(1) in context; only full-attn KV grows")

def gE():  # working set: domain (code) vs general, per model (is expert locality workload-specific?)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag in ("q35", "q80"):
        tot = M[tag]["experts"]
        for wl, ls in (("general", "-"), ("domain", "--")):
            d = load(tag, f"wl_{wl}.csv")
            if d is None: continue
            ax.plot(d["tok"], 100*d["distinct"]/tot, ls, color=M[tag]["color"], lw=1.8,
                    label=f"{tag} {wl}")
    ax.set_xlabel("generated token index"); ax.set_ylabel("distinct experts touched (% of model)")
    ax.set_title("E - Expert working set: domain (code, dashed) vs general (solid)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "cmp_E_workingset.png", "lower = more concentrated routing = cache works better")

def gF():  # steady-state tier split, domain vs general, per model
    groups = []
    for tag in ("q35", "q80"):
        for wl in ("general", "domain"):
            d = load(tag, f"wl_{wl}.csv")
            if d is None: continue
            # steady-state from last-half deltas
            h = len(d["tok"])//2
            dq = d["reqs"][-1]-d["reqs"][h]
            if dq <= 0: continue
            v = 100*(d["vram_hit"][-1]-d["vram_hit"][h])/dq
            r = 100*(d["ram_hit"][-1]-d["ram_hit"][h])/dq
            n = 100*(d["nvme_fall"][-1]-d["nvme_fall"][h])/dq
            groups.append((f"{tag}\n{wl}", v, r, n))
    if not groups: return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(groups)); labels=[g[0] for g in groups]
    v=[g[1] for g in groups]; r=[g[2] for g in groups]; n=[g[3] for g in groups]
    ax.bar(x, v, color="#2ca02c", label="VRAM hit")
    ax.bar(x, r, bottom=v, color="#ff7f0e", label="RAM hit")
    ax.bar(x, n, bottom=np.array(v)+np.array(r), color="#d62728", label="NVMe miss")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("steady-state fetch share (%)")
    ax.set_title("F - Cache tier split by workload (K=24, 2048-expert RAM tier)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "cmp_F_tiersplit.png", "domain vs general: does workload change the disk-miss floor?")

NFULL = {"q35": 10, "q80": 12}; NLAY = {"q35": 40, "q80": 48}

def gG():  # VRAM frontier: context where KV+slots cross 8 GB dedicated (spill to shared RAM begins) vs K
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); CEIL = 8192.0
    for tag in ("q35", "q80"):
        d = load(tag, "ctx_cliff.csv")
        if d is None: continue
        Ks = sorted(set(int(k) for k in d["K"])); fK = []; fctx = []
        for K in Ks:
            m = d["K"] == K; mk = d["maxkv"][m]; sv = d["static_vram_mb"][m]
            o = np.argsort(mk); mk, sv = mk[o], sv[o]
            if sv.max() < CEIL or sv.min() > CEIL: continue
            i = int(np.searchsorted(sv, CEIL))
            ctx = mk[i-1] + (CEIL - sv[i-1]) / (sv[i] - sv[i-1]) * (mk[i] - mk[i-1])
            fK.append(K); fctx.append(ctx / 1000)
        c = M[tag]["color"]; ratio = NFULL[tag] / NLAY[tag]
        ax.plot(fK, fctx, "o-", color=c, lw=2, label=f"{tag} hybrid (KV in {NFULL[tag]}/{NLAY[tag]} layers)")
        ax.plot(fK, [x*ratio for x in fctx], "x--", color=c, alpha=0.55, label=f"{tag} if pure-transformer (all {NLAY[tag]})")
    ax.set_xlabel("K = VRAM expert slots per layer"); ax.set_ylabel("max context before VRAM spill (K tokens)")
    ax.set_title("G - VRAM frontier: context where KV+slots cross 8 GB (spill begins)\n"
                 "hybrid reaches ~4x the context of a pure-transformer MoE at the same K", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
    save(fig, "cmp_G_vram_frontier.png", "measured static VRAM crossing 8 GB; pure-transformer = x (n_full/n_layer) projection")

def gH():  # no hard cliff: tok/s as VRAM grows past 8 GB (slot pool spills to shared RAM, still beats NVMe)
    d = load("q80", "k_spill.csv")
    if d is None: return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(d["peak_vram_mb"]/1000, d["tok_s"], "o-", color="C3", lw=2)
    for x, y, k in zip(d["peak_vram_mb"]/1000, d["tok_s"], d["K"]):
        ax.annotate(f"K={int(k)}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.axvline(8, ls=":", color="0.5"); ax.annotate("8 GB dedicated", (8, d["tok_s"].min()), rotation=90, fontsize=7, va="bottom", color="0.4")
    ax.set_xlabel("peak VRAM (GB) - past 8 GB spills to shared system RAM over PCIe"); ax.set_ylabel("decode tok/s")
    ax.set_title("H - No hard VRAM cliff (80B, growing K): spill stays > NVMe\n"
                 "tok/s keeps rising past 8 GB - PCIe-spilled slots (~16 GB/s) beat disk (~2 GB/s)", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "cmp_H_no_cliff.png", "WDDM over-commits to shared RAM; the 8 GB 'limit' is soft on this box")

def gI():  # context past the spill point (K=64, KV spills at ~12K): graceful tax, not a cliff
    d = load("q80", "ctx_spill.csv")
    if d is None: return
    tok, dt = d["tok"], d["dt_ms"]; mm = dt > 0; tps = 1000.0/dt[mm]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(tok[mm], _roll(tps, 128), "-", color="C3", lw=1.4)
    ax.axvline(12000, ls=":", color="0.5")
    ax.annotate("KV crosses 8 GB\n-> spill to shared RAM", (12000, np.nanmax(_roll(tps,128))*0.5), fontsize=7.5, color="0.4")
    ax.set_xlabel("context length (tokens), K=64"); ax.set_ylabel("decode tok/s (128-tok rolling)")
    ax.set_title("I - Context past the VRAM-spill point (80B, K=64)\n"
                 "decode taxes gently across the spill (PCIe KV reads), no hard cliff", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "cmp_I_ctx_spill.png", "KV spills to shared RAM at ~12K context; gradual PCIe tax, not a wall")

if __name__ == "__main__":
    for g in (gA, gB, gC, gD, gE, gF, gG, gH, gI):
        try: g()
        except Exception as e: print("FAILED", g.__name__, "->", repr(e))
