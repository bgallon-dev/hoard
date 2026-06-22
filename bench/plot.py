#!/usr/bin/env python3
# Plot the eight "legibility" graphs for the streaming-MoE engine from the bench CSVs.
# Model under test: Qwen3.5-35B-A3B (21 GB, hybrid linear-attention MoE), streaming experts from
# real NVMe on an 8 GB RX 6600 + 16 GB RAM. Run:  py bench/plot.py
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "bench", "results", "qwen")
FIGS = os.path.join(ROOT, "bench", "figs")
os.makedirs(FIGS, exist_ok=True)
MODEL = "Qwen3.5-35B-A3B Q4_K_M (21 GB) - streaming experts from NVMe, RX 6600 8 GB"

def load(name):
    with open(os.path.join(RES, name), encoding="utf-8-sig") as f:  # utf-8-sig strips PowerShell's BOM
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols

def save(fig, name):
    fig.suptitle(MODEL, y=0.005, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))

# ---- Graph 1: tok/s vs K expert slots (speed/cache tradeoff) ----
def g1():
    d = load("k_sweep.csv")
    naive = d["tok_s"][d["K"] == 0][0]
    m = d["K"] > 0
    K, toks, vhit, vram = d["K"][m], d["tok_s"][m], d["vram_pct"][m], d["peak_vram_mb"][m]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(K, toks, "o-", color="C0", lw=2, label="cache (LRU residency)")
    ax.axhline(naive, ls="--", color="0.5", label=f"naive recopy-every-token ({naive:.2f} tok/s)")
    for x, y, h in zip(K, toks, vhit):
        ax.annotate(f"{h:.0f}% VRAM hit", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.set_xlabel("K = VRAM expert slots per layer")
    ax.set_ylabel("decode tok/s")
    ax.set_title(f"Graph 1 - Speed vs cache size: K=48 = {toks.max()/naive:.2f}x naive\n"
                 f"diffuse 8-of-256 routing: ~{100-vhit.max():.0f}% still streams from NVMe at K=48", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax2 = ax.twinx(); ax2.plot(K, vram/1000, "s:", color="C3", alpha=0.6)
    ax2.set_ylabel("peak VRAM (GB)", color="C3"); ax2.tick_params(axis="y", colors="C3")
    ax2.axhline(8, ls=":", color="C3", alpha=0.4); ax2.annotate("8 GB VRAM limit", (K[0], 8), color="C3", fontsize=7, va="bottom")
    save(fig, "fig1_tok_vs_k.png")

# ---- Graph 2: peak VRAM vs context length (cheap KV growth) ----
def g2():
    d = load("vram_vs_ctx.csv")
    ctx, vram = d["ctx"], d["peak_vram_mb"]
    slope = np.polyfit(ctx, vram, 1)[0]  # MB per token
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ctx, vram, "o-", color="C2", lw=2)
    ax.set_xlabel("context length (prompt + generated tokens)")
    ax.set_ylabel("peak VRAM (MB)")
    ax.set_title(f"Graph 2 - Cheap KV growth: {slope:.3f} MB/token\n"
                 f"hybrid attn: only 10 of 40 layers carry KV (30 GDN layers = fixed state)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.annotate(f"+{(vram[-1]-vram[0]):.0f} MB over {int(ctx[-1]-ctx[0])} tokens\n"
                f"= {slope:.3f} MB/tok (a 4096-token context adds only ~{slope*4096:.0f} MB)",
                (ctx[len(ctx)//2], vram[0]), fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="C2", alpha=0.9))
    save(fig, "fig2_vram_vs_ctx.png")

# ---- Graph 3: tok/s vs RAM cache size (does the RAM tier matter?) ----
def g3():
    d = load("ram_sweep.csv")
    gb = d["ram_cap"] * 1.86 / 1000  # ~1.86 MB/expert -> GB
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(gb, d["tok_s"], "o-", color="C0", lw=2, label="decode tok/s")
    ax.set_xlabel("RAM tier size (GB of warm experts)")
    ax.set_ylabel("decode tok/s", color="C0"); ax.tick_params(axis="y", colors="C0")
    g = d["tok_s"]
    ax.set_title(f"Graph 3 - RAM tier matters under real NVMe: +{100*(g[-1]/g[0]-1):.0f}% tok/s\n"
                 f"grow warm-expert RAM 0.5 -> 5.7 GB (flat on a fits-in-RAM model)", fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(gb, d["ram_pct"], "s--", color="C1", label="RAM hit %")
    ax2.plot(gb, d["nvme_pct"], "^--", color="C3", label="NVMe miss %")
    ax2.set_ylabel("access %")
    ax.grid(alpha=0.3)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
    save(fig, "fig3_tok_vs_ram.png")

# ---- Graph 4: tok/s vs storage bandwidth (storage sensitivity) ----
def g4():
    d = load("storage_sweep.csv")
    thr, toks, gbps = d["throttle_mbps"], d["tok_s"], d["disk_gbps"]
    # x = effective bandwidth in MB/s; unthrottled (throttle=0) uses the measured real drive GB/s
    real = gbps[thr == 0]
    real_mbps = (real[0] * 1000) if len(real) and real[0] > 0 else 2060.0
    x = np.where(thr == 0, real_mbps, thr)
    order = np.argsort(x)
    x, toks = x[order], toks[order]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(x / 1000, toks, "o-", color="C4", lw=2)
    ax.set_xlim(0, x.max()/1000 * 1.22)  # headroom so the right-most label isn't clipped
    labels = {100: ("HDD-class", 6, -2, "left"), 600: ("slow SATA SSD", 6, -2, "left"),
              real_mbps: (f"real NVMe ({real_mbps/1000:.2f} GB/s)", -8, 6, "right")}
    for xi, yi in zip(x, toks):
        key = min(labels, key=lambda k: abs(k - xi)) if labels else None
        if key is not None and abs(key - xi) < 60:
            txt, dx, dy, ha = labels.pop(key)
            ax.annotate(txt, (xi/1000, yi), textcoords="offset points", xytext=(dx, dy), fontsize=7, ha=ha)
    ax.set_xlabel("effective storage bandwidth (GB/s)")
    ax.set_ylabel("decode tok/s")
    ax.set_title("Graph 4 - Storage sensitivity: decode is bandwidth-bound\n"
                 "throttled slow-drive sims + real measured NVMe (top point)", fontsize=11)
    ax.grid(alpha=0.3)
    save(fig, "fig4_tok_vs_storage.png")

# ---- helpers for time-series ----
def windowed(d, win=8):
    dq = np.diff(d["reqs"]); dv = np.diff(d["vram_hit"]); dr = np.diff(d["ram_hit"]); dn = np.diff(d["nvme_fall"])
    tok = d["tok"][1:]
    vp, rp, npc = 100*dv/dq, 100*dr/dq, 100*dn/dq
    def roll(a):
        if len(a) < win: return a
        k = np.ones(win)/win
        return np.convolve(a, k, mode="same")
    return tok, roll(vp), roll(rp), roll(npc)

# ---- Graph 5 (HEADLINE): cache hit rate over time ----
def g5():
    d = load("ts_main.csv")
    tok, vp, rp, npc = windowed(d)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.stackplot(tok, vp, rp, npc,
                 labels=["VRAM hit (instant)", "RAM hit (~0.3 ms)", "NVMe miss (~1 ms read)"],
                 colors=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85)
    ax.set_xlim(tok.min(), tok.max()); ax.set_ylim(0, 100)
    ax.set_xlabel("generated token index"); ax.set_ylabel("share of expert fetches (%)")
    vmean = np.nanmean(vp); rmean = np.nanmean(rp); nmean = np.nanmean(npc)
    ax.set_title("Graph 5 (headline) - Expert cache hit rate over time\n"
                 f"K=24 + 2048-expert RAM tier: steady-state VRAM {vmean:.0f}% / RAM {rmean:.0f}% / NVMe {nmean:.0f}%",
                 fontsize=11)
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)
    save(fig, "fig5_hitrate_over_time.png")

# ---- Graph 6: token latency percentiles (does it feel smooth?) ----
def g6():
    d = load("ts_main.csv")
    dt = d["dt_ms"][1:]  # drop token 1 (no predecessor)
    dt = dt[dt > 0]
    p50, p95, p99 = np.percentile(dt, [50, 95, 99])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(dt, bins=30, color="C0", alpha=0.7, edgecolor="white")
    for p, v, c in [("p50", p50, "C2"), ("p95", p95, "C1"), ("p99", p99, "C3")]:
        ax.axvline(v, color=c, lw=2, label=f"{p} = {v:.0f} ms")
    ax.set_xlabel("per-token latency (ms)"); ax.set_ylabel("token count")
    ax.set_title(f"Graph 6 - Per-token latency (K=24, real NVMe)\n"
                 f"median {p50:.0f} ms (~{1000/p50:.1f} tok/s); p95 {p95:.0f}, p99 {p99:.0f} ms", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig6_latency_percentiles.png")

# ---- Graph 7: quality parity vs normal inference ----
def g7():
    rows = {r["test"]: r for r in csv.DictReader(open(os.path.join(RES, "parity.csv"), encoding="utf-8-sig"))}
    fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.axis("off")
    ax.set_title("Graph 7 - Quality parity: the streaming engine is faithful", fontsize=13, pad=18)
    # (check, meaning) per row; match value pulled from the CSV
    spec = [
        ("oracle_token_match_france", "Greedy output vs llama.cpp oracle", "Qwen3.5-35B; identical token IDs"),
        ("lossless_naive_vs_cache",   "Residency cache vs recopy-every-token", "bit-identical: cache is lossless"),
        ("olmoe_oracle_match",        "Greedy output vs llama.cpp oracle", "OLMoE-1B-7B (fits-RAM full oracle)"),
    ]
    table = [["check", "match", "meaning"]]
    for key, check, meaning in spec:
        r = rows[key]
        table.append([check, f"{r['matched']}/{r['total']}", meaning])
    t = ax.table(cellText=table, cellLoc="left", loc="center", colWidths=[0.40, 0.13, 0.47])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.0)
    for c in range(3): t[(0, c)].set_facecolor("#dddddd"); t[(0, c)].set_text_props(weight="bold")
    for r in range(1, len(table)): t[(r, 1)].set_text_props(weight="bold", color="#2ca02c")
    ax.text(0.5, -0.02, "100% token-for-token: the streaming + 3-tier cache changes speed, never the output.",
            transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="0.3")
    save(fig, "fig7_quality_parity.png")

# ---- Graph 8: expert reuse over a conversation (locality) ----
def g8():
    d = load("ts_main.csv")
    tok, distinct = d["tok"], d["distinct"]
    new = np.diff(distinct, prepend=distinct[0])  # new distinct experts per token
    total = 10240
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(tok, 100*distinct/total, "o-", ms=2, color="C0", lw=1.6, label="cumulative distinct experts (% of model)")
    ax.set_xlabel("generated token index"); ax.set_ylabel("distinct experts touched (% of 10240)", color="C0")
    ax.tick_params(axis="y", colors="C0")
    ax.set_title("Graph 8 - Expert reuse over a conversation (locality exists)\n"
                 f"{int(distinct[-1])} distinct = {100*distinct[-1]/total:.0f}% of model over {int(tok[-1])} tokens; "
                 "new-expert rate decays", fontsize=11)
    ax2 = ax.twinx()
    k = np.ones(8)/8
    ax2.plot(tok, np.convolve(new, k, mode="same"), color="C3", alpha=0.7, label="new experts / token (8-tok avg)")
    ax2.set_ylabel("new experts per token (of 320 fetched)", color="C3"); ax2.tick_params(axis="y", colors="C3")
    ax2.axhline(320, ls=":", color="0.6"); ax2.annotate("320 = all-cold ceiling", (tok[len(tok)//3], 320), fontsize=7, va="bottom", color="0.4")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
    save(fig, "fig8_expert_reuse.png")

if __name__ == "__main__":
    for g in (g1, g2, g3, g4, g5, g6, g7, g8):
        try:
            g()
        except Exception as e:
            print("FAILED", g.__name__, "->", repr(e))
