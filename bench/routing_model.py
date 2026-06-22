"""
routing_model.py -- close the loop: can m_nvme(K) be PREDICTED rather than measured?

The feasibility law (bench/feasibility.py) consumes a measured miss curve m_nvme(K). That is the
one non-a-priori input (b_e is now closed-form; see the GGUF b_e check). Here we test whether m(K)
follows from a low-dimensional ROUTING MODEL + config, by replaying SYNTHETIC routing through the
same validated LRU simulator and comparing to the real trace's miss curve.

Models, in increasing knowledge:
  uniform   : each token draws n_used of n_expert uniformly        (0 parameters, config only)
  zipf(s)   : draw from a Zipf(s) over experts, s fit to mean entropy  (1 parameter)
  iid-emp   : draw i.i.d. from the REAL per-layer frequency table   (tests: does temporal order matter?)
  real      : the actual recorded access sequence                  (ground truth = sysmap surface)

If iid-emp reproduces real m(K), then m(K) depends only on the frequency DISTRIBUTION (memoryless),
and that distribution is a 1-parameter (skew) object -> m(K) becomes predictable from (n_expert,
n_used, n_layer, skew). That converts the law's last measured input into a modeled one.
Reproduce: py bench/routing_model.py [gatedump]
"""
import sys, os, re, math
import numpy as np
from collections import OrderedDict, Counter, defaultdict

GD = sys.argv[1] if len(sys.argv) > 1 else "bench/results/gatedump_80b.txt"
rng = np.random.default_rng(0)

# ---- parse the bloated gatedump: per line "tok layer wk e:p:sel:tier ..."; keep selected experts ----
sel_re = re.compile(r'(\d+):[^:\s]+:1:\d')
seq = []                       # list over (token,layer) in order: arrays of selected expert ids
per_layer = defaultdict(list)  # layer -> list of selected-id arrays (per token)
nE = 0
cur_tok = None
for line in open(GD):
    if not line or line[0] == '#': continue
    sp = line.find(' '); sp2 = line.find(' ', sp+1)
    if sp < 0 or sp2 < 0: continue
    try: tok = int(line[:sp]); L = int(line[sp+1:sp2])
    except ValueError: continue
    es = [int(e) for e in sel_re.findall(line)]
    if not es: continue
    seq.append((L, es)); per_layer[L].append(es)
    m = max(es)
    if m+1 > nE: nE = m+1
nL = max(per_layer) + 1
ntok = max(len(v) for v in per_layer.values())
u = max(len(es) for _, es in seq)
print(f"trace {GD}: {ntok} tok, {nL} layers, {nE} experts/layer, n_used={u}")

# per-layer empirical frequency tables + entropy
freq = {L: Counter() for L in range(nL)}
for L, es in seq:
    for e in es: freq[L][e] += 1
ent = []
for L in range(nL):
    n = sum(freq[L].values()); H = -sum((c/n)*math.log2(c/n) for c in freq[L].values())
    ent.append(H)
meanH = sum(ent)/len(ent); maxH = math.log2(nE)
print(f"per-layer route entropy: mean={meanH:.2f} / {maxH:.2f} bits  (normalized {meanH/maxH:.3f}; 1.0=uniform)")

# ---- LRU per-layer VRAM(K) simulator (matches sysmap/profiler), returns cold reads/token ----
def miss_of(access_by_layer, K, ram_cap=2048):
    vram = [OrderedDict() for _ in range(nL)]; ram = OrderedDict(); cold = 0; ntk = 0
    # access_by_layer: list of (L, es) in the SAME interleaving as decode (token-major, layer order)
    for L, es in access_by_layer:
        if L == 0: ntk += 1
        for e in es:
            if e in vram[L]: vram[L].move_to_end(e); continue
            if (L, e) in ram: ram.move_to_end((L, e))
            else:
                cold += 1
                if len(ram) >= ram_cap: ram.popitem(last=False)
                ram[(L, e)] = 1
            while len(vram[L]) >= K: vram[L].popitem(last=False)
            vram[L][e] = 1
    return cold / max(1, ntk)

# build per-layer probability tables
emp_p = {}
for L in range(nL):
    p = np.zeros(nE)
    for e, c in freq[L].items(): p[e] = c
    emp_p[L] = p / p.sum()

def zipf_p(s):
    r = np.arange(1, nE+1); w = 1.0/np.power(r, s); return w/w.sum()

# measure per-layer lag-1 reuse (the temporal-locality parameter alpha)
ov_num = np.zeros(nL); ov_den = np.zeros(nL)
prev = {}
for L, es in seq:
    s_ = set(es)
    if L in prev: ov_num[L] += len(s_ & prev[L]); ov_den[L] += len(s_)
    prev[L] = s_
alpha = ov_num / np.maximum(1, ov_den)        # per-layer persistence (expected fraction reused)
print(f"per-layer lag-1 reuse alpha: mean={alpha.mean():.3f}  (L0={alpha[0]:.2f} .. max={alpha.max():.2f})")

def synth(kind, s=None):
    """generate a token-major synthetic access sequence with the same shape as the real trace."""
    out = []
    zp = zipf_p(s) if s is not None else None
    prev_sel = {}
    for t in range(ntok):
        for L in range(nL):
            if kind == "uniform":
                es = rng.choice(nE, size=u, replace=False)
            elif kind == "zipf":
                es = rng.choice(nE, size=u, replace=False, p=zp)        # same skew, random labels
            elif kind == "iid-emp":
                es = rng.choice(nE, size=u, replace=False, p=emp_p[L])  # real per-layer freq
            elif kind.startswith("sticky"):                              # persistence + a 'fresh' law
                fp = {"sticky":emp_p[L], "sticky-zipf":zp, "sticky-unif":None}[kind]
                if L not in prev_sel:
                    es = rng.choice(nE, size=u, replace=False, p=fp)
                else:
                    nk = min(int(rng.binomial(u, alpha[L])), len(prev_sel[L]))
                    kept = list(rng.choice(prev_sel[L], size=nk, replace=False)) if nk else []
                    keptset = set(kept); fresh = []
                    while len(fresh) < u - nk:                            # draw rest fresh, avoid dups
                        c = int(rng.choice(nE, p=fp))
                        if c not in keptset: keptset.add(c); fresh.append(c)
                    es = np.array(kept + fresh)
                prev_sel[L] = es.tolist()
            out.append((L, list(es)))
    return out

# fit zipf s to match mean entropy
def ent_of_zipf(s):
    p = zipf_p(s); return -np.sum(p*np.log2(p))
lo, hi = 0.01, 3.0
for _ in range(40):
    mid = (lo+hi)/2
    if ent_of_zipf(mid) > meanH: lo = mid       # higher s -> lower entropy
    else: hi = mid
s_fit = (lo+hi)/2
print(f"fitted Zipf exponent s={s_fit:.3f} (entropy {ent_of_zipf(s_fit):.2f} vs target {meanH:.2f})\n")

Ks = [8, 16, 24, 32, 48, 64, 96, 128]
real_seq = seq
models = {
    "real (ground truth)": real_seq,
    "sticky-emp (freq+reuse)": synth("sticky"),
    "sticky-zipf (skew+reuse)": synth("sticky-zipf", s_fit),
    "sticky-unif (reuse only)": synth("sticky-unif"),
    "iid-emp (freq, no time)": synth("iid-emp"),
    "uniform (0-param)": synth("uniform"),
}
print(f"  {'cold reads/token':<26}" + "".join(f"K={k:<5}" for k in Ks))
real_m = None
rows = {}
for name, sq in models.items():
    ms = [miss_of(sq, K) for K in Ks]
    rows[name] = ms
    if name.startswith("real"): real_m = ms
    print(f"  {name:<26}" + "".join(f"{m:<7.1f}" for m in ms))
print(f"\n  {'MAPE vs real (across K)':<26}")
for name, ms in rows.items():
    if name.startswith("real"): continue
    mape = np.mean([abs(a-b)/b for a, b in zip(ms, real_m)])*100
    print(f"    {name:<26} {mape:5.1f}%")
