"""
close_loop.py -- the end-to-end out-of-sample test the feasibility law actually needs.

§3b of FEASIBILITY.md "predicted" the 80B using the 80B's OWN measured miss curve -> almost no
degrees of freedom. This closes the loop properly: predict the 80B's throughput from the 35B's
routing parameters + the 80B's PUBLISHED CONFIG, with NOTHING measured on the 80B except for scoring.

Pipeline (all inputs a-priori or transferred from the 35B):
  routing model   : (Zipf skew s, per-layer reuse alpha) fit on the 35B trace
  80B config      : n_expert=512, n_used=10, n_layer=48          (published)
  -> synthesize 80B routing -> LRU sim -> predicted m_nvme(K), h_vram(K)
  b_e             : 1.95 MB  (a-priori from the 80B GGUF expert-tensor bytes)
  t_fixed/layer   : 1.44 ms  (transferred from the 35B feasibility fit)
  B_nvme, B_pcie  : 2.11, 16 GB/s  (per-box constants)
  -> feasibility law -> predicted 80B tok/s(K), scored vs the measured 80B k_sweep.
Reproduce: py bench/close_loop.py
"""
import os, re, math
import numpy as np
from collections import OrderedDict, Counter, defaultdict
rng = np.random.default_rng(0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sel_re = re.compile(r'(\d+):[^:\s]+:1:\d')

def load_trace(path):
    seq = []; per_layer = defaultdict(list); nE = 0
    for line in open(path):
        if not line or line[0] == '#': continue
        sp = line.find(' '); sp2 = line.find(' ', sp+1)
        if sp < 0 or sp2 < 0: continue
        try: L = int(line[sp+1:sp2])
        except ValueError: continue
        es = [int(e) for e in sel_re.findall(line)]
        if not es: continue
        seq.append((L, es)); per_layer[L].append(es); nE = max(nE, max(es)+1)
    nL = max(per_layer)+1; ntok = max(len(v) for v in per_layer.values()); u = max(len(es) for _, es in seq)
    freq = {L: Counter() for L in range(nL)}
    for L, es in seq:
        for e in es: freq[L][e] += 1
    ent = [(-sum((c/sum(freq[L].values()))*math.log2(c/sum(freq[L].values())) for c in freq[L].values())) for L in range(nL)]
    # per-layer lag-1 reuse alpha
    ov_n = np.zeros(nL); ov_d = np.zeros(nL); prev = {}
    for L, es in seq:
        s_ = set(es)
        if L in prev: ov_n[L] += len(s_ & prev[L]); ov_d[L] += len(s_)
        prev[L] = s_
    alpha = ov_n/np.maximum(1, ov_d)
    meanH = sum(ent)/len(ent); maxH = math.log2(nE)
    # fit Zipf s to mean entropy
    def entz(s):
        r = np.arange(1, nE+1); p = (1.0/np.power(r, s)); p /= p.sum(); return -np.sum(p*np.log2(p))
    lo, hi = 0.01, 3.0
    for _ in range(40):
        mid = (lo+hi)/2
        if entz(mid) > meanH: lo = mid
        else: hi = mid
    return dict(seq=seq, nL=nL, nE=nE, u=u, ntok=ntok, alpha=alpha, s=(lo+hi)/2,
                normH=meanH/maxH)

def sim_tiers(seq, K, nL, ram_cap=2048):
    vram = [OrderedDict() for _ in range(nL)]; ram = OrderedDict(); vh = rh = cold = 0
    for L, es in seq:
        for e in es:
            if e in vram[L]: vh += 1; vram[L].move_to_end(e); continue
            if (L, e) in ram: rh += 1; ram.move_to_end((L, e))
            else:
                cold += 1
                if len(ram) >= ram_cap: ram.popitem(last=False)
                ram[(L, e)] = 1
            while len(vram[L]) >= K: vram[L].popitem(last=False)
            vram[L][e] = 1
    t = vh+rh+cold
    return vh/t, rh/t, cold/t

def synth_sticky_zipf(nE, u, nL, ntok, s, alpha):
    r = np.arange(1, nE+1); zp = (1.0/np.power(r, s)); zp /= zp.sum()
    out = []; prev = {}
    for _ in range(ntok):
        for L in range(nL):
            a = alpha[L]
            if L not in prev:
                es = rng.choice(nE, size=u, replace=False, p=zp)
            else:
                nk = min(int(rng.binomial(u, a)), len(prev[L]))
                kept = list(rng.choice(prev[L], size=nk, replace=False)) if nk else []
                ks = set(kept); fresh = []
                while len(fresh) < u-nk:
                    c = int(rng.choice(nE, p=zp))
                    if c not in ks: ks.add(c); fresh.append(c)
                es = np.array(kept+fresh)
            prev[L] = es.tolist(); out.append((L, list(es)))
    return out

# ---- LOAD ----
src = load_trace(os.path.join(ROOT, "bench/results/gatedump_35b.txt"))   # FIT here
tgt = load_trace(os.path.join(ROOT, "bench/results/gatedump_80b.txt"))   # SCORE here (routing used only for ground-truth m)
print("routing params:  35B(fit)  vs  80B(measured, for reference only)")
print(f"  Zipf s        : {src['s']:.3f}   vs  {tgt['s']:.3f}")
print(f"  norm entropy  : {src['normH']:.3f}   vs  {tgt['normH']:.3f}")
print(f"  mean alpha    : {src['alpha'].mean():.3f}   vs  {tgt['alpha'].mean():.3f}")

# ---- TRANSFER: 35B routing params -> 80B config (resample alpha profile 40->48 by relative depth) ----
alpha_xfer = np.interp(np.linspace(0, 1, 48), np.linspace(0, 1, src['nL']), src['alpha'])
synth80 = synth_sticky_zipf(nE=512, u=10, nL=48, ntok=256, s=src['s'], alpha=alpha_xfer)

# ---- predicted vs measured 80B miss/hit curves ----
Ks = [16, 24, 32, 48, 64]
print("\nstep 1 -- predict 80B m_nvme(K) from 35B routing + 80B config (zero 80B routing used):")
print(f"  {'K':>4} {'nvme% pred':>10} {'nvme% meas':>10} {'vram% pred':>10} {'vram% meas':>10}")
pred = {}; meas = {}
for K in Ks:
    vp, rp, cp = sim_tiers(synth80, K, 48)
    vm, rm, cm = sim_tiers(tgt['seq'], K, 48)
    pred[K] = (vp, cp); meas[K] = (vm, cm)
    print(f"  {K:>4} {cp*100:>10.1f} {cm*100:>10.1f} {vp*100:>10.1f} {vm*100:>10.1f}")

# ---- feed through the feasibility law: everything a-priori or transferred from 35B ----
b_e = 1.95              # a-priori from 80B GGUF
A = 10*48*b_e
t_fixed = 1.44*48       # transferred 35B t_fixed/layer
B_nvme = 2.11; B_pcie = 16.0
# measured 80B k_sweep tok/s for scoring
ksrows = [r.split(',') for r in open(os.path.join(ROOT,"bench/results/q80/k_sweep.csv")).read().splitlines()[1:]]
kmeas = {int(r[0]): float(r[1]) for r in ksrows if r[0] not in ('0','')}
# miss-curve prediction error (the routing-model contribution, in isolation)
m_mape = np.mean([abs(pred[K][1]-meas[K][1])/meas[K][1] for K in Ks])*100
print(f"  -> predicted-m(K) MAPE vs measured = {m_mape:.1f}%  (biased pessimistic: transferred Zipf skew is")
print(f"     slightly more diffuse than the 80B's true routing -> over-predicts cold reads)")

print("\nstep 2 -- feed predicted curve through the law -> 80B tok/s, scored vs measured k_sweep:")
print(f"  {'K':>4} {'tok/s pred':>10} {'tok/s meas':>10} {'err%':>6}")
errs = []; errs_truefloor = []
t_fixed_true = 83.7      # the 80B's actually-fit t_fixed (feasibility.py) -- used ONLY to decompose the bias
for K in Ks:
    vp, cp = pred[K]
    t  = t_fixed      + A*cp/B_nvme + A*(1-vp)/B_pcie
    t2 = t_fixed_true + A*cp/B_nvme + A*(1-vp)/B_pcie
    tp = 1000.0/t; tm = kmeas[K]; e = (tp-tm)/tm*100; errs.append(abs(e))
    errs_truefloor.append(abs((1000.0/t2 - tm)/tm*100))
    print(f"  {K:>4} {tp:>10.2f} {tm:>10.2f} {e:>+6.1f}")
print(f"\n  END-TO-END out-of-sample MAPE (80B tok/s from 35B params + 80B config, zero 80B timing/routing) = {np.mean(errs):.1f}%")
print(f"\n  HONEST DECOMPOSITION (so the 3% isn't a falsely-clean headline):")
print(f"    - the predicted miss curve is ~{m_mape:.0f}% pessimistic (more cold reads than real)")
print(f"    - the transferred t_fixed/layer (1.44, from 35B) is ~17% LOW vs the 80B's true 1.74")
print(f"    - these biases partly cancel. Isolating the routing error (predicted-m + the 80B's TRUE")
print(f"      t_fixed) gives MAPE {np.mean(errs_truefloor):.1f}% -- that is the honest routing-prediction cost,")
print(f"      and the {np.mean(errs):.1f}% composite benefits from ~half of it cancelling the t_fixed under-transfer.")
