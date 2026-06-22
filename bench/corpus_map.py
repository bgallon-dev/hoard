#!/usr/bin/env python3
# Corpus map: per-domain behavioral profile + the full N×N domain-divergence matrix, from one lean
# PROFILE trace ('# turn'-delimited) + a manifest of domain labels (one per turn, in order).
#   corpus_map.py <profile.txt> <manifest.txt> [K=48] [ram=2048] [warmN=16]
import sys
from collections import OrderedDict, Counter, defaultdict

prof = sys.argv[1] if len(sys.argv) > 1 else "bench/results/corpus_profile.txt"
man  = sys.argv[2] if len(sys.argv) > 2 else "bench/results/corpus_manifest.txt"
K    = int(sys.argv[3]) if len(sys.argv) > 3 else 48
RAM  = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
WN   = int(sys.argv[5]) if len(sys.argv) > 5 else 16

labels = [l.strip() for l in open(man) if l.strip()]
# parse lean profile into per-turn access traces: trace[d] = list of (layer, [experts])
turns = []; cur = None; nL = 0
for line in open(prof):
    if line.startswith('# turn'):
        cur = []; turns.append(cur); continue
    if not line.strip() or line[0] == '#' or cur is None: continue
    p = line.split()
    if len(p) < 3: continue
    try: L = int(p[1]); es = [int(x) for x in p[2:]]
    except ValueError: continue
    nL = max(nL, L+1); cur.append((L, es))
# map turns to labels (pad/truncate defensively)
doms = {}
for i, tr in enumerate(turns):
    lab = labels[i] if i < len(labels) else f"turn{i}"
    if tr: doms[lab] = tr
names = list(doms.keys())

class Sim:
    def __init__(s, K, ram): s.K, s.ram = K, ram; s.v=[OrderedDict() for _ in range(nL)]; s.r=OrderedDict(); s.cold=s.vh=s.rh=0
    def acc(s, L, e):
        if e in s.v[L]: s.vh+=1; s.v[L].move_to_end(e)
        elif (L,e) in s.r: s.rh+=1; s.r.move_to_end((L,e))
        else:
            s.cold+=1
            if len(s.r)>=s.ram and s.ram>0: s.r.popitem(last=False)
            if s.ram>0: s.r[(L,e)]=1
        if e not in s.v[L]:
            while len(s.v[L])>=s.K: s.v[L].popitem(last=False)
            s.v[L][e]=1
def nvme_pct(tr):
    s=Sim(K,RAM)
    for L,es in tr:
        for e in es: s.acc(L,e)
    t=s.vh+s.rh+s.cold; return 100*s.cold/t if t else 0

def freq(tr):
    f=defaultdict(Counter)
    for L,es in tr:
        for e in es: f[L][e]+=1
    return f
def warm(f,N): return {L:set(e for e,_ in f[L].most_common(N)) for L in range(nL)}
def coverage(w,tr):
    n=h=0
    for L,es in tr:
        for e in es: n+=1; h+=(e in w.get(L,()))
    return 100*h/n if n else 0

# ---- per-domain profile ----
print(f"corpus: {len(names)} domains | sim K={K} ram={RAM} warmN={WN}\n")
print(f"=== PER-DOMAIN PROFILE ===")
print(f"  {'domain':<12}{'tokens':>7}{'workset':>9}{'%model':>8}{'NVMe%':>7}{'route-ent':>10}")
fr = {}
import math
for d in names:
    tr = doms[d]; ntok = sum(1 for L,_ in tr if L==0); ws = set((L,e) for L,es in tr for e in es)
    fr[d] = freq(tr)
    # mean per-layer route entropy
    ents=[]
    for L in range(nL):
        c=fr[d][L]; tot=sum(c.values())
        if tot: ents.append(-sum((v/tot)*math.log2(v/tot) for v in c.values()))
    print(f"  {d:<12}{ntok:>7}{len(ws):>9}{100*len(ws)/(nL*512):>7.1f}%{nvme_pct(tr):>6.1f}%{sum(ents)/len(ents):>10.2f}")

# ---- N×N divergence: warm set of row-domain evaluated on col-domain's accesses ----
W = {d: warm(fr[d], WN) for d in names}
glob = defaultdict(Counter)
for d in names:
    for L in range(nL):
        for e,c in fr[d][L].items(): glob[L][e]+=c
W['GLOBAL'] = warm(glob, WN)
print(f"\n=== DIVERGENCE MATRIX: row=warm-set source, col=eval domain, cell=coverage% (top-{WN}/layer) ===")
print(f"  {'warm/eval':<11}" + "".join(f"{n[:7]:>8}" for n in names))
for d in names + ['GLOBAL']:
    row = "".join(f"{coverage(W[d], doms[c]):>8.1f}" for c in names)
    print(f"  {d:<11}{row}")
# own-vs-global advantage + nearest neighbor per domain
print(f"\n=== SEPARABILITY ===")
for c in names:
    own = coverage(W[c], doms[c]); gl = coverage(W['GLOBAL'], doms[c])
    nn = max((d for d in names if d!=c), key=lambda d: coverage(W[d], doms[c]))
    nnc = coverage(W[nn], doms[c])
    print(f"  {c:<12} own={own:>5.1f}%  global={gl:>5.1f}%  advantage={own-gl:>+5.1f}pts  nearest={nn}({nnc:.1f}%)")
