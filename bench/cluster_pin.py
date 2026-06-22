#!/usr/bin/env python3
# Convert the domain-divergence advantage into a real cold-read reduction, HONESTLY (held-out):
# build a cluster's warm profile from ONE session, pin it, and evaluate on a DIFFERENT session of
# the same cluster. Compares LRU vs LRU+global-pin vs LRU+cluster-pin. This is the session-aware
# caching lever (idea 4) measured as cold reads, not just coverage.
#   cluster_pin.py <profile.txt> <manifest.txt> [K=48] [ram=2048] [pinN=16]
import sys
from collections import OrderedDict, Counter, defaultdict

prof = sys.argv[1]; man = sys.argv[2]
K   = int(sys.argv[3]) if len(sys.argv) > 3 else 48
RAM = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
PIN = int(sys.argv[5]) if len(sys.argv) > 5 else 16

labels = [l.strip() for l in open(man, encoding='utf-8-sig') if l.strip()]
turns = []; cur=None; nL=0
for line in open(prof):
    if line.startswith('# turn'): cur=[]; turns.append(cur); continue
    if not line.strip() or line[0]=='#' or cur is None: continue
    p=line.split()
    if len(p)<3: continue
    try: L=int(p[1]); es=[int(x) for x in p[2:]]
    except ValueError: continue
    nL=max(nL,L+1); cur.append((L,es))
doms={ (labels[i] if i<len(labels) else f"t{i}"): tr for i,tr in enumerate(turns) if tr }
# strip BOM that can sneak into the first label
doms={ k.lstrip('﻿'): v for k,v in doms.items() }

class Sim:
    def __init__(s,K,ram,pin=None):
        s.K,s.ram=K,ram; s.v=[OrderedDict() for _ in range(nL)]; s.r=OrderedDict()
        s.pin=pin or [set() for _ in range(nL)]; s.cold=0
    def acc(s,L,e):
        if e in s.pin[L]: return                      # pinned: always resident, never cold
        if e in s.v[L]: s.v[L].move_to_end(e)
        elif (L,e) in s.r: s.r.move_to_end((L,e))
        else:
            s.cold+=1
            if len(s.r)>=s.ram and s.ram>0: s.r.popitem(last=False)
            if s.ram>0: s.r[(L,e)]=1
        if e not in s.v[L]:
            cap=s.K-len(s.pin[L])
            while len(s.v[L])>=cap: s.v[L].popitem(last=False)
            s.v[L][e]=1
    def run(s,tr):
        for L,es in tr:
            for e in es: s.acc(L,e)
        return s.cold

def freq(tr):
    f=defaultdict(Counter)
    for L,es in tr:
        for e in es: f[L][e]+=1
    return f
def warm(f,N): return [set(e for e,_ in f[L].most_common(N)) for L in range(nL)]

glob=defaultdict(Counter)
for tr in doms.values():
    for L,es in tr:
        for e in es: glob[L][e]+=1
warm_global=warm(glob,PIN)

# held-out sibling map (same cluster, different session)
sib={'code_c':'code_py','code_py':'code_c','math_proof':'math_calc','math_calc':'math_proof',
     'creative':'poetry','poetry':'creative'}
ntok=lambda tr: sum(1 for L,_ in tr if L==0)
print(f"held-out cluster pinning (pin top-{PIN}/layer, K={K} ram={RAM}); profile built from the SIBLING session\n")
print(f"  {'domain (eval)':<14}{'profile(held-out)':<18}{'LRU':>7}{'+global':>9}{'+cluster':>10}{'  cluster vs LRU':>16}")
red_g=[]; red_c=[]
for d,s in sib.items():
    tr=doms[d]; nt=ntok(tr)
    c_lru = Sim(K,RAM).run(tr)/nt
    c_glb = Sim(K,RAM,pin=warm_global).run(tr)/nt
    c_sib = Sim(K,RAM,pin=warm(freq(doms[s]),PIN)).run(tr)/nt
    red_g.append(100*(c_lru-c_glb)/c_lru); red_c.append(100*(c_lru-c_sib)/c_lru)
    print(f"  {d:<14}{s:<18}{c_lru:>7.1f}{c_glb:>9.1f}{c_sib:>10.1f}{100*(c_lru-c_sib)/c_lru:>+14.1f}%")
print(f"\n  mean cold-read reduction vs LRU:  global-pin {sum(red_g)/len(red_g):+.1f}%   cluster-pin {sum(red_c)/len(red_c):+.1f}%")
print(f"  (cluster-pin = session-aware: detect domain cluster, pin its warm set. Honest: profile from a different same-cluster session.)")
