#!/usr/bin/env python3
# Gate 4 (session/domain-aware caching): is the router's expert usage SEPARABLE by domain?
# If a warm set built from domain A covers A's accesses much better than a domain-agnostic global
# warm set does, then detecting the session domain and swapping warm sets has real value (idea 4).
# If own-domain and global warm sets cover equally, session-awareness adds nothing over a plain cache.
#   domain_gate.py <traceA.txt> <labelA> <traceB.txt> <labelB> ...
import sys, re
from collections import Counter, defaultdict

sel_re = re.compile(r'(\d+):[^:\s]+:1:\d')
def load(path):
    freq = defaultdict(Counter)          # layer -> expert -> count
    acc  = []                            # list of (layer, [experts]) per (tok,layer)
    nL = 0
    for line in open(path):
        if not line or line[0] == '#': continue
        sp = line.find(' '); sp2 = line.find(' ', sp+1)
        if sp < 0 or sp2 < 0: continue
        try: layer = int(line[sp+1:sp2])
        except ValueError: continue
        es = [int(e) for e in sel_re.findall(line)]
        if not es: continue
        nL = max(nL, layer+1)
        acc.append((layer, es))
        for e in es: freq[layer][e] += 1
    return freq, acc, nL

doms = []
for i in range(1, len(sys.argv), 2):
    path, label = sys.argv[i], sys.argv[i+1]
    freq, acc, nL = load(path)
    doms.append((label, freq, acc, nL))
nL = max(d[3] for d in doms)

def warm(freq, N):                        # warm set: top-N experts per layer
    return {L: set(e for e, _ in freq[L].most_common(N)) for L in range(nL)}
def coverage(warmset, acc):               # fraction of accesses whose expert is in the warm set
    n = h = 0
    for L, es in acc:
        for e in es:
            n += 1; h += (e in warmset.get(L, ()))
    return 100*h/n if n else 0

# activated-set overlap between the first two domains
A, B = doms[0], doms[1]
setA = set((L, e) for L in range(nL) for e in A[1][L]); setB = set((L, e) for L in range(nL) for e in B[1][L])
print(f"domains: " + ", ".join(f"{d[0]} ({len(d[2])} tok*layer rows)" for d in doms))
print(f"\nactivated (layer,expert) sets: {A[0]}={len(setA)}  {B[0]}={len(setB)}  "
      f"Jaccard={len(setA&setB)/len(setA|setB):.3f}  (1=identical usage, 0=disjoint)")

# global warm set (both domains pooled)
glob = defaultdict(Counter)
for _, freq, _, _ in doms:
    for L in range(nL):
        for e, c in freq[L].items(): glob[L][e] += c

print("\n=== coverage: warm set (rows) evaluated on each domain's accesses (cols) ===")
print("   higher = more accesses served from the pinned warm set (fewer cold reads)")
for N in (8, 16, 32):
    print(f"\n  -- top-{N} experts/layer --")
    wsets = {label: warm(freq, N) for label, freq, _, _ in doms}
    wsets["GLOBAL"] = warm(glob, N)
    hdr = "".join(f"{d[0][:10]:>12}" for d in doms)
    label_col = "warm-set / eval"
    print(f"  {label_col:<18}{hdr}")
    for wlabel, ws in wsets.items():
        row = "".join(f"{coverage(ws, d[2]):>11.1f}%" for d in doms)
        print(f"  {wlabel:<18}{row}")
    # session-aware advantage: own-domain warm coverage minus GLOBAL warm coverage, per domain
    adv = [coverage(wsets[d[0]], d[2]) - coverage(wsets['GLOBAL'], d[2]) for d in doms]
    print(f"  session-aware advantage (own - GLOBAL): " + ", ".join(f"{d[0]}={a:+.1f}pts" for d, a in zip(doms, adv)))
