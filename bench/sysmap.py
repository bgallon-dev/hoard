#!/usr/bin/env python3
# System map: pull the cache-behavior surface OFFLINE from one access trace, via the validated
# (98.8%) per-layer-VRAM(K) + global-RAM(ram_cap) LRU simulator. One engine run -> the whole
# Miss-Ratio surface, working-set growth, and per-layer I/O hot-spots. No re-running the engine.
#   sysmap.py <trace.txt> [turn_index]
import sys, re
from collections import OrderedDict, defaultdict, Counter

path = sys.argv[1] if len(sys.argv) > 1 else "bench/results/gatedump_80b.txt"
want_turn = int(sys.argv[2]) if len(sys.argv) > 2 else None

# ---- format-agnostic parse: bloated gatedump ("e:p:sel:tier") -> sel==1; lean profile ("e e e") -> all ----
sel_re = re.compile(r'(\d+):[^:\s]+:1:\d')
def parse(path, want_turn):
    trace = []        # list of (layer, [experts]) in access order
    turn = -1; nL = 0
    for line in open(path):
        if line.startswith('# turn'): turn += 1; continue
        if not line or line[0] == '#': continue
        if want_turn is not None and turn != want_turn: continue
        if ':' in line:                                   # bloated gatedump
            sp = line.find(' '); sp2 = line.find(' ', sp+1)
            if sp < 0 or sp2 < 0: continue
            try: L = int(line[sp+1:sp2])
            except ValueError: continue
            es = [int(e) for e in sel_re.findall(line)]
        else:                                             # lean profile: "tok layer e0 e1 ..."
            p = line.split()
            if len(p) < 3: continue
            try: L = int(p[1]); es = [int(x) for x in p[2:]]
            except ValueError: continue
        if not es: continue
        nL = max(nL, L+1); trace.append((L, es))
    return trace, nL
trace, nL = parse(path, want_turn)
ntok = sum(1 for L, _ in trace if L == 0)
naccess = sum(len(es) for _, es in trace)
print(f"trace: {path}" + (f" turn={want_turn}" if want_turn is not None else ""))
print(f"  {ntok} tokens, {nL} layers, {naccess} accesses ({naccess/max(1,ntok):.0f}/token)")

class Sim:
    def __init__(self, K, ram_cap):
        self.K, self.ram_cap = K, ram_cap
        self.vram = [OrderedDict() for _ in range(nL)]; self.ram = OrderedDict()
        self.vh = self.rh = self.cold = 0
        self.cold_by_layer = [0]*nL
    def access(self, L, e):
        if e in self.vram[L]: self.vh += 1; self.vram[L].move_to_end(e); return
        if (L, e) in self.ram:
            self.rh += 1; self.ram.move_to_end((L, e))
        else:
            self.cold += 1; self.cold_by_layer[L] += 1
            if len(self.ram) >= self.ram_cap and self.ram_cap > 0: self.ram.popitem(last=False)
            if self.ram_cap > 0: self.ram[(L, e)] = 1
        if e in self.vram[L]: self.vram[L].move_to_end(e); return
        while len(self.vram[L]) >= self.K: self.vram[L].popitem(last=False)
        self.vram[L][e] = 1
def run(K, ram):
    s = Sim(K, ram)
    for L, es in trace:
        for e in es: s.access(L, e)
    return s

# ---- 1) Miss-Ratio surface: cold reads/token vs K (VRAM slots) x ram_cap (RAM tier) ----
Ks = [8, 16, 24, 32, 48, 64, 96, 128]; rams = [0, 1024, 2048, 4096, 6144, 8192]
print("\n=== MISS-RATIO SURFACE: cold (NVMe) reads / token ===")
corner = "K / ram"
print(f"  {corner:>7}" + "".join(f"{('ram='+str(r)):>10}" for r in rams))
print(f"  {'RAM GB:':>7}" + "".join(f"{(str(round(r*1.87/1024,1))):>10}" for r in rams) + "   (expert ~1.87MB; box has 16GB)")
for K in Ks:
    row = "".join(f"{run(K, r).cold/max(1,ntok):>10.1f}" for r in rams)
    print(f"  {K:>7}{row}")
print(f"  (operating point K=48 ram=2048; total VRAM-slot experts = K*{nL} layers)")

# ---- 2) tier split at the operating point ----
s = run(48, 2048); tot = s.vh+s.rh+s.cold
print(f"\n=== TIER SPLIT @ K=48 ram=2048: VRAM={100*s.vh/tot:.1f}% RAM={100*s.rh/tot:.1f}% NVMe={100*s.cold/tot:.1f}% ===")

# ---- 3) working-set growth: distinct (layer,expert) touched vs tokens ----
print("\n=== WORKING-SET GROWTH: distinct experts touched by token N ===")
seen = set(); marks = [1, 4, 16, 32, 64, 128, 256, 512, 1024]; t = 0; out = []
for L, es in trace:
    if L == 0:
        t += 1
        if t in marks: out.append((t, len(seen)))
    for e in es: seen.add((L, e))
out.append((t, len(seen)))
for tk, d in out: print(f"  by token {tk:>5}: {d:>6} distinct ({d/(nL*max(1,1)):.0f}/layer-equiv)  growth-rate={d/tk:.1f}/tok")
print(f"  total working set: {len(seen)} experts = {100*len(seen)/(nL*512):.1f}% of model; "
      f"fits in RAM tier? need ram_cap>={len(seen)} (have 2048)")

# ---- 4) per-layer I/O hot-spots @ operating point ----
cbl = run(48, 2048).cold_by_layer
order = sorted(range(nL), key=lambda L: -cbl[L])
print("\n=== PER-LAYER I/O HOT-SPOTS (cold reads/token) @ K=48 ram=2048 ===")
print("  hottest:  " + ", ".join(f"L{L}={cbl[L]/max(1,ntok):.1f}" for L in order[:6]))
print("  coldest:  " + ", ".join(f"L{L}={cbl[L]/max(1,ntok):.1f}" for L in order[-6:]))
print(f"  spread: max={max(cbl)/max(1,ntok):.1f} min={min(cbl)/max(1,ntok):.1f} mean={sum(cbl)/nL/max(1,ntok):.1f}/token-layer")
