#!/usr/bin/env python3
# Per-layer MEMORY FLOW (measured): for each of the 48 layers, the expert-byte traffic NVMe->RAM->VRAM
# per decode token, from the validated simulator. Quantifies WHERE in the layer stack memory moves.
#   per_layer_flow.py <trace.txt> [K=48] [ram=2048]
import sys, re
from collections import OrderedDict, Counter, defaultdict
MB = 1.87  # per expert (gate+up+down), MiB

path = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv)>2 else 48; RAM = int(sys.argv[3]) if len(sys.argv)>3 else 2048
sel_re = re.compile(r'(\d+):[^:\s]+:1:\d')
trace=[]; nL=0
for line in open(path):
    if line.startswith('#') or not line.strip(): continue
    if ':' in line:
        sp=line.find(' '); sp2=line.find(' ',sp+1)
        if sp<0 or sp2<0: continue
        try: L=int(line[sp+1:sp2])
        except ValueError: continue
        es=[int(e) for e in sel_re.findall(line)]
    else:
        p=line.split()
        if len(p)<3: continue
        try: L=int(p[1]); es=[int(x) for x in p[2:]]
        except ValueError: continue
    if not es: continue
    nL=max(nL,L+1); trace.append((L,es))
ntok=sum(1 for L,_ in trace if L==0)

class Sim:
    def __init__(s):
        s.v=[OrderedDict() for _ in range(nL)]; s.r=OrderedDict()
        s.cold=[0]*nL; s.ram=[0]*nL; s.vh=[0]*nL
    def acc(s,L,e):
        if e in s.v[L]: s.vh[L]+=1; s.v[L].move_to_end(e)
        elif (L,e) in s.r: s.ram[L]+=1; s.r.move_to_end((L,e))
        else:
            s.cold[L]+=1
            if len(s.r)>=RAM and RAM>0: s.r.popitem(last=False)
            if RAM>0: s.r[(L,e)]=1
        if e not in s.v[L]:
            while len(s.v[L])>=K: s.v[L].popitem(last=False)
            s.v[L][e]=1
s=Sim()
for L,es in trace:
    for e in es: s.acc(L,e)
ws=[set() for _ in range(nL)]
for L,es in trace:
    for e in es: ws[L].add(e)

# full-attention layers: "full_attn_every 4" -> every 4th layer. Mark both plausible phases for reference.
full = set(L for L in range(nL) if (L+1)%4==0)   # 3,7,11,...,47

print(f"trace={path}  {ntok} tokens, {nL} layers, K={K} ram={RAM}")
print(f"\n=== PER-LAYER EXPERT FLOW (per decode token) ===")
print(f"  L   attn   acc  vram  ram  cold   NVMe-in   H2D-in   workset")
tot_cold=tot_h2d=0
for L in range(nL):
    acc=(s.vh[L]+s.ram[L]+s.cold[L])/ntok
    cold=s.cold[L]/ntok; ram=s.ram[L]/ntok; vh=s.vh[L]/ntok
    h2d=(s.cold[L]+s.ram[L])/ntok   # experts staged into VRAM = non-vram-hit accesses
    tot_cold+=cold; tot_h2d+=h2d
    tag="FULL" if L in full else "lin"
    print(f"  {L:>2} {tag:>5} {acc:>5.1f} {vh:>5.1f} {ram:>4.1f} {cold:>5.1f}  {cold*MB:>6.1f}MB {h2d*MB:>6.1f}MB  {len(ws[L]):>5}")
print(f"\n  TOTAL/token: cold={tot_cold:.0f} -> NVMe-in={tot_cold*MB:.0f}MB ; H2D-staged={tot_h2d:.0f} -> {tot_h2d*MB:.0f}MB to VRAM")
print(f"  early L0-3 cold/tok={sum(s.cold[L] for L in range(4))/ntok:.1f}  vs  late L44-47={sum(s.cold[L] for L in range(44,48))/ntok:.1f}")
fc=sum(s.cold[L] for L in full)/ntok; lc=sum(s.cold[L] for L in range(nL) if L not in full)/ntok
print(f"  full-attn layers (12): {fc:.1f} cold/tok ({fc/12:.2f}/layer)  vs linear layers (36): {lc:.1f} ({lc/36:.2f}/layer)")
