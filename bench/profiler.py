#!/usr/bin/env python3
# Expert cache profiler + offline policy simulator (the "cache profiler" keystone).
# Parses a GATEDUMP trace (per decode token, per layer: which experts the router selected, and the
# engine's residency tier at decision time), builds the behavioral MAP, and replays the access trace
# under alternative cache policies to GATE the build decisions for tiers (5) and prefetch (3).
#
#   profiler.py <gatedump.txt> [K=48] [ram_cap=2048]
#
# A faithful per-layer-VRAM(K) + global-RAM(ram_cap) LRU simulator is validated against the engine's
# recorded tiers first; only then do its policy predictions (static-pin, temporal-prefetch) count.
import sys, os, re, math
from collections import OrderedDict, defaultdict, Counter

gd  = sys.argv[1] if len(sys.argv) > 1 else "bench/results/gatedump_80b.txt"
K   = int(sys.argv[2]) if len(sys.argv) > 2 else 48
RAM = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

# ---- parse: per line "tok layer wk  e:prob:sel:tier ...". Pull only the SELECTED experts (sel==1). ----
sel_re = re.compile(r'(\d+):[^:\s]+:1:(\d)')          # expert_id : prob : selected=1 : residency_tier
trace = defaultdict(dict)                              # trace[tok][layer] = [(expert, engine_tier), ...]
n_expert = 0
for line in open(gd):
    if not line or line[0] == '#': continue
    sp = line.find(' '); sp2 = line.find(' ', sp+1)
    if sp < 0 or sp2 < 0: continue
    try: tok = int(line[:sp]); layer = int(line[sp+1:sp2])
    except ValueError: continue
    picks = [(int(e), int(t)) for e, t in sel_re.findall(line)]
    if not picks: continue
    trace[tok][layer] = picks
    for e, _ in picks:
        if e+1 > n_expert: n_expert = e+1

toks = sorted(trace.keys()); n_layer = max(max(d.keys()) for d in trace.values()) + 1
n_used = max(len(v) for d in trace.values() for v in d.values())
naccess = sum(len(v) for d in trace.values() for v in d.values())
print(f"trace: {len(toks)} decode tokens (t={toks[0]}..{toks[-1]}), {n_layer} layers, {n_expert} experts/layer, "
      f"~{n_used} sel/layer, {naccess} expert-accesses | sim K={K} ram_cap={RAM}")

# ============================ MAP ============================
freq = Counter()                                       # global (layer,expert) -> count
for t in toks:
    for L, picks in trace[t].items():
        for e, _ in picks: freq[(L, e)] += 1
total_slots = n_layer * n_expert
activated = len(freq)
counts = sorted(freq.values(), reverse=True)
cum = 0; share_top1 = 0
top1pct = max(1, activated // 100)
share_top1 = 100 * sum(counts[:top1pct]) / naccess
# per-layer selection entropy (how spread the routing is, in bits; max = log2(n_expert))
layer_ent = []
for L in range(n_layer):
    c = Counter(); n = 0
    for t in toks:
        for e, _ in trace[t].get(L, []): c[e] += 1; n += 1
    if n: H = -sum((v/n)*math.log2(v/n) for v in c.values()); layer_ent.append(H)
print("\n=== MAP ===")
print(f"  experts ever activated : {activated}/{total_slots} ({100*activated/total_slots:.1f}%)  -> dead tail = {100*(1-activated/total_slots):.1f}%")
print(f"  frequency skew         : hottest 1% of activated experts serve {share_top1:.1f}% of accesses")
print(f"  per-layer route entropy: mean={sum(layer_ent)/len(layer_ent):.2f} bits (max possible {math.log2(n_expert):.2f}); "
      f"min={min(layer_ent):.2f} max={max(layer_ent):.2f}  -> higher = more diffuse")

# ============================ SIMULATOR ============================
class Sim:
    def __init__(self, K, ram_cap, pin=None):
        self.K, self.ram_cap = K, ram_cap
        self.vram = [OrderedDict() for _ in range(n_layer)]   # per layer, expert->1, LRU (last=MRU)
        self.ram  = OrderedDict()                              # (layer,expert)->1, global LRU
        self.pin  = pin or [set() for _ in range(n_layer)]    # always-resident experts per layer (cost K slots)
        self.vh = self.rh = self.cold = 0
    def tier(self, L, e):
        if e in self.pin[L] or e in self.vram[L]: return 1
        if (L, e) in self.ram: return 2
        return 0
    def _stage(self, L, e):
        if e in self.vram[L]: self.vram[L].move_to_end(e); return
        cap = self.K - len(self.pin[L])
        while len(self.vram[L]) >= cap: self.vram[L].popitem(last=False)
        self.vram[L][e] = 1
    def prefetch(self, L, e):                                  # bring into RAM ahead of need (no hit counted)
        if e in self.pin[L] or e in self.vram[L] or (L, e) in self.ram: return
        if len(self.ram) >= self.ram_cap: self.ram.popitem(last=False)
        self.ram[(L, e)] = 1
    def access(self, L, e):
        if e in self.pin[L]: self.vh += 1; return
        if e in self.vram[L]: self.vh += 1; self.vram[L].move_to_end(e); return
        if (L, e) in self.ram: self.rh += 1; self.ram.move_to_end((L, e)); self._stage(L, e); return
        self.cold += 1
        if len(self.ram) >= self.ram_cap: self.ram.popitem(last=False)
        self.ram[(L, e)] = 1; self._stage(L, e)

# ---- validate the LRU simulator against the engine's recorded tiers ----
base = Sim(K, RAM); match = tot = 0; eng = Counter(); sim = Counter()
for t in toks:
    for L in range(n_layer):
        picks = trace[t].get(L)
        if not picks: continue
        for e, et in picks:                                   # snapshot tier (pre-fetch), like the engine
            st = base.tier(L, e); tot += 1; eng[et] += 1; sim[st] += 1
            if st == et: match += 1
        for e, _ in picks: base.access(L, e)                  # then fetch/update
print("\n=== SIMULATOR VALIDATION (LRU vs engine-recorded tiers) ===")
print(f"  per-access tier agreement: {100*match/tot:.1f}%   (1=VRAM 2=RAM 0=NVMe)")
print(f"  engine tiers : VRAM={100*eng[1]/tot:.1f}% RAM={100*eng[2]/tot:.1f}% NVMe={100*eng[0]/tot:.1f}%")
print(f"  sim    tiers : VRAM={100*sim[1]/tot:.1f}% RAM={100*sim[2]/tot:.1f}% NVMe={100*sim[0]/tot:.1f}%")
base_cold = base.cold
print(f"  LRU baseline cold (NVMe) reads: {base_cold}  ({base_cold/len(toks):.1f}/token)")

# ============================ GATE 5: residency tiers (static-pin vs LRU) ============================
# Pin the top-N most frequent experts/layer, learned from the FIRST HALF, evaluated on the SECOND HALF
# (honest generalization, not oracle). Reduces effective LRU capacity but guarantees the hot head.
half = toks[len(toks)//2]
train_freq = defaultdict(Counter)
for t in toks:
    if t >= half: break
    for L, picks in trace[t].items():
        for e, _ in picks: train_freq[L][e] += 1
test_toks = [t for t in toks if t >= half]
def cold_on_test(pin):
    s = Sim(K, RAM, pin=pin)
    # warm the cache on the train half first so the test isn't cold-start biased
    for t in toks:
        if t >= half: break
        for L in range(n_layer):
            for e, _ in trace[t].get(L, []): s.access(L, e)
    s.vh = s.rh = s.cold = 0
    for t in test_toks:
        for L in range(n_layer):
            for e, _ in trace[t].get(L, []): s.access(L, e)
    return s.cold
lru_test = cold_on_test([set() for _ in range(n_layer)])
print("\n=== GATE 5 (residency tiers): static-pin top-N/layer vs LRU, train=1st half eval=2nd half ===")
print(f"  {'policy':<26}{'cold reads (test)':>18}{'vs LRU':>12}")
print(f"  {'pure LRU':<26}{lru_test:>18}{'—':>12}")
for N in (4, 8, 16):
    pin = [set(e for e, _ in train_freq[L].most_common(N)) for L in range(n_layer)]
    c = cold_on_test(pin)
    print(f"  {'pin top-'+str(N)+'/layer + LRU':<26}{c:>18}{100*(lru_test-c)/lru_test:>+11.1f}%")

# ============================ GATE 3: predictive prefetch ============================
# Predictability signal: does token t reuse token t-1's experts at the same layer? (temporal locality)
overlap_num = overlap_den = 0
for i, t in enumerate(toks[1:], 1):
    tp = toks[i-1]
    for L in range(n_layer):
        cur = set(e for e, _ in trace[t].get(L, [])); prev = set(e for e, _ in trace[tp].get(L, []))
        if cur: overlap_num += len(cur & prev); overlap_den += len(cur)
print("\n=== GATE 3 (predictive prefetch): temporal locality + simulated upper bound ===")
print(f"  token-to-token expert reuse (same layer): {100*overlap_num/overlap_den:.1f}% of a token's experts were also used by the previous token")
# Simulate LRU + oracle-timing temporal prefetch: before token t, prefetch token t-1's experts/layer.
s = Sim(K, RAM);
for L in range(n_layer):                                   # process token-by-token so we can prefetch from t-1
    pass
sp = Sim(K, RAM); prev_layer_picks = {}
for i, t in enumerate(toks):
    for L in range(n_layer):
        if i > 0:
            for e in prev_layer_picks.get(L, ()): sp.prefetch(L, e)   # prefetch last token's experts (assume completes during compute)
        for e, _ in trace[t].get(L, []): sp.access(L, e)
    prev_layer_picks = {L: [e for e, _ in trace[t].get(L, [])] for L in range(n_layer)}
print(f"  LRU cold reads                : {base_cold}")
print(f"  LRU + temporal-prefetch (UB)  : {sp.cold}   ({100*(base_cold-sp.cold)/base_cold:+.1f}% cold reads)")
print(f"  (upper bound: assumes every prefetch finishes in the compute window; real gain <= this)")

# ============================ GATE 2: co-usage / clustering ============================
# Within a layer the n_used selected experts co-fire. Do the SAME groups recur (clusterable), or is each
# token's set fresh? Measure set stability = mean Jaccard(token t, t-1) per layer (== prefetch signal),
# and the recurrence of the single most common selected-set per layer.
jac_n = jac_d = 0
for i, t in enumerate(toks[1:], 1):
    tp = toks[i-1]
    for L in range(n_layer):
        a = frozenset(e for e, _ in trace[t].get(L, [])); b = frozenset(e for e, _ in trace[tp].get(L, []))
        if a or b: jac_n += len(a & b)/len(a | b); jac_d += 1
setrep = []
for L in range(n_layer):
    c = Counter(frozenset(e for e, _ in trace[t].get(L, [])) for t in toks if L in trace[t])
    if c: setrep.append(c.most_common(1)[0][1] / sum(c.values()))
print("\n=== GATE 2 (co-usage clustering) ===")
print(f"  per-layer selected-set stability (mean Jaccard t vs t-1): {jac_n/jac_d:.3f}")
print(f"  most-common exact set recurrence/layer (mean)          : {100*sum(setrep)/len(setrep):.1f}%")
print(f"  NOTE: block-size sweep showed the drive is read-size-insensitive (3.5 GB/s at 512KB == 8MB),")
print(f"        so contiguous layout buys ~0 bandwidth; clustering only matters as a prefetch enabler.")
