#!/usr/bin/env python3
# Stage 1 (DB-reframe plan): residency-substitution CEILING from the GATEDUMP trace.
# Softmax over n_expert is tiny-valued, so "near-tie" is RELATIVE: a resident non-selected expert
# qualifies as a substitute for a cold pick if its prob >= alpha*wk (carries >= alpha of the
# selection-threshold weight). Ceiling = fraction of cold (NVMe) picks that have a resident
# substitute, per (decode token, layer), over the warm tail.
#   gate: >=25% -> build Stage 2 ; 10-25% -> lowered bar ; <10% -> CEILING KILL.
import sys, os, csv as _csv, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench", "results", "gatedump_80b.txt")
csvp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "bench", "results", "gate_csv_80b.csv")

# parse compactly: per (tok,layer) keep wk, n_cold, and the list of resident non-selected probs
recs = []  # (tok, layer, wk, n_cold, [resident_nonselected_probs])
for line in open(gd):
    if line.startswith('#') or not line.strip(): continue
    p = line.split()
    if len(p) < 3: continue
    tok, layer, wk = int(p[0]), int(p[1]), float(p[2])
    n_cold = 0; res_cands = []
    for e in p[3:]:
        a = e.split(':')
        if len(a) != 4: continue
        prob, sel, res = float(a[1]), int(a[2]), int(a[3])
        if sel == 1 and res == 0: n_cold += 1                 # selected & NVMe = a cold read
        elif sel == 0 and res in (1, 2): res_cands.append((prob, res))  # resident non-selected = a substitution candidate
    recs.append((tok, layer, wk, n_cold, res_cands))

maxtok = max(r[0] for r in recs)
nlayer = max(r[1] for r in recs) + 1
print(f"gatedump: {len(recs)} (tok,layer) rows, decode tokens 0..{maxtok}, {nlayer} layers")

# validation: cold/token vs CSV decode-only
cpt = {}
for tok, layer, wk, nc, rc in recs: cpt[tok] = cpt.get(tok, 0) + nc
gd_cold = sum(v for t, v in cpt.items() if t >= 1) / max(1, len([t for t in cpt if t >= 1]))
try:
    cr = list(_csv.DictReader(open(csvp, encoding='utf-8-sig')))
    nv1, nvN = float(cr[0]['nvme_fall']), float(cr[-1]['nvme_fall']); ng = float(cr[-1]['tok']) - float(cr[0]['tok'])
    print(f"VALIDATION cold-reads/token: gatedump={gd_cold:.1f}  CSV decode-only={(nvN-nv1)/ng:.1f}  (should agree)")
except Exception as ex:
    print(f"CSV cross-check skipped: {ex}")

def sweep(tail, label):
    win = [r for r in recs if r[0] >= tail]
    ncold = sum(r[3] for r in win)
    print(f"\n=== ceiling ({label}, tokens>={tail}): {ncold} cold picks; substitute if resident non-selected prob >= alpha*wk ===")
    print(f"{'RBANK':<10}" + "".join(f"{'a='+str(a):>9}" for a in [0.99, 0.9, 0.75, 0.5, 0.25]))
    for rbn, rb in [("VRAM", {1}), ("VRAM|RAM", {1, 2})]:
        cells = []
        for a in [0.99, 0.9, 0.75, 0.5, 0.25]:
            tc = ts = 0
            for tok, layer, wk, nc, rc in win:
                if nc == 0: continue
                ncand = sum(1 for prob, res in rc if res in rb and prob >= a * wk)
                tc += nc; ts += min(nc, ncand)
            cells.append(100 * ts / tc if tc else 0)
        print(f"{rbn:<10}" + "".join(f"{c:>8.1f}%" for c in cells))

sweep(0, "ALL decode")
sweep(maxtok // 2, "warm half")
sweep(3 * maxtok // 4, "warmest quarter")

# near-tie structure: for cold (tok,layer), how close (relatively) is the best resident non-selected expert to wk
print("\n=== near-tie structure (warm half): best resident non-selected cand as a fraction of wk ===")
fr = []
for tok, layer, wk, nc, rc in recs:
    if tok < maxtok // 2 or nc == 0 or not rc: continue
    fr.append(max(prob for prob, res in rc) / wk if wk > 0 else 0)
if fr:
    fr.sort()
    print(f"  best-resident-cand / wk : median={statistics.median(fr):.3f}  p75={fr[3*len(fr)//4]:.3f}  p90={fr[min(len(fr)-1,9*len(fr)//10)]:.3f}  max={fr[-1]:.3f}")
    for a in [0.99, 0.9, 0.75, 0.5]:
        print(f"    fraction of cold (tok,layer) with a resident cand >= {a}*wk : {100*sum(1 for x in fr if x>=a)/len(fr):.1f}%")
