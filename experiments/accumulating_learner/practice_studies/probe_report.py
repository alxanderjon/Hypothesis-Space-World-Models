# Summary of the development-half accumulation probe (probe_accum.py output).
import json, glob, collections
import numpy as np

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
plan = json.load(open(f"{SP}/probe/plan.json"))
order = plan["order"]
N = len(order)
recs = {}
for p in glob.glob(f"{SP}/probe/*_M*_*.json"):
    r = json.load(open(p))
    recs[(r["name"], r["M"], r["arm"])] = r
print(f"{len(recs)} records; library grows to {len(plan['final_library'])} entries over {N} systems")
for M in (8, 16, 32):
    for half, sel in (("all", range(N)), ("first half", range(N // 2)), ("second half", range(N // 2, N))):
        names = [order[k] for k in sel]
        row = {}
        for arm in ("scratch", "library", "decoy"):
            rs = [recs.get((n, M, arm)) for n in names]
            rs = [r for r in rs if r is not None]
            if rs:
                row[arm] = (sum(bool(r.get("exact")) for r in rs), len(rs))
        txt = "  ".join(f"{a} {e}/{n}" for a, (e, n) in row.items())
        print(f"M={M:2d} {half:12s} {txt}")
# paired comparison (same system, same M) where both arms exist
for arm in ("library", "decoy"):
    wins = losses = 0
    for (n, M, a), r in recs.items():
        if a != arm:
            continue
        s = recs.get((n, M, "scratch"))
        if s is None:
            continue
        wins += bool(r.get("exact")) and not s.get("exact")
        losses += bool(s.get("exact")) and not r.get("exact")
    print(f"{arm} vs scratch (paired): {arm} exact & scratch not = {wins}; scratch exact & {arm} not = {losses}")
used = [(k, r["reused"]) for k, r in recs.items() if k[2] == "library" and r.get("reused")]
print("library runs that used a reused piece:", len(used), [(k[0][-4:], k[1], v, recs[k].get("exact")) for k, v in used][:20])
