"""The K rule's quick check (owner, 19 November; research log "Owner: the K rule", where the pass rule was set before the
run). For each piece with K inside (1/(K+x), x/(K+x), 1/(K+x^2), x/(K+x^2), Hill x^2/(K+x^2)) and each rule (old: the
natural-range rule; new: positive K between the 25th and 90th percentiles of the shifted quantity, checked on the
recordings), 60 rung-e-style development systems (3 variables, 3 terms, every system with a term of the piece), then for
every true term of the piece (counts only), no search, no training:
  q = max shifted quantity / |K| over the recordings, in the learner's coordinates (the bend is reached when q >= 1);
  the best single look-alike column's R^2 against the term's column (same variable, the other pieces);
  at one short, whether a same-variable look-alike has stronger evidence than the true term.
Usage: k_check.py <workers> <gen|diag|report>"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "k_check")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

OUT = LD.OUT
D, N, SEEDS = 3, 20, (1, 2, 3)
PIECES = {5: "1/(K+x)", 6: "x/(K+x)", 7: "1/(K+x^2)", 8: "x/(K+x^2)", 10: "x^2/(K+x^2)"}
LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
LOOK = (1, 3, 4, 5, 6, 7, 8, 10)                                 # single-variable pieces a term can be confused with


def variables(key):
    return sorted({int(m) for m in re.findall(r"Symbol\('x(\d)'", key)})


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    keys = LD.piece_keys(D)
    allowed = {lv: keys[lv] for lv in LEVELS}
    old = LD.earlier_keys("kcheck")
    for p in (LD.G2H / "ladder" / "gen").glob("*.json"):
        old |= {LD.data_key(r["bursts"]) for sp in json.loads(p.read_text()).values() for r in sp}
    for lv in PIECES:
        for rule in (False, True):
            p = OUT / "gen" / f"dev-L{lv}-{'new' if rule else 'old'}.json"
            if p.exists():
                continue
            t0 = time.time()
            # in small fresh pools: builds load compiled libraries that are never unloaded
            res = []
            jobs = [(f"kcheck-L{lv}-{'new' if rule else 'old'}-s{s}", k, allowed, (3, 3), D, keys[lv], rule) for s in SEEDS for k in range(N)]
            for c in range(0, len(jobs), 60):
                with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                    res += list(ex.map(LD.gen_job, jobs[c:c + 60]))
            acc = [dict(r, id=f"kcheck-L{lv}-{'new' if rule else 'old'}-{n}") for n, r in enumerate(res) if r["accepted"]]
            clash = sum(LD.data_key(r["bursts"]) in old for r in acc)
            why = {}
            for r in res:
                for w in (r.get("reasons") or []):
                    why[w.split(":")[0][:40]] = why.get(w.split(":")[0][:40], 0) + 1
            print(f"{PIECES[lv]} {'new' if rule else 'old'} rule: {len(acc)} of {len(jobs)} accepted in {time.time() - t0:.0f}s; median tries "
                  f"{sorted(r['tries'] for r in acc)[len(acc) // 2] if acc else None}; identical to earlier {clash}; last rejections {why}", flush=True)
            if clash:
                raise SystemExit("not new")
            p.write_text(json.dumps({"accepted": len(acc), "of": len(jobs), "tries": [r["tries"] for r in acc], "systems": acc}))


def diag_job(args):
    rec, lv = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import fitter_g1 as FG
    import muzero_checks as MZ
    import search_g1 as SG
    of = {k: l_ for l_, ks in LD.piece_keys(D).items() for k in ks}
    ctx, fitter = MZ.system(rec)
    v, W = ctx["vocab"], ctx["W"]
    out = []
    for i, true_keys in enumerate(rec["structure"]):
        if any(k not in v.index for k in true_keys):
            continue
        mine = [k for k in true_keys if of.get(DR.role_key(k, i, D)) == lv]
        if not mine:
            continue
        S = tuple(sorted(v.index[k] for k in true_keys))
        eq = MZ.Eq(ctx, fitter, i)
        node = eq.node(S)
        if not node["exact"]:
            out.append({"exact": False})
            continue
        for k in mine:
            t = v.index[k]
            term = [u for u in node["terms"] if u.key == k][0]
            K = float(term.theta[0])
            j = LD.shifted_var(k) or variables(k)[0]
            w = W[:, j - 1]
            qty = w.abs() if LD.K_SHIFT[lv] == "x" else w ** 2
            q = float(qty.max()) / abs(K)
            col = FG.value(term, W)
            c = col / (col.norm() + 1e-300)
            best = 0.0
            for u in range(v.V):
                ku = v.keys[u]
                lu = of.get(DR.role_key(ku, i, D))
                if u == t or lu not in LOOK or lu == lv or variables(ku) != [j] or ku not in ctx["bank"].bank:
                    continue
                C = ctx["bank"].bank[ku][0]
                C = C / C.norm(dim=0, keepdim=True)
                best = max(best, float(((C.T @ c) ** 2).max()))
            Sp = tuple(x for x in S if x != t)
            gain, _ = SG.evidence(eq.node(Sp), eq.se.ctx)
            gain = np.nan_to_num(np.asarray(gain), nan=0.0)
            look = [u for u in range(v.V) if u != t and u not in Sp and j in variables(v.keys[u])
                    and of.get(DR.role_key(v.keys[u], i, D)) in LOOK and of.get(DR.role_key(v.keys[u], i, D)) != lv]
            out.append({"exact": True, "K": K, "q": q, "lookalike_R2": best, "stronger": any(gain[u] > gain[t] for u in look),
                        "rank": int((gain > gain[t]).sum()) + 1})
    return out


def step_diag(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs, meta = [], []
    for lv in PIECES:
        for rule in ("old", "new"):
            g = json.loads((OUT / "gen" / f"dev-L{lv}-{rule}.json").read_text())
            jobs += [(r, lv) for r in g["systems"]]
            meta += [(lv, rule)] * len(g["systems"])
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(diag_job, jobs))
    rows = [dict(x, piece=m[0], rule=m[1]) for m, part in zip(meta, res) for x in part]
    (OUT / "diag.json").write_text(json.dumps(rows))
    print(f"diag: {len(rows)} rows")


def step_report():
    import numpy as np
    rows = json.loads((OUT / "diag.json").read_text())
    lines = ["== K rule check: per piece and rule, accepted systems; terms; q = max shifted quantity / |K| (share >= 1); best single "
             "look-alike R^2 (median, share > 0.999); look-alike stronger at one short; true term's evidence rank at one short (median)"]
    verdict = {}
    for lv, name in PIECES.items():
        for rule in ("old", "new"):
            g = json.loads((OUT / "gen" / f"dev-L{lv}-{rule}.json").read_text())
            R = [r for r in rows if r["piece"] == lv and r["rule"] == rule and r["exact"]]
            ne = sum(1 for r in rows if r["piece"] == lv and r["rule"] == rule and not r["exact"])
            if not R:
                lines.append(f"   {name:>11} {rule}: accepted {g['accepted']}/{g['of']}; no exact terms ({ne} true structures not exact)")
                continue
            q = np.array([r["q"] for r in R])
            lb = np.array([r["lookalike_R2"] for r in R])
            st = np.mean([r["stronger"] for r in R])
            verdict[(lv, rule)] = (g["accepted"] / g["of"], float(np.mean(q >= 1)), float(np.median(lb)), float(st))
            lines.append(f"   {name:>11} {rule}: accepted {g['accepted']}/{g['of']} (median tries {int(np.median(g['tries']))}); {len(R)} terms "
                         f"({ne} not exact); q median {np.median(q):.2f}, >= 1 in {np.mean(q >= 1):.0%}; look-alike R^2 median {np.median(lb):.4f}, "
                         f"> 0.999 in {np.mean(lb > 0.999):.0%}; look-alike stronger at one short {st:.0%}; rank median {np.median([r['rank'] for r in R]):.0f}")
    v8 = verdict.get((8, "new"))
    ok8 = bool(v8) and v8[0] >= 0.9 and v8[1] >= 0.75 and v8[2] <= 0.99 and v8[3] <= 0.15
    lines.append(f"== x/(K+x^2) under the new rule: {'PASSES' if ok8 else 'does not pass'} (needs accepted >= 90%, q >= 1 in >= 75%, "
                 f"look-alike R^2 median <= 0.99, look-alike stronger at one short <= 15%)")
    harm = [PIECES[lv] for lv in PIECES if (lv, "new") in verdict and (lv, "old") in verdict
            and (verdict[(lv, "new")][0] < 0.9 or verdict[(lv, "new")][3] > verdict[(lv, "old")][3] + 0.05)]
    lines.append(f"== one rule for every K piece: {'yes' if not harm else 'no: the new rule is worse for ' + ', '.join(harm)}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen": lambda: step_gen(workers), "diag": lambda: step_diag(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
