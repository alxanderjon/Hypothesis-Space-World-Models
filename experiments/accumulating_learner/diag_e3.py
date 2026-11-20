"""Diagnosis of sub-rung e3 (x/(K+x^2)), as for rung a: every held-out equation left unsolved in a lineage's final round
(counts only): does it hold a new-piece term; does its true structure fit exactly; and does the same search (the
lineage's final prior, the test budget, its seed) ever try the true structure, and how did the prior rank its terms at
the empty start. Read only on the ladder. Usage: diag_e3.py <workers>"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

RUNG = os.environ.get("DIAG_RUNG", "e3")


def job(args):
    rec, i, spec, seed, newset = args
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
    d = len(rec["structure"])
    keys = rec["structure"][i]
    new = any(DR.role_key(k, i, d) in newset for k in keys)
    ctx = SG.system_context(np.array(rec["bursts"]), seed=seed)
    fitter = FG.make_fitter(ctx, seed=seed)
    v = ctx["vocab"]
    if any(k not in v.index for k in keys):
        return {"new": new, "fits": "not in language"}
    S = tuple(sorted(v.index[k] for k in keys))
    exact = MZ.Eq(ctx, fitter, i).node(S)["exact"]
    prior = LD.make_prior(spec)
    budget, cap = LD.TEST_BUDGET
    se = SG.EquationSearchL1(i, ctx, budget, prior, fitter, np.random.default_rng([seed, i]), time_cap=LD.C.TIME_CAP / d,
                             thorough_cap=cap).run()
    tried = S in se.table
    sub = sum(1 for T in se.table if set(T) < set(S) and T)       # proper subsets of the truth the search visited
    # the prior's ranks of the true terms at the empty start
    pr = prior(se.root, se.ctx)
    legal = MZ.legal_adds(v.V, ())
    sc = np.log(np.maximum(np.where(legal, pr.p_swap_add, 0), 1e-300))
    ranks = {v.keys[t]: MZ.rank(sc, legal, t) for t in S}
    new_ranks = [r for k, r in ranks.items() if k in newset]
    of = {k_: lv for lv, ks in LD.piece_keys(d).items() for k_ in ks}
    worst = max(ranks, key=ranks.get)
    return {"new": new, "fits": "fits exactly" if exact else "does not fit exactly", "tried": tried, "subsets_tried": sub,
            "structures": len(se.table), "worst_rank": max(ranks.values()), "new_term_rank": min(new_ranks) if new_ranks else None,
            "pieces": sorted(str(of.get(DR.role_key(k_, i, d), "other")) for k_ in keys),
            "worst_piece": str(of.get(DR.role_key(worst, i, d), "other"))}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    workers = int(sys.argv[1])
    keys = LD.piece_keys(3)
    newset = set(LD.new_keys(RUNG, keys) or [])
    jobs, meta = [], []
    for arm in LD.ARMS:
        for s in LD.SEEDS:
            p = LD.OUT / RUNG / f"{arm}-s{s}.json"
            if not p.exists():
                continue
            st = json.loads((LD.OUT / RUNG / f"{arm}-s{s}.state.json").read_text())
            tables = {int(k): v for k, v in st["tables"].items()}
            spec = ("confidence", st["net"], tables) if arm == "blend" else ("net", st["net"])
            test = {x["id"]: x for x in json.loads((LD.OUT / "gen" / f"{RUNG}-s{s}.json").read_text())["test"]}
            for y in json.loads(p.read_text())["rounds"][-1]["test"]:
                eqs = [int(e["eq"]) for e in y["equations"] if e["first_exact"] is None] if y["status"] == "done" else [0, 1, 2]
                for i in eqs:
                    jobs.append((test[y["id"]], i, spec, s, newset))
                    meta.append((arm, s))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, jobs))
    out = {}
    for arm in LD.ARMS:
        for s in LD.SEEDS:
            R = [r for m, r in zip(meta, res) if m == (arm, s)]
            if not R and not (LD.OUT / RUNG / f"{arm}-s{s}.json").exists():
                continue
            ok = [r for r in R if r["fits"] == "fits exactly"]
            out[f"{arm}-s{s}"] = {
                "unsolved equations": len(R), "with the new piece": sum(r["new"] for r in R),
                "fits": dict(Counter(r["fits"] for r in R)),
                "true structure tried (of those that fit)": f"{sum(r['tried'] for r in ok)} of {len(ok)}",
                "median proper subsets of the truth visited": float(np.median([r["subsets_tried"] for r in ok])) if ok else None,
                "median worst true-term rank at the empty start": float(np.median([r["worst_rank"] for r in ok])) if ok else None,
                "median rank of the new-piece term": float(np.median([r["new_term_rank"] for r in ok if r["new_term_rank"] is not None]))
                if any(r["new_term_rank"] is not None for r in ok) else None}
            out[f"{arm}-s{s}"]["worst-ranked true term's piece"] = dict(Counter(r.get("worst_piece") for r in ok))
            print(f"{arm}-s{s}: {json.dumps(out[f'{arm}-s{s}'])}", flush=True)
    allok = [r for r in res if r.get("fits") == "fits exactly"]
    print("all lineages: worst-ranked true term's piece in unsolved exact equations:", dict(Counter(r["worst_piece"] for r in allok)),
          "; pieces present:", dict(Counter(pc for r in allok for pc in r["pieces"])))
    (LD.OUT / RUNG / "diagnosis.json").write_text(json.dumps({"summary": out, "rows": [dict(r, arm=m[0], seed=m[1]) for m, r in zip(meta, res)]}, indent=1))


if __name__ == "__main__":
    main()
