"""Rung e generator pilot (proposal, not approved): rung c's systems (3 variables, 3 terms per equation) over the mastered
pieces plus one new piece, with at least one term of the new piece per system (in a random equation). Acceptance and
cost only. Usage: rung_e_pilot.py <workers> <n per sub-rung>"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "rung_e_pilot")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402  (read only)

CONST = "Integer(1)"
NEW = 100                                                        # the new piece's group in the allowed pools


def new_piece_keys(d):
    """{name: keys} for the four pieces the ladder has not covered (curriculum levels 4 and 8, the constant, Hill)."""
    import search_g1 as SG
    pk = LD.piece_keys(d)
    order = SG.vocabulary_order(d)
    hill = [k for k in order if k.startswith("Mul(Pow(Symbol") and "Integer(2)), Pow(Add(Symbol('K0', real=True), Pow(" in k
            and k.count("Symbol('x") == 2 and k.split("Pow(Symbol(")[1][:4] == k.split("Pow(Symbol(")[2][:4]]
    return {"constant": [CONST], "x^3": pk[4], "x/(K+x^2)": pk[8], "x^2/(K+x^2)": hill}


def pilot_job(args):
    prefix, k, allowed, name, d = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    new = set(new_piece_keys(d)[name])
    t0, why = time.time(), {}
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = LD.draw(seed, {**allowed, NEW: sorted(new)}, (3, 3), d)
        w = DR.prescreen(desc)
        if w is not None:
            why["pre-screen"] = why.get("pre-screen", 0) + 1
            continue
        try:
            syn = DG.build(desc)
        except Exception as ex:
            key = str(ex)[:40]
            why[key] = why.get(key, 0) + 1
            continue
        hits = sum(DR.role_key(k_, i, d) in new for i, eq in enumerate(syn.structure) for k_ in eq["keys"])
        if hits == 0:                                            # at least one term of the new piece per system
            why["no new-piece term"] = why.get("no new-piece term", 0) + 1
            continue
        return {"k": k, "accepted": True, "tries": r + 1, "seconds": time.time() - t0, "why": why, "new_terms": hits}
    return {"k": k, "accepted": False, "tries": 200, "seconds": time.time() - t0, "why": why}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    workers, n = int(sys.argv[1]), int(sys.argv[2])
    keys = LD.piece_keys(3)
    base = {lv: keys[lv] for lv in (1, 2, 3, 5, 6, 7)}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name in ("constant", "x^3", "x/(K+x^2)", "x^2/(K+x^2)"):
            res = list(ex.map(pilot_job, [(f"rung-e-pilot-{name}", k, base, name, 3) for k in range(n)]))
            acc = [r for r in res if r["accepted"]]
            why = {}
            for r in res:
                for k_, v in r["why"].items():
                    why[k_] = why.get(k_, 0) + v
            print(f"{name}: {len(acc)} of {n} accepted; median draws {np.median([r['tries'] for r in acc]) if acc else None}; median seconds "
                  f"{np.median([r['seconds'] for r in acc]) if acc else float('nan'):.0f}; new-piece terms per system "
                  f"{np.mean([r['new_terms'] for r in acc]) if acc else float('nan'):.1f}; top rejections "
                  f"{dict(sorted(why.items(), key=lambda x: -x[1])[:4])}", flush=True)


if __name__ == "__main__":
    main()
