"""Rung d's generator, proposed (not yet approved): rung c's systems (3 variables, 3 terms per equation, the six pieces)
with exactly one coupling: one term, with its rate, shared by two equations, as a transfer (it leaves one variable and
enters another: coefficient -a in one equation and +a in the other, the same constants K). The shared term counts as
one of the 3 terms in each of the two equations; the rest are drawn as in rung c. Usage (pilot acceptance only, no
ladder change): coupling_gen.py <workers> <n>"""

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "coupling_pilot")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402  (read only)


def draw_coupled(seed, allowed, n_terms, d):
    import numpy as np
    import sympy as sp
    import torch
    import domains_g1 as DG
    import fitter_g1 as FG
    import language_g1 as LG
    rng = DG.rng_of(seed)
    Y, X = DG.Y4, LG.variables(4)
    params = {}
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    actual = {i: [i] + [j for j in range(d) if j != i] for i in range(d)}

    def term(key, i, tag):
        """The term's expression for equation i's frame, its constants drawn and named by tag; returns (expr, keys)."""
        expr = LG.expr_of(key).xreplace({X[r]: Y[actual[i][r]] for r in range(d)})
        inf = FG.info(key)
        sub = {}
        if inf.k:
            Wp = torch.tensor(np.asarray(ic) * rng.uniform(0.5, 1.5, (64, d)))
            th = FG.natural_thetas(inf, rng.random((1, inf.k)), Wp[:, actual[i]])[0]
            if not np.all(np.isfinite(th)):
                th = np.ones(inf.k)
            for a in range(inf.k):
                name = f"c_{tag}_{a}"
                params[name] = float(th[a])
                sub[sp.Symbol(f"K{a}", real=True)] = sp.Symbol(name)
        return expr.xreplace(sub)

    def coef(name):
        params[name] = float(rng.choice([-1.0, 1.0]) * math.exp(rng.uniform(math.log(0.1), math.log(2.0))))
        return sp.Symbol(name)

    # the coupling: a pair of equations, one piece, one rate
    i0, j0 = [int(x) for x in rng.choice(d, size=2, replace=False)]
    lv = int(rng.choice(sorted(allowed)))
    skey = allowed[lv][int(rng.integers(len(allowed[lv])))]
    shared = term(skey, i0, "shared")
    params["a_shared"] = float(math.exp(rng.uniform(math.log(0.1), math.log(2.0))))
    k = sp.Symbol("a_shared")
    rhs = [-k * shared if i == i0 else (k * shared if i == j0 else 0) for i in range(d)]
    base = LG.expr_of(skey).xreplace({X[r]: Y[actual[i0][r]] for r in range(d)})     # the shared term without constants

    def same_as_shared(key, i):
        return i in (i0, j0) and sp.simplify(LG.expr_of(key).xreplace({X[r]: Y[actual[i][r]] for r in range(d)}) - base) == 0
    for i in range(d):
        n = int(rng.integers(n_terms[0], n_terms[1] + 1)) - (1 if i in (i0, j0) else 0)
        picks = []
        while len(picks) < n:
            lv = int(rng.choice(sorted(allowed)))
            key = allowed[lv][int(rng.integers(len(allowed[lv])))]
            if key not in picks and not same_as_shared(key, i):
                picks.append(key)
        for t, key in enumerate(picks):
            rhs[i] = rhs[i] + coef(f"a{i}_{t}") * term(key, i, f"{i}_{t}")
    return {"seed": seed, "domain": "curriculum", "form": "coupled", "d": d, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False, "coupling": {"from": i0, "to": j0, "piece": lv}}


def pilot_job(args):
    prefix, k, allowed, n_terms, d = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    t0, why = time.time(), {}
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw_coupled(seed, allowed, n_terms, d)
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
        s = [eq["keys"] for eq in syn.structure]
        return {"k": k, "accepted": True, "tries": r + 1, "seconds": time.time() - t0, "why": why,
                "n_terms": [len(x) for x in s], "coupling": desc["coupling"]}
    return {"k": k, "accepted": False, "tries": 200, "seconds": time.time() - t0, "why": why}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    workers, n = int(sys.argv[1]), int(sys.argv[2])
    keys = LD.piece_keys(3)
    allowed = {lv: keys[lv] for lv in (1, 2, 3, 5, 6, 7)}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(pilot_job, [("coupling-pilot", k, allowed, (3, 3), 3) for k in range(n)]))
    acc = [r for r in res if r["accepted"]]
    why = {}
    for r in res:
        for k_, v in r["why"].items():
            why[k_] = why.get(k_, 0) + v
    print(f"coupled pilot: {len(acc)} of {n} accepted; median tries {np.median([r['tries'] for r in acc]) if acc else None}; "
          f"median seconds {np.median([r['seconds'] for r in acc]) if acc else None:.0f}; terms per equation "
          f"{sorted({tuple(r['n_terms']) for r in acc})}; coupled pieces {sorted(r['coupling']['piece'] for r in acc)}")
    print("rejections:", dict(sorted(why.items(), key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    main()
