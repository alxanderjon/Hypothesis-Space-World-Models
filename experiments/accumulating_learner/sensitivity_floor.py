"""The judgeable floor (24 Nov): how far the true law itself moves on the solved checks when its constants are perturbed
by one part in 1e14 (about the best any fit to the recordings can do: they are converged to about 1e-14 per step).
For each system: the one-step and rollout errors, from its 20 test starts over the horizon, of the true system with every
numeric constant multiplied by (1 + 1e-14 z), z ~ N(0, 1), against the unperturbed truth (median of 5 draws). A system
whose floor exceeds a bar cannot be judged solved at float64 precision by any fitter. Training systems for diagnosis;
held-out systems for counts only. Usage: sensitivity_floor.py <workers> <rung> <train|test>"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("G2H_RUN", "ladder")
import ladder as LD                                              # noqa: E402

REL = 1e-14


def job(rec):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import sympy as sp
    from scipy.integrate import solve_ivp
    import domains_g1 as DG
    import synthetic28 as SY
    desc = rec["desc"]
    d = desc["d"]
    xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
    rhs = [sp.sympify(e) for e in DG.numeric_rhs(desc)]
    f0 = sp.lambdify(xs, rhs, "numpy")
    starts = [np.asarray(x, float) for x in rec["test_starts"]]
    B = np.asarray(rec["bursts"], float).reshape(-1, d)
    scale = max(float(np.abs(B).max()), 1e-12)

    def F(f):
        return lambda t, x: np.array([float(v) for v in f(*x)])
    T = float(rec["horizon"])
    tt = np.linspace(0, T, 201)
    ref = []
    for x0 in starts:
        s = solve_ivp(F(f0), (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14 * scale, t_eval=tt)
        ref.append(s.y if s.success else None)
    rng = np.random.default_rng(LD.stable(rec["id"], "sensitivity"))
    ones, rolls = [], []
    for _ in range(5):
        pert = [e.xreplace({c: c * (1 + REL * rng.standard_normal()) for c in e.atoms(sp.Float)}) for e in rhs]
        f1 = sp.lambdify(xs, pert, "numpy")
        one = max(np.linalg.norm(np.array(f1(*x)) - np.array(f0(*x))) / (np.linalg.norm(np.array(f0(*x))) + 1e-300) for x in starts)
        roll = 0.0
        for x0, y0 in zip(starts, ref):
            if y0 is None:
                continue
            s = solve_ivp(F(f1), (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14 * scale, t_eval=tt)
            roll = max(roll, float(np.abs(s.y - y0).max() / (np.abs(y0).max() + 1e-300)) if s.success and s.y.shape == y0.shape else np.inf)
        ones.append(one)
        rolls.append(roll)
    return {"id": rec["id"], "one_step_floor": float(np.median(ones)), "rollout_floor": float(np.median(rolls))}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers, rung, split = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    recs = [r for s in LD.SEEDS for r in json.loads((LD.OUT / "gen" / f"{rung}-s{s}.json").read_text())[split]]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, recs))
    path = LD.OUT.parent / f"sensitivity_{rung}_{split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in res))
    import numpy as np
    o = np.array([r["one_step_floor"] for r in res]); ro = np.array([r["rollout_floor"] for r in res])
    print(f"{rung} {split}: {len(res)} systems; one-step floor quantiles (50/90/max) {np.percentile(o, [50, 90, 100])}; "
          f"rollout floor {np.percentile(ro, [50, 90, 100])}; floor above the bars: one step {(o > 1e-12).sum()}, rollout {(ro > 1e-9).sum()}")


if __name__ == "__main__":
    main()
