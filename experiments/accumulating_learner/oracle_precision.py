"""Precision failures of true structures (24 Nov): optimizer or data? For a system whose true structure passes the
recordings but misses one step (1e-12) or rollouts (1e-9): (1) the accepted fit refined by shooting (as the check does);
(2) the oracle: the same structure's parameters fitted to the TRUE right-hand side on dense states (recorded states,
the test starts, random states in 1.5x the box), no data involved; (3) the shooting fit started from the oracle.
For each: the shooting residual on the recordings and the one-step error from the 20 test starts. If the oracle has a
clearly lower shooting residual, the optimizer stopped short; if the residuals tie but one step differs, the recordings
cannot pin the constants (a data limit). Training systems only. Usage: oracle_precision.py <workers>"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("G2H_RUN", "ladder")
import ladder as LD                                              # noqa: E402


def job(args):
    rec, s = args
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import sympy as sp
    import torch
    torch.set_num_threads(1)
    from scipy.optimize import least_squares
    import domains_g1 as DG
    import fitter_g1 as FG
    import gen_wide as GW
    import precise_fit as PF
    desc = rec["desc"]
    d = desc["d"]
    syn = GW.rebuild_wide(desc)
    dt = float(syn.ref.sys.dt)
    xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
    f_ = sp.lambdify(xs, DG.numeric_rhs(desc), "numpy")
    ftrue = lambda X: np.stack([np.asarray(v, float) * np.ones(len(X)) for v in f_(*[X[:, j] for j in range(d)])], 1)
    ctx = FG.system_context(rec["bursts"], seed=s)
    fitter = FG.make_fitter(ctx, seed=s)
    sig = np.asarray(ctx["view"]["sig"], float)
    models = []
    for i, e in enumerate(rec["structure"]):
        nd = FG.evaluate_structure(tuple(e["keys"] if isinstance(e, dict) else e), ctx, i, fitter)
        if not nd["exact"]:
            return {"id": rec["id"], "skip": "recordings"}
        models.append((nd["terms"], np.asarray(nd["coefs"], float)))
    tpe = [[(t.key, tuple(t.theta)) for t in terms] for terms, _ in models]
    cpe = [c for _, c in models]
    _, _, r0, r_acc, model, p_acc = PF.precise(tpe, cpe, rec["bursts"], sig, dt)
    B = np.asarray(rec["bursts"], float)
    flat = B.reshape(-1, d)
    starts = np.asarray(rec["test_starts"], float)
    lo, hi = flat.min(0), flat.max(0)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    rng = np.random.default_rng(0)
    keep = (lo > 0) | (hi < 0)
    rnd = c + h * 1.5 * (2 * rng.random((400, d)) - 1)
    rnd = np.where(keep & (np.sign(rnd) != np.sign(c)), c, rnd)
    X = np.concatenate([flat, starts, rnd])
    FT = ftrue(X)
    good = np.all(np.isfinite(FT), 1)
    X, FT = X[good], FT[good]
    scl = np.abs(FT).max(0) + 1e-300

    def r_or(p):
        with np.errstate(all="ignore"):
            return np.nan_to_num(((model.rhs(X, p) - FT) / scl).ravel(), nan=1e6)
    fit = least_squares(r_or, p_acc, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200 * len(p_acc), x_scale="jac")
    p_or = fit.x
    x0, x1 = B[:, :-1].reshape(-1, d), B[:, 1:].reshape(-1, d)
    norm = np.abs(x1 - x0).max(0) + 1e-300

    def shoot_res(p):
        r = (PF.shoot(model, p, x0, dt) - x1) / norm
        return float(np.sqrt(np.mean(np.nan_to_num(r, nan=1e6) ** 2)))

    def one_step(p):
        a, b = model.rhs(starts, p), ftrue(starts)
        return float(max(np.linalg.norm(a[k] - b[k]) / (np.linalg.norm(b[k]) + 1e-300) for k in range(len(starts))))

    def res_vec(p):
        return np.nan_to_num(((PF.shoot(model, p, x0, dt) - x1) / norm).ravel(), nan=1e6)
    fit2 = least_squares(res_vec, p_or, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=60 * len(p_or), x_scale="jac")
    p_sh = fit2.x
    return {"id": rec["id"], "n_params": len(p_acc),
            "accepted": {"shoot": shoot_res(p_acc), "one_step": one_step(p_acc)},
            "oracle": {"shoot": shoot_res(p_or), "one_step": one_step(p_or), "fit_to_truth": float(np.sqrt(np.mean(fit.fun ** 2)))},
            "shoot_from_oracle": {"shoot": shoot_res(p_sh), "one_step": one_step(p_sh)},
            "param_rel_diff_acc_vs_oracle": float(np.max(np.abs(p_acc - p_or) / (np.abs(p_or) + 1e-300)))}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers = int(sys.argv[1])
    dev = [json.loads(l) for l in (LD.OUT.parent / "diag_truth_q43s-dev.jsonl").read_text().splitlines()]
    want = {r["id"] for r in dev if not r["passes"] and r.get("recordings")}
    recs = [(r, s) for s in LD.SEEDS for r in json.loads((LD.OUT / "gen" / f"q43s-s{s}.json").read_text())["train"] if r["id"] in want]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, recs))
    out = LD.OUT.parent / "oracle_precision.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in res))
    for r in res:
        if r.get("skip"):
            print("skip", r["skip"])
            continue
        a, o, so = r["accepted"], r["oracle"], r["shoot_from_oracle"]
        print(f"params {r['n_params']:2d} | accepted: shoot {a['shoot']:.1e} one {a['one_step']:.1e} | oracle: fit-to-truth {o['fit_to_truth']:.1e} "
              f"shoot {o['shoot']:.1e} one {o['one_step']:.1e} | shoot from oracle: shoot {so['shoot']:.1e} one {so['one_step']:.1e} | "
              f"param diff {r['param_rel_diff_acc_vs_oracle']:.1e}")


if __name__ == "__main__":
    main()
