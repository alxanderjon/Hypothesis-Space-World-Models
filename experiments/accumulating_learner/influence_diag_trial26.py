"""Trial 26 exploratory diagnostic (not pre-registered): does the learned guide predict WHO AFFECTS WHOM on systems
with never-seen building blocks better than the statistical screen and the scrambled-lesson guide?

True influence of variable j on component i is measured numerically from the true vector field (evaluation only):
max over attractor states of |df_i/dx_j| * s_j / max|f_i|, by central differences, thresholded at 1e-6.
No equations are parsed or printed. Predictions are the ones the arms actually used (stored in the test records)."""

import json

import numpy as np

import exact_harness as H
from trial26 import G_FAMILIAR, G_NOVEL, MS, OUT


def true_influence(ref, n=60, rel=1e-6):
    xs = ref.held_states(n)
    d = xs.shape[1]
    f0 = ref.sys.f(xs)
    fscale = np.abs(f0).max(0) + 1e-300
    J = np.zeros((d, d))
    for j in range(d):
        h = 1e-5 * ref.scale[j]
        e = np.zeros(d); e[j] = h
        df = (ref.sys.f(xs + e) - ref.sys.f(xs - e)) / (2 * h)
        J[:, j] = np.abs(df).max(0) * ref.scale[j] / fscale
    return (J > rel).astype(int)


def one(args):
    name, seed = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    p64 = OUT / f"test_s{seed}" / name / "M64_scratch0.json"
    if p64.exists() and json.loads(p64.read_text()).get("status") in ("reference_timeout", "reference_error"):
        return name, {"skipped": "reference not available"}
    try:
        ref = H.Reference(name, seed=seed, traj_steps=300, burn_periods=10)   # light reference: attractor states only
        return name, {"T": true_influence(ref).tolist()}
    except Exception as e:
        return name, {"error": repr(e)[:200]}


def main(seed=1, workers=8):
    from multiprocessing import get_context
    names = G_NOVEL + G_FAMILIAR
    with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        truth = dict(pool.map(one, [(n, seed) for n in names]))
    rows = {}
    for name in names:
        if "T" not in truth[name]:
            rows[name] = truth[name]
            continue
        T = np.asarray(truth[name]["T"])
        out = {"group": "novel" if name in G_NOVEL else "familiar"}
        for arm in ("full45", "scrambled45", "stats0", "full8", "full24"):
            acc = {}
            for M in MS:
                p = OUT / f"test_s{seed}" / name / f"M{M}_{arm}.json"
                if not p.exists():
                    continue
                g = (json.loads(p.read_text()).get("guidance") or {}).get("influence")
                if g is None:
                    continue
                P = (np.asarray(g) >= 0.5).astype(int)
                acc[M] = float((P == T).mean())
            out[arm] = acc
        rows[name] = out
        print(name, {k: v for k, v in out.items() if k != "group"}, flush=True)
    (OUT / f"influence_diag_s{seed}.json").write_text(json.dumps(rows, indent=1))
    for grp in ("novel", "familiar"):
        print(grp)
        for arm in ("full45", "scrambled45", "stats0", "full8", "full24"):
            for M in MS:
                v = [r[arm][M] for r in rows.values() if r.get("group") == grp and M in r.get(arm, {})]
                if v:
                    print(f"   {arm:12s} M={M:2d}: mean entry accuracy {np.mean(v):.3f} over {len(v)} systems")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
