"""One-shot confirmation run (owner-approved 29 October). Protocol: docs/research-log.md, "Proposed confirmation run".

Frozen models only; no retraining, no changes after looking.
  python make_data.py --confirmation --out ../../outputs/accumulating_learner/confirmation/data
  python confirm.py eval --device cuda:0
  python confirm.py analyze
"""

import argparse
import json
import math

import numpy as np
import sympy as sp
import torch
from scipy.stats import spearmanr

import systems as S
from trial1 import K_GRID, N_GRID, SEEDS, episode, poly_features, ridge_loo, rff_eval, scratch_mlp_eval
from trial3 import Shared
from trial3 import OUT as TRIAL3
from trial4 import Library, krr_with_score
from trial4 import OUT as TRIAL4
from trial5 import OUT as TRIAL5
from trial5 import library_fb

OUT = S.ROOT / "outputs/accumulating_learner/confirmation"
DATA = OUT / "data"
PRIMARY = "seqreplay400"


def confirmation_names():
    return S.eligible_split()["confirmation"]


def available():
    return [n for n in confirmation_names() if (DATA / f"{n}.npz").exists()]


def load_conf(name, dev):
    z = np.load(DATA / f"{name}.npz")
    return torch.tensor(z["x"], device=dev), torch.tensor(z["dx"], device=dev)


def lib_from(path, dev):
    lib = Library().to(dev)
    lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return lib.eval()


def evaluate(args):
    dev = args.device
    models = {}
    for t in N_GRID:
        for s in SEEDS:
            models[f"{PRIMARY}_N{t:02d}_s{s}"] = ("libfb", lib_from(TRIAL5 / "learners" / f"{PRIMARY}_N{t:02d}_s{s}.pt", dev))
            models[f"jointlibfb_N{t:02d}_s{s}"] = ("libfb", lib_from(TRIAL4 / "learners" / f"library_N{t:02d}_s{s}.pt", dev))
            w = Shared().to(dev)
            w.load_state_dict(torch.load(TRIAL3 / "learners" / f"wide_N{t:02d}_s{s}.pt", map_location=dev, weights_only=True))
            models[f"wide_N{t:02d}_s{s}"] = ("shared", w.eval())
    rows = []
    for i, name in enumerate(confirmation_names()):
        if not (DATA / f"{name}.npz").exists():
            continue
        x, dx = load_conf(name, dev)
        for K in K_GRID:
            gen = torch.Generator().manual_seed(6000 + 97 * i + K)
            eps = [episode(x[j:j + 1], dx[j:j + 1], K, gen, dev) for j in range(x.shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            res = {}
            with torch.no_grad():
                krr = krr_with_score(uc, yc, uq, yq)
                res["krr"] = krr[0]
                res["rff"] = rff_eval(uc, yc, uq, yq, torch.Generator().manual_seed(9500 + 97 * i + K))
                res["poly2"] = ridge_loo(poly_features(uc, 2), yc, poly_features(uq, 2), yq)
                res["poly3"] = ridge_loo(poly_features(uc, 3), yc, poly_features(uq, 3), yq)
                for tag, (kind, m) in models.items():
                    if kind == "libfb":
                        res[tag] = library_fb(m, uc, yc, uq, yq, krr)[0]
                    else:
                        res[tag] = ridge_loo(m(uc, None), yc, m(uq, None), yq)
            res["scratch_mlp"] = scratch_mlp_eval(uc, yc, uq, yq, seed=31 * i + K + 1)
            for m, v in res.items():
                rows.append({"system": name, "K": K, "method": m, "episodes": v.float().cpu().tolist()})
            print(name, K, {m: round(float(v.median()), 4) for m, v in res.items()
                            if m in ("krr", "poly2", f"{PRIMARY}_N51_s1", f"{PRIMARY}_N00_s1")}, flush=True)
    (OUT / "eval_confirmation.json").write_text(json.dumps({"rows": rows}))


def is_polynomial(name):
    s = S.System(name, allow_confirmation=True)
    x, y, z = sp.symbols("x y z")
    try:
        vals = [s.params[k] for k in sorted(s.params)]
        out = s._rhs(x, y, z, 0.0, *[sp.Float(float(v)) if sp.sympify(v).is_number else v for v in vals])
        return all(sp.sympify(o).is_polynomial(x, y, z) for o in out)
    except Exception:
        return False


def analyze(args):
    rows = json.loads((OUT / "eval_confirmation.json").read_text())["rows"]
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    systems = sorted({r["system"] for r in rows})
    n = len(systems)
    poly = {s: is_polynomial(s) for s in systems}
    agg = lambda m, K, names=systems: float(np.median([score[(s, K, m)] for s in names]))
    thr = lambda m, s: next((K for K in K_GRID if score[(s, K, m)] <= 0.1), np.inf)
    out = {"n_systems": n, "systems_polynomial": poly, "per_seed": {}}
    for seed in SEEDS:
        L = lambda t, arm=PRIMARY: f"{arm}_N{t:02d}_s{seed}"
        ctrls = ["rff", "krr", L(0)]
        curve = [agg(L(t), 32) for t in N_GRID]
        rho = spearmanr(N_GRID, curve).correlation
        logr = np.log([score[(s, 32, L(51))] / score[(s, 32, L(0))] for s in systems])
        rng = np.random.default_rng(seed)
        boot = [np.median(rng.choice(logr, len(logr))) for _ in range(10000)]
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        best32 = min(ctrls, key=lambda m: agg(m, 32))
        wins32 = sum(score[(s, 32, L(51))] < score[(s, 32, best32)] for s in systems)
        best8 = min(ctrls, key=lambda m: agg(m, 8))
        wins8 = sum(score[(s, 8, L(51))] < score[(s, 8, best8)] for s in systems)
        ratio8 = agg(L(51), 8) / agg(best8, 8)
        t_l, t_c = float(np.median([thr(L(51), s) for s in systems])), float(np.median([thr(best32, s) for s in systems]))
        crit = {"CF1": bool(rho <= -0.8 and ci[1] < 0), "CF2": bool(wins32 / n >= 0.7),
                "CF3": bool(wins8 / n >= 0.7 and ratio8 <= 0.75)}
        strata = {}
        for lab, names in [("polynomial", [s for s in systems if poly[s]]), ("nonpolynomial", [s for s in systems if not poly[s]])]:
            if names:
                strata[lab] = {"n": len(names), "primary_N51_K32": agg(L(51), 32, names), "krr_K32": agg("krr", 32, names),
                               "poly2_K32": agg("poly2", 32, names),
                               "wins_vs_krr_K32": sum(score[(s, 32, L(51))] < score[(s, 32, "krr")] for s in names)}
        out["per_seed"][seed] = {
            "curve_K32": dict(zip(N_GRID, curve)), "spearman": rho, "bootstrap_ci": ci,
            "best_control_K32": best32, "wins_K32": wins32, "best_control_K8": best8, "wins_K8": wins8, "agg_ratio_K8": ratio8,
            "C3_median_threshold_K": {"learner": t_l, "best_control": t_c, "pass": bool(t_l <= t_c / 2)},
            "criteria": crit, "pass": all(crit.values()), "strata": strata,
            "secondary": {arm: {"curve_K32": {t: agg(L(t, arm), 32) for t in N_GRID},
                                "wins_vs_krr_K32_N51": sum(score[(s, 32, L(51, arm))] < score[(s, 32, "krr")] for s in systems)}
                          for arm in ("jointlibfb", "wide")},
        }
    methods = sorted({r["method"] for r in rows})
    out["aggregate"] = {m: {K: agg(m, K) for K in K_GRID} for m in methods if not m.endswith("_s2")}
    out["per_system_K32"] = {s: {m: score[(s, 32, m)] for m in ("krr", "rff", "poly2", f"{PRIMARY}_N00_s1", f"{PRIMARY}_N51_s1",
                                                              f"{PRIMARY}_N51_s2", "jointlibfb_N51_s1", "wide_N51_s1")}
                             for s in systems}
    (OUT / "summary_confirmation.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "aggregate"}, indent=1, default=float))
    print("aggregate (median over confirmation systems), K =", K_GRID)
    for m in ["krr", "rff", "poly2", "poly3", "scratch_mlp", f"{PRIMARY}_N00_s1", f"{PRIMARY}_N51_s1", "jointlibfb_N51_s1", "wide_N51_s1"]:
        print(f"  {m:22s}", " ".join(f"{out['aggregate'][m][K]:9.5f}" for K in K_GRID))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "analyze"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"eval": evaluate, "analyze": analyze}[a.cmd](a)
