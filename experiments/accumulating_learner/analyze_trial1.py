"""Compute the pre-registered Trial 1 criteria (C1-C3) and stratified readouts from eval_development.json."""

import json

import numpy as np
from scipy.stats import spearmanr

import systems as S
from trial1 import K_GRID, N_GRID, OUT, SEEDS

NONPOLY = {"SaltonSea", "Thomas", "WindmiReduced", "YuWang"}
CONTROLS = ["rff", "krr"]


def main():
    rows = json.loads((OUT / "eval_development.json").read_text())["rows"]
    systems = S.eligible_split()["development"]
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    agg = lambda m, K, names=systems: float(np.median([score[(s, K, m)] for s in names]))
    thr = lambda m, s: next((K for K in K_GRID if score[(s, K, m)] <= 0.1), np.inf)
    out = {"per_seed": {}, "tables": {}}
    for seed in SEEDS:
        L = lambda n: f"learned_N{n:02d}_s{seed}"
        ctrls = CONTROLS + [L(0)]
        best = min(ctrls, key=lambda m: agg(m, 32))
        curve = [agg(L(n), 32) for n in N_GRID]
        rho = spearmanr(N_GRID, curve).correlation
        logr = np.log([score[(s, 32, L(51))] / score[(s, 32, L(0))] for s in systems])
        rng = np.random.default_rng(seed)
        boot = [np.median(rng.choice(logr, len(logr))) for _ in range(10000)]
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        wins = [s for s in systems if score[(s, 32, L(51))] < score[(s, 32, best)]]
        t_learn = float(np.median([thr(L(51), s) for s in systems]))
        t_best = float(np.median([thr(best, s) for s in systems]))
        c = {
            "C1": bool(rho <= -0.8 and ci[1] < 0),
            "C2": len(wins) >= 10,
            "C3": bool(t_learn <= t_best / 2) if np.isfinite(t_best) else bool(np.isfinite(t_learn)),
        }
        strata = {}
        for label, names in [("polynomial", [s for s in systems if s not in NONPOLY]), ("nonpolynomial", sorted(NONPOLY))]:
            strata[label] = {
                "curve_K32": {n: agg(L(n), 32, names) for n in N_GRID},
                "best_control_K32": agg(best, 32, names),
                "poly2_K32": agg("poly2", 32, names),
                "learnedlam_N51_K32": agg(f"learnedlam_N51_s{seed}", 32, names),
                "wins_vs_best": sum(score[(s, 32, L(51))] < score[(s, 32, best)] for s in names),
                "n": len(names),
            }
        out["per_seed"][seed] = {
            "best_control": best, "curve_K32": dict(zip(N_GRID, curve)), "spearman": rho, "bootstrap_ci_log_ratio_51_vs_0": ci,
            "wins_vs_best": len(wins), "win_systems": wins, "median_threshold_K_learner": t_learn,
            "median_threshold_K_best_control": t_best, "criteria": c, "pass": all(c.values()), "strata": strata,
        }
    methods = sorted({r["method"] for r in rows})
    out["tables"]["aggregate"] = {m: {K: agg(m, K) for K in K_GRID} for m in methods}
    out["tables"]["per_system_K32"] = {s: {m: score[(s, 32, m)] for m in methods} for s in systems}
    out["tables"]["per_system_threshold"] = {s: {m: thr(m, s) for m in methods} for s in systems}
    (OUT / "trial1_summary.json").write_text(json.dumps(out, indent=1, default=float))
    for seed, r in out["per_seed"].items():
        print(f"seed {seed}: best control {r['best_control']}, rho {r['spearman']:.3f}, CI {r['bootstrap_ci_log_ratio_51_vs_0']}, "
              f"wins {r['wins_vs_best']}/13, thrK learner {r['median_threshold_K_learner']} vs {r['median_threshold_K_best_control']}, "
              f"criteria {r['criteria']}")
        print("  curve K=32:", {n: round(v, 4) for n, v in r["curve_K32"].items()})
        for lab, st in r["strata"].items():
            print(f"  {lab}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k != 'curve_K32'}),
                  {n: round(v, 4) for n, v in st["curve_K32"].items()})
    print("aggregate (median over dev systems):")
    for m in methods:
        if m.endswith("_s2"):
            continue
        print(f"  {m:22s}", " ".join(f"{out['tables']['aggregate'][m][K]:9.4f}" for K in K_GRID))


if __name__ == "__main__":
    main()
