"""Pre-registered Trial 4 mechanism criteria E1-E4 (C1-C3 come from analyze_trial1.py --prefix libraryfb)."""

import json

import numpy as np

import systems as S
from analyze_trial1 import NONPOLY
from trial1 import N_GRID, SEEDS
from trial3 import OUT as TRIAL3
from trial4 import M, OUT


def main():
    ev = json.loads((OUT / "eval_development.json").read_text())
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in ev["rows"]}
    t3 = json.loads((TRIAL3 / "eval_development.json").read_text())["rows"]
    score.update({(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t3 if r["method"].startswith("wide")})
    systems = S.eligible_split()["development"]
    poly = [s for s in systems if s not in NONPOLY]
    med = lambda m, K, names=systems: float(np.median([score[(s, K, m)] for s in names]))
    ret = json.loads((OUT / "eval_retention.json").read_text())
    krr32 = med("krr", 32)
    out = {}
    for seed in SEEDS:
        lf = lambda n: f"libraryfb_N{n:02d}_s{seed}"
        curve = {n: med(lf(n), 32) for n in N_GRID}
        np_wins = sum(score[(s, 32, lf(51))] < score[(s, 32, "krr")] for s in NONPOLY)
        np_ratio = med(lf(51), 32, sorted(NONPOLY)) / med(f"widefb_N51_s{seed}", 32, sorted(NONPOLY))
        poly_ratio = med(lf(51), 32, poly) / med(f"wide_N51_s{seed}", 32, poly)

        def drift(arm):
            cell = lambda pos, n: next(r["median_nmse"] for r in ret
                                       if r["arm"] == arm and r["seed"] == seed and r["position"] == pos and r["N"] == n)
            return float(np.median([cell(p, 51) / cell(p, 8) for p in range(8)]))

        d_l, d_w = drift("libraryfb"), drift("widefb")
        crit = {
            "E1": all(v <= 1.1 * krr32 for v in curve.values()),
            "E2": bool(np_wins >= 3 and np_ratio <= 0.8),
            "E3": bool(poly_ratio <= 2.0),
            "E4": bool(d_l <= 2.0 and d_l <= d_w),
        }
        sel = {s: np.sum([x["counts"] for x in ev["selections"] if x["system"] == s and x["seed"] == seed
                          and x["N"] == 51 and x["K"] == 32], 0).tolist() for s in systems}
        gap = {K: med(lf(51), K) / med(f"oracle_N51_s{seed}", K) for K in (8, 16, 32)}
        out[seed] = {"libraryfb_curve_K32": curve, "krr_K32": krr32, "nonpoly_wins_vs_krr": np_wins,
                     "nonpoly_ratio_vs_widefb": np_ratio, "poly_ratio_vs_wide": poly_ratio,
                     "retention_drift": {"libraryfb": d_l, "widefb": d_w}, "criteria": crit,
                     "selection_counts_N51_K32 (bases 0-3, krr)": sel, "libraryfb_over_oracle": gap,
                     "aggregates": {arm: {K: med(f"{arm}_N51_s{seed}", K) for K in (8, 16, 32)}
                                    for arm in ("library", "libraryfb", "oracle", "widefb", "wide")}}
        print(f"seed {seed}:", json.dumps(out[seed], default=float))
    (OUT / "summary_mechanism.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
