"""Pre-registered Trial 3 mechanism criteria D1-D4 (C1-C3 come from analyze_trial1.py --prefix context)."""

import json

import numpy as np

import systems as S
from analyze_trial1 import NONPOLY
from trial1 import SEEDS
from trial3 import OUT


def main():
    rows = json.loads((OUT / "eval_development.json").read_text())["rows"]
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    systems = S.eligible_split()["development"]
    poly = [s for s in systems if s not in NONPOLY]
    med = lambda m, K, names: float(np.median([score[(s, K, m)] for s in names]))
    ret = json.loads((OUT / "eval_retention.json").read_text())
    out = {}
    for seed in SEEDS:
        c, w = f"context_N51_s{seed}", f"wide_N51_s{seed}"
        np_ratio = med(c, 32, sorted(NONPOLY)) / med(w, 32, sorted(NONPOLY))
        np_wins_krr = sum(score[(s, 32, c)] < score[(s, 32, "krr")] for s in NONPOLY)
        poly_ratio = med(c, 32, poly) / med(w, 32, poly)
        wrong_ratio = med(f"contextwrong_N51_s{seed}", 32, systems) / med(c, 32, systems)

        def drift(arch):
            cell = lambda pos, n: next(r["median_nmse"] for r in ret
                                       if r["arch"] == arch and r["seed"] == seed and r["position"] == pos and r["N"] == n)
            return float(np.median([cell(p, 51) / cell(p, 8) for p in range(8)]))

        d_c, d_w = drift("context"), drift("wide")
        crit = {
            "D1": bool(np_ratio <= 0.8 and np_wins_krr >= 3),
            "D2": bool(poly_ratio <= 2.0),
            "D3": bool(wrong_ratio >= 1.5),
            "D4": bool(d_c <= 2.0 and d_c <= d_w),
        }
        out[seed] = {"nonpoly_ratio_context_vs_wide": np_ratio, "nonpoly_wins_vs_krr": np_wins_krr,
                     "poly_ratio_context_vs_wide": poly_ratio, "wrong_context_ratio": wrong_ratio,
                     "retention_drift_N51_over_N8": {"context": d_c, "wide": d_w}, "criteria": crit}
        print(f"seed {seed}:", json.dumps(out[seed], default=float))
    (OUT / "summary_mechanism.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
