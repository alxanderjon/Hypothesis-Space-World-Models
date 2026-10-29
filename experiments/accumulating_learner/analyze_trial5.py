"""Pre-registered Trial 5 criteria F2-F4 (F1 = C1 and C2 from analyze_trial1.py --prefix <arm>)."""

import json

import numpy as np

import systems as S
from trial1 import SEEDS
from trial4 import OUT as TRIAL4
from trial5 import ARMS, OUT


def main():
    rows = json.loads((OUT / "eval_development.json").read_text())["rows"]
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    t4 = json.loads((TRIAL4 / "eval_development.json").read_text())["rows"]
    score.update({(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t4
                  if r["method"].startswith("libraryfb_N51")})
    systems = S.eligible_split()["development"]
    med = lambda m, K: float(np.median([score[(s, K, m)] for s in systems]))
    ret = json.loads((OUT / "eval_retention.json").read_text())
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            logf = OUT / "learners" / f"{arm}_s{seed}.json"
            if not logf.exists():
                continue
            stages = json.loads(logf.read_text())["stages"]
            f2 = med(f"{arm}_N51_s{seed}", 32) / med(f"libraryfb_N51_s{seed}", 32)
            cell = lambda pos, t: next(r["median_nmse"] for r in ret
                                       if r["arm"] == arm and r["seed"] == seed and r["position"] == pos and r["N"] == t)
            f3 = float(np.median([cell(p, 51) / cell(p, 8) for p in range(8)]))
            lr = np.log([st["encounter_libraryfb"] / st["encounter_krr"] for st in stages])
            first, last = float(np.median(lr[:17])), float(np.median(lr[-17:]))
            crit = {"F2": bool(f2 <= 2.0), "F3": bool(f3 <= 2.0), "F4": bool(last < first and last < 0)}
            out[f"{arm}_s{seed}"] = {"F2_ratio_vs_joint": f2, "F3_retention_drift": f3,
                                     "F4_median_log_ratio_first17": first, "F4_median_log_ratio_last17": last,
                                     "K32_curve": {t: med(f"{arm}_N{t:02d}_s{seed}", 32) for t in (0, 1, 2, 4, 8, 16, 32, 51)},
                                     "criteria": crit}
            print(f"{arm} seed {seed}:", json.dumps(out[f"{arm}_s{seed}"], default=float))
    (OUT / "summary_mechanism.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
