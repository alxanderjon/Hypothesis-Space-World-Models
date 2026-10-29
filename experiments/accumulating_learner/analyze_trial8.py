"""Pre-registered Trial 8 criterion T1 plus comparison with the derivative contract (C1, C2 come from analyze_trial1.py
--out <trial8> --prefix seqtraj400 / frozen5)."""

import json

import numpy as np

import systems as S
from trial1 import SEEDS
from trial5 import OUT as TRIAL5
from trial8 import ARM, OUT


def main():
    rows = json.loads((OUT / "eval_development.json").read_text())["rows"]
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    t5 = json.loads((TRIAL5 / "eval_development.json").read_text())["rows"]
    deriv = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t5}
    names = S.eligible_split()["development"]
    med = lambda sc, m, K: float(np.median([sc[(s, K, m)] for s in names]))
    out = {}
    for arm in (ARM, "frozen5"):
        for s in SEEDS:
            m = f"{arm}_N51_s{s}"
            ctrls = ["rff", "krr", f"{arm}_N00_s{s}"]
            best = min(ctrls, key=lambda c: med(score, c, 32))
            t1 = med(score, m, 32) / med(score, best, 32)
            out[m] = {"K8": med(score, m, 8), "K16": med(score, m, 16), "K32": med(score, m, 32), "best_control_K32": best,
                      "best_control_value_K32": med(score, best, 32), "T1_ratio": t1, "T1": bool(t1 <= 1 / 3),
                      "ratio_vs_derivative_contract_K32": med(score, m, 32) / med(deriv, f"seqreplay400_N51_s{s}", 32),
                      "ratio_vs_derivative_contract_K16": med(score, m, 16) / med(deriv, f"seqreplay400_N51_s{s}", 16)}
            print(m, json.dumps(out[m], default=float))
    print("controls under trajectory contract K=8/16/32:",
          {c: [round(med(score, c, K), 5) for K in (8, 16, 32)] for c in ("krr", "rff", "poly2")})
    (OUT / "summary_T1.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
