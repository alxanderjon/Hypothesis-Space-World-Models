"""Trial 26 cost report: for every exactly recovered test system (full learner, N = 45, largest M recovered), compile
the learned model to native code and time it against the original simulation (scipy DOP853, rtol 1e-13, true f).
Also re-checks exactness of the compiled artifact itself. Reporting only."""

import argparse
import json
import time

import numpy as np

import compile_artifact as CA
import exact_harness as H
from trial26 import G_FAMILIAR, G_NOVEL, MS, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="full45")
    a = ap.parse_args()
    rows = []
    for name in G_NOVEL + G_FAMILIAR:
        rec = None
        for M in sorted(MS, reverse=True):
            p = OUT / f"test_s{a.seed}" / name / f"M{M}_{a.arm}.json"
            if p.exists():
                r = json.loads(p.read_text())
                if r.get("exact"):
                    rec = r
                    break
        if rec is None:
            continue
        ref = H.Reference(name, seed=a.seed)
        ref.envelope()
        roll = CA.build(rec["model"], rec["mu"], rec["sig"], ref.sys.dt)
        ev = H.evaluate_rollout(ref, roll, rec["nsub"])
        c = H.cost_compiled(ref, roll, rec["nsub"])
        rows.append({"name": name, "group": "novel" if name in G_NOVEL else "familiar", "M": rec["M"],
                     "compiled_exact": ev["exact"], "compiled_one_step": ev["one_step_max_scaled_error"], **c})
        print(f"{name}: compiled exact {ev['exact']} (one-step {ev['one_step_max_scaled_error']:.1e}) | single "
              f"{c['single']['speedup']:.0f}x batch {c['batch']['speedup']:.0f}x | rhs evals/step artifact "
              f"{c['single']['artifact_rhs_evals_per_step']} vs reference {c['single']['reference_rhs_evals_per_step']:.1f}", flush=True)
    (OUT / f"cost_s{a.seed}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
