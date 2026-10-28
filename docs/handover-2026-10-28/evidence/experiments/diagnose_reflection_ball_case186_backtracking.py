#!/usr/bin/env python3
"""Inspect rejected backtracking trials for case-186 N3 snapshots."""
from __future__ import annotations
import json
import os
from pathlib import Path
import torch
import numpy as np

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from experiments import expand_chronological_archive_handoff_dataset as expanded
from experiments import validate_d5_production_pipeline as production
from experiments.diagnose_reflection_ball_case186_n3 import atomic_json, topology
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path("outputs/chronological_archive_handoff_supervision_expanded")


def main():
    report_path = ROOT / "case186_n3_diagnosis/report.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    spec = next(row for row in manifest["trajectories"]
                if row["system"] == "reflection_ball" and row["case_id"] == 186)
    trajectory, _ = expanded.trajectory_for(spec)
    _, objective = production._objectives("reflection_ball", trajectory)
    lower_np, upper_np, _ = production._config("reflection_ball")
    lower = torch.from_numpy(lower_np); upper = torch.from_numpy(upper_np)
    derivatives, direction_fn, _ = staged._make_n3(objective, lower_np, upper_np)
    snapshots = {row["accepted_step_before"]: row for row in report["n3"]["snapshots"]}
    output = []
    for step in (0, 64):
        saved = snapshots[step]
        theta = torch.tensor(saved["theta"], dtype=torch.float64)
        current_loss = float(saved["loss"])
        _, gradient, hessian = derivatives(theta, torch.empty(0))
        direction, qp = dimension_general_box_qp_direction(
            theta, gradient, hessian, lower, upper, direction_fn)
        trials = []
        for backtrack in range(51):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            in_bounds = bool(torch.all(candidate >= lower) and torch.all(candidate <= upper))
            row = {"backtrack": backtrack, "alpha": alpha, "in_bounds": in_bounds}
            if in_bounds:
                with torch.no_grad():
                    loss = float(objective(candidate))
                candidate_np = candidate.numpy().copy()
                row.update(loss=loss, strict_decrease=loss < current_loss,
                           topology=topology(trajectory, candidate_np),
                           displacement_norm=float(np.linalg.norm(candidate_np - theta.numpy())))
            trials.append(row)
            if row.get("strict_decrease"):
                break
        output.append({"step": step, "theta": theta.tolist(), "loss": current_loss,
                       "direction": direction.tolist(), "box_qp": qp, "trials": trials})
    path = ROOT / "case186_n3_diagnosis/backtracking_trials.json"
    atomic_json(path, output)
    for row in output:
        changed = [trial for trial in row["trials"]
                   if trial.get("topology", {}).get("sha256") !=
                   row["trials"][-1].get("topology", {}).get("sha256")]
        print(row["step"], len(row["trials"]), row["trials"][-1]["backtrack"],
              row["trials"][-1].get("loss"), "changed_topology_trials", len(changed))


if __name__ == "__main__":
    main()
