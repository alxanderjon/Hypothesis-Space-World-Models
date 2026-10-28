#!/usr/bin/env python3
"""Calibrate, without selecting, an accepted-step stagnation signature.

Consumes only existing trajectory artifacts.  No optimizer call is made.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from experiments.audit_n3_accepted_step_stagnation import (
    saved_newton_relation_metric,
)


SOURCE = Path("outputs/n3_accepted_step_stagnation_audit/accepted_steps.jsonl")
EARLY = Path("outputs/early_n3_basin_trajectory_geometry/traces.jsonl")
POINTS = Path("outputs/direct_n3_correctability_geometry/points.jsonl")
OUT = Path("outputs/n3_accepted_step_stagnation_calibration")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quantiles(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows
                         if row.get(key) is not None and math.isfinite(float(row[key]))],
                        dtype=np.float64)
    if not len(values): return {"count": 0}
    return {"count": int(len(values)), "minimum": float(values.min()),
            "p10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "maximum": float(values.max())}


def enrich(row: dict) -> dict:
    row = dict(row)
    movements = np.abs(np.asarray(row["displacement_local_ulps"], dtype=np.float64))
    row["displacement_local_ulps_total"] = float(np.sum(movements))
    spacing = abs(float(np.spacing(np.float64(row["loss"]))))
    row["predicted_full_loss_ulps"] = (
        float(row["predicted_decrease_full_direction"]) / spacing if spacing else None)
    row["predicted_accepted_loss_ulps"] = (
        float(row["predicted_decrease_accepted_displacement"]) / spacing if spacing else None)
    return row


def direct_geometry_rows() -> list[dict]:
    point_map = {(row["system"], int(row["sobol_index"])): row for row in (
        json.loads(line) for line in POINTS.read_text().splitlines())}
    output = []
    dummy_rollout = {"continuation_max_error": None, "continuation_mse": None,
                     "rollout_finite": True, "collision_topology_regressions": None}
    for trace in (json.loads(line) for line in EARLY.read_text().splitlines()):
        source = point_map[(trace["system"], int(trace["sobol_index"]))]
        teacher = np.asarray(source["offline_teacher"], dtype=np.float64)
        theta_history = [np.asarray(value, dtype=np.float64)
                         for value in trace["replay"]["theta_history"]]
        loss_history = trace["replay"]["loss_history"]
        if trace["label_PASS"]:
            cohort = "successful_direct_n3"
        elif trace["replay"]["stop_reason"] == "max_accepted_steps":
            cohort = "ordinary_nonconverged_resource_cap"
        else:
            cohort = "ordinary_nonconverged_genuine_stop"
        identity = f"{trace['system']}:sobol_{trace['sobol_index']}"
        for detailed in trace["detailed_iterations"]:
            if not detailed.get("accepted"): continue
            index = int(detailed["iteration_index"])
            row = saved_newton_relation_metric(
                cohort=cohort, identity=identity, step=index,
                theta=theta_history[index], next_theta=theta_history[index + 1],
                loss=float(loss_history[index]), next_loss=float(loss_history[index + 1]),
                gradient_norm=float(detailed["gradient_norm"]),
                direction=np.asarray(detailed["newton_direction"], dtype=np.float64),
                gradient_dot_direction=float(detailed["gradient_dot_direction"]),
                alpha=float(detailed["accepted_alpha"]),
                backtrack=int(detailed["accepted_backtrack_index"]), teacher=teacher,
                rollout_before=dummy_rollout, rollout_after=dummy_rollout,
                extra={"trajectory_pass": bool(trace["label_PASS"]),
                       "terminal_stop_reason": trace["replay"]["stop_reason"],
                       "system": trace["system"],
                       "saved_prefix_length": len(trace["detailed_iterations"])})
            output.append(enrich(row))
    return output


def max_consecutive(rows: list[dict], predicate) -> tuple[int, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        window = row.get("window", "complete")
        grouped[(row["identity"], window)].append(row)
    best = 0; winners = []
    for (identity, window), values in grouped.items():
        values.sort(key=lambda row: row["accepted_step"])
        run = 0; local_best = 0
        for row in values:
            if predicate(row): run += 1; local_best = max(local_best, run)
            else: run = 0
        if local_best > best:
            best = local_best; winners = [{"identity": identity, "window": window,
                                           "length": local_best}]
        elif local_best == best and local_best:
            winners.append({"identity": identity, "window": window,
                            "length": local_best})
    return best, winners


def main() -> None:
    prior = [enrich(json.loads(line)) for line in SOURCE.read_text().splitlines()]
    direct = direct_geometry_rows()
    rows = prior + direct

    # Non-overlapping analysis classes.  The targeted five-trajectory subset
    # in the prior audit is omitted in favor of all 105 saved PASS steps.
    classes = {
        "case186_tail": [row for row in prior
                         if row["cohort"] == "case186_event_blocked" and
                         row["accepted_step"] >= 5],
        "case186_entry": [row for row in prior
                          if row["cohort"] == "case186_event_blocked" and
                          row["accepted_step"] < 5],
        "kepler_alias_late_tail": [row for row in prior
                                    if row["cohort"] == "old_kepler_boundary_alias_tail" and
                                    row.get("window") == "late_exact_window"],
        "kepler_alias_early_tail": [row for row in prior
                                     if row["cohort"] == "old_kepler_boundary_alias_tail" and
                                     row.get("window") == "early_exact_window"],
        "normal_high_precision": [row for row in prior
                                  if row["cohort"] == "normal_high_precision"],
        "successful_direct_n3": [row for row in direct
                                 if row["cohort"] == "successful_direct_n3"],
        "ordinary_nonconverged_genuine_stop": [row for row in direct
                                                if row["cohort"] ==
                                                "ordinary_nonconverged_genuine_stop"],
        "ordinary_nonconverged_resource_cap": [row for row in direct
                                                if row["cohort"] ==
                                                "ordinary_nonconverged_resource_cap"],
    }
    metric_keys = [
        "accepted_model_over_full_model", "accepted_displacement_norm",
        "displacement_over_direction", "displacement_local_ulps_max",
        "displacement_local_ulps_total", "actual_loss_decrease",
        "actual_improvement_loss_ulps", "actual_over_predicted_accepted",
        "actual_over_predicted_full", "gradient_norm", "projected_gradient_norm",
        "predicted_decrease_full_direction", "predicted_full_loss_ulps",
        "direction_norm", "accepted_alpha", "backtracking_count",
    ]
    distributions = {name: {key: quantiles(value, key) for key in metric_keys}
                     for name, value in classes.items()}

    core = classes["case186_tail"] + classes["kepler_alias_late_tail"]
    success = classes["normal_high_precision"] + classes["successful_direct_n3"]
    # This is the observed bounding box of known late pathological states.  It
    # is used only to count overlap and persistence, never as a proposed rule.
    envelope = {
        "absolute_Rm_max": max(abs(row["accepted_model_over_full_model"]) for row in core),
        "parameter_ulp_max_max": max(row["displacement_local_ulps_max"] for row in core),
        "parameter_ulp_total_max": max(row["displacement_local_ulps_total"] for row in core),
        "loss_ulp_max": max(row["actual_improvement_loss_ulps"] for row in core),
        "actual_over_full_max": max(abs(row["actual_over_predicted_full"]) for row in core),
        "full_prediction_loss_ulps_min": min(row["predicted_full_loss_ulps"] for row in core),
    }
    def in_joint_envelope(row):
        return (abs(row["accepted_model_over_full_model"]) <= envelope["absolute_Rm_max"] and
                row["displacement_local_ulps_max"] <= envelope["parameter_ulp_max_max"] and
                row["displacement_local_ulps_total"] <= envelope["parameter_ulp_total_max"] and
                row["actual_improvement_loss_ulps"] <= envelope["loss_ulp_max"] and
                abs(row["actual_over_predicted_full"]) <= envelope["actual_over_full_max"] and
                row["predicted_full_loss_ulps"] >= envelope["full_prediction_loss_ulps_min"])
    def small_rm(row):
        return abs(row["accepted_model_over_full_model"]) <= envelope["absolute_Rm_max"]

    overlap = {}
    persistence = {}
    for name, values in classes.items():
        overlap[name] = {"steps": len(values),
                         "small_Rm_steps": sum(small_rm(row) for row in values),
                         "joint_envelope_steps": sum(in_joint_envelope(row) for row in values)}
        rm_run, rm_winners = max_consecutive(values, small_rm)
        joint_run, joint_winners = max_consecutive(values, in_joint_envelope)
        persistence[name] = {
            "maximum_consecutive_small_Rm": rm_run,
            "small_Rm_winners": rm_winners,
            "maximum_consecutive_joint_envelope": joint_run,
            "joint_winners": joint_winners,
            "observation_limit": (3 if "direct_n3" in name or
                                        "ordinary_nonconverged" in name else None),
        }

    positive_success_rm = [row for row in success
                           if row["accepted_model_over_full_model"] > 0.0]
    closest_success_rm = min(positive_success_rm,
                             key=lambda row: row["accepted_model_over_full_model"])
    least_tail_like = max(core, key=lambda row: (
        abs(row["accepted_model_over_full_model"]),
        row["displacement_local_ulps_total"],
        row["actual_improvement_loss_ulps"]))
    # The early alias state is the strongest pathological counterexample to a
    # strict late-tail box; it precedes that trajectory's later ULP tail.
    closest_pathological = max(classes["kepler_alias_early_tail"],
                               key=lambda row: row["accepted_model_over_full_model"])

    trajectory_counts = {}
    for name, values in classes.items():
        trajectory_counts[name] = len(set(row["identity"] for row in values))
    stop_counts = Counter(row.get("terminal_stop_reason") for row in direct
                          if not row["trajectory_pass"])

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "accepted_steps.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    report = {
        "diagnostic_only": True,
        "existing_corpus_only": True,
        "class_step_counts": {name: len(value) for name, value in classes.items()},
        "class_trajectory_counts": trajectory_counts,
        "ordinary_nonconverged_stop_counts": dict(stop_counts),
        "distributions": distributions,
        "observed_pathological_late_envelope": envelope,
        "envelope_is_not_a_proposed_threshold": True,
        "overlap": overlap,
        "persistence": persistence,
        "closest_successful_Rm_counterexample": closest_success_rm,
        "least_tail_like_late_pathological_step": least_tail_like,
        "closest_pathological_entry_counterexample": closest_pathological,
        "formulas": {
            "Rm": "Delta_m(s)/Delta_m(d)",
            "Delta_m(v)": "-(g.T v + 0.5 v.T (H + lambda I) v)",
            "parameter_ulp_total": "sum_j abs(s_j / spacing(theta_j))",
            "full_prediction_loss_ulps": "Delta_m(d) / abs(spacing(loss))",
        },
    }
    atomic_json(OUT / "report.json", report)
    print(json.dumps({
        "class_step_counts": report["class_step_counts"],
        "class_trajectory_counts": report["class_trajectory_counts"],
        "envelope": envelope, "overlap": overlap,
        "maximum_consecutive_joint": {name: value["maximum_consecutive_joint_envelope"]
                                      for name, value in persistence.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
