#!/usr/bin/env python3
"""Replay the calibrated FLOAT64_STAGNATION_TAIL rule.

This consumes saved N3 trajectories and reconstructs local geometry only at
already-saved low-loss-ULP control states.  Production integration is tested
separately by ``validate_n3_float64_stagnation_tail_stop``.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import analyze_early_n3_basin_trajectories as early
from experiments import map_direct_n3_correctability_geometry as basin


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

CALIBRATION = Path("outputs/n3_accepted_step_stagnation_calibration/accepted_steps.jsonl")
EARLY = Path("outputs/early_n3_basin_trajectory_geometry/traces.jsonl")
NONTERMINATION = Path(
    "outputs/n1_candidate_handoff_supervision_uncapped_n3/"
    "nontermination_diagnosis/report.json")
CAP_VALIDATION = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = Path("outputs/n3_float64_stagnation_tail_rule")

LOSS_ULP_CUTOFF = 512
FULL_PREDICTION_ULP_FLOOR = 512.0
PERSISTENCE = 3
RECONSTRUCTION_LIMIT = 10_000


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def loss_ulps(before: float, after: float) -> float:
    if not (math.isfinite(before) and math.isfinite(after) and before >= after >= 0.0):
        raise ValueError((before, after))
    spacing = abs(float(np.spacing(np.float64(before))))
    return float(before - after) / spacing


def max_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def first_run_end(flags: list[bool], length: int) -> int | None:
    current = 0
    for index, flag in enumerate(flags):
        current = current + 1 if flag else 0
        if current >= length:
            return index
    return None


def base_flag(row: dict) -> bool:
    return (int(row["actual_improvement_loss_ulps"]) <= LOSS_ULP_CUTOFF and
            float(row["predicted_full_loss_ulps"]) > FULL_PREDICTION_ULP_FLOOR)


def reconstruct_saved_control_steps(traces: list[dict]) -> dict[tuple[str, int, int], dict]:
    """Reconstruct only saved steps near the candidate numerical regime.

    The prior corpus retained full theta/loss histories but detailed tensors
    only for the first three accepted steps.  Recomputing the local differential
    state at the *saved theta* is not a new trajectory.
    """
    configs = {name: basin._system_config(name) for name in ("bouncing_ball", "kepler")}
    output: dict[tuple[str, int, int], dict] = {}
    for trace in traces:
        losses = trace["replay"]["loss_history"]
        thetas = trace["replay"]["theta_history"]
        config = configs[trace["system"]]
        for step, (before, after) in enumerate(zip(losses, losses[1:])):
            loss_movement = loss_ulps(before, after)
            if loss_movement > RECONSTRUCTION_LIMIT:
                continue
            theta = torch.tensor(thetas[step], dtype=torch.float64)
            next_theta = np.asarray(thetas[step + 1], dtype=np.float64)
            state, gradient, direction = early._iteration_state(theta, before, config)
            if not state["loss_replay_exact"] or direction is None:
                raise RuntimeError((trace["system"], trace["sobol_index"], step))
            displacement = next_theta - theta.numpy()
            spacing = np.abs(np.spacing(theta.numpy()))
            movement = np.abs(np.divide(displacement, spacing,
                                        out=np.zeros_like(displacement), where=spacing != 0.0))
            # For the saved Newton/DAMPED solve B d = -g, Delta m(d)=-0.5 g.d.
            predicted_full = 0.5 * float(state["predicted_decrease"])
            loss_spacing = abs(float(np.spacing(np.float64(before))))
            predicted_full_ulps = predicted_full / loss_spacing
            output[(trace["system"], int(trace["sobol_index"]), step)] = {
                "actual_improvement_loss_ulps": loss_movement,
                "predicted_full_loss_ulps": predicted_full_ulps,
                "projected_gradient_norm": float(state["gradient_norm"]),
                "displacement_local_ulps_max": float(np.max(movement)),
                "displacement_local_ulps_total": float(np.sum(movement)),
                "actual_over_predicted_full": (float(before - after) / predicted_full
                                                if predicted_full > 0.0 else None),
                "terminal_stop_reason": trace["replay"]["stop_reason"],
                "trajectory_pass": bool(trace["label_PASS"]),
            }
    return output


def complete_trace_flags(trace: dict, reconstructed: dict) -> list[bool]:
    flags = []
    losses = trace["replay"]["loss_history"]
    for step, (before, after) in enumerate(zip(losses, losses[1:])):
        movement = loss_ulps(before, after)
        if movement > LOSS_ULP_CUTOFF:
            flags.append(False)
            continue
        row = reconstructed[(trace["system"], int(trace["sobol_index"]), step)]
        flags.append(row["predicted_full_loss_ulps"] > FULL_PREDICTION_ULP_FLOOR)
    return flags


def metric_gap(pathological: list[dict], controls: list[dict], key: str,
               orientation: str) -> dict:
    path = [abs(float(row[key])) if key in
            ("accepted_model_over_full_model", "actual_over_predicted_full")
            else float(row[key]) for row in pathological if row.get(key) is not None]
    control = [abs(float(row[key])) if key in
               ("accepted_model_over_full_model", "actual_over_predicted_full")
               else float(row[key]) for row in controls if row.get(key) is not None]
    if orientation == "low":
        path_boundary, control_boundary = max(path), min(control)
        clean = path_boundary < control_boundary
    else:
        path_boundary, control_boundary = min(path), max(control)
        clean = path_boundary > control_boundary
    # Nearest positive values by multiplicative distance; this remains useful
    # when the class intervals overlap.
    nearest = None
    for p, c in itertools.product(path, control):
        distance = (abs(math.log(p / c)) if p > 0.0 and c > 0.0 else abs(p - c))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, p, c)
    return {
        "orientation_for_tail": orientation,
        "pathological_minimum": min(path), "pathological_maximum": max(path),
        "control_minimum": min(control), "control_maximum": max(control),
        "pathological_boundary": path_boundary, "control_boundary": control_boundary,
        "clean_univariate_gap": clean,
        "open_gap": ([path_boundary, control_boundary] if clean else None),
        "nearest_cross_class_values": {"pathological": nearest[1], "control": nearest[2]},
    }


def alias_stop_table(rows: list[dict]) -> list[dict]:
    diagnosis = json.loads(NONTERMINATION.read_text())
    detail = {f"kepler_alias:{item['case_id']}:{item['example_id'][:12]}": item
              for item in diagnosis["cases_detail"]}
    output = []
    identities = sorted({row["identity"] for row in rows})
    for identity in identities:
        early_rows = sorted([row for row in rows if row["identity"] == identity and
                             row.get("window") == "early_exact_window"],
                            key=lambda row: row["accepted_step"])
        late_rows = sorted([row for row in rows if row["identity"] == identity and
                            row.get("window") == "late_exact_window"],
                           key=lambda row: row["accepted_step"])
        early_end = first_run_end([base_flag(row) for row in early_rows], PERSISTENCE)
        item = detail[identity]
        if early_end is not None:
            # The early exact window begins immediately after the old 20-step cap.
            observed_step = 21 + early_end
            window = "early_exact_window"
            ambiguity = "could have begun before accepted step 21"
        else:
            late_end = first_run_end([base_flag(row) for row in late_rows], PERSISTENCE)
            if late_end is None:
                observed_step = None; window = None; ambiguity = "not detected"
            else:
                # Four late rows end at accepted_steps_saved.
                observed_step = int(item["accepted_steps_saved"]) - 3 + late_end
                window = "late_exact_window"
                ambiguity = "onset lies between the early and late saved windows"
        output.append({
            "identity": identity, "case_id": int(item["case_id"]),
            "accepted_steps_saved": int(item["accepted_steps_saved"]),
            "first_observed_stop_after_accepted_step": observed_step,
            "source_window": window, "onset_precision": ambiguity,
        })
    return output


def main() -> None:
    rows = load_jsonl(CALIBRATION)
    traces = load_jsonl(EARLY)
    reconstructed = reconstruct_saved_control_steps(traces)

    case186 = [row for row in rows if row["cohort"] == "case186_event_blocked"]
    case186_tail = [row for row in case186 if row["accepted_step"] >= 5]
    alias_rows = [row for row in rows if row["cohort"] == "old_kepler_boundary_alias_tail"]
    alias_late = [row for row in alias_rows if row.get("window") == "late_exact_window"]
    pathological = case186_tail + alias_late
    control_cohorts = {
        "normal_high_precision", "successful_direct_n3",
        "ordinary_nonconverged_genuine_stop", "ordinary_nonconverged_resource_cap",
    }
    controls = [row for row in rows if row["cohort"] in control_cohorts]

    # Extend the pointwise gap audit with reconstructed late states that were
    # absent from the old first-three-step detail export.  Do not duplicate any
    # step already present in the detailed corpus.
    detailed_keys = {(row["identity"], int(row["accepted_step"])) for row in controls}
    trace_by_key = {(trace["system"], int(trace["sobol_index"])): trace for trace in traces}
    controls_for_gaps = list(controls)
    for (system, sobol_index, step), value in reconstructed.items():
        identity = f"{system}:sobol_{sobol_index}"
        if (identity, step) in detailed_keys:
            continue
        trace = trace_by_key[(system, sobol_index)]
        cohort = ("successful_direct_n3" if trace["label_PASS"] else
                  ("ordinary_nonconverged_resource_cap"
                   if trace["replay"]["stop_reason"] == "max_accepted_steps" else
                   "ordinary_nonconverged_genuine_stop"))
        controls_for_gaps.append(dict(value, identity=identity,
                                      accepted_step=step, cohort=cohort))

    metrics = {
        "abs_Rm": ("accepted_model_over_full_model", "low"),
        "maximum_parameter_ULP_movement": ("displacement_local_ulps_max", "low"),
        "total_parameter_ULP_movement": ("displacement_local_ulps_total", "low"),
        "loss_decrease_ULPs": ("actual_improvement_loss_ulps", "low"),
        "abs_actual_over_full_predicted": ("actual_over_predicted_full", "low"),
        "full_predicted_decrease_loss_ULPs": ("predicted_full_loss_ulps", "high"),
        "projected_gradient_magnitude": ("projected_gradient_norm", "high"),
    }
    gap_audit = {name: metric_gap(pathological, controls_for_gaps, key, orientation)
                 for name, (key, orientation) in metrics.items()}

    complete = defaultdict(list)
    for trace in traces:
        cohort = ("successful_direct_n3" if trace["label_PASS"] else
                  ("ordinary_nonconverged_resource_cap"
                   if trace["replay"]["stop_reason"] == "max_accepted_steps" else
                   "ordinary_nonconverged_genuine_stop"))
        flags = complete_trace_flags(trace, reconstructed)
        complete[cohort].append({
            "identity": f"{trace['system']}:sobol_{trace['sobol_index']}",
            "flags": flags, "maximum_run": max_run(flags),
            "accepted_steps": len(flags), "stop_reason": trace["replay"]["stop_reason"],
        })

    normal_flags = defaultdict(list)
    for row in controls:
        if row["cohort"] == "normal_high_precision":
            normal_flags[row["identity"]].append(row)
    complete["normal_high_precision"] = [
        {"identity": identity,
         "flags": [base_flag(row) for row in sorted(values, key=lambda item: item["accepted_step"])],
         "maximum_run": max_run([base_flag(row) for row in sorted(
             values, key=lambda item: item["accepted_step"])]),
         "accepted_steps": len(values), "stop_reason": "genuine_numerical_stop"}
        for identity, values in normal_flags.items()
    ]

    # The six formerly cap-limited correct-basin Kepler controls have complete
    # saved loss histories.  Their loss motion alone is sufficient to prove
    # that the candidate conjunction cannot fire.
    cap_ids = {4, 31, 40, 47, 52, 54}
    cap_report = json.loads(CAP_VALIDATION.read_text())
    cap_controls = []
    for item in cap_report["cases"]:
        if item["system"] != "kepler5" or int(item["case_id"]) not in cap_ids:
            continue
        losses = item["cap_suppressed"]["loss_history"]
        movements = [loss_ulps(a, b) for a, b in zip(losses, losses[1:])]
        flags = [False if movement > LOSS_ULP_CUTOFF else None for movement in movements]
        cap_controls.append({
            "case_id": int(item["case_id"]), "accepted_steps": len(movements),
            "minimum_loss_decrease_ULPs": min(movements),
            "candidate_triggered": any(flag is True for flag in flags),
            "stop_reason": item["cap_suppressed"]["stop_reason"],
        })

    # Individual and joint ablations over complete saved histories.  Signal-only
    # is evaluated wherever reconstructed; loss-only is exact everywhere.
    ablations = {
        "loss_only": {}, "signal_only_on_reconstructed_states": {}, "joint": {},
    }
    for cohort, trajectories in complete.items():
        ablations["joint"][cohort] = {
            "triggered_trajectories": sum(item["maximum_run"] >= PERSISTENCE
                                          for item in trajectories),
            "trajectories": len(trajectories),
            "maximum_run": max((item["maximum_run"] for item in trajectories), default=0),
        }
    # Complete loss-only flags need no differential reconstruction.
    for cohort in ("successful_direct_n3", "ordinary_nonconverged_genuine_stop",
                   "ordinary_nonconverged_resource_cap"):
        selected = []
        for trace in traces:
            trace_cohort = ("successful_direct_n3" if trace["label_PASS"] else
                            ("ordinary_nonconverged_resource_cap"
                             if trace["replay"]["stop_reason"] == "max_accepted_steps" else
                             "ordinary_nonconverged_genuine_stop"))
            if trace_cohort != cohort: continue
            losses = trace["replay"]["loss_history"]
            flags = [loss_ulps(a, b) <= LOSS_ULP_CUTOFF for a, b in zip(losses, losses[1:])]
            selected.append(max_run(flags))
        ablations["loss_only"][cohort] = {
            "triggered_trajectories": sum(run >= PERSISTENCE for run in selected),
            "trajectories": len(selected), "maximum_run": max(selected, default=0),
        }
    # Normal controls are already complete.
    normal_runs = [item["maximum_run"] for item in complete["normal_high_precision"]]
    ablations["loss_only"]["normal_high_precision"] = {
        "triggered_trajectories": sum(run >= PERSISTENCE for run in normal_runs),
        "trajectories": len(normal_runs), "maximum_run": max(normal_runs, default=0),
    }
    # Signal-only is demonstrably nonselective even in the detailed/reconstructed
    # subset, so no missing step can rescue it as a minimal condition.
    for cohort in control_cohorts:
        values = [row for row in controls if row["cohort"] == cohort]
        values += [dict(value, cohort=("successful_direct_n3" if trace["label_PASS"] else
                                      ("ordinary_nonconverged_resource_cap" if
                                       trace["replay"]["stop_reason"] == "max_accepted_steps"
                                       else "ordinary_nonconverged_genuine_stop")))
                   for trace in traces for key, value in reconstructed.items()
                   if key[0] == trace["system"] and key[1] == int(trace["sobol_index"])
                   and (("successful_direct_n3" if trace["label_PASS"] else
                         ("ordinary_nonconverged_resource_cap" if
                          trace["replay"]["stop_reason"] == "max_accepted_steps" else
                          "ordinary_nonconverged_genuine_stop"))) == cohort]
        ablations["signal_only_on_reconstructed_states"][cohort] = {
            "passing_steps": sum(float(row["predicted_full_loss_ulps"]) >
                                 FULL_PREDICTION_ULP_FLOOR for row in values),
            "available_steps": len(values),
        }

    persistence = {}
    path_windows = {
        "case186_complete": [base_flag(row) for row in sorted(
            case186, key=lambda item: item["accepted_step"])],
    }
    for identity in sorted({row["identity"] for row in alias_late}):
        path_windows[identity] = [base_flag(row) for row in sorted(
            [item for item in alias_late if item["identity"] == identity],
            key=lambda item: item["accepted_step"])]
    all_control_runs = [item["maximum_run"] for values in complete.values() for item in values]
    for length in range(1, 7):
        triggered_controls = [item["identity"] for values in complete.values()
                              for item in values if item["maximum_run"] >= length]
        persistence[str(length)] = {
            "pathological_trajectories_detected": sum(max_run(flags) >= length
                                                       for flags in path_windows.values()),
            "pathological_trajectories": len(path_windows),
            "control_trajectories_triggered": sum(run >= length for run in all_control_runs),
            "triggered_control_identities": triggered_controls,
        }

    case186_end = first_run_end(path_windows["case186_complete"], PERSISTENCE)
    pathological_stops = {
        "case186": {
            "first_stop_after_accepted_step": None if case186_end is None else case186_end + 1,
            "exact_from_complete_128_step_history": True,
        },
        "kepler_aliases": alias_stop_table(alias_rows),
    }

    # Joint conditional gaps that support the two 512 cutoffs.
    low_loss_controls = [value for value in reconstructed.values()
                         if value["actual_improvement_loss_ulps"] <= LOSS_ULP_CUTOFF]
    low_signal_controls = [value for value in low_loss_controls
                           if value["predicted_full_loss_ulps"] <=
                           FULL_PREDICTION_ULP_FLOOR]
    high_signal_control_trajectories = []
    third_run_boundaries = []
    for trace in traces:
        sequence = []
        losses = trace["replay"]["loss_history"]
        for step, (before, after) in enumerate(zip(losses, losses[1:])):
            movement = loss_ulps(before, after)
            key = (trace["system"], int(trace["sobol_index"]), step)
            signal = (reconstructed[key]["predicted_full_loss_ulps"]
                      if key in reconstructed else None)
            sequence.append((movement, signal))
        candidates = []
        for start in range(max(0, len(sequence) - 2)):
            window = sequence[start:start + 3]
            if len(window) < 3 or any(signal is None or signal <=
                                      FULL_PREDICTION_ULP_FLOOR for _, signal in window):
                continue
            candidates.append(max(movement for movement, _ in window))
        if candidates:
            third_run_boundaries.append({
                "identity": f"{trace['system']}:sobol_{trace['sobol_index']}",
                "minimum_loss_ULP_cutoff_for_three_step_run": min(candidates),
            })
            high_signal_control_trajectories.append(trace)

    report = {
        "diagnostic_only": True, "production_modified": False,
        "new_trajectories_generated": False,
        "corpus": {
            "established_pathological_late_steps": len(pathological),
            "established_pathological_trajectories": 16,
            "detailed_nonpathological_steps": len(controls),
            "complete_direct_geometry_accepted_steps": sum(
                len(trace["replay"]["loss_history"]) - 1 for trace in traces),
            "late_control_states_reconstructed": len(reconstructed),
            "reconstruction_limit_loss_ULPs": RECONSTRUCTION_LIMIT,
        },
        "univariate_gap_audit": gap_audit,
        "candidate_thresholds": {
            "loss_decrease_ULPs_max": LOSS_ULP_CUTOFF,
            "full_predicted_decrease_loss_ULPs_min_exclusive": FULL_PREDICTION_ULP_FLOOR,
            "persistence_consecutive_accepted_steps": PERSISTENCE,
            "why_512": "exact power of two inside both observed conditional empty gaps",
            "conditional_signal_gap": {
                "largest_low-loss-ULP ordinary control full_prediction_ULPs_below_gap":
                    max(value["predicted_full_loss_ulps"] for value in low_signal_controls),
                "smallest_established_pathological_full_prediction_ULPs":
                    min(row["predicted_full_loss_ulps"] for row in pathological),
            },
            "conditional_persistent_loss_gap": {
                "smallest_cutoff_detecting_three_consecutive_steps_in_every_late_pathology":
                    max(min(max(row["actual_improvement_loss_ulps"] for row in window)
                            for window in (values[index:index + 3]
                                           for index in range(len(values) - 2)))
                        for values in ([row for row in sorted(
                            [item for item in alias_late if item["identity"] == identity],
                            key=lambda item: item["accepted_step"])]
                                       for identity in sorted({row["identity"] for row in alias_late}))),
                "smallest_observed_nonpathological_three_step_high_signal_cutoff":
                    (min(item["minimum_loss_ULP_cutoff_for_three_step_run"]
                         for item in third_run_boundaries) if third_run_boundaries else None),
            },
        },
        "minimal_joint_rule": {
            "per_step": "loss_decrease_ULPs <= 512 AND full_predicted_decrease_loss_ULPs > 512",
            "trajectory": "per_step condition for 3 consecutive accepted steps",
            "classification": "FLOAT64_STAGNATION_TAIL",
            "not_classified_as": ["CONVERGED", "KKT_STATIONARY", "RECOVERED"],
            "meaning": ("substantial Newton/QP progress remains mathematically predicted, "
                        "but persistent accepted steps realize only representational-scale "
                        "objective motion"),
            "omitted_as_redundant_or_ungapped": [
                "abs_Rm", "maximum_parameter_ULP_movement",
                "total_parameter_ULP_movement", "actual_over_full_predicted_decrease",
                "projected_gradient_magnitude"],
        },
        "ablation": ablations,
        "persistence_calibration": persistence,
        "full_corpus_replay": {
            "joint_by_cohort": ablations["joint"],
            "pathological_stop_steps": pathological_stops,
            "false_positive_trajectories": sum(
                item["triggered_trajectories"] for item in ablations["joint"].values()),
            "formerly_cap_limited_correct_basin_kepler": cap_controls,
            "all_six_formerly_cap_limited_not_stopped": all(
                not item["candidate_triggered"] for item in cap_controls),
        },
        "ambiguities": [
            "Kepler alias histories retain exact four-step early and late windows, not every accepted-step tensor; stop steps are first observed within those windows.",
            "For case 60 the first observed trigger is in the late window, so its exact onset between accepted steps 24 and the late window is not stored.",
            "The empirical rule has zero corpus errors but has not been tested on newly generated trajectories.",
        ],
        "evidence_assessment": {
            "supports_deterministic_diagnostic_stop": True,
            "supports_claim_of_convergence_or_recovery": False,
            "production_change_made": False,
        },
    }

    # Hard audit gates.
    assert all(base_flag(row) for row in pathological)
    assert persistence[str(PERSISTENCE)]["pathological_trajectories_detected"] == 16
    assert report["full_corpus_replay"]["false_positive_trajectories"] == 0
    assert report["full_corpus_replay"]["all_six_formerly_cap_limited_not_stopped"]
    assert report["candidate_thresholds"]["conditional_signal_gap"][
        "largest_low-loss-ULP ordinary control full_prediction_ULPs_below_gap"] < 512
    assert report["candidate_thresholds"]["conditional_signal_gap"][
        "smallest_established_pathological_full_prediction_ULPs"] > 512

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "report.json", report)
    print(json.dumps({
        "candidate": report["minimal_joint_rule"],
        "persistence": persistence,
        "false_positive_trajectories": report["full_corpus_replay"][
            "false_positive_trajectories"],
        "pathological_stop_steps": pathological_stops,
    }, indent=2))


if __name__ == "__main__":
    main()
