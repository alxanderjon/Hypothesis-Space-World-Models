"""Build and screen a Newton-correctability router from full-K SHO traces.

Labels come only from replaying the unchanged safeguarded N3 procedure.  The
teacher matrix is retained for offline diagnostics and is never included in
the runtime feature matrix.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

from experiments.sho.deterministic_full_k_framework import hashes, prior
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/stage1B/checkpointed_resource_edirect")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/stage1B/newton_correctability_router")
FINAL_RECOVERY = 1.0e-20
LOWER = np.full(4, prior.LOWER, dtype=np.float64)
UPPER = np.full(4, prior.UPPER, dtype=np.float64)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _case_math(states: np.ndarray):
    x = torch.from_numpy(states[:-1].copy())
    y = torch.from_numpy(states[1:].copy())

    def objective(theta: torch.Tensor) -> torch.Tensor:
        return torch.mean((x @ theta.reshape(2, 2).T - y) ** 2)

    def derivatives(theta: torch.Tensor, _unused: torch.Tensor):
        differentiable = theta.detach().clone().requires_grad_(True)
        loss = objective(differentiable)
        gradient = torch.autograd.grad(loss, differentiable, create_graph=True)[0]
        hessian = torch.stack([
            torch.autograd.grad(gradient[index], differentiable, retain_graph=True)[0]
            for index in range(len(differentiable))
        ])
        hessian = 0.5 * (hessian + hessian.T)
        return loss.detach(), gradient.detach(), hessian.detach()

    namespace = dict(polish.__dict__)
    namespace["_scalar_loss"] = lambda theta, _unused: objective(theta)
    namespace["_gradient_hessian"] = derivatives
    source = inspect.getsource(polish._newton_direction)
    source = source.replace(
        "torch.eye(2, dtype=DTYPE)",
        "torch.eye(len(gradient), dtype=DTYPE)",
    )
    exec(compile(source, polish.__file__, "exec"), namespace)
    run_n3 = types.FunctionType(polish._polish.__code__, namespace)
    return objective, derivatives, namespace["_newton_direction"], run_n3


def _safe_log(value: float) -> float:
    return math.log10(max(abs(value), np.finfo(np.float64).tiny))


FEATURE_NAMES = (
    "dimension_log2",
    "log10_loss",
    "log10_gradient_norm",
    "hessian_min_eigenvalue_over_scale",
    "hessian_max_eigenvalue_over_scale",
    "log10_hessian_condition",
    "hessian_positive",
    "direction_available",
    "log10_newton_step_norm",
    "descent_cosine",
    "log10_predicted_decrease",
    "predicted_decrease_over_loss",
    "normalized_position_abs_mean",
    "normalized_position_abs_max",
    "normalized_bound_margin_min",
    "normalized_bound_margin_mean",
    "search_progress",
    "log10_cell_extent",
    "incumbent_potentially_optimal",
    "log10_potentially_optimal_count",
)


def _features(theta: np.ndarray, objective, derivatives, direction_fn,
              search: dict) -> tuple[np.ndarray, dict]:
    tensor = torch.from_numpy(theta.copy())
    loss_tensor, gradient_tensor, hessian_tensor = derivatives(tensor, torch.empty(0))
    loss = float(loss_tensor)
    gradient = gradient_tensor.numpy()
    hessian = hessian_tensor.numpy()
    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    singular_values = np.linalg.svd(hessian, compute_uv=False)
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values[-1] > 0.0 else math.inf)
    direction_tensor, teacher_condition, damping, direction_status = direction_fn(
        gradient_tensor, hessian_tensor
    )
    direction = None if direction_tensor is None else direction_tensor.numpy()
    gradient_norm = float(np.linalg.norm(gradient))
    direction_norm = float(np.linalg.norm(direction)) if direction is not None else 0.0
    dot = float(np.dot(gradient, direction)) if direction is not None else 0.0
    cosine = (-dot / (gradient_norm * direction_norm)
              if gradient_norm > 0.0 and direction_norm > 0.0 else 0.0)
    predicted_decrease = max(0.0, -dot)
    lower = np.full_like(theta, prior.LOWER, dtype=np.float64)
    upper = np.full_like(theta, prior.UPPER, dtype=np.float64)
    normalized = 2.0 * (theta - lower) / (upper - lower) - 1.0
    margin = np.minimum(theta - lower, upper - theta) / (upper - lower)
    progress = float(search.get("evaluation", 0)) / float(search.get("case_cap", 5000))
    extent = float(search.get("incumbent_cell_extent", 0.0))
    po_count = float(search.get("potentially_optimal_cell_count", 0.0))
    vector = np.asarray((
        math.log2(len(theta)),
        _safe_log(loss),
        _safe_log(gradient_norm),
        float(eigenvalues[0] / scale),
        float(eigenvalues[-1] / scale),
        math.log10(condition) if math.isfinite(condition) else 308.0,
        float(eigenvalues[0] > 0.0),
        float(direction is not None),
        _safe_log(direction_norm),
        cosine,
        _safe_log(predicted_decrease),
        predicted_decrease / max(abs(loss), np.finfo(np.float64).tiny),
        float(np.mean(np.abs(normalized))),
        float(np.max(np.abs(normalized))),
        float(np.min(margin)),
        float(np.mean(margin)),
        progress,
        _safe_log(extent) if extent > 0.0 else -308.0,
        float(search.get("incumbent_is_potentially_optimal", False)),
        math.log10(max(1.0, po_count)),
    ), dtype=np.float64)
    diagnostic = dict(
        loss=loss,
        gradient=gradient.tolist(),
        gradient_norm=gradient_norm,
        hessian=hessian.tolist(),
        hessian_eigenvalues=eigenvalues.tolist(),
        hessian_condition=condition,
        teacher_direction_condition=teacher_condition,
        teacher_damping=damping,
        direction_status=direction_status,
        newton_direction=(direction.tolist() if direction is not None else None),
        newton_step_norm=(direction_norm if direction is not None else None),
        predicted_decrease=predicted_decrease,
    )
    return vector, diagnostic


def _run_label(theta: np.ndarray, loss: float, category: str, run_n3):
    endpoint = polish.N2Endpoint(
        trajectory_id=0,
        theta=theta.copy(),
        saved_loss=loss,
        source="newton_correctability_offline_label",
        category=category,
        previous_plateau_case=False,
    )
    result = run_n3(endpoint, torch.empty(0))
    final_theta = result.theta_history[-1]
    final_loss = float(result.loss_history[-1])
    return result, final_theta, final_loss, bool(final_loss < FINAL_RECOVERY)


def _split(case_id: int, category: str) -> str:
    digest = hashlib.sha256(f"{category}:{case_id}:newton-correctability".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "held_out"
    return "fresh"


def build_rows(limit_cases: int | None = None) -> tuple[list[dict], np.ndarray]:
    results = json.loads((ROOT / "results_partial.json").read_text())
    with np.load(DATA) as data:
        names = [str(value) for value in data["names"]]
        categories = [str(value) for value in data["categories"]]
        teachers = data["teachers"].reshape(-1, 4)
        prefixes = [data[f"prefix_{index}"].copy() for index in range(len(names))]
    by_name = {row["case"]: row for row in results}
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    for case_id, (name, category, teacher, states) in enumerate(
        zip(names, categories, teachers, prefixes)
    ):
        if limit_cases is not None and case_id >= limit_cases:
            break
        saved = by_name[name]
        objective, derivatives, direction_fn, run_n3 = _case_math(states)
        samples: list[tuple[str, str, np.ndarray, dict]] = []
        events = json.loads((ROOT / "cases" / name / "events.json").read_text())
        first_event = events[0]
        first_theta = LOWER + (UPPER - LOWER) * np.asarray(
            first_event["point"], dtype=np.float64
        )
        samples.append((
            "n1_initial",
            "evaluation_1",
            first_theta,
            dict(evaluation=1, case_cap=5000),
        ))
        first_loss_crossing = saved.get("first_loss_crossing")
        if first_loss_crossing is not None:
            crossing_evaluation = int(first_loss_crossing["evaluation"])
            crossing_event = events[crossing_evaluation - 1]
            crossing_theta = LOWER + (UPPER - LOWER) * np.asarray(
                crossing_event["point"], dtype=np.float64
            )
            samples.append((
                "n1_old_loss_crossing",
                f"evaluation_{crossing_evaluation}",
                crossing_theta,
                dict(evaluation=crossing_evaluation, case_cap=5000),
            ))
        for checkpoint_index, checkpoint in enumerate(saved["checkpoints"]):
            samples.append((
                "n1_checkpoint",
                f"checkpoint_{checkpoint_index}",
                np.asarray(checkpoint["best_theta"], dtype=np.float64),
                dict(checkpoint, case_cap=5000),
            ))
        handoff_evaluation = int(
            saved["first_both_checkpoint"]
            if saved["first_both_checkpoint"] is not None
            else saved["checkpoints"][-1]["evaluation"]
        )
        handoff = next(
            checkpoint for checkpoint in saved["checkpoints"]
            if int(checkpoint["evaluation"]) == handoff_evaluation
        )
        samples.append((
            "n1_handoff", "handoff", np.asarray(handoff["best_theta"], dtype=np.float64),
            dict(handoff, case_cap=5000),
        ))
        samples.append((
            "n2_endpoint", "endpoint", np.asarray(saved["n2_theta"], dtype=np.float64),
            dict(evaluation=saved["n2_evaluations"], case_cap=1500),
        ))
        for source_stage, source_id, theta, search in samples:
            features, diagnostic = _features(theta, objective, derivatives, direction_fn, search)
            n3, final_theta, final_loss, correctable = _run_label(
                theta, diagnostic["loss"], category, run_n3
            )
            truth_displacement = teacher - theta
            truth_distance = float(np.linalg.norm(truth_displacement))
            direction = diagnostic["newton_direction"]
            if direction is None or truth_distance == 0.0:
                alignment = None
                ratio = None
            else:
                direction_array = np.asarray(direction)
                direction_norm = float(np.linalg.norm(direction_array))
                alignment = (float(np.dot(direction_array, truth_displacement)
                                   / (direction_norm * truth_distance))
                             if direction_norm > 0.0 else None)
                ratio = direction_norm / truth_distance
            runtime_search_state = {
                key: search[key]
                for key in (
                    "evaluation", "iteration", "best_index", "best_theta", "best_loss",
                    "box_lower", "box_upper", "incumbent_cell_extent",
                    "incumbent_owned_probe_count", "incumbent_is_potentially_optimal",
                    "potentially_optimal_cell_count", "candidate_pool_count",
                    "incumbent_support_lower", "incumbent_support_upper",
                    "continuous_calls", "continuous_duplicate_rejections",
                    "continuous_surrogate_evaluations", "case_cap",
                )
                if key in search
            }
            row = dict(
                case=name,
                case_id=case_id,
                category=category,
                split=_split(case_id, category),
                source_stage=source_stage,
                source_id=source_id,
                source_evaluation=int(search.get("evaluation", 0)),
                theta=theta.tolist(),
                runtime_features=features.tolist(),
                offline_ground_truth=teacher.tolist(),
                offline_truth_distance=truth_distance,
                offline_newton_truth_alignment=alignment,
                offline_step_truth_distance_ratio=ratio,
                final_n3_theta=final_theta.tolist(),
                final_n3_loss=final_loss,
                newton_correctable=correctable,
                accepted_n3_steps=int(n3.accepted_steps),
                n3_objective_evaluations=int(n3.actual_loss_evaluations),
                n3_stop_reason=n3.stop_reason,
                **diagnostic,
                runtime_search_state=runtime_search_state,
            )
            rows.append(row)
            feature_rows.append(features)
    return rows, np.stack(feature_rows)


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return dict(count=0)
    return dict(
        count=int(len(array)),
        minimum=float(np.min(array)),
        median=float(np.median(array)),
        p90=float(np.quantile(array, 0.90, method="higher")),
        maximum=float(np.max(array)),
    )


def _group(rows: list[dict], key: str) -> dict:
    output = {}
    for value in sorted({row[key] for row in rows}):
        selected = [row for row in rows if row[key] == value]
        positives = sum(row["newton_correctable"] for row in selected)
        output[str(value)] = dict(
            states=len(selected),
            correctable=positives,
            not_correctable=len(selected) - positives,
            correctable_fraction=positives / len(selected),
        )
    return output


def save_dataset(rows: list[dict], features: np.ndarray, before: dict, after: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row["newton_correctable"] for row in rows], dtype=np.int64)
    np.savez_compressed(
        OUT / "newton_correctability_dataset.npz",
        features=features,
        labels=labels,
        feature_names=np.asarray(FEATURE_NAMES),
        case_ids=np.asarray([row["case_id"] for row in rows]),
        cases=np.asarray([row["case"] for row in rows]),
        categories=np.asarray([row["category"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        source_stages=np.asarray([row["source_stage"] for row in rows]),
        source_evaluations=np.asarray([row["source_evaluation"] for row in rows]),
        theta=np.asarray([row["theta"] for row in rows]),
        offline_ground_truth=np.asarray([row["offline_ground_truth"] for row in rows]),
        final_n3_theta=np.asarray([row["final_n3_theta"] for row in rows]),
        final_n3_loss=np.asarray([row["final_n3_loss"] for row in rows]),
    )
    _atomic_text(OUT / "label_records.json", json.dumps(rows, indent=2))
    compact_fields = (
        "case", "case_id", "category", "split", "source_stage", "source_id",
        "source_evaluation", "theta", "loss", "gradient_norm", "hessian_condition",
        "newton_step_norm", "predicted_decrease", "newton_correctable",
        "final_n3_theta", "final_n3_loss", "accepted_n3_steps", "n3_stop_reason",
        "offline_ground_truth", "offline_truth_distance",
        "offline_newton_truth_alignment", "offline_step_truth_distance_ratio",
    )
    with (OUT / "dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in compact_fields} for row in rows)
    positives = [row for row in rows if row["newton_correctable"]]
    negatives = [row for row in rows if not row["newton_correctable"]]
    case_names = sorted({row["case"] for row in rows})
    earliest_correctable = {
        case: min(
            row["source_evaluation"]
            for row in rows
            if row["case"] == case
            and row["source_stage"].startswith("n1_")
            and row["newton_correctable"]
        )
        for case in case_names
    }
    source_results = json.loads((ROOT / "results_partial.json").read_text())
    old_handoffs = {
        row["case"]: int(
            row["first_both_checkpoint"]
            if row["first_both_checkpoint"] is not None
            else row["total_n1_evaluations"]
        )
        for row in source_results
    }
    old_loss_crossings = {
        row["case"]: (
            int(row["first_loss_crossing"]["evaluation"])
            if row["first_loss_crossing"] is not None else None
        )
        for row in source_results
    }
    diagnostic_savings = [
        old_handoffs[case] - earliest_correctable[case] for case in case_names
    ]
    report = dict(
        label_definition=(
            "NEWTON_CORRECTABLE iff unchanged N3 started at theta ends with "
            "final_loss < 1e-20, the existing full-K final-recovery criterion"
        ),
        states=len(rows),
        cases=len({row["case"] for row in rows}),
        correctable=len(positives),
        not_correctable=len(negatives),
        by_source_stage=_group(rows, "source_stage"),
        by_boundary_category=_group(rows, "category"),
        by_split=_group(rows, "split"),
        correctable_truth_distance=_stats([
            row["offline_truth_distance"] for row in positives
        ]),
        not_correctable_truth_distance=_stats([
            row["offline_truth_distance"] for row in negatives
        ]),
        largest_correctable=max(
            positives, key=lambda row: row["offline_truth_distance"], default=None
        ),
        feature_names=list(FEATURE_NAMES),
        ground_truth_features_excluded=True,
        runtime_feature_representation=(
            "Twenty scalar invariants aggregate local objective geometry, Newton state, "
            "normalized bound position, and available search geometry; vector length does "
            "not depend on parameter dimension. Raw ground truth and truth-derived "
            "distances/alignment are excluded."
        ),
        n2_intermediate_availability=(
            "The completed artifacts contain N2 endpoints but no Powell iterate history."
        ),
        checkpoint_hashes_unchanged=before == after,
        sequential_diagnostic=dict(
            earliest_correctable_evaluation=_stats(list(earliest_correctable.values())),
            earliest_correctable_all_cases=all(
                value == 1 for value in earliest_correctable.values()
            ),
            old_loss_crossing_missing=sum(
                value is None for value in old_loss_crossings.values()
            ),
            old_loss_crossing_evaluation=_stats([
                value for value in old_loss_crossings.values() if value is not None
            ]),
            old_handoff_evaluation=_stats(list(old_handoffs.values())),
            diagnostic_evaluations_saved_vs_old_handoff=_stats(diagnostic_savings),
            diagnostic_total_evaluations_saved=int(sum(diagnostic_savings)),
            old_n2_gate_successes=sum(row["n2_gate"] for row in source_results),
            old_n2_gate_failures=sum(not row["n2_gate"] for row in source_results),
        ),
    )
    if len(np.unique(labels)) < 2:
        report.update(
            status="BLOCKED_NO_NEGATIVE_CLASS",
            architecture_screen=dict(
                model="zero-parameter constant NEWTON_CORRECTABLE baseline",
                parameter_count=0,
                train_accuracy=1.0,
                held_out_accuracy=1.0,
                fresh_accuracy=1.0,
                false_positives=0,
                false_positive_rate=None,
                false_negatives=0,
                negative_support=0,
                rejected_reason=(
                    "No NOT_NEWTON_CORRECTABLE state exists, so false-positive "
                    "generalization and a decision boundary are untestable."
                ),
            ),
            training_performed=False,
            checkpoint_saved=False,
            diagnosis=(
                "The unchanged N3 procedure recovered every saved state. The full-K SHO "
                "objective is a convex quadratic with a state-independent positive Hessian, "
                "so these traces do not identify a NEWTON-correctability boundary."
            ),
        )
    else:
        report.update(status="READY_FOR_ARCHITECTURE_SCREEN")
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    return report


def main() -> None:
    before = hashes()
    rows, features = build_rows()
    after = hashes()
    report = save_dataset(rows, features, before, after)
    print(json.dumps({
        key: report[key]
        for key in ("status", "states", "cases", "correctable", "not_correctable",
                    "by_source_stage", "by_boundary_category",
                    "checkpoint_hashes_unchanged")
    }, indent=2))


if __name__ == "__main__":
    main()
