"""Build a cross-system Newton-correctability corpus without changing N1/N2/N3.

Labels are obtained only by running the current safeguarded N3 procedure.  The
ground truth is retained for offline diagnostics and is never included in the
runtime feature array.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

from experiments.bouncing_ball import run_full_k_transfer as ball
from experiments.kepler import run_four_parameter_peelback as kepler
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/cross_system_newton_correctability")
CASES = OUT / "cases"
SHO_DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
KEPLER_DATA = Path(
    "outputs/kepler/four_parameter_deterministic_peelback/correctability_states.json"
)
FINAL_GATE = 1.0e-20
FINAL_HALF_WIDTH = 31.84
SHO_LOWER = np.full(4, -FINAL_HALF_WIDTH, dtype=np.float64)
SHO_UPPER = np.full(4, FINAL_HALF_WIDTH, dtype=np.float64)
BALL_LOWER = SHO_LOWER.copy()
BALL_UPPER = SHO_UPPER.copy()
KEPLER_LOWER = kepler.LOWER.copy()
KEPLER_UPPER = kepler.UPPER.copy()
SHORT_CAP = 150
CHECKPOINTS = (50, 100, 150)
INITIAL_SAMPLE_INDICES = (0, 4, 9, 14, 19)
SHO_CASES = 12
BALL_CASES = 18

FEATURE_NAMES = (
    "dimension_log2",
    "log10_loss_over_gradient_scale",
    "log10_gradient_over_curvature_scale",
    "log10_loss_over_curvature_scale",
    "scaled_hessian_min_over_spectral_scale",
    "scaled_hessian_max_over_spectral_scale",
    "negative_curvature_fraction",
    "log10_scaled_hessian_condition",
    "direct_route",
    "direction_available",
    "log10_normalized_step_l2",
    "log10_normalized_step_linf",
    "descent_cosine",
    "log10_predicted_decrease_over_loss",
    "log10_damping_over_curvature_scale",
    "normalized_position_abs_mean",
    "normalized_position_abs_max",
    "normalized_bound_margin_min",
    "normalized_bound_margin_mean",
    "proposed_step_inside_bounds",
    "search_progress",
)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    tiny = np.finfo(np.float64).tiny
    value = math.log10(max(abs(numerator), tiny) / max(abs(denominator), tiny))
    return float(np.clip(value, -308.0, 308.0))


def _split(system: str, case_id: int) -> str:
    # Whole-case partitions, fixed before labels are inspected.
    if system == "kepler":
        return "train" if case_id <= 4 else ("held_out" if case_id == 5 else "fresh")
    if system == "bouncing_ball":
        return "train" if case_id <= 8 else ("held_out" if case_id <= 12 else "fresh")
    return "train" if case_id <= 7 else ("held_out" if case_id <= 9 else "fresh")


def _objective_geometry(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                        derivatives, direction_fn) -> tuple[np.ndarray, dict]:
    tensor = torch.from_numpy(theta.copy())
    loss_t, gradient_t, hessian_t = derivatives(tensor, torch.empty(0))
    loss = float(loss_t)
    gradient = gradient_t.numpy()
    hessian = hessian_t.numpy()
    width = upper - lower
    scaled_gradient = gradient * width
    scaled_hessian = hessian * np.outer(width, width)
    eigenvalues = np.linalg.eigvalsh(scaled_hessian)
    singular = np.linalg.svd(scaled_hessian, compute_uv=False)
    curvature_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    gradient_scale = float(np.linalg.norm(scaled_gradient))
    condition = (float(singular[0] / singular[-1])
                 if singular[-1] > 0.0 else math.inf)
    direction_t, teacher_condition, damping, status = direction_fn(gradient_t, hessian_t)
    direction = None if direction_t is None else direction_t.numpy()
    normalized_step = (np.zeros_like(theta) if direction is None else direction / width)
    gradient_norm = float(np.linalg.norm(gradient))
    direction_norm = 0.0 if direction is None else float(np.linalg.norm(direction))
    dot = 0.0 if direction is None else float(np.dot(gradient, direction))
    descent_cosine = (-dot / (gradient_norm * direction_norm)
                      if gradient_norm > 0.0 and direction_norm > 0.0 else 0.0)
    predicted_decrease = max(0.0, -dot)
    normalized_position = 2.0 * (theta - lower) / width - 1.0
    margin = np.minimum(theta - lower, upper - theta) / width
    inside = bool(direction is not None
                  and np.all(theta + direction >= lower)
                  and np.all(theta + direction <= upper))
    direct = bool(eigenvalues[0] > 0.0 and condition <= polish.CONDITION_LIMIT)
    damping_scaled = float(damping) * float(np.max(width) ** 2)
    diagnostics = dict(
        loss=loss,
        gradient=gradient.tolist(),
        gradient_norm=gradient_norm,
        hessian=hessian.tolist(),
        scaled_hessian_eigenvalues=eigenvalues.tolist(),
        scaled_hessian_condition=condition,
        teacher_direction_condition=teacher_condition,
        teacher_damping=damping,
        direction_status=status,
        newton_direction=(None if direction is None else direction.tolist()),
        normalized_newton_step=normalized_step.tolist(),
        newton_step_norm=direction_norm,
        predicted_decrease=predicted_decrease,
        direct_route=direct,
    )
    features = np.asarray((
        math.log2(len(theta)),
        _safe_log_ratio(loss, gradient_scale),
        _safe_log_ratio(gradient_scale, curvature_scale),
        _safe_log_ratio(loss, curvature_scale),
        float(eigenvalues[0] / curvature_scale),
        float(eigenvalues[-1] / curvature_scale),
        float(np.mean(eigenvalues < 0.0)),
        (math.log10(condition) if math.isfinite(condition) else 308.0),
        float(direct),
        float(direction is not None),
        _safe_log_ratio(float(np.linalg.norm(normalized_step)), 1.0),
        _safe_log_ratio(float(np.max(np.abs(normalized_step))), 1.0),
        descent_cosine,
        _safe_log_ratio(predicted_decrease, loss),
        _safe_log_ratio(damping_scaled, curvature_scale),
        float(np.mean(np.abs(normalized_position))),
        float(np.max(np.abs(normalized_position))),
        float(np.min(margin)),
        float(np.mean(margin)),
        float(inside),
        0.0,  # filled by the caller
    ), dtype=np.float64)
    return features, diagnostics


def _label(theta: np.ndarray, loss: float, system: str, run_n3) -> dict:
    result = kepler._run_n3(theta, loss, f"cross_system_{system}", run_n3)
    return dict(
        newton_correctable=bool(result["final_loss"] < FINAL_GATE),
        final_theta=result["final_theta"],
        final_loss=result["final_loss"],
        accepted_steps=result["accepted_steps"],
        objective_evaluations=result["objective_evaluations"],
        stop_reason=result["stop_reason"],
        backtracking_rejections=result["backtracking_rejections"],
        damping_steps=result["damping_steps"],
    )


def _short_search(objective, lower: np.ndarray, upper: np.ndarray, seed: int) -> dict:
    width = upper - lower
    api = edirect.composition(4)
    traces: list[dict] = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
    points: list[np.ndarray] = []
    losses: list[float] = []
    snapshots: list[dict] = []
    initial = edirect.initial_design(4)

    def evaluate(unit: np.ndarray, source: str, iteration: int) -> None:
        theta = lower + width * np.asarray(unit, dtype=np.float64)
        value = float(objective(theta))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite objective during short E-DIRECT trace")
        points.append(np.asarray(unit, dtype=np.float64).copy())
        losses.append(value)
        if len(losses) in CHECKPOINTS:
            best = int(np.argmin(losses))
            snapshots.append(dict(
                source="n1_checkpoint", evaluation=len(losses), iteration=iteration,
                theta=(lower + width * points[best]).tolist(), loss=float(losses[best]),
                best_index=best, continuous_proposals=len(traces),
            ))

    for index, unit in enumerate(initial):
        evaluate(unit, "initial", 0)
    initial_samples = [dict(
        source="n1_initial", evaluation=index + 1, iteration=0,
        theta=(lower + width * initial[index]).tolist(),
        loss=float(losses[index]), best_index=index, continuous_proposals=0,
    ) for index in INITIAL_SAMPLE_INDICES]
    iteration = 0
    while len(losses) < SHORT_CAP:
        proposals = api["_cell_proposals"](
            np.asarray(points), np.asarray(losses), seed, iteration
        )
        for unit in proposals:
            if len(losses) >= SHORT_CAP:
                break
            evaluate(unit, "cell", iteration)
        if len(losses) < SHORT_CAP:
            proposal = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), seed, iteration
            )
            if proposal is not None:
                evaluate(proposal, "continuous_surrogate", iteration)
        iteration += 1
    return dict(
        samples=initial_samples + snapshots,
        evaluations=len(losses), iterations=iteration,
        continuous_proposals=len(traces),
        duplicate_rejections=int(sum(row["exact_duplicate_rejected"] for row in traces)),
    )


def _sho_case(case_id: int) -> tuple[list[dict], dict]:
    with np.load(SHO_DATA) as data:
        total = len(data["names"])
        indices = np.linspace(0, total - 1, SHO_CASES, dtype=int)
        source_index = int(indices[case_id])
        name = str(data["names"][source_index])
        category = str(data["categories"][source_index])
        teacher = data["teachers"].reshape(-1, 4)[source_index].copy()
        states = data[f"prefix_{source_index}"].copy()
    x = torch.from_numpy(states[:-1].copy())
    y = torch.from_numpy(states[1:].copy())

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        return torch.mean((x @ theta.reshape(2, 2).T - y) ** 2)

    def numpy_objective(theta: np.ndarray) -> float:
        prediction = states[:-1] @ np.asarray(theta).reshape(2, 2).T
        return float(np.mean((prediction - states[1:]) ** 2))

    derivatives, direction_fn, run_n3 = kepler._make_n3(
        torch_objective, SHO_LOWER, SHO_UPPER
    )
    search = _short_search(numpy_objective, SHO_LOWER, SHO_UPPER, 7_000_000 + case_id)
    rows = _rows_from_samples(
        "sho", case_id, name, category, teacher, SHO_LOWER, SHO_UPPER,
        search["samples"], derivatives, direction_fn, run_n3,
    )
    return rows, {key: value for key, value in search.items() if key != "samples"}


def _ball_case(case_id: int) -> tuple[list[dict], dict]:
    trajectories, _, _, _ = ball.ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=BALL_CASES, n_timesteps=ball.ball.TOTAL_STEPS,
        dt=ball.ball.DT, seed=20261101,
    )
    trajectory = trajectories[case_id]
    numpy_objective, torch_objective = ball.objectives(trajectory)
    derivatives, direction_fn, run_n3 = kepler._make_n3(
        torch_objective, BALL_LOWER, BALL_UPPER
    )
    search = _short_search(
        numpy_objective, BALL_LOWER, BALL_UPPER, 8_000_000 + case_id
    )
    rows = _rows_from_samples(
        "bouncing_ball", case_id, f"trajectory_{case_id:02d}", "event_driven",
        ball.TEACHER, BALL_LOWER, BALL_UPPER, search["samples"],
        derivatives, direction_fn, run_n3,
    )
    return rows, {key: value for key, value in search.items() if key != "samples"}


def _rows_from_samples(system: str, case_id: int, case_name: str, category: str,
                       teacher: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                       samples: list[dict], derivatives, direction_fn, run_n3) -> list[dict]:
    rows = []
    consecutive_checkpoint_positives = 0
    for sample in samples:
        theta = np.asarray(sample["theta"], dtype=np.float64)
        features, diagnostic = _objective_geometry(
            theta, lower, upper, derivatives, direction_fn
        )
        features[-1] = float(sample["evaluation"]) / SHORT_CAP
        label = _label(theta, diagnostic["loss"], system, run_n3)
        if sample["source"] == "n1_checkpoint":
            consecutive_checkpoint_positives = (
                consecutive_checkpoint_positives + 1
                if label["newton_correctable"] else 0
            )
        displacement = teacher - theta
        direction = diagnostic["newton_direction"]
        direction_array = None if direction is None else np.asarray(direction)
        distance = float(np.linalg.norm(displacement))
        alignment = None
        if direction_array is not None and distance > 0.0:
            norm = float(np.linalg.norm(direction_array))
            if norm > 0.0:
                alignment = float(np.dot(direction_array, displacement) / (norm * distance))
        rows.append(dict(
            system=system, case_id=case_id, case_name=case_name, category=category,
            split=_split(system, case_id), source=sample["source"],
            evaluation=sample["evaluation"], theta=theta.tolist(),
            bounds=[lower.tolist(), upper.tolist()], runtime_features=features.tolist(),
            offline_ground_truth=teacher.tolist(), offline_truth_distance=distance,
            offline_newton_truth_alignment=alignment,
            **diagnostic, **label,
        ))
        # Three consecutive correctable checkpoints establish sustained entry.
        if sample["source"] == "n1_checkpoint" and consecutive_checkpoint_positives >= 3:
            break
    return rows


def _kepler_rows() -> list[dict]:
    saved = json.loads(KEPLER_DATA.read_text())
    selected = [row for row in saved
                if row["source"] == "n1_state" and row["evaluation"] in (1, 100, 500, 1000)]
    output = []
    width = KEPLER_UPPER - KEPLER_LOWER
    for row in selected:
        theta = np.asarray(row["theta"], dtype=np.float64)
        gradient = np.asarray(row["gradient"], dtype=np.float64)
        eigenvalues = np.asarray(row["hessian_eigenvalues"], dtype=np.float64) * width[0] ** 2
        curvature_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
        scaled_gradient_norm = float(np.linalg.norm(gradient * width))
        direction = np.asarray(row["newton_direction"], dtype=np.float64)
        normalized_step = direction / width
        loss = float(row["loss"])
        predicted = max(0.0, -float(np.dot(gradient, direction)))
        normalized = 2.0 * (theta - KEPLER_LOWER) / width - 1.0
        margin = np.minimum(theta - KEPLER_LOWER, KEPLER_UPPER - theta) / width
        features = np.asarray((
            2.0,
            _safe_log_ratio(loss, scaled_gradient_norm),
            _safe_log_ratio(scaled_gradient_norm, curvature_scale),
            _safe_log_ratio(loss, curvature_scale),
            float(eigenvalues[0] / curvature_scale),
            float(eigenvalues[-1] / curvature_scale),
            float(np.mean(eigenvalues < 0.0)),
            math.log10(float(row["hessian_condition"])),
            float(eigenvalues[0] > 0.0 and row["hessian_condition"] <= polish.CONDITION_LIMIT),
            1.0,
            _safe_log_ratio(float(np.linalg.norm(normalized_step)), 1.0),
            _safe_log_ratio(float(np.max(np.abs(normalized_step))), 1.0),
            float(-np.dot(gradient, direction)
                  / (np.linalg.norm(gradient) * np.linalg.norm(direction))),
            _safe_log_ratio(predicted, loss),
            _safe_log_ratio(float(row["newton_damping"]) * width[0] ** 2,
                            curvature_scale),
            float(np.mean(np.abs(normalized))), float(np.max(np.abs(normalized))),
            float(np.min(margin)), float(np.mean(margin)),
            float(np.all(theta + direction >= KEPLER_LOWER)
                  and np.all(theta + direction <= KEPLER_UPPER)),
            float(row["evaluation"]) / 5000.0,
        ), dtype=np.float64)
        copied = dict(row)
        copied.update(
            system="kepler", case_name=f"trajectory_{row['case_id']:02d}",
            category="eccentric", split=_split("kepler", int(row["case_id"])),
            bounds=[KEPLER_LOWER.tolist(), KEPLER_UPPER.tolist()],
            runtime_features=features.tolist(),
            newton_correctable=bool(row["n3_recovered"]),
            offline_ground_truth=kepler.TEACHER.tolist(),
            offline_truth_distance=row["truth_distance"],
            offline_newton_truth_alignment=row["newton_truth_alignment"],
        )
        output.append(copied)
    return output


def _write_case(system: str, case_id: int) -> None:
    path = CASES / system / f"case_{case_id:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    rows, search = _sho_case(case_id) if system == "sho" else _ball_case(case_id)
    _atomic_text(path, json.dumps(dict(rows=rows, search=search), indent=2))
    print(json.dumps(dict(system=system, case=case_id, states=len(rows),
                          positives=sum(row["newton_correctable"] for row in rows))), flush=True)


def _domain_probe_record() -> dict:
    # Exact aggregates from the deterministic direct-N3 preflight performed before
    # short traces. Each level used all 20 dimension-general initial-design points.
    return dict(
        schedule_half_widths=[1.99, 3.98, 7.96, 15.92, 31.84],
        policy="symmetric doubling from the existing 1.99 half-width",
        sho=[dict(half_width=value, states=20, correctable=20, not_correctable=0)
             for value in (1.99, 3.98, 7.96, 15.92, 31.84)],
        bouncing_ball=[
            dict(half_width=1.99, states=20, correctable=20, not_correctable=0,
                 objective_min=0.5034450411286512, objective_max=25.06282161164525),
            dict(half_width=3.98, states=20, correctable=20, not_correctable=0,
                 objective_min=0.7915658351560889, objective_max=89.93591428519356),
            dict(half_width=7.96, states=20, correctable=20, not_correctable=0,
                 objective_min=1.094377923319283, objective_max=342.05734831583607),
            dict(half_width=15.92, states=20, correctable=20, not_correctable=0,
                 objective_min=3.1708925211912717, objective_max=1335.8012111113035),
            dict(half_width=31.84, states=20, correctable=20, not_correctable=0,
                 objective_min=13.075248922414703, objective_max=5281.292915638972),
        ],
        stopped_reason=(
            "No meaningful negative class appeared after a systematic 16x expansion; "
            "further expansion would target numerical overflow rather than a natural basin boundary."
        ),
    )


def _summary(rows: list[dict], before: dict, after: dict) -> dict:
    grouped = {}
    for system in ("sho", "bouncing_ball", "kepler"):
        values = [row for row in rows if row["system"] == system]
        grouped[system] = dict(
            states=len(values), correctable=sum(row["newton_correctable"] for row in values),
            not_correctable=sum(not row["newton_correctable"] for row in values),
            cases=len({row["case_id"] for row in values}),
            by_split={split: dict(
                states=sum(row["split"] == split for row in values),
                correctable=sum(row["split"] == split and row["newton_correctable"] for row in values),
                not_correctable=sum(row["split"] == split and not row["newton_correctable"] for row in values),
            ) for split in ("train", "held_out", "fresh")},
        )
    negative_systems = [system for system, value in grouped.items()
                        if value["not_correctable"] > 0]
    positive_systems = [system for system, value in grouped.items()
                        if value["correctable"] > 0]
    sequential = {}
    for system in grouped:
        cases = sorted({row["case_id"] for row in rows if row["system"] == system})
        records = []
        for case_id in cases:
            values = sorted(
                [row for row in rows if row["system"] == system and row["case_id"] == case_id],
                key=lambda row: (row["evaluation"], row["source"]),
            )
            actual = [row for row in values if row["newton_correctable"]]
            checkpoints = [row for row in values if row["source"] == "n1_checkpoint"]
            stable = all(row["newton_correctable"] for row in checkpoints[
                next((i for i, row in enumerate(checkpoints) if row["newton_correctable"]),
                     len(checkpoints)):
            ])
            records.append(dict(
                case_id=case_id,
                first_actual_correctable_evaluation=(min(row["evaluation"] for row in actual)
                                                     if actual else None),
                sustained_after_entry=stable,
            ))
        sequential[system] = records
    status = ("READY_FOR_TRAINING" if len(negative_systems) >= 2 and len(positive_systems) >= 2
              else "BLOCKED_NEGATIVE_CLASS_ONLY_ONE_SYSTEM")
    return dict(
        status=status,
        label=("NEWTON_CORRECTABLE iff unchanged current N3 reaches the existing "
               "ground-truth-equivalent final recovery criterion (final loss < 1e-20)"),
        feature_names=FEATURE_NAMES,
        runtime_feature_count=len(FEATURE_NAMES),
        ground_truth_features_excluded=True,
        final_bounds=dict(
            sho=[SHO_LOWER.tolist(), SHO_UPPER.tolist()],
            bouncing_ball=[BALL_LOWER.tolist(), BALL_UPPER.tolist()],
            kepler=[KEPLER_LOWER.tolist(), KEPLER_UPPER.tolist()],
        ),
        systems=grouped,
        total_states=len(rows),
        correctable=sum(row["newton_correctable"] for row in rows),
        not_correctable=sum(not row["newton_correctable"] for row in rows),
        systems_with_positive_class=positive_systems,
        systems_with_negative_class=negative_systems,
        training_performed=False,
        checkpoint_saved=False,
        architecture_screen=None,
        block_reason=(None if status == "READY_FOR_TRAINING" else
                      "Only Kepler supplies NOT_NEWTON_CORRECTABLE states; a shared router's "
                      "cross-system false-positive behavior cannot be validated."),
        sequential_diagnostic=sequential,
        checkpoint_hashes_unchanged=before == after,
        protected_checkpoint_count=len(before),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-system", choices=("sho", "bouncing_ball"))
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.case_system is not None:
        _write_case(args.case_system, int(args.case_index))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    before = hashes()
    _atomic_text(OUT / "checkpoint_hashes_before.json", json.dumps(before, indent=2))
    probe = _domain_probe_record()
    _atomic_text(OUT / "domain_probe.json", json.dumps(probe, indent=2))
    jobs = [("sho", index) for index in range(SHO_CASES)]
    jobs += [("bouncing_ball", index) for index in range(BALL_CASES)]
    workers = max(1, min(args.workers, 6))
    for offset in range(0, len(jobs), workers):
        batch = jobs[offset:offset + workers]
        children = []
        for system, case_id in batch:
            path = CASES / system / f"case_{case_id:02d}.json"
            if path.exists():
                continue
            command = [sys.executable, "-m", "experiments.cross_system_newton_correctability",
                       "--case-system", system, "--case-index", str(case_id)]
            children.append((system, case_id, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(system, case_id, child.wait()) for system, case_id, child in children]
        failures = [value for value in failures if value[2] != 0]
        if failures:
            raise RuntimeError(f"case failures: {failures}")
    rows = _kepler_rows()
    search_summaries = []
    for system, count in (("sho", SHO_CASES), ("bouncing_ball", BALL_CASES)):
        for case_id in range(count):
            record = json.loads((CASES / system / f"case_{case_id:02d}.json").read_text())
            for row in record["rows"]:
                row["split"] = _split(system, case_id)
            rows.extend(record["rows"])
            search_summaries.append(dict(system=system, case_id=case_id, **record["search"]))
    rows.sort(key=lambda row: (row["system"], row["case_id"], row["evaluation"], row["source"]))
    _atomic_text(OUT / "label_records.json", json.dumps(rows, indent=2))
    _atomic_text(OUT / "search_summaries.json", json.dumps(search_summaries, indent=2))
    features = np.asarray([row["runtime_features"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["newton_correctable"] for row in rows], dtype=np.int64)
    np.savez_compressed(
        OUT / "cross_system_correctability_dataset.npz",
        features=features, labels=labels,
        systems=np.asarray([row["system"] for row in rows]),
        case_ids=np.asarray([row["case_id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    with (OUT / "dataset.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("system", "case_id", "case_name", "split", "source", "evaluation",
                         "newton_correctable", *FEATURE_NAMES))
        for row in rows:
            writer.writerow((row["system"], row["case_id"], row["case_name"], row["split"],
                             row["source"], row["evaluation"], row["newton_correctable"],
                             *row["runtime_features"]))
    after = hashes()
    _atomic_text(OUT / "checkpoint_hashes_after.json", json.dumps(after, indent=2))
    report = _summary(rows, before, after)
    report["domain_probe"] = probe
    report["short_search"] = dict(
        maximum_evaluations=SHORT_CAP, checkpoints=list(CHECKPOINTS),
        exhaustive_5000_evaluation_runs=0,
        summaries=search_summaries,
    )
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "status", "total_states", "correctable", "not_correctable",
        "systems_with_negative_class", "training_performed", "checkpoint_hashes_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()
