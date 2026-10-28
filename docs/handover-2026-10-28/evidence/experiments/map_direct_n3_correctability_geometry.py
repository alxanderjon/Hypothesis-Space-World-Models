#!/usr/bin/env python3
"""Map direct safeguarded-N3 correctability on existing parameter domains.

This is a diagnostic only.  It does not run N1/N2, alter N3, train a model,
or use the known teacher in any runtime feature.  The teacher is retained only
for offline distance diagnostics after N3 has supplied the operational label.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch
from scipy.stats import qmc

from experiments import audit_known_structure as known
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/direct_n3_correctability_geometry")
ROWS = OUT / "points.jsonl"
FINAL_GATE = 1.0e-20
INITIAL_POWER = 5  # 32 points
MAX_POWER = 8      # at most 256 points; staged stopping usually ends earlier
MINORITY_TARGET = 20


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _system_config(name: str) -> dict:
    if name == "sho":
        data = np.load("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
        case = 0
        states = np.asarray(data[f"prefix_{case}"], dtype=np.float64)
        teacher = np.asarray(data["teachers"], dtype=np.float64).reshape(-1, 4)[case]
        inputs = torch.from_numpy(states[:-1].copy())
        targets = torch.from_numpy(states[1:].copy())

        def objective(theta: torch.Tensor) -> torch.Tensor:
            return torch.mean((inputs @ theta.reshape(2, 2).T - targets) ** 2)

        lower = np.full(4, -1.99, dtype=np.float64)
        upper = np.full(4, 1.99, dtype=np.float64)
        seed = 20261021
        label = "validation_case_000"
    elif name == "bouncing_ball":
        case = 3  # existing two-prefix-impact reflection trajectory
        trajectory = ball._trajectories()[case]
        _, objective = ball.objectives(trajectory)
        teacher = ball.TEACHER.copy()
        lower, upper = ball.LOWER.copy(), ball.UPPER.copy()
        seed = 20261022
        label = "reflection_source_case_003"
    elif name == "kepler":
        case = 0
        trajectory = staged.trajectories()[case]
        inputs = torch.from_numpy(trajectory[:127].copy())
        targets = torch.from_numpy(trajectory[1:128].copy())

        def objective(theta: torch.Tensor) -> torch.Tensor:
            return (known.step(inputs, theta, torch) - targets).square().mean()

        teacher = known.GT.copy()
        lower, upper = known.LOW.copy(), known.HIGH.copy()
        seed = 20261023
        label = "five_parameter_case_000"
    else:
        raise ValueError(name)

    derivatives, direction_fn, run_n3 = staged._make_n3(objective, lower, upper)
    return dict(name=name, case=case, label=label, teacher=teacher,
                lower=lower, upper=upper, objective=objective,
                derivatives=derivatives, direction_fn=direction_fn,
                run_n3=run_n3, seed=seed)


def _local_diagnostics(theta: np.ndarray, config: dict) -> dict:
    lower, upper = config["lower"], config["upper"]
    width = upper - lower
    tensor = torch.from_numpy(theta.copy())
    loss_t, gradient_t, hessian_t = config["derivatives"](tensor, torch.empty(0))
    gradient = gradient_t.numpy()
    hessian = hessian_t.numpy()
    scaled_gradient = gradient * width
    scaled_hessian = hessian * np.outer(width, width)
    raw_eigenvalues = np.linalg.eigvalsh(hessian)
    scaled_eigenvalues = np.linalg.eigvalsh(scaled_hessian)
    raw_singular = np.linalg.svd(hessian, compute_uv=False)
    scaled_singular = np.linalg.svd(scaled_hessian, compute_uv=False)
    raw_condition = (float(raw_singular[0] / raw_singular[-1])
                     if raw_singular[-1] > 0.0 else math.inf)
    scaled_condition = (float(scaled_singular[0] / scaled_singular[-1])
                        if scaled_singular[-1] > 0.0 else math.inf)
    direction_t, direction_condition, damping, direction_status = config["direction_fn"](
        gradient_t, hessian_t
    )
    direction = None if direction_t is None else direction_t.numpy()
    normalized_position = (theta - lower) / width
    bound_margins = np.minimum(theta - lower, upper - theta) / width
    if direction is None:
        normalized_step = np.full_like(theta, np.nan)
        dot = math.nan
        descent_cosine = math.nan
        predicted_decrease = math.nan
        inside = False
        step_norm = math.nan
    else:
        normalized_step = direction / width
        dot = float(np.dot(gradient, direction))
        gradient_norm = float(np.linalg.norm(gradient))
        step_norm = float(np.linalg.norm(direction))
        descent_cosine = (-dot / (gradient_norm * step_norm)
                          if gradient_norm > 0.0 and step_norm > 0.0 else 0.0)
        predicted_decrease = max(0.0, -dot)
        candidate = theta + direction
        inside = bool(np.all(candidate >= lower) and np.all(candidate <= upper))
    direct = bool(direction is not None and float(damping) == 0.0)
    damping_trial_count = None
    if direction is not None and math.isfinite(float(damping)):
        if float(damping) == 0.0:
            damping_trial_count = 1
        else:
            scale = max(float(np.max(np.abs(raw_eigenvalues))), np.finfo(np.float64).tiny)
            direct_allowed = bool(raw_eigenvalues[0] > 0.0 and
                                  raw_condition <= polish.CONDITION_LIMIT)
            initial = max(scale * polish.DAMPING_RELATIVE_FLOOR,
                          -float(raw_eigenvalues[0]) +
                          scale * polish.DAMPING_RELATIVE_FLOOR)
            base = (scale * polish.DAMPING_RELATIVE_FLOOR
                    if direct_allowed else initial)
            exponent = int(round(math.log10(float(damping) / base)))
            damping_trial_count = exponent + (2 if direct_allowed else 1)
    return dict(
        loss=float(loss_t),
        gradient=gradient.tolist(),
        gradient_norm=float(np.linalg.norm(gradient)),
        scaled_gradient_norm=float(np.linalg.norm(scaled_gradient)),
        hessian=hessian.tolist(),
        hessian_eigenvalues=raw_eigenvalues.tolist(),
        scaled_hessian_eigenvalues=scaled_eigenvalues.tolist(),
        hessian_condition=_finite(raw_condition),
        scaled_hessian_condition=_finite(scaled_condition),
        negative_curvature_count=int(np.count_nonzero(raw_eigenvalues < 0.0)),
        minimum_eigenvalue=float(raw_eigenvalues[0]),
        maximum_eigenvalue=float(raw_eigenvalues[-1]),
        direction_available=bool(direction is not None),
        route=("DIRECT" if direct else ("DAMPED" if direction is not None else "NONE")),
        direction_condition=_finite(direction_condition),
        initial_damping=_finite(damping),
        damping_trial_count=damping_trial_count,
        direction_status=direction_status,
        newton_direction=(None if direction is None else direction.tolist()),
        newton_step_norm=_finite(step_norm),
        normalized_step_l2=(None if direction is None else float(np.linalg.norm(normalized_step))),
        normalized_step_linf=(None if direction is None else float(np.max(np.abs(normalized_step)))),
        gradient_dot_direction=_finite(dot),
        descent_cosine=_finite(descent_cosine),
        predicted_decrease=_finite(predicted_decrease),
        normalized_position=normalized_position.tolist(),
        normalized_bound_margins=bound_margins.tolist(),
        minimum_bound_margin=float(np.min(bound_margins)),
        mean_bound_margin=float(np.mean(bound_margins)),
        first_newton_step_in_bounds=inside,
    )


def _run_n3(theta: np.ndarray, local: dict, config: dict) -> dict:
    endpoint = polish.N2Endpoint(
        0, theta.copy(), float(local["loss"]), "direct_sobol_basin_map",
        config["name"], False,
    )
    result = config["run_n3"](endpoint, torch.empty(0))
    final_theta = result.theta_history[-1]
    final_loss = float(result.loss_history[-1])
    return dict(
        passed=bool(final_loss < FINAL_GATE),
        final_theta=final_theta.tolist(),
        final_loss=final_loss,
        stop_reason=result.stop_reason,
        accepted_steps=int(result.accepted_steps),
        rejected_newton_steps=int(result.rejected_newton_steps),
        objective_evaluations=int(result.actual_loss_evaluations),
        backtracking_rejections=int(result.backtracking_rejections),
        accepted_with_backtracking=int(result.accepted_with_backtracking),
        hessian_evaluations=int(result.hessian_evaluations),
        damping_steps=int(result.damping_steps),
        condition_numbers=[_finite(value) for value in result.condition_numbers],
        damping_values=[_finite(value) for value in result.damping_values],
        gradient_norms=[float(value) for value in result.gradient_norms],
        theta_history=[value.tolist() for value in result.theta_history],
        loss_history=[float(value) for value in result.loss_history],
    )


def _load_rows() -> list[dict]:
    if not ROWS.exists():
        return []
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]


def _append(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ROWS.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _sample_system(config: dict, existing: list[dict]) -> list[dict]:
    name = config["name"]
    rows = [row for row in existing if row["system"] == name]
    done = {int(row["sobol_index"]) for row in rows}
    unit = qmc.Sobol(len(config["lower"]), scramble=True, seed=config["seed"]).random_base2(MAX_POWER)
    width = config["upper"] - config["lower"]
    targets = [2 ** power for power in range(INITIAL_POWER, MAX_POWER + 1)]
    for target in targets:
        for index in range(target):
            if index in done:
                continue
            started = time.perf_counter()
            theta = config["lower"] + width * unit[index]
            local = _local_diagnostics(theta, config)
            n3 = _run_n3(theta, local, config)
            row = dict(
                system=name,
                objective_case=config["label"],
                sobol_seed=int(config["seed"]),
                sobol_index=index,
                unit_point=unit[index].tolist(),
                theta=theta.tolist(),
                lower=config["lower"].tolist(),
                upper=config["upper"].tolist(),
                offline_teacher=config["teacher"].tolist(),
                offline_teacher_l2_distance=float(np.linalg.norm(theta - config["teacher"])),
                offline_final_teacher_l2_distance=float(
                    np.linalg.norm(np.asarray(n3["final_theta"]) - config["teacher"])
                ),
                local=local,
                n3=n3,
                seconds=float(time.perf_counter() - started),
            )
            _append(row)
            rows.append(row)
            done.add(index)
            print(json.dumps(dict(system=name, index=index, passed=n3["passed"],
                                  loss=local["loss"], final_loss=n3["final_loss"],
                                  steps=n3["accepted_steps"], seconds=row["seconds"])),
                  flush=True)
        pass_count = sum(bool(row["n3"]["passed"]) for row in rows)
        fail_count = len(rows) - pass_count
        if name == "sho":
            # The full-K objective is a constant-Hessian quadratic; the first
            # stage is sufficient when direct replay confirms all sampled starts.
            if fail_count == 0:
                break
        elif min(pass_count, fail_count) >= MINORITY_TARGET:
            break
    return rows


def _quantiles(values: list[float]) -> dict:
    array = np.asarray([value for value in values if value is not None and math.isfinite(value)],
                       dtype=np.float64)
    if not len(array):
        return dict(count=0)
    return dict(count=len(array), minimum=float(np.min(array)), q10=float(np.quantile(array, .1)),
                median=float(np.median(array)), q90=float(np.quantile(array, .9)),
                maximum=float(np.max(array)))


def _scalar(row: dict, key: str) -> float | None:
    if key.startswith("n3."):
        value = row["n3"].get(key[3:])
    else:
        value = row["local"].get(key)
    if value is None or isinstance(value, (list, dict, bool, str)):
        return None
    return float(value)


def _nearest_opposite(rows: list[dict]) -> dict | None:
    best = None
    for left_index, left in enumerate(rows):
        left_point = np.asarray(left["unit_point"], dtype=np.float64)
        for right in rows[left_index + 1:]:
            if left["n3"]["passed"] == right["n3"]["passed"]:
                continue
            distance = float(np.linalg.norm(left_point - np.asarray(right["unit_point"])))
            if best is None or distance < best["normalized_parameter_distance"]:
                passing, failing = (left, right) if left["n3"]["passed"] else (right, left)
                best = dict(
                    normalized_parameter_distance=distance,
                    pass_sobol_index=passing["sobol_index"],
                    fail_sobol_index=failing["sobol_index"],
                    pass_theta=passing["theta"],
                    fail_theta=failing["theta"],
                    pass_loss=passing["local"]["loss"],
                    fail_loss=failing["local"]["loss"],
                    pass_route=passing["local"]["route"],
                    fail_route=failing["local"]["route"],
                    pass_step_inside=passing["local"]["first_newton_step_in_bounds"],
                    fail_step_inside=failing["local"]["first_newton_step_in_bounds"],
                    pass_stop=passing["n3"]["stop_reason"],
                    fail_stop=failing["n3"]["stop_reason"],
                )
    return best


def _summarize(all_rows: list[dict], configs: dict[str, dict]) -> dict:
    scalar_keys = (
        "loss", "gradient_norm", "scaled_gradient_norm", "minimum_eigenvalue",
        "maximum_eigenvalue", "hessian_condition", "scaled_hessian_condition",
        "initial_damping", "damping_trial_count", "newton_step_norm", "normalized_step_l2",
        "normalized_step_linf", "descent_cosine", "predicted_decrease",
        "minimum_bound_margin", "mean_bound_margin", "n3.accepted_steps",
        "n3.backtracking_rejections", "n3.damping_steps",
    )
    report = dict(final_gate=FINAL_GATE, systems={})
    for name, config in configs.items():
        rows = [row for row in all_rows if row["system"] == name]
        passed = [row for row in rows if row["n3"]["passed"]]
        failed = [row for row in rows if not row["n3"]["passed"]]
        distributions = {}
        for key in scalar_keys:
            distributions[key] = dict(
                PASS=_quantiles([_scalar(row, key) for row in passed]),
                FAIL=_quantiles([_scalar(row, key) for row in failed]),
            )
        categorical = {}
        for key in ("direction_available", "first_newton_step_in_bounds"):
            categorical[key] = dict(
                PASS_true=sum(bool(row["local"][key]) for row in passed),
                PASS_total=len(passed),
                FAIL_true=sum(bool(row["local"][key]) for row in failed),
                FAIL_total=len(failed),
            )
        categorical["route"] = {
            label: dict(PASS=sum(row["local"]["route"] == label for row in passed),
                        FAIL=sum(row["local"]["route"] == label for row in failed))
            for label in ("DIRECT", "DAMPED", "NONE")
        }
        categorical["direct_and_inside"] = dict(
            PASS_true=sum(row["local"]["route"] == "DIRECT" and
                          row["local"]["first_newton_step_in_bounds"] for row in passed),
            PASS_total=len(passed),
            FAIL_true=sum(row["local"]["route"] == "DIRECT" and
                          row["local"]["first_newton_step_in_bounds"] for row in failed),
            FAIL_total=len(failed),
        )
        stop_reasons = {}
        for row in rows:
            key = ("PASS:" if row["n3"]["passed"] else "FAIL:") + row["n3"]["stop_reason"]
            stop_reasons[key] = stop_reasons.get(key, 0) + 1
        sho_quadratic = None
        if name == "sho" and rows:
            hessians = np.asarray([row["local"]["hessian"] for row in rows])
            directions = []
            ratios = []
            for row in rows:
                direction = np.asarray(row["local"]["newton_direction"], dtype=np.float64)
                displacement = config["teacher"] - np.asarray(row["theta"], dtype=np.float64)
                directions.append(float(np.linalg.norm(direction - displacement)))
                denominator = float(np.linalg.norm(displacement))
                ratios.append(float(np.linalg.norm(direction) / denominator) if denominator else 1.0)
            sho_quadratic = dict(
                maximum_hessian_difference_from_first=float(np.max(np.abs(hessians - hessians[0]))),
                minimum_hessian_eigenvalue=float(np.min(np.linalg.eigvalsh(hessians[0]))),
                maximum_direction_minus_teacher_displacement=float(max(directions)),
                step_to_teacher_norm_ratio=_quantiles(ratios),
            )
        report["systems"][name] = dict(
            objective_case=config["label"],
            dimension=len(config["lower"]),
            lower=config["lower"].tolist(),
            upper=config["upper"].tolist(),
            samples=len(rows), PASS=len(passed), FAIL=len(failed),
            route_counts=categorical["route"], categorical=categorical,
            stop_reasons=stop_reasons,
            distributions=distributions,
            nearest_opposite_label_pair=_nearest_opposite(rows),
            sho_constant_quadratic_check=sho_quadratic,
            seconds=float(sum(row["seconds"] for row in rows)),
        )
    return report


def _write_csv(rows: list[dict]) -> None:
    fields = (
        "system", "objective_case", "sobol_index", "passed", "theta", "loss",
        "gradient_norm", "scaled_gradient_norm", "minimum_eigenvalue",
        "maximum_eigenvalue", "hessian_condition", "scaled_hessian_condition",
        "route", "initial_damping", "damping_trial_count", "newton_step_norm", "normalized_step_l2",
        "descent_cosine", "minimum_bound_margin", "first_newton_step_in_bounds",
        "accepted_steps", "backtracking_rejections", "damping_steps", "final_loss",
        "stop_reason", "offline_teacher_l2_distance",
    )
    with (OUT / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            local, n3 = row["local"], row["n3"]
            writer.writerow(dict(
                system=row["system"], objective_case=row["objective_case"],
                sobol_index=row["sobol_index"], passed=n3["passed"],
                theta=json.dumps(row["theta"]), loss=local["loss"],
                gradient_norm=local["gradient_norm"],
                scaled_gradient_norm=local["scaled_gradient_norm"],
                minimum_eigenvalue=local["minimum_eigenvalue"],
                maximum_eigenvalue=local["maximum_eigenvalue"],
                hessian_condition=local["hessian_condition"],
                scaled_hessian_condition=local["scaled_hessian_condition"],
                route=local["route"], initial_damping=local["initial_damping"],
                damping_trial_count=local["damping_trial_count"],
                newton_step_norm=local["newton_step_norm"],
                normalized_step_l2=local["normalized_step_l2"],
                descent_cosine=local["descent_cosine"],
                minimum_bound_margin=local["minimum_bound_margin"],
                first_newton_step_in_bounds=local["first_newton_step_in_bounds"],
                accepted_steps=n3["accepted_steps"],
                backtracking_rejections=n3["backtracking_rejections"],
                damping_steps=n3["damping_steps"], final_loss=n3["final_loss"],
                stop_reason=n3["stop_reason"],
                offline_teacher_l2_distance=row["offline_teacher_l2_distance"],
            ))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {name: _system_config(name) for name in ("sho", "bouncing_ball", "kepler")}
    existing = _load_rows()
    for config in configs.values():
        _sample_system(config, existing)
        existing = _load_rows()
    report = _summarize(existing, configs)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    _write_csv(existing)
    print(json.dumps({name: {key: value for key, value in values.items()
                            if key in ("samples", "PASS", "FAIL", "seconds")}
                      for name, values in report["systems"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
