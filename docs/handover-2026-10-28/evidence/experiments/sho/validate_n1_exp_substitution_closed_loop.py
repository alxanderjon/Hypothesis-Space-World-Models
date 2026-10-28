"""Closed-loop replay of only the case changed by the recovered N1 exponential."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit
from scipy.spatial import cKDTree

from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as base
from experiments.sho import run_stage1b_edirect_vs_direct as teacher_module
from experiments.sho.deterministic_full_k_framework import deterministic_finish, hashes, prior
from experiments.sho.generate_checkpointed_edirect_teacher_data import (
    CAP, DATA, LOSS_GATE, _case_objectives, _checkpoint,
)
from experiments.sho.recover_n1_scalar_exp import (
    COEFFICIENTS, INV_LN2, LN2_HIGH, LN2_LOW, SPLITTER, assembled_exp,
)
from experiments.sho.train_newton_correctability_router import _case_math, _run_label


OUT = Path("outputs/sho/stage1B/n1_scalar_exp_downstream/closed_loop_single_face_32")
CASE_INDEX = 128
WEIGHT_FLOOR = np.float64(1.0e-15)
COEFFICIENT_HIGH = np.asarray([value[0] for value in COEFFICIENTS], np.float64)
COEFFICIENT_LOW = np.asarray([value[1] for value in COEFFICIENTS], np.float64)


@njit(cache=True)
def _two_prod(a, b):
    product = a * b
    ca = SPLITTER * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = SPLITTER * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    tail = (((a_hi * b_hi - product) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo
    return product, tail


@njit(cache=True)
def _two_sum(a, b):
    total = a + b
    b_virtual = total - a
    return total, (a - (total - b_virtual)) + (b - b_virtual)


@njit(cache=True)
def _quick_two_sum(a, b):
    total = a + b
    return total, b - (total - a)


@njit(cache=True)
def _dd_add(a_hi, a_lo, b_hi, b_lo):
    high, tail = _two_sum(a_hi, b_hi)
    tail = tail + (a_lo + b_lo)
    return _quick_two_sum(high, tail)


@njit(cache=True)
def _dd_multiply(a_hi, a_lo, b_hi, b_lo):
    high, tail = _two_prod(a_hi, b_hi)
    tail = tail + a_hi * b_lo + a_lo * b_hi + a_lo * b_lo
    return _quick_two_sum(high, tail)


@njit(cache=True)
def _assembled_exp_scalar(x):
    k = int(np.rint(x * INV_LN2))
    k_float = float(k)
    product_hi, product_lo = _two_prod(k_float, LN2_HIGH)
    product_lo = product_lo + k_float * LN2_LOW
    product_hi, product_lo = _quick_two_sum(product_hi, product_lo)
    reduced_hi, reduced_lo = _two_sum(x, -product_hi)
    reduced_lo = reduced_lo - product_lo
    reduced_hi, reduced_lo = _quick_two_sum(reduced_hi, reduced_lo)
    high, low = COEFFICIENT_HIGH[24], COEFFICIENT_LOW[24]
    for index in range(23, -1, -1):
        high, low = _dd_multiply(high, low, reduced_hi, reduced_lo)
        high, low = _dd_add(high, low, COEFFICIENT_HIGH[index], COEFFICIENT_LOW[index])
    high, low = math.ldexp(high, k), math.ldexp(low, k)
    rounded, _ = _two_sum(high, low)
    return rounded


@njit(cache=True)
def assembled_exp_compiled(values):
    output = np.empty(values.size, np.float64)
    for index in range(values.size):
        output[index] = _assembled_exp_scalar(values[index])
    return output


class CorrectlyRoundedGaussianLocalResponse:
    def __init__(self, points: np.ndarray, losses: np.ndarray) -> None:
        self.points = points
        self.values = np.log10(np.maximum(losses, teacher_module.LOSS_FLOOR))
        self.tree = cKDTree(points)

    @staticmethod
    def _weights(distances: np.ndarray) -> np.ndarray:
        distances = np.asarray(distances, np.float64)
        bandwidth = np.maximum(distances[..., -1:], np.float64(1.0e-12))
        z = np.multiply(np.float64(-9.0), np.square(np.divide(distances, bandwidth)))
        return np.add(assembled_exp_compiled(z.ravel()).reshape(z.shape), WEIGHT_FLOOR)

    def predict(self, query: np.ndarray) -> np.ndarray:
        query = np.atleast_2d(query)
        count = min(teacher_module.RBF_NEIGHBORS, len(self.points))
        distances, indices = self.tree.query(query, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        weights = self._weights(distances)
        return np.sum(weights * self.values[indices], axis=1) / np.sum(weights, axis=1)

    def loo_errors(self, site_indices: np.ndarray) -> np.ndarray:
        if len(self.points) <= 2:
            return np.ones(len(site_indices), np.float64)
        count = min(teacher_module.RBF_NEIGHBORS + 1, len(self.points))
        distances, indices = self.tree.query(self.points[site_indices], k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        result = np.empty(len(site_indices), np.float64)
        for row, site_index in enumerate(site_indices):
            keep = indices[row] != site_index
            local_indices = indices[row, keep][:teacher_module.RBF_NEIGHBORS]
            local_distances = distances[row, keep][:teacher_module.RBF_NEIGHBORS]
            if not len(local_indices):
                result[row] = 0.0
                continue
            weights = self._weights(local_distances[None, :])[0]
            prediction = float(np.sum(weights * self.values[local_indices]) / np.sum(weights))
            result[row] = abs(self.values[site_index] - prediction)
        return result


def run_search(objective, teacher: np.ndarray):
    traces = []
    # Process-local substitution only. No repository source or checkpoint is edited.
    original = teacher_module.GaussianLocalResponse
    teacher_module.GaussianLocalResponse = CorrectlyRoundedGaussianLocalResponse
    try:
        api = base.composition(4)
        api["_greedy_proposal"] = (
            lambda points, losses, current_case, iteration:
            continuous.proposal(points, losses, current_case, iteration, traces)
        )
        points, losses, events, checkpoints = [], [], [], []
        iteration = 0

        def evaluate(unit_point, source):
            value = float(objective(prior.LOWER + (prior.UPPER - prior.LOWER) * unit_point))
            if not np.isfinite(value):
                raise ValueError("nonfinite objective")
            points.append(np.asarray(unit_point).copy())
            losses.append(value)
            events.append(dict(
                evaluation=len(losses), iteration=iteration, source=source,
                point=np.asarray(unit_point).tolist(), loss=value,
                best_loss=float(min(losses)),
                boundary=bool(np.any((unit_point == 0.0) | (unit_point == 1.0))),
            ))

        for unit_point in base.initial_design(4):
            evaluate(unit_point, "initial")

        stop = False
        while len(losses) < CAP and not stop:
            proposals = api["_cell_proposals"](
                np.asarray(points), np.asarray(losses), CASE_INDEX, iteration
            )
            for unit_point in proposals:
                if len(losses) >= CAP:
                    break
                evaluate(unit_point, "cell")
                if len(losses) % 100 == 0:
                    record = _checkpoint(
                        np.asarray(points), np.asarray(losses), teacher,
                        prior.LOWER, prior.UPPER, CASE_INDEX, iteration, traces,
                    )
                    checkpoints.append(record)
                    print(json.dumps(dict(evaluations=len(losses), iteration=iteration,
                                          best_loss=float(min(losses)), both=record["both_ok"])), flush=True)
                    if record["both_ok"]:
                        stop = True
                        break
            if stop or len(losses) >= CAP:
                break
            unit_point = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), CASE_INDEX, iteration
            )
            if unit_point is not None:
                evaluate(unit_point, "greedy")
            iteration += 1
            if len(losses) % 100 == 0:
                record = _checkpoint(
                    np.asarray(points), np.asarray(losses), teacher,
                    prior.LOWER, prior.UPPER, CASE_INDEX, iteration, traces,
                )
                checkpoints.append(record)
                print(json.dumps(dict(evaluations=len(losses), iteration=iteration,
                                      best_loss=float(min(losses)), both=record["both_ok"])), flush=True)
                if record["both_ok"]:
                    stop = True
        return dict(points=np.asarray(points), losses=np.asarray(losses), events=events,
                    traces=traces, checkpoints=checkpoints, iterations=iteration)
    finally:
        teacher_module.GaussianLocalResponse = original


def earliest_correctable(events: list[dict], states: np.ndarray, category: str) -> dict | None:
    _, _, _, run_n3 = _case_math(states)
    best_loss = math.inf
    best_theta = None
    for event in events:
        loss = float(event["loss"])
        if loss >= best_loss:
            continue
        best_loss = loss
        unit = np.asarray(event["point"], np.float64)
        best_theta = prior.LOWER + (prior.UPPER - prior.LOWER) * unit
        n3, final_theta, final_loss, correctable = _run_label(
            best_theta, best_loss, category, run_n3
        )
        if correctable:
            return dict(
                evaluation=int(event["evaluation"]), source=event["source"],
                theta=best_theta.tolist(), loss=best_loss,
                final_theta=final_theta.tolist(), final_loss=final_loss,
                accepted_n3_steps=int(n3.accepted_steps),
            )
    return None


def finish(search: dict, teacher: np.ndarray, states: np.ndarray, objective, torch_objective):
    checkpoint = next((row for row in search["checkpoints"] if row["both_ok"]),
                      search["checkpoints"][-1])
    start = dict(theta=np.asarray(checkpoint["best_theta"]), loss=checkpoint["best_loss"],
                 lower=np.asarray(checkpoint["box_lower"]), upper=np.asarray(checkpoint["box_upper"]))
    result = deterministic_finish(objective, torch_objective, start)
    final = np.asarray(result["final_theta"])
    delta = final - teacher
    true_state = states[-1].copy()
    predicted_state = true_state.copy()
    errors = []
    for _ in range(4096):
        true_state = teacher.reshape(2, 2) @ true_state
        predicted_state = final.reshape(2, 2) @ predicted_state
        errors.append(predicted_state - true_state)
    errors = np.asarray(errors)
    best = int(np.argmin(search["losses"]))
    n1_theta = prior.LOWER + (prior.UPPER - prior.LOWER) * search["points"][best]
    return dict(
        handoff_checkpoint=int(checkpoint["evaluation"]),
        n1_evaluations=len(search["losses"]), n1_iterations=search["iterations"],
        n1_theta=n1_theta.tolist(), n1_loss=float(search["losses"][best]),
        n1_teacher_distance=float(np.linalg.norm(n1_theta - teacher)),
        continuous_calls=len(search["traces"]),
        continuous_surrogate_evaluations=int(sum(
            start["nfev"] for trace in search["traces"] for start in trace["starts"]
        )),
        **result,
        recovered=bool(result["final_loss"] < 1.0e-20),
        matrix_error=float(np.linalg.norm(delta)), matrix_max_error=float(np.max(np.abs(delta))),
        rollout_max_error=float(np.max(np.abs(errors))), rollout_mse=float(np.mean(errors ** 2)),
    )


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    before = hashes()
    with np.load(DATA) as data:
        name = str(data["names"][CASE_INDEX])
        category = str(data["categories"][CASE_INDEX])
        teacher = data["teachers"][CASE_INDEX].ravel()
        states = data[f"prefix_{CASE_INDEX}"].copy()
    assert name == "single_face_32"
    objective, torch_objective = _case_objectives(states)
    # Execution-only compiled graph must remain bit-identical to the validated
    # vector assembly before it is allowed into the closed-loop diagnostic.
    control = np.linspace(-9.0, np.nextafter(0.0, -1.0), 1_000_001, dtype=np.float64)
    compiled = assembled_exp_compiled(control)
    reference = assembled_exp(control)
    if not np.array_equal(compiled.view(np.uint64), reference.view(np.uint64)):
        raise RuntimeError("compiled exponential graph differs from validated assembly")
    production_dir = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases") / name
    production = json.loads((production_dir / "result.json").read_text())
    production_events = json.loads((production_dir / "events.json").read_text())
    production_correctable = earliest_correctable(production_events, states, category)

    candidate_search = run_search(objective, teacher)
    candidate_correctable = earliest_correctable(candidate_search["events"], states, category)
    candidate_finish = finish(
        candidate_search, teacher, states, objective, torch_objective
    )
    after = hashes()
    production_summary = dict(
        n1_evaluations=production["total_n1_evaluations"],
        n1_theta=production["checkpoints"][-1]["best_theta"],
        n1_loss=production["checkpoints"][-1]["best_loss"],
        n1_teacher_distance=float(np.linalg.norm(
            np.asarray(production["checkpoints"][-1]["best_theta"]) - teacher
        )),
        first_newton_correctable=production_correctable,
        n2_theta=production["n2_theta"], n2_loss=production["n2_loss"],
        n2_evaluations=production["n2_evaluations"], n2_gate=production["n2_gate"],
        final_theta=production["final_theta"], final_loss=production["final_loss"],
        n3_accepted_steps=production["n3_accepted_steps"],
        recovered=bool(production["final_loss"] < 1.0e-20),
        matrix_error=production["matrix_error"], matrix_max_error=production["matrix_max_error"],
        rollout_max_error=production["rollout_max_error"], rollout_mse=production["rollout_mse"],
    )
    result = dict(
        component="n1_exp_substitution_closed_loop",
        case=name, case_id=CASE_INDEX, category=category,
        substitution="only np.exp in GaussianLocalResponse -> zero-parameter correctly-rounded assembly",
        production=production_summary,
        recovered_exp=dict(**candidate_finish, first_newton_correctable=candidate_correctable),
        comparison=dict(
            evaluations_to_first_newton_correctable_difference=(
                None if production_correctable is None or candidate_correctable is None else
                candidate_correctable["evaluation"] - production_correctable["evaluation"]
            ),
            n1_evaluation_difference=candidate_finish["n1_evaluations"] - production["total_n1_evaluations"],
            same_final_recovery=bool((production["final_loss"] < 1.0e-20) == candidate_finish["recovered"]),
            same_basin=bool(candidate_finish["recovered"] and production["final_loss"] < 1.0e-20),
        ),
        checkpoints_unchanged=before == after,
        checkpoint_count=len(before), elapsed_seconds=time.monotonic() - started,
    )
    (OUT / "events.json").write_text(json.dumps(candidate_search["events"]) + "\n")
    (OUT / "continuous.json").write_text(json.dumps(candidate_search["traces"]) + "\n")
    np.savez_compressed(OUT / "n1_state.npz", points=candidate_search["points"], losses=candidate_search["losses"])
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if before != after:
        raise SystemExit("checkpoint changed")


if __name__ == "__main__":
    run()
