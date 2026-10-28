"""Stage 1B noisy operator-identification priors diagnostic.

Tests whether generic priors on the prefix-fitted local operator K_h can lower
the noisy finite-prefix recovery floor without using velocity, labels, analytic
SHO structure, raw-window teacher losses, or an MLP-generated K.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_noisy_short_prefix_learned_delay_diagnostic import (
    AnchoredSoftDelay,
    FixedDelay,
    _noisy_positions,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _apply_k,
    _fit_k_ridge,
    _rollout_matrix,
    _spectral_radius,
    _window_sequence,
    _write_csv,
    _write_json,
)
from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import _rollout_metrics


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/noisy_operator_identification_priors_diagnostic")
RECOVERY_LIMIT_DIR = Path("outputs/sho/stage1B/recovery_limit_and_curriculum_diagnostic")

M = 8
VAL_NUM = 6
TEST_NUM = 6
ROLLOUT_PERIODS = 10.0
R_VALUES = [8, 16, 32]
SIGMAS = [0.0, 1e-3, 1e-2]
PRIOR_SIGMAS = [1e-3, 1e-2]
COORDINATES = ["fixed_delay", "uniform_mixed_delay", "recency_weighted_mixed_delay"]
PRIMARY_COORDINATES = ["uniform_mixed_delay", "recency_weighted_mixed_delay"]
RIDGES = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

POWER_LAMBDAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
SPEC_LAMBDAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
SPEC_DELTAS = [1e-4, 1e-3, 1e-2]
MULTI_WEIGHT_KINDS = ["inverse_j", "uniform"]
DENOISE_LAMBDA_DATA = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
DENOISE_LAMBDA_MULTI = [0.0, 1e-2, 1e-1, 1.0]

# The fitting objectives are tiny 2x2 per-trajectory problems. These compact
# optimizer budgets keep the diagnostic short while preserving the comparison.
OPT_STEPS = 120
PATIENCE = 35
ADAM_LRS = [1e-3, 1e-4]
LBFGS_STEPS = 12


def _full_rollout_steps(periods: float = ROLLOUT_PERIODS) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_long_data(num_trajectories: int, seed: int) -> torch.Tensor:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed, dtype=DTYPE)
    total_steps = max(R_VALUES) + M + _full_rollout_steps() + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _model_for_name(name: str) -> Any:
    if name == "fixed_delay":
        return FixedDelay()
    if name == "uniform_mixed_delay":
        return AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE)
    if name == "recency_weighted_mixed_delay":
        return AnchoredSoftDelay("recency_weighted", train_alpha=False).to(dtype=DTYPE)
    raise ValueError(name)


def _observed_windows(positions: torch.Tensor, r_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = _window_sequence(positions, M, M - 1, r_value + 1)
    return q[:, :-1], q[:, 1:]


def _future_windows(positions: torch.Tensor, r_value: int) -> torch.Tensor:
    return _window_sequence(positions, M, M - 1 + r_value, _full_rollout_steps() + 1)


def _solve_k(h_in: torch.Tensor, h_out: torch.Tensor, solver: str, ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if solver == "ridge":
        return _fit_k_ridge(h_in, h_out, ridge=ridge)
    k_list, residuals, conds = [], [], []
    for x, y in zip(h_in, h_out):
        if solver == "qr":
            sol = torch.linalg.lstsq(x, y).solution
        elif solver == "svd":
            sol = torch.linalg.pinv(x) @ y
        else:
            raise ValueError(solver)
        k = sol.T
        k_list.append(k)
        residuals.append(torch.mean((x @ sol - y) ** 2))
        conds.append(torch.linalg.cond(x))
    return torch.stack(k_list), torch.stack(residuals).mean(), torch.stack(conds)


def _fit_baseline(model: Any, observed: torch.Tensor, r_value: int, solver: str, ridge: float) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    q_in, q_out = _observed_windows(observed, r_value)
    h_in, h_out = model.encode(q_in), model.encode(q_out)
    k, residual, conds = _solve_k(h_in, h_out, solver, ridge)
    return k, {"observed_prefix_transition_residual": float(residual.item()), "ls_covariance_condition_number": float(conds.mean().item())}, h_in, h_out, model.encode(q_out[:, -1:])


def _multi_step_residual(k: torch.Tensor, h_seq: torch.Tensor, steps: list[int]) -> float:
    losses = []
    for j in steps:
        if h_seq.shape[1] <= j:
            continue
        pred = _batch_power(k, h_seq[:, :-j], j)
        losses.append(torch.mean((pred - h_seq[:, j:]) ** 2))
    if not losses:
        return float("nan")
    return float(torch.stack(losses).mean().item())


def _evaluate_k(
    model: Any,
    observed: torch.Tensor,
    clean: torch.Tensor,
    r_value: int,
    k: torch.Tensor,
    initial_h: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q_in, q_out = _observed_windows(observed, r_value)
    h_in, h_out = model.encode(q_in), model.encode(q_out)
    if initial_h is None:
        initial_h = model.encode(_future_windows(observed, r_value)[:, :1])[:, 0]
    truth = _future_windows(clean, r_value)
    h_rollout = _rollout_matrix(k, initial_h, truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    metrics = _rollout_metrics(pred_x, truth)
    eval_row = {
        **metrics,
        "future_10_period_mse": metrics["rollout_mse"],
        "future_rollout_horizon_periods": ROLLOUT_PERIODS,
        "observed_prefix_transition_residual": float(torch.mean((_apply_k_time(k, h_in) - h_out) ** 2).item()),
        "multi_step_observed_prefix_residual": _multi_step_residual(k, torch.cat([h_in[:, :1], h_out], dim=1), [2, 4, 8]),
        "ls_covariance_condition_number": float(torch.linalg.cond(h_in).mean().item()),
        "spectral_radius_k_h": _spectral_radius(k),
        "eigenvalue_magnitudes": json.dumps(torch.abs(torch.linalg.eigvals(k)).detach().cpu().numpy().tolist()),
        **_threshold_flags(metrics["rollout_mse"]),
    }
    if extra:
        eval_row.update(extra)
    return eval_row


def _batch_power(k: torch.Tensor, h: torch.Tensor, steps: int) -> torch.Tensor:
    out = h
    for _ in range(steps):
        out = _apply_k_time(k, out)
    return out


def _apply_k_time(k: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    if h.ndim == 2:
        return _apply_k(k, h)
    return torch.einsum("bij,btj->bti", k, h)


def _optimize_k(
    k0: torch.Tensor,
    h_seq: torch.Tensor,
    objective: str,
    hyper: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    k = k0.detach().clone().requires_grad_(True)
    optimizer_name = hyper.get("optimizer", "Adam")
    lr = float(hyper.get("lr", 1e-3))
    if optimizer_name == "LBFGS":
        opt = torch.optim.LBFGS([k], lr=lr, max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")
        best_loss = float("inf")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = _k_objective(k, h_seq, objective, hyper)
            loss.backward()
            return loss

        loss = opt.step(closure)
        best_loss = float(loss.item())
        return k.detach(), {"optimizer_final_loss": best_loss, "optimizer_steps": LBFGS_STEPS}
    opt = torch.optim.Adam([k], lr=lr)
    best_state = k.detach().clone()
    best_loss = float("inf")
    stale = 0
    for step in range(1, OPT_STEPS + 1):
        opt.zero_grad()
        loss = _k_objective(k, h_seq, objective, hyper)
        loss.backward()
        opt.step()
        value = float(loss.detach().item())
        if value + 1e-15 < best_loss:
            best_loss = value
            best_state = k.detach().clone()
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_state, {"optimizer_final_loss": best_loss, "optimizer_steps": step}


def _k_objective(k: torch.Tensor, h_seq: torch.Tensor, objective: str, hyper: dict[str, Any]) -> torch.Tensor:
    h_in, h_out = h_seq[:, :-1], h_seq[:, 1:]
    one = torch.mean((_apply_k_time(k, h_in) - h_out) ** 2)
    if objective == "stable_power":
        denom = torch.mean(h_seq[:, :-1] ** 2).detach() + 1e-12
        penalties = []
        for j in [1, 2, 4, 8, 16]:
            penalties.append(torch.mean(_batch_power(k, h_in, j) ** 2) / denom)
        return one + float(hyper["lambda_power"]) * torch.stack(penalties).mean()
    if objective.startswith("spectral"):
        rho = torch.max(torch.abs(torch.linalg.eigvals(k)), dim=1).values
        if objective == "spectral_stable":
            penalty = torch.mean(torch.relu(rho - (1.0 + float(hyper["delta"]))) ** 2)
        else:
            penalty = torch.mean((rho - 1.0) ** 2)
        return one + float(hyper["lambda_spec"]) * penalty
    if objective == "multi_step":
        losses = []
        for j in [1, 2, 4, 8]:
            if h_seq.shape[1] <= j:
                continue
            pred = _batch_power(k, h_seq[:, :-j], j)
            weight = 1.0 / j if hyper["weight_kind"] == "inverse_j" else 1.0
            losses.append(weight * torch.mean((pred - h_seq[:, j:]) ** 2))
        return torch.stack(losses).sum()
    raise ValueError(objective)


def _optimize_denoised(
    k0: torch.Tensor,
    h_seq: torch.Tensor,
    hyper: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    k = k0.detach().clone().requires_grad_(True)
    z = h_seq.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([k, z], lr=float(hyper.get("lr", 1e-3)))
    best_loss = float("inf")
    best_k, best_z = k.detach().clone(), z.detach().clone()
    stale = 0
    for step in range(1, OPT_STEPS + 1):
        opt.zero_grad()
        data = torch.mean((z - h_seq) ** 2)
        dyn = torch.mean((_apply_k_time(k, z[:, :-1]) - z[:, 1:]) ** 2)
        multi_terms = []
        for j in [2, 4, 8]:
            if z.shape[1] <= j:
                continue
            multi_terms.append((1.0 / j) * torch.mean((_batch_power(k, z[:, :-j], j) - z[:, j:]) ** 2))
        multi = torch.stack(multi_terms).sum() if multi_terms else torch.zeros((), dtype=DTYPE)
        loss = float(hyper["lambda_data"]) * data + dyn + float(hyper["lambda_multi"]) * multi
        loss.backward()
        opt.step()
        value = float(loss.item())
        if value + 1e-15 < best_loss:
            best_loss = value
            best_k, best_z = k.detach().clone(), z.detach().clone()
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    stats = {
        "optimizer_final_loss": best_loss,
        "optimizer_steps": step,
        "denoised_data_fidelity": float(torch.mean((best_z - h_seq) ** 2).item()),
        "denoised_dynamics_consistency": float(torch.mean((_apply_k_time(best_k, best_z[:, :-1]) - best_z[:, 1:]) ** 2).item()),
        "denoised_latent_norm": float(torch.mean(best_z**2).item()),
        "denoised_initial_state_delta": float(torch.mean((best_z[:, -1] - h_seq[:, -1]) ** 2).item()),
    }
    return best_k, best_z[:, -1], stats


def _candidate_hypers(method: str) -> list[dict[str, Any]]:
    if method == "baseline_ls":
        return [{"solver": s, "ridge": 0.0} for s in ["qr", "svd"]] + [{"solver": "ridge", "ridge": r} for r in RIDGES]
    if method == "stable_power":
        return [{"lambda_power": l, "optimizer": "Adam", "lr": 1e-3} for l in POWER_LAMBDAS]
    if method == "spectral_stable":
        return [{"lambda_spec": l, "delta": d, "optimizer": "Adam", "lr": 1e-3} for l in SPEC_LAMBDAS for d in SPEC_DELTAS]
    if method == "spectral_unit":
        return [{"lambda_spec": l, "optimizer": "Adam", "lr": 1e-3} for l in SPEC_LAMBDAS]
    if method == "multi_step":
        return [{"weight_kind": w, "optimizer": opt, "lr": lr} for w in MULTI_WEIGHT_KINDS for opt in ["Adam", "LBFGS"] for lr in ADAM_LRS]
    if method == "denoised_latent":
        return [{"lambda_data": ld, "lambda_multi": lm, "lr": 1e-3} for ld in DENOISE_LAMBDA_DATA for lm in DENOISE_LAMBDA_MULTI]
    raise ValueError(method)


def _run_candidate(
    data_obs: torch.Tensor,
    data_clean: torch.Tensor,
    coordinate: str,
    r_value: int,
    method: str,
    hyper: dict[str, Any],
    baseline_hyper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _model_for_name(coordinate)
    if method == "baseline_ls":
        k, stats, _, _, _ = _fit_baseline(model, data_obs, r_value, str(hyper["solver"]), float(hyper["ridge"]))
        row = _evaluate_k(model, data_obs, data_clean, r_value, k, extra=stats)
    else:
        if baseline_hyper is None:
            baseline_hyper = {"solver": "ridge", "ridge": 1e-8}
        k0, _, h_in, h_out, _ = _fit_baseline(model, data_obs, r_value, str(baseline_hyper["solver"]), float(baseline_hyper["ridge"]))
        h_seq = torch.cat([h_in[:, :1], h_out], dim=1)
        if method == "denoised_latent":
            k, initial_h, opt_stats = _optimize_denoised(k0, h_seq, hyper)
            row = _evaluate_k(model, data_obs, data_clean, r_value, k, initial_h=initial_h, extra=opt_stats)
        else:
            k, opt_stats = _optimize_k(k0, h_seq, method, hyper)
            row = _evaluate_k(model, data_obs, data_clean, r_value, k, extra=opt_stats)
    return {
        **row,
        "coordinate_type": coordinate,
        "operator_method": method,
        "selected_hyperparameters": json.dumps(hyper, sort_keys=True),
        "solver_optimizer": hyper.get("solver", hyper.get("optimizer", "Adam")),
    }


def _select_and_test_method(
    val_clean: torch.Tensor,
    test_clean: torch.Tensor,
    sigma: float,
    r_value: int,
    coordinate: str,
    method: str,
    baseline_selection: dict[tuple[float, int, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    val_obs = _noisy_positions(val_clean, sigma, TRAIN_DATA_SEED + int(sigma * 1e8) + 501)
    test_obs = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 501)
    base = baseline_selection.get((sigma, r_value, coordinate))
    base_hyper = json.loads(base["selected_hyperparameters"]) if base else {"solver": "ridge", "ridge": 1e-8}
    candidate_rows = []
    best_row = None
    best_hyper = None
    for hyper in _candidate_hypers(method):
        row = _run_candidate(val_obs, val_clean, coordinate, r_value, method, hyper, base_hyper)
        row.update({"sigma": sigma, "observed_ls_transitions_R": r_value, "total_observed_positions_required": M + r_value})
        row["validation_future_10_period_mse"] = row["future_10_period_mse"]
        candidate_rows.append(row)
        if best_row is None or float(row["future_10_period_mse"]) < float(best_row["future_10_period_mse"]):
            best_row, best_hyper = row, hyper
    assert best_hyper is not None
    test_row = _run_candidate(test_obs, test_clean, coordinate, r_value, method, best_hyper, base_hyper)
    test_row.update({
        "sigma": sigma,
        "observed_ls_transitions_R": r_value,
        "total_observed_positions_required": M + r_value,
        "validation_future_10_period_mse": best_row["future_10_period_mse"],
        "test_future_10_period_mse": test_row["future_10_period_mse"],
        "selection_split": "validation_to_test",
    })
    return test_row, candidate_rows


def _run_all() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    torch.manual_seed(SEED)
    val_clean = _make_long_data(VAL_NUM, TRAIN_DATA_SEED + 1000)
    test_clean = _make_long_data(TEST_NUM, TEST_DATA_SEED + 1000)
    baseline_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    baseline_selection: dict[tuple[float, int, str], dict[str, Any]] = {}
    for sigma in SIGMAS:
        for r_value in R_VALUES:
            for coordinate in COORDINATES:
                test_row, candidates = _select_and_test_method(val_clean, test_clean, sigma, r_value, coordinate, "baseline_ls", {})
                baseline_rows.append(test_row)
                selection_rows.extend(candidates)
                baseline_selection[(sigma, r_value, coordinate)] = test_row

    method_results: dict[str, list[dict[str, Any]]] = {
        "stable_prior_results": [],
        "spectral_prior_results": [],
        "multi_step_fit_results": [],
        "denoised_latent_results": [],
    }
    method_map = {
        "stable_power": "stable_prior_results",
        "spectral_stable": "spectral_prior_results",
        "spectral_unit": "spectral_prior_results",
        "multi_step": "multi_step_fit_results",
        "denoised_latent": "denoised_latent_results",
    }
    # Run priors on the primary coordinates over the full grid, plus fixed delay
    # at the two most informative noisy short-prefix settings.
    for sigma in PRIOR_SIGMAS:
        for r_value in R_VALUES:
            coords = list(PRIMARY_COORDINATES)
            if sigma in [1e-3, 1e-2] and r_value in [16, 32]:
                coords.append("fixed_delay")
            for coordinate in coords:
                for method, table in method_map.items():
                    test_row, candidates = _select_and_test_method(val_clean, test_clean, sigma, r_value, coordinate, method, baseline_selection)
                    method_results[table].append(test_row)
                    selection_rows.extend(candidates)

    all_rows = baseline_rows + [r for rows in method_results.values() for r in rows]
    best_by_setting: dict[tuple[float, int, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (float(row["sigma"]), int(row["observed_ls_transitions_R"]), row["coordinate_type"])
        if key not in best_by_setting or float(row["test_future_10_period_mse"]) < float(best_by_setting[key]["test_future_10_period_mse"]):
            best_by_setting[key] = row
    error_rows = _error_decomposition(test_clean, best_by_setting)
    threshold_rows = [
        {
            "sigma": r["sigma"],
            "observed_ls_transitions_R": r["observed_ls_transitions_R"],
            "coordinate_type": r["coordinate_type"],
            "operator_method": r["operator_method"],
            "test_future_10_period_mse": r["test_future_10_period_mse"],
            **{f"crosses_{threshold:g}": r[f"crosses_{threshold:g}"] for threshold in THRESHOLDS},
        }
        for r in all_rows
    ]
    outputs = {
        "baseline_ls_results": baseline_rows,
        **method_results,
        "hyperparameter_selection": selection_rows,
        "summary": all_rows,
        "full_run_table": selection_rows + all_rows + error_rows,
        "error_decomposition": error_rows,
        "threshold_crossings": threshold_rows,
    }
    setup = {
        "diagnostic": "stage1B_noisy_operator_identification_priors",
        "dtype": str(DTYPE),
        "window_size_m": M,
        "observed_ls_transitions_R": R_VALUES,
        "sigmas": SIGMAS,
        "prior_sigmas": PRIOR_SIGMAS,
        "coordinates": COORDINATES,
        "primary_coordinates": PRIMARY_COORDINATES,
        "optimizer_steps": OPT_STEPS,
        "patience": PATIENCE,
        "runtime_note": "Baseline LS uses the prioritized compact grid R={8,16,32}, sigma={0,1e-3,1e-2}; expensive operator-prior sweeps use the noisy sigmas only, as allowed by the spec when runtime is high.",
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher_loss": False,
        "uses_mlp_generated_k": False,
        "K_h_identified_from_observed_latent_transitions_only": True,
    }
    return outputs, setup


def _error_decomposition(test_clean: torch.Tensor, best: dict[tuple[float, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (sigma, r_value, coordinate), best_row in best.items():
        if r_value not in [16, 32, 64] or sigma == 0.0:
            continue
        model = _model_for_name(coordinate)
        noisy = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 501)
        method = best_row["operator_method"]
        hyper = json.loads(best_row["selected_hyperparameters"])
        base = {"solver": "ridge", "ridge": 1e-8}
        if method != "baseline_ls":
            # Use the selected baseline for this setting if it is available.
            base = {"solver": "ridge", "ridge": 1e-8}
        clean_k, _, clean_h_in, clean_h_out, _ = _fit_baseline(model, test_clean, r_value, "ridge", 1e-8)
        noisy_k, _, noisy_h_in, noisy_h_out, _ = _fit_baseline(model, noisy, r_value, "ridge", 1e-8)
        h_clean_seq = torch.cat([clean_h_in[:, :1], clean_h_out], dim=1)
        h_noisy_seq = torch.cat([noisy_h_in[:, :1], noisy_h_out], dim=1)
        if method not in ["baseline_ls"]:
            if method == "denoised_latent":
                clean_k, clean_init, _ = _optimize_denoised(clean_k, h_clean_seq, hyper)
                noisy_k, noisy_init, _ = _optimize_denoised(noisy_k, h_noisy_seq, hyper)
            else:
                clean_k, _ = _optimize_k(clean_k, h_clean_seq, method, hyper)
                noisy_k, _ = _optimize_k(noisy_k, h_noisy_seq, method, hyper)
                clean_init = model.encode(_future_windows(test_clean, r_value)[:, :1])[:, 0]
                noisy_init = model.encode(_future_windows(noisy, r_value)[:, :1])[:, 0]
        else:
            clean_init = model.encode(_future_windows(test_clean, r_value)[:, :1])[:, 0]
            noisy_init = model.encode(_future_windows(noisy, r_value)[:, :1])[:, 0]
        cases = [
            ("clean_prefix_K_clean_initial_state", clean_k, clean_init, test_clean),
            ("noisy_prefix_K_noisy_initial_state", noisy_k, noisy_init, noisy),
            ("clean_K_noisy_initial_state", clean_k, noisy_init, noisy),
            ("noisy_K_clean_initial_state", noisy_k, clean_init, test_clean),
        ]
        for case, k, init, observed_source in cases:
            row = _evaluate_k(model, observed_source, test_clean, r_value, k, initial_h=init)
            row.update({
                "case": case,
                "sigma": sigma,
                "observed_ls_transitions_R": r_value,
                "coordinate_type": coordinate,
                "operator_method": method,
                "selected_hyperparameters": best_row["selected_hyperparameters"],
            })
            rows.append(row)
    return rows


def _plot_method_comparison(rows: list[dict[str, Any]], filename: str) -> None:
    selected = [r for r in rows if int(r["observed_ls_transitions_R"]) in [16, 32, 64]]
    labels = [f"{r['operator_method']}\n{r['coordinate_type']}\ns={float(r['sigma']):g} R={r['observed_ls_transitions_R']}" for r in selected[:80]]
    vals = [float(r.get("test_future_10_period_mse", r["future_10_period_mse"])) for r in selected[:80]]
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=6)
    plt.ylabel("test future 10-period MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_lines(rows: list[dict[str, Any]], xkey: str, group_keys: list[str], filename: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = " / ".join(str(row[k]) for k in group_keys)
        groups.setdefault(key, []).append(row)
    for key, group in sorted(groups.items())[:18]:
        group.sort(key=lambda r: float(r[xkey]))
        plt.plot([float(r[xkey]) for r in group], [float(r["test_future_10_period_mse"]) for r in group], marker="o", label=key)
    if "R" in xkey:
        plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ylabel)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], xkey: str, filename: str) -> None:
    plt.figure(figsize=(7, 5))
    xs = [float(r[xkey]) for r in rows if r.get(xkey, "") not in ["", "nan"]]
    ys = [float(r["test_future_10_period_mse"]) for r in rows if r.get(xkey, "") not in ["", "nan"]]
    plt.scatter(xs, ys, alpha=0.65)
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel("test future 10-period MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_heatmap(rows: list[dict[str, Any]], filename: str) -> None:
    coords = sorted({r["coordinate_type"] for r in rows})
    methods = sorted({r["operator_method"] for r in rows})
    matrix = np.full((len(coords), len(methods)), np.nan)
    for i, coord in enumerate(coords):
        for j, method in enumerate(methods):
            vals = [float(r["test_future_10_period_mse"]) for r in rows if r["coordinate_type"] == coord and r["operator_method"] == method]
            if vals:
                matrix[i, j] = min(vals)
    plt.figure(figsize=(9, 4))
    plt.imshow(np.log10(matrix), aspect="auto", cmap="viridis")
    plt.colorbar(label="log10 best MSE")
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.yticks(range(len(coords)), coords)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], setup: dict[str, Any]) -> None:
    rows = outputs["summary"]
    best = sorted(rows, key=lambda r: float(r["test_future_10_period_mse"]))[:20]
    hard = [r for r in rows if float(r["sigma"]) in [1e-3, 1e-2] and int(r["observed_ls_transitions_R"]) == 16]
    hard_best = sorted(hard, key=lambda r: float(r["test_future_10_period_mse"]))[:12]
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Noisy Operator Identification Priors</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}code{background:#f6f6f6;padding:2px 4px}</style>",
        "</head><body>",
        "<h1>Stage 1B Noisy Operator-Identification Priors Diagnostic</h1>",
        "<p>K_h is always identified from observed latent prefix transitions only. Clean future trajectories are used for validation/test scoring and hyperparameter selection, not for fitting K_h.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Best Overall Test Runs</h2>",
        _html_table(best, ["sigma", "observed_ls_transitions_R", "coordinate_type", "operator_method", "test_future_10_period_mse", "validation_future_10_period_mse", "spectral_radius_k_h", "selected_hyperparameters"]),
        "<h2>Hard Short-Prefix Settings</h2>",
        _html_table(hard_best, ["sigma", "observed_ls_transitions_R", "coordinate_type", "operator_method", "test_future_10_period_mse", "max_abs_trajectory_error", "selected_hyperparameters"]),
        "<h2>Interpretation</h2>",
        "<ul>",
        "<li>Noiseless cases remain at numerical floor; the question is whether priors lower noisy finite-prefix K_h error.</li>",
        "<li>Compare baseline_ls against stable_power, spectral_stable/unit, multi_step, and denoised_latent at the same sigma/R/coordinate.</li>",
        "<li>Error decomposition separates noisy K from noisy initial state for selected best methods.</li>",
        "</ul>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _write_outputs(outputs: dict[str, list[dict[str, Any]]], setup: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_map = {
        "summary": "summary.csv",
        "full_run_table": "full_run_table.csv",
        "baseline_ls_results": "baseline_ls_results.csv",
        "stable_prior_results": "stable_prior_results.csv",
        "spectral_prior_results": "spectral_prior_results.csv",
        "multi_step_fit_results": "multi_step_fit_results.csv",
        "denoised_latent_results": "denoised_latent_results.csv",
        "hyperparameter_selection": "hyperparameter_selection.csv",
        "error_decomposition": "error_decomposition.csv",
        "threshold_crossings": "threshold_crossings.csv",
    }
    for key, filename in file_map.items():
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    summary = outputs["summary"]
    _plot_method_comparison(summary, "method_comparison_by_sigma_R.png")
    _plot_lines([r for r in summary if float(r["sigma"]) in [1e-3, 1e-2]], "observed_ls_transitions_R", ["operator_method", "coordinate_type"], "future_mse_vs_R_by_method.png", "test future 10-period MSE")
    _plot_lines(summary, "sigma", ["operator_method", "coordinate_type"], "future_mse_vs_noise_by_method.png", "test future 10-period MSE")
    _plot_lines(outputs["stable_prior_results"], "sigma", ["coordinate_type"], "stable_prior_lambda_sweep.png", "test future 10-period MSE")
    _plot_scatter(summary, "spectral_radius_k_h", "spectral_radius_vs_future_mse.png")
    _plot_lines(outputs["multi_step_fit_results"], "observed_ls_transitions_R", ["coordinate_type"], "multi_step_fit_comparison.png", "test future 10-period MSE")
    _plot_lines(outputs["denoised_latent_results"], "observed_ls_transitions_R", ["coordinate_type"], "denoised_latent_lambda_sweep.png", "test future 10-period MSE")
    _plot_method_comparison(outputs["error_decomposition"], "error_decomposition_best_methods.png")
    _plot_heatmap(outputs["threshold_crossings"], "threshold_crossings_heatmap.png")
    _plot_scatter(summary, "ls_covariance_condition_number", "condition_number_vs_method.png")
    _write_html(outputs, setup)


def main() -> None:
    outputs, setup = _run_all()
    _write_outputs(outputs, setup)


if __name__ == "__main__":
    main()
