"""Stage 1B operator-identification follow-up diagnostics.

This script writes only new follow-up output directories. It does not mutate
prior completed experiment outputs.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
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
from torch import nn

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    DIAGNOSTIC_NUM_TEST,
    DIAGNOSTIC_NUM_TRAIN,
    _parameter_count,
)
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import (
    _angle_error,
    _eig_angle,
    _true_k,
)
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import (
    CURRICULUM_STAGES,
    OBS_PREFIX_TRANSITIONS,
    RIDGE,
    ROLLOUT_PERIODS,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _evaluate_raw_ls,
    _fit_dense_k_qr,
    _make_prefix_future_data,
    _rollout_matrix,
    _rollout_truth,
    _state_windows,
    _trajectory_errors,
)
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

CAPACITY_DIR = Path("outputs/sho/stage1B/learned_state_ls_capacity_screen")
WEIGHTED_DIR = Path("outputs/sho/stage1B/weighted_direct_state_ls_diagnostic")
COMBINED_DIR = Path("outputs/sho/stage1B/operator_identification_followups")

SEED = 0
DTYPE = torch.float64
LEARNING_RATE = 1e-3
MAX_EPOCHS_PER_STAGE = 20_000
PATIENCE = 5_000
STOPPING_MIN_DELTA = 0.0
EVAL_EVERY = 1_000
EARLY_BAD_STAGE = 64
EARLY_BAD_FACTOR = 100.0
REFERENCE_STAGE64_MSE = 8.084020032823974e-09
REFERENCE_PREVIOUS_H2_BY_STAGE = {
    "1": 4.4476597579586386e-08,
    "2": 2.756431526099045e-08,
    "4": 2.3085658662735505e-08,
    "8": 1.2119021259712952e-08,
    "16": 1.0707948690785407e-08,
    "32": 8.925246004226555e-09,
    "64": 8.084020032823974e-09,
    "128": 7.903793980252591e-09,
    "full_10_period": 8.110565470240112e-09,
}
WEIGHT_MAX_EPOCHS = 20_000
WEIGHT_PATIENCE = 5_000
TRAIN_BATCH_SIZE = 32

CAPACITY_CONFIGS = [
    {"config": "h2_shallow", "h_dim": 2, "depth": "shallow", "width": 32, "layers": 1},
    {"config": "h4_shallow", "h_dim": 4, "depth": "shallow", "width": 32, "layers": 1},
    {"config": "h8_shallow", "h_dim": 8, "depth": "shallow", "width": 32, "layers": 1},
    {"config": "h4_deep64", "h_dim": 4, "depth": "deep64", "width": 64, "layers": 2},
    {"config": "h8_deep64", "h_dim": 8, "depth": "deep64", "width": 64, "layers": 2},
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _mlp(in_dim: int, out_dim: int, *, width: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = in_dim
    for _ in range(layers):
        modules.extend([nn.Linear(current, width), nn.SiLU()])
        current = width
    modules.append(nn.Linear(current, out_dim))
    return nn.Sequential(*modules)


def _batch_positions(positions: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if positions.shape[0] <= TRAIN_BATCH_SIZE:
        return positions
    indices = torch.randperm(positions.shape[0], generator=generator, device=positions.device)[:TRAIN_BATCH_SIZE]
    return positions.index_select(0, indices)


class CapacityLearnedStateLS(nn.Module):
    def __init__(self, h_dim: int, *, width: int, layers: int) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.encoder = _mlp(2, h_dim, width=width, layers=layers)
        self.decoder = _mlp(h_dim, 2, width=width, layers=layers)

    def encode_state(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode_state(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_states)

    def identify_k(self, prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_matrix, residual = self.identify_k_for_loss(prefix_in, prefix_out)
        conditions = torch.linalg.cond(self.encode_state(prefix_in))
        return k_matrix, residual, conditions

    def identify_k_for_loss(self, prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = self.encode_state(prefix_in)
        h_out = self.encode_state(prefix_out)
        h_dim = h_in.shape[-1]
        eye = torch.eye(h_dim, dtype=h_in.dtype, device=h_in.device).unsqueeze(0)
        gram = torch.bmm(h_in.transpose(1, 2), h_in)
        rhs = torch.bmm(h_in.transpose(1, 2), h_out)
        solution = torch.linalg.solve(gram + RIDGE * eye, rhs)
        residual = torch.mean((torch.bmm(h_in, solution) - h_out) ** 2)
        return solution.transpose(1, 2), residual

    def rollout_decoded(self, s0: torch.Tensor, k_matrix: torch.Tensor, steps: int) -> torch.Tensor:
        h0 = self.encode_state(s0)
        latent_rollout = _rollout_matrix(k_matrix, h0, steps)
        return self.decode_state(latent_rollout)


def _k_spectral(k_matrix: torch.Tensor) -> dict[str, Any]:
    k_np = k_matrix.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(k_np)
    return {
        "k_spectral_radius_mean": float(np.mean(np.abs(eigvals).max(axis=1))),
        "k_determinant_mean": float(np.mean(np.linalg.det(k_np))),
    }


def _capacity_loss_and_parts(
    model: CapacityLearnedStateLS,
    positions: torch.Tensor,
    stage: int | str,
) -> tuple[torch.Tensor, dict[str, Any], float, float]:
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    ls_start = time.perf_counter()
    k_matrix, residual, conditions = model.identify_k(prefix_in, prefix_out)
    ls_time = time.perf_counter() - ls_start
    true_states = _rollout_truth(positions, stage)
    rollout_start = time.perf_counter()
    predicted = model.rollout_decoded(true_states[:, 0], k_matrix, true_states.shape[1] - 1)
    rollout_time = time.perf_counter() - rollout_start
    loss = torch.mean((predicted - true_states) ** 2)
    with torch.no_grad():
        decoded = model.decode_state(model.encode_state(true_states))
        ae_mse = torch.mean((decoded - true_states) ** 2)
    parts = {
        **_trajectory_errors(predicted, true_states),
        "state_autoencoder_mse": float(ae_mse.item()),
        "latent_ls_residual": float(residual.item()),
        "ls_condition_number_mean": float(conditions.mean().item()),
        "ls_condition_number_max": float(conditions.max().item()),
        "avg_ls_solve_time_sec": ls_time / max(1, positions.shape[0]),
        "avg_rollout_time_sec": rollout_time / max(1, positions.shape[0]),
        **_k_spectral(k_matrix),
    }
    return loss, parts, ls_time, rollout_time


def _capacity_train_loss(model: CapacityLearnedStateLS, positions: torch.Tensor, stage: int | str) -> torch.Tensor:
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    k_matrix, _ = model.identify_k_for_loss(prefix_in, prefix_out)
    true_states = _rollout_truth(positions, stage)
    predicted = model.rollout_decoded(true_states[:, 0], k_matrix, true_states.shape[1] - 1)
    return torch.mean((predicted - true_states) ** 2)


def _evaluate_capacity(
    model: CapacityLearnedStateLS,
    positions: torch.Tensor,
    stage: int | str,
    config: dict[str, Any],
    *,
    best_epoch: int,
    wall_clock_sec: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, _, _ = _capacity_loss_and_parts(model, positions, stage)
    return {
        "experiment": "learned_state_ls_capacity_screen",
        "model": "learned_state_ls_operator",
        "config": config["config"],
        "h_dim": config["h_dim"],
        "depth": config["depth"],
        "dtype": str(DTYPE).replace("torch.", ""),
        "stage": stage,
        "best_epoch": best_epoch,
        "parameter_count": _parameter_count(model),
        "wall_clock_sec": wall_clock_sec,
        **parts,
    }


def _train_capacity_config(config: dict[str, Any], train_positions: torch.Tensor, test_positions: torch.Tensor) -> list[dict[str, Any]]:
    run_dir = CAPACITY_DIR / config["config"]
    final_path = run_dir / "metrics.json"
    if final_path.exists():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        rows = list(payload.get("history", []))
        if payload.get("final"):
            rows.append(payload["final"])
        return rows

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = CapacityLearnedStateLS(config["h_dim"], width=config["width"], layers=config["layers"]).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    global_epoch = 0
    run_start = time.perf_counter()
    stopped_run_reason = ""

    _write_json(
        run_dir / "setup.json",
        {
            **config,
            "dtype": str(DTYPE),
            "position_only": True,
            "uses_velocity": False,
            "uses_omega_supervision": False,
            "uses_coefficient_supervision": False,
            "uses_analytic_decoder": False,
            "uses_spectral_loss": False,
            "uses_contrastive_loss": False,
            "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
            "patience": PATIENCE,
            "stopping_min_delta": STOPPING_MIN_DELTA,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "ridge": RIDGE,
            "obs_prefix_transitions": OBS_PREFIX_TRANSITIONS,
        },
    )

    for stage in CURRICULUM_STAGES:
        batch_generator = torch.Generator(device=train_positions.device)
        batch_generator.manual_seed(SEED + int(config["h_dim"]) * 1000 + len(history))
        best_stage_metric = float("inf")
        best_stage_state = copy.deepcopy(model.state_dict())
        best_stage_epoch = 0
        epochs_since_improvement = 0
        stopped_because = "max_epochs_reached"
        force_stop_run = False
        for epoch in range(1, MAX_EPOCHS_PER_STAGE + 1):
            global_epoch += 1
            batch_positions = _batch_positions(train_positions, batch_generator)
            model.train()
            optimizer.zero_grad()
            loss = _capacity_train_loss(model, batch_positions, stage)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            if epoch % EVAL_EVERY == 0 or epoch == 1:
                wall_clock = time.perf_counter() - run_start
                eval_row = _evaluate_capacity(
                    model,
                    test_positions,
                    stage,
                    config,
                    best_epoch=global_epoch,
                    wall_clock_sec=wall_clock,
                )
                eval_row.update(
                    {
                        "curriculum_stage": stage,
                        "stage_epoch": epoch,
                        "global_epoch": global_epoch,
                        "train_loss": float(loss.item()),
                    },
                )
                history.append(eval_row)
                _write_csv(run_dir / "history.csv", history)
                _write_json(run_dir / "training_history.json", {"history": history})
                metric = eval_row["rollout_mse"]
                if metric < best_stage_metric - STOPPING_MIN_DELTA:
                    best_stage_metric = metric
                    best_stage_state = copy.deepcopy(model.state_dict())
                    best_stage_epoch = epoch
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += EVAL_EVERY
                reference = REFERENCE_PREVIOUS_H2_BY_STAGE.get(str(stage))
                if reference is not None and metric > EARLY_BAD_FACTOR * reference:
                    stopped_because = f"stopped_during_stage_{stage}_more_than_100x_worse_than_previous_h2_same_stage"
                    force_stop_run = True
                    break
                if epochs_since_improvement >= PATIENCE:
                    stopped_because = "patience_exhausted"
                    break

        model.load_state_dict(best_stage_state)
        best_epoch_global = global_epoch - (epoch - best_stage_epoch)
        stage_summary = _evaluate_capacity(
            model,
            test_positions,
            stage,
            config,
            best_epoch=best_epoch_global,
            wall_clock_sec=time.perf_counter() - run_start,
        )
        stage_summary.update(
            {
                "curriculum_stage": stage,
                "stage_epoch": best_stage_epoch,
                "global_epoch": best_epoch_global,
                "stopped_because": stopped_because,
            },
        )
        history.append(stage_summary)
        _write_csv(run_dir / "history.csv", history)
        _write_json(run_dir / "training_history.json", {"history": history})

        if stage == EARLY_BAD_STAGE and stage_summary["rollout_mse"] > EARLY_BAD_FACTOR * REFERENCE_STAGE64_MSE:
            stopped_run_reason = "stopped_after_stage64_more_than_100x_worse_than_previous_h2"
            break
        reference = REFERENCE_PREVIOUS_H2_BY_STAGE.get(str(stage))
        if reference is not None and stage_summary["rollout_mse"] > EARLY_BAD_FACTOR * reference:
            stopped_run_reason = f"stopped_after_stage_{stage}_more_than_100x_worse_than_previous_h2_same_stage"
            break
        if force_stop_run:
            stopped_run_reason = stopped_because
            break

    final_stage = "full_10_period"
    final_row = _evaluate_capacity(
        model,
        test_positions,
        final_stage,
        config,
        best_epoch=history[-1].get("best_epoch", 0),
        wall_clock_sec=time.perf_counter() - run_start,
    )
    final_row["run_stop_reason"] = stopped_run_reason or "completed_screen"
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _write_json(final_path, {"history": history, "final": final_row})
    return history + [final_row]


class TransitionWeightNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, states_in: torch.Tensor, states_out: torch.Tensor) -> torch.Tensor:
        features = torch.cat((states_in, states_out), dim=-1)
        logits = self.net(features).squeeze(-1)
        return torch.softmax(logits, dim=1) * states_in.shape[1]


def _weighted_lstsq_k(
    states_in: torch.Tensor,
    states_out: torch.Tensor,
    weights: torch.Tensor,
    *,
    compute_condition: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    sqrt_w = torch.sqrt(torch.clamp(weights, min=1e-12)).unsqueeze(-1)
    a_matrix = states_in * sqrt_w
    b_matrix = states_out * sqrt_w
    eye = torch.eye(2, dtype=states_in.dtype, device=states_in.device).unsqueeze(0)
    gram = torch.bmm(a_matrix.transpose(1, 2), a_matrix)
    rhs = torch.bmm(a_matrix.transpose(1, 2), b_matrix)
    solution = torch.linalg.solve(gram + 1e-12 * eye, rhs)
    conditions = torch.linalg.cond(a_matrix) if compute_condition else None
    return solution.transpose(1, 2), conditions


def _k_error_metrics(k_matrix: torch.Tensor, omegas: torch.Tensor) -> dict[str, Any]:
    k_true = _true_k(omegas)
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    determinant = np.linalg.det(k_np)
    true_determinant = np.linalg.det(true_np)
    return {
        "k_frobenius_error": float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2)))),
        "trace_c_error": float(np.mean(np.abs(trace - true_trace))),
        "determinant_error": float(np.mean(np.abs(determinant - true_determinant))),
        "eigen_angle_error": float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np)))),
    }


def _weight_stats(weights: torch.Tensor) -> dict[str, Any]:
    normalized = weights / torch.sum(weights, dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -torch.sum(normalized * torch.log(normalized.clamp_min(1e-12)), dim=1)
    effective_n = 1.0 / torch.sum(normalized**2, dim=1)
    return {
        "learned_weight_entropy": float(entropy.mean().item()),
        "effective_num_transitions": float(effective_n.mean().item()),
        "max_transition_weight": float(normalized.max(dim=1).values.mean().item()),
    }


def _evaluate_weighted(
    model: TransitionWeightNet | None,
    positions: torch.Tensor,
    omegas: torch.Tensor,
    *,
    variant: str,
    best_epoch: int,
    wall_clock_sec: float,
) -> dict[str, Any]:
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    ls_start = time.perf_counter()
    if model is None:
        weights = torch.ones(prefix_in.shape[:2], dtype=prefix_in.dtype, device=prefix_in.device)
        k_matrix, conditions = _fit_dense_k_qr(prefix_in, prefix_out)
    else:
        weights = model(prefix_in, prefix_out)
        k_matrix, conditions = _weighted_lstsq_k(prefix_in, prefix_out, weights)
    ls_time = time.perf_counter() - ls_start
    true_states = _rollout_truth(positions, "full_10_period")
    rollout_start = time.perf_counter()
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    rollout_time = time.perf_counter() - rollout_start
    return {
        "experiment": "weighted_direct_state_ls_diagnostic",
        "model": variant,
        "dtype": str(DTYPE).replace("torch.", ""),
        "stage": "full_10_period",
        "best_epoch": best_epoch,
        "parameter_count": 0 if model is None else _parameter_count(model),
        "wall_clock_sec": wall_clock_sec,
        "ls_condition_number_mean": float(conditions.mean().item()) if conditions is not None else float("nan"),
        "ls_condition_number_max": float(conditions.max().item()) if conditions is not None else float("nan"),
        "avg_ls_solve_time_sec": ls_time / max(1, positions.shape[0]),
        "avg_rollout_time_sec": rollout_time / max(1, positions.shape[0]),
        **_trajectory_errors(predicted, true_states),
        **_k_error_metrics(k_matrix, omegas),
        **_weight_stats(weights),
    }


def _train_weighted_ls(train_positions: torch.Tensor, test_positions: torch.Tensor, test_omegas: torch.Tensor) -> list[dict[str, Any]]:
    final_path = WEIGHTED_DIR / "metrics.json"
    if final_path.exists():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        return payload["rows"]
    WEIGHTED_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = TransitionWeightNet().to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improvement = 0
    run_start = time.perf_counter()

    prefix_in_train, prefix_out_train = _state_windows(train_positions, 0, OBS_PREFIX_TRANSITIONS)
    true_states_train = _rollout_truth(train_positions, "full_10_period")
    batch_generator = torch.Generator(device=train_positions.device)
    batch_generator.manual_seed(SEED + 777)
    for epoch in range(1, WEIGHT_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        if train_positions.shape[0] <= TRAIN_BATCH_SIZE:
            batch_in = prefix_in_train
            batch_out = prefix_out_train
            batch_truth = true_states_train
        else:
            indices = torch.randperm(train_positions.shape[0], generator=batch_generator, device=train_positions.device)[:TRAIN_BATCH_SIZE]
            batch_in = prefix_in_train.index_select(0, indices)
            batch_out = prefix_out_train.index_select(0, indices)
            batch_truth = true_states_train.index_select(0, indices)
        weights = model(batch_in, batch_out)
        k_matrix, _ = _weighted_lstsq_k(batch_in, batch_out, weights, compute_condition=False)
        predicted = _rollout_matrix(k_matrix, batch_truth[:, 0], batch_truth.shape[1] - 1)
        loss = torch.mean((predicted - batch_truth) ** 2)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            row = _evaluate_weighted(
                model,
                test_positions,
                test_omegas,
                variant="learned_transition_weighted_ls",
                best_epoch=epoch,
                wall_clock_sec=time.perf_counter() - run_start,
            )
            row["train_loss"] = float(loss.item())
            row["epoch"] = epoch
            history.append(row)
            _write_csv(WEIGHTED_DIR / "history.csv", history)
            metric = row["rollout_mse"]
            if metric < best_metric - STOPPING_MIN_DELTA:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            if epochs_since_improvement >= WEIGHT_PATIENCE:
                break

    model.load_state_dict(best_state)
    uniform_row = _evaluate_weighted(
        None,
        test_positions,
        test_omegas,
        variant="uniform_direct_state_ls",
        best_epoch=0,
        wall_clock_sec=0.0,
    )
    weighted_row = _evaluate_weighted(
        model,
        test_positions,
        test_omegas,
        variant="learned_transition_weighted_ls",
        best_epoch=best_epoch,
        wall_clock_sec=time.perf_counter() - run_start,
    )
    rows = [uniform_row, weighted_row]
    torch.save(model.state_dict(), WEIGHTED_DIR / "learned_weight_parameters.pt")
    _write_json(WEIGHTED_DIR / "metrics.json", {"history": history, "rows": rows})
    _write_csv(WEIGHTED_DIR / "summary.csv", rows)
    return rows


def _plot_capacity(rows: list[dict[str, Any]]) -> None:
    stages = [str(stage) for stage in CURRICULUM_STAGES]
    stage_to_x = {stage: idx for idx, stage in enumerate(stages)}
    history = [row for row in rows if row.get("curriculum_stage")]
    if not history:
        return
    for metric, filename, ylabel in [
        ("rollout_mse", "rollout_mse_vs_stage.png", "rollout MSE"),
        ("state_autoencoder_mse", "state_autoencoder_mse_vs_stage.png", "state AE MSE"),
        ("max_abs_trajectory_error", "max_error_vs_stage.png", "max abs error"),
    ]:
        plt.figure(figsize=(9.0, 5.2))
        for config in sorted({row["config"] for row in history}):
            selected = [row for row in history if row["config"] == config]
            best_by_stage: dict[str, dict[str, Any]] = {}
            for row in selected:
                stage = str(row["curriculum_stage"])
                if stage not in best_by_stage or float(row[metric]) < float(best_by_stage[stage][metric]):
                    best_by_stage[stage] = row
            ordered = sorted(best_by_stage.values(), key=lambda row: stage_to_x[str(row["curriculum_stage"])])
            plt.plot(
                [stage_to_x[str(row["curriculum_stage"])] for row in ordered],
                [float(row[metric]) for row in ordered],
                marker="o",
                label=config,
            )
        plt.yscale("log")
        plt.xticks(range(len(stages)), stages, rotation=30, ha="right")
        plt.xlabel("curriculum stage")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(CAPACITY_DIR / filename, dpi=160)
        plt.close()


def _plot_weighted(rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in rows]
    for metric, filename, ylabel in [
        ("rollout_mse", "rollout_mse_comparison.png", "10-period rollout MSE"),
        ("k_frobenius_error", "k_error_comparison.png", "K Frobenius error"),
        ("effective_num_transitions", "effective_transitions_comparison.png", "effective transitions"),
    ]:
        plt.figure(figsize=(7.0, 4.4))
        plt.bar(labels, [float(row[metric]) for row in rows])
        if metric != "effective_num_transitions":
            plt.yscale("log")
        plt.ylabel(ylabel)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(WEIGHTED_DIR / filename, dpi=160)
        plt.close()


def _write_simple_html(path: Path, title: str, rows: list[dict[str, Any]], plots: list[str], note: str) -> None:
    columns = [
        "model",
        "config",
        "h_dim",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "state_autoencoder_mse",
        "latent_ls_residual",
        "ls_condition_number_mean",
        "max_abs_trajectory_error",
        "final_time_abs_trajectory_error",
        "parameter_count",
        "wall_clock_sec",
        "best_epoch",
        "run_stop_reason",
        "k_frobenius_error",
        "trace_c_error",
        "determinant_error",
        "eigen_angle_error",
        "effective_num_transitions",
        "max_transition_weight",
    ]
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    images = "\n".join(f'<img src="{escape(plot)}" alt="{escape(plot)}">' for plot in plots)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    .note {{ background: #f7f9fc; border-left: 4px solid #5577aa; padding: 12px 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="note">{escape(note)}</div>
  <table>
    <thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in columns)}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <div class="grid">{images}</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _best_final_capacity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("curriculum_stage")
        and (row.get("stage") == "full_10_period" or row.get("run_stop_reason"))
    ]


def _load_previous_h2_result() -> dict[str, Any] | None:
    path = Path("outputs/sho/stage1B/learned_state_ls_operator_curriculum/summary.csv")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("model") == "learned_state_ls_operator" and row.get("h_dim") == "2" and row.get("stage") == "full_10_period" and not row.get("curriculum_stage"):
                row["model"] = "previous_learned_state_ls_h2"
                return row
    return None


def _write_combined(
    raw_row: dict[str, Any],
    capacity_rows: list[dict[str, Any]],
    weighted_rows: list[dict[str, Any]],
) -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw = dict(raw_row)
    raw["model"] = "raw_direct_state_ls"
    rows.append(raw)
    previous = _load_previous_h2_result()
    if previous is not None:
        rows.append(previous)
    rows.extend(_best_final_capacity_rows(capacity_rows))
    rows.extend(weighted_rows)
    _write_csv(COMBINED_DIR / "summary.csv", rows)

    best = min(rows, key=lambda row: float(row.get("rollout_mse", row.get("ten_period_rollout_mse", "inf")) or "inf"))
    note = (
        "Combined comparison across raw LS, previous learned-state LS h_dim=2, "
        "new learned-state capacity screen, and weighted direct-state LS. "
        f"Best rollout MSE row: {best.get('model')} {best.get('config', '')}."
    )
    _write_simple_html(
        COMBINED_DIR / "summary.html",
        "Stage 1B Operator Identification Follow-ups",
        rows,
        [],
        note,
    )
    _write_json(COMBINED_DIR / "summary.json", {"rows": rows, "interpretation": note})


def main() -> None:
    for directory in [CAPACITY_DIR, WEIGHTED_DIR, COMBINED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_positions, _, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_positions, test_omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    train_positions = train_positions.to(dtype=DTYPE)
    test_positions = test_positions.to(dtype=DTYPE)
    test_omegas = test_omegas.to(dtype=DTYPE)

    raw_row = _evaluate_raw_ls(test_positions, test_omegas)
    all_capacity_rows: list[dict[str, Any]] = []
    for config in CAPACITY_CONFIGS:
        all_capacity_rows.extend(_train_capacity_config(config, train_positions, test_positions))
        _write_csv(CAPACITY_DIR / "summary.csv", _best_final_capacity_rows(all_capacity_rows))
        _plot_capacity(all_capacity_rows)

    _write_csv(CAPACITY_DIR / "history_all.csv", all_capacity_rows)
    _write_json(CAPACITY_DIR / "metrics.json", {"rows": all_capacity_rows, "dtype": str(DTYPE)})
    _write_simple_html(
        CAPACITY_DIR / "summary.html",
        "Learned-State LS Capacity Screen",
        _best_final_capacity_rows(all_capacity_rows),
        ["rollout_mse_vs_stage.png", "state_autoencoder_mse_vs_stage.png", "max_error_vs_stage.png"],
        "Shallow is the previous one-hidden-layer H/D. Deep64 uses two hidden layers of width 64 with SiLU activations.",
    )

    weighted_rows = _train_weighted_ls(train_positions, test_positions, test_omegas)
    _plot_weighted(weighted_rows)
    _write_simple_html(
        WEIGHTED_DIR / "summary.html",
        "Weighted Direct-State LS Diagnostic",
        weighted_rows,
        ["rollout_mse_comparison.png", "k_error_comparison.png", "effective_transitions_comparison.png"],
        "K is always identified from the observed prefix. The learned network only weights prefix transitions.",
    )

    _write_combined(raw_row, all_capacity_rows, weighted_rows)


if __name__ == "__main__":
    main()
