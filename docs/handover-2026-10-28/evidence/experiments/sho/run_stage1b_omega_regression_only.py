"""Stage 1B diagnostic: regress trajectory omega directly from two states.

This experiment intentionally does not use the SHO decoder, coefficient targets,
trajectory reconstruction, or curriculum. It tests whether the map

    (x0, v0, x1, v1) -> omega_i

is learnable in isolation with the same Stage 1B train/test split.
"""

from __future__ import annotations

import copy
import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from experiments.sho.run_stage1b_two_state_omega import (
    BATCH_SIZE,
    DELTA_T,
    MAX_EPOCHS_PER_STAGE,
    NUM_TEST,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    STOPPING_MIN_DELTA,
    STOPPING_PATIENCE,
    _inverse_softplus,
    _make_base_conditions,
    _make_residual_branch,
    _omega_correlation,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/omega_regression_only")
LOG_EVERY = 5000


class OmegaRegressor(nn.Module):
    """Predict one positive trajectory frequency from two phase-space states."""

    def __init__(
        self,
        *,
        hidden_dims: list[int] | None = None,
        initial_omega_hat: float = 2.0,
    ) -> None:
        super().__init__()
        if initial_omega_hat <= 0:
            raise ValueError("initial_omega_hat must be positive")
        self.hidden_dims = list(hidden_dims or [])
        self.linear = nn.Linear(4, 1)
        self.residual = (
            _make_residual_branch(self.hidden_dims)
            if self.hidden_dims
            else None
        )
        self._initialize_omega_bias(initial_omega_hat)

    def _initialize_omega_bias(self, initial_omega_hat: float) -> None:
        with torch.no_grad():
            self.linear.bias.fill_(_inverse_softplus(initial_omega_hat - 1e-8).item())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_omega = self.linear(inputs)
        if self.residual is not None:
            raw_omega = raw_omega + self.residual(inputs)[..., 2:3]
        return (F.softplus(raw_omega.squeeze(-1)) + 1e-8)


def _variants() -> list[dict[str, Any]]:
    return [
        {
            "run_name": "linear",
            "hidden_dims": [],
            "learning_rate": 1e-2,
        },
        {
            "run_name": "tiny_residual",
            "hidden_dims": [4],
            "learning_rate": 1e-2,
        },
        {
            "run_name": "moderate_residual",
            "hidden_dims": [8, 8],
            "learning_rate": 1e-2,
        },
        {
            "run_name": "large_residual",
            "hidden_dims": [32, 32],
            "learning_rate": 3e-2,
        },
    ]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _make_inputs(
    initial_conditions: torch.Tensor,
    omegas: torch.Tensor,
) -> torch.Tensor:
    x0 = initial_conditions[:, 0]
    v0 = initial_conditions[:, 1]
    phases = omegas * DELTA_T
    x1 = x0 * torch.cos(phases) + (v0 / omegas) * torch.sin(phases)
    v1 = -x0 * omegas * torch.sin(phases) + v0 * torch.cos(phases)
    return torch.stack((x0, v0, x1, v1), dim=-1)


def _make_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    return (
        _make_inputs(train_initial_conditions, train_omegas),
        train_omegas,
        _make_inputs(test_initial_conditions, test_omegas),
        test_omegas,
    )


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual_sum = (prediction - target).square().sum()
    total_sum = (target - target.mean()).square().sum()
    if total_sum <= 0:
        return float("nan")
    return (1.0 - residual_sum / total_sum).item()


def _metrics(
    omega_hat: torch.Tensor,
    omega_true: torch.Tensor,
    *,
    train_loss: float,
    test_loss: float,
) -> dict[str, float]:
    error = omega_hat - omega_true
    return {
        "omega_mse": error.square().mean().item(),
        "mean_omega_error": error.mean().item(),
        "mean_abs_omega_error": error.abs().mean().item(),
        "max_omega_error": error.abs().max().item(),
        "omega_correlation": _omega_correlation(omega_hat, omega_true),
        "r2_score": _r2_score(omega_hat, omega_true),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }


def _evaluate(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_omegas: torch.Tensor,
    test_inputs: torch.Tensor,
    test_omegas: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        train_prediction = model(train_inputs)
        test_prediction = model(test_inputs)
        train_loss = mse_loss(train_prediction, train_omegas).item()
        test_loss = mse_loss(test_prediction, test_omegas).item()
    return (
        _metrics(
            test_prediction,
            test_omegas,
            train_loss=train_loss,
            test_loss=test_loss,
        ),
        test_prediction.detach().cpu(),
    )


def _train_variant(
    variant: dict[str, Any],
    train_inputs: torch.Tensor,
    train_omegas: torch.Tensor,
    test_inputs: torch.Tensor,
    test_omegas: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    run_name = str(variant["run_name"])
    learning_rate = float(variant["learning_rate"])
    model = OmegaRegressor(hidden_dims=list(variant["hidden_dims"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_train = train_inputs.shape[0]
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    final_train_loss = float("inf")
    final_test_loss = float("inf")

    for epoch in range(1, MAX_EPOCHS_PER_STAGE + 1):
        model.train()
        shuffled_indices = torch.randperm(num_train)
        loss_sum = 0.0
        example_count = 0
        for start in range(0, num_train, BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            omega_hat = model(train_inputs[batch_indices])
            loss = mse_loss(omega_hat, train_omegas[batch_indices])
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            batch_count = len(batch_indices)
            loss_sum += loss.item() * batch_count
            example_count += batch_count
        if stopped_because == "nan_or_inf_loss":
            break

        final_train_loss = loss_sum / example_count
        model.eval()
        with torch.no_grad():
            final_test_loss = mse_loss(model(test_inputs), test_omegas).item()

        if final_test_loss < best_test_loss - STOPPING_MIN_DELTA:
            best_test_loss = final_test_loss
            best_train_loss = final_train_loss
            best_epoch = epoch
            best_model_state = _clone_state_dict(model.state_dict())
            best_optimizer_state = _clone_state_dict(optimizer.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                (
                    f"{run_name} epoch {epoch:06d} "
                    f"train_loss={final_train_loss:.6e} "
                    f"test_loss={final_test_loss:.6e} "
                    f"best_test={best_test_loss:.6e}"
                ),
                flush=True,
            )

        if epochs_since_improvement >= STOPPING_PATIENCE:
            stopped_because = "patience_exhausted"
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)

    metrics, test_prediction = _evaluate(
        model,
        train_inputs,
        train_omegas,
        test_inputs,
        test_omegas,
    )
    metrics.update(
        {
            "run_name": run_name,
            "model_type": "omega_regressor",
            "hidden_dims": list(variant["hidden_dims"]),
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "batch_size": BATCH_SIZE,
            "epochs_completed": epoch if "epoch" in locals() else 0,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "final_epoch_train_loss": final_train_loss,
            "final_epoch_test_loss": final_test_loss,
            "stopped_because": stopped_because,
        },
    )
    run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = {
        "stage": "stage1B",
        "experiment": "omega_regression_only",
        "run_name": run_name,
        "input": "(x0, v0, x1, v1)",
        "target": "omega_i",
        "uses_sho_decoder": False,
        "predicts_trajectories": False,
        "predicts_coefficients": False,
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "model": "linear raw omega path plus optional residual raw omega path",
        "hidden_dims": list(variant["hidden_dims"]),
        "omega_parameterization": "omega_hat = softplus(raw_omega) + 1e-8",
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS_PER_STAGE,
        "stopping_monitor": "test_loss",
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "random_seed": SEED,
    }
    _write_json(run_dir / "setup.json", setup)
    _write_json(run_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _save_predicted_vs_true_plot(
        OUTPUT_DIR / f"predicted_vs_true_omega_{run_name}.png",
        run_name,
        test_prediction,
        test_omegas,
    )
    _write_run_html(run_dir, setup, metrics)
    return metrics, test_prediction


def _save_predicted_vs_true_plot(
    path: Path,
    run_name: str,
    omega_hat: torch.Tensor,
    omega_true: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(omega_true.detach().cpu(), omega_hat.detach().cpu(), s=18, alpha=0.75)
    plt.plot([1.5, 2.5], [1.5, 2.5], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("predicted omega")
    plt.title(f"Stage 1B Omega Regression: {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_comparison_plots(
    predictions: dict[str, torch.Tensor],
    test_omegas: torch.Tensor,
    rows: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, prediction in predictions.items():
        error = prediction - test_omegas.detach().cpu()
        plt.hist(error, bins=40, alpha=0.45, label=run_name)
    plt.xlabel("omega_hat - omega_true")
    plt.ylabel("count")
    plt.title("Stage 1B Omega Regression Error Histograms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "omega_error_histograms.png")
    plt.close()

    names = [str(row["run_name"]) for row in rows]
    correlations = [float(row["omega_correlation"]) for row in rows]
    r2_scores = [float(row["r2_score"]) for row in rows]
    x_positions = torch.arange(len(names)).numpy()
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar(x_positions - width / 2, correlations, width=width, label="correlation")
    plt.bar(x_positions + width / 2, r2_scores, width=width, label="R^2")
    plt.xticks(x_positions, names, rotation=20, ha="right")
    plt.ylabel("score")
    plt.title("Stage 1B Omega Regression Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "omega_correlation_comparison.png")
    plt.close()


def _write_run_html(run_dir: Path, setup: dict[str, Any], metrics: dict[str, Any]) -> None:
    setup_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><pre>{escape(str(value))}</pre></td></tr>"
        for key, value in setup.items()
    )
    metric_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Omega Regression {escape(str(setup["run_name"]))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Stage 1B Omega Regression: {escape(str(setup["run_name"]))}</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Metrics</h2>
  <table>{metric_rows}</table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "omega_mse",
        "mean_omega_error",
        "mean_abs_omega_error",
        "max_omega_error",
        "omega_correlation",
        "r2_score",
        "train_loss",
        "test_loss",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
        "learning_rate",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in [
            "predicted_vs_true_omega_linear.png",
            "predicted_vs_true_omega_tiny_residual.png",
            "predicted_vs_true_omega_moderate_residual.png",
            "predicted_vs_true_omega_large_residual.png",
            "omega_error_histograms.png",
            "omega_correlation_comparison.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Omega Regression Only</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Omega Regression Only</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (OUTPUT_DIR / "omega_regression_summary.html").write_text(html, encoding="utf-8")


def run_omega_regression_only() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_inputs, train_omegas, test_inputs, test_omegas = _make_split()
    rows = []
    predictions = {}
    for variant in _variants():
        print(
            f"running omega-only regressor: {variant['run_name']} "
            f"lr={float(variant['learning_rate']):g}",
            flush=True,
        )
        metrics, prediction = _train_variant(
            variant,
            train_inputs,
            train_omegas,
            test_inputs,
            test_omegas,
        )
        rows.append(metrics)
        predictions[str(variant["run_name"])] = prediction

    columns = [
        "run_name",
        "model_type",
        "hidden_dims",
        "learning_rate",
        "optimizer",
        "batch_size",
        "omega_mse",
        "mean_omega_error",
        "mean_abs_omega_error",
        "max_omega_error",
        "omega_correlation",
        "r2_score",
        "train_loss",
        "test_loss",
        "epochs_completed",
        "best_epoch",
        "best_train_loss",
        "best_test_loss",
        "final_epoch_train_loss",
        "final_epoch_test_loss",
        "stopped_because",
    ]
    _write_csv(
        OUTPUT_DIR / "omega_regression_summary.csv",
        [{column: row.get(column, "") for column in columns} for row in rows],
    )
    _write_json(OUTPUT_DIR / "omega_regression_summary.json", {"runs": rows})
    _save_comparison_plots(predictions, test_omegas, rows)
    _write_summary_html(rows)
    return rows


def main() -> None:
    rows = run_omega_regression_only()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
