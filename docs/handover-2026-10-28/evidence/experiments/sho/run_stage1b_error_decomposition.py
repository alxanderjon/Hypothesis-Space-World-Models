"""Stage 1B final 10-period trajectory error decomposition.

For the final feedforward residual model and the final GRU model, compare:

1. learned omega_hat + learned coefficients
2. true omega_i + learned coefficients
3. learned omega_hat + true coefficients
4. true omega_i + true coefficients
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.sho.run_stage1b_gru_two_state_curriculum import (
    GRU_HIDDEN_SIZE,
    SHOTwoStateGRUWindowModel,
)
from experiments.sho.run_stage1b_late_horizon_refinement_curriculum import HIDDEN_DIMS
from experiments.sho.run_stage1b_two_state_omega import (
    NUM_TEST,
    NUM_TIME_STEPS,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    _build_data_from_base,
    _make_base_conditions,
)
from hypothesis_space_world_models.evaluation.metrics import mean_squared_error


OUTPUT_DIR = Path("outputs/sho/stage1B/error_decomposition")
FINAL_HORIZON_PERIODS = 10.0
GRU_PARAMETERS = Path("outputs/sho/stage1B/gru_two_state_curriculum/learned_parameters.pt")
RESIDUAL_PARAMETERS = Path(
    "outputs/sho/stage1B/late_horizon_refinement_curriculum/learned_parameters.pt",
)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"missing parameters: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _decode(
    coefficients: torch.Tensor,
    omegas: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    a = coefficients[:, 0].unsqueeze(-1)
    b = coefficients[:, 1].unsqueeze(-1)
    phases = omegas.unsqueeze(-1) * times
    return a * torch.cos(phases) + b * torch.sin(phases)


def _omega_corr(omega_hat: torch.Tensor, omega_true: torch.Tensor) -> float:
    pred = omega_hat.detach().cpu()
    true = omega_true.detach().cpu()
    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * true_centered.square().sum(),
    )
    if denominator <= 0:
        return float("nan")
    return (pred_centered * true_centered).sum().div(denominator).item()


def _evaluate_model_outputs(
    model_name: str,
    model: torch.nn.Module,
    data: Any,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        learned_coefficients, _, learned_omega = model(data.two_state_inputs, data.times)
    true_coefficients = data.coefficients
    true_omega = data.omegas
    combinations = [
        (
            "learned_omega_learned_coefficients",
            learned_coefficients,
            learned_omega,
        ),
        (
            "true_omega_learned_coefficients",
            learned_coefficients,
            true_omega,
        ),
        (
            "learned_omega_true_coefficients",
            true_coefficients,
            learned_omega,
        ),
        (
            "true_omega_true_coefficients",
            true_coefficients,
            true_omega,
        ),
    ]
    rows = []
    baseline_mse = None
    for combination_name, coefficients, omegas in combinations:
        trajectory = _decode(coefficients, omegas, data.times)
        trajectory_mse = mean_squared_error(trajectory, data.trajectories)
        if combination_name == "learned_omega_learned_coefficients":
            baseline_mse = trajectory_mse
        rows.append(
            {
                "model": model_name,
                "combination": combination_name,
                "trajectory_mse": trajectory_mse,
                "coefficient_mse": mean_squared_error(
                    coefficients,
                    true_coefficients,
                ),
                "omega_mse": mean_squared_error(omegas, true_omega),
                "omega_correlation": _omega_corr(omegas, true_omega),
            },
        )
    if baseline_mse is not None:
        for row in rows:
            row["trajectory_mse_fraction_of_learned_both"] = (
                row["trajectory_mse"] / baseline_mse
                if baseline_mse > 0
                else float("nan")
            )
            row["trajectory_mse_improvement_vs_learned_both"] = (
                baseline_mse / row["trajectory_mse"]
                if row["trajectory_mse"] > 0
                else float("inf")
            )
    return rows, learned_coefficients.detach(), learned_omega.detach()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _save_bar_plot(rows: list[dict[str, Any]]) -> None:
    model_names = []
    for row in rows:
        if row["model"] not in model_names:
            model_names.append(row["model"])
    combinations = [
        "learned_omega_learned_coefficients",
        "true_omega_learned_coefficients",
        "learned_omega_true_coefficients",
        "true_omega_true_coefficients",
    ]
    fig, axes = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 4.8))
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names, strict=True):
        model_rows = [row for row in rows if row["model"] == model_name]
        values = [
            next(row for row in model_rows if row["combination"] == combo)[
                "trajectory_mse"
            ]
            for combo in combinations
        ]
        ax.bar(range(len(combinations)), values)
        ax.set_yscale("log")
        ax.set_xticks(range(len(combinations)))
        ax.set_xticklabels(
            [
                "learned omega\nlearned coeffs",
                "true omega\nlearned coeffs",
                "learned omega\ntrue coeffs",
                "true omega\ntrue coeffs",
            ],
            rotation=15,
            ha="right",
        )
        ax.set_ylabel("10-period trajectory MSE")
        ax.set_title(model_name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_mse_decomposition.png")
    plt.close(fig)


def _save_html(rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys()) if rows else []
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Error Decomposition</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Final 10-Period Error Decomposition</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <section>
    <h2>Trajectory MSE Decomposition</h2>
    <img src="trajectory_mse_decomposition.png" alt="trajectory MSE decomposition">
  </section>
</body>
</html>
"""
    (OUTPUT_DIR / "error_decomposition_summary.html").write_text(html, encoding="utf-8")


def run_error_decomposition() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=FINAL_HORIZON_PERIODS,
    )

    residual_model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    residual_model.load_state_dict(_load_state_dict(RESIDUAL_PARAMETERS))

    gru_model = SHOTwoStateGRUWindowModel(hidden_size=GRU_HIDDEN_SIZE)
    gru_model.load_state_dict(_load_state_dict(GRU_PARAMETERS))

    rows = []
    for model_name, model in (
        ("feedforward_residual", residual_model),
        ("gru_hidden_16", gru_model),
    ):
        model_rows, _, _ = _evaluate_model_outputs(model_name, model, test_data)
        rows.extend(model_rows)

    _write_csv(OUTPUT_DIR / "error_decomposition_summary.csv", rows)
    _write_json = lambda path, data: path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    _write_json(OUTPUT_DIR / "error_decomposition_summary.json", {"rows": rows})
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "stage": "stage1B",
            "diagnostic": "final_10_period_error_decomposition",
            "final_horizon_periods": FINAL_HORIZON_PERIODS,
            "feedforward_residual_parameters": str(RESIDUAL_PARAMETERS),
            "gru_parameters": str(GRU_PARAMETERS),
            "num_test": NUM_TEST,
            "num_time_steps": NUM_TIME_STEPS,
            "seed": SEED,
        },
    )
    _save_bar_plot(rows)
    _save_html(rows)
    return rows


def main() -> None:
    rows = run_error_decomposition()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
