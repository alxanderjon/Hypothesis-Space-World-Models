"""Stage 1B two-state identifiability diagnostics.

This script does not train models. It rebuilds the deterministic Stage 1B
train/test data, computes the analytic two-state omega estimate, and compares
saved learned model omega errors against the two-state denominator size.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn

from hypothesis_space_world_models.models import SHOTwoStateTrajectoryOmegaLinearModel

from experiments.sho.run_stage1b_two_state_omega import (
    CURRICULUM_HORIZONS,
    DELTA_T,
    LINEAR_OUTPUT_DIR,
    MAX_EPOCHS_PER_STAGE,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    RESIDUAL_OUTPUT_DIR,
    RUN_DATE,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    STOPPING_MIN_DELTA,
    STOPPING_PATIENCE,
    _build_data_from_base,
    _make_base_conditions,
    _metrics,
    _residual_variants,
    analytic_two_state_omega,
)


DIAGNOSTIC_OUTPUT_DIR = Path("outputs/sho/stage1B/identifiability_diagnostics")
DENOMINATOR_EPS = 1e-10


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _denominator_and_ratio(two_state_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = two_state_inputs[:, 0]
    v0 = two_state_inputs[:, 1]
    x1 = two_state_inputs[:, 2]
    v1 = two_state_inputs[:, 3]
    denominator = x0.square() - x1.square()
    ratio = (v1.square() - v0.square()) / denominator
    return denominator, ratio


def _compute_analytic_diagnostics(
    split_name: str,
    two_state_inputs: torch.Tensor,
    true_omega: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    denominator, ratio = _denominator_and_ratio(two_state_inputs)
    omega_analytic, valid = analytic_two_state_omega(
        two_state_inputs,
        denominator_eps=DENOMINATOR_EPS,
    )
    finite_ratio = torch.isfinite(ratio)
    nonpositive_ratio = finite_ratio & (ratio <= 0.0)
    small_denominator = denominator.abs() < DENOMINATOR_EPS
    valid_error = omega_analytic[valid] - true_omega[valid]
    diagnostics = {
        "split": split_name,
        "num_trajectories": int(true_omega.numel()),
        "denominator_eps": DENOMINATOR_EPS,
        "fraction_small_denominator": small_denominator.float().mean().item(),
        "fraction_nonpositive_ratio": nonpositive_ratio.float().mean().item(),
        "fraction_valid_analytic_omega": valid.float().mean().item(),
        "omega_analytic_mse": valid_error.square().mean().item()
        if valid_error.numel() > 0
        else float("nan"),
        "omega_analytic_mean_abs_error": valid_error.abs().mean().item()
        if valid_error.numel() > 0
        else float("nan"),
        "omega_analytic_max_abs_error": valid_error.abs().max().item()
        if valid_error.numel() > 0
        else float("nan"),
        "min_abs_denominator": denominator.abs().min().item(),
        "median_abs_denominator": denominator.abs().median().item(),
        "max_abs_denominator": denominator.abs().max().item(),
    }
    tensors = {
        "denominator": denominator.detach().cpu(),
        "ratio": ratio.detach().cpu(),
        "valid": valid.detach().cpu(),
        "omega_analytic": omega_analytic.detach().cpu(),
        "true_omega": true_omega.detach().cpu(),
        "omega_error": (omega_analytic - true_omega).detach().cpu(),
    }
    return diagnostics, tensors


def _save_analytic_plots(output_dir: Path, tensors_by_split: dict[str, dict[str, torch.Tensor]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    for split_name, tensors in tensors_by_split.items():
        valid = tensors["valid"]
        plt.scatter(
            tensors["true_omega"][valid],
            tensors["omega_analytic"][valid],
            s=16,
            alpha=0.65,
            label=split_name,
        )
    plt.plot([1.5, 2.5], [1.5, 2.5], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("analytic omega")
    plt.title("Stage 1B Analytic Two-State Omega: Predicted vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_analytic_predicted_vs_true.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for split_name, tensors in tensors_by_split.items():
        plt.hist(
            tensors["denominator"],
            bins=60,
            alpha=0.55,
            label=split_name,
        )
    plt.axvline(-DENOMINATOR_EPS, color="k", linestyle="--", linewidth=1)
    plt.axvline(DENOMINATOR_EPS, color="k", linestyle="--", linewidth=1)
    plt.xlabel("x0^2 - x1^2")
    plt.ylabel("count")
    plt.title("Stage 1B Two-State Denominator Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "denominator_histogram.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for split_name, tensors in tensors_by_split.items():
        valid = tensors["valid"]
        plt.hist(
            tensors["omega_error"][valid],
            bins=60,
            alpha=0.55,
            label=split_name,
        )
    plt.xlabel("omega_analytic - omega_true")
    plt.ylabel("count")
    plt.title("Stage 1B Analytic Omega Error Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "analytic_omega_error_histogram.png")
    plt.close()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "run_name": "linear_two_state_omega",
            "output_dir": LINEAR_OUTPUT_DIR / f"{RUN_DATE}_linear_two_state_omega",
            "model": SHOTwoStateTrajectoryOmegaLinearModel(initial_omega_hat=2.0),
        },
    ]
    for variant in _residual_variants():
        specs.append(
            {
                "run_name": variant["run_name"],
                "output_dir": RESIDUAL_OUTPUT_DIR / f"{RUN_DATE}_{variant['run_name']}",
                "model": SHOTwoStateTrajectoryOmegaLinearResidualModel(
                    hidden_dims=list(variant["hidden_dims"]),
                    initial_omega_hat=2.0,
                ),
            },
        )
    return specs


def _evaluate_saved_model(
    model: nn.Module,
    output_dir: Path,
    test_data: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    state_path = output_dir / "learned_parameters.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing learned parameters: {state_path}")
    model.load_state_dict(_load_state_dict(state_path))
    model.eval()
    with torch.no_grad():
        coefficients, trajectories, omega_hat = model(
            test_data.two_state_inputs,
            test_data.times,
        )
    metrics = _metrics(
        coefficients=coefficients,
        trajectories=trajectories,
        omega_hat=omega_hat,
        data=test_data,
    )
    return metrics, omega_hat.detach().cpu()


def _binned_error_by_denominator(
    denominator_abs: torch.Tensor,
    model_abs_error: torch.Tensor,
    *,
    num_bins: int = 8,
) -> list[dict[str, Any]]:
    order = torch.argsort(denominator_abs)
    bins = torch.tensor_split(order, num_bins)
    rows = []
    for bin_index, indices in enumerate(bins, start=1):
        if indices.numel() == 0:
            continue
        bin_denominator = denominator_abs[indices]
        bin_error = model_abs_error[indices]
        rows.append(
            {
                "bin": bin_index,
                "count": int(indices.numel()),
                "min_abs_denominator": bin_denominator.min().item(),
                "median_abs_denominator": bin_denominator.median().item(),
                "max_abs_denominator": bin_denominator.max().item(),
                "mean_abs_omega_error": bin_error.mean().item(),
                "max_abs_omega_error": bin_error.max().item(),
            },
        )
    return rows


def _save_model_error_plots(
    output_dir: Path,
    run_name: str,
    denominator: torch.Tensor,
    true_omega: torch.Tensor,
    omega_hat: torch.Tensor,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    denominator_abs = denominator.abs()
    omega_error = omega_hat - true_omega
    abs_error = omega_error.abs()

    plt.figure(figsize=(7, 4.8))
    plt.scatter(denominator_abs, abs_error, s=18, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|x0^2 - x1^2|")
    plt.ylabel("|omega_hat - omega_true|")
    plt.title(f"Stage 1B {run_name}: Omega Error vs Denominator Size")
    plt.tight_layout()
    plt.savefig(output_dir / "model_omega_error_vs_denominator.png")
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.scatter(denominator_abs, omega_error, s=18, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("|x0^2 - x1^2|")
    plt.ylabel("omega_hat - omega_true")
    plt.title(f"Stage 1B {run_name}: Signed Omega Error vs Denominator Size")
    plt.tight_layout()
    plt.savefig(output_dir / "model_signed_omega_error_vs_denominator.png")
    plt.close()

    return _binned_error_by_denominator(denominator_abs, abs_error)


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


def _write_html_table(path: Path, title: str, rows: list[dict[str, Any]], plots: list[str]) -> None:
    columns = list(rows[0].keys()) if rows else []
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in plots
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run_identifiability_diagnostics() -> dict[str, Any]:
    DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    train_data = _build_data_from_base(
        train_initial_conditions,
        train_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    )
    test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    )

    analytic_rows = []
    tensors_by_split = {}
    for split_name, data in (("train", train_data), ("test", test_data)):
        diagnostics, tensors = _compute_analytic_diagnostics(
            split_name,
            data.two_state_inputs,
            data.omegas,
        )
        analytic_rows.append(diagnostics)
        tensors_by_split[split_name] = tensors
    _write_csv(DIAGNOSTIC_OUTPUT_DIR / "analytic_identifiability_summary.csv", analytic_rows)
    _write_json(
        DIAGNOSTIC_OUTPUT_DIR / "analytic_identifiability_summary.json",
        {"splits": analytic_rows},
    )
    _save_analytic_plots(DIAGNOSTIC_OUTPUT_DIR, tensors_by_split)

    model_rows = []
    model_bin_rows = []
    test_denominator = tensors_by_split["test"]["denominator"]
    true_omega = tensors_by_split["test"]["true_omega"]
    for spec in _model_specs():
        output_dir = Path(spec["output_dir"])
        if not (output_dir / "learned_parameters.pt").exists():
            continue
        metrics, omega_hat = _evaluate_saved_model(spec["model"], output_dir, test_data)
        model_diag_dir = output_dir / "identifiability_diagnostics"
        binned_rows = _save_model_error_plots(
            model_diag_dir,
            str(spec["run_name"]),
            test_denominator,
            true_omega,
            omega_hat,
        )
        for row in binned_rows:
            model_bin_rows.append({"run_name": spec["run_name"], **row})
        top_error_count = max(1, int(0.1 * true_omega.numel()))
        abs_error = (omega_hat - true_omega).abs()
        top_error_indices = torch.argsort(abs_error, descending=True)[:top_error_count]
        denominator_abs = test_denominator.abs()
        correlation = torch.corrcoef(
            torch.stack(
                (
                    torch.log10(torch.clamp(denominator_abs, min=1e-12)),
                    torch.log10(torch.clamp(abs_error, min=1e-12)),
                ),
            ),
        )[0, 1].item()
        model_rows.append(
            {
                "run_name": spec["run_name"],
                "trajectory_mse": metrics["trajectory_mse"],
                "coefficient_mse": metrics["coefficient_mse"],
                "omega_mse": metrics["omega_mse"],
                "mean_abs_omega_error": metrics["mean_abs_omega_error"],
                "max_omega_error": metrics["max_omega_error"],
                "omega_correlation": metrics["omega_correlation"],
                "log_abs_denominator_vs_log_abs_omega_error_corr": correlation,
                "median_abs_denominator_all_test": denominator_abs.median().item(),
                "median_abs_denominator_top_10pct_model_error": denominator_abs[
                    top_error_indices
                ].median().item(),
                "min_abs_denominator_top_10pct_model_error": denominator_abs[
                    top_error_indices
                ].min().item(),
                "output_dir": str(output_dir),
                "diagnostic_plots": str(model_diag_dir),
            },
        )

    _write_csv(DIAGNOSTIC_OUTPUT_DIR / "model_error_vs_denominator_summary.csv", model_rows)
    _write_csv(DIAGNOSTIC_OUTPUT_DIR / "model_error_by_denominator_bin.csv", model_bin_rows)
    _write_json(
        DIAGNOSTIC_OUTPUT_DIR / "model_error_vs_denominator_summary.json",
        {"models": model_rows, "bins": model_bin_rows},
    )
    _write_html_table(
        DIAGNOSTIC_OUTPUT_DIR / "index.html",
        "Stage 1B Identifiability Diagnostics",
        analytic_rows + model_rows,
        [
            "omega_analytic_predicted_vs_true.png",
            "denominator_histogram.png",
            "analytic_omega_error_histogram.png",
        ],
    )
    _write_json(
        DIAGNOSTIC_OUTPUT_DIR / "setup.json",
        {
            "stage": "stage1B",
            "diagnostic": "two_state_identifiability",
            "input": "(x0, v0, x1, v1)",
            "delta_t": DELTA_T,
            "denominator_eps": DENOMINATOR_EPS,
            "num_train": NUM_TRAIN,
            "num_test": NUM_TEST,
            "num_time_steps": NUM_TIME_STEPS,
            "final_curriculum_horizon_periods": CURRICULUM_HORIZONS[-1],
            "stopping_min_delta": STOPPING_MIN_DELTA,
            "stopping_patience": STOPPING_PATIENCE,
            "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
            "note": "No training is performed by this diagnostic.",
        },
    )
    return {"analytic": analytic_rows, "models": model_rows}


def main() -> None:
    diagnostics = run_identifiability_diagnostics()
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
