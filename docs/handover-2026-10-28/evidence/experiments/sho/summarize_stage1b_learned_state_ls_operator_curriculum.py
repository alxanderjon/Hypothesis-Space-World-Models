"""Summarize completed learned-state LS operator curriculum runs.

This companion is intentionally summarize-only: it does not train or resume
unfinished h_dim runs.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import DIAGNOSTIC_NUM_TEST
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import (
    CURRICULUM_STAGES,
    H_DIMS,
    OBS_PREFIX_TRANSITIONS,
    OUTPUT_DIR,
    RIDGE,
    LearnedStateLSKoopman,
    _evaluate_learned_state_model,
    _evaluate_neural_generated_k,
    _evaluate_raw_ls,
    _loss_and_parts,
    _make_prefix_future_data,
)


def _completed_model(h_dim: int) -> LearnedStateLSKoopman | None:
    run_dir = OUTPUT_DIR / f"learned_state_h{h_dim}"
    checkpoint = run_dir / "learned_parameters.pt"
    metrics = run_dir / "metrics.json"
    if not checkpoint.exists() or not metrics.exists():
        return None
    model = LearnedStateLSKoopman(h_dim).to(dtype=torch.float64)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def _load_completed_rows(h_dim: int) -> list[dict[str, Any]]:
    metrics = OUTPUT_DIR / f"learned_state_h{h_dim}" / "metrics.json"
    if not metrics.exists():
        return []
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return list(payload.get("history", []))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _plot_history(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str) -> None:
    stage_order = {str(stage): index for index, stage in enumerate(CURRICULUM_STAGES)}
    learned = [
        row
        for row in rows
        if row.get("model") == "learned_state_ls_operator" and "curriculum_stage" in row and metric in row
    ]
    if not learned:
        return
    plt.figure(figsize=(8.0, 5.0))
    for h_dim in sorted({int(row["h_dim"]) for row in learned}):
        selected = [row for row in learned if int(row["h_dim"]) == h_dim]
        best_by_stage: dict[str, dict[str, Any]] = {}
        for row in selected:
            stage = str(row["curriculum_stage"])
            current = best_by_stage.get(stage)
            if current is None or float(row[metric]) < float(current[metric]):
                best_by_stage[stage] = row
        ordered = sorted(best_by_stage.values(), key=lambda row: stage_order.get(str(row["curriculum_stage"]), 10**9))
        plt.plot(
            [stage_order[str(row["curriculum_stage"])] for row in ordered],
            [float(row[metric]) for row in ordered],
            marker="o",
            linewidth=1.4,
            label=f"h_dim={h_dim}",
        )
    plt.yscale("log")
    plt.xticks(range(len(CURRICULUM_STAGES)), [str(stage) for stage in CURRICULUM_STAGES], rotation=30, ha="right")
    plt.xlabel("curriculum rollout stage")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs curriculum stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_examples(model: LearnedStateLSKoopman, positions: torch.Tensor) -> None:
    with torch.no_grad():
        _, _, predicted, true_states = _loss_and_parts(model, positions[:4], "full_10_period")
    steps = np.arange(true_states.shape[1])
    plt.figure(figsize=(9.0, 5.2))
    for index in range(true_states.shape[0]):
        plt.plot(steps, true_states[index, :, 1].detach().cpu().numpy(), linestyle="--", alpha=0.7)
        plt.plot(steps, predicted[index, :, 1].detach().cpu().numpy(), alpha=0.9, label=f"traj {index}")
    plt.xlabel("rollout step after observed prefix")
    plt.ylabel("position")
    plt.title("learned-state LS rollout examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_examples.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9.0, 5.2))
    for index in range(true_states.shape[0]):
        error = torch.abs(predicted[index, :, 1] - true_states[index, :, 1]).detach().cpu().numpy()
        plt.plot(steps, error, label=f"traj {index}")
    plt.yscale("log")
    plt.xlabel("rollout step after observed prefix")
    plt.ylabel("|position error|")
    plt.title("learned-state LS error growth examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_growth_examples.png", dpi=160)
    plt.close()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _write_html(rows: list[dict[str, Any]], completed_h_dims: list[int], incomplete_h_dims: list[int]) -> None:
    final_rows = [
        row
        for row in rows
        if row.get("stage") == "full_10_period" and "curriculum_stage" not in row
    ]
    table_columns = [
        "model",
        "h_dim",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "max_abs_trajectory_error",
        "final_time_abs_trajectory_error",
        "best_epoch",
        "parameter_count",
        "state_autoencoder_mse",
        "latent_transition_residual",
        "ls_condition_number_mean",
        "error_growth_label",
    ]
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in table_columns) + "</tr>"
        for row in final_rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Learned-State LS Operator Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .note {{ background: #f7f9fc; border-left: 4px solid #5577aa; padding: 12px 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>Learned-State Least-Squares Operator Curriculum</h1>
  <p>This diagnostic learns <code>H(s)</code> and <code>D(h)</code>, but identifies each trajectory operator by least squares in latent state space from a fixed observed prefix. The rollout loss compares decoded future rollouts against the analytic position trajectory.</p>
  <div class="note">
    Completed latent dimensions: {escape(str(completed_h_dims))}. Incomplete latent dimensions not included in final comparisons: {escape(str(incomplete_h_dims))}.
    Operator identification uses {OBS_PREFIX_TRANSITIONS} prefix transitions and ridge <code>{RIDGE:g}</code>.
  </div>
  <h2>Final Full 10-Period Comparison</h2>
  <table>
    <thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in table_columns)}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Diagnostics</h2>
  <div class="grid">
    <img src="rollout_mse_vs_curriculum_stage.png" alt="rollout MSE">
    <img src="one_step_mse_vs_curriculum_stage.png" alt="one-step MSE">
    <img src="max_error_vs_curriculum_stage.png" alt="max error">
    <img src="rollout_examples.png" alt="rollout examples">
    <img src="error_growth_examples.png" alt="error growth examples">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_positions, test_omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=1000)

    rows: list[dict[str, Any]] = [
        _evaluate_raw_ls(test_positions, test_omegas),
        _evaluate_neural_generated_k(test_positions),
    ]
    completed_h_dims: list[int] = []
    incomplete_h_dims: list[int] = []
    last_model: LearnedStateLSKoopman | None = None

    for h_dim in H_DIMS:
        model = _completed_model(h_dim)
        if model is None:
            incomplete_h_dims.append(h_dim)
            continue
        completed_h_dims.append(h_dim)
        last_model = model
        rows.extend(_load_completed_rows(h_dim))
        metrics = json.loads((OUTPUT_DIR / f"learned_state_h{h_dim}" / "metrics.json").read_text(encoding="utf-8"))
        best_epoch = int(metrics.get("final", {}).get("best_epoch", 0))
        rows.append(_evaluate_learned_state_model(model, test_positions, "full_10_period", best_epoch=best_epoch))

    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "completed_h_dims": completed_h_dims,
                "incomplete_h_dims": incomplete_h_dims,
                "obs_prefix_transitions": OBS_PREFIX_TRANSITIONS,
                "ridge": RIDGE,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_html(rows, completed_h_dims, incomplete_h_dims)
    _plot_history(rows, "rollout_mse", "rollout_mse_vs_curriculum_stage.png", "rollout MSE")
    _plot_history(rows, "one_step_mse", "one_step_mse_vs_curriculum_stage.png", "one-step MSE")
    _plot_history(rows, "max_abs_trajectory_error", "max_error_vs_curriculum_stage.png", "max absolute error")
    if last_model is not None:
        _plot_examples(last_model, test_positions)


if __name__ == "__main__":
    main()
