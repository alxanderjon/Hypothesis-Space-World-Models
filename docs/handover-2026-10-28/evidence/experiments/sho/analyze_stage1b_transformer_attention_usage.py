"""Analyze attention usage for completed transformer dynamics encoder seed study."""

from __future__ import annotations

import csv
import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_transformer_dynamics_encoder_seed_stability import (
    DIAGNOSTIC_NUM_TEST,
    SEEDS,
    _make_position_data,
)


INPUT_DIR = Path("outputs/sho/stage1B/transformer_dynamics_encoder_seed_stability")
OUTPUT_DIR = Path("outputs/sho/stage1B/transformer_attention_usage_analysis")
NUM_WINDOWS = 8


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


def _load_seed_attention(seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = INPUT_DIR / f"seed_{seed}" / "transformer_attention_8_windows" / "latents_and_attention.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing attention artifact: {path}")
    data = torch.load(path, map_location="cpu")
    weights = data["attention_weights"].detach().cpu().numpy()
    _, omegas, _ = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=seed + 1000)
    return weights, omegas.detach().cpu().numpy()


def _attention_metrics(weights: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(weights, 1e-12, 1.0)
    max_weight = weights.max(axis=1)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    normalized_entropy = entropy / math.log(weights.shape[1])
    n_eff = 1.0 / np.square(weights).sum(axis=1)
    return {
        "max_weight": max_weight,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "n_eff": n_eff,
    }


def _summarize_seed(seed: int, weights: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "seed": seed,
        "num_trajectories": int(weights.shape[0]),
        "num_windows": int(weights.shape[1]),
        "mean_max_attention_weight": float(metrics["max_weight"].mean()),
        "std_max_attention_weight": float(metrics["max_weight"].std(ddof=0)),
        "mean_normalized_entropy": float(metrics["normalized_entropy"].mean()),
        "std_normalized_entropy": float(metrics["normalized_entropy"].std(ddof=0)),
        "mean_n_eff": float(metrics["n_eff"].mean()),
        "std_n_eff": float(metrics["n_eff"].std(ddof=0)),
        "min_n_eff": float(metrics["n_eff"].min()),
        "max_n_eff": float(metrics["n_eff"].max()),
        "mean_attention_by_window": [float(value) for value in weights.mean(axis=0)],
        "std_attention_by_window": [float(value) for value in weights.std(axis=0, ddof=0)],
    }


def _save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str, bins: int = 18) -> None:
    plt.figure(figsize=(6.6, 4.4))
    plt.hist(values, bins=bins, color="#3178a8", alpha=0.86, edgecolor="white")
    plt.xlabel(xlabel)
    plt.ylabel("trajectory count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_heatmap(path: Path, weights: np.ndarray, title: str) -> None:
    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(weights, aspect="auto", interpolation="nearest", cmap="magma")
    plt.xlabel("window index")
    plt.ylabel("test trajectory")
    plt.title(title)
    plt.colorbar(label="attention weight")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(x, y, s=28, alpha=0.82, color="#2f7d59")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_combined_histogram(path: Path, rows: list[dict[str, Any]], key: str, title: str, xlabel: str) -> None:
    plt.figure(figsize=(7.0, 4.6))
    for seed in SEEDS:
        values = np.array([row[key] for row in rows if row["seed"] == seed], dtype=float)
        plt.hist(values, bins=18, alpha=0.48, label=f"seed {seed}")
    plt.xlabel(xlabel)
    plt.ylabel("trajectory count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _format(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _write_html(seed_rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "seed",
        "mean_max_attention_weight",
        "std_max_attention_weight",
        "mean_normalized_entropy",
        "std_normalized_entropy",
        "mean_n_eff",
        "std_n_eff",
        "min_n_eff",
        "max_n_eff",
        "mean_attention_by_window",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in seed_rows
    )
    interpretation_html = "".join(f"<p>{escape(line)}</p>" for line in interpretation)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Transformer Attention Usage Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Transformer Attention Usage Analysis</h1>
  <p>Post-processing only. Uses completed transformer/attention dynamics encoder seed-study attention tensors on fixed test windows.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>Combined Histograms</h2>
  <img src="all_seeds_max_attention_weight_histogram.png">
  <img src="all_seeds_normalized_entropy_histogram.png">
  <img src="all_seeds_n_eff_histogram.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        weights, omegas = _load_seed_attention(seed)
        metrics = _attention_metrics(weights)
        seed_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_rows.append(_summarize_seed(seed, weights, metrics))
        for trajectory_index in range(weights.shape[0]):
            trajectory_rows.append(
                {
                    "seed": seed,
                    "trajectory_index": trajectory_index,
                    "omega": float(omegas[trajectory_index]),
                    "max_attention_weight": float(metrics["max_weight"][trajectory_index]),
                    "entropy": float(metrics["entropy"][trajectory_index]),
                    "normalized_entropy": float(metrics["normalized_entropy"][trajectory_index]),
                    "n_eff": float(metrics["n_eff"][trajectory_index]),
                    "attention_weights": [float(value) for value in weights[trajectory_index]],
                }
            )

        _save_histogram(seed_dir / "max_attention_weight_histogram.png", metrics["max_weight"], f"seed {seed}: max attention weight", "max_j alpha_j")
        _save_histogram(seed_dir / "normalized_entropy_histogram.png", metrics["normalized_entropy"], f"seed {seed}: normalized entropy", "H / log(8)")
        _save_histogram(seed_dir / "n_eff_histogram.png", metrics["n_eff"], f"seed {seed}: effective attended windows", "N_eff")
        _save_heatmap(seed_dir / "attention_heatmap.png", weights, f"seed {seed}: trajectory attention weights")
        _save_scatter(seed_dir / "n_eff_vs_omega.png", omegas, metrics["n_eff"], f"seed {seed}: N_eff vs omega", "omega", "N_eff")
        _save_scatter(seed_dir / "max_attention_weight_vs_omega.png", omegas, metrics["max_weight"], f"seed {seed}: max attention vs omega", "omega", "max_j alpha_j")

    _write_csv(OUTPUT_DIR / "summary.csv", seed_rows)
    _write_csv(OUTPUT_DIR / "per_trajectory_attention.csv", trajectory_rows)

    _save_combined_histogram(
        OUTPUT_DIR / "all_seeds_max_attention_weight_histogram.png",
        trajectory_rows,
        "max_attention_weight",
        "All seeds: max attention weight",
        "max_j alpha_j",
    )
    _save_combined_histogram(
        OUTPUT_DIR / "all_seeds_normalized_entropy_histogram.png",
        trajectory_rows,
        "normalized_entropy",
        "All seeds: normalized entropy",
        "H / log(8)",
    )
    _save_combined_histogram(
        OUTPUT_DIR / "all_seeds_n_eff_histogram.png",
        trajectory_rows,
        "n_eff",
        "All seeds: effective attended windows",
        "N_eff",
    )

    overall_n_eff = np.array([row["n_eff"] for row in trajectory_rows], dtype=float)
    overall_max = np.array([row["max_attention_weight"] for row in trajectory_rows], dtype=float)
    overall_entropy = np.array([row["normalized_entropy"] for row in trajectory_rows], dtype=float)
    interpretation = [
        f"Across all seeds, mean N_eff is {overall_n_eff.mean():.3f} out of {NUM_WINDOWS}, with min {overall_n_eff.min():.3f} and max {overall_n_eff.max():.3f}.",
        f"Across all seeds, mean max attention weight is {overall_max.mean():.3f}; a one-window selector would be close to 1.0.",
        f"Across all seeds, mean normalized entropy is {overall_entropy.mean():.3f}; uniform attention would be 1.0.",
    ]
    if overall_n_eff.mean() > 4.0 and overall_max.mean() < 0.5:
        interpretation.append("The attention encoder is genuinely integrating multiple windows per trajectory, not selecting one dominant window.")
    else:
        interpretation.append("The attention encoder shows meaningful concentration on a subset of windows; inspect the per-seed histograms for single-window selection.")

    _write_json(
        OUTPUT_DIR / "summary.json",
        {
            "seed_summaries": seed_rows,
            "interpretation": interpretation,
        },
    )
    _write_html(seed_rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "seed_summaries": seed_rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
