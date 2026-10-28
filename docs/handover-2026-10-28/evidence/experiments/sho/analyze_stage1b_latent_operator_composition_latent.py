"""Analyze learned latent coordinates from the Stage 1B composition diagnostic."""

from __future__ import annotations

import csv
import json
import os
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
import torch

from experiments.sho.run_stage1b_latent_operator_composition import (
    DIAGNOSTIC_NUM_TEST,
    LATENT_DIM,
    OUTPUT_DIR,
    LatentOperatorCompositionModel,
    _fixed_eval_chunks,
    _make_position_sequences,
)
from experiments.sho.run_stage1b_two_state_omega import SEED


ANALYSIS_DIR = OUTPUT_DIR / "latent_analysis"


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


def _load_latents() -> tuple[np.ndarray, np.ndarray]:
    model = LatentOperatorCompositionModel()
    state = torch.load(OUTPUT_DIR / "learned_parameters.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    sequences, omegas = _make_position_sequences(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    chunks = _fixed_eval_chunks(sequences)
    with torch.no_grad():
        z = model.encode(chunks[:, :20])
    return z.numpy(), omegas.numpy()


def _linear_r2(features: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, float]:
    model = LinearRegression()
    model.fit(features, target)
    return float(model.score(features, target)), model.coef_.copy(), float(model.intercept_)


def _save_per_dim_scatter(path: Path, z: np.ndarray, omega: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.reshape(-1)
    for dim in range(LATENT_DIM):
        axes[dim].scatter(z[:, dim], omega, s=24, alpha=0.8)
        axes[dim].set_xlabel(f"z[{dim}]")
        axes[dim].set_ylabel("omega")
        axes[dim].set_title(f"omega vs z[{dim}]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_pca_plot(path: Path, pca_embedding: np.ndarray, omega: np.ndarray) -> None:
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(
        pca_embedding[:, 0],
        pca_embedding[:, 1],
        c=omega,
        cmap="viridis",
        s=32,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of learned latent, colored by true omega")
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_umap_plot_or_placeholder(path: Path, z: np.ndarray, omega: np.ndarray) -> str:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        plt.figure(figsize=(7, 4.5))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "UMAP unavailable: package 'umap-learn' is not installed in NNLearning.",
            ha="center",
            va="center",
            wrap=True,
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return "unavailable: umap-learn is not installed in NNLearning"

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(z)
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=omega, cmap="viridis", s=32)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of learned latent, colored by true omega")
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return "generated"


def _write_html(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    pca_summary: dict[str, Any],
) -> None:
    row_html = ""
    for row in rows:
        corr = f"{row['pearson_corr_omega']:.6f}"
        abs_corr = f"{row['abs_pearson_corr_omega']:.6f}"
        mi = f"{row['mutual_information']:.6f}"
        r2 = f"{row['linear_r2']:.6f}"
        row_html += (
            "<tr>"
            f"<td>{escape(str(row['latent_dim']))}</td>"
            f"<td>{escape(corr)}</td>"
            f"<td>{escape(abs_corr)}</td>"
            f"<td>{escape(mi)}</td>"
            f"<td>{escape(r2)}</td>"
            "</tr>"
        )
    summary_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    pca_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in pca_summary.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Latent Operator Composition Latent Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Latent Operator Composition Latent Analysis</h1>
  <h2>Summary</h2>
  <table>{summary_rows}</table>
  <h2>Per-Dimension Omega Association</h2>
  <table>
    <thead><tr><th>latent_dim</th><th>pearson_corr_omega</th><th>abs_pearson_corr_omega</th><th>mutual_information</th><th>linear_r2</th></tr></thead>
    <tbody>{row_html}</tbody>
  </table>
  <h2>PCA Summary</h2>
  <table>{pca_rows}</table>
  <section><h2>Per-Dimension Scatter</h2><img src="omega_vs_latent_dims.png" alt="Omega vs latent dimensions"></section>
  <section><h2>PCA Visualization</h2><img src="latent_pca_omega.png" alt="PCA latent visualization"></section>
  <section><h2>UMAP Visualization</h2><img src="latent_umap_omega.png" alt="UMAP latent visualization"></section>
</body>
</html>
"""
    (ANALYSIS_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    z, omega = _load_latents()

    rows: list[dict[str, Any]] = []
    for dim in range(z.shape[1]):
        corr = float(np.corrcoef(z[:, dim], omega)[0, 1])
        mi = float(mutual_info_regression(z[:, [dim]], omega, random_state=SEED)[0])
        r2, coef, intercept = _linear_r2(z[:, [dim]], omega)
        rows.append(
            {
                "latent_dim": f"z[{dim}]",
                "pearson_corr_omega": corr,
                "abs_pearson_corr_omega": abs(corr),
                "mutual_information": mi,
                "linear_r2": r2,
                "linear_coef": float(coef[0]),
                "linear_intercept": intercept,
            },
        )

    multi_r2, multi_coef, multi_intercept = _linear_r2(z, omega)
    standardized_z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    standardized_omega = (omega - omega.mean()) / (omega.std() + 1e-8)
    _, standardized_coef, _ = _linear_r2(standardized_z, standardized_omega)

    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(z)
    pc1_r2, pc1_coef, pc1_intercept = _linear_r2(pca_embedding[:, [0]], omega)
    pc2_r2, pc2_coef, pc2_intercept = _linear_r2(pca_embedding[:, [1]], omega)
    pc_both_r2, pc_both_coef, pc_both_intercept = _linear_r2(pca_embedding, omega)

    best_dim = max(rows, key=lambda row: row["linear_r2"])
    summary = {
        "num_test_trajectories": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "best_single_dimension_by_r2": best_dim["latent_dim"],
        "best_single_dimension_r2": best_dim["linear_r2"],
        "best_single_dimension_abs_corr": best_dim["abs_pearson_corr_omega"],
        "multi_dimensional_linear_r2": multi_r2,
        "multi_dimensional_coefficients": multi_coef.tolist(),
        "multi_dimensional_intercept": multi_intercept,
        "standardized_multi_dimensional_coefficients": standardized_coef.tolist(),
        "umap_status": _save_umap_plot_or_placeholder(
            ANALYSIS_DIR / "latent_umap_omega.png",
            z,
            omega,
        ),
    }
    pca_summary = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_linear_r2": pc1_r2,
        "pc1_linear_coef": pc1_coef.tolist(),
        "pc1_linear_intercept": pc1_intercept,
        "pc2_linear_r2": pc2_r2,
        "pc2_linear_coef": pc2_coef.tolist(),
        "pc2_linear_intercept": pc2_intercept,
        "pc1_pc2_linear_r2": pc_both_r2,
        "pc1_pc2_linear_coef": pc_both_coef.tolist(),
        "pc1_pc2_linear_intercept": pc_both_intercept,
    }

    _write_csv(ANALYSIS_DIR / "per_dimension_latent_omega_analysis.csv", rows)
    _write_json(
        ANALYSIS_DIR / "latent_omega_analysis.json",
        {
            "per_dimension": rows,
            "summary": summary,
            "pca": pca_summary,
        },
    )
    _save_per_dim_scatter(ANALYSIS_DIR / "omega_vs_latent_dims.png", z, omega)
    _save_pca_plot(ANALYSIS_DIR / "latent_pca_omega.png", pca_embedding, omega)
    _write_html(rows, summary, pca_summary)
    print(json.dumps({"analysis_dir": str(ANALYSIS_DIR), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
