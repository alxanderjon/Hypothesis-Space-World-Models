"""Soft plausible-region Network 1 diagnostic for unconstrained K(a,b)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch

import experiments.sho.run_stage1b_direct2d_learned_partition as direct
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as data_source
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    UPPER,
    VALIDATION_DATA_SEED,
    _posthoc_reference,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    os.environ.get(
        "SOFT2D_OUTPUT_DIR",
        "outputs/sho/stage1B/soft_region_2d_search",
    )
)
TRAIN_N = int(os.environ.get("SOFT2D_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("SOFT2D_VALIDATION_N", "512"))
TEST_N = int(os.environ.get("SOFT2D_TEST_N", "512"))
HORIZONS = tuple(
    int(value) for value in os.environ.get("SOFT2D_HORIZONS", "20,40,80").split(",")
)
UPDATE_BUDGETS = tuple(
    int(value) for value in os.environ.get("SOFT2D_UPDATES", "800,600,400").split(",")
)
if len(HORIZONS) != len(UPDATE_BUDGETS):
    raise ValueError("Each soft-region horizon requires an update budget")
BATCH_SIZE = int(os.environ.get("SOFT2D_BATCH_SIZE", "32"))
EVAL_INTERVAL = int(os.environ.get("SOFT2D_EVAL_INTERVAL", "50"))
PATIENCE = int(os.environ.get("SOFT2D_PATIENCE", "250"))
PER_QUERY_GRID_RESOLUTION = int(os.environ.get("SOFT2D_QUERY_GRID", "24"))
SUMMARY_GRID_RESOLUTION = int(os.environ.get("SOFT2D_SUMMARY_GRID", "64"))
BOUNDARY_FRACTION = 0.05
MODEL_SEED = SEED + 2_350_000
LOG_EPSILON = 1e-30
DOMAIN_AREA = DOMAIN_WIDTH**2
DOMAIN_DIAGONAL = (2.0**0.5) * DOMAIN_WIDTH

# The imported controller/trainer retains exactly the preceding experiment's
# architecture and optimization setup. These assignments only make compact
# smoke runs configurable without introducing another selector variant.
direct.BATCH_SIZE = BATCH_SIZE
direct.EVAL_INTERVAL = EVAL_INTERVAL
direct.PATIENCE = PATIENCE


@dataclass
class SoftSupport:
    centers: torch.Tensor
    radii: torch.Tensor
    importance: torch.Tensor
    spacing: torch.Tensor


@dataclass
class SoftDiagnostics:
    covered: torch.Tensor
    area: torch.Tensor
    overlap_area: torch.Tensor
    disconnected_components: torch.Tensor
    nearest_parameter_error: torch.Tensor
    best_prefix_loss: torch.Tensor
    boundary_associated_miss: torch.Tensor


def _soft_support(
    evaluated_theta: torch.Tensor,
    evaluated_loss: torch.Tensor,
) -> SoftSupport:
    """Build overlapping supports without labels or partition membership."""
    batch_size, count, _ = evaluated_theta.shape
    if count == 1:
        spacing = torch.full((batch_size, 1), DOMAIN_DIAGONAL, dtype=DTYPE)
    else:
        pairwise = torch.cdist(evaluated_theta, evaluated_theta)
        diagonal = torch.eye(count, dtype=torch.bool)[None, :, :]
        pairwise = pairwise.masked_fill(diagonal, float("inf"))
        neighbor_rank = min(2, count - 1)
        spacing = torch.topk(
            pairwise, k=neighbor_rank, dim=2, largest=False
        ).values[:, :, -1]

    log_loss = torch.log(evaluated_loss + LOG_EPSILON)
    relative_log_loss = log_loss - log_loss.min(dim=1, keepdim=True).values
    temperature = torch.quantile(relative_log_loss, 0.5, dim=1).clamp_min(0.5)
    importance = torch.exp(-relative_log_loss / temperature[:, None])
    relative_sparsity = spacing / spacing.max(dim=1, keepdim=True).values.clamp_min(
        1e-12
    )

    # Every tested hypothesis retains support. Low loss and sparse local
    # sampling enlarge it; denser sampling automatically contracts it.
    radii = spacing * (
        0.50 + 0.50 * importance + 0.25 * relative_sparsity
    )
    radii = radii.clamp(min=DOMAIN_WIDTH / (32.0 * max(count, 1)), max=DOMAIN_DIAGONAL)
    return SoftSupport(evaluated_theta, radii, importance, spacing)


def _domain_grid(resolution: int) -> torch.Tensor:
    step = DOMAIN_WIDTH / resolution
    axis = torch.linspace(
        LOWER + 0.5 * step,
        UPPER - 0.5 * step,
        resolution,
        dtype=DTYPE,
    )
    aa, bb = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((aa.reshape(-1), bb.reshape(-1)), dim=1)


def _union_area_and_overlap(
    support: SoftSupport,
    grid_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = _domain_grid(grid_resolution)
    area_parts: list[torch.Tensor] = []
    overlap_parts: list[torch.Tensor] = []
    chunk_size = 8 if grid_resolution >= 48 else 32
    for start in range(0, support.centers.shape[0], chunk_size):
        centers = support.centers[start : start + chunk_size]
        radii = support.radii[start : start + chunk_size]
        squared_distance = torch.sum(
            (grid[None, :, None, :] - centers[:, None, :, :]).square(), dim=3
        )
        multiplicity = (
            squared_distance <= radii[:, None, :].square()
        ).sum(dim=2)
        area_parts.append((multiplicity > 0).to(DTYPE).mean(dim=1) * DOMAIN_AREA)
        overlap_parts.append(
            (multiplicity - 1).clamp_min(0).to(DTYPE).mean(dim=1) * DOMAIN_AREA
        )
    return torch.cat(area_parts), torch.cat(overlap_parts)


def _component_counts(support: SoftSupport) -> torch.Tensor:
    pairwise = torch.cdist(support.centers, support.centers)
    adjacency = pairwise <= (
        support.radii[:, :, None] + support.radii[:, None, :]
    )
    counts: list[int] = []
    for batch_index in range(adjacency.shape[0]):
        graph = adjacency[batch_index]
        remaining = set(range(graph.shape[0]))
        components = 0
        while remaining:
            components += 1
            frontier = [remaining.pop()]
            while frontier:
                node = frontier.pop()
                neighbors = torch.nonzero(graph[node], as_tuple=False).flatten().tolist()
                newly_reached = [neighbor for neighbor in neighbors if neighbor in remaining]
                for neighbor in newly_reached:
                    remaining.remove(neighbor)
                frontier.extend(newly_reached)
        counts.append(components)
    return torch.tensor(counts, dtype=torch.long)


def _boundary_association(
    lower: torch.Tensor,
    upper: torch.Tensor,
    reference: torch.Tensor,
    covered: torch.Tensor,
) -> torch.Tensor:
    contains = (
        (reference[:, None, :] >= lower - 1e-14)
        & (reference[:, None, :] <= upper + 1e-14)
    ).all(dim=2)
    edge_distance = torch.minimum(
        reference[:, None, :] - lower,
        upper - reference[:, None, :],
    ).min(dim=2).values
    normalized = edge_distance / (upper - lower).min(dim=2).values.clamp_min(1e-15)
    normalized = torch.where(
        contains, normalized, torch.full_like(normalized, float("inf"))
    )
    near_boundary = normalized.min(dim=1).values <= BOUNDARY_FRACTION
    return (~covered) & near_boundary


@torch.no_grad()
def _soft_diagnostics(
    rollout: direct.Direct2DRollout,
    reference: torch.Tensor,
    query_number: int,
    grid_resolution: int,
) -> SoftDiagnostics:
    index = query_number - 1
    theta = rollout.query_theta[:, :query_number]
    loss = rollout.query_loss[:, :query_number]
    support = _soft_support(theta, loss)
    squared_reference_distance = torch.sum(
        (reference[:, None, :] - support.centers).square(), dim=2
    )
    covered = (
        squared_reference_distance <= support.radii.square()
    ).any(dim=1)
    nearest_error = torch.sqrt(squared_reference_distance.min(dim=1).values)
    area, overlap_area = _union_area_and_overlap(support, grid_resolution)
    components = _component_counts(support)
    boundary_miss = _boundary_association(
        rollout.partition_lower[index],
        rollout.partition_upper[index],
        reference,
        covered,
    )
    return SoftDiagnostics(
        covered=covered,
        area=area,
        overlap_area=overlap_area,
        disconnected_components=components,
        nearest_parameter_error=nearest_error,
        best_prefix_loss=loss.min(dim=1).values,
        boundary_associated_miss=boundary_miss,
    )


def _metrics_row(
    trained_horizon: int,
    query_number: int,
    diagnostics: SoftDiagnostics,
    rollout: direct.Direct2DRollout,
) -> dict[str, Any]:
    failures = int((~diagnostics.covered).sum().item())
    boundary_failures = int(diagnostics.boundary_associated_miss.sum().item())
    overlap_ratio = diagnostics.overlap_area / diagnostics.area.clamp_min(1e-15)
    return {
        "trained_horizon": trained_horizon,
        "evaluations": query_number,
        "coverage": float(diagnostics.covered.to(DTYPE).mean().item()),
        "covered_count": int(diagnostics.covered.sum().item()),
        "median_total_plausible_area": float(torch.median(diagnostics.area).item()),
        "mean_total_plausible_area": float(diagnostics.area.mean().item()),
        "worst_total_plausible_area": float(diagnostics.area.max().item()),
        "median_nearest_parameter_error_posthoc": float(
            torch.median(diagnostics.nearest_parameter_error).item()
        ),
        "worst_nearest_parameter_error_posthoc": float(
            diagnostics.nearest_parameter_error.max().item()
        ),
        "median_best_prefix_loss": float(
            torch.median(diagnostics.best_prefix_loss).item()
        ),
        "worst_best_prefix_loss": float(diagnostics.best_prefix_loss.max().item()),
        "median_disconnected_neighborhoods": float(
            torch.median(diagnostics.disconnected_components.to(DTYPE)).item()
        ),
        "mean_disconnected_neighborhoods": float(
            diagnostics.disconnected_components.to(DTYPE).mean().item()
        ),
        "worst_disconnected_neighborhoods": int(
            diagnostics.disconnected_components.max().item()
        ),
        "median_overlap_area": float(torch.median(diagnostics.overlap_area).item()),
        "mean_overlap_area": float(diagnostics.overlap_area.mean().item()),
        "worst_overlap_area": float(diagnostics.overlap_area.max().item()),
        "median_excess_overlap_to_union_ratio": float(
            torch.median(overlap_ratio).item()
        ),
        "exploration_fraction_cumulative": float(
            rollout.exploration_action[:, :query_number].to(DTYPE).mean().item()
        ),
        "exploitation_fraction_cumulative": float(
            (~rollout.exploration_action[:, :query_number]).to(DTYPE).mean().item()
        ),
        "bookkeeping_boundary_miss_count": boundary_failures,
        "bookkeeping_boundary_fraction_among_misses": float(
            boundary_failures / max(failures, 1)
        ),
        "area_grid_resolution": 0,
    }


def _per_query_rows(
    trained_horizon: int,
    rollout: direct.Direct2DRollout,
    reference: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_number in range(1, trained_horizon + 1):
        diagnostics = _soft_diagnostics(
            rollout,
            reference,
            query_number,
            PER_QUERY_GRID_RESOLUTION,
        )
        row = _metrics_row(trained_horizon, query_number, diagnostics, rollout)
        row["area_grid_resolution"] = PER_QUERY_GRID_RESOLUTION
        rows.append(row)
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for horizon in sorted({int(row["trained_horizon"]) for row in rows}):
        subset = [row for row in rows if int(row["trained_horizon"]) == horizon]
        evaluations = [int(row["evaluations"]) for row in subset]
        axes[0].plot(
            evaluations,
            [float(row["coverage"]) for row in subset],
            label=f"trained for {horizon}",
        )
        axes[1].plot(
            evaluations,
            [float(row["median_total_plausible_area"]) for row in subset],
            label=f"median, trained for {horizon}",
        )
        axes[1].plot(
            evaluations,
            [float(row["worst_total_plausible_area"]) for row in subset],
            linestyle="--",
            alpha=0.7,
            label=f"worst, trained for {horizon}",
        )
    axes[0].set_ylabel("soft-region coverage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prefix-loss evaluations")
    axes[1].set_ylabel("plausible union area")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "plausible_area_and_coverage_vs_queries.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_hidden = data_source._make_exact_labeled_split(
        TRAIN_N, TRAIN_DATA_SEED
    )
    validation_positions, _, validation_hidden = data_source._make_exact_labeled_split(
        VALIDATION_N, VALIDATION_DATA_SEED
    )
    test_positions, _, test_hidden = data_source._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED
    )
    del train_hidden, validation_hidden

    torch.manual_seed(MODEL_SEED)
    model = direct.Direct2DSelector().to(dtype=DTYPE)
    batch_generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    summary_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    cumulative_updates = 0
    prior_coverage: float | None = None
    stop_reason = "completed_all_horizons"

    for horizon, budget in zip(HORIZONS, UPDATE_BUDGETS):
        _, training, cumulative_updates = direct._train_stage(
            model,
            train_positions,
            validation_positions,
            horizon,
            budget,
            batch_generator,
            cumulative_updates,
        )
        rollout = direct._evaluate(model, test_positions, horizon)

        # Reference parameters are materialized only after search finishes and
        # never enter support construction, selector inputs, rewards, or loss.
        reference = _posthoc_reference(test_hidden)
        stage_query_rows = _per_query_rows(horizon, rollout, reference)
        query_rows.extend(stage_query_rows)
        final_diagnostics = _soft_diagnostics(
            rollout, reference, horizon, SUMMARY_GRID_RESOLUTION
        )
        final_metrics = _metrics_row(
            horizon, horizon, final_diagnostics, rollout
        )
        final_metrics["area_grid_resolution"] = SUMMARY_GRID_RESOLUTION
        del reference
        final_metrics.update(
            {
                "row_type": "horizon_summary",
                **training,
            }
        )
        summary_rows.append(final_metrics)
        coverage = float(final_metrics["coverage"])
        if prior_coverage is not None and coverage < prior_coverage - 1e-12:
            stop_reason = f"coverage_degraded_at_{horizon}"
            summary_rows.append(
                {
                    "row_type": "stop",
                    "trained_horizon": horizon,
                    "coverage": coverage,
                    "prior_coverage": prior_coverage,
                    "stop_reason": stop_reason,
                    "runtime_seconds_total": time.perf_counter() - start,
                }
            )
            print(
                {
                    "status": stop_reason,
                    "coverage": coverage,
                    "prior_coverage": prior_coverage,
                    "median_area": final_metrics["median_total_plausible_area"],
                },
                flush=True,
            )
            break
        prior_coverage = coverage

    if not summary_rows or summary_rows[-1].get("row_type") != "stop":
        summary_rows.append(
            {
                "row_type": "stop",
                "trained_horizon": summary_rows[-1]["trained_horizon"],
                "coverage": summary_rows[-1]["coverage"],
                "prior_coverage": prior_coverage,
                "stop_reason": stop_reason,
                "runtime_seconds_total": time.perf_counter() - start,
            }
        )
    _write_csv(OUTPUT_DIR / "soft_region_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", query_rows)
    _plot(query_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "stop_reason": stop_reason,
            "completed_horizons": sorted(
                {int(row["trained_horizon"]) for row in query_rows}
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
