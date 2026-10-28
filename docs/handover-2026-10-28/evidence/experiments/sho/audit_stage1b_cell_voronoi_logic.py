#!/usr/bin/env python3
"""Audit and minimally trace the deterministic eDIRECT Cell/Voronoi rule."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
)


OUTPUT_DIR = Path("outputs/sho/stage1B/cell_voronoi_logic_audit")
AUDIT_PATH = OUTPUT_DIR / "cell_voronoi_logic_audit.md"
TRACE_PATH = OUTPUT_DIR / "tiny_curation_trace.csv"
RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
STATE_COUNT = 15
DETAIL_COUNT = 3


def _representative_indices(states: list[dict[str, Any]]) -> list[int]:
    cell_indices = [
        index for index, state in enumerate(states) if state["candidate_type"] == "cell"
    ]
    by_k: dict[int, list[int]] = {}
    for index in cell_indices:
        by_k.setdefault(len(states[index]["teacher_curated_points"]), []).append(index)
    keys = sorted(by_k)
    target_positions = np.unique(
        np.rint(np.linspace(0, len(keys) - 1, STATE_COUNT)).astype(int)
    )
    chosen: list[int] = []
    for position in target_positions:
        key = keys[int(position)]
        options = sorted(
            by_k[key], key=lambda index: int(states[index]["history_count"])
        )
        chosen.append(options[len(options) // 2])
    if len(chosen) < STATE_COUNT:
        remaining = [index for index in cell_indices if index not in chosen]
        remaining.sort(
            key=lambda index: (
                len(states[index]["teacher_curated_points"]),
                int(states[index]["history_count"]),
            )
        )
        chosen.extend(remaining[: STATE_COUNT - len(chosen)])
    return sorted(
        chosen,
        key=lambda index: (
            len(states[index]["teacher_curated_points"]),
            int(states[index]["history_count"]),
        ),
    )[:STATE_COUNT]


def _set_matches(left: np.ndarray, right: np.ndarray) -> bool:
    if len(left) != len(right):
        return False
    distances, _ = cKDTree(right).query(left, k=1)
    reverse, _ = cKDTree(left).query(right, k=1)
    return bool(max(float(np.max(distances)), float(np.max(reverse))) <= 1.0e-12)


def _physical(point: np.ndarray) -> np.ndarray:
    return LOWER + DOMAIN_WIDTH * point


def _fmt_points(points: np.ndarray) -> str:
    return "; ".join(f"({point[0]:.9g},{point[1]:.9g})" for point in points)


def _trace_state(
    state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
    detailed: bool,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    history_count = int(record["history_count"])
    iteration = int(record["iteration"])
    generator_id = int(record["generator_id"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    data = edirect._cell_data(points, losses, generator_id, iteration)
    surface = edirect.GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    size_weights = edirect._normalize(selected_sizes)
    loo_errors = surface.loo_errors(data.selected)
    loo_weights = edirect._normalize(loo_errors)
    combined_weights = 0.5 * size_weights + 0.5 * loo_weights
    largest_position = int(np.argmax(selected_sizes))
    combined_weights[largest_position] = 1.0
    reconstructed = edirect._cell_proposals(
        points, losses, generator_id, iteration
    )
    saved = np.asarray(record["teacher_curated_points"], dtype=np.float64)
    teacher_action = np.asarray(record["teacher_selected_point"], dtype=np.float64)
    requested = min(edirect.MC_FACTOR_PER_SITE * history_count, edirect.MC_CANDIDATE_CAP)
    edge_count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    selected_owner_mask = np.isin(data.owners, data.selected)
    rows: list[dict[str, object]] = [
        {
            "state_index": state_index,
            "row_type": "state_summary",
            "detailed_state": detailed,
            "trajectory_id": int(record["trajectory_id"]),
            "generator_id": generator_id,
            "iteration": iteration,
            "history_count": history_count,
            "raw_candidate_count": len(data.points),
            "requested_sobol_interior_count": requested,
            "edge_points_per_axis": edge_count,
            "generated_boundary_points_before_unique": 4 * edge_count,
            "evaluated_site_count": len(points),
            "sites_owning_raw_candidates": len(np.unique(data.owners)),
            "potentially_optimal_site_count": len(data.selected),
            "raw_candidates_in_selected_cells": int(np.sum(selected_owner_mask)),
            "raw_candidates_rejected_with_unselected_cells": int(
                np.sum(~selected_owner_mask)
            ),
            "final_curated_count": len(reconstructed),
            "saved_teacher_curated_count": len(saved),
            "curated_set_matches_saved_teacher": _set_matches(reconstructed, saved),
            "first_curated_matches_teacher_action": bool(
                np.linalg.norm(reconstructed[0] - teacher_action) <= 1.0e-12
            ),
            "final_curated_set_normalized": _fmt_points(reconstructed),
        }
    ]

    cell_summaries: list[dict[str, Any]] = []
    output_position = 0
    for selected_position, site_index in enumerate(data.selected):
        global_indices = np.flatnonzero(data.owners == site_index)
        candidates = data.points[global_indices]
        distances = data.distances[global_indices]
        if not len(candidates):
            continue
        predictions = surface.predict(candidates)
        normalized_distance = edirect._normalize(distances)
        normalized_prediction = edirect._normalize(predictions)
        if selected_position == largest_position:
            utilities = distances.copy()
            rule = "largest_selected_cell:farthest_raw_distance"
        else:
            weight = float(combined_weights[selected_position])
            utilities = weight * normalized_distance - (
                1.0 - weight
            ) * normalized_prediction
            rule = "weighted_cell_utility:u*d_norm-(1-u)*surrogate_norm"
        local_order = np.argsort(utilities)[::-1]
        winner_local = int(local_order[0])
        winner_global = int(global_indices[winner_local])
        winner = candidates[winner_local]
        duplicate = bool(
            np.min(np.linalg.norm(points - winner[None, :], axis=1)) <= 1.0e-12
        )
        accepted = not duplicate
        physical = _physical(winner)
        cell_row = {
            "state_index": state_index,
            "row_type": "selected_cell",
            "detailed_state": detailed,
            "trajectory_id": int(record["trajectory_id"]),
            "iteration": iteration,
            "history_count": history_count,
            "raw_candidate_count": len(data.points),
            "selected_cell_position": selected_position,
            "selected_site_index": int(site_index),
            "selected_site_x": float(points[site_index, 0]),
            "selected_site_y": float(points[site_index, 1]),
            "selected_site_loss": float(losses[site_index]),
            "cell_size": float(selected_sizes[selected_position]),
            "normalized_cell_size": float(size_weights[selected_position]),
            "loo_error": float(loo_errors[selected_position]),
            "normalized_loo_error": float(loo_weights[selected_position]),
            "combined_exploration_weight": float(
                combined_weights[selected_position]
            ),
            "is_largest_selected_cell": selected_position == largest_position,
            "candidate_count_in_cell": len(candidates),
            "selection_rule": rule,
            "winner_raw_index": winner_global,
            "winner_x": float(winner[0]),
            "winner_y": float(winner[1]),
            "winner_physical_a": float(physical[0]),
            "winner_physical_b": float(physical[1]),
            "winner_distance_to_owner": float(distances[winner_local]),
            "winner_surrogate_log10_loss": float(predictions[winner_local]),
            "winner_normalized_distance": float(normalized_distance[winner_local]),
            "winner_normalized_surrogate": float(
                normalized_prediction[winner_local]
            ),
            "winner_utility": float(utilities[winner_local]),
            "winner_duplicate_of_history": duplicate,
            "winner_accepted": accepted,
            "within_cell_rejected_count": len(candidates) - int(accepted),
            "curated_output_order": output_position if accepted else "",
        }
        rows.append(cell_row)
        cell_summaries.append(cell_row)
        if accepted:
            output_position += 1

        if detailed:
            for candidate_rank, local_index in enumerate(local_order[:3], start=1):
                candidate = candidates[int(local_index)]
                candidate_physical = _physical(candidate)
                rows.append(
                    {
                        "state_index": state_index,
                        "row_type": "candidate_detail_top3_within_selected_cell",
                        "detailed_state": True,
                        "trajectory_id": int(record["trajectory_id"]),
                        "iteration": iteration,
                        "history_count": history_count,
                        "selected_cell_position": selected_position,
                        "selected_site_index": int(site_index),
                        "candidate_rank_within_cell": candidate_rank,
                        "candidate_raw_index": int(global_indices[int(local_index)]),
                        "candidate_x": float(candidate[0]),
                        "candidate_y": float(candidate[1]),
                        "candidate_physical_a": float(candidate_physical[0]),
                        "candidate_physical_b": float(candidate_physical[1]),
                        "candidate_distance_to_owner": float(distances[int(local_index)]),
                        "candidate_surrogate_log10_loss": float(
                            predictions[int(local_index)]
                        ),
                        "candidate_normalized_distance": float(
                            normalized_distance[int(local_index)]
                        ),
                        "candidate_normalized_surrogate": float(
                            normalized_prediction[int(local_index)]
                        ),
                        "candidate_utility": float(utilities[int(local_index)]),
                        "candidate_accepted": int(local_index) == winner_local
                        and accepted,
                        "candidate_rejection_reason": ""
                        if int(local_index) == winner_local and accepted
                        else "lower within-cell utility",
                    }
                )

    summary = {
        "state_index": state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": iteration,
        "history_count": history_count,
        "raw_count": len(data.points),
        "selected_cells": len(data.selected),
        "curated_count": len(reconstructed),
        "selected_cell_rows": cell_summaries,
        "curated": reconstructed,
        "set_match": _set_matches(reconstructed, saved),
        "first_match": bool(np.linalg.norm(reconstructed[0] - teacher_action) <= 1e-12),
        "detailed": detailed,
    }
    return rows, summary


def _manual_state_section(summary: dict[str, Any]) -> str:
    lines = [
        f"### State {summary['state_index']}: trajectory {summary['trajectory_id']}, "
        f"iteration {summary['iteration']}",
        "",
        f"- History sites: {summary['history_count']}",
        f"- Raw candidates: {summary['raw_count']}",
        f"- Potentially optimal cells: {summary['selected_cells']}",
        f"- Curated candidates: {summary['curated_count']}",
        f"- Saved-set reconstruction: `{summary['set_match']}`; first action match: "
        f"`{summary['first_match']}`",
        "",
        "| Order | Site | Candidates | Rule | Weight | Winner (normalized) | Rejected |",
        "|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in summary["selected_cell_rows"]:
        lines.append(
            f"| {row['curated_output_order']} | {row['selected_site_index']} | "
            f"{row['candidate_count_in_cell']} | {row['selection_rule']} | "
            f"{row['combined_exploration_weight']:.6g} | "
            f"({row['winner_x']:.8g}, {row['winner_y']:.8g}) | "
            f"{row['within_cell_rejected_count']} |"
        )
    lines.extend(
        [
            "",
            "The CSV includes the top three within-cell competitors, their distances, "
            "surrogate predictions, normalized values, and utilities for this state.",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_markdown(summaries: list[dict[str, Any]]) -> str:
    verified_sets = sum(summary["set_match"] for summary in summaries)
    verified_first = sum(summary["first_match"] for summary in summaries)
    detailed = [summary for summary in summaries if summary["detailed"]]
    manual = "\n".join(_manual_state_section(summary) for summary in detailed)
    return f"""# eDIRECT Cell/Voronoi curation logic audit

## Result

Cell/Voronoi curation is **hierarchical group-wise selection**, not global independent
ranking and not sequential/conditional set construction. The raw points are assigned to
Voronoi cells owned by already evaluated hypotheses. eDIRECT first selects potentially
optimal owner cells, then retains exactly one within-cell winner from each selected cell.
All winners are computed from the same pre-iteration history; accepting one winner does
not change another winner in the current curated set.

The audit reconstructed {verified_sets}/{len(summaries)} saved curated sets exactly and
matched the saved first teacher action in {verified_first}/{len(summaries)} states.

## Exact code path

The audited implementation is
`experiments/sho/run_stage1b_edirect_vs_direct.py`:

1. `_candidate_pool(m, trajectory_id, iteration)` creates the raw pool in normalized
   `[0,1]^2` coordinates. It requests `min(100*m, 32768)` scrambled Sobol interior
   points, adds uniformly spaced points on all four closed-domain edges, then removes
   exact duplicates with `np.unique`.
2. `_cell_data(...)` assigns every raw point to its nearest evaluated hypothesis using
   a KD-tree. Candidate information at this stage is its coordinate, nearest-site owner,
   and distance to that owner.
3. Each evaluated site's approximate Voronoi size is the largest assigned raw-point
   distance, lower-bounded by half its nearest evaluated-site spacing.
4. `_potentially_optimal_cells(...)` groups sites with equal size, keeps the lowest-loss
   site per size, builds the DIRECT lower-right hull in `(cell_size, observed_loss)`, and
   applies the `epsilon=1e-4` sufficient-decrease slope condition. If none survives, it
   falls back to the largest cell. Selected sites are sorted by
   `(cell_size, observed_loss, site_index)`.
5. `_cell_proposals(...)` fits one `GaussianLocalResponse` to the existing history using
   `log10(max(loss, 1e-300))` and up to 24 nearest neighbors.
6. For the selected cells, it computes normalized cell size and normalized leave-one-out
   surrogate error. Their mean is the exploration weight `u`. The largest selected cell
   has `u=1` forced explicitly.
7. Raw candidates compete **only with candidates having the same nearest-site owner**.
   In the largest selected cell, the farthest raw candidate wins. In every other selected
   cell, the winner maximizes
   `u * normalize(distance) - (1-u) * normalize(surrogate_prediction)` within that cell.
   Normalization is cell-relative, so a candidate's effective score depends on the other
   candidates in its cell, but not on winners chosen in other cells.
8. A winner already within `1e-12` of history is skipped. Ordinarily this yields one
   candidate per potentially optimal cell. If no proposal survives, the globally farthest
   nonduplicate raw point is used as a fallback.
9. The resulting curated list is ordered by the sorted selected-cell order. In the
   deterministic teacher, action selection is simply `choice=0`; candidates are evaluated
   in list order. This evaluation is sequential, but curation was completed before any of
   those evaluations and is not recomputed between them.
10. Only after all Cell proposals are evaluated does `_greedy_proposal(...)` run. It builds
    a separate raw pool using `iteration+1_000_000`, adds local rings around the 12
    best-loss history points (24 angles at four radii), refits the surrogate on the updated
    history, and evaluates the nonduplicate candidate with minimum predicted loss. This
    greedy point is **not part of Cell/Voronoi curation**.

## Answers to the audit questions

1. **Raw pool:** deterministic scrambled Sobol interior plus explicit global-edge points,
   uniqued; its size grows with history and is capped at 32,768 requested interior points.
2. **Per-candidate information:** coordinate, nearest owner, owner distance, and—only for
   selected cells—local surrogate prediction plus cell-normalized distance/prediction.
   Cell-level context supplies site loss, cell size, LOO error, and exploration weight.
3. **Independent scores?** No. Candidates are grouped by Voronoi owner; normalization and
   competition happen within a cell.
4. **Global top-k?** No.
5. **Sequential selection?** No for curation. It is one simultaneous group argmax per
   selected cell. Subsequent objective evaluations are sequential.
6. **Does one retained candidate change another?** No within the same curation call.
7. **Constraints:** one winner per selected potentially optimal Voronoi cell; a special
   pure-exploration rule for the largest selected cell; cell-relative distance/surrogate
   competition elsewhere; duplicate-distance rejection. Cross-cell diversity follows
   structurally from the one-per-cell rule.
8. **Retained count:** the number of potentially optimal selected cells that produce a
   nonduplicate winner—not a learned or fixed global `k`.
9. **Order:** raw pool -> nearest-owner assignment -> cell sizes -> potentially optimal
   cells -> surrogate/LOO weights -> one within-cell argmax -> duplicate check -> ordered
   curated list -> sequential real evaluations -> separate greedy-surrogate evaluation.
10. **Dependence on retained candidates:** none during current-set construction. Future
    iterations depend on all newly evaluated Cell candidates and the greedy point.

## Pseudocode

```text
raw = unique(Sobol(min(100*m, 32768)) + all four domain edges)
owner[c], distance[c] = nearest_evaluated_site(raw[c])

size[s] = max(distance[c] for owner[c] == s)
size[s] = max(size[s], 0.5 * nearest_history_neighbor_distance[s])
selected_sites = DIRECT_lower_right_hull_with_epsilon(size, observed_loss)

surface = GaussianLocalResponse(history_points, log10(history_losses))
u = 0.5 * normalize(size[selected_sites])
    + 0.5 * normalize(LOO_error[selected_sites])
u[argmax(size[selected_sites])] = 1

curated = []
for site in selected_sites sorted by (size, loss, index):
    group = raw candidates whose nearest owner is site
    if site is largest selected cell:
        winner = argmax_group(distance)
    else:
        d = normalize_within_group(distance)
        f = normalize_within_group(surface_prediction)
        winner = argmax_group(u[site] * d - (1-u[site]) * f)
    if winner is not a duplicate of history:
        curated.append(winner)

for candidate in curated:       # evaluation order, not curation logic
    evaluate real prefix loss(candidate)

greedy = minimum-surrogate candidate from a separate global+local pool
evaluate real prefix loss(greedy)
```

## Natural neural analogue

The closest learned replacement is a **hierarchical, group-equivariant set model**, not a
flat binary filter or global ranker:

- a history/site encoder predicts or imitates the potentially-optimal-cell mask;
- deterministic nearest-site assignment groups raw candidates by Voronoi owner;
- a shared per-cell candidate scorer receives candidate features plus pooled cell context;
- a hard or differentiable **group top-1** chooses one candidate per selected cell;
- cardinality is induced by the number of selected cells, so no separate global count head
  or oversized shortlist is needed;
- action ordering among the resulting cell winners remains a separate selector problem.

This architecture preserves the teacher's main inductive biases: cell ownership,
one-per-cell coverage, cell-relative competition, and separation of Cell curation from the
later greedy-surrogate point.

## Detailed manually reproducible states

{manual}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_trace = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = filter_data._trajectory_map(raw_trace)
    indices = _representative_indices(raw_trace["states"])
    detail_positions = set(
        np.rint(np.linspace(0, len(indices) - 1, DETAIL_COUNT)).astype(int).tolist()
    )
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, Any]] = []
    for position, state_index in enumerate(indices):
        record = raw_trace["states"][state_index]
        trajectory = trajectories[int(record["trajectory_id"])]
        state_rows, summary = _trace_state(
            state_index, record, trajectory, position in detail_positions
        )
        rows.extend(state_rows)
        summaries.append(summary)
        print(
            f"traced state={state_index} k={summary['curated_count']} "
            f"raw={summary['raw_count']} history={summary['history_count']}",
            flush=True,
        )
    if not all(summary["set_match"] and summary["first_match"] for summary in summaries):
        raise RuntimeError("manual reconstruction disagrees with saved teacher trace")
    _write_csv(TRACE_PATH, rows)
    AUDIT_PATH.write_text(_audit_markdown(summaries), encoding="utf-8")
    print(f"wrote {AUDIT_PATH} and {TRACE_PATH}", flush=True)


if __name__ == "__main__":
    main()
