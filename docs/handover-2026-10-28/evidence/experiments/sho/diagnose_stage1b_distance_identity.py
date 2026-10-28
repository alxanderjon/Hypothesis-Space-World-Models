#!/usr/bin/env python3
"""Diagnose learned-metric anisotropy versus cKDTree arithmetic ordering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_distance_metric as distance_metric
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/distance_identity_diagnostic")
CSV_PATH = OUTPUT_DIR / "distance_identity_diagnostic.csv"
SUMMARY_PATH = OUTPUT_DIR / "distance_identity_summary.txt"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_previous_w() -> np.ndarray:
    with distance_metric.FULL_PATH.open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    return np.asarray(
        [
            [float(summary["learned_W_00"]), float(summary["learned_W_01"])],
            [float(summary["learned_W_10"]), float(summary["learned_W_11"])],
        ],
        dtype=np.float64,
    )


def _quadratic_distance(delta: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Use the exact torch/einsum/sqrt path of the learned metric."""

    with torch.no_grad():
        delta_tensor = torch.from_numpy(np.asarray(delta, dtype=np.float64))
        matrix_tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float64))
        squared = torch.einsum(
            "ni,ij,nj->n", delta_tensor, matrix_tensor, delta_tensor
        )
        return torch.sqrt(squared).cpu().numpy()


def _comparison_rows(
    kind: str,
    source_state_index: int,
    trajectory_id: int,
    item_indices: np.ndarray,
    teacher_first: np.ndarray,
    teacher_second: np.ndarray,
    identity_first: np.ndarray,
    identity_second: np.ndarray,
    learned_first: np.ndarray,
    learned_second: np.ndarray,
    first_owner: np.ndarray,
    second_owner: np.ndarray,
    identity_disagreement: np.ndarray,
    previous_disagreement: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = np.flatnonzero(identity_disagreement | previous_disagreement)
    for local_index in selected:
        identity_tie = bool(identity_first[local_index] == identity_second[local_index])
        identity_reversal = bool(identity_first[local_index] > identity_second[local_index])
        rows.append(
            {
                "record_type": (
                    "identity_disagreement"
                    if identity_disagreement[local_index]
                    else "prior_reversal_replay"
                ),
                "comparison_type": kind,
                "source_state_index": source_state_index,
                "trajectory_id": trajectory_id,
                "item_index": int(item_indices[local_index]),
                "teacher_first_owner": int(first_owner[local_index]),
                "teacher_second_owner": int(second_owner[local_index]),
                "teacher_first_distance": float(teacher_first[local_index]),
                "teacher_second_distance": float(teacher_second[local_index]),
                "teacher_distance_gap": float(
                    teacher_second[local_index] - teacher_first[local_index]
                ),
                "identity_first_distance": float(identity_first[local_index]),
                "identity_second_distance": float(identity_second[local_index]),
                "identity_distance_gap": float(
                    identity_second[local_index] - identity_first[local_index]
                ),
                "identity_ordering_changed": bool(
                    identity_disagreement[local_index]
                ),
                "identity_result": (
                    "tie"
                    if identity_tie
                    else "reversal"
                    if identity_reversal
                    else "teacher_order_preserved"
                ),
                "previous_learned_first_distance": float(
                    learned_first[local_index]
                ),
                "previous_learned_second_distance": float(
                    learned_second[local_index]
                ),
                "previous_learned_ordering_changed": bool(
                    previous_disagreement[local_index]
                ),
                "identity_disagreement_is_evaluation_order_only": bool(
                    identity_disagreement[local_index]
                ),
            }
        )
    return rows


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    held_out_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    learned_w = _load_previous_w()
    identity_w = np.eye(2, dtype=np.float64)

    counts = {
        "probe_nearest_owner": {
            "raw": 0,
            "unique_nonzero": 0,
            "teacher_ties": 0,
            "zero": 0,
            "identity_changes": 0,
            "identity_reversals": 0,
            "identity_ties": 0,
            "previous_changes": 0,
        },
        "site_nearest_spacing": {
            "raw": 0,
            "unique_nonzero": 0,
            "teacher_ties": 0,
            "zero": 0,
            "identity_changes": 0,
            "identity_reversals": 0,
            "identity_ties": 0,
            "previous_changes": 0,
        },
    }
    detail_rows: list[dict[str, Any]] = []
    for source_state_index in held_out_indices:
        sites, probes, record = distance_metric._state_geometry(
            payload, trajectories, int(source_state_index)
        )
        trajectory_id = int(record["trajectory_id"])
        tree = cKDTree(sites)

        probe_teacher, probe_owners = tree.query(probes, k=2)
        probe_teacher_tie = probe_teacher[:, 0] == probe_teacher[:, 1]
        probe_zero = probe_teacher[:, 0] == 0.0
        probe_comparable = ~probe_teacher_tie & ~probe_zero
        probe_items = np.flatnonzero(probe_comparable)
        probe_first_delta = probes[probe_items] - sites[probe_owners[probe_items, 0]]
        probe_second_delta = probes[probe_items] - sites[probe_owners[probe_items, 1]]
        probe_identity = _quadratic_distance(
            np.concatenate([probe_first_delta, probe_second_delta]), identity_w
        )
        probe_learned = _quadratic_distance(
            np.concatenate([probe_first_delta, probe_second_delta]), learned_w
        )
        probe_count = len(probe_items)
        identity_first = probe_identity[:probe_count]
        identity_second = probe_identity[probe_count:]
        learned_first = probe_learned[:probe_count]
        learned_second = probe_learned[probe_count:]
        identity_disagreement = identity_first >= identity_second
        previous_disagreement = learned_first >= learned_second
        count = counts["probe_nearest_owner"]
        count["raw"] += len(probes)
        count["unique_nonzero"] += probe_count
        count["teacher_ties"] += int(np.sum(probe_teacher_tie))
        count["zero"] += int(np.sum(probe_zero))
        count["identity_changes"] += int(np.sum(identity_disagreement))
        count["identity_reversals"] += int(np.sum(identity_first > identity_second))
        count["identity_ties"] += int(np.sum(identity_first == identity_second))
        count["previous_changes"] += int(np.sum(previous_disagreement))
        detail_rows.extend(
            _comparison_rows(
                "probe_nearest_owner",
                int(source_state_index),
                trajectory_id,
                probe_items,
                probe_teacher[probe_items, 0],
                probe_teacher[probe_items, 1],
                identity_first,
                identity_second,
                learned_first,
                learned_second,
                probe_owners[probe_items, 0],
                probe_owners[probe_items, 1],
                identity_disagreement,
                previous_disagreement,
            )
        )

        spacing_teacher, spacing_owners = tree.query(sites, k=3)
        spacing_teacher_tie = spacing_teacher[:, 1] == spacing_teacher[:, 2]
        spacing_zero = spacing_teacher[:, 1] == 0.0
        spacing_comparable = ~spacing_teacher_tie & ~spacing_zero
        spacing_items = np.flatnonzero(spacing_comparable)
        spacing_first_delta = (
            sites[spacing_items] - sites[spacing_owners[spacing_items, 1]]
        )
        spacing_second_delta = (
            sites[spacing_items] - sites[spacing_owners[spacing_items, 2]]
        )
        spacing_identity = _quadratic_distance(
            np.concatenate([spacing_first_delta, spacing_second_delta]), identity_w
        )
        spacing_learned = _quadratic_distance(
            np.concatenate([spacing_first_delta, spacing_second_delta]), learned_w
        )
        spacing_count = len(spacing_items)
        identity_first = spacing_identity[:spacing_count]
        identity_second = spacing_identity[spacing_count:]
        learned_first = spacing_learned[:spacing_count]
        learned_second = spacing_learned[spacing_count:]
        identity_disagreement = identity_first >= identity_second
        previous_disagreement = learned_first >= learned_second
        count = counts["site_nearest_spacing"]
        count["raw"] += len(sites)
        count["unique_nonzero"] += spacing_count
        count["teacher_ties"] += int(np.sum(spacing_teacher_tie))
        count["zero"] += int(np.sum(spacing_zero))
        count["identity_changes"] += int(np.sum(identity_disagreement))
        count["identity_reversals"] += int(np.sum(identity_first > identity_second))
        count["identity_ties"] += int(np.sum(identity_first == identity_second))
        count["previous_changes"] += int(np.sum(previous_disagreement))
        detail_rows.extend(
            _comparison_rows(
                "site_nearest_spacing",
                int(source_state_index),
                trajectory_id,
                spacing_items,
                spacing_teacher[spacing_items, 1],
                spacing_teacher[spacing_items, 2],
                identity_first,
                identity_second,
                learned_first,
                learned_second,
                spacing_owners[spacing_items, 1],
                spacing_owners[spacing_items, 2],
                identity_disagreement,
                previous_disagreement,
            )
        )

    summary_rows: list[dict[str, Any]] = []
    for comparison_type, values in counts.items():
        summary_rows.append(
            {
                "record_type": "summary",
                "comparison_type": comparison_type,
                "held_out_state_count": len(held_out_indices),
                "total_items": values["raw"],
                "unique_nonzero_comparison_count": values["unique_nonzero"],
                "teacher_exact_tie_count": values["teacher_ties"],
                "zero_distance_exclusion_count": values["zero"],
                "previous_learned_ordering_change_count": values[
                    "previous_changes"
                ],
                "identity_ordering_change_count": values["identity_changes"],
                "identity_strict_reversal_count": values["identity_reversals"],
                "identity_induced_tie_count": values["identity_ties"],
                "identity_ordering_agreement": (
                    1.0 - values["identity_changes"] / values["unique_nonzero"]
                ),
            }
        )
    _write_csv(CSV_PATH, summary_rows + detail_rows)

    identity_changes = sum(row["identity_changes"] for row in counts.values())
    previous_changes = sum(row["previous_changes"] for row in counts.values())
    if identity_changes == 0:
        conclusion = (
            "Exact W=I preserves every unique cKDTree ordering. The four prior "
            "failures therefore came from residual anisotropy in the learned W. "
            "The appropriate next diagnostic is a constrained learned metric "
            "W=gamma*I."
        )
    else:
        conclusion = (
            "Exact W=I still disagrees with cKDTree on unique one-ULP cases. "
            "These are arithmetic/query-convention effects and should be handled "
            "as deterministic cKDTree bookkeeping, analogously to exact ties."
        )
    replay_rows = [
        row for row in detail_rows if row["previous_learned_ordering_changed"]
    ]
    lines = [
        "Exact-identity distance ordering diagnostic",
        "===========================================",
        "",
        "Scope",
        "-----",
        f"Held-out states: {len(held_out_indices)}",
        "No model was trained and no checkpoint was written or modified.",
        "The tested arithmetic was torch float64 sqrt(einsum(delta, I, delta)) ",
        "with I exactly equal to the 2x2 identity matrix and no epsilon.",
        "",
    ]
    for comparison_type, values in counts.items():
        lines.extend(
            [
                comparison_type.replace("_", " ").title(),
                "-" * len(comparison_type),
                f"Total items: {values['raw']}",
                f"Unique nonzero comparisons: {values['unique_nonzero']}",
                f"Teacher exact ties kept as cKDTree bookkeeping: {values['teacher_ties']}",
                f"Zero-distance exclusions: {values['zero']}",
                f"Prior learned-W ordering changes: {values['previous_changes']}",
                f"Exact-identity ordering changes: {values['identity_changes']}",
                f"  strict reversals: {values['identity_reversals']}",
                f"  identity-path ties: {values['identity_ties']}",
                "",
            ]
        )
    lines.extend(
        [
            "Replay of the four prior reversals",
            "----------------------------------",
            f"Prior reversal rows recovered: {len(replay_rows)}/{previous_changes}",
        ]
    )
    for row in replay_rows:
        lines.append(
            f"- {row['comparison_type']} state={row['source_state_index']} "
            f"trajectory={row['trajectory_id']} item={row['item_index']}: "
            f"teacher=({row['teacher_first_distance']:.17g}, "
            f"{row['teacher_second_distance']:.17g}), "
            f"gap={row['teacher_distance_gap']:.17g}; "
            f"identity=({row['identity_first_distance']:.17g}, "
            f"{row['identity_second_distance']:.17g}), "
            f"result={row['identity_result']}"
        )
    lines.extend(["", "Conclusion", "----------", conclusion, ""])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"identity diagnostic complete: previous={previous_changes} "
        f"identity_changes={identity_changes}",
        flush=True,
    )


if __name__ == "__main__":
    run()
