#!/usr/bin/env python3
"""Validate the frozen learned metric through the assembled cell-size pipeline."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.audit_stage1b_same_size_grouping as grouping_audit
import experiments.sho.run_stage1b_distance_metric as distance_metric
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_keep_pop_threshold as hull_threshold
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_improvement_and as improvement_and
import experiments.sho.run_stage1b_improvement_and_truth_table as final_and
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data
import experiments.sho.run_stage1b_representative_margin as representative_margin
import experiments.sho.run_stage1b_representative_threshold as representative_threshold
import experiments.sho.run_stage1b_same_size_margin as same_size_margin
import experiments.sho.run_stage1b_same_size_threshold as same_size_threshold


OUTPUT_DIR = Path("outputs/sho/stage1B/distance_metric_assembled_validation")
ASSEMBLED_PATH = OUTPUT_DIR / "distance_assembled_validation.csv"
DOWNSTREAM_PATH = OUTPUT_DIR / "distance_downstream_validation.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "distance_metric_lock_summary.csv"
LOCKED_CHECKPOINT_PATH = OUTPUT_DIR / "geometric_distance_metric_locked.pt"
LOCKED_HASH_PATH = LOCKED_CHECKPOINT_PATH.with_suffix(".sha256")

CANDIDATE_PATH = Path(
    "outputs/sho/stage1B/distance_metric_symmetry/"
    "geometric_distance_metric_frozen_candidate.pt"
)
CANDIDATE_SHA256 = (
    "940a256347d739eec37e12659ff6fe736f6f72820769e3e352018335d2d9ede8"
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_changes(first: np.ndarray, second: np.ndarray) -> int:
    common = min(len(first), len(second))
    return int(np.sum(first[:common] != second[:common])) + abs(
        len(first) - len(second)
    )


def _set_difference_count(first: np.ndarray, second: np.ndarray) -> int:
    return len(set(int(value) for value in first) ^ set(int(value) for value in second))


def _load_distance_model() -> tuple[distance_metric.QuadraticMetric, dict[str, Any]]:
    actual_hash = _sha256(CANDIDATE_PATH)
    if actual_hash != CANDIDATE_SHA256:
        raise RuntimeError(
            f"distance candidate hash mismatch: {actual_hash} != {CANDIDATE_SHA256}"
        )
    checkpoint = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("distance checkpoint is not a frozen candidate")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    model = distance_metric.QuadraticMetric(generator)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _load_linear_checkpoint(
    path: Path, input_count: int, bias: bool, expected_hash: str, status: str
) -> tuple[nn.Linear, dict[str, Any]]:
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"checkpoint hash mismatch for {path}: {actual_hash}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != status:
        raise RuntimeError(f"unexpected checkpoint status for {path}")
    model = nn.Linear(input_count, 1, bias=bias).to(torch.float64)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _assemble_one_state(
    model: distance_metric.QuadraticMetric,
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    source_state_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sites, probes, record = distance_metric._state_geometry(
        payload, trajectories, source_state_index
    )
    trajectory = trajectories[int(record["trajectory_id"])]
    count = int(record["history_count"])
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:count], dtype=np.float64
    )
    production = edirect._cell_data(
        sites,
        losses,
        int(record["generator_id"]),
        int(record["iteration"]),
    )
    if not np.array_equal(probes, production.points):
        raise RuntimeError("regenerated probe pool differs from production")
    tree = cKDTree(sites)

    # k=2 is an exact geometric candidate search.  For a unique teacher
    # nearest site the frozen learned metric chooses between the nearest and
    # runner-up.  Only exact teacher ties retain production k=1 ownership.
    probe_teacher_distances, probe_candidates = tree.query(probes, k=2)
    probe_exact_tie = (
        probe_teacher_distances[:, 0] == probe_teacher_distances[:, 1]
    )
    first_delta = probes - sites[probe_candidates[:, 0]]
    second_delta = probes - sites[probe_candidates[:, 1]]
    first_learned = distance_metric._predict(model, first_delta)
    second_learned = distance_metric._predict(model, second_delta)
    choose_second = second_learned < first_learned
    learned_owners = np.where(
        choose_second, probe_candidates[:, 1], probe_candidates[:, 0]
    ).astype(np.int64)
    learned_distances = np.where(
        choose_second, second_learned, first_learned
    ).astype(np.float64)
    learned_owners[probe_exact_tie] = production.owners[probe_exact_tie]
    learned_distances[probe_exact_tie] = production.distances[probe_exact_tie]

    teacher_radius = np.zeros(count, dtype=np.float64)
    learned_radius = np.zeros(count, dtype=np.float64)
    np.maximum.at(teacher_radius, production.owners, production.distances)
    np.maximum.at(learned_radius, learned_owners, learned_distances)

    spacing_teacher_k2, spacing_teacher_owner_k2 = tree.query(sites, k=2)
    spacing_teacher_k3, spacing_candidates = tree.query(sites, k=3)
    spacing_exact_tie = spacing_teacher_k3[:, 1] == spacing_teacher_k3[:, 2]
    first_spacing_delta = sites - sites[spacing_candidates[:, 1]]
    second_spacing_delta = sites - sites[spacing_candidates[:, 2]]
    first_spacing = distance_metric._predict(model, first_spacing_delta)
    second_spacing = distance_metric._predict(model, second_spacing_delta)
    choose_second_spacing = second_spacing < first_spacing
    learned_spacing_owner = np.where(
        choose_second_spacing,
        spacing_candidates[:, 2],
        spacing_candidates[:, 1],
    ).astype(np.int64)
    learned_spacing = np.where(
        choose_second_spacing, second_spacing, first_spacing
    ).astype(np.float64)
    learned_spacing_owner[spacing_exact_tie] = spacing_teacher_owner_k2[
        spacing_exact_tie, 1
    ]
    learned_spacing[spacing_exact_tie] = spacing_teacher_k2[spacing_exact_tie, 1]
    teacher_spacing_owner = spacing_teacher_owner_k2[:, 1].astype(np.int64)
    teacher_spacing = spacing_teacher_k2[:, 1].astype(np.float64)

    learned_sizes = np.maximum(learned_radius, 0.5 * learned_spacing)
    teacher_sizes = production.sizes
    probe_owner_mismatch = learned_owners != production.owners
    radius_difference = np.abs(learned_radius - teacher_radius)
    spacing_owner_mismatch = learned_spacing_owner != teacher_spacing_owner
    spacing_difference = np.abs(learned_spacing - teacher_spacing)
    size_difference = np.abs(learned_sizes - teacher_sizes)
    row = {
        "record_type": "state",
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": int(record["iteration"]),
        "site_count": count,
        "probe_count": len(probes),
        "probe_exact_tie_count": int(np.sum(probe_exact_tie)),
        "probe_assignment_mismatch_count": int(np.sum(probe_owner_mismatch)),
        "maximum_probe_distance_absolute_difference": float(
            np.max(np.abs(learned_distances - production.distances))
        ),
        "assigned_radius_mismatch_count": int(np.sum(radius_difference != 0.0)),
        "maximum_assigned_radius_absolute_difference": float(
            np.max(radius_difference)
        ),
        "spacing_exact_tie_count": int(np.sum(spacing_exact_tie)),
        "nearest_site_winner_mismatch_count": int(
            np.sum(spacing_owner_mismatch)
        ),
        "spacing_mismatch_count": int(np.sum(spacing_difference != 0.0)),
        "maximum_spacing_absolute_difference": float(np.max(spacing_difference)),
        "final_cell_size_mismatch_count": int(np.sum(size_difference != 0.0)),
        "maximum_final_cell_size_absolute_difference": float(
            np.max(size_difference)
        ),
    }
    row["complete_state_exact"] = bool(
        row["probe_assignment_mismatch_count"] == 0
        and row["maximum_probe_distance_absolute_difference"] == 0.0
        and row["assigned_radius_mismatch_count"] == 0
        and row["nearest_site_winner_mismatch_count"] == 0
        and row["spacing_mismatch_count"] == 0
        and row["final_cell_size_mismatch_count"] == 0
    )
    return row, {
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": int(record["iteration"]),
        "state_bin": turn._state_bin(count),
        "losses": losses,
        "teacher_sizes": teacher_sizes,
        "learned_sizes": learned_sizes,
    }


def _load_downstream() -> dict[str, Any]:
    same_margin_model, _ = _load_linear_checkpoint(
        same_size_margin.CANDIDATE_PATH,
        1,
        True,
        same_size_threshold.FROZEN_MARGIN_SHA256,
        "FROZEN_CANDIDATE",
    )
    same_threshold_checkpoint = torch.load(
        same_size_threshold.THRESHOLD_PATH, map_location="cpu", weights_only=False
    )
    if same_threshold_checkpoint.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("same-size threshold is not locked")

    representative_margin_model, _ = _load_linear_checkpoint(
        representative_margin.CANDIDATE_PATH,
        2,
        False,
        representative_threshold.FROZEN_MARGIN_SHA256,
        "FROZEN_CANDIDATE",
    )
    representative_threshold_checkpoint = torch.load(
        representative_threshold.THRESHOLD_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if representative_threshold_checkpoint.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("representative threshold is not locked")

    skew_weight, skew_hash = hull_threshold._load_skew()
    homogeneous_checkpoint = torch.load(
        hull_threshold.HOMOGENEOUS_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if homogeneous_checkpoint.get("lock_status") != "LOCKED":
        raise RuntimeError("homogeneous hull margin is not locked")
    homogeneous_weight = (
        homogeneous_checkpoint["effective_weight"].detach().cpu().numpy().copy()
    )
    hull_checkpoint = torch.load(
        hull_threshold.THRESHOLD_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if hull_checkpoint.get("lock_status") != "LOCKED":
        raise RuntimeError("hull threshold is not locked")

    improvement_and._verify_frozen_hashes()
    and_checkpoint = torch.load(
        final_and.FINAL_PATH, map_location="cpu", weights_only=False
    )
    if (
        and_checkpoint.get("status") != "LOCKED_COMPONENT"
        or and_checkpoint.get("training_source") != "complete_AND_truth_table"
    ):
        raise RuntimeError("final truth-table AND is not locked")
    and_model = nn.Linear(2, 1, bias=True).to(torch.float64)
    and_model.load_state_dict(and_checkpoint["state_dict"])
    and_model.eval()
    for parameter in and_model.parameters():
        parameter.requires_grad_(False)
    return {
        "same_margin_weight": float(same_margin_model.weight.detach().item()),
        "same_margin_bias": float(same_margin_model.bias.detach().item()),
        "same_theta": float(same_threshold_checkpoint["learned_theta"].item()),
        "representative_margin_weight": representative_margin_model.weight.detach()
        .cpu()
        .numpy()[0]
        .copy(),
        "representative_theta": float(
            representative_threshold_checkpoint["learned_theta"].item()
        ),
        "skew_weight": skew_weight,
        "skew_hash": skew_hash,
        "homogeneous_weight": homogeneous_weight,
        "hull_theta": float(hull_checkpoint["learned_theta"].item()),
        "hull_alpha": float(hull_checkpoint["learned_alpha"].item()),
        "and_model": and_model,
    }


def _learned_group_and_represent(
    sizes: np.ndarray, losses: np.ndarray, locked: dict[str, Any]
) -> dict[str, np.ndarray]:
    order = np.lexsort((np.arange(len(sizes)), losses, sizes))
    groups: list[dict[str, Any]] = []
    membership = np.full(len(sizes), -1, dtype=np.int64)
    for index_value in order:
        index = int(index_value)
        size = float(sizes[index])
        loss = float(losses[index])
        if not groups:
            groups.append(
                {
                    "anchor": size,
                    "representative": index,
                    "representative_loss": loss,
                }
            )
            membership[index] = 0
            continue
        group = groups[-1]
        normalized_gap = abs(size - float(group["anchor"])) / 1.0e-14
        same_margin = (
            locked["same_margin_weight"] * normalized_gap
            + locked["same_margin_bias"]
        )
        if same_margin < locked["same_theta"]:
            groups.append(
                {
                    "anchor": size,
                    "representative": index,
                    "representative_loss": loss,
                }
            )
            membership[index] = len(groups) - 1
            continue
        membership[index] = len(groups) - 1
        best_index = int(group["representative"])
        best_loss = float(losses[best_index])
        scale = max(abs(best_loss), abs(loss))
        if not scale > 0.0:
            raise RuntimeError("nonpositive representative comparison scale")
        features = np.asarray([best_loss / scale, loss / scale])
        representative_margin_value = float(
            features @ locked["representative_margin_weight"]
        )
        if representative_margin_value > locked["representative_theta"]:
            group["representative"] = index
            group["representative_loss"] = loss
    return {
        "order": order.astype(np.int64),
        "membership": membership,
        "grouped_sizes": np.asarray(
            [float(group["anchor"]) for group in groups], dtype=np.float64
        ),
        "grouped_losses": np.asarray(
            [float(group["representative_loss"]) for group in groups],
            dtype=np.float64,
        ),
        "representatives": np.asarray(
            [int(group["representative"]) for group in groups], dtype=np.int64
        ),
    }


def _downstream_validation(
    assembled_states: list[dict[str, Any]], locked: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    intermediate: list[dict[str, Any]] = []
    learned_improvement_states: list[dict[str, Any]] = []
    teacher_po_sites: list[np.ndarray] = []
    learned_representatives: list[np.ndarray] = []
    state_rows: list[dict[str, Any]] = []
    for state in assembled_states:
        losses = state["losses"]
        teacher = grouping_audit._teacher_trace(state["teacher_sizes"], losses)
        learned = _learned_group_and_represent(
            state["learned_sizes"], losses, locked
        )
        teacher_hull, _ = turn._teacher_hull_and_triples(
            teacher["grouped_sizes"], teacher["grouped_losses"]
        )
        learned_hull = hull_threshold._threshold_hull(
            locked["skew_weight"],
            locked["homogeneous_weight"],
            locked["hull_theta"],
            locked["hull_alpha"],
            learned["grouped_sizes"],
            learned["grouped_losses"],
        )
        teacher_hull_sites = teacher["representatives"][teacher_hull]
        learned_hull_sites = learned["representatives"][learned_hull]
        teacher_po = edirect._potentially_optimal_cells(
            teacher["grouped_sizes"], teacher["grouped_losses"]
        )
        teacher_po_sites.append(teacher["representatives"][teacher_po])
        learned_representatives.append(learned["representatives"])
        membership_changes = int(
            np.sum(teacher["membership"] != learned["membership"])
        )
        representative_changes = _ordered_changes(
            teacher["representatives"], learned["representatives"]
        )
        hull_changes = _ordered_changes(teacher_hull_sites, learned_hull_sites)
        state_rows.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "site_count": len(losses),
                "teacher_group_count": len(teacher["grouped_sizes"]),
                "learned_group_count": len(learned["grouped_sizes"]),
                "same_size_membership_change_count": membership_changes,
                "representative_ordered_change_count": representative_changes,
                "representative_symmetric_difference_count": _set_difference_count(
                    teacher["representatives"], learned["representatives"]
                ),
                "teacher_hull_size": len(teacher_hull_sites),
                "learned_hull_size": len(learned_hull_sites),
                "hull_ordered_change_count": hull_changes,
                "hull_symmetric_difference_count": _set_difference_count(
                    teacher_hull_sites, learned_hull_sites
                ),
            }
        )
        learned_improvement_states.append(
            {
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "iteration": state["iteration"],
                "state_bin": state["state_bin"],
                "sizes": learned["grouped_sizes"],
                "losses": learned["grouped_losses"],
                "teacher_hull": learned_hull,
            }
        )
        intermediate.append(
            {
                "teacher": teacher,
                "learned": learned,
                "teacher_hull": teacher_hull,
                "learned_hull": learned_hull,
            }
        )

    pool = improvement_and._build_pool(
        learned_improvement_states, "assembled_held_out"
    )
    with torch.no_grad():
        logits = (
            locked["and_model"](
                torch.as_tensor(pool["inputs"], dtype=torch.float64)
            )
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    passes = logits >= 0.0
    for state_position, row in enumerate(state_rows):
        mask = pool["state_position"] == state_position
        hull_positions = pool["hull_position"][mask]
        state_passes = passes[mask]
        learned = intermediate[state_position]["learned"]
        learned_hull = intermediate[state_position]["learned_hull"]
        selected = [
            int(learned_hull[int(position)])
            for position, passed in zip(hull_positions, state_passes)
            if bool(passed)
        ]
        if not selected:
            selected.append(int(np.argmax(learned["grouped_sizes"])))
        learned_po = np.asarray(
            sorted(
                selected,
                key=lambda index: (
                    learned["grouped_sizes"][index],
                    learned["grouped_losses"][index],
                    index,
                ),
            ),
            dtype=np.int64,
        )
        learned_po_sites = learned_representatives[state_position][learned_po]
        teacher_sites = teacher_po_sites[state_position]
        row["teacher_PO_size"] = len(teacher_sites)
        row["learned_PO_size"] = len(learned_po_sites)
        row["improvement_filtered_PO_ordered_change_count"] = _ordered_changes(
            teacher_sites, learned_po_sites
        )
        row["improvement_filtered_PO_symmetric_difference_count"] = (
            _set_difference_count(teacher_sites, learned_po_sites)
        )
        row["complete_downstream_exact"] = bool(
            row["same_size_membership_change_count"] == 0
            and row["representative_ordered_change_count"] == 0
            and row["hull_ordered_change_count"] == 0
            and row["improvement_filtered_PO_ordered_change_count"] == 0
        )

    summary = {
        "record_type": "summary",
        "state_count": len(state_rows),
        "same_size_membership_change_count": sum(
            int(row["same_size_membership_change_count"]) for row in state_rows
        ),
        "same_size_changed_state_count": sum(
            int(row["same_size_membership_change_count"] > 0) for row in state_rows
        ),
        "representative_ordered_change_count": sum(
            int(row["representative_ordered_change_count"]) for row in state_rows
        ),
        "representative_symmetric_difference_count": sum(
            int(row["representative_symmetric_difference_count"])
            for row in state_rows
        ),
        "representative_changed_state_count": sum(
            int(row["representative_ordered_change_count"] > 0)
            for row in state_rows
        ),
        "hull_ordered_change_count": sum(
            int(row["hull_ordered_change_count"]) for row in state_rows
        ),
        "hull_symmetric_difference_count": sum(
            int(row["hull_symmetric_difference_count"]) for row in state_rows
        ),
        "hull_changed_state_count": sum(
            int(row["hull_ordered_change_count"] > 0) for row in state_rows
        ),
        "improvement_filtered_PO_ordered_change_count": sum(
            int(row["improvement_filtered_PO_ordered_change_count"])
            for row in state_rows
        ),
        "improvement_filtered_PO_symmetric_difference_count": sum(
            int(row["improvement_filtered_PO_symmetric_difference_count"])
            for row in state_rows
        ),
        "improvement_filtered_PO_changed_state_count": sum(
            int(row["improvement_filtered_PO_ordered_change_count"] > 0)
            for row in state_rows
        ),
        "exact_complete_downstream_state_count": sum(
            bool(row["complete_downstream_exact"]) for row in state_rows
        ),
    }
    return [summary, *state_rows], summary


def _relevant_checkpoint_paths() -> list[Path]:
    lower_margin = improvement_and.feasibility_margin.lower_margin
    paths = [
        CANDIDATE_PATH,
        same_size_margin.CANDIDATE_PATH,
        same_size_threshold.THRESHOLD_PATH,
        representative_margin.CANDIDATE_PATH,
        representative_threshold.THRESHOLD_PATH,
        hull_threshold.SKEW_CHECKPOINT_PATH,
        hull_threshold.HOMOGENEOUS_CHECKPOINT_PATH,
        hull_threshold.THRESHOLD_CHECKPOINT_PATH,
        lower_margin.NUMERATOR_PATH,
        lower_margin.LOGQUOTIENT_PATH,
        lower_margin.SLOPE_LOGMAG_PATH,
        lower_margin.SLOPE_SIGN_PATH,
        improvement_and.FEASIBILITY_MARGIN_PATH,
        improvement_and.FEASIBILITY_THRESHOLD_PATH,
        final_and.FINAL_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked checkpoint(s): {missing}")
    return paths


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    relevant_paths = _relevant_checkpoint_paths()
    hashes_before = {str(path): _sha256(path) for path in relevant_paths}
    model, candidate_checkpoint = _load_distance_model()
    locked = _load_downstream()

    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    held_out_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    state_rows: list[dict[str, Any]] = []
    assembled_states: list[dict[str, Any]] = []
    for position, source_index in enumerate(held_out_indices, start=1):
        row, state = _assemble_one_state(
            model, payload, trajectories, int(source_index)
        )
        state_rows.append(row)
        assembled_states.append(state)
        if position % 20 == 0 or position == len(held_out_indices):
            print(
                f"assembled distance validation states {position}/{len(held_out_indices)}",
                flush=True,
            )

    assembled_summary = {
        "record_type": "summary",
        "candidate_checkpoint": str(CANDIDATE_PATH),
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "state_count": len(state_rows),
        "site_count": sum(int(row["site_count"]) for row in state_rows),
        "probe_count": sum(int(row["probe_count"]) for row in state_rows),
        "probe_exact_tie_count": sum(
            int(row["probe_exact_tie_count"]) for row in state_rows
        ),
        "probe_assignment_mismatch_count": sum(
            int(row["probe_assignment_mismatch_count"]) for row in state_rows
        ),
        "maximum_probe_distance_absolute_difference": max(
            float(row["maximum_probe_distance_absolute_difference"])
            for row in state_rows
        ),
        "assigned_radius_mismatch_count": sum(
            int(row["assigned_radius_mismatch_count"]) for row in state_rows
        ),
        "maximum_assigned_radius_absolute_difference": max(
            float(row["maximum_assigned_radius_absolute_difference"])
            for row in state_rows
        ),
        "spacing_exact_tie_count": sum(
            int(row["spacing_exact_tie_count"]) for row in state_rows
        ),
        "nearest_site_winner_mismatch_count": sum(
            int(row["nearest_site_winner_mismatch_count"]) for row in state_rows
        ),
        "spacing_mismatch_count": sum(
            int(row["spacing_mismatch_count"]) for row in state_rows
        ),
        "maximum_spacing_absolute_difference": max(
            float(row["maximum_spacing_absolute_difference"])
            for row in state_rows
        ),
        "final_cell_size_mismatch_count": sum(
            int(row["final_cell_size_mismatch_count"]) for row in state_rows
        ),
        "maximum_final_cell_size_absolute_difference": max(
            float(row["maximum_final_cell_size_absolute_difference"])
            for row in state_rows
        ),
        "exact_complete_state_count": sum(
            bool(row["complete_state_exact"]) for row in state_rows
        ),
    }
    assembled_passed = bool(
        assembled_summary["probe_assignment_mismatch_count"] == 0
        and assembled_summary["maximum_probe_distance_absolute_difference"] == 0.0
        and assembled_summary["assigned_radius_mismatch_count"] == 0
        and assembled_summary["nearest_site_winner_mismatch_count"] == 0
        and assembled_summary["spacing_mismatch_count"] == 0
        and assembled_summary["final_cell_size_mismatch_count"] == 0
        and assembled_summary["exact_complete_state_count"] == len(state_rows)
    )
    assembled_summary["assembled_distance_gate_passed"] = assembled_passed
    _write_csv(ASSEMBLED_PATH, [assembled_summary, *state_rows])
    if not assembled_passed:
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "geometric_distance_metric",
                    "status": "NOT_LOCKED_ASSEMBLED_DISTANCE_GATE_FAILED",
                    "candidate_checkpoint_sha256": CANDIDATE_SHA256,
                }
            ],
        )
        print("assembled distance gate failed; stopped", flush=True)
        return

    downstream_rows, downstream_summary = _downstream_validation(
        assembled_states, locked
    )
    downstream_passed = bool(
        downstream_summary["same_size_membership_change_count"] == 0
        and downstream_summary["representative_ordered_change_count"] == 0
        and downstream_summary["hull_ordered_change_count"] == 0
        and downstream_summary[
            "improvement_filtered_PO_ordered_change_count"
        ]
        == 0
        and downstream_summary["exact_complete_downstream_state_count"]
        == len(assembled_states)
    )
    downstream_rows[0]["downstream_gate_passed"] = downstream_passed
    _write_csv(DOWNSTREAM_PATH, downstream_rows)

    hashes_after = {str(path): _sha256(path) for path in relevant_paths}
    upstream_unchanged = hashes_before == hashes_after
    if not downstream_passed or not upstream_unchanged:
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "geometric_distance_metric",
                    "status": "NOT_LOCKED_DOWNSTREAM_GATE_FAILED",
                    "candidate_checkpoint_sha256": CANDIDATE_SHA256,
                    "downstream_gate_passed": downstream_passed,
                    "upstream_checkpoints_unchanged": upstream_unchanged,
                }
            ],
        )
        print("distance downstream gate failed; stopped", flush=True)
        return

    checkpoint = {
        **candidate_checkpoint,
        "status": "LOCKED",
        "component": "geometric_distance_metric",
        "source_candidate_checkpoint": str(CANDIDATE_PATH),
        "source_candidate_sha256": CANDIDATE_SHA256,
        "assembled_validation_summary": assembled_summary,
        "downstream_validation_summary": downstream_summary,
        "upstream_checkpoints_unchanged": True,
        "training_performed_during_locking": False,
        "nearest_site_assignment_learned": False,
        "minimum_or_maximum_operator_learned": False,
        "never_retrain_without_explicit_request": True,
    }
    torch.save(checkpoint, LOCKED_CHECKPOINT_PATH)
    restored = torch.load(
        LOCKED_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if restored.get("status") != "LOCKED":
        raise RuntimeError("locked distance checkpoint readback failed")
    locked_hash = _sha256(LOCKED_CHECKPOINT_PATH)
    LOCKED_HASH_PATH.write_text(
        f"{locked_hash}  {LOCKED_CHECKPOINT_PATH.name}\n", encoding="utf-8"
    )
    lock_rows = [
        {
            "component": "geometric_distance_metric_candidate",
            "status": "PRESERVED_UNCHANGED",
            "checkpoint": str(CANDIDATE_PATH),
            "checkpoint_sha256": CANDIDATE_SHA256,
        },
        {
            "component": "locked_downstream_components",
            "status": "LOCKED_UNCHANGED",
            "checkpoint_count": len(relevant_paths) - 1,
            "all_hashes_unchanged": upstream_unchanged,
        },
        {
            "component": "geometric_distance_metric",
            "status": "LOCKED",
            "checkpoint": str(LOCKED_CHECKPOINT_PATH),
            "checkpoint_sha256": locked_hash,
            "assembled_exact_states": assembled_summary[
                "exact_complete_state_count"
            ],
            "downstream_exact_states": downstream_summary[
                "exact_complete_downstream_state_count"
            ],
            "state_count": len(assembled_states),
            "training_performed": False,
            "trajectory_search_run": False,
        },
    ]
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        f"geometric distance metric LOCKED states={len(assembled_states)} "
        f"sha256={locked_hash}",
        flush=True,
    )


if __name__ == "__main__":
    run()
