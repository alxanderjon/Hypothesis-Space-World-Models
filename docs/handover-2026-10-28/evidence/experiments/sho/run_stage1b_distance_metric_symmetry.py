#!/usr/bin/env python3
"""Symmetry-supervised smoke test for the general PSD distance metric."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_distance_metric as base
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/distance_metric_symmetry")
SMOKE_PATH = OUTPUT_DIR / "distance_symmetry_smoke.csv"
FULL_PATH = OUTPUT_DIR / "distance_symmetry_full_pool.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "geometric_distance_metric_frozen_candidate.pt"
IDENTITY_DIAGNOSTIC_PATH = Path(
    "outputs/sho/stage1B/distance_identity_diagnostic/"
    "distance_identity_diagnostic.csv"
)

SEED = 176219
MAX_EPOCHS = 5000
PLATEAU_PATIENCE = 1000
ORDERING_CHECK_INTERVAL = 25
REGRESSION_RMSE_GATE = 1.0e-15
REGRESSION_MAX_GATE = 1.0e-14
W_STABILITY_GATE = 1.0e-13


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


def _symmetry_expand(delta: np.ndarray) -> np.ndarray:
    dx = delta[:, 0]
    dy = delta[:, 1]
    variants = np.stack(
        [
            np.stack([dx, dy], axis=1),
            np.stack([-dx, dy], axis=1),
            np.stack([dx, -dy], axis=1),
            np.stack([-dx, -dy], axis=1),
            np.stack([dy, dx], axis=1),
            np.stack([-dy, dx], axis=1),
            np.stack([dy, -dx], axis=1),
            np.stack([-dy, -dx], axis=1),
        ],
        axis=1,
    )
    return variants.reshape(-1, 2)


def _expanded_pool(pool: base.PairPool) -> base.PairPool:
    return base.PairPool(
        delta=_symmetry_expand(pool.delta),
        distance=np.repeat(pool.distance, 8),
        pair_type=np.repeat(pool.pair_type, 8),
        zero_probe_site_count=pool.zero_probe_site_count,
        zero_site_site_count=pool.zero_site_site_count,
    )


@dataclass
class OrderingArrays:
    first_delta: np.ndarray
    second_delta: np.ndarray


def _build_ordering_arrays(
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
) -> OrderingArrays:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for source_index in state_indices:
        sites, probes, _ = base._state_geometry(
            payload, trajectories, int(source_index)
        )
        tree = cKDTree(sites)
        probe_distance, probe_owner = tree.query(probes, k=2)
        comparable = (
            (probe_distance[:, 0] != probe_distance[:, 1])
            & (probe_distance[:, 0] != 0.0)
        )
        items = np.flatnonzero(comparable)
        first.append(probes[items] - sites[probe_owner[items, 0]])
        second.append(probes[items] - sites[probe_owner[items, 1]])

        spacing_distance, spacing_owner = tree.query(sites, k=3)
        comparable = (
            (spacing_distance[:, 1] != spacing_distance[:, 2])
            & (spacing_distance[:, 1] != 0.0)
        )
        items = np.flatnonzero(comparable)
        first.append(sites[items] - sites[spacing_owner[items, 1]])
        second.append(sites[items] - sites[spacing_owner[items, 2]])
    return OrderingArrays(
        first_delta=np.concatenate(first, axis=0),
        second_delta=np.concatenate(second, axis=0),
    )


def _ordering_change_count(
    model: base.QuadraticMetric, arrays: OrderingArrays
) -> int:
    first = base._predict(model, arrays.first_delta)
    second = base._predict(model, arrays.second_delta)
    return int(np.sum(first >= second))


def _symmetry_consistency(
    model: base.QuadraticMetric, original: base.PairPool
) -> dict[str, Any]:
    prediction = base._predict(model, _symmetry_expand(original.delta)).reshape(-1, 8)
    target = original.distance[:, None]
    orbit_range = np.max(prediction, axis=1) - np.min(prediction, axis=1)
    orbit_mean = np.mean(prediction, axis=1, keepdims=True)
    return {
        "symmetry_transform_prediction_rmse": float(
            np.sqrt(np.mean(np.square(prediction - target)))
        ),
        "symmetry_transform_prediction_mae": float(
            np.mean(np.abs(prediction - target))
        ),
        "mean_within_orbit_distance_range": float(np.mean(orbit_range)),
        "maximum_within_orbit_distance_range": float(np.max(orbit_range)),
        "within_orbit_consistency_rmse": float(
            np.sqrt(np.mean(np.square(prediction - orbit_mean)))
        ),
    }


def _prior_reversal_keys() -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    with IDENTITY_DIAGNOSTIC_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("previous_learned_ordering_changed") != "True":
                continue
            keys.add(
                (
                    row["comparison_type"],
                    int(row["source_state_index"]),
                    int(row["item_index"]),
                )
            )
    if len(keys) != 4:
        raise RuntimeError(f"expected four prior reversal keys, found {len(keys)}")
    return keys


def _evaluate_ordering(
    model: base.QuadraticMetric,
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
    split: str,
    replay_keys: set[tuple[str, int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = {
        "probe_raw": 0,
        "probe_unique": 0,
        "probe_ties": 0,
        "probe_zero": 0,
        "probe_changes": 0,
        "spacing_raw": 0,
        "spacing_unique": 0,
        "spacing_ties": 0,
        "spacing_zero": 0,
        "spacing_changes": 0,
    }
    rows: list[dict[str, Any]] = []
    found_replays: set[tuple[str, int, int]] = set()
    for source_index in state_indices:
        sites, probes, record = base._state_geometry(
            payload, trajectories, int(source_index)
        )
        tree = cKDTree(sites)
        trajectory_id = int(record["trajectory_id"])
        for kind in ("probe_nearest_owner", "site_nearest_spacing"):
            if kind == "probe_nearest_owner":
                teacher, owners = tree.query(probes, k=2)
                teacher_first = teacher[:, 0]
                teacher_second = teacher[:, 1]
                first_owner = owners[:, 0]
                second_owner = owners[:, 1]
                points = probes
                first_delta = points - sites[first_owner]
                second_delta = points - sites[second_owner]
                prefix = "probe"
            else:
                teacher, owners = tree.query(sites, k=3)
                teacher_first = teacher[:, 1]
                teacher_second = teacher[:, 2]
                first_owner = owners[:, 1]
                second_owner = owners[:, 2]
                points = sites
                first_delta = points - sites[first_owner]
                second_delta = points - sites[second_owner]
                prefix = "spacing"
            tied = teacher_first == teacher_second
            zero = teacher_first == 0.0
            comparable = ~tied & ~zero
            learned_first = base._predict(model, first_delta)
            learned_second = base._predict(model, second_delta)
            changed = comparable & (learned_first >= learned_second)
            counts[f"{prefix}_raw"] += len(points)
            counts[f"{prefix}_unique"] += int(np.sum(comparable))
            counts[f"{prefix}_ties"] += int(np.sum(tied))
            counts[f"{prefix}_zero"] += int(np.sum(zero))
            counts[f"{prefix}_changes"] += int(np.sum(changed))
            selected = np.flatnonzero(changed)
            selected_set = set(int(value) for value in selected)
            for key in replay_keys:
                if key[0] == kind and key[1] == int(source_index):
                    selected_set.add(key[2])
                    found_replays.add(key)
            for item in sorted(selected_set):
                key = (kind, int(source_index), item)
                rows.append(
                    {
                        "record_type": (
                            "prior_one_ulp_replay"
                            if key in replay_keys
                            else "ordering_failure"
                        ),
                        "split": split,
                        "comparison_type": kind,
                        "source_state_index": int(source_index),
                        "trajectory_id": trajectory_id,
                        "item_index": item,
                        "teacher_first_owner": int(first_owner[item]),
                        "teacher_second_owner": int(second_owner[item]),
                        "teacher_first_distance": float(teacher_first[item]),
                        "teacher_second_distance": float(teacher_second[item]),
                        "teacher_distance_gap": float(
                            teacher_second[item] - teacher_first[item]
                        ),
                        "learned_first_distance": float(learned_first[item]),
                        "learned_second_distance": float(learned_second[item]),
                        "learned_distance_gap": float(
                            learned_second[item] - learned_first[item]
                        ),
                        "ordering_changed": bool(changed[item]),
                        "teacher_exact_tie": bool(tied[item]),
                        "prior_one_ulp_case": key in replay_keys,
                    }
                )
    return {
        "probe_total_item_count": counts["probe_raw"],
        "probe_unique_nonzero_comparison_count": counts["probe_unique"],
        "probe_exact_tie_deterministic_count": counts["probe_ties"],
        "probe_zero_distance_exclusion_count": counts["probe_zero"],
        "unique_probe_owner_change_count": counts["probe_changes"],
        "probe_ordering_agreement": 1.0
        - counts["probe_changes"] / counts["probe_unique"],
        "spacing_total_item_count": counts["spacing_raw"],
        "spacing_unique_nonzero_comparison_count": counts["spacing_unique"],
        "spacing_exact_tie_deterministic_count": counts["spacing_ties"],
        "spacing_zero_distance_exclusion_count": counts["spacing_zero"],
        "nearest_site_spacing_winner_change_count": counts["spacing_changes"],
        "spacing_ordering_agreement": 1.0
        - counts["spacing_changes"] / counts["spacing_unique"],
        "prior_one_ulp_replays_found": len(found_replays),
        "prior_one_ulp_replays_preserved": sum(
            row["prior_one_ulp_case"] and not row["ordering_changed"] for row in rows
        ),
    }, rows


def _train(
    training: base.PairPool,
    validation: base.PairPool,
    ordering_arrays: OrderingArrays,
) -> tuple[base.QuadraticMetric, dict[str, Any]]:
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    model = base.QuadraticMetric(generator)
    initial_l = model.L.detach().clone()
    initial_w = model.metric_matrix().detach().clone()
    train_x = torch.from_numpy(training.delta)
    train_y = torch.from_numpy(training.distance)
    validation_x = torch.from_numpy(validation.delta)
    validation_y = torch.from_numpy(validation.distance)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-2)

    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[int, float] = (2**63 - 1, math.inf)
    best_epoch = 0
    best_order_changes = 2**63 - 1
    best_validation_rmse = math.inf
    best_validation_max = math.inf
    best_regression_epoch = 0
    best_regression_rmse = math.inf
    previous_check_w: np.ndarray | None = None
    last_w_change = math.inf
    stop_epoch = MAX_EPOCHS
    stop_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        loss = torch.mean(torch.square(prediction - train_y))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_prediction = model(validation_x)
            error = validation_prediction - validation_y
            validation_rmse = float(
                torch.sqrt(torch.mean(torch.square(error))).item()
            )
            validation_max = float(torch.max(torch.abs(error)).item())
        if validation_rmse < best_regression_rmse:
            best_regression_rmse = validation_rmse
            best_regression_epoch = epoch
        if epoch % ORDERING_CHECK_INTERVAL != 0 and epoch != 1:
            continue
        order_changes = _ordering_change_count(model, ordering_arrays)
        current_w = model.metric_matrix().detach().cpu().numpy()
        if previous_check_w is not None:
            last_w_change = float(np.max(np.abs(current_w - previous_check_w)))
        previous_check_w = current_w.copy()
        key = (order_changes, validation_rmse)
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_order_changes = order_changes
            best_validation_rmse = validation_rmse
            best_validation_max = validation_max
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 250 == 0:
            print(
                f"symmetry metric epoch={epoch} val_rmse={validation_rmse:.3e} "
                f"val_max={validation_max:.3e} ordering_changes={order_changes} "
                f"W_change={last_w_change:.3e}",
                flush=True,
            )
        if (
            order_changes == 0
            and validation_rmse <= REGRESSION_RMSE_GATE
            and validation_max <= REGRESSION_MAX_GATE
            and last_w_change <= W_STABILITY_GATE
        ):
            stop_epoch = epoch
            stop_reason = "exact_ordering_and_stable_W"
            break
        if epoch - best_regression_epoch >= PLATEAU_PATIENCE:
            stop_epoch = epoch
            stop_reason = "validation_plateau"
            break
    if best_state is None:
        raise RuntimeError("symmetry training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "initial_L": initial_l.numpy(),
        "initial_W": initial_w.numpy(),
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "best_ordering_change_count": best_order_changes,
        "best_validation_rmse": best_validation_rmse,
        "best_validation_max_error": best_validation_max,
        "last_checked_W_change": last_w_change,
    }


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    source_train = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    source_held_out = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    rng = np.random.default_rng(SEED)
    shuffled = np.asarray(source_train, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    train_states = shuffled[: base.SMOKE_TRAIN_STATE_COUNT].tolist()
    validation_states = shuffled[
        base.SMOKE_TRAIN_STATE_COUNT : base.SMOKE_TRAIN_STATE_COUNT
        + base.SMOKE_VALIDATION_STATE_COUNT
    ].tolist()
    if set(train_states) & set(validation_states):
        raise RuntimeError("smoke state split overlaps")
    if (set(train_states) | set(validation_states)) & set(source_held_out):
        raise RuntimeError("smoke and full held-out state pools overlap")

    train_base = base._build_smoke_pool(
        payload,
        trajectories,
        train_states,
        base.TRAIN_PAIRS_PER_TYPE_PER_STATE,
        SEED + 1,
    )
    validation_base = base._build_smoke_pool(
        payload,
        trajectories,
        validation_states,
        base.VALIDATION_PAIRS_PER_TYPE_PER_STATE,
        SEED + 2,
    )
    training = _expanded_pool(train_base)
    validation = _expanded_pool(validation_base)
    ordering_arrays = _build_ordering_arrays(
        payload, trajectories, validation_states
    )
    model, training_info = _train(training, validation, ordering_arrays)
    train_prediction = base._predict(model, training.delta)
    validation_prediction = base._predict(model, validation.delta)
    smoke_metrics = base._metrics(validation.distance, validation_prediction)
    symmetry_metrics = _symmetry_consistency(model, validation_base)
    replay_keys = _prior_reversal_keys()
    smoke_ordering, smoke_failure_rows = _evaluate_ordering(
        model,
        payload,
        trajectories,
        validation_states,
        "smoke_validation",
        set(),
    )
    replay_state_indices = sorted({key[1] for key in replay_keys})
    replay_metrics, replay_rows = _evaluate_ordering(
        model,
        payload,
        trajectories,
        replay_state_indices,
        "prior_one_ulp_replay",
        replay_keys,
    )
    L = model.L.detach().cpu().numpy()
    W = model.metric_matrix().detach().cpu().numpy()
    smoke_passed = bool(
        smoke_metrics["rmse"] <= REGRESSION_RMSE_GATE
        and smoke_metrics["maximum_absolute_error"] <= REGRESSION_MAX_GATE
        and smoke_ordering["unique_probe_owner_change_count"] == 0
        and smoke_ordering["nearest_site_spacing_winner_change_count"] == 0
        and replay_metrics["prior_one_ulp_replays_found"] == 4
        and replay_metrics["prior_one_ulp_replays_preserved"] == 4
    )
    smoke_summary = {
        "record_type": "summary",
        "status": "passed" if smoke_passed else "failed",
        "training_state_count": len(train_states),
        "validation_state_count": len(validation_states),
        "base_training_pair_count": len(train_base.distance),
        "symmetry_augmented_training_pair_count": len(training.distance),
        "base_validation_pair_count": len(validation_base.distance),
        "symmetry_augmented_validation_pair_count": len(validation.distance),
        "training_zero_distance_exclusion_count": (
            train_base.zero_probe_site_count + train_base.zero_site_site_count
        ),
        "validation_zero_distance_exclusion_count": (
            validation_base.zero_probe_site_count
            + validation_base.zero_site_site_count
        ),
        "initial_W_00": training_info["initial_W"][0, 0],
        "initial_W_01": training_info["initial_W"][0, 1],
        "initial_W_10": training_info["initial_W"][1, 0],
        "initial_W_11": training_info["initial_W"][1, 1],
        "learned_L_00": L[0, 0],
        "learned_L_01": L[0, 1],
        "learned_L_10": L[1, 0],
        "learned_L_11": L[1, 1],
        "learned_W_00": W[0, 0],
        "learned_W_01": W[0, 1],
        "learned_W_10": W[1, 0],
        "learned_W_11": W[1, 1],
        "learned_W_diagonal_difference": abs(W[0, 0] - W[1, 1]),
        "learned_W_off_diagonal_magnitude": abs(W[0, 1]),
        "best_epoch": training_info["best_epoch"],
        "stop_epoch": training_info["stop_epoch"],
        "stop_reason": training_info["stop_reason"],
        "last_checked_W_change": training_info["last_checked_W_change"],
        **smoke_metrics,
        **symmetry_metrics,
        **{f"smoke_{key}": value for key, value in smoke_ordering.items()},
        "prior_one_ulp_replays_found": replay_metrics[
            "prior_one_ulp_replays_found"
        ],
        "prior_one_ulp_replays_preserved": replay_metrics[
            "prior_one_ulp_replays_preserved"
        ],
    }
    smoke_rows: list[dict[str, Any]] = [smoke_summary]
    smoke_rows.extend(base._metric_rows(training, train_prediction, "training_augmented"))
    smoke_rows.extend(
        base._metric_rows(validation, validation_prediction, "validation_augmented")
    )
    smoke_rows.extend(smoke_failure_rows)
    smoke_rows.extend(replay_rows)
    _write_csv(SMOKE_PATH, smoke_rows)
    print(
        f"symmetry smoke status={smoke_summary['status']} "
        f"rmse={smoke_metrics['rmse']:.3e} max={smoke_metrics['maximum_absolute_error']:.3e} "
        f"probe_changes={smoke_ordering['unique_probe_owner_change_count']} "
        f"spacing_changes={smoke_ordering['nearest_site_spacing_winner_change_count']} "
        f"replays={replay_metrics['prior_one_ulp_replays_preserved']}/4",
        flush=True,
    )
    if not smoke_passed:
        return

    full_pool = base._full_pool(model, payload, trajectories, source_held_out)
    full_prediction = base._predict(model, full_pool.delta)
    full_metrics = base._metrics(full_pool.distance, full_prediction)
    full_ordering, full_failure_rows = _evaluate_ordering(
        model,
        payload,
        trajectories,
        source_held_out,
        "full_held_out",
        replay_keys,
    )
    full_passed = bool(
        full_ordering["unique_probe_owner_change_count"] == 0
        and full_ordering["nearest_site_spacing_winner_change_count"] == 0
        and full_ordering["prior_one_ulp_replays_found"] == 4
        and full_ordering["prior_one_ulp_replays_preserved"] == 4
    )
    checkpoint_hash = ""
    if full_passed:
        torch.save(
            {
                "component": "geometric_distance_metric",
                "status": "FROZEN_CANDIDATE",
                "training": "eightfold_displacement_symmetry_supervision",
                "model_definition": "d_hat=sqrt(delta^T W delta), W=L L^T",
                "dtype": "float64",
                "state_dict": model.state_dict(),
                "L": torch.from_numpy(L.copy()),
                "W": torch.from_numpy(W.copy()),
                "seed": SEED,
                "smoke_training_state_indices": train_states,
                "smoke_validation_state_indices": validation_states,
                "full_held_out_state_indices": source_held_out,
                "exact_tie_policy": "deterministic_ckdtree_bookkeeping",
                "smoke_summary": smoke_summary,
                "full_metrics": full_metrics,
                "full_ordering": full_ordering,
            },
            CHECKPOINT_PATH,
        )
        checkpoint_hash = _sha256(CHECKPOINT_PATH)
    full_summary = {
        "record_type": "summary",
        "status": "FROZEN_CANDIDATE" if full_passed else "failed",
        "held_out_state_count": len(source_held_out),
        "full_pair_count": len(full_pool.distance),
        "probe_site_pair_count": int(np.sum(full_pool.pair_type == "probe_to_site")),
        "site_site_pair_count": int(np.sum(full_pool.pair_type == "site_to_site")),
        "probe_site_zero_distance_exclusion_count": full_pool.zero_probe_site_count,
        "site_site_zero_distance_exclusion_count": full_pool.zero_site_site_count,
        "learned_W_00": W[0, 0],
        "learned_W_01": W[0, 1],
        "learned_W_10": W[1, 0],
        "learned_W_11": W[1, 1],
        "learned_W_diagonal_difference": abs(W[0, 0] - W[1, 1]),
        "learned_W_off_diagonal_magnitude": abs(W[0, 1]),
        **full_metrics,
        **full_ordering,
        "checkpoint_path": str(CHECKPOINT_PATH) if full_passed else "",
        "checkpoint_sha256": checkpoint_hash,
    }
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(base._metric_rows(full_pool, full_prediction, "full_held_out"))
    full_rows.extend(full_failure_rows)
    _write_csv(FULL_PATH, full_rows)
    print(
        f"symmetry full status={full_summary['status']} pairs={len(full_pool.distance)} "
        f"probe_changes={full_ordering['unique_probe_owner_change_count']} "
        f"spacing_changes={full_ordering['nearest_site_spacing_winner_change_count']}",
        flush=True,
    )


if __name__ == "__main__":
    run()
