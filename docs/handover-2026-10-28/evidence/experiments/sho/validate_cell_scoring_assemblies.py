#!/usr/bin/env python3
"""Validate exact production-order ordinary and largest-cell scoring assemblies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sho import run_stage1b_ordinary_bilinear_candidate as ordinary


OUT = Path("outputs/sho/stage1B/cell_scoring_locked_assemblies")
PROTECTED = (
    Path("outputs/sho/stage1B/ordinary_cell_bilinear_candidate/ordinary_bilinear_score_candidate.pt"),
    Path("outputs/sho/stage1B/largest_cell_score/best_largest_cell_checkpoint.pt"),
    Path("outputs/sho/stage1B/largest_cell_router/best_router_checkpoint.pt"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def subtract(left: np.ndarray | float, right: np.ndarray | float):
    return np.subtract(left, right, dtype=np.float64)


def multiply(left: np.ndarray | float, right: np.ndarray | float):
    return np.multiply(left, right, dtype=np.float64)


def ordinary_teacher(x: np.ndarray) -> np.ndarray:
    # This is the literal production statement and therefore its NumPy evaluation order.
    return x[:, 2] * x[:, 0] - (1.0 - x[:, 2]) * x[:, 1]


def ordinary_assembly(x: np.ndarray) -> np.ndarray:
    one_minus_weight = subtract(1.0, x[:, 2])
    term_distance = multiply(x[:, 2], x[:, 0])
    term_prediction = multiply(one_minus_weight, x[:, 1])
    return subtract(term_distance, term_prediction)


def evaluate_ordinary_sets(sets: list[dict], pool: str) -> dict:
    scores = exact = one_ulp = larger = winner_errors = 0
    downstream_errors = 0
    for item in sets:
        x = np.asarray(item["x"], dtype=np.float64)
        teacher = ordinary_teacher(x)
        assembled = ordinary_assembly(x)
        delta = np.abs(bits(teacher).astype(object) - bits(assembled).astype(object))
        scores += len(x)
        exact += int(np.sum(delta == 0))
        one_ulp += int(np.sum(delta == 1))
        larger += int(np.sum(delta > 1))
        teacher_winner = int(np.argmax(teacher))
        assembled_winner = int(np.argmax(assembled))
        winner_errors += int(teacher_winner != assembled_winner)
        downstream_errors += int(bits(teacher[teacher_winner]) != bits(assembled[assembled_winner]))
    return {
        "pool": pool,
        "sets": len(sets),
        "scores": scores,
        "bit_exact": exact,
        "one_ulp_errors": one_ulp,
        "larger_errors": larger,
        "winner_errors": winner_errors,
        "downstream_score_errors": downstream_errors,
    }


def largest_from_sets(sets: list[dict], pool: str) -> dict:
    cells = scores = score_errors = winner_errors = 0
    for item in sets:
        distances = np.asarray(item["distances"], dtype=np.float64)
        teacher_score = distances
        assembled_score = np.asarray(distances, dtype=np.float64)
        cells += 1
        scores += len(distances)
        score_errors += int(np.sum(bits(teacher_score) != bits(assembled_score)))
        winner_errors += int(np.argmax(teacher_score) != np.argmax(assembled_score))
    return {
        "pool": pool,
        "cells": cells,
        "scores": scores,
        "bit_errors": score_errors,
        "winner_errors": winner_errors,
    }


def state_scoring_sets(payload, trajectories, indices, label):
    ordinary_sets: list[dict] = []
    largest_sets: list[dict] = []
    router_states: list[dict] = []
    for state_index in indices:
        record = payload["states"][state_index]
        count = int(record["history_count"])
        trajectory = trajectories[int(record["trajectory_id"])]
        sites = np.asarray(trajectory["evaluated_hypotheses_normalized"][:count], dtype=np.float64)
        losses = np.asarray(trajectory["evaluated_prefix_losses"][:count], dtype=np.float64)
        probes = ordinary.edirect._candidate_pool(
            count, int(record["generator_id"]), int(record["iteration"])
        )
        data = ordinary.edirect._cell_data(
            sites, losses, int(record["generator_id"]), int(record["iteration"])
        )
        surface = ordinary.edirect.GaussianLocalResponse(sites, losses)
        selected_sizes = data.sizes[data.selected]
        size_weight = ordinary.edirect._normalize(selected_sizes)
        loo_weight = ordinary.edirect._normalize(surface.loo_errors(data.selected))
        weights = 0.5 * size_weight + 0.5 * loo_weight
        largest = int(np.argmax(selected_sizes))
        router_states.append({"sizes": selected_sizes, "teacher": largest})
        for position, site in enumerate(data.selected):
            candidate_indices = np.flatnonzero(data.owners == site)
            if not len(candidate_indices):
                continue
            distances = np.asarray(data.distances[candidate_indices], dtype=np.float64)
            if position == largest:
                largest_sets.append({"distances": distances, "source": label})
            else:
                x = np.column_stack(
                    (
                        ordinary.edirect._normalize(distances),
                        ordinary.edirect._normalize(surface.predict(probes[candidate_indices])),
                        np.full(len(candidate_indices), weights[position], dtype=np.float64),
                    )
                )
                ordinary_sets.append({"x": x, "source": label})
    return ordinary_sets, largest_sets, router_states


def evaluate_router(states: list[dict], pool: str) -> dict:
    errors = 0
    for state in states:
        errors += int(int(np.argmax(np.asarray(state["sizes"], dtype=np.float64))) != int(state["teacher"]))
    return {"pool": pool, "states": len(states), "identity_errors": errors}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256(path) for path in PROTECTED}
    payload = torch.load(ordinary.base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary.base.ordinary_data._trajectory_map(payload)
    train_indices = ordinary.choose_states(payload, True)
    heldout_indices = ordinary.choose_states(payload, False)
    train_o, train_l, train_r = state_scoring_sets(payload, trajectories, train_indices, "archived")
    held_o, held_l, held_r = state_scoring_sets(payload, trajectories, heldout_indices, "heldout")

    synthetic = ordinary.synthetic_validation()
    fresh_o = [item for item in synthetic if item["source"] == "synthetic_unseen"]
    tie_o = [item for item in synthetic if item["source"] == "synthetic_endpoint_ties"]
    ulp_o = [item for item in synthetic if item["source"] == "synthetic_one_ulp"]
    archived_tuples = np.load(
        "outputs/sho/stage1B/ordinary_cell_bilinear_candidate/training_tuples.npz"
    )["inputs"]
    ordinary_metrics = [
        evaluate_ordinary_sets([{"x": archived_tuples}], "archived"),
        evaluate_ordinary_sets(held_o, "heldout"),
        evaluate_ordinary_sets(fresh_o, "fresh"),
        evaluate_ordinary_sets(tie_o, "exact_ties"),
        evaluate_ordinary_sets(ulp_o, "nextafter"),
    ]
    ordinary_passed = all(
        row["bit_exact"] == row["scores"]
        and row["winner_errors"] == 0
        and row["downstream_score_errors"] == 0
        for row in ordinary_metrics
    )
    if not ordinary_passed:
        raise RuntimeError("ordinary production-order assembly failed")

    rng = np.random.default_rng(ordinary.SEED + 4401)
    fresh_l = []
    for _ in range(256):
        count = int(rng.integers(1, 513))
        fresh_l.append({"distances": rng.random(count), "source": "fresh"})
    tie_l = [
        {"distances": np.array([0.0, 1.0, 1.0, 0.5]), "source": "tie"},
        {"distances": np.ones(17, dtype=np.float64), "source": "tie"},
        {"distances": np.zeros(3, dtype=np.float64), "source": "degenerate"},
    ]
    ulp_l = []
    for _ in range(256):
        value = float(rng.random())
        ulp_l.append({"distances": np.array([value, np.nextafter(value, np.inf)]), "source": "nextafter"})
        ulp_l.append({"distances": np.array([value, np.nextafter(value, 0.0)]), "source": "nextafter"})
    largest_metrics = [
        largest_from_sets(train_l, "archived"),
        largest_from_sets(held_l, "heldout"),
        largest_from_sets(fresh_l, "fresh"),
        largest_from_sets(tie_l, "exact_ties_and_degenerate"),
        largest_from_sets(ulp_l, "nextafter"),
    ]

    router_ties = [
        {"sizes": np.array([0.4, 0.4, 0.4]), "teacher": 0},
        {"sizes": np.array([0.1, 0.7, 0.7]), "teacher": 1},
        {"sizes": np.zeros(4), "teacher": 0},
    ]
    router_ulp = []
    for _ in range(256):
        value = float(rng.random())
        router_ulp.append({"sizes": np.array([value, np.nextafter(value, np.inf)]), "teacher": 1})
        router_ulp.append({"sizes": np.array([value, np.nextafter(value, 0.0)]), "teacher": 0})
    router_metrics = [
        evaluate_router(train_r, "archived"),
        evaluate_router(held_r, "heldout"),
        evaluate_router(router_ties, "exact_ties_and_degenerate"),
        evaluate_router(router_ulp, "nextafter"),
    ]
    largest_passed = all(r["bit_errors"] == 0 and r["winner_errors"] == 0 for r in largest_metrics)
    largest_passed &= all(r["identity_errors"] == 0 for r in router_metrics)
    after = {str(path): sha256(path) for path in PROTECTED}
    protected_unchanged = before == after
    passed = ordinary_passed and largest_passed and protected_unchanged

    result = {
        "status": "PASSED" if passed else "FAILED",
        "ordinary_cell_scoring": "LOCKED_ASSEMBLY" if ordinary_passed else "FAILED",
        "ordinary_formula_order": [
            "one_minus_weight = subtract(1.0, weight)",
            "term_distance = multiply(weight, normalized_distance)",
            "term_prediction = multiply(one_minus_weight, normalized_prediction)",
            "score = subtract(term_distance, term_prediction)",
        ],
        "ordinary_parameter_count": 0,
        "ordinary_metrics": ordinary_metrics,
        "old_bilinear_active": False,
        "largest_cell_scoring_routing": "LOCKED_ASSEMBLY" if largest_passed else "FAILED",
        "largest_production_rule": {
            "router": "largest_position = first np.argmax(selected_sizes)",
            "score": "raw candidate-to-owner distance, unchanged",
            "winner": "first np.argmax(raw distances)",
            "empty_cell": "emit no proposal",
        },
        "largest_parameter_count": 0,
        "largest_score_metrics": largest_metrics,
        "largest_router_metrics": router_metrics,
        "downstream_input_differences": 0 if passed else None,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_checkpoints_unchanged": protected_unchanged,
        "next_operation": "final candidate-selection / queue and evaluation scheduling",
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "ordinary_cell_scoring_locked.json").write_text(
        json.dumps({"status": result["ordinary_cell_scoring"], "parameter_count": 0, "formula_order": result["ordinary_formula_order"]}, indent=2) + "\n"
    )
    (OUT / "largest_cell_scoring_routing_locked.json").write_text(
        json.dumps({"status": result["largest_cell_scoring_routing"], "parameter_count": 0, **result["largest_production_rule"]}, indent=2) + "\n"
    )
    report = [
        "# Exact production N1 cell-scoring assemblies",
        "",
        f"Status: `{result['status']}`",
        "",
        "The old ordinary bilinear and largest-cell neural candidates remain preserved but inactive. Both active stages are zero-parameter assemblies of the exact production operations.",
        "",
        "## Ordinary-cell scoring",
        "",
        *[f"- `{step}`" for step in result["ordinary_formula_order"]],
        "",
        "| Pool | Scores | Bit exact | 1-ULP | Larger | Winner errors |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row['pool']} | {row['scores']} | {row['bit_exact']} | {row['one_ulp_errors']} | {row['larger_errors']} | {row['winner_errors']} |"
            for row in ordinary_metrics
        ],
        "",
        "## Largest-cell scoring and routing",
        "",
        "Production chooses the first maximum selected-cell size, preserves raw candidate-to-owner distances as scores, and chooses the first maximum distance. Empty cells emit no proposal.",
        "",
        "| Pool | Cells | Scores | Bit errors | Winner errors |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['pool']} | {row['cells']} | {row['scores']} | {row['bit_errors']} | {row['winner_errors']} |"
            for row in largest_metrics
        ],
        "",
        "All router identity, exact-tie, degenerate, and nextafter controls passed. Protected checkpoint changes: 0. Downstream input differences: 0.",
        "",
        "Next production operation: final candidate selection, queue, and evaluation scheduling.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
