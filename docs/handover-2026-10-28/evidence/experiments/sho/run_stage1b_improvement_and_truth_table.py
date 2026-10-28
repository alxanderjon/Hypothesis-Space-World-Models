#!/usr/bin/env python3
"""Train and validate the final improvement-condition AND on its truth table."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_improvement_and as observed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = observed.OUTPUT_DIR
TRUTH_TABLE_PATH = OUTPUT_DIR / "improvement_and_truth_table.csv"
FULL_PATH = OUTPUT_DIR / "improvement_and_full_pool.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "improvement_condition_operator_lock_summary.csv"
FINAL_PATH = OUTPUT_DIR / "improvement_and_locked.pt"
FINAL_HASH_PATH = FINAL_PATH.with_suffix(".sha256")
ARCHIVE_PATH = OUTPUT_DIR / "improvement_and_observed_domain_only.pt"
ARCHIVE_HASH_PATH = ARCHIVE_PATH.with_suffix(".sha256")

OBSERVED_DOMAIN_SHA256 = (
    "c5496e6d1fd456f6ab011ccd8b520583eec09e90b2870f778324cecdd8305843"
)
MODEL_SEED = observed.MODEL_SEED + 314_159
LEARNING_RATE = 2.0e-2
MAX_EPOCHS = 5_000
CLEAR_LOGIT_MARGIN = 1.0

TRUTH_INPUTS = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    dtype=np.float64,
)
TRUTH_LABELS = np.asarray([False, False, False, True], dtype=bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_observed_domain_checkpoint() -> None:
    """Preserve the prior checkpoint before replacing the final locked path."""
    if ARCHIVE_PATH.exists():
        archive_hash = _sha256(ARCHIVE_PATH)
        if archive_hash != OBSERVED_DOMAIN_SHA256:
            raise RuntimeError(
                "observed-domain archive hash mismatch: "
                f"{archive_hash} != {OBSERVED_DOMAIN_SHA256}"
            )
    else:
        if not FINAL_PATH.exists():
            raise FileNotFoundError(
                "missing observed-domain-only checkpoint to archive: "
                f"{FINAL_PATH}"
            )
        current_hash = _sha256(FINAL_PATH)
        if current_hash != OBSERVED_DOMAIN_SHA256:
            raise RuntimeError(
                "current final checkpoint is not the expected observed-domain model: "
                f"{current_hash} != {OBSERVED_DOMAIN_SHA256}"
            )
        shutil.copy2(FINAL_PATH, ARCHIVE_PATH)
    ARCHIVE_HASH_PATH.write_text(
        f"{OBSERVED_DOMAIN_SHA256}  {ARCHIVE_PATH.name}\n", encoding="utf-8"
    )


def _upstream_hashes() -> dict[str, str]:
    return {
        "slope_sign": _sha256(observed.SLOPE_SIGN_PATH),
        "feasibility_margin": _sha256(observed.FEASIBILITY_MARGIN_PATH),
        "feasibility_threshold": _sha256(observed.FEASIBILITY_THRESHOLD_PATH),
    }


def _truth_metrics(model: nn.Linear) -> tuple[np.ndarray, np.ndarray, float]:
    with torch.no_grad():
        logits = (
            model(torch.as_tensor(TRUTH_INPUTS, dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    predictions = logits >= 0.0
    return logits, predictions, float(np.min(np.abs(logits)))


def _train_truth_table() -> tuple[nn.Linear, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(2, 1, bias=True).to(torch.float64)
    initial_weight = model.weight.detach().cpu().numpy()[0].copy()
    initial_bias = float(model.bias.detach().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    inputs = torch.as_tensor(TRUTH_INPUTS, dtype=torch.float64)
    labels = torch.as_tensor(TRUTH_LABELS.astype(np.float64), dtype=torch.float64)

    stop_epoch = MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        logits = model(inputs).squeeze(-1)
        # Mean BCE gives all four truth-table rows exactly equal weight.
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        learned_logits, predictions, minimum_margin = _truth_metrics(model)
        if np.array_equal(predictions, TRUTH_LABELS) and (
            minimum_margin >= CLEAR_LOGIT_MARGIN
        ):
            stop_epoch = epoch
            break
    else:
        raise RuntimeError(
            "truth-table AND did not reach the clear-margin gate by "
            f"epoch {MAX_EPOCHS}"
        )

    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    learned_logits, predictions, minimum_margin = _truth_metrics(model)
    summary = {
        "record_type": "summary",
        "training_source": "complete_AND_truth_table",
        "equal_row_weighting": True,
        "epoch": stop_epoch,
        "stopping_rule": "all_four_correct_and_min_abs_logit_at_least_1",
        "required_minimum_absolute_logit": CLEAR_LOGIT_MARGIN,
        "learned_w1": float(weight[0]),
        "learned_w2": float(weight[1]),
        "learned_bias": bias,
        "initial_w1": float(initial_weight[0]),
        "initial_w2": float(initial_weight[1]),
        "initial_bias": initial_bias,
        "minimum_absolute_logit_margin": minimum_margin,
        "truth_table_mistakes": int(np.sum(predictions != TRUTH_LABELS)),
        "truth_table_gate_passed": bool(
            np.array_equal(predictions, TRUTH_LABELS)
            and minimum_margin >= CLEAR_LOGIT_MARGIN
        ),
    }
    rows: list[dict[str, Any]] = [summary]
    for index, (inputs_row, teacher, logit, predicted) in enumerate(
        zip(TRUTH_INPUTS, TRUTH_LABELS, learned_logits, predictions)
    ):
        rows.append(
            {
                "record_type": "truth_table_row",
                "row": index,
                "positive_upper": int(inputs_row[0]),
                "feasible": int(inputs_row[1]),
                "teacher_PASS": bool(teacher),
                "logit": float(logit),
                "predicted_PASS": bool(predicted),
                "absolute_logit_margin": float(abs(logit)),
                "correct": bool(predicted == teacher),
            }
        )
    return model, summary, rows


def _save_final_checkpoint(
    model: nn.Linear,
    truth_summary: dict[str, Any],
    full_summary: dict[str, Any],
    upstream_hashes: dict[str, str],
) -> str:
    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    checkpoint = {
        "format_version": 2,
        "component": "improvement_condition_AND",
        "status": "LOCKED_COMPONENT",
        "training_source": "complete_AND_truth_table",
        "architecture": "logit=w1*positive_upper+w2*feasible+b; no hidden layers",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_w1": torch.tensor(float(weight[0]), dtype=torch.float64),
        "learned_w2": torch.tensor(float(weight[1]), dtype=torch.float64),
        "learned_bias": torch.tensor(bias, dtype=torch.float64),
        "clear_logit_margin": CLEAR_LOGIT_MARGIN,
        "truth_table_summary": truth_summary,
        "full_pool_summary": full_summary,
        "upstream_hashes": upstream_hashes,
        "observed_domain_only_archive": str(ARCHIVE_PATH),
        "observed_domain_only_sha256": OBSERVED_DOMAIN_SHA256,
        "complete_improvement_condition_operator_locked": True,
        "new_K_upper_positivity_network_trained": False,
    }
    torch.save(checkpoint, FINAL_PATH)
    restored = torch.load(FINAL_PATH, map_location="cpu", weights_only=False)
    if restored.get("training_source") != "complete_AND_truth_table":
        raise RuntimeError("truth-table checkpoint readback failed")
    checkpoint_hash = _sha256(FINAL_PATH)
    FINAL_HASH_PATH.write_text(
        f"{checkpoint_hash}  {FINAL_PATH.name}\n", encoding="utf-8"
    )
    return checkpoint_hash


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed._verify_frozen_hashes()
    upstream_before = _upstream_hashes()
    _archive_observed_domain_checkpoint()

    model, truth_summary, truth_rows = _train_truth_table()
    observed._write_csv(TRUTH_TABLE_PATH, truth_rows)
    if not bool(truth_summary["truth_table_gate_passed"]):
        raise RuntimeError("truth-table gate failed")

    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    validation_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "truth-table AND validation"
    )
    validation = observed._build_pool(validation_states, "validation")
    full_metrics = observed._metrics(
        model, validation["inputs"], validation["teacher_pass"]
    )
    po_rows, po_summary = observed._learned_po_rows(
        model, validation, validation_states
    )
    upstream_after = _upstream_hashes()
    upstream_unchanged = upstream_before == upstream_after
    full_summary: dict[str, Any] = {
        "record_type": "final_summary",
        "held_out_event_count": len(validation["teacher_pass"]),
        "overall_accuracy": full_metrics["overall_accuracy"],
        "PASS_accuracy": full_metrics["PASS_accuracy"],
        "FAIL_accuracy": full_metrics["FAIL_accuracy"],
        "mistake_count": full_metrics["mistake_count"],
        **po_summary,
        "upstream_checkpoints_unchanged": upstream_unchanged,
        "locked_slope_sign_sha256": upstream_after["slope_sign"],
        "locked_feasibility_margin_sha256": upstream_after["feasibility_margin"],
        "locked_feasibility_threshold_sha256": upstream_after[
            "feasibility_threshold"
        ],
        "observed_domain_only_archive": str(ARCHIVE_PATH),
        "observed_domain_only_sha256": OBSERVED_DOMAIN_SHA256,
    }
    full_summary["full_pool_gate_passed"] = bool(
        full_summary["mistake_count"] == 0
        and full_summary["exact_ordered_PO_states"] == 80
        and full_summary["exact_unordered_PO_states"] == 80
        and full_summary["false_PO_additions"] == 0
        and full_summary["false_PO_removals"] == 0
        and upstream_unchanged
    )
    full_rows: list[dict[str, Any]] = [full_summary, *po_rows]
    full_rows.extend(
        observed._mistake_rows(
            model,
            validation,
            validation["inputs"],
            validation["teacher_pass"],
            "full_held_out_pool",
            np.arange(len(validation["teacher_pass"]), dtype=np.int64),
        )
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        observed._write_csv(FULL_PATH, full_rows)
        observed._write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "improvement_condition_operator",
                    "status": "NOT_LOCKED_FULL_POOL_GATE_FAILED",
                }
            ],
        )
        raise RuntimeError("truth-table AND full-pool gate failed")

    checkpoint_hash = _save_final_checkpoint(
        model, truth_summary, full_summary, upstream_after
    )
    full_summary["AND_checkpoint"] = str(FINAL_PATH)
    full_summary["AND_checkpoint_sha256"] = checkpoint_hash
    full_summary["improvement_condition_operator_status"] = "LOCKED"
    observed._write_csv(FULL_PATH, full_rows)

    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    lock_rows = [
        {
            "component": "observed_domain_only_AND",
            "status": "ARCHIVED_NOT_FINAL",
            "checkpoint": str(ARCHIVE_PATH),
            "checkpoint_sha256": OBSERVED_DOMAIN_SHA256,
        },
        {
            "component": "upstream_components",
            "status": "LOCKED_UNCHANGED",
            "slope_sign_sha256": upstream_after["slope_sign"],
            "feasibility_margin_sha256": upstream_after["feasibility_margin"],
            "feasibility_threshold_sha256": upstream_after[
                "feasibility_threshold"
            ],
            "retrained_or_modified": False,
        },
        {
            "component": "truth_table_AND",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(FINAL_PATH),
            "checkpoint_sha256": checkpoint_hash,
            "learned_w1": float(weight[0]),
            "learned_w2": float(weight[1]),
            "learned_bias": bias,
            "minimum_absolute_logit_margin": truth_summary[
                "minimum_absolute_logit_margin"
            ],
            "truth_table_mistakes": 0,
        },
        {
            "component": "improvement_condition_operator",
            "status": "LOCKED",
            "PASS_FAIL_mistakes": full_summary["mistake_count"],
            "exact_ordered_PO_states": full_summary["exact_ordered_PO_states"],
            "state_count": full_summary["state_count"],
            "upstream_checkpoints_unchanged": upstream_unchanged,
            "new_K_upper_positivity_network_trained": False,
        },
    ]
    observed._write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        "truth-table AND LOCKED "
        f"w1={weight[0]:.17g} w2={weight[1]:.17g} b={bias:.17g} "
        f"sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
