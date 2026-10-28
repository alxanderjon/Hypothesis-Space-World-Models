"""Strict 2 -> 4 -> 8 -> 16 -> 32 TCN localization curriculum."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from torch import nn

import experiments.sho.run_stage1b_learned_continuous_c_localization as localization
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    "outputs/sho/stage1B/strict_tcn_localization_curriculum"
)
STAGE_BINS = (2, 4, 8, 16, 32)
MAX_EPOCHS_PER_STAGE = 150
PATIENCE_EPOCHS = 20
BATCH_SIZE = localization.BATCH_SIZE
LEARNING_RATE = localization.LEARNING_RATE
PREVIOUS_DIRECT_TOP1 = 0.82421875
PREVIOUS_DIRECT_TOP2 = 0.9375
PREVIOUS_DIRECT_TOP3 = 0.966796875
DIFFICULT_FINAL_BINS = (7, 8, 9)


def _labels_for_bins(c_values: torch.Tensor, bin_count: int) -> torch.Tensor:
    width = (localization.UPPER - localization.LOWER) / float(bin_count)
    labels = torch.floor((c_values - localization.LOWER) / width).to(
        torch.long
    )
    return labels.clamp(0, bin_count - 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    positions: torch.Tensor,
    labels: torch.Tensor,
    bin_count: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top2 = 0
    total_top3 = 0
    bin_correct = torch.zeros(bin_count, dtype=torch.long)
    bin_total = torch.zeros(bin_count, dtype=torch.long)
    topk_count = min(3, bin_count)
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        batch_positions = positions[start:stop]
        batch_labels = labels[start:stop]
        logits = model(batch_positions)
        total_loss += float(
            torch.nn.functional.cross_entropy(
                logits, batch_labels, reduction="sum"
            ).item()
        )
        prediction = torch.argmax(logits, dim=1)
        matches = prediction == batch_labels
        total_correct += int(matches.sum().item())
        top = torch.topk(logits, k=topk_count, dim=1).indices
        if bin_count >= 2:
            total_top2 += int(
                (top[:, :2] == batch_labels[:, None]).any(dim=1).sum().item()
            )
        else:
            total_top2 += int(matches.sum().item())
        if bin_count >= 3:
            total_top3 += int(
                (top[:, :3] == batch_labels[:, None]).any(dim=1).sum().item()
            )
        else:
            total_top3 += positions[start:stop].shape[0]
        for bin_index in range(bin_count):
            selected = batch_labels == bin_index
            bin_total[bin_index] += int(selected.sum().item())
            bin_correct[bin_index] += int(matches[selected].sum().item())
    count = positions.shape[0]
    return {
        "loss": total_loss / count,
        "top1": total_correct / count,
        "top2": total_top2 / count,
        "top3": total_top3 / count,
        "bin_correct": bin_correct,
        "bin_total": bin_total,
    }


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    positions: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[float, float]:
    model.train()
    permutation = torch.randperm(positions.shape[0], generator=generator)
    total_loss = 0.0
    total_correct = 0
    for start in range(0, positions.shape[0], BATCH_SIZE):
        indices = permutation[start : start + BATCH_SIZE]
        logits = model(positions[indices])
        loss = torch.nn.functional.cross_entropy(logits, labels[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * indices.shape[0]
        total_correct += int(
            (torch.argmax(logits.detach(), dim=1) == labels[indices])
            .sum()
            .item()
        )
    return total_loss / positions.shape[0], total_correct / positions.shape[0]


def _expand_head_from_parents(
    model: localization.TCNHead, parent_count: int
) -> None:
    old_head = model.head[-1]
    if not isinstance(old_head, nn.Linear):
        raise TypeError("Expected the final TCN head module to be Linear")
    if old_head.out_features != parent_count:
        raise ValueError(
            f"Expected {parent_count} parent logits, got {old_head.out_features}"
        )
    new_head = nn.Linear(old_head.in_features, 2 * parent_count).to(
        dtype=old_head.weight.dtype, device=old_head.weight.device
    )
    with torch.no_grad():
        new_head.weight[0::2].copy_(old_head.weight)
        new_head.weight[1::2].copy_(old_head.weight)
        new_head.bias[0::2].copy_(old_head.bias)
        new_head.bias[1::2].copy_(old_head.bias)
    model.head[-1] = new_head


def _plot_history(
    history: list[dict[str, Any]], stage_rows: list[dict[str, Any]]
) -> None:
    global_epochs = [int(row["global_epoch"]) for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(
        global_epochs,
        [float(row["train_top1_accuracy"]) for row in history],
        label="train",
    )
    axes[0].plot(
        global_epochs,
        [float(row["validation_top1_accuracy"]) for row in history],
        label="validation",
    )
    axes[0].set_ylabel("top-1 accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower right")
    axes[1].plot(
        global_epochs,
        [float(row["train_cross_entropy"]) for row in history],
        label="train",
    )
    axes[1].plot(
        global_epochs,
        [float(row["validation_cross_entropy"]) for row in history],
        label="validation",
    )
    axes[1].set_xlabel("global curriculum epoch")
    axes[1].set_ylabel("cross-entropy")
    axes[1].legend(loc="upper right")

    for row in stage_rows:
        start = int(row["global_epoch_start"])
        end = int(row["global_epoch_end"])
        for axis in axes:
            axis.axvline(start - 0.5, color="0.65", linestyle="--", linewidth=1)
        midpoint = (start + end) / 2.0
        axes[0].text(
            midpoint,
            1.005,
            f"{int(row['bin_count'])} bins",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if stage_rows:
        final_end = int(stage_rows[-1]["global_epoch_end"])
        for axis in axes:
            axis.axvline(
                final_end + 0.5, color="0.65", linestyle="--", linewidth=1
            )
    figure.suptitle("Strict TCN localization curriculum")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "learning_curve.png", dpi=180)
    plt.close(figure)


def _stage_per_bin_rows(
    bin_count: int, test: dict[str, Any]
) -> list[dict[str, Any]]:
    edges = torch.linspace(
        localization.LOWER,
        localization.UPPER,
        bin_count + 1,
        dtype=DTYPE,
    )
    rows: list[dict[str, Any]] = []
    for bin_index in range(bin_count):
        total = int(test["bin_total"][bin_index].item())
        correct = int(test["bin_correct"][bin_index].item())
        rows.append(
            {
                "bin_count": bin_count,
                "bin": bin_index + 1,
                "c_lower": float(edges[bin_index].item()),
                "c_upper": float(edges[bin_index + 1].item()),
                "test_count": total,
                "test_correct": correct,
                "test_accuracy": correct / max(total, 1),
                "previously_difficult_32_bin_region": (
                    bin_count == 32
                    and bin_index + 1 in DIFFICULT_FINAL_BINS
                ),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, train_labels_32, train_c = direct32._make_exact_labeled_split(
        localization.TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, validation_c = direct32._make_exact_labeled_split(
        localization.VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels_32, test_c = direct32._make_exact_labeled_split(
        localization.TEST_N, TEST_DATA_SEED + 1_810_300
    )
    direct32._verify_test_labels(test_labels_32)
    if not bool(torch.equal(train_labels_32, _labels_for_bins(train_c, 32))):
        raise RuntimeError("Training 32-bin reconstruction mismatch")
    print(
        "Verified exact train/validation/test split reconstruction and prior "
        "test c* regions",
        flush=True,
    )

    torch.manual_seed(localization.MODEL_SEED)
    model = localization.TCNHead(STAGE_BINS[0]).to(dtype=DTYPE)
    shuffle_generator = torch.Generator().manual_seed(
        localization.MODEL_SEED + 1
    )
    history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_bin_rows: list[dict[str, Any]] = []
    global_epoch = 0
    run_start = time.perf_counter()

    for stage_index, bin_count in enumerate(STAGE_BINS, start=1):
        if stage_index > 1:
            _expand_head_from_parents(model, STAGE_BINS[stage_index - 2])
            print(
                f"Expanded {STAGE_BINS[stage_index - 2]} parent logits to "
                f"{bin_count} child logits by exact row duplication",
                flush=True,
            )
        train_labels = _labels_for_bins(train_c, bin_count)
        validation_labels = _labels_for_bins(validation_c, bin_count)
        test_labels = _labels_for_bins(test_c, bin_count)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_state = copy.deepcopy(model.state_dict())
        best_validation_accuracy = -1.0
        best_validation_loss = float("inf")
        maximum_online_train_accuracy = -1.0
        best_epoch = 0
        stale_epochs = 0
        passed = False
        stop_reason = "maximum_epochs"
        stage_start = time.perf_counter()
        global_epoch_start = global_epoch + 1

        for stage_epoch in range(1, MAX_EPOCHS_PER_STAGE + 1):
            global_epoch += 1
            train_loss, train_accuracy = _train_epoch(
                model,
                optimizer,
                train_positions,
                train_labels,
                shuffle_generator,
            )
            validation = _evaluate(
                model, validation_positions, validation_labels, bin_count
            )
            history.append(
                {
                    "stage": stage_index,
                    "bin_count": bin_count,
                    "stage_epoch": stage_epoch,
                    "global_epoch": global_epoch,
                    "train_cross_entropy": train_loss,
                    "validation_cross_entropy": validation["loss"],
                    "train_top1_accuracy": train_accuracy,
                    "validation_top1_accuracy": validation["top1"],
                }
            )
            maximum_online_train_accuracy = max(
                maximum_online_train_accuracy, train_accuracy
            )
            improved = float(validation["top1"]) > best_validation_accuracy
            if improved:
                best_validation_accuracy = float(validation["top1"])
                best_validation_loss = float(validation["loss"])
                best_epoch = stage_epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "bin_count": bin_count,
                        "stage": stage_index,
                        "stage_epoch": stage_epoch,
                        "global_epoch": global_epoch,
                        "validation_accuracy": best_validation_accuracy,
                        "validation_loss": best_validation_loss,
                        "model_seed": localization.MODEL_SEED,
                        "shuffle_seed": localization.MODEL_SEED + 1,
                        "learning_rate": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                    },
                    OUTPUT_DIR
                    / f"best_validation_checkpoint_{bin_count}_bins.pt",
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"bins={bin_count:02d} epoch={stage_epoch:03d} "
                f"train CE={train_loss:.4e} val CE={validation['loss']:.4e} "
                f"train acc={train_accuracy:.4f} "
                f"val acc={validation['top1']:.4f} stale={stale_epochs}",
                flush=True,
            )
            if float(validation["top1"]) == 1.0:
                passed = True
                stop_reason = "validation_reached_100_percent"
                break
            if stale_epochs >= PATIENCE_EPOCHS:
                stop_reason = "20_epochs_without_validation_improvement"
                break

        model.load_state_dict(best_state)
        best_train = _evaluate(model, train_positions, train_labels, bin_count)
        best_validation = _evaluate(
            model, validation_positions, validation_labels, bin_count
        )
        test = _evaluate(model, test_positions, test_labels, bin_count)
        stage_bin_rows = _stage_per_bin_rows(bin_count, test)
        per_bin_rows.extend(stage_bin_rows)
        difficult_accuracy: float | str = ""
        if bin_count == 32:
            selected = [
                row
                for row in stage_bin_rows
                if bool(row["previously_difficult_32_bin_region"])
            ]
            difficult_accuracy = sum(int(row["test_correct"]) for row in selected) / sum(
                int(row["test_count"]) for row in selected
            )
        summary_rows.append(
            {
                "stage": stage_index,
                "bin_count": bin_count,
                "status": "passed" if passed else "failed_stop",
                "stop_reason": stop_reason,
                "epochs_run": stage_epoch,
                "best_epoch": best_epoch,
                "global_epoch_start": global_epoch_start,
                "global_epoch_end": global_epoch,
                "maximum_online_train_accuracy": maximum_online_train_accuracy,
                "best_checkpoint_train_accuracy": best_train["top1"],
                "best_validation_accuracy": best_validation_accuracy,
                "best_checkpoint_validation_accuracy": best_validation["top1"],
                "best_checkpoint_validation_loss": best_validation_loss,
                "test_top1_accuracy": test["top1"],
                "test_top2_coverage": test["top2"] if bin_count == 32 else "",
                "test_top3_coverage": test["top3"] if bin_count == 32 else "",
                "difficult_bins_7_9_test_accuracy": difficult_accuracy,
                "previous_direct_32_top1": (
                    PREVIOUS_DIRECT_TOP1 if bin_count == 32 else ""
                ),
                "previous_direct_32_top2": (
                    PREVIOUS_DIRECT_TOP2 if bin_count == 32 else ""
                ),
                "previous_direct_32_top3": (
                    PREVIOUS_DIRECT_TOP3 if bin_count == 32 else ""
                ),
                "curriculum_minus_direct_top1": (
                    test["top1"] - PREVIOUS_DIRECT_TOP1
                    if bin_count == 32
                    else ""
                ),
                "curriculum_minus_direct_top2": (
                    test["top2"] - PREVIOUS_DIRECT_TOP2
                    if bin_count == 32
                    else ""
                ),
                "curriculum_minus_direct_top3": (
                    test["top3"] - PREVIOUS_DIRECT_TOP3
                    if bin_count == 32
                    else ""
                ),
                "runtime_seconds": time.perf_counter() - stage_start,
            }
        )
        print(
            {
                "bin_count": bin_count,
                "passed": passed,
                "stop_reason": stop_reason,
                "epochs": stage_epoch,
                "best_epoch": best_epoch,
                "best_train": best_train["top1"],
                "best_validation": best_validation["top1"],
                "test": test["top1"],
            },
            flush=True,
        )
        if not passed:
            print(
                f"Strict curriculum stopped at {bin_count} bins; finer stages "
                "will not run",
                flush=True,
            )
            break

    _write_csv(OUTPUT_DIR / "curriculum_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_bin_accuracy.csv", per_bin_rows)
    _plot_history(history, summary_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "attempted_stages": [int(row["bin_count"]) for row in summary_rows],
            "completed_full_curriculum": len(summary_rows) == len(STAGE_BINS)
            and all(row["status"] == "passed" for row in summary_rows),
            "total_epochs": global_epoch,
            "runtime_seconds": time.perf_counter() - run_start,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
