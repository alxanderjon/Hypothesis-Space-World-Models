"""Classifier-only learning-curve diagnostic for 32-bin continuous-c localization."""

from __future__ import annotations

import copy
import csv
import math
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

import experiments.sho.run_stage1b_learned_continuous_c_localization as previous
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    "outputs/sho/stage1B/tcn_localization_classifier_diagnostic"
)
MAX_EPOCHS = 150
PATIENCE_EPOCHS = 20
EVAL_TOPK_INTERVAL = 10
BATCH_SIZE = previous.BATCH_SIZE
LEARNING_RATE = previous.LEARNING_RATE


def _make_exact_labeled_split(
    n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the prior deterministic split and its simulator c labels."""
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.rand(n, generator=generator, dtype=DTYPE)
    unit = (torch.arange(n, dtype=DTYPE) + offsets) / float(n)
    hidden_c = previous.LOWER + (previous.UPPER - previous.LOWER) * unit
    hidden_c = hidden_c[torch.randperm(n, generator=generator)]
    omegas = torch.acos(hidden_c / 2.0) / float(DELTA_T)
    initial_conditions = 2.0 * torch.rand(
        n, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    times = torch.arange(previous.P, dtype=DTYPE).unsqueeze(0)
    times = times.expand(n, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    labels = previous._region_labels(hidden_c)
    return positions, labels, hidden_c


def _verify_test_labels(labels: torch.Tensor) -> None:
    path = Path(
        "outputs/sho/stage1B/learned_continuous_c_localization/per_test_results.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            "The existing localization test results are required to verify labels"
        )
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    prior = torch.tensor(
        [int(row["true_region"]) - 1 for row in rows], dtype=torch.long
    )
    if prior.shape != labels.shape or not bool(torch.equal(prior, labels)):
        mismatch = (
            -1
            if prior.shape != labels.shape
            else int((prior != labels).sum().item())
        )
        raise RuntimeError(
            f"Reconstructed test labels do not match prior c* labels: {mismatch}"
        )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    positions: torch.Tensor,
    labels: torch.Tensor,
    include_topk: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top2 = 0
    total_top3 = 0
    bin_correct = torch.zeros(previous.REGION_COUNT, dtype=torch.long)
    bin_total = torch.zeros(previous.REGION_COUNT, dtype=torch.long)
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
        for bin_index in range(previous.REGION_COUNT):
            selected = batch_labels == bin_index
            bin_total[bin_index] += int(selected.sum().item())
            bin_correct[bin_index] += int(matches[selected].sum().item())
        if include_topk:
            top = torch.topk(logits, k=3, dim=1).indices
            total_top2 += int(
                (top[:, :2] == batch_labels[:, None]).any(dim=1).sum().item()
            )
            total_top3 += int(
                (top == batch_labels[:, None]).any(dim=1).sum().item()
            )
    count = positions.shape[0]
    result: dict[str, Any] = {
        "loss": total_loss / count,
        "top1": total_correct / count,
        "bin_correct": bin_correct,
        "bin_total": bin_total,
    }
    if include_topk:
        result["top2"] = total_top2 / count
        result["top3"] = total_top3 / count
    return result


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
            (torch.argmax(logits.detach(), dim=1) == labels[indices]).sum().item()
        )
    return (
        total_loss / positions.shape[0],
        total_correct / positions.shape[0],
    )


def _plot_curves(history: list[dict[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        epochs,
        [float(row["train_top1_accuracy"]) for row in history],
        label="train",
    )
    axis.plot(
        epochs,
        [float(row["validation_top1_accuracy"]) for row in history],
        label="validation",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("top-1 accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "learning_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        epochs,
        [float(row["train_cross_entropy"]) for row in history],
        label="train",
    )
    axis.plot(
        epochs,
        [float(row["validation_cross_entropy"]) for row in history],
        label="validation",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, train_labels, _ = _make_exact_labeled_split(
        previous.TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, validation_labels, _ = _make_exact_labeled_split(
        previous.VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels, _ = _make_exact_labeled_split(
        previous.TEST_N, TEST_DATA_SEED + 1_810_300
    )
    _verify_test_labels(test_labels)
    print(
        "Verified exact split reconstruction against prior test c* region labels",
        flush=True,
    )

    torch.manual_seed(previous.MODEL_SEED)
    model = previous.TCNHead(previous.REGION_COUNT).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(previous.MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_train_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    validation_topk_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = _train_epoch(
            model,
            optimizer,
            train_positions,
            train_labels,
            generator,
        )
        include_topk = epoch % EVAL_TOPK_INTERVAL == 0
        validation = _evaluate(
            model,
            validation_positions,
            validation_labels,
            include_topk=include_topk,
        )
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": train_loss,
                "validation_cross_entropy": validation["loss"],
                "train_top1_accuracy": train_accuracy,
                "validation_top1_accuracy": validation["top1"],
            }
        )
        best_train_accuracy = max(best_train_accuracy, train_accuracy)
        if include_topk:
            validation_topk_rows.append(
                {
                    "epoch": epoch,
                    "top2": validation["top2"],
                    "top3": validation["top3"],
                    "per_bin": ";".join(
                        f"{float(correct.item()) / max(int(total.item()), 1):.17g}"
                        for correct, total in zip(
                            validation["bin_correct"], validation["bin_total"]
                        )
                    ),
                }
            )
        improved = float(validation["top1"]) > best_validation_accuracy
        if improved:
            best_validation_accuracy = float(validation["top1"])
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                    "regions": previous.REGION_COUNT,
                    "model_seed": previous.MODEL_SEED,
                    "shuffle_seed": previous.MODEL_SEED + 1,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                },
                OUTPUT_DIR / "best_validation_checkpoint.pt",
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        topk_text = (
            f", top2={validation['top2']:.4f}, top3={validation['top3']:.4f}"
            if include_topk
            else ""
        )
        print(
            f"epoch={epoch:03d} train CE={train_loss:.4e} "
            f"val CE={validation['loss']:.4e} train acc={train_accuracy:.4f} "
            f"val acc={validation['top1']:.4f}{topk_text} "
            f"stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= PATIENCE_EPOCHS:
            print(
                f"Early stopping after {PATIENCE_EPOCHS} epochs without "
                "validation-accuracy improvement",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    best_train = _evaluate(
        model, train_positions, train_labels, include_topk=True
    )
    best_validation = _evaluate(
        model, validation_positions, validation_labels, include_topk=True
    )
    test = _evaluate(model, test_positions, test_labels, include_topk=True)
    edges = previous._region_edges()
    per_bin_rows: list[dict[str, Any]] = []
    for bin_index in range(previous.REGION_COUNT):
        test_total = int(test["bin_total"][bin_index].item())
        test_correct = int(test["bin_correct"][bin_index].item())
        validation_total = int(best_validation["bin_total"][bin_index].item())
        validation_correct = int(
            best_validation["bin_correct"][bin_index].item()
        )
        per_bin_rows.append(
            {
                "bin": bin_index + 1,
                "c_lower": float(edges[bin_index].item()),
                "c_upper": float(edges[bin_index + 1].item()),
                "validation_count": validation_total,
                "validation_correct": validation_correct,
                "validation_accuracy": validation_correct
                / max(validation_total, 1),
                "test_count": test_total,
                "test_correct": test_correct,
                "test_accuracy": test_correct / max(test_total, 1),
                "problematic_test_bin_below_50_percent": (
                    test_correct / max(test_total, 1) < 0.5
                ),
            }
        )

    _plot_curves(history)
    _write_csv(OUTPUT_DIR / "per_bin_accuracy.csv", per_bin_rows)
    problematic = [
        int(row["bin"])
        for row in per_bin_rows
        if bool(row["problematic_test_bin_below_50_percent"])
    ]
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "maximum_online_train_accuracy": best_train_accuracy,
            "best_checkpoint_train_accuracy": best_train["top1"],
            "best_validation_accuracy": best_validation_accuracy,
            "best_checkpoint_validation_accuracy": best_validation["top1"],
            "best_checkpoint_validation_loss": best_validation_loss,
            "test_accuracy": test["top1"],
            "test_top2": test["top2"],
            "test_top3": test["top3"],
            "problematic_test_bins_below_50_percent": problematic,
            "runtime_seconds": time.perf_counter() - start,
            "validation_topk_checkpoints": len(validation_topk_rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
