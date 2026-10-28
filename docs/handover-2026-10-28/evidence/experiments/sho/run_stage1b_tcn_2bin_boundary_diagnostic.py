"""Replay the best 2-bin TCN and diagnose its learned decision boundary."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar

import experiments.sho.run_stage1b_learned_continuous_c_localization as localization
import experiments.sho.run_stage1b_strict_tcn_localization_curriculum as curriculum
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_continuous_c_search_comparison import _losses
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/tcn_2bin_boundary_diagnostic")
CHECKPOINT_PATH = OUTPUT_DIR / "best_2_bin_checkpoint.pt"
REPLAY_EPOCHS = 21
EXPECTED_VALIDATION_ACCURACY = 0.99609375
EXPECTED_TEST_ACCURACY = 0.99609375
EXPECTED_TRAIN_ACCURACY = 0.99462890625
EXPECTED_VALIDATION_LOSS = 0.03820084142802637
FIDELITY_LOSS_TOLERANCE = 1e-12
EXCLUSION_THRESHOLDS = (1e-3, 1e-2, 5e-2, 1e-1, 2e-1)
PROBE_C_COUNT = 401
PROBE_INITIAL_CONDITIONS = 64
PROBE_SEED = localization.MODEL_SEED + 2_021_000
BRENT_XATOL = 1e-14
BRENT_MAXITER = 1_000


def _stored_test_c_star() -> torch.Tensor:
    path = Path(
        "outputs/sho/stage1B/learned_continuous_c_localization/"
        "per_test_results.csv"
    )
    with path.open(encoding="utf-8", newline="") as file:
        rows = sorted(
            csv.DictReader(file), key=lambda row: int(row["trajectory_index"])
        )
    return torch.tensor([float(row["c_star"]) for row in rows], dtype=DTYPE)


@torch.no_grad()
def _predict(model: torch.nn.Module, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    logits_parts: list[torch.Tensor] = []
    for start in range(0, positions.shape[0], curriculum.BATCH_SIZE):
        stop = min(start + curriculum.BATCH_SIZE, positions.shape[0])
        logits_parts.append(model(positions[start:stop]))
    logits = torch.cat(logits_parts, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "prediction": torch.argmax(logits, dim=1),
    }


def _trajectory_rows(
    split: str,
    reference_c: torch.Tensor,
    reference_source: str,
    predictions: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    true_bin = (reference_c >= 0.0).to(torch.long)
    rows: list[dict[str, Any]] = []
    for index in range(reference_c.shape[0]):
        c_value = float(reference_c[index].item())
        predicted = int(predictions["prediction"][index].item())
        truth = int(true_bin[index].item())
        logit_negative = float(predictions["logits"][index, 0].item())
        logit_positive = float(predictions["logits"][index, 1].item())
        probability_negative = float(
            predictions["probabilities"][index, 0].item()
        )
        probability_positive = float(
            predictions["probabilities"][index, 1].item()
        )
        rows.append(
            {
                "row_type": "trajectory",
                "split": split,
                "trajectory_index": index,
                "reference_c_star": c_value,
                "reference_c_star_source": reference_source,
                "true_bin": truth,
                "predicted_bin": predicted,
                "correct": predicted == truth,
                "logit_negative": logit_negative,
                "logit_positive": logit_positive,
                "probability_negative": probability_negative,
                "probability_positive": probability_positive,
                "distance_to_zero": abs(c_value),
                "probability_confidence_margin": abs(
                    probability_positive - probability_negative
                ),
                "logit_confidence_margin": abs(
                    logit_positive - logit_negative
                ),
                "misclassified": predicted != truth,
            }
        )
    return rows


def _exclusion_rows(
    split: str,
    reference_c: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    truth = (reference_c >= 0.0).to(torch.long)
    correct = predictions["prediction"] == truth
    rows: list[dict[str, Any]] = []
    for threshold in EXCLUSION_THRESHOLDS:
        included = torch.abs(reference_c) >= threshold
        included_count = int(included.sum().item())
        rows.append(
            {
                "row_type": "exclusion_accuracy",
                "split": split,
                "exclusion_threshold_abs_c": threshold,
                "included_count": included_count,
                "excluded_count": int((~included).sum().item()),
                "accuracy_after_exclusion": float(
                    correct[included].to(DTYPE).mean().item()
                )
                if included_count
                else float("nan"),
                "remaining_error_count": int((~correct[included]).sum().item())
                if included_count
                else 0,
            }
        )
    return rows


def _probe_positions() -> tuple[torch.Tensor, torch.Tensor]:
    c_grid = torch.linspace(-0.2, 0.2, PROBE_C_COUNT, dtype=DTYPE)
    generator = torch.Generator().manual_seed(PROBE_SEED)
    initial_conditions = 2.0 * torch.rand(
        PROBE_INITIAL_CONDITIONS, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    flat_c = c_grid.repeat_interleave(PROBE_INITIAL_CONDITIONS)
    flat_initial_conditions = initial_conditions.repeat(PROBE_C_COUNT, 1)
    omegas = torch.acos(flat_c / 2.0) / float(DELTA_T)
    times = torch.arange(localization.P, dtype=DTYPE).unsqueeze(0)
    times = times.expand(flat_c.shape[0], -1) * float(DELTA_T)
    positions, _ = _state_at_time(flat_initial_conditions, omegas, times)
    return c_grid, positions


def _interpolated_crossing(
    c_grid: torch.Tensor, values: torch.Tensor
) -> float | None:
    candidates: list[float] = []
    for index in range(c_grid.shape[0] - 1):
        left = float(values[index].item())
        right = float(values[index + 1].item())
        if left == 0.0:
            candidates.append(float(c_grid[index].item()))
        elif left * right < 0.0:
            c_left = float(c_grid[index].item())
            c_right = float(c_grid[index + 1].item())
            candidates.append(c_left - left * (c_right - c_left) / (right - left))
    if float(values[-1].item()) == 0.0:
        candidates.append(float(c_grid[-1].item()))
    return min(candidates, key=abs) if candidates else None


def _probe(
    model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    c_grid, positions = _probe_positions()
    prediction = _predict(model, positions)
    logits = prediction["logits"].reshape(
        PROBE_C_COUNT, PROBE_INITIAL_CONDITIONS, 2
    )
    probabilities = prediction["probabilities"].reshape(
        PROBE_C_COUNT, PROBE_INITIAL_CONDITIONS, 2
    )
    margins = logits[:, :, 1] - logits[:, :, 0]
    positive_probability = probabilities[:, :, 1]
    fraction_positive = (margins > 0.0).to(DTYPE).mean(dim=1)
    mean_probability = positive_probability.mean(dim=1)
    probability_q10 = torch.quantile(positive_probability, 0.1, dim=1)
    probability_q90 = torch.quantile(positive_probability, 0.9, dim=1)
    mean_margin = margins.mean(dim=1)
    aggregate_boundary = _interpolated_crossing(c_grid, mean_margin)
    per_initial_condition_boundaries = [
        boundary
        for initial_condition in range(PROBE_INITIAL_CONDITIONS)
        if (
            boundary := _interpolated_crossing(
                c_grid, margins[:, initial_condition]
            )
        )
        is not None
    ]
    boundary_tensor = torch.tensor(
        per_initial_condition_boundaries, dtype=DTYPE
    )
    any_disagreement = (fraction_positive > 0.0) & (fraction_positive < 1.0)
    central_disagreement = (fraction_positive >= 0.05) & (
        fraction_positive <= 0.95
    )

    def interval(mask: torch.Tensor) -> tuple[float, float, float]:
        selected = c_grid[mask]
        if selected.numel() == 0:
            return float("nan"), float("nan"), 0.0
        lower = float(selected.min().item())
        upper = float(selected.max().item())
        return lower, upper, upper - lower

    disagreement_lower, disagreement_upper, disagreement_width = interval(
        any_disagreement
    )
    central_lower, central_upper, central_width = interval(central_disagreement)
    summary = {
        "effective_boundary_mean_logit_crossing": aggregate_boundary,
        "initial_condition_boundary_count": len(
            per_initial_condition_boundaries
        ),
        "initial_condition_boundary_mean": float(boundary_tensor.mean().item()),
        "initial_condition_boundary_median": float(
            boundary_tensor.median().item()
        ),
        "initial_condition_boundary_std": float(
            boundary_tensor.std(unbiased=False).item()
        ),
        "initial_condition_boundary_min": float(boundary_tensor.min().item()),
        "initial_condition_boundary_max": float(boundary_tensor.max().item()),
        "any_initial_condition_disagreement_lower": disagreement_lower,
        "any_initial_condition_disagreement_upper": disagreement_upper,
        "any_initial_condition_disagreement_width": disagreement_width,
        "five_to_95_percent_disagreement_lower": central_lower,
        "five_to_95_percent_disagreement_upper": central_upper,
        "five_to_95_percent_disagreement_width": central_width,
    }
    rows: list[dict[str, Any]] = []
    for index in range(PROBE_C_COUNT):
        rows.append(
            {
                "row_type": "decision_boundary_probe",
                "probe_c": float(c_grid[index].item()),
                "probe_initial_condition_count": PROBE_INITIAL_CONDITIONS,
                "fraction_classified_positive": float(
                    fraction_positive[index].item()
                ),
                "mean_probability_positive": float(
                    mean_probability[index].item()
                ),
                "probability_positive_q10": float(
                    probability_q10[index].item()
                ),
                "probability_positive_q90": float(
                    probability_q90[index].item()
                ),
                "mean_positive_logit_margin": float(mean_margin[index].item()),
            }
        )
    rows.append({"row_type": "decision_boundary_summary", **summary})
    plot_data = {
        "c_grid": c_grid,
        "fraction_positive": fraction_positive,
        "mean_probability": mean_probability,
        "probability_q10": probability_q10,
        "probability_q90": probability_q90,
    }
    return rows, summary, plot_data


def _bounded_brent(
    positions: torch.Tensor, bounds: tuple[float, float]
) -> dict[str, Any]:
    evaluations = 0

    def objective(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        candidate = torch.tensor([value], dtype=DTYPE)
        return float(_losses(candidate, positions.unsqueeze(0))[0].item())

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    return {
        "recovered_c": float(result.x),
        "prefix_loss": float(result.fun),
        "prefix_loss_evaluations": evaluations,
        "brent_success": bool(result.success),
        "brent_message": str(result.message),
    }


def _brent_rows(
    test_positions: torch.Tensor,
    test_reference_c: torch.Tensor,
    test_predictions: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    true_bin = (test_reference_c >= 0.0).to(torch.long)
    misclassified = torch.nonzero(
        test_predictions["prediction"] != true_bin, as_tuple=False
    ).squeeze(1)
    if misclassified.numel() != 2:
        raise RuntimeError(
            f"Faithful replay should have 2 test errors, got {misclassified.numel()}"
        )
    rows: list[dict[str, Any]] = []
    halves = {
        0: (localization.LOWER, 0.0),
        1: (0.0, localization.UPPER),
    }
    for tensor_index in misclassified:
        index = int(tensor_index.item())
        truth = int(true_bin[index].item())
        predicted = int(test_predictions["prediction"][index].item())
        reference = float(test_reference_c[index].item())
        for restriction, selected_bin in (
            ("predicted_wrong_half", predicted),
            ("correct_half", truth),
        ):
            bounds = halves[selected_bin]
            result = _bounded_brent(test_positions[index], bounds)
            rows.append(
                {
                    "row_type": "brent_half_comparison",
                    "split": "test",
                    "trajectory_index": index,
                    "reference_c_star": reference,
                    "true_bin": truth,
                    "predicted_bin": predicted,
                    "brent_restriction": restriction,
                    "brent_lower": bounds[0],
                    "brent_upper": bounds[1],
                    **result,
                    "absolute_c_error": abs(result["recovered_c"] - reference),
                }
            )
    return rows


def _plot_probe(
    plot_data: dict[str, torch.Tensor],
    summary: dict[str, Any],
    misclassified_test_c: torch.Tensor,
) -> None:
    c_grid = plot_data["c_grid"].numpy()
    mean_probability = plot_data["mean_probability"].numpy()
    q10 = plot_data["probability_q10"].numpy()
    q90 = plot_data["probability_q90"].numpy()
    fraction_positive = plot_data["fraction_positive"].numpy()
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.fill_between(
        c_grid,
        q10,
        q90,
        color="tab:blue",
        alpha=0.2,
        label="10th–90th percentile across initial conditions",
    )
    axis.plot(
        c_grid,
        mean_probability,
        color="tab:blue",
        linewidth=2,
        label="mean positive-class probability",
    )
    axis.plot(
        c_grid,
        fraction_positive,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="fraction classified positive",
    )
    axis.axvline(0.0, color="black", linestyle=":", label="true boundary")
    learned_boundary = summary["effective_boundary_mean_logit_crossing"]
    if learned_boundary is not None:
        axis.axvline(
            float(learned_boundary),
            color="tab:red",
            linestyle="--",
            label="effective mean boundary",
        )
    for index, c_value in enumerate(misclassified_test_c.tolist()):
        axis.axvline(
            c_value,
            color="tab:purple",
            alpha=0.6,
            linewidth=1,
            label="misclassified test c*" if index == 0 else None,
        )
    axis.set_xlim(-0.2, 0.2)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("c")
    axis.set_ylabel("positive-class probability / fraction")
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "probability_positive_vs_c.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_c = direct32._make_exact_labeled_split(
        localization.TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, validation_c = direct32._make_exact_labeled_split(
        localization.VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels_32, test_hidden_c = direct32._make_exact_labeled_split(
        localization.TEST_N, TEST_DATA_SEED + 1_810_300
    )
    direct32._verify_test_labels(test_labels_32)
    test_c_star = _stored_test_c_star()
    maximum_test_teacher_difference = float(
        torch.max(torch.abs(test_c_star - test_hidden_c)).item()
    )
    train_labels = curriculum._labels_for_bins(train_c, 2)
    validation_labels = curriculum._labels_for_bins(validation_c, 2)
    test_labels = curriculum._labels_for_bins(test_c_star, 2)

    torch.manual_seed(localization.MODEL_SEED)
    model = localization.TCNHead(2).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=curriculum.LEARNING_RATE)
    shuffle_generator = torch.Generator().manual_seed(
        localization.MODEL_SEED + 1
    )
    for epoch in range(1, REPLAY_EPOCHS + 1):
        train_loss, train_accuracy = curriculum._train_epoch(
            model,
            optimizer,
            train_positions,
            train_labels,
            shuffle_generator,
        )
        validation = curriculum._evaluate(
            model, validation_positions, validation_labels, 2
        )
        print(
            f"replay epoch={epoch:02d} train CE={train_loss:.4e} "
            f"val CE={validation['loss']:.4e} train acc={train_accuracy:.4f} "
            f"val acc={validation['top1']:.4f}",
            flush=True,
        )

    replay_train = curriculum._evaluate(model, train_positions, train_labels, 2)
    replay_validation = curriculum._evaluate(
        model, validation_positions, validation_labels, 2
    )
    replay_test = curriculum._evaluate(model, test_positions, test_labels, 2)
    faithful = (
        replay_train["top1"] == EXPECTED_TRAIN_ACCURACY
        and replay_validation["top1"] == EXPECTED_VALIDATION_ACCURACY
        and replay_test["top1"] == EXPECTED_TEST_ACCURACY
        and abs(replay_validation["loss"] - EXPECTED_VALIDATION_LOSS)
        <= FIDELITY_LOSS_TOLERANCE
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": "existing_stage1b_tcn",
        "output_bins": 2,
        "epoch": REPLAY_EPOCHS,
        "model_seed": localization.MODEL_SEED,
        "shuffle_seed": localization.MODEL_SEED + 1,
        "learning_rate": curriculum.LEARNING_RATE,
        "batch_size": curriculum.BATCH_SIZE,
        "train_accuracy": replay_train["top1"],
        "validation_accuracy": replay_validation["top1"],
        "validation_loss": replay_validation["loss"],
        "test_accuracy": replay_test["top1"],
        "faithful_replay": faithful,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(
        {
            "checkpoint": str(CHECKPOINT_PATH),
            "faithful": faithful,
            "train_accuracy": replay_train["top1"],
            "validation_accuracy": replay_validation["top1"],
            "validation_loss": replay_validation["loss"],
            "test_accuracy": replay_test["top1"],
        },
        flush=True,
    )
    if not faithful:
        raise RuntimeError("Epoch-21 replay did not faithfully reproduce the prior run")

    frozen = localization.TCNHead(2).to(dtype=DTYPE)
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    frozen.load_state_dict(saved["model_state_dict"])
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)

    validation_predictions = _predict(frozen, validation_positions)
    test_predictions = _predict(frozen, test_positions)
    rows: list[dict[str, Any]] = [
        {
            "row_type": "replay_summary",
            "faithful_replay": faithful,
            "replay_epoch": REPLAY_EPOCHS,
            "train_accuracy": replay_train["top1"],
            "validation_accuracy": replay_validation["top1"],
            "validation_cross_entropy": replay_validation["loss"],
            "test_accuracy": replay_test["top1"],
            "maximum_hidden_c_vs_stored_test_c_star_difference": (
                maximum_test_teacher_difference
            ),
        }
    ]
    validation_rows = _trajectory_rows(
        "validation",
        validation_c,
        "deterministic_clean_simulator_c",
        validation_predictions,
    )
    test_rows = _trajectory_rows(
        "test",
        test_c_star,
        "stored_prefix_search_teacher_c_star",
        test_predictions,
    )
    rows.extend(validation_rows)
    rows.extend(test_rows)
    rows.extend(
        _exclusion_rows("validation", validation_c, validation_predictions)
    )
    rows.extend(_exclusion_rows("test", test_c_star, test_predictions))
    combined_c = torch.cat((validation_c, test_c_star))
    combined_predictions = {
        key: torch.cat((validation_predictions[key], test_predictions[key]))
        for key in validation_predictions
    }
    rows.extend(_exclusion_rows("validation_and_test", combined_c, combined_predictions))

    probe_rows, probe_summary, plot_data = _probe(frozen)
    rows.extend(probe_rows)
    brent_rows = _brent_rows(test_positions, test_c_star, test_predictions)
    rows.extend(brent_rows)
    _write_csv(OUTPUT_DIR / "boundary_diagnostic.csv", rows)

    test_truth = (test_c_star >= 0.0).to(torch.long)
    test_misclassified = test_predictions["prediction"] != test_truth
    _plot_probe(plot_data, probe_summary, test_c_star[test_misclassified])
    print(
        {
            "validation_misclassified": [
                row["trajectory_index"]
                for row in validation_rows
                if row["misclassified"]
            ],
            "test_misclassified": [
                row["trajectory_index"]
                for row in test_rows
                if row["misclassified"]
            ],
            **probe_summary,
            "test_teacher_vs_hidden_c_max_difference": (
                maximum_test_teacher_difference
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
