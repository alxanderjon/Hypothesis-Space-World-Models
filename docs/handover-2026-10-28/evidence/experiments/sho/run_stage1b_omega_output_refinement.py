"""Stage 1B omega-output-only refinement diagnostic.

Starting from final Stage 1B checkpoints, freeze coefficient-related outputs and
all shared parameters, then update only the clean omega-output rows through
10-period trajectory reconstruction loss.
"""

from __future__ import annotations

import copy
import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import torch

from experiments.sho.run_stage1b_gru_two_state_curriculum import (
    GRU_HIDDEN_SIZE,
    SHOTwoStateGRUWindowModel,
)
from experiments.sho.run_stage1b_late_horizon_refinement_curriculum import HIDDEN_DIMS
from experiments.sho.run_stage1b_two_state_omega import (
    BATCH_SIZE,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/omega_output_refinement")
FEEDFORWARD_PARAMETERS = Path(
    "outputs/sho/stage1B/late_horizon_refinement_curriculum/learned_parameters.pt",
)
GRU_PARAMETERS = Path("outputs/sho/stage1B/gru_two_state_curriculum/learned_parameters.pt")
LEARNING_RATES = [1e-4, 3e-5, 1e-5]
MAX_EPOCHS = 100000
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 20000
TARGET_HORIZON_PERIODS = 10.0
LOG_EVERY = 10000


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _make_feedforward_model() -> SHOTwoStateTrajectoryOmegaLinearResidualModel:
    model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    model.load_state_dict(_load_state_dict(FEEDFORWARD_PARAMETERS))
    return model


def _make_gru_model() -> SHOTwoStateGRUWindowModel:
    model = SHOTwoStateGRUWindowModel(hidden_size=GRU_HIDDEN_SIZE)
    model.load_state_dict(_load_state_dict(GRU_PARAMETERS))
    return model


def _mask_output_rows(module: torch.nn.Linear, trainable_row: int) -> None:
    weight_mask = torch.zeros_like(module.weight)
    weight_mask[trainable_row, :] = 1.0
    module.weight.register_hook(lambda grad: grad * weight_mask.to(grad.device))
    if module.bias is not None:
        bias_mask = torch.zeros_like(module.bias)
        bias_mask[trainable_row] = 1.0
        module.bias.register_hook(lambda grad: grad * bias_mask.to(grad.device))


def _freeze_for_omega_only(model_name: str, model: torch.nn.Module) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if model_name == "feedforward_residual":
        # The architecture has shared hidden layers, but the final linear rows
        # cleanly map to (a, b, raw_omega). Keep only row 2 trainable in each
        # additive output path.
        model.linear_encoder.weight.requires_grad_(True)
        model.linear_encoder.bias.requires_grad_(True)
        _mask_output_rows(model.linear_encoder, trainable_row=2)
        residual_last = model.residual_encoder[-1]
        if not isinstance(residual_last, torch.nn.Linear):
            raise TypeError("expected final residual encoder module to be Linear")
        residual_last.weight.requires_grad_(True)
        residual_last.bias.requires_grad_(True)
        _mask_output_rows(residual_last, trainable_row=2)
        return {
            "freeze_mode": "all parameters frozen except omega row of linear_encoder and residual final layer",
            "trainable_parameter_tensors": [
                "linear_encoder.weight[row 2]",
                "linear_encoder.bias[row 2]",
                "residual_encoder[-1].weight[row 2]",
                "residual_encoder[-1].bias[row 2]",
            ],
        }

    if model_name == "gru_hidden_16":
        model.output_head.weight.requires_grad_(True)
        model.output_head.bias.requires_grad_(True)
        _mask_output_rows(model.output_head, trainable_row=2)
        return {
            "freeze_mode": "GRU frozen; only omega row of output_head trainable",
            "trainable_parameter_tensors": [
                "output_head.weight[row 2]",
                "output_head.bias[row 2]",
            ],
        }

    raise ValueError(f"unknown model_name: {model_name}")


def _train_one(
    *,
    model_name: str,
    learning_rate: float,
    train_data: Any,
    test_data: Any,
) -> dict[str, Any]:
    model = _make_feedforward_model() if model_name == "feedforward_residual" else _make_gru_model()
    freeze_info = _freeze_for_omega_only(model_name, model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    source_metrics, _, _, _ = _evaluate_model(model, test_data)

    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    final_train_loss = float("inf")
    final_test_loss = float("inf")
    num_train = train_data.two_state_inputs.shape[0]

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled_indices = torch.randperm(num_train)
        loss_sum = 0.0
        example_count = 0
        for start in range(0, num_train, BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            _, trajectories, _ = model(
                train_data.two_state_inputs[batch_indices],
                train_data.times[batch_indices],
            )
            loss = mse_loss(trajectories, train_data.trajectories[batch_indices])
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            batch_count = len(batch_indices)
            loss_sum += loss.item() * batch_count
            example_count += batch_count

        if stopped_because == "nan_or_inf_loss":
            break

        final_train_loss = loss_sum / example_count
        model.eval()
        with torch.no_grad():
            _, test_trajectories, _ = model(test_data.two_state_inputs, test_data.times)
            final_test_loss = mse_loss(test_trajectories, test_data.trajectories).item()

        if final_test_loss < best_test_loss - STOPPING_MIN_DELTA:
            best_test_loss = final_test_loss
            best_train_loss = final_train_loss
            best_epoch = epoch
            best_model_state = _clone_state_dict(model.state_dict())
            best_optimizer_state = _clone_state_dict(optimizer.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                (
                    f"{model_name} lr={learning_rate:g} epoch {epoch:06d} "
                    f"train_loss={final_train_loss:.6e} "
                    f"test_loss={final_test_loss:.6e} best_test={best_test_loss:.6e}"
                ),
                flush=True,
            )

        if epochs_since_improvement >= STOPPING_PATIENCE:
            stopped_because = "patience_exhausted"
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)

    metrics, _, _, _ = _evaluate_model(model, test_data)
    run_dir = OUTPUT_DIR / model_name / f"lr_{learning_rate:g}".replace("-", "m").replace(".", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    row = {
        "model": model_name,
        "learning_rate": learning_rate,
        "source_trajectory_mse": source_metrics["trajectory_mse"],
        "source_coefficient_mse": source_metrics["coefficient_mse"],
        "source_omega_mse": source_metrics["omega_mse"],
        "epochs_completed": epoch if "epoch" in locals() else 0,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        **metrics,
        **freeze_info,
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "learning_rate",
        "trajectory_mse",
        "omega_mse",
        "omega_correlation",
        "mean_abs_omega_error",
        "max_omega_error",
        "coefficient_mse",
        "source_trajectory_mse",
        "best_epoch",
        "stopped_because",
        "freeze_mode",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Omega Output Refinement</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
  </style>
</head>
<body>
  <h1>Stage 1B Omega Output Refinement</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
"""
    (OUTPUT_DIR / "omega_output_refinement_summary.html").write_text(html, encoding="utf-8")


def run_omega_output_refinement() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    train_data = _build_data_from_base(
        train_initial_conditions,
        train_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=TARGET_HORIZON_PERIODS,
    )
    test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=TARGET_HORIZON_PERIODS,
    )
    rows = []
    for model_name in ("feedforward_residual", "gru_hidden_16"):
        for learning_rate in LEARNING_RATES:
            rows.append(
                _train_one(
                    model_name=model_name,
                    learning_rate=learning_rate,
                    train_data=train_data,
                    test_data=test_data,
                ),
            )
    _write_csv(OUTPUT_DIR / "omega_output_refinement_summary.csv", rows)
    (OUTPUT_DIR / "omega_output_refinement_summary.json").write_text(
        json.dumps({"runs": rows}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "setup.json").write_text(
        json.dumps(
            {
                "stage": "stage1B",
                "diagnostic": "omega_output_refinement",
                "target_horizon_periods": TARGET_HORIZON_PERIODS,
                "learning_rates": LEARNING_RATES,
                "max_epochs": MAX_EPOCHS,
                "stopping_min_delta": STOPPING_MIN_DELTA,
                "stopping_patience": STOPPING_PATIENCE,
                "feedforward_parameters": str(FEEDFORWARD_PARAMETERS),
                "gru_parameters": str(GRU_PARAMETERS),
                "loss": "trajectory reconstruction only",
                "omega_supervision": False,
                "coefficient_supervision": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_html(rows)
    return rows


def main() -> None:
    rows = run_omega_output_refinement()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
