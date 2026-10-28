"""Stage 1B late-horizon refinement curriculum.

This run keeps the Stage 1B architecture, decoder, input, supervision, loss,
learning rate, and stopping settings fixed. It only inserts smaller continuation
steps near the late horizons.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import experiments.sho.run_stage1b_ultra_fine_unsupervised_curriculum as ultra
from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    SUCCESS_OMEGA_TOLERANCE,
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/late_horizon_refinement_curriculum")
RUN_NAME = "linear_moderate_residual"
HIDDEN_DIMS = [8, 8]
LEARNING_RATE = 1e-2
MAX_EPOCHS_PER_STAGE = 100000
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 20000
SUCCESS_TRAJECTORY_MSE = 1e-15
REFINEMENT_HORIZONS = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
]


def run_late_horizon_refinement_curriculum() -> list[dict[str, object]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = {
        "stage": "stage1B",
        "experiment": "late_horizon_refinement_curriculum",
        "run_name": RUN_NAME,
        "input": "(x0, v0, x1, v1)",
        "output": "(a, b, omega_hat_i)",
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "model": "linear + moderate residual encoder",
        "hidden_dims": HIDDEN_DIMS,
        "activation": "SiLU",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "loss": "trajectory reconstruction only",
        "omega_supervision": False,
        "coefficient_supervision": False,
        "auxiliary_losses": False,
        "regularization": False,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": ultra.BATCH_SIZE,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
        "curriculum_horizons_periods": REFINEMENT_HORIZONS,
        "random_seed": SEED,
        "run_date": RUN_DATE,
    }
    rows = []
    original_output_dir = ultra.OUTPUT_DIR
    original_max_epochs = ultra.MAX_EPOCHS_PER_STAGE
    original_patience = ultra.STOPPING_PATIENCE
    original_min_delta = ultra.STOPPING_MIN_DELTA
    try:
        ultra.OUTPUT_DIR = OUTPUT_DIR
        ultra.MAX_EPOCHS_PER_STAGE = MAX_EPOCHS_PER_STAGE
        ultra.STOPPING_PATIENCE = STOPPING_PATIENCE
        ultra.STOPPING_MIN_DELTA = STOPPING_MIN_DELTA
        for stage_index, horizon_periods in enumerate(REFINEMENT_HORIZONS, start=1):
            train_data = _build_data_from_base(
                train_initial_conditions,
                train_omegas,
                num_time_steps=NUM_TIME_STEPS,
                num_periods=horizon_periods,
            )
            test_data = _build_data_from_base(
                test_initial_conditions,
                test_omegas,
                num_time_steps=NUM_TIME_STEPS,
                num_periods=horizon_periods,
            )
            row = ultra._train_stage(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                test_data=test_data,
                stage_index=stage_index,
                horizon_periods=horizon_periods,
            )
            rows.append(row)
            metrics, _, _, omega_hat = _evaluate_model(model, test_data)
            ultra._save_stage_omega_plots(
                stage_dir=OUTPUT_DIR / "stage_omega_plots",
                stage_index=stage_index,
                horizon_periods=horizon_periods,
                omega_hat=omega_hat,
                true_omega=test_data.omegas,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stage_summary": row,
                    "stage_metrics_recomputed": metrics,
                },
                OUTPUT_DIR / f"checkpoint_after_stage_{stage_index:02d}.pt",
            )

        final_test_data = _build_data_from_base(
            test_initial_conditions,
            test_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=REFINEMENT_HORIZONS[-1],
        )
        ultra._save_final_trajectory_examples(
            output_dir=OUTPUT_DIR,
            model=model,
            test_data=final_test_data,
        )
        torch.save(model.state_dict(), OUTPUT_DIR / "learned_parameters.pt")
        ultra._write_json(OUTPUT_DIR / "setup.json", setup)
        ultra._write_json(
            OUTPUT_DIR / "late_horizon_refinement_curriculum_summary.json",
            {"stages": rows},
        )
        ultra._write_csv(
            OUTPUT_DIR / "late_horizon_refinement_curriculum_summary.csv",
            rows,
        )
        ultra._save_metric_plots(rows)
        ultra._write_html(rows, setup)
        (OUTPUT_DIR / "ultra_fine_unsupervised_curriculum_summary.html").rename(
            OUTPUT_DIR / "late_horizon_refinement_curriculum_summary.html",
        )
    finally:
        ultra.OUTPUT_DIR = original_output_dir
        ultra.MAX_EPOCHS_PER_STAGE = original_max_epochs
        ultra.STOPPING_PATIENCE = original_patience
        ultra.STOPPING_MIN_DELTA = original_min_delta
    return rows


def main() -> None:
    rows = run_late_horizon_refinement_curriculum()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
