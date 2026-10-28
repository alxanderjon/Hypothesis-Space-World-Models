#!/usr/bin/env python3
"""First frozen N1->N2->N3 Kepler transfer, with the required N1 bypass stop."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.kepler import deterministic_teacher as kepler
from experiments.sho import assemble_n3_learned_polisher as n3
from experiments.sho import run_stage1b_fifo_vs_ranker as n1
from experiments.sho import run_stage1b_learned_global_search_imitation as ranker
from experiments.sho import run_stage1b_learned_powell_2d as n2_scalar
from experiments.sho import train_n2_direction_controller as n2_direction


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/kepler/first_n1_n2_n3_transfer")
GATE = n1.GATE

# These are disjoint from the four geometries used by the teacher/adjoint audits.
# A phase offset is produced by advancing the unchanged teacher before exposing
# the 128-state prefix.
FRESH_CASES = (
    (0.18, 0.72, 0.17, 13),
    (0.28, 0.90, 0.83, 31),
    (0.42, 1.05, 1.47, 53),
    (0.55, 1.18, 2.08, 79),
    (0.68, 0.82, 2.71, 101),
    (0.33, 1.35, 3.22, 127),
    (0.47, 0.95, 4.13, 157),
    (0.62, 1.45, 5.21, 191),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_paths() -> dict[str, Path]:
    root = Path("outputs/sho/stage1B/n3_learned_polisher")
    paths = {
        "n1_candidate_ranker": ranker.CHECKPOINT_PATH,
        "n1_ordinary_score": n1.ordinary.CHECKPOINT_PATH,
        "n1_candidate_selector": n1.ordinary.SELECTOR_PATH,
        "n1_nearest_assignment": n1.nearest.OPERATOR,
        "n1_half_spacing": n1.cells.OUT / "half_spacing_operator_locked.pt",
        "n1_distance": n1.down.LOCKED_CHECKPOINT_PATH,
        "n2_scalar_line_controller": n2_scalar.LINE_CHECKPOINT,
        "n2_direction_controller": n2_direction.LOCAL_CHECKPOINT,
        "n3_gradient": root / "gradient_operator/n3_gradient_operator.pt",
        "n3_hessian": root / "hessian_operator/n3_hessian_operator.pt",
        "n3_curvature": root / "curvature_router/n3_curvature_router.pt",
        "n3_damping": root / "newton_step_operator/n3_damping_construction_locked.json",
        "n3_controls": root / "control_and_closed_loop/n3_learned_polisher.pt",
        "n3_variable_power": Path(
            "outputs/kepler/n3_variable_power_adjoint/n3_variable_power_adjoint.pt"
        ),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Required frozen checkpoints missing: {missing}")
    return paths


def manifest(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def trajectories() -> list[np.ndarray]:
    values = []
    for eccentricity, periapsis, orientation, phase_steps in FRESH_CASES:
        initial = kepler.initial_state(eccentricity, periapsis, orientation)
        phased = kepler.numpy_rollout(initial, steps=phase_steps)[-1]
        values.append(kepler.numpy_rollout(phased, steps=kepler.TOTAL_STEPS))
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths()
    before = manifest(paths)

    # Loading the frozen proposal chain verifies that the active N1 assembly is
    # available. The production loop does not call it when initial design meets
    # the handoff gate.
    n1.FrozenProposals()

    normalized = n1.edirect._initial_design().copy()
    physical = kepler.LOWER + normalized * (kepler.UPPER - kepler.LOWER)
    teacher_trajectories = trajectories()
    rows: list[dict] = []
    for case_id, (spec, trajectory) in enumerate(zip(FRESH_CASES, teacher_trajectories)):
        losses = kepler.prefix_loss_numpy(physical, trajectory)
        best = int(np.argmin(losses))
        theta = physical[best]
        loss = float(losses[best])
        rows.append({
            "trajectory_id": case_id,
            "eccentricity": spec[0],
            "periapsis_radius": spec[1],
            "orientation": spec[2],
            "phase_steps": spec[3],
            "prefix_states": kepler.PREFIX_STATES,
            "initial_design_best_index": best,
            "initial_design_gamma": float(theta[0]),
            "initial_design_p": float(theta[1]),
            "initial_design_loss": loss,
            "initial_design_crossed_handoff_gate": int(loss <= GATE),
            "n1_success": int(loss <= GATE),
            "n1_evaluations": len(physical),
            "n1_learned_proposal_ranking_decisions": 0,
            "n1_voronoi_cell_rounds": 0,
            "n1_handoff_gamma": float(theta[0]),
            "n1_handoff_p": float(theta[1]),
            "n1_handoff_loss": loss,
            "n1_gamma_error": float(theta[0] - kepler.TEACHER_THETA[0]),
            "n1_p_error": float(theta[1] - kepler.TEACHER_THETA[1]),
            "n2_ran": 0,
            "n3_ran": 0,
            "stop_reason": "INITIAL_DESIGN_BYPASSED_LEARNED_N1",
            "teacher_trajectory_finite": int(np.isfinite(trajectory).all()),
        })
        print(
            f"Kepler N1 {case_id + 1}/{len(FRESH_CASES)}: initial eval=10 "
            f"loss={loss:.6e} theta={theta} gate={loss <= GATE}",
            flush=True,
        )

    write_csv(OUT / "per_trajectory.csv", rows)
    all_bypassed = all(row["initial_design_crossed_handoff_gate"] for row in rows)
    after = manifest(paths)
    report = {
        "status": (
            "STOPPED_ALL_INITIAL_DESIGNS_BYPASSED_LEARNED_N1"
            if all_bypassed else "READY_FOR_FULL_STACK"
        ),
        "first_failing_or_blocking_stage": "N1_ACTIVITY_CHECK" if all_bypassed else "",
        "required_action": (
            "Stopped per experiment instruction; N2 and N3 were not run."
            if all_bypassed else "Proceed through frozen N1/N2/N3."
        ),
        "teacher": {
            "state": "(x1,y1,vx1,vy1,x2,y2,vx2,vy2)",
            "integrator": "unchanged float64 velocity Verlet",
            "dt": float(kepler.DT),
            "theta": kepler.TEACHER_THETA.tolist(),
            "bounds": np.column_stack((kepler.LOWER, kepler.UPPER)).tolist(),
            "prefix_states": kepler.PREFIX_STATES,
            "total_steps": kepler.TOTAL_STEPS,
        },
        "objective": "mean squared one-step transition reconstruction over the 128-state prefix",
        "trajectories": len(rows),
        "fresh_case_definitions": [
            {"eccentricity": e, "periapsis": r, "orientation": o, "phase_steps": phase}
            for e, r, o, phase in FRESH_CASES
        ],
        "n1_successes": sum(row["n1_success"] for row in rows),
        "n1_initial_design_handoffs": sum(
            row["initial_design_crossed_handoff_gate"] for row in rows
        ),
        "n1_genuinely_exercised": sum(
            row["n1_learned_proposal_ranking_decisions"] > 0 for row in rows
        ),
        "n1_learned_decisions": 0,
        "n1_voronoi_cell_rounds": 0,
        "n1_total_evaluations": sum(row["n1_evaluations"] for row in rows),
        "n1_median_evaluations": float(np.median([row["n1_evaluations"] for row in rows])),
        "n1_worst_handoff_gamma_error": max(abs(row["n1_gamma_error"]) for row in rows),
        "n1_worst_handoff_p_error": max(abs(row["n1_p_error"]) for row in rows),
        "n1_worst_handoff_objective": max(row["n1_handoff_loss"] for row in rows),
        "initial_design_parameter_extent": {
            "gamma": [float(physical[:, 0].min()), float(physical[:, 0].max())],
            "p": [float(physical[:, 1].min()), float(physical[:, 1].max())],
        },
        "n2_runs": 0,
        "n2_successes": None,
        "n3_runs": 0,
        "n3_successes": None,
        "continuation_rollouts": 0,
        "worst_final_objective": None,
        "worst_continuation_error": None,
        "qualitative_orbital_regime_diagnostic": (
            "No learned candidate search occurred; only the fixed broad-domain initial design "
            "was evaluated with the one-step objective. No additional orbit-regime classifier "
            "or conservation metric was introduced."
        ),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "checkpoints_unchanged": before == after,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
