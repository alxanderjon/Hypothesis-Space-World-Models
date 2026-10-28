#!/usr/bin/env python3
"""Map unchanged N2->N3 reachability on the saved static-basin samples."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

from experiments import map_direct_n3_correctability_geometry as basin
from experiments.audit_known_structure import step as kepler_step
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.kepler import run_four_parameter_peelback as staged


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SOURCE = Path("outputs/static_n3_basin_geometry/points.jsonl")
OUT = Path("outputs/n2_n3_reachable_geometry")
PRIMARY = OUT / "primary.jsonl"
SCALE = OUT / "scale_sensitivity.jsonl"
SCALE_COUNTS = (25, 100, 1_000, 5_000)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _source_index(row: dict) -> int:
    index = int(row["source_sobol_index"])
    return index - 100_000 if index >= 100_000 else index


def _stage_count(row: dict) -> int:
    # Direct-basin data were generated and assessed in completed Sobol blocks
    # of 32, 64, 128, and 256.  This count is part of the constructed handoff
    # state; it is not inferred from objective value or ground truth.
    target = max(32, _source_index(row) + 1)
    return 1 << int(math.ceil(math.log2(target)))


def _objectives() -> dict:
    ball_trajectory = ball._trajectories()[3]
    ball_numpy, _ = ball.objectives(ball_trajectory)
    kepler_trajectory = staged.trajectories()[0]

    def kepler_numpy(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = float(np.mean((kepler_step(
                kepler_trajectory[:127], np.asarray(theta, dtype=np.float64), np
            ) - kepler_trajectory[1:128]) ** 2))
        return value if math.isfinite(value) else 1.0e100

    return {"bouncing_ball": ball_numpy, "kepler": kepler_numpy}


def _run(row: dict, count: int, objective, config: dict, mode: str) -> dict:
    started = time.perf_counter()
    theta = np.asarray(row["theta"], dtype=np.float64)
    lower, upper = config["lower"], config["upper"]
    width = upper - lower
    radius = width / float(count) ** (1.0 / len(theta))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    start_loss = float(objective(theta))
    n2 = staged._run_n2(objective, theta, start_loss, box_lower, box_upper)
    n3 = basin._run_n3(np.asarray(n2["theta"]), {"loss": float(n2["loss"])}, config)
    teacher = np.asarray(config["teacher"])
    outside = np.maximum(np.maximum(box_lower - teacher, teacher - box_upper), 0.0)
    return dict(
        key=f'{row["anchor_id"]}:{row["kind"]}:{row["direction_id"]}:{row["radius"]}',
        mode=mode, system=row["system"], anchor_id=row["anchor_id"],
        anchor_role=row["anchor_role"], kind=row["kind"],
        direction_id=int(row["direction_id"]), radius=float(row["radius"]),
        source_sobol_index=int(row["source_sobol_index"]),
        n1_evaluation_count=int(count), normalized_box_halfwidth=float(count ** (-1.0 / len(theta))),
        theta=theta.tolist(), start_loss=start_loss,
        direct_n3_PASS=bool(row["n3_label"]["PASS"]),
        box_lower=box_lower.tolist(), box_upper=box_upper.tolist(),
        offline_teacher_inside_box=bool(np.all(outside == 0.0)),
        offline_teacher_box_distance=float(np.linalg.norm(outside)),
        n2_endpoint=np.asarray(n2["theta"]).tolist(), n2_loss=float(n2["loss"]),
        n2_evaluations=int(n2["evaluations"]), n2_stop_reason=n2["message"],
        n2_historical_gate=bool(n2["gate"]),
        n3_PASS=bool(n3["passed"]), n3_final_theta=n3["final_theta"],
        n3_final_loss=float(n3["final_loss"]), n3_stop_reason=n3["stop_reason"],
        n3_accepted_steps=int(n3["accepted_steps"]),
        n3_objective_evaluations=int(n3["objective_evaluations"]),
        rescued=bool(not row["n3_label"]["PASS"] and n3["passed"]),
        damaged=bool(row["n3_label"]["PASS"] and not n3["passed"]),
        static_geometry={key: row["static_geometry"][key] for key in (
            "loss", "gradient_norm", "scaled_gradient_norm", "minimum_eigenvalue",
            "maximum_eigenvalue", "condition_number", "negative_curvature_count",
            "third_derivative_frobenius_norm", "existing_route", "newton_step_norm",
            "normalized_step_l2", "descent_cosine", "predicted_decrease",
            "minimum_bound_margin", "full_newton_step_in_bounds",
        )},
        seconds=float(time.perf_counter() - started),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = _load(SOURCE)
    configs = {name: basin._system_config(name) for name in ("bouncing_ball", "kepler")}
    objectives = _objectives()

    existing = _load(PRIMARY)
    done = {row["key"] for row in existing}
    for index, row in enumerate(source):
        key = f'{row["anchor_id"]}:{row["kind"]}:{row["direction_id"]}:{row["radius"]}'
        if key in done:
            continue
        result = _run(row, _stage_count(row), objectives[row["system"]],
                      configs[row["system"]], "source_stage")
        _append(PRIMARY, result); done.add(key)
        print(json.dumps(dict(index=index + 1, total=len(source), system=row["system"],
                              anchor=row["anchor_id"], direct=result["direct_n3_PASS"],
                              reachable=result["n3_PASS"], rescued=result["rescued"],
                              damaged=result["damaged"], count=result["n1_evaluation_count"],
                              n2_evals=result["n2_evaluations"], seconds=result["seconds"])),
              flush=True)

    existing_scale = _load(SCALE)
    done_scale = {(row["key"], int(row["n1_evaluation_count"])) for row in existing_scale}
    anchors = [row for row in source if row["kind"] == "anchor"]
    for row in anchors:
        for count in SCALE_COUNTS:
            key = f'{row["anchor_id"]}:{row["kind"]}:{row["direction_id"]}:{row["radius"]}'
            if (key, count) in done_scale:
                continue
            result = _run(row, count, objectives[row["system"]], configs[row["system"]],
                          "scale_sensitivity")
            _append(SCALE, result); done_scale.add((key, count))
            print(json.dumps(dict(scale=True, system=row["system"], anchor=row["anchor_id"],
                                  count=count, direct=result["direct_n3_PASS"],
                                  reachable=result["n3_PASS"], rescued=result["rescued"],
                                  damaged=result["damaged"], n2_evals=result["n2_evaluations"],
                                  seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    main()
