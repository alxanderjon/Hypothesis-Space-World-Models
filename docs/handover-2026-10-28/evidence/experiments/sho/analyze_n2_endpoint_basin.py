"""Diagnose N2 endpoint distance and the first deterministic N3 Newton step.

This reads completed full-K artifacts only.  It does not execute N1, N2, or N3.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.sho.deterministic_full_k_framework import prior
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


ROOT = Path("outputs/sho/stage1B/checkpointed_resource_edirect")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
DTYPE = torch.float64


def _first_newton(states: np.ndarray, theta_value: np.ndarray):
    x = torch.from_numpy(states[:-1].copy())
    y = torch.from_numpy(states[1:].copy())
    theta = torch.tensor(theta_value, dtype=DTYPE, requires_grad=True)
    loss = torch.mean((x @ theta.reshape(2, 2).T - y) ** 2)
    gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
    hessian = torch.stack([
        torch.autograd.grad(gradient[index], theta, retain_graph=True)[0]
        for index in range(len(theta))
    ])
    hessian = 0.5 * (hessian + hessian.T)

    singular_values = torch.linalg.svdvals(hessian)
    smallest = float(torch.min(singular_values))
    largest = float(torch.max(singular_values))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum_eigenvalue = float(torch.min(eigenvalues))
    scale = max(float(torch.max(torch.abs(eigenvalues))), torch.finfo(DTYPE).tiny)
    direct_allowed = minimum_eigenvalue > 0.0 and condition <= polish.CONDITION_LIMIT
    initial_damping = 0.0
    if not direct_allowed:
        initial_damping = max(
            scale * polish.DAMPING_RELATIVE_FLOOR,
            -minimum_eigenvalue + scale * polish.DAMPING_RELATIVE_FLOOR,
        )
    identity = torch.eye(len(theta), dtype=DTYPE)
    direction = None
    damping = math.nan
    for attempt in range(polish.MAX_DAMPING_TRIALS):
        if attempt == 0 and direct_allowed:
            damping = 0.0
        else:
            base = initial_damping if initial_damping > 0.0 else scale * polish.DAMPING_RELATIVE_FLOOR
            damping = base * (10.0 ** (attempt if not direct_allowed else attempt - 1))
        try:
            candidate = torch.linalg.solve(hessian + damping * identity, -gradient)
        except torch.linalg.LinAlgError:
            continue
        if bool(torch.isfinite(candidate).all()) and float(torch.dot(gradient, candidate)) < 0.0:
            direction = candidate.detach().numpy()
            break
    return dict(
        gradient=gradient.detach().numpy(),
        gradient_norm=float(torch.linalg.vector_norm(gradient)),
        hessian=hessian.detach().numpy(),
        newton_direction=direction,
        newton_step_norm=(float(np.linalg.norm(direction)) if direction is not None else None),
        condition=condition,
        damping=damping,
    )


def _stats(rows):
    def values(key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)
    result = {"cases": len(rows)}
    for key in ("n2_max_entry_error", "n2_frobenius_error", "normalized_domain_distance"):
        data = values(key)
        result[key] = dict(
            median=float(np.median(data)),
            p90=float(np.quantile(data, 0.90, method="higher")),
            maximum=float(np.max(data)),
        )
    result["accepted_n3_steps"] = {
        str(step): sum(row["accepted_n3_steps"] == step for row in rows)
        for step in sorted({row["accepted_n3_steps"] for row in rows})
    }
    return result


def main():
    results = json.loads((ROOT / "results_partial.json").read_text())
    with np.load(DATA) as data:
        names = [str(value) for value in data["names"]]
        teachers = data["teachers"].reshape(-1, 4)
        prefixes = [data[f"prefix_{index}"].copy() for index in range(len(names))]
    by_name = {row["case"]: row for row in results}
    domain_width = prior.UPPER - prior.LOWER
    domain_diagonal = float(np.linalg.norm(domain_width))
    rows = []
    for index, name in enumerate(names):
        result = by_name[name]
        teacher = teachers[index]
        n2 = np.asarray(result["n2_theta"], dtype=np.float64)
        final = np.asarray(result["final_theta"], dtype=np.float64)
        error = n2 - teacher
        frobenius = float(np.linalg.norm(error))
        checkpoint = next(
            point for point in result["checkpoints"]
            if point["evaluation"] == result.get("handoff_checkpoint", result["first_both_checkpoint"])
        )
        derivative = _first_newton(prefixes[index], n2)
        step_norm = derivative["newton_step_norm"]
        row = dict(
            case=name, category=result["category"],
            n2_gate_ok=bool(result["n2_gate"]),
            ground_truth=teacher.tolist(), n2_theta=n2.tolist(),
            entrywise_errors=error.tolist(),
            n2_max_entry_error=float(np.max(np.abs(error))),
            n2_frobenius_error=frobenius,
            normalized_domain_distance=frobenius / domain_diagonal,
            n2_objective_loss=float(result["n2_loss"]),
            ground_truth_inside_n2_box=bool(checkpoint["containment_ok"]),
            ground_truth_to_n2_box_distance=float(checkpoint["teacher_outside_l2"]),
            n2_box_lower=checkpoint["box_lower"], n2_box_upper=checkpoint["box_upper"],
            gradient=derivative["gradient"].tolist(),
            gradient_norm=derivative["gradient_norm"],
            newton_direction=(derivative["newton_direction"].tolist()
                              if derivative["newton_direction"] is not None else None),
            newton_step_norm=step_norm,
            n2_to_final_n3_distance=float(np.linalg.norm(n2 - final)),
            newton_step_to_truth_distance_ratio=(
                step_norm / frobenius if step_norm is not None and frobenius > 0.0 else None
            ),
            accepted_n3_steps=int(result["n3_accepted_steps"]),
            final_n3_matrix_entry_error=float(result["matrix_max_error"]),
            final_n3_frobenius_error=float(result["matrix_error"]),
            final_rollout_error=float(result["rollout_max_error"]),
            n1_loss_ok=result["first_loss_crossing"] is not None,
            containment_ever_ok=result["first_containment_checkpoint"] is not None,
        )
        rows.append(row)

    n2_success = [row for row in rows if row["n2_gate_ok"]]
    n2_failure = [row for row in rows if not row["n2_gate_ok"]]
    containment_failure = [row for row in rows if not row["ground_truth_inside_n2_box"]]
    farthest = max(rows, key=lambda row: row["n2_frobenius_error"])
    farthest_outside = max(rows, key=lambda row: row["ground_truth_to_n2_box_distance"])
    summary = dict(
        definition=dict(
            normalized_domain_distance="N2 Frobenius error / L2 diagonal of full 4D bounds",
            full_parameter_width=domain_width.tolist(),
            full_domain_diagonal=domain_diagonal,
            newton_step="raw first deterministic N3 Newton direction before backtracking",
        ),
        all_cases=_stats(rows), n2_gate_successes=_stats(n2_success),
        n2_gate_failures=_stats(n2_failure),
        containment_failures=_stats(containment_failure),
        farthest_recovered_case=farthest,
        farthest_outside_box_recovered_case=farthest_outside,
        corner_16=next(row for row in rows if row["case"] == "corner_16"),
    )
    (ROOT / "n2_endpoint_basin.json").write_text(json.dumps(summary, indent=2))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / "n2_endpoint_basin.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
