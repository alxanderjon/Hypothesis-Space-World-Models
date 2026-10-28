"""Test-only affine-containment fixture for the generic SHO update network.

This is a representational diagnostic only.  It performs no fitting, search,
recovery, checkpoint creation, or mutation of the paused recovery artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.sho import recover_affine_bias_targeted as parent
from experiments.sho import recover_generic_update_targeted as recovery
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.generic_update_network_adapter import (
    GenericSHOUpdateNetworkAdapter,
)
from experiments.sho.validate_generic_update_network_adapter import protected


OUT = Path("outputs/sho/generic_update_network_targeted_recovery")
REPORT = OUT / "representational_feasibility_companion_6.json"
PAUSE = OUT / "manual_efficiency_pause.json"
JSONL = OUT / "companion_6/n00005_i00004_n3.jsonl"
BRANCHES = OUT / "companion_6/branches.json"
CASE = "companion_6"
CASE_ID = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else str(scalar)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def projected_gradient(gradient: torch.Tensor, phi: torch.Tensor,
                       lower: torch.Tensor, upper: torch.Tensor):
    at_lower = phi == lower
    at_upper = phi == upper
    kkt_lower = at_lower & (gradient >= 0.0)
    kkt_upper = at_upper & (gradient <= 0.0)
    projected = gradient.clone()
    projected[kkt_lower | kkt_upper] = 0.0
    return projected, at_lower, at_upper, kkt_lower, kkt_upper


def rank_geometry(eigenvalues: torch.Tensor) -> dict:
    values = eigenvalues.detach().cpu().numpy()
    maximum = float(np.max(np.abs(values)))
    tolerance = 22 * np.finfo(np.float64).eps * maximum
    positive = values > tolerance
    negative = values < -tolerance
    near_zero = ~(positive | negative)
    rank = int(np.count_nonzero(positive | negative))
    return {
        "rank_convention": "abs(eigenvalue) > 22*eps*max(abs(eigenvalue))",
        "rank_tolerance": tolerance,
        "numerical_rank": rank,
        "nullity": 22 - rank,
        "positive_count": int(np.count_nonzero(positive)),
        "negative_count": int(np.count_nonzero(negative)),
        "near_zero_count": int(np.count_nonzero(near_zero)),
        "eigenvalues": values,
        "minimum_eigenvalue": float(values[0]),
        "maximum_eigenvalue": float(values[-1]),
        "maximum_absolute_eigenvalue": maximum,
    }


def search_comparison(fixture_loss: float) -> dict:
    pause = json.loads(PAUSE.read_text())
    paused_loss = float(pause["n3"]["loss"])
    branches = json.loads(BRANCHES.read_text())
    completed = [row for row in branches if "N3" in row and "loss" in row["N3"]]
    best = min(completed, key=lambda row: float(row["N3"]["loss"]))
    best_loss = float(best["N3"]["loss"])

    def gap(loss):
        if fixture_loss == 0.0:
            return "infinite_fixture_loss_is_exact_zero"
        return math.log10(loss / fixture_loss)

    return {
        "paused_branch": {
            "evaluation_count": 5,
            "archive_index": 4,
            "accepted_step": int(pause["n3"]["accepted_step"]),
            "loss": paused_loss,
            "orders_of_magnitude_above_fixture": gap(paused_loss),
        },
        "best_completed_branch": {
            "evaluation_count": int(best["evaluation_count"]),
            "archive_index": int(best["archive_index"]),
            "loss": best_loss,
            "accepted_steps": int(best["N3"]["accepted_steps"]),
            "stop_reason": best["N3"]["stop_reason"],
            "orders_of_magnitude_above_fixture": gap(best_loss),
        },
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    paused_before = {str(path): sha256(path) for path in (JSONL, PAUSE)}
    protected_before = protected()
    checkpoints_before = hashes()

    corpus = np.load(parent.DATA)
    legacy = json.loads((parent.OLD / CASE / "result.json").read_text())
    if int(legacy["case_id"]) != CASE_ID:
        raise RuntimeError("companion_6 case identity changed")
    trajectory = np.asarray(corpus[f"prefix_{CASE_ID}"], np.float64).copy()
    matrix = np.asarray(corpus["teachers"][CASE_ID], np.float64).reshape(2, 2)
    bias = np.zeros(2, dtype=np.float64)

    adapter = GenericSHOUpdateNetworkAdapter()
    expected_layout = [
        {"name": "0.weight", "shape": [4, 2], "start": 0, "stop": 8},
        {"name": "0.bias", "shape": [4], "start": 8, "stop": 12},
        {"name": "2.weight", "shape": [2, 4], "start": 12, "stop": 20},
        {"name": "2.bias", "shape": [2], "start": 20, "stop": 22},
    ]
    if adapter.layout != expected_layout:
        raise RuntimeError(f"unexpected adapter layout: {adapter.layout}")

    phi = np.zeros(22, dtype=np.float64)
    phi[0:8] = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float64,
    ).reshape(-1)
    phi[8:12] = 0.0
    phi[12:20] = np.stack(
        (matrix[:, 0], -matrix[:, 0], matrix[:, 1], -matrix[:, 1]), axis=1,
    ).reshape(-1)
    phi[20:22] = bias

    below = np.flatnonzero(phi < adapter.lower)
    above = np.flatnonzero(phi > adapter.upper)
    bound_check = {
        "all_inside": not len(below) and not len(above),
        "minimum_parameter": float(phi.min()),
        "maximum_parameter": float(phi.max()),
        "below_lower_indices": below,
        "above_upper_indices": above,
        "lower": adapter.lower,
        "upper": adapter.upper,
    }
    if not bound_check["all_inside"]:
        report = {
            "status": "CANONICAL_AFFINE_FIXTURE_OUTSIDE_BOUNDS",
            "case": CASE,
            "teacher": {"K": matrix, "b": bias},
            "phi": phi,
            "bound_check": bound_check,
        }
        REPORT.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
        raise RuntimeError("canonical fixture violates bounds")

    states = torch.from_numpy(trajectory)
    flat = torch.from_numpy(phi)
    terms = adapter.objective_terms(flat, states)
    prediction = terms.prediction.detach()
    residual = terms.residual.detach()
    square = terms.square.detach()
    loss = float(terms.loss)
    maximum_residual = float(torch.max(torch.abs(residual)))
    rms_residual = float(torch.sqrt(torch.mean(square)))

    k_tensor = torch.from_numpy(matrix)
    b_tensor = torch.from_numpy(bias)
    direct = states[:-1] @ k_tensor.T + b_tensor
    direct_residual = direct - states[1:]
    network_minus_direct = prediction - direct
    prefix_rows = []
    for index in range(len(prediction)):
        prefix_rows.append({
            "transition": index,
            "input": states[index],
            "target": states[index + 1],
            "network_prediction": prediction[index],
            "network_residual": residual[index],
            "direct_affine_prediction": direct[index],
            "direct_affine_residual": direct_residual[index],
            "network_minus_direct": network_minus_direct[index],
        })

    rollout = recovery.rollout_diagnostic(adapter, trajectory, matrix, phi)
    derivatives = adapter.derivatives(phi, trajectory)
    gradient = derivatives.gradient
    hessian = derivatives.symmetric_hessian
    eigenvalues = torch.linalg.eigvalsh(hessian)
    lower = torch.from_numpy(adapter.lower)
    upper = torch.from_numpy(adapter.upper)
    projected, at_lower, at_upper, kkt_lower, kkt_upper = projected_gradient(
        gradient, flat, lower, upper,
    )
    nonzero_indices = lambda mask: torch.nonzero(mask, as_tuple=False).flatten().numpy()
    active_indices = nonzero_indices(at_lower | at_upper)
    geometry = {
        "loss": float(derivatives.loss),
        "gradient": gradient,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "projected_gradient": projected,
        "projected_gradient_norm": float(torch.linalg.vector_norm(projected)),
        "hessian": hessian,
        **rank_geometry(eigenvalues),
        "active_bound_indices": active_indices,
        "lower_active_indices": nonzero_indices(at_lower),
        "upper_active_indices": nonzero_indices(at_upper),
        "kkt_compatible_lower_indices": nonzero_indices(kkt_lower),
        "kkt_compatible_upper_indices": nonzero_indices(kkt_upper),
    }

    comparison = search_comparison(loss)
    feasible = loss < recovery.FINAL_GATE
    status = (
        "GENERIC_NETWORK_HANDOFF_FEASIBLE_IN_FLOAT64"
        if feasible else "CANONICAL_AFFINE_FIXTURE_DOES_NOT_REACH_HANDOFF"
    )
    report = {
        "status": status,
        "training": False,
        "optimization": False,
        "recovery_run": False,
        "fixture_use": "TEST_ONLY_REPRESENTATIONAL_DIAGNOSTIC",
        "case": CASE,
        "case_id": CASE_ID,
        "prefix_states": len(trajectory),
        "transition_count": len(trajectory) - 1,
        "teacher": {
            "K": matrix,
            "b": bias,
            "bias_source": "original full-K companion law has no additive term",
        },
        "architecture": "Linear(2,4)->SiLU->Linear(4,2)",
        "layout": adapter.layout,
        "phi": phi,
        "bound_check": bound_check,
        "prefix": {
            "transition_mse": loss,
            "maximum_absolute_one_step_residual": maximum_residual,
            "rms_residual": rms_residual,
            "handoff_threshold": recovery.FINAL_GATE,
            "below_handoff_threshold": feasible,
            "direct_affine_mse": float(torch.mean(direct_residual * direct_residual)),
            "direct_affine_maximum_absolute_residual": float(torch.max(torch.abs(direct_residual))),
            "network_minus_direct_maximum_absolute": float(torch.max(torch.abs(network_minus_direct))),
            "network_minus_direct_mse": float(torch.mean(network_minus_direct * network_minus_direct)),
            "records": prefix_rows,
        },
        "rollout": rollout,
        "derivative_geometry": geometry,
        "search_comparison": comparison,
        "adapter_invocations": adapter.calls,
    }

    paused_after = {str(path): sha256(path) for path in (JSONL, PAUSE)}
    protected_after = protected()
    checkpoints_after = hashes()
    report["integrity"] = {
        "paused_artifacts_before": paused_before,
        "paused_artifacts_after": paused_after,
        "paused_artifacts_unchanged": paused_before == paused_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "checkpoint_count": len(checkpoints_before),
        "checkpoint_hashes_unchanged": checkpoints_before == checkpoints_after,
        "resume_not_invoked": True,
        "n1_n2_n3_not_invoked": True,
    }
    report["source_hashes"] = {
        str(Path(__file__)): sha256(Path(__file__)),
        str(Path("experiments/sho/generic_update_network_adapter.py")):
            sha256(Path("experiments/sho/generic_update_network_adapter.py")),
    }
    if not all((
        report["integrity"]["paused_artifacts_unchanged"],
        report["integrity"]["protected_artifacts_unchanged"],
        report["integrity"]["checkpoint_hashes_unchanged"],
    )):
        report["status"] = "FAIL_CLOSED_INTEGRITY"
    REPORT.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean({
        "status": report["status"],
        "teacher": report["teacher"],
        "phi": report["phi"],
        "bound_check": report["bound_check"],
        "prefix": {key: value for key, value in report["prefix"].items()
                   if key != "records"},
        "rollout": report["rollout"],
        "derivative_geometry": {key: value for key, value in geometry.items()
                                if key not in ("gradient", "projected_gradient", "hessian", "eigenvalues")},
        "search_comparison": comparison,
        "integrity": report["integrity"],
        "report": str(REPORT),
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
