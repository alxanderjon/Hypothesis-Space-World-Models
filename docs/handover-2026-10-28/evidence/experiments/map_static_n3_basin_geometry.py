#!/usr/bin/env python3
"""Static local differential-geometry map of direct N3 basin labels."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

from experiments import map_direct_n3_correctability_geometry as basin
from experiments.bouncing_ball import run_reflection_peelback as ball


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/static_n3_basin_geometry")
POINTS = OUT / "points.jsonl"
SOURCE = Path("outputs/direct_n3_correctability_geometry/points.jsonl")
RADII = (0.01, 0.04, 0.12, 0.24)
SOBOL_DIRECTION_COUNT = 8

ANCHORS = (
    ("ball_nearest_PASS", "bouncing_ball", 42, "nearest_pair", 18),
    ("ball_nearest_FAIL", "bouncing_ball", 18, "nearest_pair", 42),
    ("ball_PASS_interior", "bouncing_ball", 29, "interior", None),
    ("ball_wrong_basin_FAIL", "bouncing_ball", 47, "wrong_basin", None),
    ("kepler_nearest_PASS", "kepler", 84, "nearest_pair", 99),
    ("kepler_nearest_FAIL", "kepler", 99, "nearest_pair", 84),
    ("kepler_PASS_interior", "kepler", 235, "interior", None),
    ("kepler_FAIL_exterior", "kepler", 31, "exterior", None),
    ("kepler_threshold_FAIL", "kepler", 196, "threshold_pair", 88),
    ("kepler_threshold_PASS", "kepler", 88, "threshold_pair", 196),
)


def _load(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with POINTS.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _third_tensor(hessian: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(len(theta)):
        columns = []
        for j in range(len(theta)):
            value = hessian[i, j]
            if not value.requires_grad:
                derivative = torch.zeros_like(theta)
            else:
                derivative = torch.autograd.grad(
                    value, theta, retain_graph=True, allow_unused=True
                )[0]
                if derivative is None:
                    derivative = torch.zeros_like(theta)
            columns.append(derivative)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


def _directional(name: str, direction: np.ndarray, gradient: np.ndarray,
                 hessian: np.ndarray, third: np.ndarray) -> dict:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return dict(name=name, direction=vector.tolist(), first=0.0, second=0.0, third=0.0)
    vector = vector / norm
    return dict(
        name=name, direction=vector.tolist(),
        first=float(np.dot(gradient, vector)),
        second=float(vector @ hessian @ vector),
        third=float(np.einsum("i,j,k,ijk->", vector, vector, vector, third)),
    )


def _event_regime(theta: np.ndarray, trajectory: np.ndarray) -> dict:
    signatures = [ball.event_signature(state, theta) for state in trajectory[:127]]
    encoded = json.dumps([list(value) for value in signatures], separators=(",", ":"))
    histogram: dict[str, int] = {}
    for signature in signatures:
        key = "/".join(map(str, signature)) if signature else "none"
        histogram[key] = histogram.get(key, 0) + 1
    return dict(
        hash=hashlib.sha256(encoded.encode()).hexdigest(),
        signatures=[list(value) for value in signatures],
        impacted_transitions=sum(bool(value) for value in signatures),
        total_events=sum(len(value) for value in signatures),
        signature_histogram=histogram,
    )


def _static(theta_np: np.ndarray, config: dict, offset_direction: np.ndarray | None,
            pair_direction: np.ndarray | None, trajectory: np.ndarray | None) -> dict:
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
    loss = config["objective"](theta)
    gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
    raw_hessian = torch.stack([
        torch.autograd.grad(gradient[i], theta, retain_graph=True, create_graph=True)[0]
        for i in range(len(theta))
    ])
    hessian = 0.5 * (raw_hessian + raw_hessian.T)
    third = _third_tensor(hessian, theta)
    gradient_np = gradient.detach().numpy()
    hessian_np = hessian.detach().numpy()
    third_np = third.detach().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_np)
    singular = np.linalg.svd(hessian_np, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    width = config["upper"] - config["lower"]
    scaled_hessian = hessian_np * np.outer(width, width)
    scaled_eigenvalues = np.linalg.eigvalsh(scaled_hessian)
    directional = []
    if np.linalg.norm(gradient_np) > 0.0:
        directional.append(_directional("negative_gradient", -gradient_np,
                                        gradient_np, hessian_np, third_np))
    for index in range(len(theta_np)):
        directional.append(_directional(f"eigenvector_{index}", eigenvectors[:, index],
                                        gradient_np, hessian_np, third_np))
    if offset_direction is not None:
        directional.append(_directional("anchor_offset", offset_direction,
                                        gradient_np, hessian_np, third_np))
    if pair_direction is not None:
        directional.append(_directional("paired_anchor", pair_direction,
                                        gradient_np, hessian_np, third_np))
    local = basin._local_diagnostics(theta_np, config)
    permutation_error = max(
        float(np.max(np.abs(third_np - np.transpose(third_np, axes))))
        for axes in ((1, 0, 2), (2, 1, 0), (0, 2, 1))
    )
    output = dict(
        loss=float(loss.detach()),
        gradient=gradient_np.tolist(),
        gradient_norm=float(np.linalg.norm(gradient_np)),
        scaled_gradient_norm=float(np.linalg.norm(gradient_np * width)),
        hessian=hessian_np.tolist(),
        hessian_eigenvalues=eigenvalues.tolist(),
        hessian_eigenvectors=eigenvectors.tolist(),
        scaled_hessian_eigenvalues=scaled_eigenvalues.tolist(),
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=_finite(condition),
        negative_curvature_count=int(np.count_nonzero(eigenvalues < 0.0)),
        third_derivative_tensor=third_np.tolist(),
        third_derivative_frobenius_norm=float(np.linalg.norm(third_np)),
        third_derivative_permutation_error=permutation_error,
        directional_derivatives=directional,
        existing_route=local["route"],
        existing_damping=local["initial_damping"],
        newton_step_norm=local["newton_step_norm"],
        normalized_step_l2=local["normalized_step_l2"],
        descent_cosine=local["descent_cosine"],
        predicted_decrease=local["predicted_decrease"],
        normalized_bound_margins=local["normalized_bound_margins"],
        minimum_bound_margin=local["minimum_bound_margin"],
        full_newton_step_in_bounds=local["first_newton_step_in_bounds"],
    )
    if trajectory is not None:
        output["event_regime"] = _event_regime(theta_np, trajectory)
    return output


def _directions(seed: int) -> list[np.ndarray]:
    points = qmc.Sobol(5, scramble=True, seed=seed).random_base2(2)
    vectors = []
    for point in points:
        vector = ndtri(np.clip(point, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        vector = vector / np.linalg.norm(vector)
        vectors.extend((vector, -vector))
    assert len(vectors) == SOBOL_DIRECTION_COUNT
    return vectors


def _label(theta: np.ndarray, static: dict, config: dict) -> dict:
    result = basin._run_n3(theta, {"loss": float(static["loss"])}, config)
    return dict(PASS=bool(result["passed"]), final_loss=result["final_loss"],
                accepted_steps=result["accepted_steps"], stop_reason=result["stop_reason"])


def _anchor_definitions(source: list[dict]) -> list[dict]:
    lookup = {(row["system"], int(row["sobol_index"])): row for row in source}
    output = []
    for anchor_id, system, index, role, pair_index in ANCHORS:
        record = lookup[(system, index)]
        pair = None if pair_index is None else lookup[(system, pair_index)]
        output.append(dict(anchor_id=anchor_id, system=system, sobol_index=index,
                           role=role, pair_index=pair_index, source=record, pair=pair))
    # The coarse 256-point map can make an isolated PASS look like an interior
    # point.  Use the already-saved converged endpoint of a PASS trajectory as
    # the interior reference; this is diagnostic only and is never an N3 input.
    recovered_source = dict(lookup[("kepler", 84)])
    recovered_theta = np.asarray(recovered_source["n3"]["final_theta"], dtype=np.float64)
    lower = np.asarray(recovered_source["lower"], dtype=np.float64)
    upper = np.asarray(recovered_source["upper"], dtype=np.float64)
    recovered_source["theta"] = recovered_theta.tolist()
    recovered_source["unit_point"] = ((recovered_theta - lower) / (upper - lower)).tolist()
    output.append(dict(anchor_id="kepler_recovered_solution_PASS", system="kepler",
                       sobol_index=100084, role="recovered_solution_interior",
                       pair_index=None, source=recovered_source, pair=None))
    return output


def _point_specs(anchor: dict) -> list[dict]:
    unit_anchor = np.asarray(anchor["source"]["unit_point"], dtype=np.float64)
    specs = [dict(kind="anchor", direction_id=-1, radius=0.0,
                  normalized_point=unit_anchor, normalized_offset=np.zeros(5))]
    vectors = _directions(20261101 + int(anchor["sobol_index"]) +
                          (0 if anchor["system"] == "bouncing_ball" else 10_000))
    pair_direction = None
    if anchor["pair"] is not None:
        delta = np.asarray(anchor["pair"]["unit_point"]) - unit_anchor
        pair_distance = float(np.linalg.norm(delta)); pair_direction = delta / pair_distance
        for fraction in (0.25, 0.5, 0.75, 1.0):
            offset = fraction * delta
            specs.append(dict(kind="paired_line", direction_id=100, radius=float(np.linalg.norm(offset)),
                              normalized_point=unit_anchor + offset, normalized_offset=offset,
                              pair_fraction=fraction))
    for direction_id, vector in enumerate(vectors):
        for radius in RADII:
            point = unit_anchor + radius * vector
            if np.all(point >= 0.0) and np.all(point <= 1.0):
                specs.append(dict(kind="sobol_direction", direction_id=direction_id,
                                  radius=radius, normalized_point=point,
                                  normalized_offset=radius * vector))
    return specs


def _quantiles(values: list[float]) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values: return dict(count=0)
    array = np.asarray(values)
    return dict(count=len(array), minimum=float(array.min()), median=float(np.median(array)),
                maximum=float(array.max()))


def _summarize(rows: list[dict], anchors: list[dict]) -> dict:
    report = dict(radii=list(RADII), anchors={})
    for anchor in anchors:
        selected = [row for row in rows if row["anchor_id"] == anchor["anchor_id"]]
        anchor_row = next(row for row in selected if row["kind"] == "anchor")
        by_radius = {}
        for radius in RADII:
            group = [row for row in selected if row["kind"] == "sobol_direction" and
                     abs(row["radius"] - radius) < 1e-15]
            by_radius[str(radius)] = dict(
                samples=len(group), PASS=sum(row["n3_label"]["PASS"] for row in group),
                FAIL=sum(not row["n3_label"]["PASS"] for row in group),
                PASS_fraction=(sum(row["n3_label"]["PASS"] for row in group) / len(group)
                               if group else None),
                event_same_fraction=(sum(row.get("same_event_regime_as_anchor",False) for row in group)/len(group)
                                     if group and anchor["system"] == "bouncing_ball" else None),
            )
        opposite = [row for row in selected if row["n3_label"]["PASS"] != anchor_row["n3_label"]["PASS"]]
        same = [row for row in selected if row["n3_label"]["PASS"] == anchor_row["n3_label"]["PASS"] and
                row["kind"] != "anchor"]
        direction_brackets = []
        for direction_id in sorted({row["direction_id"] for row in selected if row["kind"] != "anchor"}):
            line = sorted([row for row in selected if row["direction_id"] == direction_id],
                          key=lambda row: row["radius"])
            previous_label = anchor_row["n3_label"]["PASS"]; previous_radius = 0.0
            transitions = []
            for row in line:
                if row["n3_label"]["PASS"] != previous_label:
                    transitions.append(dict(lower_radius=previous_radius, upper_radius=row["radius"],
                                            from_PASS=previous_label,
                                            to_PASS=row["n3_label"]["PASS"],
                                            event_changed=(not row.get("same_event_regime_as_anchor",True))))
                previous_label=row["n3_label"]["PASS"];previous_radius=row["radius"]
            if transitions:
                direction_brackets.append(dict(direction_id=direction_id, transitions=transitions))
        if anchor["system"] == "bouncing_ball":
            changed_label = [row for row in selected if row["kind"] != "anchor" and
                             row["n3_label"]["PASS"] != anchor_row["n3_label"]["PASS"]]
            event_relation = dict(
                label_changes=len(changed_label),
                label_changes_with_event_change=sum(not row["same_event_regime_as_anchor"] for row in changed_label),
                label_changes_without_event_change=sum(row["same_event_regime_as_anchor"] for row in changed_label),
                event_changes=sum(not row["same_event_regime_as_anchor"] for row in selected
                                  if row["kind"] != "anchor"),
            )
        else: event_relation=None
        report["anchors"][anchor["anchor_id"]] = dict(
            system=anchor["system"], role=anchor["role"], sobol_index=anchor["sobol_index"],
            anchor_PASS=anchor_row["n3_label"]["PASS"],
            anchor_static=anchor_row["static_geometry"],
            samples=len(selected)-1, by_radius=by_radius,
            nearest_opposite_distance=(min(row["radius"] for row in opposite) if opposite else None),
            nearest_same_distance=(min(row["radius"] for row in same) if same else None),
            direction_boundary_brackets=direction_brackets,
            event_relation=event_relation,
            paired_line=[dict(radius=row["radius"], fraction=row.get("pair_fraction"),
                              PASS=row["n3_label"]["PASS"], loss=row["static_geometry"]["loss"],
                              event_same=row.get("same_event_regime_as_anchor"))
                         for row in sorted(selected,key=lambda row:row["radius"])
                         if row["kind"] == "paired_line"],
        )
    for system in ("bouncing_ball", "kepler"):
        selected=[row for row in rows if row["system"]==system]
        for label in (True,False):
            group=[row for row in selected if row["n3_label"]["PASS"]==label]
            report.setdefault("static_distributions",{}).setdefault(system,{})["PASS" if label else "FAIL"]={
                "loss":_quantiles([row["static_geometry"]["loss"] for row in group]),
                "gradient_norm":_quantiles([row["static_geometry"]["gradient_norm"] for row in group]),
                "minimum_eigenvalue":_quantiles([row["static_geometry"]["minimum_eigenvalue"] for row in group]),
                "condition_number":_quantiles([row["static_geometry"]["condition_number"] for row in group]),
                "third_norm":_quantiles([row["static_geometry"]["third_derivative_frobenius_norm"] for row in group]),
            }
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = _load(SOURCE); anchors = _anchor_definitions(source)
    configs = {system: basin._system_config(system) for system in ("bouncing_ball","kepler")}
    ball_trajectory = ball._trajectories()[3]
    existing = _load(POINTS)
    done={(row["anchor_id"],row["kind"],row["direction_id"],row["radius"]) for row in existing}
    for anchor in anchors:
        config=configs[anchor["system"]]; width=config["upper"]-config["lower"]
        unit_anchor=np.asarray(anchor["source"]["unit_point"])
        pair_direction=None
        if anchor["pair"] is not None:
            pair_direction=width*(np.asarray(anchor["pair"]["unit_point"])-unit_anchor)
        anchor_event_hash=None
        for spec in _point_specs(anchor):
            key=(anchor["anchor_id"],spec["kind"],spec["direction_id"],spec["radius"])
            if key in done: continue
            theta=config["lower"]+width*np.asarray(spec["normalized_point"])
            offset=np.asarray(spec["normalized_offset"])
            physical_offset=width*offset
            static=_static(theta,config,(physical_offset if np.linalg.norm(physical_offset)>0 else None),
                           pair_direction,(ball_trajectory if anchor["system"]=="bouncing_ball" else None))
            if spec["kind"]=="anchor" and anchor["system"]=="bouncing_ball":
                anchor_event_hash=static["event_regime"]["hash"]
            if anchor_event_hash is None and anchor["system"]=="bouncing_ball":
                prior=next((row for row in existing if row["anchor_id"]==anchor["anchor_id"] and
                            row["kind"]=="anchor"),None)
                if prior is not None: anchor_event_hash=prior["static_geometry"]["event_regime"]["hash"]
            label=_label(theta,static,config)
            row=dict(anchor_id=anchor["anchor_id"],anchor_role=anchor["role"],
                     source_sobol_index=anchor["sobol_index"],system=anchor["system"],
                     kind=spec["kind"],direction_id=spec["direction_id"],radius=spec["radius"],
                     pair_fraction=spec.get("pair_fraction"),normalized_point=np.asarray(spec["normalized_point"]).tolist(),
                     normalized_offset=offset.tolist(),theta=theta.tolist(),
                     static_geometry=static,n3_label=label)
            if anchor["system"]=="bouncing_ball":
                row["same_event_regime_as_anchor"]=bool(static["event_regime"]["hash"]==anchor_event_hash)
            _append(row);existing.append(row);done.add(key)
            print(json.dumps(dict(anchor=anchor["anchor_id"],kind=spec["kind"],
                                  direction=spec["direction_id"],radius=spec["radius"],
                                  PASS=label["PASS"])),flush=True)
    rows=_load(POINTS); report=_summarize(rows,anchors)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:{"samples":value["samples"],"PASS":value["anchor_PASS"],
                                "nearest_opposite":value["nearest_opposite_distance"]}
                      for key,value in report["anchors"].items()},indent=2),flush=True)


if __name__=="__main__":main()
