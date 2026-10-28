#!/usr/bin/env python3
"""Validate the new two-body authority and record transfer-interface facts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.kepler import deterministic_teacher as teacher

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
OUT = Path("outputs/kepler/transfer_hypothesis_audit")


def residual_jacobian(trajectory: np.ndarray, prefix_states: int) -> torch.Tensor:
    theta = torch.from_numpy(teacher.TEACHER_THETA.copy()).requires_grad_(True)
    observed = torch.from_numpy(trajectory)
    residual = torch.cat([teacher.torch_step(observed[index], theta) - observed[index + 1]
                          for index in range(prefix_states - 1)])
    rows = [torch.autograd.grad(residual[index], theta, retain_graph=True)[0]
            for index in range(len(residual))]
    return torch.stack(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [(0.10, 0.8, 0.0), (0.35, 1.0, 0.3), (0.60, 1.2, 1.1), (0.25, 1.4, 2.4)]
    rows = []; overall_max = 0.0; exact_values = total_values = 0
    default_trajectory = None
    for serial, (eccentricity, periapsis, orientation) in enumerate(cases):
        initial = teacher.initial_state(eccentricity, periapsis, orientation)
        numpy_values = teacher.numpy_rollout(initial)
        torch_values = teacher.torch_rollout(torch.from_numpy(initial)).detach().numpy()
        difference = np.abs(numpy_values - torch_values)
        exact = int(np.count_nonzero(numpy_values == torch_values)); total = numpy_values.size
        rows.append(dict(case=serial, eccentricity=eccentricity, periapsis=periapsis,
                         orientation=orientation, exact_values=exact, total_values=total,
                         maximum_absolute_error=float(np.max(difference)),
                         rms_error=float(np.sqrt(np.mean(difference*difference))),
                         finite=bool(np.isfinite(torch_values).all())))
        overall_max = max(overall_max, float(np.max(difference)))
        exact_values += exact; total_values += total
        if serial == 1: default_trajectory = numpy_values
    assert default_trajectory is not None
    identifiability = []
    for prefix_states in (2, 3, 4, 8, 16, 32, 64, 128):
        jacobian = residual_jacobian(default_trajectory, prefix_states)
        singular = torch.linalg.svdvals(jacobian).detach().numpy()
        rank = int(np.linalg.matrix_rank(jacobian.detach().numpy()))
        identifiability.append(dict(prefix_states=prefix_states, residual_values=len(jacobian),
                                    rank=rank, singular_values=singular.tolist(),
                                    condition=float(singular[0]/singular[-1]) if singular[-1] > 0 else "inf"))
    teacher_loss_numpy = float(teacher.prefix_loss_numpy(teacher.TEACHER_THETA, default_trajectory)[0])
    teacher_loss_torch = float(teacher.prefix_loss_torch(torch.from_numpy(teacher.TEACHER_THETA),
                                                         torch.from_numpy(default_trajectory)))
    report = dict(
        status="TEACHER_AND_HYPOTHESIS_DEFINED",
        repository_search="No suitable Kepler/two-body implementation found in working tree or Git history.",
        teacher=dict(state_order=("x1","y1","vx1","vy1","x2","y2","vx2","vy2"),
                     units="dimensionless", dt=float(teacher.DT), integrator="velocity Verlet (kick-drift-kick)",
                     G=float(teacher.G), masses=(float(teacher.MASS_1),float(teacher.MASS_2)),
                     no_softening=True, prefix_states=teacher.PREFIX_STATES,
                     trajectory_shape=[teacher.TOTAL_STEPS+1,8]),
        hypothesis=dict(parameters=("shared_gravitational_strength_gamma","radial_denominator_exponent_p"),
                        teacher=teacher.TEACHER_THETA.tolist(),
                        bounds=np.stack((teacher.LOWER,teacher.UPPER),axis=1).tolist(),
                        acceleration="a1=+gamma*G*m2*dr/||dr||^p; a2=-gamma*G*m1*dr/||dr||^p"),
        adapter_validation=dict(cases=rows, exact_values=exact_values,total_values=total_values,
                                maximum_absolute_error=overall_max,
                                tolerance=2e-13, passed=overall_max <= 2e-13,
                                teacher_loss_numpy=teacher_loss_numpy,
                                teacher_loss_torch=teacher_loss_torch),
        identifiability=identifiability,
        interface=dict(parameter_dimension=2,
                       n1="Reuse normalized 2-D pool/search adapter with vector physical bounds.",
                       n2="Existing dimension-generic direction controller and scalar line controller; no dimension change.",
                       n3="Control/solve path is dimension-generic. Add a Kepler objective dispatch and validate variable-power/log adjoints before learned gradient/Hessian use.",
                       rollout="Use velocity-Verlet step recursively from the last observed state."),
        exact_teacher_containment=(teacher_loss_numpy == 0.0),
    )
    (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
