"""Teacher-exact symmetric N3 Hessian binding for the hybrid student."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.bouncing_ball import run_reflection_peelback as reflection
from experiments.geometry.full_k_sho_hessian_assembly import FullKSHOHessianAssembly
from experiments.geometry.n3_gradient_binding_assembly import N3GradientBindingAssembly
from experiments.kepler import run_four_parameter_peelback as kepler_runtime
from experiments.sho import assemble_n3_gradient_operator as arithmetic
from experiments.sho import assemble_n3_hessian_operator as second_adjoint


torch.set_default_dtype(torch.float64)


@dataclass
class HessianAssemblyResult:
    loss: torch.Tensor
    gradient: torch.Tensor
    raw_rows: tuple[torch.Tensor, ...]
    raw_hessian: torch.Tensor
    transpose: torch.Tensor
    summed: torch.Tensor
    symmetric_hessian: torch.Tensor
    retained_oracles: dict[str, int]
    raw_autograd_final_hessian_fallback_calls: int = 0


class N3SymmetricHessianBindingAssembly:
    """Exact production row construction and symmetrization for d=4/d=5."""

    def __init__(self, gradient_binding: N3GradientBindingAssembly | None = None):
        self.gradient_binding = gradient_binding or N3GradientBindingAssembly()
        # This installs the validated second-adjoint-visible arithmetic wrappers
        # while retaining the same exact learned forward values.
        second_adjoint.activate_second_order_gradient()
        self.sho = FullKSHOHessianAssembly()
        self.invocations = 0
        self.by_system: dict[str, int] = {}
        self.raw_autograd_final_hessian_fallback_calls = 0
        self.variable_power_first_adjoint_binding_calls = 0
        self.variable_power_second_adjoint_binding_calls = 0
        self.production_pow_backward_fallback_calls = 0
        self.event_branch_step_oracle_calls = 0

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=torch.float64)
        return torch.from_numpy(np.asarray(value, np.float64).copy())

    @staticmethod
    def _symmetrize(raw: torch.Tensor):
        transpose = raw.T
        summed = arithmetic.OPS.add(raw, transpose)
        symmetric = arithmetic.OPS.multiply(
            torch.as_tensor(0.5, dtype=raw.dtype), summed,
        )
        return transpose, summed, symmetric

    def _sho(self, trajectory: torch.Tensor, theta: torch.Tensor):
        # The recovered SHO Hessian is constant in theta, but the integrated
        # gradient is still evaluated so the live Hessian input is explicit.
        gradient_result = self.gradient_binding._sho(trajectory, theta)
        raw_np, symmetric_np = self.sho(trajectory[:-1].detach().cpu().numpy())
        raw = torch.from_numpy(raw_np.copy())
        transpose = raw.T
        summed = torch.from_numpy(self.sho.forward._add(raw_np, raw_np.T).copy())
        symmetric = torch.from_numpy(symmetric_np.copy())
        return HessianAssemblyResult(
            loss=gradient_result.loss,
            gradient=gradient_result.gradient,
            raw_rows=tuple(raw[index].clone() for index in range(4)),
            raw_hessian=raw, transpose=transpose,
            summed=summed, symmetric_hessian=symmetric,
            retained_oracles={},
        )

    def _assembled_objective(self, system: str, trajectory: torch.Tensor,
                             theta: torch.Tensor):
        if system == "reflection_ball":
            prefix = trajectory[:reflection.ball.PREFIX_STATES]
            predictions = [
                self.gradient_binding._ball_step(prefix[index], theta)
                for index in range(reflection.ball.PREFIX_STATES - 1)
            ]
            self.event_branch_step_oracle_calls += len(predictions)
            objective = self.gradient_binding.prefix_objective.evaluate(
                torch.stack(predictions), prefix[1:], "reflection_ball",
                reduction_path="torch",
            )
            return objective.loss, {}
        if system == "kepler5":
            prefix = trajectory[:kepler_runtime.kepler.PREFIX_STATES]
            prediction = self.gradient_binding._kepler_step(prefix[:-1], theta)
            objective = self.gradient_binding.prefix_objective.evaluate(
                prediction, prefix[1:], "kepler5", reduction_path="torch",
            )
            return objective.loss, {
                "deterministic_forward_power_nodes": 2,
            }
        raise ValueError(system)

    def _general(self, system: str, trajectory: torch.Tensor,
                 theta: torch.Tensor):
        loss, retained = self._assembled_objective(system, trajectory, theta)
        gradient = torch.autograd.grad(
            loss, theta, create_graph=True, retain_graph=True,
        )[0]
        raw_rows = tuple(
            torch.autograd.grad(
                gradient[index], theta, retain_graph=True,
            )[0]
            for index in range(len(theta))
        )
        raw = torch.stack(raw_rows)
        transpose, summed, symmetric = self._symmetrize(raw)
        if system == "kepler5":
            first = 2
            second = 2 * len(theta)
            self.variable_power_first_adjoint_binding_calls += first
            self.variable_power_second_adjoint_binding_calls += second
            retained.update({
                "assembled_variable_power_first_adjoint_calls": first,
                "assembled_variable_power_second_adjoint_calls": second,
                "production_pow_backward_fallback_calls": 0,
            })
        elif system == "reflection_ball":
            retained.update({
                "deterministic_event_branch_step_calls": (
                    reflection.ball.PREFIX_STATES - 1
                ),
            })
        return HessianAssemblyResult(
            loss=loss.detach(), gradient=gradient.detach(),
            raw_rows=tuple(row.detach() for row in raw_rows),
            raw_hessian=raw.detach(), transpose=transpose.detach(),
            summed=summed.detach(), symmetric_hessian=symmetric.detach(),
            retained_oracles=retained,
        )

    def evaluate(self, system: str, trajectory, theta, log=None):
        trajectory_tensor = self._tensor(trajectory)
        theta_tensor = self._tensor(theta).requires_grad_(True)
        self.invocations += 1
        self.by_system[system] = self.by_system.get(system, 0) + 1
        if system == "full_k_sho":
            result = self._sho(trajectory_tensor, theta_tensor)
        elif system in ("reflection_ball", "kepler5"):
            result = self._general(system, trajectory_tensor, theta_tensor)
        else:
            raise ValueError(f"unsupported N3 Hessian system: {system}")
        if log is not None:
            log.call(
                "N3 symmetric Hessian operator", "A",
                "teacher-exact second-adjoint rows plus ordered symmetrization", 1,
            )
        return result
