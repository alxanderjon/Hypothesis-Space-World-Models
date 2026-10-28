"""Teacher-exact N3 gradient binding for the current hybrid systems.

The autograd engine remains deterministic graph-traversal scaffolding for the
assembled reflection-ball and Kepler reverse graphs.  The final gradient is
therefore produced by locked local adjoints, not by differentiating the raw
teacher objective.  Kepler's positive-base variable power deliberately keeps
production ``PowBackward`` as the declared temporary primitive oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from experiments import audit_known_structure as kepler5
from experiments.bouncing_ball import run_reflection_peelback as reflection
from experiments.geometry.torch_matmul2_fma_assembly import TorchMatmul2FMAAssembly
from experiments.geometry.torch_mm_backward_dot_assembly import TorchMMBackwardDotAssembly
from experiments.geometry.prefix_residual_mse_assembly import PrefixResidualMSEAssembly
from experiments.kepler.assemble_n3_variable_power_adjoint import (
    variable_power as assembled_variable_power,
)
from experiments.kepler import run_four_parameter_peelback as kepler_runtime
from experiments.sho import assemble_n3_gradient_operator as arithmetic


torch.set_default_dtype(torch.float64)


@dataclass
class GradientAssemblyResult:
    loss: torch.Tensor
    gradient: torch.Tensor
    prediction: torch.Tensor | None
    prediction_adjoint: torch.Tensor | None
    retained_oracles: dict[str, int]
    raw_autograd_final_gradient_fallback_calls: int = 0


class _ExactMatvec2(torch.autograd.Function):
    """Exact active 2x2 Torch matvec value with locked product adjoints."""

    forward_assembly = TorchMatmul2FMAAssembly()

    @staticmethod
    def forward(ctx, matrix, vector):
        ctx.save_for_backward(vector)
        value = _ExactMatvec2.forward_assembly(
            vector.detach().cpu().numpy()[None, :],
            matrix.detach().cpu().numpy(),
        )[0]
        return torch.from_numpy(value.copy())

    @staticmethod
    def backward(ctx, incoming):
        (vector,) = ctx.saved_tensors
        values = []
        for output in range(2):
            for coordinate in range(2):
                # Use the active locked arithmetic binding rather than the
                # checkpoint module directly.  The Hessian integration swaps
                # this binding for its validated second-adjoint-visible form.
                values.append(arithmetic.multiply(
                    incoming[output], vector[coordinate],
                ))
        return torch.stack(values).reshape(2, 2), None


class N3GradientBindingAssembly:
    """Live teacher-exact gradient source for SHO, reflection ball, and Kepler."""

    def __init__(self):
        if arithmetic.OPS is None:
            arithmetic.OPS = arithmetic.LearnedArithmetic()
        self.sho_forward = TorchMatmul2FMAAssembly()
        self.sho_backward = TorchMMBackwardDotAssembly()
        self.prefix_objective = PrefixResidualMSEAssembly()
        self.invocations = 0
        self.by_system: dict[str, int] = {}
        self.variable_power_oracle_calls = 0
        self.variable_power_binding_invocations = 0
        self.production_pow_backward_fallback_calls = 0
        self.raw_autograd_final_gradient_fallback_calls = 0

    @staticmethod
    def _trajectory(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=torch.float64)
        return torch.from_numpy(np.asarray(value, np.float64).copy())

    def _sho(self, trajectory: torch.Tensor, theta: torch.Tensor):
        inputs = trajectory[:-1]
        targets = trajectory[1:]
        matrix = theta.reshape(2, 2)
        prediction_np = self.sho_forward(
            inputs.detach().cpu().numpy(), matrix.detach().cpu().numpy(),
        )
        prediction = torch.from_numpy(prediction_np.copy()).requires_grad_(True)
        objective = self.prefix_objective.evaluate(
            prediction, targets, "full_k_sho", reduction_path="torch",
        )
        loss = objective.loss
        incoming = torch.autograd.grad(loss, prediction)[0].detach()
        gradient_np = self.sho_backward(
            incoming.detach().cpu().numpy().T,
            inputs.detach().cpu().numpy(),
        ).reshape(-1)
        return GradientAssemblyResult(
            loss=loss.detach(),
            gradient=torch.from_numpy(gradient_np.copy()),
            prediction=prediction.detach(),
            prediction_adjoint=incoming,
            retained_oracles={},
        )

    @staticmethod
    def _ball_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        matrix = theta[:4].reshape(2, 2)
        rho = theta[4]
        x, y, vx, vy = state.unbind()
        ux, wx = _ExactMatvec2.apply(matrix, torch.stack((x, vx))).unbind()
        uy, wy = _ExactMatvec2.apply(matrix, torch.stack((y, vy))).unbind()
        remaining: float | torch.Tensor = reflection.ball.DT
        for _ in range(reflection.ball.MAX_COLLISIONS):
            if float(remaining) <= reflection.ball.EPS:
                break
            tx: float | torch.Tensor = math.inf
            ty: float | torch.Tensor = math.inf
            if float(ux.detach()) > reflection.ball.EPS:
                tx = arithmetic.divide(
                    arithmetic.subtract(reflection.ball.XMAX, x), ux,
                )
            elif float(ux.detach()) < -reflection.ball.EPS:
                tx = arithmetic.divide(
                    arithmetic.subtract(reflection.ball.XMIN, x), ux,
                )
            if float(uy.detach()) > reflection.ball.EPS:
                ty = arithmetic.divide(
                    arithmetic.subtract(reflection.ball.YMAX, y), uy,
                )
            elif float(uy.detach()) < -reflection.ball.EPS:
                ty = arithmetic.divide(
                    arithmetic.subtract(reflection.ball.YMIN, y), uy,
                )
            tx_value, ty_value = float(tx), float(ty)
            hit_time: float | torch.Tensor = tx if tx_value <= ty_value else ty
            if float(hit_time) > float(remaining) or math.isinf(float(hit_time)):
                x = arithmetic.add(x, arithmetic.multiply(ux, remaining))
                y = arithmetic.add(y, arithmetic.multiply(uy, remaining))
                remaining = 0.0
                break
            x = arithmetic.add(x, arithmetic.multiply(ux, hit_time))
            y = arithmetic.add(y, arithmetic.multiply(uy, hit_time))
            remaining = arithmetic.subtract(remaining, hit_time)
            hit_x = abs(tx_value - float(hit_time)) < reflection.ball.EPS
            hit_y = abs(ty_value - float(hit_time)) < reflection.ball.EPS
            if hit_x:
                x = torch.as_tensor(
                    reflection.ball.XMAX if float(ux.detach()) > 0.0
                    else reflection.ball.XMIN,
                )
                ux = arithmetic.multiply(rho, ux)
                wx = arithmetic.multiply(rho, wx)
            if hit_y:
                y = torch.as_tensor(
                    reflection.ball.YMAX if float(uy.detach()) > 0.0
                    else reflection.ball.YMIN,
                )
                uy = arithmetic.multiply(rho, uy)
                wy = arithmetic.multiply(rho, wy)
        return torch.stack((x, y, wx, wy))

    def _ball(self, trajectory: torch.Tensor, theta: torch.Tensor):
        prefix = trajectory[:reflection.ball.PREFIX_STATES]
        predictions = [
            self._ball_step(prefix[index], theta)
            for index in range(reflection.ball.PREFIX_STATES - 1)
        ]
        prediction_stack = torch.stack(predictions)
        objective = self.prefix_objective.evaluate(
            prediction_stack, prefix[1:], "reflection_ball",
            reduction_path="torch",
        )
        loss = objective.loss
        incoming = torch.autograd.grad(
            loss, predictions, retain_graph=True,
        )
        gradient = torch.autograd.grad(loss, theta)[0]
        return GradientAssemblyResult(
            loss=loss.detach(), gradient=gradient.detach(),
            prediction=prediction_stack.detach(),
            prediction_adjoint=torch.stack([value.detach() for value in incoming]),
            retained_oracles={},
        )

    def _kepler_field(self, states: torch.Tensor, theta: torch.Tensor):
        c1, c2, _, _, exponent = theta.unbind()
        dx = arithmetic.subtract(states[..., 4], states[..., 0])
        dy = arithmetic.subtract(states[..., 5], states[..., 1])
        radius_squared = arithmetic.add(
            arithmetic.multiply(dx, dx), arithmetic.multiply(dy, dy),
        )
        # Fixed sqrt has an already validated adjoint. The following custom
        # variable-power node retains only the production forward power value.
        radius = torch.sqrt(radius_squared)
        # Forward stays production torch.pow inside the frozen custom Function;
        # only PowBackward/PowBackward double-backward are replaced.
        denominator = assembled_variable_power(radius, exponent)
        self.variable_power_binding_invocations += 1
        common = arithmetic.divide(
            arithmetic.multiply(float(kepler_runtime.kepler.G), 1.0),
            denominator,
        )
        base1 = arithmetic.multiply(common, float(kepler_runtime.kepler.MASS_2))
        base2 = arithmetic.multiply(common, float(kepler_runtime.kepler.MASS_1))
        scale1 = arithmetic.multiply(
            base1, arithmetic.divide(c1, float(kepler5.GT[0])),
        )
        scale2 = arithmetic.multiply(
            base2, arithmetic.divide(c2, float(kepler5.GT[1])),
        )
        return (
            arithmetic.multiply(scale1, dx),
            arithmetic.multiply(scale1, dy),
            arithmetic.multiply(-scale2, dx),
            arithmetic.multiply(-scale2, dy),
        )

    def _kepler_step(self, states: torch.Tensor, theta: torch.Tensor):
        _, _, eta, drift, _ = theta.unbind()
        dt = float(kepler_runtime.kepler.DT)
        half_dt = 0.5 * dt
        a1x, a1y, a2x, a2y = self._kepler_field(states, theta)
        vx = arithmetic.add(states[..., 2], arithmetic.multiply(half_dt, a1x))
        vy = arithmetic.add(states[..., 3], arithmetic.multiply(half_dt, a1y))
        wx = arithmetic.add(states[..., 6], arithmetic.multiply(half_dt, a2x))
        wy = arithmetic.add(states[..., 7], arithmetic.multiply(half_dt, a2y))
        x = arithmetic.add(states[..., 0], arithmetic.multiply(
            dt, arithmetic.multiply(drift, vx),
        ))
        y = arithmetic.add(states[..., 1], arithmetic.multiply(
            dt, arithmetic.multiply(drift, vy),
        ))
        z = arithmetic.add(states[..., 4], arithmetic.multiply(
            dt, arithmetic.multiply(drift, wx),
        ))
        w = arithmetic.add(states[..., 5], arithmetic.multiply(
            dt, arithmetic.multiply(drift, wy),
        ))
        drifted = torch.stack((x, y, vx, vy, z, w, wx, wy), dim=-1)
        b1x, b1y, b2x, b2y = self._kepler_field(drifted, theta)
        return torch.stack((
            x, y,
            arithmetic.add(vx, arithmetic.multiply(
                half_dt, arithmetic.multiply(eta, b1x))),
            arithmetic.add(vy, arithmetic.multiply(
                half_dt, arithmetic.multiply(eta, b1y))),
            z, w,
            arithmetic.add(wx, arithmetic.multiply(
                half_dt, arithmetic.multiply(eta, b2x))),
            arithmetic.add(wy, arithmetic.multiply(
                half_dt, arithmetic.multiply(eta, b2y))),
        ), dim=-1)

    def _kepler(self, trajectory: torch.Tensor, theta: torch.Tensor):
        prefix = trajectory[:kepler_runtime.kepler.PREFIX_STATES]
        prediction = self._kepler_step(prefix[:-1], theta)
        objective = self.prefix_objective.evaluate(
            prediction, prefix[1:], "kepler5", reduction_path="torch",
        )
        loss = objective.loss
        incoming = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
        gradient = torch.autograd.grad(loss, theta)[0]
        return GradientAssemblyResult(
            loss=loss.detach(), gradient=gradient.detach(),
            prediction=prediction.detach(), prediction_adjoint=incoming.detach(),
            retained_oracles={
                "deterministic_forward_power_calls": 2,
                "assembled_variable_power_adjoint_calls": 2,
                "production_pow_backward_fallback_calls": 0,
            },
        )

    def evaluate(self, system: str, trajectory, theta, log=None):
        trajectory_tensor = self._trajectory(trajectory)
        theta_tensor = (
            theta.detach().clone().to(dtype=torch.float64)
            if isinstance(theta, torch.Tensor)
            else torch.from_numpy(np.asarray(theta, np.float64).copy())
        ).requires_grad_(True)
        self.invocations += 1
        self.by_system[system] = self.by_system.get(system, 0) + 1
        if system == "full_k_sho":
            result = self._sho(trajectory_tensor, theta_tensor)
        elif system == "reflection_ball":
            result = self._ball(trajectory_tensor, theta_tensor)
        elif system == "kepler5":
            result = self._kepler(trajectory_tensor, theta_tensor)
        else:
            raise ValueError(f"unsupported N3 gradient system: {system}")
        if log is not None:
            log.call(
                "N3 gradient operator", "A",
                "frozen learned/recovered exact reverse-adjoint assembly", 1,
            )
        return result
