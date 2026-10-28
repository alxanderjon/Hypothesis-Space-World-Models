"""Isolated per-case raw-state SHO hypothesis; no recovery or initialization policy."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call

from hypothesis_space_world_models.models.sho_exact_solution_decoder import _make_mlp_encoder
from experiments.geometry.prefix_residual_mse_assembly import PrefixResidualMSEAssembly


@dataclass
class NetworkDerivatives:
    loss: torch.Tensor
    gradient: torch.Tensor
    raw_hessian: torch.Tensor
    symmetric_hessian: torch.Tensor


class GenericSHOUpdateNetworkAdapter:
    derivative_oracle = 'TEMPORARY_TORCH_AUTOGRAD_D_ORACLE'
    recovery_status = 'NOT_YET_TESTED'

    def __init__(self):
        # Meta tensors provide architecture only: no persistent numerical weights,
        # no RNG consumption, and no initialization offered to the search.
        with torch.device('meta'):
            self.template = _make_mlp_encoder(hidden_dims=[4], activation='silu').to(torch.float64)
        self.template.requires_grad_(False)
        self.layout = []
        offset = 0
        for name, parameter in self.template.named_parameters():
            count = parameter.numel()
            self.layout.append(dict(name=name, shape=list(parameter.shape), start=offset, stop=offset+count))
            offset += count
        self.dimension = offset
        if self.dimension != 22:
            raise RuntimeError('unexpected generic network architecture')
        self.lower = np.full(self.dimension, -1.99, dtype=np.float64)
        self.upper = np.full(self.dimension, 1.99, dtype=np.float64)
        self.mse = PrefixResidualMSEAssembly()
        self.calls = dict(forward=0, objective=0, derivative_oracle=0, rollout_steps=0)

    @staticmethod
    def _tensor(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float64 or value.device.type != 'cpu':
                raise ValueError('CPU float64 tensors required')
            return value
        return torch.as_tensor(np.asarray(value, dtype=np.float64))

    def parameters_from_flat(self, phi):
        phi = self._tensor(phi)
        if tuple(phi.shape) != (self.dimension,):
            raise ValueError(f'phi must have shape [{self.dimension}]')
        return {entry['name']: phi[entry['start']:entry['stop']].reshape(entry['shape'])
                for entry in self.layout}

    def forward(self, phi, inputs):
        inputs = self._tensor(inputs)
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError('inputs must have shape [N,2]')
        self.calls['forward'] += 1
        return functional_call(self.template, self.parameters_from_flat(phi), (inputs,), strict=True)

    def _trajectory(self, trajectory):
        states = self._tensor(trajectory)
        if states.ndim != 2 or states.shape[1] != 2 or len(states) < 2:
            raise ValueError('trajectory must have shape [T,2], T >= 2')
        return states

    def objective_terms(self, phi, trajectory):
        states = self._trajectory(trajectory)
        self.calls['objective'] += 1
        return self.mse.evaluate(self.forward(phi, states[:-1]), states[1:],
                                 'full_k_sho', reduction_path='torch')

    def objective(self, phi, trajectory):
        return self.objective_terms(phi, trajectory).loss

    def derivatives(self, phi, trajectory):
        states = self._trajectory(trajectory)
        flat = self._tensor(phi).detach().clone().requires_grad_(True)
        self.parameters_from_flat(flat)
        # Explicit native derivative oracle for the same residual/square/sum/divide
        # graph. Do not differentiate through frozen arithmetic model internals.
        prediction = self.forward(flat, states[:-1])
        residual = prediction - states[1:]
        loss = (residual * residual).sum() / residual.numel()
        with torch.no_grad():
            assembled = self.objective(flat, states)
        if not torch.equal(loss.detach().view(torch.int64), assembled.view(torch.int64)):
            raise RuntimeError('native derivative-oracle scalar differs from assembled objective')
        gradient = torch.autograd.grad(loss, flat, create_graph=True)[0]
        raw = torch.stack([torch.autograd.grad(gradient[i], flat, retain_graph=True)[0]
                           for i in range(self.dimension)])
        symmetric = 0.5 * (raw + raw.T)
        self.calls['derivative_oracle'] += 1
        return NetworkDerivatives(loss.detach(), gradient.detach(), raw.detach(), symmetric.detach())

    def rollout(self, phi, initial_state, steps=4096):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        flat = self._tensor(phi).detach().clone()
        self.parameters_from_flat(flat)
        state = self._tensor(initial_state).detach().clone()
        if tuple(state.shape) != (2,):
            raise ValueError('initial state must have shape [2]')
        rows = [state]
        with torch.no_grad():
            for _ in range(steps):
                state = self.forward(flat, state[None, :])[0]
                rows.append(state)
                self.calls['rollout_steps'] += 1
        return torch.stack(rows)

    def archive_box(self, phi, evaluation_count):
        flat = self._tensor(phi).detach().numpy()
        self.parameters_from_flat(phi)
        if not isinstance(evaluation_count, int) or evaluation_count < 1:
            raise ValueError('positive actual evaluation count required')
        d = len(flat)
        radius = (self.upper - self.lower) / float(evaluation_count) ** (1.0 / d)
        return np.maximum(self.lower, flat-radius), np.minimum(self.upper, flat+radius)
