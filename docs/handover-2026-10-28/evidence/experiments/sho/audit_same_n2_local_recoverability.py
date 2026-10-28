"""Isolated monotone local-path witness search; never imported by live N3.

Uses SciPy 1.13's trust-region-reflective step primitives, bounded in physical
parameter displacement by the historical first accepted stabilized-N3 move.
Acceptance is the unchanged assembled float64 MSE, with additional segment
checks. No fixture, teacher parameters, restarts, or alternative N2 endpoints.
"""
from pathlib import Path
import json
import os
import time
import platform
import numpy as np
import torch
import scipy
from scipy.linalg import svd
from scipy.optimize._lsq.common import (
    CL_scaling_vector, solve_lsq_trust_region, update_tr_radius,
    make_strictly_feasible,
)
from scipy.optimize._lsq.trf import select_step
from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import (
    OptInResidualN3Runtime, STABILIZED_RESIDUAL, residual_and_jacobian,
)
from experiments.sho.run_companion6_stabilized_residual_live import (
    N2_SOURCE, DATA, bits_equal, clean, atomic, sha256,
)
from experiments.sho.validate_generic_update_network_adapter import protected
from experiments.sho.deterministic_full_k_framework import hashes

OUT = Path('outputs/sho/same_n2_local_recoverability_audit')
REFERENCE = Path('outputs/sho/stabilized_residual_n3_from_n2/start_verification.json')
HISTORY = Path('outputs/sho/stabilized_residual_n3_from_n2/stabilized.jsonl')


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if (OUT / 'accepted.jsonl').exists():
        raise RuntimeError('Existing isolated path: refusing to overwrite/restart')
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected()
    checkpoint_before = hashes()
    source_before = {str(p): sha256(p) for p in Path('experiments').rglob('*.py')}
    n2 = json.loads(N2_SOURCE.read_text())['result']
    reference = json.loads(REFERENCE.read_text())
    trajectory = torch.from_numpy(np.load(DATA)['prefix_6'].copy())
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
    x0 = np.array(n2['theta'], dtype=np.float64)
    theta = torch.from_numpy(x0.copy())
    result, grad, hess, gn, pre, route, direction, qp = runtime.derivatives_and_direction(theta)
    r0, J0 = residual_and_jacobian(adapter, trajectory, theta)
    checks = {
        'n2_evaluations': n2['evaluations'] == 1500,
        'phi': bits_equal(x0, reference['phi']),
        'loss': bits_equal(result.loss, reference['loss']) and bits_equal(result.loss, n2['loss']),
        'expected_loss': bits_equal(n2['loss'], 0.0018503643453732527),
        'gradient': bits_equal(grad, reference['gradient']),
        'hessian': bits_equal(hess, reference['hessian']),
        'residual': bits_equal(r0, reference['residual']),
        'jacobian': bits_equal(J0, reference['J']),
        'mu': bits_equal(qp['damping'], reference['mu']),
        'bounds': bool(np.all(adapter.lower == -1.99) and np.all(adapter.upper == 1.99)),
    }
    if not all(checks.values()):
        raise RuntimeError(f'Start verification failed: {checks}')
    with HISTORY.open() as f:
        old0, old1 = json.loads(next(f)), json.loads(next(f))
    step_cap = float(np.linalg.norm(np.array(old1['phi']) - old0['phi']))
    lb, ub = adapter.lower.copy(), adapter.upper.copy()
    eig, vec = torch.linalg.eigh(hess)
    u, singular, vh = torch.linalg.svd(J0, full_matrices=False)
    start = dict(checks=checks, phi=x0, loss=n2['loss'], residual=r0, jacobian=J0,
                 gradient=grad, hessian=hess, mu=qp['damping'], bounds=[lb, ub],
                 hessian_eigenvalues=eig, hessian_condition=float(torch.linalg.cond(hess)),
                 jacobian_singular_values=singular, jacobian_rank=int(torch.linalg.matrix_rank(J0)),
                 hessian_rank=int(torch.linalg.matrix_rank(hess)),
                 gradient_eigenprojections=vec.T @ grad,
                 gradient_singular_projections=vh @ grad,
                 weakest_singular_directions=vh[-6:],
                 negative_curvature_directions=vec[:, eig < 0],
                 active_bounds=np.where((x0 == lb) | (x0 == ub))[0],
                 step_cap=step_cap, stabilized_primary_direction=direction)
    atomic(OUT / 'start_verification.json', start)
    manifest = dict(method='SciPy 1.13 trust-region reflective primitives; exact physical MSE acceptance',
                    step_cap=step_cap, step_cap_source=str(HISTORY),
                    locality='Each move <= historical first accepted N3 displacement; no random/global jumps. Quarter-segment losses must strictly decrease.',
                    trust_controls='SciPy Coleman-Li scaling, theta=max(.995,1-||g*v||inf), standard .25/.75 quality ratio radius updates; all singular values retained',
                    success=1e-20, ftol=None, gtol=None, xtol=None,
                    fixture_access=False, teacher_parameter_access=False,
                    environment=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, machine=platform.machine()),
                    n2_source_hash=sha256(N2_SOURCE), reference_hash=sha256(REFERENCE),
                    script_hash=sha256(Path(__file__)), source_before=source_before,
                    frozen_before=before, checkpoint_before=checkpoint_before)
    atomic(OUT / 'manifest.json', manifest)
    x = x0.copy()
    loss = float(n2['loss'])
    r, J = r0.numpy().copy(), J0.numpy().copy()
    m, n = J.shape
    Delta = step_cap / np.sqrt(np.max(ub - lb))
    alpha = 0.0
    accepted = 0
    trials = 0
    objective_calls = 0
    path_length = 0.0
    max_step = 0.0
    max_distance = 0.0
    rows = []
    rejected = 0
    started = time.monotonic()
    stop = None

    def objective(value):
        nonlocal objective_calls
        objective_calls += 1
        return float(adapter.objective(torch.from_numpy(value.copy()), trajectory))

    def save_state(update=None):
        nonlocal r, J
        t = torch.from_numpy(x.copy())
        derivatives = adapter.derivatives(t, trajectory)
        residual, jacobian = residual_and_jacobian(adapter, trajectory, t)
        if not bits_equal(derivatives.loss, loss):
            raise RuntimeError('Accepted loss reconstruction differs')
        r, J = residual.numpy().copy(), jacobian.numpy().copy()
        e = torch.linalg.eigvalsh(derivatives.symmetric_hessian)
        s = torch.linalg.svdvals(jacobian)
        g = derivatives.gradient.numpy()
        projected = g.copy()
        projected[((x == lb) & (g >= 0)) | ((x == ub) & (g <= 0))] = 0
        row = dict(step=accepted, phi=x.copy(), loss=loss,
                   gradient=g, gradient_norm=float(np.linalg.norm(g)),
                   projected_gradient_norm=float(np.linalg.norm(projected)),
                   hessian_eigenvalues=e, jacobian_singular_values=s,
                   hessian_condition=float(torch.linalg.cond(derivatives.symmetric_hessian)),
                   jacobian_condition=float(s[0]/s[-1]),
                   active_bounds=np.where((x == lb) | (x == ub))[0],
                   minimum_bound_margin=float(np.min(np.minimum(x-lb, ub-x))),
                   cumulative_path_length=path_length, distance_from_start=float(np.linalg.norm(x-x0)),
                   update=update)
        rows.append(clean(row))
        with (OUT / 'accepted.jsonl').open('a') as f:
            f.write(json.dumps(clean(row), allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        if accepted < 6 or accepted % 50 == 0 or loss < 1e-20:
            print(f'LOCAL step={accepted} loss={loss:.17g} path={path_length:.6g} distance={np.linalg.norm(x-x0):.6g} trials={trials}', flush=True)

    save_state()
    try:
        while loss >= 1e-20:
            g = J.T @ r
            if np.linalg.norm(g) == 0:
                stop = 'zero_residual_model_gradient'
                break
            v, dv = CL_scaling_vector(x, g, lb, ub)
            d = np.sqrt(v)
            if not np.all(d > 0):
                stop = 'degenerate_reflective_scaling'
                break
            Delta = min(Delta, step_cap / np.max(d))
            diag_h = g * dv
            gh = d * g
            Jh = J * d
            augmented = np.vstack((Jh, np.diag(np.sqrt(diag_h))))
            U, s, Vt = svd(augmented, full_matrices=False)
            uf = U.T @ np.concatenate((r, np.zeros(n)))
            theta_backoff = max(.995, 1 - np.linalg.norm(g*v, ord=np.inf))
            accepted_trial = False
            seen_trials = set()
            while not accepted_trial:
                ph, alpha, nit = solve_lsq_trust_region(n, m, uf, s, Vt.T, Delta, initial_alpha=alpha)
                p = d * ph
                step, step_h, prediction = select_step(x, Jh, diag_h, gh, p, ph, d, Delta, lb, ub, theta_backoff)
                candidate = make_strictly_feasible(x + step, lb, ub, rstep=0)
                if not np.all(np.isfinite(candidate)):
                    stop = 'nonfinite_diagnostic_direction'
                    break
                if np.array_equal(candidate, x) or candidate.tobytes() in seen_trials:
                    stop = 'floating_point_no_further_improvement'
                    break
                seen_trials.add(candidate.tobytes())
                displacement = candidate-x
                norm_step = float(np.linalg.norm(displacement))
                if not np.all((candidate >= lb) & (candidate <= ub)):
                    raise RuntimeError('Infeasible diagnostic candidate')
                if norm_step > step_cap and not np.isclose(norm_step, step_cap, rtol=8*np.finfo(float).eps, atol=0):
                    stop = 'LOCALITY_GUARD_STEP_CAP_EXCEEDED'
                    break
                trials += 1
                candidate_loss = objective(candidate)
                segment_losses = []
                segment_ok = candidate_loss < loss and float(g @ displacement) < 0
                if segment_ok:
                    previous = loss
                    for fraction in (.25, .5, .75):
                        segment_loss = objective(x + fraction*displacement)
                        segment_losses.append(segment_loss)
                        if not (candidate_loss < segment_loss < previous):
                            segment_ok = False
                            break
                        previous = segment_loss
                reduction = (loss-candidate_loss)*m/2
                new_delta, ratio = update_tr_radius(Delta, reduction, prediction, np.linalg.norm(step_h), np.linalg.norm(step_h)>.95*Delta)
                if not segment_ok:
                    new_delta = min(new_delta, .25*np.linalg.norm(step_h))
                if not np.isfinite(new_delta) or new_delta <= 0:
                    stop = 'numerical_trust_radius_exhausted'
                    break
                old_delta = Delta
                alpha *= Delta/new_delta
                Delta = new_delta
                if segment_ok:
                    update = dict(displacement=displacement, displacement_norm=norm_step,
                                  previous_loss=loss, actual_decrease=loss-candidate_loss,
                                  predicted_decrease=prediction*2/m, quality_ratio=ratio,
                                  radius=old_delta, regularization_multiplier=alpha*new_delta/old_delta,
                                  segment_losses=segment_losses, directional_derivative_mse=float(g @ displacement)*2/m)
                    x = candidate
                    loss = candidate_loss
                    path_length += norm_step
                    max_step = max(max_step, norm_step)
                    max_distance = max(max_distance, float(np.linalg.norm(x-x0)))
                    accepted += 1
                    save_state(update)
                    accepted_trial = True
                else:
                    rejected += 1
            if stop:
                break
        if loss < 1e-20:
            stop = 'HANDOFF_THRESHOLD_REACHED'
    except KeyboardInterrupt:
        stop = 'EXTERNAL_INTERRUPTION_DIAGNOSTIC_INCOMPLETE'
    final = dict(status='SAME_N2_ENDPOINT_LOCALLY_RECOVERABLE' if loss < 1e-20 else 'LOCAL_RECOVERABILITY_NOT_ESTABLISHED',
                 stop_reason=stop, accepted_steps=accepted, trial_candidates=trials,
                 rejected_trials=rejected, objective_evaluations=objective_calls,
                 final_phi=x, final_loss=loss, handoff=loss < 1e-20,
                 path_length=path_length, max_displacement=max_step, max_distance_from_start=max_distance,
                 elapsed_seconds=time.monotonic()-started, locality_step_cap=step_cap,
                 frozen_unchanged=before == protected(), checkpoints_unchanged=checkpoint_before == hashes(),
                 source_unchanged=source_before == {str(p):sha256(p) for p in Path('experiments').rglob('*.py')},
                 accepted_trace_hash=sha256(OUT/'accepted.jsonl'))
    atomic(OUT/'result.json', final)
    print(json.dumps(clean(final), indent=2), flush=True)


if __name__ == '__main__':
    main()
