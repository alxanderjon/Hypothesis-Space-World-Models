# Hypothesis Space Recovery

## Core Question

How much structure must be built into a model class before the true dynamics can be reliably recovered?

The goal is to study this by starting with a model whose output space is restricted so that the true solution is guaranteed to be representable, then progressively relaxing those restrictions.

## Initial Test System

Simple Harmonic Oscillator:

x'' + omega^2 x = 0

with solution family:

x(t) = a cos(omega t) + b sin(omega t)

## Stage 0: Exact Solution-Space Model

The first model should not be a generic neural network.

It should have a neural encoder that maps initial conditions to coefficients:

(x0, v0) -> (a, b)

Then a fixed decoder constructs the trajectory:

x_hat(t) = a cos(omega t) + b sin(omega t)

The decoder is not learned. This guarantees the model can only output trajectories in the SHO solution family.

Stage 0 is only a proof of concept that the architecture contains the exact solution family. Since the dynamics are analytically enforced by the decoder, the primary metrics are:

- coefficient recovery
- trajectory reconstruction error

## Stage 1: Unknown Global Omega

Stage 1 keeps the SHO analytic solution family fixed, but removes omega from the model.

The data are still generated from a true SHO system:

x'' + omega^2 x = 0

but the true omega is used only for:

- dataset generation
- evaluation and recovery metrics

The true omega must not be provided to the model.

The Stage 1 model learns both:

- the coefficient encoder
- one global shared frequency parameter

The initial Stage 1 architecture is the most structured version:

(x0, v0) -> linear encoder -> (a, b)

with one learned global scalar omega_hat shared across every trajectory.

The decoder remains analytic:

x_hat(t) = a cos(omega_hat t) + b sin(omega_hat t)

The first Stage 1 experiment is:

stage1_linear_encoder_global_omega

Its purpose is to test whether the model can jointly recover the exact coefficient map and the global system frequency when omega is unknown to the decoder.

Stage 1 metrics include:

- omega_hat
- true_omega
- omega_error
- relative_omega_error
- coefficient recovery
- trajectory reconstruction error

## Relaxation Experiments

After Stage 0 works, progressively relax the hypothesis space:

1. Exact SHO solution-space decoder with known omega
2. SHO solution-space decoder with learnable omega
3. Larger Fourier-style basis
4. General linear dynamics model
5. Hamiltonian dynamics model
6. Fully neural autoregressive dynamics model

## Evaluation

For Stage 0, compare models using:

- coefficient recovery
- trajectory reconstruction error

For later stages with learned dynamics, compare models using:

- coefficient recovery
- trajectory reconstruction error
- long-horizon rollout error
- recovery of governing equation structure
- energy conservation error

## Comparison Philosophy

The primary comparison in this project is a best-achieved recovery comparison.

The goal is not to determine which model performs best under one fixed optimizer configuration. The goal is to understand how hypothesis-space structure affects recovery of the true dynamics.

When comparing model classes, keep the data and evaluation protocol fixed:

- dataset generation
- train/test split
- random seed, unless explicitly studying robustness
- omega and other system parameters
- trajectory length
- evaluation metrics

Optimizer settings may vary by model class when needed:

- optimizer
- learning rate
- batch size
- number of epochs
- stopping criterion
- optimizer-specific hyperparameters

Every run must record its optimization settings in setup.json.

### Best-Achieved Comparisons

Best-achieved comparisons are the primary scientific comparison.

They allow model-specific optimization choices and train each model until convergence or until a documented stopping condition is reached. These comparisons ask:

Can this hypothesis space recover the true structure when trained with a reasonable configuration for that model?

### Shared-Budget Comparisons

Shared-budget comparisons are secondary.

They use the same optimizer settings and training budget across models. These comparisons ask:

How do different model classes optimize under identical training resources?

Shared-budget results should not be treated as the primary evidence of representational power.

## Experiment Output Convention

Experiment outputs are organized by system, stage, and comparison mode:

outputs/
  sho/
    stage0/
      best_achieved/
        YYYY-MM-DD_run_name/
      best_achieved_long/
        YYYY-MM-DD_run_name/
      best_achieved_lr_controlled/
        YYYY-MM-DD_run_name/
      epsilon_sweep/
        YYYY-MM-DD_run_name/
      lr_sweep/
        YYYY-MM-DD_run_name/
      lr_sweep_extended/
        YYYY-MM-DD_run_name/
      sanity_check_analytic_solution/
      stage0_exact_solution_decoder/
        YYYY-MM-DD_run_name/
    stage1/
      unknown_global_omega/
        YYYY-MM-DD_linear_encoder_omega_init_1/
        YYYY-MM-DD_linear_encoder_omega_init_1_diagnostics/
    stage2/
      YYYY-MM-DD_run_name/

Stage 1 unknown-global-omega runs should use:

outputs/sho/stage1/unknown_global_omega/YYYY-MM-DD_<run_name>/

For example:

outputs/sho/stage1/unknown_global_omega/2026-06-06_linear_encoder_omega_init_1/
outputs/sho/stage1/unknown_global_omega/2026-06-06_linear_encoder_omega_init_1_diagnostics/

Each experimental model run should save the standard bundle:

- index.html
- setup.json
- metrics.json
- training_history.json
- learned_parameters.pt
- trajectory_comparison.png
- phase_space_comparison.png
- training_loss.png
- optimizer_analysis.txt

If an item is not applicable to a model, keep the section present in index.html and mark it as N/A.

setup.json must include:

- comparison_mode
- model_type
- encoder_type
- activation, if applicable
- hidden_dim, if applicable
- optimizer
- learning rate
- batch size
- number of epochs completed
- max epochs
- stopping criterion
- stopping monitor
- stopping patience, if applicable
- stopping min_delta, if applicable
- stopped_because
- dataset and evaluation settings

## Goal

Determine how relaxing architectural constraints affects whether the learned model recovers the true governing dynamics and whether it maintains accurate long-horizon prediction.

## Research handover

Research was stopped on 28 October 2026 for an independent reassessment. See the [handover and evidence snapshot](docs/research-handover-2026-10-28.md) for the objective, exact experimental contracts, results, limitations and artifact inventory. Archived continuation instructions are inactive.
