# Independent-mechanism task-generative phase

Status: design and source-validation phase; no PDE data downloaded and no model
fit executed yet.  Official source was inspected at commit
`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`.

## Why this phase

The local oscillator studies now include static and sequential coefficient
priors, compact latent posteriors, a direct TNP, exact GPs, a GP mixture,
function-output regularization, replay/distillation and shared-weight Fisher
updates.  The latest sequential 32-coefficient head population retained old
laws but was worse than an isotropic reset on every target law.  Another prior,
head, covariance or small synthetic grammar variation is not a new hypothesis.

The next test should therefore use mechanisms and splits defined independently
of learner outcomes.  PDEBench is the proposed source because its public suite
already includes separately implemented 1D advection, Burgers and
reaction-diffusion generators, multiple initial/boundary conditions and
parameters, and standard data-generation code.  This is still simulated PDE
data and still supplies a common scalar-field/coordinate interface; it is not
arbitrary physics or real-world confirmation.

## Model-visible contract

Pending source validation, each example is a scalar field `u(x,t)` on the
benchmark's published grid.  Context tokens contain only normalized
`(x,t,u)` observations from an initial time window.  Query tokens contain
`(x,t)` and request future `u`.  The learner does not receive the PDE name,
equation, coefficients, numerical solver identity, family ID or future values.
Boundary masks and coordinate normalization, if required by the official data,
are global declared interface metadata rather than per-mechanism labels.

Source validation found that the apparent common interface is not free.  The
official multi-solution configurations use `nx=1024` for all three mechanisms,
but advection saves every `.01` through `t=2` (100 trajectories), Burgers every
`.05` through `t=2` (10,000 trajectories, domain `[-1,1]`), and
reaction-diffusion every `.01` through `t=1` (100 trajectories).  Therefore a
common 64-point/32-time tensor would require a prospectively fixed spatial
downsample, time selection/interpolation, domain-coordinate normalization and
balanced 100-trajectory subsample.  Those transforms are supplied engineering
assumptions and may erase mechanism-specific scales.  They must be tested with
per-mechanism scratch baselines before any transfer interpretation.

The official downloader reports full-corpus sizes of 47 GB (advection), 93 GB
(Burgers) and 62 GB (1D reaction-diffusion).  Downloading all 202 GB is outside
the present bounded pilot.  The repository exposes individual example shards,
but their exact byte sizes and trajectory contents must be checked before any
download.  A small pilot may instead regenerate 100 trajectories per mechanism
from the pinned official code; that still incurs simulator cost and inherits
the official solver/initial-condition distributions, and must not be described
as observational real-world evidence.

The first model is a direct class-model Transformer: a context encoder with
self-attention, query-to-context cross-attention and a shared probabilistic
field decoder.  There is no analytic feature/basis library and no coefficient
target.  This resembles published in-context system-identification models and
the local direct TNP architecturally; its novelty is **not** attention.  The
causal question is whether persistent learning across independently specified
mechanisms reduces data/update cost on a held-out equation under a raw common
interface.  A failure cannot be repaired by claiming the benchmark was designed
for the model.

## Persistent and per-encounter quantities

Persistent state is the shared network weights plus a fixed-size reservoir of
real past training windows.  No hidden PDE program or task ID is stored.  A new
system first predicts purely from its observed context.  If adaptation is
enabled, ordinary gradient updates use only that system's revealed training
window plus the fixed replay reservoir; every update is counted as supervised
per-system adaptation.  No claim of zero-shot discovery is made.  An optional
adapter is excluded from the first comparison so per-system fitted quantities
are exactly the shared weights changed by the declared updates.

## Splits and controls

Source validation must establish compatible units, grid/time shapes, boundary
conditions and licenses before selection.  The intended whole-mechanism split
uses two PDE mechanisms for training/model selection and the third as a
development holdout; roles rotate only in a prospectively fixed three-fold
analysis.  Initial conditions and parameter settings remain disjoint within
each mechanism.  No held-mechanism window enters normalization, model selection
or replay.

Four matched arms are required:

1. accumulated shared model with fixed-size real replay;
2. the same pretrained model frozen before the held mechanism;
3. same architecture trained from scratch on the held mechanism with the same
   number of held-mechanism windows and update evaluations;
4. joint-offline upper comparator with equal total source+held windows and
   update evaluations, reported as non-chronological.

Replay memory, window counts, optimizer evaluations and total prediction tokens
must be matched or disclosed.  A state-only/global-mean baseline checks whether
the model merely memorizes typical field evolution.  A shuffled context-value
control checks use of observed field pairing.

## Decision evidence

Useful accumulated knowledge requires, in every fixed fold and at least two
seeds: (a) lower held-mechanism error than scratch at matched held data and
compute, (b) lower error than frozen pretraining, (c) source retention within
10% for at least 80% of source validation trajectories and aggregate <=1.1,
and (d) intact context beating shuffled and mean controls.  A prospective
effect-size threshold and absolute error scale will be fixed only after reading
the official training split, never from held-mechanism outcomes.  Any later
confirmation needs an untouched benchmark/mechanism, not another parameter
setting of an opened PDE.

## Preflight and bounds still unresolved

Before execution, acquire only the official 100-trajectory subsets (or generate
them from the pinned source if licensing and dependencies permit) and verify
actual saved shapes before choosing index maps.  Record exact file hashes, trajectory
counts, grid/time sizes, normalization, selected folds, model parameter count,
token/update ceilings, replay bytes and wall/RSS/artifact limits.  The target is
approximately one million shared parameters so scaling is measured directly;
the exact architecture and all-in budget remain unsealed until dataset shapes
are verified.  This document is therefore an implementable direction, not a
claim that training is currently running or that PDEBench guarantees transfer.
