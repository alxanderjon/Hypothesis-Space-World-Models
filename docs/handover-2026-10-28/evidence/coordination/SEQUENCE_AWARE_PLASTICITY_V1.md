# Sequence-aware plasticity discriminator v1

Status: prospective before execution.

## Hypothesis and novelty

Can a representation trained through a sequence of observation-only parameter
updates learn to improve on later systems while retaining earlier behavior?

This is an OML-style causal test, not another zero-update transfer test. OML
meta-trains representations for fast online updates with reduced interference
(Javed & White, NeurIPS 2019, <https://arxiv.org/abs/1905.12588>); ANML learns
context-dependent selective activation/plasticity through sequential updates
(Beaulieu et al., 2020, <https://arxiv.org/abs/2002.09571>); MER instead seeks
gradient alignment with replay (Riemer et al., ICLR 2019,
<https://arxiv.org/abs/1810.11910>).

Completed local studies differ materially:

- replay retained examples but did not meta-learn an update-compatible
  representation;
- function distillation consolidated old functions but retained a fixed
  per-system ridge inference rule;
- `source_pretrained_meta_objective_v1` optimized one first-order target-family
  step and did not differentiate through a multi-family sequence or include a
  post-sequence retention objective;
- callable/module/schema studies changed explicit memories or bases rather than
  the shared update dynamics.

## Supplied interface and data

This first diagnostic deliberately reuses the opened controlled oscillator
benchmark. It is not a complexity advance or arbitrary physics:

- state `(q,v)`, known `q'=v`, `dt=.05`, RK4;
- controlled context32 and disjoint query64 banks;
- source families linear, Duffing and Van der Pol;
- pendulum is the already-opened target family;
- family equations and parameters are generator-only.

The shared model is the existing `2 -> 32 SiLU -> 16` feature network (624
parameters). A system still fits a supervised 16x2 ridge head from its observed
context. Every encounter-time shared-weight update is additional supervised
adaptation and is counted explicitly.

## Update algorithm

Source laws in each family are split prospectively into 16 sequence-update laws
and eight within-source outer-evaluation laws. For each outer iteration:

1. start from the persistent shared feature parameters;
2. visit one source law from each family in a fixed response-independent cyclic
   order;
3. for each law, fit its ridge head on even context rows and take one
   differentiable shared-feature step on odd context rows;
4. after all three steps, refit heads on context32 and compute query64 losses on
   one law from each eight-law outer-evaluation pool; the pools rotate
   deterministically so all eight laws are used without outcome-dependent
   sampling;
5. backpropagate through all three inner steps.

The **retention-aware** arm minimizes the mean post-sequence loss over all three
families. The matched **current-only** arm performs the identical inner steps
and computes the same three outer losses, but differentiates only the last
family loss. This isolates the sequence-retention objective; architecture,
systems, states, update count and forward work are otherwise identical.

At target time, eight pendulum adaptation laws arrive in a fixed order. Each
provides context32 only: even16 fit a ridge head, odd16 define one shared-feature
gradient update, and the updated weights persist. Eight disjoint pendulum laws
measure later-system performance before any update of their own. Source-family
represented-development laws measure retention at stages zero and eight.

Controls use the identical target update procedure:

- retention-aware sequence checkpoint;
- current-only sequence checkpoint;
- saved ordinary source-pretrained checkpoint from the prior matched study;
- deterministic random-initialized scratch network.

No family ID, equation, parameter, query response, or future-law label is an
input to an encounter update. Query labels are generated/read only after all
target-sequence predictions are persisted.

## Decision rule

Both seeds must satisfy all of:

- pendulum evaluation error after eight encounters <= 0.8 of its stage-zero
  error;
- retention-aware stage-eight error <= 0.8 of current-only and ordinary
  source-pretrained controls, with at least 6/8 pendulum-law wins;
- source-development aggregate after/before <= 1.10 and at least 20/24 laws
  individually within 10%;
- every prediction finite.

Passing would support learned sequential plasticity inside this supplied
oscillator interface. It would not demonstrate passive trajectories, arbitrary
mechanisms, zero-update recognition, or million-parameter scalability. Failure
would close this fixed architecture/three-step objective, not all continual
meta-learning.

## Prospective exact work

- Four fresh meta fits (two arms x two seeds), 1,600 outer updates.
- 4,800 differentiable inner steps and 9,600 training ridge solves.
- 307,200 training feature rows.
- Eight target sequences, 64 persistent target updates, 1,024 target/evaluation
  ridge solves and 61,440 prediction rows.
- 10,752 repeated one-step simulations / 43,008 RK4 RHS evaluations.
- No new systems or confirmation cohort; all data are opened development.
- 600 seconds wall, 1 GiB RSS, 128 MiB artifacts. No automatic retry.

