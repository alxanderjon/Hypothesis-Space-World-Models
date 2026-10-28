# Heterogeneous common-interface pilot v1

Status: Stage 0 interface preflight complete; Stage 1 remains a frozen proposal.
No learner fitting, simulator generation, or evaluation was performed.

Stage 0 evidence is in
`outputs/heterogeneous_common_interface_stage0_v1_recovery1/result.json`.
Attempt 1 is preserved separately: it stopped before its first trajectory read
because it guessed a directory stem from a provenance-family label. Recovery 1
resolved artifacts through the authoritative corpus manifest and did not repeat
scientific work. It read 822 cached trajectories containing 398,038 states and
2,559,896 finite scalar values. All three source layouts round-tripped through
the neutral `[2,4]` state tensor at exactly zero error. Peak RSS was 34,553,856
bytes and wall time 0.6562 seconds.

## Decision question

Can retained weights learned across genuinely different state cardinalities and
transition types make identification of a new system easier from 32 observed
transitions, without giving the learner a simulator name, governing equation,
physical parameters, integrator name, or mechanism-specific adapter?

This is not a test of arbitrary physics discovery. It is a test of whether a
single observation-conditioned interface is viable across the mechanisms that
already exist in this repository. The current 822-case corpus cannot answer the
question: it contains 320 distinct SHO update matrices, but reflection-ball and
Kepler vary initial conditions around one teacher rule each.

## Existing evidence that this must not duplicate

- The eight-family oscillator work uses a supplied two-dimensional `(q,v)`
  state, `q'=v`, `dt=.05`, RK4, controlled state banks, and a compact ridge
  head. It is mechanism-diverse only inside that supplied second-order scalar
  interface.
- `cross_mechanism_passive32_bridge_v1` replaced controlled resets with two
  16-transition passive trajectories and failed its declared gate. It did not
  test different state cardinalities, collision events, or multi-entity states.
- Direct TNP, set-attention, recurrent, compact-code, learned-basis, GP, RKHS,
  program-search, and learned-callable studies already test many alternative
  readouts within narrower interfaces. This pilot does not reopen those sweeps.

## Generator-side mechanisms

All equations, parameters, resets, and numerical integrators below are
simulator-only. They are used to generate transitions and stratify folds, never
as learner inputs.

1. **Scalar oscillator families** (opened development): one entity with one
   position and one velocity; eight acceleration families; `dt=.05`; RK4. The
   existing corpus supplies 32 controlled transitions per system and passive
   episodes for a smaller bridge study.
2. **Reflection ball** (opened development): one two-dimensional entity with
   state `(x,y,vx,vy)`; `dt=.05`; event-driven elastic reflections in a fixed
   box. Existing cases vary initial conditions, not the collision law.
3. **Planar two-body/Kepler** (opened development): two two-dimensional
   entities with state `(x1,y1,vx1,vy1,x2,y2,vx2,vy2)`; `dt=.02`;
   velocity-Verlet. Existing canonical cases vary initial conditions around one
   teacher `(gamma,p)=(1,3)`, although the simulator can generate other
   `(gamma,p)` settings.
4. **Full-K SHO** (opened development): one two-channel delay-state entity;
   discrete linear update with a 2x2 matrix. This is not labeled as
   position/velocity because the stored state is a delay coordinate.

The first feasibility stage reads existing opened transitions only. Any later
scientific comparison must prospectively generate multiple laws for ball and
Kepler or add other mechanism families; otherwise it may not claim
cross-mechanism rule transfer.

The source audit confirmed the exact limitation: all 320 full-K SHO cases have
distinct teacher matrices, while all 246 reflection-ball cases share one
teacher vector and all 256 Kepler cases share one teacher vector. Consequently,
Stage 1 cannot responsibly execute the four-arm transfer gate from the frozen
822 trajectories alone. It would measure state-layout/initial-condition
transfer, not the stated new-rule capability.

### Smallest defensible acquisition prerequisite

If Stage 1 is approved for execution review, freeze the following development
acquisition before generating responses:

- 32 reflection-ball adapter laws sampled prospectively over the existing
  two-parameter `K_ball(a,b)` bounds, with one 32-transition context trajectory
  and one independent 64-transition query trajectory per law;
- 32 Kepler laws sampled prospectively over the existing `(gamma,p)` bounds,
  with the same 32+64 passive-transition schedule;
- 24 laws/mechanism are training-only in folds where that mechanism is
  represented; eight are opened development. In a whole-mechanism fold all 32
  are excluded from shared-weight training and only each system's declared
  32-transition context is visible at inference;
- 6,144 new transition steps total. Count every collision event and both Kepler
  acceleration evaluations separately. No replacement trajectories after a
  numerical rejection; the full law/reset schedule and rejection rule are
  sealed first.

This uses already implemented update families rather than inventing new
equations. It still supplies those families at generator level and does not
create an untouched mechanism confirmation. The ball `a,b` parameters are the
repository's transport/reflection adapter, not a claim of realistic material
variation. All resulting systems are development evidence.

## Learner-visible common interface

Each encounter contains an unordered context set of 32 records and a query:

```
context record = {
  x_t:       [E, C],
  x_t1:      [E, C],
  entity_mask: [E],
  channel_mask: [E, C],
  delta_t:   scalar,
  continuity_flag: scalar
}
query = {x_t, entity_mask, channel_mask, delta_t}
target = x_t1 - x_t
```

`E<=2` and `C=4` in this pilot. Channels have only common observational
semantics: two value coordinates followed by their two rate coordinates when
the source actually supplies that distinction. Full-K SHO instead uses two
generic value channels with the rate-channel mask off. `continuity_flag`
indicates whether a transition crosses a detected discontinuity, but it must be
derivable from observed transition geometry alone; collision-event metadata is
not exposed. If a robust observation-only rule cannot be frozen, the flag is
removed from every arm.

Supplied assumptions are therefore explicit: fully observed Markov state,
known record boundaries, known elapsed time, shared numeric coordinate units
within each record, and entity exchangeability. No actions are present. Reset
parameters, family IDs, masses, `G`, wall locations, restitution, force
parameters, teacher vectors, integrator identity, and simulator source are
quarantined.

The pilot is fully observed. Partial observability is a separate later problem;
mixing it into the first comparison would confound interface viability with
state reconstruction.

## Dimension-general learner and per-system inference

The proposed learner is one shared relation/set network, not a padded flat MLP:

- encode every observed entity transition with a shared MLP;
- aggregate within a transition using permutation-invariant attention;
- aggregate the 32 transition embeddings into a context representation;
- decode a query entity from its state, elapsed time, other-entity messages,
  and the context representation;
- predict a normalized state increment and an observation-noise scale.

The model has no mechanism-specific heads. Persistent knowledge is the shared
network weights and training-only normalization statistics. Encounter-time
state is the context representation computed by a forward pass; no gradient
update is used in the principal arm. A separately declared compact-adaptation
arm may fit at most one 16-dimensional context vector and must count that as
per-system supervised adaptation.

This architecture borrows the entity/interaction interface from graph-network
simulators, while the context-set conditioning addresses new-system
identification. It does not import conservation laws, force direction, contact
geometry, or an integration scheme.

## Folds and safeguards

### Stage 0: interface feasibility, zero scientific fit

- Read and hash existing opened trajectories.
- Emit only shape, mask, time-step, finite-value, ordering, and quarantine
  checks; do not copy teacher parameters into learner records.
- Verify round-trip reconstruction of source states from the neutral schema.
- Verify that removing quarantined metadata leaves no source/family string or
  mechanism-specific field.

Stage 0 can reject the protocol. Passing it is not scientific evidence.

### Stage 1: opened-development leave-one-mechanism analysis

Use opened data only. Every fold withholds a complete generator-side family;
parameter settings and initial conditions from that family may not occur in
training. Because ball and Kepler currently have only one rule each, folds that
hold them out are interface stress tests, not estimates of new-law transfer.

The four matched arms are:

1. **shared + context**: shared weights and the correct 32-transition context;
2. **shared + blind**: identical weights with the context replaced by the
   training-population mean;
3. **shared + wrong context**: context cyclically reassigned within the same
   evaluation fold, fixed before scoring;
4. **scratch**: identical architecture initialized from the same distribution
   and trained only on the target fold's permitted training split with matched
   optimizer-update and example budgets.

A small local-linear predictor fitted to the 32 context transitions is a
non-neural reference, not a capacity-matched arm.

All normalization is fit inside each training fold. `delta_t` is visible and
may become a mechanism cue because existing generators use different fixed
steps. Report a time-blind diagnostic, but do not select between time-visible
and time-blind models using held-family outcomes.

### Stage 2: future confirmation, not part of this pilot

Before opening another simulator family, freeze the interface, model,
normalization, model-selection rule, and every gate. A valid confirmation must
contain a mechanism family absent from architecture and model selection, not
merely new coefficients or initial conditions. Candidate examples are a
population interaction system or a coupled-particle system, but the choice must
be made before its responses are inspected.

## Development decision rule

All metrics are computed per system first, then aggregated by held mechanism.
The pilot supports continued work only if both seeds satisfy all of:

1. shared+context improves median normalized one-step MSE by at least 20% over
   scratch on at least three held mechanism groups;
2. shared+context improves by at least 20% over shared+blind on the same groups;
3. correct context beats cyclic wrong context on at least 75% of systems in
   those groups;
4. five-step rollout error is finite and no worse than scratch on those groups;
5. after a fixed chronology of source groups, performance on previously seen
   groups regresses by no more than 10% relative to the frozen pre-update
   checkpoints.

Failure is informative but local: it rejects this interface/model/budget, not
all heterogeneous transfer. A pass remains opened-development evidence and
cannot establish unique rule recovery, calibrated uncertainty, continual
learning, or arbitrary-mechanism generality.

## Prospective execution ceiling for Stage 1

This ceiling is intentionally small and includes repeats:

- at most 8 mechanism groups, 32 systems/group, 32 context and 64 query
  transitions/system;
- 2 seeds, 4 arms, at most 1,000 optimizer updates/fit;
- at most 16 fits and 16,000 optimizer updates total;
- at most 262,144 learner prediction blocks and 2,048 rollout blocks;
- at most 600 seconds wall time, 2 GiB peak RSS, and 128 MiB new artifacts;
- zero new teacher/simulator steps in Stage 0 and Stage 1;
- pre-call counters, atomic result persistence, and shape-derived completion
  validation are mandatory.

If existing stored data cannot furnish the declared contexts and queries
without regeneration, Stage 1 stops. Any acquisition then receives a separate,
prospectively counted protocol; cached initial-condition diversity may not be
relabeled as law diversity.

## What recurrence and accumulated knowledge mean here

No hidden motif is selected. Reuse is measured by chronological forward
benefit across complete, predeclared mechanism groups and retention on earlier
groups. The persistent object is the shared network, while each encounter
supplies only its 32 transitions. Improvement only on familiar parameter
settings, or improvement from the unconditional prior with no correct-context
advantage, does not count as system identification. One chronology is
development evidence; a claim of persistent accumulation eventually requires
multiple frozen orders or a prospectively reserved order.

## Primary-literature rationale and limits

- Graph Network-based Simulators demonstrate that shared entity/interaction
  representations scale across particle counts and several physical domains,
  but they do not by themselves solve few-shot unknown-rule identification.
- Context-Informed Dynamics Adaptation explicitly conditions adaptation on a
  new-environment trajectory, but assumes environments share learnable
  structure.
- Function-contrastive and neural-process work motivates learning a context
  representation of a function, but local set/attention/GP failures mean the
  present project must retain blind/wrong-context and scratch controls.
- Hybrid/event-aware graph ODE work supports treating collisions as a genuine
  discontinuity challenge, not smoothing them into another oscillator family.

These precedents justify the contrast, not its outcome. The pilot remains much
smaller and more structured than broad physical-simulation benchmarks.
