# Heterogeneous common-interface milestone v1

## What executed

1. Stage-0 lossless schema audit over all 822 frozen trajectories.
2. Development acquisition of 32 distinct ball-adapter laws and 32 distinct
   Kepler laws: 6,144 transition steps plus explicit collision/integrator work.
3. A fold-safe three-group package with 32 SHO, 32 ball, and 32 Kepler laws.
4. Twelve whole-group context-model fits (6,000 updates), with blind, wrong
   context, time, scratch, and five-step rollout comparisons.
5. Two represented-mechanism prerequisite fits (1,000 updates).
6. Twelve rapid-acquisition fits (1,200 new target updates) comparing retained
   source weights with matched reset weights.
7. Six training-only source/target gradient comparisons (12 backward passes).

All cohorts are development evidence. No untouched mechanism confirmation was
opened.

## Verified results

### Interface

The neutral `[2 entities, 4 channels]` tensor exactly round-trips 398,038 cached
states with zero error. This is syntactic interoperability, not semantic
neutrality. The learner still receives body order, position/velocity channel
grouping when available, masks, elapsed time, fully observed Markov states, and
one/two-entity structure.

### Rule diversity acquisition

All 32 ball and 32 Kepler parameter vectors produced distinct fixed-query
transition maps. Minimum pairwise maximum absolute differences were `0.0027861`
and `0.0010540`. This establishes different finite transition functions, not
identifiability from passive32 contexts. Ball supplied only 32 collision events
across 3,072 steps; contact-specific evidence is sparse.

### Whole-mechanism transfer: NOT_SUPPORTED

The shared model trained on two groups and held the third out entirely. It was
`13.1x-18.6x` worse than target-family scratch for SHO, `86,361x-91,880x`
worse for ball, and `282.7x-1,359x` worse for Kepler. Correct and cyclic-wrong
contexts were nearly indistinguishable. Five-step rollout also failed every
fold. The model therefore did not infer a previously unseen transition type
from context.

### Represented-mechanism context use: SUPPORTED prerequisite

With all three mechanism groups represented during shared training, the exact
same model used context strongly in both seeds:

- correct/blind aggregate error ratios: `0.064-0.098`;
- correct/wrong ratios: `0.0054-0.0725`;
- correct context won `7/8` or `8/8` laws in every group.

Thus the context pathway is functional. The failed whole-group result is
specifically an extrapolation failure beyond represented transition types, not
simple context collapse.

### Rapid full-weight acquisition: NOT_SUPPORTED

After exactly four target-mechanism training laws and 100 updates, retained
source weights did not reliably beat a reset model and caused severe source
regression (`1.35x-18,137x`). This is supervised per-mechanism adaptation, not
zero-shot recognition. It does not support persistent accumulation in this
formulation.

### Gradient mechanism

Training-only gradients do not support one universal cause:

- SHO target gradients were `412x-628x` larger than source gradients;
- ball source/target cosine was `-0.200` and `-0.023`;
- Kepler cosine was near zero and `89%-97%` of target-gradient energy lay in
  the decoder.

This mixture of scaling, interference, and decoder mismatch means another
optimizer/replay/weighting tweak would not isolate the scientific issue.

## Decision

Close this exact monolithic entity/set model, normalization, and three-group
whole-mechanism comparison. Do not tune its width, updates, learning rate,
context dimension, replay ratio, or masks on the opened results.

The next experiment should first remove the major causal confound: in the
current pilot, an unseen mechanism is also an unseen state layout, timestep,
channel semantics, and integration regime. A model can fail because it has
never learned that interface, not because retained experience cannot help
identify a new law.

### Recommended next phase: same-interface mechanism transfer

Use a prospectively broader but single neutral interface with multiple genuinely
different mechanisms per interface. The smallest defensible design is:

- **one-entity 1D position/velocity interface**: the eight existing oscillator
  mechanisms, using passive trajectories rather than controlled response banks;
- **one-entity 2D position/velocity interface**: at least two prospectively
  distinct hybrid/force mechanisms, not merely new `K_ball(a,b)` coefficients;
- **two-entity 2D position/velocity interface**: at least two interaction
  mechanisms, not merely new Kepler `(gamma,p)` values.

Within each interface, hold out a complete mechanism while keeping state
semantics, entity count, timestep distribution, and observation budget matched.
Only then ask whether retained weights improve context-conditioned prediction
or fixed-budget adaptation versus reset. Ball/Kepler cross-layout transfer
becomes a secondary stress test, not the primary accumulation gate.

Required controls remain: correct/blind/cyclic context, matched reset, fixed
data and update budgets, source retention, and whole-mechanism splits. No
mechanism ID, equation, physical parameter, or simulator-specific adapter may
enter the learner.

### Alternatives and costs

1. **Modular expert expansion with observation-only routing.** This may avoid
   destructive full-weight updates, but on the current three groups a router
   can exploit obvious masks/timesteps. It becomes meaningful only after the
   same-interface corpus above exists.
2. **Active/controlled probes.** These can reduce ambiguity but change the
   information contract. They should be compared to equal-count random probes,
   not presented as a solution to passive inference.
3. **Broad public physics benchmark immediately.** Graph simulators offer much
   richer domains, but moving now would confound corpus engineering, partial
   observability, action semantics, dimensionality, and continual learning.
   This is a later confirmation path, not the next causal test.

The recommended next phase is therefore corpus/interface work, not another
model variation. It retains the system-general objective while making the next
failure interpretable.
