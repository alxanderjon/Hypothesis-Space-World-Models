# Heterogeneous context-scaling milestone v1

Date: 2026-10-28

## Completed result

`heterogeneous_context_scaling_discriminator_v1` completed its prospectively
declared 12 fits and 6,000 updates with no new data or teacher calls. It changed
only the coordinate scale: each active channel was divided by an RMS computed
from that system's permitted 32-row context, and output increments were scaled
the same way. The 22,724-parameter architecture, data, whole-mechanism folds,
seeds, optimizer, updates, masks, timestep and original success gate stayed
fixed.

Classification: `NOT_SUPPORTED`.

- Ball shared MSE fell from about `1.45` to `0.170--0.182` (`0.117--0.125x`).
- Kepler shared MSE fell to `0.000109--0.000116` (`0.025--0.134x`).
- SHO shared MSE worsened to `1.390--1.451` (`1.55--1.63x`).
- Every original gate cell failed. Correct-context wins were only `3--5/8`;
  correct, cyclic-code-wrong and full-wrong errors were nearly tied. Shared
  error remained `16.5--935x` scaled scratch, and rollout remained
  `4.70--38.8x` scaled scratch.

Exact work was 1,536,000 context rows, 768,000 training query rows, 18,432
evaluation query rows and 480 rollout steps. Wall time was 16.398 s, peak RSS
254,640,128 bytes and artifacts before the result were 3,062,352 bytes.
Result SHA-256:
`9a77b02578542d122fda0e9268c528bb1ada51e7940d6ea25d8456a87f71d08e`.

## What this separates

Scale/statistics were a real contributor to ball and Kepler calibration, but
they were not the main explanation for cross-mechanism identity failure in this
learner. The scaled model often improved without using the correct context.
That is prior/calibration improvement, not inference of the held mechanism.

The huge original scratch ratios also did not arise solely from tiny scratch
denominators: the original shared ball MSE (`~1.45`) exceeded zero-prediction
target MSE (`0.00501`) by roughly 290x. After scaling it still exceeded that
target energy by roughly 34--36x. Kepler shared prediction moved below the zero
predictor, but the blind context was at least as good and identity still failed.

This does not isolate “new dynamics” from semantic interface mismatch. The
neutral tensor retains different meanings:

- full-K SHO: one entity, two generic delay-state channels, `dt=1`;
- ball: one entity with `(x,y,vx,vy)`, `dt=.05`, including reflections;
- Kepler: two entities with `(x,y,vx,vy)`, `dt=.02`.

Thus a common shape is not a common state semantics. The earlier same-`(q,v)`
oscillator studies already supplied `q'=v`, RK4 and controlled state banks;
the 4D common-panel studies tested engineered recurring callable structure and
later disjoint-state transport. Neither is a matched decomposition of semantic
interface shift versus a genuinely held mechanism in this heterogeneous run.

## Next research decision

Close further tuning of this exact entity/context model. The smallest honest
next discriminator is a prospective 2x2 factorial, not another broad corpus:

1. familiar dynamics in its native observational interface;
2. the same familiar dynamics under a fixed, invertible, law-independent
   interface transform;
3. held dynamics in the native interface;
4. held dynamics under that same transform.

All four cells must use matched state support, target energy, observations,
updates and model capacity. The transform and inverse must be frozen before any
responses and may not use equations, family IDs or parameters. Context-derived
RMS scaling remains fixed in all cells. A reset model and blind/wrong-context
controls remain mandatory.

This design asks whether the learner fails on an interface transformation even
when the dynamics were represented, versus failing only when the update
function itself is outside training. It is materially different from the
native-interface oscillator folds and the engineered-recurrence common-panel
studies. It should use development-only systems first; no untouched mechanism
confirmation should be opened until the transformation, learner and gates are
frozen.

The main design risk is scientific, not administrative: a transform mixing
state coordinates changes the coordinate representation of the vector field.
The permitted transform class therefore needs to be declared as a group action
with exact conjugate targets, rather than chosen after results. If no such
law-independent transform preserves a scientifically meaningful common
interface, the correct decision is to stop claiming one universal raw-state
interface and instead compare explicit interface encoders feeding a shared
function-level memory. That alternative supplies more semantics and must be
reported as such.

## Evidence status

All systems in this discriminator were already opened development evidence.
No fresh confirmation was used. The result limits one frozen learner and one
context-derived scaling intervention; it does not establish impossibility of
heterogeneous transfer, observation non-identifiability, or general failure of
learned state representations.

