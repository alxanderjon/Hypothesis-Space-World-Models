# Common-4D interface x mechanism factorial milestone v1

Date: 2026-10-28

## Evidence and recovery history

The original attempt completed four fits and 2,000 updates, then stopped before
prediction persistence because a blind-control tensor was converted to NumPy
outside `no_grad`. The attempt is preserved as an administrative stop, not a
scientific result. Its exact spent work includes 9,984 generated rows, 6,144
transform rows and 3,072 unsaved evaluation rows.

Additive recovery loaded the four checkpoints and repeated no fit, update or
gradient call. It repeated 9,984 deterministic RHS rows and 6,144 transforms,
then completed 32,768 prediction rows and 512 score blocks. Cumulative counts
are 19,968 RHS rows, 12,288 transforms, four fits, 2,000 updates and 35,840
attempted evaluation rows. Recovery used 0.162 s, 194,592,768-byte peak RSS and
1,015,138 artifact bytes before the result.

Recovered result SHA-256:
`82d4202f8b26808ad72ed59be8f7a5ab9151e6c84303c528c0cda9f205723388`.

## Result

The fixed unknown transform `z=[x2,-x0,x3,-x1]` preserved dimension, Euclidean
geometry and target energy. The same physical development laws and banks were
paired across native and transformed cells.

- Represented cubic-residual dynamics: transformed/native error was `1.0126`
  and `1.0177` across the two seeds.
- Held tanh-residual dynamics: transformed/native error was `1.0052` and
  `1.0745`.
- Held shared/scratch error was `1.103--1.220` native and `1.164--1.188`
  transformed: retained cubic-family weights did not help the tanh family.
- The blind context beat the correct context in all held cells
  (`correct/blind=1.049--1.155`).
- Represented cells had `15--16/16` correct-versus-wrong wins, but the correct
  context improved over blind by only `0--2.8%`, below the unchanged 20% gate.

The familiar/native sanity conjunction therefore did not pass, so this is not
a positive system-identification result. Nevertheless, the paired interface
ratios show that this finite signed permutation was not the dominant source of
failure. The mechanism-family shift was substantially more consequential under
the frozen learner and budget.

## Scope

This result does not establish invariance to arbitrary unknown coordinates.
It uses controlled banks, fully observed 4D states, known `dt=.05`, Euler
increments, a cubic-Legendre training family and a tanh-network held family.
It cannot be transferred directly to the heterogeneous SHO/ball/Kepler tensor,
whose masks and channel meanings differ. All systems are opened synthetic
development; no confirmation was used.

## Decision

Close further coordinate-transform and scaling variants of this 22,724-
parameter learner. The data now argue against treating layout/scale as the main
repair for its held-mechanism failure.

The next decision is genuinely conceptual:

1. **Typed interface encoders into a shared function memory.** This supplies
   state semantics (for example, generic discrete map versus position/velocity
   entity dynamics) and tests transfer at the callable-function level. It is a
   more tractable engineering path, but no longer claims a single semantics-
   free raw-state interface.
2. **Learned new-mechanism adaptation and retention.** Keep a generic observed-
   transition interface, allow bounded shared-weight or callable-memory updates
   after encountering a new mechanism, and measure later-system benefit plus
   prior retention. This better matches accumulated knowledge, but it is
   supervised per-mechanism adaptation—not immediate recognition—and requires
   a chronology with genuinely new mechanisms.

The repository already contains many compact-head, replay, callable-memory and
meta-adaptation studies, so option 2 needs a new causal mechanism rather than a
rerun. Without such a mechanism, option 1 is the smallest defensible next
phase. No further numerical study is launched from this milestone.

