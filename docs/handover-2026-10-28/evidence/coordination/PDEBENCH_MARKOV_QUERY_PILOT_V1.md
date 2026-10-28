# PDEBench Markov-query pilot v1

Status: completed; `NOT_SUPPORTED` because only one of two seeds passed every
declared gate.

## Question

The direct task Transformer could predict from `(x,t)` while almost ignoring
the observed trajectory.  This comparison asks whether the failure is caused
by indirect retrieval of the current field from attention, rather than absence
of usable transition information.  It is an observation-interface test, not a
new PDE basis or a claim of general equation discovery.

## Data and supplied assumptions

Use the existing official range subset and its existing source/development
split.  No download, simulator call or newly protected response is permitted.
Advection and Burgers remain source mechanisms; reaction-diffusion remains
opened whole-mechanism development evidence.  The learner is explicitly given
that the data are fully observed scalar fields on a common periodic 1D grid,
the spatial coordinate, time, fixed four-cell patches and chronological order.
It is not given PDE identity, coefficients, equations, derivatives, boundary
operators or solver metadata.

For a start index `s`, context consists of field states `s:s+4`, represented by
64 tokens `(normalized patch values[4], x, t)`.  Sixteen query tokens contain
the current patch at `s+3`, its `(x,t)`, and request the normalized one-step
increment `u[s+4]-u[s+3]`.  Field and increment normalization are computed from
source-training trajectories only.  This is a Markov-state interface with
history; it changes the old four-future-state contract and cannot be compared
as an identical task.

## Model and persistent learning

Use the same standard three-layer Transformer encoder, three-layer Transformer
decoder, width 128, four heads and FF width 256.  Only the query projection
changes from two inputs to six.  Offline source training updates all shared
weights.  At a new mechanism, replay/no-replay/scratch update all shared weights
for 100 steps; there is no compact per-system head.  Replay retains a fixed
reservoir of real source windows.  Therefore this tests shared-weight
adaptation with stored examples, not zero-shot recognition or compact Bayesian
identification.

## Controls and gates

Use the same two seeds and four arms (`frozen`, `replay`, `no_replay`,
`scratch`) as v1.  Evaluation starts are fixed at 0, 32, 64 and 96.

* history shuffle: replace the four-state context with the next trajectory in
  the same coefficient file while leaving the current query field intact;
* current-field shuffle: replace only the query current field with the next
  trajectory while leaving history and target intact.

These are deliberately incoherent negative controls and do not represent a
deployable input.  Useful development evidence requires both seeds to satisfy
all prior accuracy/retention gates, current-field shuffle/intact aggregate
error at least 1.10, and history-shuffle/intact at least 1.05.  The history gate
is necessary to distinguish rule/context use from a state-only transition
prior.  No threshold is changed after outcomes.

## Prospective work ceiling

At most eight fits, 600 source updates, 600 adaptation updates, 4,800 training
windows, 5,280 evaluation blocks, 600 seconds, 1 GiB RSS and 128 MiB artifacts.
No architecture, learning-rate, normalization, loss-weight or update-count
sweep follows automatically.  A negative result closes this explicit-current-
state contract at the declared budget, not attention models or PDE learning in
general.

## Outcome

The interface correction restored genuine observation dependence.  Current-
field shuffle/intact error was `1.635x/2.242x`; history shuffle/intact was
`1.082x/1.947x`.  Replay beat frozen target error strongly in both seeds
(`0.0345x/0.357x`).  Seed `202610281` passed all gates, including `0.251x`
scratch error and `115/120` retained source blocks.  Seed `202610282` failed:
replay was `1.127x` scratch and `3.361x` no-replay, with only `38/120` retained
source blocks.  The two adaptation arms used different sampled batches, so the
result establishes the value of the Markov query interface but cannot cleanly
attribute the seed-two failure to replay itself.  That causal comparison needs
an exactly paired batch schedule; no outcome-tuned replay weight is justified.

Exact work: eight fits, 600 source updates, 600 adaptation updates, 4,800
training windows, 5,280 evaluation blocks, 37.94 seconds and 298,434,560 bytes
peak RSS.  Evidence:
`data/pdebench_range_subset_v1/task_transformer_markov_v1/result.json`.
