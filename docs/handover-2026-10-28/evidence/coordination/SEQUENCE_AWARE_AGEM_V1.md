# Sequence-aware A-GEM discriminator v1

Status: prospective before execution.

The completed sequence-aware plasticity study improved every never-adapted
pendulum law in both seeds, but its retention failure was concentrated exactly
in the eight Duffing represented-development laws. This follow-up tests whether
that failure is caused by conflicting target-update gradients.

For each of the same two frozen retention-aware checkpoints, use the identical
eight-law pendulum sequence and one observation-only shared-weight update per
law. Before each update, compute a reference gradient from a fixed source
memory: the first eight source-training laws from each of linear, Duffing and
Van der Pol, using even16 context rows to fit the ridge head and odd16 rows for
loss. If the pendulum gradient conflicts with the average reference gradient,
apply the standard A-GEM half-space projection. No family ID is an inference
input; the stratified memory selection is an explicitly supplied offline corpus
design assumption.

The saved unprojected sequence is the matched baseline and is not rerun. The
follow-up passes only if, in both seeds: pendulum stage-eight error is no more
than 0.8 of stage zero and no more than 1.1 of the saved unprojected error;
source aggregate after/before is no more than 1.1; at least 20/24 source laws
remain within 10%; all predictions are finite. Passing supports gradient
interference as a causal contributor inside this benchmark. It does not prove
that replay is unnecessary, that A-GEM scales without approximation, or that
the learned representation identifies arbitrary dynamics.

Prospective work: zero fresh fits; two target sequences; 16 target updates; 16
target ridge solves; 384 memory ridge solves; 240 evaluation ridge solves;
15,360 prediction rows; 10,752 repeated one-step simulations / 43,008 RK4 RHS
evaluations (including the source-query responses required to reconstruct the
frozen normalization); 240 score blocks. Limits: 120 seconds, 1 GiB RSS, 64 MiB artifacts;
no automatic retry.
