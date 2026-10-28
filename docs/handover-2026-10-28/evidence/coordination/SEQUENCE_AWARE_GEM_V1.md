# Sequence-aware full-GEM discriminator v1

Status: prospective before execution.

The A-GEM follow-up made no projections: every target gradient aligned with the
average 24-system memory gradient, even though all eight Duffing development
laws worsened. This test asks whether the average concealed conflicts with
individual stored systems.

Reuse the same two frozen sequence-aware checkpoints, target sequence, source
memory and gates. For each target encounter, compute 24 separate observation
gradients, one per stored source system. Project the target gradient onto their
joint nonnegative half-space with the standard GEM dual, solved as a fixed
24-variable nonnegative least-squares problem. No family label participates in
the projection or target update. The saved unprojected result is the baseline.

The pass rule is unchanged: both seeds require pendulum stage8/stage0 <=.8,
GEM/saved-unprojected pendulum error <=1.1, source after/before <=1.1, at least
20/24 source laws within 10%, and finite predictions. Passing supports masked
per-system gradient interference in this supplied benchmark. Failure closes
this fixed local-gradient constraint, not all replay, consolidation, or
continual learning.

Prospective work: zero fresh fits; two sequences; 16 target updates and target
ridge solves; 384 memory gradients/ridge solves; 16 fixed 24-variable NNLS
projections; 240 evaluation ridge solves; 15,360 prediction rows; 10,752
one-step simulations / 43,008 RK4 RHS evaluations; 240 score blocks. Limits:
120 seconds, 1 GiB RSS, 64 MiB artifacts; no automatic retry.
