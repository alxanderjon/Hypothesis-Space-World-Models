# Sequence-aware function-space GEM v1

Status: prospective before execution.

Observation-row full GEM found and projected individual conflicts but did not
retain Duffing behavior on disjoint query states. This test retains function
values rather than only observation-row losses.

For each of the two frozen sequence-aware checkpoints and each of 24 stored
source-training systems, freeze the checkpoint's 64-state query prediction
after fitting its ridge head on that system's context32. These 3,072 scalar
pseudo targets are retained memory, not true simulator query labels. During the
same eight-law pendulum sequence, compute one gradient constraint per stored
system from the current model's context-fitted query prediction to its frozen
pseudo target, then apply the same fixed full-GEM NNLS projection. The new
system supplies only its context32; no target-system query answer, family ID,
equation or parameter enters its update.

The pass rule is unchanged from full GEM. Passing supports function-space
retention as the missing ingredient for this benchmark; it does not show that
the fixed 64-state bank is sufficient generally or that memory cost stays fixed
as systems accumulate. Failure limits this stored-bank mechanism, not all
generative replay or functional consolidation.

Prospective work: zero fresh fits; two sequences; 16 target updates/ridge
solves; 48 pseudo-target ridge solves and 3,072 pseudo-output rows; 384 memory
gradients/ridge solves and 24,576 memory-prediction rows; 16 NNLS projections;
240 evaluation ridge solves and 15,360 evaluation rows; 10,752 one-step
simulations / 43,008 RK4 RHS evaluations; 240 score blocks. Retained pseudo
targets are exactly 24,576 bytes in float64, plus the already counted source
contexts. Limits: 120 seconds, 1 GiB RSS, 64 MiB artifacts; no automatic retry.
