# Sequence-aware distributional functional replay v1

Status: prospective before execution.

This is not another posterior used to identify a new system. It learns a replay
distribution over old callable functions, while new-system prediction and
adaptation remain the existing context32 ridge plus one shared-weight step.

Construct one normalized 192-dimensional vector per source-training system by
concatenating its true context32 and query64 increments. Fit the full empirical
Gaussian with no rank truncation (mean plus all nonzero SVD directions), using
no family ID, equation or parameter. For each model seed, draw 24 synthetic
functions from this frozen Gaussian before adaptation. Their context portion
fits the current ridge head and their query portion supplies one GEM retention
constraint. Retained generator storage is the mean, 71x192 empirical factor,
and fixed RNG receipt; generated functions are counted separately.

The matched broken control applies a fixed response-independent cyclic
permutation to query64 functions across the 72 source systems before fitting
the Gaussian. It preserves the empirical context and query marginals and the
same SVD/sample/gradient/solve budget, while breaking their system-level joint
covariance. Neither arm sees held-system outputs during generator fitting.

Both seeds pass only if the aligned arm: retains >=20/24 source laws within
10%, has source aggregate after/before <=1.1, has pendulum stage8/stage0 <=.8,
is no more than 1.1x the saved unconstrained pendulum error, and beats the
broken generator on source aggregate with >=16/24 law wins. This is opened
development evidence. Passing supports joint distributional replay in this
supplied oscillator interface; it does not establish a calibrated function
posterior, arbitrary mechanisms, or fixed memory with continual growth.

Prospective work: two generator SVDs; four target sequences; 32 target updates
and target ridge solves; 96 generated functions / 18,432 generated scalars;
768 replay gradients/ridge solves and 49,152 replay-prediction rows; 32 NNLS
projections; 480 evaluation ridge solves and 30,720 evaluation rows; 10,752
one-step simulations / 43,008 RK4 RHS evaluations; 480 score blocks. Limits:
120 seconds, 1 GiB RSS, 64 MiB artifacts; no automatic retry.
