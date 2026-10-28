# Sequence-aware soft function-distribution retention v1

Status: prospective before execution.

Fit one full empirical Gaussian to the 72 source-training normalized
context32+query64 functions, without family labels, equations or parameters.
Use all 64 fixed query states as the inducing bank. The retained comparison
space is the QR-orthonormalized query portion of the first 16 training-only
principal directions; 16 is fixed to the existing per-system feature/head
dimension and is not selected on retention outcomes.

For each seed draw 24 synthetic functions once and reuse them in both arms. At
each pendulum encounter the current representation fits a ridge head to every
synthetic context and predicts its query values. The **distributional** arm
adds `KL(old Gaussian || current Gaussian)` over the 24 projected query
functions. The **mean-only** arm retains only the old mean under its fixed old
covariance. It uses the same samples, contexts, solves, model and update count.

The simulator is deterministic. The Gaussian covariance is empirical
across-system variation, not observation noise or calibrated uncertainty. A
fixed numerical floor `tau=1e-6*trace(C_old)/16` makes covariance inverses and
log determinants defined; it is frozen before target evaluation. For each
seed, source-training data alone determine: (1) a multiplier making the
mean-only and full regularizer gradient norms equal at the initial checkpoint,
and (2) a shared coefficient making that regularizer norm equal to one fixed
source observation-loss gradient norm. No opened pendulum/source-development
outcome enters calibration.

Both seeds pass only if distributional retention has pendulum stage8/stage0
<=.8, is <=1.1x the saved unconstrained pendulum error, retains >=20/24 source
laws within 10% with aggregate after/before <=1.1, and beats mean-only on source
aggregate with >=16/24 law wins while being no more than 1.1x its pendulum
error. Passing would support covariance-aware soft functional retention in this
supplied benchmark, not calibrated uncertainty or arbitrary-mechanism transfer.

Prospective work: one generator SVD; 48 generated functions / 9,216 generated
scalars; six calibration gradients, 50 calibration ridge solves and 3,072
calibration prediction rows; four sequences, 32 target updates/ridge solves;
32 regularizer evaluations, 768 replay ridge solves and 49,152 replay rows;
480 evaluation ridge solves and 30,720 rows; 10,752 repeated one-step
simulations / 43,008 RK4 RHS evaluations; 480 score blocks. Limits: 120 seconds,
1 GiB RSS, 64 MiB artifacts; no automatic retry.
