# Sequential compact-head hyperposterior v1

Status: prospective before execution.

This diagnostic tests a task-generative population update rather than another
shared-weight retention penalty.  The two saved sequence-aware feature networks
remain frozen.  A system is represented only by its 16x2 linear response head
(32 fitted quantities).  Seventy-two source-training systems initialize a
Gaussian population distribution over those heads.  Its observation variance
is the median source even16-to-odd16 normalized residual variance; its covariance
floor is `1e-3*trace(C)/32`.  Both are fixed before target outcomes.

For each encountered pendulum system, permitted context32 observations define
an exact Gaussian posterior over its head.  The **sequential** arm updates the
population mean and covariance with that posterior's first two moments using a
Welford moment update.  The **frozen** arm retains the source population prior.
The **reset** arm uses the source mean scale as an isotropic zero-mean prior and
discards every encounter.  All arms receive and solve the same encounter data.
An independently recomputed batch moment after every sequential update must
match the online state within `1e-10`; this is an implementation invariant, not
an evaluation gate.

This differs from the completed static learned-basis population prior, affine
chronological sufficient-statistics study, compact latent posterior, direct
TNP and K=2 GP mixture: it updates a population distribution sequentially from
observation-derived task posteriors while never changing shared weights or
using family IDs, coefficients, equations or query labels.  It remains an
empirical moment-matched Gaussian approximation, not calibrated uncertainty.

Each seed passes only if sequential stage-8 held-pendulum error is <=.8x both
frozen and reset with >=6/8 law wins against each; source-development aggregate
after/before is <=1.1 with >=20/24 laws within 10%; all batch invariants and
finiteness checks pass.  These are opened controlled-oscillator development
gates, not fresh confirmation.

Prospective work: two source constructions, 288 source head solves, six target
sequences, 48 encounter posterior solves, 720 evaluation posterior solves,
46,080 evaluation rows, 10,752 repeated transitions / 43,008 RK4 RHS calls and
720 score blocks.  Persistent sequential state is one 32-vector mean, one
32x32 covariance and one count per seed; stored encountered posterior moments
are retained only for the batch-invariant audit.  Limits: 120 seconds, 1 GiB
RSS, 64 MiB artifacts; no retry or hyperparameter sweep.
