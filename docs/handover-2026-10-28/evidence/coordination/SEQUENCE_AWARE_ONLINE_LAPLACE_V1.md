# Sequence-aware online diagonal-Laplace diagnostic v1

Status: prospective before execution.

This study asks whether the failure of output-space retention comes from using
the wrong persistent object.  Instead of retaining old predictions or replay
examples, it retains a 624-entry diagonal importance vector over the shared
feature weights.  For each source-training law, a 16-row context head is fit
and the normalized response loss on the disjoint 16 rows is differentiated;
the mean squared gradient is the source diagonal empirical Fisher.  Because
the simulator is deterministic, this is a scale-dependent Gauss--Newton style
importance diagnostic, not a calibrated Bayesian posterior.

At each of eight pendulum encounters the learner still receives only the same
permitted context32 responses.  It fits the same 16x2 ridge head, computes one
shared-weight gradient, and makes one update.  The **diagonal** arm divides the
gradient by accumulated per-parameter source/encounter importance.  The
**isotropic** arm uses the mean of the same importance, so it has identical
model capacity, observations, updates and persistent scalar scale but no
parameter-specific posterior structure.  Each arm's learning-rate multiplier
is frozen from three fixed source-training calibration laws (indices 0, 24,
48) so its initial update norm matches the established 0.01 ordinary step.
Numerical damping is fixed to 1% of mean source importance before any target
outcome is evaluated.  Importance is updated after every encounter.

The two seeds pass only if the diagonal arm has pendulum stage8/stage0 <= .8,
is <=1.1x the saved unconstrained sequence learner, retains >=20/24 source laws
within 10% with aggregate after/before <=1.1, and beats the isotropic arm on
source aggregate with >=16/24 law wins while having no more than 1.1x its
pendulum error.  These are opened controlled-oscillator development gates.
Passing would support diagonal sequential weight importance under this supplied
model and interface; failure would close this approximation, not sequential
Bayesian learning generally.

Prospective work: 144 source-importance and six calibration gradient/ridge
evaluations; four target sequences, 32 target gradients/ridge solves/updates;
480 evaluation ridge solves and 30,720 prediction rows; 10,752 repeated
one-step simulations / 43,008 RK4 RHS evaluations; 480 score blocks.  Stored
new persistent state is 1,248 float64 importance values (624 per seed), plus
two isotropic scalars.  Limits: 120 seconds, 1 GiB RSS, 64 MiB artifacts; no
automatic retry.

The operation is O(P) memory and update work after gradients are available.
At one million shared parameters a float64 diagonal is about 8 MB, but each
encounter still requires an O(P) backward pass and this experiment does not
establish million-parameter performance.
