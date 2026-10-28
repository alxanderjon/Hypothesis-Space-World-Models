# Common-4D interface x mechanism factorial v1

Status: prospective before science.

## Distinct question

Can the frozen entity/context learner transport familiar versus held dynamics
under a matched, unknown but simple observation-interface change? This crosses
two axes on the same 4D state support:

| dynamics | native interface | transformed interface |
|---|---|---|
| represented linear + cubic-Legendre residual | cell 1 | cell 2 |
| held linear + tanh-network residual | cell 3 | cell 4 |

Earlier two-body canonicalization supplied physical translation/rotation
symmetries, `meta_gp_identity_warp_v1` learned one shared nonlinear 8D warp,
and the common-panel recurrence studies used native coordinates only. None
formed this matched factorial. This study does not reopen those branches.

## Fixed interface transform

Native state is `x in R^4`. The transformed observation is
`z = [x2, -x0, x3, -x1]`; increments receive the identical signed permutation.
The transform is orthogonal, invertible, norm preserving and fixed before data
generation. The generator uses the inverse only to construct conjugate target
transitions. The learner is never given the transform, inverse, interface ID,
mechanism ID, generator parameters, equations, or future query answers.

This is a deliberately small layout test, not universal invariance to arbitrary
coordinates. Thirty-two controlled context transitions may still be
insufficient to infer a coordinate convention; failure is not a general
non-identifiability or impossibility result.

## Data and supplied assumptions

- Fully observed 4D Markov states, known `dt=.05`, Euler one-step increments.
- Fixed context32 and disjoint query64 banks, shared across all laws.
- 48 training laws from linear + elementwise cubic-Legendre residual dynamics.
- 24 tanh-residual laws for the matched scratch learner only.
- 16 represented-family and 16 held-family development laws.
- Every development law is evaluated on the same physical bank in native and
  transformed coordinates. Dimension, state distances, target norm, row count,
  model capacity, updates and scaling rule are therefore paired.
- All systems are synthetic opened development. No confirmation is opened.

Context-derived per-channel RMS scaling from the previous discriminator is
fixed in every cell. It uses context states/increments only and is not selected
per outcome.

## Learners and controls

- Shared learner: unchanged 22,724-parameter entity/context network, trained on
  the 48 represented-family laws in native coordinates only.
- Scratch: identical network and compute trained on 24 held-family laws in
  native coordinates only.
- Two fixed seeds, 500 updates per fit: four fits and 2,000 updates total.
- Correct, mean-context blind and cyclic full-wrong-context predictions for
  every shared cell; correct and cyclic wrong for held-family scratch.
- No transform augmentation, equivariance layer, architecture change, label
  selection, or post-hoc checkpoint selection.

Primary interpretations are predeclared rather than compressed into one claim:

1. Familiar/native failure makes the test inconclusive for this learner.
2. Familiar/native success plus familiar/transformed failure implicates
   coordinate/interface transport.
3. Familiar cells passing while held/native fails implicates dynamics-family
   extrapolation beyond interface alignment.
4. Held shared beating held scratch with correct-context identity in both
   interfaces supports reusable cross-family knowledge under this supplied
   4D/Euler/controlled-bank contract only.

Existing forward gates remain: correct/blind <= .8, at least 12/16 correct
wins over full-wrong context, and shared/scratch <= .8 where scratch exists.
The interface error ratio is reported, not tuned into a replacement gate.

## Prospective exact work

- 9,984 generated one-step rows/RHS evaluations; 6,144 signed-transform array rows (states and corresponding increments).
- Four fits, 2,000 optimizer updates.
- 512,000 context rows and 256,000 query rows used during training.
- 32,768 evaluation rows, 512 law-mode score blocks.
- 300 seconds wall, 1 GiB RSS, 64 MiB artifacts.
- No retries or automatic recovery; any incomplete attempt is preserved.

