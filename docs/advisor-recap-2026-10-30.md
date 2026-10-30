# Accumulating learner — recap since the handover

*As of 30 October 2026. All figures are from `docs/research-log.md`, where all 25 trials were pre-registered.*

**Summary.** Experience gives large gains on systems that resemble the learner's history, and this was confirmed on untouched systems. A new approach (Trial 24) is the first to beat kernel ridge on unfamiliar kinds of systems. Its advantage does not yet grow with history, which the success criterion requires. A scaled-up version (Trial 25) is running.

## Goal and success criterion

The goal is a learner whose accumulated history makes new dynamical systems learnable from fewer observations. It is given no equations, no term library and no ground-truth targets. The current success criterion, set by the owner:

- with at most 64 noisy observations, it beats kernel ridge and the best no-history control on **unfamiliar kinds** of systems;
- in both seeds;
- and the advantage grows with the number of systems in its history.

Reserve test sets are used only after a pass on the development systems and then on a fresh split.

## Setup

- **Systems.** 3-D chaotic flows from the dysts catalog, split by hash before any training: 51 meta-train, 13 development and 14 confirmation systems (opened once). Two reserve sets are untouched: A (11 three-variable systems excluded for code structure) and B (31 higher-dimensional systems). Real biochemical models come from BioModels, selected by a fixed rule and parsed as data (17 three-variable and 56 any-dimension oscillators).
- **Observations.** Each new system gives K noisy observations. Trials 1–8 observed derivatives with 1% noise. From Trial 9 on, the learner sees only noisy trajectory snippets (0.5% state noise), with no ground-truth targets anywhere.
- **Scoring.** Query NMSE, taking the median over episodes and then over systems, in two seeds. The controls are kernel ridge, random Fourier features and a network trained from scratch. So far 25 trials have been evaluated on the development systems.

## What was confirmed

- **Experience transfers to related systems** (Trial 5, then the confirmation run). A library of four learned feature bases, trained one system at a time with a 6% replay memory, passed every development criterion in both seeds. On 13 untouched systems its error fell as history grew (ρ = −0.81 and −0.90), and it beat the best no-history control on 12 of 13. Error was about 65× lower at 32 observations (0.00023 against 0.0159) and about 40× lower at 16.
- **The confirmation run did not pass as pre-registered.** Its 8-observation criterion failed in both seeds (8 of 13 wins, aggregate ratio 0.91–0.95). Also, 11 of the 13 systems are polynomial, like the training catalog.
- **Without supplied answers** (Trial 9, development only). Trained and scored only on noisy trajectories, error still fell with history in both seeds. The gain over kernel ridge shrank to about 2× at 32 observations and 4× at 16.

## What failed, and why

- **Unfamiliar mechanisms.** No learned-basis variant beat kernel ridge on systems unlike its history. That covers context-conditioned features (Trial 3), test-time learning (Trials 10–11), fine-tuning (Trial 12), a broader synthetic history (Trial 15), and 11 or 37 related real models (Trials 16 and 18: 0–1 of 6 and 5–8 of 19 wins). Trial 23's history of 145 real transient models was unlearnable for every method, kernel ridge included (median NMSE above 1).
- **Eight observations** (Trials 6–7). A gain from sparse fits came from a biased selection estimate and vanished under exact leave-one-out.
- **Choosing observations** (Trials 13–14, 17, 19, 21–22). Spreading observations out helps every method 3–5×, and no experience-based rule beat it. Trial 17's reported 25–39% gain came from a confounded comparison and was withdrawn.
- **Rehearsal by error** (Trial 20). Rehearsing the systems with the most error made results 1.09–1.26× worse.
- **Why.** Every variant fitted a new system with a linear combination of a fixed learned basis. That amounts to assuming "this system resembles my history". On unfamiliar systems it becomes a generic regressor that gets no more from the observations than kernel ridge does.

## Audit of Trials 1–23

| Approach family | Trials | Outcome |
| --- | --- | --- |
| Learned feature basis + per-system linear fit | 1–12, 15, 16, 18, 20, 23 (17 trials) | Large gains on related systems; never beat kernel ridge on unfamiliar ones |
| Choosing observations, built on that learner | 13, 14, 17, 19, 21, 22 (6 trials) | Generic coverage was best; no experience-based gain |
| Amortized inference (learning the inference procedure) | 24, 25 | First win on unfamiliar kinds (below) |

The 23 trials were variations of one idea plus an add-on built on it. Untried approach families were ranked, and amortized inference was tried first, ahead of trajectory-consistent estimation.

## Current status and plan

- **Trial 24** (development level). A Transformer with 2.2M parameters reads the observations and predicts directly. It was trained only on simulated polynomial, trigonometric, piecewise and tanh systems, plus 45 catalog systems. It was tested on 26 real systems of kinds it never saw: rational/saturating and exponential. In both seeds it beat kernel ridge, the best no-history control, with error ratios of 0.69 and 0.72 at 16 observations and 0.77 and 0.79 at 32. It won on 17 of 26 systems.
- **Not met in Trial 24.**
  - The advantage did not grow with more, or more varied, training systems.
  - It is 10× worse than kernel ridge on familiar polynomial systems (0.052 against 0.005).
  - It is overtaken at 64 observations.
- **Trial 25** (running). The same test with 6.5M parameters and 3.3× more training. Trial 24's training error was the same at every history size, which pointed to a capacity limit. The early sign is that training error now rises with history size: about 0.04 at 512 systems, against 0.064–0.067 at 8,192 and 16,384. Ten of 14 learners have finished, and evaluation follows.
- **Next.** If both halves of the criterion pass in both seeds, replicate on a fresh held-out-kind split and fresh seeds, then request the reserve test. Otherwise move to trajectory-consistent estimation, combined with the Transformer.

## Still tentative

- Trial 24's win is development-level and has not been replicated. Several of its 26 test systems appeared in earlier trials, and the "kind" labels come from an automatic reading of the equations.
- Trial 25 has no evaluation yet. The trend in training error is only suggestive.
- The confirmed gains are on a polynomial-majority catalog. Generality beyond that rests on Trial 24 alone.
- Trial 9's trajectory-only result is development-level.
