# Accumulating learner — recap since the handover

*30 October 2026. Figures are from `docs/research-log.md`, which records 25 pre-registered trials.*

**Summary.** Experience gives large gains on systems that resemble the learner's history, and this was confirmed on untouched systems. A new approach (Trial 24) is the first to beat kernel ridge on unfamiliar kinds of systems. But its advantage does not yet grow with history, as the success criterion requires. A scaled-up version (Trial 25) is running.

## Goal and criterion

The goal is a learner whose accumulated history makes new dynamical systems learnable from fewer observations, with no equations, term library or ground-truth targets supplied. Success means: with at most 64 noisy observations, beat kernel ridge and the best no-history control on unfamiliar kinds of systems, in both seeds, with an advantage that grows with history.

## Setup

- **Systems.** 3-D chaotic flows from the dysts catalog, split by hash before any training into 51 meta-train, 13 development and 14 confirmation systems. Reserve sets A (11 systems) and B (31) are untouched. Real biochemical models come from BioModels, selected by a fixed rule.
- **Observations.** Each system gives K noisy observations. Trials 1–8 observed derivatives; from Trial 9 on, only noisy trajectory snippets, with no ground-truth targets.
- **Scoring.** Query NMSE (median over episodes, then over systems), two seeds. The controls are kernel ridge, random Fourier features and a network trained from scratch.

## Confirmed

- **Experience transfers to related systems** (Trial 5, then the confirmation run). A library of four learned feature bases, trained one system at a time, beat the best no-history control on 12 of 13 untouched systems. Its error fell as history grew (ρ = −0.81 and −0.90). Error was about 65× lower at 32 observations (0.00023 against 0.0159) and about 40× lower at 16.
- **Limits.** The pre-registered 8-observation criterion failed (8 of 13 wins), so the confirmation did not fully pass. Also, 11 of the 13 systems are polynomial, like the training catalog. Learning only from noisy trajectories (Trial 9), the gain shrinks to about 2–4×.

## Failed, and why

- **Unfamiliar mechanisms.** No learned-basis variant beat kernel ridge on systems unlike its history:
  - context-conditioned features (Trial 3);
  - test-time learning (Trials 10–11);
  - fine-tuning (Trial 12);
  - a broader synthetic history (Trial 15);
  - 11 or 37 related real models (Trials 16 and 18: at most 1 of 6, and 5–8 of 19 wins).

  Trial 23's 145 real transient models were unlearnable for every method.
- **Eight observations** (Trials 6–7). An apparent gain came from a biased selection estimate.
- **Choosing observations** (Trials 13–14, 17, 19, 21–22). Spreading observations out helps every method 3–5×, but no experience-based rule beat it. Trial 17's reported gain came from a confounded comparison and was withdrawn.
- **Why.** Every variant fitted a new system with a linear combination of a fixed learned basis, in effect assuming "this resembles my history". On unfamiliar systems that makes it a generic regressor, no better than kernel ridge.

## Audit of Trials 1–23

| Approach family | Trials | Outcome |
| --- | --- | --- |
| Learned basis + per-system linear fit | 1–12, 15, 16, 18, 20, 23 | Large gains on related systems; never beat kernel ridge on unfamiliar ones |
| Choosing observations, built on that learner | 13, 14, 17, 19, 21, 22 | Generic coverage was best |
| Amortized inference | 24, 25 | First win on unfamiliar kinds |

The audit concluded that the 23 trials varied one idea. Amortized inference was chosen next as the most different family not yet tried.

## Status and plan

- **Trial 24.** A Transformer with 2.2M parameters predicts directly from the observations. It was trained only on simulated polynomial, trigonometric, piecewise and tanh systems. On 26 real systems of held-out kinds (rational/saturating and exponential), it beat kernel ridge in both seeds: error ratios of 0.69–0.72 at 16 observations and 0.77–0.79 at 32, with 17 of 26 wins.
- **Not met.** The advantage did not grow with more, or more varied, training systems. It is 10× worse than kernel ridge on familiar polynomial systems, and it is overtaken at 64 observations.
- **Trial 25** (running). The same test with 6.5M parameters and 3.3× more training, since Trial 24 looked capacity-limited. Training error now rises with history size (about 0.04 at 512 systems, against about 0.065 at 8,192–16,384). Evaluation follows.
- **Next.** If both halves pass in both seeds, replicate on a fresh held-out-kind split and fresh seeds, then request the reserve test. Otherwise move to trajectory-consistent estimation, combined with the Transformer.

## Tentative

- Trial 24 is development-level and unreplicated. Some of its test systems appeared in earlier trials, and its kind labels are assigned automatically.
- Trial 25's trend is only suggestive until it is evaluated.
- The confirmed gains rest on a polynomial-majority catalog.
