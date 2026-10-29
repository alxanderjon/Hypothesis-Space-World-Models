# Research log

Running record of the resumed research, started 28 October 2026. Every trial gets a written hypothesis, setup and success/failure criteria **before** it runs. Its result, including failures, is recorded **after** it runs and is used to choose the next trial. Read [the handover](research-handover-2026-10-28.md) for the prior history.

## Owner context (stated in the 28 October session)

### The real goal

Build a learner that learns from its experience and builds on that knowledge, so it learns new dynamical systems faster and more reliably. This was the original objective. The deterministic teacher was a means to it, geared toward teaching a network each new system one at a time. The owner judges that the previous agent drifted from the goal into making the teacher recover individual systems.

### The three-stage teacher

The working belief was that gradients were holding the work back by getting stuck in local minima. The teacher was therefore built in three stages, each handing off only once it passed:

1. **Stage 1: DIRECT-style global search.** Voronoi cells partitioned parameter space, and Sobol space-filling points were scored by reconstruction loss. Gradient-free.
2. **Stage 2: Powell-style line search.** Also gradient-free.
3. **Stage 3: gradient-based polishing.** Used only at this point, on the belief that it would no longer get stuck in local minima. The archived code shows N3 is Newton-type: full Hessian, safeguarded box-constrained quadratic steps.

### Hard-coded versus removed dynamics

With the dynamics hard-coded into the model, recovering system parameters went well. Once the dynamics information was removed, the work struggled with the geometry of the search. The archive agrees:

- An affine `Kz+b` model was validated on all 320 SHO cases.
- The generic 22-parameter SiLU network was start-sensitive: 0 of 8 handoffs at the companion_29 sites.
- One case succeeded only after a solver redesign (variable projection).

### Yubei's dimensionality concern

Yubei, the owner's co-advisor, pointed out that the geometry of a relatively low-dimensional parameter space is very different from that of a large one. This approach may therefore not work for, say, a million-parameter system.

Working interpretation: keep three sizes separate:

- **d:** the state dimension.
- **P:** the size of the shared model kept across systems.
- **k:** the number of quantities fitted for each new system.

The concern bites hardest when per-system adaptation is a large search, e.g. k = P ≈ 1M in the PDEBench fine-tuning.

## Rules from the owner's kickoff message

1. **Freedom of method.** Permission to try whatever is judged likely to reach the goal, continuing similar approaches or going in a completely different direction.
2. **Before each trial**, write down the hypothesis, the setup, and what result counts as success or failure.
3. **After each trial**, record the result, including failures, and use it to decide the next trial. Keep this running log in `docs/research-log.md`.
4. **Measure the goal itself where possible:** how fast or reliably a new system is learned, as a function of how many systems the learner has already seen.
5. **Keep reserved/unopened data untouched** until there is a result worth confirming on it. This covers the PDEBench reserved trajectories and the diffusion–sorption mechanism, plus any confirmation set declared below.
6. **GPUs.** A free GPU (not used by someone else's job) is free game, and as many free GPUs as help may be used. Check `nvidia-smi` before launching, and never touch GPUs or processes belonging to other users.
7. **Disk.** The disk is nearly full: about 91% per the owner, 96% used / 168 GB free when checked. Check free space before writing large outputs and clean up unneeded intermediates.
8. **Git.** Work on a new branch off `codex/research-handover-2026-10-28` (this branch: `claude/accumulating-learner-2026-10-28`), commit as you go, and push to GitHub so progress is visible. Do not commit unrelated working-tree changes (`src/hypothesis_space_world_models.egg-info/PKG-INFO`, `experiments/sho/run_stage1_per_trajectory_omega_curriculum.py`).
9. **Push notifications** when a trial finishes, when direction changes, or when blocked.

## Done so far (28 October)

- **Handover revised** in commit `bcbe83f` on `codex/research-handover-2026-10-28`, pushed. The goal is now framed as original with the teacher as a means. Added: the teacher history, the hard-coded versus removed finding, attribution of the dimensionality concern to Yubei, corrected paired-replay counts (800 unique windows / 1,600 forward passes, versus the doubled hard-coded values in the record), the five pass/fail criteria, and the real target split (adapt on trajectories 0–9, evaluate on 10–24).
- **Environment surveyed** (host `ironforge`):
  - 8x V100 16 GB. GPU 4 is in use by user `weyl` (PID 961126) and is off-limits; GPUs 0–3 and 5–7 were idle.
  - 48 CPUs, 188 GB RAM.
  - Conda env `/home/alex/miniconda3/envs/NNlearning` has torch 2.5.1 (CUDA), numpy 2.2.6, scipy 1.15.2 and matplotlib. It lacks h5py, pandas and sklearn.
  - The PDEBench arrays and checkpoints are **not** on this machine, so the prior Transformer work can't be resumed from artifacts here.
- **External system catalog checked.** PyPI is reachable, and the `dysts` 0.96 wheel was downloaded to the session scratchpad only, not installed. It is Gilpin's catalog of ~130 published chaotic ODE systems with literature parameters. Its package import pulls in `gluonts` through `dysts.utils.data_utils`, so the plan is to use only its right-hand-side definitions and parameter JSON, not to install its full dependency tree into a shared env. No experiment has been run yet.

## Plan

### Why this direction

The archive shows three repeated failure modes:

1. Benchmarks built around the component the method was meant to find.
2. Per-system adaptation that is either a large search (the teacher; full fine-tuning) or a fit inside a hand-supplied basis.
3. No measurement of the actual goal curve: performance on a new system as a function of how many systems were seen before.

The first trial therefore builds the measurement before any new method claim. It uses an externally curated, independently motivated set of systems, keeps per-system adaptation small (low k, per Yubei), and compares against equal-cost controls with no prior experience.

### Trial 1 (proposed, not yet run): does an experience curve exist?

- **Hypothesis.** A shared representation meta-trained across N previously seen dynamical systems makes a new, held-out *mechanism* learnable from fewer observations. Held-out error at a fixed observation budget K falls as N grows, and beats equal-k fixed bases (random features, RBF) that embody no experience.
- **Systems.** Three-dimensional continuous-time systems from the dysts catalog, split by **system** (mechanism) into meta-train, development and an untouched **confirmation** set, fixed by a seeded hash of system names before any training.
- **Observation contract.** For a new system, the learner sees K passive on-attractor states with their vector-field values (derivative observations, optionally with noise), standardized using statistics of those K observations only. Query states come from a separate trajectory segment.
  - Simplifications to state openly: derivatives are observed directly rather than estimated, and d = 3 is fixed.
  - Supplied: the state, the observation scaling and the fact that it is an autonomous first-order ODE. No equations, families or parameters.
- **Learner.**
  - A shared MLP φ (P parameters) maps standardized state to F features.
  - A new system gets an F x 3 linear head fitted by ridge regression from its K observations, so k = 3F is fixed and analytic, with no landscape search.
  - φ is meta-trained by differentiating query loss through the ridge solve, across N meta-train systems. Each system is re-sampled with randomized parameters and initial conditions, which gives many tasks per system.
- **Measurement.** Train separate learners for N in {0 (no meta-training), 1, 2, 4, 8, 16, 32, all}. For each learner and each development system, measure query NMSE at K in {8, 16, 32, 64, 128}. Report:
  - the K needed to reach NMSE thresholds, as a function of N;
  - per-system results, not only aggregates;
  - adaptation compute;
  - retention: whether earlier meta-train systems stay good as N grows.
- **Controls (equal k = 3F).** Random-feature ridge, RBF ridge (centres from the context), a disclosed polynomial basis as an upper structural comparator, and per-system training of a small MLP from scratch at matched optimizer budget.
- **Success.** In both seeds:
  - development NMSE at K = 32 decreases monotonically within noise from N = 0 to N = all;
  - the N = all learner beats the best experience-free control at equal k on at least 70% of development systems;
  - the threshold K at NMSE 0.1 is at least 2x lower than the best control's.
- **Failure.** No monotone N trend, or the learned features fail to beat random/RBF features at equal k. Either result redirects the next trial: toward in-context inference, a different persistent object, or richer observations.
- **Resource bounds.** At most 6 free GPUs. Outputs under `outputs/accumulating_learner/trial1/`, capped at about 2 GB, with checkpoints only for final learners. Confirmation systems are never loaded.

## Trial 1: review and revised protocol (29 October, written before any training)

### Problems found in the proposal, and adjustments

1. **The catalog is mostly polynomial.** After the eligibility filter (below), 42 of 51 meta-train systems and 9 of 13 development systems have polynomial vector fields, nearly all quadratic. A disclosed degree-2 basis (10 terms, k = 30) fits these almost exactly from ~10 observations, so meta-learning could "succeed" by rediscovering monomials. That is legitimate accumulation, but narrower than general mechanism learning. Adjustment: polynomial bases (degree 2 and 3) are a **known-prior reference**, not an experience-free control, and every result is reported separately for the polynomial (9) and non-polynomial (4) development strata. A general claim requires the non-polynomial stratum to improve too.
2. **Compute was not held fixed across N.** Adjustment: every trained learner gets the same 20,000 meta-training steps of 64 tasks, so N changes only diversity of experience, not optimizer work.
3. **Regularization choice could favour one side.** Adjustment: at evaluation, every ridge-type method (learner, random features, kernel ridge, polynomials) selects λ (and length scale where applicable) by closed-form leave-one-out on the K context points only. The learner's own meta-learned λ is reported as a secondary readout.
4. **"Equal k" for the RBF control.** It is implemented as Gaussian kernel ridge with centres at the context states (k = 3K), which equals the learner's k = 96 exactly at the primary budget K = 32 and exceeds it at K ≥ 64 (conservative against the learner).
5. **"Monotone within noise" and "best control" were undefined.** Defined below.
6. **Retention.** With joint (not sequential) meta-training this is only an in-distribution check, not a continual-learning test. Sequential accumulation is deferred to a later trial.
7. **Executing downloaded code.** dysts is not installed or imported. `flows.py`, the parameter JSON and the Apache-2.0 license are vendored in `third_party/dysts_0_96/`, along with the wheel's SHA-256. Each system's `_rhs` is compiled alone after an AST whitelist: arithmetic, numeric constants and `np` math functions only, no builtins.

### Systems and split (fixed before any training)

- **Eligible:** 3-dimensional, autonomous, non-delay, no unbounded coordinate, and a static `_rhs` that passes the whitelist. Excluded: 31 by dimension, 13 nonautonomous, 8 with custom `rhs`, 2 unbounded, 3 failing the whitelist (BelousovZhabotinsky, IsothermalChemical, VallisElNino).
- **Split:** by `sha256("accumulating-learner-trial1-split-v1:<name>")`, giving 51 meta-train, 13 development and 14 **confirmation** (names not printed, data never generated, `System` refuses to load them).
- **Development:** BurkeShaw, Halvorsen, KawczynskiStrizhak, RikitakeDynamo, SaltonSea*, ShimizuMorioka, SprottI, SprottJ, SprottP, SprottQ, Thomas*, WindmiReduced*, YuWang* (* = non-polynomial).

### Data

- **Instances per system:** published parameters (instance 0), plus perturbed instances with each parameter multiplied by exp(U(−0.1, 0.1)). Meta-train systems have 32 instances; development systems have 4.
- **Integration:** LSODA (rtol 1e-8) from the published initial condition. 20 characteristic periods are discarded, then 100 periods are sampled at 4,000 times. Instances that diverge, collapse, time out (60 s) or reach a fixed point are redrawn, at most 4 draws.
- **Context and queries:** context states come from the first half of the trajectory, queries (128) from the second half.

### Observation contract

- **Observed:** K on-attractor states with derivative values. Derivative noise is Gaussian at 1% of the context derivative std. States and derivatives are standardized with context statistics only.
- **Supplied:** d = 3 and the autonomous first-order form.
- **Not supplied:** equations, family or parameters.

### Learner

- **φ:** 3→128→128→31 SiLU, plus a constant feature, so F = 32, P = 21,024 (including a meta-learned log λ).
- **Per-system head:** ridge, 32×3, k = 96.
- **Meta-training:** Adam (lr 1e-3, cosine schedule) on query NMSE differentiated through the ridge solve. K is log-uniform in [8, 128] per batch.
- **Nested subsets:** for seed s, the meta-train list is permuted with `default_rng(s)` and learner N uses its first N systems.
- **Grid:** N ∈ {0, 1, 2, 4, 8, 16, 32, 51}, seeds {1, 2}, giving 14 trained learners plus 2 untrained (N = 0) learners.

### Evaluation (development systems only)

- **Episodes:** for each system and K ∈ {8, 16, 32, 64, 128}, 20 fixed episodes (4 instances × 5), identical across methods.
- **Episode NMSE:** query MSE / query variance, averaged over the 3 output dims.
- **System score:** the median over its 20 episodes.
- **Aggregate:** the median over the 13 development systems.

**Experience-free controls:**

- random Fourier features (F = 32, length scale by LOO);
- Gaussian kernel ridge;
- the untrained N = 0 network.

**References, not part of the criteria:**

- polynomial ridge, degree 2 and 3 (known prior);
- the φ architecture trained from scratch per episode (AdamW, 500 steps; k = P + 96).

**Best control:** the experience-free control with the lowest aggregate at K = 32.

### Success (pre-registered; must hold in both seeds)

- **C1 (experience curve):** Spearman ρ(N, aggregate NMSE at K = 32) ≤ −0.8 over the 8 N values. In addition, a paired bootstrap over development systems (10,000 resamples) puts the 95% CI of the median log(NMSE_N=51 / NMSE_N=0) entirely below 0.
- **C2:** at K = 32, the N = 51 learner beats the best control on ≥ 10 of 13 development systems.
- **C3:** per system, threshold K = the smallest K in the grid with NMSE ≤ 0.1 (∞ if never reached). The median threshold for N = 51 is ≤ half the best control's.

**Failure:** any criterion failing in either seed. The stratified readout sets the scope of any positive claim. If the non-polynomial stratum does not improve with N, the result is "experience rediscovers polynomial structure", not general mechanism learning.

**Resource bounds:**

- at most 7 free GPUs (never GPU 4);
- 14 × 20k steps;
- outputs under `outputs/accumulating_learner/trial1/`, at most 2 GB (data bank about 200 MB).
