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

## Trial 1: result (29 October)

**Classification: FAILED. C1, C2 and C3 all fail in both seeds.** No experience curve was demonstrated at the pre-registered budget, K = 32.

### Pre-registered readout (K = 32, median over the 13 development systems)

| | Seed 1 | Seed 2 |
|---|---:|---:|
| Aggregate NMSE, N = 0 → 1 → 2 → 4 → 8 → 16 → 32 → 51 | .0058 .0048 .0069 .0100 .0048 .0048 .0035 .0042 | .0057 .0144 .0281 .0153 .1015 .0392 .0086 .0047 |
| Spearman ρ(N, NMSE) (need ≤ −0.8) | −0.76 | −0.07 |
| 95% CI, median log(NMSE₅₁/NMSE₀) (need < 0) | [−0.82, 0.43] | [−0.36, 0.16] |
| Best experience-free control | kernel ridge, .0031 | kernel ridge, .0031 |
| N = 51 beats best control (need ≥ 10/13) | 3/13 | 3/13 |
| Median threshold K at NMSE ≤ 0.1, learner vs control (need ≤ ½) | 16 vs 16 | 16 vs 16 |

The other methods' aggregates at K = 32 were random Fourier features .0046, from-scratch MLP .0050, poly3 .0022 and poly2 .0001. Poly2 sits at the 1% noise floor.

### Stratified readout (K = 32)

- **Polynomial stratum (9 systems).** N = 51 scores .0032 (seed 1) and .0040 (seed 2) against kernel ridge .0025 and the poly2 reference .0001. The learner did **not** rediscover the quadratic structure, even though 42 of its 51 meta-train systems are polynomial. It got worse than N = 0 on some systems, e.g. Halvorsen .005 → .03.
- **Non-polynomial stratum (4 systems).** N = 0 scores .25 and .20, N = 32 scores .13 and .15, and N = 51 scores .12 and .13, against kernel ridge .136. There is a downward trend in both seeds, but with only 4 systems the learner is merely level with the best control.

### Exploratory observations (not pre-registered; hypotheses only)

- **Smallest budget.** At K = 8, N = 51 has aggregate NMSE .167 and .175 against .345 and .299 for N = 0, and beats kernel ridge (.358) on 8 of 13 systems in both seeds. If experience helps here, it helps most where data are scarcest, but this was found after looking.
- **Interference.** Learners are in-distribution-checked on each seed's first 8 meta-train systems (K = 32, published parameters). Error on an early system rises as N grows: QiChen goes from .00004 at N = 1 to .0035 at N = 51, ThomasLabyrinth from .23 at N = 8 to 1.03, MooreSpiegel from .0025 at N = 8 to .012. Training is joint, so this is not forgetting. It is one shared 32-feature basis being spread across more mechanisms.
- **Meta-training loss plateaus.** Final loss is ≤ .009 for N ≤ 4, but .12–.23 for N ≥ 16. The loss is the mean of NMSE clipped at 10, so a few unfit systems (LiuChen, ThomasLabyrinth, KawczynskiStrizhak; NMSE ≈ 1) dominate the gradient. The ~40 near-quadratic systems sit at about 1e-3 and contribute almost nothing. This plausibly explains why a basis that would solve most tasks was not found.
- **Seed instability at intermediate N.** Seed 2 at N = 8 has NMSE .10, so which systems happen to be seen first matters a lot. That is a property of the nested-subset design that a per-N average over more orderings would smooth.

### Deviations from the protocol

- **SprottQ** (development) kept 3 of 4 instances, because one perturbed instance failed all 4 integration draws, so it has 15 episodes instead of 20. **SprottJerk** (meta-train) kept 17 of 32 instances.
- **Code execution.** A command that executed all of `flows.py` was blocked by the session's safety classifier. It was replaced by the AST-whitelisted per-function loader described above, before any data were generated.
- **Learned-λ readout at N = 0** is meaningless (an untrained λ) and is ignored.

The confirmation systems were never loaded. Compute used:

- **Data:** 64 systems, about 25 CPU-minutes on 16 workers.
- **Training:** 14 runs × 20,000 steps, about 30 minutes each, two per GPU on GPUs 0–3 and 5–7.
- **Evaluation:** about 5 minutes.

Outputs total 157 MB. Small records are copied to `docs/research-log-records/trial1/`.

### What this decides

As implemented, "meta-learn one shared feature basis plus a ridge head" does not turn experience into a better prior at K = 32. Generic kernel ridge is at least as good. Two candidate causes are visible in the data, and each is cheap to test separately:

1. **Objective weighting.** A mean-NMSE loss lets a few unlearnable tasks drive the shared features.
2. **One fixed basis for all mechanisms.** Error on early systems grows with N, consistent with the basis being diluted across mechanisms.

### Trial 2 (proposed, not run)

Test cause 1 alone. Keep everything identical except the meta-training loss, which becomes the mean of log(NMSE + 1e-4), weighting every task's relative improvement equally. Same seeds, splits, budgets and criteria.

- **If C1–C3 then pass:** the Trial 1 failure was an objective artefact.
- **If they still fail, with polynomial-stratum error still above kernel ridge:** cause 2 becomes the lead. The next step is a context-conditioned representation (features or a basis selected from the K observations, i.e. in-context inference) rather than a single shared basis.

The K = 8 observation would be pre-registered as a secondary criterion in Trial 2, not credited to Trial 1. The development systems are now opened. Any later positive result needs the 14 untouched confirmation systems before it counts.

## Working rule update (29 October, from the owner)

Do not stop to wait for the owner when a trial ends or the direction changes. Send a push notification, log the reasoning and continue. Stop and wait only when blocked, when something only the owner can provide is needed, or **before touching the reserved confirmation systems**.

## Trial 2: does a scale-free objective fix Trial 1? (pre-registered 29 October, before training)

- **Hypothesis.** Trial 1 failed because the mean-NMSE loss let a few unfit meta-train systems dominate the shared features. With each task's log error weighted equally, meta-training will use the many near-quadratic systems, and the experience curve at K = 32 will appear.
- **Single change from Trial 1.** The meta-training loss is the mean over tasks of log(min(NMSE, 10) + 1e-4), replacing the mean of min(NMSE, 10). The floor 1e-4 is about the 1% noise floor, so tasks already at the noise level stop pulling.
- **Unchanged:** code (`trial1.py --loss lognmse --out outputs/accumulating_learner/trial2`), data bank, split, seeds, N grid, steps, architecture, evaluation episodes and controls. The N = 0 learners are identical by construction.
- **Primary criteria:** C1, C2 and C3 exactly as in Trial 1, in both seeds.
- **Secondary criterion S1** (new; motivated by Trial 1's exploratory K = 8 observation, so it is a test of that observation and not a re-scoring of Trial 1):
  - at K = 8, the N = 51 learner beats the experience-free control with the lowest K = 8 aggregate on ≥ 10/13 development systems;
  - and its aggregate is ≤ 0.75× that control's;
  - in both seeds.

  In Trial 1, S1 would have failed: 8/13 wins against the untrained network, aggregate ratio 0.48.
- **Readouts:** strata, retention and per-system tables as in Trial 1.

**What happens next, decided now:**
- **C1–C3 pass:** a development-level positive result. Stop before any confirmation run and ask the owner.
- **Primary criteria fail but S1 passes:** experience helps only at very small budgets. Trial 3 redesigns around the small-K regime, with the context-conditioned representation as the candidate.
- **Both fail:** objective weighting is ruled out as the main cause. Trial 3 tests a context-conditioned representation (cause 2) with a fresh pre-registration.

**Resources:**
- 14 runs × 20,000 steps on GPUs 0–3 and 5–7, two per GPU;
- outputs under `outputs/accumulating_learner/trial2/` (about 5 MB beyond the shared data bank).

## Trial 2: result (29 October)

**Classification: FAILED.** C1 and C3 fail in both seeds. C2 passes in both. Secondary S1 fails in both. Changing only the objective nonetheless made the N = 51 learner about 10× better than the best experience-free control at K = 16–32.

### Pre-registered readout

| | Seed 1 | Seed 2 |
|---|---:|---:|
| Aggregate NMSE at K = 32, N = 0 → 1 → 2 → 4 → 8 → 16 → 32 → 51 | .0058 .0135 .0182 .0131 .0076 .0015 .0008 .0003 | .0057 .0122 .0185 .0357 .0039 .0011 .0003 .0002 |
| C1: Spearman ρ (need ≤ −0.8) | −0.74 ✗ | −0.76 ✗ |
| C1: 95% CI, median log(NMSE₅₁/NMSE₀) (need < 0) | [−3.36, −0.69] ✓ | [−3.32, −0.47] ✓ |
| C2: N = 51 beats kernel ridge (.0031) at K = 32 | 10/13 ✓ | 10/13 ✓ |
| C3: median threshold K at NMSE ≤ 0.1 (learner vs kernel ridge) | 16 vs 16 ✗ | 16 vs 16 ✗ |
| S1: K = 8 wins against the best control (untrained net) / aggregate ratio (need ≥ 10 and ≤ 0.75) | 11/13, 0.83 ✗ | 11/13, 0.95 ✗ |

At K = 16, N = 51 has aggregate .0022 against kernel ridge .0249 and the poly2 reference .0010.

### Reading

- **Why C1 fails.** Learners trained on 1–4 systems are *worse* on new systems than the untrained network (negative transfer). The curve then falls steeply from N = 8 to N = 51 in both seeds, roughly 20× below N = 0.
- **Why C3 fails.** At K = 8 everything is above 0.1 (learner .29, kernel ridge .36), and at K = 16 everything is below it. The grid cannot resolve a 2× difference at that threshold.
  - Exploratory only: at threshold 0.01 the median is 16 for the learner against 32 for kernel ridge in both seeds. That is not credited, since the pre-registered threshold was 0.1.
  - C3 is left unchanged going forward rather than loosened.
- **Polynomial stratum (9 systems).** N = 51 is at about .0001 at K = 32, the poly2 reference level, beating kernel ridge on 8 of 9 systems. Experience rediscovered the structure that a human would supply as "use quadratic terms". This is real accumulation, but of the majority structure of this catalog.
- **Minority mechanisms are squeezed out.**
  - On development systems at K = 32: Thomas .094–.099 against kernel ridge .033, KawczynskiStrizhak (cubic) .73 against .43. WindmiReduced is level. YuWang .14–.19 and SaltonSea .0012 are better than kernel ridge.
  - In the meta-train retention check, systems that do not share the quadratic structure degrade as N grows: SanUmSrisuchinwong from .0010 at N = 8 to .014 at N = 51, Laser from .0008 to .012, LiuChen from .105 to .51, Chua from .0027 to .0072. Polynomial systems stay at about 1e-4.
  - One shared basis is spending its capacity on the majority mechanism.

### Correction to the Trial 2 decision rule

The rule said that if both the primary criteria and S1 fail, "objective weighting is ruled out as the main cause". That reading is wrong given the data: the objective change gave the largest improvement so far. The rule's *action*, to test a context-conditioned representation next, still follows from the remaining failures (minority mechanisms, interference, small-N negative transfer). Trial 3 keeps the log-NMSE objective.

## Trial 3: context-conditioned features versus a capacity-matched shared basis (pre-registered 29 October, before training)

- **Hypothesis.** A single shared basis is diluted toward the majority mechanism. If the features are instead conditioned on what the K observations reveal (amortized inference from the context), one network can serve minority mechanisms without sacrificing the majority. This should improve non-polynomial development systems and stop the retention drift on minority meta-train systems.
- **Arms** (`experiments/accumulating_learner/trial3.py`, both with the log-NMSE loss):
  - **context (79,968 parameters):**
    - A set encoder (6→128→128→64, mean-pooled) reads the K standardized (state, derivative) pairs.
    - Its summary FiLM-modulates both hidden layers of the 3→128→128→31 feature network, which has zero-initialized modulation.
    - A ridge head (k = 96) is then fitted on the context as before. Encounter cost is one encoder pass plus one ridge solve, with no optimizer steps.
  - **wide (74,784 parameters):** Trial 1's shared network at width 256. This separates conditioning from parameter count.
- **Unchanged:** data bank, split, seeds, N grid, 20,000 steps × 64 tasks, Adam lr 1e-3 cosine, K log-uniform in [8, 128], evaluation episodes and controls. The control rows are copied from Trial 1, since the episodes are identical by construction.
- **Changed:** episode sampling during training is vectorized. The distribution is the same, but the random stream differs from Trials 1–2.
- **Primary criteria (context arm):** C1, C2 and C3 exactly as in Trial 1, in both seeds.
- **Mechanism criteria** (`analyze_trial3.py`, at K = 32, N = 51, both seeds):
  - **D1 (minority mechanisms):** the context arm's non-polynomial-stratum median NMSE ≤ 0.8× the wide arm's, *and* the context arm beats kernel ridge on ≥ 3 of 4 non-polynomial systems.
  - **D2 (majority not sacrificed):** the context arm's polynomial-stratum median ≤ 2× the wide arm's.
  - **D3 (context is used):** the aggregate NMSE when the encoder reads another development system's context, divided by the intact aggregate, is ≥ 1.5. The ridge head is still fitted on the true observations.
  - **D4 (retention):** over each seed's first 8 meta-train systems, the median of NMSE(N = 51)/NMSE(N = 8) for the context arm is ≤ 2 and ≤ the wide arm's. For reference, Trial 2's shared learner gives about 2.5 in seed 1.

**What happens next, decided now:**
- **C1–C3 pass:** stop before confirmation and ask the owner.
- **D1 and D4 pass, but C1–C3 fail:** conditioning works. Trial 4 targets small-N negative transfer and keeps conditioning.
- **D1 fails and the wide arm ≈ context:** pooled-context conditioning does not separate mechanisms. Trial 4 tries an explicit modular library: several bases, chosen per system by leave-one-out on the context, with a generic fallback.
- **D3 fails:** the encoder is ignored, and that is reported as the reason.

**Resources:**
- 28 runs × 20,000 steps, four per GPU on GPUs 0–3 and 5–7;
- outputs under `outputs/accumulating_learner/trial3/`.

## Trial 3: result (29 October)

**Classification: FAILED.** The hypothesis that context-conditioned features would serve minority mechanisms is not supported.

### Context arm, primary criteria

| | Seed 1 | Seed 2 |
|---|---|---|
| C1 | ✓ (ρ = −0.83; CI [−2.48, −0.32]) | ✗ (ρ = −0.41) |
| C2 | ✓ 10/13 | ✓ 11/13 |
| C3 | ✗ 16 vs 16 | ✗ 16 vs 16 |
| S1 | ✗ (9/13, ratio 0.46) | ✓ (10/13, ratio 0.34) |

### Mechanism criteria (context against wide, N = 51, K = 32)

| | Seed 1 | Seed 2 |
|---|---|---|
| D1: non-polynomial stratum ratio (need ≤ 0.8) and wins against kernel ridge (need ≥ 3/4) | 1.49, 2/4 ✗ | 1.59, 2/4 ✗ |
| D2: polynomial stratum ratio (need ≤ 2) | 2.47 ✗ | 2.42 ✗ |
| D3: wrong-context / intact (need ≥ 1.5) | 2.63 ✓ | 1.05 ✗ |
| D4: retention drift N51/N8, context vs wide (need ≤ 2 and ≤ wide) | 8.8 vs 2.1 ✗ | 3.7 vs 1.6 ✗ |

### Aggregate NMSE at N = 51 (median over 13 development systems)

| Arm | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| Context, seeds 1 / 2 | .165 / .123 | .0039 / .0051 | .0008 / .0011 |
| Wide, seeds 1 / 2 | .279 / .277 | .0019 / .0021 | .0002 / .0002 |
| Kernel ridge | .358 | .0249 | .0031 |

### Reading

- **The wide shared arm is the strongest learner so far.** At K = 32 it is 15× better than kernel ridge and at the poly2 level on the polynomial stratum (.0001). It is roughly the same as Trial 2's width-128 learner, so capacity was not the bottleneck.
  - Its C1 fails for the same reason as Trial 2's: negative transfer at N = 1–4. Aggregates at K = 32 for N = 1, 2 and 4 are .0096, .028 and .019 in seed 1, against .009 at N = 0.
  - Wide arm primary criteria: C2 ✓ in both seeds; C1 ✗ (ρ −0.79 and −0.67); C3 ✗.
- **The context arm trades small-K gains for instability.** Its K = 8 aggregate is the best seen so far, but individual episodes blow up: Thomas reaches NMSE 5.6 and 2.9 at K = 8, and WindmiReduced 4.2 and 58. Its meta-training mean NMSE is .64–.68, against .27 for the wide arm.
  - At K = 32 it is worse than the wide arm on both strata, and its earlier meta-train systems degrade more.
  - The pooled encoder does not separate mechanisms. It adds a second, fragile inference channel.
- **Failures that persist across Trials 1–3:**
  1. negative transfer from a few systems;
  2. minority mechanisms (Thomas, KawczynskiStrizhak, WindmiReduced, YuWang) no better than generic kernel ridge;
  3. retention drift on minority meta-train systems as N grows.

### Deviations from the protocol

- GPUs 0 and 1 were in use by another user (`weyl`) at launch, so runs went on GPUs 2, 3, 5, 6 and 7 (five or six per GPU) instead of 0–3 and 5–7. This affects wall time only (about 65–70 minutes per run).
- The wrong-context diagnostic pairs each development system with the next one in alphabetical order.

## Trial 4: a modular library with leave-one-out selection and a generic fallback (pre-registered 29 October, before training)

- **Hypothesis.** The persistent failures come from forcing every mechanism through one basis. Store experience as a library of M = 4 bases that specialise during meta-training. On a new system, choose among them, plus a generic kernel ridge fallback, using only the leave-one-out error on its K observations. This should:
  - remove negative transfer: when no stored basis fits, the selection falls back to an experience-free method;
  - let minority mechanisms get their own basis;
  - stop retention drift: a basis that serves minority systems is not overwritten by the majority.
- **Training** (`experiments/accumulating_learner/trial4.py`):
  - four bases, each Trial 1's 3→128→128→31 network plus a constant, with its own learned λ;
  - 84,096 parameters in total, versus 74,784 for the Trial 3 wide arm;
  - relaxed winner-take-all: per task, loss = min over bases of log(NMSE + 1e-4), plus 0.05 × the mean over bases (so no basis goes untrained);
  - everything else as Trial 3: data, split, seeds, N grid, 20,000 steps × 64 tasks, optimizer, K distribution and the vectorized sampler.
- **Encounter procedure.** For each basis: ridge with λ chosen by closed-form leave-one-out. Kernel ridge: length scale and λ by leave-one-out. The option with the lowest leave-one-out MSE on the K standardized observations is used to predict. No query information is used. Per-system cost: k = 96 coefficients plus one discrete choice among 5 options, with 5 analytic leave-one-out sweeps.
- **Evaluated arms**, on the same development episodes as Trials 1–3, with controls copied from Trial 1:
  - **library+fb:** the main hypothesis;
  - **library:** leave-one-out among the 4 bases only, with no fallback;
  - **wide+fb:** the Trial 3 wide checkpoints with the same kernel ridge fallback, which isolates the fallback's effect;
  - **library-oracle:** the basis with the lowest *query* error. Diagnostic only, not a valid method; it measures how much the leave-one-out selection loses.
- **Primary criteria:** C1–C3 for library+fb, exactly as before, in both seeds. They are also reported for wide+fb.
- **Mechanism criteria** (both seeds):
  - **E1 (no negative transfer):** for library+fb at K = 32, aggregate(N) ≤ 1.1 × kernel ridge's aggregate (.0031) for every N in the grid.
  - **E2 (minority mechanisms):** library+fb at N = 51 beats kernel ridge on ≥ 3 of 4 non-polynomial development systems at K = 32, *and* its non-polynomial median ≤ 0.8× wide+fb's.
  - **E3 (majority kept):** library+fb at N = 51 has a polynomial-stratum median ≤ 2× the Trial 3 wide arm's.
  - **E4 (retention):** on each seed's first 8 meta-train systems at K = 32, the median NMSE(N = 51)/NMSE(N = 8) for library+fb is ≤ 2 and ≤ wide+fb's.
- **Also reported:** which option is selected, per system and N; the oracle-to-leave-one-out gap; and the S1 secondary.

**What happens next, decided now:**
- **C1–C3 pass:** stop before confirmation and ask the owner.
- **E1 passes but E2 fails:** the fallback fixes negative transfer, but stored specialists do not transfer to minority mechanisms. The next trial moves to the sequential (continual) setting with library+fallback, since it already has a never-worse-than-generic property, and measures accumulation over encounter order.
- **E1 fails:** leave-one-out selection is too noisy at these K. Investigate selection, e.g. with a held-out split of the context.

**Resources:**
- 14 runs × 20,000 steps on whichever of GPUs 0–3 and 5–7 are free at launch, checked by owner;
- outputs under `outputs/accumulating_learner/trial4/`.

## Trial 4: result (29 October)

**Classification: FAILED.** C3 fails in both seeds and C1 fails in seed 1. It is nonetheless the strongest learner so far, and the first to pass the small-budget secondary criterion (S1) in both seeds.

### library+fb (main arm)

| | Seed 1 | Seed 2 |
|---|---|---|
| Aggregate NMSE at K = 32, N = 0 → 1 → 2 → 4 → 8 → 16 → 32 → 51 | .0030 .0017 .0017 .0016 .0012 .0019 .00025 .00011 | .0036 .0043 .0034 .0031 .00083 .00027 .00014 .00009 |
| C1 (ρ; CI) | ✗ (−0.76; [−3.75, −0.81]); the N = 16 bump breaks ρ | ✓ (−0.98; [−3.58, −1.03]) |
| C2 | ✓ 10/13 | ✓ 11/13 |
| C3 | ✗ 16 vs 16 | ✗ 16 vs 16 |
| S1 (K = 8) | ✓ 12/13, ratio 0.46 | ✓ 12/13, ratio 0.44 |
| E1: every N ≤ 1.1× kernel ridge at K = 32 | ✓ | ✗ (N = 1: .0043 against a bound of .0034) |
| E2: non-polynomial wins against kernel ridge; ratio to wide+fb | ✗ 2/4; 0.65 | ✓ 3/4; 0.64 |
| E3: polynomial-stratum ratio to wide | ✓ 0.46 | ✓ 0.44 |
| E4: retention drift, library+fb vs wide+fb | ✗ 2.18 vs 2.07 | ✓ 1.39 vs 1.56 |

### Aggregate NMSE at N = 51

| | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| library+fb (seeds 1 / 2) | .162 / .158 | .0004 / .0006 | .00011 / .00009 |
| wide+fb | .27 | .0019–.0021 | .0002 |
| Kernel ridge | .358 | .0249 | .0031 |

At K = 16 the library is about 40–60× better than kernel ridge, and at K = 32 about 30× better.

### Reading

- **The bases specialise, and selection on the context alone routes systems sensibly.**
  - The Sprott family goes to one basis.
  - Halvorsen and YuWang get their own basis. YuWang improves from .24 (kernel ridge) and .12 (wide) to .05–.07.
  - Thomas mostly goes to the kernel ridge fallback, which is correct: no stored basis fits it.
  - KawczynskiStrizhak and WindmiReduced spread across all options. Nothing fits them, and they stay at kernel ridge level or worse.
  - The fallback is rarely chosen elsewhere, so library and library+fb have the same aggregates.
- **Negative transfer is mostly gone.** In 15 of 16 (seed, N) cells the K = 32 aggregate is at or below kernel ridge level. The library at N = 1–4 no longer underperforms the untrained network, as it did in Trials 2–3.
- **Weaknesses that remain:**
  - Selection noise at small K: library+fb is 2.3× worse than the oracle choice at K = 8, falling to 1.3× at K = 32.
  - The C3 grid problem persists. At K = 8, library+fb at N = 16 and 32 actually gets below 0.1 in seed 1 (.080 and .064), but N = 51 does not (.16).

### Direction decision

The pre-registered rules assumed E1 and E2 would agree across seeds; each passes in one seed only. Judgement call: the library with leave-one-out selection and a generic fallback is now the clear best method on every aggregate, and it is roughly never worse than kernel ridge. The largest gap between what has been measured and the owner's goal is now the **setting**, not the method. All trials so far retrain from scratch for each N with joint access to every earlier system's full data. The goal is a learner that accumulates over a stream of encounters and retains. Trial 5 therefore moves to the sequential setting with the Trial 4 method unchanged.

Selection noise is left for later. Tuning selection rules against the opened development set would add development-set fitting.

## Trial 5: does the library accumulate sequentially? (pre-registered 29 October, before training)

- **Hypothesis.** The library (M = 4, relaxed winner-take-all, log NMSE), trained on systems one at a time with only a small replay memory, keeps the experience curve and retention of the jointly trained Trial 4 library. A newly encountered system is also predicted better (relative to kernel ridge) the more systems have already been encountered.
- **Stream.** Each seed's meta-train order is the same permutation as before, so after t systems the learner has seen exactly the systems the joint learner N = t saw. At stage t:
  1. **Encounter evaluation, before any training on system t:** 20 episodes at K = 32 from its published-parameter instance. Score library+fb and kernel ridge NMSE (medians).
  2. **Training:** S steps × 64 tasks. Half the tasks come from system t's full data (all its instances) and half from the replay memory, all of it at stage 1. The memory holds instances 0 and 1 of every earlier system (2 of 32, about 6% of their data).
  3. **Learning-rate schedule:** each stage is cosine from 1e-3 to 1e-4, and the Adam state carries over.
- **Arms** (both seeds):
  - **seq-replay-400:** S = 400 per stage, so 20,400 steps in total, compute-matched to the joint N = 51 learner;
  - **seq-noreplay-400:** the same, but all tasks come from system t (no memory), to show how much forgetting the structure alone prevents;
  - **seq-replay-2000:** S = 2,000 per stage (102,000 steps in total), to show whether compute is the limit.
- **Checkpoints:** after t ∈ {1, 2, 4, 8, 16, 32, 51}. t = 0 is the untrained library, identical to Trial 4's N = 0. Each checkpoint is evaluated exactly like Trial 4's library+fb on the development episodes and on the retention check (each seed's first 8 systems).
- **Primary criteria (seq-replay-400, both seeds):**
  - **F1:** C1 and C2 on the checkpoints, with the same definitions as before.
  - **F2 (close to joint):** at t = 51, aggregate NMSE at K = 32 ≤ 2× Trial 4's joint library+fb N = 51 aggregate.
  - **F3 (retention):** over the first 8 systems, the median NMSE(t = 51)/NMSE(t = 8) is ≤ 2.
  - **F4 (forward transfer on the stream):** the median encounter log(NMSE_library+fb / NMSE_kernel-ridge) over the last 17 encountered systems is lower than over the first 17, *and* below 0 (better than kernel ridge).
- **Reported for all arms:** C3, S1, strata, per-system encounter ratios and selection counts.

**What happens next, decided now:**
- **F1–F4 pass for seq-replay-400:** a positive sequential development result. Stop before the confirmation run and ask the owner.
- **F3 fails but F1 passes:** forgetting is the bottleneck. Next, compare growing the library (new basis per surprising system, older bases frozen) against replay.
- **F4 fails:** the stream does not make encounters easier, even though checkpoints help development systems. Investigate the difference between encounter and development evaluation.
- **seq-replay-2000 passes where 400 fails:** compute-limited. Note the cost and continue with 2,000.

**Resources:**
- 6 runs (about 15 minutes each for S = 400 and about 70 minutes for S = 2,000);
- outputs under `outputs/accumulating_learner/trial5/`.

## Trial 5: result (29 October)

**Classification: PASSED (development only).** All pre-registered primary criteria (F1–F4) pass in both seeds for the compute-matched sequential arm, seq-replay-400. Per the pre-registered rule and the owner's working rule, **work stops here before any confirmation run.**

### seq-replay-400: library trained one system at a time, 400 steps per stage, replay of 2 of 32 instances per earlier system

| | Seed 1 | Seed 2 |
|---|---|---|
| Aggregate NMSE at K = 32 over checkpoints t = 0 → 1 → 2 → 4 → 8 → 16 → 32 → 51 | .0030 .0042 .0031 .0031 .0017 .0010 .00031 .00017 | .0036 .0034 .0034 .0031 .0031 .00098 .00023 .00009 |
| F1: C1 (ρ; CI) | ✓ (−0.85; [−3.44, −0.81]) | ✓ (−0.97; [−3.45, −0.61]) |
| F1: C2 | ✓ 11/13 | ✓ 12/13 |
| F2: t = 51 vs joint Trial 4 N = 51 (need ≤ 2) | ✓ 1.47 | ✓ 1.02 |
| F3: retention drift, t = 51 over t = 8 (need ≤ 2) | ✓ 1.56 | ✓ 1.41 |
| F4: median encounter log(lib/kernel ridge), first 17 → last 17 | ✓ −0.22 → −1.05 | ✓ −0.28 → −3.01 |
| C3 (reported, not required) | ✗ 16 vs 16 | ✗ 16 vs 16 |
| S1 (reported) | ✓ 10/13, 0.39 | ✓ 12/13, 0.35 |

**Aggregate NMSE at t = 51, K = 8 / 16 / 32:**

| | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| seq-replay-400, seed 1 | .139 | .00080 | .00017 |
| seq-replay-400, seed 2 | .127 | .00046 | .00009 |
| Kernel ridge | .358 | .0249 | .0031 |

### Other arms

- **seq-noreplay-400:** C2 passes. F2 fails badly: the t = 51 learner is 7× and 17× worse than joint training. F4 fails in seed 1. Without the replay memory the library does not keep its earlier gains.
- **seq-replay-2000:** F1–F4 pass in both seeds, and C3 also passes in seed 2 (threshold K 8 vs 16). It is not clearly better than the 400-step arm. Five times the compute is not needed.

### Exploratory within-system check of F4 (not pre-registered)

The first-17 versus last-17 comparison is confounded by which systems happen to arrive late. Each seed streams the same 51 systems in a different order, so every system is encountered at two positions. I compared each system's encounter log NMSE at its later position with its earlier one.

- **seq-replay-400:** the later encounter is better for 36 of 51 systems (Wilcoxon p = 2e-4). Spearman correlation between the position difference and the log-ratio difference is −0.54 (p = 4e-5).
- **seq-replay-2000:** 30 of 51 (p = 0.13).
- **seq-noreplay-400:** 23 of 51 (p = 0.72).

With replay, the same mechanism is learned better the more systems came before it. This is the owner's target quantity, measured on systems that were genuinely unseen at encounter time.

### Caveats (these limit what the result means)

1. **Development reuse.** The 13 development systems have now steered five trials of method choices, so these numbers are optimistic. The 14 confirmation systems are untouched and are the real test.
2. **Catalog composition.** Most gains are on the polynomial majority.
   - Non-polynomial systems at K = 32 (seed 1 / seed 2 against kernel ridge): YuWang .037/.012 vs .24 and SaltonSea .0007/.0010 vs .0017 improve strongly. WindmiReduced .25/.27 vs .29 improves slightly. Thomas .076/.037 vs .033 is at or below kernel ridge level. KawczynskiStrizhak (cubic) .46/.36 vs .43 is mixed.
   - This is accumulation over a catalog, not evidence for arbitrary mechanisms.
3. **The fallback** makes "about as good as kernel ridge at worst" partly true by construction. The gains *beyond* kernel ridge come from the stored bases.
4. **Retention is not perfect.** LiuChen drifts from .22 at t = 8 to .42 at t = 51, and SanUmSrisuchinwong and Laser fluctuate between 1e-4 and 8e-3. ThomasLabyrinth is never fit by any basis; the fallback handles it throughout.
5. **The replay memory** (6% of earlier systems' data, never any development or confirmation data) is necessary. The no-replay arm fails.
6. **The observation contract** is still simplified: derivatives are observed directly with 1% noise, d = 3, and states are on-attractor.

### Resources used

- 6 streams: about 20 minutes each at 400 steps per stage and about 80 minutes each at 2,000;
- GPUs 0–3, 5 and 6;
- outputs about 15 MB.

## Proposed confirmation run (awaiting owner approval; nothing below has been run)

The confirmation systems' names have never been printed, their vector fields never compiled, and their data never generated. The proposal is fixed here before any of that happens.

- **Data.** Generate the 14 confirmation systems with `make_data.py`'s procedure:
  - published parameters plus 3 perturbed instances, same seed and integration settings;
  - this requires an explicit `--confirmation` flag that lifts `System`'s refusal, shown in the diff before use;
  - a system whose published instance fails integration is dropped and reported, not replaced.
- **Models, frozen.**
  - Primary: the seq-replay-400 checkpoints, both seeds, t ∈ {0, 1, 2, 4, 8, 16, 32, 51}, exactly as trained.
  - Secondary: Trial 4's joint library+fb and Trial 3's wide arm.
  - No retraining and no hyperparameter changes.
- **Episodes.**
  - Per system and K ∈ {8, 16, 32, 64, 128}: 4 instances × 5 episodes, same episode function, generator seeds 6000 + 97i + K.
  - Controls as before: random Fourier features, kernel ridge, poly2/3, from-scratch MLP.
- **Criteria** (seq-replay-400, both seeds, on all confirmation systems that generated):
  - **CF1:** C1 (Spearman ρ ≤ −0.8 over the t grid at K = 32, and the bootstrap 95% CI of the median log(NMSE_t51/NMSE_t0) below 0).
  - **CF2:** t = 51 beats the best experience-free control at K = 32 on ≥ 70% of systems.
  - **CF3:** at K = 8, t = 51 beats the best K = 8 control on ≥ 70% of systems, and its aggregate is ≤ 0.75× that control's.
  - C3 and the polynomial/non-polynomial strata (classified symbolically at confirmation time) are reported but not required.
- **Pass = CF1–CF3 in both seeds.** One shot: no reruns, no changes after looking, and the result is recorded whatever it is.
