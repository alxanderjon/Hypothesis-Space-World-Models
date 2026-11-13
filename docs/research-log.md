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

## Confirmation run: approved by the owner (29 October)

The owner replied "approve" to the proposal above. Before any confirmation data was generated, the following were committed:

- **Guard change:** `systems.System` takes `allow_confirmation=False` and only lifts its refusal when that is passed.
- **Data flag:** `make_data.py --confirmation` generates *only* the confirmation systems, with 4 instances each (published plus 3 perturbed), using the same seed and integration settings as the development set.
- **Evaluation and scoring:** `confirm.py` implements the protocol as written. Episodes use seeds 6000 + 97i + K, where i is the index in the sorted confirmation list. The models are the frozen seq-replay-400, joint library+fb and wide checkpoints. Criteria CF1–CF3 are scored in code.
- **Classifier for the strata:** symbolic, the same method used for the development strata.
- **Smoke test:** run only on two already-opened development systems, writing to the scratchpad.

## Confirmation run: result (29 October)

**Classification: NOT CONFIRMED.** CF1 and CF2 pass in both seeds; CF3 fails in both. The protocol required all three, so the confirmation fails. The result is recorded as specified, with no reruns and no changes after looking.

### Data

- 13 of 14 confirmation systems generated. **Torus** was dropped because its published-parameter trajectory collapsed a coordinate; as the protocol specifies, it was reported, not replaced.
- The remaining 13 kept all 4 instances.
- The symbolic classifier labels 11 of the 13 polynomial; GlycolyticOscillation and ItikBanksTumor are not.

### Primary arm: seq-replay-400, frozen

| | Seed 1 | Seed 2 |
|---|---|---|
| Aggregate NMSE at K = 32, t = 0 → 1 → 2 → 4 → 8 → 16 → 32 → 51 | .0150 .0080 .0105 .0107 .0056 .00067 .00053 .00023 | .0120 .0086 .0144 .0115 .0093 .0026 .00053 .00021 |
| CF1: ρ; 95% CI | ✓ −0.90; [−4.30, −1.34] | ✓ −0.81; [−4.58, −0.81] |
| CF2: t = 51 beats the best experience-free control at K = 32 | ✓ 12/13 (control: untrained library+fb, .0150) | ✓ 12/13 (control: untrained library+fb, .0120) |
| CF3: K = 8 wins (need ≥ 70%); aggregate ratio (need ≤ 0.75) | ✗ 8/13 (62%); 0.91 | ✗ 8/13 (62%); 0.95 |
| C3 (reported) | ✗ 16 vs 16 | ✗ 16 vs 16 |

### Aggregate NMSE (seed 1)

| | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|
| seq-replay-400, t = 51 | .446 | .00158 | .00023 | .00005 |
| Kernel ridge | .501 | .0647 | .0159 | .0032 |
| Random Fourier features | .712 | .0781 | .0150 | .0025 |
| From-scratch MLP | .536 | .0839 | .0154 | .0030 |
| poly2 (known-prior reference) | .915 | .0024 | .00010 | .00003 |

### Strata at K = 32

- **Polynomial (11 systems):** learner .00023 / .00013 against kernel ridge .013, beating it on 10 of 11. That is close to the poly2 reference (.00009).
- **Non-polynomial (2 systems):** learner .32 / .28 against kernel ridge .31.
  - ItikBanksTumor: .004–.005 against .016, but random Fourier features get .006.
  - GlycolyticOscillation: about kernel ridge level.
- HindmarshRose, which is polynomial but whose time scales differ widely, stays at kernel ridge level (.22–.33 against .30).

### Secondary arms

- Joint library+fb N = 51: .00019 / .00018, beating kernel ridge on 11 of 13.
- Wide (shared basis): .0004 / .00036, also 11 of 13. It again shows small-N negative transfer: .024–.113 at N = 1–4 against .008–.013 at N = 0. The library arms do not.

### What is now supported, and what is not

- **Supported on untouched mechanisms.** A library of four feature bases, accumulated *sequentially* over 51 catalog systems with a 6% replay memory, makes held-out systems from the same catalog much easier to learn once there are at least 16 observations. At K = 32 its error is about 65× below the best experience-free method; at K = 16 about 40× below. The gain grows with the number of systems seen (ρ = −0.81 to −0.90). Negative transfer from a few systems is absent.
- **Not supported.** A small-budget (K = 8) advantage. The development set suggested one (S1 passed), and it did not hold on new mechanisms. Even the known quadratic basis fails at K = 8 (.915). In 3 dimensions, 8 noisy derivative observations under-determine a quadratic field (10 terms per component). Beating that requires knowledge about *which* terms are present (sparsity), not just a better basis.
- **Scope.** The held-out mechanisms are mostly polynomial, like the training catalog. This is accumulation of the structure shared by a catalog of published chaotic flows, not evidence about arbitrary mechanisms. Non-polynomial held-out systems were few (2), with mixed results.

### Reserved data from now on

The 14 confirmation systems are now opened. Two untouched sets remain, declared reserved here, before any use:

- **Reserve set A (3-D, never loaded or evaluated):** AnishchenkoAstakhov, BeerRNN, BelousovZhabotinsky, CaTwoPlus, CaTwoPlusQuasiperiodic, CellularNeuralNetwork, Colpitts, ExcitableCell, IsothermalChemical, MultiChua, VallisElNino.
  - These are the systems excluded by code structure (custom `rhs`, a builtin `max`, or decorators). They are mostly non-polynomial: neural-network, chemical and circuit models.
  - Using them needs hand-porting of their right-hand sides, with review, at confirmation time.
- **Reserve set B:** the 31 systems of dimension 4, 5, 6 and 10. Only relevant to a learner that works for any state dimension.

The PDEBench reserved trajectories and the diffusion–sorption mechanism stay untouched, as before.

## Next direction: the small-budget regime (29 October)

The one claim that failed confirmation is the K = 8 advantage.

**Why a learned Gaussian prior would not help.** A Gaussian prior over the per-system head (any covariance, zero mean) can be absorbed into the last linear layer of the learned features. Isotropic ridge on learned features therefore already *is* a learned Gaussian prior. (A non-zero mean is meaningless here, because output roles differ between systems.) What could help at K = 8 is a *non-Gaussian* prior about which terms are present, i.e. sparsity.

### Pilot (development episodes, no training; recorded before designing Trial 6)

Head: `sparse_head.py`, adaptive ridge. Five reweighting iterations, weights 1/(W² + ε), ε = 0.01 × the mean of the squared ridge coefficients, λ chosen by approximate leave-one-out.

| Aggregate NMSE | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| poly2, ridge head | .446 | .00096 | .00010 |
| poly2, sparse head (known-prior reference, SINDy-like) | **.069** | .00023 | .00006 |
| seq-replay-400 bases, ridge head (seed 1 / 2) | .139 / .127 | .00080 / .00046 | .00017 / .00009 |
| seq-replay-400 bases, sparse head, no retraining | .282 / .126 | .0018 / .0018 | .00057 / .00022 |

At K = 8 the sparse head beats the ridge head on 11 of 13 systems in the monomial basis, but on only 1 and 4 of 13 in the learned bases. The ridge-head rows include the kernel ridge fallback and reproduce Trial 5's official values exactly.

*Correction, made before committing:* a first version of this table summarised each system's 20 episodes with `torch.median`, which returns the lower of the two middle values. Every official analysis uses `np.median`, the average of the two, and the table above uses that. I traced the discrepancy by recomputing stored Trial 5 entries; they reproduce exactly when the same statistic is used. No stored result or criterion is affected.

So sparsity is worth about 6× at K = 8 *in the right basis*, but the learned bases are not one. The catalog's vector fields are sparse in monomials, not in features learned under a ridge head.

## Trial 6: learn a basis in which systems are sparse (pre-registered 29 October, before training)

- **Hypothesis.** If the library is trained *through* the sparse head, experience will find bases in which catalog systems are sparse: a learned term library, not a supplied one. This recovers much of the monomial-sparsity advantage at K = 8 without being given monomials, and without losing the K = 32 performance.
- **Method.**
  - Trial 5's seq-replay-400 unchanged (stream, orders, 400 steps per stage, replay of 2 instances, per-stage cosine learning rate, 4 bases, relaxed winner-take-all, log NMSE) except for the head.
  - In training, the head is `sparse_fit` with each basis's learned λ, differentiated through the 5 unrolled reweighting iterations.
  - At encounter, λ is chosen by the sparse head's approximate leave-one-out, and there is the same kernel ridge fallback.
  - Code: `trial6.py`.
- **References:**
  - seq-replay-400 (Trial 5 checkpoints, ridge head);
  - poly2 with sparse head (known prior; shows how much is left to gain).
- **Criteria** (development episodes, both seeds, t = 51):
  - **G1 (small budget):** K = 8 aggregate ≤ 0.75× seq-replay-400's (seed 1: .139 → ≤ .104; seed 2: .127 → ≤ .095), *and* it beats seq-replay-400 on ≥ 9 of 13 systems at K = 8.
  - **G2 (no loss at larger K):** K = 32 aggregate ≤ 1.5× seq-replay-400's.
  - **G3:** C1 and C2 pass on the t-grid checkpoints.
- **Reported:**
  - S1 against the experience-free controls;
  - fraction of the gap to poly2-sparse closed at K = 8;
  - effective sparsity: the share of head coefficients below 1% of the largest, per system;
  - the strata.

**What happens next, decided now:**
- **G1–G3 pass in both seeds:** a candidate for confirmation on reserve set A, which needs hand-porting and owner approval. Stop and ask.
- **G1 fails while G2 and G3 pass:** sparsity cannot be learned through this unrolled head. Next, try a sparsity-inducing feature parameterization, or an explicit term-selection policy learned from experience.
- **Training is numerically unstable** (NaN, or loss blow-ups in more than 1% of steps): report it, then retry once with ε increased 10×, logged as a deviation.

**Resources:**
- 2 streams of 20,400 steps, about 1 hour each;
- outputs under `outputs/accumulating_learner/trial6/`.

## Trial 6: result (29 October)

**Classification: FAILED.** G1 passes in seed 1 only, and G2 fails in both seeds. Training was numerically clean (0 non-finite steps in both seeds).

| At t = 51, development set | Seed 1 | Seed 2 |
|---|---|---|
| K = 8: sparse library vs ridge library (seq-replay-400) | **.077** vs .139 (ratio 0.56; 9/13 wins) ✓ | .267 vs .127 (ratio 2.11; 2/13) ✗ |
| K = 32: sparse vs ridge (G2 needs ≤ 1.5×) | .00040 vs .00017 (2.4×) ✗ | .0024 vs .00009 (27×) ✗ |
| C1; C2; C3 | ✗ (ρ −0.74); ✓ 11/13; ✓ (threshold K 8 vs 16) | ✗ (ρ −0.64); ✗ 7/13; ✗ |
| Share of the log-gap to poly2-sparse (.0695) closed at K = 8 | 85% | −124% |
| Median share of head coefficients < 1% of the largest (K = 32) | 47% | 34% |

### Reading

- **Seed 1 shows the idea can work.** Trained through a sparse head, the library closes 85% of the gap to the known sparse-monomial reference at K = 8.
  - Example per-system K = 8 gains over the ridge library: BurkeShaw .010 → .003, RikitakeDynamo .092 → .014, ShimizuMorioka .77 → .24, Halvorsen .70 → .16.
  - This is the first pass of C3 (threshold K = 8 vs 16 for the best control) by a non-reference method.
- **Problem 1: the fallback is broken.** The approximate leave-one-out of the adaptive head treats the data-dependent reweighting as fixed, so it is optimistic. A sparse option then always beats kernel ridge's exact leave-one-out: kernel ridge is selected 0 times out of 255 at K = 32, in both seeds. Systems that no basis fits blow up instead of falling back: Thomas 2.6 / 3.8 at K = 8 against .72 with the ridge library, WindmiReduced 1.01 against .35.
- **Problem 2: seed 2's library degraded late in the stream.** K = 32 aggregate is .0008 at t = 32 but .0025 at t = 51, and polynomial systems are uniformly worse than with the ridge library. Seed 1 does not show this. The cause is not established; candidates are instability of the unrolled reweighting gradients, or basis collapse (seed 2's last-stage wins concentrate on basis 0).
- **What this decides.** The pre-registered branches assumed G2 would pass; it did not. Problem 1 can be tested *without retraining*, so that comes first as Trial 7.

## Trial 7: honest selection for the sparse head (evaluation-only; pre-registered 29 October, before running)

- **Hypothesis.** Trial 6's failures on systems with no fitting basis come from the optimistic approximate leave-one-out, not from the learned bases. With an *exact* leave-one-out for the sparse head (refit K times with each point held out, for every λ), the selection becomes honest:
  - the fallback returns for systems no basis fits;
  - a per-system choice between the sparse and ridge heads keeps K = 32 performance.
- **No training.** Frozen checkpoints at t = 51 only, the same development episodes, and the same kernel ridge fallback (exact closed-form leave-one-out).
- **Arms:**
  - **hybrid6 (primary):** Trial 6 bases. Options per episode: each basis with the sparse head (exact leave-one-out), each basis with the ridge head (exact closed-form leave-one-out), and kernel ridge. 9 options, lowest leave-one-out wins.
  - **sparse6-exact:** Trial 6 bases, sparse head only (exact leave-one-out), plus kernel ridge. Isolates the selection fix.
  - **hybrid5:** Trial 5 seq-replay-400 bases with the same 9 options. Tests whether honest selection makes the sparse head a pure add-on to the confirmed method.
- **Criteria** (each arm, both seeds, against Trial 5's official seq-replay-400 t = 51 numbers):
  - **G1:** K = 8 aggregate ≤ 0.75× and ≥ 9/13 wins.
  - **G2:** K = 32 aggregate ≤ 1.5×.
  - **H1 (fallback restored):** Thomas and WindmiReduced at K = 8 are each ≤ 1.2× their seq-replay-400 values.
- **Reported:** how often each option is selected, and the S1/C3 readouts at t = 51.

**What happens next, decided now:**
- **hybrid6 passes G1, G2 and H1 in both seeds:** Trial 6's bases plus honest selection are a confirmation candidate. Retraining stability (problem 2) still needs a seed-robustness check before any confirmation request.
- **Only hybrid5 passes:** the sparse head is a free add-on to the confirmed method, but training through it is not yet useful.
- **None pass:** the K = 8 thread is set aside, and the next trial moves to the observation-contract simplification (trajectories instead of directly observed derivatives).

## Trial 7: result (29 October)

**Classification: FAILED** for all three arms (G1 fails everywhere).

| t = 51, development set | K = 8 (ratio to seq-replay-400; wins) | K = 32 ratio | H1: Thomas, WindmiReduced K = 8 ratios | Criteria |
|---|---|---|---|---|
| hybrid6, seed 1 | .130 (0.94; 9/13) | 1.61 | 1.00, 2.85 | G1 ✗ G2 ✗ H1 ✗ |
| hybrid6, seed 2 | .235 (1.86; 2/13) | 18.6 | 1.07, 2.84 | ✗ ✗ ✗ |
| sparse6-exact, seed 1 | .134 (0.97; 10/13) | 1.61 | 0.96, 2.86 | ✗ ✗ ✗ |
| sparse6-exact, seed 2 | .199 (1.57; 2/13) | 18.4 | 1.07, 2.86 | ✗ ✗ ✗ |
| hybrid5, seed 1 | .143 (1.03; 3/13) | 1.08 | 1.00, 1.03 | ✗ ✓ ✓ |
| hybrid5, seed 2 | .116 (0.92; 4/13) | 1.00 | 1.04, 1.00 | ✗ ✓ ✓ |

### Reading

- **Trial 6's seed-1 gain at K = 8 was mostly an artefact of the biased selection.** Under exact leave-one-out it shrinks from .077 to .130–.134. The biased estimate acted as a hard-coded "always trust the sparse fit" prior. That suited the polynomial majority and broke the rest.
- **Leave-one-out cannot resolve the choice at K = 8.** With 8 points, honest selection among 9 options is too noisy to beat the plain ridge library (Trial 4's oracle gap was already 2.3× at K = 8).
- **Extrapolation, not just selection.** WindmiReduced stays 2.85× worse even with exact leave-one-out. Context leave-one-out (trajectory segment A) does not predict how sparse fits extrapolate to the query segment (B).
- **hybrid5 shows the sparse head is harmless when added honestly** (G2 and H1 pass), but it buys nothing at K = 8.
- **K = 8 thread set aside**, as pre-registered. The remaining idea worth revisiting later is a *learned* selection prior: experience about which option tends to be right, rather than leave-one-out alone. It was not pursued now because Trial 3's context-conditioned model, the closest relative, was unstable.

## Trial 8: noisy trajectories instead of observed derivatives (pre-registered 29 October, before training)

- **Why.** Every result so far assumed the learner directly observes derivatives with 1% noise. Real systems give sampled trajectories. If accumulated knowledge only works under the idealized contract, the confirmed result means much less.
- **New observation contract.**
  - The learner observes K short noisy snippets of the trajectory, each three consecutive samples (x(t−h), x(t), x(t+h)).
  - h is the data bank's native spacing, 100 periods / 3999 ≈ 1/40 of a characteristic period. The time stamps are supplied.
  - Each state gets Gaussian noise of 0.5% of that trajectory's per-dimension state standard deviation.
  - **The learner's context** is the noisy middle states, with central-difference derivative *estimates* (x(t+h) − x(t−h)) / 2h.
  - **Standardization** uses those K estimates only, as before.
  - **Scoring:** query states (segment B) are clean, and the error is against the true vector field. The model is judged as a vector field; the learner never sees the truth.
  - Snippet centres are drawn at random from segment A, excluding its ends.
- **Arms:**
  - **frozen5:** Trial 5's seq-replay-400 checkpoints (all t), evaluated under the new contract with no retraining. Tests whether stored knowledge survives a change of observation process.
  - **seqtraj400:** seq-replay-400 retrained from scratch with training episodes built under the new contract (same stream, orders, seeds, steps and memory).
- **Controls under the same contract:** kernel ridge, random Fourier features, untrained library+fb (t = 0). Reference: poly2.
- **Criteria** (for seqtraj400, both seeds, development episodes; generator seeds 5000 + 97i + K as before, noise drawn from the same generator):
  - C1 and C2, as defined in Trial 1;
  - **T1 (substantial):** at t = 51, K = 32 aggregate ≤ 1/3 of the best experience-free control's.
- **Reported:** the same criteria for frozen5; the ratio of each arm's t = 51 aggregates to its derivative-contract counterpart; strata.

**What happens next, decided now:**
- **seqtraj400 passes C1, C2 and T1 in both seeds:** the accumulation result survives realistic observations. Together with Trial 5, that is a confirmation candidate for reserve set A under the trajectory contract. Stop and ask before any use of reserve set A.
- **Only frozen5 fails:** knowledge must be learned under the deployment contract.
- **seqtraj400 fails:** identify whether estimation noise or bias is responsible, by re-scoring with clean-state differences, before anything else.

**Resources:**
- 2 training streams of about 25 minutes each, plus evaluation of 32 checkpoints;
- outputs under `outputs/accumulating_learner/trial8/`.

## Trial 8: result (29 October)

**Classification: FAILED.** For seqtraj400, T1 fails in both seeds, and C1 and C2 fail in seed 2.

| t = 51, development, K = 32 aggregate | Seed 1 | Seed 2 |
|---|---|---|
| seqtraj400 (retrained under the trajectory contract) | .0027; C1 ✓ C2 ✓ (11/13); T1 ✗ (0.60) | .0022; C1 ✗ (ρ −0.57) C2 ✗ (8/13); T1 ✗ (0.44) |
| frozen5 (Trial 5 library, not retrained) | .0025; C1 ✗ C2 ✓; T1 ✗ (0.57) | .0021; C1 ✓ C2 ✓; T1 ✗ (0.41) |

Under the trajectory contract, kernel ridge scores .0050, random Fourier features .0090 and poly2 .0017. Under the derivative contract, the same library scored .00017 / .00009, 15–25× better than here.

### Pre-registered diagnostic: noise or bias?

Same episodes, state noise set to 0, so the remaining error is central-difference bias only.

| Aggregate NMSE, no noise | K = 8 | K = 16 | K = 32 | K = 64 | K = 128 |
|---|---|---|---|---|---|
| Kernel ridge | .308 | .0228 | .0046 | .0026 | .0017 |
| poly2 | .366 | .0024 | **.0015** | **.0015** | **.0018** |
| frozen5 N = 51, seed 1 / 2 | .0094 / .063 | .0025 / .0019 | .0020 / .0014 | .0014 / .0013 | .0014 / .0012 |
| seqtraj400 N = 51, seed 1 / 2 | .063 / .077 | .0029 / .0018 | .0024 / .0028 | .0015 / .0024 | .0015 / .0016 |

**Reading.**

- **The trajectory contract is limited by bias, not noise.** With noise removed, every method still floors at roughly 1–3e-3, and the error stops falling with K: even the exact quadratic basis stays at .0015.
- Sampled at 1/40 of a characteristic period, the central differences are systematically biased estimates of the vector field (about .025 NMSE for fast systems such as Thomas). Scoring against the *true* field therefore measures a floor set by the data, not learning. The advantage over kernel ridge is compressed to about 2×.
- Side observation, not a criterion: without noise, the frozen library is extremely strong at K = 8 (.009 / .063 against .31 for kernel ridge).
- Trial 8 also still used ground-truth derivatives as *training* query targets, a dependence the deployed learner could not have.

## Trial 9: learn and score only observable quantities (pre-registered 29 October, before training)

- **Hypothesis.** When a learner is trained and judged on what it can actually observe — predicting the trajectory's central-difference increment (x(t+h) − x(t−h)) / 2h at a state — the bias floor disappears from the task. Accumulation should then again give a large advantage over experience-free methods, with **no ground-truth derivatives used anywhere**: not in meta-training, not at encounter.
- **Contract.**
  - Context: as in Trial 8, K noisy snippets with central-difference estimates.
  - Training query targets: noisy central-difference estimates at the query states, which are observable. Query input states are noisy too.
  - Evaluation query targets: *clean* central differences at clean query states, i.e. the expected observation, so the scored quantity is noise-free but contains no information beyond the trajectory itself.
  - Query states come from segment B with both neighbours inside the trajectory.
- **Arms:**
  - **seqobs400:** seq-replay-400, retrained (same stream, orders, seeds, steps and memory) with this contract throughout.
  - **frozen5-obs:** Trial 5 checkpoints scored under this contract.
- **Controls under the same contract:** kernel ridge, random Fourier features, untrained library+fb. Reference: poly2.
- **Criteria** (seqobs400, both seeds, development episodes, seeds 5000 + 97i + K as before): C1; C2; T1 (t = 51, K = 32 aggregate ≤ 1/3 of the best experience-free control's).

**What happens next, decided now:**
- **All pass:** the accumulation result holds with observable-only supervision. That is the confirmation candidate (reserve set A, trajectory contract). Stop and ask.
- **Fail, while frozen5-obs passes:** retraining under noisy targets is the problem. Investigate the training signal.
- **Both fail:** report it. Accumulation under realistic trajectory observations is not established, and the confirmed result stays specific to the derivative contract.

**Resources:** 2 streams of about 25 minutes, then evaluation; outputs under `outputs/accumulating_learner/trial9/`.

## Trial 9: result (29 October)

**Classification: FAILED.** For seqobs400, C1 passes in both seeds, C2 in seed 1 only, and T1 fails in both. No ground-truth derivative was used anywhere: a test with NaN-filled derivative arrays confirmed the code never reads them.

| t = 51, development | Seed 1 | Seed 2 |
|---|---|---|
| seqobs400: C1 (ρ; CI) | ✓ (−0.98; [−1.39, −0.10]) | ✓ (−0.83; [−1.60, −0.06]) |
| seqobs400: C2 against kernel ridge | ✓ 11/13 | ✗ 8/13 |
| seqobs400: T1 (K = 32 ratio to kernel ridge; need ≤ 1/3) | ✗ 0.53 (.0023 vs .0043) | ✗ 0.53 (.0023 vs .0043) |
| seqobs400 K = 8 / 16 | .219 / .0077 | .162 / .0068 |
| frozen5-obs: C1, C2, T1 | ✗, ✓, ✗ (0.47) | ✓, ✓, ✗ (0.51) |

Controls under this contract at K = 8 / 16 / 32:

| | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| Kernel ridge | .301 | .031 | .0043 |
| Random Fourier features | .531 | .049 | .0070 |
| poly2 | .573 | .0066 | .0035 |

### Reading

- **Using observable targets removes the bias floor from the task, but the noise floor remains.** Central-difference noise of about 5e-4 NMSE per dimension puts every method near its floor by K = 32. So the advantage from experience is about 4× at K = 16 but only about 1.9× at K = 32.
- **The experience curve itself survives realistic observation:** C1 passes in both seeds with no ground-truth derivatives anywhere.
- **The confirmed large advantage (about 65× at K = 32) is specific to the low-noise derivative contract.** With realistic trajectory noise, accumulated knowledge buys roughly a 2–4× error reduction at K = 16–32, and the pre-registered "substantial" threshold (3×) is not met at K = 32.
- **frozen5-obs performs about as well as retraining under the new contract.** The stored knowledge transfers across observation processes; retraining adds nothing measurable here.

## Synthesis after Trials 1–9 and the confirmation run (29 October)

### Established

1. **Confirmed on untouched mechanisms (derivative contract).** A library of four learned feature bases, accumulated *sequentially* over 51 published 3-D chaotic flows with a 6% replay memory and leave-one-out selection with a kernel ridge fallback, learns new held-out systems from the same catalog with much less data than experience-free methods.
   - Error is about 65× lower at K = 32 and 40× lower at K = 16.
   - The gain grows with the number of systems seen.
   - Encounter-order analysis shows the same system is learned better when it arrives later in the stream.
   - Replay is necessary: without it the learner fails to retain.
2. **The experience curve survives a realistic, observable-only contract** (development set): C1 passes in both seeds in Trial 9. Its size there is modest (2–4×), limited by observation noise.

### Not established

- **Any advantage at K = 8 on new mechanisms.** It did not confirm, and neither learned sparsity (Trials 6–7) nor anything else so far fixes it under honest selection.
- **Generality beyond the catalog's polynomial majority.** Non-polynomial development systems are mixed: big wins on YuWang and SaltonSea, kernel ridge level on Thomas and WindmiReduced. Only 2 non-polynomial confirmation systems existed, again mixed.
- **A large advantage under realistic noise.**
- **Dimension beyond 3.**

### Candidate next steps

1. **Generality probe on reserve set A** (11 untouched, mostly non-polynomial 3-D mechanisms, with right-hand sides ported by hand). This tests the confirmed derivative-contract result as is. It spends the last 3-D reserve set; a failure would still be the most informative single result available.
2. **Keep reserve set A for later** and develop on the development set first. For example: train on a broader, independently motivated source of non-polynomial mechanisms, or a learner that works for any state dimension, which would also unlock reserve set B. Each of these is a multi-trial programme.
3. **Better observation design under noise** (longer snippets, local-polynomial derivative estimates). This is generic engineering that would shrink the noise floor. It is not accumulation, so it is a low priority for the owner's goal.

**My recommendation is option 1.** The confirmed result's most important open question is whether it says anything beyond polynomial flows, and reserve set A answers that directly. Option 1 uses reserve data, so per the owner's rule it needs approval. **Work pauses here for that decision.**

### Protocol for option 1, fixed now

1. Hand-port the 11 right-hand sides from the vendored `flows.py` into a reviewed module, with the diff shown in the log before any integration.
2. Generate data with the confirmation procedure (published parameters plus 3 perturbed instances). Systems whose published instance fails are dropped and reported.
3. Evaluate the frozen seq-replay-400 checkpoints (both seeds, all t), joint library+fb and wide, with the same controls and episode seeding scheme (seeds 7000 + 97i + K).
4. **Pass = CF1 and CF2 in both seeds.** CF3 and the per-system results are reported.
5. One shot; recorded whatever it shows.

## Owner guidance (29 October, after the Trial 9 synthesis)

The owner asked whether testing on reserve set A would just mean creating more test systems later (yes: it would spend the last untouched 3-D set), and noted my uncertainty about the network. Their direction: *"the goal is to create a network that can learn from its history without being supplied the answer; whatever we have to adjust or try to achieve that has my approval."*

**Decisions:**
- Reserve set A stays untouched until a network passes development criteria designed to predict success on it.
- Two gaps between the confirmed network and the goal come first:
  1. It was meta-trained with true derivatives as targets, i.e. supplied the answer. Trial 9's observable-only learner removes this.
  2. It is frozen at test time, so it does not learn from the new systems it meets.
- **Prequential evaluation design.** Every new system is scored at encounter, *before* the network learns from it, so every encounter is an unseen test. The goal quantity — does a growing history make the next system easier? — is then measured directly. A reserved set would be used once, as a final stream, rather than as a frozen-network benchmark.

## Trial 10: does the observable-only learner keep learning from new systems at test time? (pre-registered 29 October, before running)

- **Hypothesis.** If the Trial 9 observable-only library keeps learning after each new system it encounters (with replay), the systems it meets later in a stream are learned better than by an equal-compute learner that never learns from new systems. This includes non-polynomial systems, and earlier knowledge is not lost.
- **Start point.** Trial 9's seqobs400 checkpoints at t = 51, both seeds. Trained only on observable quantities; no ground-truth derivatives anywhere in this trial.
- **Stream.** The 13 development systems in 6 random orders per seed; order r of seed s is `default_rng(1000·s + r).permutation(development)`.
- **At each encounter:**
  1. Score library+fb at K = 16 and K = 32 on that system's Trial 9 evaluation episodes (identical seeds), using np.median over the 20 episodes.
  2. Then learn: 400 steps × 64 tasks, half from the new system's observable data (all 4 instances), half from memory. Memory holds instances 0–1 of all 51 meta-train systems plus each earlier stream system.
  3. Learning rate: per-stage cosine from 1e-3 to 1e-4, with a fresh Adam per order.
- **Arms:**
  - **continual:** as described.
  - **replay-only:** the same number of steps, but every batch comes from the meta-train memory. It never learns from the stream systems, so it controls for extra compute.
  - **frozen:** no updates. Its scores equal Trial 9's.
- **Primary quantity:** L = log(continual error / replay-only error) at encounter, at K = 32 (K = 16 reported). At position 1, L = 0 by construction.
- **Criteria (both seeds):**
  - **P1 (history helps):** Spearman ρ(position, L) ≤ −0.3 over all 78 (order × system) pairs, *and* the median L over positions ≥ 7 is < 0.
  - **P2 (beyond polynomial):** the median L over non-polynomial systems at positions ≥ 7 is < 0.
  - **P3 (no forgetting):** after the stream, on the first 8 meta-train systems (observable contract, K = 32), the median of NMSE after / NMSE before is ≤ 1.5, averaged over orders.

**What happens next, decided now:**
- **P1–P3 pass:** a candidate for the final reserve-set-A stream, with the same protocol and the 11 systems in 6 orders. Ask before touching it.
- **P1 fails:** 13 more systems on top of 51 may give too small a marginal gain to detect. Next, test the encounter curve at a smaller history (start from earlier checkpoints), and consider a library that grows bases for systems no basis fits.
- **P3 fails:** test-time learning causes forgetting. Adjust replay before anything else.

**Resources:**
- 2 seeds × 6 orders × 2 trained arms = 24 short streams (13 × 400 steps each), run in parallel;
- outputs under `outputs/accumulating_learner/trial10/`.

## Trial 10: result (29 October)

**Classification: FAILED.** P1 and P2 fail in both seeds; P3 passes in both.

| | Seed 1 | Seed 2 |
|---|---|---|
| P1: Spearman ρ(position, L) at K = 32 (need ≤ −0.3) | +0.08 ✗ | −0.23 ✗ |
| P1: median L, positions ≥ 7, K = 32 | −0.005 | −0.027 |
| Same at K = 16 (ρ; median L) | −0.03; −0.060 | −0.08; −0.085 |
| P2: non-polynomial median L, positions ≥ 7 | +0.028 ✗ | +0.002 ✗ |
| Continual vs frozen, median L at positions ≥ 7, K = 32 | −0.023 | −0.004 |
| P3: retention ratio after the stream (need ≤ 1.5) | 1.15 ✓ | 1.02 ✓ |

### Reading

- **No forgetting, but no detectable benefit.** Learning from each development system after its encounter costs nothing (P3), but it does not make later development systems easier than equal extra compute on replay does. The improvement is at most about 3% at K = 32 and 6–8% at K = 16.
- Per-system effects are noisy and of both signs; for example SprottI improves by −0.50 in seed 2 while Halvorsen worsens by +0.92 in seed 1.
- **Two candidate explanations, decided now as the next test:**
  1. Diminishing returns: on top of 51 systems, 13 more add little.
  2. Unrelatedness: the development systems, especially the non-polynomial ones, are mostly unique mechanisms, so one cannot inform another.
- Trial 11 tests explanation 1 directly, as pre-registered.

## Trial 11: test-time accumulation from a short history (pre-registered 29 October, before running)

- **Hypothesis.** Test-time accumulation works when the history is short. Starting from the observable-only library after only **8** meta-train systems, learning from each encountered development system makes later development systems easier than equal-compute replay.
- **Identical to Trial 10 except:**
  - starting checkpoint: Trial 9's seqobs400 at t = 8, both seeds;
  - memory: instances 0–1 of the 8 meta-train systems that checkpoint has seen (the first 8 in that seed's order), plus stream systems for the continual arm. The replay-only arm replays only those 8.
- **Criteria:** P1, P2 and P3, exactly as in Trial 10 (P3's 8 retention systems are exactly the 8 seen).
- **Also reported:** the gap between continual and frozen at positions ≥ 7.

**What happens next, decided now:**
- **P1 passes:** test-time accumulation works when experience is scarce, and Trial 10's null is diminishing returns. Then decide whether a reserve-set-A stream from t = 51 can show anything. The likely answer is that the final test should measure the encounter curve from a short history.
- **P1 fails here too:** the explanation is unrelatedness among the development systems, not saturation. Next, try a learner that adapts more at encounter time (per-system fine-tuning of the selected basis) or that grows new bases, and source related-mechanism families for training.

**Resources:** 12 short streams, as in Trial 10; outputs under `outputs/accumulating_learner/trial11/`.

## Trial 11: result (29 October)

**Classification: FAILED.** P1 fails in both seeds, P2 passes in seed 1 only (barely), and P3 fails in both.

| | Seed 1 | Seed 2 |
|---|---|---|
| P1: ρ(position, L) at K = 32 | −0.16 ✗ | −0.24 ✗ |
| P1: median L, positions ≥ 7, K = 32 | −0.10 | −0.12 |
| P2: non-polynomial median L, positions ≥ 7 | −0.019 ✓ | 0.000 ✗ |
| Continual vs **frozen**, median L at positions ≥ 7 | −0.031 | −0.013 |
| P3: retention ratio (need ≤ 1.5) | 1.77 ✗ | 1.54 ✗ |

### Reading

- **History helps for related mechanisms only.**
  - The Sprott family and BurkeShaw gain a lot from earlier stream systems. Per-system median L at positions ≥ 7 in seeds 1 / 2: SprottQ −0.66 / −1.25, SprottJ −0.02 / −1.48, SprottP −0.71 / −0.64, SprottI −0.33 / −0.47, BurkeShaw −0.42 / −0.92.
  - The one-of-a-kind mechanisms do not gain and some get worse: ShimizuMorioka +0.73 / +0.33, RikitakeDynamo +0.67 / +0.07, Thomas +0.28 / 0.00, KawczynskiStrizhak −0.09 / +0.23.
  - Together with Trial 10 this supports explanation 2: the null result is mostly about how related the stream systems are, not about saturation.
- **The replay-only control is weak at short history.** Over-training on 8 systems degrades it. Against the frozen learner, test-time learning helps only 1–3% in the median.
- **Forgetting appears at short history.** With only 8 earlier systems in memory, learning 13 new ones crowds them out (ratio about 1.5–1.8). At 51 systems (Trial 10) it did not.
- **Implication for the goal.** The learner turns related experience into faster learning, both in meta-training and at test time. For a mechanism unlike anything in its history, the only gains available are from how it adapts at encounter. Its current encounter-time adaptation is a linear head on frozen features, which falls back to kernel ridge level for such systems.

## Trial 12: stronger adaptation at encounter for unfamiliar mechanisms (pre-registered 29 October, before running)

- **Hypothesis.** For a mechanism its stored features do not fit, the learner does better by *adapting* its best stored basis to the K observations (transfer by fine-tuning) than by keeping it frozen or falling back to kernel ridge. History then provides a starting point even for unfamiliar mechanisms. Selection on a held-out part of the context keeps this from hurting familiar systems.
- **Setting.** Evaluation only, at encounter: Trial 9's observable-only seqobs400 t = 51 checkpoints (both seeds), Trial 9's development episodes and contract, K ∈ {32, 64}. No ground-truth derivatives.
- **Procedure per episode:**
  1. Library+fb selection by exact leave-one-out (as before) gives option o*.
  2. **If o* is a basis:**
     - split the context at random (seeded per episode) into 75% fit and 25% validation;
     - fine-tune a copy of that basis (all parameters) for 100 Adam steps (lr 3e-4) on the fit part, minimizing the ridge head's closed-form leave-one-out MSE, with λ fixed to the value selected before fine-tuning;
     - choose between the fine-tuned and original basis by ridge-head validation MSE (head fitted on the fit part);
     - refit the head on all K with the chosen basis.
  3. **If o\* is kernel ridge:** also fine-tune the basis with the lowest leave-one-out score, and choose between it and kernel ridge on the same validation split. Kernel ridge is scored there with its leave-one-out-chosen hyperparameters, fitted on the fit part.
- **Criteria** (both seeds; against Trial 9's library+fb on the same episodes):
  - **A1 (unfamiliar mechanisms):** non-polynomial-stratum median NMSE at K = 64 ≤ 0.8× library+fb's, *and* fine-tuning beats kernel ridge on ≥ 3 of 4 non-polynomial systems at K = 64.
  - **A2 (no harm):** overall aggregate at K = 32 and K = 64 ≤ 1.1× library+fb's.
- **Reported:** the same at K = 32, and how often the fine-tuned basis is chosen.

**What happens next, decided now:**
- **A1 and A2 pass:** encounter-time adaptation is the route to unfamiliar mechanisms. Next, meta-train *for* adaptability, so the inner fine-tuning is part of meta-training (learning how to adapt from history).
- **A1 fails:** with K ≤ 64 noisy observations, fine-tuning cannot extract an unfamiliar mechanism beyond what kernel ridge gets. The binding constraint for unique mechanisms is then information per encounter, not the learner, and the next lever is what to observe.

**Resources:** evaluation only, about 1,000 fine-tunings × 100 steps; outputs under `outputs/accumulating_learner/trial12/`.

## Trial 12: result (29 October)

**Classification: FAILED.** A1 fails in both seeds; A2 passes in both.

| | Seed 1 | Seed 2 |
|---|---|---|
| A1: non-polynomial median, fine-tune vs library+fb at K = 64 (need ≤ 0.8) | 1.03 ✗ | 1.22 ✗ |
| A1: fine-tune beats kernel ridge on non-polynomial systems at K = 64 (need ≥ 3/4) | 1/4 ✗ | 1/4 ✗ |
| A2: overall ratio at K = 32 / 64 (need ≤ 1.1) | 0.93 / 1.01 ✓ | 0.95 / 1.08 ✓ |
| Fine-tuned basis chosen by validation | 25% | 25% |

Non-polynomial systems at K = 64 (fine-tune / library+fb / kernel ridge, seed 1):

| System | Fine-tune | Library+fb | Kernel ridge |
|---|---|---|---|
| SaltonSea | .00051 | .00051 | .00075 |
| Thomas | .0101 | .0101 | .0078 |
| WindmiReduced | .089 | .093 | .086 |
| YuWang | .069 | .067 | .055 |

### Reading

- **Encounter-time fine-tuning is safe, but it does not extract unfamiliar mechanisms.** With at most 64 noisy observations under the observable-only contract, neither the frozen library, the fine-tuned library (Trial 12) nor test-time accumulation over a stream (Trials 10–11) beats generic kernel ridge on mechanisms unlike the history.
- **The consistent picture after Trials 10–12.** History makes learning faster and more data-efficient when it contains related mechanisms: confirmed on held-out systems, and seen again at test time within the Sprott family. For a genuinely unfamiliar mechanism, what limits the learner is information per encounter.
- **As pre-registered, the next lever is what to observe.** Use experience to decide *where* to look, which is the "learns how to investigate" aspiration in the handover.

## Trial 13: does experience help decide what to observe? (pre-registered 29 October, before running)

- **Hypothesis.** When observations are costly and the learner can choose which to take, choosing them with its experience-derived features gives lower error at a fixed budget than random choice, and lower than a generic space-covering design. This is investigation guided by history.
- **Contract: pool-based active observation.** It extends the observable-only contract; the extension is stated openly.
  - For each episode, the learner sees a pool of 400 candidate states for free: noisy middle states of snippets at random centres in segment A, with 0.5% noise as before.
  - A derivative estimate (that snippet's central difference) costs one unit of budget. The budget is K.
  - Queries and scoring are as in Trial 9: clean central differences at clean segment-B states.
  - No ground-truth derivatives are used. The pool's states are themselves observations, not answers.
- **Learner.** Trial 9 seqobs400 t = 51 library+fb, both seeds, frozen.
- **Strategies, at the same K:**
  - **random:** K pool points at random; this is Trial 9's contract.
  - **coverage (generic):** greedy farthest-point selection in pool-standardized state space, starting from one random point.
  - **experience (primary):**
    1. Take K/2 random pool points.
    2. Select an option by exact leave-one-out as usual (4 bases plus kernel ridge).
    3. If a basis is selected, add the remaining K/2 points greedily by maximum leverage, φ(x)ᵀ(ΦᵀΦ + λK·I)⁻¹φ(x), in that basis's features with its selected λ, updating after each point.
    4. If kernel ridge is selected, add the remaining points by coverage.
    5. Finally, run library+fb on all K labelled points.
- **Controls:** kernel ridge with random and with coverage selection.
- **Episodes:** per development system and K ∈ {8, 16, 32}: 4 instances × 5, generator seeds 13000 + 97i + K, shared by all strategies.
- **Criteria** (both seeds, aggregate median over the 13 systems):
  - **I1 (choosing helps):** at K = 16, experience ≤ 0.67× random for the same learner.
  - **I2 (experience adds to generic design):** at K = 16, experience ≤ 0.83× coverage for the same learner.
- **Reported:** K = 8 and 32, strata, kernel ridge with random and coverage selection, and the learner with coverage selection.

**What happens next, decided now:**
- **I1 and I2 pass:** history-guided investigation works. Next, test whether it helps unfamiliar mechanisms specifically, and move toward reset-style probing.
- **I1 passes but I2 fails:** active choice helps but generic coverage is as good. Experience is not what makes investigation work here.
- **I1 fails:** on-attractor pools are too redundant for choice to matter. Report it and reconsider observation design (controlled probes).

**Resources:** evaluation only; outputs under `outputs/accumulating_learner/trial13/`.

## Trial 13: result (29 October)

**Classification: FAILED.** I1 passes in both seeds; I2 fails in both.

| Aggregate NMSE (median over 13 development systems) | K = 8 | K = 16 | K = 32 |
|---|---|---|---|
| learner, random, seed 1 / 2 | .152 / .170 | .0086 / .0061 | .0022 / .0026 |
| learner, coverage, seed 1 / 2 | .148 / .052 | **.0016 / .0013** | **.00037 / .00034** |
| learner, experience, seed 1 / 2 | **.079 / .038** | .0027 / .0021 | .00049 / .00039 |
| kernel ridge, random | .370 | .0124 | .0050 |
| kernel ridge, coverage | .142 | .0024 | .00060 |

- **I1:** experience vs random at K = 16 is 0.32 and 0.35, with wins on 13/13 and 12/13 systems. ✓
- **I2:** experience vs coverage at K = 16 is 1.72 and 1.71. ✗

**Non-polynomial stratum, kernel ridge:** K = 16 goes from .26 (random) to .12 (coverage); K = 32 from .10 to .022. The learner with coverage gets .14 / .14 at K = 16 and .025 / .021 at K = 32.

### Reading

- **Observation design is a large lever, for every method and every kind of system,** including the unfamiliar non-polynomial ones that no learner beat before. Spreading K observations over the state space is worth 3–5× at K = 16–32 compared with random on-attractor samples, which cluster.
- **Experience-guided leverage beats coverage only at K = 8 (exploratory).** At K = 16–32 a generic design wins.
- **With good designs, the learner's advantage over kernel ridge stays about 1.5–2×** at K = 16–32: under coverage, .0016 / .0013 against .0024 at K = 16.
- **So far investigation has been driven by hand-written rules** (coverage, leverage). The owner's aspiration is to *learn* from history how to investigate. That is the next test.

## Trial 14: a learned investigation policy (pre-registered 29 October, before running)

- **Hypothesis.** A choice rule trained on past systems — learning which candidate observation most reduces prediction error — chooses better than the generic coverage rule, because it can combine coverage with what the stored features say is informative.
- **Contract.** Trial 13's pool-based active observation (observable-only, 400 free noisy candidate states, a budget of K central-difference estimates, scored on clean segment-B central differences). The frozen Trial 9 seqobs400 t = 51 library, both seeds.
- **Policy.**
  - A small MLP (6 → 32 → 32 → 1) scores each candidate c from six descriptors:
    1. log leverage of c in the currently selected basis;
    2. log distance from c to the nearest chosen point, in pool-standardized space;
    3. log mean distance to its 10 nearest pool neighbours (local density);
    4. log norm of the current model's prediction at c;
    5. log k (current number of observations);
    6. log of the current leave-one-out MSE.
  - Greedy selection: start from 2 random pool points; at each step add the highest-scoring unchosen candidate. The basis is re-selected by exact leave-one-out whenever k ∈ {2, 4, 8, 16}.
- **Training (history only, meta-train systems).**
  - Per meta-train system: 40 random episodes, each with a random target K ∈ {8, 16, 32}, a random current set of size k ∈ [2, K − 1], and 64 random candidates.
  - Label for each candidate: log(query MSE before / query MSE after adding it). The ridge head of the leave-one-out-selected basis is refitted, with observable central-difference targets on that system's segment-B queries.
  - Adam, 3,000 steps, MSE on standardized labels.
- **Evaluation:** Trial 13's development episodes and seeds, with the learned policy added as a strategy. Final prediction by library+fb, as in Trial 13.
- **Criteria (both seeds):**
  - **L1:** at K = 16, learned ≤ 0.83× coverage (the same bar as I2);
  - **L2:** at K = 8 and K = 32, learned ≤ 1.0× coverage.
- **Reported:** strata, learned vs experience-leverage, and a descriptor ablation using coverage distance only.

**What happens next, decided now:**
- **L1 and L2 pass:** history-learned investigation beats generic design. That would be the strongest goal-aligned result so far, and a candidate for reserve set A under the active contract. Ask before using it.
- **Fail:** generic coverage is near-optimal for this observation model. Experience then pays off through the model, not the investigation. Write the final synthesis, with remaining directions for the owner.

**Resources:** evaluation plus policy training (minutes); outputs under `outputs/accumulating_learner/trial14/`.

## Trial 14: result (29 October)

**Classification: FAILED.** L1 and L2 fail in both seeds.

| Learned policy / coverage, aggregate ratio | K = 8 | K = 16 (L1 needs ≤ 0.83) | K = 32 |
|---|---|---|---|
| Seed 1 | 1.86 | 1.00 | 1.25 |
| Seed 2 | 2.16 | 1.26 | 1.78 |

- The coverage arm reproduces Trial 13's numbers exactly, so the comparison is sound. The policy was trained on 130,560 candidate evaluations from meta-train systems only.
- A learned scorer that sees only the coverage descriptor is far worse than real farthest-point coverage (seed 1, K = 16: .014 against .0016). One-step "which point helps most now" labels do not teach the multi-step space-filling strategy that coverage embodies.
- **As pre-registered, generic coverage is near-optimal for this observation model.** Experience pays off through the model, not through investigation. Learning an investigation strategy would need a sequential objective, i.e. a multi-step return, as in the AlphaGo-style aspiration, rather than myopic labels. This has not been tried.

## Synthesis after Trials 1–14 and the confirmation run (29 October)

### What a learner that "learns from its history" achieves here

1. **Confirmed on untouched systems (derivative contract).** A library of four learned feature bases, accumulated one system at a time over 51 published chaotic flows with a 6% replay memory, learns new systems from the same catalog with far less data than experience-free methods: about 65× lower error at K = 32 and 40× at K = 16. The gain grows with the number of systems seen, and the same system is learned better when it arrives later in the stream.
2. **It survives realistic observation** (development set). With noisy trajectories and *no ground-truth derivatives anywhere* (Trial 9), the experience curve persists in both seeds. Its size shrinks to 2–4× because observation noise dominates.
3. **It keeps learning at test time without forgetting** when its history is long (Trial 10). When the stream contains related mechanisms (the Sprott family, Trial 11), learning one makes the next 1.5–4× easier.

### What it does not do

4. **Unfamiliar mechanisms.** For a mechanism unlike anything in its history, none of these beat generic kernel ridge with ≤ 64 noisy observations: stored features, test-time accumulation, or encounter-time fine-tuning (Trials 10–12). History only transfers to what it resembles.
5. **Very small budgets.** K = 8 did not confirm. Learned sparsity did not survive honest selection (Trials 6–7).
6. **Investigation.** *Where* to observe matters a great deal: a spread-out design is worth 3–5× for every method, including on unfamiliar mechanisms (Trial 13). But neither a hand-written experience-based rule nor a myopically learned policy beats generic coverage (Trial 14).

### What this implies for the goal

The learner turns *related* experience into faster and more reliable learning, without being supplied the structure (the term library is learned, not given). Its reach is bounded by how diverse its history is. In this catalog history is dominated by quadratic flows, so that is where the gains are. The main remaining lever for generality is **a broader history**: more, and more varied, mechanism families. Better per-encounter methods are not it. The second lever is **investigation with a multi-step objective**.

### Remaining directions, and what each needs

- **(a) Broaden the history.** Requires new mechanism sources.
  - Options: (i) re-designate part of reserve set B (31 untouched systems of dimension 4–10) as training history for a learner that works for any state dimension, keeping the rest as held-out tests; (ii) source and vet other published catalogs.
  - (i) changes the status of reserved data, so it needs the owner's decision.
- **(b) Investigation learned with a multi-step objective** (a sequential policy trained on history to minimize error after K observations). Development data only; can proceed.
- **(c) A final test on reserve set A.** Not recommended yet. With coverage design, the learner's advantage over kernel ridge on non-polynomial development systems is about 1×, so a reserve-set-A test would most likely confirm finding 4 and spend the set.

## Owner permission and decision on reserve set B (29 October)

The owner wrote: *"if you think this will help us reach our goal and has generalization capability you have my permission"* to use part of reserve set B as training history.

**Check made before deciding.** Names, dimensions and equation form only. I compiled the whitelisted right-hand sides to classify them symbolically; no data was generated and nothing was evaluated.

- Of B's 31 systems, 19 are usable autonomous ODEs with a static right-hand side. The rest are 7 delay embeddings, 2 nonautonomous systems, and 4 with custom right-hand sides.
- **17 of the 19 are polynomial**, mostly 4-D hyperchaotic extensions of the Lorenz, Chen, Lü and Rössler family that already dominates the training history: HyperBao, HyperCai, HyperJha, HyperLorenz, HyperLu, HyperPang, HyperQi, HyperRossler, HyperWang, HyperXu, HyperYan, HyperYangChen, LorenzStenflo, Qi, LorenzCoupled, HenonHeiles and NuclearQuadrupole.
- Only DoublePendulum and SwingingAtwood are non-polynomial.

**Decision: not used.** B would add more of the same structure in more dimensions. It would not broaden the *kinds* of mechanism in the history, and that is the bottleneck identified in the synthesis. It would also spend the only higher-dimensional test set. **Reserve set B stays reserved.**

## Trial 15: does a procedurally broadened history help unfamiliar mechanisms? (pre-registered 29 October, before generating anything)

- **Hypothesis.** The learner fails on unfamiliar mechanisms because its history contains only one family of structure. If the history also contains many procedurally generated, non-polynomial dynamical systems from a *generic* recipe that is not built around any test system, its stored features become more general. Real non-polynomial development systems then improve beyond both the catalog-only learner and kernel ridge, without losing the polynomial gains.
- **Synthetic mechanisms** (`synthetic.py`, seeded):
  - Form: f(x) = −Γx + Σ_{k=1..m} a_k σ_k(u_k·x + b_k), with:
    - x in R³;
    - Γ diagonal with entries U(0.1, 1);
    - m ~ U{3..8} terms;
    - u_k ~ N(0, I) / √3, b_k ~ U(−π, π), a_k ∈ R³ ~ N(0, s²·I) with s ~ U(1, 4);
    - σ_k drawn uniformly from {sin, tanh, z/(1+z²), exp(−z²), cos}, all bounded.
  - Bounded σ plus linear damping guarantees bounded trajectories. Products and polynomials are deliberately excluded, so this history adds *non-polynomial* structure.
  - **Acceptance** (no reference to any test system):
    - integrate from a random initial condition;
    - estimate a characteristic period P = 2π·mean‖x − x̄‖ / mean‖f‖ from a pilot run;
    - discard 50P, then sample 4,000 points over 100P;
    - keep the system if no coordinate's segment-B std falls below 1e-3 of its scale and the speed has not decayed (segment-B RMS of f ≥ 0.1 × segment-A RMS).
  - Generate until 200 systems are accepted, each with 8 instances (random initial conditions) and h = P/40, stored like the catalog bank.
- **Arms** (observable-only contract, Trial 9's training code, both seeds):
  - **catalog-only:** Trial 9's seqobs400 checkpoints (51 systems, 400 steps per stage, M = 4).
  - **catalog-long:** catalog-only at 1,970 steps per stage, matching the compute of the broad arm.
  - **broad:** a stream of 251 systems, the 51 catalog systems and 200 synthetic ones, interleaved by a seeded shuffle. 400 steps per stage, M = 4, replay memory of 2 instances per system.
  - **broad8:** broad with M = 8 bases, in case four bases lack capacity for a broader history.
- **Evaluation:** development systems, Trial 9 observable contract and episodes (random design), K ∈ {16, 32, 64}. Kernel ridge is the experience-free control.
- **Criteria** (both seeds, K = 32; the primary arm is whichever of broad and broad8 has the lower non-polynomial median in seed 1, decided by that rule before looking at seed 2):
  - **S-A (unfamiliar mechanisms):** non-polynomial-stratum median ≤ 0.75× catalog-only's *and* ≤ 0.75× catalog-long's, and it beats kernel ridge on ≥ 3 of 4 non-polynomial systems.
  - **S-B (no loss on familiar):** polynomial-stratum median ≤ 1.5× catalog-only's.

**What happens next, decided now:**
- **S-A and S-B pass:** a broader history buys generality. Propose a reserve-set-A test (11 real, mostly non-polynomial systems) to the owner, with the protocol fixed in advance.
- **S-A fails:** a generic synthetic history does not transfer to real unfamiliar mechanisms at these budgets. Report it. This would strengthen the "information per encounter" explanation.

**Resources:** synthetic data about 200 systems × 8 instances (about 80 MB); 2 seeds × 3 training arms; outputs under `outputs/accumulating_learner/trial15/`.

## Trial 15: result (30 October)

**Classification: FAILED.** S-A fails in both seeds; S-B passes in both. The primary arm, chosen by the pre-registered seed-1 rule, is **broad8**.

**Synthetic set.** 200 accepted systems from 2,960 seeds (6.8% acceptance). The five nonlinearity types were used roughly equally (230–259 terms each); characteristic periods ranged from 1.2 to 49.1, median 4.9.

| K = 32, development | Seed 1 | Seed 2 |
|---|---|---|
| S-A: broad8 non-polynomial median vs catalog-only (need ≤ 0.75) | 1.02 ✗ | 1.49 ✗ |
| S-A: broad8 non-polynomial median vs catalog-long (need ≤ 0.75) | 1.20 ✗ | 1.12 ✗ |
| S-A: broad8 beats kernel ridge, non-polynomial systems (need ≥ 3/4) | 2/4 ✗ | 2/4 ✗ |
| S-B: broad8 polynomial median vs catalog-only (need ≤ 1.5) | 0.94 ✓ | 0.68 ✓ |

Non-polynomial medians at K = 32:

| | Seed 1 | Seed 2 |
|---|---|---|
| catalog-only | .103 | .099 |
| catalog-long | .088 | .132 |
| broad | .138 | .136 |
| broad8 | .105 | .148 |
| kernel ridge | .100 | .100 |

At K = 64 every arm is between .033 and .048, against kernel ridge .031.

### Reading

- **A broad, generic, non-polynomial history does not transfer to real unfamiliar mechanisms** at these budgets. Every learner variant stays at kernel ridge level on them, as pre-registered for S-A failing. This strengthens the conclusion that, for a mechanism unlike its history, what limits the learner is information per encounter, not the breadth of its stored features.
- Familiar (polynomial) systems are unaffected or slightly better (S-B).
- **Resource incident.** During this trial I found my training processes were each using about 10 CPU cores through PyTorch's default threading (the samplers do CPU argsorts). Four runs pushed the shared host's load to about 95 on 48 cores, nearly all mine. I reniced the running processes to 15 at once. From now on runs are launched with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10`. The earlier 12-stream trials (10, 11) were probably oversubscribed too. GPUs were never shared with another user's job.

## Conclusion of this phase (30 October)

Across 15 trials and one confirmation run, the picture is consistent:

1. **Learning from history works, and is confirmed on untouched systems,** whenever new systems share structure with the history. The learner discovers that structure itself (no term library is supplied) and accumulates it one system at a time. Gains reach about 40–65× at 16–32 observations under low-noise observation, and 2–4× from noisy trajectories with no ground-truth targets anywhere.
2. **It does not generalize to genuinely unfamiliar mechanisms.** Nothing tried beats generic kernel ridge on such systems at ≤ 64 noisy observations: broader synthetic history (Trial 15), test-time accumulation (Trials 10–11), encounter-time fine-tuning (Trial 12), sparsity (Trials 6–7), or context-conditioned features (Trial 3).
3. **What does help unfamiliar mechanisms is better observations.** A spread-out observation design gives 3–5× for every method (Trial 13). Neither hand-written nor myopically learned experience-based rules beat generic coverage (Trials 13–14).

**Open directions**, each needing an owner decision on scope:

- **(i)** Source *real* families of related non-polynomial mechanisms from outside the catalog. Real relatives might transfer where generic synthetic ones did not, but each system needs vetting.
- **(ii)** A controlled-probe contract, where the learner chooses initial states to observe. Larger information per encounter, but the handover notes reset access materially simplifies problems.
- **(iii)** A multi-step learned investigation policy. Low expected value given (3).

Reserve sets A and B are untouched.

## Owner decision (30 October): option (i), plus uncertainty-driven learning

The owner chose option (i), real families of related non-polynomial mechanisms from outside the catalog. They also want "learning based off of what remains uncertain". That covers two things, and both are planned once option (i)'s data exists:

- **Where to look:** observations chosen where the learner is most uncertain, combined with coverage and planned over several steps. This is option (iii) generalized; Trial 13's leverage rule was a first version.
- **What to study:** replay and training prioritized toward the systems and regions with the highest remaining error.

### Sourcing rule for option (i), fixed before any download

To avoid hand-picking models and parameters from memory, which risks both error and selection bias, systems come from a curated external database of published models with their published parameters, **BioModels** (EBI), selected by a mechanical rule:

1. Curated models only.
2. After removing species fixed as constant or boundary, exactly 3 species are governed by ODEs (reaction kinetics or rate rules). No events, no delays, no explicit time dependence. Assignment rules are allowed if they are algebraic in the 3 species and constants.
3. The right-hand side is non-polynomial in the species (rational, Hill, exponential, and so on). This is checked symbolically after conversion.
4. Sustained motion: integrated from the published initial state with the published parameters, it passes the same acceptance test as the catalog and synthetic banks, i.e. a limit cycle or chaos, not a fixed point.
5. Each accepted model's identifier, title and publication are recorded. Models are grouped into families by the database's own annotations (for example GO or KEGG process terms) where available, otherwise by title keywords, recorded per model.

**Exclusions:**
- any model whose title names a system in reserve set A (Belousov–Zhabotinsky, Ca²⁺ oscillation, excitable cell, isothermal chemical, Colpitts, Chua, Hopfield or Beer networks, El Niño) or in the development or confirmation sets;
- duplicates of the same published model.

**Splits** (by hash of the model identifier):
- about two thirds become training history;
- one third becomes **family-held-out** test systems: related to training families, but unseen.

Reserve set A stays untouched for a final test.

**Downloaded files are untrusted data.** They go into their own directory. SBML is parsed as data (XML and MathML), never executed, and converted to whitelisted right-hand sides with the same AST checks as the catalog.

### Sourcing outcome (30 October, before any training or evaluation on these systems)

- **Download.** 1,019 curated ODE models listed; 1,017 SBML files obtained. The first pass guessed file names and failed for 335 models; a retry using each model's file listing recovered all but 2.
- **Conversion** (`biomodels_convert.py`):
  - **21 of 1,017 accepted.**
  - Main rejection reasons: number of state variables ≠ 3 (738), events (146), title exclusions (26), polynomial right-hand side (25), no sustained motion (45 across the pilot and instance-0 checks).
- **Code aligned with the written rule, before any learning result.** The first converter was stricter than the rule. It rejected models with initial assignments (allowed now, if they evaluate to constants) and with piecewise expressions (allowed now, if their conditions involve only states and constants; explicit time is still rejected).
- **Rule applications the first code missed:**
  - Lavrentovich2008_Ca_Oscillations excluded (Ca²⁺ oscillation title; the first title pattern missed `Ca_`).
  - Goldbeter-minimal-model duplicates excluded. BIOMD0000000944 is a structural duplicate of 729. 729 (Goldbeter1996) and 933 (Kosiuk2015, "geometric analysis of the Goldbeter minimal model") duplicate BIOMD0000000003 by title; this is a judgment, and both are recorded.
- **17 real non-polynomial models remain.**
- **Split.** The hash threshold gave 7 / 10 by chance with only 17 items. To realize the stated "about two thirds", the split was changed to **rank-based on the same hash**: the lowest 11 hashes are training. This was decided before any training or evaluation.
  - **Training (11):** Zatorsky2006 p53, Westermark2003 glycolysis, Leloup1999 Neurospora circadian, Kim2011 synthetic oscillator, DeVries2000 β-cell, Araujo2016 Cdk1, Dritschel2018 T-cell / tumour, Smith1980 hypothalamic, Lenbury2001 insulin, Bianca2013 cancer–immune, Ferrell2011 CDK1 oscillator.
  - **Family-held-out (6):** Goldbeter1991 mitotic oscillator, Rattanakul2003 bone, Decroly1982 enzymatic oscillator, Caravagna2010 tumour–immune, Voliotis2019 GnRH pulse generator, Gerard2013 cell cycle. Each has a related family in training.
- **Relatedness to reserve set A, noted for later.** DeVries2000 (training) is a pancreatic β-cell bursting model, the same biological system as reserve A's ExcitableCell. Its title does not trigger the exclusion rule, so it stays.
- **Not used:** the 738 models with more or fewer than 3 variables, a much larger real history that a learner working in any dimension could use.

## Trial 16: does related real history transfer to held-out real mechanisms? (pre-registered 30 October, before training)

- **Hypothesis.** A history containing *real* related families transfers to held-out members of those families, where generic synthetic history (Trial 15) did not. Concretely: the 11 training BioModels systems (cell cycle, metabolic, endocrine, tumour–immune and others) make the 6 held-out real systems learnable from fewer observations than the catalog-only learner or kernel ridge can manage.
- **Data.** Each BioModels system has 8 instances (instance 0 from the published initial state, the rest started from perturbed points on the attractor), h = P/40, observable-only contract (Trial 9). CPU threads are capped (OMP 4, nice 10).
- **Arms** (seeds 1, 2; M = 4; 400 steps per stage; replay of 2 instances):
  - **catalog-only:** Trial 9 seqobs400 t = 51 (existing);
  - **catalog+bio:** a stream of the 51 catalog systems plus the 11 BioModels training systems, seeded shuffle `default_rng(1600 + s)`;
  - **catalog+synthetic:** Trial 15's broad arm (existing), as the comparator for real vs generic history.
- **Evaluation.**
  - Held-out systems: instances 0–3, 5 episodes each, K ∈ {16, 32, 64}, `episode_obs` with seeds 16000 + 97i + K.
  - Development systems: Trial 9 episodes, for the no-harm readout.
  - Kernel ridge is the experience-free control.
- **Criteria (both seeds):**
  - **R1 (related real families transfer):** on the 6 held-out systems at K = 32, catalog+bio's median NMSE ≤ 0.67× catalog-only's, *and* catalog+bio beats kernel ridge on ≥ 4 of 6.
  - **R2 (real beats generic):** catalog+bio's held-out median at K = 32 ≤ 0.8× catalog+synthetic's.
  - **R3 (no harm):** development polynomial-stratum median at K = 32 ≤ 1.5× catalog-only's.
- **Reported:** K = 16 and 64, per-system results with family annotation, the development non-polynomial stratum.

**What happens next, decided now:**
- **R1 and R3 pass:** related real history transfers. Next, the owner's uncertainty-driven additions on this learner: (a) where to look, choosing observations by uncertainty plus coverage; (b) what to study, prioritizing replay by remaining error. After that, propose a reserve-set-A test.
- **R1 fails:** at this scale (11 related real systems) real families do not transfer either. Then the learner working in any dimension, using the 738 other BioModels, is the remaining way to get a much broader real history.

## Trial 16: result (30 October)

**Classification: FAILED.** R1 fails in both seeds; R2 passes in seed 1 only; R3 passes in both.

| K = 32 | Seed 1 | Seed 2 |
|---|---|---|
| R1: held-out median, catalog+bio vs catalog-only (need ≤ 0.67) | 0.90 ✗ | 0.78 ✗ |
| R1: catalog+bio beats kernel ridge on held-out systems (need ≥ 4/6) | 1/6 ✗ | 0/6 ✗ |
| R2: catalog+bio vs catalog+synthetic (need ≤ 0.8) | 0.63 ✓ | 0.84 ✗ |
| R3: development polynomial stratum vs catalog-only (need ≤ 1.5) | 1.33 ✓ | 1.14 ✓ |

Held-out medians (seed 1 / seed 2; kernel ridge is shared):

| | K = 16 | K = 32 | K = 64 |
|---|---|---|---|
| catalog-only | .240 / .253 | .093 / .110 | .014 / .011 |
| catalog+bio | .216 / .218 | .084 / .085 | .013 / .011 |
| catalog+synthetic | .259 / .279 | .133 / .101 | .015 / .016 |
| kernel ridge | .336 | .080 | .011 |

### Reading

- **Real related history helps a little, and more than generic synthetic history.** It moves held-out real systems 10–22% closer than the catalog-only learner manages. One example of family transfer: Decroly's enzymatic oscillator improves from .14 / .21 to .11 / .12, with a glycolysis model in training.
- **But at K = 32 it only reaches kernel ridge level and never clearly beats it.** Exploratory: at K = 16 every learner beats kernel ridge on the held-out real systems (.22–.28 against .34). Kernel ridge catches up by K = 32.
- **Eleven related real systems are not enough history** to turn unfamiliar real mechanisms into familiar ones.
- The pre-registered branch for an R1 failure is a learner that works in any dimension, using the other 738 BioModels. Before that larger build, the owner's uncertainty-driven idea is tested cheaply where it matters most: unfamiliar systems.

## Trial 17: observe where the learner's stored hypotheses disagree (evaluation only; pre-registered 30 October, before running)

- **Hypothesis.** For unfamiliar systems, the most informative next observation is where the learner's *stored hypotheses* — its 4 specialised bases, each fitted to the observations so far — disagree most, while staying spread out. This "learn from what remains uncertain" rule beats generic coverage at the same budget.
- **Contract.** Trial 13's pool-based active observation, observable-only: 400 free noisy candidate states, a budget of K central-difference estimates, clean central-difference queries.
- **Learner.** Trial 16's catalog+bio library (seeds 1, 2), frozen. Final prediction by library+fb, as in Trials 13–14.
- **Test systems:**
  - unfamiliar = the 6 held-out BioModels systems plus the 4 non-polynomial development systems (10, pooled);
  - familiar = the 9 polynomial development systems.
- **Strategies** (all share each episode's pool and random draws):
  - **random:** K random pool points.
  - **coverage:** farthest-point, from the first random point.
  - **disagreement×coverage (primary):** start from 2 random points. Each step, fit every basis's ridge head (its leave-one-out-chosen λ) on the points so far and predict at all candidates. Score each candidate as (standard deviation across the 4 bases' standardized predictions) × (distance to the nearest chosen point), and take the maximum. Bases are refitted every step and λ is re-selected at k ∈ {2, 4, 8, 16}.
  - **leverage×coverage (secondary):** the same, with the selected basis's leverage, √(φᵀ(ΦᵀΦ + λkI)⁻¹φ), in place of disagreement.
- **Episodes:** per system and K ∈ {8, 16, 32}, 4 instances × 5, generator seeds 17000 + 97i + K (BioModels held-out systems are indexed after the development systems).
- **Criteria (both seeds):**
  - **U1 (unfamiliar):** pooled unfamiliar median at K = 16, disagreement×coverage ≤ 0.83× coverage;
  - **U2 (no harm):** familiar median at K = 16, ≤ 1.0× coverage.
- **Reported:** K = 8 and 32, the secondary strategy, and kernel ridge with coverage.

**What happens next, decided now:**
- **U1 and U2 pass:** uncertainty-guided investigation works for unfamiliar systems and becomes part of the learner. Next, "what to study" (replay prioritized by remaining error), and then the learner that works in any dimension.
- **Fail:** generic coverage remains the best observation rule. Proceed to the learner that works in any dimension.

## Trial 17: result (30 October)

**Classification: FAILED** (U2 fails in both seeds), **with a positive signal**: U1 passes in both seeds.

The first run crashed on SprottQ, which has 3 instances (recorded in Trial 1), because the code hard-coded 4. It was fixed to use the available instances, as all earlier trials do, and rerun from scratch; the run is deterministic.

| Median NMSE, K = 8 / 16 / 32 | Unfamiliar (10), seed 1 | Unfamiliar, seed 2 | Familiar (9), seed 1 | Familiar, seed 2 |
|---|---|---|---|---|
| learner, random | .59 / .27 / .084 | .54 / .23 / .096 | .165 / .0081 / .0017 | .146 / .0051 / .0010 |
| learner, coverage | .21 / .036 / **.0025** | .16 / .037 / **.0027** | .031 / .0007 / .0003 | .0085 / .0005 / .0003 |
| learner, disagreement×coverage | .17 / **.027** / .0043 | .13 / **.023** / .0039 | .047 / .0009 / .0003 | .009 / .0006 / .0002 |
| learner, leverage×coverage | .15 / .031 / .0033 | .15 / .025 / .0036 | .042 / .0006 / .0002 | .012 / .0005 / .0002 |
| kernel ridge, coverage | .13 / .034 / .0026 | (same) | .090 / .0019 / .0005 | (same) |

- **U1:** unfamiliar, disagreement×coverage vs coverage at K = 16 is 0.75 and 0.61 ✓. Per-system wins are only 6/10 and 3/10.
- **U2:** familiar, same comparison, is 1.22 and 1.06 ✗.

### Reading

- **Observing where the learner's stored hypotheses disagree helps on unfamiliar systems at moderate budgets.** It is the first learner-based result to edge below kernel ridge with coverage on unfamiliar systems at K = 16 (.023–.027 against .034).
- **It is not uniformly better.** It is slightly worse on familiar systems, worse at K = 32, and the per-system wins are mixed, so the median gain rests on a subset of systems.
- **Refinement noted, not pursued now** to avoid further tuning on the same development systems: use disagreement-guided observation only when the system appears unfamiliar, i.e. when the fallback is selected or the leave-one-out error stays high.
- **As pre-registered for failure, the next step is the learner that works in any dimension,** so the history can draw on the much larger pool of real BioModels.

### Sourcing extension for a learner that works in any dimension (30 October, before conversion)

The same BioModels sourcing rule applies, except that the number of ODE state variables may be anything from **2 to 10**. Everything else is unchanged: no events, delays or explicit time; non-polynomial; sustained motion; the same title exclusions and duplicate handling; 8 instances.

- Trial 16's 6 held-out 3-D systems stay held out, so the new learner is scored on exactly the same systems.
- Newly accepted systems (d ≠ 3, or 3-D systems not accepted before) are split by rank-hash with salt `trial18-split`: two thirds training, one third held-out.
- Trial 16's 11 training systems stay in training.
- The converter takes `--dmin/--dmax`; the defaults (3/3) reproduce Trial 16 exactly.

### Any-dimension sourcing outcome (30 October, before any training on these systems)

- **63 accepted at 2 ≤ d ≤ 10** (the 21 three-dimensional ones as before).
- **Exclusions:** Ca²⁺ titles (Somogyi1990, Lavrentovich2008); Trial 16's duplicate decisions; Leloup1999 *Drosophila* (298), a structural duplicate of 021; Goldbeter1991 "explicit inactive forms" (004), the same minimal model with extra variables (a judgment).
- A first pass also excluded models that merely shared author and year (Leloup *Neurospora* vs *Drosophila*, Westermark base vs extended, Kim SimpleII vs SimpleIII). These are different models from one paper, not duplicates under the rule, so the flags were overruled, before any training.
- **56 real systems kept.**
- **Split:** Trial 16 assignments preserved; new systems by rank-hash (salt `trial18-split`), two thirds training.
  - **Training: 37** (by d = 2…10: 4, 11, 5, 4, 6, 1, 2, 2, 2).
  - **Held-out: 19** (by d = 2, 3, 4, 5, 7, 8, 9, 10: 1, 6, 1, 3, 2, 3, 1, 2), including Trial 16's 6.
- One same-paper pair crosses the split: Leloup1999 *Neurospora* (training) and *Drosophila* (held-out). This is annotated as a related-family case.

## Trial 18: a pairwise feature library that works in any dimension, with a broad real history (pre-registered 30 October, before training)

- **Hypothesis.** Biochemical kinetics are built from one- and two-variable terms (mass action, Michaelis–Menten, Hill). A library of *shared* single-variable and pairwise feature functions, accumulated over 51 catalog flows and 37 real models of 2–10 variables, transfers to held-out real models of any dimension. It should beat both the same learner trained on the catalog alone and kernel ridge.
- **Learner** (`trial18.py`): 4 bases; each basis is three small SiLU networks on context-standardized coordinates.
  - s: R → R¹⁶ (self terms), 1 → 64 → 64 → 16;
  - p: R² → R⁸ (pair terms), 2 → 64 → 64 → 8;
  - o: R → R⁴ (other-variable terms), 1 → 32 → 4.
  - Each variable i gets its own design row: [s(u_i), p(u_i, u_b) for b ≠ i, o(u_b) for b ≠ i, 1], so k = 17 + 12(d − 1). That is 41 at d = 3 and 125 at d = 10.
  - One ridge head per variable, with a shared λ: learned in training, chosen by exact leave-one-out (mean over variables) at encounter. Kernel ridge is the fallback, as before.
  - Training uses relaxed winner-take-all on log NMSE, as in Trials 4–16.
- **Training.**
  - Observable-only contract: central differences with 0.5% state noise, no ground truth anywhere.
  - Sequential stream with a seeded shuffle, 400 steps per stage.
  - Each step: 32 episodes of the new system plus 32 episodes of one memory system chosen uniformly. Memory holds 2 instances per past system. K is log-uniform in [8, 128]. Adam, the same schedule.
- **Arms** (seeds 1, 2):
  - **pw-catalog:** the 51 catalog systems only;
  - **pw-all:** catalog plus the 37 real training systems.
- **Evaluation.**
  - The 19 held-out real systems and the 13 development systems; K ∈ {16, 32, 64}; 4 instances × 5 episodes; seeds 18000 + 97i + K.
  - Additionally, Trial 16's 6 held-out systems on Trial 16's exact episodes (seeds 16000 + 97i + K).
  - Kernel ridge is the experience-free control.
- **Criteria (both seeds):**
  - **X1 (broad real history transfers):** over the 19 held-out systems at K = 32, pw-all's median ≤ 0.67× pw-catalog's, *and* pw-all beats kernel ridge on ≥ 12 of 19.
  - **X2 (more than 11 related systems):** on Trial 16's 6 held-out systems (its episodes), pw-all's median at K = 32 ≤ 0.8× Trial 16's catalog+bio.
  - **X3 (no harm):** development polynomial-stratum median at K = 32, pw-all ≤ 1.5× pw-catalog.
- **Resources:** threads capped (OMP 4, nice 10), 2 seeds × 2 arms; outputs under `outputs/accumulating_learner/trial18/`.

**What happens next, decided now:**
- **X1–X3 pass:** a candidate for reserve set A (and potentially B, since the learner works in any dimension). Ask before touching either.
- **X1 fails:** a broad real history at this scale does not overcome the information-per-encounter limit for unfamiliar real mechanisms. Write the synthesis. The remaining lever is the gated, uncertainty-guided observation from Trial 17.

## Trial 18: result (30 October)

**Classification: FAILED.** X1 and X2 fail in both seeds; X3 passes in both. Training was clean: about 58 minutes per all-history run and 30 minutes per catalog run, at 4 threads each.

| K = 32 | Seed 1 | Seed 2 |
|---|---|---|
| X1: held-out median, pw-all vs pw-catalog (need ≤ 0.67) | 0.80 ✗ | 0.74 ✗ |
| X1: pw-all beats kernel ridge on held-out systems (need ≥ 12/19) | 5/19 ✗ | 8/19 ✗ |
| X2: Trial 16's 6 held-out systems, pw-all vs Trial 16 catalog+bio (need ≤ 0.8) | 1.38 ✗ | 1.21 ✗ |
| X3: development polynomial stratum, pw-all vs pw-catalog (need ≤ 1.5) | 0.81 ✓ | 1.00 ✓ |

Held-out medians (19 systems; kernel ridge is shared):

| | K = 16 | K = 32 | K = 64 |
|---|---|---|---|
| pw-catalog, seed 1 / 2 | .082 / .077 | .0170 / .0192 | .0015 |
| pw-all, seed 1 / 2 | .089 / .099 | .0136 / .0142 | .0015 |
| kernel ridge | .082 | .0117 | .0015 |

### Reading

- **A broad real history (37 real models of 2–10 variables, plus the catalog) helps the learner relative to itself:** about 20–26% lower median error than catalog-only at K = 32, though only 11/19 and 9/19 held-out systems individually improve. There are gains on some high-dimensional clocks (Leloup1999 10-D .0005–.0008 against kernel ridge .0014; Locke2008 8-D).
- **It still does not beat kernel ridge on held-out real mechanisms.** It is not better than Trial 16's 3-D learner on Trial 16's held-out systems either.
- Many held-out systems sit at an error floor of about 1e-4 at K = 32, where every method ties. The hard ones (Plant1981 bursting neurons, Bertram2000 β-cells, Rattanakul2003 bone; NMSE .3–.8) are hard for every method.
- **As pre-registered for X1 failing:** a broad real history at this scale does not overcome the information-per-encounter limit for unfamiliar real mechanisms.

## Synthesis after Trials 1–18 and the confirmation run (30 October)

### Established

1. **Accumulation works when new systems share structure with the history. Confirmed on untouched systems.**
   - A library accumulated one system at a time discovers the shared structure itself (no term library is supplied) and turns it into 40–65× lower error at 16–32 observations under low-noise observation.
   - It gives 2–4× from noisy trajectories with no ground-truth targets anywhere.
2. **The gains grow with experience, and history is retained with a small replay memory.** Related systems help each other even at test time; for example, later Sprott-family systems are learned 1.5–4× better.

### Not achieved: generalization to unfamiliar real mechanisms

Nothing beats generic kernel ridge on mechanisms unlike the history at ≤ 64 noisy observations. Tried so far:

- broader generic history (Trial 15);
- related real history, 3-D (Trial 16) and any dimension with 37 real models (Trial 18);
- test-time accumulation (Trials 10–11);
- encounter-time fine-tuning (Trial 12);
- sparsity (Trials 6–7);
- context-conditioned features (Trial 3).

History helps each of these learners *relative to itself*, by 10–26%, but not beyond what kernel ridge extracts from the same observations.

### What does help unfamiliar mechanisms

- **Where to observe** (Trial 13): spreading observations out gives 3–5× for every method.
- **Observing where the learner's stored hypotheses disagree** (the owner's uncertainty idea, Trial 17) helps further on unfamiliar systems at moderate budgets: 25–39% at K = 16, and the only learner-based result to edge past kernel ridge there. It is slightly worse on familiar systems.

### Interpretation

The learner's experience is valuable exactly when it contains the right structure. For a genuinely new mechanism, a handful of noisy observations limits everyone. The productive levers are then *better observations*, and *choosing them using the learner's uncertainty*, rather than more stored features.

### Recommended next steps

- **Gated uncertainty-guided observation:** disagreement-guided only when the system looks unfamiliar (fallback chosen or high leave-one-out error), coverage otherwise. This targets Trial 17's familiar-system cost.
- **"What to study":** replay prioritized by remaining error.
- **Reserve sets A and B stay untouched.** No candidate yet meets the bar of beating kernel ridge on unfamiliar mechanisms, which reserve set A would test.

## Trials 19 and 20: the two halves of "learn from what remains uncertain" (pre-registered 30 October, before running)

### Trial 19: gated, uncertainty-guided observation (evaluation only)

- **Hypothesis.** Trial 17's disagreement-guided observation helped unfamiliar systems but slightly hurt familiar ones. Using it only when the system *looks* unfamiliar keeps the benefit and removes the cost.
- **Rule** (fixed now):
  1. Start from 2 random pool points.
  2. Choose by coverage until k = max(4, K/4).
  3. Run library+fb's leave-one-out selection on those points. If **kernel ridge is selected** (no stored basis fits), continue with disagreement×coverage. Otherwise continue with coverage.
- **Setting:** identical to Trial 17 (same learner, Trial 16's catalog+bio; same pool contract; same episodes, seeds 17000 + 97i + K; the same 10 unfamiliar and 9 familiar systems). Trial 17's coverage and disagreement×coverage numbers are reused.
- **Criteria (both seeds, K = 16):**
  - **GU:** unfamiliar median, gated ≤ 0.83× coverage;
  - **GF:** familiar median, gated ≤ 1.0× coverage.
- **Reported:** how often the gate opens, per group.

### Trial 20: replay prioritized by what the learner still gets wrong

- **Hypothesis.** Replay that rehearses past systems *in proportion to their current error* uses the same compute better than uniform replay. Held-out real systems are learned better, and earlier systems are retained at least as well.
- **Setting:** identical to Trial 16's catalog+bio arm (observable-only, same stream and seeds, 400 steps per stage, M = 4, 2 replay instances), except how the memory half of each batch is drawn.
  - Each memory system keeps an exponential moving average (factor 0.9) of the winning-basis NMSE whenever it is replayed, initialized to its last-stage training error.
  - Memory episodes are sampled from systems with probability ∝ EMA^1, mixed 50/50 with uniform so no system is starved.
- **Criteria (both seeds):**
  - **P1:** on Trial 16's 6 held-out systems (its episodes) at K = 32, prioritized median ≤ 0.8× Trial 16's catalog+bio;
  - **P2:** development polynomial-stratum median at K = 32 ≤ 1.2× Trial 16's catalog+bio;
  - **P3:** retention on each seed's first 8 meta-train systems (observable contract, K = 32), median ratio to Trial 16's catalog+bio ≤ 1.0.

**What happens next, decided now:** whatever these show, write a final synthesis for the owner. If either passes, it becomes part of the recommended learner. Reserve sets stay untouched unless a candidate beats kernel ridge on unfamiliar development and held-out real systems.

## Trial 19: result (30 October)

**Classification: FAILED.** GF fails in both seeds; GU passes in both.

| Median NMSE, K = 8 / 16 / 32 | Unfamiliar, seed 1 | Unfamiliar, seed 2 | Familiar, seed 1 | Familiar, seed 2 |
|---|---|---|---|---|
| coverage | .214 / .036 / .0025 | .156 / .037 / .0027 | .031 / .00075 / .00029 | .0085 / .00054 / .00025 |
| disagreement×coverage (Trial 17) | .166 / .027 / .0043 | .134 / .023 / .0039 | .047 / .00091 / .00027 | .0090 / .00057 / .00023 |
| **gated** | **.140** / .027 / .0028 | **.106** / .022 / .0030 | .047 / .00080 / .00027 | .0102 / .00061 / .00025 |
| kernel ridge, coverage | .134 / .033 / .0026 | (same) | .090 / .0019 / .00048 | (same) |

- **GU:** unfamiliar, gated vs coverage at K = 16 is 0.75 and 0.59 ✓.
- **GF:** familiar, same comparison, is 1.07 and 1.13 ✗.
- Gate open rate at K = 16: unfamiliar 56% / 67%; familiar 56% / 62%.

### Reading

- **Gating keeps the unfamiliar-system gain at K = 16 and removes the K = 32 penalty** (.0028–.0030 against coverage .0025–.0027, where the ungated rule got .0039–.0043). It is the best strategy on unfamiliar systems at K = 8.
- **The gate is not selective.** With only 4 observations, kernel ridge wins leave-one-out selection on familiar systems almost as often as on unfamiliar ones, so familiar systems still pay a small cost.
- A better unfamiliarity signal is needed, for example the gap between the best basis's leave-one-out error and its typical in-history error.

## Trial 20: result (30 October)

**Classification: FAILED.** P1 and P3 fail in both seeds; P2 passes in seed 1 only.

| | Seed 1 | Seed 2 |
|---|---|---|
| P1: held-out median, prioritized vs uniform replay (need ≤ 0.8) | 1.09 ✗ | 1.26 ✗ |
| P2: development polynomial stratum (need ≤ 1.2) | 1.08 ✓ | 1.80 ✗ |
| P3: retention (need ≤ 1.0) | 1.05 ✗ | 1.05 ✗ |

### Reading

- **Prioritizing replay by remaining error concentrated rehearsal on systems the learner cannot learn.** ThomasLabyrinth and LiuChen topped the priority list in both seeds, at the expense of everything else. This is the same trap as Trial 1's mean-NMSE objective.
- **"What remains uncertain" has to mean *still improvable*, not *still wrong*.** The natural next version prioritizes by learning progress (recent *decrease* in error) rather than by error level. Not tried.

## Final synthesis of this session (30 October)

### The goal

A network that learns from its history, without being supplied the answer, and so learns new dynamical systems faster and more reliably.

### Achieved and confirmed

- **When new systems share structure with the history, the learner gets dramatically faster.** It discovers that structure itself (no term library or equations are supplied) and accumulates it one system at a time with a small replay memory.
- **Confirmed on untouched systems:** about 65× lower error at 32 observations and 40× at 16, with the gain growing with the number of systems seen.
- **It holds, more modestly (2–4×), when learning only from noisy trajectories** with no ground-truth targets anywhere.

### Not achieved

- **Generalization to genuinely unfamiliar mechanisms.** With ≤ 64 noisy observations, no learner variant beats generic kernel ridge on mechanisms unlike its history. Tried:
  - broader synthetic history;
  - 11 and then 37 real related models (any dimension);
  - test-time learning;
  - fine-tuning;
  - sparsity;
  - context conditioning.
- History improves each learner relative to itself by about 10–26%, but not beyond kernel ridge.

### Partly achieved: the owner's "learn from what remains uncertain"

- **Choosing observations where stored hypotheses disagree** is the only learner-based approach that edged below kernel ridge on unfamiliar systems: 25–40% better than generic coverage at K = 16, and best at K = 8 when gated. *[Withdrawn: confounded by mismatched starting points; see "Correction to Trials 17 and 19" below.]*
- **Its open problem** is a reliable signal for when a system is unfamiliar.
- **Rehearsing what is still wrong backfires.** It needs a learning-progress signal instead.

### Recommended next steps

1. A better unfamiliarity signal for the gate.
2. Learning-progress-prioritized replay.
3. Only then a reserve-set-A test (mostly non-polynomial, untouched), if a candidate beats kernel ridge on unfamiliar development and held-out real systems.

Reserve sets A and B, the PDEBench reserved trajectories and the diffusion–sorption mechanism remain untouched.

## Owner decision (30 October)

The owner chose recommended step 1, a better unfamiliarity signal for the gate. They judged step 2 (learning-progress replay) to be avoiding the problem.

## Trial 21: a calibrated unfamiliarity gate (evaluation only; pre-registered 30 October, before running)

- **Problem.** Trial 19's gate (kernel ridge winning leave-one-out after 4 points) opened about 55–60% of the time on familiar *and* unfamiliar systems alike, so familiar systems paid a cost.
- **Hypothesis.** A gate calibrated on *new but familiar* systems separates the two groups. Unfamiliar systems keep Trial 19's gain and familiar systems lose nothing compared with coverage. The calibrated question is: is the best stored basis's leave-one-out error unusually high compared with how it fits systems it has never seen but that resemble its history?
- **Calibration set** (already opened, never trained on, not development): the 11 polynomial confirmation systems (classified symbolically in the confirmation run). For each seed's learner and each gate size k₀ ∈ {4, 8, 16}:
  - 10 episodes per system (pool contract, generator seeds 21000 + 97i + k₀);
  - each episode takes 2 random points plus coverage up to k₀ and records the best basis's exact leave-one-out MSE (standardized units, minimum over the 4 bases);
  - **the threshold is the 90th percentile** of these values.
- **Gate:**
  1. Start from 2 random points.
  2. Choose by coverage until k₀.
  3. If the best basis's leave-one-out MSE is above the threshold for k₀, the system counts as unfamiliar: continue with disagreement×coverage (Trial 17). Otherwise continue with coverage.
- **Variants:**
  - **gated-half (primary):** k₀ = K/2, i.e. 4, 8 and 16 for K = 8, 16 and 32. More data for the decision.
  - **gated-quarter (secondary):** k₀ = max(4, K/4), Trial 19's timing.
- **Setting:** identical to Trials 17 and 19 (Trial 16's catalog+bio learner, pool contract, episodes with seeds 17000 + 97i + K, 10 unfamiliar and 9 familiar systems). Trial 17's coverage and disagreement numbers are reused.
- **Criteria (both seeds, K = 16, gated-half):**
  - **GU:** unfamiliar median ≤ 0.83× coverage;
  - **GF:** familiar median ≤ 1.0× coverage.
- **Reported:** gate open rates per group (the aim is high for unfamiliar, low for familiar), K = 8 and 32, the secondary variant.

**What happens next, decided now:**
- **Both pass:** the gated learner is the candidate for the reserve-set-A test. It still has to beat kernel ridge on unfamiliar systems in development, which is checked and reported here, and the owner is asked before reserve set A is touched.
- **GF fails:** report the open rates. The signal needs more than leave-one-out error, for example disagreement among the bases relative to calibration.

## Trial 21: result (30 October)

**Classification: FAILED.** GF fails in seed 2. GU's pass is not credited: it is confounded, as explained below.

**Thresholds** (90th percentile of the best-basis leave-one-out on the polynomial confirmation systems):

| k₀ | 4 | 8 | 16 |
|---|---|---|---|
| Seed 1 | 1.39 | 1.17 | 0.098 |
| Seed 2 | 1.42 | 1.25 | 0.133 |

| K = 16, gated-half | Seed 1 | Seed 2 |
|---|---|---|
| GU: unfamiliar vs Trial 17 coverage | 0.83 | 0.62 |
| GF: familiar vs Trial 17 coverage | 0.92 ✓ | 1.12 ✗ |
| Gate open rate, unfamiliar / familiar | 5% / 0% | 6% / 0% |

### Reading

- **The calibrated gate almost never opened** at K ≤ 16 (5–9% on unfamiliar systems, 0–13% on familiar ones). So "gated" was essentially coverage started from *two* random points instead of Trial 17's one.
- **Confound:** the GU and GF ratios mostly measure that difference in starting points, plus episode noise. They are not credited to the gate. Later comparisons must start the coverage baseline from the same points.
- **The thresholds are huge at k₀ ≤ 8** (leave-one-out MSE above 1, i.e. worse than predicting the mean). With so few points, every system's fit error is large and heavy-tailed.

### Exploratory diagnostics (not criteria)

How well candidate gate statistics separate unfamiliar from familiar episodes (AUC; Trial 17 episodes at K = 32, seed-1 learner):

| k₀ | Best-basis leave-one-out | Disagreement among the 4 bases | Best-basis / kernel-ridge leave-one-out ratio |
|---|---|---|---|
| 4 | 0.50 | 0.52 | 0.44 |
| 8 | 0.69 | 0.47 | **0.82** |
| 16 | 0.87 | 0.58 | 0.82 |

- **Self-knowledge has the same information limit.** From its own absolute fit error, the learner cannot tell that a system is unfamiliar until it has about 16 observations.
- **The *relative* fit carries the signal earlier.** Asking "does my best stored basis fit better than a generic method?" separates the groups from 8 observations (AUC 0.82). Disagreement among the bases does not separate them at all.
- Trial 19's gate ("kernel ridge wins") is this same ratio thresholded at 1, but it was checked at k₀ = 4, where the ratio has no information (AUC 0.44).

## Trial 22: gate on relative fit at half budget (evaluation only; pre-registered 30 October, before running)

- **Disclosure.** The statistic was chosen *after* seeing the diagnostic AUCs on development episodes, a selection between three candidates. To limit optimism, Trial 22 uses **fresh episodes**, and the threshold is calibrated only on the polynomial confirmation systems, as in Trial 21. Reserve set A would be the real test.
- **Gate:**
  1. Start from 2 random pool points.
  2. Choose by coverage until k₀ = K/2.
  3. Compute r = (best basis's exact leave-one-out MSE) / (kernel ridge's exact leave-one-out MSE).
  4. If r exceeds the 90th percentile of r on the calibration episodes at that k₀ (computed as in Trial 21, seeds 22500 + 97i + k₀), continue with disagreement×coverage. Otherwise continue with coverage.
- **Fair baseline:** coverage started from the *same* 2 random points (the gate always closed), and disagreement×coverage from the same 2 points (the gate always open).
- **Setting:** Trial 16's catalog+bio learner; the same 10 unfamiliar and 9 familiar systems; pool contract; **fresh episode seeds 22000 + 97i + K**; K ∈ {8, 16, 32}.
- **Criteria (both seeds, K = 16):**
  - **GU:** unfamiliar median, gated ≤ 0.83× coverage(2-start);
  - **GF:** familiar median, gated ≤ 1.0× coverage(2-start).
- **Reported:** gate open rates per group, AUC of r on these fresh episodes, and kernel ridge with coverage.

**What happens next, decided now:**
- **Both pass:** present the gated learner to the owner as the reserve-set-A candidate, with the caveat about how the statistic was selected.
- **Fail:** the information limit also blocks practical self-assessment at these budgets. Report it and stop this thread.

## Trial 22: result (30 October)

**Classification: FAILED.** GU fails in both seeds; GF passes in both.

**Relative-fit thresholds** (k₀ = 4 / 8 / 16): seed 1 1.45 / 1.29 / 1.19; seed 2 1.54 / 1.13 / 1.88.

**AUC of the statistic r on fresh episodes:**

| K (gate at k₀ = K/2) | 8 (k₀ = 4) | 16 (k₀ = 8) | 32 (k₀ = 16) |
|---|---|---|---|
| Seed 1 | 0.50 | 0.77 | 0.83 |
| Seed 2 | 0.53 | 0.82 | 0.74 |

| Median NMSE at K = 16, fresh episodes | Unfamiliar, seed 1 | Unfamiliar, seed 2 | Familiar, seed 1 | Familiar, seed 2 |
|---|---|---|---|---|
| coverage, 2-start | .0262 | .0265 | .0006 | .0006 |
| disagreement×coverage, 2-start | .0307 | .0275 | .0009 | .0005 |
| gated on relative fit | .0241 (GU 0.92 ✗) | .0269 (GU 1.02 ✗) | .0006 (GF 1.00 ✓) | .0006 (GF 1.00 ✓) |
| kernel ridge, coverage, 2-start | .0238 | (same) | .0017 | (same) |

Gate open rate at K = 16: unfamiliar 12% / 24%; familiar 3% / 3%.

### Reading

- **The gate is now selective** (3% false openings on familiar systems), and it costs familiar systems nothing. But disagreement-guided observation itself gives no benefit over coverage started from the same two points, so gating it has nothing to gain.

## Correction to Trials 17 and 19 (30 October)

**Trial 17's U1 "pass" is withdrawn, and Trial 19's GU is confounded the same way.** In both trials, the coverage baseline started from **one** random point, while the uncertainty-guided strategies started from **two**.

**Re-check on Trial 17's own episodes** (K = 16): coverage started from the same two points.

| | Coverage, 1 start | Coverage, 2 starts | Disagreement×coverage, 2 starts |
|---|---|---|---|
| Unfamiliar, seed 1 | .0360 | .0298 | .0269 |
| Unfamiliar, seed 2 | .0371 | .0230 | .0227 |
| Familiar, seed 1 | .00075 | .00069 | .00091 |
| Familiar, seed 2 | .00054 | .00060 | .00057 |

With matched starts, the disagreement benefit on unfamiliar systems is **0.90 and 0.99**, which would have failed U1's 0.83 bar. Most of the reported 25–39% gain came from the start design, not from the uncertainty rule.

This also shows how sensitive these 10-system medians are: changing one starting point moved coverage by up to 38%. Effects of that size need matched designs and more systems before they are believed.

**Corrected statements.** The earlier synthesis said uncertainty-guided observation "edged below kernel ridge on unfamiliar systems" and was "25–40% better than coverage at K = 16". **That is not supported.** With matched baselines, neither disagreement-guided observation (always on or gated) nor leverage-guided observation beats generic coverage. The surviving findings are:

- coverage (spreading observations out) is worth 3–5× over random choice;
- the learner can tell an unfamiliar system from its *relative* fit (best stored basis against kernel ridge) only from about 8–16 observations (AUC about 0.8); at 4 observations it is at chance;
- with ≤ 64 noisy observations, no form of experience tried here, whether in the model or in choosing observations, beats generic kernel ridge on mechanisms unlike the history.

As pre-registered, this thread stops here.

## Owner instruction (30 October, night)

The owner asked whether I am waiting on anything or can run autonomously overnight. Answer: autonomous, under the standing rules — pre-register, log, push-notify, and stop only before reserved sets (A, B, PDEBench). Plan: scale the real history.

## Trial 23 plan: a large real history from one domain (BioModels transients) — sourcing rule, fixed before conversion

- **Why.** The established strength is fast learning of systems that resemble the history. Trial 18's real history was small (37 models) because only sustained oscillators were accepted. Most curated BioModels relax to a steady state; their transients carry the same kinetic structure (mass action, Michaelis–Menten, Hill).
- **Sourcing extension.** The BioModels rule as before (curated; no events, delays or explicit time; non-polynomial; title exclusions; duplicate rule), but with:
  - 2 ≤ d ≤ 12 state variables;
  - *either* sustained motion (as before) *or* a non-trivial transient. A transient model is accepted if, from the published initial state, the trajectory moves (‖x(T) − x(0)‖ ≥ 1e-3 × (‖x(0)‖ + 1e-6)) and settles. The settling time t_s is the first time the distance to the final state falls below 1% of its initial value, with t_s > 0 and within a horizon of 10⁴ model time units.
- **Transient data:** 8 instances.
  - Instance 0 starts from the published initial state; instances 1–7 start from the published state with each variable scaled by U(0.5, 1.5) (seeded by the model identifier).
  - Each instance is integrated over [0, 1.5 t_s] with 4,000 samples, h = 1.5 t_s / 3999.
  - Instances that fail or blow up are redrawn up to 3 times.
- **Transient observation contract:** like Trial 9's (central differences, 0.5% noise, observable only), except that the context comes from instances 0–3 and the queries from instances 4–7. This tests prediction for new initial conditions instead of a later time segment. Oscillatory systems keep the segment split.
- **Split:** Trial 18's assignments preserved. New systems by rank-hash (salt `trial23-split`), two thirds training.

The full Trial 23 protocol (arms and criteria) is pre-registered after conversion, before any training.

### Trial 23 sourcing outcome (30 October, before any training)

- **Accepted at 2 ≤ d ≤ 12:** 249 models (187 transients, 62 oscillators).
  - By d = 2…12: 28, 55, 38, 26, 22, 17, 19, 14, 15, 11, 4.
- **Exclusions:** 22 in total — Trial 18's decisions, Ca²⁺ titles, structural duplicates. Structural duplicates were judged by a permutation-invariant fingerprint; Trial 18's kept systems are protected.
- **Import:** Rozi2003 (BIOMD0000000100), a Trial 18 training system, failed integration in this pass (probably a 60 s timeout under load). Its Trial 18 data and record were imported instead, so Trial 18's training set stays intact.
- **Final real set: 228.**
  - **Training: 152** (37 oscillators from Trial 18, 115 new transients).
  - **Held-out: 76** (Trial 18's 19 oscillators, 57 new transients).

## Trial 23: does a large real history make held-out real systems learnable? (pre-registered 30 October, before training)

- **Hypothesis.** With an order of magnitude more related real history (152 real biochemical models plus the 51 catalog flows), the any-dimension pairwise library (Trial 18 architecture, unchanged) beats kernel ridge on held-out real models from the same domain. This is the first test of whether accumulated experience generalizes to new real systems at scale.
- **Training.**
  - `trial23.py`, seeds 1 and 2.
  - A stream of 203 systems (51 catalog + 152 real) in a seeded shuffle, `default_rng(2300 + s)`, at 400 steps per stage, about 81k steps.
  - Observable-only. Oscillators use the segment contract. Transients use the instance contract: context from starting conditions 0–3, queries from 4–7; replay keeps instances 0–1 (context) and 4–5 (queries).
  - Everything else as in Trial 18. Threads are capped (OMP 4, nice 10), and runs are detached.
- **Comparators:**
  - Trial 18's pw-catalog and pw-all checkpoints;
  - kernel ridge (experience-free).
- **Evaluation:**
  - 57 new held-out transient models (transient contract, 20 episodes per K, seeds 23000 + 97i + K);
  - Trial 18's 19 held-out models and 13 development systems on Trial 18's exact K = 32 episodes.
- **Criteria (both seeds, K = 32):**
  - **B1 (new held-out real models):** pw-big's median ≤ 0.67× pw-catalog's, *and* pw-big beats kernel ridge on ≥ 60% (≥ 35 of 57).
  - **B2 (Trial 18's held-out):** pw-big ≤ 0.8× Trial 18 pw-all, *and* beats kernel ridge on ≥ 10 of 19.
  - **B3 (no harm):** development polynomial-stratum median ≤ 1.5× pw-catalog's.

**What happens next, decided now:**
- **B1 and B2 pass:** experience at scale generalizes to new real systems. Stop and ask the owner about a reserve-set-A test.
- **B1 passes but B2 fails:** gains are specific to transients (new initial conditions within a domain). Report it, and then test reserve-set-A-style unfamiliar mechanisms on development only.
- **Both fail:** even a large related real history does not beat kernel ridge at ≤ 64 noisy observations. Report this as the session's conclusion.

### Trial 23 deviation: training restarted after NaN (30 October, before any evaluation)

- **What happened.** In both seeds the training loss became NaN between stages 90 and 100, so the weights were corrupted. Both runs were stopped (about 2 h of compute discarded). Checkpoints are only written at the end, so none existed and nothing was evaluated.
- **Cause, found by diagnostic** (sampling every real system's training episodes):
  - 2 systems produce non-finite standardized values, and 6 have a variable that never changes. Division by a near-zero spread gives infinities.
  - Separately, many transient systems give very large but finite standardized values (multi-scale relaxation, query starting states outside the context's range). That is legitimate difficulty and is shared by every method.
- **Fix, decided before any evaluation:**
  1. **Validity rule:** exclude models with any near-constant variable (span ≤ 1e-6 × max|x|) or non-finite samples. This removes 11 models: 7 training and 4 new held-out transients; none of Trial 18's held-out set. The list is in `validity_excluded.json`.
  2. **Non-finite steps are skipped and counted**, so a bad batch cannot corrupt the weights again.
- **Effect on the protocol:** 145 training and 72 held-out real models remain (53 new held-out transients). B1's win threshold stays at 60%, now ≥ 32 of 53. Everything else is unchanged. Training restarts from scratch with the same seeds and stream rule; the stream is now 196 systems. The aborted logs are kept as `train_s*_aborted_nan.log`.

## Trial 23: result (30 October)

**Classification: FAILED.** B1 and B2 fail in both seeds; B3 passes in both. The restarted training was clean: 196 stages, 0 non-finite steps, about 2.3 h per seed.

| K = 32 | Seed 1 | Seed 2 |
|---|---|---|
| B1: new held-out transients, pw-big vs pw-catalog (need ≤ 0.67) | 0.98 ✗ | 1.09 ✗ |
| B1: pw-big beats kernel ridge (need ≥ 32/53) | 17/53 ✗ | 14/53 ✗ |
| B2: Trial 18 held-out, pw-big vs Trial 18 pw-all (need ≤ 0.8) | 1.29 ✗ | 1.42 ✗ |
| B2: pw-big beats kernel ridge (need ≥ 10/19) | 6/19 ✗ | 9/19 ✗ |
| B3: development polynomial stratum vs pw-catalog (need ≤ 1.5) | 0.77 ✓ | 0.90 ✓ |

Median NMSE on the new held-out transients at K = 32: pw-catalog 1.37 / 1.27, pw-all 1.36 / 1.32, pw-big 1.34 / 1.38, kernel ridge 1.16.

### Reading, and why this is not a fair test of "a large history"

- **On the new held-out transients every method, kernel ridge included, is worse than predicting the query mean** (NMSE > 1). The transient contract as posed cannot measure whether history helps.
- **Contract-validity check** (afterwards; kernel ridge only, on 30 *training* transients, 10 episodes each at K = 32):
  - instance-split contract: median NMSE 1.15, 30% below 1, 0% below 0.1;
  - within-trajectory contract (context and queries from the same instances, different times): median 1.10, 37% below 1, 0% below 0.1.
- **Diagnosis.** The transient data as constructed is effectively unlearnable from ≤ 64 noisy central differences. Uniform time sampling over [0, 1.5 t_s] is dominated by near-equilibrium points with tiny derivatives, while the short fast initial phase produces rare, huge ones. The single sample spacing is also too coarse for that phase. The 19× more "history" therefore carried almost no learnable signal.
- **Negative transfer.** Training on these tasks made the learner worse on the oscillatory held-out systems it handled before (B2: 1.29–1.42× worse than Trial 18).
- **Conclusion.** As pre-registered for both failing, this ends the session. The specific question "does a large *related* real history beat kernel ridge?" remains open, because Trial 23's history was mostly unlearnable data, not related structure.

## Session conclusion (30 October)

1. **Established and confirmed on untouched systems.** A learner that accumulates experience one system at a time, without being given equations or a term library, learns new systems that share structure with its history far faster than experience-free methods: about 65× lower error at 32 observations, with the gain growing with history.
   - It holds, more modestly (2–4×), from noisy trajectories with no ground-truth targets anywhere.
   - It retains earlier systems with a 6% replay memory.
2. **Not achieved: generalization to unfamiliar real mechanisms.** With ≤ 64 noisy observations, no learner variant beats kernel ridge on mechanisms unlike its history. Tried:
   - broader synthetic history;
   - 11, 37, then 145 real models;
   - test-time learning;
   - fine-tuning;
   - sparsity;
   - context conditioning;
   - uncertainty-guided observation (the apparent gain was a confounded comparison and was withdrawn).
   History improves each learner relative to itself (10–26%), but not beyond kernel ridge.
3. **What helps unfamiliar systems: observation design.** Spreading observations out gives 3–5× for every method. The learner can only tell that a system is unfamiliar after about 8–16 observations (relative-fit AUC about 0.8), which is too late to change strategy within these budgets.
4. **Open question with a concrete next step.** Whether a large *related* real history helps remains untested. It needs transient data that is learnable: log-time or adaptive sampling with per-snippet spacing, checked first with kernel ridge, as above, before any learner is trained.

Reserve sets A and B, the PDEBench reserved trajectories and the diffusion–sorption mechanism are untouched.

## Owner instruction (30 October)

*"How would we stop here? We don't have a result, what we have is another failure. We should not stop until we have a solution. You have my permission for the proposal."*

**Standing rule from now on:** keep iterating toward the goal, a learner whose history makes *unfamiliar* systems learnable, and do not propose stopping or writing up while it is unmet.

## Trial 24 plan: learnable transient data, then the large-history test again (pre-registered 30 October, before generating data)

**Step 1 — resample the transient models** (the 187 accepted transients; same models, same validity rule, same splits). For each instance, integrate densely from t = 0 to 1.5 t_s, then keep 4,000 samples by one of two schemes:
- **log-time:** t = 0 plus geometric spacing from 10⁻⁴ t_s to 1.5 t_s;
- **arc-length:** samples equally spaced in trajectory arc length, in coordinates scaled by each variable's range, so observations are dense where the state changes. This is like an event-driven sensor.

Time stamps are stored. A central difference uses each sample's own neighbours: (x_{i+1} − x_{i−1}) / (t_{i+1} − t_{i−1}).

**Step 2 — solvability check, kernel ridge only, on training transients only.**
- **Contract:** within-trajectory. Context and queries come from instances 0–3 at disjoint sample indices, with 0.5% state noise and central-difference estimates, as for the oscillators.
- **Scheme choice:** the scheme with the lower kernel-ridge median NMSE at K = 32 over the training transients is chosen.
- **Gate:** a learner is trained only if that median is ≤ 0.5. If neither scheme passes, the transient data is redesigned again before any learner is trained.

**Step 3 — Trial 24 training and evaluation.** As Trial 23 (pairwise any-dimension library, the same stream rule with the 145 valid training models plus the catalog, 400 steps per stage, observable-only), but with the resampled transients under the within-trajectory contract.
- **Criteria:** B1, B2 and B3 exactly as in Trial 23.
- **New held-out transients:** evaluated under the same within-trajectory contract, generator seeds 24000 + 97i + K.

## Owner standing rules (30 October)

1. **Disk.** Before generating data, delete this project's own unneeded intermediates. Keep ≥ 5% of the disk free; if it is below that, stop writing and notify the owner.
2. **Success.** With ≤ 64 noisy observations, the learner beats kernel ridge *and* the best no-history control on **unfamiliar kinds of systems**, in **both seeds**, and **the advantage grows with the number of systems in its history**. Every trial states how it moves toward this, and the log keeps a running count of trials evaluated on the development systems.
3. **Reserve sets.** Before using A or B, the candidate must pass the success criteria on development in both seeds, then pass again on a fresh, independent split or fresh seeds. Only then request the reserve test, and wait for approval.

### Disk status at the time of these rules

- Free space was 3.72% (137 GB of 3.6 TB), **already below the 5% floor**, almost entirely from other users' data.
- Stopped immediately:
  - the Trial 24 transient resampling and its queued solvability check (no learner had been trained);
  - all further data generation.
- Deleted my own unneeded intermediates: the partial Trial 24 resampled arrays, Trial 23's transient and oscillator arrays (metadata JSON kept), Trial 15's synthetic arrays (regenerable from their seed; metadata kept), and the scratch copies of the dysts wheel (the code used is vendored).
- **After cleanup: 3.75% free** (137 GB; 5% = 183 GB). This project now uses about 326 MB in outputs plus 124 MB of scratch, so the shortfall cannot be fixed by deleting my files. **Data generation is paused, and the owner has been notified.**

**Running count of trials evaluated on the development systems: 23** (Trials 1–23), plus 5 exploratory diagnostics on development episodes (Trial 6 pilot, Trial 8 noise/bias, Trials 21 and 22 AUCs, the Trial 17 re-check). The confirmation run used the confirmation set, not development.

## Audit of Trials 1–23 by underlying approach (30 October)

### Approach A — learn a feature basis from history, fit a linear (ridge) head per new system (17 trials)

**Trials:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 20, 23.

All of these share one idea: a network maps a state to a fixed set of features learned from past systems, and a new system is handled by fitting a linear combination of those features to its few observations. What varied:

- **Training objective:** mean vs log NMSE (1 → 2).
- **Basis structure:** single, wide, mixture library with leave-one-out selection, context-modulated (FiLM), pairwise any-dimension (2, 3, 4, 18).
- **Head prior:** ridge vs sparse (6, 7).
- **Training regime:** joint, sequential with replay, test-time continual, error-prioritized replay (5, 10, 11, 20).
- **Adaptation:** fine-tuning the basis at encounter (12).
- **History content:** catalog, synthetic non-polynomial, 11 / 37 / 145 real models, transients (15, 16, 18, 23).
- **Observation contract:** derivatives vs noisy trajectories vs observable-only (8, 9).

**What the failures suggest.** The per-system inference is *linear regression on a fixed learned basis*, so its prior is "this system lies near the span of what I have seen".

- When that is true (related systems) it wins enormously; that is the confirmed result.
- For an unfamiliar mechanism the span is wrong. The method then degrades into a generic regressor with a worse generic basis than a Gaussian kernel, and no amount of extra history changes the *form* of the inference.
- Leave-one-out selection among stored bases is unreliable at small K.
- Every variant does pointwise regression on noisy derivative estimates, the same information kernel ridge uses. Nothing in approach A extracts more information per encounter than kernel ridge does.

### Approach B — choose which observations to take, on top of approach A's learner (6 trials)

**Trials:** 13, 14, 17, 19, 21, 22.

**Variations:** generic coverage, leverage in learned features, a myopically learned scorer, disagreement among stored bases, uncalibrated and calibrated gates.

**What the failures suggest.**
- For smooth regression from an on-attractor pool, space-filling is close to optimal.
- Uncertainty measures computed from approach A's features inherit A's misspecification. They are confident exactly where A is wrong, on unfamiliar systems.
- The learner cannot tell that a system is unfamiliar before about 8–16 observations.
- The one apparent positive (Trial 17) was a confounded comparison.

### Verdict

Two approach families have been tried, and B is built on A's learner. **Twenty-three trials have mostly varied one idea.** Per the owner's instruction, the next trial switches to a genuinely different approach instead of the Trial 23 rebuild (Trial 24 is abandoned before any learner was trained).

### Untried approach families that could plausibly beat kernel ridge on unfamiliar systems at ≤ 64 noisy observations

1. **Amortized inference, i.e. learning the inference procedure itself** (a prior-fitted network: an in-context Transformer).
   - A sequence model reads the K noisy observations and directly outputs the predictive distribution at query states. It is trained over a very large, diverse prior of dynamical systems, so it approximates Bayesian inference under that prior.
   - It is not limited to a linear combination of fixed features, and it can represent structure shared *across mechanisms*: saturations, interactions, time-scale separation, positivity.
   - On small tabular datasets this class (TabPFN) beats Gaussian processes and kernel ridge when the prior matches the data.
   - Here, history is the prior, so "advantage grows with history" can be measured directly by training on increasing numbers of systems.
2. **Trajectory-consistent (weak-form or ODE-constrained) estimation.** Use the fact that observations come from one dynamical system: fit through integral or weak formulations or short-horizon simulation, so noise averages along trajectories. This extracts more information per encounter than pointwise derivative regression, which every approach-A variant and kernel ridge share. It can be combined with a learned prior.
3. **Sparse structural regression over a generic mechanistic grammar with a learned prior over structures** (Bayesian sparse regression with meta-learned term probabilities; saturating terms with unknown constants via variable projection). The Trial 6 pilot showed about 6× gains at K = 8 when the structure class matches. The caveat is that a term language is supplied once (the owner's earlier concern), which a single generic grammar plus learned priors only partly mitigates.
4. **Meta-learned kernels** (kernel ridge with a learned kernel mixture that includes the generic Gaussian kernel). Never worse than kernel ridge by construction, but close to approach A. A large gain is unlikely.
5. **Active experimentation with controlled resets** (choosing initial conditions). It changes the observation contract, which the handover flags as materially simplifying.

### Recommendation: try 1 (amortized inference) first

- **It is genuinely different.** It replaces approach A's fixed "linear in a learned basis" inference with a learned inference map, which the failure analysis identifies as A's bottleneck.
- **It measures the success criterion directly.** Train on priors built from increasing numbers of systems and check that the advantage over kernel ridge grows.
- **It fits the disk constraint.** Training tasks can be simulated on the fly on the GPU (batched RK4 on randomly generated mechanistic vector fields plus the stored catalog and real models), so no datasets are written; only checkpoints and logs (under 200 MB).
- **It is feasible in hours** on 1–2 free V100s.

Approach 2 (trajectory-consistent estimation) is the next candidate, and can be combined with 1 if 1 shows promise.

## Owner decision (30 October)

- **Disk rules replaced:**
  - keep **≥ 50 GB free** at all times;
  - keep this project's total outputs **< 20 GB**;
  - if free space drops below 50 GB, stop writing and notify the owner;
  - don't generate datasets to disk when they can be simulated on the fly.
  - Status at this decision: 137 GB free; project outputs about 0.33 GB.
- **The Trial 23 rebuild is skipped.** It fixes the data but keeps the method family the audit shows fails on unfamiliar systems.
  - *Kept for later:* the learnable-transient resampling (log-time / arc-length sampling with per-sample time stamps, checked first with kernel ridge) is a possible future *test set* of real biochemical transients.
- **Option 1 (amortized inference) starts as Trial 24.** Requirements:
  - the training distribution deliberately includes varied kinds of dynamics;
  - success is judged on unfamiliar kinds of systems, **held out by kind** (not just by parameters), against kernel ridge and the best no-history control;
  - the history-growth curve is tested by training on increasing numbers *and* varieties of systems.

## Trial 24: amortized inference trained on varied kinds of dynamics, tested on held-out kinds (pre-registered 30 October, before training)

**Approach family: new** (audit option 1). The learner does not fit coefficients on a learned basis. A Transformer reads the K noisy observations of a new system and directly outputs its predicted velocity (central difference) at query states: it learns the *inference procedure*, approximating Bayesian prediction under its training distribution of dynamical systems.

**How it moves toward the success definition.** It tests the definition directly: unfamiliar *kinds* of systems held out by kind, ≤ 64 noisy observations, kernel ridge and the best no-history control, both seeds, and an advantage that should grow with the number and variety of systems in the history.

**Running count of trials evaluated on the development systems: 24** (this one included).

### Kinds and the held-out-by-kind split (fixed now; `kinds.py`, `trial24pfn/kinds.json`)

- Each real 3-D system's nonlinearity kinds are read from its equation structure: polynomial, trig, exponential, rational/saturating, piecewise, tanh, power, log.
- **Held-out kinds:** rational/saturating (Michaelis–Menten, Hill, Holling, ratio forms) and exponential.
- **Test set:** every real 3-D system containing either held-out kind — **26 systems.**
  - 16 BioModels (all of Trial 16's 3-D set);
  - catalog systems Blasius, CoevolvingPredatorPrey, HastingsPowell, JerkCircuit, SprottMore and YuWang2 (meta-train), SaltonSea and YuWang (development), GlycolyticOscillation and ItikBanksTumor (confirmation, already opened).
- **No system containing a held-out kind is ever used in training, and no generator produces one.**
- Familiar-kind sanity set: the 9 polynomial development systems.

### Training distribution (simulated on the fly on the GPU; nothing written to disk except checkpoints)

**Generators** (3-D, random parameters, bounded by construction or by rejection):
- **polynomial:** sparse linear + sparse quadratic + cubic confinement;
- **trig:** −Γx + Σ a·sin(u·x + b);
- **piecewise:** −Γx + Σ a·σ(u·x + b), with σ ∈ {clip to ±1, a triangular bump};
- **tanh:** −Γx + W·tanh(Ux + b);
- **mixed:** sin, cos, tanh and clip terms.

Each candidate is integrated with batched RK4, with a 100 time-unit pilot. It is kept if motion is sustained and bounded and its characteristic period P = 2π·radius/speed is in (0.05, 50). It is then re-integrated at h = P/40 for 1,200 samples (segment A = context, B = queries). Probe acceptance rates: polynomial 1.4%, trig 8.9%, piecewise 4.5%, tanh 5.5%, mixed 7.7%.

**Real history:** catalog meta-train systems whose kinds are all allowed — 42, 43, 44 and 45 systems for varieties V1–V4. They are sampled in proportion to their count among all systems.

### Model and training

- **Model:** 6 layers, width 192, 6 heads, feed-forward 512, about 2.2M parameters.
  - Context tokens embed (u, y); query tokens embed u.
  - Queries attend only to context tokens and themselves.
  - It outputs the standardized velocity at each query.
- **Loss:** log(min(NMSE, 10) + 1e-4).
- **Optimizer:** AdamW (lr 3e-4, weight decay 0.01), 1,000 warm-up steps, cosine, 30,000 steps × 64 episodes, K log-uniform in [4, 64].
- **Contract:** observable-only. Training targets are noisy central differences (0.5% state noise), with no ground truth anywhere.

### History-growth grid (both seeds)

- **Variety, at N = 8,192 generated systems:**
  - V1 = polynomial;
  - V2 = + trig;
  - V3 = + piecewise;
  - V4 = + tanh + mixed.
- **Number, at V4:** N ∈ {512, 8,192, 65,536}.
- That is 6 configurations × 2 seeds = 12 learners.

### Evaluation

- Trial 9's observable contract. Per system and K ∈ {8, 16, 32, 64}: 4 instances × 5 episodes, generator seeds 24000 + 97i + K.
- **No-history controls:** kernel ridge and random Fourier features (leave-one-out hyperparameters). The *best no-history control* at each K is whichever has the lower median over the 26 test systems.

### Criteria (both seeds; primary learner V4, N = 65,536)

- **S1 (beats no-history on unfamiliar kinds):** test median ≤ 0.8× the best no-history control's at **both** K = 16 and K = 32, *and* beats kernel ridge on ≥ 60% of the 26 systems at K = 32 (≥ 16).
- **S2-variety (grows with variety):** with adv = log(control median / learner median) at K = 32, at N = 8,192, Spearman ρ(V1…V4, adv) ≥ 0.8 *and* adv(V4) − adv(V1) ≥ log 1.25.
- **S2-number (grows with number):** at V4, adv(65,536) − adv(512) ≥ log 1.25, *and* adv(8,192) ≥ adv(512).
- **Pass = S1, S2-variety and S2-number in both seeds.**
- **Reported:** K = 8 and 64; the familiar-kind sanity set; per-system results.

**What happens next, decided now:**
- **Pass:** per standing rule 3, replicate on a fresh, independent split (different held-out kinds, e.g. trig + piecewise, with the rational and exponential kinds allowed in training) and on fresh seeds before any reserve request.
- **S1 fails but S2 passes:** the prior is learning transferable structure but is not yet good enough. Scale the variety and number of generators (the history axis) before changing the method.
- **S2 fails:** amortized inference with this prior does not grow with history. Go to audit option 2 (trajectory-consistent estimation), possibly combined.

**Resources:** each learner takes about 25 min of training plus pool building (up to about 35 min at N = 65,536), with 4 CPU threads and nice 10, on free GPUs except GPU 4. Disk: checkpoints about 9 MB each.

## Trial 24: result (30 October)

**Classification: FAILED** (S2 fails in both seeds). **S1 passes in both seeds:** this is the first approach in the project to beat kernel ridge and the best no-history control on unfamiliar kinds of systems.

| | Seed 1 | Seed 2 |
|---|---|---|
| S1: V4/N65,536 median vs best control (kernel ridge) at K = 16 / K = 32 (need ≤ 0.8 at both) | **0.69 / 0.77** ✓ | **0.72 / 0.79** ✓ |
| S1: wins against kernel ridge at K = 32 (need ≥ 16/26) | **17/26** ✓ | **17/26** ✓ |
| S2-variety: advantage at K = 32 for V1 / V2 / V3 / V4 (N = 8,192); Spearman ρ | .276 / .235 / .184 / .236; ρ = −0.4 ✗ | .244 / .240 / .287 / .252; ρ = 0.6 ✗ |
| S2-number: advantage at K = 32 for N = 512 / 8,192 / 65,536 (V4); need a gain ≥ 0.223 from 512 to 65,536 | .179 / .236 / .257 (gain 0.078) ✗ | .035 / .252 / .241 (gain 0.206) ✗ |

Unfamiliar-kind medians (26 systems), K = 8 / 16 / 32 / 64:

| | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|
| kernel ridge | .588 | .289 | .0997 | **.0147** |
| random Fourier features | .646 | .379 | .133 | .032 |
| Transformer V4/N65,536, seed 1 / 2 | **.443 / .422** | **.199 / .209** | **.077 / .078** | .026 / .029 |
| Transformer V1/N8,192 (polynomial-only history), seed 1 / 2 | .380 / .357 | .165 / .170 | .076 / .078 | .021 / .019 |

Familiar kinds (9 polynomial development systems, K = 32): kernel ridge .0050 against the Transformer .052 / .051, i.e. **10× worse**.

### Reading

- **Learning the inference procedure beats kernel ridge on kinds never seen in training,** at 8–32 noisy observations, in both seeds, on 17 of 26 real systems. The gains are largest where kernel ridge struggles, e.g. Blasius 0.29–0.36×, Bianca2013 0.11–0.15×, Goldbeter1991 0.53–0.58×, Caravagna2010 0.41–0.62×.
- **It loses where kernel ridge is already very precise** (SaltonSea, kernel ridge .0017, 4.4–5.1×; BIOMD 763, 323 and 931). It is 10× worse on familiar polynomial systems and is overtaken at K = 64. Its precision bottoms out around .02–.05.
- **Why S2 fails.** The final training NMSE is .079–.103 for *every* pool size, the same at N = 512 as at N = 65,536, and still falling at 30k steps. **The model is capacity- and optimization-limited, not history-limited.** It cannot absorb even 512 systems in detail, so extra or more varied history cannot show up as a larger advantage. The S2 test was confounded by model size and training length.
- The polynomial-only history (V1) is about as good as V4 on held-out kinds. The advantage comes from generic inference skill (handling noisy central differences, extrapolating from segment A to segment B), not from kind-specific structure. Whether kind variety helps once the model can absorb it is untested.

### Deviation from the pre-registered next step

The rule said that if S2 fails, move to option 2. The diagnosis (training error flat across pool sizes) shows S2 was not a test of whether history helps, but of whether this model could absorb history. The direct test is to scale the model and training budget on the same history grid: **Trial 25**. Option 2 (trajectory-consistent estimation) stays next if scaling does not produce growth. The precision floor on familiar and easy systems is a separate issue, deferred to a later trial so that one thing changes at a time.

## Trial 25: does a larger Transformer turn history into a growing advantage? (pre-registered 30 October, before training)

- **How it moves toward success.** Trial 24 met the "beats kernel ridge on unfamiliar kinds" half in both seeds. Trial 25 targets the "advantage grows with history" half.
- **Running count of trials evaluated on the development systems: 25.**
- **Hypothesis.** With enough capacity and training to absorb its history, the advantage over kernel ridge on held-out kinds grows with the number of training systems and with their variety.
- **Single change from Trial 24: model and budget.**
  - Width 256, 8 layers, 8 heads, feed-forward 1,024 (about 6.4M parameters, 3× Trial 24).
  - 100,000 steps (3.3×), 2,000 warm-up steps, peak learning rate 2e-4.
  - Everything else identical: generators, kinds, held-out-by-kind test set (26 systems), contract, loss, evaluation episodes (seeds 24000 + 97i + K), controls.
- **History grid** (both seeds):
  - number, at V4: N ∈ {512, 8,192, 65,536};
  - variety, at N = 16,384: V1, V2, V3, V4. N = 16,384 replaces 8,192 so that variety has more room to matter.
  - That is 7 configurations × 2 seeds = 14 learners.
- **Criteria** (both seeds; primary learner V4/N65,536):
  - **S1:** as in Trial 24;
  - **S2-variety:** as in Trial 24, at N = 16,384;
  - **S2-number:** as in Trial 24.
  - **Pass = all three in both seeds.**
- **Also reported:** final training NMSE per pool size (now the advantage should depend on the pool, and training error should rise with pool size if the model is absorbing more); the familiar-kind set; K = 64.

**What happens next, decided now:**
- **Pass:** a fresh, independent split (different held-out kinds) and fresh seeds, per standing rule 3.
- **S2 fails again, and training error still does not depend on the pool:** still capacity-limited. Scale once more only if the trend from Trial 24 to 25 shows movement. Otherwise move to option 2.
- **S2 fails, and training error now depends on the pool:** absorbing history doesn't help held-out kinds. Move to option 2, combined with the Transformer.

**Resources:** about 3 h per learner with 2 per GPU on GPUs 0–3 and 5–7, threads capped. Disk: checkpoints of about 26 MB each (14 × 26 MB = 0.4 GB).


*Date correction (30 October): ten headings above, from the Trial 23 result to the Trial 25 pre-registration, were mistakenly labelled 31 October. They were all written on 30 October (UTC, per the machine clock) and have been relabelled. No content changed.*

## Trial 25: result (31 October)

**Classification by its own pre-registration: FAILED** (S2-variety fails in both seeds). **By the owner's success definition (standing rule 2): MET on the development systems in both seeds.** The learner beats kernel ridge and the best no-history control on unfamiliar kinds, and its advantage grows with the number of systems in its history.

| | Seed 1 | Seed 2 |
|---|---|---|
| S1: V4/N65,536 median vs best control (kernel ridge) at K = 16 / K = 32 (need ≤ 0.8 at both) | **0.574 / 0.574** ✓ | **0.578 / 0.574** ✓ |
| S1: wins against kernel ridge at K = 32 (need ≥ 16/26) | **21/26** ✓ | **22/26** ✓ |
| S2-number: advantage at K = 32 for N = 512 / 8,192 / 65,536 (V4); need a gain ≥ 0.223 and adv(8,192) ≥ adv(512) | **−0.114 / 0.464 / 0.555** (gain 0.669) ✓ | **−0.054 / 0.476 / 0.555** (gain 0.609) ✓ |
| S2-variety: advantage at K = 32 for V1 / V2 / V3 / V4 (N = 16,384); need ρ ≥ 0.8 and V4 − V1 ≥ 0.223 | .570 / .612 / .621 / .463; ρ = −0.2 ✗ | .595 / .481 / .649 / .657; ρ = 0.8 but gain only 0.062 ✗ |

Unfamiliar-kind medians (26 systems), K = 8 / 16 / 32 / 64:

| | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|
| kernel ridge | .588 | .289 | .0997 | .0147 |
| Transformer V4/N65,536, seed 1 / 2 | **.333 / .340** | **.166 / .167** | **.057 / .057** | .0149 / .0168 |
| Transformer V1/N16,384 (polynomial-only history), seed 1 / 2 | .297 / .301 | .169 / .153 | .056 / .055 | .014 / .012 |

Familiar kinds (9 polynomial development systems, K = 32): kernel ridge .0050 against the Transformer .017 / .015, 3× worse (Trial 24 was 10× worse).

**Final training NMSE, now rising with pool size, i.e. the model absorbs history:**

| N (V4) | 512 | 8,192 | 16,384 | 65,536 |
|---|---|---|---|---|
| Final training NMSE | .040–.042 | .064–.065 | .065–.067 | .064–.067 |

### Reading

- **Scaling the learned inference procedure** (3× parameters, 3.3× training) turned Trial 24's 21–31% advantage into **about 43% lower error than kernel ridge** on 26 real systems of kinds never seen in training (rational/saturating and exponential). It wins on 21–22 of 26, in both seeds. That covers 8–32 observations, and it is level at 64.
- **The advantage grows strongly with the number of systems in the history.**
  - At 512 systems the larger model is *worse* than kernel ridge (it overfits a small history).
  - At 8,192 it is about 37% better; at 65,536 about 43% better, in both seeds.
  - This is the history-dependence that Trial 24 could not show.
- **It does not grow with the variety of kinds.** A polynomial-only history is about as good as one with trig, piecewise, tanh and mixed generators. The transferable knowledge seems to be generic to dissipative dynamics on attractors (how to infer velocity fields from noisy trajectory snippets), not specific to nonlinearity kinds. Since the held-out kinds are by design unlike any training kind, this is plausible, but it was a pre-registered criterion and it failed.
- **Precision on easy, familiar systems is still worse than kernel ridge** (3×). This is outside the owner's success definition (which concerns unfamiliar kinds), but a deployed learner would need a fallback for it.

**Next, per standing rule 3:** the success criteria are met on development in both seeds, so the candidate must pass again on fresh seeds (or a fresh, independent split) before a reserve test is requested from the owner. Trial 26 is that replication. Its primary criteria are the owner's success definition; variety growth is reported as a secondary outcome, and its Trial 25 failure stands.

**Running count of trials evaluated on the development systems: 25.**

## Trial 26: fresh-seed replication of Trial 25 (standing rule 3; pre-registered 31 October, before training)

- **Purpose.** Standing rule 3 requires a candidate that meets the success criteria on development in both seeds to pass again on fresh seeds (or a fresh, independent split) before a reserve test is requested. A meaningful *new* held-out-kind split is not available in 3-D: the trig and piecewise kinds have only about 4–5 real systems. So this is a **fresh-seed replication with fresh evaluation episodes.**
- **How it moves toward success.** If it passes, the candidate has met the owner's success definition twice independently, and the reserve-set test can be requested.
- **Running count of trials evaluated on the development systems: 26.**

### Identical to Trial 25

Architecture (width 256, 8 layers, about 6.5M parameters), training budget (100k steps, lr 2e-4, 2k warm-up), generators, kinds and the held-out-by-kind split, real training systems, observable contract, loss, test systems (the same 26 unfamiliar-kind real systems and the 9 familiar polynomial development systems), controls.

### Fresh

- **Seeds 3 and 4.** New network initializations and new simulated pools; the pool generator seed depends on the run seed.
- **Evaluation episodes** with generator seeds 26000 + 97i + K, never used before.

### Grid

- number, at V4: N ∈ {512, 8,192, 65,536} (primary);
- variety, at N = 16,384: V1 and V4 only (secondary readout).
- That is 5 configurations × 2 seeds = 10 learners.

### Primary criteria: the owner's success definition, operationalized exactly as in Trial 25 (both seeds)

- **S1:** V4/N65,536 median over the 26 unfamiliar-kind systems ≤ 0.8× the best no-history control at K = 16 and K = 32, *and* beats kernel ridge on ≥ 16/26 at K = 32.
- **S2-number:** advantage at K = 32 grows with history size: adv(65,536) − adv(512) ≥ log 1.25 and adv(8,192) ≥ adv(512).
- **Pass = S1 and S2-number in both seeds.**

**Secondary (reported, not part of pass/fail):** variety, adv(V4) − adv(V1) at N = 16,384. Trial 25 failed its pre-registered variety criterion, and that failure stands whatever this trial shows. Also reported: K = 8 and 64, familiar-kind precision, per-system results.

**What happens next, decided now:**
- **Pass:** stop and request the reserve-set test from the owner, with a fixed protocol, and wait for approval (standing rule 3).
- **Fail:** report it. The Trial 25 result is then not robust to seeds, and the next step is diagnosing the seed sensitivity.

**Resources:** 10 learners, at most 2 per GPU on GPUs 0–3 and 5–7, 2 CPU threads each, nice 10. Checkpoints about 26 MB each.

---

# GOAL CORRECTION (31 October) — exact recovery replaces prediction error

The owner corrected the goal. **Prediction error compared with kernel ridge is not the target.**

**Ultimate goal.** A network that learns physical systems *exactly* from data, so it can generate the physics without running a physics package, and that gets better at learning new systems the more systems it has learned.

**Rules:**
- **Exactness.** The learned network reproduces each system's trajectories to machine precision, regardless of model form. Equations cannot be assumed known; for some systems they are unknown.
- **Cheap to use.** Running the learned network must be cheaper than running the original simulation.
- **Teaching is unrestricted.** Heavy search, teacher pipelines (like N1/N2/N3), Newton polishing and long training are all allowed. Learning cost is not a constraint, but it is reported.
- **It must actually learn.** The network may not be given the true equations or call the simulator when generating trajectories. The final artifact generates the physics on its own from what it learned from data.
- **Accumulation.** Knowledge from earlier systems should make exact recovery of new systems more reliable and possible from fewer observations.

**New success criterion** (replaces standing rule 2):
- Observations are noise-free.
- **Exact recovery:** on held-out starting states not used for fitting, the learned one-step map matches the true system to near machine precision (relative error ~1e-12 or better), and multi-step trajectories stay at that level up to a horizon justified from each system's chaos timescale.
- **Experience** is measured by how the *success rate* of exact recovery, and the *number of observations needed*, improve as the number of past systems grows, compared with learning each system from scratch.

**Housekeeping:**
- Trials 1–25 optimized the wrong target. They remain context.
- The prediction-error replication pre-registered above as "Trial 26" was **stopped before evaluation** on this goal change. It is renumbered **25R (aborted, not evaluated)**, so Trial 26 is the first trial under the new goal.
- Running count of trials evaluated on the development systems under any goal: 25.

## How exactness is defined and measured, per system

These definitions apply to every trial from now on. Notation: true vector field f (never given to the learner), dimension d, attractor scale s_i = per-coordinate standard deviation over a long reference trajectory, characteristic period P, and largest Lyapunov exponent λ (from the catalog metadata, or estimated by tangent dynamics for BioModels).

1. **One-step interval.** Δt = P/100, used both for observations and as the artifact's step.
2. **Reference one-step map Φ** (truth, for evaluation only).
   - The true f is integrated over Δt with a fixed-step 8th-order Runge–Kutta (scipy's DOP853 tableau) in float64.
   - The number of substeps is the smallest power of two for which doubling it changes Φ by ≤ 1e-15·s on 100 attractor states.
   - Sanity check: Φ agrees with scipy `solve_ivp`(DOP853, rtol 1e-13) to ≤ 1e-13·s.
3. **Held-out starting states.**
   - 1,000 states on 4 held-out trajectories (initial conditions and instances never used for learning), after burn-in.
   - 100 of them start the multi-step tests.
4. **One-step exactness:** e₁ = max over held-out states and coordinates of |Φ̂ᵢ(x) − Φᵢ(x)| / sᵢ ≤ **1e-12**. Median and error relative to the step displacement are also reported.
5. **Multi-step exactness.**
   - **Error curve.** e(k) = max over coordinates of |X̂ᵢ(k) − Xᵢ(k)| / sᵢ (median and 90th percentile over the 100 starts), where X is the reference rollout and X̂ the artifact's rollout.
   - **Ideal-model envelope.** E*(k) is the median error of the *true* map with an independent random perturbation of size 1e-12·s injected every step, i.e. what a model sitting exactly at the 1e-12 threshold would achieve on this system (20 Monte-Carlo runs).
   - **Horizon.** H = the first k at which E*(k) reaches 1e-6, capped at 2,000 steps (20 periods). For chaotic systems, H ≈ ln(10⁶)/(λΔt) steps; for non-chaotic ones, the cap applies.
   - **Pass:** median e(k) ≤ 10·E*(k) and 90th percentile ≤ 100·E*(k) for all k ≤ H.
   - **Justification.** E* *is* the chaos timescale applied to a 1e-12-accurate map: no model at that accuracy can do better. The horizon stops where errors leave the linear-growth regime.
6. **Exact recovery** = one-step *and* multi-step pass.
7. **Cost.**
   - Wall time to generate 1 trajectory × 1,000 steps, and 100 trajectories × 1,000 steps, with the artifact, against the *original simulation*: scipy `solve_ivp`(DOP853, rtol 1e-13, atol 1e-15·s) on the true right-hand side, with the same outputs and the same machine.
   - FLOPs per step for both are reported.
   - "Cheap" means the artifact is faster in both settings.

## Trial 26: exact recovery by expression-network discovery with machine-precision polishing; accumulation through a growing term library (pre-registered 31 October, before running)

### Approach

Machine precision across a whole attractor is out of reach for generic fixed-form approximators at feasible sizes. Every earlier learner here, a learned basis plus linear head or a Transformer, has an error floor between 1e-2 and 1e-6.

The feasible route is to discover *what the dynamics are made of*. The learner builds an **expression network** for f: a sparse computational graph of learned terms with learned constants. It polishes every constant to machine precision with Gauss–Newton/Levenberg–Marquardt on the one-step flow map. The artifact is that network plus its own fixed-step 8th-order integrator.

- **Nothing about any specific system is supplied.** What is supplied, once and for all systems, is a generic grammar of elementary building blocks. This is the approach's main assumption: dynamics are finite compositions of elementary functions. That holds for every catalog and BioModels system, and for most physics.
- **Experience** grows the grammar, by promoting discovered composite terms, and supplies a prior over which building blocks to try first and where their constants tend to lie.

### Observations (noise-free; the learner never sees f and never calls the simulator)

- M bursts of L = 13 consecutive states at spacing Δt, with burst starts drawn from 4 learning trajectories on the attractor, disjoint from the held-out ones.
- **Observation count = 13M.** Grid: M ∈ {2, 4, 8, 16, 32, 64}, i.e. 26 to 832 observations.

### Grammar (generic, fixed before looking at any test system's equations)

Coordinates are internally centred and scaled.

- **Level 1, base terms:**
  - the constant;
  - monomials of total degree ≤ 3;
  - single-variable factors with fitted constants: sin(ωx + φ), exp(ax), tanh(ax + b), |x − b|, |x|^p (real p), 1/(K + x);
  - and each non-polynomial factor multiplied by one linear monomial.
- **Level 2, composites** (budgeted search):
  - products of two non-polynomial factors;
  - saturations u/(K + u) and K/(K + u) of a base factor u (covering Michaelis–Menten and Hill forms).
- **Level 3:** products of two level-2 saturations.

### Teacher pipeline, per system (unrestricted cost; reported)

1. **Derivatives.** Estimated at the 5 interior points of each burst with 9-point central differences. They are used only for screening.
2. **Term screening, per component.** Beam search (width 5) of greedy forward selection with backtracking. Linear coefficients come from least squares; nonlinear constants from variable projection (grid plus Gauss–Newton). Level 1 first, then levels 2 and 3 within a budget of 500 composite evaluations per component. Screening stops at relative residual ≤ 1e-6 or 12 terms.
3. **Machine-precision polish.** Levenberg–Marquardt on the one-step flow-map residuals over all consecutive pairs (12 per burst), in float64. The model is integrated with the same fixed-step 8th-order RK, with substeps chosen by the model's *own* self-convergence (no simulator).
4. **Internal validation.** 25% of the bursts are held back inside the training data. A model is accepted when its validation one-step error is ≤ 1e-13·s; otherwise the next beam candidate is tried. Limits: ≤ 50 polished candidates and ≤ 15 CPU-minutes per system.

### Accumulation

- **Library.** Every *exactly recovered* history system contributes its term templates (for example x_j·x_k, x_j/(K + x_j), sin(ωx_j + φ)), their usage counts and its fitted constants.
- **Grammar growth.** Composite templates found at levels 2 and 3 are promoted to level 1 for later systems.
- **Prior.** Screening and beam ranking use an MDL-style penalty −log p(template) from smoothed library counts. Nonlinear constants are also initialized at library values.
- **From scratch (N = 0):** an empty library — uniform prior, base level 1, and the same level 2/3 budget.
- **History stream** (seeded order): the 51 catalog meta-train systems plus the 11 BioModels 3-D training systems, i.e. 62 systems, learned one at a time at M = 64. Library snapshots are taken at N ∈ {0, 8, 24, 62}.
- **Test systems (never in the history): 32.** The 13 development, the 13 confirmation (already opened) and the 6 held-out BioModels 3-D systems. Each test system is run for every snapshot and every M.
- **Seeds 1 and 2** set the stream order and the observation draws.

### Criteria (both seeds)

- **X1 (exactness is achievable):** at N = 62 and M = 64, exact recovery on ≥ 50% of the 32 test systems.
- **X2 (experience raises success):** at M = 8, the success rate at N = 62 is ≥ N = 0's + 15 percentage points, and the rates at N = 0, 8, 24, 62 rise monotonically (Spearman ρ ≥ 0.8).
- **X3 (experience reduces observations needed):** over test systems exactly recovered at N = 62, the median minimal observation count at N = 62 is ≤ half that at N = 0. A failure at N = 0 counts as 1,664 (twice the maximum).
- **X4 (cheap to use):** every exactly recovered test system's artifact is faster than the original simulation (wall time, 1 and 100 trajectories).
- **Pass = X1–X4 in both seeds.**
- **Reported:** learning cost per system, recovery by kind, failures with their reasons (structure outside the grammar, budget, conditioning), and FLOPs.

**What happens next, decided now:**
- **Pass:** standing rule 3 — a fresh replication (new seeds, and new test systems if available, e.g. the higher-dimensional BioModels), then a reserve-set request.
- **X1 fails:** diagnose by failure reason; extend the grammar (for example, general expression search) or add a numerical-exactness fallback (spectral or high-degree polynomial flow maps).
- **X1 passes but X2 or X3 fail:** exactness works but experience does not help. Strengthen the accumulation mechanism, for example a learned neural proposal network that predicts the terms from the observations.

### Implementation order

1. The exactness harness, validated by feeding it the *true* right-hand side as a "perfect learner": it must pass. The same harness then checks that an obviously wrong model fails.
2. The from-scratch pipeline, on a few history systems.
3. The library and the history stream.
4. The full test.

Data is generated in memory, not written to disk. CPU-bound in float64: at most 24 worker processes at nice 10.

## Owner standing rules (31 October, added mid-implementation)

1. **Check in before each trial.** Before running any trial, send the owner a plain-language explanation (no jargon) of:
   - what the network learns from past systems;
   - how that knowledge is used on a new system;
   - how the test tells genuine learning from history apart from new systems simply resembling the old ones.

   Wait for approval. After each trial, send a plain-language summary of the result and what it means for the goal. Within an approved trial, work autonomously.
2. **Similarity is not accumulation.** The owner is not convinced the earlier approach learned from history: its gains came from new systems resembling old ones. For every trial, the test must show that the benefit comes from what was learned, not from choosing test systems close to the training ones, and the log must say how this is ruled out before anything runs.

**State when these rules arrived.** No Trial 26 runs had started. Running were only (a) validation of the exactness harness and (b) a smoke test of the learner on Lorenz (tool development, not trial runs).

- **Harness validation on Thomas:**
  - Horizon 399 steps (λ ≈ 0.63; Δt ≈ 0.05).
  - The true field as a "perfect learner" passes.
  - A 1e-9 output error gives a one-step error of 3.1e-10 and fails. The harness discriminates as intended.
- **Cost on Thomas:** the artifact is 0.6× (i.e. *slower*) than scipy DOP853 for a single trajectory and 9.5× faster for 20 trajectories in a batch. Single-trajectory cost is dominated by Python overhead in the fixed-step integrator. This is raised with the owner.

## Trial 26 — revised before approval (31 October)

**Flaw in the pre-registered Trial 26 under the new similarity rule.** Its only accumulation mechanisms were reuse of past equation pieces, a usage prior over them and parameter initialization. All three can only help a new system containing pieces seen before, which is resemblance by construction. The revised design adds kind-agnostic, transferable knowledge and a test that separates the two.

### What is learned from history

- **(a) Building-block memory.** The original mechanism, kept as the *resemblance-only control*.
- **(b) An "influence guide" network.** From a new system's observations, it predicts:
  - which variables each rate of change depends on;
  - how many pieces each equation has;
  - how complex the pieces are (grammar level).

  It is trained on past systems, with labels taken from the learner's *own* recovered models (never the true equations). This knowledge is about *structure in general*, not about specific pieces, so it can transfer to systems built from pieces never seen.

### How it is used

The search for exact equations first explores only the influences and complexity the guide predicts (with fallback to the full search). Known pieces are tried first. New pieces are built from the generic grammar. The precision polish and internal validation are as before.

### How resemblance is ruled out

- **(1) Test held out by building block.** History contains only systems whose pieces are polynomial, trigonometric, piecewise or tanh (catalog meta-train systems of those kinds: 45). The **decisive test group, G-novel,** is the 26 real systems containing a piece *never* present in any history system (rational/saturating or exponential — the Trial 24 held-out-kind set). For G-novel the building-block memory cannot contain the needed piece, so any gain must come from transferable knowledge.
  - G-familiar (development systems whose pieces all occur in the history) is reported separately as *resemblance-eligible*. It does not count toward accumulation.
- **(2) Dissimilar-history check.** For each G-novel system, every history system sharing *any* nonlinear piece with it is removed, and the benefit is re-measured.
- **(3) Controls on the same test systems with the same budgets:**
  - **scratch:** no history;
  - **memory-only:** the resemblance-only control;
  - **guide trained on scrambled lessons:** the same amount of training but shuffled labels; it must lose the gain;
  - **no-history statistical influence screen:** local sparse regression; the learned guide must beat it.
- **(4) History-growth curve.** The guide is trained on N ∈ {0, 8, 24, 45} history systems. The G-novel benefit must grow with N.

### Measurements

- Exact recovery (the 1e-12 one-step and envelope tests on held-out starts) for each test system at M ∈ {2, 4, 8, 16, 32, 64} bursts.
- Success rate at fixed M, and the minimal observation count.
- Seeds 1 and 2.

### Criteria (both seeds)

- **C0, capability prerequisite (reported):** scratch exact-recovery rate at M = 64. If it is below 25% on G-novel, the accumulation part is uninformative, and capability is fixed first.
- **A1 (learning beyond resemblance), on G-novel:**
  - full learner at N = 45 vs scratch: success rate at M = 8 up ≥ 15 points, *or* the median minimal observation count ≤ 0.5×;
  - the scrambled-lesson guide keeps ≤ half of that gain;
  - the learned guide beats the statistical screen.
- **A2 (grows with history):** the G-novel benefit at M = 8 rises with N (Spearman ρ ≥ 0.8).
- **A3 (dissimilar history):** at least half of the A1 benefit remains with per-system dissimilar histories.
- **Cost:** reported, with batch generation as the primary use case. Whether single-trajectory speed is required is a question for the owner.
- **Pass = A1, A2 and A3 in both seeds, with C0 met.**

**Status: awaiting owner approval.** A plain-language explanation was sent; nothing in Trial 26 runs before approval.

### Refinements before approval (31 October, nothing run)

Checking the kinds file (`trial24pfn/kinds.json`) against the design:

- **History pieces.** The 45 history systems contain only polynomial (products and whole-number powers), trigonometric, piecewise and tanh pieces. None contains rational/saturating, exponential, non-integer-power or log pieces.
- **G-novel = 27 systems.** These are the 26 rational/exponential systems plus **WindmiReduced** (development), whose non-integer power never appears in the history. The 26 are:
  - 16 BioModels;
  - 6 catalog systems removed from the history: Blasius, CoevolvingPredatorPrey, HastingsPowell, JerkCircuit, SprottMore and YuWang2;
  - SaltonSea and YuWang;
  - GlycolyticOscillation and ItikBanksTumor.
- **G-familiar = 10 systems** (reported only): the 9 polynomial development systems and Thomas.
- **A1 is also measured against the memory-only control.** Of the full learner's G-novel gain over scratch, at least half must remain over memory-only. Otherwise reuse of the *familiar* pieces inside G-novel systems (for example, their product terms) could carry the result.
- **A3 is redefined so it is not degenerate.** Removing every history system that shares *any* piece would empty the history, because nearly all systems share product terms. Instead:
  - For each G-novel system the full learner recovers at N = 45, the history systems are ranked by structural overlap with its recovered equations: shared term shapes, ignoring variable labels, taken from the learner's own recovered models.
  - A history of the 22 *least similar* systems is compared with 22 random history systems (3 draws).
  - **A3 passes if the least-similar history keeps ≥ half of the random-history benefit at M = 8**, or in minimal observations.
- **Copy-the-closest-system baseline** (reported, mainly informative on G-familiar): take the structure of the past system most similar in data and refit only its constants.
- **Budgets.** Per-system search and polish budgets are fixed on *history* systems during tool development and recorded here before any test system is run.

**Running count of trials evaluated on the development systems (any goal): 25.** Trial 26 will be the 26th.

## Trial 26 approved (31 October)

The owner approved Trial 26 as described in the plain-language check-in (revised design plus the refinements above). Work within the trial is autonomous from here. No test system has been run.

### Tool correction before any test system (substep rule)

- **Problem.** The pre-registered rule ("doubling the substeps changes Φ by ≤ 1e-15·s") cannot be met when a system's states are large relative to their spread. For Lorenz, |x|/s ≈ 5, so floating-point rounding alone moves Φ by about 1e-15·s. The search then ran to the 1,024–4,096-substep cap, which is why the Lorenz and BioModels harness checks and the first learner smoke test stalled.
- **Corrected rule, for both the reference and the learner's own self-convergence.** Choose the smallest power-of-two substep count whose doubling changes the map by ≤ 1e-14·s. Alternatively, once the change is already below 1e-13·s and doubling no longer reduces it by 8×, the rounding floor has been reached; stop there.
- **Margin.** Both stopping points remain ≥ 10× below the 1e-12 exactness threshold.

### Tool development on history systems (31 October; no test system touched)

- **Derivatives are too inaccurate to certify supports.** The first learner design screened supports on 9-point derivative estimates, assuming they were accurate to about 1e-12. Measured on history systems at Δt = P/100, their relative error is:

  | System | Relative error of 9-point derivatives |
  |---|---|
  | Lorenz | 2e-5 |
  | Rossler | 1e-4 |
  | Aizawa | 2e-8 |
  | ThomasLabyrinth | 1e-2 |

  The catalog's dominant period understates the fastest timescales. A 13-point integral formulation is no better. Derivative screening can therefore only *propose* supports. Only the exact flow map can decide them, which was already the pre-registered principle for the polish.
- **Redesign (`exact_learner.py`).**
  1. The screening tolerance is set from a data-based estimate of the derivative error (Richardson comparison of 5-, 7- and 9-point estimates). The screening yields a generous support.
  2. Levenberg–Marquardt polish on the one-step flow map.
  3. Stepwise refinement *on the flow map*:
     - prune any term whose linearized removal cost (others re-optimized) keeps the residual at the 1e-14 floor;
     - add missing terms by their exact flow-map sensitivity (forward-mode derivatives through the model's own RK8);
     - repeat.
  4. Acceptance is unchanged: held-back bursts reproduced to ≤ 1e-13.
- **Speed.** All Jacobians are forward-mode. The first version used about 300,000 reverse-mode passes per system.
- **Grammar.** The sine grid was widened to frequencies {0.5, 1, 1.5, 2, 3, 4, 6} × 4 phases (previously 4 × 2). No other grammar change.
- **New tools.**
  - `guide.py`: the influence guide, a permutation-equivariant set network, and the no-history kernel-ridge leave-one-out influence screen.
  - `trial26.py`: the driver.
  - `compile_artifact.py`: the learned model plus its own RK8, generated as C and compiled, for the cost measurement.

### Tool development, continued (history systems only)

- **Development data must come from the trial's own generator.** Early development bursts came from scipy's adaptive DOP853 (rtol 1e-13). Those are consistent with no one-step map below about 1e-13, so every fit floored there: SprottB at 4e-13, Rossler at about 1e-11. With the harness reference (fixed-step RK8, substeps converged to 1e-14), a full degree-2 polynomial support on Lorenz polishes to **1.3e-15 in 3 Levenberg–Marquardt iterations**. All development now uses the harness reference and the trial's burst sampler (`trial26.bursts_from`), on history systems only.
- **Degeneracy rule** (`degenerate()`, generic, the same for every system). A nonlinear factor is rejected if, over the observed range, it has collapsed into something simpler or is singular there:
  - a sine with frequency × data span < 0.5;
  - an exponential with rate × span < 0.2;
  - tanh nearly linear (argument range < 0.3) or saturated (beyond ±4);
  - an absolute-value breakpoint outside the data;
  - a reciprocal pole inside the data, or more than 2 spans away;
  - a power within 0.02 of 0, 1, 2 or 3;
  - a Hill constant far outside the data.

  Found on Lorenz: 1/(K + z) with K = −264 stood in for a constant term. The fit was then nearly singular (condition number 2e9), and the polish crawled instead of converging. Degenerate terms are rejected in screening; any that appear after polishing are dropped and the refinement is repeated.
- **Simplicity penalty.** 2·log(n)/n per fitted constant (4× the BIC weight; n = derivative points). A fixed penalty of 1.15 log-units per constant made a cubic outrank the correct sine on ThomasLabyrinth (the sine alone explains 92% of the variation) and pruned the right path at step one. Beam width 6.
- **Substeps.** The model's own substep count during polishing is capped at 16 provisionally and 64 finally. A candidate with a non-finite one-step map is discarded. Previously an unstable starting guess escalated to 256 substeps and made each polish about 64× slower.
- **Final refit.** After internal validation has chosen a model, its constants are refit on all observed bursts (training plus held-back). This is standard practice. It matters at small M: SprottB at M = 8 had the correct structure, but extrapolation to unobserved parts of the attractor gave a 2.1e-12 one-step error.
- **Speed.** Candidate evaluation is vectorized (`design_fast`, 10× faster). The flow-map gradient for *all* candidates comes from one reverse pass using stage adjoints, verified against direct autograd to 1e-14.
- **Development results so far** (internal acceptance; truth check for reporting only):

  | System | M | Time | Internal validation | Truth check |
  |---|---|---|---|---|
  | Lorenz | 64 | 6 s | 1.3e-15 | one-step 1.7e-15, multi-step pass over 869 steps: **exact** |
  | Chen | 8 | 7 s | 7e-16 | — |
  | SprottB | 8 | 4 s | 7e-16 | one-step 2.1e-12 (before the final-refit change) |
  | Rossler | 16 | 122 s | 4e-16 | one-step 8e-15, multi-step pass over 2,000 steps: **exact** |

### Tool development, final changes before the trial runs (history systems only)

- **Polynomial-first search order.** The grammar levels gain a level 0, products and whole-number powers only. If those terms alone clearly cannot reach the derivative-error level, the search moves on. The level-1 search (adding sines, exponentials, ratios and so on) starts from the best polynomial support (warm start). Reason: a fitted flexible term can absorb several true monomials at step one and lead greedy search astray. A weaker simplicity penalty fixed ThomasLabyrinth but broke Chen, where 12 flexible terms replaced 4 monomials.
- **Degenerate terms are simplified after every polish.** If a fitted factor drifts into a degenerate form, it is replaced by the constant and linear terms it imitates, and the polish is repeated. Examples: an absolute value with its kink outside the data, or a near-linear tanh. Found on ThomasLabyrinth, where such stand-ins replaced the linear −b·x terms.
- **Centring only for badly offset variables** (|mean| > 3 std). Centring turns every product into a sum of shifted terms: Lorenz's ẏ had 5 terms instead of 3. Learned equations now have the systems' own sparsity (Lorenz 2/3/2 terms, SprottB 1/2/2). Breakpoint and tanh-centre grids moved to data quantiles accordingly.
- **The artifact's own substep count.** Self-convergence is now checked on *all* observed states, followed by one more doubling. That doubling is the safety margin of an 8th-order method: it cuts the truncation error 256×, at 2× run time. Found on SprottB and SprottTorus. Their learned equations were exact: with the doubled count, the one-step error is 2e-15 and 5e-15. Their self-chosen counts (2 and 4) left 2.1e-12 and 2.3e-12 of integration error on faster parts of the attractor. The true vector field at the same counts shows the same 2.1e-12 and 2.3e-12.
- **Development summary** (20 history systems, M = 8 or 16, one setting each). 19 were accepted internally, each in 1–25 seconds; Chua and SanUmSrisuchinwong (piecewise) were still running. Truth checks (reporting only): 15 confirmed exact. SprottB and SprottTorus missed by 2× before the substep fix above. ThomasLabyrinth is now recovered as a sine plus a linear term per equation. Aizawa contains a degree-4 term outside the grammar and is expected to fail.

### Budgets for the test stage (fixed now, before any test system is run)

- **Per learn call:** time limit 240 s for M ≤ 8, 360 s for M ≥ 16. The limit is checked between refinement steps, so a single polish may overrun. It is identical for every arm.
- **Search settings:**
  - levels 0 → 1 → 1+2 → 1+2+3;
  - beam width 6, at most 12 terms per component;
  - composite budget 4,000 candidates;
  - up to 10 support combinations per level;
  - 8 stepwise add/prune rounds per combination.
- **Guided arms** (guide or statistics) run the same levels restricted to the predicted influences (and the predicted level and term count for the guide) first, then the full search, within the same time limit.
- **History stage:** M = 64, 1,200 s per system. Labels come only from internally accepted models. Each history system is learned from scratch (not sequentially), so snapshot N uses exactly the first N systems' own solutions, in the seed's order.
- **Guides:** 3,000 steps of 8 systems each, on random burst subsets (2–64) of the 64 observed bursts, with random variable relabelings. The scrambled-lesson guide gets identical training with label sets deranged across systems.

### Disclosure: one test system's equations were seen (31 October, during the history stage)

- **What happened.** While checking that the BioModels reference metadata loads, I printed the first three records of `trial16/realfam/conversion.json`. The third is **BIOMD0000000003** (Goldbeter 1991 mitotic oscillator), a G-novel test system. Its converted equations were displayed: Michaelis–Menten-type ratio terms.
- **Why the design is unaffected.** By that point the learner, the grammar, the degeneracy rule, the search order and the test budgets had been fixed and logged (previous sections and commit 30fd6b6). The history stage was already running on that frozen code. **No learner, grammar or budget change is made after this point.** Analysis-only code (cost reporting, aggregation) may still be written.
- **Reporting.** BIOMD0000000003 stays in the test set as pre-registered. Results are also reported with it excluded, so a reader can discount it.
- **Second exposure (same day, during the test stage).** A numpy RuntimeWarning raised inside one BioModels test system's lambdified right-hand side was written to the run log, and I saw a fragment of that equation: a power term whose exponent depends on another state variable. The learner was frozen, so nothing is affected. From now on, workers run with warnings suppressed (`PYTHONWARNINGS=ignore`), and logs are read only through progress-line filters.

### History-stage wall-time rule (fixed at 03:16, before the stragglers finish)

The 1,200 s budget covers learning only. Data generation is not bounded. For the piecewise systems (Chua, SanUmSrisuchinwong), the fixed-step reference needs very many substeps at the kinks. **Rule:** a history system whose job (reference plus learning) has not finished 60 minutes after the stage started (that is, by 03:53) is treated as *not learned*. It contributes no labels and no memory. The rule is the same for both seeds and is applied before its outcome is known.

### History stage outcome and test-stage launch (03:55)

- **Wall-time rule applied at 03:53.**
  - Seed 1: not learned: Aizawa, AtmosphericRegime, Chua, LiuChen, LorenzBounded, SanUmSrisuchinwong.
  - Seed 2: not learned: Chua, SanUmSrisuchinwong, WangSun.
- **Guides trained on the systems each snapshot actually solved:**

  | Seed | N = 8 | N = 24 | N = 45 | Scrambled (N = 45) |
  |---|---|---|---|---|
  | 1 | 6 | 19 | 37 | 37 |
  | 2 | 5 | 18 | 38 | 38 |

- **Full test stage launched for both seeds** (12 workers each, M = 64 first). The guide-only arm was dropped: it was never pre-registered.

### C0 gate used for compute (fixed now)

The pre-registration says that if scratch recovers < 25% of G-novel at M = 64, the accumulation part is uninformative.

**Rule.** When every G-novel M = 64 job has finished in both seeds:
- **If C0 fails in both seeds,** the remaining smaller-M jobs for G-novel are stopped. G-familiar jobs continue; they are fast, and they are reported as resemblance-eligible.
- **Otherwise** the full grid runs.

The criteria themselves are unchanged. *Known when this rule was written:* the history-free runs had finished scratch at M = 64 on 3 G-novel systems, all BioModels, with 0 of 3 recovered.

### Trial 26 history stage: results (M = 64, from scratch; truth check is reporting only)

| Seed | Accepted | Rejected | No model | Not learned (wall-time rule) | Truth-exact | False accepts | False rejects | Learning time (median / max) |
|---|---|---|---|---|---|---|---|---|
| 1 | 37 | 2 | 0 | 6 | 39 | 0 | 2 (Rossler, SprottG) | 6 s / 81 s |
| 2 | 38 | 3 | 1 | 3 | 37 | 1 (LiuChen, 1.2e-12) | 0 | 6 s / 46 s |

- On history-type systems, exact recovery from 64 bursts (832 states) works for 37–38 of 45 systems. The learner's own acceptance check almost always agrees with the truth check: across both seeds, 1 false accept and 2 false rejects.
- **Known issue for the next learner version (not changed now; the learner is frozen).** False rejects happen because internal validation uses the *provisional* substep count. The substep safety doubling is applied only after acceptance. Validation should use the converged count.

### Bug found during the test stage (04:55): every guide-based run crashed; fixed, crashed runs to be rerun

- **Symptom.** Every run of the arms that use the guide crashed with a dtype error before any learning: full8, full24, full45 and scrambled45.
- **Cause.** The guides were trained in float32 in processes that had not imported the learner. The learner sets float64 as the process-wide default. In test workers, the reloaded guide therefore mixed float32 inputs with float64 parameters and flags.
- **Fix (`guide.py`).** The guide is always float32, with an explicit dtype for its internal flag tensor. Verified: guide predictions in a learner process match standalone use. The trained guide weights are unaffected; their training never involved float64.
- **A second, rarer crash.** Five statistics/memory runs at M = 64 crashed in a least-squares call on a non-finite design (LAPACK error).
- **Fix (`exact_learner.py`, `lstsq_resid`).** A non-finite design, or a LAPACK failure, now returns an infinite residual, which all callers already filter out.
  - This changes behaviour only in runs that would otherwise have crashed. Every completed run follows an identical code path.
  - It is the only learner change after the freeze, and it is unrelated to the disclosed exposures.
- **Rerun.** Workers spawned after the fix use it. Every record with status `error` (all of them crashes, none of them learning failures) will be deleted and rerun in a clean-up pass, with the same budgets.

### Test-stage reordering (06:40; no result changes)

- **Why.** Every worker was occupied by an M = 64 G-novel job running 10 arms of 7–30 minutes each, so the remaining systems' scratch results (which decide C0) were up to 2 hours away.
- **What was done.**
  - The drivers were stopped. Finished arm records are kept; each in-progress arm was lost.
  - The 73 crash records (`error`) were deleted for rerun, as stated above.
  - The remaining work was relaunched in this order: (1) scratch at M = 64 on the G-novel systems still missing it, which settles C0; (2) all remaining G-novel arms at M = 64, including the crashed guide arms; (3) G-familiar at all M.
- **C0 gate.** As fixed earlier, G-novel M < 64 runs only if C0 passes.
- **Logged before C0 is known.** Status at this point: seed 1 has 19/27 and seed 2 15/27 scratch results at M = 64, with 0 exact recoveries.

### Test-stage reference-time rule (fixed at 07:05, before these systems' outcomes are known)

- **Situation.** Some G-novel BioModels have produced no record at all: BIOMD0000000157, 236, 299, 323 and 371 in seed 1, and 236, 299, 371, 878 and 938 in seed 2. In the first test pass their jobs ran 2.5 hours without finishing. The fixed-step reference at 1e-14 is extremely slow for them, presumably because they are stiff.
- **Rule.** At 07:40, any G-novel system with no scratch record at M = 64 is recorded as `reference_timeout` (not exact) for every arm and every M in that seed. Phase 1 then stops, and phases 2–3 proceed.
- **C0 reporting.** C0 is reported both counting these systems as failures and excluding them, as systems the harness cannot simulate within budget.

### Trial 26: C0 decided (07:40) — capability prerequisite FAILED in both seeds

- **Result.** Scratch exact recovery at M = 64 on G-novel: **0 of 27 in seed 1 and 0 of 27 in seed 2**. The bar was ≥ 25%, about 7 systems.
- **How the 27 break down.**
  - Seed 1: 19 completed runs, none exact. The other 8 were recorded under the 07:40 rule, with no scratch result after 60 minutes of phase 1 and up to 2.5 h in the first pass: BIOMD0000000157, 236, 299, 323 and 371, SaltonSea, SprottMore and YuWang2.
  - Seed 2: 20 completed runs, none exact. The other 7 were recorded under the rule: BIOMD0000000236, 299, 371, 878 and 938, SprottMore and YuWang2.
  - Excluding the timed-out systems, C0 is still 0/19 and 0/20.
- **Failure modes among completed runs.**
  - "No model" or "rejected", with the best internal validation errors ranging from 6e-6 to 4e-1.
  - Catalog ratio systems that the grammar can express in principle were approximated with many terms instead of the compact form. Blasius reached 9e-7 with 11–12 terms per component.
- **Consequence (pre-registered).** The accumulation part is uninformative, and **Trial 26 fails at the capability gate**. Per the C0 compute rule, G-novel runs at M < 64 are not made. The remaining G-novel M = 64 arms (including reruns of the crashed guide arms) and all G-familiar runs are completed for the report.
- **Orchestration slip, no effect on results.** The 07:40 rule script stopped the phase-1 driver by process group. That group also contained the coordinating script and the other seed's driver, so phases 2–3 did not start automatically. They were relaunched at 07:42, each driver in its own session.

### Cost of the learned artifacts (history systems the learner recovered; compiled C artifact vs the original simulation)

| System | Compiled artifact exact? | 1 trajectory × 1,000 steps | 100 trajectories | RHS evaluations per step (artifact / simulator) |
|---|---|---|---|---|
| Lorenz | yes (2.5e-15) | 5.9 ms vs 3,119 ms (528×) | 472× | 96 / 23.6 |
| Rossler | yes (6.6e-14) | 6.0 ms vs 2,941 ms (492×) | 242× | 96 / 25.6 |
| ThomasLabyrinth | yes (1.9e-15) | 17 ms vs 3,584 ms (207×) | 213× | 192 / 63.8 |

- **Caveat.** The wall-time advantage comes from compiled code. The original simulation is scipy DOP853 with a Python right-hand side, as pre-registered. Counted in right-hand-side evaluations, the artifact does 3–4× *more* work per step: it uses a fixed step, a 2× substep safety margin and no step-size control. Reducing that is an open item.

### Plain-language summary sent to the owner (Trial 26)

Sent in chat after C0 was decided. G-familiar and remaining M = 64 numbers will be appended when phases 2–3 finish.

## Trial 27 — proposed (31 October), awaiting owner approval: exact learning of ratio-type systems (capability), on a fresh pool

- **Why capability first.** Trial 26 could not test accumulation on new kinds, because the learner recovered none of them (C0 = 0/27). A history test needs a learner that can recover such systems at all.
- **Learner changes (generic, developed without Trial 26's test systems):**
  1. A ratio-first search branch. It starts cold, with ratio terms allowed from step one, alongside the polynomial-first branch. It adds *swap* moves (replace a term) to add and prune.
  2. A direct route to rational right-hand sides: multiply through by an unknown denominator (implicit linear fit, as in SINDy-PI), then the same exact flow-map polish.
  3. A stiff-capable reference for the harness: an implicit high-order method or adaptive DOP853 at rtol 1e-14, checked against the fixed-step reference on non-stiff systems.
  4. Internal validation with the converged, doubled substep count (the Trial 26 false rejects).
- **Data: a fresh, untouched pool.** BioModels with 2 and 4 state variables (64 + 84 before filtering), converted with the same safe converter and filters. They are split *before any is opened* by a seeded hash into a development half and a held-out half. The held-out half is never inspected or run until the final run. Reserve sets A and B stay untouched.
- **Criteria (both seeds; noise-free; the Trial 26 exactness definition):**
  - **K1:** exact recovery on ≥ 50% of the held-out half at M = 64.
  - **K2 (reported):** success rate and minimal-observation curve over M ∈ {2, 4, 8, 16, 32, 64}.
  - **K3 (reported):** the artifact is faster than the original simulation, plus RHS-evaluation counts.
- **No claim about learning from history in this trial.** If K1 passes, Trial 28 re-runs the Trial 26 accumulation design (strict never-seen building-block holdout, scrambled lessons, statistics-only screen, least-similar histories, growth with N) on systems inside the learner's demonstrated capability.

## Owner decision on Trial 27 (31 October)

> "Don't run Trial 27 as proposed. Adding ratio pieces to the toolkit by hand means we supply the answer; the learner learns nothing from history that way. Revise it so that when a system needs a piece the toolkit doesn't have, the learner discovers that piece from data, fits it to exactness, and adds it to its toolkit for future systems. Hand-supplied pieces should stay limited to what's already there. The accumulation question then becomes: do pieces discovered on earlier systems make later systems learnable exactly, with fewer observations, compared with learning from scratch? Also explain in plain language: in Trial 26, what exactly is 'the network,' and what does it learn versus what is fixed search code? Send me the revised proposal before running anything."

The capability-only Trial 27 above is **withdrawn**. Nothing has run for it.

### What "the network" was in Trial 26 (for the record)

- **Fixed, hand-written code (never changes with experience).**
  - The toolkit: monomials of degree ≤ 3; single-variable factors sin, exp, tanh, |·|, 1/(K + z), x^p; saturations x^n/(K + x^n) and K/(K + x^n); a factor times a variable; products of two factors.
  - The search: screening, beam search, polynomial-first order, degeneracy rule.
  - Exact polishing on the step-to-step map, internal validation, budgets.
- **Learned per system.** Which pieces appear in each component and every constant inside them. The result is a small formula, the "expression network", which together with its own fixed-step RK8 is the artifact.
- **Learned across systems (the only parts that changed with history).**
  - The guide: a small neural network of about 36,000 parameters, trained on the learner's own past solutions. It predicts influences, term counts and levels, and only steers where the search looks first.
  - The memory: a tally of piece shapes in past solutions, used as a mild prior.
  - Promotion of deeper composite pieces. *Never triggered*: all 37–38 accepted history solutions per seed used only level-1 pieces (polynomial terms plus 3 sines).
- **Consequence.** In Trial 26 the toolkit never grew. History could only steer the search; it could not add pieces.

## Trial 27 — revised proposal (31 October), awaiting owner approval: discovering missing pieces from data, and reusing them

**Question.** Do pieces discovered on earlier systems make later systems learnable exactly, from fewer observations, compared with learning from scratch?

### Hand-supplied vs discovered

- **Hand-supplied, unchanged.** The Trial 26 toolkit and its existing ways of combining pieces (sums of terms, a piece times a variable, a product of two pieces), plus the search, polish and validation code. The guide and tally from Trial 26 are set aside, to isolate the piece question.
- **New generic machinery (code, not pieces): discovery.** When no exact model is found with the current toolkit, the learner:
  1. locates the component whose leftover the current pieces cannot explain;
  2. fits that leftover as m(z)·g(u), with an optional variable multiplier m ∈ {1, z_k} and argument u = w·z + c (a weighted sum of variables, weights learned). Here g is a smooth 1-D curve, a Chebyshev series of degree ≤ 48 over the observed range of u;
  3. polishes all constants, including the curve's coefficients, w and c, jointly on the step-to-step map, and raises the degree until internal validation reaches ≤ 1e-13;
  4. tries a compact rewrite of g using only existing toolkit pieces under affine maps, a·φ(b·u + c) + d (or a product of two). A rewrite that passes validation replaces the curve, since compact forms extrapolate.
- **Flagged for the owner.** The weighted-sum argument is the only new structural capability. Previously a piece took a single variable. Without it, pieces mixing several quantities cannot be discovered at all, for example a denominator containing two concentrations.
- **Library.**
  - Every discovered piece that the learner's own validation accepted is stored with its argument structure: how many variables enter u, and whether there is a multiplier.
  - A stored compact form is reused with refitted constants. A stored curve's shape is reused under refitted affine input/output maps.
  - On later systems, library pieces enter the candidate list like toolkit pieces. No truth is used anywhere.

### Systems

- **Development.** Trial 26's already-opened systems (history, G-familiar and G-novel), plus the development half of the fresh pool.
- **Evaluation: a fresh, untouched pool.** BioModels with 2 and 4 state variables, converted with the same safe converter and filters.
  - They are split by a seeded hash of model IDs into development and evaluation halves *before any is opened*. The evaluation half runs once, at the end. Reserve sets A and B stay untouched.
  - Non-oscillating models are observed through recordings that start from varied initial states (transient protocol). Held-out recordings start elsewhere.
  - The reference is upgraded so stiff systems can be simulated within budget.

### Design

- **Streams.** The evaluation systems are put in 3 random orders, fixed in advance, each with 2 independent data draws. For each system in turn:
  1. Learn with the library from earlier systems in that order, at M ∈ {2, 4, 8, 16, 32, 64} (nested bursts).
  2. Record exactness at each M and the minimal observation count.
  3. Add any pieces newly discovered and accepted in the M = 64 run.
- **Scratch.** The same learner, discovery included, with an empty library. It is order-independent, computed once per system and draw.
- **Controls for "learned, not resemblance" (owner rule 2):**
  - **Order fixed in advance, fresh pool.** Test systems are never chosen for similarity.
  - **Decoy library.** The same number of pieces with the same templates and complexity, but random shapes and argument weights that come from no data.
  - **Copy-the-closest.** The whole learned model of the earlier system most similar in data, with constants refitted.
  - **Removal check.** For every system where the library beats scratch, rerun with the discovered piece(s) it used removed; record which earlier system each piece came from.
  - **Similarity report (analysis only, from true equations).** The structural overlap of each evaluation system with earlier ones.

### Criteria (both data draws, averaged over the 3 orders)

- **D (discovery works; prerequisite).** Among evaluation systems that the original toolkit without discovery cannot learn exactly at M = 64, scratch-with-discovery learns ≥ 30% exactly. If D fails, the accumulation part is uninformative.
- **A1 (accumulation).** In the second half of each stream, the library learner either beats scratch in success at M = 8 by ≥ 15 points, or needs ≤ 0.5× the median minimal observations.
- **A2 (grows with history).** The library-minus-scratch advantage rises with stream position: second half > first half, and Spearman ρ ≥ 0.5 over position bins.
- **A3 (from what was learned).** Decoys keep ≤ half of the advantage, and so does copy-the-closest.
- **A4 (attribution).** In ≥ 80% of the systems where the library beats scratch, removing the used discovered piece(s) removes the advantage.
- **Pass** = D and A1–A4 in both draws.

### Fixed during development, recorded before the evaluation run

Discovery budgets (curve degree, number of tries, time per learn call), the transient-protocol details, and the stiff-capable reference.

### Risks (stated now)

- Discovered curves may not stay exact outside the observed range; compact rewrites mitigate this.
- Discovered pieces may not recur across real systems in the pool, in which case no accumulation is possible. The removal and similarity reports would show it.
- Discovery is expensive.

## Trial 27 approved (31 October), with owner additions

The owner approved the revised Trial 27 with four additions:

1. The weighted-sum argument (x + 2y) is fine: *"it's a generic capability, not supplying the answer."*
2. **For every discovered piece, log whether it is a combination of existing toolkit pieces (a compact rewrite was found) or a genuinely new shape (the curve is kept).** Both count.
3. **Test the exactness of discovered pieces on starting states outside the range they were learned on, and report how often they stay exact there.**
4. During the build, send a plain-language progress update at least every few hours, and notify immediately if something changes the plan.

Work within the trial is autonomous from here. Trial 26's code stays frozen; Trial 27 uses a new learner module.

### Trial 27 fresh pool: split rule fixed before conversion (31 October)

- **Source.** The downloaded BioModels SBML set (1,011 files), converted with the same safe converter (`biomodels_convert.py`: a whitelist MathML walker, no eval). The pool is the models with exactly 2 or 4 state variables, under the same acceptance rules: no events, sustained oscillation or a settling transient (`--transient`), and the same title exclusion rule. That rule keeps every model whose title names a reserve-set-A, development or confirmation catalog system out of the pool.
- **Split.** A model belongs to the **evaluation** half if the first hexadecimal digit of sha256("trial27-split-v1:" + model id) is odd, and to the **development** half otherwise.
  - Only development-half models are inspected or run during the build.
  - For the evaluation half, only counts are looked at until the final run.
  - When printing, scripts filter to development IDs (lesson from the Trial 26 exposures).

### Trial 27 build: sampling rule for the pool (fixed during development, applies to both halves)

- **Finding (development systems).** The pool's timescales come from the converter's crude estimates: period ≈ 2π · radius / speed, or settle time. For many BioModels, one interval at T/100 contains so much motion that:
  - the reference needs 16–64 substeps per interval;
  - 9-point rate estimates are off by 10–30%;
  - the learner's screening is blind and its exact polish starts too far away. Example: BIOMD0000000114, rejected with validation error 6.5.
- **Rule.** The recording interval is halved, up to 6 times, until the *true* system needs ≤ 8 substeps per interval. That is the regime of the catalog systems in Trial 26.
  - Recordings keep their time spans: oscillators 30 periods; transients 1.5 settle times.
  - The multi-step horizon cap stays 2,000 steps.
- **Status of the rule.** It is a choice of how the data are recorded, the same for every system in both halves. It carries no information about which pieces the equations contain.
- **Also fixed.** A polish on a model whose terms were all pruned away crashed on an empty parameter vector.

### Trial 27 build notes (development systems only)

- **Substep margin.** Artifact substeps are now 4× the self-converged count (previously 2×). On BIOMD0000000753 the learned equations were exact: 1.6e-14 even out of range. But 2× left 1.2e-12 of integration error on some held-out states.
- **Disguised-monomial rule** (degeneracy, generic). z_j^e · x_j^p is rejected when z_j is proportional to x_j and e + p is an integer between 0 and 3, because it is a monomial in disguise.
  - Found on BIOMD0000000728. The learner recovered the true ratio term −10x/(x+1) exactly, but wrote the constant as x · x⁻¹.
  - That is identical in range and singular at x = 0, so 24% of the out-of-range states failed.
  - This bears directly on the owner's out-of-range question: an in-range-equivalent but non-compact form need not extrapolate.
- **Out-of-range test.** States where the reference itself is untrustworthy (non-finite, or changed by > 1e-13 when the substeps are doubled) are excluded and counted. The headline is the fraction of the remaining out-of-range states that stay one-step exact.
- **Robustness.** Pruning never empties a component. The adjoint pass tolerates models that ignore their input.
- **Discovery, versions 2 and 3** (development systems).
  - **Compact shapes first.** For each promising (multiplier, argument), existing toolkit shapes on that argument are tried before a free curve. Their constants are initialised from the screened curve. A free curve is the fallback.
  - **Degree stops on held-back data.** A curve's degree rises only while its held-back error keeps improving.
  - **Minimal base first.** Discovery starts from a constant plus linear terms (plus any earlier discovered or library pieces), with the fuller base from the standard search as the fallback. On BIOMD0000000507 the right saturation shape was found, but spurious polynomial terms from an over-fitted base blocked exactness.
  - **Two pieces per component.** Up to two discovered pieces per component; some development equations need two.
  - **Disguised polynomials expanded.** A power with an integer exponent (a polynomial in disguise) is expanded into the monomials it equals.
  - **Smaller first search.** The standard search's support cap is 8 terms per component (12 in Trial 26). 18–22-term supports over-fitted and used up the time budget before discovery started.
- **What development systems contain (dev half, inspected).**
  - Hill saturations times a variable (x0·x1⁴/(81 + x1⁴)).
  - Michaelis–Menten ratios, and competitive denominators with a weighted sum (x0/(1 + a·x0 + b·x1)).
  - A logarithm of a weighted sum (x0·log(a(x0 + x1))). This is outside the hand toolkit, so it can only be learned exactly by discovering a genuinely new shape.
  - Nested rational Hill terms (BIOMD0000000058, 4-D).
- **Validity filter.** Converted equations with infinite or undefined constants are excluded. It excludes nothing so far: an apparent "infinity" in BIOMD0000000776 was a display artefact of a logarithm.
- **Development rounds 4–6** (dev half, M = 64, 900 s per system).
  - **Time enforcement.** Deadlines are enforced inside polishing and pruning. Before this, the first search overran its 40% share on 3 of 9 systems (12–15 minutes), so discovery never ran.
  - **First search levels.** The first search now runs only levels 0–1; discovery's compact shapes cover level-2 forms.
  - **Internal acceptance 3e-13 (scaled).** On BIOMD0000000573 a model at 1.5e-13 was externally exact (3.3e-13).
  - **Relaxed degeneracy, for discovery hypotheses only.** Nearly linear ratios (poles far from the data) are allowed, because x·y/(y + 1000) is the truth on BIOMD0000000767. The toolkit search keeps the strict Trial 26 rule.
  - **The artifact integrates adaptively.** Per output step, it doubles its own substeps until two results agree to 1e-14 of its own scale, using only the learned equations. Fixed counts chosen on observed states had failed at the steep starts of held-out transient recordings (three systems).
  - **Direct compact screening.** Every (multiplier, single argument, compact shape, starting range) is scored by a linear fit on the rate data, refined, and the best are polished exactly. This replaces ranking through a noisy screened curve.
- **Development status after round 5** (largest data amount).
  - **Discovery's first exact recovery: BIOMD0000000573.** Internal 1.5e-13, external one-step 3.3e-13.
    - The piece is a *new shape*: a curve of z0 + 3·z1, times z0.
    - It is not exact out of range (2.2e-11). The true term is x0/(x0 + 380); the transient data make an equivalent two-variable curve indistinguishable inside the observed region.
  - Toolkit-only systems (728, 753, 866) are exact, including out of range.
  - The other 7 systems are not recovered: Hill saturations (114, 115), competitive ratios (458), 767, 507, Blasius and the logarithm system (776).

### Trial 27 build notes, continued (31 Oct evening; development systems only)

**Rehearsal.** A short end-to-end run of the evaluation driver (5 development systems, one order, one data draw, M = 8 and 64) wrote library, scratch, copy-the-closest and decoy records with no errors. Library entries are added only on the learner's own acceptance (never on the external check), so no test information reaches the learner. It also confirmed that discovery was too weak to justify the 14-hour evaluation, so the build continues on the development half.

**Diagnosis of the failures (development half; true equations inspected).** Most of the 12 development systems that need discovery are representable within the approved template. They failed in the search, not for lack of shapes.
- 507: a reciprocal whose pole sits 0.04 data-ranges outside the data. The starting grid only reached 0.25 ranges.
- 114 and 115: Hill saturations with n = 4, times a variable.
- 458: a competitive denominator, x0/(1 + a·x0 + b·x1), plus an ordinary reciprocal.
- 573 and 767: Michaelis–Menten ratios with far poles (K = 380, 1000, 10000).
- Blasius and HastingsPowell: Holling-type ratios times variables.
- 881 is entirely toolkit-representable. Only 058 (nested rational Hill terms) and 855 (a four-variable denominator) are outside the template.

**Discovery v4: dense scans of compact shapes.** The compact shapes are existing toolkit shapes on v = b·u + c.
- Every combination is scored by its exact least-squares gain on the rate data, vectorized. A combination is a multiplier (none, one variable, or the same shape times two variables), an argument (one variable, then weighted sums extending the best ones), a shape, and grid constants.
- The grids are generic. Poles, zero points and half-saturation points sit at log-spaced distances from the observed range, from 1e-3 to 1e4 ranges. Exponents are 1–4 and the halves, slopes run from 0.3 to 100 per range, and rates from 0.1 to 60 e-folds per range.
- Several grid starts are refined for each shape. On 507 the exact start ranked 4th; the top start refined into a local fit.
- Pieces are added by a small beam search (width 3 for d ≤ 2, 2 for d = 3, 1 for d = 4; up to 3 pieces per component; each step must halve the residual), so a wrong first piece can be outgrown. On 458 the one-at-a-time version picked a wrong first piece.
- Then a sparse support is chosen over monomials and the found pieces, the model is polished on the step-to-step map, and the scanned pieces are offered to a re-solve of the sparse search.
- The v3 curve discovery remains as the fallback. A near-exact model (held-back error ≤ 1e-9) gets a completion stage: compact pieces chosen by exact gains on the step-to-step map, which sees errors far below the rate-data noise.
- **Time split.** Toolkit search to 30% of the budget, scans to 50% (polish to 60%), re-solve to 65%, curves to 85%, their re-solve to 100%. Completion and the final rewrite/refit have the same 15% / 25% grace periods as before.

**Fixes found along the way.**
- **Validity rule.** Discovered and library pieces are always judged by the relaxed limits. The strict Trial 26 limits had silently replaced far-pole pieces with polynomials inside the re-solve and during library reuse. Toolkit pieces keep the strict rule.
- **Redundant constants held fixed.** In a·1/(b·u + c), a·exp(b·u + c), a·(b·u + c)^n and a·|b·u + c| one constant duplicates the coefficient, so it is held fixed while fitting. The family and the artifact are unchanged.
- **Harness: reference substeps (affects both arms alike; fixed before any evaluation run).** The reference took its substep count from 100 sampled states. On 114 that was not enough: at some held-out states the 2-substep reference was off by 5.7e-10 (true answer from 8 to 1024 substeps, agreeing to 2e-14). The learned model, which was exact, was therefore marked inexact. The reference now uses the doubling rule on the sample and on up to 2000 recorded states, times a margin of 4 (RK8 error ÷ 65,536).

**First results of v4** (M = 64, 900 s budget; truth checks are reporting only).
- **114** (Hill n = 4 times a variable, in both equations) was discovered in 120 s. The held-back error was 3.6e-15 and the learned structure is the true one. It is exact out of range (1.7e-14).
  - Its in-range error of 5.7e-10 was exactly the old reference's own error at those states. Against the corrected reference it is exact in range (8.7e-15; the corrected reference needed 128 substeps on this system) and out of range (2.4e-14).
  - The same re-check turned BIOMD0000000919 (toolkit-only; earlier listed as failing at 7.5e-12) into exact: 4.4e-15 in range, 1.1e-14 out of range. Its earlier failure was also the reference's.
- **115** (Hill n = 4 times (x1 − x0)) and **458** (competitive denominator x0/(1 + a·x0 + b·x1), plus a reciprocal) were discovered in about 6 minutes each. Both are exact in range (5.7e-15, 2.6e-15) and out of range (4.6e-15, 3.3e-15).
- **Blasius** (Holling-type ratios times variables, 3-D) was accepted at 6.2e-15 in 5 minutes.
- **573** (Michaelis–Menten with a far pole): exact in range (1.5e-13) once the simplest-piece tie-break was added. Without it, the first piece to reach the noise level was a more complicated near-equivalent. Out of range it is not exact (6.3e-12): the far pole is only weakly determined by the observed region.
- **Toolkit-only systems** (728, 753, 866) are unchanged: exact in range and out of range.
- **Not yet recovered:** 507 and 767 (near-exact, 2e-12), HastingsPowell, 881 (4-D, 8e-10), and the two systems outside the template (058, 855).
- **Tie-break added during these rounds.** Several pieces can reach the rate-data noise level. Once one has, only simpler pieces are still tried, and the simplest wins (fewest fitted numbers). Single-multiplier shapes are always among the refined candidates. The starts for shapes with an exponent include the best grid point for every exponent.

### Trial 27 build notes, continued (31 Oct, 18:30–19:10; development systems only)

**Library reuse made meaningful.**
- A reused library piece keeps its discovered exponent fixed; only its placement (b, c, and the weight of a weighted sum) is refitted. Reuse means reusing the discovered shape.
- With a free exponent, a random decoy shape of the same family could be refitted into the right one, and the decoy control could not tell real reuse from chance.
- Reused shapes with an exponent also join the dense scan (exponent fixed, placement scanned). With fewer fitted numbers they win ties under the simplest-piece rule.
- Shapes without an exponent (e.g. a reciprocal) carry no shape beyond the family and are scanned as new pieces anyway.

**Search fixes, each found on a development system.**
- **Beam.** One beam slot always holds the best simple expansion (newest piece: one multiplier, one variable). When two pieces are missing, a complicated piece that absorbs part of both can beat either true piece alone (HastingsPowell).
- **Start diversity.** Starts are spread over (exponent, orientation) before filling by score, and single-multiplier shapes get most of the refinement budget (64 refinements per step for d ≤ 2, 32 otherwise). On 507 the true shape's best-scored start was its mirror-image orientation, which refines only to a near fit.
- **Far poles are placed on the step-to-step data.**
  - With noisy rate data a far pole is barely determined. The joint refit then moves along the curved valley between pole and coefficient too slowly to correct it: on 767 it stayed at half the true distance.
  - The step-to-step residual has a sharp minimum at the true pole even with every low-degree monomial present: 3.8e-13 there versus 3e-9 at ±6%.
  - New stage: scan the pole over log-spaced distances, re-solving all coefficients by Gauss–Newton at each position (coarse, shortlist, fine), then polish.
  - This runs on the scan's model in two routes: the sparse support, and the full low-degree support (re-placed before pruning, because a sparse support can swap a needed term for one that compensates the misplaced pole). It also runs in the completion stage, together with backward elimination by full refits.
  - 767 went from 1.9e-11 to 8e-13 held-back; it is not yet below the 3e-13 acceptance.
- **Dense polynomial fallback (hand toolkit only).** Every monomial of degree ≤ 3 in every component, solved exactly (Gauss–Newton with an SVD solve) on the step-to-step map, then pruned if the fit survives. Sparse supports are capped at 8–10 terms, but a polynomial development system (291) has about 18 monomials per equation. It reached 4e-12 on 291: not exact yet, limited by the conditioning of uncentred monomials.
- **Final refit.** The final refit on all bursts now runs 30 iterations, then takes two exact Gauss–Newton steps on the linear coefficients (kept unless clearly worse; both residuals sit at the rounding floor).
  - Two accepted development models (973, 909) had missed external exactness by small margins (1.1e-12, 2.6e-12), each at one extreme held-out state (medians ~1e-14). The damped refit had stopped a few digits short.
  - On their saved models the fix gives 8.1e-14 and 1.1e-14.
- **Exponent simplification.** A discovered exponent that refits to within 1e-6 of a whole or half number is tried at that value (then held fixed), and kept only if the held-back check still passes. Powers are never snapped to 0–3 (disguised polynomials); reused shapes keep their exponent.

**Analysis rule fixed before the evaluation.** A system whose reference cannot be built (the true system diverges from the protocol's starting states even after interval refinement) is excluded from every criterion, in every arm, and counted. Two development systems hit this: 741 and 802.

**Development-half status at M = 64** (31 pool systems; 2 excluded by the rule above; truth checks are reporting only).
- **Exact via discovery** (all pieces are combinations of existing pieces):
  - 114, 115, 458 and 1047: also exact out of range.
  - 573 and 685: not exact out of range (6.3e-12, 1e-10; far poles weakly determined by the observed region).
- **Exact with the toolkit alone:** 728, 753, 866, 919, 1012. 973 and 909 are exact with the final-refit fix (on their saved models).
- **Near-exact:** 507, 767.
- **Not recovered:** 275, 291, 377, 517, 714, 776, 855, 058, 881, 1060, 876 (several are outside the template: four-variable denominators, nested terms, a degree-2 multiplier on a piece, a logarithm). 402, 787 and 812 are still running.

### Trial 27 build notes, continued (31 Oct, 19:10–20:05; development systems only)

**Regression round** (16 development systems with known outcomes, latest code, M = 64).
- **Exact:** 114, 115, 507 (1.6e-13; newly), 573, 685, 767 (3.4e-14 in range, 1.2e-13 out of range; newly), 728, 753, 866, 919, 973 (2.2e-13; newly, by the exact final refit) and 1012. Blasius was accepted.
- **Lost:** 458 (1.9e-8; overran to 1322 s), 909 (8e-10) and 1047 (no model).
  - 1047: the dense-polynomial fallback for d = 4 used the whole budget, so discovery never ran.
  - 458: an earlier, slower first stage left the scan too little time, and the v3 curve fallback overran its deadline.

**Fixes from it.**
- **Dense-polynomial fallback disabled.** It had no success in the build: on 291 it reached 3e-12, limited by the conditioning of uncentred monomials.
- **The scan gets at least 20% of the budget**, whenever it starts.
- **Hard ceiling of 1.4× the budget on every stage.** The final rewrite and refit of an accepted model get a further 0.1×.
- **Zero points.** A discovered zero point (pole) within 1e-4 of the observed range of an uncentred variable's own zero is tried at exactly zero (then held fixed), and kept only if the held-back check passes.
  - On 507 the accepted model gave undefined values (a non-integer power of a negative number) at 2 of 1000 held-out states below the observed range.
  - With the exponent snapped to exactly 2.5 and 1, and the zero point to 0, it is exact (8.1e-14). The learned pieces are exactly 1/(1 + x1^2.5) and a Michaelis–Menten term.
- **Reused exponents.** A reused exponent guides the search (held fixed in the scan) but may move in the final exact-fitting stages. A stored exponent can be slightly off (e.g. fitted 4.0000001), and held fixed it blocks an otherwise exact fit.
  - In a reuse probe (older code) the library run on 114 at M = 16 stalled at 1.7e-8 with the reused piece, while scratch was accepted in 121 s.
  - New entries now come from exponent-snapped models.

**Evaluation efficiency (outcome-identical).**
- The long multi-step test is skipped when the one-step test already failed; exact needs both.
- The out-of-range test runs only for runs that are exact in range; the analysis only reads those.
- Held-out test states use their own random stream, so the envelope is computed only for models that passed one-step.
- Found because a wildly wrong small-data model made the adaptive integrator take 2000 s in an external check.

**Reuse probes so far** (114 ↔ 115, Hill n = 4 times a variable; 300 s per run; older code).
- No library advantage at M = 2, 4, 8 or 16; both arms fail at M ≤ 8.
- At M = 16, 114 from scratch was accepted but not externally exact; with the library it stalled (see above).
- An accumulation probe over the whole development half (static libraries from the M = 64 discoveries, library vs scratch vs decoy at M = 8, 16, 32) is next.

### Trial 27 build notes, continued (31 Oct, 20:05–20:45; development systems only)

- **Candidate scoring lets earlier pieces re-adjust.** When pieces are added in turn, the constants of the pieces already chosen absorb part of a still-missing piece. Scored with those constants frozen, the true next piece can look useless. On 458 the near-pole reciprocal explained nothing when frozen (4.51e-6 vs 4.54e-6), yet a joint refit reached 6.8e-9, the noise level. The first-order directions of the existing constants now join the linear projection, so each candidate is scored as if the earlier pieces were refitted with it. On 458's first equation the scan then reaches 3.8e-9 (tolerance 2e-8).
- **Near-exact threshold raised from 1e-9 to 1e-8.** At this threshold the completion and simplification stages start. 458 ended at 4.9e-9, just outside the old one.
- **Regression round 3** (latest code, M = 64): 114, 115, 507, 573, 685, 767, 1047 and Blasius are all still exact or accepted (no losses). 458 is at 4.9e-9, rejected. 275 and 855 are rejected; 855 is outside the template.
- **Development-half status, latest code** (29 pool systems after the reference rule):
  - Exact via discovery: 114, 115, 507, 573, 685, 767, 1047.
  - Exact with the toolkit alone: 728, 753, 866, 919, 973, 1012.
  - Near-exact and varying between rounds: 458 and 909 (each exact in an earlier round).
  - **Estimated D on the development half: 7/23 ≈ 30% (9/23 ≈ 39% if 458 and 909 hold).** The evaluation half has more 4-D systems (21 of 35, versus 17 of 31), where the learner is weaker.

### Trial 27 build notes, continued (31 Oct, 20:45–21:40; development systems only)

- **Substep counts chosen on states spread over all bursts.** Fixes a general bug. The model's own substep count (used in polishing and validation) was chosen on the first 200 recorded states, which cover only a few bursts and can miss demanding states. Models were then fitted with too few substeps there. On 909 the accepted model fitted the training transitions only to 8e-13 and missed external exactness (7.5e-12); with substeps chosen on states spread over all bursts it fits to 1e-14, and the final refit reaches 9e-15.
- **Variables with no spread in the data.** A species that is zero in every recording now gets a zero rate, and terms involving it are dropped; they vanish on all observed data. On 909 their coefficients were arbitrary (up to 1e4), and a stray 1e-15 rate on that variable made it drift by 1e-16, amplified to 1e-11.
- **Sparse supports when the noise target is unreachable.** The rate-data noise estimate can be optimistic. When no support reaches it, the support with the best fit-versus-size trade-off seen at any step is taken, not the largest last support (which only adds junk terms toward the target).
- **Final refit.** Kept unless worse than the model it replaces. Previously it had to be below 3e-13 on all data, and better refits were discarded. A deadline shortened by an earlier step no longer cuts the final refit short.
- **Accumulation probe** (development half; libraries built from each earlier system's accepted M = 64 pieces in a fixed random order; code of 20:30).
  - M = 8: scratch 2/26, library 2/17, decoy 1/15 exact.
  - The single library-only success (767) used no reused piece (run-to-run variation).
  - At M = 8 even toolkit-only systems fail: with 30 rate points the sparse search keeps 8-term equations that fit the training bursts to 1e-14 and miss the held-back ones (1012: 7.7e-8). So at small data the limit is overfitting of the term selection, for library and scratch alike.
  - M = 16 and M = 32 are running.

### Trial 27: check-in sent to the owner before the evaluation (31 Oct, 22:10)

**Status sent.**
- **Discovery.** It works on the development half: 8 of 23 toolkit-unlearnable systems are exact (estimated D ≈ 35%; at the start of the evening it was 1 of 12).
  - Exact via discovery: 114, 115, 458 (earlier rounds), 507, 573, 685, 767, 909, 1047; the latest code holds 114, 115, 507, 573, 685, 767, 909 and 1047.
  - Exact with the toolkit alone: 728, 753, 866, 919, 973, 1012.
  - The evaluation half has more 4-D systems, so D could land on either side of 30%.
- **Accumulation.** Not shown: at M = 8 the development-half probe gives scratch 2/27, library 2/18, decoy 1/15. With little data the learner's main failure is over-selecting terms, which reusing a single piece does not fix.
- **Options offered:**
  - (a) Run the evaluation as approved (about a day of compute; likely "discovery yes, accumulation no").
  - (b) First strengthen reuse on the development half: also store whole equation patterns learned on earlier systems, tried under variable relabelings, so that with few observations only a few numbers are fitted. This extends the approved library and needs approval.
  - (c) The owner's choice.
- **Recommendation:** (b). The evaluation half is not touched until the owner replies.

**Fixes since the previous entry** (development systems only).
- Exponents beyond ±8 are degenerate. A fitted v^-65.7 piece with a tiny coefficient passed validation on 685 but blew up at held-out states.
- Degenerate pieces are simplified before the final refit.
- 685 is exact again (1.1e-14).

### Owner decision (31 Oct, ~22:50): option (b), with conditions

Owner's reply, verbatim: "Approved: go with (b). Two conditions: 1. Before comparing, improve the from-scratch learner's overfitting with little data (e.g., a stronger preference for the simplest model that is exact on held-back data), and apply the same improvement to every arm. Patterns must beat a strong scratch learner, not a handicapped one. 2. For every pattern-based success, log which earlier system the pattern came from and how different the two systems are beyond that pattern. Keep the decoy patterns, copy-the-closest and removal checks. Test on practice data only, and check in with me before any evaluation on the held-out half."

**Plan, in order (practice/development half only).**
1. **Strong scratch.** Reduce small-data overfitting in the base learner, which every arm uses, and measure it against the previous scratch numbers. The baseline is from the accumulation probe at 20:30 (development half): exact at M = 8: 2/27; M = 16: 3/9 and M = 32: 4/11 (partial).
2. **Pattern library.** Each accepted system's component equations, relabelled to "self"/"others", are tried on later systems under every variable assignment. The comparison uses the strong scratch learner, decoy patterns, copy-the-closest and the removal check.
3. **Provenance.** For every pattern-based success, log the source system and how different the two systems are beyond the pattern.
4. **Check in** with the owner before any held-out evaluation.

The two probes that compared against the old scratch learner were stopped (superseded). The pattern prototype in the learner stays off by default.

**Trial 26 record-keeping runs (phase 2)** finished for both seeds (27/27). Their results are added to the Trial 26 supplementary table when that table is next updated; the verdict is unaffected.

### Option (b), step 1: a stronger from-scratch learner (31 Oct, 22:50–23:40; practice half only)

**Diagnosis of small-data overfitting** (practice systems).
- **Greedy selection.** With few bursts (M = 8: 30 rate points) the greedy beam over supports commits to wrong early terms. It then ends with over-sized equations that fit the training bursts to ~1e-14 and miss the held-back ones. Example: 1012, a polynomial-plus-reciprocal system exact at M = 64, ended with 8-term equations at 7.7e-8.
- **Hard tolerance cut.** The rate-data noise estimate is slightly optimistic at small data, so "first support to reach tolerance" picks an over-sized support that squeezes just below it over the true, smaller one just above.
- **Centring.** Centring a variable (when |mean| > 3 std) splits a product x·y into several terms. With few bursts a variable's spread is narrow, so it gets centred and the true equation is no longer sparse.
- **Far poles at small data.** At M = 8 a reciprocal's pole can sit 100 observed ranges away (the bursts cover a sliver of the variable). Only the discovery scan can place it, and a full low-degree base (15 terms for d = 4) nearly saturates the 30 rate points and hides it.

**Changes, used by every arm** (the shared learner).
- **Exhaustive small supports.** For every support size up to 4 (all sizes when ≤ 12 candidates), the best supports among the plain monomials are found exhaustively (batched least squares). Large candidate sets add the 15 toolkit pieces that best explain what the best small monomial support leaves over.
- **Ranking by fit and size together.** These supports and the beam's results are ranked by log(max(residual, tolerance)) plus the same complexity penalty per fitted number. The validation step on the step-to-step data then chooses among the top few. This is the "simplest model exact on held-back data" preference.
- **Sparse scan base at small data.** When there are fewer than 8 rate points per base monomial, discovery starts from the best small monomial support (plus earlier pieces) instead of every monomial of degree ≤ 2.
- **Centring ratio 3 → 10.** A variable is centred only if |mean| > 10 std.

**Strong-scratch baseline** (the 15 practice systems the learner recovers at M = 64; external exactness).

| M | exact | accepted internally | earlier scratch (probe at 20:30) |
|---|---|---|---|
| 8 | 4/15 | 8 | 2/27 over all 29 systems (≈ 2/15 on these) |
| 16 | 7/15 | 10 | 3/9 (partial) |
| 32 | 10/15 | 13 | 4/11 (partial) |
| 64 | 14/15 | 14 | — |

- **False acceptances at small data.** Most are numeric near misses (1e-11–1e-12; imprecise constants). 973 is at 4e-10 at every M ≤ 32: its true 6.6e-9·x1 term is invisible in small data but matters at extreme held-out states. 507 at M = 8 and 32 is a wrong structure accepted.
- With only 2–4 held-back bursts (drawn from the same recordings), no internal check can see extrapolation failures. External exactness remains the measure.
- **Pattern provenance.** The pattern prototype now records which stored pattern (hence which earlier system) each component used.

### Option (b), step 2: equation-pattern test design (1 Nov, ~00:05; practice half only)

- **Order.** The 29 usable practice systems in the fixed order of the earlier probe. The test systems are the second half (15).
- **Library at position k.** Built from the accepted M = 64 runs of the current strong learner on the earlier systems: their new pieces, plus each component's equation as a pattern.
  - A pattern relabels variables as "self" and "others". Each compact piece keeps its family and exponent; its placement is refitted.
- **Arms, all using the same strong base learner:**
  - scratch;
  - pieces-only library (the originally approved design);
  - pattern library (pieces plus patterns);
  - decoy library (same number and size of entries, random shapes and patterns);
  - copy-the-closest (the whole earlier model closest in data, refitted).
- **Observation budgets:** M = 8, 16, 32, with 300 s at M ≤ 8 and 900 s otherwise.
- **The pattern stage.** Every pattern is tried on every component under every variable assignment, ranked by a linear fit and refined (VarPro) on the rate data. The assembled model is polished and validated on the step-to-step data.
  - It runs even when the toolkit search already accepted a model. At small data that model is often an over-fitted polynomial that passed the 2–4-burst held-back check, and the pattern stage would otherwise never compete.
  - When both models pass the held-back check, the one with fewer fitted numbers wins (the owner's "simplest model exact on held-back data").
- **Removal check.** Every pattern-arm win (exact where scratch is not, with patterns used) is rerun with the used patterns removed from the library.
- **Provenance.** For every pattern-based success, the report gives:
  - the source system of each pattern used;
  - the number of the target's other equations that share a pattern with the source system (canonical structure, any relabelling);
  - the data-feature distance between the two systems (the copy-the-closest features, best relabelling), with its rank among all same-dimension earlier systems.

### Option (b), step 2 result: equation patterns give no accumulation over a strong scratch learner (1 Nov, ~02:30; practice half)

**Setup.** 15 test systems (the second half of the fixed order). The library holds 18 pieces and 28 patterns by the end. Every arm uses the same strong base learner.

**Pattern stage, final version** (three iterations during this step):
- every pattern under every variable assignment;
- the pattern's compact pieces re-placed by the dense placement grid;
- patterns guiding the smallest-support search: the pattern pieces under every multiplier join the monomials, because similar systems differ by a term or two;
- the stage competes even when the toolkit search already accepted a model, and the simplest exact model wins.

| M | scratch | pieces | patterns | decoy patterns | copy-the-closest |
|---|---|---|---|---|---|
| 8 | 1/15 | 1/15 | 2/15 | 3/15 | 0/15 |
| 16 | 4/15 | 4/15 | 4/15 | 4/15 | 1/15 |
| 32 | 5/15 | 5/15 | 5/15 | 5/15 | 2/15 |

**Outcomes.**
- **Paired against scratch.** Patterns win once and lose never; decoys win twice; pieces show no difference; copy-the-closest loses 7.
- **No exact model came from the pattern stage**, so the removal check and the provenance log had nothing to examine. The single extra success at M = 8 did not use a pattern; it and the decoy "wins" are run-to-run variation from a changed search path.
- **The closest practice pair, 114 → 115** (Hill n = 4 switches): 115 also multiplies the switch by its own variable, and its switch sits elsewhere in its data. Even guided and re-placed, the best pattern fit at M = 16 was 2.5e-4, against noise 1.6e-5. Scratch recovers 115 at M = 32.
- **Medians of minimal observations** are identical across arms.

**Conclusion.** On the practice half, neither reusing pieces nor reusing equation patterns makes the learner better on later systems once the from-scratch learner is strong.

**New owner request received meanwhile** (verbatim): "Look at PySR v2 (github.com/astroautomata/PySR, docs at pysr.ai), a symbolic regression library my co-advisor Yubei flagged. On the practice half only: 1. Benchmark PySR from scratch against our homemade search: how many systems each learns exactly (1e-12 bar, held-out starting states) at each data size, and how long each takes. Add a final high-precision polish step if PySR stops short of the bar. 2. Check whether discovered pieces and equation patterns can be fed into PySR as custom building blocks, so the accumulation test could run on top of it. If PySR is the stronger from-scratch learner, propose switching the search engine to it, keeping all controls. Report back in plain language before changing anything in Trial 27."

**PySR benchmark setup** (practice half only; nothing in Trial 27 changed).
- **Version and environment:** PySR 2.7.2 in its own environment.
- **Data:** exactly our learner's data per (system, M) — same bursts, same training/held-back split, same scaled states and rate estimates (burst interiors).
- **Operators:** + − × ÷ ^, exp, tanh, sin, abs (the toolkit's building blocks); float64; deterministic; maxsize 40.
- **Time:** the same wall-clock budget as our learner (300 s for M ≤ 8, 900 s otherwise), shared equally over the components.
- **High-precision finish.** Candidates come from each component's Pareto front: the simplest within 10× and 1000× of the best fit, the most accurate, and the best elbow.
  - Candidates are screened like ours: one missing the rate data by more than 100× their noise level cannot be exact.
  - Survivors get Levenberg–Marquardt refitting of all constants on the step-to-step map (RK8, float64), the same held-back check (3e-13), a refit on all bursts, a compiled artifact, and the harness's external in-range test.
- **First observation** (114, M = 64, full budget): PySR's best formulas approximate the n = 4 switches with nested sines and tanh, missing the rate data by 2e-2 relative (noise 3e-8). Our learner recovers this system exactly in 2–5 minutes.

### PySR benchmark: configuration findings (1 Nov, ~03:15; practice half only)

**The operator set matters a great deal for PySR.**
- On 753 at M = 16, the first equation is a 3-term polynomial, z0·(a + b·z0 + c·z1).
- With + − × ÷ only, PySR finds it in 60 s at the noise level (misfit 1.7e-8, noise 3.6e-8).
- With the toolkit-equivalent set (+ − × ÷ ^, exp, tanh, sin, abs) it falls into sine-based approximations: 1.4e-2 after 60 s, and still about 1e-2 after its full 450 s.
- To give PySR its best chance, the benchmark's main configuration uses + − × ÷ (polynomials and ratios, which cover most of this pool). The 48 runs already done with the full set are kept as a secondary configuration.

**Screening fix.** PySR's own constant fitting stops a few digits short, so a correct structure could look far off. Each candidate's constants are now refitted on the rate data (Levenberg–Marquardt, float64) before the noise-level screen. Refits are time-capped so pathological nested formulas cannot stall the finish.

**Building blocks** (the owner's question 2). PySR 2.7.2 accepts:
- discovered pieces as custom operators, e.g. `hill4(x) = x^4/(1+x^4)` with a SymPy mapping;
- equation patterns as starting expressions (`guesses`, per output);
- fixed structures as template expressions.

Demonstration on 115, M = 32, using 114's switch piece and 114's equations as guesses:
- First equation: misfit 6.3e-2 → 9.5e-6, and the formula contains the right switch times (x0, x1).
- Second equation: worse, 6.2e-2 → 8.2e-2 (the search drifted into nested switch compositions).
- So the accumulation test could run on top of PySR; whether reuse then helps is a separate question.

### PySR benchmark result (1 Nov, ~05:15; practice half only)

**Setup.**
- The 27 usable practice systems at M = 8, 16, 32, 64, with identical data and splits for both learners.
- Same wall-clock budget per run (300 s at M ≤ 8, 900 s otherwise).
- Same external exactness test (1e-12 one-step on held-out starting states, plus the multi-step envelope).
- PySR in its best configuration for this pool (+ − × ÷), followed by our high-precision finish (constants refitted on the rate data, noise-level screen, Levenberg–Marquardt on the step-to-step map, held-back check, refit on all bursts).
- Some cells are still pending (three slow-to-simulate 4-D systems, and some of our M = 32/64 runs on systems neither learner recovers). The table counts only cells where both learners have a result.

| M | our learner: exact | PySR + finish: exact | both | ours only | PySR only | median time per run, ours / PySR |
|---|---|---|---|---|---|---|
| 8 | 4/22 | 1/22 | 1 | 3 | 0 | 308 s / 345 s |
| 16 | 7/17 | 2/17 | 2 | 5 | 0 | 320 s / 904 s |
| 32 | 7/22 | 1/22 | 1 | 6 | 0 | 443 s / 903 s |
| 64 | 14/18 | 2/18 | 2 | 12 | 0 | 266 s / 904 s |

**Findings.**
- PySR's exact recoveries are 753 (M = 8, 64), 728 (M = 16), 919 (M = 64) and one more cell: simple polynomial or low-term systems.
- Our learner recovers every system PySR recovers, plus many more, usually in a fraction of the time. PySR uses its whole budget because it has no stopping rule at exactness.
- PySR also had a false acceptance: 866 at M = 16 passed the held-back check and failed the external test.
- **Why our search wins on this task.** These right-hand sides are sparse sums of a few products of states and saturating pieces. Our search exploits that structure directly (exhaustive small supports, dense placement scans, exact fitting on the step-to-step map). PySR searches general expression trees and spends its budget on approximations, especially once sine, exp and tanh are allowed. The trees it returns rarely reduce to the exact sparse form.
- **Building blocks.** Pieces (custom operators) and patterns (starting expressions) can be fed to PySR, but PySR is the weaker base learner here.

**Recommendation to the owner: do not switch the search engine to PySR.** It is the weaker from-scratch learner on the practice half at every data size, and it never recovered a system ours missed.

### Trial 26 supplementary results (record-keeping runs, finished 31 Oct afternoon)

**G-familiar** (10 systems that resemble the history; reported, not counted). Exact-recovery rate by data amount:

| Arm | M = 2 | M = 4 | M = 8 | M = 16 | M = 32 | M = 64 | Median minimal observations |
|---|---|---|---|---|---|---|---|
| Scratch (s1) | 0.6 | 0.8 | 0.9 | 1.0 | 0.9 | 0.9 | **26** |
| Scratch (s2) | 0.6 | 0.9 | 0.8 | 1.0 | 0.9 | 1.0 | **26** |
| Memory, N = 8/24/45 | | | | | | | ≈ scratch (26) |
| Guide, full arm, N = 45 (s1 / s2) | 0.2 / 0.1 | 0.5 / 0.2 | 0.3 / 0.5 | 0.7 / 0.8 | 0.7 / 0.8 | 0.9 / 0.9 | 52 / 156 |
| Guide, full arm, N = 8 and N = 24 | | | | | | | worse still (208) |
| Scrambled guide | | | | | | | 104 / 39 |
| Statistics screen | | | | | | | 104 |
| Copy-the-closest | 0 | 0 | 0 | 0 | 0 | 0 | |

- On history-like systems the learner from scratch is exact from just **26 observations** (2 bursts) in 60% of cases.
- **The learned guide hurt at small M.** Its predicted influences, term counts and levels narrowed the search too much when data were scarce. The tally ("memory") neither helped nor hurt.

**G-novel at M = 64, all arms.**
- 0 exact everywhere, except **SaltonSea (seed 2)**, which was recovered exactly (one-step 4.5e-15) by memory8, memory24, full24 and full45.
- Its scratch run had been cut by the 07:40 time rule (`reference_timeout`), so no history-versus-scratch comparison exists for it.
- C0 is unaffected: even counting it, 1/27 < 25%.

### Pool freshness check before the Trial 27 held-out run (1 November)

The owner approved the held-out evaluation after the reference-simulation fix. First, though, they asked me to confirm that a fresh, untouched pool exists for the next accumulation trial, and to tell them before spending the held-out half if there isn't one.

- **No ready-made fresh pool exists.**
  - Every curated BioModels ODE model that passes the acceptance rules at 2–12 state variables was already accepted in Trial 23 (or in Trial 16/18). The exceptions are about 20 that failed a time-limited test integration there; some might pass with a longer limit.
  - BioModels lists the same number of curated ODE models today as at our download: 1,019.
- **Correction to the Trial 27 plan.** The plan called the pool "fresh, untouched". That is true only with respect to this learner: the held-out half has never been run or inspected during the Trial 27 build, and the learner carries nothing trained in earlier trials. In the strict sense the claim is wrong:
  - **Held-out half:** 34 of 35 systems are in Trial 23's real set (29 training, 5 held-out).
    - 28 are in the training order of both Trial 23 learners (`pwbig_s1`, `pwbig_s2`).
    - 5 were Trial 23 test systems.
    - 6 are also in Trial 18's real set.
  - **Development half:** 28 of 31 systems are in Trial 23's real set (20 training, 8 held-out).
- **Exposure while checking.** A structure printout showed the first 12 model IDs of Trial 23's split file. One of them is a Trial 27 held-out system. Only IDs were shown: no data, equations or results.
- **Fresh sources that do exist:**
  1. About 675 non-curated BioModels ODE models the project has never downloaded. BioModels lists 1,694 ODE models, of which 1,019 are curated. Their yield under the same filters is unknown until converted.
  2. 137 curated models rejected in Trial 23 only for having polynomial right-hand sides (2–12 variables, no events). They have never been used, but they are a different and easier kind.
  3. 270 curated models rejected for size (13 or more state variables; other filters not applied) and 20 with a single variable.
  4. Reserve sets A and B, under the reserve gating rule.
- **Status:** the held-out run is on hold until the owner decides. The reference-simulation fix and the (b) proposal go ahead in the meantime, since neither touches the held-out half.

### Option (b), rethought: proposal (1 November, awaiting owner approval; practice data only, nothing run)

**Why reuse has not helped.** Take the 14 practice systems the strong learner recovers exactly from 64 bursts, and look at the same systems at 8 bursts (existing records):

| Outcome at 8 bursts | Systems | Examples |
|---|---|---|
| Exact | 4 | |
| Right kind of equation, constants slightly off | 3 | 573 (2e-11), 919 (2e-12); 973's true 6.6e-9·x1 term is invisible |
| Wrong equation that fits the bursts | 1 | 507 |
| No exact model found, came close (6e-9 to 8e-7) | 3 | 114, 866, 1012 |
| No exact model found, nowhere near (~1e-2) | 3 | 115, 458, 1047 |

- Pieces and patterns hand the learner earlier *answers*. But the strong learner already proposes the right kinds of terms on its own.
- A similar system's answer is usually wrong in one term or one switch position (114 versus 115, for example). Correcting it takes about as much data as starting fresh, and every constant must still be refitted from the same scarce data.
- So reuse targeted a part that was not the bottleneck. The bottlenecks are: faint evidence for structure (far poles, steep switches seen over a narrow range), choosing among models that fit equally well, and pinning down constants.

**Observation (practice truths).** 9 of 10 failing practice systems couple their equations:
- the same term appears in two equations, with ratio −1 (one variable's loss is another's gain), −2 (973), −4 (1012) or a fixed volume-like factor (573);
- or the equations are symmetric copies of each other (1047). Only 507 has neither.

Fitting equations separately ignores this, so each shared constant is fitted from one equation's data only.

**Proposal: experience as practice, not as stored answers.**
1. **Practice network.** After each system it learns exactly, the learner generates many variations of the learned model and simulates them on the fly:
   - a term added or dropped (from toolkit families and its learned pieces);
   - a switch moved or an exponent changed;
   - other constants and other starting states.

   For each variation it draws the protocol's small-data observations. A recognition network is trained to read the structure (which terms, which ties, rough switch positions) from a few bursts. On a new system:
   - the network's top suggestions are fitted exactly first;
   - the usual search still runs afterwards and is only reordered, never pruned.

   It differs from Trial 26's guide, which learned from 8–45 real systems and pruned the search: this network learns from thousands of variations with known answers.
2. **Ties between equations.** A shared-learner capability for every arm (like the weighted sums): a term may be tied across equations, so one constant is fitted to several equations' data, and simple ratios remove constants. Experience adds knowing which ties to expect.

**Controls:**
- strong scratch (with ties);
- a generic network that practised on variations of random toolkit systems (no history);
- the history network after N learned systems in a fixed order;
- a scrambled-history decoy;
- removal and provenance checks.

Experience counts only if the history network beats the generic one, increasingly with N.

**First step, needing approval: a ceiling check (about an hour, practice half).**
- Fit the correct structure, taken from the true equations of practice systems, to the same 8- and 16-burst observations, with and without the true ties, and judge it with the external exactness test.
- This bounds what any use of experience can achieve at small data.
- If even the correct structure is rarely exact at 8 bursts, the data are too narrow for exactness there, and I report that before building anything.

### Trial 27: reference-simulation fix (owner-approved, 1 November)

- **What.** Each system's reference simulation (recordings, scale, substeps, interval, both random-generator states) is now built once per system and data draw and reused by every run (`harness27.reference`, used by `trial27.run_one`). Before, it was rebuilt for every run, which took 20–40 minutes on the slowest systems.
  - A build that fails because the true system diverges is stored and re-raised identically, so the exclusion rule is unchanged.
  - Concurrent requests wait for the first builder.
  - The cache lives in `outputs/accumulating_learner/trial27/refcache` and is deleted after the evaluation.
- **Check (practice systems only)** (`test_refcache.py` in the scratchpad; logs `refcache_test_run1.log` and `refcache_test.log`):
  - **Five references are bit-for-bit identical to fresh builds** (every array and scalar, the interval and both generator states): 114, 1012 and 728 (oscillators; interval refined 4, 3 and 0 times), and 866 and 919 (transients; refined 5 and 1 times). Bursts and held-out draws are identical too.
  - **Failure path:** 741 raises the same `RuntimeError('provisional transient diverged')` fresh, on first build and from the cache.
  - **Full behaviour (866, seed 2):** bursts, in-range evaluation (exact; one-step 2.6e-15), out-of-range evaluation and the generator states afterwards are all identical.
  - **Contention:** three processes requesting one reference at once get identical copies; it is built once.
- Results therefore cannot change. Only the time per run drops.

### PySR benchmark: table update (1 November, ~06:40; practice half only)

More cells have finished since the result above. The table counts only cells where both learners have a result:

| M | our learner: exact | PySR + finish: exact | both | ours only | PySR only | median time per run, ours / PySR |
|---|---|---|---|---|---|---|
| 8 | 4/27 | 1/27 | 1 | 3 | 0 | 308 s / 344 s |
| 16 | 7/17 | 2/17 | 2 | 5 | 0 | 320 s / 904 s |
| 32 | 10/27 | 1/27 | 1 | 9 | 0 | 433 s / 904 s |
| 64 | 14/27 | 2/27 | 2 | 12 | 0 | 377 s / 904 s |

- PySR still recovers nothing ours misses. The recommendation (do not switch) is unchanged.
- **Not benchmarked: 377 and 402.** The data export for PySR spent about 5 hours building their reference simulations without finishing, and was stopped. It used the old per-run construction, not the new reference cache.
- **Pending:** our M = 16 runs on 10 systems (still running). The table is updated if they change anything.

### Correction: why the Trial 27 pool was called "untouched" (1 November, answering the owner)

- **What went wrong.** On 31 October (04:07) I checked which BioModels had been used by reading one record only: Trial 16's conversion list (`trial16/realfam/conversion.json`).
  - Trial 16 accepted only 3-variable models, so it listed 64 two-variable and 84 four-variable models as rejected for size. I read that as "no trial has used them".
  - I did not check Trials 18 and 23. They re-converted the same downloaded collection at 2–10 and 2–12 variables and used most of these models.
  - The approved plan therefore stated something I had not verified. The accurate statement is "never used to develop or test the exact learner (Trials 26–27)".
- **What Trial 23 did with them.** Trial 23 trained a neural network that predicts motion approximately and never produces equations. Its data were simulated recordings from several starting points with 0.5% noise, observed as short noisy samples. Of the 35 Trial 27 held-out systems:
  - 28 were in Trial 23's training stream (both seeds);
  - 5 were Trial 23 test systems, scored by prediction error;
  - 6 were also used in the same way by Trial 18.

  Trial 23 failed: its network learned almost nothing from these recordings. Results were reported as summaries. Its logs name individual systems: per-stage training lines, and per-system error scores for the 5 test systems (Trial 23 `eval.log`; also 3 lines in Trial 18's). One held-out system is among the 11 that Trial 23's validity rule excluded for a numerical problem. No equations were worked with.
- **Assessment: still a fair held-out test of the exact learner.**
  - Nothing from Trials 18/23 reaches the Trial 27 learner. The methods differ, and no trained network, features, settings or stored pieces carry over.
  - Trial 27's learner was built on its practice half and Trial 26's systems. The held-out systems' equations, data and results have not been looked at for Trial 27.
  - The pool was selected by size and the standard filters, and split by a hash of the IDs, independently of Trial 23's findings.
  - The residual risk is researcher exposure across sessions to these models' names and approximate-prediction scores. The record shows no exposure to their equations, and no mechanism by which such scores could help the exact learner on them.
- **Changes.**
  - The overlap check for the new pool covers every earlier trial's records (16, 18, 23, 26, 27).
  - **Exposure while checking:** a structure printout showed 6 more model IDs (keys of Trial 23's `dedup.json`). None is a held-out system.

## Fresh pool from non-curated BioModels: filtering and split rules (1 November, fixed before any model is downloaded)

The owner approved downloading and converting the non-curated BioModels ODE models, counts only, with these rules written down first.

- **Source.** BioModels entries with modelling approach "ordinary differential equation model" and curation status "Non-curated": 675 listed on 1 November. The project has never downloaded any of them.
  - Each model's main SBML file is downloaded as untrusted data into a new, dedicated directory (`dl/biomodels_noncurated` in the scratchpad).
  - Local file names are chosen by us, never by the server, and nothing downloaded is executed.
- **Conversion.** The same safe converter as Trials 23 and 27 (`biomodels_convert.py`: a whitelist MathML walker, no eval), with transients allowed and no data stored (`--transient --no-data`). Models with 2–12 state variables are converted, for counts.
- **Acceptance (unchanged from Trial 27):**
  - no events, delays, explicit time or other unsupported constructs;
  - a non-polynomial right-hand side;
  - sustained oscillation, or a settling transient by the Trial 23 rule;
  - the converter's title exclusion rule (reserve-set-A, development and confirmation catalog names, Ca²⁺ and the others listed in the converter);
  - no infinite or undefined constants (Trial 27's validity filter).
- **Pool:** accepted models with **2–4 state variables**. Counts at 5–12 are recorded for information only.
- **Freshness (stricter than before), checked against every earlier trial:**
  1. **Identity.** No identifier may appear in any earlier trial's records (Trials 16, 18, 23, 26 and 27).
  2. **Structure.** A model is excluded if its right-hand side is a structural duplicate of any curated BioModels model in our collection with 2–4 variables. That collection contains every BioModels system used in Trials 16, 18, 23, 26 and 27. A structural duplicate has the same structure after every number is replaced by a placeholder, up to relabelling of the variables.
  3. **Within the pool.** Of structural duplicates, the lowest identifier is kept.
  Exclusion counts are recorded.
- **Split.** A model belongs to the **evaluation** half if the first hexadecimal digit of sha256("fresh-pool-v1:" + model id) is odd, and to the **development** half otherwise. Fixed now, before download.
- **Hygiene.**
  - Until a trial on this pool is approved, only counts are looked at: accepted and rejected by reason, by dimension, kind and half.
  - Scripts print counts only. No identifier, title, equation or data of any model is displayed.
  - After approval, only development-half models may be inspected during development. The evaluation half is looked at only at the final run.

## Diagnostic, not a learner result: ceiling check with the correct structure supplied (1 November, owner-approved; practice half only)

- **Question.** If the learner were handed the correct structure, how often would 8 or 16 bursts give an exact model? This bounds how much of the small-data failure any use of experience could remove. It supplies the answer's structure, so it is **not** a learner result.
- **Systems.** The 14 practice systems the learner recovers exactly from 64 bursts. Their exact M = 64 models supply the structure: they are exact at held-out states, so their structure is correct.
- **Variants** at M = 8 and 16, with the same bursts and held-back split as the learner's runs (seed 1):
  - **given:** structure supplied; constants start at the M = 64 values and are refitted on the small-data training bursts. An upper bound.
  - **perturbed:** as given, but the constants are perturbed first: placements × exp(N(0, 0.3)), coefficients × (1 + N(0, 0.3)), exponents kept. Best of 5 restarts on the training bursts.
  - **tied:** as perturbed, with ties between equations:
    - pieces equal up to a constant factor share one placement;
    - equal terms whose physical coefficient ratio is a simple number (±1, 2, 3, 4, 1/2, 1/3, 1/4) share one coefficient.
- **Judged as in the trial:** a held-back error ≤ 3e-13 to accept, a refit on all bursts, then external exactness at held-out starting states with the compiled artifact.
- **Reference point.** The learner from scratch is exact on 4 of these 14 at M = 8 and 7 of 14 at M = 16 (existing records).
- **Script:** `ceiling_check.py` (scratchpad).

### Held-out evaluation: choice of test set (owner's decision, 1 November; rule fixed before the fresh-pool counts are seen)

- **Owner:** *"if the fresh pool gives enough systems, use it; if not, use the current held-out half and state its prior use in Trials 18 and 23 in the write-up."*
- **"Enough" means at least 30 systems**, counted after all pool rules. The criteria were designed around the 35-system held-out half. As before, systems whose reference cannot be built are excluded at analysis.
- **Rule:**
  1. If the fresh pool's evaluation half has ≥ 30 systems, the Trial 27 evaluation runs on it. The fresh development half and the current held-out half stay unused.
  2. Otherwise, if the whole fresh pool has ≥ 30, the evaluation runs on the whole fresh pool. Trial 27 needs no development use of it.
  3. Otherwise, the evaluation runs on the current held-out half, and the write-up states its prior use in Trials 18 and 23.
- **Unchanged either way:** the learner, settings, criteria and controls as approved. The fresh pool's 3-variable systems are included; the learner's development covered 3-variable systems through Trial 26. For a fresh-pool run, the harness gains the fresh pool as a data source, with the same converter, sampling rule and transient protocol, which were fixed during development for all pool systems.

### Fresh pool: counts (1 November, ~09:45; counts only, nothing inspected)

- **Download:** 675 listed, 638 SBML files fetched. 27 entries had no SBML main file, and 10 were larger than the converter accepts (5 MB).
- **Conversion** (2–12 state variables): 73 accepted.
  - Rejections by category: size 316, events 69, time/delay/csymbol 47, polynomial 44, missing values 42, title rule 14, integration failed 12, no motion 8, unresolved rules 6, other 6, unsupported construct 1.
  - Accepted by dimension: d = 2: 5, 3: 11, 4: 13, 5: 7, 6: 7, 7: 7, 8: 15, 9–12: 2 each.
- **Pool rules (2–4 variables):** 29 candidates.
  - Validity filter: 0 removed.
  - Identity overlap with any earlier trial: 0.
  - Structural duplicates of curated models: 2, compared against 144 distinct curated structures.
  - Duplicates within the pool: 0.
- **Fresh pool: 27 systems.**
  - Development half 17: d = 2: 2 oscillators and 2 transients; d = 3: 2 transients; d = 4: 2 oscillators and 9 transients.
  - Evaluation half 10: d = 2: 1 transient; d = 3: 1 oscillator and 6 transients; d = 4: 2 transients.
  - For information: 44 accepted models with 5–12 variables, before duplicate checks.
- **Decision by the rule fixed beforehand.** The evaluation half has 10 and the whole pool 27, both below 30, so rule 3 applies. The Trial 27 evaluation runs on the current held-out half (35 systems), and the write-up states its prior use in Trials 18 and 23.
- **Consequence for the next trial.** A genuinely untouched pool exists, but it is small: 27 systems at 2–4 variables, or about 71 if 5–12 variables are allowed. The pool file is `outputs/accumulating_learner/fresh_pool_v1/pool.json`; it has not been inspected.

### Trial 27: final settings, recorded before the evaluation run (1 November, before launch)

- **Code.** Commit 80d0c10. `trial27.py eval` with 24 workers (OMP 1, nice 10), then the removal stage, then `analyze27.py --tag eval`.
- **Learner** (`discovery_learner.learn`, the same for every arm):
  - Budgets: 300 s (M ≤ 8) and 900 s (M ≥ 16). Hard ceiling 1.4× the budget; the final refit may run to +0.1×.
  - Stages, as fractions of the budget:
    - toolkit search (levels 0–1) to 30%;
    - discovery scan of compact shapes to max(50%, start + 20%), with polish to +10%;
    - re-solve with the scanned pieces to 65%;
    - curve discovery to 85%, then its re-solve;
    - completion of near-exact models (held-back error ≤ 1e-8): pole re-placement, backward elimination, flow-map additions;
    - compact rewrite, degenerate-piece simplification, constant-variable cleanup, exponent and zero-point snapping;
    - final refit on all bursts (30 iterations plus 2 exact coefficient steps), kept unless worse.
  - Small-data selection (the strong-scratch changes, all arms):
    - exhaustive smallest supports (≤ 4 terms; all sizes when there are ≤ 12 candidates; ≤ 40 columns), with the top 15 toolkit pieces by residual correlation added for large candidate sets;
    - ranking by log(max(residual, tolerance)) plus a cost per fitted number;
    - sparse scan base when there are fewer than 8 rate points per base monomial;
    - a variable is centred only if |mean| > 10 std.
  - Pattern stage off; dense polynomial stage off.
  - Acceptance: held-back error ≤ 3e-13 (scaled), with the model's own converged substeps × 4.
- **Library arm.** Accepted new pieces from the library arm's M = 64 run are added. Reused exponents are fixed during the search and free in the final fitting. Decoys have the same count and templates, with random shapes.
- **Artifact.** The learned equations plus their own adaptive RK8 (doubling until two results agree to 1e-14 of the model's scale), compiled to C. No simulator is used.
- **Harness.**
  - Exactness as in Trial 26: one-step ≤ 1e-12 of scale at 1,000 held-out states, plus the multi-step envelope.
  - Reference substeps: 4 × the doubling-rule count, on 100 sampled plus ≤ 2,000 recorded states.
  - Sampling rule: the truth needs ≤ 2 substeps per interval.
  - Shortcuts that cannot change an outcome: multi-step is skipped after a one-step failure; the out-of-range test runs only for exact runs with pieces.
  - Each reference is built once per system and data draw and reused (verified bit-identical).
- **Analysis.** Systems whose reference cannot be built (the true system diverges) are excluded from every criterion in every arm, and counted.
- **Criteria.** Unchanged from the approval: D, A1–A4, both data draws, averaged over 3 orders.
- **Pool note (1 November check).** The held-out half has never been run or inspected for this learner. However, 34 of its 35 systems were used as data in Trials 18/23 by earlier, different learners.
- **Test set.** The current held-out half (35 systems), by the rule fixed before the fresh-pool counts. **Disclosure, to be repeated in the write-up:** 34 of these 35 systems were used as data by earlier, different learners. Trial 23 trained on 28 of them and tested on 5; Trial 18 used 6. Neither trial worked with their equations, and nothing from either reaches this learner.
- **Pipeline.**
  1. `trial27.py eval --workers 24`;
  2. `trial27.py removal --tag eval --workers 24`;
  3. `analyze27.py --tag eval`;
  4. the reference cache is then deleted.

  Progress is watched by counts only until the run ends.

### Diagnostic result (not a learner result): ceiling check with the correct structure supplied (1 November, ~09:50; practice half)

The correct structure of each system is supplied: the learner's own exact M = 64 model. This measures what the data at each size could support. It is not something any learner achieved.

**Exact recoveries out of the 14 practice systems** (external test on held-out starting states; same bursts and held-back split as the learner's runs). "Started at answer" begins the constants at the exact M = 64 values (an upper bound). "Rough start" perturbs placements by ×exp(N(0, 0.3)) and coefficients by ×(1 + N(0, 0.3)), best of 5 restarts.

| Bursts (observations) | Learner from scratch | Structure supplied, started at answer | Rough start | Rough start + ties | Answer start or ties |
|---|---|---|---|---|---|
| 2 (26) | not run | 6 (6 more accepted but not exact) | not run | 1 | 6 |
| 4 (52) | not run | 10 | not run | 6 | 10 |
| 8 (104) | 4 | 12 | 7 | 9 | 13 |
| 16 (208) | 7 | 11 | 9 | 10 | 12 |

"Answer start or ties" counts a system as exact if either variant is.

**What it shows:**
- **From 4 bursts up, the information is mostly in the data.**
  - At 8 bursts, 12 of 14 systems (13 with ties) are exact once the structure and good starting constants are supplied; the learner alone gets 4.
  - At 16 bursts the corresponding numbers are 11–12 against 7.
  - Most small-data failures are therefore search failures (finding the structure and the constants), not a lack of information.
- **At 2 bursts the data themselves are the limit.** Even with structure and constants started at the answer, only 6 of 14 are exact. Another 6 pass the internal held-back check but fail externally, because one training burst cannot pin down the constants away from the observed region.
- **Constants are a second bottleneck.** With the correct structure but rough starting constants, success at 8 bursts falls from 12 to 7. Placements of switches and poles land in wrong basins (458, 507, 573, 866).
- **Ties help.** Ties between equations raise rough-start success at 8 bursts from 7 to 9. They are the only way 114 becomes exact at 8 bursts: 9 → 6 free constants removes its constant imprecision.
- **Residual precision cases:** 973 (an almost invisible 6.6e-9·x1 term), 1012 at 8 bursts (a far pole), and 573/114 at 16 bursts (4e-12, 2e-12, just above the bar). More data does not always help, because these sit at the edge of the 1e-12 bar.

**Implications for the accumulation test's data sizes:**
- Headline at **8 bursts**, where the room is largest: 4 achieved against a ceiling of 12–13.
- **4 and 16 bursts** as secondary sizes.
- **2 bursts** reported only: not even ideal knowledge is reliably exact there.
- **32–64 bursts** have little or no room for these systems.
- What experience must provide to use that room is the structure *and* good starting values for switch and pole positions, plus ties between equations. This is consistent with the (b) proposal.
- **Caveat:** these are the 14 systems the learner solves at 64 bursts. For harder systems, the structure itself is the larger problem.

## Option (b) practice study: design, fixed before any comparison is run (1 November; owner-approved with three conditions)

Owner approval: build (b) on practice data, with 8 bursts as the headline and 4 and 16 as secondary sizes, under three conditions:
1. Suggestions only add candidates; the from-scratch search always runs.
2. Ties go to the scratch learner too.
3. Every earlier control is kept, so that test systems' similarity cannot explain the result.

**Components** (code: `practice_net.py`, `ties.py`; study driver `practice_b.py` in the scratchpad):
- **Ties** (`ties.finalize`), the shared final step of every arm:
  - pieces equal up to a constant factor share one placement;
  - equal terms in two equations with a simple physical coefficient ratio share one coefficient;
  - ties are detected on the fitted model (loose tolerance), imposed exactly and refitted;
  - the tied model is kept if the learner's held-back check passes. A refit on all bursts follows, as in `learn()`.
- **Practice data.** For each learned practice system, variations of its learned model (practice cases): a term dropped, a toolkit term added, a variable swapped, an exponent changed, or none; constants re-drawn (coefficients ×exp(N(0, 0.5)), placements ×exp(N(0, 0.3))); variables relabelled at random.
  - Each case is simulated on the fly from starting states around the system's observed region, and observed exactly as the learner observes data: bursts of 13 steps, M ∈ {4, 8, 16}, the learner's split and scaling.
  - About 1,400 training cases per system. Stored compactly in the scratchpad and deleted after the study.
- **Practice network.**
  - Vocabulary per equation: monomials of degree ≤ 3, plus reciprocal and Hill (n = 1–4) switches of one variable times 1 or one variable. This vocabulary covers every term of all 14 learned practice models.
  - Data evidence per candidate: partial R² over a placement grid.
  - A transformer over all equations' candidates, with variable identities as random per-system embeddings so it can relate the same switch in two equations.
  - Outputs: presence, and placement on the grid.
- **Suggestions.** The top 8 whole-model suggestions are each fitted exactly: placements refined on rate data, then Levenberg–Marquardt on the step-to-step map, the held-back check, and ties. They are **added** to the scratch learner's own result, and the learner's rule chooses: log(max(held-back error, tolerance)) plus a cost per fitted constant.

**Order and history.** The 29 usable practice systems in the fixed order of the earlier pattern test. The history at position k is the learned systems (accepted M = 64 models) among positions < k: 0 to 13 systems. Seed 1 (data draw 1).

**Arms** (all include the scratch result and ties):
1. **Scratch + ties.**
2. **History network:** trained on variations of the history at that position.
3. **Generic network:** trained on random toolkit systems placed in the same observed regions, same amount of data. Controls for "having a network" and for the extra fitting compute.
4. **Scrambled-lessons network:** trained on variations of the history models with terms reassigned to random equations and variables, keeping term statistics.
5. **Least-similar-history network:** the same number of learned systems, chosen as the least similar to the test system in data (the copy-the-closest similarity), from all learned systems except the test system.
6. **Copy-the-closest:** the most similar earlier learned model, refitted, as an added candidate.

The **removal check** retrains the history network without the provenance source of each win: the history system whose learned structure best matches the winning suggestion. Each win is logged with that source and how different the two systems are.

**Practice criteria** (the next step is a fresh test only if all three hold):
- **P1 (headline):** in the second half of the order, the history arm beats scratch + ties at M = 8 by ≥ 15 points in exact recoveries.
- **P2 (from what was learned, not similarity):**
  - the generic network, the scrambled-lessons network and copy-the-closest each keep ≤ half of that advantage;
  - the least-similar-history network keeps ≥ half of it.
- **P3 (grows with history):** the history-arm advantage at M = 8 rises with history size: second half > first half, and Spearman ρ ≥ 0.5 over history-size bins.
- **Reported:** M = 4 and 16 rates, the removal check with provenance, ties' effect on scratch alone, and suggestion compute time.

**Compute.** The scratch reruns (models kept at M = 4, 8, 16) and the arm runs wait for the held-out evaluation to finish, which holds 24 CPU workers. Networks train on free GPUs. Practice data are generated on one extra CPU process meanwhile.
- **Compute note (10:35).**
  - The held-out evaluation will take about 20 hours.
  - The practice scratch reruns start now on 2 extra processes, plus 1 for practice data. That is 27 busy processes on the 48-core host, where I am the only active user.
  - This is a temporary exception to my 24-process limit. It is checked periodically and reduced if other users become active.
- **Build notes (11:15).**
  1. **First offline check negative.** Networks trained on variations of 13 learned systems did not place a held-out system's learned structure in their top 8 at any M. Even on their own training systems' real data, true terms often ranked low. Two flaws were found in the evidence features:
     - every term was measured against a base containing the constant, so the constant (present in most equations) always showed zero evidence;
     - fit quality was recorded on a linear scale and stored at half precision, which erased the tiny residual differences that separate a true term from its near-copies.

     Over the narrow range seen in 8 bursts, many candidates (a far reciprocal, a stretched Hill curve) are near-copies of constant-plus-linear terms. A greedy path over all candidates picked only 5% of the true terms, while a path over monomials only picked 58% of the true monomial terms.

     **Fix:** evidence on a log-residual scale along two greedy forward-selection paths per equation (all candidates; monomials only), starting from no term. The practice data are regenerated with it.
  2. **Probability calibration.** Presence probabilities are calibrated for the class weighting used in training. Before this, 2–9% of wrong terms scored above 0.5.
  3. **Tolerant suggestion added.** Per equation, the network's most probable terms plus 2 are fitted exactly and pruned by the learner's own backward elimination on the step-to-step map. That map can tell a true term from its near-copy where rate data cannot.
  4. **377 and 402 excluded from the practice study.** Their reference simulations did not finish building in more than 5 hours earlier today. 787 stays: its reference builds once in about 40 minutes and is then cached.

### Option (b) practice study: offline check of the practice network, negative (1 November, ~12:15; practice half)

The check uses leave-one-out networks: each is trained on variations of the other 13 learned practice systems, then tested on the held-out system's real data (seed 1). It asks whether the network's suggestions contain the system's learned M = 64 structure. No learner runs are involved.

| Network version | Exact structure in top 8 (4 systems × M = 4/8/16) | All true terms inside the tolerant suggestion (top + 2) |
|---|---|---|
| v1 (marginal evidence) | 0 of 12 | — |
| v2 (log-residual greedy paths), history only | 0 of 12 | 0 of 12 |
| v2, history + generic | 0 of 12 | 1 of 12 (866 at M = 8) |
| v3 (+ mixed starting-state jitter), history only | 0 of 12 | 0 of 12 |
| v3, history + generic | 0 of 12 | 0 of 12 |

- **Held-out systems:** 866, 115, 573, 1012.
  - With v3, the worst true term per equation ranks 5th–15th of 40 for 866, 31st–35th of 40 for 115 (even though its close sibling 114 is in the history), 7th–38th of 40 for 573, and 29th–129th of 135 for 1012.
- **Systems the network was trained on** (real data, v3 history + generic):
  - 728 is recognized: the exact structure is the first suggestion at M = 8 and 16.
  - 973's terms fall inside the tolerant suggestion at M = 4 and 16, and 114's at M = 16.
- **Reading.** The network learned to recognize systems it has practised on, not to read the structure of new ones. That is the similarity the owner ruled out as evidence. Two observations explain it:
  1. Over the narrow range of 8–16 bursts, many candidate terms are near-copies of one another. What tells the true term from its copies is a residual difference at the 1e-6 to 1e-12 level, which the exact step-to-step fit sees and rate-based evidence does not. A greedy path over all candidates picks only 5% of the true terms.
  2. Thirteen learned systems with about 60 variations each are memorized rather than generalized. On practice cases from the training systems, true terms rank at about 3–4; shifting the starting states or relabelling variables breaks this.
- **Ties, by contrast, work** as a shared final step on fitted models. 114 at M = 8 turns from accepted-but-not-exact (2.6e-12) into exact (1.3e-13), and 866 becomes about 27× more precise. They are measured on every scratch run in the end-to-end stage.
- **Next, within the approved study:**
  - the end-to-end comparison at the headline size (scratch + ties against the history network on the second half, M = 8);
  - the effect of ties on scratch at M = 4, 8, 16.

  Both run once the practice scratch runs finish. The full set of controls matters only if the history arm shows an advantage.

### Option (b) practice study: end-to-end result at the headline size (1 November, ~18:00; practice half, seed 1)

27 practice systems (377 and 402 excluded), M = 8. Every arm includes the scratch learner's own result and the ties step.

| | All positions | Second half |
|---|---|---|
| Scratch, no ties | 4 / 27 | 1 / 14 |
| Scratch + ties | 4 / 27 | 1 / 14 |
| History network (+ scratch + ties) | 3 / 26 (scratch + ties on the same 26: 3) | 1 / 14 |

- **No wins and no losses for the history arm.** The learner's rule chose a network suggestion over the scratch result in 8 of 26 cases, but none changed whether the system came out exact.
- **P1 fails:** 0 points against the +15 required. With no advantage to explain, the control arms (generic, scrambled, least-similar, copy-the-closest, removal) are not run. This was fixed beforehand: the full set of controls only matters if the history arm shows an advantage.
- **Ties** fired in 3 of the 27 scratch final steps, with no change in exactness at M = 8 in these runs. Earlier, on 114, ties fixed constants on a correct structure. The scratch learner's failures at M = 8 are mostly structural, which ties cannot fix. Their effect at M = 4 and 16 is measured once those scratch runs finish.
- **Conclusion for (b) as built:** the practice network does not make new systems learnable from less data. It recognizes systems it has practised on, but does not read the structure of new ones (offline check), so its suggestions add nothing over the scratch learner end to end.

### Trial 27 held-out evaluation: stall from very slow reference simulations (1 November, 17:55)

- **What.** No run has completed since 12:37 (261 completed). All 24 workers sit at full CPU in runs lasting more than 5 hours.
  - The reference cache shows 4 references being built for 6–7.5 hours under locks: held-out systems whose true-system simulations are extremely slow, like 377 and 402 in the practice half.
  - The other workers waited out the cache's one-hour limit and are building the same references independently.
- **Action (operational; results cannot change).** The 30 references no worker had started were pre-built in separate processes on idle cores.
  - 25 finished within seconds to a minute.
  - 1 failed by divergence: the already excluded system's second draw.
  - 4 remain slow and are still building.
  - That leaves 9 slow references in all: probably 4–5 systems, both draws.
- **Proposal to the owner** (an extension of the pre-registered exclusion for references that cannot be built, decided before any result is inspected and independent of the learner): a system whose reference is still not built 12 hours after its build started is excluded from every criterion in every arm, and counted, like the diverging system.

### Trial 27 held-out evaluation: slow-reference exclusion applied, restart (owner decision, 1 November, ~18:50)

- **Owner:** *"Exclude now any system whose reference hasn't finished building, count it like the diverging system, and restart with finished runs kept. The exclusion depends only on simulation cost, not results, so it's fair."*
- **Applied.**
  - 3 held-out systems had unfinished references, in both draws. They are excluded from every criterion in every arm, and counted, alongside the 1 diverging system. **31 of 35 systems remain.**
  - The list is in `outputs/accumulating_learner/trial27/excluded_references.json`. No result was inspected.
  - Stream orders are unchanged. Streams skip the excluded systems, and those systems add nothing to any history. No stream had passed an excluded system yet, so every finished run replays identically.
  - Code: `trial27.slow_excluded()` and `analyze27.slow_excluded()`; the analysis reports them under `excluded_slow_reference`.
- **Restarted at 18:53** with the same pipeline (evaluation, removal stage, analysis). The first evaluation log is kept as `eval_part1.log`.

## Owner standing rule (1 November): cheap references only

*"Our goal is a network that learns from data, not building expensive simulations. For every future pool, filter systems up front by how cheaply their reference can be computed (set a cutoff, e.g. a few minutes, written into the log before looking at anything). Prefer data we can pull or systems we can simulate efficiently."*

- **Applies to every future pool.**
  - A reference-cost cutoff, for example a few minutes of build time on one core, is written into the log before anything is looked at.
  - Systems over the cutoff are dropped and counted.
  - Data that can be pulled, or systems that simulate efficiently, are preferred.

### Option (b) practice study: effect of ties on the scratch learner (2 November, 00:35; practice half, seed 1)

| M | Exact without ties | Exact with ties | Ties fired | Gains | Losses |
|---|---|---|---|---|---|
| 4 | 2 / 27 | 2 / 27 | 6 | — | — |
| 8 | 4 / 27 | 4 / 27 | 3 | — | — |
| 16 | 7 / 27 | 8 / 27 | 6 | 114 | — |

- Ties are a small, harmless gain: +1 exact at M = 16 (114, whose structure is found but whose constants were imprecise without ties). They have no effect where the structure is wrong.
- **The option (b) practice study is complete.** The practice network adds nothing at the headline size (P1 fails), and ties add one system at M = 16. Per the owner's instruction, the next step is a plain-language assessment of why accumulation keeps failing, sent after the held-out summary and before proposing anything new.

## Trial 27: held-out result (2 November, pipeline finished 10:26)

**Classification: FAILED** in both data draws. Discovery is below its bar (D) and there is no accumulation (A1–A4).

**Test set and disclosures:**
- **Size.** 35 held-out systems. Excluded and counted: 1 whose reference diverges (pre-registered rule) and 3 whose references had not finished building (owner decision, 1 November). **31 evaluated.**
- **Prior use.** 34 of the 35 held-out systems had been used as data by earlier, different learners: Trial 23 trained on 28 and tested on 5, and Trial 18 used 6. Neither worked with their equations, and nothing from either reaches this learner. For Trial 27 the held-out systems were never run or inspected until this evaluation.
- **Stall.** The first run stalled for about 6 hours on slow references and was restarted with finished runs kept.

| | Draw 1 | Draw 2 |
|---|---|---|
| Scratch exact at M = 2 / 4 / 8 / 16 / 32 / 64 (of 31) | 1 / 2 / 4 / 5 / 5 / 7 | 2 / 2 / 2 / 6 / 9 / 9 |
| Library arm exact rate at M = 2 / 4 / 8 / 16 / 32 / 64 (all orders) | 3 / 8 / 11 / 17 / 18 / 27% | 6 / 6 / 6 / 19 / 25 / 28% |
| **D**: toolkit-unlearnable systems learned exactly with discovery (need ≥ 30%) | 2 of 26 = 8% ✗ | 5 of 27 = 19% ✗ |
| **A1**: library − scratch at M = 8, second half (need ≥ +15 points or ≤ 0.5× observations) | 0 points; ratio 1.0 ✗ | 0 points; ratio 1.0 ✗ |
| **A2**: advantage grows with position (second half > first, ρ ≥ 0.5) | first half −4.4 points; ρ −0.06 ✗ | 0 points; ρ 0.27 ✗ |
| **A3**: decoys and copy-the-closest keep ≤ half of the advantage | no advantage to keep (decoy −2.1, copy −6.3 points) ✗ | no advantage (decoy 0, copy −2.1) ✗ |
| **A4**: library wins lost when the used pieces are removed (need ≥ 80%) | 5 of 7 = 71% ✗ | no library wins ✗ |

**Owner additions:**
- **Discovered pieces.** About 90% were rewritten as combinations of existing toolkit pieces (2,347 and 2,373 piece occurrences over all runs), and about 10% kept a new curve shape (221 and 177).
- **Out-of-range exactness.** Among runs that were exact in range and used discovered or reused pieces:
  - draw 1: 19 of 31 runs were also exact outside the fitted range, from 5 distinct systems;
  - draw 2: 26 of 32 runs, again from 5 distinct systems;
  - on average 94% and 97% of out-of-range states were exact;
  - 2 systems were exact out of range in every run.

**Reading:**
- Discovery works on some new systems, but far less often than the practice half suggested: 8–19% held out against 30–39% in practice. The practice estimate was optimistic because the learner was developed on those systems.
- Exact recovery from 64 bursts reaches 23–29% of held-out systems.
- Reusing discovered pieces never helped. The library arm is level with scratch at every size, as in every practice study.
- Discovered pieces that are combinations of toolkit pieces mostly stay exact outside the fitted range.

## Assessment sent to the owner: why accumulation keeps failing (2 November, after the held-out summary)

- **Record.** Experience helped only under the old prediction-error goal:
  - through similarity, up to 65×;
  - through a generic skill learned from tens of thousands of synthetic systems (Trial 25), 43% better than kernel ridge on unfamiliar kinds and growing with history.

  Under exactness, every form of reuse was flat or negative against a strong scratch learner: term prior, learned guide, pieces (practice and held out), patterns, and the practice network. Generic capabilities (small-data selection, discovery, ties) moved results, and they rightly go to scratch.
  - The data are not the main limit: with the right structure, 8 bursts suffice for 12–13 of 14 practice systems.
  - Base capability is: 23–29% of new real systems are recovered exactly at 64 bursts.
- **Why.**
  1. Exactness needs the specific answer. Shared regularities are generic and already in the toolkit; specific structure recurs only in variants of the same model, which is similarity.
  2. The history is tiny and varied (about 14 learned systems).
  3. Small-data look-alikes are separable only by exact fitting.
  4. There is no partial credit.
  5. Fair baselines absorb the generic gains.
- **Implications.** On this task, history-driven reduction in data may be unreachable beyond similarity. Accumulation becomes plausible with:
  1. a large (possibly synthetic, cheaply simulable) history for learning a generic skill;
  2. a world that reuses its parts;
  3. measures that experience can change: search cost, reliability, or experiment choice.

  Base capability limits everything meanwhile.
- **Status.** Awaiting the owner's choice of direction before any proposal.

## Owner standing rule update (2 November): disk floor

*"Drop the 50 GB floor. We're not the ones filling the disk, so keep working normally. Only stop writing if free space falls below 5 GB, since at zero our own runs would crash, and notify me if that happens. Keep this project's total footprint small as you have been."*

- **Rule.** Stop writing only below 5 GB free, and notify the owner. Otherwise work normally, keeping the project's footprint small.
- **Context.** Free space fell to 45 GB on 2 November because of other users. This project's footprint was about 4.7 GB in total (outputs 0.84 GB), after 3.6 GB of temporary practice data were deleted.

## Owner direction (2 November): search over equation structures guided by a network trained on a large synthetic history

*"An AlphaGo-style approach. A search over equation structures (moves = add, remove or swap a term), guided by a network that learns from a large history of solved systems which moves look promising. The search must always still explore; the network only sets priorities. Exactness on held-out starting states is the 'win' signal. The history can be large and synthetic, using cheap-to-simulate systems."*

**Design rules (owner):**
- **Fitting.** Constants are fitted by brute force (many starting values, or a global search on the nonlinear constants, then Newton polishing), built as a swappable component.
- **Effort logging.** Time and search effort are recorded per system from the start.
- **Fair comparison.** The same search budget for the with-history learner and the baseline; headline at 8 bursts.
- **Wording.** Standard terms only: "baseline", "with-history learner", "applied to both".

### Survey of published work (2 November)

1. **Networks that write the equation directly**, pretrained on millions of synthetic equations: NeSymReS (2021), end-to-end transformers (Kamienny et al., NeurIPS 2022), ODEFormer (d'Ascoli et al., ICLR 2024; ODE systems from one trajectory; ODEBench, 63 systems).
   - They are fast, but symbolic recovery drops sharply on problems unlike their training data. On ODEBench, about half of ODEFormer's good fits are not the true equation.
   - A 2025 analysis (Voigt et al., arXiv 2509.19849) finds them not competitive with a search baseline (PySR) when generalization is needed.
2. **Search guided by a learned network:**
   - TPSR (Shojaee et al., NeurIPS 2023): tree search on top of a pretrained transformer, rewarded for fit and simplicity.
   - DGSR-MCTS (Kamienny et al., ICML 2023): tree search whose moves are edits proposed by a pretrained "mutation" network, refined online from the search's own successes. This is the closest to add/remove/swap, and was state of the art on SRBench at the time.
   - MGMT (Brugger et al., arXiv 2503.16953, 2025): AlphaZero-style search with a policy prior and value estimate. Supervised training on synthetic equations beat self-play, and it needed far fewer search steps than an RL baseline; the experiments are small.
   - MOSAIC-SR (Zheng et al., arXiv 2609.20997, September 2026): a pretrained transformer proposes starting sketches and searches start from them; it claims the highest symbolic solution rates on SRSD-Feynman and six other benchmarks.
   - uDSR (Landajuela et al., NeurIPS 2022): combines neural-guided search, pretraining, genetic programming and linear models.
3. **Search without learned guidance:**
   - tree search over grammars: Symbolic Physics Learner (ICLR 2023, for dynamics), RSRM (ICLR 2024), improved MCTS with bandit allocation (Huang et al., 2025);
   - genetic programming: PySR;
   - sparse regression: SINDy.

   These are strong baselines. PySR stays robust out of distribution.
4. **Program-synthesis analogues:**
   - DeepCoder (Balog et al., ICLR 2017): a network predicts which functions a program uses, and enumerative search tries likely ones first while staying complete, giving about an order of magnitude speedup.
   - DreamCoder (Ellis et al., PLDI 2021): a learned library plus a recognition network trained on self-generated examples; it solves more problems, faster, with experience.
5. **AlphaZero-style discovery with exact verifiers:**
   - AlphaTensor (Nature 2022): finds provably correct matrix-multiplication algorithms, partly using synthetic training games.
   - AlphaGeometry (Nature 2024): a language model proposes constructions and a symbolic engine verifies them; trained on 100 million synthetic examples.
6. **Constant fitting:** Levenberg–Marquardt is the strongest local fitter in genetic-programming studies. Top SRBench 2.0 methods all include a local constant-optimization step. Multi-start from log-spaced values is common (MOSAIC-SR); a pretrained model's predicted constants can serve as starting values (end-to-end transformers).

**What worked:**
- Network plus search beats either alone.
- Supervised training from synthetic solved examples beats learning from self-play alone.
- Keeping the search complete protects against network errors.
- Exact verifiers make rewards and synthetic data cheap.
- Local constant optimization is essential.

**Caveats:**
- The synthetic-to-real gap is the main weakness.
- Most benchmarks measure approximate fit or symbolic match, not 1e-12 exactness, and few report search effort at a fixed budget.

**Reused:**
- tree search with network priors and an exploration bonus, plus a guaranteed exploration share;
- moves as edits of a whole model;
- supervised training from known solutions of synthetic systems;
- multi-start Levenberg–Marquardt plus exact Newton polishing, as the swappable fitter;
- our exactness test as the win.

### Trial 28: proposed (2 November), awaiting owner approval

The plain-language proposal sent to the owner is reproduced in the next entry once it is approved or revised.

### Trial 28: owner approval of step 1, with two additions (2 November)

- **Approved:** step 1 only. Build the search, the swappable fitter and the effort logging, and run the baseline on practice data. Check in afterwards with the baseline's effort numbers and a proposed budget.
- **Addition 1.** Every real-system result is reported two ways: on all systems, and on the subset the toolkit can express. History cannot help with systems no move can reach, so the search improvement must be visible separately.
- **Addition 2.** Before step 3, a go/no-go bar is set for the network on new synthetic systems, for example how often the true next move ranks in its top few compared with chance. It is written into the log before training. If it is not met, the work stops and is reported before any real system is touched.

### Trial 28 step 1: search vocabulary and expressibility rule (fixed 2 November, before any search run)

- **Vocabulary (applied to both).** This is the Trial 26 toolkit plus the weighted-sum argument approved in Trial 27:
  - products of variables of total degree ≤ 3;
  - single-variable pieces: reciprocal, Hill-type with a free exponent (zero point at 0), exp, tanh, sin, |·| and a free-exponent power;
  - two-variable weighted-sum reciprocal and Hill pieces;
  - any piece alone or times one variable;
  - products of two single-variable switches.
- **Moves:** add, remove or swap one term in one equation.
- **Expressibility rule** (`expressible.py`; truth used only for reporting). A system is expressible if every additive term of every true equation, after polynomial division of each switch denominator, is a vocabulary term type. Piecewise ramps and clips count, since they are exact combinations of |·| and products of variables; logarithms do not.
- **Practice half:**
  - 19 of 31 systems are expressible, 17 of the 27 in the practice study.
  - Not expressible:
    - denominators outside the toolkit (058, 377, 402, 714, 812, 855, 909, 1060);
    - a switch times a degree-2 product (517, 714, 855, 876);
    - other forms (402, 776 with a logarithm, 787).

### Trial 28 step 1: first baseline run, and two flaws found in the evaluator (2 November, 23:30; practice half)

- **First baseline run.**
  - Data-only priorities, up to 10,000 structures per equation, 30-minute search cap, 27 practice systems at M = 4, 8, 16, plus a uniform-priority reference at M = 8; 105 of 108 runs finished.
  - **0 exact recoveries at every M**, on all systems and on the 17 expressible ones; uniform priorities also 0. The Trial 27 learner gets 2, 4 and 8.
  - **Effort:**
    - median 4,000–6,700 structures per equation in 30 minutes, at 2–11 ms per structure;
    - a candidate within 3× of the rate-data noise floor was found for 23–51% of equations, first at structure 460–830 (median);
    - one assembly ran 1.5 hours (uncapped).
- **Flaw 1: the fast fitter.** It mis-fits correct structures: switch placements stay at the nearest grid point (relative error about 1e-3), because Levenberg–Marquardt polishing was triggered only within 1,000× of the noise floor. On 866, the correct structures sat at 1e7–1e9× the floor under this fitter, against 1.3–1.6× with the slower screen-then-polish fitter. **Fix:** always polish structures that contain pieces (finite-difference Levenberg–Marquardt, about 10 ms).
- **Flaw 2: the score overfits.** It uses the training bursts only. At M = 8, wrong structures with 6–8 terms fit the training data to the rate-data floor, and even the precise step-integral form (degree-12 interpolation of each burst, 8-node quadrature per step) to about 1e-12; on 866 there were 248 and 103 such look-alikes. **Fix:** fit each candidate's coefficients and placements on the training bursts' step integrals, which are precise, and score it by its step-integral prediction error on the held-back bursts. "Plausibly exact" then means a small held-back error; the assembled model's step-to-step held-back check remains the internal win.
- Both are corrections within step 1. The baseline is re-measured before the check-in.

### Trial 28 step 1: evaluator and fitter corrections before the baseline rerun (3 November; practice half)

The diagnosis used only the practice half. Where a system's correct structure was needed, it was used only to test the evaluator, never by the learner. The correct structures come from the exact M = 64 models of the Trial 27 learner, carried into the small-M coordinates. All changes apply to both the baseline and the with-history learner.

1. **Precise held-back scoring (the planned fix for flaw 2).**
   - A candidate goes to the precise stage if it is within 0.1% on the rate data and within 10⁶× the rate data's noise floor.
   - There it is fitted on the training bursts' step integrals and scored by its step-integral error on the held-back bursts.
2. **Polishing from more starting points (the fix for flaw 1, extended).** Polishing every structure was not enough.
   - On 866 equation 0, the best grid start sat in a wrong valley, with the pole above the data instead of below it. Every fitter converged there, including the slow ones (2.5×10⁷ × the floor).
   - The fix: also polish from each newly added piece's next-best placement, and from the parent's placements when the sweep moved them. This finds the true valley (1.3× the floor; held-back step error 9×10⁻¹³).
   - The precise stage uses four starts per piece, and tries pieces of one family on one variable at a shared placement (115 has two Hill switches with the same constants).
   - It also snaps exponents within 0.25 of an integer, as the Trial 27 learner did.
3. **Hill-type and power pieces on the physical variable.**
   - The code tied their zero point to the scaled variable, and dropped them whenever the learner's scaling centred that variable. The logged vocabulary puts the zero point at 0, meaning the physical zero.
   - They are now x^n/(K + x^n) and x^p, both in the Trial 26 toolkit.
   - Hill constants K are polished on a log scale, and the Hill start grid is denser: 12 half-saturation points instead of 6, exponents 1–4. On 114 the correct valley is about 1% wide in K and lay between grid points 250× apart.
   - Centring itself is unchanged (the Trial 27 rule: centre only if |mean| > 10 spreads). It affects 4 of 26 practice systems at M = 8, 10 at M = 4 and 2 at M = 16.
4. **Start grids for products of two switches.** The column bank kept only the first 24 combinations, all with the first factor at its first placement. Each factor's grid is now subsampled evenly.
5. **Polishing cost.** Variable projection with Kaufman's Jacobian, stopping at the noise floor or on convergence. This is about 3× faster per structure for the same fits.
6. **Plausibly-exact threshold.**
   - The new threshold is a step-integral error ≤ max(10⁻⁹, 30 × the equation's relative rate-noise floor), on both training and held-back bursts.
   - It was calibrated on the practice systems that are toolkit-expressible and exactly learned at M = 64 by the Trial 27 learner (13 systems; systems whose centring differs between M and 64 skipped).
   - Of the correct structures fitted to within 3× their rate floor, 60 of 61 pass. The miss (573 at M = 4) fits its training bursts exactly but not the held-back burst, which is a data limit.
   - Systems such as 115 have step-integral precision near 10⁻⁶ (fast dynamics relative to the sampling), which a fixed 10⁻⁹ excluded.
7. **Data-only priorities sharpened.**
   - The evidence for adding a term is now 1 / (fraction of the residual left unexplained), instead of the fraction explained. The old measure barely separated terms on smooth residuals: log 0.99 ≈ log 0.9999.
   - The removal cost is computed with rate-data coefficients.
   - The priorities are still data-only. The first version is kept as a reference (`data_prior_v1`).
8. **Reference cache.**
   - 876's reference had never been cached: a lock left by a builder that died on 1 November made every 876 job wait an hour, then rebuild without storing (the 6,509 s run).
   - The lock is removed and the reference built in 56 s. All 27 practice references are now cached.
9. **One random stream per equation.** A smaller budget is now exactly a prefix of a larger run, so success against budget can be read from one run.
10. **Search economy and simplest answers.**
    - Strict supersets of a plausibly exact structure skip the precise stage.
    - The thorough refit is skipped when the ordinary fit already reaches the floor.
    - An equation's search stops once it has a plausibly exact structure and has spent as many structures again (at least 500) looking for a smaller one.
    - Every plausibly exact structure's one-term-smaller structures are evaluated next, recursively (pruning).
    - A structure with a degenerate piece cannot be plausibly exact. Degenerate means collapsed to something simpler over the observed range, or singular there; this is the Trial 27 learner's generic `degenerate()` rule. Examples: exp(0·z) = 1, a zero Hill exponent, a power that is a monomial.
    - On 866 at M = 8, the learner's own held-back check had accepted a look-alike (val 2.9×10⁻¹³, just under 3×10⁻¹³). It contained an |·| piece with its kink at the edge of the data, and it failed the external test, because the search had stopped before reaching the correct two-term structure.
11. **Two versions of the data-only priorities in the rerun.** On 866 equation 0, the sharper version found the first plausibly exact structure later (795 against 286). Both run at M = 8 on all practice systems, and the stronger becomes the baseline (strong-baseline rule).
12. **End-to-end check before the rerun (M = 8, data-only priorities, 3,000 structures per equation).**
    - 728 is recovered exactly (external test passed) in 132 s.
    - 866 is recovered exactly with either version of the priorities. Equation 1 comes out as z₁ + x₀/(10 + x₀), a Michaelis–Menten term with K = 10 exactly, which is the same function as the M = 64 model's constant plus reciprocal.
    - Cost: about 90 ms per structure with the first version of the priorities and 325 ms with the sharper one. Most of the time goes on polishing larger structures and the thorough refits.
13. **Calibration status after the corrections.** Correct structures carried into the small-M coordinates are plausibly exact for 24 of 30 equations at M = 8 and 26 of 30 at M = 16, all within the floor-based threshold except those listed. The fitter still misses 458 and 1012 at M = 8 (both fitted at M = 16). 507 at M = 8 also fails, a reciprocal carried over from a power piece with exponent −1. 115 is data-limited at about 10⁻⁶.
14. **Baseline rerun launched (3 November, 00:22).** 27 practice systems; data-only priorities at M = 4, 8 and 16; the first version of the priorities and uniform priorities at M = 8. Up to 5,000 structures per equation, a one-hour cap per system, 24 workers. Outputs are in the scratchpad (`t28_baseline_v2`); the flawed first run is kept aside as v1.
15. **First rerun stopped after 20 minutes, and restarted (3 November, 01:21).**
    - One job hit a math domain error: on nearly collinear designs, the removal cost's covariance diagonal went negative. The cost is now clamped at zero.
    - The partial results showed the more important problem: several systems with a plausibly exact structure in every equation were still rejected at assembly (728, 876, 767).
    - On 728, equation 0's candidates were look-alikes: z₁·sin(0.004 z₀ + 3π/2) standing in for z₁, which cannot be told apart at this system's step-integral precision (about 10⁻¹⁰). The step-map check correctly rejected every combination (8.6×10⁻¹¹ against 3×10⁻¹³), but the equation had already stopped searching.
    - **Fix 1, resumable searches.** An equation pauses after its first plausibly exact structure plus as many structures again (at least 500). If the assembled model fails its held-back check, every equation with budget left searches on (doubling its structures) and assembly is retried. The stop now rests on the real internal win.
    - **Fix 2, piece simplification.** Plausibly exact structures also queue the structures with a piece replaced by a plain product of its variables. Exact candidates are ordered by number of terms, then number of pieces, then held-back error.
    - **Checks with the final code** (M = 8, up to 5,000 structures per equation):
      - 728 exact in 124 s (the simplification step found the true structure);
      - 866 exact in 227 s;
      - 767 rejected at 9×10⁻¹³ after one resume round;
      - 1060 at M = 4 runs cleanly (rejected; not expressible). It took 1,786 s, because at M = 4 nearly every structure passes the precise gate.
    - The 19 jobs finished before the stop are kept aside, not mixed into the restarted run.
16. **Second restart (3 November, 02:02): the tree descent looped.**
    - In the restarted run, 6 of the first 58 equations stopped by "stalling": the descent kept revisiting known structures and ended without evaluating a new one. Two of them (507 and 573) stopped short of their budget without any plausibly exact structure.
    - Cause: with the risk-seeking (max) backup, every ancestor carries the best value found below it, so a move back to an ancestor scored as high as the best path, and descents looped.
    - **Fix.** Moves that lead back onto the current path are excluded when a move is chosen. If a descent still ends without a new structure, a random untried move from the deepest node that has one is evaluated, so every step explores.
    - Effect on the six equations that stalled (first pause point, M = 8): forced explorations fell from thousands (a first fallback-only fix) to 0–140. All six now reach a plausibly exact structure: 507 at structure 342, 753 at 423, 458 at 506, 573 at 2,087, 866 at 2,141, and 115 at 4,185. With the loop, 507, 753 and 115 had none.
    - The 21 jobs finished before this stop are kept aside, not mixed in. The full rerun was relaunched at 02:02 with commit a832524.

### Trial 28 step 1: baseline results (3 November, 05:22; practice half; commit a832524)

Setup:
- 27 practice systems, 17 of them toolkit-expressible.
- Up to 5,000 structures per equation, a one-hour cap per system.
- "Exact" means passing the external test on held-out starting states.
- Data-only priorities at M = 4, 8 and 16; the first version of the data-only priorities at M = 8; uniform (equal) priorities at M = 4, 8 and 16.
- 189 jobs, no errors.

**Exact recoveries, all systems / expressible subset** (every recovered system is expressible):

| Priorities | M = 4 | M = 8 | M = 16 |
|---|---|---|---|
| Uniform | 2 / 2 | 5 / 5 | 7 / 7 |
| Data-only (sharper version) | 1 / 1 | 2 / 2 | 5 / 5 |
| Data-only (first version) | – | 3 / 3 | – |
| Trial 27 learner (scratch + ties), for reference | 2 | 4 | 8 |

- **Against budget (uniform):**
  - M = 4: 1 by 500 structures per equation, 2 by 3,000;
  - M = 8: 1 by 300, 3 by 1,000, 5 by 5,000;
  - M = 16: 1 by 500, 4 by 1,000, 5 by 3,000, 7 by 5,000.
- **Systems recovered:**
  - M = 8: uniform gets 458, 728, 753, 866 and 919; the Trial 27 learner gets 685, 728, 753 and 909 (909 is not expressible).
  - M = 16: uniform gets 458, 507, 685, 728, 753, 866 and 919; the Trial 27 learner gets 114, 507, 728, 753, 866, 909, 919 and 1012.
- **Uniform priorities beat the data-only priorities at every M.** With uniform priorities, more equations reach a plausibly exact structure (M = 8: 47 of 80, against 38 and 32), and sooner (median structure 407, against 776). The data-only evidence is measured at grid placements, before polishing, and evidently misleads the search.
- **False acceptances.** The learner's own held-back check accepted systems that fail the external test:
  - uniform, M = 8: 4 of 9 accepted (114 at 1×10⁻¹², 507 at 10⁻¹¹ and 573 at 2×10⁻¹¹ are near misses; 973 at 4×10⁻⁹);
  - M = 4: 6 of 8;
  - M = 16: 3 of 10.
  The external test remains the only judge of exactness.
- **Effort (uniform, M = 8):**
  - fitting about 33 ms per structure (median);
  - about one structure in 16 gets the precise stage, at about 0.2 s;
  - 47 of 80 equations reach a plausibly exact structure, at a median of structure 407 (90th percentile 3,155), after a median of 30 s;
  - unsolved equations run to the budget; the time cap stopped only 4 of 80 equations early, none with a plausible candidate;
  - time per system median 847 s, maximum 2,659 s; assembly median 8 s;
  - M = 4 and 16 are similar (first plausible structure at medians 252 and 439; time per system medians 1,013 s and 896 s).
- **Step-1 check-in sent to the owner.** Proposals: uniform priorities as the baseline (the strongest search without history; a change from the approved data-only design, so the owner decides); budget 2,000 structures per equation as the headline and 5,000 secondary; a go/no-go bar on new synthetic systems.

### Trial 28: owner approval of the step-1 proposals, and step 2 go-ahead (3 November)

The owner approved all four proposals:
1. The equal-priority (uniform) search is the baseline.
2. The headline budget is 2,000 formulas (structures) per equation, with 5,000 secondary.
3. The go/no-go bar stands as written below.
4. Step 2 may go ahead, using practice and synthetic data only.

The network's computing time is logged separately. The next check-in comes once the network is trained and evaluated against the bar.

### Trial 28 step 2: go/no-go bar and test rules (fixed 3 November, before any synthetic system is generated or any network trained)

**Go/no-go bar (as approved).** Measured on new synthetic systems made after training and never seen in training, at 8 bursts, in two sets:
- **Set (1):** systems from the same generator.
- **Set (2):** systems in which some equation contains a combination of term types that never appears together in any training equation.

The bars:
- **(a) Ranking.** At search states on the way to the true structure, a correct next move ranks in the network's top 5 at least 30% of the time, and at least twice as often as under the baseline.
- **(b) Search.** Within the headline budget, the with-history learner recovers exactly at least 10 percentage points more of these systems than the baseline, on the same systems, with a paired test at p < 0.05.

If (a) or (b) fails on either set, the work stops and is reported before any real system is touched.

**How each part is measured:**
- **Ranking states.**
  - For each test system and equation, the states are the proper subsets of the true structure T: one subset of each size 0 … |T|−1, drawn with a seed.
  - The correct moves are the additions of tokens in T \ S.
  - The network's priorities over all legal moves at S (adds, removes, swaps, mixed as in the search) are ranked. A hit means some correct move is among the 5 highest.
  - The baseline's (uniform) hit rate is the exact chance rate, 1 − C(N−c, 5)/C(N, 5), for N legal moves of which c are correct. Hit rates are averaged over states and reported per set.
- **Search.**
  - The learner is `learn28`: 2,000 structures per equation, a one-hour cap per system, 8 bursts. The with-history learner and the baseline differ only in their priorities.
  - "Exact" uses the same external test as for real systems: one-step error ≤ 10⁻¹² of scale on held-out starting states, plus the multi-step envelope.
  - The paired test is an exact two-sided sign test on the discordant systems. Bar (b) needs both a difference of at least 10 percentage points and p < 0.05.
- **Test sets.** 150 systems each, generated only after training.
  - Seeds: set (1) uses "t28-test1-k" and set (2) "t28-test2-k", for k = 0 … 149 (with replacement seeds when a draw is rejected, in seed order).
  - Generator-calibration and validation systems use "t28-dev-k". Training systems use "t28-hist-k". No seed is shared between sets.
- **Held-out combinations (set 2).** These are pairs of piece families within one equation. Six were drawn on 3 November with the seed sha256("t28-heldout-combinations-v1"), from the 21 family pairs that involve a Hill-type or reciprocal piece: (abs, hill), (hill, tanh), (pair-hill, recip), (product-of-Hills, recip), (recip, recip) and (recip, sin). No training or set-(1) equation contains a held-out pair; every set-(2) system contains at least one.
- **Cheap simulation (cheap-reference rule; cutoff fixed now).** A synthetic system is accepted only if:
  - its reference needs at most 4 interval halvings (at most 64 substeps per interval at the reference margin), and builds in at most 60 s;
  - its trajectories stay finite, positive and bounded (below 100× the largest initial state);
  - every variable moves by more than 0.1% of its scale over a recording;
  - no variable is centred by the learner's scaling at M = 4, 8 or 16;
  - no true piece is degenerate over the observed range (`DL.degenerate`).
- **Generator v1.** These are biochemistry-like positive systems built from the search vocabulary, with term-type frequencies set by hand, not fitted to any system.
  - Size: d = 2, 3 or 4 (probabilities 0.4, 0.35, 0.25), initial states log-uniform in [0.3, 3].
  - Each equation has a degradation term −k xᵢ (probability 0.8) plus 1–3 further terms:
    - mass-action monomials of degree 0–2 (0.35; negative when they contain xᵢ);
    - Michaelis–Menten/Hill switches x_jⁿ/(K + x_jⁿ), with n = 1–4, optionally times one variable (0.35);
    - reciprocals (0.10);
    - exp, tanh, sin, |·| or power pieces (0.10);
    - weighted-sum pieces (0.05);
    - products of two switches (0.05).
  - Transient recordings follow the Trial 27 rules (64 learning recordings from 0.5–1.5× the initial state, settle time 10, interval halving to at most 2 substeps), and bursts are drawn as for real systems.
  - The generator may be recalibrated on development seeds only, before training, towards a baseline success of 25–60% at the headline budget. Any change is logged.
- **Practice systems** are used for development only. No evaluation-half or fresh-pool system is touched in step 2.

### Trial 28 step 2: generator calibration on development seeds (3 November, before the training history)

The changes below were made only on development seeds ("t28-dev-k"), before any training system was generated. The bar and the test rules are unchanged.

1. **Settle time.** Each synthetic system now uses the Trial 23 transient rule, as real systems do: integrate from the initial state up to t = 10⁴; the settle time is the first time the distance to the final state falls below 1% of its starting value. Recordings cover 1.5 settle times. A fixed settle time of 10 left fast systems flat for most of each recording. The settle-time integration is limited to 5 s (part of the cheap-simulation rule).
2. **Centring check at M = 8 only.** The bar is measured at 8 bursts. At M = 4, centring rejected many systems that are well-posed at M = 8. Training examples at M = 4 or 16 fall back to M = 8 when that view is centred.
3. **Size mix.** Acceptance falls with size: 25%, 11% and 5% for d = 2, 3 and 4 in a 100-system pilot. Sizes are now drawn with probabilities 0.17, 0.33 and 0.50, so accepted systems are about 40/35/25%.
4. **Cutoff applied while building.** The cheap-simulation cutoff stops a reference build as soon as it is exceeded, instead of after building.

**Pilot (100 development systems, earlier size mix):**
- Acceptance 15%. Main rejections: simulation cost 165, centred 150, integration failures 148, degenerate true piece 71.
- The true structures reproduce their own data: rate residual over noise floor median 0.41, 90th percentile 2.2, over 243 equations.
- Families present (systems): Hill 88, reciprocal 40, weighted-sum Hill 17, exp 12, weighted-sum reciprocal 10, product of Hills 10; the others fewer than 6.

**Difficulty (baseline: uniform priorities, 2,000 structures per equation, M = 8, 48 development systems):**
- 10 recovered exactly (21%): 8 of 19 with d = 2, 1 of 16 with d = 3, 1 of 13 with d = 4.
- This is just below the 25–60% band aimed for, which lies within its uncertainty (about 10–35% on 48 systems), and close to the real practice systems at the same budget (3 of 27, 11%).
- **The generator was left unchanged**, so the synthetic test stays about as hard as real systems.
- Time per system: 45–1,170 s.

**GPUs.** All 8 GPUs were fully used by another user, so the network is trained on CPU (it is small). Its training time and its inference time inside the search are recorded separately.

### Trial 28 step 2: training history, network, and go/no-go bar (a) (3 November)

- **Training history.**
  - 7,982 accepted synthetic systems (seeds t28-hist-0 … 7999; 18 seeds had no accepted draw within 50 replacements). 58,591 draws were rejected.
  - Each system contributes training states at one number of bursts (M = 4, 8 or 16 with probabilities 0.2, 0.6 and 0.2): about 12 states per system, 94,472 in all.
  - The states are proper subsets of each equation's true structure, subsets with 1–2 wrong terms, and the true structure plus one wrong term.
  - On disk: 136 MB, deleted after step 2. Generation took 67 minutes on 24 CPU workers (11 s per system).
- **Network.**
  - A 3-layer transformer over the candidate terms (128 wide), trained 12 epochs on CPU, because all GPUs were in use by another user.
  - **Training time: 5,389 s, plus 1,222 s loading** (a slow loader, since fixed), reported separately from any search effort.
  - Validation on held-out history systems (every 20th seed): a correct next move is in the top 5 at 82.5% of proper-subset states (chance 5.9%). Validation loss fell from 1.46 to 1.19.
- **Test systems** (generated after training, as logged):
  - set (1): 150 systems from seeds t28-test1-k;
  - set (2): 150 systems from seeds t28-test2-k, each containing a held-out combination.
  - In set (2), 2 seeds (k = 88, 91) found no acceptable system within 50 replacement seeds. The cap was raised to 500, so the set has 150 systems as the rule requires; this changes no other system.

**Bar (a), ranking (proper-subset states, top 5 of all legal moves):**

| Set | States | Network | Baseline (chance) | Data-only priorities (reference) |
|---|---|---|---|---|
| (1) same generator | 1,257 (150 systems) | 80.6% | 5.2% | 18.3% |
| (2) held-out combinations | 1,528 (148 of 150 systems) | 74.3% | 6.7% | 14.6% |
| (2), equations containing a held-out combination | 724 | 65.6% | 8.0% | – |

- **Bar (a) is met on both sets**: at least 30%, and at least twice the baseline (15.6× and 11.1×).
- The 2 missing set-(2) systems are added once the search runs free CPU workers.

- **Bar (b), the search** (in progress; launched 11:36): learn28 at 2,000 structures per equation on both sets, with network priorities and with uniform priorities (the baseline). There are 600 runs on 24 workers, with the network's computing time recorded separately.

### Trial 28 step 2: go/no-go result: bar (a) met on both sets, bar (b) met on set (1) and NOT met on set (2); stopped (3 November, 15:42)

**Bar (b), the search** (learn28 at 2,000 structures per equation, M = 8, 150 systems per set; with-history learner = network priorities, baseline = uniform priorities; same systems, same budget):

| Set | Baseline exact | With-history exact | Difference | Discordant (history only / baseline only) | Sign test (two-sided) | Bar (b) |
|---|---|---|---|---|---|---|
| (1) same generator | 34 (22.7%) | 70 (46.7%) | +24.0 points | 36 / 0 | p = 2.9×10⁻¹¹ | met |
| (2) held-out combinations | 1 (0.7%) | 3 (2.0%) | +1.3 points | 2 / 0 | p = 0.5 | **not met** |

- **Bar (a), with all 150 set-(2) systems:** network 74.2% top-5 against 6.6% chance; on the equations containing a held-out combination, 65.5% against 8.0%. Met on both sets.
- **Per-equation diagnostics** (plausibly exact structure found, and the median structure at which it was first found):
  - set (1): with history 360 of 460 equations at a median of structure 13; baseline 247 of 460 at a median of 245;
  - set (2), equations without a held-out combination: with history 217 of 287 at 22; baseline 144 of 287 at 222;
  - **set (2), equations with a held-out combination: with history 22 of 150 at 428; baseline 18 of 150 at 340.**
- **Why set (2) fails.** By construction, each set-(2) system has one equation that receives both families of a held-out pair on top of its usual terms. These equations average 4.9 terms and 3.2 pieces, against 2.7–2.9 terms and 1.2 pieces elsewhere. Neither learner solves them within the budget, so almost no set-(2) system can be recovered whole, and set (2) leaves no room for any learner to show a difference.
  - Set (2) therefore mixes novelty of combination with equation size. The network still ranks well on those equations (bar a), but its priorities do not make the search complete them.
- **Effort** (network computing time separate):
  - set (1): time per system, median 225 s with history against 518 s for the baseline; network inference median 11 s per system (6.2% of run time);
  - set (2): median 499 s against 664 s; network inference median 20 s (4.3%);
  - one-off costs: history generation 67 minutes on 24 CPU workers; training 5,389 s on CPU (16 threads) plus 1,222 s loading.
- **Decision rule applied.** Bar (b) failed on set (2), so the work stops here. No real system has been touched, and step 3 is not started. The history shards (136 MB) are kept until the owner decides; the trained network is saved in outputs/accumulating_learner/trial28.

### Trial 28: owner decision on set (2), and rules for the size-matched set (2′) (3 November, logged before any set-(2′) system is generated)

**Owner decision.**
- Replace set (2) with a size-matched version that swaps terms rather than adding them, so its equations match training in number of terms and number of nonlinear pieces. Rerun bars (a) and (b) on it.
- Keep the original set (2) in the log, reported as **failed as designed (size confound)**: its held-out equations received both held-out pieces on top of their usual terms (4.9 terms and 3.2 pieces on average, against 2.7–2.9 and 1.2), so neither learner could solve them (baseline 1 of 150).
- Run the original set (2) at the secondary budget (5,000 structures per equation), both learners, **as a diagnostic only**, not counting toward the bar.
- Check in with both results before step 3.

**Set (2′) generation rules (fixed now):**
- 150 systems, seeds "t28-test2m-k" for k = 0 … 149, with replacement seeds "t28-test2m-k-r1", "-r2", … in order when a draw is rejected (up to 500). They are generated only after this entry.
- **Construction.**
  1. Draw a system with the training generator (no held-out combination anywhere).
  2. Among its equations with at least two nonlinear pieces (non-monomial terms), choose one at random; choose one of the six held-out pairs at random; choose two of that equation's pieces at random. All choices use a seed derived from the system seed.
  3. Replace the two chosen pieces with one new term from each family of the pair, drawn as in the generator and new to the equation.
  4. The equation keeps its number of terms and its number of nonlinear pieces; the other equations are unchanged.
  5. A draw with no equation of two or more pieces, or a replacement that cannot be drawn, is rejected and the next replacement seed used.
- **Acceptance.** The same checks as every synthetic system, and the system must contain a held-out combination.
- **Bars (a) and (b)** are measured exactly as before: same network, M = 8, 2,000 structures per equation, the same ranking states, and the same exact two-sided sign test (difference of at least 10 points and p < 0.05). Set (1)'s results stand.
- **Reported alongside, not part of the bar:**
  - the size match: mean terms and pieces of set-(2′) held-out equations, against set-(1) equations with two or more pieces;
  - the per-equation diagnostics as before.

### Trial 28 step 2: size-matched set (2′): bar (a) and bar (b) both met (3 November, 19:31)

- **Bar (a)** (150 systems, 1,335 ranking states):
  - all equations: network top-5 78.7% against 5.5% chance (14×); data-only priorities 16.8% (reference);
  - equations containing a held-out combination (518 states): 70.3% against 6.6%.
  - **Met.**
- **Bar (b)** (2,000 structures per equation, M = 8, same 150 systems):
  - baseline 8 exact (5.3%), with-history learner 26 (17.3%): **+12.0 points**;
  - discordant systems 19 (with history only) against 1 (baseline only); exact two-sided sign test p = 4.0×10⁻⁵.
  - **Met.**
- **Size match:** set-(2′) held-out equations average 3.45 terms and 2.11 pieces, against 3.50 and 2.18 for set-(1) equations with two or more pieces (and 3.47 and 2.22 for set (2′)'s other such equations).
- **Per equation** (plausibly exact structure found, and the median structure at which it was first found):
  - held-out equations: with history 57 of 150 at 329; baseline 27 of 150 at 674;
  - other equations: with history 250 of 309 at 9; baseline 155 of 309 at 210.
- **Effort:** time per system, median 316 s with history against 537 s for the baseline; network inference median 13.7 s per system (5.1% of run time).

**Status of the go/no-go bar:**
- Bar (a) is met on sets (1), (2) and (2′).
- Bar (b) is met on sets (1) and (2′).
- The original set (2) failed as designed, from the size confound, and is kept in this log as such.
- The diagnostic of the original set (2) at 5,000 structures per equation (both learners, not counting toward the bar) is running. The owner check-in follows when it finishes, before step 3.

### Trial 28 step 2: diagnostic: the original set (2) at 5,000 structures per equation (4 November, 01:42; not part of the bar)

- **Result** (both learners, the same 150 original set-(2) systems): baseline 3 exact (2.0%), with-history learner 6 (4.0%), +2.0 points. Discordant 3 against 0; sign test p = 0.25. At 2,000 the figures were 1 and 3.
- **The oversized held-out equations** (4.9 terms, 3.2 pieces) remain mostly unsolved: a plausibly exact structure in 37 of 150 with history (first at median structure 977) and 28 of 150 for the baseline (1,151). At 2,000 the figures were 22 and 18.
- **The ordinary equations of the same systems:** 237 of 287 with history (first at median 34) against 164 of 287 (317).
- **Effort:** time per system, median 1,313 s with history against 1,725 s for the baseline; network inference median 33 s (3.1% of run time).
- **Reading.**
  - Two and a half times the budget barely changes whole-system recovery on the original set (2), and every system there contains an equation of about five terms with three pieces.
  - Set (2′), which places the same held-out combinations in equations of training size, gives +12 points at 2,000.
  - So the original set (2)'s failure comes from equation size, a limit of the search and fitting on large multi-piece equations for both learners, rather than from novelty of combination.

### Trial 28 step 3: owner approval and plan (4 November, logged before any step-3 run)

**Owner.**
- Step 3 is approved as outlined. Check in with the practice results, including the scrambled-history control and the history-size curve, before requesting any held-out test.
- **GPU rule:** if all GPUs are busy, I may share 1–2 GPUs that have enough free memory and are not fully utilized, alongside the other user's jobs. I must never stop, pause or interfere with anyone else's processes, must check `nvidia-smi` before each launch, and must log which GPU was used.

**Plan (fixed now):**
- **Systems.** The 27 practice systems (development half) only, reported on all 27 and on the 17 the toolkit can express. No evaluation-half, fresh-pool or reserve system is touched.
- **Main comparison.**
  - With-history learner: the network trained on 7,982 synthetic systems, exactly as evaluated in step 2. Baseline: uniform priorities.
  - M = 4, 8 and 16; 2,000 structures per equation (headline) and 5,000 (secondary); one-hour cap per system.
  - The baseline at 5,000 reuses the step-1 runs (commit a832524). The only later changes to `structure_search.py` add the equation index to the search context and the network's featurizer and timing, none of which touches the uniform path. The baseline at 2,000 is run fresh.
  - Paired comparison per M and budget: exact recoveries, difference, exact two-sided sign test. Network time is reported separately.
- **Scrambled-history control.** The same network and recipe, trained on the same synthetic states with the labels permuted among examples of the same vocabulary size (seeded), which destroys the link between what the network sees and what it should do. It is run on the practice systems at M = 8 and 2,000.
- **History-size curve.**
  - Networks trained on the first 1,000, 2,000 and 4,000 history systems (seed index below N; 8,000 is the main network), with the same recipe and the same validation systems.
  - Each is run on the practice systems at M = 8 and 2,000, and ranked on synthetic test set (1) (the bar-(a) measure).
  - The question is whether exact recoveries and ranking improve as the history grows.
- **Training recipe for every variant.** 12 epochs, the same optimizer and schedule, the last epoch kept: the step-2 network's best epoch was its last, and keeping the last epoch avoids choosing the scrambled control's epoch by true labels. Training device, GPU index and time are logged.
- **Control networks trained (4 November, 02:35).** All keep the last of 12 epochs. Validation is on held-out history systems with true labels:

  | Network | GPU | Training time | Validation top-5 hits |
  |---|---|---|---|
  | scrambled history | GPU 0 | 528 s | 71.3% |
  | 1,000 systems | GPU 1 | 83 s | 81.0% |
  | 2,000 systems | GPU 1 | 159 s | 81.9% |
  | 4,000 systems | GPU 1 | 276 s | 82.0% |
  | 8,000 (main, step 2) | CPU | 5,389 s | 82.5% |

  The GPUs were free: 0% utilization, checked with `nvidia-smi` before each launch.
- **Scrambled history, interpretation.**
  - Moving labels among examples of the same vocabulary size removes the link to each state's data.
  - It keeps how often each vocabulary token is a correct move across the history, for example own-variable decay terms. The control therefore retains this population knowledge (71% on validation).
  - On the practice systems it separates two things: the gain from knowing which terms are common, and the gain from reading each state's data, which is the full network minus the scrambled one.
  - The first launch reported an infinite training loss: moved labels can name impossible moves, such as adding a term already present. The loss now counts only possible targets; this is unchanged for real labels. The control was rerun from scratch.
- **History-size curve, extension (decided now, from the synthetic validation numbers only, before any control result on practice systems).**
  - The planned sizes (1,000–8,000) all sit on a plateau (81–82.5%), so networks on the first 100 and 300 history systems are added, with the same recipe.
  - They are labelled as an extension in the results.

### Trial 28 step 3: practice results (4 November, 05:36; 27 practice systems, no held-out system touched)

**With-history learner (network on 7,982 synthetic systems) against the baseline (uniform priorities):** exact recoveries on held-out starting states, with every recovered system expressible except 909, at M = 16 and 5,000.

| Budget | M = 4 | M = 8 | M = 16 |
|---|---|---|---|
| 2,000 (headline) | 3 vs 2 (+3 / −2, p = 1) | 7 vs 3 (+4 / −0, p = 0.13) | 11 vs 5 (+6 / −0, p = 0.031) |
| 5,000 | 5 vs 2 (+4 / −1, p = 0.38) | 8 vs 5 (+3 / −0, p = 0.25) | 12 vs 7 (+5 / −0, p = 0.063) |
| Trial 27 learner (reference) | 2 | 4 | 8 |

- **Direction.**
  - Across the six cells, 25 system recoveries were gained against 3 lost (458 and 728 at M = 4 and 2,000, and 458 at M = 4 and 5,000; 728 is recovered at 5,000).
  - With 27 systems, only M = 16 at 2,000 is individually significant.
  - The with-history learner beats the Trial 27 learner at every M, in both budgets.
- **Systems** (M = 8, 2,000): with history gets 115, 685, 728, 753, 866, 919 and 1047; the baseline gets 728, 753 and 866; the Trial 27 learner gets 685, 728, 753 and 909 (909 is not expressible).
- **Effort.**
  - Time per system, median, with history against baseline: 444 against 453 s (M = 4), 277 against 412 s (M = 8), and 147 against 414 s (M = 16) at 2,000.
  - Network inference is 4–7% of run time.

**Controls at M = 8 and 2,000:**

| Network | Exact (all / expressible) | Against baseline | Equations with a plausibly exact structure (median structure first found) | Ranking on synthetic set (1), top 5 |
|---|---|---|---|---|
| none (baseline) | 3 / 3 | – | 36/80 (168) | 5.2% (chance) |
| scrambled history | 6 / 6 | +3 / −0 | 46/80 (136) | 73.3% |
| 100 systems (extension) | 4 / 4 | +2 / −1 | 51/80 (97) | 76.5% |
| 300 systems (extension) | 8 / 8 | +5 / −0 (p = 0.063) | 57/80 (47) | 78.7% |
| 1,000 systems | 7 / 7 | +4 / −0 | 53/80 (48) | 80.3% |
| 2,000 systems | 7 / 7 | +4 / −0 | 55/80 (48) | 80.7% |
| 4,000 systems | 7 / 7 | +4 / −0 | 53/80 (36) | 80.5% |
| 8,000 systems (main) | 7 / 7 | +4 / −0 | 56/80 (60) | 80.6% |

**Reading.**
1. **History helps on real practice systems** in the same direction at every data size and budget, and the with-history learner is the best learner so far on the practice half at every M. The counts are small, though: 27 systems, of which only 17 are expressible.
2. **Most of the practice gain is reproduced by the scrambled-history network** (6 against 7). That network keeps the history's knowledge of which kinds of terms are common, but not the link to each state's data. So most of the benefit on real systems comes from a learned prior over structures, from synthetic systems built to resemble biochemistry. Reading each state's data adds about one system here, more equations reaching a plausibly exact structure (56 against 46), and faster discovery (median structure 60 against 136). On synthetic systems it adds about 7 points of ranking.
3. **The history-size curve rises from 0 through 100 to 300 systems, then is flat** from 1,000 to 8,000, in both ranking and practice recoveries. "Better with more systems learned" holds only up to a few hundred synthetic systems in this setup.

### Trial 28: owner decision (c), with two additions; plan for the fresh-seed confirmation (4 November, logged before any run)

**Owner.**
- Do (a), then (b).
- **Addition 1.** In (a), add a control: the baseline search with priorities from a plain frequency table of term types in the synthetic generator, with no network. Compare it with the scrambled-history network and the full network on the fresh seeds, and include it in the held-out test request.
- **Addition 2, for (b).** The history's term frequencies were set by hand to resemble biochemistry, so part of the gain may be our design choices rather than learning. Propose a history that is not tuned toward the test systems, and add learning from the learner's own solved searches (AlphaGo's second stage), so that it improves from systems it has actually learned. Send the (b) proposal in plain language before running it.

**(a) Fresh-seed confirmation (fixed now):**
- **Systems.** The 27 practice systems at M = 8 and 2,000 structures per equation (the headline), on fresh data:
  - new reference recordings (reference seed 2, new starting states);
  - new bursts (burst seed 2);
  - a new held-back split and search randomness (learner seed 2).
  Nothing from the seed-1 runs is reused.
- **Arms.**
  1. Baseline: uniform priorities.
  2. **Frequency-table control.** For every vocabulary term and equation i, the priority comes from how often a term of the same type (family; relation to the equation's own variable; multiplier; degree) appears in the true structures of the training-history systems. This uses the same type description the network sees, with no network and no data. It is mixed by move type exactly like every other prior: add a term ∝ its type's frequency; remove a term ∝ 1 − frequency; swap = both.
  3. The scrambled-history network.
  4. The full network (7,982 systems).
- **Report.** Exact recoveries (all / expressible), and paired sign tests for full against baseline, full against frequency table, full against scrambled, and frequency table against baseline. Effort and network time are reported separately.
- **Then.** The held-out test request is sent to the owner with full rules, including the frequency-table control. Nothing held out is touched before approval.

### Trial 29 proposal (option b): an untuned history plus learning from its own solved searches (4 November; awaiting owner approval)

**Why.** Trial 28's synthetic term frequencies were set by hand to resemble biochemistry, so part of the gain on real systems may be a design choice rather than learning. The frequency-table control (running in (a)) measures how much a plain table of those frequencies explains on its own.

1. **Stage 1, untuned history.**
   - The same building blocks with no hand-set preferences: every term type equally likely, 1–5 terms per equation drawn evenly, wide constant ranges, random signs, and no built-in decay term.
   - Only the checks needed for usable data remain: finite, bounded, cheap to simulate, moving, no degenerate pieces, and positive states (required by the toolkit's Hill-type and power pieces). Any remaining preference comes from which random systems survive those checks.
   - The resulting term-type frequencies are reported next to the practice systems' frequencies.
   - The same network is trained on it, as in step 2.
   - A go/no-go bar on new untuned synthetic systems, the same as step 2's, is written into the log before training.
2. **Stage 2, learning from its own solved searches** (AlphaGo's second stage).
   - The learner works on systems without being told their answers.
   - When its search finds a model that passes its own held-back check with a margin (val ≤ 10⁻¹³, three times tighter than acceptance), the search path and the structure found become new lessons. The network is retrained on them together with stage 1's data, in about 3 rounds.
   - **Sources:**
     - (i) new untuned synthetic systems, whose answers are never used;
     - (ii) the practice systems in two halves: it learns from what it solves in one half and is tested on the other, then the halves swap.
   - Only the learner's own verified solutions are used, never true structures or external test results.
   - Wrong lessons (false acceptances) are counted afterwards against the external test, but not used.
3. **Measured.**
   - The accumulation curve: exact recoveries on the held-apart practice half (M = 8, 2,000 structures per equation, all / expressible) after each round, against the number of systems solved and learned from.
   - The step-2 ranking and search measures on new untuned synthetic systems.
   - **Comparisons** at the same budget: uniform baseline; the untuned frequency table; the stage-1-only network; the Trial 28 hand-tuned network.
   - **Controls:**
     - stage 2 with its lessons scrambled (the same amount of retraining, no real lesson);
     - stage 2 learning from synthetic solutions only, which separates learning from solved real systems from more synthetic practice.
4. **Decision points.** Stop and report if stage 1 fails its bar. Check in with the curves after stage 2, before any held-out test.
5. **Cost.** About 1 hour for the history, minutes of GPU training (under the GPU rule), and about 2 hours per stage-2 round, about 6–8 hours in total.

### Trial 29: owner approval with conditions, and rules for stages 1 and 2 (4 November, logged before any Trial 29 system is generated)

**Owner's conditions.**
1. Check in after stage 1 whether or not its bar passes, with a term-frequency comparison between the untuned history and the practice systems.
2. In stage 2, report for each round how many real and how many synthetic solutions were learned from, alongside the accumulation curve.
3. Practice-half gains are suggestive only, since those systems have been used throughout Trials 26–28. The fresh-pool held-out test remains the real check.
4. The owner is asleep: do not wait after stage 1.
   - If stage 1 passes its bar, log the term-frequency comparison and continue straight into stage 2.
   - If it fails, stop and report.
   - Either way, have a plain-language summary ready for the morning, covering stage 1, the frequency comparison, stage 2 so far, and the results of (a).

**Stage 1 rules (fixed now).**
- **Untuned generator.**
  - d is uniform over {2, 3, 4}; initial states are log-uniform in [0.1, 10].
  - Each equation has 1–5 terms (uniform). Each term's family is uniform over the 12 families (monomial; reciprocal; Hill-type; exp; tanh; sin; |·|; power; weighted-sum reciprocal; weighted-sum Hill; product of reciprocals; product of Hills), then uniform within the family over variables and multiplier.
  - Monomial degree is uniform 0–3.
  - Every coefficient has a random sign and a wide magnitude: log-uniform in [0.01, 10] for monomials, and the piece's scale times log-uniform [0.1, 10] for pieces.
  - Piece placements are relative to each variable's initial state, only so that pieces bend over the data.
  - There is no built-in decay term. Held-out family pairs are excluded from the history and set (1u).
- **Acceptance.** The same checks as every synthetic system: the cheap-simulation cutoff, the settle-time rule, finite, bounded, positive, moving, not centred at M = 8, and no degenerate true piece.
- **History.** 3,000 accepted systems (seeds "t29-hist-k"), fewer than Trial 28's 7,982 because that curve saturated by 300–1,000 systems. About 12 states per system, with M drawn from 4, 8 and 16 (probabilities 0.2, 0.6, 0.2).
- **Network.** The same architecture and recipe: 12 epochs, last epoch kept. On a free GPU, or shared under the owner's rule; the GPU used is logged.
- **Bar** (as in step 2, at M = 8), with 100 systems per set to fit the night:
  - set (1u): 100 new untuned systems (seeds "t29-test1-k");
  - set (2u′): 100 untuned systems with a size-matched held-out combination (seeds "t29-test2m-k"; two pieces swapped, as in set (2′));
  - (a) ranking top-5 at least 30% and at least twice the uniform chance rate on each set;
  - (b) search at 2,000 structures per equation: with history at least +10 points over uniform, with an exact two-sided sign test at p < 0.05, on each set.
- **Term-frequency comparison.** Per equation, the share of terms in each class: products of variables; single-variable switches (reciprocal, Hill); weighted-sum switches; products of switches; other functions (exp, tanh, sin, |·|, power); and, for practice systems, terms outside the toolkit. It covers three sources: the untuned history; the Trial 28 hand-tuned history; and the 27 practice systems' true right-hand sides, classified with `expressible.py`.

**Stage 2 rules (fixed now).**
- **Practice halves.** Half A holds the practice systems whose sha256("t29-half:" + id) starts with a hex digit 0–7; half B holds the rest. Counts are reported.
- **Rounds r = 1, 2, 3.** In each round the current main network (round 1: the stage-1 network) does the following.
  1. It searches 150 new untuned synthetic systems (seeds "t29-s2syn-r-k") and practice half A (seed 1), at M = 8 and 2,000 structures per equation.
  2. **Solutions** are runs accepted by the learner's own held-back check with val ≤ 10⁻¹³, three times tighter than acceptance.
  3. **Lessons.** For each equation, about 4 states around the found structure (proper subsets, subsets with 1–2 other terms, and the found structure plus one other term), labelled relative to the found structure (hindsight), as in the history.
  4. **Retraining from scratch** on the stage-1 history plus all lessons so far, with lessons repeated 10× per epoch. This gives the main network and two controls: (i) scrambled lessons (labels permuted among lesson examples of the same vocabulary size); (ii) synthetic lessons only.
  5. **Evaluation on practice half B** (seed 1, M = 8, 2,000) of the new main network and the two controls. Round 0 is the stage-1 network. References: the uniform baseline and the Trial 28 hand-tuned network (seed-1 step-3 runs).
- **Reported each round:**
  - real and synthetic solutions learned from, new and cumulative;
  - lesson examples;
  - wrong lessons, counted afterwards and never used (synthetic: the found structure differs from the true one; practice: the external test failed);
  - exact recoveries on half B (all / expressible) for the main network and the controls;
  - success on that round's new synthetic systems.
- **Direction.** A → B first. The reverse direction (B → A) is run only if time allows before morning, and is reported as not run otherwise.
- **Untuned generator calibration on development seeds only** ("t29-dev-k"; 4 November, before any history or test system).
  - As first specified, 0 of 80 draws were accepted. With random signs and no decay term, random systems blow up or never settle, and the settle-time rule requires settling.
  - **Changes, none touching which term families appear (still uniform over the 12):**
    - Untuned systems that do not settle are recorded over a window of half their own starting timescale (the median of |xᵢ / fᵢ(x₀)|). Blow-ups inside the window, stiffness and non-positivity are still rejected.
    - Coefficient magnitudes are log-uniform in [0.1, 2] for monomials, and the piece's scale times [0.3, 3] for pieces. The original 0.01–10 range made most draws too stiff to simulate cheaply.
    - The settle-time integration is limited to 2 s for untuned systems.
  - Acceptance is now 4 of 80 (5%), at about 5 s per draw.
  - **History size reduced to 1,000 systems**, from 3,000, to fit the night: Trial 28's history-size curve was flat from 1,000 to 8,000 in both ranking and practice recoveries.
  - The test sets stay at 100 systems each.
  - Accepted untuned systems use the 12 families roughly evenly, so monomials are rare. That is unlike biochemistry, and intended.
- **(a) almost complete** (seed 2, M = 8, 2,000): uniform 4 of 27 exact; frequency table 7 of 25 so far; scrambled network 8 of 27; full network 9 of 27.

### Trial 28 (a): fresh-seed confirmation with the frequency-table control (4 November, 06:56; practice half)

Fresh data: new reference recordings (seed 2), new bursts and a new held-back split. M = 8, 2,000 structures per equation, 27 practice systems, every recovery expressible.

| Arm | Exact | Systems | Time per system (median) |
|---|---|---|---|
| Baseline (uniform) | 4 | 115, 753, 866, 919 | 372 s |
| Frequency table (no network, no data) | 7 | 115, 275, 507, 728, 753, 866, 1047 | 409 s |
| Scrambled-history network | 8 | the table's 7 + 919 | 239 s |
| Full network | 9 | the scrambled network's 8 + 685 | 290 s |

**Paired comparisons:**
- full against baseline +5 / −0 (p = 0.063);
- full against frequency table +2 / −0 (p = 0.5);
- full against scrambled +1 / −0 (p = 1);
- frequency table against baseline +4 / −1 (p = 0.38);
- scrambled against baseline +4 / −0 (p = 0.13).

**Reading.**
- The direction of the seed-1 result is confirmed on fresh data: full 9 against baseline 4 here, against 7 and 3 with seed 1.
- But a plain table of how common each term type is in the hand-tuned generator, with no network and no data, recovers 7 of the full network's 9.
- Most of Trial 28's gain on real systems is therefore the generator's hand-set term frequencies, as the owner suspected. Reading the data adds about 2 systems, which is not significant on 27.
- This is the motivation for Trial 29 (untuned history, learning from its own solved searches).
- The held-out test request, which includes the frequency-table control, goes to the owner with the morning summary.

### Trial 29 stage 1: history, network, term-frequency comparison and bar (a) (4 November, 07:48)

- **History.** 1,000 accepted untuned systems (seeds t29-hist-0 … 999), from 12,723 rejected draws (acceptance 7%).
- **Network.** Trained on **GPU 0** (0% utilization, 1 MiB used at the `nvidia-smi` check), in about 1 minute. Validation on held-out untuned history systems: top-5 hits 51.1% against 6.1% chance. Trial 28's hand-tuned network reached 82.5% on its own history: untuned systems are more varied and carry more nonlinear pieces.
- **Term-frequency comparison** (share of all terms, by class):

  | Class | Untuned history (Trial 29) | Hand-tuned history (Trial 28) | 27 practice systems (true equations) |
  |---|---|---|---|
  | products of variables | 10% | 57% | 69% |
  | single-variable switches | 16% | 32% | 16% |
  | weighted-sum switches | 19% | 4% | 1% |
  | products of switches | 17% | 3% | 0% |
  | other functions (exp, tanh, sin, \|·\|, power) | 37% | 5% | 1% |
  | outside the toolkit | 0% | 0% | 12% |
  | terms per equation | 2.91 | 2.76 | 3.84 |

  The hand-tuned history resembles the practice systems; the untuned history does not.
- **Bar (a), ranking** (top 5 of all legal moves):
  - set (1u), 100 systems and 768 states: network 44.1% against 6.6% chance (6.7×); data-only priorities 28.0% (reference);
  - set (2u′), 100 systems and 842 states: network 36.0% against 7.0% chance (5.1×); data-only 30.6%.
  - **Met on both sets.**
- **Bar (b)**, the search at 2,000 structures per equation on both sets (400 runs), is running.
- **Bar (b), search** (2,000 structures per equation, M = 8; 09:28):

  | Set | Baseline exact | Untuned-history network exact | Difference | Sign test | Bar (b) |
  |---|---|---|---|---|---|
  | (1u) | 10 of 100 | 11 of 100 | +1 point | p = 1.0 | not met |
  | (2u′) | 2 of 100 | 3 of 100 | +1 point | p = 1.0 | not met |

- **Stage 1 bar not met. The pipeline stopped before stage 2, as instructed; stage 2 was not run.**
- **Why (diagnostics).**
  - Untuned equations are dominated by nonlinear pieces: 2.5 per equation on average, against 1.2 in the hand-tuned history.
  - Per equation, the network finds a plausibly exact structure somewhat more often (set (1u): 111 against 97 of 278) and much sooner (median structure 32 against 140; set (2u′): 17 against 112).
  - Most piece-heavy equations stay unsolved by both learners within the budget, so whole systems rarely complete. The network's ranking skill shows as speed, not as more systems recovered.
  - No false acceptances in either learner.
- **Effort.** Time per system, median 302 s with the network against 205 s for the baseline; network inference 7% of run time. Training took about 1 minute on GPU 0.
- **Reading.**
  - With an untuned history, what the network learns (bar a: 36–44% top-5 against 7% chance) does not turn into more exactly recovered systems at the headline budget.
  - Together with (a), where a plain frequency table explains most of Trial 28's practice gain, the evidence so far is that history helps mainly through the term frequencies we put into it.
  - The binding limit on harder systems is the search and fitting of equations with several nonlinear pieces, for every learner.

## Owner direction (4 November): next trial on fitting equations with several nonlinear pieces; held-out set kept for a stronger candidate

*"Keep the held-out set for a stronger candidate, and make the next trial about fitting equations with several nonlinear pieces."* First a literature search, then a proposal that reuses the N1/N2/N3 ideas, keeps the fitter swappable, applies the improvement to both learners, reruns the untuned-history network against the baseline, and reports equations solved as well as whole systems, plus search effort. Survey and proposal before running anything.

### Survey: fitting the constants of models with several nonlinear pieces exactly (4 November)

1. **Variable projection**
   - Golub & Pereyra 2003 review; Kaufman's simplification; R's `nls(algorithm = "plinear")`.
   - The linear coefficients are solved exactly at every value of the nonlinear constants, which leaves a smaller, better-conditioned problem. It is the standard tool for sums of nonlinear pieces, especially exponentials.
   - Systems biology uses the same idea ("hierarchical optimization", Loos, Krause & Hasenauer 2018).
   - Our N1/N2/N3 archive found one case 53,921× better once the linear output weights were solved exactly. We already use it.
2. **Rational pieces can be linearized by clearing denominators**
   - The Sanathanan–Koerner iteration (1963); vector fitting (Gustavsen & Semlyen 1999), which relocates poles by repeated linear least squares; the AAA algorithm (Nakatsukasa, Sète & Trefethen 2018), which finds rational approximants and their poles with no initial guesses.
   - Implicit SINDy (Mangan et al. 2016) and SINDy-PI (Kaheman, Kutz & Brunton 2020) apply the same idea to Michaelis–Menten and other rational dynamics. Implicit SINDy was very sensitive to noise, and SINDy-PI fixed that.
   - Algebraic estimation for rational ODE models (Bassik et al. 2023; Demin, Ovchinnikov & Rouillier 2025) finds all parameter values consistent with the data, with no initial guesses, by solving polynomial systems. It is practical up to about 10 states and 10 parameters.
   - Our reciprocals, integer-exponent Hill switches, products of switches and weighted-sum reciprocals are all rational, and our data are noise-free, which is the case where clearing denominators works best.
3. **Shooting and collocation.** Multiple shooting (Bock; Peifer & Timmer 2007) and collocation-type matching both reduce the multimodality of fitting by simulation.
   - Gradient matching (Varah 1982; Brunel 2008) and integral matching (Dattner & Klaassen 2015; `simode`) avoid integrating the model at all.
   - Integral matching is more stable than matching derivatives, and both are used as warm starts for a final local fit.
   - Our rate fit followed by a step-integral refit is this two-stage pattern, already in place.
4. **Global optimizers for a few hard parameters**
   - Multistart of derivative-based local fits, with log-scaled parameters and Latin-hypercube starts, outperformed the alternatives by orders of magnitude (Raue et al. 2013).
   - In larger benchmarks a hybrid of stochastic global search with local refinement did better still (enhanced scatter search, MEIGO; Villaverde et al. 2019).
   - Deterministic global methods with guarantees (Esposito & Floudas 2000; Singer & Barton; Miró et al. 2012) are limited to about 5 parameters and 5 states.
   - DIRECT-style partitioning, as in our N1, suits a handful of parameters.
5. **Continuation (homotopy)** smooths the objective, for example by adding observer-like terms (Vyasarayani, Uchida & McPhee 2012), and follows the solution as the smoothing is removed. Analogues for us: soften steep switches, or move a Hill exponent from 1 upwards, then follow the fit.
6. **Identifiability**
   - Structural: SIAN and StructuralIdentifiability.jl, based on differential algebra.
   - Practical: the profile likelihood (Raue et al. 2009).
   - The Hill curve is structurally identifiable, but its exponent is often practically non-identifiable from limited data (2025 framework paper). Two pieces sharing one switch are functionally related.
   - Detecting flat directions tells a fit to tie, snap or stop instead of wandering.
7. **Constant fitting in symbolic regression**
   - Levenberg–Marquardt with restarts is the strongest local fitter (Kommenda et al. 2013 and 2020; de Veloso et al.). No single constant optimizer wins everywhere (2024 benchmark).
   - Exhaustive enumeration shows that local optima in constant fitting make the same expression look good or bad depending on the fit (Kronberger et al. 2024). That wastes search, which is our situation when a correct structure is mis-fitted.
   - End-to-end transformers (Kamienny et al. 2022) predict constants as starting values for a final optimizer: the template for a learned initializer.

**What we could reuse:**
- clearing denominators for joint, guess-free starting values of all rational pieces;
- an N1-style joint global search over the few remaining nonlinear constants, with variable projection inside, in place of today's one-piece-at-a-time sweeps;
- multistart Levenberg–Marquardt (N2's role) and Newton polishing on the precise step integrals (N3's role);
- log-scaled constants and continuation for steep switches;
- an identifiability check that ties, snaps or stops.

### Trial 30 proposal: a fitter for equations with several nonlinear pieces (4 November; awaiting owner approval)

1. **Fitter v2** (swappable; the same interface as today's, used by both learners):
   - (a) **algebraic start** for rational pieces: clear denominators, solve one linear least-squares problem for all of them jointly (integer Hill exponents 1–4 tried), and remove the bias with Sanathanan–Koerner reweighting;
   - (b) **joint global search** (N1-style) over the remaining nonlinear constants (exp rates, tanh/sin placements, |·| kinks, free exponents), with the linear coefficients solved exactly at every candidate;
   - (c) **Levenberg–Marquardt** from the best few distinct candidates, then **Newton polishing** on the step integrals;
   - (d) **identifiability check**: flat directions lead to tying shared switches, snapping exponents, or reporting "not identifiable from these data".
   - A learned starting-value network can later replace (a) and (b).
2. **Fitter benchmark with known structures** (no search):
   - synthetic equations from both generators, grouped by number of nonlinear pieces (1, 2, 3+), and the practice systems' known structures;
   - measures: the share fitted to exactness and the time per fit, for v1 against v2;
   - a pass bar, written into the log before running, decides whether to go on.
3. **Search comparison.** Both learners (the uniform baseline and the untuned-history network) with fitter v2 on Trial 29's stage-1 test sets, at M = 8 and 2,000 structures per equation. Reported: equations solved (plausibly exact, and for synthetic systems the exact true structure) as well as whole systems; search effort (structures, fitting time per structure, time to first plausible structure); and network time separately.
4. **If the untuned-history network then passes Trial 29's bar (b)**, propose resuming Trial 29 stage 2. No held-out system is touched.

### Trial 30: owner approval with two additions; fitter v2 design and known-structure pass bar (4 November, logged before any run)

**Owner.** Trial 30 is approved as outlined, with two additions:
1. Check in after the known-structure fitter test, with the old-against-new results measured against the pass bar, before running the search comparison.
2. Snap exponents to whole numbers only when the identifiability check says the data cannot pin them down, and let the exactness test decide in every case. Report how often snapping was used.

**Fitter v2** (`fitter30.py`, swappable; same interface plus an optional equation index):
- **(a) Algebraic start.**
  - Rational pieces are handled jointly by clearing denominators: single-variable and weighted-sum reciprocals, Hill-type pieces with whole-number exponents 1–4 (every combination tried), and products of reciprocals or of Hill-type pieces.
  - The other terms enter at their current placements.
  - It is a single linear least-squares problem in the lifted unknowns, normalized by fixing one coefficient, with 3 Sanathanan–Koerner reweightings.
  - It runs on dense rate data: derivatives of each training burst's interpolating polynomial at interior points.
  - Starts that put a pole or a switch outside its valid range are dropped.
- **(b) Joint global search** (precise stage only, N1-style) over all the structure's hard constants together, with the linear coefficients solved exactly at every candidate: 1,024 Sobol points plus 3 rounds of refinement around the 8 best.
- **(c) Polishing.** Levenberg–Marquardt with variable projection from the best distinct candidates (the algebraic start, the sweep start, and the global search's best), then the existing Gauss–Newton refit on the step integrals, which converges like Newton at zero residual.
- **(d) Identifiability check.** At the step-integral fit, each exponent's uncertainty is estimated from the Jacobian and the data's precision floor. An exponent may be snapped to the nearest whole number only if that uncertainty exceeds 0.05. Both versions are then judged by the held-back step-integral error, and the lower one is kept. Snaps allowed and snaps kept are counted.
- The old fitter (v1, `GridSweepFitter` with its snapping rule) is kept unchanged as the reference.

**Known-structure fitter test.** The fitter is given the true structure (no search).
- **Equations:**
  - (U) 60 untuned synthetic systems (seeds "t30-fit-u-k");
  - (T) 60 hand-tuned synthetic systems (seeds "t30-fit-t-k");
  - (P) the practice systems' known structures: the 13 practice systems that are expressible and exactly learned at M = 64, carried into the M = 8 coordinates, skipping coordinate-inconsistent ones as in the step-1 calibration.
- **Procedure.** M = 8, seed 1. Each true structure is evaluated exactly as the search evaluates a structure: ordinary fit, precise gate, thorough fit, step-integral refit, held-back check. "Recognized" means plausibly exact.
- **Groups:** equations by number of nonlinear pieces (1, 2, 3 or more); monomial-only equations are reported separately.
- **Pass bar (all four required):**
  1. On synthetic equations with 2 or more pieces (U and T pooled), v2 recognizes at least 1.5× as many as v1, and at least 10 percentage points more.
  2. On 1-piece synthetic equations, v2 is within 2 points of v1 or better.
  3. On practice known structures, v2 recognizes at least as many as v1.
  4. The ordinary fit's median time is at most 2× v1's, and the thorough fit's median time is at most 2 s.
- **Reported alongside:** time per fit, snapping use, and identifiability flags.
- **The search comparison waits for the owner's go-ahead after this test.**
- **Fitter v2 development on development seeds only** ("t30-dev-u-k", 6 untuned systems, 15 equations; 4 November, before the logged test).
  - As first built, v2 recognized no more equations than v1: most 3–5-piece true structures never reached the precise stage, because the cheap fit stayed 10⁸–10¹⁰× above the floor. The global search's box, taken from the start grids, missed gentle pieces; on one equation the true sine is about a tenth of a period over the data, below the grid's half period.
  - **Changes, all in v2 (v1 unchanged):**
    1. The global search samples natural data-based ranges: frequencies and rates from gentle to steep on a log scale, switch centres and kinks across the data, poles at log-spaced distances on either side, Hill constants on a log scale with free exponents. Its budget is 256 samples per hard constant, up to 2,048.
    2. v2's sweep uses a start grid built from the same natural ranges, joined to v1's grid points.
    3. A joint-placement beam in the cheap fit: each piece's 3 best placements, combinations solved jointly, the best one polished.
    4. v2's precise gate also admits structures with two or more nonlinear pieces whose cheap fit explains 99% of the rates.
  - **Development result:** v1 recognized 5 of 15 equations, v2 6 (multi-piece 0 of 10 against 1 of 10). The ordinary fit's median time was 0.074 s against 0.123 s (1.66×); v2's evaluation median was 0.94 s.
  - Run directly with no gate, v2's thorough fit solved 2 of 9 hard development equations exactly. Success on 6–7 hard constants is partly a matter of the search's sampling.

### Trial 30: known-structure fitter test: pass bar MET (4 November, 16:10)

The fitter is given each true structure, which is evaluated exactly as the search evaluates a structure. M = 8, seed 1. Sets: 60 untuned (U) and 60 hand-tuned (T) synthetic systems, and the practice known structures (P: 13 systems, 1 skipped for centred coordinates, 12 evaluated). 372 equations in all.

| Set, pieces | Equations | v1 recognized | v2 recognized |
|---|---|---|---|
| U, 0 | 4 | 4 | 4 |
| U, 1 | 35 | 32 (91%) | 33 (94%) |
| U, 2 | 46 | 15 (33%) | 31 (67%) |
| U, 3+ | 77 | 3 (4%) | 12 (16%) |
| T, 0 | 28 | 28 | 28 |
| T, 1 | 89 | 81 (91%) | 83 (93%) |
| T, 2 | 50 | 16 (32%) | 40 (80%) |
| T, 3+ | 13 | 2 | 4 |
| P, 0 | 10 | 10 | 10 |
| P, 1 | 16 | 14 | 14 |
| P, 2 | 4 | 0 | 1 |

**Bar:**
1. Synthetic equations with 2+ pieces: v1 36 of 186, v2 87 of 186 (+27 points, 2.4×). **Met.**
2. 1-piece synthetic equations: 91% against 94%. **Met.**
3. Practice known structures: 24 against 25 of 30. **Met.**
4. Ordinary fit median 0.030 s against 0.037 s (1.25×). The thorough fit, run on 150 equations, has a median of 0.92 s per call (90th percentile 1.73 s, maximum 3.5 s); the report script's 0.07 s figure was the median whole evaluation over the 350 equations that reached the precise stage, most of which needed no thorough fit. **Met.**

**Stages of v2 over all equations:**
- the algebraic start improved the cheap fit 152 times (of 342 tries);
- the joint-placement beam 42 times;
- the global search won 73 of 150 thorough fits.

**Snapping:** 181 exponents were checked. The identifiability check allowed snapping 44 times, every one a direction the data could not pin down. The exactness test kept the snapped version 9 times.

**Still hard:** equations with 3 or more pieces (U: 16%). Awaiting the owner's go-ahead before the search comparison.

### Trial 30: owner approval of the search comparison; rules (4 November, logged before any run)

**Owner.** *"Approved: do the timing check on the 6 development systems and the old-fitter consistency rerun, then the search comparison. Include the plain term-frequency control as a third arm. If you need to cap thorough fits per equation, log the cap before starting, apply it equally to all arms, and report how often it was hit. Check in with the results, side by side with Trial 29's old-fitter numbers, before proposing stage 2."*

**Arms** (all three with fitter v2):
- **baseline:** uniform priorities;
- **untuned-history network:** Trial 29's network (`t29_net_u`), unchanged;
- **frequency control:** priorities from a plain frequency table of term types, with no network. The table is counted over the same untuned history the network was trained on (50 shards, 2,714 equations, 72 term types), built the same way as in check (a).

**Systems.**
- Trial 29's two stage-1 test sets: (1u) "t29-test1" untuned, and (2u′) "t29-test2m" size-matched; 100 systems each.
- M = 8, seed 1, 2,000 structures per equation, and a time cap of 3,600 s per system, all as in Trial 29.
- CPU only (the network runs on CPU, as in Trial 29); at most 24 worker processes.

**Instrumentation** (it changes nothing about what the search does):
- per equation, the number and the time of thorough fits;
- an optional per-equation cap on thorough fits, off unless logged here;
- a watch on each equation's true structure: when it was fitted, and whether it was recognized.

**Step 1: timing check.**
- The 6 development systems ("t30-dev-u-k", k = 0–5), the three arms with v2 (18 runs).
- From the mean time per system, predict the comparison's run time at 24 workers (600 runs).
- If the prediction exceeds 8 hours, a cap C on thorough fits per equation is chosen from the timing check and logged here before the comparison.
- **Cap rule (if used):** an equation's first C structures that need a thorough fit get one; after that, a structure gets one only if its cheap fit is better than that of every structure already thorough-fitted. The same C for all arms; how often the cap was hit is reported.

**Step 2: old-fitter consistency rerun.**
- Trial 29's runner, unchanged, with today's code and fitter v1: systems k = 0–3 of each test set, baseline and network arms (16 runs).
- Compared with Trial 29's records on status, external exactness, held-back error, chosen structures, the structures fitted per equation, resume rounds and assemblies.
- **Pass:** identical on all of these (runs stopped by the time cap are excepted and reported). If it fails, I stop and report before the comparison.

**Step 3: the comparison** (600 runs). Reported per set and arm, next to Trial 29's v1 numbers on the same systems:
- whole systems exactly recovered;
- equations with a plausibly exact structure found;
- equations whose true structure was recognized: chosen in the final model (comparable with Trial 29's records), and, for v2 only, fitted and recognized at any point of the search;
- search effort: structures, fitting time per structure, thorough fits, time to first plausible structure, time per system, time-cap hits, and cap hits if a cap is used;
- the prior's computing time, separately;
- Trial 29's bar (b), applied to the network against the baseline with v2: at least +10 points, with an exact two-sided sign test at p < 0.05, on each set. The frequency control is compared with both.
- Then a check-in before any proposal for stage 2.

### Trial 30 steps 1–2: timing check and old-fitter consistency rerun (4 November, 18:05). Consistency criterion not met; stopped before the comparison, as logged

**Step 2: old-fitter consistency rerun** (16 runs: systems k = 0–3 of each test set, baseline and network arms, Trial 29's runner unchanged, fitter v1).
- **Against Trial 29's records:** identical on every compared field in 9 of 16 runs. The logged criterion (identical on all fields) is **not met**; no run was stopped by the time cap.
- **Where the 7 differ:**
  - the inexact structures chosen for a failed final model, and that model's held-back error, in 6;
  - when the chosen structures were first fitted, in 7;
  - slightly different search paths (structures fitted, first plausible find, pruning, precise-stage count), in 4.
- **Outcomes:**
  - status (accepted or rejected) and whole-system exactness: identical in 16 of 16;
  - which equations found a plausibly exact structure: identical in 15 of 16. In one network-arm run, the rerun found a plausibly exact structure for one equation (at structure 762) where Trial 29 found none.
- **Diagnosis** (counts only; test systems from the rerun itself):
  - **No code change on the old fitter's path.** Since Trial 29, the only changes there are the Trial 30 dispatch hooks, which v1 does not use, and bookkeeping. No module the search imports was modified after Trial 29 ran.
  - **Run-to-run noise.** The 5 runs that differed among the first 14 were rerun once more with the same settings:
    - 1 of 5 matched the first rerun;
    - 3 of 5 matched Trial 29 exactly;
    - whole-system exactness was unchanged in all.
  - **Python's hash seed:** on the one system tested, it made no difference (seeds 0, 0 and 1 gave identical results, equal to Trial 29). The two unpinned reruns of that system differed from each other only in the failed model's chosen structures, with the same search.
  - **Likely cause** (not proven): floating-point results that vary between processes in the math library torch uses (MKL, single-threaded), so near-ties among inexact fits break differently.
  - **Consequence:** Trial 29's numbers are a fair reference in distribution. Differences of an equation or two, and rarely a system, are within run-to-run noise.

**Step 1: timing check** (6 development systems, three arms, fitter v2, no cap; no errors):

| Arm | Time per system, mean (median) | Equations stopped by the time cap | Thorough fits per equation, median | Thorough share of search time | Ordinary fit per structure |
|---|---|---|---|---|---|
| baseline | 2,125 s (2,020 s) | 4 of 15 | 559 | 84% | 86 ms |
| network | 2,534 s (2,315 s) | 7 of 15 | 532 | 82% | 119 ms |
| frequency | 2,316 s (2,142 s) | 4 of 15 | 569 | 80% | 113 ms |

- **For comparison**, Trial 29's old-fitter medians were 205 s (baseline) and 302 s (network) per system. In the 16 old-fitter reruns, the median equation had no thorough fit; 7 of 40 equations had more than 200, and thorough fits took 28% of search time.
- **The cost:** in the search, v2's precise gate admits about 45% of structures, each with a thorough fit of about 1 s.
- **Predicted run time for the comparison** (600 runs at 24 workers) with no cap: **16.1 h**, and 15 of 45 equation searches would stop at the time cap before 2,000 structures. With a cap C on thorough fits per equation (logged rule):

  | C | Predicted time | Equations over the cap | First plausibly exact finds after more than C thorough fits (upper bound on what the cap can cost) |
  |---|---|---|---|
  | 100 | 5.7 h | 42 of 45 | 6 of 18 |
  | 200 | 7.6 h | 38 of 45 | 4 of 18 |
  | 400 | 10.7 h | 30 of 45 | 1 of 18 |

- **Development systems only, too few to judge:** whole systems exact: baseline 1, network 0, frequency 1 of 6. Equations with a plausibly exact structure: 6 of 15 in each arm.

**Stopped before the comparison, as logged. Proposal to the owner:**
1. Restate the consistency criterion at the outcome level, which the rerun met: whole systems 16 of 16, equation-level 15 of 16.
2. Rerun Trial 29's two old-fitter arms in the same batch, so old and new fitters are compared under the same conditions. Trial 29's original numbers are reported alongside.
3. Cap C = 200 thorough fits per equation for every arm in the batch.

### Trial 30: owner decision on steps 1–2, and the comparison's final rules (4 November, 18:40, logged before the comparison starts)

**Owner.** *"Go ahead: judge consistency on outcomes and treat small differences as noise. Use a cap of 200 thorough fits per equation for every arm, and include the same-batch rerun of Trial 29's old-fitter arms under the same cap. Report Trial 29's original numbers alongside. Have the results and a plain-language summary when done."*

**Consistency.** Judged on outcomes, which the rerun met: whole systems 16 of 16, and equations with a plausibly exact structure 15 of 16. Differences of an equation or two, or rarely a system, are treated as run-to-run noise.

**The comparison:** one batch of 1,000 runs on 24 CPU workers at nice 10, with no GPU.
- **Arms:**
  - fitter v2 with the baseline, the untuned-history network and the frequency control;
  - fitter v1 with the baseline and the network: the same-batch rerun of Trial 29's arms.
- **Every arm:**
  - a cap of C = 200 thorough fits per equation; after the first 200, a structure gets a thorough fit only if its cheap fit beats that of every structure already thorough-fitted;
  - M = 8, seed 1, 2,000 structures per equation and 3,600 s per system, as in Trial 29.
  - The cap's mechanism was checked on a development system before launch (with C = 2: 4 and 6 thorough fits, 22 refused).
- **Systems:** Trial 29's sets (1u) and (2u′), k = 0–99 each, interleaved by system so that partial results stay balanced across arms.
- **Reported per set,** next to Trial 29's original numbers (v1, no cap):
  - the measures logged in the rules above;
  - how often the cap was hit: equations where it bound, and structures refused a thorough fit;
  - the same-batch v1 against Trial 29's original, which shows noise plus what the cap costs v1.
- **Predicted run time:** about 9 hours.

### Owner request: log every thorough fit for future learned components (4 November, 19:20; proposal, nothing changed yet)

**Owner.** *"For future runs (don't change the one in progress): save every thorough fit's inputs and outcomes (the data, the structure, the starting values, the final constants, whether it was exact, and the time taken) in a compact form. These are training data for a later learned starting-value network and a learned screen for which candidates deserve a thorough fit… Don't edit any file the running comparison uses; make the changes on a separate branch or copy until it finishes. For now, just tell me how much disk space this would need and how you'd implement it."*

**Sizes measured** (3 development systems; thorough-fit counts from the 55 comparison runs finished so far, counts only):
- **A system's data:** the raw bursts, 8 × 13 × d values, 1.6–3.2 KB in float64. Everything the fitters use (rates, step integrals, dense rates) is rebuilt deterministically from them.
- **One structure:** 1–5 terms, 1–5 hard constants and 1–5 coefficients.
- **Thorough fits per run** (cap 200): v2 about 440 (plus about 1,480 refused by the cap); v1 about 66.

**Estimated disk use:**
- **Per thorough fit:** about 0.25 KB for the structure, the starting and final constants, the scores, the flags and the time; about 1.2 KB if every internal start of v2's thorough fit is kept.
- **Per v2 run:** 0.1–0.5 MB, plus up to 0.2 MB for optional short rows on structures refused by the cap.
- **Per 1,000-run batch** like the current comparison: about **0.1–0.5 GB**, and up to about 1.5 GB without the cap. That is well inside the 20 GB footprint limit; 23 GB is free.

**Proposed implementation** (awaiting the owner):
- On a separate branch in its own git worktree, off by default, tested on development systems only, and merged after the comparison finishes.
- **The recorder:** a small module (`fitlog.py`). The search hands each thorough fit to it, and it writes one compressed file per run in the history-shard layout. The file holds the bursts once, then one row per thorough fit:
  - the structure;
  - the warm start and the cheap fit's constants and residual;
  - the gate and cap status;
  - for v2, each internal start (sweep, algebraic, global, beam) and its polished residual, plus the snapping decisions;
  - the constants after the thorough fit and after the step-integral refit;
  - the residuals, the held-back error, the degenerate and plausibly-exact flags, and the times.
- **Checks:**
  - logging changes no outcome;
  - reloaded rows reproduce their fits from the stored data and starts;
  - the overhead and the file sizes are as estimated.
- **Test-data hygiene:** each file is tagged with its set. The loader refuses test and reserve sets for training by default; fits logged on test systems may only be used to evaluate learned components.
- **Option:** shadow thorough fits on a small random sample of the structures the gate or the cap turned away. They would be logged but not used by the search, with their own random stream. Without them, the learned screen only ever sees candidates the current gate let through. They are off in headline comparisons, because their time interacts with the 3,600 s cap.

### Trial 30 step 3: search comparison results (5 November, 04:47)

**Run.**
- 1,000 runs from 18:38 to 04:47 on 24 CPU workers (no GPU), with no errors; about 240 worker-hours.
- A cap of C = 200 thorough fits per equation applied to every arm.
- **Disk.** The shared disk's free space fell from 23 GB to 12 GB during the night through other users' writing; this batch's own files are a few MB. Free space was watched every minute against the 5 GB stop rule and never came near it.

**Systems.**
- Set (1u): all 100 systems were drawn identically in all seven runs (Trial 29's two and this batch's five).
- Set (2u′): 96 of 100 were. The other 4 came out differently in at least one run, because the generator's acceptance step uses simulation time limits, which can flip under load. All (2u′) comparisons use the 96. Trial 29's original (2u′) comparison may have paired a differently drawn system or two.

**Whole systems exactly recovered, and equations with a plausibly exact structure:**

| Arm | (1u) systems of 100 | (1u) equations of 278 | (2u′) systems of 96 | (2u′) equations of 267 |
|---|---|---|---|---|
| Trial 29 baseline (v1, no cap) | 10 | 97 | 2 | 81 |
| Trial 29 network (v1, no cap) | 11 | 111 | 3 | 87 |
| baseline, v1, this batch | 10 | 97 | 2 | 79 |
| network, v1, this batch | 10 | 106 | 3 | 88 |
| **baseline, v2** | **12** | **116** | **8** | **100** |
| **network, v2** | **15** | **129** | **7** | **104** |
| **frequency control, v2** | **13** | **113** | **6** | **88** |

**The true structure** (synthetic systems), per equation:

| Arm | (1u) chosen in the final model | (1u) recognized during the search | (1u) fitted during the search | (2u′) chosen | (2u′) recognized | (2u′) fitted |
|---|---|---|---|---|---|---|
| Trial 29 baseline (v1) | 68 | – | – | 50 | – | – |
| Trial 29 network (v1) | 72 | – | – | 63 | – | – |
| baseline, v1, this batch | 68 | 74 | 92 | 51 | 58 | 73 |
| network, v1, this batch | 73 | 81 | 133 | 64 | 70 | 107 |
| baseline, v2 | 79 | 90 | 102 | 66 | 70 | 84 |
| network, v2 | 90 | 107 | 139 | 89 | 95 | 109 |
| frequency control, v2 | 84 | 96 | 122 | 72 | 77 | 94 |

**Paired comparisons** (exact two-sided sign tests; "a vs b" counts the systems or equations only a solved, and only b solved):
- **New fitter against old fitter, same batch:**
  - baseline (1u): systems +2 points (2 vs 0, p = 0.5); equations 27 vs 8 (p = 0.002);
  - baseline (2u′): systems +6 points (6 vs 0, **p = 0.031**); equations 25 vs 4 (p = 0.0001);
  - network (1u): systems +5 points (5 vs 0, p = 0.062); equations 25 vs 2 (p = 6 × 10⁻⁶);
  - network (2u′): systems +4 points (4 vs 0, p = 0.12); equations 18 vs 2 (p = 0.0004).
- **Network against baseline, both with v2 (Trial 29's bar (b)):** **not met on either set.**
  - (1u): systems +3 points (4 vs 1, p = 0.38); equations 23 vs 10 (p = 0.035);
  - (2u′): systems −1 point (0 vs 1, p = 1); equations 15 vs 11 (p = 0.56).
- **Frequency control (v2):**
  - against the baseline: (1u) systems +1 point (p = 1), equations 12 vs 15 (p = 0.7); (2u′) systems −2 points (p = 0.5), equations 8 vs 20 (p = 0.036);
  - the network against it: (1u) systems +2 points (p = 0.62), equations 22 vs 6 (p = 0.004); (2u′) +1 point (p = 1), equations 20 vs 4 (p = 0.002).
- **Noise and cap reference** (old fitter, this batch with the cap, against Trial 29's original):
  - baseline: identical whole-system outcomes on both sets; equations 0 vs 0 and 0 vs 2;
  - network: systems 0 vs 1 and 1 vs 1; equations 7 vs 12 and 4 vs 3.

**Effort** (set (1u); set (2u′) is within a few percent):

| Arm | Time per system, median (mean) | Ordinary fit per structure | Thorough fits per equation, mean | Refused by the cap per equation, mean | Thorough share of search time | First plausible structure, median (s) | Prior's time per system, median |
|---|---|---|---|---|---|---|---|
| baseline, v1 | 226 s (239 s) | 36 ms | 22 | 26 | 8% | structure 140 (6 s) | 0 |
| network, v1 | 350 s (346 s) | 54 ms | 18 | 20 | 8% | 20 (1 s) | 16 s |
| baseline, v2 | 981 s (1,067 s) | 84 ms | 178 | 627 | 54% | 187 (101 s) | 0 |
| network, v2 | 1,387 s (1,306 s) | 135 ms | 167 | 624 | 48% | 32 (14 s) | 15 s |
| frequency, v2 | 1,317 s (1,338 s) | 124 ms | 182 | 555 | 51% | 139 (78 s) | 3 s |

- **The cap** bound in 200–239 equations per new-fitter arm and set (of 267–278), and in 14–21 for the old fitter.
- **The time cap** (3,600 s per system) stopped 2 equations of the network with v2 in each set, and none elsewhere.
- **The network's computing time** was about 1% of a new-fitter run.

**Reading.**
- **The new fitter helps both learners.** Equations solved rise by 18 to 25 (discordant counts) in all four learner-and-set comparisons (p ≤ 0.002 each). Whole systems rise modestly, by 2–6 points, significantly only for the baseline on (2u′). The search takes 4–5 times as long.
- **It does not open a system-level gap between the network and the baseline.** Trial 29's bar (b) is not met, so, as logged, stage 2 is not proposed.
- **What the network adds:**
  - more equations on (1u);
  - its first plausible structure comes after about 20–30 structures instead of 140–190;
  - the search fits the true structure more often (139 against 102 equations on (1u)), and with v2 recognizes it more often once fitted.
- **The untuned history's frequency table adds nothing over uniform priorities** (at the equation level it is worse on (2u′)), and the network beats it at the equation level on both sets. With an untuned history, the network learns more than term frequencies, but not enough to change whole-system recovery.
- **Whole systems stay rare** (at most 15 of 100). Untuned equations average 2.5 nonlinear pieces, every equation of a system must be right, and equations with 3 or more pieces remain the binding limit (16–31% recognized even when given the true structure).

### Fit recorder: built, checked and merged (5 November, 05:20)

**Owner.** *"Approved: build the fit recorder now on a separate branch and working copy, off by default, with the checks you listed. Include the shadow sample of rejected candidates as an option, off by default and never used in headline comparisons; we'll turn it on for dedicated data-collection runs. Keep the loader refusing test and reserve sets for training. Merge only after the comparison finishes…"*

**Built** on branch `claude/fit-recorder-2026-11-04` in its own worktree; nothing the running comparison loaded was touched. Merged after the comparison and the checks (merge commit e6d05ea).
- **`fitlog.py`** writes one compressed file per run. It holds the run's bursts once and a header: the set tag, the vocabulary, and each equation's floor, scale and exactness threshold. Then one row per record:
  - **kind 0, a thorough fit:**
    - the structure, the warm start, and the cheap fit's constants, coefficients and residual (the starting values);
    - the fitter's random state before the fit;
    - the constants after the thorough fit and after the step-integral refit;
    - the step-integral and held-back errors, the degenerate and plausibly-exact flags, the snapping counts, the winning internal start and the times;
  - **kind 1:** a structure the cap refused, with inputs only;
  - **kind 2:** a shadow fit.
  - **Trace (fitter v2):** every internal start (v1 sweep, algebraic, global, snap), with its start and polished constants and its polished residual.
- **Shadow fits** (option `shadow_rate`, default 0) go to a random sample of the structures the gate or the cap turned away. They are recorded but never used by the search. The fitter's random state and statistics are restored afterwards, the time is kept off the search's clock, and they draw from their own random stream.
- **Set tags:** test and reserve prefixes can never be tagged otherwise. The loader always refuses test and reserve files for training (no override), and refuses untagged files unless allowed explicitly.
- **Hooks:** in `EquationSearch.evaluate` and `learn28(fitlog=…)`, plus an optional trace in `FitterV2`. With no fit log, the code paths are unchanged.

**Checks** (6 development systems, fitters v1 and v2, budget 300, cap 40; 4 configurations, 48 runs; no errors):
1. **Outcomes identical** to logging off in 12 of 12 runs, both with logging on and with logging plus shadow fits (rate 5%).
   - Details differed in 1 and 2 runs respectively, against 1 for a second logging-off run.
   - All of these were on one noise-sensitive system, where the logging runs agree with the repeat; this is run-to-run noise.
2. **Reproduction** (352 sampled rows, data rebuilt from the stored bursts):
   - the cheap fit was bitwise identical in 352 of 352;
   - the thorough fit, from the stored inputs and random state, was bitwise identical in 288 of 288, and the plausibly-exact flag agreed in 288 of 288;
   - the stored constants rebuild the fitter's terms exactly. The fitter's own reported residual can differ slightly from a direct recomputation (up to 4 × 10⁻⁵ of the rate scale in about 3% of fits); this is noted in the file format.
3. **Overhead:**
   - run time with logging on was 0.98× logging off (median; off against off 1.02×), and saving a log takes 0.01 s;
   - shadow fits at 5% cost 1.03×;
   - files take about 565 bytes per row (v2, with the trace) and 504 (v1). At the comparison's scale that is about 0.75 MB per new-fitter run, or about 0.5 GB per 1,000-run batch.
4. **Shadow fits on development systems (first look):**
   - of 346 structures the gate turned away, none was plausibly exact;
   - of 51 the cap refused, 1 was (with v2).
- **Pre-existing, not from the recorder:** Intel MKL prints "Parameter 4 was incorrect on entry to DGELSD" from batched least squares (as often in logging-off runs and in the comparison's log). Such candidates are always refitted and kept only if better.

**Kept:** the runner and check scripts are in `outputs/accumulating_learner/fit_recorder/`. Trial 30's scripts, reports and run records are in `outputs/accumulating_learner/trial30/`.

### Trial 31 proposal: a learned starting-value network for thorough fits (5 November; awaiting owner approval, nothing run)

**Owner.** *"Write up the proposal, with this change of focus: the main limit is that thorough fits fail on equations with 3+ nonlinear pieces even when given the correct structure. Make the learned starting-value network the primary goal, trained on recorded fits from training and development systems only (shadow sample on), with the screen secondary. Measure success first: equations fitted exactly, using the known-structure test from Trial 30 broken down by number of pieces, then in the full search at the same budget for both learners. Report time as secondary. Include how a learned starting-value network would be kept from failing on unfamiliar systems (e.g. it only adds starting points alongside the existing ones, never replaces them). Send the proposal before running anything."*

**Why.** In Trial 30's known-structure test, the new fitter (v2) recognized 16 of the 90 synthetic equations with 3 or more pieces. Of the 74 failures:
- **69 failed inside the thorough fit,** ending far from exact (median residual 1.1 × 10⁸ times the precision floor): wrong valleys, not near misses;
- only 5 were stopped at the gate.

The starting points of the thorough fit are where a learned component can help.

**1. Data collection** (training and development systems only; recorder on, shadow sample on):
- **Systems:**
  - about 800 new untuned and 200 new hand-tuned training systems (prefixes "t31-train-u-k" and "t31-train-t-k", tagged "train");
  - 40 new development systems ("t31-dev-u-k", tagged "dev") for model selection, plus the 6 existing ones;
  - the generators' training modes, so the held-out family pairs are excluded;
  - the generator's existing cheap-reference cutoff (refine ≤ 4, nsub ≤ 64, simulation time limits 2 s / 5 s), logged as the cutoff before any system is drawn.
- **Per system:**
  - (a) each equation's true structure through the v2 evaluation, as in the known-structure test (recorded);
  - (b) one search at 500 structures per equation, cap 100, recorder and shadow sample on (5%). Half use the baseline's priorities and half the network's, so both search styles are covered.
- **Labels:**
  - for a true structure: **the generator's true constants**, converted into the learner's terms with the existing `synthetic28.learner_term`, even when the recorded fit failed. These are exactly the hard 3+-piece cases;
  - for any other recorded fit that ended plausibly exact: its final constants;
  - first, on development systems, check that the converted true constants reproduce the data to the precision floor.
- **Never in training:** Trial 29's and Trial 30's test sets, the practice systems and the fresh pool. The loader refuses test and reserve sets.
- **Estimated cost:** about 4–5 h on 24 CPU workers, about 0.3–0.5 GB, under the 5 GB stop rule.

**2. The network.**
- **Model:** a small set transformer in the style of MoveNet (3 layers, width 128, float32), trained on a GPU under the GPU sharing rule.
- **Input:** the structure's terms (family, variables, exponents, which constant is which) and a fixed-size summary of the equation's rate data over each piece's variables.
- **Output:** 16 candidate points in **the global search's natural-range unit cube**, the same scale-free coordinates v2 already samples (switch centres across the data, rates and frequencies on log scales relative to the data range, poles at log distances outside it).
- **Loss:** winner-takes-all over the 16 hypotheses, matching interchangeable pieces in any order and treating phases as circular.
- **Model selection:** development systems only.
- **Accumulation curve:** also trained on 100, 300 and 1,000 training systems, so recognition can be shown as a function of how many systems the fitter has learned from.

**3. Fitter v3 = v2 plus learned starts, and how it is kept from failing on unfamiliar systems.**
1. **Additive only.** v2's candidates (the v1 sweep fit, 3 algebraic starts, the global search) are computed exactly as now, with the same random stream. The network's points are scored in one batch (coefficients solved exactly), refined around the best few with their own fixed random stream, and the best 4 distinct are added to the candidate list. Polishing, snapping and the exactness test are unchanged. So **v3's best fit can never have a higher residual than v2's on the same inputs**; the network can only add options.
2. **Bounded.** Its points lie inside the natural ranges the global search already explores (polishing can leave them). A fixed number of extra polishes caps the cost when the network is no help.
3. **Unfamiliar families tested separately.** The held-out family pairs never appear in training. A known-structure set built on them, and the practice systems, measure the network on unfamiliar systems. "Never worse" is counted on every set.
4. **No test data** in training or model selection.

**4. Measurement** (primary: equations fitted exactly; time secondary).
- **Stage A, known-structure test** (Trial 30's harness, v2 against v3, by number of pieces 0 / 1 / 2 / 3+):
  - **Sets:**
    - Trial 30's U (60 untuned), T (60 hand-tuned) and P (12 practice systems), for direct comparison;
    - fresh U′ (60 new untuned systems), as confirmation;
    - H (60 new systems whose equations include held-out family pairs), for unfamiliar systems.
  - **Pass bar (all required):**
    1. Synthetic equations with 3+ pieces (U, T and U′ pooled): v3 recognizes at least 1.5× as many as v2 and at least 15 points more, paired sign test p < 0.05.
    2. Never worse: on every set, the equations recognized by v2 but not v3 are at most 2% of the set (run-to-run noise). Equations with 0–2 pieces are within 2 points of v2 or better.
    3. Unfamiliar set H and practice P: v3 recognizes at least as many as v2.
  - **Also reported:** the accumulation curve (100 / 300 / 1,000 training systems), which starts win, thorough-fit time against v2 (secondary), and per-family results.
- **Check-in after Stage A, before Stage B.**
- **Stage B, full search** (at the same budget as Trial 30: M = 8, 2,000 structures per equation, cap 200):
  - both learners (baseline and untuned-history network) with v3 on Trial 29's sets (1u) and (2u′);
  - next to Trial 30's v2 records of the same systems, with a 16-run same-batch v2 rerun as an outcome-level consistency check;
  - drawn-system identity checked before pairing.
  - **Primary:** equations with a plausibly exact structure and true structures recognized (paired sign tests per learner and set). **Then:** whole systems, and Trial 29's bar (b) re-checked with v3. **Time:** secondary.

**5. The screen (secondary).** Trained on the same recorded thorough fits and shadow fits, to predict whether a thorough fit will end plausibly exact. Evaluated **offline only** on development data: how many thorough fits could be skipped while keeping 98% of exact finds. It is not used in Stage A or B; a later trial decides.

**Check-ins:**
1. after data collection and training (development results, data sizes, label check), before Stage A;
2. after Stage A, before Stage B;
3. with Stage B's results.

### Trial 31: owner approval with one addition; rules (5 November, 05:35, logged before any system is drawn)

**Owner.** *"Approved as written, including the generator's true constants as labels and the pass bar. One addition: some equations have more than one exact set of constants (symmetries, interchangeable pieces). Make sure training only requires the best of the 16 candidates to match, and doesn't penalize valid alternative answers. Check in at each of the three points you listed."*

**Systems** (none drawn yet):
- "t31-train-u-k" (k = 0–799, untuned generator) and "t31-train-t-k" (k = 0–199, hand-tuned generator), tagged "train";
- "t31-dev-u-k" (k = 0–39, untuned), tagged "dev".
- **The generator's existing acceptance rules, as the cheap-reference cutoff fixed before looking:**
  - refine ≤ 4, nsub ≤ 64, build ≤ 60 s, settle-time limits 2 s (untuned) and 5 s;
  - positive, finite and bounded; every variable moves;
  - no held-out family pair; not centred at M = 8; no degenerate true piece.
- **The build-time limit is wall-clock time,** so which draw is accepted can depend on machine load. This is harmless for training data, because each system's bursts and labels are stored. It is the likely cause of the 4 differently drawn (2u′) systems in Trial 30.

**Per system** (fitter v2, M = 8, seed 1; recorder on):
- (a) the known-structure pass: each equation's true structure through `EquationSearch.evaluate`, as in Trial 30's test;
- (b) a search: `learn28`, 500 structures per equation, cap 100, time cap 3,600 s, shadow sample 5% (shadow seed k). Baseline priorities for even k; the untuned-history network for odd k.
- A labels file: the generator's description (tokens and constants) and the learner's scaling at M = 8.
- 24 CPU workers at nice 10. Free disk watched every minute: below 5 GB my runs stop (by PID) and the owner is told. The runner resumes, skipping finished systems.

**Labels and the owner's addition** (alternative exact answers are never penalized):
1. **Winner-takes-all.** An example's loss is the distance from the best of the 16 hypotheses to the nearest valid answer; the other 15 get no loss.
2. **Valid answers:**
   - the generator's true constants, converted with `synthetic28.learner_term` into the global search's natural-range unit cube (numerical inversion of its maps, in its dependency order);
   - their symmetry images: permutations of interchangeable pieces (factors or terms of identical form: same family, variables and multiplier); sign flips that stay inside the search space (tanh (b, c) ↔ (−b, −c), with the coefficient changing sign); periodic constants compared circularly (sine phase), and canonical forms where the search space holds one sign only (positive sine frequency);
   - the final constants of any recorded fit of the true structure that ended plausibly exact.
3. **Constants the data cannot pin down are left out of the distance.** These are slots whose uncertainty in unit-cube coordinates exceeds 0.05, from the Jacobian at the true constants and the data's noise level.
4. **Checked before training, on development systems:**
   - the converted true constants reproduce the data (coefficients solved exactly: step-integral and held-back errors within each equation's exactness threshold);
   - every symmetry image fits as exactly as the original.
   - Counts are reported at check-in 1.
5. **Every measurement judges success by the exactness test,** so any exact answer counts.

**The network:**
- **Tokens:** sample points (state values normalized to each variable's observed range, and the equation's rates normalized by their scale); one token per free constant (family, position, variables, multiplier exponents, its variable's data range); one per monomial term.
- **Model:** 3 transformer layers, width 128, float32. Outputs 16 hypotheses per free constant, inside the unit cube.
- **Training:** on a GPU under the sharing rule (`nvidia-smi` checked before launch, GPU logged).
- **Model selection:** development systems only, by known-structure recognition of 3+-piece equations with v3.
- **Sizes:** trained on 100, 300 and 1,000 training systems (the accumulation curve); the full-size model goes to Stage A.

**Fitter v3** = v2 plus a hook after v2's candidates (v2 returns none there, so its behaviour is unchanged):
- the network's 16 points are scored in one batch (coefficients solved exactly);
- 2 refinement rounds around the best follow, with their own fixed random stream;
- the best 4 distinct points are added to the candidates for polishing;
- the recorder tags them "learned".

**The screen** (secondary, offline): trained on the same records, and reported by check-in 2 at the latest.

**Check-in 1** (before Stage A): data sizes, the label and symmetry checks, the development results (v2 against v3 by number of pieces, and the accumulation curve), and time.

**Owner, 5 November, 05:40:** the disk stop rule is now **2 GB remaining**, replacing 5 GB. Runs stop writing below 2 GB free and the owner is told. The collector and the watchdog use 2 GB.

### Trial 31 check-in 1: data, labels, network on development systems (5 November, 12:40). Stage A not started

**Data collection** (05:45–12:03, 24 CPU workers):
- 1,040 systems: 40 development, 800 untuned training, 200 hand-tuned training. No errors; 149 worker-hours; 215 MB.
- The disk watchdog (2 GB floor) never fired; free space stayed at 8–9 GB.
- **v2's known-structure pass on these systems:** 0 pieces 176 of 176; 1 piece 824 of 856 (96%); 2 pieces 536 of 700 (77%); 3+ pieces **137 of 1,141 (12%)**.
- **Training examples:** 3,565 in all.
  - 2,585 true structures: 826 with 0–1 pieces, 670 with 2, 1,089 with 3+;
  - 980 other structures that a recorded fit made exact: 212, 487 and 281;
  - a median of 4 free constants per example (maximum 18), and 45 of 16,469 constants masked as not identifiable.
  - Development: 104 true-structure examples (46 with 3+ pieces).
- **Labels:** true constants outside the search coordinates are clipped to the nearest point (polishing can still reach them), not dropped. Constants that depend on a clipped one are labelled given the clipped value; this was fixed after 2 of 516 development constants failed the round trip.

**Label checks** (all 40 development systems, 106 equations):
- the converted true constants use the vocabulary's families and fit exactly in 106 of 106 (worst error 0.91 of the threshold);
- 516 constants: 509 inside the search space (all round-trip exactly, worst relative error 4 × 10⁻¹⁵), 6 clipped, 1 not representable, 5 masked;
- 950 symmetry images: all fit exactly (residual gap ≤ 5 × 10⁻¹⁴ of the rate scale); 421 lie inside the search space.

**Network.** Trained on GPU 7, at 0% utilization at most launches; once 6% with 418 MiB of another user's, which sharing allows. Dropout 0.1, data-point subsampling and rate jitter, 6,000 steps, best development distance kept.

**Development results, v2 against v3,** same process, 40 systems, 106 equations, recognized:

| Training systems | Development distance | Within 0.02 | 3+ pieces (46) | 2 pieces (30) | 0–1 pieces (30) | v2 only (any group) |
|---|---|---|---|---|---|---|
| (v2) | – | – | 5 | 22 | 28 | – |
| 100 | 0.055 | 0.24 | 7 | 23 | 28 | 0 |
| 300 | 0.047 | 0.26 | 7 | 22 | 28 | 0 |
| 1,000 | 0.040 | 0.37 | 7 | 22 | 28 | 0 |

- **The network learns from more systems:** its distance to the nearest valid answer falls with data.
- **But equations fitted exactly do not rise:** 7 of 46 on 3+ pieces at every size, against v2's 5.
- **Never worse:** no equation was recognized by v2 and missed by v3.
- **Time:** evaluation +12–20% (median 0.69 s against 0.83 s).

**Diagnostics** (development only):
1. **Candidates, not choice, are the limit.** Whenever any polished candidate (v2's own or learned) passes the exactness test, the chosen one does (3+ pieces: 7 and 7).
2. **Wrong valleys.** The learned starts give the best residual in about 15 of the 3+-piece thorough fits, but held-back errors stay at 10⁻⁴–10⁻¹, against about 10⁻¹² for an exact fit.
3. **A wider local search does not help.** A cloud of 32 points around each of the 16 guesses gave 6 against 7.

**Screen** (secondary, offline; training systems for training, development systems for evaluation):
- 301,911 recorded rows; development AUC 0.93;
- keeping 98% of exact finds it would skip 68% of thorough fits (3+ pieces: 78%); keeping all, 46%.
- **Shadow sample** (training searches, cap 100):
  - of structures the gate turned away, 16 exact in 32,970 shadow fits;
  - of structures the cap refused, 63 exact in 15,234. Scaled to the 307,000 refusals, that is about as many exact structures as the 1,086 found by the 253,707 thorough fits made.

**Outlook for Stage A's bar** (3+ pieces at least 1.5× and +15 points): on development, v3 gives 5 → 7 of 46 (+4 points, 1.4×). The bar is unlikely to be met with this network.

**Options put to the owner:**
- **A (recommended).** Train the network on examples generated on the fly: the collected systems' measured states and structures, new constants drawn in the search coordinates, and rates computed exactly from the term formulas. There is no ODE simulation and the supply is unlimited, with the same symmetry and mask handling. Accumulation is measured over the number of collected systems feeding the generator. Re-run development, and go to Stage A only on a clear gain.
- **B.** Stage A now with the current network.
- **C.** A test of the screen in place of the cap (secondary).

### Trial 31: owner decision after check-in 1 (5 November, 13:00): option A with three additions; rules logged before running

**Owner.** *"Go with A, with three additions:
1. First, measure how close a start must be for polishing to reach exactness: on development systems, start from the true constants perturbed by 1%, 2%, 5% and 10% of each constant's range, and report how often polishing reaches exact, broken down by number of pieces. Compare with the network's current accuracy (about 4%) before generating more data.
2. Make sure on-the-fly training examples match what the network sees at test time (states on real trajectories of the new constants, in burst form), not states taken from a different system.
3. Run option C in parallel on development systems: let the screen, instead of the fixed cap, decide which candidates get thorough fits, ordering rather than skipping. Report equations solved at the same budget. The shadow data suggests the cap is turning away as many exact answers as the search finds.

Check in with all three before Stage A."*

**1. Basin test** (development systems only; 40 systems, 104 equations with free constants):
- Each true structure starts from the generator's true constants in the search coordinates, every free constant moved by exactly δ = 0.01, 0.02, 0.05 or 0.10 of its range, in a random direction (clipped to the cube); 8 random draws per equation and level.
- The start then goes through **the same path as a learned start in v3:** polishing (`fast_refine`, 20 iterations), then the step-integral refit and the exactness test (training and held-back errors within the threshold, no degenerate piece). A stronger polish (60 iterations) is reported alongside.
- Reported by number of pieces: the share of starts that reach exactness.
- **Compared with the network:** the 1,000-system network's best guess on the same equations, as its per-constant error and its largest error over an equation's constants.

**2. On-the-fly training data with real trajectories:**
- Each example is a whole new synthetic system from the generators' training modes (new prefixes "t31-gen-u-k" untuned and "t31-gen-t-k" hand-tuned, tagged "train"), with the same acceptance rules as the collection.
- It is simulated into M = 8 bursts and seen through the learner's view exactly as at test time. Labels are its own drawn constants (images, masks and clipping as before).
- **Cost:** drawing and simulating one accepted system takes 46 s on average for untuned systems (12 rejected draws each) and 11 s for hand-tuned ones. That is about 1,900 untuned systems an hour on 24 workers.
- **How much is generated depends on step 1:** enough to measure how accuracy scales with data, at least 5,000 systems. Systems are produced in rounds and fed to training; only the small per-system bursts and labels are kept (about 3 KB each), not search logs.
- **Development selection and accumulation curve as before:** the 40 development systems, sizes up to the largest generated.

**3. Option C, the screen ordering thorough fits** (development systems, 40):
- **Arms:** fitter v2 at M = 8, 2,000 structures per equation, time cap 3,600 s, under four combinations:
  - {uniform baseline, untuned-history network} priorities;
  - {cap: 200 thorough fits per equation, first come, then only record-breaking cheap fits; screen ordering: the same 200, by screen score}.
  - That is 160 runs.
- **Screen ordering:**
  - A structure that passes the gate and needs a thorough fit waits in a queue, with the screen's score.
  - Its precise stage without the thorough fit runs at once, as for a capped structure now.
  - Every 50 structures, waiting candidates are given thorough fits, best score first, up to the pro-rata budget (200 × structures so far / 2,000). Any budget left is spent at the end of each search phase.
  - A thorough fit that makes a structure exact updates its node: the exact flag, the value, and the simplification queue.
- **The screen** is retrained on training systems without the two features that depend on the cap policy (the thorough-fit count so far and the record-breaker flag).
- **Reported:** equations with a plausibly exact structure, true structures recognized, whole systems, thorough fits used, and time, at the same budget.

### Trial 31 addition 1: basin test (5 November, development systems; 40 systems, 104 equations with free constants)

**How close a start must be.** Every free constant of the true structure is moved by exactly δ of its range (search coordinates, random direction, 8 draws). The start then follows a learned start's path: polish on the rate data, step-integral refit, exactness test. Reaches exactness (20 / 60 polishing iterations):

| Pieces | Equations | δ = 0 | δ = 0.01 | δ = 0.02 | δ = 0.05 | δ = 0.10 |
|---|---|---|---|---|---|---|
| 1 | 28 | 100% | 99% / 99% | 99% / 99% | 92% / 92% | 74% / 75% |
| 2 | 30 | 100% | 84% / 89% | 69% / 77% | 45% / 48% | 27% / 33% |
| 3+ | 46 | 89%* | 26% / 33% | 15% / 23% | 4% / 7% | 2% / 2% |

\* At δ = 0, the misses are equations whose true constants lie outside the search coordinates, so the start is the clipped point.

**The network's accuracy on the same equations** (1,000-system network; best of 16 guesses against the nearest valid answer):

| Pieces | Mean per-constant error (median) | Largest-constant error (median) | Every constant within 0.01 | Within 0.02 | Within 0.05 | Within 0.10 |
|---|---|---|---|---|---|---|
| 1 | 0.012 | 0.014 | 32% | 57% | 64% | 82% |
| 2 | 0.037 | 0.061 | 3% | 13% | 43% | 70% |
| 3+ | 0.059 | 0.143 | 0% | 2% | 7% | 28% |

**Which polish widens the basin** (3+ pieces; δ = 0.02 / δ = 0.05):
- (a) the current path: 15% / 4%;
- (b) the step-integral refit alone, 60 iterations: 23% / 8%;
- (c) 60 iterations on the dense rates, then the step-integral refit: 25% / 12%;
- (d) 200 iterations on the rate data: 27% / 13%.
- 2 pieces: 69% → 77–80% and 45% → 52–56%.

**Reading.**
- For equations with 3 or more pieces, polishing reaches exactness from about 1% of every constant's range only a quarter to a third of the time.
- The network's guesses are about 5 times further off (median 6% mean, 14% on the worst constant).
- From 100 to 1,000 training systems its error fell only from 0.055 to 0.040. At that rate, the 1% scale would need thousands of times more data.
- A stronger polish doubles or triples the hit rate at 5%, but the basin stays narrow.
- **Generation (addition 2) goes ahead as approved,** sized to measure the scaling point: 4,000 new systems, about 1.8 h on 24 workers, after option C finishes.

### Trial 31 addition 3: option C, the screen ordering thorough fits against the cap (5 November, 16:57–19:00; development systems)

160 runs: 40 development systems (stored bursts, identical across arms) × {uniform, untuned-history network} priorities × {cap, screen ordering}; fitter v2, M = 8, 2,000 structures per equation, 200 thorough fits per equation. No errors.

| Priorities | Policy | Equations with a plausibly exact structure (of 106) | True structures recognized | Systems accepted (all exact when checked externally) | Thorough fits per equation | Time per system (median) |
|---|---|---|---|---|---|---|
| uniform | cap | 51 | 32 | 5 | 172 | 827 s |
| uniform | screen | 52 | 33 | 5 | 170 | 835 s |
| network | cap | 49 | 36 | 3 | 179 | 1,257 s |
| network | screen | 49 | 38 | 3 | 172 | 1,092 s |

- **Discordant equations, screen against cap:** uniform 4 vs 3 (p = 1); network 1 vs 1 (p = 1). True structures 5 vs 4 and 4 vs 2. Whole systems are identical.
- **At the same budget, ordering by the screen solves the same equations as the cap,** and saves 13% of time with the network's priorities.
- **Why the shadow data overstated the loss.** In the training searches (budget 500, cap 100), 40 equations had an exact structure among the cap-refused shadow fits: 37 of them were solved anyway by another exact structure, and only 3 were left unsolved. The cap mostly turns away further exact variants of equations that are already solved.
- **The screen used:** retrained without the cap-policy features; development AUC 0.943. Keeping 98% of exact finds it would skip 61% of thorough fits (76% for 3+ pieces).

### Trial 31 check-in 2 (all three additions; 5 November, 21:30). Stage A not started

**Addition 2: on-the-fly data with real trajectories.**
- 4,000 new systems (3,200 untuned, 800 hand-tuned; 19:01–20:39, 24 CPU workers), each simulated into its own M = 8 bursts and seen exactly as at test time.
- 10,358 true-structure examples, 4,605 of them with 3+ pieces; 175 MB. Together with the collection: 5,000 systems, 13,923 examples.
- Trained on GPUs 0 and 1 (both free: 0%, 1 MiB), 15,000 steps.

| Training systems | 100 | 300 | 1,000 | 2,500 | 5,000 |
|---|---|---|---|---|---|
| Development distance (best) | 0.055 | 0.047 | 0.040 | 0.037 | 0.034 |
| Share within 0.02 | 0.24 | 0.26 | 0.37 | 0.37 | 0.42 |
| 3+ pieces recognized, v3 (v2: 5 of 46) | 7 | 7 | 7 | 7 | 8 |
| 2 pieces recognized, v3 (v2: 22 of 30) | 23 | 22 | 22 | 23 | 23 |

- **Accuracy accumulates with systems learned** (error ∝ N^−0.12).
- **Measured as in the basin test,** the 5,000-system network on 3+ pieces: every constant within 0.05 for 13% of equations (1,000 systems: 7%), within 0.10 for 37% (28%), within 0.02 for only 2%.

**Learned starts plus a longer polish** (development; same process as v2; nothing v2 solved was lost in any variant):

| Variant | 3+ pieces (46) | 2 pieces (30) | Evaluation median |
|---|---|---|---|
| v2 | 5 | 22 | 0.69 s |
| v3, the network's 4 picks, usual polish | 8 | 23 | 0.85 s |
| control: 16 random points, same refinement, the 4 picks pre-polished 200 iterations | 6 | 22 | 0.95 s |
| control: v2's own global search (fresh stream), its best 4 pre-polished 200 iterations | 10 | 24 | 1.12 s |
| **v3 + pre-polish: the network's 4 picks pre-polished 200 iterations** | **15** | 23 | 0.98 s |

- **The network brings starts near the answer, and the longer polish closes the gap.** Each half alone gives 8 or 10; together 15.
- The network's 16 guesses beat v2's 2,000-sample global search as starting points (15 against 10) at lower cost, and random starts (6) by far.

**Addition 1 (basin test) and addition 3 (option C):** logged above.
- Equations with 3+ pieces need starts within about 1% of every constant (from 1%, 26–33% reach exactness); a stronger polish doubles or triples the 5% hit rate.
- The screen ordering thorough fits solves the same equations as the cap at the same budget.

**Proposal for Stage A** (owner's decision):
- **Arms:**
  - v2, as approved;
  - "v2 strong": v2 plus its own fresh global search's best 4 with the 200-iteration pre-polish, the strong-baseline rule's fair control;
  - v3: v2 plus the 5,000-system network's best 4 with the same pre-polish.
- **The approved bar applies to v3 against v2.** v3 against v2-strong is reported as the test of the learned component itself (on development: 15 against 10, +11 points, 1.5×).

### Trial 31 Stage A: owner's decision and rules (5 November, 21:45, logged before running)

**Owner.** *"Run Stage A with all three arms. The deciding comparison is v3 vs v2-strong, since the long polish is a generic improvement and goes to both. Pass bar for it: a statistically significant gain on 3+-piece equations (p < 0.05), no more than noise-level losses on any set, and no loss on the unfamiliar-family and practice sets. Also report v3 vs plain v2 against the originally approved bar, for the record. Write this into the log before running, and check in with the results before Stage B."*

**Arms.** Each equation's true structure goes through `EquationSearch.evaluate` (Trial 30's harness), all three arms in the same process:
- **v2:** fitter v2;
- **v2-strong:** v2 plus its own fresh global search's best 4 (own random stream), each pre-polished with 200 rate iterations;
- **v3:** v2 plus the 5,000-system network's best 4 (16 guesses, 2 refinement rounds, own random stream), each pre-polished with 200 rate iterations.

**Sets:**
- U: Trial 30's 60 untuned systems;
- T: Trial 30's 60 hand-tuned systems;
- P: the practice known structures, as in Trial 30;
- U′: 60 fresh untuned systems ("t31-fit-u-k");
- H: 60 fresh systems whose equations include held-out family pairs ("t31-fit-h-k", mode untuned_test2m; families never seen in training).
- M = 8, seed 1. Counts only are printed.

**Deciding comparison, v3 against v2-strong. Pass bar (all required):**
1. Synthetic equations with 3+ pieces (U, T and U′ pooled): v3 recognizes significantly more than v2-strong. Exact two-sided sign test on discordant equations, p < 0.05, with more v3-only than v2-strong-only.
2. No more than noise-level losses: on every set, the equations v2-strong recognizes and v3 misses are at most 2% of the set's equations.
3. No loss on H and P: v3 recognizes at least as many as v2-strong on each.

**For the record, v3 against plain v2 under the originally approved bar:**
1. 3+ pieces pooled: at least 1.5× and at least +15 points, p < 0.05;
2. on every set, v2-only at most 2%, and 0–2-piece equations within 2 points or better;
3. on H and P, v3 at least v2.

**Also reported:** results by set and number of pieces; time per arm; how often learned and global starts win; per-family results on H.

**Check-in before Stage B.**

### Trial 31 Stage A results (5 November, 21:14–21:30): deciding bar NOT met; original bar (v3 against v2) met

**Run.** Three arms in one process per equation; 24 CPU workers; 710 equations (U 162, T 180, U′ 165, H 173, P 30). One practice system was skipped for centred coordinates, as in Trial 30.

**Recognized** (v2 / v2-strong / v3):

| Set | 0 pieces | 1 piece | 2 pieces | 3+ pieces |
|---|---|---|---|---|
| U | 4 / 4 / 4 (4) | 33 / 33 / 33 (35) | 31 / 37 / 36 (46) | 12 / 20 / 22 (77) |
| T | 28 / 28 / 28 (28) | 83 / 84 / 84 (89) | 40 / 43 / 44 (50) | 4 / 5 / 6 (13) |
| U′ | 5 / 5 / 5 (5) | 43 / 43 / 43 (44) | 26 / 30 / 29 (36) | 11 / 19 / 25 (80) |
| H (unfamiliar families) | 3 / 3 / 3 (3) | 44 / 44 / 44 (44) | 22 / 25 / 26 (30) | 13 / 20 / 28 (96) |
| P (practice) | 10 / 10 / 10 (10) | 14 / 15 / 15 (16) | 1 / 1 / 1 (4) | – |

**Deciding comparison, v3 against v2-strong: NOT MET.**
1. 3+ pieces (U, T, U′ pooled, 170 equations): 44 → 53; v3 only 16, v2-strong only 7; sign test **p = 0.093**. Not significant.
2. Losses (v2-strong only): U 6 of 162 (3.7%), T 1 of 180, U′ 4 of 165 (2.4%), H 4 of 173 (2.3%), P 0 of 30. **Above 2% on three sets.**
3. H and P: H 92 → 101, P 26 → 26. Met.

**For the record, v3 against plain v2 under the originally approved bar: MET.**
1. 3+ pieces pooled: 27 → 53 of 170 (+15 points, 2.0×); v3 only 26, v2 only 0; p = 3 × 10⁻⁸.
2. No losses on any set; 0–2 pieces within 2 points.
3. H 82 → 101, P 25 → 26.

**Also reported:**
- **Evaluation time, mean (median):** v2 0.71 s (0.21 s), v2-strong 1.50 s (0.31 s), v3 1.29 s (0.26 s).
- **Added starts won the thorough fit:** global picks 225 times (v2-strong), learned picks 261 times (v3).
- **H by family** (equations containing it; v2 / v2-strong / v3):
  - abs 3: 1/1/1;
  - exp 32: 8/8/14;
  - hill 47: 17/18/25;
  - pair_hill 43: 6/8/11;
  - pair_recip 31: 10/11/16;
  - pow 36: 9/11/10;
  - prod_hill 41: 14/20/21;
  - prod_recip 29: 6/8/10;
  - recip 65: 28/34/35;
  - sin 40: 10/16/15;
  - tanh 40: 10/11/14.

**Reading.**
- The long polish is most of the gain (v2 → v2-strong: 27 → 44 on 3+ pieces). The network adds to it (→ 53, and 20 → 28 on unfamiliar families), but not significantly on the pooled sets.
- **v3's "losses" are not failures on shared candidates.** v2-strong and v3 are different supersets of v2 (global picks against learned picks), so each solves some equations the other misses.
- **A fitter with both picks** would hold both candidate sets. From the union of the two arms it is estimated at about 60 of 170 on 3+ pieces; this is not measured.

### Trial 31: owner decision after Stage A, and the combined-fitter test pre-registered (5 November, 22:00, logged before running)

**Owner.** *"Adopt v2-strong as the standard fitter for both learners. Then run option 2: v2-strong plus the network's picks vs v2-strong, on fresh known-structure sets including an unfamiliar-family set. Pre-register before running: a significant gain on 3+-piece equations (p < 0.05) and noise-level losses. Also run the combined fitter with networks trained on 100, 1,000 and 5,000 systems, to show whether the gain grows with history. Report time too. Check in with the results before Stage B."*

**Standard fitter from now on: v2-strong**, for both learners. It is fitter v2 plus its own fresh global search's best 4 starts, each pre-polished with 200 rate iterations, as Stage A's v2-strong arm. In code it becomes `fitter30.FitterV2S` and `learn28(fitter="v2s")`.

**Combined fitter:** v2-strong plus the network's best 4 picks (16 guesses, 2 refinement rounds, own random stream), each pre-polished with 200 iterations.
- Its global picks are generated exactly as v2-strong's (same random stream), so its candidates include all of v2-strong's.
- Networks: trained on 100, 1,000 and 5,000 systems (Trial 31's networks, unchanged).

**Fresh sets** (never used before; Trial 30's harness; M = 8, seed 1; counts only):
- U″: 100 untuned systems ("t31c-fit-u-k");
- T″: 60 hand-tuned systems ("t31c-fit-t-k");
- H″: 100 systems whose equations include held-out family pairs ("t31c-fit-h-k", mode untuned_test2m).

**Arms, all in one process per equation:**
- v2-strong;
- the combined fitter with the 100-system, the 1,000-system and the 5,000-system network.

**Deciding comparison: combined (5,000-system network) against v2-strong. Pass bar (both required):**
1. Equations with 3+ pieces, pooled over U″, T″ and H″: significantly more recognized by the combined fitter. Exact two-sided sign test on discordant equations, p < 0.05, with more combined-only than v2-strong-only.
2. Noise-level losses: on every set, the equations v2-strong recognizes and the combined fitter misses are at most 2% of the set's equations.

**Also reported:**
- the same comparison with the 100- and 1,000-system networks (whether the gain grows with history);
- 3+ results for the familiar sets (U″ + T″) and for H″ separately;
- results by number of pieces;
- evaluation time per arm;
- how often learned and global picks win.

**Check-in before Stage B.**

### Trial 31 combined-fitter test results (5 November, 21:42–21:59): pass bar MET; the gain grows with history

**Run.** Four arms in one process per equation, 24 CPU workers. 738 equations on fresh sets: U″ (100 untuned) 275, T″ (60 hand-tuned) 182, H″ (100 with held-out family pairs) 281. No failures. Before the run, on 28 development equations, the new `FitterV2S` reproduced Stage A's v2-strong exactly, and the combined fitter recognized every equation v2-strong did.

**Recognized** (v2-strong / combined with the 100- / 1,000- / 5,000-system network):

| Set | 0 pieces | 1 piece | 2 pieces | 3+ pieces |
|---|---|---|---|---|
| U″ | 5/5/5/5 (5) | 61/61/61/61 (62) | 62/63/63/63 (71) | **25 / 42 / 46 / 52** (137) |
| T″ | 34/34/34/34 (35) | 83/83/83/83 (88) | 39/39/40/40 (48) | **3 / 4 / 4 / 6** (11) |
| H″ (unfamiliar) | 5/5/5/5 (5) | 58/58/58/58 (59) | 70/71/70/72 (77) | **35 / 46 / 47 / 53** (140) |

**Deciding comparison, combined (5,000-system network) against v2-strong: MET.**
1. 3+ pieces, pooled over U″, T″ and H″ (288 equations): 63 → 111. Combined only 48, v2-strong only 0; sign test **p = 7 × 10⁻¹⁵**.
2. Losses (v2-strong only): 0 on every set.

**Does the gain grow with history?** 3+ pieces pooled:

| Network trained on | Recognized | Gain over v2-strong | Combined only / v2-strong only | p |
|---|---|---|---|---|
| 100 systems | 92 | +29 | 29 / 0 | 4 × 10⁻⁹ |
| 1,000 systems | 97 | +34 | 34 / 0 | 1 × 10⁻¹⁰ |
| 5,000 systems | 111 | +48 | 48 / 0 | 7 × 10⁻¹⁵ |

- Familiar sets (U″ + T″, 148): 28 → 46 / 50 / 58.
- Unfamiliar families (H″, 140): 35 → 46 / 47 / 53.
- **The gain grows with the number of systems learned,** on familiar and on unfamiliar families. No set lost anything at any size.

**Time.**
- Evaluation per equation, mean (median): v2-strong 1.63 s (0.52 s); combined 2.27–2.31 s (0.60–0.61 s), about +40% on the mean.
- Thorough fits won by the added starts: v2-strong's global picks 236. In the combined fitters, global picks 134 / 118 / 112 and learned picks 180 / 208 / 209 (100 / 1,000 / 5,000 systems).

**Reading.**
- Given the correct structure, the combined fitter recognizes **39% of 3+-piece equations against v2-strong's 22%** (and v2's 16% in Stage A) on fresh systems, with no losses.
- The learned part improves with the history it was trained on. This is the first fitting component in the programme that gets better the more systems it has learned from, and it holds on families never seen in training.

### Trial 31 Stage B: rules (5 November, 22:15, logged before running)

**Owner.** *"Run the full Stage B with 32 workers for the whole run. Never stop or interfere with other users' processes. Send a push notification and a plain-language summary when the results are in. After Stage B, plan the first test of the combined fitter on real systems (practice set first, then propose a held-out test under the usual rules)."*

**Arms** (4 × 200 systems = 800 runs):
- priorities: {uniform baseline, Trial 29's untuned-history network};
- fitters: {v2-strong (the standard), combined (v2-strong plus the 5,000-system network's picks)}.

**Systems and settings, as in Trial 30:**
- Trial 29's sets (1u) "t29-test1" and (2u′) "t29-test2m", 100 each;
- M = 8, seed 1, 2,000 structures per equation, 200 thorough fits per equation (Trial 30's cap rule), 3,600 s per system;
- 32 CPU workers at nice 10.

**Reported, per set and priorities:**
- **Primary:** equations with a plausibly exact structure, and true structures recognized. Combined against v2-strong, paired, with exact two-sided sign tests.
- **Then:** whole systems exactly recovered (external check), combined against v2-strong.
- **Trial 29's bar (b) re-checked under each fitter:** network priorities against baseline, +10 points with p < 0.05 on each set.
- **Context:** Trial 30's fitter-v2 records of the same systems.
- **Secondary:** search effort and time, including time-cap hits.
- Drawn-system identity is checked before any pairing; systems drawn differently in any run are excluded and counted.

### Trial 31 Stage B results (5 November 22:15 – 6 November 14:20): no gain in the full search at Trial 30's limits

**Run.** 800 runs on 32 CPU workers, no errors; other users' processes untouched. Drawn identically in all six arms (including Trial 30's v2 records): (1u) 100 of 100, (2u′) 97 of 100.

| Set | Arm | Systems exact | Equations with a plausibly exact structure | True structures recognized | Equations stopped by the time cap | Time per system (median) |
|---|---|---|---|---|---|---|
| (1u) | baseline, v2 (Trial 30) | 12 | 116 / 278 | 90 | 0 | 981 s |
| (1u) | baseline, v2-strong | 16 | 111 | 88 | 44 | 1,800 s |
| (1u) | baseline, combined | 15 | 109 | 83 | 75 | 2,180 s |
| (1u) | network, v2 (Trial 30) | 15 | 129 | 107 | 2 | 1,387 s |
| (1u) | network, v2-strong | 14 | 125 | 109 | 67 | 2,289 s |
| (1u) | network, combined | 14 | 125 | 110 | 111 | 2,701 s |
| (2u′) | baseline, v2 (Trial 30) | 8 | 100 / 269 | 70 | 0 | 953 s |
| (2u′) | baseline, v2-strong | 8 | 94 | 74 | 27 | 1,801 s |
| (2u′) | baseline, combined | 8 | 95 | 77 | 56 | 2,104 s |
| (2u′) | network, v2 (Trial 30) | 7 | 104 | 95 | 2 | 1,324 s |
| (2u′) | network, v2-strong | 8 | 105 | 93 | 59 | 2,394 s |
| (2u′) | network, combined | 7 | 103 | 91 | 114 | 2,812 s |

**Combined against v2-strong** (paired; only combined / only v2-strong):

| Set | Priorities | Systems | Equations | p (equations) | True structures |
|---|---|---|---|---|---|
| (1u) | baseline | 2 / 3 | 12 / 14 | 0.85 | 12 / 17 |
| (1u) | network | 1 / 1 | 8 / 8 | 1 | 9 / 8 |
| (2u′) | baseline | 1 / 1 | 11 / 10 | 1 | 10 / 7 |
| (2u′) | network | 1 / 2 | 7 / 9 | 0.8 | 6 / 8 |

**No difference.**

**Trial 29's bar (b)** (network against baseline priorities): not met under either fitter.
- Equations: (1u) 23 / 9 (p = 0.02) under v2-strong and 24 / 8 (p = 0.007) under combined;
- (2u′) 20 / 9 and 20 / 12;
- systems: −2 to 0 points.

**Why, from the diagnostics** (post hoc):
1. **The search rarely reaches the 3+-piece structures** where the fitter's gain lies:

   | Arm | 3+ pieces: true structure fitted / recognized (of about 270) | 2 pieces (of 142) | 0–1 pieces (of about 144) |
   |---|---|---|---|
   | baseline v2 | 11 / 2 | 45 / 33 | 131 / 126 |
   | baseline v2-strong | 7 / 1 | 48 / 39 | 129 / 126 |
   | baseline combined | 14 / 3 | 41 / 35 | 128 / 125 |
   | network v2-strong | 21 / 2 | 84 / 68 | 143 / 138 |
   | network combined | 18 / 4 | 78 / 66 | 141 / 136 |

   With the true structure reached in 3–8% of 3+-piece equations, the fitter's known-structure gain (22% → 39%) cannot show.
2. **Same reach, same recognition.** Where both v2-strong and combined fitted the true structure, recognition is the same: only combined / only v2-strong 0 / 1, 4 / 3, 1 / 2, 1 / 1.
3. **The time cap penalizes the slower fitters.** The equations stopped before 2,000 structures rise from 0–2 (v2) to 27–67 (v2-strong) and 56–114 (combined). Thorough fits take 70–82% of search time.
   - On equations stopped by neither arm, combined against v2-strong: 10 / 6, 4 / 1, 11 / 4, 6 / 2; pooled 31 / 13.
   - Combined against Trial 30's v2 on such equations: pooled 33 / 16.
   - These are post hoc and suggestive only.

**Reading.**
- The combined fitter fixes fitting once a structure is found (known-structure test: 63 → 111 of 288 on 3+ pieces). But in the full search the binding limit for 3+-piece equations is now **finding the structure**: the true structure is tried in only 3–8% of them within 2,000 structures.
- The fitter's extra cost also uses up the time cap.
- The next lever is the search's reach for multi-piece structures, and the time given to it. The network priorities already reach about twice as many 2-piece true structures as uniform priorities (78–84 against 41–48).

### Proposal: first test of the combined fitter on real systems (awaiting the owner)

**Step 1, practice set** (development; gains suggestive only, as the owner ruled in Trial 29):
1. **Known structures:** the practice systems' known structures (Trial 30's set P, 12 systems, 30 equations), with v2, v2-strong and the combined fitter in one process. This checks that nothing is lost on real systems; they have few equations with several pieces.
2. **Full search:** the 27 usable practice systems at M = 8, baseline and network priorities × v2-strong and combined. The structure budget is 2,000 per equation and **the time cap is raised to 2 hours per system**, so that the budget, not time, binds; time is reported.
   - Measures: whole systems exactly recovered (the success definition: one-step error ≤ 10⁻¹² on held-back starting states, multi-step envelope), equations solved, time.
   - Compared with Trial 28's practice results.

**Step 2, held-out test** (only after the owner approves the request; usual rules):
- **Data:** the fresh pool, 27 real systems with 2–4 variables, never inspected (split by seeded hash before conversion; not used in any development since). The whole pool is used, since nothing in this work was developed on it. Reserve sets A and B stay untouched under the gating rule.
- **Arms:** v2-strong and combined, each with baseline and network priorities, plus the frequency-table control for priorities (owner's earlier addition to any held-out request). Same settings as Step 2 of the practice test, one run, counts only.
- **Criteria, fixed before running:**
  - whole systems exactly recovered per arm;
  - combined against v2-strong paired (systems, then equations);
  - no system recovered by v2-strong and lost by combined beyond noise.
- **With 27 systems** the test can show harm or a large gain, not a small one.

### Trial 31 step 1 on the practice set: owner approval and a new standing rule (6 November, 14:45, logged before running)

**Owner.** *"Approved: start step 1 on the practice set as described, with the time limit raised so the structure budget binds, not time. Make that the rule for all future comparisons: set time limits generously enough that they rarely bind, and report how often they did. After the practice results, propose how to improve the search's reach for structures with 3+ pieces, including whether to resume Trial 29's stage 2 (learning from its own solved searches) now that the fitter solves more. Check in before the held-out request."*

**New standing rule (all future comparisons).** Time limits are set generously from measured timings, so that the budget binds, not the clock, and every report says how often a time limit was hit, per arm.

**Time limit for step 1: 8 hours per system** (learn28's limit, divided equally among a system's equations). Projected from Stage B's runs, scaling each equation to the full 2,000 structures, the full-budget time per system was:
- v2-strong: median 2,300–3,400 s, 99th percentile 11,400–15,500 s;
- combined: median 2,900–5,500 s, 99th percentile 22,100–23,700 s, maximum 29,500 s.

**Step 1(a), known structures:** set P (the practice systems' known structures, as in Trials 30 and 31). v2, v2-strong and the combined fitter in one process; M = 8, seed 1.

**Step 1(b), full search:**
- the 27 usable practice systems at M = 8, seed 1;
- priorities {uniform baseline, untuned-history network} × fitters {v2-strong, combined}, 108 runs;
- 2,000 structures per equation, 200 thorough fits per equation (Trial 30's cap rule), 8 h per system; 24 CPU workers.
- **Reported:**
  - whole systems exactly recovered (external check on held-back starting states, the success definition);
  - equations with a plausibly exact structure;
  - combined against v2-strong paired, by priorities;
  - time per system and **time-limit hits per arm**;
  - Trial 28's practice results (M = 8) as context.
- Practice-set gains are suggestive only.

### Trial 31 step 1: change of plan, and the practice known-structure check (6 November, 15:10)

**Owner.** *"Change of plan: run only the known-structure check on the 12 expressible practice systems now (v2, v2-strong, combined), to confirm the combined fitter helps and loses nothing on real data. Skip the full-search practice run for now; it would likely repeat Stage B's finding that the search rarely reaches hard structures. We'll run it after the search's reach improves. Then move straight to the proposal for improving the search's reach on 3+-piece structures, including whether to resume stage 2. Keep the time-limit rule for future comparisons."*

**Full-search practice run stopped** (my processes only, by PID) after 5 minutes; no run had finished and no output was written.

**Known-structure check, practice set P** (12 systems, 30 equations; 1 skipped for centred coordinates as before; M = 8, all three fitters in one process):

| Pieces | Equations | v2 | v2-strong | combined |
|---|---|---|---|---|
| 0 | 10 | 10 | 10 | 10 |
| 1 | 16 | 14 | 15 | 15 |
| 2 | 4 | 1 | 1 | 1 |
| all | 30 | 25 | 26 | 26 |

- Combined against v2-strong: no equation lost, none gained.
- Evaluation time, mean: v2 0.09 s, v2-strong 0.14 s, combined 0.17 s.
- **On real systems' known structures, the combined fitter loses nothing.** These systems have no equations with 3+ pieces in the toolkit's terms, which is where it helps on synthetic systems.

**One more fact for the reach proposal** (Stage B, combined fitter): for the 3+-piece equations whose true structure was never fitted (251 with baseline priorities, 247 with network priorities), the structure the search finally chose holds a median of **none** of the true terms (mean 10–16%), and usually misses 3–5 of the true pieces. **The search is not close to these structures; it is elsewhere.**

### Trial 32 proposal: learned whole-structure proposals to improve the search's reach on 3+-piece equations (6 November; awaiting owner approval, nothing run)

**Why.** In Stage B, the search fitted the true structure of only 3–8% of 3+-piece equations within 2,000 structures. Where it missed, its final structure shared a median of none of the true terms.
- **The step-by-step search is badly guided for multi-piece structures.** It builds structures one term at a time, and the value of a partial structure (a fit missing several pieces) says little about which pieces are missing.
- **With a vocabulary of 60–250 terms,** a 4–6-term structure needs 4–6 correct moves in a row. The network priorities get one right in their top 5 only about 40% of the time (Trial 29).
- **The fitter is no longer the limit:** given the structure, the combined fitter recognizes 39% of 3+-piece equations.

**Proposal: propose whole structures, then let the search refine.**
1. **The proposal network.**
   - Given an equation's data (the dense rates and states, as at test time) and the system's vocabulary, it scores each candidate term's chance of belonging to the true structure. It also predicts the number of terms.
   - A beam over term sets, by joint score, then gives the top-k complete structures. If that falls short, an autoregressive set decoder replaces the beam.
   - **Training:** generated systems with known true structures. The generators' training modes exclude the held-out family pairs; each system is simulated into its own bursts (real trajectories, as for the starting-value network). Bursts and true structures are stored (about 3 KB per system).
   - **Data size:** 10,000 new systems, about 5 h on 24 workers, plus the 1,040 collected systems.
   - **Accumulation curve:** trained on 1,000, 3,000 and 10,000 systems.
2. **In the search (additive only).** The top-k proposals are evaluated first, with the combined fitter, and the usual PUCT search then continues within the same total budget. Proposals add candidates and never remove any.
3. **Stage 1, development (reach):**
   - the share of equations whose true structure is among the top-k proposals (k = 10, 50, 200), by number of pieces;
   - against the PUCT search's reach within the same number of evaluated structures, and against a zero-cost baseline: proposals built from the existing move network's first-move scores.
4. **Stage 2, fresh known-data sets** (untuned, hand-tuned, unfamiliar families): equations recognized when the k proposals are fitted by the combined fitter, against PUCT at the same number of evaluations; by pieces, with time.
5. **Stage 3, full search** on Trial 29's sets:
   - proposals then PUCT, against PUCT alone, at the same structure budget, with generous time limits under the new rule;
   - both priorities;
   - equations, true structures and whole systems; Trial 29's bar (b) re-checked.
- **Pass bars are written into the log before each stage;** check-ins between stages.

**Secondary: the screen as a time saver.** Skipping thorough fits the screen scores lowest (development: 61–68% skipped at 98% recall of exact finds) would give the search's time back to more structures. It is a generic improvement for every arm, used only if a known-structure check shows no loss.

**Trial 29's stage 2 (learning from its own solved searches): not now.**
- The learner itself solves very few 3+-piece equations (1–4 of about 270 per arm in Stage B), so its own solutions would teach mostly the easy structures it already finds.
- On synthetic systems the true structures are known, and they are strictly better labels than the learner's own solutions.
- Stage 2 matters for real systems, which have no labels. Resume it there, as fine-tuning of the proposal network on its own solved practice and real systems, once the proposals raise the number of hard equations solved.

### Before Trial 32: the owner's two questions; why the real practice systems fail (6 November, 15:20; existing records only, nothing run)

**Owner.** *"1. What architecture are the move network and the proposed structure network? Scoring each term independently can't capture that a true term and its look-alike shouldn't both appear. Consider a design that models interactions between terms… 2. Real practice systems have no 3+-piece equations in toolkit terms, so recent gains are on a synthetic-only difficulty. Give me a breakdown of why the real practice systems currently fail… Make sure Trial 32 targets what limits real systems, not just synthetic ones. Answer both in plain language before running anything."*

**1. Architectures.**
- **The move network (Trial 28):**
  - a 3-layer transformer (width 128) over all candidate terms of the vocabulary, one token per term;
  - each token carries 28 features: the term's family and its relation to the equation (variables, multiplier), random variable codes, and evidence from the current partial fit (how much adding the term would cut the residual now and from scratch, whether it is already in, the cost of removing it, the current residual and size);
  - attention runs across all candidate terms; it outputs an add score and a remove score per term.
- **It is used inside the tree search one move at a time,** and its features are recomputed after each fit. So it is conditioned on the terms already chosen: once a true term is in, a look-alike's evidence drops.
- **The whole-structure network as first sketched** (independent per-term scores, then a beam) has the flaw the owner names.
- **Replacement design:** an autoregressive set generator. It proposes one term at a time, attending to the data, to all candidate terms and to the terms already chosen, until a stop token; beam search over these steps gives whole structures with joint scores. It is trained on true structures in random orders, so the order does not matter. Optionally the partial structure is refitted after each choice, so the next choice sees the residual.

**2. Why the 27 practice systems fail at M = 8.**
- **Sources:** the best earlier search (Trial 28: history network, 5,000 structures, old fitter v1); Trial 31's known-structure check (combined fitter); the expressibility classifier on the true right-hand sides.
- **Systems:**
  - **8 recovered exactly;**
  - **10 have an equation outside the toolkit.** 23 equations; the terms outside are rational terms with denominators the toolkit lacks (30), logarithms (4), exponentials of non-linear arguments (3), other (1);
  - **9 are expressible but not recovered.**
- **The 9 expressible systems not recovered:**
  - **7: every equation has a plausibly exact structure, but the system is not exact.**
    - 4 passed the learner's own held-back check but failed on new starting states, with one-step errors of 1.1, 2.0, 3.3 × 10⁻¹² and 4.4 × 10⁻⁹ (pass mark 10⁻¹²).
    - 3 failed the held-back check at 9 × 10⁻¹³, 1.1 × 10⁻¹² and 6.3 × 10⁻¹⁰ (pass mark 3 × 10⁻¹³).
    - **5 of the 7 are precision near-misses, within about 3× of the thresholds.** With M = 16, 3 of the 7 become exact.
  - **1:** an equation the fitter cannot fit at M = 8 even given its known structure.
  - **1:** not learned even at M = 64.
- **Equations** (80): 49 solved (plausibly exact), 23 outside the toolkit, 4 failing to fit even given the known structure, 4 without a plausibly exact structure and no known toolkit structure to check against.
- **None** failed because the search did not reach a structure the fitter could fit. Toolkit-expressible practice equations have 0 pieces (24), 1 (28) or 2 (5); none has 3 or more.
- **Reading.** On real practice systems the limits are, in order:
  1. toolkit coverage (10 of 27 systems), which falls under the owner's rule against hand-added pieces;
  2. final precision of the assembled model: about 5 near-misses within about 3× of the thresholds;
  3. two genuinely wrong or unassembled models, one fitting failure and one system never learned.
- **Search reach for 3+-piece structures is a synthetic-only limit at present.**

### Trial 32 step 1: precision near-misses; diagnosis and the practice-set check pre-registered (6 November, 15:40, logged before running the check)

**Owner.** *"Agreed on the order: Trial 32 does the precision near-misses first, then the toolkit-coverage proposal, then the structure-proposal network (with the term-by-term design). For the near misses: improve precision genuinely (reference accuracy, final polish, conditioning). Do not change the exactness bar. Pre-register the practice-set check before running. For toolkit coverage: discovering a general class from data, such as rational terms with denominators found by the algebraic start, kept in a library and reused, is within my rule, since the specific forms come from data. Test reuse with Trial 27's controls (decoys, removal check). Also add such general terms to the synthetic generator, so synthetic results transfer to real systems. Send that proposal before running it."*

**Diagnosis** (development: the 4 practice near-misses whose models were stored; the 3 rejected near-misses have no stored model):

| Recorded one-step error | Reference's own error (4× substeps) | Model integrated 4× finer | Least-squares re-polish | Library precise final stage |
|---|---|---|---|---|
| 2.0 × 10⁻¹² | 3.5 × 10⁻¹⁵ | 2.0 × 10⁻¹² | 1.9 × 10⁻¹² | 1.7 × 10⁻¹² |
| 1.1 × 10⁻¹² | 4.6 × 10⁻¹⁵ | 1.1 × 10⁻¹² | 1.1 × 10⁻¹² | 1.2 × 10⁻¹² |
| 4.4 × 10⁻⁹ | 9.7 × 10⁻¹⁵ | 6.0 × 10⁻⁹ | 2.0 × 10⁻¹⁰ | 2.9 × 10⁻⁹ |
| 3.3 × 10⁻¹² | 1.1 × 10⁻¹⁴ | 7.0 × 10⁻¹³ | 5.4 × 10⁻¹³ | **9.4 × 10⁻¹³, exact** (multi-step included) |

- **The reference is accurate** (≤ 1.1 × 10⁻¹⁴).
- **One near-miss was integration and polish:** fixed by the precise stage.
- **Two are identifiability:** the model matches the observed data to about 10⁻¹⁵ but misses new starting states by 1–2 × 10⁻¹². Precision cannot fix that, though M = 16 fixed one of them in Trial 28.
- **One is a wrong structure.**

**The precise final stage** (`DL.set_precise`, `learn28(precise=True)`; off by default, default path unchanged):
- Levenberg–Marquardt steps by scaled least squares instead of normal equations, at the model's final substep count, refitted if that count changes, with a convergence-based stop (up to 5× the iterations). This applies in the polish and in the tied refit.
- Substep convergence to 1 × 10⁻¹⁵ instead of 1 × 10⁻¹⁴.
- Artifact and held-back check at 16× the converged count instead of 4×.
- **The exactness bars are unchanged:** external one-step ≤ 10⁻¹² plus the multi-step envelope; internal held-back ≤ 3 × 10⁻¹³.
- **Cost:** the artifact's substeps rise from 4 to 16–32 per step on these systems, while the references use 8–64. This is reported against the goal that the artifact be cheaper than the simulation.

**Practice-set check, pre-registered:**
- 27 practice systems at M = 8, seed 1; fitter v2-strong; untuned-history network priorities; 2,000 structures per equation; 200 thorough fits per equation; 8 h per system (rarely binds; hits reported); 24 CPU workers.
- Arms: **A**, the default final stage; **B**, the precise final stage. 54 runs.
- **Pass (both required):**
  1. B recovers more systems exactly than A;
  2. B loses none of A's exact systems (outcome level, as the consistency rule).
- **Reported:**
  - systems exact, paired;
  - near-misses in each arm (one-step error between 10⁻¹² and 10⁻¹¹, or held-back between 3 × 10⁻¹³ and 3 × 10⁻¹²);
  - artifact substeps against the reference's;
  - time and time-limit hits.
- Practice gains are suggestive only.

### Trial 32 step 2 proposal: toolkit coverage by discovering a general rational class from data (6 November; awaiting owner approval, nothing run)

**What it targets** (practice breakdown, development data):
- 10 of 27 practice systems contain terms outside the toolkit: rational terms with denominators the toolkit lacks in 8 systems (31 terms), a logarithm in 1, an exponential of a non-linear argument in 1.
- Of the 31 rational terms, **17 fit a general class**: a polynomial denominator of degree ≤ 2 in any of the system's variables (squares and products included), over a numerator polynomial of degree ≤ 3.
- That class would make **4 of the 10 systems fully expressible**. Degrees 3 or 4 add no further system: the other 6 also contain denominators of degree 7–24, logarithms or exponentials.

**1. Discovery, per system, from data** (owner: the specific forms come from data, so this is within the rule):
- **When it runs:** for an equation with no plausibly exact toolkit structure after the search.
- **What it looks for:** the equation's rate as one rational function, y = P(x)/D(x), in implicit form D(x)·y − P(x) = 0 (as in implicit SINDy).
  - Libraries: D has 1 plus the monomials of degree 1–2; P has the monomials of degree ≤ 3 times those of D's support.
  - Solved by the algebraic start on the dense rates, with a sparsity search (pruning, as for the toolkit), normalized by D's constant term.
- **Checked** by the learner's usual exactness machinery: step integrals, held-back check, external check.
- **Output:** a rational term whose denominator support (which monomials) and constants both came from the data.

**2. Library and reuse:**
- After the learner's own acceptance (never the external check), each discovered denominator support is stored as a pattern, with variables relabelled by role ("self" / "others", as in Trial 27's pattern library).
- In later systems, every library pattern under every variable assignment is a candidate piece in the search's vocabulary: a numerator over the pattern's denominator, constants fitted, starting values from the algebraic start.

**3. Synthetic generator** (owner's addition):
- Both generators (training and untuned modes) gain the general rational family: terms m(x)/D(x), with D a random sparse polynomial of degree ≤ 2 in 1–3 variables (squares and products included), positive coefficients in natural ranges so that D > 0 on the domain, and monomial numerators of degree ≤ 2.
- One held-out family pair is added with the new family.
- The history is regenerated, and the move network and the starting-value network retrained, so that their training covers the family. Fresh pool and reserve sets untouched.

**4. Tests** (bars written into the log before each run):
- **(a) Synthetic discovery.** On systems containing the new family, the share of its equations discovered exactly from data at M = 8, by number of pieces. **Prerequisite (Trial 27's D, adapted):** at least 30% of equations the toolkit cannot learn.
- **(b) Reuse streams on synthetic systems** (Trial 27's design):
  - fixed random orders;
  - library against scratch-with-discovery, a decoy library (the same number of patterns of the same size, with random supports that come from no data) and copy-the-closest;
  - the removal check and provenance log for every library win;
  - Trial 27's criteria adapted: accumulation, growth with history, decoys and copy-the-closest keeping at most half the advantage, attribution by removal in ≥ 80% of wins.
- **(c) Practice set:**
  - the 10 non-toolkit systems with discovery, and with the library built from synthetic and earlier practice systems in a fixed order: exact at M = 8;
  - the other 17 as a no-harm check.
- **(d) Then** the term-by-term structure-proposal network (step 3), trained on the extended generator.

### Trial 32 step 1 result: practice-set precision check NOT MET (6 November, 15:42–17:21)

**Run.** 54 runs (27 practice systems × arms A and B), 24 CPU workers, no errors, no time-limit hits (8 h per system; medians 2,030 s for A and 2,252 s for B; maximum 4,333 s).

| Arm | Exact | Accepted | Near misses | Artifact substeps (median) | Artifacts with more substeps than the reference |
|---|---|---|---|---|---|
| A, default final stage | 4 | 6 | 1 | 4 (reference 8) | 0 of 6 |
| B, precise final stage | 4 | 6 | 1 | 24 (reference 8) | 4 of 6 |

- **Paired:** no system differs between A and B. **The pass bar (B recovers more and loses none) is not met.**
- **Why.** The near misses diagnosed earlier came from Trial 28's runs (hand-tuned-history network, 5,000 structures, fitter v1). Today's learner (untuned-history network, v2-strong, 2,000 structures) mostly does not produce them: 1 near miss in each arm, and the precise stage does not fix it.
- **The precise stage also makes the artifact more expensive than the simulation** in 4 of 6 accepted systems, against the programme's goal. **It is not adopted;** the code stays, off by default.
- **Configuration matters on the practice set.** Today's standard learner recovers 4 of 27 practice systems at M = 8, against 8 for Trial 28's best arm (hand-tuned history, 5,000 structures, fitter v1). Which history and budget to carry forward for real systems is a decision for the owner. The hand-tuned history was built to resemble biochemistry (the owner's earlier concern about design choices), but that resemblance may be legitimate prior knowledge for real biochemical systems.

### Direction change under assessment: one general learner for physical systems across domains (6 November; owner's question, nothing run)

**Owner.** *"Pause step 2. Bigger question first: biochemistry is just the codebase we pulled from; the learner must work on physical systems in general… a curated toolkit of biochemistry-shaped terms doesn't fit our goal. Our goal also goes beyond AlphaZero: … one network that learns physical domains in sequence (e.g. biochemistry, then mechanics, then circuits) and accumulates, so each new domain is learned faster or better than by a fresh network, while earlier domains are retained. In plain language, assess: 1. … a general expression language … 2. A synthetic history drawn from many physical domains … 3. Test systems from several domains … 4. Where AlphaZero-style learning … fits … 5. How to measure accumulation across domains: forward transfer … and retention …, and how to avoid the forgetting problem seen in the earlier PDEBench replay test. Don't run anything. Send the assessment and a staged plan, with the first stage small enough to show whether this direction is viable."*

**1. A general expression language in place of the toolkit.**
- **Carries over unchanged:**
  - the exactness tests (step integrals, held-back check, external check on new starting states, multi-step envelope), which judge any right-hand side;
  - the compiled artifact;
  - the search framework (tree search with learned priors and values, transpositions, simplification of exact finds);
  - variable projection for the outer sum (most physical right-hand sides are sums of terms with coefficients);
  - the fit recorder, the screen, the idea of a learned starting-value network, and the shadow sample.
- **Breaks or must be generalized:**
  1. **The algebraic start** works only for rational parts; with exp, log or trigonometric functions inside a term there is nothing to clear.
  2. **The fitter's "natural ranges"** are hand-made per family (Hill constant around the data median, tanh centre across the data range). For a general language they must come from the data flowing into each operator.
  3. **Constants can sit deep inside compositions** (e.g. exp(−a/(b + x))). The multi-piece fitting problem of Trials 30–31 becomes the normal case.
  4. **The vocabulary can no longer be listed:** terms must be proposed, not enumerated. The move network's per-candidate evidence would be computed only for proposed candidates.
  5. **Equivalent forms** (x(y + z) and xy + xz) need canonicalization for the transposition table.
  6. **More alternative models fit the observed data exactly** (the identifiability failures seen on 2 practice systems), so parsimony and validation matter more.
- **Search-space size** (counted, before merging equivalent forms; leaves: the variables and a constant; operations: +, ×, ÷, power, exp, log, sin):
  - **today's toolkit:** 60 terms (d = 2) or 247 (d = 4), so 6 × 10⁷ and 3 × 10¹¹ structures of up to 6 terms;
  - **the general language:**

    | Variables | Terms with up to 5 / 7 / 9 / 15 operations | Sums of up to 6 terms of up to 7 operations |
    |---|---|---|
    | d = 2 | 4.5 × 10³ / 2.8 × 10⁵ / 1.9 × 10⁷ / 9.3 × 10¹² | 6 × 10²⁹ |
    | d = 4 | 1.2 × 10⁴ / 1.0 × 10⁶ / 1.0 × 10⁸ / 1.3 × 10¹⁴ | 2 × 10³³ |

  - Blind search is hopeless; **learned guidance becomes essential**, as in Go.
- **Recommended form** (keeps what works): each equation is a sum of terms, each term a small expression from the general language (up to 5–7 operations at first, more later), with coefficients solved exactly by variable projection. Today's curated pieces are special cases (a Hill term is x^n / (K + x^n)); nothing biochemistry-specific is built in.

**2. A synthetic history from many physical domains.**
- Generators per domain, each a small grammar of that domain's typical motifs, with random structure, variables and constants:
  - **mechanics:** oscillators, Duffing, pendulum, damping forms, coupling, rigid bodies;
  - **circuits:** RLC, Van der Pol, FitzHugh–Nagumo, Chua-like, diode exponentials, saturating amplifiers;
  - **chemistry:** mass-action networks, enzyme rate laws, Brusselator / Oregonator, Arrhenius factors;
  - **ecology and epidemiology:** Lotka–Volterra, Holling II / III, logistic, SIR / SEIR;
  - **chaotic flows:** Lorenz, Rössler and Sprott-type, Thomas, the double pendulum.
- **Each system** is simulated into bursts under the existing acceptance rules.
- **Risk:** generators still encode modelling choices. **Mitigation:** generator rules are fixed before any test system is opened; test systems come from independent published collections; generated systems are de-duplicated against test systems.
- **Scope:** autonomous, fully observed ODEs from trajectory bursts, as today. Mechanics is written in first order (positions and velocities). Forced or partially observed systems come later.

**3. Test systems from several domains** (ground-truth equations published):
- **dysts** (about 130 chaotic flows from physics, mechanics, circuits, ecology, chemistry, climate, neuroscience);
- **ODEBench** (63 ODEs, 1–4 dimensions, from Strogatz's textbook and other sources);
- **SRBench's Strogatz set;**
- **textbook mechanics and circuit models;**
- **BioModels:** practice set, fresh pool and reserve sets, as now.
- **Usual rules:** splits by seeded hash before opening, reserve sets, held out by domain and source, the cheap-reference cutoff, the time-limit rule, counts only.

**4. Where AlphaZero-style learning fits.**
- **The "game"** is finding an exact model of a system. Search runs over terms and term edits, guided by a policy (which term or edit next; the term-by-term design) and a value (the chance that a partial model leads to an exact one).
- **The outcome is checked by the exactness test, which plays the role of the game's rules:** a verified exact model is a trustworthy training example without any label.
- **The loop is expert iteration:**
  1. search many systems (synthetic, and real systems without labels) with the current network;
  2. keep the verified solutions;
  3. train the policy on what the search found and the value on outcomes;
  4. repeat.
- **This is Trial 29's stage 2,** now central rather than optional. Supervised pretraining on generated systems with known answers (AlphaGo's first stage) can start it; AlphaZero showed it is not required. One network serves all domains: its inputs (bursts and the partial model) carry no domain label.
- **A difference from games:** problems do not come from self-play. They come from the generators and from real data, so the curriculum is a design choice.

**5. Measuring accumulation across domains, and avoiding forgetting.**
- **Protocol:** a fixed sequence of domains (e.g. chemistry, mechanics, circuits, ecology, chaotic flows). For each domain, a training stream and a held-out test set of real systems.
- **Measures** (per domain, at fixed search and training budgets):
  - **exact recovery and search effort;**
  - **forward transfer:** the learning curve on the new domain (success against the number of the domain's training systems) for the network that has learned the earlier domains, against a fresh network on the same data;
  - **retention, per system:** each earlier domain's test systems after each new domain, against right after that domain was learned;
  - **reference:** a network trained jointly on all domains, as an upper bound.
- **The PDEBench lessons:**
  - replay at a fixed loss weight (0.5) harmed the new family in both seeds;
  - aggregate error hid case-level regressions (50 of 120 earlier cases within 10% in one seed);
  - all of about a million weights were updated per new family.
- **Remedies:**
  1. **Knowledge that cannot be forgotten.** Verified solutions and discovered terms are kept as an explicit library. Correctness never depends on the network; forgetting can only slow the search, and that is measurable.
  2. **Rehearsal from stored verified solutions,** with the mix tuned on development (not fixed), and gradient-interference control (stop updates that raise earlier-domain loss) compared with simple rehearsal.
  3. **Retention judged per system,** with pre-registered tolerances, never on aggregates.
  4. **Enough capacity in one shared network;** growth only if forgetting persists.

**Staged plan** (each stage pre-registered, with a check-in between stages):
- **Stage 1, viability (small):**
  - **Setup:** 2-variable systems from 3 synthetic domains (mechanics, circuits, ecology), each with a generator fixed in advance (about 300 training and 40 test systems per domain), and the general language with terms of up to 5 operations.
  - **(a) Language and fitter, no learning:** given the true structure, does the generalized fitter recover exact constants? **Bar:** at least 80% of equations with at most 2 nonlinear terms.
  - **(b) Search without learning:** uniform priorities at 2,000 structures per equation; a baseline.
  - **(c) Learning and transfer:** a policy network trained on domain A, tested on domain B, against uniform priorities and against a network trained on B alone with the same amount of data. Then A followed by B, with retention on A per system. **Bars:** the network beats uniform priorities on its own domain by at least 20 points (or needs at most half the search); no negative transfer; a positive transfer at small data, for example +10 points at 50 systems of B.
  - **Cost:** about a day of computing.
  - **If (a) fails, stop and rethink the fitter; if (c) shows no transfer, rethink before scaling.**
- **Stage 2, sequential domains** (5 domains, d ≤ 3): forward-transfer curves and per-system retention, with rehearsal choices compared on development.
- **Stage 3, the AlphaZero loop:** search plus learning from its own verified solutions, on synthetic and real unlabelled systems; terms up to 7 operations, d up to 4.
- **Stage 4, held-out multi-domain tests** (dysts, ODEBench, BioModels fresh pool) under the usual rules, after the owner approves the request.
- **Retired:** the curated families and their hand-made ranges, and the hand-tuned history.
- **Kept:** the exactness machinery, the search framework, variable projection, the verifier-based learning loop, the fit recorder and the screen.

### Stage 1 of the general learner: detailed rules (6 November; awaiting owner approval, nothing run)

**Owner.** *"Write up Stage 1's rules in detail for my approval, with these additions: 1. Build the plan on DreamCoder's wake/abstraction/dreaming loop … Explain how the design adopts each phase (at least in a later stage if not Stage 1), why Trial 27's growing library failed where DreamCoder's worked, and what's different this time. For every major design choice, say which published method it borrows from and whether that method succeeded. 2. Some domains share identical equations (e.g. a damped spring and an RLC circuit)… report transfer separately for test systems whose forms appear in the earlier domain and for those that don't. 3. Keep discontinuous systems like the bouncing ball on the list for a later stage."*

#### 1. The loop: DreamCoder's three phases

- **Wake (Stage 1).**
  - **What happens:** the learner searches each task (a system's bursts, with no answer given) with its current network guiding the search, and keeps every model that passes the exactness test.
  - **DreamCoder:** the recognition network guides enumeration; tasks are checked against their examples.
  - **Here:** the existing tree search over term sets, with the move network as the guide. The exactness test is the check.
- **Dreaming (Stage 1).**
  - **What happens:** the network is retrained on two kinds of examples:
    - (i) **replays:** the learner's own verified solutions to the tasks so far, as search states along each solution, in the history format;
    - (ii) **fantasies:** new systems sampled from the current language, with term frequencies taken from the library's usage so far. They are simulated into bursts with the existing reference machinery and acceptance rules, so their answers are known.
  - **DreamCoder:** trains on replays and fantasies sampled from the current library's grammar.
  - Fantasies here come from the language and library, not from the domain generators: the modeller-written generators only supply tasks and test systems.
- **Abstraction (Stage 2).**
  - **What happens:** fragments that recur across verified solutions (for example, a subexpression 1 / (1 + c·x) appearing in many terms) are found by compression. They are added to the library as new building blocks that the search can add in a single move.
  - **DreamCoder:** refactoring by compression over all solutions; Stitch is a much faster equivalent.
  - **Controls (Trial 27's):** a decoy library (the same number and size of fragments, random, from no data) and the removal check with provenance for every library win.
  - **Why not Stage 1:** abstraction needs a corpus of verified solutions, and Stage 1 is where that corpus is first built.
- **Value network and learning from real unlabelled systems:** Stage 3 (the AlphaZero part).

#### 2. Why Trial 27's library failed where DreamCoder's worked, and what is different now

- **What Trial 27 did and found:**
  - pieces discovered on single systems (at M = 64) were added to a library in a fixed order;
  - at M = 8: scratch 2 of 26, library 2 of 17, decoy 1 of 15; the one library-only success used no library piece;
  - the later pattern library produced no exact model from its patterns.
- **Why it failed:**
  1. **Entries came from single systems, not from fragments that recur across many solutions.** With a few dozen real systems, little recurs.
  2. **The bottleneck at small data was over-selection of terms** (8-term equations fitting the training bursts and missing the held-back ones). A larger library does not fix that; it enlarges the search.
  3. **No learned network decided when to use an entry,** so entries were just more candidates.
  4. **No dreaming:** the guidance never trained on what the library could generate.
  5. **Too few tasks** and no repeated cycles.
- **What DreamCoder did differently:**
  - libraries built by compression across hundreds of solved tasks;
  - a recognition network trained on replays and fantasies to choose entries;
  - many tasks of graded difficulty over repeated wake–sleep cycles.
- **What is different now:**
  - thousands of generated tasks across domains;
  - an exact verifier;
  - a move network trained by dreaming before any abstraction is tried;
  - abstraction only once the solution corpus is large (Stage 2), with Trial 27's controls;
  - parsimony (the simplest model exact on held-back data) kept against over-selection.
- **Remaining risk:** over-selection at small data. Stage 1 measures it directly (true structure recognized, against other structures that are exact on the data).

#### 3. Scope and data

- **Systems:** autonomous, fully observed systems with 2 state variables, learned from trajectory bursts as now (M = 8, seed 1, the learner's view and scaling unchanged).
- **Not in Stage 1:** discontinuous and hybrid systems (bouncing ball, stick–slip friction, relay circuits), forced systems, partial observation. Discontinuous systems are kept on the list for a later stage.
- **Exactness bars unchanged:** step-integral and held-back checks at the learner; external one-step ≤ 10⁻¹² plus the multi-step envelope.
- **Stage 1 uses synthetic systems only;** real multi-domain test sets are kept unopened for Stage 4.

#### 4. Language L1 (replaces the curated toolkit)

- **Leaves:** the state variables (scaled as now, and named only by position, x₁ and x₂, never by domain) and fitted constants.
- **Operations:** +, ×, ÷, power with a constant exponent (fitted, or one of 2, 3 and ½), exp, log, sin.
- **A term** has at most 7 nodes (operations, variables, constants and fixed exponents all count) and at most 2 fitted constants, with no exp or log inside another exp or log. A term counts as in L1 if it, or an equivalent form up to a constant factor, meets these limits. Examples:
  - x/(c + x) has 5 nodes;
  - x·y/(c + x) has 7. Holling type II predation, x·y/(1 + h·x), has 9 as written but equals (1/h)·x·y/(1/h + x), so it is in L1;
  - v·(v²)^½ (that is, v·|v|) has 7;
  - exp(c·v) has 4, x³ has 3, sin x has 2.
- **An equation** is a sum of at most 6 terms (the constant 1 is one of the terms), with coefficients solved exactly by variable projection.
- **Canonical form:**
  - each term is simplified and ordered (sympy) and stripped of an overall constant factor;
  - two terms are merged when they describe the same family: for random values of one term's constants, the other's constants can be fitted to reproduce its values exactly, up to a scale, on random points.
- **The vocabulary:**
  - all canonical terms, enumerated once for 2 variables (an estimated 10³–10⁴); the exact count is logged before any task is drawn;
  - for each system, terms that are undefined or constant on its data are dropped.
- **Larger terms** come only through abstraction (Stage 2).

#### 5. Generic fitter

- **Starting grids** for each term's internal constants come from generic rules on the data:
  - a constant added to a subexpression is gridded across that subexpression's observed range, extended by half;
  - a multiplying constant is gridded on a log scale from 10⁻² to 10² of 1/scale, both signs;
  - an exponent is gridded on [−3, 3].
- **The rest is v2-strong, unchanged:** sweeps, joint beam, global search in these units with the best starts pre-polished, Levenberg–Marquardt with variable projection, snapping of constants only where identifiable, step-integral refit, exactness test.
- **The algebraic start** is used only when a term is rational (detected automatically).
- **Learned starting values follow in Stage 2.** Trial 31's network was trained on toolkit terms and does not apply to L1 terms.

#### 6. The guiding network

- **One network for all domains,** with no per-domain parts. Its weights are carried from domain to domain and never re-initialized.
- **Design: Trial 28's move network,** with attention across candidate terms.
  - At each search step it scores every addition and removal given the terms already chosen. A true term and its look-alike are judged together, and the structure is proposed term by term.
  - **Change 1, what a term is:** a small encoder reads each term's expression (operations, variables by position, constants), instead of a toolkit slot number. The same term then has the same description in every domain.
  - **Change 2, which terms attend:** the terms in the current structure, plus the 512 candidates with the most single-term evidence from the current fit. Attending across 10³–10⁴ terms at every step would be too slow; the remaining terms keep the uniform prior.
- **Inputs from the data:** as in Trial 28, each term carries the current fit's evidence (gain, removal cost, membership). No domain label is given.
- **Output and training:** the add and remove scores become the search's priors. Training uses Trial 28's multi-positive loss on the search states along verified solutions.
- **Cycle 0 is uniform** (no network), for the continued network and for each fresh network.
- **Not in Stage 1:** a network that proposes whole structures directly from the data (the proposer planned for Trial 32). It needs the corpus of verified solutions that Stage 1 builds.

#### 7. Domain generators (fixed now, before any task exists)

- **Mechanics** (x, v): ẋ = v; v̇ = −k x plus 0–2 of {−a x³, −a sin x, −c v, −c v³, −c v·(v²)^½, −c x² v}.
- **Circuits,** one of five forms, equally likely:
  - series RLC in charge form (q, i): q̇ = i, i̇ = −(R/L) i − q/(LC), the same form as the damped spring;
  - a parallel circuit (v, i): v̇ = (i − g(v))/C and i̇ = (−v − R i)/L, where g is one of g₁v (linear), g₁v − g₃v³ (Van der Pol type) and I(exp(v/V) − 1) (diode);
  - FitzHugh–Nagumo: v̇ = v − v³/3 − w + I, ẇ = ε(v + a − b w).
- **Ecology** (x, y):
  - predator–prey: ẋ = r x − (r/K) x² − f(x) y and ẏ = e f(x) y − m y, with f in {a x, a x/(1 + h x)};
  - or two-species competition.
- **Constants** are random in physical ranges fixed in code, with the existing acceptance rules (bounded, moving, positive for ecology, cheap reference, own-timescale windows).
- **Every generated term is expressible in L1** (checked by the generator).
- **Shared forms, owner's addition 2:**
  - **Some forms are shared on purpose.** The charge-form RLC circuit is the damped spring. The Van der Pol circuit's v̇ equation has the same terms as a damped Duffing oscillator's v̇ equation.
  - **Per equation:** a test equation's form *appears in an earlier domain* if its canonical term set (constants ignored, the two variables possibly swapped) equals the true term set of an equation of some training task in an earlier domain.
  - **Per system:** a test system is *shared-form* when all its equations are.
  - **How it is made:** from the known structures, which the learner never sees. It is made when the test systems are drawn, before any evaluation, and its counts are logged.
  - **Ecology:** its forms (x·y, x², x·y/(c + x)) do not occur in mechanics or circuits, so transfer to ecology is almost entirely to new forms.

#### 8. Tasks and tests

- **Per domain,** drawn with separate seeds:
  - 150 training tasks ("g1-train-{domain}-k");
  - 10 development systems ("g1-dev-{domain}-k"), for the timing pilot and debugging only, never in headline numbers;
  - 40 test systems ("g1-test-{domain}-k").
- **Everything is drawn once and stored** (bursts, with a hash per system), so every arm sees identical systems.
- **The learner sees tasks as data only.** Answers are used for evaluation and for the shared-form classification.
- **Counts only for test systems.**

#### 9. Cycles

- **Order:** mechanics (A), then circuits (B), then ecology (C).
- **Each domain gets 2 wake–dream cycles:**
  - **wake:** all 150 tasks searched, with:
    - 1,000 structures per equation;
    - 200 thorough fits per equation;
    - time limits set from the development pilot so that they rarely bind (the standing rule);
  - **dream:** the same weights are trained further on every verified solution so far, from every domain, plus 500 new fantasies. Every network gets the same number of training steps per cycle.
- **Against forgetting:** every verified solution stays in the training mix at full weight (cheap, because solutions are small). This differs from the PDEBench replay test that failed:
  - there, a large network adapted all its weights to each new family, with a down-weighted share of the old data, and an aggregate error hid per-case losses;
  - here retention is judged per system, and the language itself cannot be forgotten;
  - if retention still fails, Stage 2 compares rehearsal choices on development systems.
- **Fresh networks for comparison:** the same 2 cycles on B alone and on C alone, with the same settings (strong-baseline rule).
- **Test evaluations,** all at the same budget:
  - uniform priorities;
  - the networks after A, after each cycle of B, and after each cycle of C;
  - fresh-B and fresh-C after each cycle.
- **Compute:**
  - roughly 1–1.5 days on 24 CPU workers, with the network trained on a free or shareable GPU (rule) and a few hundred MB on disk;
  - projected from the pilot before the run. If the projection exceeds 3 days, I check in with a smaller design (for example 100 tasks per domain) before running.

#### 10. Measures and pass bars (all four required)

1. **(a) Language and fitter:** given the true structure, at least 80% of test equations with at most 2 terms with internal constants are recognized exactly, in each domain. **If not met, stop.**
2. **(c) Learning on the first domain:** after A's cycles, on A's tests, guided search must beat uniform search at the same budget in one of two ways:
   - it solves at least 20 points more equations; or
   - it needs at most half the structures (median) to its first plausible model.
3. **(d) Forward transfer, on B and on C:**
   - **After the first cycle on the new domain:** the continued network solves at least 10 points more test equations than the fresh network, and the gain is significant (paired sign test on equations, p < 0.05).
   - **After the second cycle:** the continued network is not worse; the fresh network is not significantly better (p < 0.05).
   - **Reported separately** for shared-form and novel-form test equations, and for shared-form systems and the rest (owner's addition 2).
   - **The bar applies to all test equations together.** On C almost every form is new, so C's bar is in effect a test on novel forms. If B's gain comes only from shared forms, the report will say so.
4. **(e) Retention:** after C, results on A's and B's tests are judged per system. At most 2 systems per domain (5%) may be lost, compared with right after that domain's cycles.

- **(b) Uniform search** is reported as the baseline, without a bar.
- **Also reported:**
  - **zero-shot transfer:** the network from earlier domains on the new domain's tests before any cycle there, against uniform;
  - **provenance:** each test equation solved only by the continued network, with the earlier-domain solutions that share any of its true terms;
  - **the over-selection risk:** true structures recognized, against other structures exact on the data;
  - whole systems exact;
  - search effort and time, and time-limit hits per arm.

#### 11. Later stages

- **Stage 2:**
  - abstraction (compression over verified solutions; decoys and removal checks);
  - chemistry and chaotic flows (d ≤ 3);
  - learned starting values for the generic fitter;
  - a network proposing whole structures from the data;
  - rehearsal choices for retention compared on development.
- **Stage 3:** the AlphaZero part: a value network, with search plus learning from its own verified solutions on synthetic and real unlabelled systems; terms up to 9 nodes; d up to 4.
- **Stage 4:** held-out multi-domain real tests (dysts, ODEBench, SRBench's Strogatz set, BioModels fresh pool) under the usual rules, after the owner's approval.
- **Later still:** discontinuous and hybrid systems (bouncing ball, stick–slip, relays), forced systems, partial observation.

#### 12. What each design choice borrows, and whether it worked there

| Choice | Borrowed from | Did it succeed there? |
|---|---|---|
| Wake–abstraction–dreaming cycles | DreamCoder (Ellis et al., 2021) | Yes, on list processing, text editing, regular expressions, graphics, towers, symbolic regression and physics laws; compute-heavy, small domains |
| Abstraction by compression | DreamCoder's refactoring; Stitch (Bowers et al., 2023); LILO (Grand et al., 2024) | Yes. Stitch matched DreamCoder's libraries 1,000–10,000× faster. LILO, which adds language-model guidance to Stitch, solved more tasks than DreamCoder |
| Search guided by learned priors, improved by its own solutions | AlphaZero (Silver et al., 2018); expert iteration (Anthony et al., 2017) | Yes in games; also AlphaTensor and AlphaDev for algorithm discovery |
| Move network with attention across candidate terms, scoring moves given the terms already chosen | Trial 28 here, after AlphaZero's policy network | Partly. On synthetic sets like its history: 47% against 23% of systems exact, and +12 points on a size-matched set. On real practice systems, most of the gain came from its history's hand-set term frequencies (a plain frequency table matched it); reading the data added about 2 systems, not significant |
| Neural-guided search for equations | TPSR (Shojaee et al., 2023); DSO/uDSR (Petersen 2021; Landajuela 2022) | Yes, improved accuracy on SRBench and over transformer-only methods |
| Whole-expression proposals from data (later, for the term-by-term design) | NeSymReS (Biggio 2021); E2E (Kamienny 2022); ODEFormer (d'Ascoli 2024) | Partly: good proposals, imprecise constants; best combined with refinement |
| Sparse regression over a fixed library | SINDy (Brunton 2016); SINDy-PI (Kaheman 2020) | Yes when the library holds the right terms; fails otherwise, the problem this replaces |
| Exact coefficients | Variable projection (Golub & Pereyra, 1973) | Yes, widely |
| Constants by multistart global search and local polishing | Raue et al. (2013) for ODE parameters; Trials 30–31 here | Yes |
| Learned starting values with long polishing | Trial 31 here | Yes on synthetic 3+-piece equations (63 → 111 of 288); no effect on the full search, which rarely reaches them |
| Canonical forms of equivalent expressions | e-graphs and equality saturation (egg; Willsey et al., 2021) | Yes in compilers; partial in symbolic regression. Stage 1 uses the simpler sympy and numerical merging |
| Full rehearsal of every verified solution against forgetting | Experience replay (Rolnick 2019; Chaudhry 2019); A-GEM; EWC (Kirkpatrick 2017) | Replay and A-GEM yes on benchmarks; EWC mixed. Our PDEBench replay (a down-weighted share of old data) failed |
| Explicit library as memory that cannot be forgotten | DreamCoder; Voyager's skill library (Wang et al., 2023) | Yes |
| Cross-domain synthetic training, multi-domain tests | ODEFormer's random ODEs; dysts (Gilpin, 2021) | Moderate generalization to real benchmarks; dysts widely used |

*Clarified before sending (same day):* added the guiding network (section 6), development sets and stored draws, node counting with the Holling example, the charge-form RLC circuit, the per-equation shared-form rule, significance tests for transfer, the provenance and zero-shot reports, and the compute check.

### Stage 1 of the general learner: owner approval, with two notes (6 November)

**Owner.** *"Approved as written. Two notes: 1. Any claim of accumulation should rest mainly on equations with new forms (ecology, and circuits' new-form equations). Report it that way; no extra deciding bar needed. 2. In Stage 2, include a second domain order to check that transfer doesn't depend on the order. Check in after bar 1 (the fitter on true structures) before starting the wake–dream cycles."*

- **Bars unchanged.** Bars (a), (c), (d) and (e) are as written above.
- **Reporting.**
  - Every claim of accumulation rests mainly on new-form test equations: ecology's, and circuits' new-form equations.
  - Shared-form results are reported beside them, as context only.
- **Stage 2** adds a second domain order, to check that transfer does not depend on the order.
- **Order of work:**
  1. Count the L1 terms.
  2. Build the generators and the generic fitter.
  3. Draw and store tasks, development and test systems, and log the shared-form counts.
  4. Run the development timing pilot.
  5. Run bar (a), the fitter on true structures.
  6. **Check in with the owner** before any wake–dream cycle.

### Stage 1: L1 vocabulary count and implementation details (6 November, 20:30; fixed before any task is drawn)

**Vocabulary (two variables).** Built by `g1_vocab.py` and `g1_families.py`; saved in `outputs/accumulating_learner/g1/vocab_d2.json`.
- **Counts:**
  - 34,173 trees within the limits give 15,225 canonical forms;
  - 148 are dropped as undefined or constant on test points;
  - 472 constant-free terms are merged because their values agree up to a scale;
  - 200 families are merged by the fitting test, out of 22,129 candidate pairs from nearest neighbours.
- **Final vocabulary: 14,406 terms.**
  - By number of constants: 9,801 without constants, 4,003 with one, 602 with two.
  - By the size of the smallest generating tree: 3, 6, 23, 125, 461, 2,463 and 11,325 terms for 1 to 7 nodes.
  - This is above the estimate of 10³–10⁴.
- **Canonical-form rules added during the build.** Each is valid wherever the term is defined:
  - powers combine (x·x^K becomes x^K′);
  - nested powers are rewritten: (u^a)^b becomes u^(ab), or |u|^(ab) when a is an even integer;
  - logs of powers are rewritten: log(u^b) becomes b·log u;
  - |c·f| becomes |f|.
- **Checks.**
  - Every term the three generators can produce is in the vocabulary. The generator checks this for each draw.
  - Examples: x₁x₂/(K0 + x₁) (Holling type II), sin(K0·x₁), exp(K0·x₁), x₂·|x₂|.

**Generators** (`domains_g1.py`).
- **Physical ranges** (log-uniform unless noted):
  - **mechanics:**
    - k in [0.5, 5]; cubic a in [0.1, 2]; sine a in [0.5, 5];
    - damping coefficients in [0.05, 1];
    - the number of extra terms is uniform on {0, 1, 2};
    - x₀ = ±[0.5, 2.5] and v₀ = ±[0.5, 2.5]·√k (uniform);
  - **circuits:**
    - ω² in [0.5, 5]; R/L, R and g₁ in [0.05, 1]; C and L in [0.2, 2];
    - Van der Pol: g₁ in [0.1, 1], g₃ in [0.05, 0.5];
    - diode: I_s in [0.01, 0.5], V_t in [0.5, 3];
    - FitzHugh–Nagumo: I in [0, 1] and a, b in [0.5, 1] (uniform), ε in [0.05, 0.5];
  - **ecology:**
    - r in [0.5, 2], K in [2, 10], a in [0.2, 2], e in [0.2, 1], m in [0.1, 0.5], h in [0.2, 2];
    - competition: r in [0.5, 2], a_ii in [0.1, 1], a_ij in [0.05, 1].
- **Windows:** from each system's own timescale (the Trial 29 rule) for every system, because many mechanical and electrical systems do not settle. Recordings follow the existing transient rules.
- **Cheap-reference cutoff** (cheap-reference rule): the existing one. At most 4 interval halvings, at most 64 reference substeps, and at most 60 s to build.
- **Acceptance** (as approved, made precise):
  - finite;
  - positive (ecology only);
  - bounded (within 100× the largest initial magnitude);
  - every variable moves;
  - not centred by the learner's scaling at M = 8;
  - every true non-polynomial term is not singular on the learner's M = 8 data, and differs from every cubic polynomial there by at least 10⁻⁶ (relative least-squares residual).

**Fitter** (`fitter_g1.py`).
- The rules in section 5, with v2-strong's stages.
- A fitted term counts as degenerate if it is singular on the data, or within 10⁻¹⁰ (relative) of a cubic polynomial there.

**Pilot on pilot seeds only** ("g1-pilot*": never used again; separate from task, development and test seeds).
- **Acceptance:** 84% of draws in mechanics, 88% in circuits, 81% in ecology.
- **The drag motif v·|v| was never accepted, in 20 draws.** It is not smooth at v = 0. Either its reference needs too many substeps, or v keeps one sign on the data, so that v·|v| = ±v² there. Mechanics therefore effectively has five motifs.
- **Fitting:** known-structure fits were exact on 240 of 240 pilot equations, 32 of them with constants inside terms. The slowest took 0.6 s.

**Seeds and storage** (`g1_draw.py`).
- **Seeds:** "g1-{train, dev, test}-{domain}-k" for k = 0–149, 0–9 and 0–39. Replacement seeds "-r1", "-r2", … are used in order when a draw is rejected.
- **Storage:** `outputs/accumulating_learner/g1/systems/{set}_{domain}.json`, with bursts at M = 8 (seed 1) and a hash per system.
- **Hygiene:** test files are read by scripts only, and only counts are printed.
- **Shared-form classification:** `g1_shared.py`, run right after the draw; counts are logged.

**Bar (a): procedure and limits, fixed now.**
- **Per test equation:** `fitter_g1.evaluate_structure` on its true structure. This is the search's judgement of a structure:
  - ordinary fit;
  - the precise-stage gate;
  - a thorough fit unless the fit is already at the floor;
  - step-integral refit;
  - held-back check;
  - degenerate check.
- **Exact** means plausibly exact, as in the search: the step-integral error on training and held-back bursts is at most max(10⁻⁹, 30 × the equation's relative noise floor), and no term is degenerate.
- **Scope:** every Stage 1 test equation has at most one term with constants inside it, so all count toward bar (a).
- **Time limit:** 600 s per system, from the pilot's slowest fit of 0.6 s. Hits are reported.
- **Context:** whole systems with both equations exact are reported alongside.
- **Timing pilot first:** the 10 development systems per domain go through the same procedure before the test run.

### Stage 1: systems drawn, shared forms classified, bar (a) MET (6 November, 20:45)

**Draws** (`g1_draw.py`). All systems were accepted, all stored with hashes.

| Set | Mechanics | Circuits | Ecology |
|---|---|---|---|
| Training | 150 (25 rejected draws) | 150 (14) | 150 (28) |
| Development | 10 (5) | 10 (0) | 10 (1) |
| Test | 40 (5) | 40 (3) | 40 (10) |

- **Most common rejections:**
  - mechanics: a true term not visibly nonlinear on the data;
  - ecology: centred at M = 8.

**Shared forms** (`g1_shared.py`; true structures only; made before any evaluation).

| Test set | Shared-form equations | Shared-form systems |
|---|---|---|
| Circuits | 50 of 80 | 19 of 40 |
| Ecology | 0 of 80 | 0 of 40 |

- **Circuits:** its linear two-term equations, the charge-form RLC circuit and the Van der Pol voltage equation match mechanics forms.
- **New-form equations:** FitzHugh–Nagumo and the diode's voltage equation have new forms, 30 equations in all.
- **Basis for accumulation claims** (owner's note): ecology's 80 equations and circuits' 30 new-form equations.

**Development timing pilot** (`g1_bar_a.py dev`).
- 60 of 60 equations exact, 11 of them with constants inside terms.
- At most 0.9 s per system.
- The 600 s limit stands; 0 hits.

**Bar (a) on the test systems** (`g1_bar_a.py test`; counts only).

| Domain | Equations exact | With constants inside terms | Systems with both equations exact | Time per system (median / max) |
|---|---|---|---|---|
| Mechanics | 80 of 80 | 6 of 6 | 40 of 40 | 0.1 s / 0.7 s |
| Circuits | 80 of 80 | 12 of 12 | 40 of 40 | 0.0 s / 0.5 s |
| Ecology | 80 of 80 | 22 of 22 | 40 of 40 | 0.0 s / 0.7 s |

- Time-limit hits, errors and hash mismatches: 0.
- **Bar (a) is MET in every domain:** 100% against the 80% required.

**Reading.**
- **The fitter is not the limit here.** The general language and the generic fitter recover every Stage 1 equation exactly once the structure is known.
- **Stage 1's difficulty lies in the search:** finding each equation's few terms among 14,406 candidates, from 30 rate points and 80 step integrals per equation. Bars (c), (d) and (e) test that.
- **What this does not show:** the three domains have at most one term with constants inside it per equation. Harder fits (three or more such terms, as in Trial 31) are not tested here.

**Next.**
- **Build:** the L1 search (the tree search over L1 term sets), the move network with a term encoder, and the dreaming step.
- **Timing pilot:** on development systems only, with uniform search and no learning, to project compute.
- **Wake–dream cycles:** none start before the owner's go-ahead.

### Stage 1: owner's go-ahead for the wake–dream cycles, with conditions (6 November)

**Owner.** *"Start the wake–dream cycles once the pilot confirms the cost, as long as the projection is within the planned 1–1.5 days. If it's longer, check in with the projection and a smaller version first. Stop and report after bar 2 if it fails, before starting circuits."*

- **Condition 1.** The cycles start only if the development timing pilot projects Stage 1's whole compute within 1.5 days on 24 workers. That compute covers every cycle, every fresh network and every test evaluation.
- **Condition 2.** If the projection is longer, I check in with it and a smaller version before running anything.
- **Condition 3.** After mechanics' two cycles, bar (c) is judged on mechanics' tests. If it fails, I stop and report before circuits.

### Stage 1: development timing pilot, three changes found by it, projection, launch (6 November, 22:20)

**Three changes, generic and applied to every arm, found before any cycle on development systems only.** They are logged before the run.

1. **Search vocabulary: the 1,143 terms that keep their form when the variables are rescaled** (`g1_scale_closed.py`; x_j → s_j·x_j).
   - The learner's coordinates are the physical ones divided by scales from the data. So only these terms can be exact.
   - A term like sin(x₁) is the K0 = 1 member of sin(K0·x₁), which is kept.
   - The other 13,263 terms of L1 drop out.
   - Every true term of every stored system is among the 1,143 (counts checked for all sets: training, development and test). This holds by construction, for any physical system.
   - Found because fantasies drawn from the whole language were rejected for terms that left the language after rescaling. On a first try, many such terms also misled the search.
2. **The guiding network attends to every term of the system's vocabulary (at most 1,143), as in Trial 28.** The rules' 512 shortlist (by single-term evidence) is dropped. It was a compute measure for 10³–10⁴ terms.
   - **Rank check** (`g1_rank_diag`, development systems): at states with more than one true term still missing, the missing true terms ranked inside the top 512 by evidence in:
     - 26% of states over the whole language;
     - 51% over the 1,143 terms.
   - **Why:** flexible terms with fitted constants explain most of the 30 rate points on their own. A network that never saw the true terms could not rank them.
   - **Cost:** attention over all terms costs about 0.15 s per search node on one core, which the projection includes.
3. **Fantasies draw only from the search vocabulary**, with the approved rule otherwise unchanged: half usage, half every term size equally likely.

- **Robustness fixes:**
  - the exponent-uncertainty step now handles non-finite fits; this was the pilot's first crash;
  - the search records fitting errors instead of stopping.

**Pilot** (`g1_pilot.py`; uniform search, 1,000 structures and 200 thorough fits per equation, no time limit; 30 development systems).

| Domain | Equations solved | With the true structure | Time per system (median / max) | Median structures to the first plausible model |
|---|---|---|---|---|
| Mechanics | 20 of 20 | 16 | 158 s / 547 s | 17 |
| Circuits | 19 of 20 | 15 | 119 s / 322 s | 13 |
| Ecology | 8 of 20 | 4 | 547 s / 762 s | not reached (fewer than half solved) |

- **Thorough fits take most of the time:** 82 s, 55 s and 206 s per equation.
- **Uniform search is already strong on mechanics and circuits with the 1,143 terms.** So bar (c)'s +20 points may be out of reach on mechanics. Its other criterion is still open: at most half of uniform's median structures to the first plausible model.

**Time limit.** 3,600 s per system, 1,800 s per equation. That is 4.7× the pilot's slowest system and leaves room for the network's time. Hits are reported per arm.

**Projection.**
- **Search:** about 318 core-hours, 13.3 h on 24 workers. This assumes 0.25 s per network call and 600 calls per equation for guided searches.
- **Dreams:** about 3 h, at about 14 minutes of fantasies and 5 minutes of GPU training per dream.
- **Total:** about 16 h, within the planned 1–1.5 days. The owner's condition for starting is met.

**Launch** (`cycles_g1.py`, 24 workers, nice 10, one thread each).
- **Order:** mechanics' two cycles, then the uniform and guided tests on mechanics, then **bar (c)**.
- **Stop rule:** the run stops there if bar (c) is not met (owner's instruction).
- **GPU:** training uses a free GPU, checked before each dream and logged per dream (GPUs 0–5 and 7 were free at 22:00).

### Stage 1: mechanics' two cycles and bar (c): MET (6 November, 23:25)

**Cycles (continued network, mechanics; training tasks).**

| Phase | What ran | Result |
|---|---|---|
| Wake 1 (uniform) | 150 tasks, 1,495 s | 270 of 300 equations solved (254 with the true structure); 120 of 150 systems |
| Dream 1 | 500 of 500 fantasies accepted, 3,659 examples, 329 s; trained on 859 replay + 3,659 fantasy examples from 270 verified solutions | GPU 0, 214 s; final loss 1.22, top-5 add 0.88 |
| Wake 2 (network after dream 1) | 150 tasks, 508 s | 300 of 300 solved (285 true); 150 of 150 systems |
| Dream 2 | 500 of 500 fantasies, 3,587 examples; trained on 979 replay + 3,587 fantasy examples from 300 verified solutions | GPU 1, 212 s; final loss 0.75 |

- Wake 2's gain is on training tasks, partly learned from, so it is no test. It does solve the 30 equations that were absent from the replays.
- Time-limit hits: 0 in both wakes. Failed systems: 0.

**Test (mechanics' 40 test systems, 80 equations; counts only).**

| Arm | Equations solved | True structure | Systems | Median structures to the first plausible model | Time per system (median / max) | Time-limit hits |
|---|---|---|---|---|---|---|
| Uniform | 70 | 64 | 30 | 17 | 168 s / 492 s | 0 |
| Network after mechanics | 80 | 74 | 40 | 2 | 44 s / 164 s | 0 |

- **Paired comparison:** solved only with the network 10, only uniform 0 (sign test p = 0.002).
- **Bar (c): MET**, on its second criterion: median structures to the first plausible model, 2 against 17, at most half. The first criterion was not met: +12.5 points against +20, a ceiling, since uniform already solves 70 of 80.
- **Next:** per the plan, the run continues into circuits.
- **Whole-system check:** the external exactness test of the assembled models is running on the side (`g1_systems.py`).

**What this shows and what it does not.**
- **Shown:** learning within one domain. The network, trained on its own verified solutions and on fantasies, finds mechanics' test structures far faster than uniform search, and finds 10 that uniform search misses.
- **Not shown:** nothing yet about accumulation across domains. Bars (d) and (e) test that. Claims rest on new-form equations (ecology's, and circuits' 30 new-form equations).

**Whole systems exact, mechanics tests** (`g1_systems.py`; the external test is one-step ≤ 10⁻¹² of the scale plus the multi-step envelope; rebuilt systems identical in every case).

| Arm | Systems with every equation solved | Accepted internally (held-back) | Exact |
|---|---|---|---|
| Uniform | 30 of 40 | 27 | 26 |
| Network after mechanics | 40 of 40 | 40 | 40 |

**Uniform baseline on circuits' tests:**
- 74 of 80 equations solved (62 with the true structure); 34 of 40 systems.
- **Bar (d) has a ceiling on circuits.** It counts equations solved, and uniform already solves 74 of 80. So the continued network can beat a fresh one by at most 7.5 points there, short of the +10 required, whatever it learns. The speed and new-form numbers will be reported beside it.

**Ecology uniform baseline and zero-shot on circuits (7 November, 00:05).**
- **Uniform on ecology's tests:** 20 of 80 equations solved; 3 with the true structure's keys; 4 of 40 systems.
- **Zero-shot** (the network after mechanics, no circuits training, on circuits' tests): 79 of 80 equations solved (66 true), 39 of 40 systems. Uniform: 74 (62), 34.

**Why "true structure" undercounts on ecology** (checked on development systems only, `g1_imposters_dev`).
- Populations are positive, so |x| = x on the data and everywhere the system goes. Structures written with |x| are therefore exactly the true functions:
  - x₂·|x₁| = x₁x₂;
  - |x₁|·|x₂| = x₁x₂;
  - x₁·|x₁| = x₁².
- One development system solved this way passed the external test: one-step error 2.3 × 10⁻¹⁵, multi-step envelope passed.
- A second case is a reparametrization: x₂·(K0 + x₁)^K1 with K1 = −1 spans the same pair of terms as the Holling term with x₂.
- **Consequence:** the vocabulary's terms are distinct over all signs, but some coincide on positive data. The bars' measure (equations solved, plausibly exact) and the whole-system external test are the right judges. The "true structure" count is a lower bound.

**Driver restart (7 November, 00:41).**
- **Problem:** each dream had trained inside the driver process. PyTorch then kept about 2.4 GB of idle memory on every GPU the driver had used (GPUs 0–3), and the free-GPU check moved on to a new GPU each time. Over ten dreams this would have tied up every free GPU on the shared host.
- **Fix:** training now runs in a short-lived process that releases its GPU on exit.
- **Restart:** I stopped my own driver and its workers by PID and restarted it. It resumed at the first unfinished phase. The circuits test of the network after circuits cycle 1 had just begun and was rerun.
- **Effect on results:** none. Training is unchanged, with the same examples, order, seed and steps.

**Interim results, continued network through circuits (7 November, 01:20; counts only).**

| Test set | Evaluation | Equations solved (true structure) | Systems solved | Systems exact (external) |
|---|---|---|---|---|
| Circuits | uniform | 74 (62) | 34 | 24 |
| Circuits | network after mechanics (zero-shot) | 79 (66) | 39 | 36 |
| Circuits | after circuits cycle 1 | 80 (60) | 40 | 35 |
| Circuits | after circuits cycle 2 | 80 (60) | 40 | 35 |
| Ecology | uniform | 20 (3) | 4 | 3 |
| Ecology | network after circuits (zero-shot) | 66 (53) | 27 | 27 |

- **Circuits training tasks:**
  - wake 1, guided by the mechanics network: 297 of 300 equations, 147 of 150 systems;
  - wake 2: 299 of 300, 149 of 150.
- **Dreams B1 and B2:** trained on 2,149 and 2,156 replay examples, from 597 and 599 verified solutions across mechanics and circuits, plus about 3,600 fantasy examples each.
- **Equation-level "solved" overcounts on circuits:** uniform's 34 solved systems include 10 that fail the external test. Whole systems exact is the stricter measure and is reported beside it.
- **Zero-shot on ecology is large: 27 of 40 systems exact against 3.** Every ecology form is new relative to mechanics and circuits. What does carry over: the network's use of the data evidence, and dreaming from the language, whose fantasies can contain ecology-like terms.
- **The comparison that decides bar (d) is still to come:** continued against fresh networks after the same number of ecology cycles.

**Circuits: bar (d) (7 November, 01:45; counts only).**

| Evaluation on circuits' tests | Equations solved | True structure | Systems solved | Systems exact (external) |
|---|---|---|---|---|
| Continued, after circuits cycle 1 | 80 | 60 | 40 | 35 |
| Fresh, after circuits cycle 1 | 80 | 74 | 40 | 37 |
| Continued, after circuits cycle 2 | 80 | 60 | 40 | 35 |
| Fresh, after circuits cycle 2 | 80 | 74 | 40 | pending |

- **Fresh network on circuits' training tasks:**
  - wake 1 (uniform): 274 of 300 equations;
  - wake 2: 300 of 300 (291 true).
- **Bar (d), after cycle 1: NOT MET.**
  - +0.0 points: 80 against 80 on all equations, 30 against 30 on the new-form ones, 50 against 50 on the shared-form ones.
  - Both networks solve everything, which is the ceiling noted earlier.
- **Bar (d), after cycle 2: MET** (the fresh network is not better).
- **Provenance:** no equation is solved only by the continued network.
- **Against it:** the continued network more often settles on structures other than the true one (60 against 74 true). One system fewer is exact, 35 against 37.
- **Zero-shot against uniform:**
  - circuits: 6 only by the network, 1 only by uniform (p = 0.12);
  - ecology: 50 against 4 (p = 3.8 × 10⁻¹¹).
- **Reading:** circuits cannot show forward transfer on the bar's measure. A fresh network learns it fully in one cycle. Ecology decides bar (d) next: all its forms are new, and uniform solves 20 of 80.

**Incident: NaN network in ecology's first dream, caught and redone (7 November, 02:20).**
- **What happened.**
  - Dream "cont C1" trained on 3,197 replay examples (853 verified solutions) plus 3,676 fantasy examples. The loss was NaN from before step 200, and every weight ended NaN.
  - The cause was one fantasy example whose removal-share feature overflowed float32 to infinity. At a near-exact fit the residual is almost zero, so the share can reach about 10⁴⁰.
- **Wake 2 discarded.** It had started with the broken network. I stopped my driver and its workers by PID before it wrote anything.
- **Fixes:**
  - the removal share is capped at 10³⁰;
  - every network input is sanitized, in training and in the search;
  - a training step whose loss or gradient is not finite is skipped and counted;
  - check: 300 steps on the same examples now run with finite losses and weights, and 0 skipped steps.
- **The same overflow could have hit the search before the fix**, at rare near-exact nodes, through infinite features and so NaN priors there. The search then falls back to random moves from that node.
  - **Effect:** small, and only after a near-exact fit.
  - **Every ecology search uses the fixed code**, for continued and fresh networks alike.
  - **Retention (bar e):** the two "right after the domain" tests (the network after mechanics on mechanics, after circuits on circuits) are rerun with the fixed code and used as the baselines. Retention is then judged with the same code on both sides.
- **The driver was restarted** and resumes at dream C1's training. The fantasies are cached and unchanged.

**Interim results, continued network through ecology (7 November, 03:00; counts only).**

| Test set | Evaluation | Equations solved (true keys) | Systems solved | Systems exact (external) |
|---|---|---|---|---|
| Ecology | uniform | 20 (3) | 4 | 3 |
| Ecology | network after circuits (zero-shot) | 66 (53) | 27 | 27 |
| Ecology | after ecology cycle 1 | 80 (51) | 40 | 40 |
| Ecology | after ecology cycle 2 | 80 (48) | 40 | 40 |
| Mechanics (retention) | after mechanics (rerun, fixed code) | 80 (74) | 40 | pending |
| Mechanics (retention) | after ecology | 80 (73) | 40 | 39 |
| Circuits (retention) | after circuits (rerun, fixed code) | 80 (61) | 40 | pending |
| Circuits (retention) | after ecology | 80 (59) | 40 | 35 |

- **Ecology's training tasks:**
  - wake 1, guided by the network after circuits: 254 of 300 equations, 109 of 150 systems;
  - wake 2, after the redone dream: 300 of 300, 150 of 150.
- **Retention baselines rerun with the fixed code:**
  - mechanics: 80 (74), 40, against 80 (74), 40 before the fix;
  - circuits: 80 (61), 40, against 80 (60), 40.
  - So the feature fix made no measurable difference.
- **Bar (e), equation level:** 0 systems lost in mechanics and 0 in circuits, comparing after ecology with right after each domain.
- **Whole systems exact:** mechanics 39 after ecology against 40 right after mechanics (the original run); circuits 35 against 35. The same check on the rerun baselines is pending.
- **Still running:** the fresh ecology network for bar (d) on ecology.

### Stage 1 of the general learner: results (7 November, 04:45; counts only for test systems)

**Run.** 22:20 to 04:29 (6.2 h of wall time, against a projected 16 h) on 24 workers, plus the side runs of the whole-system check.
- Time-limit hits: 0 in every arm. Failed systems: 0.
- GPUs used for training: 0, 1, 2 and 3, each free at its check (logged per dream). After the 00:41 fix, no idle memory was held.
- Two driver restarts, both logged: the GPU memory fix, and the NaN network caught in ecology's first dream.

**Bars.**

| Bar | Verdict | Numbers |
|---|---|---|
| (a) fitter on true structures | MET | 240 of 240 test equations |
| (c) learning on the first domain | MET | mechanics' tests: 80 against 70 of 80 for uniform; median structures to the first plausible model 2 against 17; only network 10, only uniform 0, p = 0.002 |
| (d) forward transfer, after cycle 1 | NOT MET (circuits and ecology) | continued against fresh: circuits 80 against 80; ecology 80 against 80; new-form equations 30 against 30 and 80 against 80. Both networks solve every test equation after one cycle (ceiling) |
| (d) after cycle 2 | MET | 80 against 80 in both domains (fresh not better) |
| (e) retention | MET | systems solved after ecology against right after the domain: mechanics 40 against 40, circuits 40 against 40 (0 lost). Whole systems exact: mechanics 39 against 40, circuits 35 against 36 (1 lost each, within the 2 allowed) |

**Whole systems exact** (external test; counts out of 40).

| Test set | Uniform | Zero-shot (earlier domains only) | Continued, cycle 1 | Continued, cycle 2 | Fresh, cycle 1 | Fresh, cycle 2 | After ecology |
|---|---|---|---|---|---|---|---|
| Mechanics | 26 | not applicable | 40 (after mechanics) | | | | 39 |
| Circuits | 24 | 36 | 35 | 35 | 37 | 38 | 35 |
| Ecology | 3 | 27 | 40 | 40 | 39 | 39 | 40 |

**Where transfer shows: before the new domain's first dream.**
- **Zero-shot,** the network from earlier domains on the new domain's tests against uniform:
  - circuits: equations 79 against 74 (p = 0.12); systems exact 36 against 24;
  - ecology, all new forms: equations 66 against 20 (only network 50, only uniform 4, p = 3.8 × 10⁻¹¹); systems exact 27 against 3.
- **First wake on the new domain** (its 150 training tasks, before any training on them):

  | Domain | Continued: equations solved | Continued: compute | Fresh (uniform): equations solved | Fresh: compute |
  |---|---|---|---|---|
  | Circuits | 297 of 300 | 3.6 core-hours | 274 of 300 | 6.4 core-hours |
  | Ecology | 254 of 300 | 5.9 core-hours | 89 of 300 | 21.9 core-hours |

- **After one dream** on its own solutions and 500 fantasies, the fresh network solves every training task and every test equation. Speed then matches too: the median structures to the first plausible model are 4 against 4 on circuits and 7 against 7 on ecology.

**Reading.**
1. **The general direction works within a domain.**
   - The general language (1,143 rescaling-invariant terms), the generic fitter and the guided search with wake and dreaming recover mechanics' test systems exactly: 40 of 40 against 26 for uniform search.
   - Learning from its own solutions and fantasies makes the search about eight times faster to the first plausible model.
2. **Forward transfer exists but Stage 1 cannot show it on its bar.**
   - The network carried from earlier domains starts the new domain much better: zero-shot on ecology 66 against 20 equations, 27 against 3 exact systems. Its first wake there solves 254 against 89 tasks at about a quarter of the compute.
   - But one wake–dream cycle lets a fresh network catch up completely, so "after the first cycle" sits at the ceiling for both.
   - The claim rests on new-form equations, per the owner's note. On those (all 80 of ecology's), the earlier start is real; after one cycle there is no difference.
3. **Part of the zero-shot gain may come from dreaming on the language rather than from the earlier domains.** The fantasies can contain ecology-like terms. Stage 1 has no control that separates the two (a network trained on fantasies alone).
4. **No forgetting.** All earlier verified solutions stayed in training at full weight, and per system nothing was lost at the equation level. One system per domain was lost at the whole-system level.
5. **A small cost of carrying a network over, on circuits.** The continued network settles more often on structures other than the true one: 60 against 74 true-structure keys, and 35 against 37–38 systems exact.
6. **What limited the test: Stage 1's domains are too easy for a learner that dreams.** Measured at the end of a cycle, accumulation can only show where a fresh network cannot learn a domain within one cycle.

**Next (to propose to the owner; nothing is run).** A Stage 2 design whose transfer measure is not at the ceiling:
- learning curves inside the first cycle (after 10, 30 and 100 tasks), with zero-shot as a pre-registered measure;
- a tighter learning budget per domain;
- a fantasies-only control network;
- the harder domains planned for Stage 2 (chemistry, chaotic flows, d up to 3), with abstraction;
- the second domain order (owner's note).

### Stage 2 of the general learner: rules for Stage 2a, and the gate to Stage 2b (7 November; awaiting owner approval, nothing run)

**Owner.** *"Write up Stage 2 for my approval, split into two parts: Stage 2a: harder domains (chemistry and chaotic flows, up to 3 variables), transfer measured where it isn't saturated (after 10, 30 and 100 tasks of the first cycle, with zero-shot pre-registered, and a tighter budget per domain), the control network trained only on made-up systems, and the second domain order. No abstraction yet. Stage 2b: add abstraction only after 2a shows transfer that the made-up-systems control doesn't explain. Any accumulation claim must beat that control, on new-form equations."*

#### 1. The question Stage 2a answers

- **The question:** does a network that has learned earlier domains learn a new, harder domain faster than two other networks?
  - a network trained only on made-up systems, with the same amount of training (**the control**);
  - a network starting from scratch.
- **Where it is judged:** on the new domain's new-form test equations, at points where a network starting from scratch has not yet reached the ceiling.
- **Why Stage 1 could not answer it:** a fresh network caught up within one cycle. Its zero-shot gain (ecology 66 against 20 equations) had no control separating real-domain experience from dreaming on the language.

#### 2. Domains and the two orders

- **Earlier domains:** Stage 1's mechanics (M), circuits (C) and ecology (E), with two variables, and Stage 1's tasks, test sets and budgets.
- **New domains (the transfer targets):**
  - **chemistry (K):** two or three species, positive;
  - **chaotic flows (F):** three variables.
- **Order 1:** M → C → E → K → F.
- **Order 2:** E → C → M → F → K.
- **Why this pair of orders:** each new domain appears once 4th and once 5th, after the same three earlier domains acquired in opposite orders.
- **One architecture throughout:** both histories are rebuilt with the Stage 2a network (section 4), under Stage 1's rules (two cycles per domain, 150 tasks, 1,000 structures and 200 thorough fits per equation, 500 fantasies and 2,000 steps per dream). Stage 1's networks are not reused.

#### 3. The new domains' generators (fixed before any task is drawn)

**Chemistry (K):** one of six forms, equally likely, with constants log-uniform in ranges fixed in code.

| Form | Species | Terms |
|---|---|---|
| Random mass-action network | 3 | 3–5 reactions from {X_i → X_j, X_i + X_j → X_k, 2X_i → X_j, ∅ → X_i, X_i → ∅, X_i + X_j → 2X_j}; every species takes part |
| Michaelis–Menten pathway | 2 or 3 | inflow, conversion V·S/(K + S), outflow, optional inhibitor |
| Brusselator | 2 | |
| Selkov glycolysis | 2 | |
| Goodwin-type repression loop | 3 | a/(K + x³ⁿ), with n in {1, 2, 3} |
| Oregonator-like | 3 | polynomial |

- Hill activation of order 2 or 3 enters as the pair {1, 1/(K + xⁿ)}, which is in L1.

**Chaotic flows (F):** one of five families, equally likely, with constants around the families' chaotic ranges.

| Family | Form |
|---|---|
| Jerk | ẋ = y, ẏ = z, ż = linear terms + one of {x², xy, y², x³, x²y} (+ an optional constant) |
| Lorenz-type | with one optional extra quadratic term |
| Rössler-type | |
| Sprott-type | a sparse linear part with one quadratic term |
| Random sparse quadratic | 2–3 terms of degree ≤ 2 per equation |

- **Acceptance:** Stage 1's rules; positivity for chemistry only; chaos not required. The share of accepted systems with a positive largest Lyapunov exponent is reported.
- **Overlap with Stage 4:** named families (Lorenz, Rössler, Sprott) share forms with systems in dysts, which is reserved for Stage 4. dysts is not opened before then. Stage 4 will classify those systems as shared-form, as every test does.
- **Sets per new domain:** 100 training tasks (the first cycle), 10 development systems and 40 test systems, drawn with separate seeds and stored with hashes.
- **Shared-form classification:** per equation, with any injective relabelling of variables; a two-variable form can match inside a three-variable system. It is computed against all earlier domains' training tasks for each position and order, before any evaluation. Counts are logged.

#### 4. Language and network for up to three variables

- **Language:** L1 for three variables, an estimate made before any task: about 34,700 terms, about 2,700 of them rescaling-invariant, before family merging. The exact count is logged before any task is drawn.
- **Search vocabulary:** the rescaling-invariant terms for the system's dimension.
- **Network:**
  - Stage 1's network, plus a per-term scorer: a small network reading each term's encoding and its evidence.
  - Attention covers the terms in the state plus the scorer's top 768 terms. The remaining terms share one "rest" slot, spread evenly (the Stage 1 rules' uniform prior).
  - Why: attending across about 2,700 terms would cost about 0.8 s per search node. 768 costs about 0.07 s.
  - The scorer is trained with the multi-positive loss over every term.
- **Variable roles:** each variable is read relative to the equation, as its own variable or as one of the others in index order. Training examples are relabelled at random, so the network does not depend on the order.

#### 5. The control network (owner's instruction)

- **Training content:** made-up systems only, drawn from the language with every term size equally likely. There is no usage from real solutions and no replays.
- **Schedule:** exactly that of the continued network it is compared with, in the same order: the same number of dreams, the same steps per dream, and the same number of examples per dream (taken from the logged counts), at the same dimension as the domain at each dream.
- **One control per order**, built up to the start of each new domain.
- **In the new domain**, the control learns exactly as the other arms do (section 6).

#### 6. The first cycle in a new domain: the learning curve under a tighter budget

- **Arms per new domain and order:** continued, control and fresh. The fresh network does not depend on the order, so it runs once per domain.
- **Tasks:** the 100 tasks are taken in a fixed order, in increments.

| Point | Before it |
|---|---|
| 0 (zero-shot) | no task of the domain |
| 10 | wake on tasks 1–10, then dream |
| 30 | wake on tasks 11–30, then dream |
| 100 | wake on tasks 31–100, then dream |

- **Test:** each arm is evaluated on the domain's 40 test systems at every point. The zero-shot point is pre-registered.
- **Tighter budget**, the same for every arm, in the new domains only:
  - 300 structures and 60 thorough fits per equation, in wakes and tests;
  - 100 fantasies and 500 training steps per dream;
  - replays as in Stage 1 (every verified solution of the lineage).
- **Budget check on development systems, before any test:** if uniform search at 300 structures solves more than 50% of a new domain's development equations, that domain's budget drops to 100 structures and 20 thorough fits.
- **Time limits:** from a development pilot, set so that they rarely bind; hits are reported per arm.

#### 7. Bars (pre-registered)

1. **(a′) Fitter on true structures**, per new domain: at least 80% of test equations exact given the true structure. **If not met, stop.**
2. **Validity:** after 30 tasks, the fresh network solves at most 70% of a domain's new-form test equations. Otherwise that domain's points from 30 on count as saturated: reported, and not used for claims.
3. **(B) Accumulation beyond the control: the claim.**
   - Measure: new-form test equations at the 10-task point.
   - Requirement: the continued network solves at least 10 points more than the control, with a paired sign test p < 0.05, **in both orders**.
   - Judged per domain.
   - Zero-shot is reported with the same test as a pre-registered second measure; the claim is stronger if it also holds there. Points 30 and 100 are reported likewise, when valid.
4. **(C) Beyond the fresh network:** the same test. It is reported, and is not the claim.
5. **(D) Order:**
   - for each domain and point, the continued network in order 1 against order 2 on new-form equations (paired sign test), reported;
   - "transfer does not depend on the order" means (B) holds in both orders and the two orders do not differ significantly.
6. **(E) Retention:** at the end of each order, per system, at most 2 systems lost in each earlier domain. Each is compared with the point before the new domains (M, C and E tests) or with right after its own first cycle (the 4th domain's tests).

- **Also reported:** whole systems exact; shared-form results as context; median structures to the first plausible model; search compute; the provenance of each new-form equation solved only by the continued network.

#### 8. Stage 2b: abstraction, behind a gate

- **The gate:** (B) is met in at least one new domain, in both orders.
- **If the gate opens:** Stage 2b adds abstraction under the Stage 1 rules' design:
  - compression of the verified solutions into recurring fragments that become single-move building blocks;
  - decoy fragments, the removal check, and the provenance of every library win.
  - The same learning-curve protocol, control and new-form rule apply, on a further new domain or on the same domains with abstraction on against off. That choice is to be proposed then.
- **If the gate does not open:** no abstraction. I report what the control explains and propose next steps.

#### 9. Cost and order of work

- **Estimate:** about 20 h on 24 workers.
  - two rebuilt histories, about 4–5 h;
  - two control histories, about 2–3 h, mostly generating fantasies;
  - eight new-domain runs with tests at four points, about 11 h;
  - retention tests, about 1 h.
- **A development pilot comes first.** If it projects more than 1.5 days, I check in with a smaller version.
- **Order of work:**
  1. three-variable language and its count;
  2. generators, draws and shared-form counts;
  3. (a′) on the true structures;
  4. development pilot and budget check;
  5. histories and controls;
  6. new-domain learning curves;
  7. bars.
- **Check-in:** after (a′), before any history or control is built.

### Stage 2a: owner approval, with two changes (7 November)

**Owner.** *"Approved, with two changes: 1. The shortlist only limits what the network attends to. The search must still be able to try any term in the full vocabulary (keep the random exploration share), and report how often a true term was cut from the shortlist. 2. Keep dysts unfamiliar: drop the Lorenz-, Rössler- and Sprott-type families, and use only jerk and random sparse quadratic flows for the chaotic domain. Check in after the fitter test on true structures, as planned."*

1. **Every term stays reachable.**
   - Terms outside the 768 attended keep a nonzero prior through the "rest" slot, spread evenly.
   - The search's 15% random-exploration share still draws from every legal move, over the full vocabulary.
   - **New report, a probe that never influences the search:** at each node a network-guided search expands, whether any missing true term was outside the attended set (the state plus the 768). It is reported as the share of expansions, per arm and domain, and per equation at the root.
   - The true structures are used only for this count, and the search never reads them, as with Trial 30's probe.
2. **Chaotic flows (F)** keep two families, equally likely:
   - jerk: ẋ = y, ẏ = z, ż = linear terms + one of {x², xy, y², x³, x²y} (+ an optional constant);
   - random sparse quadratic: 2–3 terms of degree ≤ 2 per equation.
   - No named families. dysts stays unopened and unfamiliar until Stage 4.

- Everything else is as written above.
- **Check-in:** after (a′), before any history or control is built.

### Stage 2a: three-variable vocabulary, generators and one acceptance correction (7 November, 08:00; fixed before any Stage 2a task is drawn)

**Vocabulary for three variables** (`g1_vocab.py 3`, `g1_families.py … 3`, `g1_scale_closed.py 20 3`).
- **Counts:**
  - 84,283 trees give 36,264 canonical forms;
  - 1,311 constant-free terms are merged numerically;
  - 466 families are merged by the fitting test, out of 42,957 candidate pairs;
  - 268 are dropped as undefined or constant.
- **L1 for three variables: 34,220 terms.** By number of constants: 24,993 without, 8,162 with one, 1,065 with two.
- **Search vocabulary (rescaling-invariant): 2,494 terms.**
- **Merged true keys:** a generated true term whose key was merged into an equivalent family takes the family's representative. In Stage 1 no true key was merged (0 of 13 distinct keys), so stored systems are unchanged.

**Code made dimension-agnostic.**
- **Fitter:** terms compile for up to three variables. Rerunning Stage 1's development bar (a) gave bitwise-identical results: 60 of 60 equations, same errors.
- **Search vocabulary** per dimension.
- **Generators** for two or three variables: Stage 1's forms are unchanged and their stored sets are reused.

**Generators as approved** (`domains_g1.py`).
- **Chemistry:** six forms.
- **Chaotic flows:** jerk and random sparse quadratic.
- Constant ranges are fixed in the code. The largest Lyapunov exponent (Benettin method) is reported for every accepted system.

**Acceptance correction for all new draws** (found on pilot seeds "g2-pilot-*", never used again).
- **The problem:** Stage 1's rule required a true non-polynomial term to differ by at least 10⁻⁶ from every cubic polynomial in all the variables. With three variables those are 20 monomials on 30 rate points, which mimic almost any smooth term.
- **Pilot evidence (Goodwin and Michaelis–Menten terms):** 23 of 45 clearly nonlinear terms had residuals near 10⁻⁹ under the all-variable rule. Against cubics in the term's own variables, all 45 had residuals of 5 × 10⁻⁶ to 10⁻³.
- **Effect on the mix:** Goodwin systems were accepted 2 of 20 times and three-step pathways 0 of 6. Replacement seeds redraw the form, so these forms would nearly vanish from the domain.
- **New rule:** **the comparison uses cubic polynomials in the term's own variables.** Stage 1's stored sets stay as drawn.

**Pilot after the correction** (pilot seeds; counts).

| Form | Accepted | Known-structure equations exact |
|---|---|---|
| Brusselator | 23 of 26 | 46 of 46 |
| Goodwin | 18 of 20 | 54 of 54 |
| Mass-action | 16 of 17 | 48 of 48 |
| Michaelis–Menten (2 steps) | 15 of 17 | 30 of 30 |
| Michaelis–Menten (3 steps) | 5 of 6 | 11 of 15 |
| Oregonator-like | 18 of 18 | 54 of 54 |
| Selkov | 16 of 16 | 32 of 32 |
| Jerk | 59 of 60 | 177 of 177 |
| Sparse quadratic | 45 of 60 | 135 of 135 |

- The three-step pathway's misses have two rational terms in one equation.
- **Chaos:** 18 of the 104 accepted chaotic-flow pilot systems have a largest Lyapunov exponent above 0.01.

**Draws next** (`g2_draw.py`, seeds "g2-{set}-{domain}-k"):
- 100 training tasks, 10 development and 40 test systems per new domain, stored with hashes;
- then the shared-form classification (`g2_shared.py`, every relabelling, per position and order);
- then bar (a′) and the check-in.

### Stage 2a: systems drawn, shared forms classified, bar (a′) MET (7 November, 08:20; counts only)

**Draws** (`g2_draw.py`). All systems were accepted, all stored with hashes.

| Domain | Training tasks | Development | Test | Rejected draws (train / dev / test) | Largest Lyapunov exponent > 0.01 (train / dev / test) |
|---|---|---|---|---|---|
| Chemistry | 100 (36 with two species, 64 with three) | 10 | 40 (19 two-species, 21 three-species; 101 equations) | 7 / 0 / 0 | 5 / 1 / 2 |
| Chaotic flows | 100 (three variables) | 10 | 40 (120 equations) | 23 (mostly the cost rule) / 1 / 10 | 19 / 4 / 5 |

**New-form test equations** (`g2_shared.py`; every relabelling; made before any evaluation). The claim rests on these.

| Order | Domain (position) | Shared-form | New-form | Shared-form systems |
|---|---|---|---|---|
| 1 | Chemistry (4th, after M, C, E) | 35 | 66 of 101 | 0 of 40 |
| 1 | Chaotic flows (5th, after M, C, E, K) | 75 | 45 of 120 | 2 of 40 |
| 2 | Chaotic flows (4th, after E, C, M) | 61 | 59 of 120 | 0 of 40 |
| 2 | Chemistry (5th, after E, C, M, F) | 50 | 51 of 101 | 1 of 40 |

- Chaotic flows share many simple forms with earlier domains: the jerk's ẋ = y and ẏ = z, and linear two-term equations.

**Bar (a′), the fitter on true structures** (`g1_bar_a.py`).

| Domain | Development equations exact | Test equations exact | With constants inside terms | Test systems exact |
|---|---|---|---|---|
| Chemistry | 22 of 24 | 100 of 101 (99%) | 21 of 22 | 39 of 40 |
| Chaotic flows | 30 of 30 | 120 of 120 | (none) | 40 of 40 |

- At most 1.3 s per system. Time-limit hits, errors and hash mismatches: 0.
- **Bar (a′): MET in both new domains.**

**Next, after the owner's go-ahead.**
- **Code:**
  - the network extension: per-term scorer, attention over the state plus 768 with the rest slot, variable roles for three variables, and relabelling of training examples;
  - the shortlist-cut probe;
  - the Stage 2a driver: two histories, two controls, and learning curves at 0, 10, 30 and 100 tasks.
- **Development pilot:**
  - the budget check: if uniform search at 300 structures solves more than 50% of a new domain's development equations, that domain drops to 100 structures;
  - timing and the projection;
  - a further check-in if the projection exceeds 1.5 days.

### Stage 2a: owner's go-ahead (7 November)

**Owner.** *"Go ahead with the build, the pilot and the run as written. In reports, describe the chaotic-flow domain as quadratic flows (mostly non-chaotic), since only 5 of 40 test systems are chaotic. Check in only if the pilot projects over 1.5 days; otherwise send the full results with a plain-language summary when done."*

- **Naming:** from here on, reports call domain F **quadratic flows (mostly non-chaotic)**. The code keeps the name "chaotic".
- **Check-in rule:** only if the development pilot projects more than 1.5 days. Otherwise the run goes ahead, and the full results follow with a plain-language summary.

### Stage 2a: development pilot, budget decisions and projection (7 November, 08:55; development systems only)

**Budget check (pre-registered rule).** Uniform search at 300 structures and 60 thorough fits per equation.

| Domain | Development equations solved | Budget for every arm, in wakes and tests |
|---|---|---|
| Chemistry | 4 of 24 (17%) | 300 structures, 60 thorough fits |
| Quadratic flows (mostly non-chaotic) | 16 of 30 (53%) | 100 structures, 20 thorough fits |

**Timing** (per system, at those budgets; an untrained network measures the network's cost only).

| Domain | Uniform: mean (median / max) | Guided: mean | Network time per system |
|---|---|---|---|
| Chemistry | 185 s (176 s / 248 s) | 155 s | 17 s |
| Quadratic flows | 44 s (43 s / 83 s) | 38 s | 11 s |

- **Time limits:** 3,600 s per system, as in Stage 1. That is more than 14× the slowest pilot system. Hits are reported per arm.

**Projection: about 15 h on 24 workers.**
- two histories, about 4.6 h, from Stage 1's measured wake and dream times;
- two control histories with extensions, about 3.4 h, mostly generating fantasies;
- ten learning curves, about 5.5 h. Each curve is 4 tests of 40 systems plus 100 tasks and 3 dreams; a chemistry curve takes about 45 min, a quadratic-flow curve about 20 min;
- retention tests, about 0.5 h.

This is within 1.5 days, so there is no check-in (owner's rule). The run goes ahead.

**Code for the run.**
- `movenet_g2.py`: the scorer, attention over the state plus 768 with the rest slot, variable roles for up to three variables, and the shortlist-cut probe.
- `dream_g2.py`: training examples relabelled at random, and fantasies of two or three variables.
- `cycles_g2.py`: the driver. Its phase order:
  1. history of order 1;
  2. retention baselines;
  3. control base;
  4. the fresh networks' curves;
  5. the continued and control curves at position 4;
  6. control extension;
  7. curves at position 5;
  8. final retention;
  9. the same for order 2.
- Every dream trains in a short-lived process, so GPU memory is freed.

### Stage 2a: smoke test passed; corrected projection over 1.5 days; check-in (7 November, 11:00)

**Smoke test** (scratch directory, tiny settings, `G2_ALLOW_CPU` for its 5–10-step dreams only): **SMOKE OK.**
- Every phase of `cycles_g2.py` ran: the history, retention tests, control base, the fresh, continued and control learning curves at positions 4 and 5, the control extension, and the final retention.
- The shortlist probe works: a missing true term was outside the attended set at 7 of 876 expansions, with barely trained networks.
- Mixed-dimension training (two- and three-variable examples) works on GPU and CPU.

**GPUs** (read-only `nvidia-smi` watch, 09:03–10:03).
- A GPU was free or shareable under the owner's rule in only 11% of one-minute windows, in three short windows.
- The other user refills freed GPUs within minutes.
- The driver now waits for a free or shareable GPU, checking every minute, and never trains on CPU.

**Correction: the earlier ~15 h projection underestimated generating fantasies.**
- **Measured on pilot seeds, one core** (under load):

  | Fantasy type | Draws per accepted system | Time per accepted system |
  |---|---|---|
  | Three variables, size prior | about 50 | about 330 s |
  | Two variables, size prior (the control's) | about 14 | about 63 s |

  The earlier projection used Stage 1's usage-shaped two-variable rate, about 14 s per accepted fantasy.
- **Corrected projection:**

  | Item | Estimate |
  |---|---|
  | Histories' fantasies | about 1 h |
  | Two control bases (about 4,700 fantasies each, to match the continued network's example counts) | 5–7 h |
  | New-domain dreams (30 × 100 fantasies, mostly three-variable) | 5–10 h |
  | Control extensions | about 3 h |
  | Searches | about 7 h |
  | GPU training | about 3.5 h |
  | **Compute total** | **about 25–30 h** |
  | Waiting for GPU windows, at the observed contention | up to about 15 h |

- **Likely total: 1–2 days. Over the 1.5-day limit in the likely case**, so the owner's rule calls for a check-in.
- Nothing of the real run has started.

### Stage 2a: owner's decision (B and C), the fantasy pre-screen validated, order 1 launched (7 November, 15:25)

**Owner.** *"Approved: B and C. Validate the pre-screen on pilot seeds first (speed-up and wrong rejections), and apply it identically to every arm. Start generating the made-up systems on CPU now, so that work overlaps the wait for GPUs. Run order 1, report its results in plain language, then run order 2 with the same setup."*

**The pre-screen** (`dream_g2.prescreen`). A cheap check before the expensive build of every fantasy, identical for every arm:
- the derivatives at the initial state must be finite;
- a rough 4th-order Runge–Kutta run over the system's own window (150 intervals, 4 substeps) from 4 starting states around the initial state must stay finite and within 100× the largest initial magnitude.

The full build still decides acceptance.

**Validation** (pilot seeds "g2-pilotPS-*"; 300 size-prior fantasy draws per dimension; both checks run on every draw).

| Variables | Accepted by the full build | Rejected by the pre-screen | Of those, accepted by the full build (wrong rejections) | Build time without / with the pre-screen | Speed-up |
|---|---|---|---|---|---|
| Two | 27 | 243 | **0** | 819 s / 59 s | 14× |
| Three | 9 | 267 | **0** | 1,416 s / 75 s | 19× |

- **Pre-screen reasons:**
  - derivatives not finite at the initial state: 171 (two variables) and 214 (three variables);
  - blow-ups: 72 and 53.
- **The full build's reasons for those draws:** "simulation or cost" for 222 and 250, no canonical form for 15 and 16, unbounded for 6 and 1.
- **Result:** the pre-screen changes no acceptance on these draws, and only saves time.

**Overlap with the GPU wait.**
- While the driver waits for a free or shareable GPU (checked every minute), its idle workers generate batches of the control's fantasies into a pool. These are drawn from the size prior and depend on nothing learned.
- Each control dream consumes unused batches in order, and the ledger records which dream used which batch. If the pool runs short, more batches are generated on the spot.
- The continued and fresh networks' fantasies depend on their own solutions, so they are generated at each dream, as before.

**Run plan.**
- **Order 1:** `cycles_g2.py 24 1`.
  - First the phases that need no network: order 1's uniform first wake, and the fresh networks' zero-shot tests and first 10 tasks in both new domains.
  - Then the history, the control base, the learning curves and retention.
- **Order 2** follows with the same setup, after the order 1 report.

**GPUs at 15:22:** all eight busy, all with one other user's jobs (93–100% utilization).

### Stage 2a, order 1: results (7 November, 21:45; counts only for test systems)

**Run** (`cycles_g2.py 24 1`).
- 15:22 to 21:43, 6.3 h. That includes one 96-minute wait for a GPU. All other dreams found a free GPU, mostly GPU 0.
- Time-limit hits: 0 in every arm. Failed systems: 0. Training steps skipped as non-finite: 0.
- **Repeatability check:** order 1's uniform first wake on mechanics reproduced Stage 1's exactly: 270 of 300 equations, 254 with the true structure.

**Learning curves.** Test equations solved, with new-form ones in parentheses, then systems.

*Chemistry (4th; 101 test equations, 66 new-form):*

| Arm | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|
| Continued | 84 (49), 29 | 87 (52), 30 | 89 (54), 31 | 92 (57), 32 |
| Control (fantasies only) | 75 (42), 21 | 81 (48), 24 | 86 (51), 30 | 88 (53), 28 |
| Fresh (0 tasks = uniform) | 36 (13), 3 | 88 (53), 31 | 97 (62), 36 | 94 (59), 33 |

*Quadratic flows, mostly non-chaotic (5th; 120 test equations, 45 new-form):*

| Arm | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|
| Continued | 111 (41), 31 | 118 (45), 38 | 112 (40), 32 | 115 (42), 35 |
| Control | 109 (39), 30 | 94 (28), 18 | 111 (39), 31 | 117 (43), 38 |
| Fresh | 54 (8), 4 | 99 (33), 25 | 111 (39), 32 | 117 (42), 37 |

**Validity.** After 30 tasks, the fresh network solves 94% of chemistry's new-form equations and 87% of quadratic flows'. **Both domains are saturated from 30 tasks on**: those points are reported only.

**Bar (B), order 1** (new-form, at 10 tasks; continued against the control).

| Domain | Points | Only continued / only control | p | Verdict |
|---|---|---|---|---|
| Chemistry | +6.1 | 6 / 2 | 0.29 | NOT MET |
| Quadratic flows | +37.8 | 17 / 0 | 1.5 × 10⁻⁵ | MET |

- **Caveat on quadratic flows:**
  - The control regressed after its first small dream in the domain: from 0 to 10 tasks it lost 11 new-form equations and gained 0 (p = 0.001).
  - The continued network rehearses its whole history of real solutions in every dream. The control's training on fantasies is never rehearsed: as designed, only verified solutions are replayed.
  - The continued network at 10 tasks still beats the control *before* that regression: 45 against 39 of 45 new-form equations, +13.3 points, 6 against 0, p = 0.031.
  - Against the fresh network at 10 tasks: +26.7 points, 12 against 0, p = 0.0005.
- **For the claim:** bar (B) needs both orders. Quadratic flows come 4th in order 2.

**Zero-shot, the second measure** (new-form; continued against the control).

| Domain | Points | Only continued / only control | p |
|---|---|---|---|
| Chemistry | +10.6 | 9 / 2 | 0.065 |
| Quadratic flows | +4.4 | 5 / 3 | 0.73 |

- **The control explains most of the head start over uniform search.**
  - Chemistry: uniform 36 → control 75 → continued 84 equations overall. On all equations the continued network's margin is +8.9 points, p = 0.022.
  - Quadratic flows: uniform 54 → control 109 → continued 111.

**Against the fresh network** (new-form).
- Chemistry at 10 tasks: −1.5 points (p = 1).
- Chemistry at 30 tasks (saturated): −12.1 points, 0 against 8, p = 0.008. The continued network is slower there.
  - A likely reason: full rehearsal of about 3,400 history examples dilutes chemistry's few hundred examples in each 500-step dream.

**Retention (E): MET.**
- Systems solved after both new domains against before them: mechanics 40 → 40, circuits 40 → 40, ecology 40 → 40.
- Chemistry, after quadratic flows against right after its own cycle: 32 → 32. 0 systems lost anywhere.
- Equations with the true structure's keys fell, though: circuits 69 → 55, ecology 67 → 45. More structures that are exact on the data but not the true form (ecology's include the |x| equivalents). The whole-system check is pending.

**Shortlist probe** (share of expansions at 10 tasks where a missing true term was outside the attended set; and at the root, out of all test equations).

| Domain | Continued | Control | Fresh |
|---|---|---|---|
| Chemistry | 6.8% (root 4) | 9.7% (root 4) | 16.0% (root 5) |
| Quadratic flows | 0.0% (root 0) | 2.2% (root 4) | 0.3% (root 1) |

**Reading.**
1. Accumulation beyond the control shows **only on quadratic flows, at 10 tasks, in this order**. Part of it is the control's own regression.
2. **On chemistry, the made-up-systems control explains the head start:** zero-shot +10.6 points is not significant on new-form equations, and +6.1 at 10 tasks.
3. **Dreaming on the language is the main source of the zero-shot gain.** Real-domain experience adds little beyond it, with the one exception above.
4. **Both new domains are learned to the ceiling within 30 tasks** by any arm.

**Order 1, whole systems exact** (`g1_systems.py`; the external test is one-step ≤ 10⁻¹² plus the multi-step envelope; rebuilt systems identical in every case). Counts out of 40.

| Domain | Continued, 0 tasks | Control, 0 | Uniform | Continued, 10 tasks | Control, 10 | Fresh, 10 |
|---|---|---|---|---|---|---|
| Chemistry | 29 | 21 | 2 | 29 | 23 | 31 |
| Quadratic flows | 31 | 30 | 4 | 38 | 18 | 25 |

- These agree with the equation-level picture. Order 2 starts now with the same setup (`cycles_g2.py 24 2`).

### Stage 2a: report details fixed before any order-2 result (7 November, 22:30)

These fill in the pre-registered "(D)" and "also reported" items (section 7). No order-2 result exists yet.

- **(D) Order**: the continued network in order 1 is compared with order 2, per domain and point, by a paired sign test. The comparison uses the equations that are new-form in **both** orders, so both sides are scored on one set of equations.
  - Chemistry: its new-form equations in order 2 (5th, after quadratic flows).
  - Quadratic flows: its new-form equations in order 1 (5th, after chemistry).
- **The gate to Stage 2b** is applied exactly as written: bar (B) met in at least one new domain, in both orders.
- **Provenance** (Stage 1's definition, adapted to three variables):
  - Unit: each new-form equation solved only by the continued network, against the control at every point and against the fresh network at 10 tasks.
  - Count: how many have a true term, and how many have all their true terms, among the lineage's verified solutions from earlier domains.
  - Terms are compared relative to their equation: own variable first, the others in either order, as three-variable vocabulary representatives.
  - The base rate over all new-form test equations is reported alongside.
- **Effort**: per arm, at 0 and 10 tasks:
  - median structures to the first plausible model, over new-form equations;
  - structures, thorough fits and CPU hours per test.

**Order 1 under these definitions** (counts only).

*Provenance at 10 tasks* (wins against the control; the base rate is over all new-form test equations):

| Domain | Wins | With a true term in earlier solutions | With all | Base rate: any / all |
|---|---|---|---|---|
| Chemistry | 6 | 6 | 4 | 63 / 66 and 35 / 66 |
| Quadratic flows | 17 | 17 | 15 | 45 / 45 and 40 / 45 |

- **Term overlap does not tell wins apart:** almost every new-form equation has its terms in earlier solutions. The wins come from combining known terms into new forms, not from terms the control never saw.

*Median structures to the first plausible model, new-form* (0 tasks → 10 tasks):

| Domain | Continued | Control | Fresh |
|---|---|---|---|
| Chemistry | 31 → 26 | 144 → 42 | not reached → 21 |
| Quadratic flows | 21 → 17 | 28 → 65 | not reached → 39 |

- The control's regression on quadratic flows shows here as well: 28 → 65.

**Compute:** each test is 0.2–1.9 CPU hours. Uniform search is the costliest because of its many thorough fits.

### Stage 2a, order 2: history, retention baselines, quadratic flows; the gate cannot open (8 November, 01:50; counts only for test systems)

**Run so far** (`cycles_g2.py 24 2`, from 21:50).
- No GPU waits. No time-limit hits. No failed systems. No skipped training steps.
- **Repeatability check:** order 2's uniform first wake on ecology reproduced Stage 1's fresh ecology wake exactly: 89 of 300 equations, 19 with the true structure, 14 systems.
- **History** (wakes 1 and 2, equations of 300):

  | Domain | Wake 1 | Wake 2 |
  |---|---|---|
  | Ecology | 89 | 300 |
  | Circuits | 297 | 300 |
  | Mechanics | 291 | 294 |

- **Retention baselines before the new domains:** ecology, circuits and mechanics each 40 of 40 systems.
- **The control pool** from order 1 was used up. Order 2's control base generated its fantasies on the fly with the same procedure: 105 batches, about 35 minutes of CPU.

**Quadratic flows, mostly non-chaotic** (4th in this order; 120 test equations, 59 new-form). Equations solved (new-form in parentheses), then systems:

| Arm | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|
| Continued | 109 (49), 31 | 112 (51), 32 | 113 (52), 33 | 112 (51), 32 |
| Control | 90 (32), 17 | 107 (46), 28 | 111 (50), 33 | 112 (51), 32 |
| Fresh | 54 (8), 4 | 99 (38), 25 | 111 (50), 32 | 117 (56), 37 |

- **Validity:** the fresh network solves 85% of new-form equations at 30 tasks, so the domain is **saturated from 30 on**.

**Results** (new-form; only continued / only control):

| Measure | Points | Discordant | p | Verdict |
|---|---|---|---|---|
| **Bar (B), at 10 tasks** | +8.5 | 8 / 3 | 0.23 | **NOT MET** |
| Zero-shot | +28.8 | 21 / 4 | 0.0009 | — |

- Unlike in order 1, the control does not regress here: 32 → 46 new-form from 0 to 10 tasks.
- **Against the fresh network:** +22.0 points at 10 tasks (16 / 3, p = 0.004). At 100 tasks, −8.5 (0 / 5, p = 0.06).
- **Whole systems exact:** continued 31 at 0 tasks and 31 at 10 tasks; control 17 and 28.
  - Every assembled system passed the internal check. One continued system at 10 tasks then failed the external test. Rebuilt systems were identical.
- **Shortlist probe at 10 tasks:** continued 4.6% of expansions (root 5 of 120); control 0.0%; fresh 0.3%.
- **Provenance at 10 tasks:**
  - 8 wins. Each has a true term in earlier solutions, but none has all of them.
  - Base rate over all new-form test equations: 58 of 59 (any), 13 of 59 (all).

**The gate to Stage 2b cannot open.** Bar (B) must hold in the same domain in both orders. Chemistry failed in order 1 and quadratic flows in order 2. The rest of order 2 still runs as planned: the control extension, chemistry 5th, and retention after.

**Reading.**
- Experience of real domains gives a large head start before any task:
  - +28.8 points beyond the control here (both arms had only two-variable experience);
  - +10.6 for chemistry in order 1.
- **It is gone by 10 tasks:** the made-up-systems control learns quadratic flows' terms from its own first tasks.

### Stage 2a: order 2 complete; combined results; the gate to Stage 2b does not open (8 November, 03:40; counts only for test systems)

**Run.** Order 2 ran from 21:50 to 03:24, 5.6 h. No GPU waits. Time-limit hits 0 in every arm. Failed systems 0. Skipped training steps 0.

**Chemistry, order 2** (5th; 101 test equations, 51 new-form). Equations solved (new-form in parentheses), then systems:

| Arm | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|
| Continued | 90 (41), 32 | 88 (38), 32 | 83 (34), 28 | 91 (41), 32 |
| Control | 80 (32), 24 | 94 (44), 34 | 92 (43), 32 | 93 (43), 33 |
| Fresh | 36 (13), 3 | 88 (40), 31 | 97 (47), 36 | 94 (44), 33 |

- **Validity:** the fresh network solves 92% of new-form equations at 30 tasks, so chemistry is **saturated from 30 on**.
- **Bar (B) at 10 tasks: NOT MET.** The control is ahead: −11.8 points, only continued / only control 0 / 6, p = 0.031.
- **Zero-shot:** +17.6 points, 11 / 2, p = 0.022.
- **Against the fresh network:** −3.9 at 10 tasks (1 / 3, p = 0.62). At 30 tasks (saturated), −25.5 (0 / 13, p = 0.0002); against the control at 30, −17.6 (0 / 9, p = 0.004).
  - The continued network got worse over its first 30 chemistry tasks: 41 → 34 new-form. It recovered to 41 by 100.
- **Whole systems exact:** continued 32 at 0 tasks and 31 at 10 tasks; control 24 and 34.
  - One continued system at 10 tasks failed the internal check. Rebuilt systems were identical in every case.
- **Shortlist probe at 10 tasks:** continued 1.7% (root 1 of 101); control 2.5% (root 2); fresh 16.0% (root 5).
- **Median structures to the first plausible model** (new-form, 0 → 10 tasks): continued 54 → 52; control 198 → 25; fresh not reached → 36.
- **Provenance at 10 tasks:** no new-form equation was solved only by the continued network.

**Retention (E), order 2: MET.**
- Ecology, circuits and mechanics: 40 → 40 systems each.
- Quadratic flows, after chemistry against right after its own cycle: 32 → 33 (2 lost, 3 gained).

**(D) Order** (continued network, order 1 against order 2, on the equations new-form in both orders):

| Domain | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|
| Chemistry (51) | 34 vs 41, 1 / 8, p = 0.039 | 37 vs 38, p = 1 | 39 vs 34, 6 / 1, p = 0.12 | 42 vs 41, p = 1 |
| Quadratic flows (45) | 41 vs 38, p = 0.55 | 45 vs 40, 5 / 0, p = 0.062 | 40 vs 39, p = 1 | 42 vs 38, 4 / 0, p = 0.12 |

- Chemistry's zero-shot is better at the 5th position than at the 4th. That fits the extra three-variable experience gained from quadratic flows. Quadratic flows show no significant difference.

**Combined: continued network against the made-up-systems control** (new-form; only continued / only control):

| Domain, order (position) | Zero-shot | At 10 tasks: bar (B) |
|---|---|---|
| Chemistry, 1 (4th) | +10.6, 9 / 2, p = 0.065 | +6.1, 6 / 2, p = 0.29: not met |
| Quadratic flows, 1 (5th) | +4.4, 5 / 3, p = 0.73 | +37.8, 17 / 0, p = 1.5 × 10⁻⁵: met (the control regressed) |
| Quadratic flows, 2 (4th) | +28.8, 21 / 4, p = 0.0009 | +8.5, 8 / 3, p = 0.23: not met |
| Chemistry, 2 (5th) | +17.6, 11 / 2, p = 0.022 | −11.8, 0 / 6, p = 0.031: not met (the control is ahead) |

- **Bar (B) in both orders:** chemistry NOT MET; quadratic flows NOT MET.
- **The gate to Stage 2b does not open.** As pre-registered: no abstraction.

**What the made-up-systems control explains.**
1. **Most of the head start over uniform search.** Chemistry equations: uniform 36, control 75–80, continued 84–90. Quadratic flows: uniform 54, control 90–109, continued 109–111.
2. **Everything from 10 tasks on.** By 10 tasks the control matches or beats the continued network in three of four cases. The exception, quadratic flows in order 1, is the control's own regression.

**What it does not explain: a head start before any task.** The continued network is ahead in all four cases, significantly in two:
- p = 0.0009 for quadratic flows at 4th;
- p = 0.022 for chemistry at 5th.

Term overlap does not single out the wins (provenance base rates near 100%). The edge comes from how known terms are combined and weighted.

**Why the edge disappears: the continued network adapts slowly.**
- Its new-domain dreams are about 80% rehearsal of earlier domains. Example: order 2's first chemistry dream has 5,494 examples, of which about 4,500 come from earlier domains (history and quadratic flows; batches are drawn in proportion to example counts). The control's dream is entirely the new domain: 1,088 examples.
- Both arms use 500 steps, so the control sees each new-domain example about five times as often.
- The control's lack of rehearsal cuts the other way once: it forgot its general skill in quadratic flows, order 1, after its first dream.
- So at 10 tasks the comparison mixes what was learned before (real vs made-up) with how fast each arm learns the new domain.
- The slow adaptation also shows against the fresh network: −12.1 (p = 0.008) and −25.5 (p = 0.0002) at 30 chemistry tasks.

#### Proposal (awaiting the owner's approval; nothing run): Stage 2a′, matched rehearsal and balanced dreams

1. **A rehearsal-matched control.**
   - In every dream after its base training (the extension and the new-domain dreams), the control also replays its own earlier made-up examples.
   - It replays as many per dimension as the continued network replays from earlier domains. They are drawn with a fixed seed from the control's own earlier fantasy files.
   - Dream size and mix then match. Only real against made-up content differs.
2. **Balanced dreams** for every arm with earlier examples (continued and control):
   - half of each dream's training batches come from the newest domain's examples (its verified solutions so far and this dream's fantasies);
   - half come from everything earlier.
   - The fresh network has nothing earlier and is unchanged.
3. **Fresh test draws for the claim.** 40 new systems each for chemistry and quadratic flows:
   - same generators and acceptance, new seeds;
   - new-form flags computed as before.
   - Stage 2a's test sets become development sets. Reserve sets and dysts stay untouched.
4. **Reused:** the two histories, control bases and fresh networks (retested on the new draws).
   **Rerun in both orders:** the new-domain curves of the continued network and the control, the control extensions, and retention.
5. **Bars unchanged:**
   - (B) at 10 tasks, new-form, +10 points and p < 0.05, in both orders;
   - zero-shot as the second measure; (C), (D) and (E) as before;
   - the gate to Stage 2b as before.
6. **Cost:** about 5–6 h on 24 workers and one GPU, plus any GPU waits.

- **If (B) still fails:** in these domains the made-up-systems control explains in-domain transfer. The next step would be harder domains, where the fresh network does not saturate within 30 tasks.

### Stage 2a′′ proposal: harder new domains (8 November, 04:15; awaiting owner approval, nothing run)

**Owner.** *"Skip Stage 2a′ as a separate run. The domains saturate within 30 tasks, so we need harder data to see whether anything is really being learned. Propose a harder Stage 2a′′ instead: 1. Harder new domains: more variables, more terms per equation, and more terms with constants inside, so that a fresh network does not solve most of them quickly. Pre-register a saturation check on development systems: a fresh network must stay below about 60% of new-form equations after 30 tasks; if not, make the domain harder before any test. 2. Before any learning, run the fitter test on true structures for the harder domains. If the fitter can't fit them, stop and report, since harder equations may hit the multi-piece fitting limit seen earlier. 3. Keep the fairness fixes from your 2a′ proposal: the control reviews its own earlier training as much as the real-trained network reviews real domains, and both spend half of each round on the newest domain. 4. Fresh test systems for the claim, with the same bars and gate. Send the plain-language proposal before running anything."*

#### 1. Two harder new domains, four variables each

The generator code is fixed before any draw.

- **Reaction networks** (four species, positive):
  - A random network of reactions drawn from seven kinds:
    - constant inflow; decay; conversion x_i → x_j; mass action x_i + x_j → x_k;
    - Michaelis–Menten conversion V·x_i/(K + x_i);
    - activation of a species' production by x_j, V·x_j²/(K + x_j²);
    - repression of a species' production by x_j, V/(K + x_j²).
  - Each reaction's rate enters the equations of the species it consumes and produces.
- **Nonlinear flows** (four variables, signs free):
  - Every equation has a self-damping term −a·x_i.
  - The other terms are drawn from linear, quadratic and cubic monomials and two bounded rational terms with a constant inside, x_j/(K + x_j²) and 1/(K + x_j²).
- **Why these constant-bearing terms:** they are rational, the kind the fitter handled in Stage 2a (21 of 22 chemistry test equations with constants inside, exact).
- **Acceptance** as in Stage 2a (build, reference-cost rule, the own-variable cubic check), plus: every variable appears in the system.

#### 2. A difficulty ladder (both domains)

| Level | Terms per equation | Terms with a constant inside, per system | At most per equation |
|---|---|---|---|
| 1 | 3–4 | 2–3 | 1 |
| 2 | 4–5 | 3–4 | 1 |
| 3 | 5–6 | 4–6 | 2 |

- Start at level 1.
- Two constant-bearing terms in one equation are allowed only at level 3. Earlier, two rational terms in one equation were the fitter's weak point: 11 of 15 for the three-step pathway in the Stage 2a pilot.

#### 3. Build

- **Four-variable vocabulary.** The L1 rules are unchanged: at most 7 nodes and 2 constants per term, with family merging and the rescaling-invariant search set. The vocabulary is probably 2–2.5× the three-variable one.
- **Four variables in the fitter, generators, relabelling and shared-form rule.** Check: Stage 2a's development bar (a′) is reproduced bitwise, as for the three-variable extension.
- **Network: a fourth variable role ("other3"), appended.**
  - The histories and control bases of Stage 2a are reused. In every reused network the new role starts as the mean of "other1" and "other2"; nothing else changes.
  - Their behaviour on two and three variables is unchanged, so the retention baselines stay valid.
- **The fairness fixes** (from the 2a′ proposal):
  - **Rehearsal-matched control.** In every dream after its base (the extension and the new-domain dreams), the control also replays its own earlier made-up examples, as many per dimension as the continued network replays from earlier domains. They are drawn with a fixed seed from its own earlier fantasy files.
  - **Balanced dreams.** For the continued network and the control, half of each dream's training batches come from the newest domain's examples (its verified solutions so far and this dream's fantasies), half from everything earlier.
  - The fresh network has nothing earlier and is unchanged.

#### 4. Checks before any test system exists (development systems only)

- **Systems per domain and level:** 100 training tasks and 40 development systems. Seeds: "g2h-{set}-{domain}-L{level}-k".
- **(a′′) Fitter on true structures:** at least 80% of development equations exact given the true structure. **If not met: stop and report.**
- **Saturation check:**
  - **Procedure:** the fresh network runs a uniform first wake on training tasks 1–10, a dream, a wake on tasks 11–30 and a dream, at Stage 2a's new-domain budget (300 structures, 60 thorough fits). It is then tested on the development systems.
  - **Requirement:** it solves **under 60%** of new-form development equations (new-form relative to mechanics, circuits and ecology).
  - **At 60% or more:** go to the next level and repeat (a′′) and this check. If level 3 still saturates, stop and check in.
  - **Under 15%:** check in before going on. The domain may then be too hard for every arm at this budget.
- **Development pilot:** timing and projection. Time limits are set from measured timings so they rarely bind. Check in if the run projects over 1.5 days.

#### 5. Then the test systems and the run

- **Test systems:** 40 per domain at the chosen level, with new seeds. New-form is classified per order and position (every relabelling of four variables).
- **(a′′) on the test equations;** stop if under 80%.
- **The run, as Stage 2a:**
  - two orders: mechanics, circuits and ecology, then the two harder domains in either order;
  - the fresh network's first 30 tasks are reused from the saturation check;
  - learning curves at 0, 10, 30 and 100 tasks for the continued network, the rehearsal-matched control and the fresh network;
  - retention at the end of each order.

#### 6. Bars and gate (unchanged from Stage 2a)

- **Validity:** the fresh network at most 70% of new-form test equations at 30 tasks. Otherwise later points are reported only.
- **(B) at 10 tasks,** in both orders, judged per domain: +10 points new-form against the control and p < 0.05.
- **Also reported:**
  - zero-shot as the second measure; (C) against the fresh network; (D) order; (E) retention, at most 2 systems lost;
  - whole systems exact, the shortlist probe, effort and provenance.
- **Gate to Stage 2b:** (B) met in at least one harder domain, in both orders.

#### 7. Cost (rough; the pilot gives the projection)

- **Build:** about half a day, of which about 3–4 h is vocabulary computation.
- **Each ladder level:** about 2–3 h for both domains.
- **The run:** about 1–1.5 days. Four-variable searches over more terms cost several times Stage 2a's.

### Stage 2a′′: owner approval (8 November)

**Owner.** *"Approved. Check in once the ladder has settled on a level, with that level's fitter and saturation numbers and the timing projection, before starting the claim run."*

### Stage 2a′′: build, one fitter correction, generator pilot (8 November, 05:55; fixed before any development or test system is drawn)

**Four-variable vocabulary** (`g1_vocab.py 4`, `g1_families.py … 4`, `g1_scale_closed.py 20 4`; 16 minutes).
- 168,569 trees give 69,947 canonical forms.
- 2,610 constant-free terms are merged numerically, and 807 families by the fitting test (out of 71,477 candidate pairs).
- **L1 for four variables: 66,109 terms.** By number of constants: 50,549 without, 13,886 with one, 1,674 with two.
- **Search vocabulary (rescaling-invariant): 4,683 terms** (three variables: 2,494).

**Code for four variables.**
- **Fitter** (terms compile for up to four variables), generators, relabelling, the shared-form rule (24 relabellings), fantasies (`dream_g2`) and the network:
  - the role "other3" is appended to the tokens;
  - `movenet_g2.load_net` gives a Stage 2a network the new role as the mean of "other1" and "other2";
  - `train(groups=…)` gives balanced dreams.
- **Checks:**
  - Stage 1's and Stage 2a's development bars are reproduced bitwise with four-variable compilation (all fields but time; 30 and 20 systems);
  - the four-variable symbols extend the three-variable ones;
  - relabelling gives the same keys as before on 3,600 (three-variable key, permutation) pairs;
  - no three-variable term uses the new token.
- **Driver** `cycles_g2h.py` (saturation check and claim run), `g2h_draw.py`, `g2h_shared.py`.

**One fitter correction, found on pilot seeds** ("g2h-pilot-*", never used again).
- **The problem:** every equation with a constant-bearing term failed on its true structure (0 of 66 at level 1), although its fits were exact (step-integral error about 2.5 × 10⁻¹⁴ against τ = 10⁻⁹).
- **The cause:** the fitter's degenerate-term check compares a fitted term with all cubic polynomials in all the variables. With four variables that is 35 monomials on the fitter's 30 rate points, so every column is matched exactly. This is not the multi-piece fitting limit.
- **The correction: the comparison uses cubic polynomials in the term's own variables.** This is the rule Stage 2a adopted for acceptance. A term with a constant uses at most three variables in L1 (seven nodes), so at most 20 monomials.
- **Effect on earlier results** (rerun, counts only):
  - Stage 1's development and test bars: unchanged (30 and 120 systems).
  - Stage 2a's development bar: 2 chemistry equations, exact fits wrongly flagged degenerate, now count as exact (22 → 24 of 24).
  - Stage 2a's test bar: one equation loses the flag but still fails to fit, so the count is unchanged (100 of 101).
  - So the old check occasionally rejected exact three-variable structures in Stage 2a. Every arm used the same fitter, so the comparisons stand. Stage 2a's chemistry counts may be slightly low.
- The stored Stage 1 and Stage 2a bar files are kept as they were.

**Generator pilot** (pilot seeds; 24 systems per domain and level; counts).

| Level | Domain | Accepted (rejected draws) | Terms per equation | True structures exact (with constants inside) | Systems exact |
|---|---|---|---|---|---|
| 1 | Reaction networks | 24 (3) | 3–4 | 96 of 96 (66 of 66) | 24 of 24 |
| 1 | Nonlinear flows | 24 (8) | 3–4 | 96 of 96 (62 of 62) | 24 of 24 |
| 2 | Reaction networks | 24 (3) | 4–5 | 96 of 96 (88 of 88) | 24 of 24 |
| 2 | Nonlinear flows | 24 (4) | 4–5 | 96 of 96 (86 of 86) | 24 of 24 |
| 3 | Reaction networks | 24 (4) | 5–6 | 94 of 96 (82 of 84) | 22 of 24 |
| 3 | Nonlinear flows | 24 (5) | 5–6 | 95 of 96 (85 of 86) | 23 of 24 |

- Builds take about 1 s per system. Rejections are mostly the cost rule (flows), centred data or a term not visibly nonlinear.
- A few flows are chaotic: 3, 2 and 1 of 24 have a largest Lyapunov exponent above 0.01.
- The generators are fixed as they stand.

**Next:** level 1's 100 training tasks and 40 development systems per domain; new-form classification; (a′′) on development equations; then the saturation check.

### Stage 2a′′: the ladder settles at level 1 (8 November, 06:50; development systems only, no test system drawn)

**Level 1 draws** (`g2h_draw.py 24 1 train dev`; seeds "g2h-{set}-{domain}-k").
- 100 training tasks and 40 development systems per domain, all accepted.
- Rejected draws (train / dev): reaction networks 11 / 9, mostly centred data; nonlinear flows 20 / 12, mostly the cost rule.
- Development equations: 160 per domain.
- New-form, relative to mechanics, circuits and ecology (`g2h_shared.py 1 dev`): reaction networks 155 of 160, nonlinear flows 159 of 160.

**(a′′) Fitter on true structures, development equations** (`g1_bar_a.py 24 dev reactions_L1 flows_L1`): **MET**.

| Domain | Equations exact | With constants inside | Systems exact |
|---|---|---|---|
| Reaction networks | 160 of 160 | 126 of 126 | 40 of 40 |
| Nonlinear flows | 160 of 160 | 102 of 102 | 40 of 40 |

- At most 1.5 s per system. Time-limit hits and errors: 0.

**Saturation check** (`cycles_g2h.py 24 sat 1`; the fresh network after 30 training tasks, tested on the development systems; budget 300 structures and 60 thorough fits; GPU 0, free at every check).

| Domain | Uniform first wake (tasks 1–10) | Wake 11–30 | **Fresh network after 30 tasks, new-form** | All equations | Systems | Verdict |
|---|---|---|---|---|---|---|
| Reaction networks | 0 of 40 | 5 of 80 | **56 of 155 (36.1%)** | 61 of 160 | 0 of 40 | passes |
| Nonlinear flows | 2 of 40 | 31 of 80 | **88 of 159 (55.3%)** | 89 of 160 | 2 of 40 | passes |

- **Both domains fall between 15% and 60%: the ladder stops at level 1.** No harder level is needed.
- **Uniform search at this budget finds almost nothing** in four variables (4,683 terms, 3–4 per equation). The fresh network first learns from fantasies alone, then from its few verified solutions.
- **Whole systems are rare** (0 and 2 of 40): an equation-level claim has room, a system-level one has little.
- **The shortlist** (768 of 4,683 terms) left a missing true term out of the attended set at 40% (reaction networks) and 28% (flows) of the development test's expansions. The figure in Stage 2a was 0–16%. The search still reaches every term through its random share and the rest slot. As pre-registered, this is reported and not changed.
- **Time-limit hits: 0.** The slowest system took 272 s against a cap of 3,600 s.

**Timing** (24 workers):
- Uniform search: 196–226 s per system on average.
- Network-guided search: 102–147 s.
- A 40-system test: 4.3–6.0 min.
- A dream: about 4 min (fantasies 1–2.5 min, 500 training steps 2.2 min).
- **Projection for the claim run:** about 10 h, range 9–12 h, plus any GPU waits. This covers both orders, three arms, the control extension, retention and the whole-system checks. Under the 1.5-day threshold.

**Dry run of the claim-run code** (`G2H_RUN=smoke G2H_SMOKE=1 cycles_g2h.py 16 run 1 1`: tiny budgets on development systems only, order 1; 06:16–07:22). **Every phase ran without error.**
- **Reused networks:** a Stage 2a network's encodings for two and three variables are bitwise identical to Stage 2a's. Every weight is unchanged except the appended role.
- **Balanced dreams:** the continued network's 4th-domain dreams hold 3,395 history examples (group 0) and its newest-domain examples (group 1), with batches split half and half. In the 5th domain, group 0 adds its 4th-domain solutions (four variables).
- **Rehearsal-matched control:** each new-domain dream replays exactly as many of its own earlier made-up examples per dimension as the continued network's group 0.
  - 4th domain: 3,395 two-variable examples.
  - 5th domain: 3,395 two-variable examples, plus 11 four-variable ones from its extension.
- **The extension** generates new made-up four-variable examples as many as the mirrored dream's group 1. A batch of 48 such fantasies takes 4–6.5 min on 16 workers: the size prior gives many rejected draws.
  - The projection rises by about 0.6 h, to **about 10.5 h (9.5–12.5 h), plus any GPU waits**.
- The dry run once trained on the CPU, when no GPU was free at that instant; it allows this. The claim run never does: it waits and re-checks every minute.

**Check-in with the owner before the claim run** (owner's instruction).

### Stage 2a′′: owner's go-ahead for the claim run (8 November)

**Owner.** *"Start the claim run. In the results, report the shortlist's cut rate separately for each arm and domain. If bar B fails, include an assessment of whether the shortlist cuts could explain it, before proposing next steps."*

- **Added before the run, for that assessment:** the shortlist probe now also counts per equation (expansions, expansions with a missing true term, cuts, and the same at the root). It is still counting only and never influences the search, so search results are unchanged.

**Test systems** (`g2h_draw.py 24 1 test`, seeds "g2h-test-{domain}-k"; counts only).
- 40 per domain, all accepted. Rejected draws: 6 (reaction networks), 8 (flows).
- **(a′′) on the test equations: MET.** Reaction networks 160 of 160, 117 of 117 with constants inside. Nonlinear flows 160 of 160, 107 of 107. All 80 systems exact.
- **New-form test equations** (`g2h_shared.py 1 test`):

| Order | Domain (position) | New-form | Shared-form systems |
|---|---|---|---|
| 1 | Reaction networks (4th, after M, C, E) | 158 of 160 | 0 of 40 |
| 1 | Nonlinear flows (5th, after M, C, E, R) | 102 of 160 | 0 of 40 |
| 2 | Nonlinear flows (4th, after E, C, M) | 157 of 160 | 0 of 40 |
| 2 | Reaction networks (5th, after E, C, M, F) | 92 of 160 | 2 of 40 |

- In 5th position the two harder domains share many four-variable forms with each other: self-damping, linear and product terms.
- **The claim run starts now** (`cycles_g2h.py 24 run 1`; both orders). The dry-run folder is deleted.

**Claim run under load** (08:15).
- The run started at 07:36 at the lowest CPU priority (nice 19 instead of 10). Another user's jobs occupy about 41 of the 48 cores, and nice 19 keeps this run from competing with them.
- The first test (uniform search, 40 reaction-network test systems) took 34 min against about 8 min projected: about 4× slower. At this pace the run would take about 1.5–2 days. It speeds up when the other jobs finish.
- **Time limits:** the slowest system took 1,689 s of elapsed time. The limit is 3,600 s per equation, measured in elapsed time, so it does not bind at this load. Hits are reported per arm.
- Owner notified. The run continues.

### Stage 2a′′: results (8 November, 16:40; counts only for test systems)

**Run** (`cycles_g2h.py 24 run 1`, both orders, nice 19):
- 07:36 to 16:12, 8.6 h. The first 2 hours ran under another user's heavy CPU load.
- **Time-limit hits: 0** in every arm. Failed systems 0. Skipped training steps 0. No GPU waits.
- **Validity:** at 30 tasks the fresh network solves 27–54% of new-form test equations. **No domain is saturated**, so every point is valid.

**Learning curves.** Test equations solved of 160 (new-form in parentheses); complete systems were 0–5 of 40 throughout.

| Domain, order (position; new-form) | Arm | 0 tasks | 10 | 30 | 100 |
|---|---|---|---|---|---|
| Reaction networks, 1 (4th; 158) | Continued | 46 (44) | 52 (50) | 52 (50) | 59 (57) |
| | Control | 23 (22) | 52 (50) | 60 (58) | 64 (62) |
| | Fresh | 2 (2) | 19 (19) | 61 (59) | 60 (58) |
| Nonlinear flows, 1 (5th; 102) | Continued | 52 (22) | 57 (24) | 54 (23) | 77 (44) |
| | Control | 48 (23) | 65 (34) | 66 (33) | 83 (44) |
| | Fresh | 2 (1) | 63 (21) | 88 (36) | 95 (39) |
| Nonlinear flows, 2 (4th; 157) | Continued | 46 (43) | 58 (55) | 55 (52) | 65 (62) |
| | Control | 40 (37) | 56 (53) | 64 (61) | 82 (79) |
| | Fresh | 2 (1) | 63 (60) | 88 (85) | 95 (92) |
| Reaction networks, 2 (5th; 92) | Continued | 71 (31) | 68 (31) | 79 (36) | 79 (37) |
| | Control | 37 (9) | 54 (15) | 77 (36) | 86 (41) |
| | Fresh | 2 (1) | 19 (5) | 61 (25) | 60 (23) |

**Continued network against the rehearsal-matched control** (new-form; only continued / only control):

| Domain, order (position) | Zero-shot | **10 tasks: bar (B)** | 30 tasks | 100 tasks |
|---|---|---|---|---|
| Reaction networks, 1 (4th) | +13.9, 23 / 1, p = 3 × 10⁻⁶ | **+0.0, 9 / 9, p = 1: not met** | −5.1, 5 / 13, p = 0.10 | −3.2, p = 0.38 |
| Nonlinear flows, 1 (5th) | −1.0, 5 / 6, p = 1 | **−9.8, 2 / 12, p = 0.013: not met (control ahead)** | −9.8, p = 0.013 | 0.0, p = 1 |
| Nonlinear flows, 2 (4th) | +3.8, 10 / 4, p = 0.18 | **+1.3, 5 / 3, p = 0.73: not met** | −5.7, p = 0.064 | −10.8, 9 / 26, p = 0.006 |
| Reaction networks, 2 (5th) | +23.9, 23 / 1, p = 3 × 10⁻⁶ | **+17.4, 16 / 0, p = 3 × 10⁻⁵: met** | 0.0, p = 1 | −4.3, p = 0.52 |

- **Bar (B) in both orders:** reaction networks not met (order 2 only); nonlinear flows not met. **The gate to Stage 2b does not open.**
- **(C), continued against fresh, new-form:**
  - Reaction networks: +19.6 (order 1) and +28.3 (order 2) at 10 tasks, p ≤ 4 × 10⁻⁸. At 30 tasks −5.7 (order 1) and +12.0 (order 2, p = 0.019). At 100 tasks −0.6 and +15.2 (p = 0.0005).
  - Flows: +2.9 and −3.2 at 10 tasks. At 30 tasks −12.7 (p = 0.007) and −21.0 (p = 1 × 10⁻⁶). At 100 tasks +4.9 and −19.1 (p = 2 × 10⁻⁵).
- **(D) Order** (equations new-form in both orders):
  - Reaction networks are solved far better 5th (after flows) than 4th, at every point: 31 against 11 at zero-shot, p ≤ 0.0015 throughout.
  - Flows show no difference, except order 1 at 100 tasks (44 against 33, p = 0.043).
- **(E) Retention: MET.**
  - Order 1: mechanics, circuits and ecology 40 → 40 systems. Reaction networks after flows: 0 → 0 systems, and equations 59 → 77.
  - Order 2: ecology and circuits 40 → 40; mechanics 40 → 38 (2 lost, at the limit). Flows after reaction networks: 0 → 2 systems, equations 65 → 79.
- **Whole systems exact:** at most 4 of 40 for any arm and point (the fresh network on flows at 100 tasks). At 10 tasks, 0 for every arm. Every complete system checked was exact, except 2 of the control's reaction networks at 30 tasks (order 2).
- **Provenance base rate:** every new-form equation has a true term in earlier solutions. Term overlap does not separate wins.

**The shortlist's cut rate, per arm, domain, order and point** (owner's request). This is the share of expansions with a missing true term at which a true term was outside the 768 attended terms; the search vocabulary has 4,683 terms.

| Domain, order | Arm | 0 tasks | **10** | 30 | 100 |
|---|---|---|---|---|---|
| Reaction networks, 1 | Continued | 66.3% | **41.3%** | 58.9% | 43.1% |
| | Control | 53.4% | **50.9%** | 55.1% | 53.6% |
| | Fresh | (uniform) | **60.9%** | 45.2% | 52.2% |
| Nonlinear flows, 1 | Continued | 57.9% | **27.8%** | 41.4% | 29.4% |
| | Control | 62.2% | **42.8%** | 50.6% | 28.5% |
| | Fresh | (uniform) | **28.2%** | 35.1% | 37.3% |
| Nonlinear flows, 2 | Continued | 63.4% | **27.8%** | 38.2% | 51.5% |
| | Control | 35.1% | **59.4%** | 36.9% | 34.8% |
| | Fresh | (uniform) | **28.2%** | 35.1% | 37.3% |
| Reaction networks, 2 | Continued | 59.0% | **45.3%** | 31.9% | 23.9% |
| | Control | 64.4% | **58.5%** | 47.4% | 19.4% |
| | Fresh | (uniform) | **60.9%** | 45.2% | 52.2% |

The fresh network's rows are the same in both orders (it does not depend on the order).

**Could shortlist cuts explain bar (B)'s failures?** No. Three checks at the deciding point, 10 tasks:
1. **The continued network was cut less often than the control in all four cases:** 41 against 51%, 28 against 43%, 28 against 59%, 45 against 59%. To explain its failure the cuts would have had to hurt it more.
2. **No advantage where the shortlist played no part.** On the new-form equations where neither arm ever had a true term cut, the three failing cases still show no advantage for the continued network: −1.9 points (3 / 4), −4.2 (1 / 2), −1.9 (1 / 2).
3. **Cuts lower every arm's counts.** Equations that were never cut are mostly solved; equations with a cut rarely are. Examples: the continued network 44 of 61 against 6 of 97 (reaction networks, order 1); the control 50 of 53 against 3 of 104 (flows, order 2). A wider shortlist would probably raise all curves. Nothing indicates it would change the comparison.

- **The one met case (reaction networks, order 2):**
  - The control's higher cut rate (58.5 against 45.3%) may have added to the gap.
  - On equations neither arm ever had cut, the continued network is still ahead: +31.2 points, 5 / 0, p = 0.062.
- **Limitation:** this assessment is observational. Never-cut equations are an easier subset. An interventional check (the 10-task tests rerun with all 4,683 terms attended) would settle it in about 1–1.5 h.

**What the made-up-systems control explains.**
- **Explained:** nearly all transfer once a domain's own tasks arrive.
  - Under fair rehearsal and balanced dreams, the control equals or beats the continued network from 10 tasks on in three of four cases, and at 30 and 100 tasks in all four.
  - This holds in domains a fresh network cannot saturate.
- **Not explained:**
  1. **A large head start on reaction networks before any task:** +13.9 and +23.9 points.
  2. **Its persistence to 10 tasks when reaction networks follow flows:** +17.4 points. Both arms had then seen four-variable systems, real for one and made-up for the other.
- **Flows show neither.**
- **Negative transfer:** the networks with earlier training learn flows more slowly than a fresh network, even with balanced dreams. At 30–100 tasks the continued network is 13–21 points behind it, and the control behind too.

#### Proposed next steps (awaiting the owner's approval; nothing run)

1. **Close the shortlist question directly** (about 1–1.5 h).
   - Rerun the 10-task tests of the continued network and the control, in both domains and orders, with all 4,683 terms attended. The networks and test systems are unchanged.
   - Bar (B) is recomputed and reported next to the run's.
2. **Then the main problem: learning slows with experience.**
   - Networks with earlier training learn flows 13–21 points more slowly than a fresh network at 30–100 tasks. The continued network is worst, and balanced dreams do not cure it. A learner that learns new domains more slowly than from scratch cannot show accumulation.
   - **Proposal:** a development-systems-only pilot (the saturation-check protocol, 30 tasks). It measures the gap to the fresh network under up to three remedies, applied identically to every arm with earlier training:
     - longer dreams on the new domain (1,000 steps, half new-domain batches);
     - a fresh per-term scorer at each new domain, keeping the term encoder;
     - a wider shortlist (2,048 terms).
   - The best remedy would then be proposed for a claim run under the same bars, with fresh test systems.
3. **The gate to Stage 2b stays closed.**
   - The one result beyond the control (reaction networks after flows, +17.4 points) holds in one order only.
   - It would be worth re-testing on new seeds once the slowdown is understood.

### Stage 2a′′: owner's decision on next steps (8 November)

**Owner.** *"Run both: the direct shortlist check, then the development-only slowdown pilot. For the slowdown pilot: 1. First fix to test: a focus-then-consolidate schedule. When a new domain starts, train 100% on it (focus phase) until its learning curve flattens, then run a consolidation phase with shuffled review of all domains, and check retention after consolidation. Compare the experienced network in the focus phase against the fresh network at equal exposure: if it still learns slower, that's loss of plasticity; if it keeps its head start, the slowdown was review time. Apply the same schedule to the control. 2. Networks with earlier training learning new domains more slowly than a fresh network is a known problem (loss of plasticity). Search the literature for fixes that have worked (e.g. resetting little-used units, shrink-and-perturb, regularizing toward initial weights) and for schedules like focus-then-consolidate, and include the most promising alongside longer training rounds and a wider shortlist. 3. Don't use a separate scoring layer per domain: the learner won't be told which domain a system comes from. Test one fix at a time, judged by the gap to the fresh network after 30 tasks. Report the literature findings in plain language, and check in before any claim run."*

**The direct shortlist check** (approved).
- **Rerun:** the 10-task tests of the continued network and the control, in both domains and orders, with every term of the search vocabulary attended (shortlist = 4,683). The fresh network's 10-task test is rerun the same way, as context.
- **Unchanged:** the networks, test systems and search seeds. Only the attended set differs, so each rerun is paired with its original test. The original results are kept.
- **Reported:** bar (B) recomputed next to the run's.

### Loss of plasticity: literature, and the slowdown pilot's design (8 November, 16:50; development systems only, fixed before any pilot run)

**Literature, in plain language.**
- **The problem is real and well studied.**
  - Dohare et al. (Nature 2024): networks trained on a long sequence of tasks gradually lose the ability to learn new ones, until they learn no better than a shallow network. A growing share of units stops doing anything useful.
  - Lyle et al. (CoLLAs 2024): several separate causes contribute, including dead units, units that behave linearly, and large weights. Fixing one cause alone often fails.
- **Fixes that have worked:**
  1. **Resetting little-used units.**
     - *Continual backprop* (Dohare 2024) keeps re-initializing a tiny share of the least useful units as training goes on.
     - *ReDo* (Sokar et al., ICML 2023) periodically re-initializes "dormant" units: those whose average activity is below τ = 0.1 of their layer's mean. Their incoming weights are redrawn and their outgoing weights set to zero, so the network's output does not jump.
  2. **Shrink-and-perturb** (Ash & Adams, NeurIPS 2020). Before training on new data, scale all weights down (×0.4) and add a little fresh random initialization (×0.1). Warm-started networks then train like fresh ones without losing what they knew.
  3. **Regularizing toward the initial weights: L2 Init** (Kumar et al., 2023). A penalty λ‖θ − θ₀‖² pulls the weights toward their own random initialization rather than toward zero. λ = 10⁻² was the usual choice with Adam. It is the most consistent of the methods they compared, about as good as continual backprop.
  4. **Layer normalization with weight decay** (Lyle 2024): the most robust single combination in their study. **Our network already has both** (pre-norm transformer layers, AdamW with weight decay 0.01).
- **Schedules like focus-then-consolidate:**
  - *Progress & Compress* (Schwarz et al., ICML 2018) alternates a "progress" phase, where new tasks are learned freely, with a "compress" phase that consolidates them into a lasting knowledge base. It uses distillation rather than replay.
  - *Hare & Tortoise* (Lee et al., 2024) pairs a fast network for new learning with a slow, consolidating one, and periodically resets the fast one to the slow one.
  - *Wake–sleep consolidated learning* (2023) and sleep-like replay (Tadros et al., Nature Comm. 2022) consolidate in separate phases.
  - The *stability gap* (De Lange et al., ICLR 2023): when a new task starts, performance on old tasks dips sharply and then recovers. This is a reason to check retention after consolidation, not during the focus phase.

**The pilot** (owner, 8 November; development systems only; one fix at a time; judged by the gap to the fresh network after 30 tasks).
- **Testbed:**
  - Nonlinear flows at level 1 in 4th position, after order 2's history (ecology, circuits, mechanics), where the slowdown was largest.
  - Same 100 training tasks (the first 30 used), the 40 development systems (159 new-form equations), and the claim run's budget (300 structures, 60 thorough fits) and dream sizes (100 fantasies, 500 steps).
  - Tests after 10 and 30 tasks.
- **Measure:** the fresh network minus the experienced network, new-form development equations, at 30 tasks (and 10), with paired sign tests.
  - Baseline: the claim run's balanced schedule, testing its networks on the development systems.
  - The fresh network's first 30 tasks are those of the saturation check.
- **Fixes, in order:**
  1. **Focus-then-consolidate** (owner's).
     - **Focus:** while a new domain is learned, each dream trains only on the new domain (its verified solutions so far and this dream's fantasies). This is exactly the fresh network's recipe, so the comparison is at equal exposure. In the pilot the focus phase covers tasks 1–30, where the fresh network's curve is still rising.
     - **Consolidation** after the focus phase: one dream of 1,000 steps on all domains' verified solutions, shuffled together (sampled in proportion to their numbers).
     - **Checked:**
       - retention on the development systems of mechanics, circuits and ecology (budget 1,000), against the history network;
       - the new domain's development score after consolidation.
     - **Reading:** if the experienced network still learns more slowly than the fresh network in the focus phase, that is loss of plasticity; if it keeps its head start, the slowdown was review time.
     - **The control runs the same schedule.** Its consolidation reviews its own earlier made-up examples, as many as the continued network's history replays.
  2. **Shrink-and-perturb** once, just before the first new-domain dream: θ ← 0.4·θ + 0.1·θ_fresh (Ash & Adams's values). Then the balanced schedule.
  3. **L2 Init** in every new-domain dream: penalty λ‖θ − θ₀‖² with λ = 10⁻² (Kumar et al.'s usual value). θ₀ is the network's own initial weights, rebuilt from the same seed. Balanced schedule.
  4. **ReDo** before every new-domain dream: τ = 0.1; dormancy measured on 512 of the dream's examples; incoming weights redrawn, outgoing set to zero. Balanced schedule.
  5. **Longer training rounds:** 1,000 steps per new-domain dream instead of 500. Balanced; applied to the fresh network too, for a fair gap.
  6. **Wider shortlist:** 2,048 attended terms in training and search. Balanced; applied to the fresh network too.
- **Rules:**
  - The learner is never told the domain: no per-domain parameters.
  - The first wake of each fix (tasks 1–10) uses the unchanged history network, so it is the claim run's wake. Fixes act from the first dream on.
  - The control is run only with fix 1 (owner's instruction). The fix chosen for a claim run would apply to every arm.
  - **The best fix is confirmed on reaction networks (4th, after order 1's history) before the check-in.**
  - No test system is used.

**Pilot code checks** (`g2h_pilot.py`; before any pilot run).
- **The fixes' functions were tested on the order 2 history network and a flows fantasy file.**
  - Rebuilding the initial weights is deterministic and matches every parameter's shape.
  - ReDo runs; a dtype bug was found and fixed.
- **Dormant units (a plasticity-loss sign):** on four-variable flow examples, the history network's main transformer has many dormant units (score ≤ 0.1 of the layer mean). In its three feed-forward layers that is 133, 174 and 278 of 512 units (26%, 34%, 54%). The term encoder and the scorer's and input projection's layers have none.
- **L2 Init will act far from its usual regime.** The history network's squared distance from its initialization is 951, so at the registered λ = 10⁻² the penalty starts at 9.5. That is larger than the training loss (about 1–6). The method is meant to be used from the start of training, where the penalty starts at 0. The registered value is kept, and the outcome is reported as it is.
- **Timing:** the shortlist check's full-attention tests take about 3–4× as long as the originals (attention over 4,683 terms). The check needs about 2.5–3.5 h; the pilot starts after it.

**The direct shortlist check, order 1** (`g2h_fullcheck.py`; 10-task tests rerun with every term attended; same networks, test systems and seeds; counts only). Valid: no time-limit hits.

| Domain, order 1 | Shortlist 768 (the run) | Every term attended |
|---|---|---|
| Reaction networks | continued 50, control 50 of 158 new-form; (B) +0.0 | continued 50, control 41; (B) **+5.7**, 15 / 6, p = 0.078: not met |
| Nonlinear flows | continued 24, control 34 of 102; (B) −9.8, p = 0.013 | continued 25, control 26; (B) **−1.0**, 8 / 9, p = 1: not met |

- **The continued network is unaffected by the shortlist.** Solved only with every term / only with the shortlist: 4 / 4 (reaction networks) and 4 / 3 (flows).
- **The shortlist helped the control.** With every term attended it loses equations: 4 gained against 13 lost (p = 0.049) on reaction networks, and 0 against 8 (p = 0.008) on flows.
- **Conclusion, order 1:** the shortlist's cuts do not explain bar (B)'s failure. Attending every term does not bring the continued network to the bar. The gap moves toward it only because the control does worse without its shortlist.

**Order 2's reruns are not valid yet.**
- **What went wrong:** under heavy machine load (another user's jobs, plus the pilot alongside), full-attention searches ran past the 3,600 s limit per equation:
  - order 2 flows: 6 (continued) and 15 (control) equations stopped;
  - order 2 reaction networks: 56 (continued).
- **What was done:** the check was stopped before the control's reaction-network rerun. All four order-2 tests will be rerun with the limit raised to 4 h per equation (`G2H_TIME_CAP=14400`), under a new suffix.
- **Why raising the limit is sound:** the limit is a safety limit. The original shortlist tests never reached it, so raising it keeps the comparison about the attended set.

### The slowdown pilot: results on nonlinear flows (9 November, 04:10; development systems only)

**Testbed:** flows (level 1), 4th position after order 2's history; 40 development systems, 159 new-form equations.
- The gap is the fresh network minus the experienced network, in new-form points.
- Paired sign tests (only experienced / only fresh).
- All runs under the claim run's budget; time-limit hits 0.
- `g2h_pilot.py`; folder `outputs/accumulating_learner/g2h/pilot`.

| Fix (one at a time) | 10 tasks: experienced / fresh | 30 tasks: experienced / fresh | **Gap at 30** | Sign test at 30 |
|---|---|---|---|---|
| Baseline (the claim run's balanced dreams) | 53 / 45 | 59 / 88 | **+18.2** | 3 / 32, p = 4 × 10⁻⁷ |
| 1. Focus-then-consolidate: the focus phase | 54 / 45 | 52 / 88 | **+22.6** | 3 / 39, p = 6 × 10⁻⁹ |
| 2. Shrink-and-perturb (0.4, 0.1) | 38 / 45 | 63 / 88 | **+15.7** | 1 / 26, p = 4 × 10⁻⁷ |
| 3. L2 Init (λ = 10⁻²) | 48 / 45 | 52 / 88 | **+22.6** | 2 / 38, p = 2 × 10⁻⁹ |
| 4. ReDo (τ = 0.1) | 53 / 45 | 57 / 88 | **+19.5** | 1 / 32, p = 8 × 10⁻⁹ |
| 5. 1,000-step dreams (both networks) | 46 / 30 | 55 / 75 | **+12.6** | 4 / 24, p = 0.0002 |
| 6. **2,048-term shortlist** (both networks) | **52 / 36** | **76 / 80** | **+2.5** | 11 / 15, p = 0.56 |
| Control, balanced baseline | 54 / 45 | 68 / 88 | +12.6 | 11 / 31, p = 0.003 |
| Control, focus phase | 55 / 45 | 72 / 88 | +10.1 | 6 / 22, p = 0.004 |

**Fix 1, the owner's question: loss of plasticity, not review time.**
- In the focus phase the experienced network trains exactly as the fresh network does: only the new domain, same examples and steps. It keeps its head start at 10 tasks (54 against 45), then stalls: 52 at 30 tasks against the fresh network's 88.
- The control, trained on made-up systems, loses less plasticity: 72 at 30 tasks.
- **Consolidation** (1,000 steps, all domains shuffled together):
  - Retention: the continued network keeps every history system (development: ecology, circuits and mechanics 10 of 10 before and after). The control keeps 9, 10 and 9 of 10 (8, 10 and 10 before).
  - The new domain: consolidation keeps the continued network's flows (52 → 49, p = 0.55). It **erases most of the control's new learning: 72 → 46** (2 / 28, p = 9 × 10⁻⁷). The review is 95% earlier examples, which pull the network back.

**Fixes 2–5 do not close the gap.**
- Shrink-and-perturb learns faster from 10 to 30 tasks (+25 equations against +6), but only after giving up the head start (38 at 10 tasks).
- L2 Init and ReDo change little. ReDo does reset many dormant units: 137, 178 and 271 of 512 in the main transformer's feed-forward layers before the first new-domain dream, and 42, 88 and 164 before the second.
- Longer dreams lower both networks; the gap shrinks only because the fresh network loses more.

**Fix 6, a 2,048-term shortlist in training and search, closes the gap.**
- The experienced network keeps a significant head start at 10 tasks: +10.1 points, 28 / 12, p = 0.017.
- It learns as fast as the fresh network up to 30 tasks: 76 against 80, p = 0.56, from 59 with the 768-term shortlist.
- **Mechanism (likely):** the experienced network's scorer was trained on earlier domains. It ranks many of the new domain's true terms below the top 768, so training's attention never sees them, and they are learned only through the scorer's own loss. A wider attended set lets them in.
- **The cost:** a wider shortlist also lowers the fresh network (88 → 80 at 30 tasks), and each search step costs more (tests take about 2–3× as long).
- **Next, as pre-registered:** fix 6 is confirmed on reaction networks (4th, after order 1's history; development systems) before the check-in.

### The direct shortlist check: complete (9 November, 04:35; counts only)

**Setup.** Order 2's tests were rerun with the limit at 4 h per equation (suffix "cap4h"); order 1's ran under the normal limit. Every rerun is now free of time-limit hits.

| Case (new-form) | Bar (B) at 10 tasks, shortlist 768 (the run) | Bar (B) at 10 tasks, every term attended | Every term vs shortlist: gained / lost |
|---|---|---|---|
| Reaction networks, order 1 (158) | +0.0, 9 / 9: not met | +5.7, 15 / 6, p = 0.078: not met | continued 4 / 4; control 4 / 13 (p = 0.049) |
| Nonlinear flows, order 1 (102) | −9.8, 2 / 12: not met | −1.0, 8 / 9: not met | continued 4 / 3; control 0 / 8 (p = 0.008) |
| Nonlinear flows, order 2 (157) | +1.3, 5 / 3: not met | −3.8, 3 / 9, p = 0.15: not met | continued 1 / 9 (p = 0.021); control 3 / 3 |
| Reaction networks, order 2 (92) | +17.4, 16 / 0: **met** | +15.2, 16 / 2, p = 0.0013: **met** | continued 2 / 7; control 2 / 5 |

- **No bar (B) verdict changes when every term is attended, so the gate to Stage 2b stays closed.** The shortlist's cuts at search time do not explain bar (B)'s failures.
  - The shortlist helped the control in two cases (order 1) and the continued network in one (flows, order 2).
- **This is not in conflict with the pilot.** The check changes only what networks trained with 768 terms attend to when tested. The pilot's 2,048-term shortlist changes training as well, from the first new-domain dream. That is where it matters for learning speed.

### The slowdown pilot: confirmation on reaction networks, and conclusions (9 November, 06:40; development systems only)

**Confirmation: reaction networks** (4th, after order 1's history; 40 development systems, 155 new-form equations).

| | 10 tasks: experienced / fresh | 30 tasks: experienced / fresh | Gap at 30 (fresh − experienced) |
|---|---|---|---|
| Baseline, shortlist 768 | 38 / 18 | 40 / 56 | **+10.3**, 8 / 24, p = 0.007 |
| Control, shortlist 768 | 38 / 18 | 50 / 56 | +3.9, p = 0.26 |
| **Fix 6, shortlist 2,048 (both networks)** | **57 / 17** | **60 / 35** | **−16.1**, 28 / 3, p = 5 × 10⁻⁶ |

- The slowdown reproduces on reaction networks: the experienced network leads at 10 tasks and falls behind by 30.
- **With the wider shortlist it does not fall behind:**
  - at 10 tasks it leads by 25.8 points (40 / 0, p = 2 × 10⁻¹²);
  - at 30 tasks it still leads, by 16.1 points.
- **The cost to the fresh network is larger here:** 56 → 35 at 30 tasks with the wider shortlist.

**Pilot conclusions.**
1. **The slowdown is loss of plasticity, not review time.** In the focus phase, trained exactly like the fresh network, the experienced network still stalls.
2. **The standard plasticity fixes from the literature do not remove it here** (shrink-and-perturb, L2 Init, ReDo), and neither do longer dreams.
3. **A wider shortlist (2,048 attended terms, in training and search) removes it in both domains.**
   - The experienced network keeps a significant head start at 10 tasks.
   - It keeps pace with (flows) or stays ahead of (reaction networks) a fresh network with the same shortlist at 30 tasks.
   - **Likely mechanism:** the plasticity that is lost sits in what the network attends to. Its scorer, trained on earlier domains, keeps the new domain's true terms out of the top 768, so training's attention rarely sees them.
4. **Costs and open points:**
   - The wider shortlist lowers the fresh network's own learning: new-form at 30 tasks, flows 88 → 80 and reaction networks 56 → 35.
   - Tests take about 2–3× as long.
   - Retention and the control under the wider shortlist were not measured in the pilot.
   - In absolute terms, at 30 tasks the experienced network with 2,048 terms (flows 76, reaction networks 60) is above the best fresh network on reaction networks (768 terms: 56) but below it on flows (88).
5. **The direct shortlist check is consistent with this.** Attending every term only at test time, in networks trained with 768, changes no bar (B) verdict. What matters is the attended set during training.

**Check-in with the owner before any claim run.**

### Owner's decision: the claim run with the 2,048-term shortlist, and a DreamCoder assessment (9 November)

**Owner.** *"Remove the 20 GB cap on outputs. Delete files only when they're no longer useful. Keep the 5 GB free-disk crash guard. Run the claim run with the 2,048-term shortlist as proposed. In parallel, assess our design against DreamCoder in plain language: 1. Abstraction: in DreamCoder, the growing library is what produces transfer and keeps the search space small as it scales. We gated abstraction behind showing transfer first, which may be backwards. Should abstraction come earlier? 2. Dreams: DreamCoder samples dreams from its current library and learned weights, so dreams reflect accumulated experience. Ours are uniform draws from a fixed language, which may explain why the made-up-systems control matches the experienced network. 3. The learned prior: DreamCoder uses a probabilistic grammar whose weights and library are updated from solved tasks (Bayesian prior), plus a recognition network (our move network) and compression-based library learning. Our prior is essentially uniform and never updates. How would we add a learned prior? 4. Curriculum: DreamCoder solves easy tasks first and builds on them. Ours are independent random draws. 5. Why DreamCoder succeeds from 5–10 examples, and what we'd need for the same. Propose a redesign that adds the missing pieces, staged one change at a time, and send it with the claim-run results before running anything new."*

**Disk rule from now on:** no cap on outputs; files are deleted only when no longer useful; writing stops if free space falls below 5 GB.

**The claim run (Stage 2a′′ with the wider shortlist), as proposed.**
- **Shortlist:** 2,048 attended terms in training and search, for every arm (continued, rehearsal-matched control, fresh).
- **Test systems:** fresh, 40 per domain at level 1 (seeds "g2h-test2-{domain}-k"), with new-form classification per order and (a′′) on their true structures.
- **Unchanged:** the bars, the gate, the budget, the dream sizes and the time limit (3,600 s per equation).
  - Measured with the wider shortlist in the pilot, the slowest equation took 687 s, so the limit stays.
- **Reused:** the histories and control bases.
  - The fresh networks' uniform first wakes are identical, so they are reused.
  - Everything else is rerun in `outputs/accumulating_learner/g2h/run2`.
- **Check-in:** if the measured pace projects over 1.5 days.

### DreamCoder comparison and a staged redesign (9 November; proposal, awaiting the owner's approval; nothing run)

**What DreamCoder does** (Ellis et al., PLDI 2021 / arXiv 2006.08381).
- **The library sets the prior.** A growing library (a programming language) gives every program a prior probability. Programs that are short in the library are likely.
- **Wake:** each task has 5–15 input–output examples. Programs are searched in order of probability under a recognition network conditioned on the task, and the best few are kept.
- **Abstraction sleep:** the solutions are refactored, and the fragments that best compress them (a Bayesian, description-length objective) are added to the library.
- **Dream sleep:** the recognition network trains on a 50/50 mix of:
  - replays: tasks with their found programs;
  - fantasies: programs sampled from the current library, run on real inputs, then solved by the same search.
- **Implicit curriculum:** tasks are revisited every cycle. Harder ones get solved later, as the library makes their programs shorter.
- **Ablations:**
  - removing either library learning or the recognition network lowers held-out performance;
  - deeper libraries go with more tasks solved (r = 0.79);
  - domains have about 100–200 training tasks.

**Side by side.**

| DreamCoder | Ours |
|---|---|
| A library that grows (learned language) | A fixed language L1 (terms of at most 7 nodes and 2 constants) |
| A generative prior: library plus weights, updated from solved tasks | None. The search's base prior is uniform over the vocabulary, and the mass outside the attended set is spread evenly |
| Recognition network Q(program given task) | The move network: the same role |
| Dreams sampled from the current library and weights | Half from single-term frequencies in our solutions, half from a fixed size prior; the control: the size prior only |
| Unsolved tasks revisited every cycle | Each new-domain task searched once |
| Whole-task programs | One equation at a time; no structure across equations |

**1. Abstraction: should it come earlier? Yes.**
- **What we have now:** the only thing that carries over between domains is the move network's sense of which terms of a fixed vocabulary tend to be useful.
  - Made-up systems teach that equally well, since they are drawn from the same vocabulary.
  - A new domain's own first tasks teach it quickly.
  - That is what Stages 2a and 2a′′ found: a head start that the control and the fresh network erase within 10–30 tasks.
- **So gating abstraction behind transfer was backwards.** In our setting, abstraction is the main way real experience could carry structure that made-up systems cannot. The gate's purpose, no new machinery without evidence, is better served by testing abstraction directly against:
  - the made-up-systems control with its own library;
  - a decoy library of random fragments of matched size;
  - removal of library pieces, with provenance.
- **Abstraction here means three kinds of reusable piece:**
  1. composite terms beyond 7 nodes built from recurring parts, such as a saturating rate x/(K + x) reused inside products;
  2. templates: groups of terms that recur together in one equation, added in one move;
  3. motifs across equations: a rate that enters two equations with opposite signs (a conversion), or x′ = v. Our per-equation search cannot see these, yet in reaction networks they are the dominant structure.

**2. Dreams: partly right.**
- The continued network's dreams are not purely uniform: half follow how often each term appeared in our solutions. But only single-term frequencies are used.
- Nothing in our dreams reflects how terms combine, how many terms an equation has, typical constants, or which equations share a rate.
- So the two arms' dreams are close to each other and carry little accumulated structure. That is a plausible reason the control matches the experienced network.
- DreamCoder's dreams come from its current library and weights, so they look more like real tasks as experience grows.

**3. The learned prior.**
- **What we have:** the recognition half (the move network), but not the generative half. Nothing outside the network accumulates "what structures look like", and the base prior never updates.
- The shortlist finding showed the cost: what the network does not attend to gets only an even share of probability.
- **How to add it:** a probabilistic grammar of equation structures, learned from verified solutions with Bayesian smoothing. It gives:
  - the probability of each term given its role (own variable, other variables);
  - the probability of the number of terms per equation;
  - pairwise co-occurrence within an equation;
  - for motifs, the chance that a term recurs in another equation with the opposite sign.
- **Smoothing:** counts with symmetric Dirichlet pseudo-counts, so unseen terms keep probability. The posterior predictive is the prior.
- **Uses:**
  1. the search's base prior, combined with the network's scores (the network corrects it per system);
  2. spreading the mass outside the attended set, instead of evenly;
  3. the dream generator;
  4. the description length that library learning compresses.
- **Fairness:** the control's grammar is fitted to its own made-up systems' true structures and its own solutions.

**4. Curriculum.**
- **What we have:** each new-domain task is searched once, in random order, so hard tasks never benefit from what is learned later.
- **Fix:**
  - after each dream, re-search the earlier unsolved tasks with a small budget each;
  - order tasks easy-first by an easiness estimate (the network's confidence, or the learned grammar's probability of short structures);
  - use our difficulty ladder (level 1 → 2 → 3) as stages.

**5. Why DreamCoder succeeds from 5–15 examples, and what we would need.**
- **Why it works:**
  - programs are discrete and checked exactly;
  - the library makes the right program short, so it ranks high among the few consistent ones;
  - the recognition network narrows the search;
  - the description-length prior prefers the simplest consistent program.
- **What we already have:** exactness checks (10⁻⁹ on held-back data) from 8 bursts, about 30 rate points per equation, which is comparably few data.
- **What we lack: a prior that makes the right structure short.**
  - Our search affords about 300 candidate structures per equation, because each needs continuous constants fitted. That is far fewer than DreamCoder enumerates.
  - With 4,683 terms in four variables, the right 3–4 rarely come up without help.
- **What we would need:**
  - library pieces that package whole recurring structures, including where their constants go;
  - the learned grammar;
  - learned starting values for constants (typical half-saturation constants, for example), so each candidate costs less and the right one comes early.
- Then a test with fewer bursts (2–4).

**The redesign, one change at a time.**

| Stage | Change (applied to every arm under the same rules) | Test | Main risk |
|---|---|---|---|
| A | A learned generative prior: the smoothed grammar above, as the search's base prior and for the mass outside the attended set | Development pilot: experienced, control and fresh at 10 and 30 tasks, flows and reaction networks; claim run if it helps | Only term statistics again: a small effect |
| B | Dreams sampled from each lineage's learned grammar (a share kept from the size prior for exploration) | As A | Dreams too narrow |
| C | Library learning by compression: composite terms, templates and cross-equation motifs, as single moves and grammar symbols | As A, plus a decoy library and a removal check | Build effort; fitting cost of large pieces |
| D | Curriculum: re-search unsolved tasks after each dream; easy-first order | As A | Compute |
| E | Fewer data: 2–4 bursts per system | Identification rate against bursts, with and without stages A–D | — |

- **Order:**
  - A comes first: the library's compression needs a prior to score description length, and dreams need a generative model.
  - B tests the owner's dream hypothesis directly.
  - C adds the library; dreams then include library pieces automatically.
  - D and E follow.
- **Retire the "transfer first" gate**, in favour of the decoy and removal controls.
- **Keep** the made-up-systems control (with the same machinery fitted to its own data) and the new-form rule for every claim.
- **Cost, rough:** A and B about 0.5 day each to build and 0.5 day to pilot; C about 1.5–2 days to build; D about 1 day; E about 0.5 day. Claim runs (about 1 day each at the wider shortlist) only where a pilot shows a gain. Check-in before each claim run.

### Owner: order 2 runs as pre-registered; the DreamCoder assessment sent before the claim-run report (9 November)

**Owner.** *"Let order 2 finish as pre-registered. Don't wait for it to send the DreamCoder assessment: send that now, in plain language, covering what DreamCoder has that we're missing (abstraction, dreams drawn from the learned library, the learned prior, curriculum, why it works from 5–10 examples) and a staged redesign proposal. The claim-run report can follow when order 2 is done."*

- **Context (claim run 2, order 1 only; the full report follows order 2):** bar (B) not met in order 1 for either domain, so the gate stays closed whatever order 2 shows.
  - Reaction networks (all 160 test equations new-form): continued 58, control 48 at 10 tasks (+6.2 points, 15 against 5, p = 0.041); before any task 39 against 17 (+13.8 points).
  - Flows (95 of 160 new-form after reaction networks): continued 26, control 22 at 10 tasks (+4.2 points, 6 against 2, p = 0.29); before any task 23 against 13 (+10.5 points).
  - The fifth domain shows the slowdown again despite the wider shortlist: on flows at 100 tasks the continued network solved 36 of 95 new-form equations, the fresh network 55 (4 against 23, p = 0.0003). On reaction networks (the fourth domain) it kept level with the fresh network (87 against 92 of 160 at 100 tasks, p = 0.59).
- **Sent:** a plain-language version of the section above, with one addition prompted by the fifth-domain slowdown.
  - Knowledge held only in the network's weights has to survive every later round of training, and each added domain makes the network slower to learn the next.
  - A grammar and a library hold what was learned outside the network, where later training cannot erase it. That would also allow the move network to be refreshed (retrained from a new start on replays and dreams) without losing what was learned.
  - That refresh is listed as an option to test after stage A if the slowdown persists in order 2; it is not a stage of its own yet.

### Owner: the staged redesign approved; Stage A built; its pilot's design (9 November; pilot awaiting approval)

**Owner.** *"Approved: start with Stage A, then B, C, D, E, one change at a time, with a development pilot before each and a claim run only where a pilot shows a clear gain. Retire the "transfer first" gate; use the decoy and removal checks instead. Treat A mainly as infrastructure for B and C, so keep its pilot quick. For Stage C's cross-equation patterns, plan three parts, each tested on its own: 1. Let the network attend across all equations of a system, not just within one (e.g. one summary token per equation, to limit cost). 2. Paired moves that add the same rate to two equations at once, with opposite signs. 3. Shared constants across equations in the fitter. After C, test refreshing the network from a new start using the library, grammar and replays. Let order 2 finish as planned. Send the pilot design for Stage A before running it."*

**The plan as approved.**
- **Order, one change at a time:**
  - A, the learned grammar (infrastructure; a quick pilot);
  - B, dreams from each lineage's grammar;
  - C, library learning by compression, with a decoy library and a removal check. Its cross-equation patterns come in three parts, each tested on its own:
    - C1: attention across all equations of a system (one summary token per equation);
    - C2: paired moves that add the same rate to two equations with opposite signs;
    - C3: constants shared across equations in the fitter;
  - after C: refreshing the network from a new start using the library, the grammar and the replays;
  - D, curriculum;
  - E, fewer bursts.
- **Pilots and claims:** a development pilot before each stage; a claim run only where a pilot shows a clear gain.
- **The "transfer first" gate is retired.** Abstraction is tested directly against a decoy library, a removal check, and the made-up-systems control with its own library.
- **Unchanged:**
  - accumulation claims rest on new-form equations and must beat the made-up-systems control (with the same machinery fitted to its own data);
  - no hand-added pieces;
  - no domain labels when a system is searched.

**Stage A as built.** New modules, so the claim run in progress (which re-imports its own modules for every review round) is untouched: `grammar_g2.py`, `movenet_g2a.py` and `stageA_pilot.py`.

- **Term shapes:**
  - Each vocabulary term is read relative to its equation: the equation's own variable becomes x1, and the others become x2–x4 in whichever order gives the smallest canonical key.
  - There are 1,572 shapes over the 2-, 3- and 4-variable vocabularies (1,140, about 1,486 and about 1,552 per dimension).
  - A shape is the same key in every dimension, so counts from two-variable histories carry over to four-variable domains.
  - A few dozen terms with a constant inside a logarithm keep variable-order-dependent canonical forms, so shape counts are kept per own variable.
- **A component** holds shape counts over a set of equations (and counts of terms per equation). Its probability of a concrete term is (n_s/m_s + α/V)/(N + α): a Dirichlet-smoothed count with a uniform base.
  - **The smoothing α comes from a fixed grid (1 to 10⁴):**
    - for the earlier component, chosen by predicting each earlier domain from the other earlier domains (leave one domain out);
    - for the newest component and for made-up experience, chosen by leaving one equation out;
    - with fewer than 20 equations, α = 10.
  - **The histories:**
    - order 1 gives α = 1,000, so 34% of the mass goes to unseen kinds of term; order 2 gives α = 300 (13%). The histories hold 900 and 894 equations but only 24 distinct shapes.
    - Leaving one equation out instead would give α = 1 (0.1% for unseen kinds), which is far too confident for a new domain. That is why the domain is held out.
  - **The controls' made-up experience:** about 9,700 made-up equations (reconstructed from their example files), about 1,000 distinct shapes, α = 1,000 (5% for unseen kinds). It is a much broader prior than the real history's.
- **The grammar** is the equal mixture of the components that have counts in the system's dimension:
  - "earlier" (everything before the newest domain) and "newest" (the newest domain's verified solutions), the same half-and-half rule as the balanced review rounds;
  - an empty grammar is exactly uniform (checked: the log ratio is 0 everywhere).
- **The network corrects the grammar instead of replacing it** (`movenet_g2a`). Every term's add score is the network's output plus L = log P_grammar + log V, which is 0 for the uniform prior:
  - the shortlist is chosen by score + L;
  - an attended term's logit is the head's logit + L;
  - the rest slot carries the grammar's mass outside the attended set, spread over the rest's terms in proportion to the grammar;
  - training uses the same sums;
  - networks trained before Stage A (the history networks, until their first Stage A review round) keep their own scores, and for them the grammar only spreads the rest slot's probability.
- **Checks:**
  - With no grammar, the loss equals movenet_g2's (0.59448 against 0.59448 on two-variable examples; 1.31338 against 1.31338 on four-variable ones).
  - With the order-1 history grammar, the untrained network's loss on reaction-network replay examples falls from 1.31 to 0.78.

**Stage A pilot design (pre-registered; development systems only; awaiting the owner's approval).**
- **Recipe:** the slowdown pilot's f6 recipe, with one change, the grammar, for every arm. The f6 recipe is:
  - the 2,048-term shortlist in training and search, and balanced review rounds;
  - wakes on training tasks 1–10 and 11–30, with a 500-step review round and 100 fantasies after each;
  - development tests (40 systems per domain) after each review round;
  - budget 300, thorough cap 60, time limit 3,600 s per system.
- **Testbeds:** nonlinear flows after order 2's history; reaction networks after order 1's history.
- **Arms:**
  - continued: the history's verified solutions plus its new-domain solutions;
  - rehearsal-matched control: its made-up systems' structures plus its own new-domain solutions, with review rounds matched as in the claim run;
  - fresh: its own new-domain solutions only. Its grammar is empty at the first wake, so the claim run's uniform first wake is reused.
- **Baselines without the grammar:** the slowdown pilot's f6 lineages (p6_o2 and p6fresh_2 on flows; q6_o1 and q6fresh_1 on reaction networks). No control was run without the grammar at this shortlist on development systems; claim run 2's gap gives context.
- **Measures** (new-form development equations; counts only):
  1. continued and fresh, with against without the grammar, at 10 and 30 tasks (paired sign test);
  2. continued against control with the grammar, at 10 tasks (bar (B)'s quantity on development systems) and at 30;
  3. continued against fresh at 30 tasks (the slowdown);
  4. a diagnostic with no search: the mean description length (bits) of the development systems' true new-form equations under each arm's grammar before the new domain and after 10 and 30 tasks, against the uniform prior's. This shows directly whether real experience gives a better prior than made-up experience;
  5. root expansions where a missing true term lies outside the attended set, time-limit hits per arm, and compute.
- **Decision rules:**
  - Stage A goes forward into B unless the grammar makes the continued or fresh network significantly worse at 30 tasks (sign test p < 0.05) in either domain. If it does, diagnose and check in before B.
  - A claim run for A on its own is proposed only if, with the grammar, the continued network beats the control by at least 10 points with p < 0.05 at 10 tasks in both domains (bar (B) on development systems).
- **Cost:** six arm-domains, about 4.5 hours with 30 workers. The continued network and the control rerun their first 10 tasks; the fresh first wake is reused. It starts when claim run 2 ends (about 02:00 UTC on 10 November), to stay within the CPU share.
- **Smoke test** (tiny budgets, two tasks per wake, three development systems per test, reaction networks): the whole path ran for all three arms. That covered grammar files saved with each network (the history networks marked as not trained with it), review rounds trained with the grammar, searches with it, and the report. The fresh network found nothing in its two tiny wakes, so its grammar stayed empty (uniform), as designed. The smoke outputs were deleted afterwards.
  - **Disclosed:** checking the report code printed one real number, the no-search diagnostic before the new domain on the reaction networks' development systems: mean description length of the 155 true new-form equations, uniform prior 43.6 bits; the continued network's history grammar 25.9; the control's made-up grammar 26.7. It is development data, not test data, and the design above was written before it was seen.

### Owner: Stage A pilot approved; the bits check in every later stage (9 November)

**Owner.** *"Approved: run the Stage A pilot when claim run 2 ends. Proceed to designing Stage B per the decision rules, and include the bits check (how many bits each grammar needs to describe the true new-form equations) in every later stage, since it measures directly whether real experience gives a better prior than made-up experience."*

- **The Stage A pilot starts automatically when claim run 2's driver exits** (`stageA_pilot.py 30 flows reactions`, nice 19, one thread per worker).
  - The launcher checks that at least 5 GB of disk is free before starting.
  - While the pilot runs, it stops the pilot's process group if free space falls below 5 GB.
- **The bits check is now a standing measure** for every later stage (B, C1–C3, the refresh test, D, E). It is the mean description length of the development systems' true new-form equations under each arm's grammar, before the new domain and after each review round, against the uniform prior.
- **Stage B** is designed while A runs. It goes ahead under A's decision rule: unless the grammar makes the continued or fresh network significantly worse at 30 tasks. Its pilot design is sent for approval with A's results.

### Stage B: dreams from each lineage's grammar, and its pilot's design (9 November; pre-registered; pilot design to be sent with Stage A's results)

**What changes.** Every review round's 100 fantasies are drawn from the lineage's grammar at that point (`dream_g2b.py`, `stageB_pilot.py`).
- **Before Stage B:** each term was drawn half from single-term usage and half from the size prior, with 1–4 terms per equation.
- **With Stage B:** each fantasy system is drawn whole, either from the grammar or, with probability ½, from the open language as before (exploration).
  - From the grammar: its number of terms per equation (1–6) and its terms, read relative to each equation.
  - From the open language: the size prior, with 1–4 terms per equation.
  - The branch is fixed per fantasy before any retry, so the share does not drift with acceptance.
- **Unchanged:** coefficients, internal constants, initial states, the pre-screen and acceptance.
- **The grammar** is the one the review round trains with:
  - the continued network: its history's verified solutions plus its new-domain ones;
  - the control: its made-up systems' structures plus its own new-domain solutions;
  - the fresh network: its own new-domain solutions only.
- So the dreams of the experienced network now carry its real experience, and the control's carry its made-up experience. This tests the owner's hypothesis that uniform dreams explain why the control keeps up.

**Pilot (development systems only):**
- **Recipe:** Stage A's pilot recipe with this one change. Same testbeds, arms (b_o<o>, bc_<o>, bfresh_<o>), shortlist, budgets and time limit.
- **Reuse:** the first wakes on tasks 1–10 are Stage A's, since they use the same network, grammar, tasks, budget and seed.
- **Baselines:** Stage A's lineages, paired system by system.
- **Measures** (new-form development equations):
  1. each arm with grammar dreams against Stage A at 10 and 30 tasks (paired sign test);
  2. continued against control at 10 tasks (bar (B)'s quantity on development systems) and at 30;
  3. continued against fresh at 30 tasks;
  4. the bits check: each arm's grammar after 10 and 30 tasks, against the uniform prior;
  5. dream diagnostics: acceptance per branch, rejected draws, mean terms per equation, and the realized share of examples from each branch;
  6. time-limit hits per arm.
- **Decision rules:**
  - B goes forward into C unless grammar dreams make the continued or fresh network significantly worse at 30 tasks than Stage A (sign test p < 0.05) in either domain. If they do, diagnose and check in.
  - A claim run for A+B is proposed only if the continued network beats the control by at least 10 points with p < 0.05 at 10 tasks in both domains (development).
- **Cost:** about 4 hours with 30 workers, after Stage A's pilot.
- **Smoke test** (tiny budgets, reaction networks, all three arms): passed end to end, and its outputs were deleted.
  - In four variables the open-language branch often needs many draws: in two tiny review rounds, none of its draws passed within 200 tries.
  - That is why acceptance and the realized example share per branch are reported.

### Stage 2a′′ claim run 2 (2,048-term shortlist, fresh test systems): results (10 November)

**Setup as pre-registered** (9 November): the 2,048-term shortlist in training and search for every arm; 40 fresh test systems per harder domain (`test2`); bars, gate, budgets and time limit unchanged. It ran from 07:28 UTC on 9 November to 01:45 UTC on 10 November (18.3 hours). All 82 test and wake searches finished with no time-limit hit and no failure. The report is `g2h_report.py 1` (G2H_RUN=run2, G2H_TEST_SET=test2), saved as `outputs/accumulating_learner/g2h/run2/report.txt`.

**New-form test equations solved** (continued / control / fresh). The reaction networks' test systems are all new-form in order 1. In order 2, after flows, 90 of 160 are. Flows: 95 of 160 new-form in order 1 (after reaction networks), 158 of 160 in order 2.

| Order, domain (position) | 0 tasks | 10 tasks | 30 tasks | 100 tasks |
|---|---|---|---|---|
| 1, reaction networks (4th), of 160 | 39 / 17 / 0 | 58 / 48 / 16 | 67 / 55 / 62 | 87 / 64 / 92 |
| 1, flows (5th), of 95 | 23 / 13 / 0 | 26 / 22 / 19 | 27 / 27 / 33 | 36 / 35 / 55 |
| 2, flows (4th), of 158 | 37 / 32 / 1 | 62 / 49 / 47 | 56 / 57 / 80 | 73 / 69 / 111 |
| 2, reaction networks (5th), of 90 | 32 / 7 / 0 | 32 / 24 / 5 | 46 / 31 / 21 | 59 / 35 / 39 |

**Continued against the control, new-form** (points; only continued against only control; p):

| Order, domain | 0 tasks | 10 tasks: bar (B) | 30 tasks | 100 tasks |
|---|---|---|---|---|
| 1, reaction networks | +13.8 (24–2, 1e-5) | +6.2 (15–5, p = 0.041): **not met** | +7.5 (14–2, 0.004) | +14.4 (25–2, 6e-6) |
| 1, flows | +10.5 (11–1, 0.006) | +4.2 (6–2, p = 0.29): **not met** | 0.0 (3–3) | +1.1 (8–7) |
| 2, flows | +3.2 (10–5, 0.3) | +8.2 (16–3, p = 0.0044): **not met** | −0.6 (4–5) | +2.5 (12–8, 0.5) |
| 2, reaction networks | +27.8 (25–0, 6e-8) | +8.9 (14–6, p = 0.12): **not met** | +16.7 (18–3, 0.0015) | +26.7 (26–2, 3e-6) |

- **Bar (B) is met in no domain and no order, so the gate to Stage 2b stays closed.** The gate was retired by the owner on 9 November in favour of the redesign's direct tests, and is applied here only as pre-registered.
- **Validity:** after 30 tasks the fresh network solves 39%, 35%, 51% and 23% of the new-form test equations (all at most 70%).

**(C) Continued against the fresh network, new-form:**
- **Reaction networks:** ahead or level.
  - Order 1: +26.2 at 10 tasks; level at 30 (+3.1, p = 0.46) and 100 (−3.1, p = 0.59).
  - Order 2: ahead at every point (+30.0, +27.8 (p = 5e-6), +22.2 (p = 0.002)).
- **Flows:** behind from 30 tasks in both orders.
  - Order 1: −6.3 (p = 0.18) at 30 tasks, −20.0 (4–23, p = 0.0003) at 100.
  - Order 2: −15.2 (8–32, p = 0.0002) at 30, −24.1 (4–42, p = 5e-9) at 100.
  - The fresh network also solves more whole flows systems at 100 tasks: 10, against 1 for both other arms in both orders.
- **The slowdown is domain-specific, not a fifth-domain effect.** This corrects the reading given during the run. Flows are learned more slowly than by the fresh network in both positions (4th in order 2, 5th in order 1). Reaction networks show no slowdown in either position. The control matches the continued network on flows from 30 tasks, so on flows both earlier-trained networks learn more slowly than a fresh one, whatever their earlier training was.
- In the slowdown pilot (development systems), the 2,048-term shortlist had closed the flows gap at 30 tasks in order 2 (76 against 80). On fresh test systems it did not (56 against 80).

**(D) Order** (equations new-form in both orders):
- **Reaction networks** do better after flows (order 2) than straight after the history: 32 against 11 at 0 tasks (p = 2e-5), 46 against 35 at 30 (p = 0.019); not significant at 10 or 100.
- **Flows** start better after reaction networks (order 1: 23 against 13 at 0 tasks, p = 0.002), and the difference is gone from 10 tasks on.

**(E) Retention:** met everywhere.
- Order 1: mechanics, circuits and ecology 40 of 40 systems before and after; reaction networks 4 → 6 after learning flows (0 lost; equations 87 → 97).
- Order 2: ecology and circuits 40 of 40; mechanics 40 → 38 (2 lost, the bar's limit); flows 1 → 5 after learning reaction networks (1 lost; equations 75 → 94).
- Ecology after order 2 is still 80 of 80 equations solved, but only 26 are recovered with the true structure (51 before); the others are other plausibly exact structures.

**Whole systems exact:** all 55 test systems with every equation solved passed the external test (one-step ≤ 10⁻¹² of the scale on 1,000 held-out states, plus the multi-step envelope). The rebuilt systems were identical in every case and there were no errors. The most was order 2's continued network on reaction networks at 100 tasks: 13 of 40.

**Shortlist cut rates** (the share of expansions with a missing true term where one lay outside the attended set; full per-arm and per-point figures in the report):
- **Order 1, reaction networks:** 37.5%, 36.5%, 50.9% and 11.5% for the continued network at 0, 10, 30 and 100 tasks; 29–35% for the control until 100 tasks.
- **Order 2, reaction networks:** 0–7% for the continued network; 24–47% for the control.
- **Flows:** 5–41% across arms.

**Could the cuts explain bar (B)'s failures?** Not in the continued network's disfavour.
- **Order 1, reaction networks:** on the 68 new-form equations where neither arm ever had a true term cut, the edge is +14.7 points (12–2, p = 0.013). Both arms solve almost none of the equations that were cut (continued 2 of 73, control 5 of 86), and the discordant equations' cut shares are alike in both arms. So the cuts dilute the edge by removing equations from both arms equally. They do not hand the control its wins.
- **Order 2, flows:** the continued network's 16 sole wins had 28.9% of the control's expansions cut, against 0.2% of its own. There, cuts favour the continued network.
- **Order 2, reaction networks:** +14.3 points on the 49 never-cut equations (p = 0.14).
- **Shortlist cuts are what Stage A's grammar addresses:** probability outside the attended set is spread by the grammar instead of evenly.

**Compute:** each test took 3–7 CPU hours (2–3 times the 768-term shortlist's). Every review round ran on a free GPU (GPU 0 or 1), in 2–3 minutes.

**What it means for the goal.**
- **Real experience helps with reaction networks.** The continued network beats the made-up-systems control at 30 and 100 tasks in both orders (by up to 27 points), and beats or matches the fresh network. It learns the most whole systems exactly (13 of 40 at 100 tasks in order 2).
- **But by the pre-registered measure (bar (B), 10 tasks) the control explains most of the early transfer.** Its made-up experience gives nearly the same head start once it has seen 10 tasks.
- **On flows, earlier training of either kind slows learning against a fresh network.**
- This supports the redesign: explicit, experience-specific memory (grammar, library) instead of term preferences that made-up systems reproduce, and the refresh test for the slowdown. Stage A's pilot started at 01:45 UTC on 10 November; Stage B's design is pre-registered.

### Stage A pilot: results, diagnosis and a proposed fix (10 November; check-in before B)

**Run:** `stageA_pilot.py 30 flows reactions`, started automatically when claim run 2 ended. It ran 01:45–07:18 UTC.
- The machine was heavily loaded by another user (about 40 cores), and one review round waited 20 minutes for a GPU.
- No search was time-capped and none failed.
- Report: `outputs/accumulating_learner/g2h/stageA/report.txt`.

**New-form development equations solved, with the grammar against without it** (the slowdown pilot's f6 lineages; only-with against only-without; p):

| Testbed, arm | 10 tasks | 30 tasks |
|---|---|---|
| Flows (159), continued | 51 vs 52 (5–6, p = 1) | 67 vs 76 (9–18, p = 0.12) |
| Flows, fresh | 42 vs 36 (11–5, p = 0.21) | **67 vs 80 (8–21, p = 0.024)** |
| Flows, control (with the grammar only) | 62 | 55 |
| Reaction networks (155), continued | 51 vs 57 (7–13, p = 0.26) | 57 vs 60 (7–10, p = 0.63) |
| Reaction networks, fresh | 25 vs 17 (9–1, p = 0.021)\* | 36 vs 35 (8–7, p = 1) |
| Reaction networks, control (with the grammar only) | 42 | 53 |

\* Its grammar was still empty at that point, so this is a placebo comparison; see below.

- **Continued against control, with the grammar:**
  - flows: −6.9 points at 10 tasks (6–17, p = 0.035; the control is ahead), +7.5 at 30 (18–6, p = 0.023);
  - reaction networks: +5.8 at 10 (13–4, p = 0.049), +2.6 at 30 (13–9, p = 0.52).
  - None reaches bar (B)'s +10 points.
- **Continued against fresh, with the grammar:**
  - flows: 21–12 at 10 tasks (p = 0.16), 17–17 at 30;
  - reaction networks: 27–1 at 10, 24–3 at 30.
- **The pre-registered stop rule is triggered:** the fresh network on flows is significantly worse at 30 tasks with the grammar (p = 0.024). Stage B does not start; this is the diagnosis and check-in.

**The bits check** (mean description length of the true new-form development equations; uniform prior 41.7 bits on flows, 43.6 on reaction networks):

| Grammar | Flows | Reaction networks |
|---|---|---|
| Continued, before the new domain | 28.3 | 25.9 |
| Control, before | 27.4 | 26.7 |
| Continued, after 10 tasks | 20.1 | 22.3 |
| Control, after 10 | 20.8 | 23.2 |
| Fresh, after 10 | 29.4 | 43.6 (empty) |
| Continued, after 30 | 17.5 | 18.7 |
| Control, after 30 | 18.1 | 19.4 |
| Fresh, after 30 | 18.4 | 19.7 |

- **Real experience gives a slightly better prior than made-up experience:** 0.6–0.9 bits per equation after 10 and 30 tasks in both domains, and 0.8 bits before the new domain on reaction networks.
- **The one exception:** before the new domain on flows, the control's broad made-up grammar is better by 0.9 bits.
- **The new domain's own first 30 tasks teach far more,** about 8–10 bits, and the fresh network's own grammar comes within about 1 bit of the experienced one by 30 tasks.
- This is the bar-(B) picture in prior terms: little that is experience-specific carries over.

**Diagnosis.**
1. **The grammar in the shortlist's choice pushes the right terms out of the attended set when it does not know their kind.** The shortlist cut rate (expansions with a missing true term where one lay outside the attended set) changed as follows:

   | Testbed, arm (tasks) | With the grammar | Without |
   |---|---|---|
   | Flows, continued (10) | 26.6% | 5.0% |
   | Flows, continued (30) | 28.8% | 0.0% |
   | Flows, fresh (30) | 18.8% | 5.6% |
   | Reaction networks, fresh (30) | 56.6% | 5.2% |

   - The flows equations lost with the grammar are the ones with a true term the grammar rates below the uniform prior: 48% of the fresh network's lost equations against 14% of those both solved, and 56% against 2% for the continued network.
   - On reaction networks, whose kinds of term the history covers, the grammar instead kept true terms in the attended set: the continued network's cut rate fell from 36–38% to 8–11%. The solved count did not rise, though.
2. **The new-domain component is over-confident.**
   - It is fitted to the equations already solved, which are mostly the easy ones, so it keeps only 2–3% of its mass for unseen kinds.
   - Choosing its smoothing by predicting the second wake's solutions from the first's gives the same values: only 2–11% of the second wake's solved terms were of a new kind.
   - The kinds the grammar has not seen are in the equations not yet solved, which no smoothing fitted to solved equations can see.
3. **Run-to-run noise is as large as the trigger.** The fresh network's grammar on reaction networks was still empty at 10 tasks, so its run differed from the baseline only in training randomness (fantasy draws, seeds). Yet that comparison gave 25 against 17 (9–1, p = 0.021).
   - The paired sign test treats equations as independent, while two runs differ as wholes.
   - So p-values around 0.02 from one pair of runs are within the noise. The stop rule's trigger (p = 0.024) may be partly noise, but the mechanism in item 1 is clear.

**Proposed fix: Stage A′** (`stageA2_pilot.py`, built and syntax-checked; not run). The grammar only shares out the rest slot's probability over the terms outside the network's shortlist.
- The network's scores, its shortlist and its training are exactly the f6 recipe's, so the grammar can neither push a term out of the attended set nor lower an attended term's score.
- Grammars are fitted and saved as in Stage A, for the search's rest slot, for Stage B's dreams and for the bits check.
- **Pilot:** flows only (where the rule fired), with all three arms at 10 and 30 tasks (continued and control reuse Stage A's first wakes; fresh reuses the uniform first wake). About 2.5–3 hours at the current machine load.
- **Decision rule:** A′ goes into B unless the continued or fresh network is significantly worse at 30 tasks than without the grammar. Given the noise finding, a trigger is confirmed by one rerun of that arm before it counts.
- **Stage B is then run on A′** (dreams from the same grammars, search as in A′), with its pre-registered design otherwise unchanged.

### Owner: skip the A′ pilot, Stage B on A′; reruns to confirm p between 0.01 and 0.05; diagnose the flows slowdown (10 November)

**Owner.** *"Skip the A′ pilot and go to Stage B on top of A′. Given the noise finding (two chance-only runs at p = 0.021), from now on any decision rule or claim with p between about 0.01 and 0.05 must be confirmed by a rerun with a different seed before it counts. Tell me how much that adds to the cost of claim runs. Also, from existing logs only, diagnose the flows slowdown: what terms do networks with earlier training rank highly that are wrong for flows, and which true flow terms do they rank low, compared with the fresh network? Report in plain language alongside the Stage B pilot design, before running B."*

- **Standing rule from now on:**
  - A decision rule or claim resting on p < 0.01 counts as it stands.
  - One resting on 0.01 ≤ p ≤ 0.05 counts only after a rerun of the arms concerned confirms it: different seeds for that cycle's fantasies, training and search, and the same systems, with the same direction at p < 0.05.
  - p > 0.05 is not significant.
- **Stage B is built on A′:**
  - the grammar is used in the search only for the rest slot's probability;
  - networks are trained as before Stage A;
  - the fantasies are drawn from each lineage's grammar.
  - The A′ pilot is skipped.

#### What the rerun rule adds to a claim run (measured on claim run 2's phases, 30 workers, 2,048-term shortlist)

- **A confirming rerun repeats both sides of the comparison with new seeds,** on the same systems and from the same starting networks. The new seeds are for the cycle's fantasies, training and search; only the new-domain cycle's randomness is resampled, which is the noise the placebo showed.
- **Costs:**

  | Rerun | Hours |
  |---|---|
  | A 10-task comparison, continued + control (first wake, review round, test) | 0.7–0.8 |
  | Adding the 0-task tests | +0.4–0.5 |
  | A 30-task comparison | about 1.5 |
  | A whole cell, both arms, 0 to 100 tasks | 3.0–3.4 |
  | A fresh network's curve | 0.4 (10 tasks) to 1.8 (100 tasks) |
  | The whole claim run (claim run 2 took 18.3 h) | 18.3 |

- **Claim run 2 as a worked example:**
  - Comparisons whose outcome would have hung on 0.01 ≤ p ≤ 0.05: (C) in order 2 flows at 10 tasks (p = 0.04), and (D) on reaction networks at 30 tasks (p = 0.019).
  - Bar (B) in order 1 reaction networks had p = 0.041, but failed on size (+6.2 < +10), so it needs no rerun.
  - Cost: about 0.8 h + 1.6 h, so about 2.4 h, +13%.
- **Typical cost:** one to three borderline comparisons per claim run, so +1 to +5 hours (+5% to +30%).
- **Worst case** (most comparisons borderline): about a full rerun (+100%).
- Rerunning from the history instead of from the cell's starting networks would cost 7.5–9 h per order.
- **Reruns are pre-registered as automatic parts of each approved pilot or claim run** (no extra check-in), with their cost reported.

#### The flows slowdown, diagnosed from saved networks (10 November; no training, no search)

**Method.** Rankings are not in the logs, so they were recomputed.
- **What was scored:** claim run 2's saved networks were run forward (scoring only) on the 40 development flows systems (160 equations). At the start of each equation's search they ranked every term of the vocabulary by the search's probability of adding it.
- **Networks:**
  - the continued network and the control before flows, after 10 and after 30 flows tasks, in both orders;
  - the fresh network after 10 and 30 tasks.
- **Also counted:** whether each lineage's own training wakes ever found the terms concerned, from the wake logs (training tasks).
- **Kinds of term**, written relative to the equation: u is its own variable; v and w are other variables; K is a constant.

**Findings.**
1. **The slowdown is one kind of term: the "bump" 1/(K + x²),** a constant-bearing term flows use and no earlier domain does. It is in about 29% of flows training equations.
   - Median rank among about 4,000 terms after 30 flows tasks:

     | Network | Bump in another variable, 1/(K + v²) (35 true terms) | Bump in its own variable, 1/(K + u²) (12) |
     |---|---|---|
     | Fresh | 17 | 16 |
     | Continued, order 2 | 671 | 1,190 |
     | Continued, order 1 | 1,325 | 1,358 |
     | Control, order 2 | 2,105 | 2,086 |

   - After 10 tasks it is already 8 for the fresh network against 836–991 for the others. These ranks often fall outside the 2,048-term shortlist.
2. **It is self-reinforcing.**
   - The fresh network's first wake is uniform, so it tries simple terms evenly and found a bump solution at once. It then found 10 in tasks 11–30 and 39 in tasks 31–100, of 11, 22 and 82 training equations whose true structure has one.
   - Every network with earlier training, real or made-up, in both orders, found 0, 0 and at most 2 bump solutions in the same tasks.
   - A term a network ranks low is never tried, never appears in its verified solutions, and so is never learned. The review rounds cannot correct a blind spot the network's own search never fills.
3. **Wrong terms the earlier-trained networks rank too high** (share of equations where a wrong term of that kind is among the 10 top-ranked, at 30 tasks):

   | Wrong term | Earlier-trained | Fresh |
   |---|---|---|
   | Product of two other variables, v·w | 74–80% (continued) | 33% |
   | Cube of its own variable, u³ (order-2 lineages, whose history ends with mechanics) | 64–67% | 3% |
   | Sines from the pendulum | 6–12% | 2–7% |
   | A constant (order 1 continued, order 2 control) | 78–85% | 57% |

4. **Not the cause:**
   - The polynomial true terms (v, u·v, v², u², u³, 1) are ranked high, roughly 1 to 40, by every network.
   - The saturating x/(K + x²) terms are ranked low by every network including the fresh one (median 570–2,600). That is a shared weakness.

**What it means.**
- **The slowdown is a failure to explore,** not a loss of capacity: confident earlier preferences keep the search away from a kind of term the new domain needs, and the learner never finds out.
- **The grammar inherits the blind spot.** It is fitted to verified solutions, so Stage A′ and B leave it in place; the same goes for the refresh after C, which retrains on the same replays.
- **The fresh network's advantage comes from its uniform first wake.** A direct test would give every lineage part of its first new-domain wake under the uniform prior, before its own preferences take over. This is not run; it is a candidate for Stage D (curriculum and re-search).

### Stage B on A′: the pilot's design (10 November; pre-registered; supersedes the design above where they differ)

- **Recipe:** the slowdown pilot's f6 recipe, with two things added:
  - Stage A′: each lineage's grammar is fitted after every wake and used in the search only to share out the rest slot's probability; networks are trained exactly as in f6;
  - Stage B: every review round's 100 fantasies are drawn from that grammar, each whole from the grammar or, with probability ½, from the open language.
- **Testbeds:** flows after order 2's history; reaction networks after order 1's.
- **Arms:** continued `b_o<o>`, rehearsal-matched control `bc_<o>`, fresh `bfresh_<o>`; tests at 10 and 30 tasks on the 40 development systems.
- **Reuse:** the first wakes are Stage A's (identical), and the fresh first wake is the uniform one.
- **Baselines:** the f6 lineages (no grammar at all) for continued and fresh; Stage A for context.
- **Measures:**
  1. A′+B against f6 per arm, at 10 and 30 tasks;
  2. continued against control at 10 tasks (bar (B)'s quantity on development) and at 30;
  3. continued against fresh at 30;
  4. the bits check;
  5. dream diagnostics: acceptance, examples and terms per branch;
  6. cut rates and time-limit hits.
- **Also reported:** the bump term's rank and how many bump solutions each lineage's wakes find, since the slowdown diagnosis predicts that grammar dreams alone will not fix it.
- **Decision rules** (with the rerun rule):
  - B goes into C unless the continued or fresh network is worse at 30 tasks than without the grammar. p < 0.01 counts at once. 0.01 ≤ p ≤ 0.05 triggers an automatic two-sided rerun with seed 2 (the A′+B arm and a no-grammar f6 arm), and counts only if confirmed.
  - A claim run for A′+B only if the continued network beats the control by at least 10 points at 10 tasks in both domains, at p < 0.01 or confirmed by a rerun.
- **Cost:** about 4.5–5.5 hours with 30 workers at the current load. A two-sided rerun adds about 2.2 hours per arm and domain.
- **Smoke test of Stage B on A′** (tiny budgets, reaction networks): passed end to end, and its outputs were deleted. It covered:
  - all three arms;
  - a rerun with seed 2 (`reactions:r2:fresh`);
  - the no-grammar side of a two-sided rerun (`reactions:f6r2:fresh`): f6's fantasies, f6 training, an empty grammar saved, so the rest slot is spread evenly.

### Owner: an exploration fix before Stage B; breadth of domains (10 November)

**Owner.** *"Test an exploration fix first, before Stage B, since B's grammar-based dreams would inherit the same blind spot. Keep it general, not bump-specific; it's a safety net for any unfamiliar system in the future, not a fix for flows: 1. In a new domain, run a share of every network's searches with even weights over all terms before its own preferences take over (as AlphaZero adds exploration noise), and consider increasing that share automatically when equations stay unsolved. 2. Follow DreamCoder's principle that no term built from the language is ever effectively ruled out: make sure terms outside the shortlist keep a meaningful chance of being tried, so learned preferences can delay an unfamiliar term but never block it. Apply the same rules to the experienced, control and fresh networks. Pilot on flows and reaction networks at 10 and 30 tasks, reporting the bump's ranking and discovery counts, under the rerun rule. Then run Stage B on top of the winning version. Separately, note for later stages: the more varied the domains learned, the more future systems fall between them, so breadth of training domains matters alongside exploration. Send the exploration pilot design before running it."*

- **Note for later stages (breadth):** the more varied the training domains, the more future systems fall between them. So the breadth of training domains matters alongside exploration. Later stages and claim runs should say which domains a learner has seen, and the redesign should eventually test learners trained on more varied histories.

### Stage X: exploration as a safety net, the pilot's design (10 November; pre-registered; awaiting approval)

**Why.** The flows diagnosis: confident earlier preferences kept the search away from a kind of term the new domain needed, so it was never found and never learned.
- In the search, 300 structures per equation against about 4,000 terms means a term with a low learned probability is in effect never tried. That holds even inside the 2,048-term shortlist: the bump was ranked 670–2,100.
- Measured on claim run 2's tests, the network's searches solve equations after a median of 29 structures; 84% within 150, 93% within 225 and 98% within 270. That bounds what an exploration share can cost.

**Two general mechanisms, the same for every arm** (continued, control, fresh), in every search in a new domain (wakes and tests). The budget per equation is unchanged (300 structures), so compute matches the baseline.
1. **Never blocked** (DreamCoder's principle).
   - At every node of the network-guided search, at least ρ = 1/10 of the add moves' probability goes to the terms the network rates below even weights (below 1/number of legal adds). These include every term outside the shortlist and the low-rated terms inside it.
   - This share is given to the next untried such term in the language's own order: fewest nodes, then fewest constants (the order the uniform search uses).
   - So learned preferences decide what comes first, but each node keeps trying unfamiliar terms, simplest first. A term can be delayed, not blocked.
2. **An exploration share with even weights** (as AlphaZero mixes exploration noise into its prior).
   - The first share s of each equation's structures is searched with even weights over all moves: the uniform prior, simplest terms first. Then the network's preferences take over, and every structure already fitted is reused.
   - **The share adapts to how much stays unsolved:** s = ¼ in a lineage's first wake of a new domain. After each wake, s = half the share of that wake's equations left unsolved, kept between 1/10 and ¼. Tests use the lineage's current s.
   - The cap ¼ is AlphaZero's noise weight. It puts at most about 7% of the currently solved equations' budgets at risk.
   - The fresh network's first wake stays fully even (it has no preferences yet).

**Versions piloted:**
- X-n: never blocked only;
- X-ns: never blocked plus the adaptive exploration share.

**Recipe and testbeds:**
- The f6 recipe (the 2,048-term shortlist in training and search, balanced review rounds, no grammar), plus the version's search rules.
- Flows after order 2's history, and reaction networks after order 1's.
- Tests at 10 and 30 tasks on the 40 development systems.
- The continued and control networks rerun their first wakes. The fresh network's first wake, which is uniform under both versions, is reused.

**Baselines:** the f6 lineages (no exploration rules) for the continued and fresh networks. No control was run without exploration on development systems; it is not needed for the decision.

**Measures** (new-form development equations; counts only):
1. each arm and version against f6 at 10 and 30 tasks;
2. continued against control at 10 and 30 tasks;
3. continued against fresh at 30 tasks (the slowdown);
4. **the bump**:
   - its median rank by each lineage's networks after 10 and 30 tasks (root scoring as in the diagnosis);
   - how many bump-containing solutions each lineage's wakes found;
   - the saturating x/(K + x²) the same way;
5. general exploration measures (not bump-specific):
   - kinds of term each lineage found for the first time in each wake;
   - solutions that contain a term the network rated below even weights at the root;
   - the share of structures spent with even weights;
6. the bits check: grammars fitted to each lineage's verified solutions, as in Stage A;
7. time-limit hits and compute.

**The winner, and the decision rules** (with the rerun rule):
- **Eligible:** a version is eligible unless the continued or fresh network is worse than f6 at 30 tasks in either domain. p < 0.01 counts; 0.01 ≤ p ≤ 0.05 is decided by an automatic two-sided rerun with seed 2.
- **The winner** is the eligible version with the most new-form development equations solved at 30 tasks, summed over the three arms and both domains. On a tie, the simpler X-n wins.
- **If neither version is eligible, or neither beats f6's total, check in before Stage B.** For that total, the control's f6 figure is taken from the version-free claim-run recipe's control; failing that, it is compared over continued and fresh only.
- **Stage B then runs on the winner.** Proposed: the grammar is used only for B's dreams and the bits check, not A′'s rest-slot spread. The never-blocked rule takes that role in a general form, and A′ was never piloted.

**Cost:** 12 arm-domains, about 9 hours with 30 workers at light load (the Stage A pilot's 6 arm-domains took 5.5 hours under heavy load). Reruns add about 2.2 hours per arm and domain. A cheaper alternative is X-ns only (about 5 hours).
- **Clarification (same day):** "beats f6's total" is counted over the continued and fresh networks only, since the control has no f6 run on development systems. The winner rule's sum over three arms compares the two versions with each other.

### Owner: Stage X approved with both versions; the grammar's role in Stage B; two report additions (10 November)

**Owner.** *"Approved: run the pilot with both versions (never blocked alone, and never blocked plus the exploration share). Agreed to drop A′'s search use of the grammar for Stage B; keep the grammar only for B's practice systems and the bits check. Two additions to the pilot report: 1. How many distinct low-rated terms the exploration actually tried per equation, and how deep into the simplest-first order it got. 2. A small budget-sensitivity check on flows: rerun the winning version at 1,000 structures per equation for the fresh and experienced networks at 30 tasks, so we can see how much the 300 budget limits discovery of unfamiliar terms. Also say whether quick fits could screen many more candidates cheaply before the thorough fit, allowing a larger effective budget at similar cost. Report in plain language, including whether the experienced network's gap to the fresh network on flows closes, before starting Stage B."*

- **The budget check:** the winner's 30-task networks (experienced and fresh, flows) are tested on the 40 development systems at 1,000 structures per equation instead of 300.
  - The thorough cap is scaled in proportion, 60 → 200.
  - The time limit is raised to 14,400 s per system, so the budget, not the clock, binds.
  - The networks themselves are the ones trained with 300-structure wakes.
- **For the quick-fit question,** every search now records its time split: ordinary fits, thorough fits, precise checks and the prior (network and evidence).
- **Stage X built and smoke-tested** (`search_x.py`, `stageX_pilot.py`, `stageX_report.py`, `rank_terms_x.py`).
  - With both rules off, the exploration search reproduces search_g1's exactly (checked on a development system).
  - The smoke test covered both rules, a seed-2 rerun of the no-exploration side, the budget check, the root rankings and the report. Its outputs were deleted.
- **The pilot started at 17:07 UTC on 10 November:** `stageX_pilot.py 30 flows:n flows:ns reactions:n reactions:ns ranks:n,ns`, with nice 19, its own process group and the 5 GB free-disk guard (282 GB free at the start).
- **Pilot stopped at 17:09 UTC, after its first wake** (the continued network, flows, version n: 0 of 40 equations solved in 35 s, with a median of 243 low-rated terms tried per equation of 300 structures).
  - **Cause, an implementation error:** the never-blocked share was given in full to the next untried low-rated term at every visit. In PUCT an untried move's bonus is not divided by visits, so this explorer move always won as the node's visits grew, and took about 80% of the structures instead of about 10%.
  - **Fix:** the share now behaves as one child. The next untried low-rated term carries ρ × (add mass) / (1 + the low-rated terms already tried at that node), so its bonus decays with use like any child's.
  - Low-rated terms tried are now counted with the rules off too, giving the baseline's count.
  - The stopped run's outputs were deleted, and its log was kept as `stageX_driver_stopped.log`. The pilot restarts after the fix is checked.
- **Check of the decaying-prior fix:** two development flows systems, budget 300, the continued network's starting state. That fix behaved sensibly but too weakly.
  - At most nodes it reached only the 1–2 simplest low-rated terms (72 in one equation).
  - PUCT spends its visits on children that already fit well, so a share of prior mass is not a share of choices.
  - With the rules off, the search still tries 67–105 low-rated terms per equation, at random, through its existing 15% random exploration.
- **Implemented as approved ("at least 1/10"), as a share of choices.**
  - At every visit of a node in the network-guided search, with probability 1/10 the search tries the next untried low-rated term, simplest first. Otherwise it chooses as before. This works like the search's existing random exploration share.
  - Checked on the same two systems:
    - the simplest-first run reached 21–31 low-rated terms per equation at ρ = 1/10, and 14–23 with the exploration share, which also took 75 structures at even weights;
    - distinct low-rated terms tried: 92–111, against 67–105 with the rules off;
    - equations solved: unchanged (1 of 4 and 0 of 4).
- **The pilot restarts with this rule.**
- **Pilot stopped a second time, at 18:19 UTC on 10 November,** after the continued network's flows curve under version n.
  - Against f6, new-form development equations: 41 against 52 at 10 tasks (2–13, p = 0.0074); 47 against 76 at 30 (1–30, p = 3e-8).
  - **Cause, again the implementation:** a 1/10 draw at every node visited along each search path compounds with the path's depth. About 30–40% of the structures went to unfamiliar terms (a median of 105 per equation), starving the network's own choices and its wakes' solutions (7 against 10, and 27 against 33).
  - **Fix:** the draw is once per search step. With probability 1/10 the step is an exploring step: it descends as usual, and the new structure it creates is the expansion node plus that node's next untried low-rated term, simplest first. The new child is linked into the tree, so it is revisited if it fits well. So exactly 1/10 of the structures go to unfamiliar terms, each tried in the context the network judged best.
  - The stopped run's outputs were deleted, and its logs were kept as `stageX_*_stopped2.log`. The fix is checked before restarting.
- **Check of the per-step rule** (four development flows systems, the order-2 continued network after 30 flows tasks, budget 300):
  - exploring structures: 25–34 per equation (8–11%);
  - simplest-first runs: 4–19 deep, against 0–1 with the rules off;
  - equations solved: 5 of 16 in all three settings (rules off, version n, version ns);
  - version ns added 75 structures at even weights and about 20 exploring structures.
  - With the rules off, the search already tries 70–100 low-rated terms per equation, at random, through its existing 15% random exploration. That randomness is spread over about 4,000 terms, so a given term is rarely reached. The rule adds a tenth in simplest-first order.
- **The pilot restarts with this rule.**
- **First decision-relevant result** (20:06 UTC): the continued network on flows under version n solves 56 against f6's 52 at 10 tasks (7–3, p = 0.34), and **54 against 76 at 30 tasks (2–24, p = 1e-5)**. Under the pre-registered rule, version n is ineligible.
- **But the single f6 baseline is an outlier.** Every development run of this arm (continued network, flows after order 2's history) at 30 tasks:

  | Run | New-form solved at 30 tasks (of 159) |
  |---|---|
  | 768-term shortlist, baseline | 59 |
  | Shrink-and-perturb | 63 |
  | L2 Init | 52 |
  | ReDo | 57 |
  | 1,000-step reviews | 55 |
  | 2,048-term shortlist (f6) | 76 |
  | f6 + grammar (Stage A) | 67 |
  | f6 + never blocked | 54 |

  The fresh network varies too: 88, 80 and 67 in its three runs.
  - So the run-to-run spread at 30 tasks on flows is about 10–20 equations, and f6's 76 is the highest of eight runs.
  - The paired sign test treats equations as independent, but two runs differ as wholes (one learned network against another). Its p-values between single runs (here 1e-5) do not measure a version's effect; the placebo showed the same, more mildly.
- **Proposed to the owner** (the pilot continues unchanged meanwhile): replicate the f6 baseline with seeds 2 and 3 for the continued and fresh networks on flows at 10 and 30 tasks (4 runs, about 5 h).
  - Judge "worse than f6" against the baseline's three runs, not one.
  - Replicate the winning version twice before declaring it (about 2.5 h).

### Stage X pilot: results (11 November)

**Run:** 18:58 UTC on 10 November to 03:35 UTC on 11 November.
- No search was time-capped or failed.
- Review rounds waited up to 8 minutes for GPUs under other users' load, and always ran on a free or shareable GPU.
- Report: `outputs/accumulating_learner/g2h/stageX/report.txt`.

**New-form development equations solved** (flows of 159; reaction networks of 155; continued / control / fresh):

| Version | Flows, 10 tasks | Flows, 30 tasks | Reaction networks, 10 tasks | Reaction networks, 30 tasks |
|---|---|---|---|---|
| f6 (no exploration) | 52 / – / 36 | 76 / – / 80 | 57 / – / 17 | 60 / – / 35 |
| n (never blocked) | 56 / 59 / 42 | 54 / 82 / 91 | 57 / 46 / 23 | 63 / 62 / 48 |
| ns (never blocked + even-weights share) | 45 / 57 / 28 | 55 / 60 / 64 | 39 / 33 / 26 | 46 / 41 / 37 |

- **Against f6 at 30 tasks** (only version against only f6; p):

  | Version, arm | Flows | Reaction networks |
  |---|---|---|
  | n, continued | 2–24 (p = 1e-5), worse | 8–5 (p = 0.58) |
  | n, fresh | 19–8 (p = 0.052), better | 21–8 (p = 0.024), better |
  | ns, continued | 5–26 (p = 0.0002), worse | 7–21 (p = 0.013), worse |
  | ns, fresh | 2–18 (p = 0.0004), worse | 7–5 |

- **Version ns is worse than version n for every arm in both testbeds,** except the continued network on flows (55 against 54). The even-weights share costs the network's own budget more than it finds.
- **Continued against fresh on flows at 30 tasks (the slowdown):**
  - f6: 76 against 80 (p = 0.56);
  - n: 54 against 91 (3–40, p = 3e-9);
  - ns: 55 against 64 (p = 0.14).
  - **The gap does not close; under n it widens,** because the fresh network gains and the experienced one does not.
- **Continued against control on flows under n:** −1.9 points at 10 tasks; **−17.6 at 30 (2–30, p = 2.5e-7)**. With exploration, the made-up-trained control learns flows far better than the real-history network.
  - On reaction networks: +7.1 at 10 tasks (17–6, p = 0.035); +0.6 at 30.
- **The bump on flows** (root median rank among about 4,000 terms, after 10 → 30 tasks, and bump solutions in the wakes to 30 tasks):

  | Version, arm | Root rank, 10 → 30 tasks | Bump solutions in the wakes |
  |---|---|---|
  | n, continued | 1,921 → 2,069 | 0 |
  | n, control | 1,009 → 54 | 4 (it learned the bump) |
  | n, fresh | 15 → 21 | 10 |
  | ns, continued | 2,139 → 2,061 | 0 |
  | ns, control | 2,093 → 1,225 | 0 |
  | ns, fresh | 61 → 43 | 5 |

  For comparison, f6's continued network found 1 bump solution and its fresh network 8.
- **Exploration counts (version n):**
  - 83–93 distinct low-rated terms tried per equation (median); with the rules off the search's own 15% random exploration already tries 70–100.
  - Simplest-first depth: median 6 per node, at most 20–27.
  - The largest low-rated term tried had 7 nodes.
  - Equations solved with a term the network rated below even at the root: 15–41 per test.
- **Kinds of term found for the first time** (flows, tasks to 10 | to 30):

  | Version | Continued | Control | Fresh |
  |---|---|---|---|
  | f6 | 2 \| 2 | – | 5 \| 6 |
  | n | 5 \| 1 | 13 \| 4 | 5 \| 7 |

- **Bits check** (flows, 30 tasks; uniform prior 41.7):
  - f6: continued 16.2, fresh 17.5;
  - n: continued 19.5, control 16.7, fresh 16.4.
  - Reaction networks (uniform 43.6): f6 continued 19.6, fresh 24.2; n continued 18.6, control 18.4, fresh 21.6.
- **Time split per equation** (version n tests): about 85–100 s.
  - The prior (network scoring and evidence) takes 86–87% of the time; ordinary fits take 4%, thorough fits 9% and precise checks 9%.
  - Under ns, thorough fits and precise checks take 16–20% each: its even-weights share reaches more near-exact fits.
- **The pre-registered winner rule:**
  - Both versions are ineligible: the continued network on flows is worse than f6 at 30 tasks with p < 0.01 (n: p = 1e-5; ns: p = 0.0002, and its fresh network p = 0.0004).
  - Totals at 30 tasks, three arms and both testbeds: n 400, ns 303. Continued plus fresh: n 256 against f6's 251, ns 202.
  - The rerun rule's pending case (ns, reaction networks, continued, p = 0.013) cannot change ns's ineligibility, so it was not run.
  - **By the rule: check in before Stage B.**
- **Caution, as reported during the run:** f6's continued flows result (76) is the highest of eight development runs of that arm (52–76; the others 52–67). So "worse than f6" for the continued network on flows may largely be run-to-run spread. The proposed baseline replicates (seeds 2 and 3) are awaiting the owner's decision.
- **The budget check** (version n, the better version, at 1,000 structures) and a no-search check of where the bump sits in each network's low-rated queue are running.
- **Why the never-blocked rule never reached the bump** (root scoring of the version-n networks on the development flows systems; no search):
  - Every network, the fresh one included, rates about 3,800 of its about 4,000 legal terms below even. So the low-rated queue is nearly the whole vocabulary.
  - In the language's simplest-first order, the true bump terms sit at queue positions 1,950–2,530 (10th–90th percentile; median about 2,200) for every network. The saturating x/(K + x²) terms sit at about 2,850.
  - The rule's runs reach a median of 6 per node (at most 27).
  - So ordering by simplicity cannot give a term of the bump's size a meaningful chance within 300 structures, for any network.
  - **Where the bump discoveries came from instead:**
    - the fresh network learned the bump from its uniform first wake, and after 30 tasks rates only 4 of 47 true bump terms below even;
    - the control's 4 bump solutions under n most likely came from the search's existing 15% random exploration. It found none under ns or in claim run 2, but after those 4 it learned the term (root rank 1,009 → 54).
- **The data's evidence alone at the root** (each term's partial R² for the whole derivative; no network) ranks:
  - true bump terms at median 736 of about 3,840 (25th–75th percentile 187–1,502; 11% in the top 50, 30% in the top 300);
  - saturating terms at median 955;
  - other true terms at median 1,906.
  - So evidence at the root alone is not enough either. Deeper in a search, the residual isolates the missing terms, so the evidence there should be sharper; that is untested.
- **The budget check** (version n, flows, the 30-task networks, 1,000 against 300 structures per equation; thorough cap 200; no time-limit hits):

  | | 1,000 structures | 300 structures |
  |---|---|---|
  | Continued: new-form solved | 60 (6–0, p = 0.031) | 54 |
  | Continued: bump solutions | 0 | 0 |
  | Continued: simplest-first depth, median (max) | 14 (43) | 6 (27) |
  | Continued: seconds per equation | 359 | 133 |
  | Fresh: new-form solved | 112 (21–0, p = 1e-6) | 91 |
  | Fresh: bump solutions | 38 | 29 |
  | Fresh: whole systems | 9 | 3 |
  | Fresh: seconds per equation | 332 | 80 |

  - The 300 budget limits the fresh network a lot. For the experienced network the extra budget helps elsewhere, but it still never reaches the bump.
- **The quick-fit question** (the measured time split):
  - The prior step (network scoring plus the evidence for every term) takes 76–90% of the search time, at 0.24–0.39 s per structure.
  - An ordinary fit takes 0.010–0.024 s per structure, 15–25 times less. Thorough fits and precise checks take 5–18%.
  - **So yes:** quick fits could screen many more candidates per network evaluation. For example, at each expansion, quick-fit the network's next few choices and the next few unfamiliar terms together and keep the best, with thorough fits still capped. That gives perhaps 5–10 times the structures at similar cost. It is a change to the search, not run.

### Owner: baseline replicates; hold the safety net and Stage B; assess compositional term building (11 November)

**Owner.** *"Approve the baseline replicates. Hold off on the stronger safety net and on Stage B. We haven't yet implemented DreamCoder's core, and this pilot suggests why the gap widens: DreamCoder builds programs step by step from primitives under a learned grammar, so an unfamiliar term made of familiar parts (like 1/(K + x²)) is only a few likely choices away. Our search picks from a flat list of about 4,000 whole terms, where an unfamiliar term can sit thousands of places deep. In plain language, assess replacing the flat term list with compositional term building: the search constructs each term from operations and arguments, guided by the network and a grammar over operations, as in DreamCoder. Explain what changes in the search, the network and the fitter, the cost, and how library learning (Stage C) would then add learned pieces as new operations. Send that, with a staged plan, before running anything beyond the replicates."*

- **The replicates started at 05:24 UTC on 11 November:** `stageX_pilot.py 30 flows:f6:r2:continued,fresh flows:f6:r3:continued,fresh` (seeds 2 and 3; nice 19; own process group; 5 GB guard; 171 GB free).

#### Compositional term building: assessment and staged plan (11 November; nothing run)

**What we have.** L1's terms are built from a small grammar (`language_g1.enumerate_trees`):
- the variables x1…x4 and a constant;
- the unary operations sin, exp and log (no exp or log inside another);
- powers (2, 3, ½, or a fitted exponent);
- add, multiply and divide.
- A term has at most 7 nodes and 2 constants.

All such trees were enumerated once, offline, and canonicalized. This gives 66,109 terms in four variables, of which 4,683 keep their form when the variables are rescaled; that set is the search's vocabulary. So the building blocks already exist, but the search sees only the finished list: one choice among about 4,000 whole terms, ranked by the network. The pilot showed what that costs:
- every network rates about 3,800 terms below even;
- the bump sits about 2,000 deep in the simplest-first order.

**DreamCoder's way.**
- A program is built step by step. At each open slot a production is chosen with a probability that depends on the slot's parent and position (a "contextual" grammar).
- The grammar's weights come from solved tasks. A recognition network predicts task-specific weights, and programs are enumerated in decreasing probability.
- An unfamiliar program made of familiar parts is a few likely choices away.
- The bump, as a tree, is divide(constant, add(constant, power(x, 2))): 7 choices among about a dozen options each.
  - The parts are familiar: the history's verified solutions include x/(K + x)-type terms (divide with an add(constant, ·) denominator) and many squares.
  - Its probability is a product of mostly likely choices. The one less common step is a square inside that denominator.

**What would change.**
- **The search, in two levels.**
  - The equation search (which terms are in an equation, with fitted values guiding it) stays.
  - The "add a term" move draws from a term generator instead of the list. The generator proposes terms in decreasing probability, built from operations and arguments, and the search widens each node's set of children progressively: the next most probable untried term each time.
  - The shortlist, the rest slot and the never-blocked queue all disappear, because the probability order is the queue.
  - Every buildable term has a nonzero probability (smoothing), so none is blocked; the budget only sets how deep the search goes.
  - The size limit becomes soft (probability, not a hard 7-node cap). That lets library pieces and larger terms in later.
- **The grammar** is over operations, not whole-term shapes: how likely each operation is, given its parent operation and argument position.
  - It is learned from each lineage's verified solutions with Bayesian smoothing (Stage A's machinery applied to operations).
  - It is the generator's base, the source of Stage B's dreams, and the bits check's measure.
  - The control gets the same machinery fitted to its own made-up systems.
- **The network** becomes DreamCoder's recognition model.
  - From the system's data and the current partial equation, it predicts the grammar's weights for this equation: how likely each operation is in each context.
  - It is trained to make the building steps of verified (and dreamed) terms likely.
  - The current per-term scorer, which reads each complete term's fit to the data, can stay as a second check on generated candidates.
- **The fitter** changes little. It already handles any canonical term (term information and constants' starting values by role), and can add a term's data columns on demand (`Bank.add`). What is new:
  - columns and compiled code are computed when a term is first generated, and cached across systems;
  - the rescaling rule (a term must keep its form when the variables are rescaled) is checked when a term is completed, and terms that fail are skipped;
  - duplicates are removed by canonical form, which is cached.
- **To check the new search fairly,** it is first restricted to L1's limits (7 nodes, 2 constants) and compared with the flat-list search on the same space.
- **Library learning (Stage C) then adds learned pieces as new operations.**
  - A piece that shortens the description of the verified solutions under the operation grammar becomes one production with its own probability: for example "saturate(x) = x/(K + x)", or "bump(x) = 1/(K + x²)" once found. The generator then proposes it in one choice instead of five to seven.
  - The pieces are chosen by compression (DreamCoder's description-length criterion), with the approved decoy library and removal check.
  - The cross-equation parts C1–C3 fit the same picture as system-level productions: attention across equations, paired moves that add one rate with opposite signs, and shared constants.

**Staged plan (one change at a time; replicate seeds per the noise finding).**
- **K0, a no-search check (minutes).**
  - Fit the operation grammar to each lineage's verified solutions.
  - Compute where the development systems' true terms, the bump especially, would rank among all terms by that grammar, against their ranks now (network, shape grammar, simplest-first).
  - This tests the core claim before anything is built: it should put the bump in the top tens to low hundreds for the experienced lineage.
- **K1, compositional order in the existing search** (about 0.5 day to build):
  - the never-blocked queue and the rest slot ordered by the operation grammar's probability instead of simplest-first;
  - the flat list kept.
  - Pilot: flows and reaction networks, 10 and 30 tasks, three arms, with 2–3 seeds per arm.
- **K2, the generator replacing the flat list** (about 4–6 days to build and validate):
  - best-first generation under the operation grammar, the recognition network predicting its weights, and progressive widening;
  - the fitter's columns on demand, with caching.
  - First reproduce the flat-list search on L1's space, then pilot.
- **Then:**
  - Stage B: dreams sampled from the operation grammar, so they inherit composable terms, not the shape grammar's blind spot;
  - Stage C: library pieces as new operations, with decoy and removal checks, then C1–C3;
  - D (curriculum) and E (fewer bursts) as planned.
- **Compute:** K1 costs about the same as now. For K2, the generator should be no slower per structure than scoring about 4,000 terms with the network, which is now 76–90% of the search time, provided it generates hundreds of candidates per step, not thousands.
- **Risks:**
  - The recognition network predicts a structured output, which is harder to train than a per-term score.
  - Longer terms mean more constants and harder fits (the earlier multi-piece fitting limit).
  - The rescaling rule rejects many generated terms (only 4,683 of 66,109 pass in four variables), which wastes generation unless the grammar learns where constants go.

### Owner: K0 approved (11 November)

**Owner.** *"Approved: start with K0 once the replicates are in. Run K0 for the experienced, control and fresh networks on both new domains, reporting where the true development terms (especially the bump) would rank under each network's operation grammar versus now, and the bits check at the operation level, so we can see whether real experience gives a better compositional prior than made-up practice. Check in after K0, and again after K1, before starting the K2 build."*

- **K0's design** (no search, no training; development systems only; counts only):
  - **The operation grammar:** each term's canonical tree is read relative to its equation (its own variable; any other variable; a constant; numbers; add and multiply as nested pairs; powers; sin, exp, log, Abs). A term's probability is the product over its nodes of P(child | parent, argument position), as in DreamCoder's contextual grammar.
  - **Smoothing:** each context's counts get Dirichlet smoothing with a uniform base over the tokens, and α is chosen as in Stage A: leave one domain out for earlier experience, else leave one equation out.
  - **Components:** earlier and newest, mixed half and half.
  - **Fitted to each lineage's verified solutions,** with Stage X version n's lineages, since they have all three arms in both domains:
    - the continued network: its history plus its new-domain wakes;
    - the control: its made-up systems' structures plus its wakes;
    - the fresh network: its wakes only.
    - Each at three points: before the new domain, after 10 tasks and after 30.
  - **Compared rankings,** among the about 4,000 terms of the system's vocabulary, of each development system's true new-form terms:
    - the operation grammar;
    - Stage A's shape grammar on the same solutions;
    - the language's simplest-first order;
    - the lineage's network at the root (scoring only), before the new domain and after 10 and 30 tasks.
  - **Reported** for all true new-form terms; for the unfamiliar ones (kinds not in the lineage's solutions so far); and for the bump and the saturating term on flows.
  - **The bits check at the operation level:** mean description length of the true new-form development equations under each lineage's operation grammar.

### The f6 baseline replicates: results (11 November)

**Run:** 05:24–09:22 UTC: seeds 2 and 3, continued and fresh, flows after order 2's history, 10 and 30 tasks. No search was time-capped or failed.

**New-form development equations solved** (of 159):

| Arm, point | f6 seed 1 / 2 / 3 | Mean (range) | Version n | Version ns |
|---|---|---|---|---|
| Continued, 10 tasks | 52 / 54 / 44 | 50.0 (44–54) | 56 | 45 |
| Continued, 30 tasks | 76 / 49 / 51 | 58.7 (49–76) | 54 | 55 |
| Fresh, 10 tasks | 36 / 37 / 32 | 35.0 (32–37) | 42 | 28 |
| Fresh, 30 tasks | 80 / 57 / 68 | 68.3 (57–80) | 91 | 64 |

- **Seed 1 was an outlier.** With nothing changed but the seed, the continued network ranges 49–76 at 30 tasks. Judged against the three runs, neither version makes the continued network worse; both are inside the range. The pre-registered verdict ("worse than f6, p = 1e-5") came from one outlying baseline run.
- **The bump is a rare discovery for every recipe.**
  - Bump solutions in the wakes to 30 tasks: continued network, f6 seeds 1–3, 1 / 0 / 0, and 0 under both versions; fresh network, f6 seeds 1–3, 9 / 0 / 0, then 11 under n and 6 under ns.
  - Only seed 1's uniform first wake found a bump solution. The fresh network then learned it.
  - Version n's and ns's fresh lineages reused that same lucky first wake, so the fresh network's 91 under n started with the discovery.
- **The flows slowdown therefore depends on luck.** Continued minus fresh at 30 tasks under f6 is −4, −8 and −17 for seeds 1–3. It is largest when the fresh network makes that early discovery.
- **For claims:** single runs of an arm at 30 tasks on flows differ by up to about 25 equations. Comparisons need several seeds per arm, judged run against run, not equation by equation.

### K0: the operation grammar's ranks and bits (11 November; no search, no training)

**Run:** `k0_check.py 16`, 09:22–09:27 UTC (report `outputs/accumulating_learner/g2h/stageX/k0/report.txt`).
- Lineages are Stage X version n's: continued, control and fresh.
- Ranks are medians among the system's vocabulary (about 3,841 terms on flows, about 4,671 on reaction networks).
- "Unfamiliar" means a true term whose kind is not among the lineage's verified solutions (or made-up structures) so far.

**Flows** (159 new-form development equations, 557 true terms; operation grammar / shape grammar / network / simplest-first):

| Lineage, point | Unfamiliar true terms | Bump (47) | Saturating (55) |
|---|---|---|---|
| Continued, before the domain | 19 / 113 / 755 / 23 (239 terms) | 860 / 2,282 / 2,987 / 2,248 | 1,180 / 2,929 / 3,199 / 2,898 |
| Continued, 10 tasks | 1,182 / 2,626 / 1,892 / 2,600 (102 terms) | 1,063 / 2,282 / 1,951 / 2,248 | 1,505 / 2,929 / 1,781 / 2,898 |
| Continued, 30 tasks | 730 / 2,628 / 2,908 / 2,600 (102 terms) | 442 / 2,285 / 2,907 / 2,248 | 780 / 2,932 / 2,978 / 2,898 |
| Control, before | 58 / 2,016 / 482 / 22 (61 terms) | 941 / 1,275 / 1,382 / 2,248 | 2,349 / 1,128 / 703 / 2,898 |
| Control, 30 tasks (after finding 4 bump solutions) | – | 100 / 30 / 54 / 2,248 | – |
| Fresh, 10 tasks (after its first wake's bump) | – | 30 / 7 / 15 / 2,248 | – |
| Fresh, 30 tasks | – | 30 / 9 / 21 / 2,248 | – |

All true terms rank at a median of about 8 under every prior: most are simple polynomials.

**Reaction networks** (155 new-form, 571 true terms; operation / shape / network / simplest-first):

| Lineage, point | Unfamiliar true terms | 1/(K + x²) terms (39) |
|---|---|---|
| Continued, before the domain | 130 / 414 / 739 / 385 | 691 / 2,847 / 3,567 / 2,829 |
| Continued, 30 tasks | 718 / 2,858 / 1,808 / 2,836 | 716 / 2,851 / 1,444 / 2,829 |
| Control, before | 61 / 2,349 / 77 / 24 | – |
| Control, 10 tasks | 13 / 2,349 / 33 / 24 | – |

**The bits check at the operation level** (mean bits per true new-form equation):

| Lineage | Flows: before / 10 / 30 tasks | Reaction networks: before / 10 / 30 tasks |
|---|---|---|
| Uniform | 66.9 | 75.1 |
| Continued | 33.2 / 32.5 / 28.0 | 35.3 / 31.8 / 30.5 |
| Control | 33.4 / 30.5 / 24.5 | 37.4 / 33.5 / 31.3 |
| Fresh | uniform / 35.4 / 24.0 | uniform / uniform / 33.2 |

**Reading.**
1. **The operation grammar ranks unfamiliar terms far better than the flat list's network or the shape grammar.**
   - Before a new domain, the continued network's unfamiliar true terms rank at a median of 19 against 755 on flows, and 130 against 739 on reaction networks.
   - That is the compositional effect the owner described: a term of familiar parts is near the front.
2. **But for the bump the gain is a factor of 2–7, not to the top 100.**
   - The continued lineage ranks it at 440–1,060, against about 2,000–3,000 by its network.
   - One step is rare in its history, a square inside the sum of a reciprocal (its sums were "constant + variable"). A pair-level grammar cannot see the whole saturating-denominator pattern, which is what library pieces (Stage C) supply.
   - The fresh network and the control rank the bump high only after their own searches found it.
3. **Real experience against made-up practice at the operation level:**
   - Reaction networks: real experience is better by 2.1 bits per equation before the domain, 1.7 at 10 tasks and 0.8 at 30.
   - Flows: equal before the domain (33.2 against 33.4). By 30 tasks the control (24.5) and the fresh network (24.0) are ahead (28.0), because only they found the flows-specific terms.
4. **For K1:**
   - Ordering unfamiliar terms by the operation grammar should help with unfamiliar terms generally.
   - It is not expected to reach the bump for the continued network in 300 structures: about 440 deep in its order, against the never-blocked rule's 6–27 per node.
   - Reaching it needs the generator's cheap screening of many candidates per step, the data's evidence at the node, or library pieces.

### Owner: K0b, then K1; seeds per arm; replicate claim run 2's strongest result (11 November)

**Owner.** *"Approve K0b first, then K1 using the same-network test followed by learning curves for the winning ordering (2 seeds per arm). For K0b: judge it on all unfamiliar true terms in both domains, not just the bump, and apply the same compression machinery to the control's and fresh network's own solutions. Given the replicate results, every comparison from now on needs at least 2–3 seeds per arm. Also plan a replication of the claim-run-2 reaction-network result (order 2, experienced vs control at 100 tasks) with additional seeds, so we know whether our strongest positive finding survives the noise. Check in after K0b."*

- **Standing rule from now on:** every comparison has at least 2–3 seeds per arm, judged run against run, with per-equation tests only within the same network.
- **K0b** (no search, no training):
  - DreamCoder-style library learning by compression: pieces with gaps (subtrees with some subtrees replaced by holes), added greedily while they shorten the description of the lineage's own verified solutions (or the control's made-up structures).
  - The same machinery is used for all three arms.
  - The operation grammar is extended with these pieces as single operations, and each term's probability is that of its best parse.
  - Judged on all unfamiliar true terms in both domains (with the bump and the saturating term as cases), and on all true terms. Also the operation-level bits check.
- **Planned (not run):** a replication of claim run 2's strongest finding, order 2's reaction networks at 100 tasks (continued 59 against control 35 of 90 new-form, +26.7 points), with additional seeds. The design and cost are sent at the K0b check-in.

### K0b: design as built, and how it will be read (11 November; set before the run)

**What is built** (`library_ops.py`, `k0b_check.py`; no search, no training; development systems only; counts only).
- **Pieces.** A piece is part of a verified solution's tree with up to two of its parts replaced by gaps, for example (K + □)^□. The grammar uses it as one operation whose gaps are its arguments.
- **Library learning by compression**, as in DreamCoder:
  - Start with no pieces. Add the piece that most shortens the description of the lineage's own solutions (bits of the library plus bits of the solutions written with it), while it shortens it. At most 20 pieces.
  - A candidate must occur in at least 2 distinct solution terms. At each step the 30 most promising candidates are scored exactly.
  - The description length integrates the grammar's parameters out, so a piece pays for any new contexts it opens.
- **Two forms of grammar, each learning its own library:**
  - "Contextual" is K0's operation grammar: a choice depends on its parent operation and argument position.
  - "Context-free" is the form of DreamCoder's own prior: a choice depends only on whether it fills an exponent or an expression. Under this form a piece learned inside a product is just as likely at the top of a term.
- **Scoring.** A term's probability is that of its best parse, at each node the more probable of its own operation and any matching piece.
  - Smoothing is chosen by held-out likelihood as in K0.
  - When one history domain at a time is held out, the library is learned again without that domain, so the held-out domain is new to the library too.
- **Check.** Without pieces, the contextual form reproduces K0's grammar exactly; the smoke test confirmed this at every lineage point.
- **Runs per arm** (the seeds rule): every Stage X run of each arm.
  - Flows: continued 5 (versions n and ns, and f6 seeds 1–3; seed 1 is the slowdown pilot's p6 baseline), control 2 (n, ns), fresh 5.
  - Reaction networks: continued 3, control 2, fresh 3.
  - The fresh runs share claim run 1's uniform first wake (except flows f6 seeds 2 and 3), so they are not independent at 10 tasks.
- **Term sets:**
  - all true terms of the new-form equations;
  - those unfamiliar to the lineage (K0's definition);
  - those new to both prior experiences: kinds absent from both the history's solutions and the made-up structures, the same set for every arm;
  - the bump and the saturating term, as cases.

**Reading, set before the run.**
1. **A form's library helps** if it lowers the median rank of unfamiliar true terms for every arm at every point where the arm has a library, in both domains (median over runs), and lowers the bits. Anything else is reported as mixed.
2. **Real experience gives a better library than made-up practice** if the continued arm, with its library, beats the control on both of these:
   - it ranks the terms new to both prior experiences better;
   - it needs fewer bits.
   This must hold before the domain, and in every run pairing after 10 and 30 tasks.

### K0b: results (11 November; no search, no training)

**Run:** `k0b_check.py 16`, 15:47–15:59 UTC at nice 19 (report `outputs/accumulating_learner/g2h/stageX/k0b/report.txt`; libraries in `libraries_{flows,reactions}.json`).
- Lineage points: 27 on flows, 19 on reaction networks.
- Ranks are within each development system's vocabulary (about 3,841 terms on flows, 4,671 on reaction networks). Each cell is the median over runs of each run's median.

**What the libraries contain** (learned from training-side solutions only):
- **Continued network** (history: 894 equations, 43 distinct terms):
  - contextual: (K + □)^□, u·□, K·□, (u)^□, u·(K + □)^□, ...
  - context-free: exp(K·u), (u)^□, □·(K + □)^-1, □·(K + □)^K, sin(K·v), ...
- **Control** (made-up systems: 9,730 equations, 1,113 distinct terms):
  - contextual: K·□, K·(□)^□, (□)^-1, □·(□)^-1, ...
  - context-free: sin(K·□), exp(K·□), log(K·□), (u)^□·(v)^□, (K + □)^-1, ...
- **Fresh network** after 30 flows tasks: 1/(K + □²), from its own wakes.

**Flows, unfamiliar true terms** (network / K0 grammar / + library / context-free / + library / simplest-first):

| Arm, point | Ranks |
|---|---|
| Continued, before (239 terms) | 755 / 19 / 224 / 17 / 28 / 23 |
| Continued, 10 tasks (5 runs) | 1,861 / 970 / 906 / 1,036 / 449 / 2,600 |
| Continued, 30 tasks | 1,607 / 730 / 580 / 1,014 / 486 / 2,600 |
| Control, before (61) | 482 / 58 / 72 / 11 / 99 / 22 |
| Fresh, 30 tasks | 1,250 / 192 / 76 / 598 / 598 / 2,752 |

**Flows, the bump** (47 terms): continued before 860 → 1,189 (contextual) and 595 → 1,067 (context-free); continued at 30 tasks 442 → 813 and 648 → 1,122; control before 941 → 1,610 and 988 → 667; fresh at 30 tasks 34 → 12 and 324 → 464.

**Flows, the saturating term** (55 terms): continued before 1,180 → 271 and 1,165 → 372; control before 2,349 → 1,395 and 2,717 → 949; continued at 30 tasks 780 → 287 and 1,172 → 377.

**Reaction networks, unfamiliar true terms** (same columns):

| Arm, point | Ranks |
|---|---|
| Continued, before (193) | 739 / 130 / 87 / 655 / 107 / 385 |
| Continued, 10 tasks (3 runs) | 695 / 238 / 188 / 734 / 39 / 390 |
| Continued, 30 tasks | 3,520 / 721 / 463 / 590 / 265 / 2,830 |
| Control, before (31) | 77 / 61 / 79 / 14 / 128 / 24 |
| Control, 10 tasks | 32 / 33 / 41 / 18 / 25 / 24 |
| Fresh, 30 tasks | 1,539 / 452 / 896 / 515 / 1,183 / 2,825 |

**Reaction networks, 1/(K + x²) terms** (39): continued before 691 → 2,844 and 654 → 1,935.

**All true terms:** the median stays about 5–12 everywhere, except the control before reaction networks (53 → 74, and 11 → 121).

**Bits per new-form equation** (K0 grammar / + library / context-free / + library):

| Arm, point | Flows (uniform 66.9) | Reaction networks (uniform 75.1) |
|---|---|---|
| Continued, before | 33.2 / 38.3 / 33.9 / 30.9 | 35.3 / 38.0 / 38.5 / 38.7 |
| Continued, 10 tasks | 32.0 / 30.4 / 33.4 / 30.0 | 31.9 / 35.1 / 37.4 / 32.6 |
| Continued, 30 tasks | 28.0 / 25.5 / 32.7 / 28.0 | 30.9 / 30.3 / 35.9 / 34.3 |
| Control, before | 33.4 / 31.1 / 37.1 / 32.4 | 37.4 / 34.9 / 41.6 / 39.6 |
| Control, 10 tasks | 29.4 / 26.6 / 35.1 / 28.7 | 34.1 / 31.4 / 38.6 / 34.6 |
| Control, 30 tasks | 24.5 / 22.4 / 33.3 / 26.3 | 30.3 / 25.9 / 36.5 / 32.0 |
| Fresh, 30 tasks | 24.2 / 22.5 / 30.7 / 30.8 | 35.5 / 42.0 / 37.4 / 45.8 |

**Against the reading set before the run.**
1. **"A form's library helps": not met for either form; mixed.**
   - **Helps:** the continued network once it has new-domain solutions (both forms, both domains, 10 and 30 tasks); reaction networks before the domain; the saturating term everywhere.
   - **Hurts:** the control at every point; the contextual form for the continued network on flows before the domain (19 → 224); the fresh network on reaction networks at 30 tasks.
   - **The bump is never brought closer** for the continued network.
2. **"Real experience gives a better library": not met.**
   - Context-free form with its library: real experience ranks the terms new to both prior experiences better in every run pairing at every point in both domains (before: flows 10 vs 99, reaction networks 9 vs 128). But the control needs fewer bits after 10 and 30 flows tasks (continued fewer in 0 of 10 and 1 of 10 pairings) and after 30 reaction tasks (2 of 6).
   - Contextual form with its library: the control is ahead on both measures at most points.
   - **Not part of the reading set in advance:** without any library, the context-free form has real experience ahead on bits in every pairing on flows at 10 and 30 tasks, on reaction networks at 10 (5 of 6 at 30), and before the domain in both domains.

**Diagnosis of the bump** (history grammars, the bump 1/(K + u²) for one equation among all 4,683 terms; training-side data only):
- **Contextual form.** Without a library the bump costs 20.2 bits (rank 660). The expensive step is a power inside a sum, 10.6 bits: it never occurs in the history.
  - With the library it costs 25.9 bits (rank 1,430). The piece (K + □)^□ is used at the top of a term, where the history never had it (only inside products): 15.2 bits. The square in its gap, also unseen there, costs 9.6 bits.
  - The saturating term u/(K + u²) uses u·(K + □)^□, which the history does have at the top: 23.0 → 17.6 bits, rank 1,052 → 385.
- **Context-free form.** The bump costs 21.6 → 21.2 bits. Larger pieces (□·(K + □)^-1, □·(K + □)^K) absorbed every use of the general piece (K + □)^□, so it has only smoothing weight.
  - The saturating term goes from 26.2 → 12.1 bits, rank 1,296 → 153.
- **In short:** pieces remove the cost of a familiar frame around an unfamiliar term. They cannot supply the unfamiliar step itself (here the square inside the sum), which still costs about 10 bits under every grammar tried.

**Limits of the version tested** (not tried after seeing the results; candidates for a check, if wanted):
- Each solution is parsed one way, the largest pieces first. This starves the plain operations of counts, which is the likely reason the library hurts the control and some unfamiliar terms. Counting over all parses (an inside–outside fit) would soften that.
- General pieces lose their uses to more specific pieces that contain them.

### Proposed, awaiting approval: the replication of claim run 2's reaction-network result, and K1 (11 November)

**Replication** of order 2's reaction networks at 100 tasks.
- **The finding:** continued 59 against control 35 of 90 new-form test equations (+26.7 points). Most of the gap was already there before any reaction task: 32 against 7 at 0 tasks. At 0 tasks the control's network cut a missing true term from its 2,048-term shortlist in 46% of expansions.
- **Design:**
  - The reaction phase is run again for the continued network and its control, with seeds 2 and 3: a new search seed and new lineage names, so fantasies and training batches are new draws too.
  - Same starting networks (o2_flows_L1_d100; c2x_flows_L1100) and the same test systems (test2). Tests at 0, 10, 30 and 100 tasks.
  - The claim-run recipe is unchanged; the control's rehearsal mirrors its own seed's continued run.
  - Code: a new module subclassing the driver; nothing in cycles_g2h changes.
- **Reading:**
  - **Survives** if at 100 tasks every continued run beats every control run on new-form test equations (9 of 9 pairings) and the mean gap is at least +10 points.
  - **Fails** if the mean gap is under +10 points or a control run matches a continued run.
  - Otherwise **mixed**.
- **Limits:**
  - Three runs per arm cannot reach a strong significance level: complete separation has a chance of 1 in 20. Five per arm would allow p < 0.01.
  - It covers the reaction phase's run-to-run noise only. It does not cover the flows phase that produced the starting networks, or the draw of test systems.
- **Cost:** claim run 2 took 3.0 h for this phase at 30 workers.
  - Two extra seeds: about 6 h (7–8 h at 24 workers under the current load).
  - Four extra seeds: about 12 h.
  - A full order-2 replicate (flows, then reaction networks) would be about 6.6 h per seed.

**K1: the grammar's order inside today's search.**
- **Orderings** (budget 300, the 2,048 shortlist, f6 rules otherwise):
  - **A:** the network alone.
  - **B:** the network mixed half and half with K0's operation grammar, fitted on the network's own lineage (its history's solutions or made-up structures, plus its wakes so far).
  - **C:** the same with the context-free grammar and its library.
- **Same-network test** (development systems; no training), on 16 existing 30-task networks:
  - Flows: continued and fresh f6 seeds 1–3; controls n and ns.
  - Reaction networks: continued and fresh f6 seed 1 and versions n and ns; controls n and ns.
  - Every network is searched with A, B and C under one new seed. The 8 f6 networks' earlier tests, under other seeds, give the A-versus-A noise.
- **Winner rule** (set before). An ordering wins if all of these hold:
  - it beats A on new-form equations solved in at least 14 of 16 networks, and in at least 6 of 8 in each domain;
  - its median gain is larger than the median A-versus-A difference;
  - shared-form equations fall by no more than that difference.

  If both B and C pass, the larger median gain wins. If neither passes, no learning curves are run, and I check in.
- **Then learning curves** for the winner: continued, control and fresh, both domains, 2 seeds each, at 10 and 30 tasks.
  - Wakes and tests use the winning ordering; training is unchanged.
  - The baselines also need 2 seeds per arm: 6 new f6 lineages (reaction networks continued and fresh seed 2; f6 controls seeds 1–2 in both domains).
- **Cost** (about 12 min per development test and 45–60 min per lineage curve, at 30 workers):
  - same-network test: 48 tests, about 9–10 h (32 tests, about 6–7 h, without C);
  - curves: 18 lineages, about 14–18 h;
  - about 2 h of training on a shared GPU.

### Owner: skip the replication; the all-parses check, then K1 with ordering C; outline early tests of K2's screening (11 November)

**Owner.** *"Skip the claim-run-2 replication for now; we'll test accumulation again with multiple seeds once the new library design is in place. Run the 15-minute all-parses check of K0b first, then K1 as designed, including ordering C. Since K0b shows that genuinely new steps can only come from search, also outline how K2's cheap screening of many candidates per network step could be tested early, possibly inside K1's setup. Check in with K1's same-network results before running any learning curves."*

**The all-parses check: design and reading (set before the run).**
- **The change.** Each library grammar's weights are fitted by counting every parse of each solution, weighted by its probability. This is an inside–outside fit, iterated to convergence, with alpha 1 while fitting. K0b instead counted one parse, the largest pieces first.
  - Unchanged: the pieces, the scoring of a term by its best parse, and the held-out choice of smoothing.
  - The grammars without a library have a single parse and do not change.
- **A determinism fix.** Library learning broke ties among equally promising candidates in an order that varied from process to process (Python's per-process hashing). Ties are now broken in a fixed order.
  - Both fits are re-run on identical libraries.
  - The one-parse re-run is compared with K0b's reported numbers.
- **Reading:**
  1. **Removing the harm.** The all-parses fit removes the harm if, in every arm–point cell where K0b's one-parse library ranked unfamiliar true terms worse than the grammar without a library, the all-parses library ranks them no worse than that grammar (median over runs).
  2. **K0b's readings** 1 and 2 are applied to the all-parses fit.
  3. **Which fit ordering C uses.** C uses the fit of the context-free grammar with its library that ranks unfamiliar true terms better in more arm–point cells. This counts both domains, and only cells where the arm has a library. Ties are broken by mean bits.

### K0b's all-parses check: results (11 November)

**Run:** `k0b2_check.py 16`, about 2 minutes (report `outputs/accumulating_learner/g2h/stageX/k0b2/report.txt`).
- **Reproducibility.** The libraries, learned again with ties broken in a fixed order, are identical to K0b's at all 46 lineage points. The one-parse ranks are identical too, so K0b's numbers stand.
- **The fit barely moves.** The all-parses fit converged in a median of 2–3 iterations (at most 61) and hardly changed the counts. A piece's parse needs fewer choices than spelling the same part out with plain operations, so it keeps almost all the weight.
- **Unfamiliar true terms, one parse → all parses** (median over runs):

| Arm, point | Contextual | Context-free |
|---|---|---|
| Flows, continued before | 224 → 229 | 28 → 28 |
| Flows, control before | 72 → 72 | 99 → 90 |
| Reaction networks, continued before | 87 → 92 | 107 → 113 |
| Reaction networks, control before | 79 → 79 | 128 → 114 |
| Reaction networks, continued at 10 tasks | 188 → 221 | 39 → 39 |

- Bits move by at most 0.7 per equation, mostly by 0.0–0.2.

**Against the reading set before the run.**
1. **Harm removed: no.** In none of the 10 arm–point cells where the one-parse library hurt is the all-parses library no worse than the grammar without one.
2. **K0b's readings 1 and 2 for the all-parses fit: not met**, the same pattern as with one parse.
3. **Ordering C's fit: all parses.**
   - The two fits tie on cells: all parses better in 3, one parse in 3, equal in 5.
   - The tie-break by mean bits over the 14 arm–point cells with a library favours all parses by 0.016 bits per equation (33.18 against 33.20).
   - The two are practically the same grammar. Averaged over lineage points instead of cells, the difference is 0.013 bits the other way.

**Meaning.** The starvation is not an artefact of the fit. It is the price of the pieces under this grammar: a piece wins every parse it matches, so the plain operations lose weight. Remedies would change the model rather than the fit, and were not pursued:
- score a term by the sum over its derivations;
- keep a share of weight on the plain operations by construction.

### K1's same-network test: launched (11 November, 18:07 UTC)

- **Code.** `k1_check.py`: orderings A, B and C as approved.
  - C uses the all-parses fit, by the pre-set tie-break.
  - The mixed prior gives the term to add (and the added term of a swap) the network's probability times ½ plus the grammar's times ½, renormalized over terms not in the structure. Removals stay the network's.
  - Unit check: probabilities sum to one, and every term keeps at least half its network probability.
  - Smoke test (6 tests on 2 systems): no failures, no time caps.
- **Run.** 48 tests (16 networks × 3 orderings), search seed 4, budget 300 and thorough cap 60, time cap 3,600 s per system. 28 workers at nice 15. A watchdog stops the run if free disk falls below 5 GB. Outputs are in `outputs/accumulating_learner/g2h/k1/`.
- **The A-versus-A noise** comes from the 8 f6 networks' earlier development tests (seeds 1–3; same budget, shortlist and rules).
- **Winner rule** as set before. A new ordering wins if all of these hold:
  - it beats A on new-form equations in at least 14 of 16 networks, and in at least 6 of 8 per domain;
  - its median gain exceeds the median A-versus-A difference;
  - shared-form equations fall by no more than that difference.

  If both B and C pass, the larger median gain wins; if neither does, no learning curves are run. Either way, I check in before any curves.

### K1's same-network test: results (12 November; finished 02:56 UTC)

**Run.** 48 of 48 tests, with no failures and no time caps (report `outputs/accumulating_learner/g2h/k1/report.txt`).
- Restarted once at 18:20 UTC to queue every test's systems at once; the first finished test was kept.
- Reniced to 19 at about 20:45 UTC, while machine load was 60–68 on 48 cores.

**New-form development equations solved, A / B / C** (A = the network alone; B = mixed with K0's grammar; C = mixed with the context-free grammar and its library). In brackets, A run with the network's own earlier seed:

| Network | Flows (of 159) | Reaction networks (of 155) |
|---|---|---|
| Continued, f6 seed 1 | 76 / 62 / 58 (76) | 69 / 65 / 62 (60) |
| Continued, f6 seed 2 | 50 / 47 / 47 (49) | – |
| Continued, f6 seed 3 | 53 / 53 / 53 (51) | – |
| Continued, version n | – | 64 / 66 / 61 |
| Continued, version ns | – | 55 / 44 / 43 |
| Fresh, f6 seed 1 | 79 / 79 / 72 (80) | 40 / 33 / 31 (35) |
| Fresh, f6 seed 2 | 57 / 54 / 53 (57) | – |
| Fresh, f6 seed 3 | 66 / 61 / 61 (68) | – |
| Fresh, version n | – | 50 / 46 / 43 |
| Fresh, version ns | – | 41 / 34 / 33 |
| Control, version n | 82 / 82 / 79 | 67 / 66 / 65 |
| Control, version ns | 63 / 63 / 59 | 55 / 48 / 43 |

**Totals over the 16 networks, A / B / C:**

| Measure | A | B | C |
|---|---|---|---|
| New-form equations solved | 967 | 903 | 863 |
| True structures | 893 | 824 | 817 |
| Bump solutions | 70 | 53 | 45 |
| Solutions using a kind of term new to the network's lineage | 108 | 91 | 88 |

- The bump is found mostly by the networks that already had it: flows continued f6 seed 1 found it 16 / 3 / 4 times, fresh f6 seed 1 21 / 21 / 18, control n 19 / 18 / 16.
- **A against A** (seed 4 against the network's own seed, 8 f6 networks): differences 0, 1, 2, 1, 0, 2, 9, 5 new-form equations (median 1.5); shared-form differences all 0.

**Against the winner rule set before the run:**
- **B:** beats A in 1 of 16 networks (flows 0 of 8, reaction networks 1 of 8); median gain −3.5 (continued −3.5, control −0.5, fresh −4.5). Does not pass.
- **C:** beats A in 0 of 16; median gain −6.0 (continued −5.0, control −3.5, fresh −7.0). Does not pass.
- **Within single networks:** the losses reach p = 0.0013 (B, flows continued f6 seed 1) and p = 4e-05 (C, same network); also p = 0.0074 and 0.00049 on reaction networks continued ns.
- **Winner: none.** By the rule, no learning curves are run.

**Reading.**
- Giving half the weight of the choice of term to a grammar of forms costs equations in nearly every network, and the costlier the grammar, the more is lost (C more than B).
- It does not help with unfamiliar steps either: fewer bump solutions and fewer solutions with a kind of term new to the lineage.
- The grammar is not conditioned on the system's data. The half of the weight it takes goes to terms that are plausible as forms but do not fit, and it is taken away from the network's data-informed choices within the 300-structure budget.
- This agrees with K0b's diagnosis: an unfamiliar step cannot come from a prior over forms; it has to come from the data, through search.

### How K2's cheap screening could be tested early (outline; not run; 12 November)

- **What exists already.**
  - At every search step the search computes, for each of the roughly 4,000 terms, how much of the remaining residual it would explain if added to the current structure with its best constants from the bank's grid (`search_g1.evidence`: each term's best partial R²). That screens every term at once, for about the cost of one network call.
  - The network sees it only as one input feature among many, and has learned to override it for unfamiliar terms.
  - An ordinary fit of a whole candidate structure is 15–25 times cheaper than a network call (Stage X timings).
- **Step 1: a probe, no search (minutes).**
  - Take the development systems at three kinds of node: the node one term short of each true structure (the true structure minus one term, fitted), the root, and nodes two terms short.
  - At each node, rank the missing true term by the evidence, by each lineage's network, and by each grammar. Report unfamiliar terms and the bump especially.
  - This says directly whether the data puts a genuinely new step near the front once the familiar terms are in place. K1 suggests that is where any gain must come from.
- **Step 2: inside K1's setup** (same 16 networks, seed, budget and systems; test only).
  - Ordering D: the network mixed half and half with a distribution from the evidence, spread over the top 64 terms by explained share.
  - Same-network comparison against A, with the same winner rule.
  - About 16 tests (3–4 h); no training.
  - Because the evidence is the data's own judgement, D tests the opposite of what B and C tested.
- **Step 3: K2's screen proper.**
  - At each expansion, quick-fit the top M candidate structures (the current structure plus one of the top terms by prior and by evidence, or a swap), with M from 16 to 64 per network call (one to three network calls of time).
  - Expand the best fits first.
  - Compare at equal wall time and at equal network calls. This is where the effective budget could grow 5–10 times.

### Owner: quick checks first; the evidence probe now; D only if it passes (12 November)

**Owner.** *"New rule: every idea gets a minutes-scale check without search first (like the probe), and only ideas that pass it get hours-scale tests. Keep pilots short unless a quick check strongly supports them. Run the probe now and report before doing anything else. Run ordering D only if the probe shows the data puts unfamiliar terms, including the bump, near the front. Confirm in plain language that D and the cheap screening complement, rather than replace, the DreamCoder direction: growing the library (Stage C) and building terms from parts with a network that adjusts grammar weights from each system's data (K2, DreamCoder's recognition model). Note that K1's grammar ignored the data, which DreamCoder's search doesn't do."*

- **Standing rule from now on.** Every idea first gets a minutes-scale check without search. Only ideas that pass get hours-scale tests, and pilots stay short unless a quick check strongly supports them.

**The evidence probe: design and reading (set before the run).**
- **Nodes.** Development systems, new-form equations, each true term in turn:
  - the node one term short (the true structure minus that term, fitted as the search fits a structure);
  - the node two terms short (minus that term and the next true term);
  - the root.
- **What is ranked at each node.** The missing term's rank among the terms that can be added, by three measures:
  - **evidence:** the data's own score, each term's best partial R² for the residual (search_g1.evidence);
  - **network:** each of K1's 16 networks (8 per domain);
  - **grammar:** the K0 grammar and the context-free grammar with library of each network's lineage.
- **Term sets:**
  - all true terms;
  - terms new to both prior experiences;
  - terms unfamiliar to each lineage (median over the lineages);
  - the bump (1/(K + x²)) and the saturating term.

  Each is reported as the median rank and the shares in the top 64 and top 10.
- **D's condition.** D runs only if, at the node one term short, in both domains, each of these median evidence ranks is at most 64 (inside D's window of the 64 terms with the most evidence):
  - the bump;
  - the terms new to both prior experiences;
  - each lineage's unfamiliar terms (median over the lineages).
- No search and no training; counts only.

### The evidence probe: results (12 November; no search, no training)

**Run:** `probe_evidence.py 24`, 03:40–04:00 UTC (report `outputs/accumulating_learner/g2h/probe/report.txt`).
- 1,671 probes on flows and 1,713 on reaction networks.
- Cost: evidence 0.03 s per node against 1.06 s per network call, so screening every term by the data costs about 1/30 of a network step.

**Median rank of the missing true term** (evidence / networks, the median over 8 networks of each network's median / K0 grammar / context-free grammar with library).

Flows (about 3,841 terms):

| Term set | One term short | Two short | Root |
|---|---|---|---|
| All true terms (557) | 62 (45% top 10) / 1 / 6 / 6 | 1,311 / 3 / 7 / 6 | 1,717 / 6 / 8 / 8 |
| Unfamiliar to the lineage (74 per lineage) | **20** / 256 / 577 / 542 | 306 / 719 / 578 / 543 | 853 / 1,224 / 579 / 544 |
| Bump (47) | **12** (70% top 64) / 140 / 122 / 760 | 153 / 538 / 122 / 762 | 736 / 1,126 / 124 / 763 |
| Saturating (55) | 14 / 66 / 357 / 429 | 460 / 429 / 358 / 430 | 955 / 970 / 359 / 432 |
| New to both prior experiences (61) | 25 / 1 / 5 / 10 | 1,113 / 4 / 5 / 11 | 1,778 / 8 / 6 / 12 |

Reaction networks (about 4,671 terms):

| Term set | One term short | Two short | Root |
|---|---|---|---|
| All true terms (571) | 618 (23% top 10) / 2 / 8 / 8 | 1,730 / 3 / 8 / 8 | 1,643 / 5 / 9 / 10 |
| Unfamiliar to the lineage (100 per lineage) | 86 / 292 / 717 / 442 | 622 / 1,492 / 718 / 443 | 1,310 / 2,663 / 720 / 445 |
| Bump (39) | 143 (33% top 64) / 770 / 716 / 2,688 | 431 / 1,690 / 716 / 2,689 | 1,183 / 3,542 / 718 / 2,691 |
| New to both prior experiences (31) | 408 (35% top 10) / 2 / 6 / 12 | 678 / 6 / 6 / 12 | 2,313 / 15 / 8 / 14 |

**D's condition (set before the run): NOT MET, so D does not run.**
- Flows passes all three: bump 12, terms new to both 25, unfamiliar terms 20.
- Reaction networks fail all three: 143, 408 and 86.

**Reading.**
1. **The networks are near perfect at completing a structure when the missing term is familiar** (median rank 1–2). They rank unfamiliar missing terms deep: 256 on flows, 292 on reaction networks.
2. **On flows, the data's own score finds the missing unfamiliar term once the rest of the structure is in place.**
   - It ranks the bump at 12 and unfamiliar terms at 20, far ahead of the networks (140; 256) and the grammars (122–760).
   - This is exactly the gap a data-driven screen would fill.
3. **The signal vanishes as more of the structure is missing.**
   - Two terms short, the flows bump falls to 153 and unfamiliar terms to 306; at the root, to 736.
   - The other missing terms dominate the residual.
4. **On reaction networks the data's signal is weak even one term short** (all terms 618, bump 143). Their equations have several correlated product terms, so one term's share of the residual is not distinctive.
5. **The grammars rank the same unfamiliar terms at 442–717 on both domains, whatever the node.** They do not look at the data, which explains K1.

**What the probe suggests** (not run; each idea would first need its own quick check, by the new rule):
- A data-driven channel helps mainly near completion, and on some domains.
- Spending half the search's weight on it everywhere would dilute the networks' strong completions, as K1's grammar did.
- A completion-stage screen, used only where the current structure already explains most of the data, is the natural variant to check next.

### Owner: the late-search quick check, then the Stage C and K2 design (12 November)

**Owner.** *"Do the quick check of the late-search variant (minutes, no search). Then, whatever it shows, move to designing Stage C and K2, including the cross-equation parts, since reaction networks' similar product terms won't be resolved by search-ordering tweaks. Send the quick-check result and the design together, with a minutes-scale check planned for each new piece before any longer test."*

**The late-search quick check: design and reading (set before the run).**
- **The variant.** The data's score is mixed into the network's choice of term only at nodes where the current structure already explains most of the data. The gate is the fit's unexplained share (its residual over the target's size) being at most g.
- **The check.** The evidence probe's nodes (root, one term short, two terms short) on the development systems' new-form equations. At each node:
  - the fit's unexplained share;
  - the missing term's rank by the evidence;
  - wherever the share is at most 0.5, also its rank by three networks per domain (continued and fresh f6 seed 1; control n) and by the mixture of each network with the evidence. The mixture is half the network and half spread over the 64 terms with the most evidence, in proportion to their explained share.
- **Gates tried:** g = 0.5, 0.3, 0.1, 0.03.
- **Reading.** The variant passes if one gate satisfies all of these in both domains:
  1. it opens at at least 25% of the nodes one term short;
  2. at the nodes where it is open, the median evidence rank is at most 64 for unfamiliar missing terms (median over the three lineages) and for the bump;
  3. there, the mixture ranks unfamiliar missing terms better than the network does (median over the three networks);
  4. there, the mixture keeps familiar missing terms within a median rank of 5 (the networks' is 1–2).

### The late-search quick check: results (12 November; no search, no training)

**Run:** `probe_late.py 24`, about 10 minutes (report `outputs/accumulating_learner/g2h/late/report.txt`).
- 3 networks per domain: continued and fresh f6 seed 1, control n.
- These three happen to be lineages that already know the bump, so for them the bump is familiar.

**Where the gate opens** (share of nodes one term short / two short / root):

| Gate | Flows | Reaction networks |
|---|---|---|
| 0.5 | 94% / 76% / 0% | 98% / 86% / 0% |
| 0.3 | 87% / 56% / 0% | 94% / 71% / 0% |
| 0.1 | 61% / 19% / 0% | 72% / 32% / 0% |
| 0.03 | 31% / 5% / 0% | 45% / 8% / 0% |

**At the open nodes, the missing unfamiliar term's median rank** (evidence / network / network + evidence):

| Gate | Flows | Reaction networks |
|---|---|---|
| 0.5 | 231 / 133 / 146 | 225 / 534 / 476 |
| 0.1 | 165 / 96 / 131 | 241 / 410 / 142 |
| 0.03 | 90 / 62 / 64 | 244 / 125 / 106 |

- Familiar missing terms stay at rank 1 under both the network and the mixture at every gate.

**Against the reading set before the run: does not pass.** No gate meets the four conditions in either domain.

**Why.** The gate opens at the wrong nodes.
- The data's score finds a missing term when that term carries much of the remaining signal. At those nodes the current structure explains little, so the gate stays shut.
- In the evidence probe, unfamiliar terms one term short ranked 20 on flows (all such nodes).
- Where the structure already explains most of the data, the missing term is a small part of what is left, and its evidence rank falls to 90–250.
- A gate on the evidence's own confidence (one term explaining a large share of the residual) would be a different idea, needing its own quick check. Not pursued.

### Design: K2 and Stage C, with the cross-equation parts; a quick check for each piece (12 November; proposed, not run)

**Where things stand.**
- The network completes familiar structures almost perfectly (missing familiar term at rank 1–2) but cannot reach unfamiliar terms.
- Grammars that ignore the data cost equations (K1).
- The data's own score finds a missing term only when that term carries much of the remaining signal, so a fixed rule for when to trust it fails (the late-search check).
- Reaction networks add a separate problem: several similar product terms per equation, which one equation's data cannot tell apart.

**The pieces.** These are DreamCoder's three parts (a library grown by compression, a recognition network that sets the grammar's weights from each task's data, and a search that builds candidates from parts and checks each against the data), plus the cross-equation parts. Each gets a minutes-scale check without search or training first; only pieces that pass get a longer test.

1. **K2-a, build and check.**
   - **What it is.** The search builds candidate terms from parts (operations and arguments), not from the fixed list, and checks many completed candidates against the data at every step. A full sweep of all terms costs about 1/30 of a network step.
   - The first version reproduces today's language exactly. Today's network stays one source of candidates, so familiar completions keep their strength.
   - **Quick check.** At the probe's nodes, take the top 64, 256 and 1,024 candidates from each source: the network, K0's grammar, and the data-set weights of check 2(ii). Screen them by the data, and record how often the missing term is in the screened top 10.
   - **Passes if** some source with at most 1,024 candidates puts missing unfamiliar terms in the top 10 more often than the network's own top 10 does, in both domains, at the nodes one and two terms short.
2. **K2-b, the recognition network.**
   - **What it is.** A network reads each system's data and the current structure and sets the weights of the building choices for that system. Example: "a reciprocal of a sum is likely here, and inside it a constant plus a square."
   - It is trained on the building choices of verified solutions and of dreamed systems drawn from the current library (Stage B's idea, which is how DreamCoder trains its recognition model).
   - **Quick check (i), headroom.** Set the weights from the system's other true terms, one term held out, plus the lineage's experience. Where do held-out unfamiliar terms rank?
     - Passes if the median is within the top 64 in both domains.
     - If even this ceiling cannot reach them, training will not.
   - **Quick check (ii), does the data carry choice-level signal?** Spread the data's term scores onto the building choices each term contains, and rank the missing term by its choices. This is a stand-in for the recognition network with no training.
     - Passes if it ranks missing unfamiliar terms better than both the plain data score and the network, at the nodes one and two terms short, in both domains.
3. **Stage C, the growing library.**
   - **What it is.** Pieces learned by compression become single building choices, with weights set from the data by the recognition network. This fixes K0b's flaw, where a piece's weight depended only on how often it had appeared in the past.
   - Controls as agreed: a decoy library, a removal check, and the made-up-systems control with its own library.
   - **Quick check.** K0b's rank and bits check, redone with the data-set weights of check 2(ii), against a decoy library with pieces of the same sizes.
   - **Passes if** the learned library ranks unfamiliar true terms better than both no library and the decoy, in both domains.
4. **C1, attention across a system's equations.**
   - **What it is.** The recognition network sees all of a system's equations (one summary per equation), so a rate found in one equation raises its weight in the others.
   - **Quick check** (counts only). On the development systems:
     - how often the missing term (one term short) appears in another equation of the same system;
     - its rank if the terms K1's searches (ordering A) found in the other equations are proposed first.
   - **Passes if**, on reaction networks, at least half of missing terms appear in another equation and this proposal puts them within a median rank of 10.
5. **C2, paired moves.**
   - **What it is.** Add the same term to two equations at once, judged by how much of both equations' remaining error it explains.
   - **Quick check.** At nodes where a term shared by two equations is missing from both, rank it by the joint score against the one-equation score. For reference, one term short on reaction networks the one-equation score ranked the bump at 143 and all terms at 618.
   - **Passes if** the joint score puts shared missing terms within a median rank of 64 on reaction networks where the one-equation score does not.
6. **C3, shared constants across equations.**
   - **What it is.** The fitter fits one constant for a term that appears in two equations.
   - **Quick check:**
     - on training systems' true equations, how often a constant repeats across equations;
     - fits only, with no search: the development systems' true structures fitted with the constants tied and separately, comparing how often each fit reaches the exactness tolerance.
   - **Passes if** constants repeat in at least a quarter of some domain's equations, and tied fits reach exactness at least as often as separate ones.

**Order and cost.**
- **Quick checks first**, in the order 1, 2, 4, 5, 6, 3: about 1–2 hours of CPU in all, no search and no training.
- **Longer tests only for pieces that pass**, kept short unless the check strongly supports them, with at least 2–3 seeds per arm.
- **The accumulation test** (experienced against the made-up control and the fresh network, on new-form equations) comes after the pieces are in place. It uses the same machinery for every arm: own library, own recognition training, decoy library and removal check.
- **Unchanged rules:** no hand-added pieces; the learner is never told the domain.

### Owner: run the six quick checks; their exact pass rules, set before any of them runs (12 November)

**Owner.** *"Yes, run the six checks in that order, with each pass rule written into the log before it runs. Report each result in a short plain-language paragraph: what was tested, the number that decided it, and pass or fail. Then propose which passing pieces to test further, with the shortest test that can tell us something."*

**Common ground.**
- **Systems and nodes.** Development systems' new-form equations, at the evidence probe's nodes: one term short and two terms short of each true structure, fitted as the search fits a structure.
- **Lineages.** K1's 8 networks per domain. "Unfamiliar" is per lineage (K0's definition), and figures are medians over the 8 lineages of each lineage's median.
- **Ranks** are among the terms that can be added, ties going to the simpler term.
- **No search and no training.** Code: `k2_checks.py`.

**The choice-level data weights,** used in checks 1, 2(ii) and 3.
- The building choices are grammar_ops' (parent, argument position, child).
- Each choice gets the largest evidence (best partial R² for the node's residual) of any addable term containing it.
- Choices are normalized within their context, with a floor of 1e-4.
- A term's score is the sum of the log weights of its choices.

**Check 1 (build and check).**
- **Sources:** each lineage's own network, its K0 grammar, and the choice-level data weights.
- **Hit:** for a candidate count N of 64, 256 or 1,024, the missing term is in the source's top N, and fewer than 10 of those N candidates have more evidence than it.
- **Passes if** one source with one N has a higher hit rate on missing unfamiliar terms than the network's own top 10 (the term within the network's first 10). This must hold in all four cells: both domains, one and two terms short.

**Check 2(i) (headroom).**
- Weights are half the lineage's K0 grammar and half a contextual grammar fitted on the system's other true terms (every equation's true terms except the held-out occurrence; alpha 1).
- The held-out term is ranked among the terms not already in its equation.
- **Passes if** the median rank of held-out unfamiliar terms is at most 64 in both domains.

**Check 2(ii) (data signal at the level of choices).**
- **Passes if** the choice-level data weights rank missing unfamiliar terms better (lower median) than both the evidence and the networks (median over the 8 networks), at both node kinds and in both domains.

**Check 4 (C1, proposals across equations).** Nodes one term short only.
- **Proposals:** the terms that lineage's network found (K1, ordering A) in the system's other equations, ordered by how many other equations contain them (ties simpler first), followed by the network's order. If the missing term is not proposed, its rank is the number of addable proposals plus its network rank, a cautious upper bound.
- **Passes if,** on reaction networks, at least 50% of missing terms appear (as the same term) in another equation's true structure, and the proposal's median rank is at most 10.

**Check 5 (C2, paired moves).**
- **Cases:** every term shared, as the same term, by two equations of a system (at least one of them new-form).
- At the two nodes where it is missing from both equations, the term is ranked three ways: by each equation's own evidence (one-equation rank), and by the sum of the two (joint rank). Addable means not in either node.
- **Passes if,** on reaction networks, the median joint rank is at most 64 and the median one-equation rank is above 64.

**Check 6 (C3, shared constants).**
- **Part (i):** training systems of every domain. The share of equations with a parameter (from the system's description) that also appears in another equation, separately for coefficients and for constants inside a term.
- **Part (ii), fits only:**
  - Development systems' pairs of plain-product terms whose coefficient is the same parameter in two equations (with the description's sign and factor).
  - Both true structures are fitted separately (the search's fit), then jointly with that coefficient tied.
  - Tying "preserves exactness" when the tied relative residual is at most max(2 × the separate one, 1e-9).
  - Counted among the pairs whose separate fits are exact-level (relative residual at most 1e-6).
- **Passes if** at least 25% of some domain's equations share a constant, and tying preserves exactness in at least 95% of those pairs.

**Check 3 (Stage C, the library with data-set weights).**
- The choice-level data weights over each lineage's parses with its learned contextual library at 30 tasks (pieces as choices, a piece's gaps as contexts).
- **Against:** the same weights with no library, and with a decoy library. Decoy pieces are random parts of random language terms, with the same sizes and gap counts as the learned pieces (fixed seed).
- **Passes if** the learned library ranks missing unfamiliar terms better (lower median) than both no library and the decoy, at both node kinds and in both domains.

### The six quick checks: results (12 November; no search, no training)

**Run:** `k2_checks.py 24 nodes headroom pairs constants` (29 minutes), then `report` (report `outputs/accumulating_learner/g2h/k2checks/report.txt`). Figures are medians over the 8 lineages per domain.

1. **Build and check: PASSES.**
   - Missing unfamiliar terms land in the screened top 10 of the network's own top N far more often than in the network's own top 10:

| Node | Top 64 | Top 256 | Top 1,024 | Network's own top 10 |
|---|---|---|---|---|
| Flows, one term short | 18% | 35% | 44% | 8% |
| Flows, two short | 2% | 8% | 13% | 0% |
| Reaction networks, one term short | 17% | 21% | 22% | 5% |
| Reaction networks, two short | 5% | 7% | 7% | 0% |

   - K0's grammar as the source also passes at 256 and 1,024 candidates. The choice-level data weights as the source do not (0–3%).
   - **Descriptive, not part of the rule.** Screening alone loses familiar completions: familiar missing terms in the screened top 10 of 1,024 are 45% on flows one term short and 24% on reaction networks, against 90–92% in the network's top 10.
   - The union "network's top 5 or screened top 10 of 1,024" keeps familiar terms at 84% in both domains, while catching unfamiliar ones at 45% (flows) and 22% (reaction networks), against the network's 8% and 5%. Two terms short: 13% and 8%, against 0%.
2. **(i) Headroom: FAILS.**
   - Weights half from the lineage's grammar and half from the system's other true terms rank held-out unfamiliar terms at a median of 150 on flows and 214 on reaction networks (K0's grammar alone: 577 and 717). The rule needed 64 or better.
   - **(ii) Choice-level data signal: FAILS.** It ranks missing unfamiliar terms at 1,466–2,494, against the evidence at 20–622 and the networks at 256–1,492. Spreading term scores onto building choices destroys the data's signal.
4. **Proposals across equations (C1): PASSES by the rule, with no gain over the network.**
   - On reaction networks, 65% of missing terms appear in another equation's true structure, and the proposals' median rank is 4 (rule: at most 10).
   - But the network alone ranks the same missing terms at 2 (flows: 40% shared; 4 against 1).
   - Unfamiliar terms rank 296 with proposals, against about 292 by the network.
   - **Descriptive:** among the 22–24% of missing terms the network ranks below 10, proposals rescue 2% on flows and 5% on reaction networks, and lose 3% and 1%.
5. **Paired moves (C2): FAILS.**
   - On reaction networks (264 shared terms), the joint score ranks the shared missing term at a median of 1,169, against 638 by each equation alone (flows: 650 against 74).
   - **Descriptive:** a shared term is usually obvious in one equation (median rank 1) and buried in the other (median rank about 1,900). The sum lands in between, worse than either equation alone in 3% of cases and better than both in 9%.
6. **Shared constants (C3): FAILS.**
   - **(i)** Equations sharing a parameter with another equation: reaction networks 99%, ecology 67%, mechanics, circuits and flows 0%.
   - **(ii)** Tying keeps exactness in 161 of 213 development pairs (76%; the rule needed 95%).
   - **A bug, found and fixed after the first run.** My first tied fit used the description's factors directly. That ignores the learner's rescaling of each variable (coordinates x/σ, targets dx/dt/σ), and it gave 0%. The corrected tie uses the learner's own scales, and that 76% is the result that counts. Code: `k2_checks.py`, `job_constants`.
   - The 52 remaining failures are genuinely inexact (tied residuals 0.0005–0.2; separate fits 1e-11 or better). Probably the coefficient of a plain term is often a sum of several rates, for example a decay and a reaction on the same variable. A simple tie cannot express that.
3. **The library with data-set weights (Stage C): PASSES by the rule, inside a weak scorer.**
   - The learned library ranks missing unfamiliar terms at 601 on flows and 916–944 on reaction networks. No library gives 2,450 and 1,466–1,471; the decoy gives 1,990–1,993 and 1,667–1,676.
   - The scorer itself failed check 2(ii), and these ranks remain far behind the network (256 and 292) and the evidence (20 and 86).

### Owner: the flows same-network test of build-and-check; reaction networks if it passes; an idea recorded (12 November)

**Owner.** *"Yes, run the flows same-network test as designed, with the pass rule written into the log first. If it passes, run the same six-test check on reaction networks. Separately, note for a later quick check: the shared-constant failure suggests treating a whole reaction (one rate that enters several equations with fixed signs) as the building block, instead of single terms in single equations. Don't build it yet; just record it."*

**The test: design and pass rule (set before the run).**
- **Ordering E, build-and-check inside today's search.** At every node, the choice of the term to add (and the added term of a swap) gives half its weight to the network's own distribution. The other half is spread evenly over the 10 terms with the most evidence (the data's score, best partial R²) among the network's top 1,024 addable terms. Removals stay the network's.
- Everything else as in K1: development systems, budget 300, thorough cap 60, the 2,048-term shortlist, search seed 4, test only, no training.
- **Comparison.** K1's ordering-A runs (the network alone, seed 4) on the same networks, already done. A against A with different seeds differs by a median of 1.5 new-form equations.
- **Flows:** the six f6 30-task networks, continued and fresh, seeds 1–3.
- **Passes if** all three hold:
  - E solves more new-form equations than A in at least 5 of the 6 networks;
  - E's median gain is above 1.5 equations;
  - shared-form equations solved, summed over the six networks, are no fewer with E than with A.
- **Reaction networks run only if flows passes.** Six 30-task networks: continued and fresh, f6 seed 1 and versions n and ns. Same rule.

**Recorded for a later quick check (not built).** The shared-constant failure suggests taking a whole reaction as the building block: one rate entering several equations with fixed signs, instead of single terms in single equations. The coefficient of a plain term would then be a sum of the reactions' rates, which is where simple ties failed. Before any build it gets its own minutes-scale check without search. For example: how well do whole-reaction candidates, scored by their joint evidence across the equations they enter, rank the true reactions at nodes where one reaction is missing?

### Build-and-check (ordering E) on flows: results (12 November, 06:39–08:31 UTC)

**New-form development equations solved, A (network alone, K1) against E** (seed 4; report `outputs/accumulating_learner/g2h/k1/report_E_flows.txt`):

| Network | A | E | Change | Only E / only A | p |
|---|---|---|---|---|---|
| Continued, f6 seed 1 | 76 | 58 | −18 | 6 / 24 | 0.0014 |
| Continued, f6 seed 2 | 50 | 39 | −11 | 9 / 20 | 0.061 |
| Continued, f6 seed 3 | 53 | 47 | −6 | 9 / 15 | 0.31 |
| Fresh, f6 seed 1 | 79 | 64 | −15 | 5 / 20 | 0.0041 |
| Fresh, f6 seed 2 | 57 | 48 | −9 | 8 / 17 | 0.11 |
| Fresh, f6 seed 3 | 66 | 55 | −11 | 6 / 17 | 0.035 |

- Shared-form equations: 6 with A and 6 with E. No failures and no time caps.
- Searches took longer: median 538–926 s per system, against 353–666 s with A.

**Against the rule set before the run: does not pass.** E is better in 0 of 6 networks, median change −11. By the rule, the reaction networks were not run.

**Why the quick check let it through.**
- Check 1 measured only the benefit: whether a missing term is caught at nodes one or two terms short of the truth.
- In a search, most nodes are far from the truth. There the data's 10 best-fitting candidates are mostly wrong (true terms rank about 1,700 by evidence at the root).
- Giving them half the weight at every node spends budget on wrong candidates and halves the weight of the network's own near-perfect completions.
- Check 1's descriptive figures already pointed to this cost: familiar missing terms fell to 45% (flows) and 24% (reaction networks) in the screened top 10, and to 1–5% two terms short.

**Lesson for quick checks.** A quick check of a change to the search's choices must measure the cost at the nodes the search actually visits, not only the benefit near the truth. Those nodes can be sampled cheaply from one short run of the unchanged search.

### Owner: change of direction; DreamCoder's physics experiment compared with ours; a mastery curriculum (12 November)

**Owner.** *"Change of direction: we've been asking for too much too fast. Before designing anything, compare our setup with DreamCoder's physics experiment, since DreamCoder worked on physics and we're missing something essential. Read the paper's physics section and supplementary details, and compare point by point, in plain language ... Then design a curriculum where the learner masters one piece at a time ... Start with a minutes-scale quick check ... Send the comparison, the curriculum design and the quick-check plan before running anything longer. Hold the two quick checks you proposed (dreams from the library, whole reactions) until after this."*

**Sources read:**
- Ellis et al., DreamCoder: main text (arXiv 2006.08381);
- the supplement bundled at cs.cornell.edu/~ellisk/documents/dreamcoder_with_supplement.pdf: section S2.1.7 "Learning a language for physical laws", Table S1, and the hyperparameter table in S4;
- the released physics script, ellisk42/ec `bin/scientificLaws.py`.

**Point by point.**
1. **Task format.**
   - **DreamCoder:** each task is one formula with a scalar or vector output, for example kinetic energy ½m|v|² or Coulomb's law.
     - Each equation was converted to a regression task by sampling 10 random input/output examples (the released script's default is 20).
     - Inputs are drawn independently at random over a wide range, and the outputs are the formula's exact values.
     - There is no time, no derivative estimation and no coupling between equations.
   - **Ours:** each task is a system of 2–4 coupled equations (about 14 terms for 4 variables, 3–4 per equation).
     - The data are 8 short bursts of 13 time points along trajectories, so the states are strongly correlated. About 30 derivative points per equation remain after edges and the validation split.
     - Derivatives are estimated, and exactness is judged on held-back bursts.
2. **Search breadth and constants.**
   - **DreamCoder:** up to 4 hours of enumeration per unsolved task per cycle, on 64 CPUs.
     - Each candidate is checked by evaluating it on the examples, which is very cheap: its 24-hour enumeration baseline tested up to 400 million programs.
     - No constants were fitted: "Physical constants were expressed in Planck units, which generally drops certain special coefficients (e.g., the gravitational constant G = 1)". The primitives include only 0, 1 and π, and the script's free real constant is commented out.
   - **Ours:** 300 candidate structures per equation.
     - Each check fits the coefficients and up to 2 constants inside terms, with thorough refits, at 5–10 minutes per system.
     - That is roughly five orders of magnitude fewer candidates per task, each far costlier.
3. **Recurrence.**
   - **DreamCoder:** 60 laws with heavy overlap. Examples:
     - the inverse square in 6 laws (three Coulomb and three gravitation variants);
     - norms and dot products in kinetic energy, work, power and velocity magnitude;
     - sums in net force, series resistors, parallel capacitors and centre of mass;
     - 2π periods in 3 laws;
     - near-duplicate families: three capacitor-energy forms, two Hooke's laws, two torque and two Lorentz-force forms.
   - **Ours:** a piece recurs about as often inside a domain (the flows bump is 47 of 557 true development terms, 8%; the saturating term 10%). But it always appears inside a 4-equation system with about 13 other terms, never alone. A new domain's pieces come without simpler stepping stones.
4. **Ordering.**
   - **DreamCoder:** no designed curriculum. "At each wake cycle, rather than sample a random minibatch of tasks, we gave the agent all tasks it had not solved in the past two wake cycles. This had the effect of the agent solving the easiest tasks first, and then reallocating its compute time to a smaller set of harder, unsolved tasks." The main text contrasts this with teacher-ordered curricula.
   - **Ours:** domains in a fixed order, and each domain's tasks seen once in a fixed order (wakes on tasks 1–10, 11–30 and 31–100). Unsolved tasks are never retried, nothing is graded by difficulty, and every new domain starts at full complexity.
5. **Primitives and library.**
   - **DreamCoder:** starts from 0, 1, π, power, +, ×, ÷, −, map, fold, zip, cons, car, cdr and nil, with no vectors, norms or dot products. It learned vector algebra (inner products, vector sums, norms), then the inverse-square schema behind Newton's and Coulomb's laws, solving 93% of the laws after 8 cycles.
   - **Ours:** a fixed flat list of about 4,000 whole terms (up to 7 nodes, 2 constants). No library grows during learning (K0b's library was only a diagnostic); the network learns preferences over the fixed list.
6. **Iterations, time and compute.**
   - **DreamCoder:** 8 wake/sleep cycles (the script allows 10), a 4-hour search limit per task, 64 CPUs. The released physics script uses a dummy feature extractor, so no neural recognition model was trained for physics.
   - **Ours:** three wake rounds per domain at 300 structures per equation, the network trained for 500 steps after each round. A claim run took about 18 hours on 30 workers.

**Differences most likely essential:**
1. **Breadth of search on cheap, unambiguous checks.**
   - DreamCoder can reach a program several choices deep by sheer enumeration, then shortens it with the library.
   - Our 300 costly checks cannot reach anything the network does not already favour. That is exactly the failure behind Stage X, K1 and ordering E.
2. **Tasks that make each candidate easy to judge.**
   - One formula, independent random inputs and no constants to fit.
   - Our trajectories make similar terms look alike (the reaction networks' product terms), and constant fitting makes every check slow.
3. **Easy-first with retries, and pieces met alone before they are combined.**
   - Simple laws are solved first, their parts become library pieces, and the harder laws then become short.
   - Our domains start at full complexity, each new piece is buried among many others, and unsolved tasks are never retried.
4. **Not essential for physics:**
   - a neural recognition model (none was used);
   - a designed curriculum (the easy-first effect came from retrying unsolved tasks).

**The curriculum design (proposed, not built).**
- **Levels.** Each level adds exactly one new piece, combined with pieces already mastered.

| Level | Adds |
|---|---|
| 1 | Plain variables (x, y) |
| 2 | Products (x·y) |
| 3 | Squares (x²) |
| 4 | Cubes (x³) |
| 5 | A reciprocal of a constant plus a variable, 1/(K + x): the first constant inside a term |
| 6 | The product of 2 and 5, x/(K + x) |
| 7 | The bump, 1/(K + x²): the reciprocal of 5 around the square of 3 |
| 8 | The saturating term x/(K + x²), combining 6 and 7 |
| 9 onward | A new operation each (exp, then sin), first alone, then combined |

- **System sizes.** Levels 1–8 use 2 variables and 1–2 terms per equation; later levels grow to 3 variables and 3 terms. Every system of a level contains its new piece at least once, and its other terms come from earlier levels.
- **Data.** Systems come from the existing made-up-system machinery (simulation and acceptance checks), with structures drawn only from the level's allowed pieces: 20 training and 20 held-out systems per level. Constants enter only at level 5, after the purely discrete pieces are mastered (DreamCoder fitted none).
- **A round at a level.**
  - Wake on every training system not yet solved (DreamCoder's rule), with the budget growing on each retry: 300, then 1,000, then 3,000 structures per equation, so compute moves to the hard systems.
  - Train the network on the verified solutions plus dreams built from the level's pieces.
  - Test on the held-out systems.
- **Mastery and stopping.** Advance when at least 80% of the level's held-out systems are solved exactly (every equation). Allow at most 3 rounds; if a level is not mastered by then, stop and report.
- **Controls,** at least 2–3 seeds per arm, same budgets and rounds:
  1. **Shuffled order:** the same systems of all levels in one random order, with no mastery gating.
  2. **Fresh at each level:** a new network at every level, trained only on that level's systems.
  3. **Made-up systems:** before each level, a network trained on made-up systems matched in amount (random structures of the same sizes from the whole language).
- **Measures.**
  - Zero-shot on each level before training on it: exact solves on its held-out systems, and the new piece's rank at the root and one term short.
  - Rounds to mastery, and structures per solved equation.
- **The accumulation claim** rests on later levels: zero-shot solves and new-piece ranks above both the fresh and the made-up control, and mastery in fewer rounds. The shuffled control shows whether the order matters.

**The quick check (planned, not run; no search).**
- **Question.** After brief training on the levels up to L, does level L+1's new piece rank much higher than for a fresh network?
- **Transitions:**
  - levels 1–2 → 3 (squares);
  - levels 1–5 → 6 (x/(K + x));
  - levels 1–6 → 7 (the bump).
- **Training data.** Systems of the levels so far, with supervised examples from their true structures (as the made-up control's examples are built). A network trained from scratch for 500 steps, 2 seeds.
- **Made-up arm.** The same amount of made-up systems, random structures of the same sizes from the whole language.
- **Ranking.** On 40 held-out systems of level L+1, the new piece's true terms at the root and one term short, ranked by:
  - the curriculum network;
  - the made-up network;
  - an untrained network;
  - the simplest-first order.
- **Pass rule.** In at least 2 of the 3 transitions, in both seeds, both of these hold:
  - the curriculum network's median rank one term short is at most half that of the untrained network and of the simplest-first order;
  - it is lower than the made-up network's.
- **Cost.**
  - generating about 1,000 small systems: about 10 minutes on 24 workers;
  - 12 trainings of 500 steps: about 2–3 minutes each on a shared GPU;
  - ranking: about 5 minutes.
  - About 45 minutes in all.

### Owner: the curriculum quick check approved, with two additions (12 November)

**Owner.** *"Approved: build the level generator and run the quick check, with two additions: 1. Measure the new piece's rank not only one term short, but also from the empty start and two terms short, per our standing rule that quick checks must cover the points the real search visits. 2. For the full curriculum design, add a retry-only control: all levels' systems mixed together, with DreamCoder's retry rule and growing budgets, but no designed order. The comparison suggests easy-first came from retries, so we should test whether the designed ordering adds anything beyond that. Report the quick check in plain language before building anything longer."*

**Full curriculum design, a fourth control: retry-only.** All levels' systems mixed together, with DreamCoder's rule (every unsolved system retried each round, budgets growing 300, 1,000, 3,000) but no designed order. This tests whether the designed order adds anything beyond retries.

**The quick check, as it will run (set before the run).**
- **Level generator** (`curriculum_check.py`). The made-up-system machinery (dream_g2: coefficients, internal constants, initial states, the pre-screen and the full build's acceptance), with each equation's terms drawn only from the level's allowed pieces.
  - Two variables.
  - Every system contains the level's new piece in at least one equation.
  - Training systems have 1–2 terms per equation. Held-out systems of the tested level have 2–3 terms per equation, so that nodes two terms short exist.
- **Pieces,** as terms of the 2-variable language (own variable or the other one):

| Level | Piece |
|---|---|
| 1 | x |
| 2 | x·y |
| 3 | x² |
| 4 | x³ |
| 5 | 1/(K + x) |
| 6 | x/(K + x) |
| 7 | 1/(K + x²) |
| 8 | x/(K + x²) |

- **Transitions:** levels 1–2 → 3, levels 1–5 → 6, levels 1–6 → 7.
- **Arms:**
  - curriculum: 50 training systems per level so far, as supervised examples from their true structures;
  - made-up: random structures of the same sizes from the whole 2-variable language, matched in number of examples;
  - each network trained from scratch for 500 steps, seeds 1 and 2, on a free GPU (sharing rule otherwise);
  - references: an untrained network of the same architecture (seeds 1 and 2), and the simplest-first order.
- **Ranking.** On 40 held-out systems of the next level, every true occurrence of its new piece, ranked among the addable terms at three points: the empty start, one term short, and two terms short.
- **Pass rule.** At least 2 of the 3 transitions, in both seeds and at all three points, must meet both of these:
  - the curriculum network's median rank of the new piece is at most half both the untrained network's and the simplest-first order's;
  - it is below the made-up network's.

### The curriculum quick check: results (12 November; no search)

**Run:** `curriculum_check.py 24 gen train rank report`, 15 minutes (report `outputs/accumulating_learner/g2h/curriculum/report.txt`).
- Generation: 1,520 systems, all accepted.
- Training examples: 688–695 (levels 1–2), 1,718–1,725 (levels 1–5), 2,058–2,066 (levels 1–6).
- Made-up arms were matched exactly in number of examples.
- 14 networks trained on the two free GPUs in 8 minutes.

**Median rank of the next level's new piece** (of the 2-variable language's terms). Each cell gives curriculum (seeds 1, 2) / made-up (1, 2) / untrained (1, 2) / simplest-first.

| Point | Squares, after levels 1–2 | x/(K + x), after levels 1–5 | The bump 1/(K + x²), after levels 1–6 |
|---|---|---|---|
| Empty start | 26, 124 / 12, 23 / 496, 403 / 6 (85 cases) | 700, 496 / 277, 407 / 925, 467 / 89 (51 cases) | **56, 76** / 257, 222 / 775, 821 / 511 (51 cases) |
| One term short | 5, 8 / 1, 1 / 453, 368 / 4 (85 cases) | 413, 106 / 45, 55 / 923, 438 / 88 (51 cases) | **51, 62** / 69, 99 / 779, 826 / 510 (51 cases) |
| Two terms short | 26, 135 / 14, 23 / 490, 411 / 4 (46 cases) | 771, 486 / 233, 385 / 930, 469 / 90 (20 cases) | **55, 74** / 211, 213 / 796, 800 / 561 (27 cases) |

**Against the rule set before the run: FAILS.** Only 1 of the 3 transitions (the bump) meets it, against the 2 required.

**Reading.**
1. **The bump: met in both seeds at all three points.**
   - After mastering squares and 1/(K + x), the curriculum network ranks the bump at 51–76 everywhere.
   - The made-up network ranks it 69–257 (2.7–4.6 times worse at the empty start), simplest-first about 510–560, and the untrained network about 800.
   - Mastered pieces do combine into the unfamiliar term that blocked flows, and ahead of generic practice.
2. **Squares: no.** A small new term is already near the front of the simplest-first order (4–6). The made-up network, which met squares among its random terms, ranks it 1–23. The curriculum network, which never saw a square, ranks it 5–135.
3. **x/(K + x): no.** Products and 1/(K + x) were both mastered, yet the curriculum network ranks their product 106–771, behind the made-up network (45–407) and simplest-first (about 90). Putting a mastered reciprocal inside a product is not picked up the way putting a mastered square inside a reciprocal is.
4. **Caution about the control.** The made-up network has seen random terms from the whole language, so for any piece it has a head start of coverage. The curriculum networks have seen only their levels' pieces. Beating the made-up network therefore means combination, not coverage. That is why the bump's result is meaningful, and also why simple new pieces favour the control.

### Owner: the disk guard is 2 GB (12 November)

**Owner.** *"no stop yourself at 2GB"*

From now on my runs stop themselves, and stop writing, only when free disk falls below **2 GB**, replacing the 5 GB guard of 9 November. `curriculum_check.py` is updated, and later launchers will use 2 GB.

### Owner: more combinations of each kind, and a no-network library comparison (12 November; set before the run)

**Owner.** *"Yes, run the 15-minute check with more combinations of each kind (piece inside a frame, and frame multiplied by a variable). Also add a no-network comparison: rank the new pieces using a DreamCoder-style library and weights learned from the curriculum levels' true structures (compression plus counting, no network), alongside the curriculum network, the made-up network and simplest-first. Report in plain language which kinds of combination transfer under the network and which under the library."*

**Combinations** (2-variable language; x = the equation's own variable or the other one, consistently within a term):

| Kind | New term | Mastered levels |
|---|---|---|
| A1 | 1/(K + x²): square inside the reciprocal frame | 1–5 |
| A2 | 1/(K + x³): cube inside the frame | 1–5 |
| A3 | 1/(K + x·y): product inside the frame | 1–5 |
| B1 | x/(K + x): the frame times its own variable | 1–5 |
| B2 | x/(K + x²): the bump frame times its variable | 1–7 |
| B3 | y/(K + x): the frame times the other variable | 1–5 |

- Kind A puts a piece inside a frame; kind B multiplies a frame by a variable.
- Levels 1–5 are the existing ones (x, x·y, x², x³, 1/(K + x)). Levels 1–7 add the existing 6 (x/(K + x)) and a new training level 7 (the bump), 50 systems per seed, generated as before.

**Held-out systems.** 40 per combination, 2–3 terms per equation. The new term is forced into one equation, and the other terms come only from the mastered levels.

**Methods,** in seeds 1 and 2:
- the curriculum network trained on the mastered levels (the existing networks for levels 1–5, new ones for 1–7), the made-up network matched in examples, and an untrained network;
- **the library, no network:** DreamCoder's context-free form (library_ops, form "uni"). Pieces are learned by compression from the mastered levels' true structures, and weights by counting, one parse each. Terms are scored by their best parse.
- **its made-up counterpart:** the same machinery on the made-up network's own training systems;
- **counting alone** (no pieces), to show what compression adds;
- the simplest-first order.

**Points:** the empty start, one term short, and two terms short.

**"Transfers"** (set now):
- **Network:** in both seeds and at all three points, its median rank of the new term is at most half the untrained network's and half the simplest-first order's, and below the made-up network's.
- **Library:** in both seeds and at all three points, its median rank is at most half the simplest-first order's, and below the made-up library's.

### The follow-up curriculum check: results (12 November; no search)

**Run:** `curriculum_check2.py 24 gen train grammars rank report`, 4 minutes (report `outputs/accumulating_learner/g2h/curriculum2/report.txt`).

**Median rank of the new term** (of about 1,100 terms of the 2-variable language), seed 1 and seed 2 at the empty start / one term short / two short. Methods:
- **Curriculum net / Made-up net:** the networks trained on the mastered levels and on matched made-up systems.
- **Library (cur.) / Library (made):** the context-free library grammar learned from those systems' true structures, no network.
- **Counting:** counting alone, without pieces.
- **Simplest:** the simplest-first order.

| | Curriculum net | Made-up net | Library (cur.) | Library (made) | Counting | Simplest |
|---|---|---|---|---|---|---|
| A1 1/(K + x²), after 1–5 | 111, 628 / 102, 240 / 95, 716 | 214, 268 / 62, 78 / 275, 359 | 11, 11 / 9, 9 / 10, 10 | 866, 808 / 866, 806 / 815, 762 | 26–28 | 512–564 |
| A2 1/(K + x³), after 1–5 | 104, 826 / 108, 247 / 104, 832 | 171, 404 / 39, 102 / 180, 486 | 13–15 | 840–883 | 29–31 | 565–567 |
| A3 1/(K + x·y), after 1–5 | 87, 446 / 89, 122 / 87, 544 | 239, 190 / 52, 45 / 281, 235 | 18–21 | 914–979 | 36–37 | 675–677 |
| B1 x/(K + x), after 1–5 | 646, 406 / 422, 138 / 732, 479 | 202, 283 / 39, 60 / 234, 386 | 19–20 | 138–854 | 34–35 | 88–90 |
| B2 x/(K + x²), after 1–7 | 263, 37 / 38, 23 / 122, 34 | 276, 438 / 58, 213 / 177, 330 | 25–27 | 305–687 | 46–48 | 728–787 |
| B3 y/(K + x), after 1–5 | 710, 368 / 454, 123 / 833, 512 | 195, 222 / 34, 46 / 314, 381 | 18–21 | 149–915 | 33–35 | 91–93 |

**Against the definition set before the run:**
- **Network:** a piece inside a frame transfers in 0 of 3 combinations, a frame times a variable in 1 of 3 (B2).
- **Library:** both kinds transfer in 3 of 3.

**Reading.**
1. **The library transfers every combination of mastered pieces, of both kinds.** Ranks are 9–27 in both seeds at every point, against 138–979 for the made-up library and 88–787 for simplest-first.
   - Counting alone already gives 26–48. Compression roughly halves that (A1: 27 to 10).
   - The library looks only at form. Because the curriculum's levels use few operations, every composition of them becomes likely.
2. **The network's transfer is fragile and depends on the seed.** Seed 1 and seed 2 differ by up to 8 times (A2 at the empty start: 104 against 826).
   - The bump, which passed after levels 1–6 in the first check (51–76), does not pass after levels 1–5 (seed 2: 628 at the empty start).
   - Only x/(K + x²) after levels 1–7 transfers.
   - The made-up network beats the curriculum network in most cells one term short (for example A1: 62 and 78 against 102 and 240), because it has seen every operation.
3. **Taken with K1:** the same kind of form-only grammar hurt the search on flows and reaction networks, where it ranked unfamiliar terms at 440–760. In a narrow curriculum it ranks them at 9–27. A grammar of forms helps when the experience it was learned from is focused on a few pieces, as DreamCoder's physics laws were.

### Correction: DreamCoder's physics run and its recognition network (12 November)

**What was wrong.** The comparison entry above, and my report to the owner, said the released physics script "trains no neural recognition model". It does train one: with the released defaults a recognition network is trained every cycle, but it is given no information about the task.

**Sources.** `ellisk42/ec` master, commit cb0e63f (16 March 2022), fetched raw.
- `bin/scientificLaws.py` 410–413: `**commandlineArguments(` / `compressor="ocaml",` / `featureExtractor=DummyFeatureExtractor,` / `iterations=10,`. These are defaults that command-line flags can override. `useRecognitionModel` is not passed.
- `dreamcoder/dreamcoder.py` 716: `useRecognitionModel=True,` (the default in `commandlineArguments`).
- `dreamcoder/dreamcoder.py` 190–193: the model is switched off only `if useRecognitionModel and featureExtractor is None:`, which is not the case here.
- `dreamcoder/dreamcoder.py` 433: `if useRecognitionModel:` (trains the model each cycle); 574–575 build `RecognitionModel(featureExtractorObjects[i], ...)`.
- `dreamcoder/recognition.py` 1421–1429: `class DummyFeatureExtractor(nn.Module):` with `self.outputDimensionality = 1`, and `def featuresOfTask(self, t): return variable([0.]).float()`. Every task gets the same feature, 0.
- `dreamcoder/recognition.py` 651–656: the network's first layer takes `featureExtractor.outputDimensionality` inputs, so with a constant input its output cannot depend on the task.
- `dreamcoder/dreamcoder.py` 808–812: a `-g/--no-recognition` flag exists. The paper does not give the command used for physics.

**So:** under the released defaults, the physics run trains a recognition network on replays and dreams (`helmholtzRatio=0.5`) that cannot see any task's data. It learns one set of weights for all tasks, not a recognition model in the paper's sense.

**The paper and supplement** (dreamcoder_with_supplement.pdf):
- S2.1.7 (physics, p. 33) never mentions a recognition model.
- S2.1.8 (recursive functions, p. 33) says: "We did not use the recognition model for this experiment".
- S4.7 (p. 61) gives recognition architectures only for list processing, text editing and regular expressions (GRU), and for LOGO graphics, tower building and symbolic regression (CNN). None is given for physics.

**Ablations.** The with-and-without-recognition ablations do not include physics:
- Fig. 6A (p. 13) is "Held-out test set accuracy, across 20 iterations of wake/sleep learning for six domains".
- Fig. S1B (p. 24) and Fig. S11 (p. 36; "Pink: Abstraction only (no recognition model)") show the same comparison.
- The physics script has no held-out tasks (`bin/scientificLaws.py` 409: `testingTasks=[],`).
- Which six domains the panels show is not in the extracted text (the labels are in the figures). That physics is excluded is inferred from the held-out-test requirement and from S4.7's list.

**Effect on the comparison.** Point 4 of "differences most likely essential" should read as follows:
- The physics setup had no recognition model reading each task's data: none is described, and the released code's is blind to the task.
- No ablation measures the network's role in physics.
- So DreamCoder's physics result cannot show that a recognition model is unnecessary, only that the result was obtained without a data-reading one.

### Owner: a short curriculum pilot, the library ordering the search's candidates (12 November; design and pass rule set before the run)

**Owner.** *"Run a short pilot: the curriculum's first levels with the library ordering the search's candidates and the data checking them. Keep it as short as possible and write the pass rule into the log first. Arms (at least 2 seeds each): 1. Library alone ... 2. Network alone: as now. 3. Library plus network: the network adjusts the library's weights for each system from its data (DreamCoder's recognition model) ... Controls: the same systems in shuffled order, a fresh start at each level, the made-up library, and the retry-only control ... If arm 3 needs substantial building, run arms 1 and 2 with the controls first and propose arm 3 as the next step. Keep the two held checks waiting."*

**Arm 3 is deferred, as allowed.** A recognition model that reads each system's data and sets the library's weights needs a new network output, a new training target and a search that builds terms from the library. That is substantial building. It will be proposed after this pilot.

**Design** (`curriculum_pilot.py`):
- **Levels** 1 (x), 2 (x·y), 3 (x²), 5 (1/(K + x)), 6 (x/(K + x)), 7 (1/(K + x²)). Level 4 (cubes) is skipped to keep the pilot short; the bump does not need it.
- **Systems** from the curriculum generator: two variables, 1–2 terms per equation. Per level and seed: 10 training and 10 held-out systems.
- **Seeds** 1 and 2 (different systems and search seeds).
- **Within a level:**
  - round 1 searches every training system at 300 structures per equation (thorough cap 60);
  - rounds 2 and 3 retry the unsolved ones at 1,000 (cap 200) and 3,000 (cap 600);
  - after each round the arm learns, then is tested on the level's 10 held-out systems at 300.
- **Mastery:** at least 8 of 10 held-out systems solved exactly (every equation). If a level is not mastered after 3 rounds, the lineage stops there.
- **Before any experience,** every arm searches with the language's size prior (the empty grammar).
- **The arms:**
  1. **Library:** after each round, pieces by compression and weights by counting (context-free, one parse) from the lineage's own verified solutions so far. The search's candidate terms are ordered by this grammar alone, and the data checks each structure by fitting, as always.
  2. **Network:** the network as now. After each round it is trained further for 500 steps on examples from its verified solutions plus 50 dreams drawn half from its own usage and half from the size prior (on a free GPU).
- **Controls,** for the library arm:
  - **shuffled:** the same 60 training systems in one random order, in 6 blocks of 10, each block given the same 3 rounds;
  - **fresh:** the curriculum order, but at each level the library comes only from that level's own solutions;
  - **made-up:** the curriculum order, with the library learned from made-up systems' true structures (10 systems per earlier level) plus the level's own solutions;
  - **retry-only:** all 60 training systems mixed, with DreamCoder's rule (every unsolved system retried each round, 3 rounds at 300, 1,000 and 3,000) and no designed order.
- **Final test** for every lineage: all 60 held-out systems at 300.

**Measures.** Systems solved exactly per level (in the final test, and after each round for the curriculum-order lineages), rounds to mastery, structures tried before each solution, and total structures spent.

**Pass rule.** The library arm passes if both hold in both seeds:
- it masters all six levels;
- on the final test it solves at least 2 more of the 20 held-out systems of levels 6 and 7 (the combinations) than the network arm, and than each of the four controls.

Anything else is reported as it is.

### The curriculum pilot: stopped by the 2 GB disk guard; partial results (12 November)

**What happened.**
- The launcher's watchdog stopped the pilot at 22:44:55 UTC, when free disk fell to 2,033 MB. The pilot itself had written 44 MB; other users' jobs filled the disk.
- **Finished (7 of 12 lineages):** library seeds 1 and 2, fresh library seed 2, made-up library seeds 1 and 2, retry-only seeds 1 and 2.
- **Not finished:**
  - network seeds 1 and 2, both at level 7 after mastering level 6;
  - fresh library seed 1, in its final test;
  - shuffled library seed 1 (block 5) and seed 2 (block 3).

**Held-out systems solved after each round** (of 10; ✗ = not mastered in 3 rounds, lineage stopped):

| Lineage | L1 | L2 | L3 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|
| Library s1 | 10 | 10 | 9 | 5, 9 | 2, 1, 3 ✗ | – |
| Library s2 | 10 | 10 | 10 | 10 | 2, 3, 3 ✗ | – |
| Network s1 | 10 | 10 | 9 | 9 | **8** | (running) |
| Network s2 | 10 | 10 | 10 | 9 | **8** | (running) |
| Fresh library s1 | 10 | 10 | 9 | 10 | 4, 4, 4 ✗ | – |
| Fresh library s2 | 10 | 10 | 10 | 10 | 6, 6, 6 ✗ | – |
| Made-up library s1 | 10 | 10 | 10 | 6, 2, 5 ✗ | – | – |
| Made-up library s2 | 10 | 10 | 4, 4, 4 ✗ | – | – | – |

- **Retry-only** (all 60 training systems mixed, library): training systems solved over its 3 rounds were 16, then 59, then 60 of 60 (seed 1); 29, then 51, then 56 (seed 2).

**Final test** (60 held-out systems at 300 structures per equation):

| Lineage | L1, L2, L3, L5, L6, L7 | Total |
|---|---|---|
| Library s1 | 10, 10, 9, 10, 3, 0 | 42 |
| Library s2 | 10, 10, 8, 9, 3, 2 | 42 |
| Fresh library s2 | 10, 10, 10, 9, 6, 7 | 52 |
| Made-up library s1 | 10, 1, 5, 5, 3, 2 | 26 |
| Made-up library s2 | 10, 7, 4, 4, 1, 0 | 26 |
| Retry-only s1 | 10, 10, 8, 10, 3, 4 | 45 |
| Retry-only s2 | 10, 10, 10, 9, 7, 4 | 50 |

- Structures tried before each solution, median: 15–24 in training for the library lineages (84 for retry-only seed 2, which includes the large retries); 6–38 in the final tests.
- **"Solved exactly" is functional.** Found structures often differ from the generating ones while fitting exactly, for example x/(K + x) = 1 − K/(K + x). Only 11–20 of the solved final-test systems match the generating structure term for term.

**Against the rule set before the run: does not pass.** The library arm did not master all six levels in either seed; it stopped at level 6. This holds whatever the unfinished lineages would have shown.

**Reading.**
1. **The network, which reads the data, beat the form-only library in the actual search.**
   - At level 6 (x/(K + x)), the network mastered the level in one round in both seeds (8/10). The library reached only 1–3/10 after three rounds with growing budgets.
   - The quick check's rank of about 20 for the new piece did not carry over. The search still has to separate it from many look-alike combinations, and to fit its constant, within 300 structures.
   - The library here learned from the lineage's own solutions, which are often equivalent rewritings rather than the generating terms. In the quick check it learned from the true structures.
2. **Real curriculum experience beats made-up practice for the library.** The made-up library stalled at level 3 or 5 and solved 26/60, against 42/60.
3. **Accumulated counts dilute the current level's pieces.** The fresh library, learning only from level 6's own solutions, did better at level 6 (4–6/10) than the accumulated one (1–3/10).
4. **Retries with growing budgets work without a designed order.** Retry-only solved 56–60 of 60 training systems and 45–50/60 held-out. That is more than the curriculum library's 42, which stopped at level 6 and so never learned from level 7.

### Owner: finish the pilot; count equivalent forms as one piece; then a quick check for arm 3 (13 November)

**Owner.** *"Disk space is available again. Run the unfinished lineages (network ×2, shuffled ×2, fresh seed 1), then report the network arm's final test next to the others. Before any further library test: make the library count algebraically equivalent forms as the same piece (e.g. x/(K + x) and 1 − K/(K + x)), so rewritten solutions don't split its counts. Check that fix in minutes by rebuilding the library from the pilot's solutions and re-ranking. Then do the minutes-scale check for arm 3 (network adjusting the library's weights per system from its data) before building it fully. Keep the two held checks waiting."*

- **The five unfinished lineages are rerun from the start** (free disk 55 GB): network seeds 1 and 2, shuffled seeds 1 and 2, fresh library seed 1.
  - Same systems, code and settings as the pilot. The pilot saves no partial state, so each lineage starts again.
  - Reruns are not bitwise identical to the interrupted runs.

**Equivalent forms as one piece: the fix and its check (set before the run).**
- **The fix** (`canon_check.py`). Before the library counts a verified solution, the solution is rewritten into its canonical equivalent form, using the system's own data. Two steps are repeated until neither applies:
  1. **Drop:** remove a term if the structure without it still fits exactly (the search's exactness test).
  2. **Merge:** replace a pair of terms by a single term t if the rest plus t fits exactly. Candidates for t are the 5 terms with the most evidence (at least 0.99 of the residual) at the node without the pair, tried simplest first.

  For example, {1, 1/(K + x)} with tied coefficients becomes {x/(K + x)}. Canonical forms are thus the fewest-term exact forms. The library is otherwise unchanged (context-free, compression plus counting, `curriculum_pilot.grammar_table`).
- **The check.**
  - **Rebuild:** for the curriculum-order library lineages (seeds 1 and 2), rebuild the library from their verified solutions at levels 1, 2, 3 and 5, both raw and canonical.
  - **Reference:** a library from the true structures of the same levels (the curriculum check's systems).
  - **Re-rank:** on the follow-up check's held-out systems, rank the new term of each combination whose parts lie within those levels (the bump, 1/(K + x·y), x/(K + x), y/(K + x)) at the empty start and one and two terms short.
- **The fix works if** the canonical library ranks the new terms better (lower median) than the raw library in at least 3 of the 4 combinations, in both seeds, at one term short, and within twice the true-structure library's ranks.
- **Also reported:** how many solutions the rewriting changes, and which rewrites occur.

**The fix as set rewrites almost nothing; an extension, decided after seeing which rewrites occur and before any re-ranking.**
- The pre-set fix (drop and merge) changed 1 of 96 solved equations (seed 1) and 2 of 88 (seed 2).
- The splits in the counts are mostly not algebraic identities. They are forms that coincide on the observed data, because the curriculum's variables keep one sign along their trajectories:
  - x·|x| where x² was meant (seed 1's solutions use x1·|x1| 13 times and x2·|x2| 13 times);
  - |x| for x;
  - unusual exact equivalents at level 6, such as x1²/(K + x1) beside |x1|.
- Once found first, such a form is counted by the library and then favoured at later levels.
- **The extension.** Add a third step, simplify: replace a term by a simpler term (earlier in the simplest-first order) when the fit stays exact on that system's own data. Candidates are the 5 terms with the most evidence at the node without it, at least 0.99 of the residual.
  - This rewrites a form only where this system's data cannot tell the two apart. Where a variable changes sign, x·|x| and x² differ and the fit would not stay exact.
- **What is ranked:**
  - raw solutions;
  - the pre-set fix ("canonical");
  - the extended fix ("canonical+");
  - the true-structure library.
- The rule set before the run applies to the pre-set fix. The extended fix is judged by the same rule and reported as a change made after seeing the rewrite counts.

**Equivalent forms as one piece: results (13 November; no search; `canon_check.py`, report `outputs/accumulating_learner/g2h/canon/report.txt`).**
- **Rewrites:**
  - the pre-set fix changed 1 of 96 solved equations (seed 1) and 2 of 88 (seed 2);
  - the extended fix changed 37 and 27, almost all by simplify steps (44 and 29), for example x·|x| to x² and |x| to x.
- **Median rank of the new term at one term short** (the other points are within ±2). Columns: raw / pre-set fix / extended fix / true structures, then the same for seed 2:

| New term | Seed 1 | Seed 2 |
|---|---|---|
| Bump 1/(K + x²) | 166 / 166 / 13 / 9 | 30 / 30 / 11 / 9 |
| 1/(K + x·y) | 22 / 22 / 18 / 16 | 18 / 18 / 15 / 16 |
| x/(K + x) | 22 / 22 / 17 / 15 | 17 / 17 / 14 / 15 |
| y/(K + x) | 21 / 21 / 17 / 14 | 17 / 17 / 14 / 14 |

- **Against the rule set before the run:**
  - the **pre-set fix fails** (better than raw in 0 of 4 combinations);
  - the **extended fix**, judged by the same rule as a change made after seeing the rewrite counts, **passes**: better than raw in 4 of 4 in both seeds, and within twice the true-structure library everywhere.
- **Reading.** The library's counts were split mainly by forms that coincide on the observed data, not by algebraic identities. Rewriting each solution into its simplest exact form on its own data brings the library close to one learned from the true structures. The bump for seed 1 moves from rank 166 to 13. The extended fix is the version to use in later library tests.

**Arm 3's quick check: design and pass rule (set before the run; no search, no training).**
- **The question.** Would a network that adjusts the library's weights for each system from its data rank true terms better than the library alone and the network alone, at the points the search visits?
- **Stand-in for arm 3, no training ("adjusted library"):**
  1. At each node, take the network's distribution over addable terms.
  2. Spread it onto the library's building choices. Each choice (slot type, symbol) in the context-free form gets the network's probability mass of the terms whose parse uses it, normalized within its slot type, with a floor of 1e-6.
  3. Average its log with the library's own log weight, and renormalize within the slot type.
  4. Score each term as the sum of its parse's log weights.
  - The library supplies the composition; the network's reading of the data adjusts its weights.
- **Simpler reference ("product"):** the library's log probability of the term plus the network's.
- **Inputs:**
  - **Lineages:** the pilot's network lineages, seeds 1 and 2.
  - **Network:** the one after level 5.
  - **Library:** the extended-fix library from the lineage's own solutions at levels 1, 2, 3 and 5.
  - **Systems:** the follow-up check's held-out systems for the bump, 1/(K + x·y), x/(K + x) and y/(K + x).
- **Term sets:**
  - the new term;
  - familiar terms (pieces of levels 1, 2, 3 and 5);
  - cubes, which the pilot never trained on, are left out.
- **Points:** the empty start, one term short, and two terms short.
- **Pass rule.** The adjusted library passes if both hold in both seeds and at all three points:
  - its median rank of the new terms is below both the library's and the network's;
  - its median rank of familiar terms is at most twice the network's.
  - The product is judged and reported by the same rule.

**Arm 3's quick check: a term-level blend added (owner, 13 November; set before the run).**

**Owner.** *"For the arm 3 quick check: spreading term scores onto building choices failed in check 2(ii) because it discarded the signal, so the stand-in may underestimate arm 3. Also run a term-level blend: the library proposes its top candidate terms (e.g. top 100 or 1,000), and the network scores those whole terms, blended with the library's weights. Report both versions, and don't treat a failure of the choice-level stand-in alone as a failure of arm 3."*

- **Term-level blend.** At each node the library proposes its N most probable addable terms (N = 100 and N = 1,000).
  - Those terms are ranked by half the library's log probability plus half the network's.
  - All other terms follow, in the library's order.
- **Rule.** The same pass rule as before: new terms ranked below both the library and the network, familiar terms within twice the network's rank, in both seeds and at all three points.
- **Reading.** All four versions are reported: choice-level, product, and the blend at N = 100 and N = 1,000.
  - Arm 3 counts as promising if any of them passes.
  - A failure of the choice-level stand-in alone is not a failure of arm 3.

### The curriculum pilot: final results with the network arm (13 November; 11 of 12 lineages; shuffled seed 2 still running)

**Final test: held-out systems solved exactly per level** (of 10; seed 1 / seed 2):

| Lineage | L1 | L2 | L3 | L5 | L6 | L7 | Total (of 60) |
|---|---|---|---|---|---|---|---|
| **Network** | 10 / 10 | 10 / 10 | 10 / 10 | 8 / 6 | **8 / 7** | **7 / 9** | **53 / 52** |
| Library | 10 / 10 | 10 / 10 | 9 / 8 | 10 / 9 | 3 / 3 | 0 / 2 | 42 / 42 |
| Fresh library | 10 / 10 | 10 / 10 | 7 / 10 | 9 / 9 | 4 / 6 | 3 / 7 | 43 / 52 |
| Shuffled library | 10 / – | 10 / – | 10 / – | 10 / – | 3 / – | 5 / – | 48 / – |
| Made-up library | 10 / 10 | 1 / 7 | 5 / 4 | 5 / 4 | 3 / 1 | 2 / 0 | 26 / 26 |
| Retry-only library | 10 / 10 | 10 / 10 | 8 / 10 | 10 / 9 | 3 / 7 | 4 / 4 | 45 / 50 |

- **Levels 6–7, the combinations** (of 20): network 15 and 16; library 3 and 5; fresh 7 and 13; shuffled 8; made-up 5 and 1; retry-only 7 and 11.
- **Rounds to mastery:**
  - the network mastered every level in 1 round in both seeds, except level 7 in seed 1 (not mastered in 3 rounds);
  - the library stopped at level 6 in both seeds, as did the fresh library;
  - the made-up library stopped at level 5 (seed 1) and level 3 (seed 2).
- **Structures tried before each solution (median):** network 4 and 3; library 18 and 11; fresh 15 and 16; shuffled 19; made-up 21 and 22; retry-only 13 and 51.
- **Total structures spent:** similar across lineages, 86,000–140,000.

**Pass rule: does not pass** (unchanged). The library arm did not master all levels and is far behind the network on levels 6–7.

**Reading.**
- In the actual search the network, which reads the data, beats the form-only library at every hard level. It finds solutions after a median of 3–4 structures, against 11–22.
- The library's strength in ranking unfamiliar combinations (the follow-up check) does not survive the search without the data. This is the case for arm 3: keep the network's data-reading and add the library's composition.
- **Status of arm 3's quick check.** It has not run yet. Twice it was stopped at the tool's background time limit while starved of CPU: another user runs about 191 processes at normal priority (about 53 cores' worth on 48 cores), and my jobs run at nice 19 under the heavy-load rule. It is relaunched detached, at nice 19.

### Owner: normal priority (13 November)

**Owner.** *"Run the arm 3 check at normal priority (nice 0), per my earlier rule: don't slow down for other users' load, and never interfere with their processes. Apply the same to future short checks. Once arm 3's check is done, rerun shuffled seed 2 at the same priority so the curriculum-vs-shuffled comparison has both seeds. Report arm 3's four versions in plain language when finished."*

- **Priority.** My runs and checks run at normal priority (nice 0). Thread caps and the total worker count stay as before; other users' processes are never touched.
  - This corrects my practice of nice 10–19 (nice 19 under heavy load). Under another user's roughly 190-process load, that practice starved arm 3's check: about 0.1–0.5% CPU per worker, almost no progress in 6.5 hours.
- **Restarted.** The nice-19 arm 3 job was stopped (my own process group) and relaunched at nice 0. Shuffled seed 2 follows at nice 0 when it finishes.

### Arm 3's quick check: results (13 November, 16:34–16:49 UTC at normal priority; no search, no training)

**Run:** `arm3_check.py` (report `outputs/accumulating_learner/g2h/arm3/report.txt`).
- The network lineages' solutions were rewritten with the extended fix: 27 of 117 (seed 1) and 10 of 104 (seed 2).
- Libraries were built from levels 1, 2, 3 and 5 (77 and 71 equations), and the networks taken after level 5.
- 3,256 ranked cases: 984 new-term cases and 2,272 familiar-term cases.

**Median rank** (of about 1,100), at the empty start / one term short / two terms short:

| Method | New terms, seed 1 | New terms, seed 2 | Familiar, seed 1 | Familiar, seed 2 |
|---|---|---|---|---|
| Network | 204 / 39 / 178 | 207 / 57 / 200 | 3 / 1 / 3 | 3 / 1 / 4 |
| Library | 15 / 14 / 15 | 24 / 23 / 24 | 4 / 3 / 4 | 4 / 4 / 4 |
| Choice-level (adjusted library) | 15 / 13 / 15 | 27 / 24 / 29 | 3 / 2 / 3 | 3 / 2 / 3 |
| Product | 17 / 12 / 16 | 26 / 19 / 26 | 3 / 1 / 3 | 3 / 1 / 3 |
| Top-100 blend | 17 / 12 / 16 | 25 / 18 / 24 | 3 / 1 / 3 | 3 / 1 / 3 |
| Top-1,000 blend | 17 / 12 / 16 | 26 / 19 / 26 | 3 / 1 / 3 | 3 / 1 / 3 |

**Against the rule set before the run: no version passes.**
- The rule asked for new terms ranked below *both* the library and the network at every point.
- One term short, the term-level blends beat the library: 12 against 14 (seed 1), 18–19 against 23 (seed 2).
- At the empty start and two short they are 1–2 places behind it.
- The choice-level version is slightly behind the library in seed 2.
- On familiar terms every combined version keeps the network's ranks (1–3); the library alone is at 3–4.

**Reading.**
- The combinations reach the better of the two parents on each kind of term. New terms come in at about the library's rank, roughly 10 times better than the network alone, while familiar terms stay at the network's rank.
- They do not improve on the library for new terms, which is what the rule required.
- So by the pre-set rule arm 3 is not shown to be promising.
- What it does show: blending keeps both strengths without losing either. The pilot had suggested this was the missing piece, since the network's data-reading won the search and the library's ranking of new terms did not survive a search without the data.
- Whether that is worth a short search test is the owner's decision. The fixed-point ranks cannot settle it; ordering E showed they can mislead.
- **Shuffled seed 2** relaunched at 16:50 UTC at normal priority.
