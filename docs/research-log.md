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
