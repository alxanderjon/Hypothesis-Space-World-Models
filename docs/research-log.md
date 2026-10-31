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
