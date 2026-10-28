# Hypothesis Space Recovery: research handover

Prepared 28 October 2026 for a new agent taking an independent approach.

**Research is stopped at the owner's request.** Dot's chat, “Inspect interrupted SHO recovery” (`01a13cc5-3739-77c4-93bd-98a03f69e01d`, host `durable`), confirmed that no experiment or delegated agent remained running. The continuation automation `keep-dynamics-research-progressing` is paused. This document authorizes no experiment. Archived protocols, `NEXT_TASK.md` files and continuation instructions are historical records, not instructions to resume.

## Start here

The objective remains unfinished. There is no demonstrated general learner that accumulates reusable dynamics knowledge, reliably adapts faster to unfamiliar mechanisms, and retains earlier capabilities. There are narrower positive results and useful failures, but no breakthrough.

The latest completed work used roughly one million parameter Transformers on official PDEBench trajectories. It replaced the earlier hand-selected oscillator term language with learned field representations and independently defined PDE families. Learning a new family helped compared with a frozen source model, but source-data replay did not reliably improve adaptation or preserve individual earlier cases. In the most recent controlled comparison, replay was worse than otherwise identical adaptation without replay in both seeds.

A new agent should examine the goal and evidence before choosing a method. There is no obligation to continue the latest architecture or the original deterministic teacher. The user wants an independent approach, not more variations of a method that has lost its connection to the goal.

Read this document first, then the [cumulative state](handover-2026-10-28/evidence/coordination/RESEARCH_STATE.md), [results ledger](handover-2026-10-28/evidence/coordination/RESULT.md), and the [latest paired comparison record](handover-2026-10-28/evidence/pdebench_records/task_transformer_markov_paired_replay_v1/result.json). The ledgers contain historical terminology and superseded proposals; use source and result records to resolve discrepancies.

## What the owner is actually asking for

Build a learner whose accumulated experience makes unfamiliar dynamics easier to infer. On encountering a new system, it should be able to discover what it needs from observations, improve its predictions, and retain useful knowledge for later encounters. Per-system observation and computation cost should become more scalable, with diminishing dependence on a bespoke teacher.

This has been the objective from the start. The deterministic N1/N2/N3 teacher was built as a means to it: a way to produce supervision for an observations-to-dynamics student. It is one possible bootstrap, not a required solution. The research drifted when making the teacher recover individual systems became the working objective in its own right (see [How the teacher became the objective](#how-the-teacher-became-the-objective)). New-system learning, gradients and adaptation are allowed. Immediate prediction without adaptation is not the sole objective.

The owner's strongest objections to earlier work are substantive:

- Choosing among nine supplied combinations of position and velocity does not establish general discovery. A new hand-written list for every system defeats the purpose.
- A benchmark constructed around the exact repeated function the method is designed to recover can test implementation, but cannot establish broad reusable-component discovery.
- A scalar oscillator is a simpler diagnostic than general two-body dynamics. Moving between them is a change of task, not automatically scientific progress.
- Descriptions must expose exactly what is observed, what is supplied, what is trained beforehand, what changes on a new encounter, and what is held out. Terms such as “broader experience,” “knowledge base,” or “zero-shot” are insufficient.
- Analytic coefficient fitting is still learning from a new system. Freezing a shared network does not make the complete procedure zero-shot if system-specific coefficients are fitted.

Keep three costs separate: state dimension, total retained model size, and the number of quantities adjusted per new system. Also count observations, optimizer work, evaluation work, retained examples and offline search. A million-parameter first-order learner and a million-parameter full-Hessian search have very different costs.

## How the research reached its present position

| Phase | Actual method and observations | What the evidence supports |
|---|---|---|
| Deterministic teacher | Global normalized search, bounded Powell refinement, then full-objective Hessian/box-QP polishing; system-specific simulation adapters | Reusable numerical search machinery on restricted tasks. No general observations-to-rule student demonstrated. |
| Passive two-body student | Context transitions from one central-force trajectory; predict transitions on other trajectories | Many encoders and adaptation variants failed reliable context use and transfer under the tested budgets. A known two-parameter inverse solver succeeded, which establishes much less. |
| Controlled two-body observations | Reset simulator at selected states; learn observation-to-response maps | Some strong transfer within the same central-force family. Active resets and supplied interfaces materially simplify the problem. |
| Scalar oscillators | Learned shared features plus per-system ridge coefficients from 32 controlled transitions | Narrow adaptation successes; passive-trajectory bridge and continual retention/adaptation tradeoffs remained unresolved. |
| Guided symbolic/library search | Supplied term language, then growing expression language and callable memories | Search guidance sometimes helped restricted search. Reusable nonlinear discovery and reliable retention were not established. |
| Four-dimensional/component tests | Shared nonlinear residuals, common panels, neural/kernel modules and acquisition policies | Recognizing shared systems did not imply reconstructing a useful transferable function. One graph study exceeded caps and is censored. |
| Heterogeneous interfaces | Lossless common formatting across earlier tasks; context models and coordinate/mechanism shifts | Common formatting did not solve semantic or mechanism transfer. |
| PDEBench Transformer | Official trajectories from advection, Burgers and reaction–diffusion; shared learned model with new-system weight updates | Feasible first-order training at approximately one million parameters; current replay scheme failed its complete transfer/retention test. |

These are different experimental contracts. Comparing their errors as though they measured one unchanged task would be misleading.

### 1. Deterministic teacher: what exists and what it does not prove

The teacher was a three-stage pipeline. Each stage handed off to the next only once it passed:

1. **N1: gradient-free global search.** The working belief was that gradient methods were holding the work back by getting stuck in local minima. N1 therefore searched the normalized parameter box DIRECT-style. It partitioned the box into Voronoi cells around evaluated sites, approximated the cells with scrambled Sobol space-filling points, scored sites by reconstruction loss, and refined the potentially optimal cells. The two-parameter version compares standard DIRECT with eDIRECT (Liu et al. 2015): [source](handover-2026-10-28/evidence/experiments/sho/run_stage1b_edirect_vs_direct.py).
2. **N2: gradient-free local search.** Bounded Powell line search (`maxfev=1500`) from N1's handoff.
3. **N3: derivative-based polishing.** Only after N1 and N2 passed did the pipeline use derivatives, on the belief that it would no longer get stuck in a poor local minimum. N3 is Newton-type rather than plain gradient descent: it computes the objective's full Hessian, tries a safeguarded box-constrained quadratic step, adds damping when needed, and backtracks.

The archive confirms these mechanics. It does not record the local-minima rationale, which comes from the owner. The later [assessment](handover-2026-10-28/evidence/coordination/NEURAL_GUIDED_DYNAMICS_DISCOVERY_ASSESSMENT.md) (section 6) notes that in high-dimensional neural problems, saddles and flat directions may matter more than poor local minima. The premise is therefore weakest exactly where scale matters.

Adapters specify how candidate parameters generate observations for SHO, reflection/event-driven motion and Kepler-like dynamics.

The SHO sequence expanded from two parameters to stiffness matrices, affine dynamics and finally a generic 22-parameter `2 -> 4 -> 2` SiLU update network with no skip connection. Some long polishing runs reached extremely small objective values. They did not establish exact unrestricted function recovery. Earlier numerical stop rules were replaced; do not treat old accepted-step limits as current.

### Hard-coded versus removed dynamics

With the dynamics built into the hypothesis, parameter recovery went well. Examples are the Stage 0 exact-solution decoder on `main`, two-parameter search, and an affine update `Kz+b` validated on all 320 SHO cases.

Once the dynamics were removed, by replacing the structured hypothesis with the generic 22-parameter network, the difficulty became the geometry of the search. The network can represent the SHO update: an in-box fixture reaches transition MSE `3.67e-29`. Finding it from data was the hard part:

- Eight bounded companion_29 sites gave 0 handoffs, five resource caps, and final losses spanning about seven orders of magnitude depending on the start.
- Case (4,3) remained unresolved, and an extended (5,4) run stalled at `7.9e-11`.
- One case (companion_6) succeeded only after the solver was redesigned to solve the ten output weights exactly at every hidden-layer candidate (variable projection). That gave a 53,921x lower loss than the saved stabilized run.

Part of the geometry problem was therefore output/hidden coupling that added structure could fix. Even the 22-parameter version still supplied the state `(q,v)`, the time step, the one-step update form and the parameter bounds. Sources: [cumulative state](handover-2026-10-28/evidence/coordination/RESEARCH_STATE.md), especially the frozen-baseline and generic-recovery entries dated 13 October.

### Dimensionality: Yubei's concern

Yubei, the owner's co-advisor, pointed out that the geometry of a relatively low-dimensional parameter space is very different from that of a large one. This approach may not work for, say, a million-parameter system. The evidence already shows strain between 2 and 22 parameters. Partition-based global search needs coverage that grows exponentially with dimension. Full dense Hessians scale quadratically in storage: one million parameters would require about 8 TB just for a float64 Hessian. This limits that teacher implementation, not all neural learning. See [teacher source](handover-2026-10-28/evidence/experiments/sho/generic_update_network_adapter.py) and the cumulative ledgers for the recovery and stop-rule history.

Large networks also have many weight settings with identical behavior, through symmetries and flat directions. Parameter recovery therefore stops being a meaningful target at scale, and success has to be judged on predictions. The concern applies to later work whenever the number of quantities adjusted per new system is large. In particular, the PDEBench adaptation below updates all of its roughly one million weights for every new system.

### How the teacher became the objective

For weeks the standing scope was making the unchanged 22-parameter network recover reliably, one case and one start at a time, each under its own approval ([decisions ledger](handover-2026-10-28/evidence/coordination/DECISIONS.md), from the 13 October entries). That work never asked whether experience made the *next* system easier. The ledger itself concedes that the generic recovery cache had only four teacher trajectories/laws, “inadequate for a credible cross-law learned discovery/update policy”. The accumulation question returned only in the 27 October entries. Even then, no study measured how fast or reliably a new system was learned as a function of how many systems the learner had already seen.

The saved 822-case collection contains 320 SHO, 246 reflection and 256 Kepler cases, but this is not 822 unrelated mechanisms. Reflection-ball and Kepler each vary initial conditions around one teacher law. Some successful “hybrid student” work replayed successful solver branches rather than learning an observations-to-rule mapping. Preserve that distinction.

### 2. Two-body learning: passive observations versus reset access

The central-force experiments varied force strength and radial exponent inside a supplied family, with fixed masses, a known observation interface, and simulated integration. State had eight entries: each body's two positions and two velocities. A context often contained 32 selected one-step state-to-increment pairs from one trajectory; prediction queries came from other trajectories of that law.

Many variants failed to consistently distinguish correct context from shuffled or wrong-system context. A solver supplied the true two-parameter force family achieved excellent recovery. That shows information was useful under that prior; it does not show a general learner can find an unknown mechanism.

Later controlled versions could place the simulator at chosen states and measure the response. One successful map used four reset observations, each with eight response values, as 32 scalar inputs to predict a fixed response grid. It transferred within that same family. Reset access, a fixed output grid and substantial offline design were supplied. It was not passive inference from an arbitrary trajectory.

### 3. Oscillators: the exact meaning of shared features and coefficient fitting

These experiments used a scalar position `q` and velocity `v`. The supplied kinematic relation was `dq/dt = v`; the unknown part was acceleration. Simulated transitions used RK4 with step 0.05. The observation protocol reset each system at 32 Sobol-selected coordinates in `q ∈ [-1.5,1.5]`, `v ∈ [-1,1]`, and measured one transition from each coordinate. Those were coordinates in state space, not timestep numbers along a trajectory. Queries used separate states.

A shared MLP took `(q/1.5, v)`, passed through 32 SiLU units, and output 16 learned feature values. Dividing by 1.5 normalized the sampled position range to approximately `[-1,1]`; it was not a physical law. The network had 624 shared parameters.

For a new system, evaluate those 16 features at each of its 32 observation states. Fit a `16 x 2` coefficient matrix so weighted feature sums predict the two components of the observed transition. This fits 32 system-specific numbers. At a new query state, evaluate its 16 features and multiply by that fitted matrix. The shared MLP can remain frozen while those coefficients change. The whole procedure nevertheless learns from the new system.

During prior training, losses on query states were differentiated through the coefficient solve to update the 624 shared weights across source systems. Thus shared features were learned to make later coefficient fitting useful. They were not first trained on full trajectories and then secretly retrained on the 32 states in every frozen-feature evaluation.

“Narrow” versus “broad” source arms referred to numbers of simulated systems: 48 coefficient settings from each of three families versus 24 from each of six families, totaling 144 systems in either arm. They were not 48 different network weights. Families included linear, Duffing, Van der Pol, pendulum, saturating and Rayleigh mechanisms. Rayleigh damping depends nonlinearly on velocity, for example `a(1-v²)v - bq`; it can inject energy at small velocity and dissipate it at larger velocity.

Some held-family controlled-observation results were positive. Subsequent replay, function constraints, KL regularization, Laplace importance and sequential head-posterior studies did not jointly establish adaptation benefit and retention. For example, the sequential head posterior retained earlier cases but accumulated history was worse than resetting it in both seeds. See the archived sequence protocols and results ledger. Success on this simplified contract should not be advertised as arbitrary dynamics discovery.

### 4. Symbolic guidance, callable libraries and the retention issue

A small guide ranked individual terms in a supplied nine-term language. It reduced error versus random candidate search, while a larger classifier over complete combinations failed on unseen combinations. Each new system still supplied transitions and each candidate still fitted coefficients. This was guidance within an externally supplied language.

A later expression language used addition, multiplication, sine and tanh; a learned macro could reuse an expression that was already expressible. This demonstrated useful compression/search reuse on an engineered recurring-motif task, not discovery of an unrestricted new operator.

A callable-function library gave narrow future-system gains. However, adding a function worsened an earlier mechanism's error by 36.3%. The old functions themselves were frozen: expanding the basis and refitting coefficients could still change the resulting predictor and hurt retention. It is too simplistic to describe this solely as shared-network weights forgetting.

Four-dimensional tests subsequently distinguished detecting a recurring component from learning its nonlinear shape on different states. Detection of all eight relevant systems did not produce an admitted reusable nonlinear function. Earlier apparent gains were heavily explained by nearly affine corrections. Neural and RBF variants did not resolve this.

The graph-module study has a separate validity problem: recovery repeated unsaved work, taking total generation/training to 13,312 transitions and 9,600 updates against previously declared ceilings of 7,168 and 6,400. The review found no specific owner authorization for the overrun. Preserve both attempts and the failure; classify the recovered bounded comparison as **censored**, not as valid negative evidence disproving the mechanism. Recording extra work is not retroactive permission.

## Latest completed work: PDEBench, in detail

This phase sought externally defined mechanisms and a representation learned from data instead of a curated `q,v` basis. It still has explicit scope: scalar, fully observed, one-dimensional fields on a common periodic-grid interface. It does not cover arbitrary sensors, partial observability or arbitrary state dimensions.

### Data and splits

Source mechanisms were advection at three coefficients and Burgers at three viscosities. Target adaptation used reaction–diffusion at three coefficient pairs. The target mechanism was not used for source training, but it is now opened development evidence, not untouched future confirmation.

The acquisition selected trajectories without inspecting their responses, using filename-seeded sampling from official PDEBench HDF5 files. There were nine coefficient files. Per file, 20 trajectories were designated training, five development, and five reserved. Only the first 25 were downloaded: 225 trajectories total. The five reserved trajectories per file remained undownloaded. Diffusion–sorption was a separately named unopened future mechanism.

The studies did not use the target files' acquisition labels as their split. Both Markov studies used the following split:

- **Source files:** training on the 20 training trajectories, evaluation on the five development ones. That is 6 files x 5 trajectories x 4 windows = 120 source blocks.
- **Target reaction–diffusion files:** adaptation on trajectories 0–9 of the 25 downloaded, evaluation on trajectories 10–24. Ten of the 15 evaluation trajectories were therefore designated training at acquisition. That is 3 files x 15 trajectories x 4 windows = 180 target blocks.

The four evaluation windows per trajectory start at times 0, 32, 64 and 96.

Selected trajectories were reduced to 64 spatial locations and 101 times. Advection/Burgers used every second stored time; reaction–diffusion used its 101 times. Spatial sampling used every sixteenth point of the original 1024-point fields. Physical coordinates were retained. About 154.6 MB were transferred by byte ranges, not the full approximately 202 GB corpus. The [manifest](handover-2026-10-28/evidence/pdebench_records/manifest.json) records provenance and hashes; partial downloads cannot validate the full remote-file MD5.

### Model and encounter procedure

The latest Markov model has **996,100 trainable parameters**, three Transformer encoder layers and three decoder layers, width 128, four attention heads and feed-forward width 256.

Four field states provide context. Each state is split into 16 patches of four neighboring field values: 64 context tokens total. Tokens include field values and position/time coordinates. For prediction, 16 query tokens contain the current patch's four values plus position and time. The model predicts the next field increment, four values per patch. Supplying the current field in the query corrected the earlier version, whose query contained coordinates only.

Source-only statistics normalize fields and increments. No PDE equation, coefficient label or solver is supplied to the model. The grid, patch structure, chronology, coordinates and fully observed current state are supplied.

For each of two seeds, source training ran 300 updates. On the new reaction–diffusion family, adaptation ran 100 updates and changed **all** model weights. This is new-system training, not zero-shot or compact coefficient fitting. Frozen-source, adaptation without replay, adaptation with source replay, and training from scratch were compared.

The initial Markov study gave inconsistent seed results, and its replay versus no-replay arms used different batch schedules. The follow-up fixed that comparison.

### Exact paired replay follow-up and result

The two arms began from identical source checkpoints with identical optimizer setup and sampled windows. Each update used two target windows and two source windows. Replay optimized `target loss + 0.5 * source loss`; the paired control optimized target loss only. Both arms used the same sampled windows. Four fits (two arms x two seeds) used 400 optimizer updates. Each seed sampled 100 x 4 = 400 windows shared by its two arms: 800 unique sampled windows in total. Each arm forwarded all four windows per update, including the no-replay arm, whose source loss was multiplied by zero. That gives 1,600 model-window training forward passes. The result record's `counts` field reports 1,600 windows and 3,200 forward passes. Those values are hard-coded in the [implementation](handover-2026-10-28/evidence/experiments/pdebench/task_transformer_markov_paired_replay_v1.py) rather than computed, and are double the counts implied by its training loop. Previously saved frozen/scratch scores were reused and are identified as such in the record.

A seed passed only if all five of these held. The study is `SUPPORTED_DEVELOPMENT_ONLY` only if both seeds pass:

1. Replay target error / paired no-replay target error `<= 0.8`.
2. Earlier-source error after replay / frozen source error `<= 1.1`.
3. At least 80% of the 120 source blocks within 10% of their frozen error.
4. Target error with history shuffled / intact `>= 1.05`.
5. Target error with the current field shuffled / intact `>= 1.1`.

Here, error is total MSE summed over blocks.

Lower error is better:

| Comparison | Seed 202610281 | Seed 202610282 |
|---|---:|---:|
| Replay error / paired no-replay error | 1.882 | 1.660 |
| Blocks where replay beat paired no-replay | 10 / 180 | 3 / 180 |
| Replay error / frozen-source error | 0.0539 | 0.2551 |
| Replay error / saved scratch error | 0.3917 | 0.8049 |
| Earlier-source aggregate error / frozen source | 0.9309 | 0.8644 |
| Earlier-source blocks within 10% of frozen error | 112 / 120 | 50 / 120 |
| History-shuffled error / intact error | 1.037 | 2.942 |
| Current-field-shuffled error / intact error | 1.504 | 3.534 |
| Criteria failed | 1, 4 | 1, 3 |

Adaptation substantially improved on a frozen model and replay beat the saved scratch baseline in aggregate. However, **replay harmed adaptation relative to the paired no-replay control in both seeds**, and good aggregate source error concealed substantial case-level regressions in the second seed. The study's classification is `NOT_SUPPORTED`.

Shuffling diagnostics tested sensitivity to current-field and history information. They do not establish a learned transferable mechanism. A low aggregate prediction error alone also does not establish that.

Source and evidence:

- [Markov protocol](handover-2026-10-28/evidence/coordination/PDEBENCH_MARKOV_QUERY_PILOT_V1.md), [implementation](handover-2026-10-28/evidence/experiments/pdebench/task_transformer_markov_v1.py), [full result](handover-2026-10-28/evidence/pdebench_records/task_transformer_markov_v1/result.json).
- [Paired protocol](handover-2026-10-28/evidence/coordination/PDEBENCH_MARKOV_PAIRED_REPLAY_V1.md), [implementation](handover-2026-10-28/evidence/experiments/pdebench/task_transformer_markov_paired_replay_v1.py), [full result](handover-2026-10-28/evidence/pdebench_records/task_transformer_markov_paired_replay_v1/result.json).

This rules against the tested replay setup. It does not establish that all continual learning, Transformers or transfer between PDE mechanisms must fail. Sweeping settings against already opened targets would require acknowledging further development, not treating the same targets as fresh confirmation.

## What a fresh approach should resolve

These are design questions for the next agent, not a prescribed experiment sequence:

1. **What knowledge persists?** Specify whether it is a representation, inference procedure, optimizer, explicit model, memory, modular component or something else. Explain how its contents can help an unfamiliar mechanism without a custom answer-containing language.
2. **What is allowed at encounter time?** State passive trajectory versus controlled-reset access, observable quantities, noise, observation count, updates, fitted quantities and search costs. Account for the possibility that the same limited observations fit multiple explanations.
3. **What would demonstrate accumulation?** Compare against a reset/scratch learner, a frozen prior, and equal-cost controls. Measure whether experience reduces the observations or computation needed to reach useful error. Also test earlier capabilities at the individual-system level.
4. **What would expose a shortcut?** Separate coefficient changes, new mechanisms, coordinate/interface changes and observation-protocol changes. Use independently motivated systems. Do not construct all targets around the desired reusable component and infer generality from that construction.
5. **How will dimensionality enter?** Increasing state dimension changes observation coverage and computational geometry. Distinguish this from increasing network size. Do not infer scalable identification merely from being able to optimize a large model.

The owner has expressed interest in a learner that learns how to investigate and adapt, with an AlphaGo-like motivation. That analogy is an aspiration, not an implemented method or evidence. Any search/evaluation loop here must score candidates using permitted observations, without a hidden simulator oracle supplying the answer.

Before running anything, propose a concrete observation-to-prediction/adaptation contract and explain why it addresses the actual goal. State assumptions and prospective resource bounds. Preserve untouched confirmation data. Do not silently repeat lost training, expand ceilings after the fact, tune success criteria to obtain a pass, or call one completed audit the completion of the research objective.

## Repository and evidence package

The publishing baseline is commit `5b95a4389779ffd3986fe7ff14936b50340e332a`. Most later research existed as untracked local files. This handover therefore includes a **verbatim evidence snapshot**, rather than relying on GitHub links to files that were never committed.

[MANIFEST.json](handover-2026-10-28/MANIFEST.json) lists each copied file, its original location, byte count and SHA-256. It contains 742 files: experiment/source text, coordination records and the available PDEBench JSON records. The snapshot is provenance material, not a second active implementation tree. Historical instructions within it are inactive. The consolidated ledgers are reported research history; this handover did not independently rerun all earlier experiments.

Excluded: raw trajectory arrays, checkpoints, virtual environments, the approximately 25 GB general outputs tree and unrelated working changes. Consequently, older experiments are not all reproducible from this Git branch alone. Many archived scripts contain absolute paths and depend on omitted artifacts. Do not run them blindly. Inspect dependencies and inputs first; historical reproduction would need a separate authorization and data/artifact recovery plan.

For recovery on the original machine:

- Research working repository: `/Users/alexanderrichardson/Documents/hypthesis-space-world-models` (spelling intentional).
- PDEBench data, JSON results and saved model artifacts: `/Users/alexanderrichardson/Documents/Codex/2026-10-14/task-2/data/pdebench_range_subset_v1`.
- Source GitHub repository: <https://github.com/alxanderjon/Hypothesis-Space-World-Models>.

The PDEBench directory is outside the research repository. The snapshot includes its JSON records, but not its arrays or checkpoints. Check actual files and hashes before treating any local artifact as the right continuation checkpoint.

No scientific computation was launched to prepare this handover. Original research files and failed/censored attempts were preserved. The next agent should receive a new instruction from the owner before resuming research.
