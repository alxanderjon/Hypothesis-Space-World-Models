# Learned guidance plus search for reusable dynamics discovery

Status: source/literature assessment and proposed protocol only. No numerical work was launched for this assessment.

## 1. Terminology and accounting correction

Future reports must say **what is fitted at encounter time**. A frozen shared network followed by a ridge solve is not “no training” in the ordinary sense. It is a form of per-system supervised adaptation, even when the solve is analytic.

For every method, report four separate objects:

1. **New-system observations**: the states and transitions supplied for the newly encountered system.
2. **Encounter-time fitted state**: coefficients, latent variables, expression trees, mixture weights, or shared weights changed using those observations, including the fitting algorithm and number of updates/factorizations.
3. **Persistent knowledge**: objects retained and reused across different systems.
4. **Discarded/transient state**: search trees, temporary gradients, or candidate populations not retained.

Use “direct frozen prediction” only when no parameter or coefficient is fitted from the new system. Otherwise say “context-conditioned analytic adaptation,” “context-conditioned gradient adaptation,” or “search plus coefficient fitting,” as applicable.

## 2. What the present benchmarks actually are

### Original planar two-body task

The original teacher state is eight-dimensional:

`(x1, y1, vx1, vy1, x2, y2, vx2, vy2)`.

It supplies fixed `dt=0.02`, `G=1`, masses `(1, 0.25)`, velocity-Verlet integration, and a central-force family

`a1 = +gamma G m2 (r2-r1) / ||r2-r1||^p`

`a2 = -gamma G m1 (r2-r1) / ||r2-r1||^p`.

Only `(gamma,p)` varies in this particular generator. Thus, even though the equations/parameters may be withheld from a learner, the task still supplies strong system-specific assumptions: planar point masses, state layout, masses, `G`, time step, central action-reaction coupling, force direction, and integrator. Source: `experiments/kepler/deterministic_teacher.py`, lines 2-12 and 22-30.

### Current cross-mechanism oscillator benchmark

This benchmark is deliberately simpler, not a complexity advance. Its state is two-dimensional `(q,v)`, every system obeys the supplied kinematic equation `q'=v`, `dt=0.05`, and fixed RK4 integration. The scalar acceleration families are:

- linear: `a=-p0 q-p1 v`
- Duffing: `a=-p0 q-p1 q^3-p2 v`
- Van der Pol: `a=p0(1-q^2)v-p1 q`
- damped pendulum: `a=-p0 sin(q)-p1 v`
- saturating spring: `a=-p0 q/(1+p1 q^2)-p2 v`
- Rayleigh: `a=p0(1-v^2)v-p1 q`
- quadratic drag: `a=-p0 q-p1 v-p2 v|v|`
- deadzone: `a=-p0 sign(q) max(|q|-p1,0)-p2 v`.

Parameter boxes and equations are in `run_cross_mechanism_diversity_matched_feature_corpus_v1.py`, lines 32-56. In the main mechanism-diversity corpus there are 24 laws per each of six training families, eight represented-development laws per family, and 16 laws in each of the two excluded development families. Every law uses the same controlled context bank of 32 states and query bank of 64 states (lines 61-65). Those controlled resets are substantially more informative than one passive trajectory.

### Supplied generic basis

The strong `generic16` comparator explicitly supplies

`[1,q,v,q^2,qv,v^2,q^3,q^2v,qv^2,v^3,sin(q),cos(q),sin(v),cos(v),tanh(q),tanh(v)]`.

It is a structural prior, not discovered knowledge. It already contains terms relevant to several benchmark mechanisms. Source: the same corpus driver, lines 55-58.

## 3. Exact adaptation and retention in the principal recent methods

### Learned integration-consistent atom model

- Persistent knowledge: a `2 -> 32 SiLU -> 16` feature network, 624 shared parameters, separately trained for each source-family fold/seed.
- New-system observations: 32 controlled `(q,v)` states and their one-step transitions.
- Encounter-time adaptation: fit 16 coefficients by ridge to observed `delta-v/dt`; this is analytic supervised adaptation. In the refinement study, those same 16 coefficients were then updated for 300 Adam steps on context transition error.
- Prediction: insert the fitted acceleration into the supplied `q'=v`, `dt=.05`, RK4 step and evaluate 64 query states.
- Result: shared atoms beat random atoms, but lost badly to fixed RBF and supplied `generic16`; coefficient refinement helped but did not close the gap. Source: `run_learned_rk4_atom_transport_v1.py`, lines 31-40 and 47-78, and the corresponding result record.

### Response-function distillation

- Persistent knowledge: two 624-parameter shared feature networks plus 73,728 bytes/seed of stored pseudo-response targets during sequential training.
- New-system observations: 32 controlled transitions.
- Encounter-time adaptation: a 32-coefficient ridge head is fitted from that system’s context. Therefore this is not a direct frozen answer.
- Shared update: two shared fits, 1,602 Adam updates total across the chronology.
- Result: it preserved encountered linear/Duffing/Van-der-Pol behavior efficiently, but did not improve Rayleigh over replay and lost to random features. This establishes bounded consolidation, not immediate recognition of an absent mechanism. Source: `coordination/RESEARCH_STATE.md`, lines 9275-9281.

### Callable RKHS memory

- Persistent knowledge: fixed `generic16`, 32 shared Gaussian centers plus bandwidth (520 bytes), and each accepted two-mode callable RKHS block (1,024 bytes).
- New-system observations: 32 context responses.
- Encounter-time adaptation: refit the entire ridge coefficient vector whenever the dictionary changes; dimensions grew from 32 to 34 or 36. This is analytic supervised adaptation.
- Old callable blocks were frozen. The 36.3% reverse-order retention regression occurred after dictionary expansion and coefficient refitting. It is inference interference under changed coordinates/regularization, not erasure of neural weights.
- Retention gate: aggregate after/before `<=1.1` and at least `8/12` earlier laws individually within 10%. Reverse-order retention was 1.3634 with 0/12 within 10%.
- Later-system benefits were demonstrated on other parameter settings of acquired quadratic-drag/deadzone families. That is not automatically transfer to entirely unseen mechanisms.
- Source: `run_cross_mechanism_rkhs_callable_memory_v1.py`, especially lines 48-51, 55-87, and 93-107; saved result `diagnostics/cross_mechanism_rkhs_callable_memory_v1/result.json`.

## 4. What the accumulated evidence says

Verified:

- Permitted observations often contain system-specific information: correct contexts commonly beat cyclic wrong contexts.
- Shared training can learn useful reusable features, and prior responses can be consolidated through distillation.
- A compact per-system solve can work when its callable span contains the relevant response form.
- The tested learned atom spans did not transport absent nonlinear structure competitively.
- Search/selection over a fixed dictionary and coefficient priors can cause inference interference even when stored callable functions are unchanged.

Not established:

- Unique recovery of a governing law from finite observations.
- Immediate prediction of a new mechanism without encounter-time fitting.
- Continual discovery across arbitrary physical mechanisms.
- That neural optimization is the unique cause of failure.
- That higher-dimensional optimization is uniformly harder.
- That the two-dimensional oscillator successes advance the system complexity of the eight-dimensional two-body task.

## 5. What an AlphaGo-style analogy does and does not justify

AlphaGo Zero combined a policy/value network with Monte Carlo tree search, then improved the network from self-play. But Go supplied exact rules and a perfect transition simulator. During a move, tree search did not update network weights; network learning occurred between self-play iterations.

The corresponding dynamics-discovery decomposition would be:

1. **Persistent guide**: a shared network scores candidate program expansions, basis elements, experiments, or model revisions based on previous systems.
2. **Encounter-time search**: search over a bounded space of candidate vector fields or programs for the new system.
3. **Continuous fitting inside each candidate**: fit that candidate’s small numeric coefficients from the new system’s observations. This is per-system supervised adaptation and must be counted.
4. **Feedback**: score candidates on an observation split not used for coefficient fitting. Query labels used for final scientific evaluation cannot guide search.
5. **Persistent update between encounters**: update the guide from completed search traces and, optionally, admit callable modules that pass an acquisition-only transfer gate.
6. **Ambiguity set**: retain several observationally compatible candidates when the available observations cannot distinguish them. Candidate agreement is not proof of truth; disagreement is evidence of unresolved predictive ambiguity only on the tested domain.

Unlike Go, the project does not have an exact unknown-system transition oracle at search time. It must declare what feedback is available: held observed transitions, actively acquired transitions, or simulator responses. Supplying active resets or extra trajectories changes the observation contract, potentially legitimately, but must be stated.

Primary precedents support the broad learned-guidance-plus-search pattern, not this project’s validity automatically. AlphaGo Zero uses learned guidance plus exact-rule search; neural-guided symbolic-regression work uses a learned production/mutation model with tree search or genetic programming. Relevant primary sources include Silver et al. (Nature 2017), Li et al. (2019) on neural-guided MCTS for symbolic regression, Kamienny et al. (ICML 2023) on context-aware neural mutation plus MCTS, and Mundhenk et al. (2021) on learned population seeding plus genetic programming.

## 6. High-dimensional landscape geometry and scaling

Three dimensions must remain separate:

### State dimension `d`

`d` controls the domain over which dynamics must be identified. Coverage of an unrestricted `d`-dimensional function becomes exponentially demanding without locality, factorization, symmetry, sparsity, or active excitation. Moving from 2D `(q,v)` to an 8D two-body state can therefore make observation coverage and candidate evaluation harder even if the shared network is unchanged.

### Shared persistent parameter count `P`

`P` is the size of the guide or reusable representation. A million shared parameters does not require a million-dimensional per-system search if they are frozen during encounters. First-order training is roughly linear in `P` per example up to architecture-dependent constants. A dense Hessian requires `O(P^2)` storage and dense factorization `O(P^3)`, which is infeasible at a million parameters. Matrix-free Hessian-vector products, conjugate-gradient/trust-region methods, and Lanczos spectral probes can inspect or exploit curvature without materializing the Hessian.

### Per-system search dimension `k`

`k` is the number of numeric coefficients/latent variables changed for a new system plus the discrete program/search choices. This controls encounter-time adaptation. A system can use a million-parameter frozen guide while fitting only, for example, six coefficients and a small expression tree. Conversely, fitting all shared weights per system recreates Yubei’s scaling objection even if the benchmark state is only 2D.

### Geometry cautions

- High-dimensional nonconvex problems can contain many saddle/near-flat directions; Dauphin et al. argued these may be more important than poor local minima in some neural problems. This is an influential diagnosis, not a universal theorem about this repository’s losses.
- Parameter redundancy and symmetries create flat directions and many equivalent weights.
- Garipov et al. showed low-loss paths can connect apparently different solutions in large neural networks. Distinct weight vectors therefore need not represent distinct functions.
- Two-dimensional loss plots are projections. Li et al. showed that filter-wise normalization materially changes meaningful comparisons. A visually sharp or flat 2D slice is not the complete landscape.
- More dimensions do not always make optimization worse: overparameterization can create connected solution sets and easier descent while worsening conditioning or identifiability. Empirical curvature/function diagnostics are required.

A scalable diagnostic stack should use function-space comparisons, Jacobian singular values/effective rank, gradient norms, and matrix-free extremal-curvature probes. It should not build a full Hessian or infer the full landscape from a 2D plot.

## 7. Proposed next study: learned-guided bounded vector-field search

This is a formulation change, not another MLP feature sweep. It should remain a proposal until its grammar and data contract are reviewed.

A repository-wide source/protocol search found no completed expression-tree, production-grammar, genetic-programming or MCTS dynamics study. The existing “search” branches optimize continuous weights, latent codes, fixed dictionary prefixes or numerical candidates; they do not learn a policy over callable program construction. This establishes local novelty of the causal mechanism, not scientific merit by itself.

### Scientific question

Does experience across source systems teach a search strategy that finds observationally supported, callable vector fields for a new system more efficiently than an identical unguided search budget, while keeping encounter-time fitted state compact and retaining multiple plausible candidates when observations are ambiguous?

### Development contract

- Use the existing opened 2D oscillator corpus first. This is a controlled feasibility benchmark, not evidence of higher system complexity.
- State supplied: `(q,v)`.
- Structural assumptions supplied: `q'=v`, `dt=.05`, RK4, scalar acceleration `a(q,v)`, and a fixed expression grammar.
- New-system observations: exactly 32 controlled transitions. Split them before search into 24 coefficient-fit transitions and eight search-validation transitions.
- Final development queries: the existing 64-state bank. Its labels are never available to candidate search or coefficient fitting.
- Whole-family folds prevent a guide from learning from the held mechanism’s systems. All currently opened families remain development evidence; no new confirmation family is opened in this pilot.

### Candidate representation

- Expression tree for scalar `a(q,v)`.
- Fixed grammar declared once: terminals `{q,v,1}`, binary `{+,-,multiply}`, unary `{sin,tanh,abs}`, and one explicitly bounded protected reciprocal. The grammar is a supplied inductive prior and must be reported as such.
- Maximum 15 nodes, depth five, and six free scalar coefficients.
- For every complete candidate, coefficients are fitted from the 24 fit transitions using one fixed bounded optimizer. This is encounter-time supervised adaptation.
- Candidate score combines the eight validation-transition errors with one predeclared complexity penalty. Up to four near-best candidates are retained; no uniqueness claim follows.

### Persistent guide

- A context encoder reads the 32 observed transitions and produces priors over grammar expansions plus a value estimate for partial programs.
- The guide weights persist across systems. Search trees and per-system fitted coefficients do not become shared knowledge automatically.
- After an encounter, successful and failed search traces may update the guide between encounters. This is persistent shared learning; it is distinct from the coefficient fitting performed within an encounter.
- An admitted expression template becomes persistent callable memory only if it passes a source-only acquisition/transfer gate fixed before later-system evaluation.

### Causal controls

1. Learned-guided tree search.
2. Uniform-prior tree search with the same candidate and coefficient-fit budget.
3. One-shot guide proposal with the same coefficient-fit optimizer, isolating the value of search.
4. Fixed `generic16` ridge as a disclosed structural upper comparator.
5. Fixed RBF and random-guide controls with matched coefficient count/search evaluations.

### What is retained and fitted

- Retained across systems: guide weights; accepted grammar templates/callable modules; aggregate search-trace training data if replayed (bytes counted).
- Fitted for each new system: one expression tree selected by search, up to six numeric coefficients per candidate, and optional weights over at most four retained candidates.
- Updated between encounters: shared guide weights only in the accumulation arm.
- Not retained: rejected candidate optimizer state and search tree, unless explicitly counted as replay memory.

### Prospective pilot ceiling

- Existing cached responses only; zero teacher calls and zero new trajectories.
- Six whole-family development folds, two fixed seeds.
- At most 128 complete candidate evaluations per system and 25 coefficient-optimizer closures per candidate.
- At most 144 held-family system encounters per seed, 73,728 candidate evaluations across guided and uniform search arms, and 1,843,200 coefficient closures.
- At most 12 guide fits and 12,000 shared-guide optimizer updates.
- Maximum shared guide: 1,000,000 parameters; maximum per-system continuous state: six coefficients per candidate plus four mixture weights.
- Wall time 30 minutes, peak RSS 1 GiB, artifacts 128 MiB. Any capped encounter is inconclusive and is not resumed under a new name.

### Decision outcomes

- Guided search beats matched uniform search and one-shot proposals across whole-family folds: evidence that prior systems taught a useful discovery strategy within this supplied grammar and observation protocol. It is not arbitrary dynamics discovery.
- One-shot equals guided search: search adds no value beyond the learned proposal under this budget.
- Uniform search succeeds but guidance does not: the grammar/search space is adequate but the learned guide fails to transfer.
- Neither succeeds while `generic16` does: the grammar, search budget, coefficient optimization, or guide representation is inadequate; the result does not establish observation non-identifiability.
- Several candidates fit observations but diverge on queries: finite observations leave predictive ambiguity within the declared grammar/domain. Report the candidate set; do not choose using query labels.

### Scaling path

Only after the controlled 2D causal test should the method return to 8D. The 8D version must not search arbitrary flat expression trees over all coordinates. It needs an explicit factorization prior (for example graph/local interactions or learned object relations), and that prior must be disclosed. A million-parameter guide is acceptable only if new-system search remains compact and matrix-free; fitting a million weights per new system is not.

## 8. Recommendation

Do not continue tuning the closed learned-atom, ridge-prior, or RKHS-prefix branches. The materially different direction is to learn **how to search and revise candidate functions**, not to demand an immediate answer from a fixed encoder. The first study should be the controlled, cached-data learned-guidance-versus-matched-search causal test above. It would test a necessary capability—whether experience improves discovery efficiency—while making every supplied prior and every encounter-time update explicit.
