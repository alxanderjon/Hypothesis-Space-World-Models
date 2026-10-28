# Abstraction-learning cross-study synthesis

## Verified gap

The completed studies do not jointly implement **solution-corpus abstraction induction plus learned search guidance**.

- `growing_callable_program_library_v1` enumerated a fixed expression pool and directly admitted `sin(q*v)`. It demonstrated efficient reuse of an expression already in the supplied closure, not induction of a new abstraction from several independently solved programs.
- `variable_graph_callable_acquisition_v1_recovery1` learned one callable block inside four supplied depth-two graph templates. It retained neural weights but had no explicit corpus of solved hypotheses, compression step, or growing search language; it failed its no-retention comparison.
- `common_panel_labelblind_callable_recurrence_v1_confirmation` extracted one recurring callable from controlled common probes without hidden reuse labels, but the recurrence distribution was engineered and there was no program search, abstraction hierarchy, or sequential guide update.
- `parameterized_piecewise_schema_discovery_v1` supplied ReLU, symmetric hinges, and a numeric hole. The expanded language improved the old language by about 30%, while retaining the selected schema added no measured benefit over the identical expanded language.

## Concrete missing mechanism

A bounded wake–compress–guide learner would maintain three distinct persistent objects:

1. **Callable library:** typed macro ASTs formed by compressing recurring subgraphs in programs that were first selected solely from observation fit/selection splits.
2. **Search prior:** production probabilities over base primitives and learned macros.
3. **Recognition guide:** weights mapping permitted observation summaries to production or expansion scores.

Per new system it would temporarily maintain:

- the permitted state/response observations;
- a bounded candidate beam or bottom-up execution table;
- fitted numeric holes and outer coefficients for each candidate;
- posterior/search scores and the selected hypothesis.

Only admitted macros, search-prior parameters, recognition weights, and a declared replay/compression archive persist. Per-system numeric coefficients and search tables are discarded unless their storage is explicitly charged as replay memory.

No true equation, hidden program, primitive identity, family label, or future query response may supervise search, compression, or admission. Generator metadata is quarantined until predictions and library decisions are frozen.

## Why this is distinct

The causal hypothesis is not that neural capacity is larger. It is that **compressing recurring subprograms reduces later search depth while a learned guide reduces search breadth**. DreamCoder-style wake/sleep learning is the relevant mechanism: successful programs are compressed into abstractions and a recognition model is retrained on solved and replayed tasks. The local studies tested isolated pieces—candidate selection, one neural module, recurrence extraction, or parameterized holes—but never this coupled update.

## Necessary controls

All arms must share the same base computational language, observations, candidate-evaluation budget, numeric-hole optimizer, and total training data.

1. **Uniform reset:** no retained macros and uniform search.
2. **Guide-only:** retained recognition/search weights but reset macro library.
3. **Library-only:** retained macros with uniform search.
4. **Wake–compress–guide:** both retained library and guide.
5. **Shuffled compression:** same macro count and memory, induced from task-shuffled or behavior-mismatched solved programs.
6. **Monolithic predictor:** matched persistent parameter count and data, with no discrete search, to distinguish abstraction/search benefit from generic capacity.

Later-system benefit must be measured before updating on later query labels. Report solve rate, held-query prediction, correct-versus-wrong-context behavior, candidates evaluated, wall time, retained bytes, replay bytes, and fitted per-system quantities.

## Outcome-blind corpus requirements

One synthetic corpus alone cannot prove open-ended discovery. A defensible first diagnostic needs two prospectively fixed strata:

- **Expressible compositional stratum:** programs sampled from a generic typed base language, with whole subtrees and compositions withheld from acquisition. Recurrence frequencies and non-reuse tasks are fixed before learner outcomes.
- **Non-expressible stratum:** smooth vector fields generated outside the bounded symbolic language. This tests whether the learner abstains or falls back rather than misrepresenting search failure as discovery.

The base language is still a supplied computational prior. A positive result would establish learned abstraction and search reuse inside that language, not assumption-free dynamics discovery.

## Scientific fork

### A. Neurosymbolic wake–compress–guide

Best direct test of persistent, inspectable knowledge acquisition. It supplies a generic typed computational substrate and pays bounded combinatorial search per encounter. Success would mean later held-composition tasks are solved with fewer candidates and/or lower error because learned abstractions and guidance persist.

### B. Implicit neural function model

Avoids discrete DSL search, but retains knowledge only in shared weights and requires a much broader training distribution. Existing learned-feature, atom, neural-process, and meta-objective results already show that ordinary shared representation training can capture signal while failing absent-mechanism transport. A new run would require a genuinely larger corpus and cannot cleanly attribute improvement to acquired reusable structure.

## Recommendation

Choose A for the next phase because it tests the missing accumulation mechanism directly. Treat it as a bounded mechanism study, not the final system-general solution. Do not launch another fixed-basis, kernel, residual-width, or monolithic architecture sweep first.
