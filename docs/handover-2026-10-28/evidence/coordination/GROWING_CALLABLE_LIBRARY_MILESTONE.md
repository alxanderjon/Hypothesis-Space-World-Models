# Growing callable library milestone

Date: 2026-10-27

## Why the fixed polynomial branch was stopped

The nine-monomial choose-three studies test search guidance inside a human-supplied final-term library. They do not demonstrate how the system would acquire an operation or reusable composition that the library does not already contain. The planned active-candidate-disagreement follow-up was therefore not executed.

The current learner history already distinguishes three different failures:

- learned 16-atom callable features did not transport absent nonlinear structure competitively (`scripts/run_learned_rk4_atom_transport_v1.py` and `diagnostics/learned_rk4_atom_transport_v1/result.json`);
- callable RKHS blocks sometimes transferred but later dictionary expansion changed the ridge inference problem and failed reverse-order retention (`scripts/run_cross_mechanism_rkhs_callable_memory_v1.py` and its result);
- polynomial guide variants ranked combinations of supplied terms but did not expand their representation (`diagnostics/balanced_guided_structure_search_v1`, `categorical_guided_structure_search_v1_recovery1`, and `pairwise_energy_guided_structure_search_v1`).

## Mechanism tested

`growing_callable_program_library_v1` replaced final-term selection with a small compositional program language. The supplied universal assumptions were explicit: 2D `(q,v)`, `q'=v`, `dt=.05`, Euler transitions, and the primitive operations `q`, `v`, addition, multiplication, `sin`, and `tanh`. Each candidate still fitted four coefficients per new system from 24 transitions and used eight transitions for candidate selection.

Twelve acquisition systems shared a hidden composition, four were distractors, and no family IDs or true programs entered search or admission. The admission rule mined subtrees from observation-selected source programs and retained the one with maximum positive description-length saving and at least six source-system occurrences. It admitted `sin(q*v)` independently under both search orders.

Development systems used four outer compositions never present as source programs. The learned primitive was compared at 256 candidates against base-language search at 256, a same-size sham primitive at 256, and base-language search at 512. Query responses were generated only after predictions persisted.

## Verified development result

Both search orders passed every frozen gate. The learned library won all 16 systems against all three controls. Mean transition MSE was about `1.84e-14` and `2.43e-14`; base-256 means were `1.28e-5` and `1.09e-5`, and base-512 means were `1.44e-6` and `8.40e-6`. Cyclic wrong-structure controls lost 16/16.

Exact work: 49,152 candidate coefficient fits, 1,179,648 fit rows, 393,216 selection rows, 128 query blocks, 2,048 generated transition rows, and 256 score blocks. Runtime was 2.75 seconds and peak RSS 46.8 MB. No reserved confirmation response was generated.

Sources: `protocols/growing_callable_program_library_v1/plan.json`, `scripts/run_growing_callable_program_library_v1.py`, and `diagnostics/growing_callable_program_library_v1/result.json`.

## Fresh confirmation

The admitted primitive, base language, sham, ordering, budgets and thresholds were frozen. Four reserved outer compositions and 16 new coefficient vectors were then evaluated once. Both search orders passed every gate. Learned/base-256 aggregate ratios were `0.02199` and `0.00783`; learned/base-512 ratios were `0.07885` and `0.03093`. The learned library won 16/16 against base-256 and sham, 12/16 against base-512, and 16/16 against cyclic wrong structures.

Exact confirmation work: 40,960 coefficient fits, 983,040 fit rows, 327,680 selection rows, 128 query blocks, 1,536 generated confirmation rows, and 256 score blocks. Runtime was 2.36 seconds and peak RSS 46.4 MB.

Sources: `protocols/growing_callable_program_library_confirmation_v1/plan.json`, `scripts/run_growing_callable_program_library_confirmation_v1.py`, and `diagnostics/growing_callable_program_library_confirmation_v1/result.json`.

## What this establishes

The project now has fresh evidence for one necessary accumulated-learning capability: an observation-derived recurring subprogram can be retained as callable knowledge and can make later search over new compositions much cheaper than matched base search. The persistent object is an executable AST, while each encounter still performs supervised coefficient fitting and bounded program search.

This is closer to DreamCoder-style library induction than to the closed polynomial guide. DreamCoder alternates program search with learned abstraction and search guidance; recent symbolic-regression work likewise reports benefits from reusable motif libraries. These precedents justify the mechanism, not generality of this result.

Primary references:

- Ellis et al., *DreamCoder* (PLDI 2021): https://arxiv.org/abs/2006.08381
- Kamienny et al., *Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search* (ICML 2023): https://proceedings.mlr.press/v202/kamienny23a.html
- Grayeli et al., *Symbolic Regression with a Learned Concept Library* (NeurIPS 2024): https://arxiv.org/abs/2409.09359
- Scherk and Chen, *SymMatika: Structure-Aware Symbolic Discovery* (2025 preprint): https://arxiv.org/abs/2507.03110

## Limits and next research decision

This is a deliberately favorable 2D unit test. The later systems intentionally reused the source motif; the base operator language already contained the operations needed to construct it; state banks were controlled; and the model class was one nonlinear callable plus linear terms. The result does not show that the learner can invent an operation outside the language, infer unrestricted dynamics from finite data, handle passive trajectories, or scale to high-dimensional interactions.

Do not return to polynomial policy tuning or spend compute widening this toy grammar. The next question is representation expansion beyond current language closure: can a system detect that all current programs are systematically inadequate, learn a callable residual module from acquisition systems using observations only, admit it with source-only transfer evidence, and then compose it with existing symbolic/library elements on later systems? This must be compared against a fixed universal approximator, a random callable, added search compute, and the frozen symbolic library. It must also declare the stored callable's parameter/byte cost and new-system coefficient/search cost.

That next phase should use whole-mechanism acquisition/transfer splits and should not claim success merely because a human-designed hidden motif recurs. A learned neural or kernel callable is an escape hatch for functions outside the symbolic closure; symbolic compression is useful only after repeated evidence shows that the callable structure recurs.
