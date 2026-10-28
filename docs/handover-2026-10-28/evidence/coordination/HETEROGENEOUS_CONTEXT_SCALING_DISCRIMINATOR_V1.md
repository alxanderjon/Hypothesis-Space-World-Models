# Heterogeneous context-scaling discriminator v1

Status: prospective before scientific execution.

## Question

Did the whole-mechanism failure primarily reflect a supplied-interface scale/statistics mismatch, or does the frozen entity/context learner still fail to transport genuinely new dynamics after that mismatch is removed?

This is not a new corpus, a new architecture, or a claim that common tensor shape gives common semantics. Earlier same-`(q,v)` oscillator studies already tested within-interface family transfer, and the 4D common-panel studies tested deliberately recurring callable structure. Here SHO, bouncing-ball, and Kepler remain mechanically different generators.

## Fixed intervention

For each system independently, derive one RMS scale per active entity/channel from its 32 permitted context state rows and one RMS scale per active entity/channel from its 32 observed increments. Divide context and query states by the state scales and divide context and supervised query increments by the increment scales. Inactive channels use scale one. No centering, clipping, hidden group/law metadata, simulator parameters, equations, query answers, or development labels enter the transform.

The architecture (22,724 parameters), Adam settings, seeds, 500 updates per fit, batches, three whole-mechanism folds, target tensors, masks, `delta_t`, and all original controls remain fixed. The intervention therefore changes only the observation/target coordinate scale supplied to the same learner. It explicitly supplies position/velocity channel identities, entity masks, timestep, and the fact that matching channels may be rescaled from a context.

## Controls and interpretation

- `correct`: correct context code and its own context-derived scales.
- `blind`: mean training-context code, but the evaluated system's own scales. This measures how much scale statistics alone help.
- `code_wrong`: cyclic wrong context code with the evaluated system's own scales.
- `full_wrong`: cyclic wrong context code and cyclic wrong scales.
- `time_blind`: correct code/scales with normalized time removed.
- `scratch`: same scaled interface, architecture, updates, and target-group data from reset.
- Saved unscaled shared/scratch predictions are comparison baselines and are not retrained.

The unchanged forward gate is primary: shared/scratch MSE <= 0.8, shared/blind <= 0.8, at least 6/8 correct-context wins against `full_wrong`, and rollout shared/scratch <= 1.0 in every group/seed cell. Scaled-versus-unscaled error changes are causal diagnostics, not substitute gates.

Improvement without correct-context identity would support calibration/interface mismatch but not rule inference. Passing the original gate would support context-derived scale canonicalization for these three supplied interfaces, not arbitrary-mechanism discovery. Failure after stable optimization would weaken scale/statistics as the main explanation for this model, while leaving representation, semantics, and genuinely new dynamics unresolved.

## Prospective all-in bounds

- Existing data only; zero teacher calls, trajectories, or transition generation.
- 12 fresh fits, 6,000 optimizer updates.
- 1,536,000 context rows and 768,000 query rows used in training.
- 18,432 one-step development query rows, 480 rollout prediction steps.
- 600 seconds wall time, 2 GiB peak RSS, 128 MiB artifacts.
- All evaluated cohorts are already-opened development evidence. No confirmation cohort is opened.

