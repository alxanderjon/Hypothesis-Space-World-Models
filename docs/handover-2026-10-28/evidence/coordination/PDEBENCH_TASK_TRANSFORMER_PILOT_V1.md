# PDEBench task-Transformer pilot v1

Status: completed; `NOT_SUPPORTED` on opened development data.

Input is the official compact subset in `pdebench_range_subset_v1`.  Only the
six advection/Burgers source files determine normalization and model selection.
Reaction-diffusion is the opened whole-mechanism development target.  Five
reserved trajectories per file and all diffusion-sorption data remain absent.

Each window contains four fully observed field states followed by four query
states.  The 64 spatial points are grouped into 16 fixed contiguous patches of
four.  A context token contains four normalized field values, physical-domain
normalized patch-center `x`, and globally scaled physical time `t/2`; there are
64 context tokens.  A query token contains only `(x,t/2)` and requests four
field values; there are 64 query tokens.  No PDE ID, coefficient, equation,
solver, family label or future field enters the model.

Architecture: `d_model=128`, four attention heads, three 256-wide pre-norm
Transformer encoder layers and three matching decoder layers, linear context
and query projections, and a four-value output head.  Exact parameter count is
established by preflight.  This is a standard direct class-model Transformer,
not a novel learning mechanism.

Prospective scientific comparison after preflight: two seeds; 300 source
updates per seed; then frozen, real-replay adaptation, no-replay adaptation and
scratch arms.  Adaptation uses 100 updates and the same two target windows per
update.  Every arm also executes two source-window forwards; replay includes
their loss, while no-replay/scratch multiply it by zero, matching forward and
target exposure.  Source replay uses a fixed reservoir of source training
windows.  No target outcome controls update count, normalization or selection.

Evaluation uses the 15 downloaded non-adaptation reaction trajectories per
setting and 30 source-development trajectories at fixed starts 0, 32 and 64,
plus a fixed cross-trajectory context shuffle.  Useful accumulation requires
both seeds to beat frozen and scratch target aggregate by at least 20%, win at
least 60% of target blocks, keep source aggregate within 10% of frozen with at
least 80% source blocks within 10%, and beat shuffled context by 10%.  This is
opened development evidence only.

Prospective ceiling after preflight: eight model constructions/fits, 1,200
optimizer updates, 4,800 training windows, 2,880 evaluation blocks, 900 seconds,
1 GiB RSS and 128 MiB artifacts.  Any change to tokens, normalization, update
counts or target split requires a new namespace; no architecture or budget
sweep follows automatically.

## Completed outcome

The run used eight fits, 600 source updates, 600 adaptation updates, 4,800
training windows and 2,880 evaluation blocks.  It took 42.46 seconds and
peaked at 340 MiB RSS.  In seed `202610281`, replay was `2.224x` frozen and
`1.084x` scratch target error.  In seed `202610282`, replay was `0.720x`
frozen and `0.475x` scratch.  No-replay adaptation beat replay in both seeds
(`replay/no-replay = 1.572x/1.445x`).  Aggregate source retention was
`1.007x/1.035x`, but only `61/90` and `57/90` source blocks remained within
10%, below the declared reliability gate.

The shuffled-context aggregate ratio rounded to `1.000x` in both seeds and all
arms.  This was not literal rowwise equality: among 135 paired target blocks
per arm, 118 differed, with maximum absolute MSE changes from `0.0357` to
`0.0565`.  The changes cancel or are too small in aggregate, so the declared
observation-use gate failed.  The result supports a shortcut diagnosis for
this architecture and token contract, not a claim that the official fields
contain no mechanism information.

Evidence: `data/pdebench_range_subset_v1/task_transformer_pilot_v1/result.json`.
