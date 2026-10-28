# PDEBench Markov paired-replay v1

Status: completed; `NOT_SUPPORTED`.

The Markov pilot established observation use but confounded replay with a
different stochastic batch sequence. This comparison loads each valid frozen
source checkpoint and makes two copies. At every one of 100 updates, replay
and no-replay receive the exact same two reaction-diffusion windows and two
source windows. Both compute both losses; replay optimizes
`target + 0.5*source`, while no-replay optimizes `target + 0*source`. Optimizer
state, update order and initialization are otherwise identical. This is a
causal replay-loss comparison on opened development data, not a new model or a
fresh confirmation.

The old frozen and scratch predictions are reused without model evaluation.
The four new fits are evaluated on the same source and target blocks, including
history and current-field shuffles. Support requires both seeds to reduce
target aggregate error by at least 20% versus paired no-replay while retaining
source aggregate within 10% of frozen and at least 80% of source blocks within
10%. Observation controls remain descriptive and must not regress below the
old declared 1.05 history and 1.10 current-field thresholds. No replay-weight,
batch, optimizer or checkpoint selection follows.

Prospective ceiling: four fits, 400 optimizer updates, 1,600 unique sampled
windows / 3,200 model-window forwards, 2,640 new evaluation blocks, 300 seconds,
1 GiB RSS and 96 MiB artifacts. No simulator, network or teacher work.

## Outcome

Replay was worse than the exactly paired no-replay arm in both seeds:
`1.882x` and `1.660x` target aggregate error, winning only `10/180` and
`3/180` target blocks.  It still improved strongly over frozen
(`0.0539x/0.255x`) and preserved aggregate source error
(`0.931x/0.864x`), but seed two retained only `50/120` individual source blocks
within 10%.  History/current-field shuffle ratios were `1.037x/1.504x` and
`2.942x/3.534x`; seed one narrowly missed the predeclared history-use gate.

This establishes a replay stability/plasticity tradeoff for the fixed update
and weight, not a universal failure of replay.  The replay coefficient is now
closed to outcome-driven tuning.  Exact work: four fits, 400 optimizer updates,
1,600 unique sampled windows / 3,200 model-window forwards, 2,640 new evaluation
blocks, 16.87 seconds and 341,147,648 bytes peak RSS.  Evidence:
`data/pdebench_range_subset_v1/task_transformer_markov_paired_replay_v1/result.json`.
