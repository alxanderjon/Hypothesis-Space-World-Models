#!/usr/bin/env python3
"""Short independent-weight float64 regression with explicit symmetry examples."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_nearest_margin as base


OUTPUT_DIR = Path("outputs/sho/stage1B/nearest_distance_margin_symmetry")
SMOKE_PATH = OUTPUT_DIR / "nearest_margin_symmetry_smoke.csv"
FULL_PATH = OUTPUT_DIR / "nearest_margin_symmetry_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "nearest_distance_margin_frozen_candidate.pt"
MAX_STEPS = 12
EQUAL_COUNT = 4096


def predict(model, inputs):
    # Evaluate the requested two products and sum without weight tying or
    # replacing the forward computation with a scaled input difference.
    w = model.weight.detach().numpy()[0]
    return w[0] * inputs[:, 0] + w[1] * inputs[:, 1]


def equal_inputs(pool):
    values = np.unique(pool.inputs.ravel())
    indices = np.linspace(0, len(values) - 1, min(EQUAL_COUNT, len(values))).astype(int)
    values = values[indices]  # Actual real input values, no interpolated quantiles.
    return np.column_stack((values, values))


def train(pool):
    torch.manual_seed(base.MODEL_SEED)
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    initial = model.weight.detach().numpy()[0].copy()
    equal = equal_inputs(pool)
    blocks = [(pool.inputs, pool.target),
              (pool.inputs[:, ::-1].copy(), -pool.target),
              (equal, np.zeros(len(equal), dtype=np.float64))]
    # Equal mass per supervision block. Both weights remain independent.
    # Newton steps minimize only continuous MSE. Computing gradients from
    # explicit residuals avoids cancellation in an expanded quadratic loss.
    gram = sum(x.T @ x / len(x) for x, _ in blocks) / len(blocks)

    def objective():
        return float(sum(np.mean((predict(model, x) - y) ** 2)
                         for x, y in blocks) / len(blocks))

    best_loss = objective()
    best = copy.deepcopy(model.state_dict())
    history = []
    for step in range(1, MAX_STEPS + 1):
        gradient_half = sum(x.T @ (predict(model, x) - y) / len(x)
                            for x, y in blocks) / len(blocks)
        delta = np.linalg.solve(gram, gradient_half)
        with torch.no_grad():
            model.weight.sub_(torch.from_numpy(delta).reshape(1, 2))
        loss = objective()
        w = model.weight.detach().numpy()[0]
        history.append(dict(record_type="training_checkpoint", step=step,
                            training_objective=loss, learned_w1=float(w[0]),
                            learned_w2=float(w[1]), w1_plus_w2=float(w.sum())))
        print(f"symmetry regression step={step} loss={loss:.3e} "
              f"w=({w[0]:.17g},{w[1]:.17g}) sum={w.sum():.3e}", flush=True)
        if loss < best_loss:
            best_loss, best = loss, copy.deepcopy(model.state_dict())
        if loss == 0.0 or not np.any(delta):
            break
    model.load_state_dict(best)
    return model, history, dict(initial_w1=float(initial[0]), initial_w2=float(initial[1]),
        training_pair_count=len(pool.target), swapped_training_pair_count=len(pool.target),
        equal_training_example_count=len(equal), optimizer="full_batch_residual_Newton",
        objective="mean(original_MSE,swapped_MSE,equal_input_MSE)",
        maximum_steps=MAX_STEPS, steps=len(history), training_objective=best_loss,
        model_seed=base.MODEL_SEED, dtype="float64", bias=False,
        independent_weights=True, coefficient_projection=False,
        checkpoint_selection="training_continuous_MSE_only")


def error_metrics(error):
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))),
                maximum_absolute_error=float(np.max(np.abs(error))))


def evaluate(model, pool, split):
    original = predict(model, pool.inputs)
    swapped = predict(model, pool.inputs[:, ::-1])
    equal_x = equal_inputs(pool)
    equal_prediction = predict(model, equal_x)
    rows = []
    summaries = {}
    masks = [("all", np.ones(len(pool.target), dtype=bool)),
             ("ordinary_pairs", pool.category < 2),
             ("nearest_runner_up_pairs", pool.category >= 2),
             ("all_one_ulp", pool.one_ulp)]
    masks.extend((name, pool.category == code) for code, name in base.CATEGORY_NAMES.items())
    for orientation, predicted, target, exact in (
        ("original", original, pool.target, pool.exact_margin),
        ("swapped", swapped, -pool.target, -pool.exact_margin),
    ):
        mistakes = np.sign(predicted) != np.sign(exact)
        reversals = predicted * np.sign(exact) < 0.0
        for name, mask in masks:
            metrics = error_metrics((predicted - target)[mask]) if mask.any() else {}
            row = dict(record_type="continuous_and_ordering_metrics", split=split,
                orientation=orientation, pair_group=name, count=int(mask.sum()), **metrics,
                ordering_mistakes=int(mistakes[mask].sum()),
                reversals=int(reversals[mask].sum()),
                false_zero_count=int(((predicted == 0.0) & mask).sum()))
            rows.append(row)
            summaries[(orientation, name)] = row
        for index in np.flatnonzero(pool.one_ulp | mistakes):
            rows.append(dict(record_type="one_ulp_case" if pool.one_ulp[index] else "ordering_mistake",
                split=split, orientation=orientation, event_index=int(index),
                source_state_index=int(pool.source_state_index[index]),
                query_index=int(pool.query_index[index]), site1=int(pool.site1[index]),
                site2=int(pool.site2[index]), pair_group=base.CATEGORY_NAMES[int(pool.category[index])],
                exact_d1=float(pool.exact_d1[index]), exact_d2=float(pool.exact_d2[index]),
                x1=float(pool.inputs[index, 0 if orientation == "original" else 1]),
                x2=float(pool.inputs[index, 1 if orientation == "original" else 0]),
                m_teacher=float(target[index]), m_hat=float(predicted[index]),
                ordering_exact=not bool(mistakes[index]), reversal=bool(reversals[index]),
                one_ulp=bool(pool.one_ulp[index])))
    rows.append(dict(record_type="swap_consistency", split=split, count=len(original),
                     **error_metrics(original + swapped)))
    rows.append(dict(record_type="equal_input_error", split=split, count=len(equal_x),
                     **error_metrics(equal_prediction)))
    # A BLAS implementation of the same two-weight model must also pass; it is
    # reported separately to expose arithmetic-path sensitivity at one ULP.
    blas_mistakes = 0
    for orientation, inputs, exact in (("original", pool.inputs, pool.exact_margin),
                                       ("swapped", pool.inputs[:, ::-1].copy(), -pool.exact_margin)):
        prediction = base._predict(model, inputs)
        count = int(np.sum(np.sign(prediction) != np.sign(exact)))
        blas_mistakes += count
        rows.append(dict(record_type="blas_ordering_check", split=split,
                         orientation=orientation, count=len(prediction), ordering_mistakes=count))
    passed = all(row["ordering_mistakes"] == 0 for row in summaries.values())
    passed = passed and blas_mistakes == 0 and pool.locked_zero_margin_count == 0
    return rows, summaries, passed


def run():
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in (SMOKE_PATH, FULL_PATH, CANDIDATE_PATH)):
        raise RuntimeError("Refusing to overwrite an existing symmetry experiment")
    locked_paths = sorted(Path("outputs/sho/stage1B").rglob("*locked*"))
    locked_hashes = {str(p): base._sha256(p) for p in locked_paths if p.is_file()}
    distance_model, checkpoint = base._load_locked_distance()
    payload = torch.load(base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    train_states = list(map(int, checkpoint["smoke_training_state_indices"]))[:base.SMOKE_TRAIN_STATES]
    validation_states = list(map(int, checkpoint["smoke_validation_state_indices"]))[:base.SMOKE_VALIDATION_STATES]
    held_out = list(map(int, checkpoint["full_held_out_state_indices"]))
    splits = [set(train_states), set(validation_states), set(held_out)]
    assert not any(splits[i] & splits[j] for i in range(3) for j in range(i))
    training = base._build_pool(distance_model, payload, trajectories, train_states,
        base.TRAIN_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED, "symmetry training")
    validation = base._build_pool(distance_model, payload, trajectories, validation_states,
        base.VALIDATION_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED + 1, "symmetry validation")
    model, history, training_summary = train(training)
    train_rows, _, train_passed = evaluate(model, training, "training")
    validation_rows, metrics, validation_passed = evaluate(model, validation, "validation")
    smoke_passed = train_passed and validation_passed
    weights = model.weight.detach().numpy()[0]
    summary = dict(record_type="summary", status="passed" if smoke_passed else "failed",
        learned_w1=float(weights[0]), learned_w2=float(weights[1]), w1_plus_w2=float(weights.sum()),
        **training_summary, training_state_count=len(train_states),
        validation_state_count=len(validation_states), validation_pair_count=len(validation.target),
        training_state_indices=json.dumps(train_states), validation_state_indices=json.dumps(validation_states),
        held_out_state_indices=json.dumps(held_out),
        validation_ordering_mistakes=metrics[("original", "all")]["ordering_mistakes"],
        validation_swapped_ordering_mistakes=metrics[("swapped", "all")]["ordering_mistakes"],
        hard_gate="zero_sign_mistakes_including_false_zeros_both_orientations",
        meaningful_ordering_definition="every_non_tied_pair_no_tolerance_exemptions",
        pair_scope="all_decisive_nearest_runner_up_and_sampled_ordinary_real_pairs",
        one_ulp_scope="every_non_tied_one_ULP_case_in_each_state_pair_pool",
        exact_tie_policy="deterministic_cKDTree_bookkeeping_not_learned",
        training_exact_ties_excluded=training.exact_tie_exclusion_count,
        validation_exact_ties_excluded=validation.exact_tie_exclusion_count,
        full_pool_evaluated=False, candidate_saved=False)

    def verify_locks():
        assert all(base._sha256(Path(p)) == digest for p, digest in locked_hashes.items())
        summary["locked_file_count_verified_unchanged"] = len(locked_hashes)
        summary["locked_distance_sha256"] = base.LOCKED_DISTANCE_SHA256

    verify_locks()
    base._write_csv(SMOKE_PATH, [summary] + train_rows + validation_rows + history)
    print(f"SMOKE {'PASS' if smoke_passed else 'FAIL'}: {summary}", flush=True)
    if not smoke_passed:
        print("STOP: smoke ordering gate failed; no full-pool evaluation or candidate.", flush=True)
        return
    full = base._build_pool(distance_model, payload, trajectories, held_out,
        base.FULL_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED + 2, "symmetry full held-out")
    full_rows, full_metrics, passed = evaluate(model, full, "full_held_out")
    full_summary = dict(record_type="summary", status="FROZEN_CANDIDATE" if passed else "failed",
        component="nearest_distance_margin", held_out_state_count=len(held_out),
        held_out_pair_count=len(full.target), learned_w1=float(weights[0]),
        learned_w2=float(weights[1]), w1_plus_w2=float(weights.sum()),
        all_ordering_mistakes=full_metrics[("original", "all")]["ordering_mistakes"],
        swapped_ordering_mistakes=full_metrics[("swapped", "all")]["ordering_mistakes"],
        locked_zero_margin_count=full.locked_zero_margin_count,
        exact_ties_excluded=full.exact_tie_exclusion_count)
    verify_locks()
    if passed:
        candidate = dict(format_version=1, component="nearest_distance_margin", status="FROZEN_CANDIDATE",
            architecture="m_hat=w1*x1+w2*x2; independent weights; no bias", dtype="float64",
            evaluation="two_float64_products_then_add", state_dict=copy.deepcopy(model.state_dict()),
            training_state_indices=train_states, validation_state_indices=validation_states,
            full_held_out_state_indices=held_out, training_summary=training_summary,
            full_summary=full_summary, locked_distance_sha256=base.LOCKED_DISTANCE_SHA256,
            exact_tie_policy=summary["exact_tie_policy"])
        torch.save(candidate, CANDIDATE_PATH)
        digest = base._sha256(CANDIDATE_PATH)
        CANDIDATE_PATH.with_suffix(".sha256").write_text(f"{digest}  {CANDIDATE_PATH.name}\n")
        loaded = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
        assert torch.equal(loaded["state_dict"]["weight"], model.weight)
        full_summary.update(checkpoint_path=str(CANDIDATE_PATH), checkpoint_sha256=digest)
        summary.update(candidate_saved=True, checkpoint_sha256=digest)
    summary.update(full_pool_evaluated=True, elapsed_seconds=time.monotonic() - started)
    base._write_csv(FULL_PATH, [full_summary] + full_rows)
    base._write_csv(SMOKE_PATH, [summary] + train_rows + validation_rows + history)
    print(f"FULL RESULT: {full_summary}", flush=True)


if __name__ == "__main__":
    run()
