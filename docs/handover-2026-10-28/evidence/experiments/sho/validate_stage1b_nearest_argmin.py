#!/usr/bin/env python3
"""Full learned site scan with exact-tie-only deterministic scaffolding."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_nearest_margin as base
import experiments.sho.run_stage1b_nearest_margin_symmetry as margin
import experiments.sho.run_stage1b_nearest_threshold as threshold
import experiments.sho.validate_stage1b_distance_assembled as downstream

OUT = Path("outputs/sho/stage1B/nearest_argmin_assembled")
ASSEMBLED = OUT / "nearest_argmin_assembled.csv"
DOWNSTREAM = OUT / "nearest_argmin_downstream.csv"
SUMMARY = OUT / "nearest_argmin_checkpoint_hash_summary.csv"
THRESHOLD = OUT / "nearest_distance_threshold_frozen_candidate.pt"
OPERATOR = OUT / "nearest_argmin_operator_locked.pt"
SCAFFOLD = "TEMPORARY DETERMINISTIC SCAFFOLDING"


def prepare_threshold(distance, frozen, checkpoint, payload, trajectories):
    """Repeat only the tiny threshold fit; never optimize upstream parameters."""
    pools = []
    for key, count, seed in (("training_state_indices", base.TRAIN_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED),
                             ("validation_state_indices", base.VALIDATION_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED+1)):
        pools.append(base._build_pool(distance, payload, trajectories, checkpoint[key], count, seed, key))
    train, valid = pools
    ti = threshold.balanced_indices(train, 8192, threshold.MODEL_SEED)
    model, history, summary = threshold.train(margin.predict(frozen, train.inputs[ti]), train.exact_margin[ti])
    rows = [dict(record_type="threshold_training_summary", **summary)] + history
    for pool, name in ((train, "threshold_training"), (valid, "threshold_validation")):
        result, passed = threshold.evaluate_pool(model, frozen, pool, name)
        assert passed, "nonzero threshold regression failed"
        rows.extend(result)
    for x, name in ((threshold.SYNTHETIC, "synthetic_boundary"), (threshold.dense_grid(), "dense_nonzero_grid")):
        result, _, _, _ = threshold.metrics(model, x, x, name, "all")
        assert result["mistakes"] == 0
        rows.append(result)
    model.eval().requires_grad_(False)
    return model, rows, summary


def scan(sites, probes, distance, frozen, threshold_model, state):
    """Scan every site. Reference distances select no non-tied winner."""
    n, p = len(sites), len(probes)
    queries = np.concatenate((probes, sites))
    qcount = len(queries)
    self_id = np.concatenate((np.full(p, -1, dtype=np.int64), np.arange(n)))
    production_tree = cKDTree(sites)
    pd, ps = production_tree.query(probes, k=2)
    sd, ss = production_tree.query(sites, k=3)
    production_distance, production_owner = production_tree.query(probes, k=1)
    spacing_distance, spacing_owner = production_tree.query(sites, k=2)
    teacher_owner = np.concatenate((production_owner, spacing_owner[:, 1]))
    teacher_distance = np.concatenate((production_distance, spacing_distance[:, 1]))
    decisive_sites = np.concatenate((ps, ss[:, 1:3]))
    decisive_distances = np.concatenate((pd, sd[:, 1:3]))
    decisive_fixed_distances = np.full((qcount, 2), np.nan)
    production_tied = decisive_distances[:, 0] == decisive_distances[:, 1]

    winner = np.full(qcount, -1, dtype=np.int64)
    winner_distance = np.zeros(qcount, dtype=np.float64)
    winner_reference = np.zeros(qcount, dtype=np.float64)
    reference_prefix_minimum = np.full(qcount, np.inf)
    ever_wrong = np.zeros(qcount, dtype=bool)
    ever_pair_error = np.zeros(qcount, dtype=bool)
    bad_streak = np.zeros(qcount, dtype=np.int64)
    max_streak = np.zeros(qcount, dtype=np.int64)
    counts = {name: np.zeros(qcount, dtype=np.int64) for name in
        ("comparisons", "exact_tie_fallbacks", "tie_replacements", "nonzero_comparisons",
         "pairwise_mistakes", "one_ulp_comparisons", "one_ulp_mistakes", "false_zero_margins",
         "prefix_nonminimal_steps")}
    details = []
    theta, alpha = float(threshold_model.theta()), float(threshold_model.alpha())
    for candidate in range(n):
        new_distance = base.distance_metric._predict(distance, queries-sites[candidate])
        # A tree with ONE fixed site cannot choose among candidates. Only its
        # returned distance is used by the exact-equality scaffolding and audit.
        candidate_reference = cKDTree(sites[candidate:candidate+1]).query(queries, k=1)[0]
        for column in (0, 1):
            mask = decisive_sites[:, column] == candidate
            decisive_fixed_distances[mask, column] = candidate_reference[mask]
        eligible = self_id != candidate
        initialize = eligible & (winner < 0)
        active = eligible & ~initialize
        idx = np.flatnonzero(active)
        current = winner[idx]
        old_d = winner_distance[idx]
        new_d = new_distance[idx]
        old_ref = winner_reference[idx]
        new_ref = candidate_reference[idx]
        ties = old_ref == new_ref
        replace = np.zeros(len(idx), dtype=bool)
        # Teacher ownership is consulted only for equal-distance comparisons,
        # and only the current or incoming site can be retained. For tied sites
        # that do not include the production owner, keep current until a
        # strictly closer site or that equally distant production owner arrives.
        replace[ties] = teacher_owner[idx[ties]] == candidate
        counts["exact_tie_fallbacks"][idx[ties]] += 1
        counts["tie_replacements"][idx[ties & replace]] += 1
        nonzero = ~ties
        m = np.full(len(idx), np.nan)
        logits = np.full(len(idx), np.nan)
        scale = np.maximum(old_d[nonzero], new_d[nonzero])
        assert np.all(scale > 0.0)
        inputs = np.column_stack((old_d[nonzero]/scale, new_d[nonzero]/scale))
        m[nonzero] = margin.predict(frozen, inputs)
        logits[nonzero] = alpha * (m[nonzero]-theta)
        replace[nonzero] = logits[nonzero] > 0.0
        # Validation-only reference signs below NEVER determine replace.
        ulp = base._ulp_separation(old_ref, new_ref) == 1
        wrong = nonzero & ((np.sign(logits) != np.sign(old_ref-new_ref))
                          | (m == 0) | ~np.isfinite(logits))
        counts["comparisons"][idx] += 1
        counts["nonzero_comparisons"][idx[nonzero]] += 1
        counts["pairwise_mistakes"][idx[wrong]] += 1
        counts["false_zero_margins"][idx[nonzero & (m == 0)]] += 1
        counts["one_ulp_comparisons"][idx[ulp]] += 1
        counts["one_ulp_mistakes"][idx[ulp & wrong]] += 1
        ever_pair_error[idx[wrong]] = True
        for j in np.flatnonzero(wrong | ulp):
            query = int(idx[j])
            details.append(dict(record_type="pairwise_error" if wrong[j] else "one_ulp_comparison",
                source_state_index=state, kind="probe" if query < p else "spacing",
                query_index=query if query < p else query-p, candidate_site=candidate,
                current_site=int(current[j]), exact_current=float(old_ref[j]), exact_candidate=float(new_ref[j]),
                learned_current=float(old_d[j]), learned_candidate=float(new_d[j]),
                margin=float(m[j]), logit=float(logits[j]), replace=bool(replace[j]),
                one_ulp=bool(ulp[j]), correct=not bool(wrong[j])))
        chosen = idx[replace]
        winner[chosen] = candidate
        winner_distance[chosen] = new_distance[chosen]
        winner_reference[chosen] = candidate_reference[chosen]
        winner[initialize] = candidate
        winner_distance[initialize] = new_distance[initialize]
        winner_reference[initialize] = candidate_reference[initialize]
        reference_prefix_minimum[eligible] = np.minimum(reference_prefix_minimum[eligible], candidate_reference[eligible])
        prefix_wrong = (winner >= 0) & (winner_reference != reference_prefix_minimum)
        ever_wrong |= prefix_wrong
        counts["prefix_nonminimal_steps"][prefix_wrong] += 1
        bad_streak = np.where(prefix_wrong, bad_streak+1, 0)
        max_streak = np.maximum(max_streak, bad_streak)

    # Cross-check the fixed-site tie-test contract against production's actual
    # nearest/runner-up cKDTree queries on EVERY query, including all near ties.
    fixed_tied = decisive_fixed_distances[:, 0] == decisive_fixed_distances[:, 1]
    tie_detector_mismatch = fixed_tied != production_tied
    wrong_owner = winner != teacher_owner
    for query in np.flatnonzero(wrong_owner | tie_detector_mismatch):
        details.append(dict(record_type="final_argmin_error" if wrong_owner[query] else "tie_detector_error",
            source_state_index=state, kind="probe" if query < p else "spacing",
            query_index=int(query if query < p else query-p), learned_owner=int(winner[query]),
            teacher_owner=int(teacher_owner[query]), production_exact_tie=bool(production_tied[query]),
            fixed_site_exact_tie=bool(fixed_tied[query]), had_pairwise_error=bool(ever_pair_error[query]),
            had_prefix_error=bool(ever_wrong[query])))
    state_rows = []
    for kind, sl in (("probe", slice(0, p)), ("spacing", slice(p, None))):
        state_rows.append(dict(record_type="scan_state", source_state_index=state, kind=kind,
            site_count=n, query_count=len(winner[sl]), **{k: int(v[sl].sum()) for k, v in counts.items()},
            final_argmin_mistakes=int(wrong_owner[sl].sum()),
            exact_owners=int(np.sum(~wrong_owner[sl])),
            authoritative_nearest_ties=int(production_tied[sl].sum()),
            authoritative_tie_detector_mismatches=int(tie_detector_mismatch[sl].sum()),
            queries_with_prefix_error=int(ever_wrong[sl].sum()),
            queries_with_pairwise_error=int(ever_pair_error[sl].sum()),
            maximum_consecutive_nonminimal_steps=int(max_streak[sl].max()),
            final_errors_following_pairwise_error=int(np.sum(wrong_owner[sl] & ever_pair_error[sl])),
            winner_reference_distance_mismatches=int(np.sum(winner_reference[sl] != teacher_distance[sl])),
            winner_learned_distance_mismatches=int(np.sum(winner_distance[sl] != teacher_distance[sl])),
            max_learned_distance_error=float(np.max(np.abs(winner_distance[sl]-teacher_distance[sl])))))
    assert counts["comparisons"][:p].sum() == p*(n-1)
    assert counts["comparisons"][p:].sum() == n*(n-2)
    assert np.all(winner[p:] != np.arange(n))
    # Downstream values come from the locked learned metric even on tie-owned
    # sites. Do not silently replace learned values with teacher distances.
    teacher_radius, learned_radius = np.zeros(n), np.zeros(n)
    np.maximum.at(teacher_radius, teacher_owner[:p], teacher_distance[:p])
    np.maximum.at(learned_radius, winner[:p], winner_distance[:p])
    teacher_size = np.maximum(teacher_radius, 0.5*teacher_distance[p:])
    learned_size = np.maximum(learned_radius, 0.5*winner_distance[p:])
    numeric = dict(record_type="cell_numeric_state", source_state_index=state, site_count=n)
    for name, learned, truth in (("assigned_radius", learned_radius, teacher_radius),
                                ("spacing", winner_distance[p:], teacher_distance[p:]),
                                ("cell_size", learned_size, teacher_size)):
        err = np.abs(learned-truth)
        numeric[name+"_mismatches"] = int(np.sum(err != 0))
        numeric[name+"_maximum_absolute_error"] = float(err.max())
    return state_rows+details, numeric, teacher_size, learned_size


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if ASSEMBLED.exists() or OPERATOR.exists():
        raise RuntimeError("Refusing to overwrite an existing assembled run")
    protected = set(downstream._relevant_checkpoint_paths())
    protected.update(p for p in Path("outputs/sho/stage1B").rglob("*locked*") if p.is_file())
    protected.update((margin.CANDIDATE_PATH, margin.CANDIDATE_PATH.with_suffix(".sha256")))
    hashes = {str(p): base._sha256(p) for p in protected}
    assert hashes[str(margin.CANDIDATE_PATH)] == threshold.MARGIN_SHA256
    distance, _ = base._load_locked_distance()
    checkpoint = torch.load(margin.CANDIDATE_PATH, weights_only=False)
    frozen = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    frozen.load_state_dict(checkpoint["state_dict"])
    frozen.eval().requires_grad_(False)
    locked_downstream = downstream._load_downstream()
    payload = torch.load(base.turn.RAW_TRACE_PATH, weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    held_out = checkpoint["full_held_out_state_indices"]
    assert not set(held_out) & (set(checkpoint["training_state_indices"]) | set(checkpoint["validation_state_indices"]))
    model, training_rows, train_summary = prepare_threshold(distance, frozen, checkpoint, payload, trajectories)
    rows, numeric_rows, downstream_states = [], [], []
    for position, state in enumerate(held_out, 1):
        sites, probes, record = base.distance_metric._state_geometry(payload, trajectories, state)
        state_rows, numeric, teacher_size, learned_size = scan(sites, probes, distance, frozen, model, state)
        rows.extend(state_rows)
        numeric_rows.append(numeric)
        losses = np.asarray(trajectories[int(record["trajectory_id"])]["evaluated_prefix_losses"][:len(sites)], dtype=np.float64)
        downstream_states.append(dict(source_state_index=state, trajectory_id=int(record["trajectory_id"]),
            iteration=int(record["iteration"]), state_bin=base.turn._state_bin(len(sites)),
            losses=losses, teacher_sizes=teacher_size, learned_sizes=learned_size))
        scan_rows = [r for r in state_rows if r["record_type"] == "scan_state"]
        print(f"assembled {position}/{len(held_out)} state={state} sites={len(sites)} "
            f"pair_errors={sum(r['pairwise_mistakes'] for r in scan_rows)} "
            f"owner_errors={sum(r['final_argmin_mistakes'] for r in scan_rows)} "
            f"cell_size_mismatches={numeric['cell_size_mismatches']} elapsed={time.monotonic()-started:.1f}s", flush=True)
        base._write_csv(ASSEMBLED, [dict(record_type="progress", status="running", completed_states=position)] + rows)
    downstream_rows, down_summary = downstream._downstream_validation(downstream_states, locked_downstream)
    summaries = []
    for kind in ("probe", "spacing"):
        selected = [r for r in rows if r["record_type"] == "scan_state" and r["kind"] == kind]
        summed = [k for k in selected[0] if k not in ("record_type", "source_state_index", "kind", "maximum_consecutive_nonminimal_steps", "max_learned_distance_error")]
        summary = dict(record_type="scan_summary", kind=kind, **{k: sum(r[k] for r in selected) for k in summed})
        summary["maximum_consecutive_nonminimal_steps"] = max(r["maximum_consecutive_nonminimal_steps"] for r in selected)
        summary["max_learned_distance_error"] = max(r["max_learned_distance_error"] for r in selected)
        summaries.append(summary)
    numeric_summary = dict(record_type="cell_numeric_summary", state_count=len(held_out))
    for name in ("assigned_radius", "spacing", "cell_size"):
        numeric_summary[name+"_mismatches"] = sum(r[name+"_mismatches"] for r in numeric_rows)
        numeric_summary[name+"_maximum_absolute_error"] = max(r[name+"_maximum_absolute_error"] for r in numeric_rows)
    # Require exact discrete behavior and exact numeric downstream values; this
    # is stricter than accepting small errors as numerically insignificant.
    passed = all(s[k] == 0 for s in summaries for k in
        ("pairwise_mistakes", "final_argmin_mistakes", "one_ulp_mistakes", "authoritative_tie_detector_mismatches", "prefix_nonminimal_steps"))
    passed &= all(numeric_summary[name+"_mismatches"] == 0 for name in ("assigned_radius", "spacing", "cell_size"))
    passed &= down_summary["exact_complete_downstream_state_count"] == len(held_out)
    assert all(base._sha256(Path(p)) == h for p, h in hashes.items())
    gate = dict(record_type="assembled_gate", component="nearest_argmin_operator", status="LOCKED" if passed else "FAILED",
        held_out_states=len(held_out), tie_routing=SCAFFOLD, scan_order="original_site_index_ascending",
        non_tied_winner_source="locked_distance_then_frozen_margin_then_learned_threshold_only",
        tie_detector="exact_equality_of_cKDTree_single_fixed_site_distances",
        tie_policy="on_equal_distances_prefer_production_owner_if_in_pair_else_keep_current",
        full_candidate_scan=True, threshold_retrained_only=True, protected_files_verified_unchanged=len(hashes),
        elapsed_seconds=time.monotonic()-started, learned_theta=float(model.theta()), learned_alpha=float(model.alpha()))
    hash_rows = [dict(record_type="existing_checkpoint", checkpoint_path=p, sha256=h, unchanged=True) for p,h in sorted(hashes.items())]
    if passed:
        threshold_ckpt = dict(format_version=1, component="nearest_distance_threshold", status="FROZEN_CANDIDATE",
            state_dict=copy.deepcopy(model.state_dict()), learned_theta=float(model.theta()), learned_alpha=float(model.alpha()),
            dtype="float64", architecture="alpha*(m-theta); alpha=softplus(raw_alpha)/scale; theta=scale*scaled_theta",
            training_summary=train_summary, tie_routing=SCAFFOLD, margin_sha256=threshold.MARGIN_SHA256)
        torch.save(threshold_ckpt, THRESHOLD)
        thash = base._sha256(THRESHOLD)
        THRESHOLD.with_suffix(".sha256").write_text(f"{thash}  {THRESHOLD.name}\n")
        operator_ckpt = dict(format_version=1, component="nearest_argmin_operator", status="LOCKED", tie_routing=SCAFFOLD,
            distance_checkpoint=str(base.LOCKED_DISTANCE_PATH), distance_sha256=base.LOCKED_DISTANCE_SHA256,
            margin_checkpoint=str(margin.CANDIDATE_PATH), margin_sha256=threshold.MARGIN_SHA256,
            threshold_checkpoint=str(THRESHOLD), threshold_sha256=thash,
            implementation_path=__file__, implementation_sha256=base._sha256(Path(__file__)),
            held_out_state_indices=held_out, assembly_summary=gate, scan_summaries=summaries,
            numeric_summary=numeric_summary, downstream_summary=down_summary)
        torch.save(operator_ckpt, OPERATOR)
        ohash = base._sha256(OPERATOR)
        OPERATOR.with_suffix(".sha256").write_text(f"{ohash}  {OPERATOR.name}\n")
        hash_rows += [dict(record_type="new_checkpoint", checkpoint_path=str(THRESHOLD), sha256=thash, status="FROZEN_CANDIDATE"),
                      dict(record_type="new_checkpoint", checkpoint_path=str(OPERATOR), sha256=ohash, status="LOCKED")]
        loaded = threshold.NonzeroThreshold()
        loaded.load_state_dict(torch.load(THRESHOLD, weights_only=False)["state_dict"])
        assert torch.equal(loaded(torch.from_numpy(threshold.dense_grid())), model(torch.from_numpy(threshold.dense_grid())))
    base._write_csv(ASSEMBLED, [gate]+summaries+rows+training_rows)
    base._write_csv(DOWNSTREAM, [numeric_summary]+numeric_rows+downstream_rows)
    base._write_csv(SUMMARY, [gate]+hash_rows)
    print(json.dumps(dict(gate=gate, scan=summaries, numeric=numeric_summary, downstream=down_summary), indent=2), flush=True)


if __name__ == "__main__":
    run()
