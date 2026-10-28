#!/usr/bin/env python3
"""Read-only audit of N1-visible state for checked Kepler GOOD/BAD candidates.

GOOD/BAD labels come from the already-saved unchanged N2->N3 counterfactual
checks.  Teacher parameters and recovery diagnostics are deliberately excluded
from every runtime feature assembled here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.sho import audit_stage1b_po_cell_selection as po_audit
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize


SOURCE = Path("outputs/d5_kepler_persistent_localization_diagnosis/report.json")
STATES = Path("outputs/d5_handoff_continuation_20000/cases")
OUT = Path("outputs/n1_kepler_good_bad_information_audit")
LOWER = np.asarray([-.25, -1., -2., -.5, 1.5], dtype=np.float64)
UPPER = np.asarray([.5, 2., 3., 2.5, 4.5], dtype=np.float64)
WIDTH = UPPER - LOWER


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _rank(values: np.ndarray, descending: bool = False) -> np.ndarray:
    ids = np.arange(len(values))
    order = np.lexsort((ids, -values if descending else values))
    out = np.empty(len(values), dtype=np.int64)
    out[order] = np.arange(1, len(values) + 1)
    return out


def _state(case_id: int, count: int, iteration: int):
    path = STATES / f"case_{case_id:03d}" / "n1_state.npz"
    saved = np.load(path)
    points = np.asarray(saved["points"][:count], dtype=np.float64)
    losses = np.asarray(saved["losses"][:count], dtype=np.float64)
    api = edirect.composition(5)
    generator_id = 6_300_000 + case_id
    data = api["_cell_data"](points, losses, generator_id, iteration)
    selected, meta = po_audit._standalone_po_selection(data.sizes, losses)
    if not np.array_equal(selected, data.selected):
        raise RuntimeError(f"case {case_id}: PO replay mismatch")
    surface = GaussianLocalResponse(points, losses)
    prefix_bytes = np.ascontiguousarray(points).tobytes() + np.ascontiguousarray(losses).tobytes()
    state_dir = OUT / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"case_{case_id:03d}_n{count}_it{iteration}.npz"
    if not state_path.exists():
        np.savez_compressed(state_path, points=points, losses=losses,
                            lower=LOWER, upper=UPPER,
                            case_id=np.int64(generator_id), iteration=np.int64(iteration))
    return dict(points=points, losses=losses, api=api, data=data, selected=selected,
                meta=meta, surface=surface, generator_id=generator_id,
                state_path=str(state_path), state_sha256=hashlib.sha256(prefix_bytes).hexdigest())


def _hull_maps(meta):
    hull_position = {int(group): pos for pos, group in enumerate(meta["hull"])}
    return hull_position


def _feature_record(case_id: int, check: dict, state_record: dict) -> dict:
    count = int(check["round_start_evaluations"])
    iteration = int(state_record["iteration"])
    s = _state(case_id, count, iteration)
    points, losses, data = s["points"], s["losses"], s["data"]
    selected, meta, surface = s["selected"], s["meta"], s["surface"]
    candidate = check["candidate"]
    ci = int(candidate["candidate_index"])
    unit = np.asarray(candidate["unit_point"], dtype=np.float64)
    if not np.array_equal(unit, data.points[ci]):
        matches = np.flatnonzero(np.all(data.points == unit[None, :], axis=1))
        if len(matches) != 1:
            raise RuntimeError(f"case {case_id}: candidate identity mismatch")
        ci = int(matches[0])
    owner = int(data.owners[ci])
    selected_set = set(map(int, selected))
    loss_rank = _rank(losses)
    size_rank = _rank(data.sizes, descending=True)
    hull_position = _hull_maps(meta)
    group_index = int(meta["cell_group"][owner])
    group = meta["groups"][group_index]
    representative = int(group["sample_index"])
    dominated = np.asarray(meta["pareto_dominated"], dtype=bool)
    be = (data.sizes >= data.sizes[owner]) & (losses <= losses[owner])
    strict = (data.sizes > data.sizes[owner]) | (losses < losses[owner])
    be[owner] = False
    dominators = np.flatnonzero(be & strict)

    k = min(24, len(points))
    nd, ni = cKDTree(points).query(unit, k=k)
    nd, ni = np.atleast_1d(nd), np.atleast_1d(ni).astype(int)
    neighbor_rows = []
    for order, (distance, site) in enumerate(zip(nd, ni), start=1):
        ng = int(meta["cell_group"][site])
        neighbor_rows.append(dict(
            order=order, site_index=site, distance=float(distance),
            loss=float(losses[site]), encoded_loss=float(surface.values[site]),
            loss_rank=int(loss_rank[site]), cell_size=float(data.sizes[site]),
            size_rank=int(size_rank[site]), selected=bool(site in selected_set),
            pareto_dominated=bool(dominated[site]),
            lower_right_hull_position=hull_position.get(ng),
        ))

    owner_mask = data.owners == owner
    owner_indices = np.flatnonzero(owner_mask)
    owner_local_pos = int(np.flatnonzero(owner_indices == ci)[0])
    owner_distance_order = np.lexsort((owner_indices, -data.distances[owner_indices]))
    owner_distance_rank = int(np.flatnonzero(owner_distance_order == owner_local_pos)[0]) + 1
    predictions = surface.predict(data.points[owner_indices])
    pred_order = np.lexsort((owner_indices, predictions))
    owner_prediction_rank = int(np.flatnonzero(pred_order == owner_local_pos)[0]) + 1
    production_role = "UNROUTED_OWNER"
    score = None
    score_rank = None
    production_owner_winner = False
    selected_position = None
    selected_weight = None
    if owner in selected_set:
        selected_position = list(map(int, selected)).index(owner)
        selected_sizes = data.sizes[selected]
        weights = .5 * _normalize(selected_sizes) + .5 * _normalize(surface.loo_errors(selected))
        largest_position = int(np.argmax(selected_sizes))
        weights[largest_position] = 1.0
        selected_weight = float(weights[selected_position])
        if selected_position == largest_position:
            production_role = "LARGEST_CELL"
            scores = data.distances[owner_indices]
        else:
            production_role = "ORDINARY_CELL"
            dbar = _normalize(data.distances[owner_indices])
            pbar = _normalize(predictions)
            scores = selected_weight * dbar - (1.0 - selected_weight) * pbar
        score = float(scores[owner_local_pos])
        score_order = np.lexsort((owner_indices, -scores))
        score_rank = int(np.flatnonzero(score_order == owner_local_pos)[0]) + 1
        production_owner_winner = score_rank == 1

    evaluation = count + 1
    unit_halfwidth = evaluation ** (-1.0 / 5.0)
    local_mask = np.all(np.abs(points - unit[None, :]) <= unit_halfwidth, axis=1)
    local_indices = np.flatnonzero(local_mask)
    local_rows = [dict(site_index=int(i), insertion_evaluation=int(i)+1,
                       unit_point=points[i].tolist(), distance=float(np.linalg.norm(points[i]-unit)),
                       loss=float(losses[i]), loss_rank=int(loss_rank[i])) for i in local_indices]
    reservoir_local = np.flatnonzero(np.all(
        np.abs(data.points - unit[None, :]) <= unit_halfwidth, axis=1))

    candidate_prediction = float(surface.predict(unit[None, :])[0])
    candidate_true_loss = float(candidate["objective_loss"])
    post_losses = np.append(losses, candidate_true_loss)
    post_rank = int(_rank(post_losses)[-1])
    would_incumbent = bool(candidate_true_loss < float(np.min(losses)))
    owner_enc = float(surface.values[owner])
    dominator_rows = [dict(site_index=int(i), loss=float(losses[i]),
                           loss_rank=int(loss_rank[i]), cell_size=float(data.sizes[i]),
                           size_rank=int(size_rank[i])) for i in dominators]
    rec = dict(
        case_id=case_id, label="GOOD" if check["recovered"] else "BAD",
        label_definition="unchanged N2->N3 reached existing final recovery criterion",
        state_reference=dict(path=s["state_path"], sha256=s["state_sha256"],
                             dimension=5, history_count=count, iteration=iteration,
                             generator_id=s["generator_id"], lower=LOWER.tolist(), upper=UPPER.tolist()),
        candidate_index=ci, candidate_unit=unit.tolist(),
        candidate_theta=(LOWER + WIDTH * unit).tolist(),
        candidate_unit_0=float(unit[0]), candidate_unit_1=float(unit[1]),
        candidate_unit_2=float(unit[2]), candidate_unit_3=float(unit[3]),
        candidate_unit_4=float(unit[4]),
        normalized_domain_center_distance=float(np.linalg.norm(unit-.5)),
        normalized_nearest_bound_margin=float(np.min(np.minimum(unit, 1.-unit))),
        owning_site=owner, candidate_owner_distance=float(data.distances[ci]),
        owner_loss=float(losses[owner]), owner_encoded_loss=owner_enc,
        owner_loss_rank=int(loss_rank[owner]), owner_cell_size=float(data.sizes[owner]),
        owner_size_rank=int(size_rank[owner]), owner_potentially_optimal=owner in selected_set,
        owner_loss_size_rank_ratio=float(loss_rank[owner]/size_rank[owner]),
        owner_pareto_dominated=bool(dominated[owner]),
        owner_dominator_count=int(len(dominators)), owner_dominators=dominator_rows,
        owner_same_size_group=group_index, owner_group_representative=representative,
        owner_is_group_representative=owner == representative,
        owner_lower_right_hull_position=hull_position.get(group_index),
        owner_group_hull_fields={k: group.get(k) for k in (
            "lower_slope", "upper_slope", "required_slope", "feasible_lower_slope", "epsilon_pass")},
        production_rejection_reason=candidate["production_rejection_reason"],
        surrogate_prediction=candidate_prediction,
        surrogate_minus_owner_encoded_loss=float(candidate_prediction-owner_enc),
        nearest_sites=neighbor_rows,
        nearest_site_indices=ni.tolist(), nearest_site_distances=nd.tolist(),
        owner_reservoir_candidate_count=int(len(owner_indices)),
        owner_distance_rank_desc=owner_distance_rank,
        owner_surrogate_prediction_rank_asc=owner_prediction_rank,
        production_score_role=production_role, production_score=score,
        production_score_rank_desc=score_rank,
        production_owner_winner=production_owner_winner,
        selected_position=selected_position, selected_cell_weight=selected_weight,
        current_reservoir_local_box_count=int(len(reservoir_local)),
        evaluated_local_box_history_count=int(len(local_indices)),
        evaluated_local_box_history=local_rows,
        most_recent_local_evaluation=(int(local_indices[-1])+1 if len(local_indices) else None),
        local_history_recency=(evaluation-(int(local_indices[-1])+1) if len(local_indices) else evaluation),
        evaluation_count_if_tested=evaluation,
        implied_n2_box_unit_halfwidth=unit_halfwidth,
        # Promotion-only N1-visible values after the physical objective is evaluated.
        promotion_candidate_loss=candidate_true_loss,
        promotion_loss_rank=post_rank,
        promotion_would_be_incumbent=would_incumbent,
        promotion_loss_minus_owner_loss=float(candidate_true_loss-losses[owner]),
        # Offline label diagnostics are isolated and never used above.
        offline_final_loss=float(check["final_loss"]),
        offline_final_parameter_l2_error=float(check["final_parameter_l2_error"]),
        offline_rollout_max_error=float(check["continuation_max_error"]),
    )
    return rec


def _scalar_table(rows, fields):
    out = {}
    for field in fields:
        values = {label: [r[field] for r in rows if r["label"] == label and r[field] is not None]
                  for label in ("GOOD", "BAD")}
        if not values["GOOD"] or not values["BAD"]: continue
        g, b = np.asarray(values["GOOD"], float), np.asarray(values["BAD"], float)
        threshold_separates = bool(np.max(g) < np.min(b) or np.max(b) < np.min(g))
        out[field] = dict(GOOD=values["GOOD"], BAD=values["BAD"],
                          ranges=dict(GOOD=[float(g.min()),float(g.max())],
                                      BAD=[float(b.min()),float(b.max())]),
                          single_threshold_separates=threshold_separates)
    return out


def _category_table(rows, fields):
    out = {}
    for field in fields:
        values = {}
        for row in rows:
            key = json.dumps(row[field], sort_keys=True)
            values.setdefault(key, {"GOOD":0,"BAD":0})[row["label"]] += 1
        out[field] = values
    return out


def _nearest_cross_label(rows, fields):
    matrix = np.asarray([[float(r[f]) for f in fields] for r in rows])
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    span = np.where(hi > lo, hi-lo, 1.)
    z = (matrix-lo)/span
    pairs=[]
    for i,r in enumerate(rows):
        options=[j for j,q in enumerate(rows) if q["label"] != r["label"]]
        j=min(options,key=lambda x: float(np.linalg.norm(z[i]-z[x])))
        pairs.append(dict(case_id=r["case_id"],label=r["label"],candidate_index=r["candidate_index"],
                          other_case_id=rows[j]["case_id"],other_label=rows[j]["label"],
                          other_candidate_index=rows[j]["candidate_index"],
                          normalized_feature_distance=float(np.linalg.norm(z[i]-z[j])),
                          feature_values={f:r[f] for f in fields},
                          other_feature_values={f:rows[j][f] for f in fields}))
    return sorted(pairs,key=lambda x:x["normalized_feature_distance"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    report = json.loads(SOURCE.read_text())
    rows=[]
    for case in report["cases_detail"]:
        case_id=int(case["case_id"])
        states={int(s["round_start_evaluations"]):s for s in case["representative_states"]}
        for check in case["candidate_handoff_checks"]:
            rows.append(_feature_record(case_id,check,states[int(check["round_start_evaluations"])]))
    after=hashes()
    selection_fields=["candidate_unit_0","candidate_unit_1","candidate_unit_2","candidate_unit_3",
        "candidate_unit_4","normalized_domain_center_distance","normalized_nearest_bound_margin",
        "candidate_owner_distance","owner_loss","owner_encoded_loss","owner_loss_rank",
        "owner_cell_size","owner_size_rank","owner_dominator_count","surrogate_prediction",
        "surrogate_minus_owner_encoded_loss","owner_reservoir_candidate_count",
        "owner_distance_rank_desc","owner_surrogate_prediction_rank_asc",
        "current_reservoir_local_box_count","evaluated_local_box_history_count",
        "local_history_recency","owner_loss_size_rank_ratio",
        "evaluation_count_if_tested","implied_n2_box_unit_halfwidth"]
    promotion_fields=selection_fields+["promotion_candidate_loss","promotion_loss_rank",
        "promotion_loss_minus_owner_loss"]
    category_fields=["owner_potentially_optimal","owner_pareto_dominated",
        "owner_is_group_representative","owner_lower_right_hull_position",
        "production_rejection_reason","production_score_role","production_owner_winner",
        "promotion_would_be_incumbent"]
    pair_fields=["candidate_owner_distance","owner_loss_rank","owner_size_rank",
        "owner_dominator_count","surrogate_prediction","surrogate_minus_owner_encoded_loss",
        "owner_reservoir_candidate_count","owner_distance_rank_desc",
        "owner_surrogate_prediction_rank_asc","current_reservoir_local_box_count",
        "evaluated_local_box_history_count","implied_n2_box_unit_halfwidth"]
    analysis=dict(
        candidate_count=len(rows), good=sum(r["label"]=="GOOD" for r in rows),
        bad=sum(r["label"]=="BAD" for r in rows),
        selection_scalar_audit=_scalar_table(rows,selection_fields),
        promotion_scalar_audit=_scalar_table(rows,promotion_fields),
        categorical_audit=_category_table(rows,category_fields),
        nearest_opposite_label_pairs=_nearest_cross_label(rows,pair_fields),
        checkpoint_hashes_unchanged=before==after,
        exact_past_reservoir_recurrence_available_at_runtime=False,
        recurrence_note=("Production stores ordered evaluated history but no prior reservoirs. "
                         "The audit therefore retains exact local evaluated history and current-reservoir "
                         "support; retrospective past-reservoir frequency is not an N1 runtime input."),
    )
    (OUT/"candidate_states.json").write_text(json.dumps(_jsonable(rows),indent=2)+"\n")
    (OUT/"analysis.json").write_text(json.dumps(_jsonable(analysis),indent=2)+"\n")
    (OUT/"checkpoint_hashes_before.json").write_text(json.dumps(before,indent=2)+"\n")
    (OUT/"checkpoint_hashes_after.json").write_text(json.dumps(after,indent=2)+"\n")
    print(json.dumps(_jsonable({k:v for k,v in analysis.items() if k not in (
        'selection_scalar_audit','promotion_scalar_audit','categorical_audit','nearest_opposite_label_pairs')}),indent=2))
    print("separating selection scalars",[k for k,v in analysis["selection_scalar_audit"].items() if v["single_threshold_separates"]])
    print("separating promotion scalars",[k for k,v in analysis["promotion_scalar_audit"].items() if v["single_threshold_separates"]])
    print("nearest opposite pair",analysis["nearest_opposite_label_pairs"][0])


if __name__ == "__main__": main()
