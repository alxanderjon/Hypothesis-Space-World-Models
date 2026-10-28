#!/usr/bin/env python3
"""Audit fixed-site BAD->GOOD transitions in chronological archive supervision.

This is a read-only diagnostic.  It reconstructs deterministic E-DIRECT geometry
from the lossless archive state, compares consecutive labels for the same site,
and matches every BAD->GOOD transition to a same-count BAD->BAD control.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse
from experiments.train_archive_handoff_screen import ArchiveSiteSelector, ArchiveState
from experiments.validate_lossless_edirect_representation import decode, replay


DATA = Path("outputs/chronological_archive_handoff_supervision_expanded")
OUT = Path("outputs/archive_same_site_transition_audit")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def finite(value: float | np.floating) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def distribution(values) -> dict:
    values = np.asarray([value for value in values if value is not None], np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9, method="higher")),
        "maximum": float(values.max()),
    }


def state_geometry(meta: dict) -> dict:
    state = decode(Path(meta["lossless_state_binary"]).read_bytes())
    reconstructed = replay(state)
    data = reconstructed["data"]
    points = state.points
    losses = state.losses
    n = len(losses)
    tree_distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    masked = tree_distance.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = (np.argmin(masked, axis=1) if n > 1
               else np.full(1, -1, np.int64))
    nearest_distance = (masked[np.arange(n), nearest] if n > 1
                        else np.full(1, np.inf))
    rank_order = np.argsort(losses, kind="stable")
    rank = np.empty(n, np.int64)
    rank[rank_order] = np.arange(n)
    incumbent = int(rank_order[0])
    surface = GaussianLocalResponse(points, losses)
    prediction = surface.predict(points)
    loo = np.empty(n, np.float64)
    for index in range(n):
        loo[index] = surface.loo_errors(np.asarray([index], np.int64))[0]
    selected_position = {int(index): position
                         for position, index in enumerate(data.selected)}
    site_rows = []
    for index in range(n):
        owned = data.points[data.owners == index]
        if len(owned):
            extent = owned.max(axis=0) - owned.min(axis=0)
            aabb_volume = float(np.prod(extent))
        else:
            extent = np.zeros(state.dimension, np.float64)
            aabb_volume = 0.0
        dominates = ((data.sizes >= data.sizes[index]) &
                     (losses <= losses[index]) &
                     ((data.sizes > data.sizes[index]) |
                      (losses < losses[index])))
        dominated = ((data.sizes <= data.sizes[index]) &
                     (losses >= losses[index]) &
                     ((data.sizes < data.sizes[index]) |
                      (losses > losses[index])))
        site_rows.append({
            "archive_index": index,
            "loss": float(losses[index]),
            "loss_rank": int(rank[index]),
            "loss_rank_fraction": float(rank[index] / max(1, n - 1)),
            "is_incumbent": bool(index == incumbent),
            "nearest_index": int(nearest[index]),
            "nearest_distance": finite(nearest_distance[index]),
            "cell_size": float(data.sizes[index]),
            "owned_probe_count": int(np.sum(data.owners == index)),
            "owned_probe_fraction": float(np.mean(data.owners == index)),
            "cell_aabb_extent": extent.tolist(),
            "cell_aabb_volume": aabb_volume,
            "dominator_count": int(dominates.sum()),
            "dominated_count": int(dominated.sum()),
            "potentially_optimal": bool(index in selected_position),
            "hull_selected_position": selected_position.get(index),
            "surrogate_prediction": float(prediction[index]),
            "loo_error": float(loo[index]),
            "history_age": int(n - 1 - index),
            "previous_index": int(index - 1) if index > 0 else None,
            "next_index": int(index + 1) if index + 1 < n else None,
            "is_newest": bool(index == n - 1),
        })
    return {
        "evaluation_count": n,
        "iteration": int(state.iteration),
        "incumbent_index": incumbent,
        "incumbent_loss": float(losses[incumbent]),
        "sites": site_rows,
    }


def site_delta(before: dict, after: dict) -> dict:
    numeric = (
        "loss_rank", "loss_rank_fraction", "nearest_distance", "cell_size",
        "owned_probe_count", "owned_probe_fraction", "cell_aabb_volume",
        "dominator_count", "dominated_count", "surrogate_prediction",
        "loo_error", "history_age",
    )
    result = {}
    for key in numeric:
        left, right = before[key], after[key]
        result[key] = None if left is None or right is None else float(right - left)
    for key in ("is_incumbent", "nearest_index", "potentially_optimal",
                "hull_selected_position", "next_index", "is_newest"):
        result[key + "_changed"] = before[key] != after[key]
    return result


def box(row: dict) -> dict:
    lower = np.asarray(row["n2_box_lower"], np.float64)
    upper = np.asarray(row["n2_box_upper"], np.float64)
    theta = np.asarray(row["theta"], np.float64)
    domain_lower = np.asarray([-1.99] * len(theta) if row["system"] == "reflection_ball"
                              else [-.25, -1., -2., -.5, 1.5], np.float64)
    domain_upper = np.asarray([1.99] * len(theta) if row["system"] == "reflection_ball"
                              else [.5, 2., 3., 2.5, 4.5], np.float64)
    nominal = (domain_upper - domain_lower) / row["current_evaluation_count"] ** (1 / len(theta))
    widths = upper - lower
    return {
        "lower": lower.tolist(), "upper": upper.tolist(),
        "effective_half_width": (0.5 * widths).tolist(),
        "nominal_radius": nominal.tolist(),
        "volume": float(np.prod(widths)),
        "clipped_coordinates": int(np.sum((lower == domain_lower) | (upper == domain_upper))),
    }


def downstream(row: dict) -> dict:
    return {
        "n2_endpoint": row["n2_endpoint"],
        "n2_loss": float(row["n2_loss"]),
        "n2_evaluations": int(row["n2_evaluations"]),
        "n2_stop_reason": row["n2_stop_reason"],
        "n3_accepted_steps": int(row["n3_accepted_steps"]),
        "n3_stop_reason": row["n3_stop_reason"],
        "final_theta": row["final_theta"],
        "final_loss": float(row["final_loss"]),
        "final_parameter_l2_error": finite(row["final_parameter_l2_error"]),
        "continuation_max_error": (finite(row["continuation_max_error"])
                                   if row["continuation_max_error"] is not None else None),
        "continuation_mse": row["continuation_mse"],
        "collision_topology_regressions": int(row["collision_topology_regressions"]),
        "recovered": bool(row["target_good"]),
    }


def comparison(kind: str, key: tuple, before: dict, after: dict,
               states: dict, geometry: dict) -> dict:
    system, case_id, archive_index = key
    before_meta = states[(system, case_id, before["current_evaluation_count"])]
    after_meta = states[(system, case_id, after["current_evaluation_count"])]
    gb = geometry[before_meta["state_id"]]
    ga = geometry[after_meta["state_id"]]
    sb = gb["sites"][archive_index]
    sa = ga["sites"][archive_index]
    before_box, after_box = box(before), box(after)
    new_indices = list(range(before["current_evaluation_count"],
                             after["current_evaluation_count"]))
    new_sites = [{
        "archive_index": index,
        "theta": after["theta"] if index == archive_index else
                 np.load(after_meta["state_npz"])["physical_sites"][index].tolist(),
        "loss": float(np.load(after_meta["state_npz"])["site_losses"][index]),
    } for index in new_indices]
    db, da = downstream(before), downstream(after)
    return {
        "kind": kind,
        "system": system,
        "case_id": int(case_id),
        "split": before["split"],
        "archive_index": int(archive_index),
        "insertion_evaluation": int(before["insertion_evaluation"]),
        "theta": before["theta"],
        "archived_loss": float(before["archived_loss"]),
        "state_id_before": before_meta["state_id"],
        "state_id_after": after_meta["state_id"],
        "from_count": int(before["current_evaluation_count"]),
        "to_count": int(after["current_evaluation_count"]),
        "box_before": before_box,
        "box_after": after_box,
        "box_nominal_radius_ratio": (np.asarray(after_box["nominal_radius"]) /
                                     np.asarray(before_box["nominal_radius"])).tolist(),
        "box_volume_ratio": float(after_box["volume"] / before_box["volume"]),
        "new_sites": new_sites,
        "incumbent_before": {"index": gb["incumbent_index"], "loss": gb["incumbent_loss"]},
        "incumbent_after": {"index": ga["incumbent_index"], "loss": ga["incumbent_loss"]},
        "site_before": sb,
        "site_after": sa,
        "site_delta": site_delta(sb, sa),
        "n2_endpoint_displacement": float(np.linalg.norm(
            np.asarray(da["n2_endpoint"]) - np.asarray(db["n2_endpoint"]))),
        "downstream_before": db,
        "downstream_after": da,
    }


def summarize(rows: list[dict]) -> dict:
    old_outside = []
    for row in rows:
        lower = np.asarray(row["box_after"]["lower"])
        upper = np.asarray(row["box_after"]["upper"])
        old_endpoint = np.asarray(row["downstream_before"]["n2_endpoint"])
        old_outside.append(not bool(np.all((old_endpoint >= lower) &
                                           (old_endpoint <= upper))))
    return {
        "pairs": len(rows),
        "count_pairs": dict(Counter(f"{r['from_count']}->{r['to_count']}" for r in rows)),
        "incumbent_changed": sum(r["incumbent_before"]["index"] != r["incumbent_after"]["index"] for r in rows),
        "site_loss_rank_changed": sum(r["site_delta"]["loss_rank"] != 0 for r in rows),
        "nearest_identity_changed": sum(r["site_delta"]["nearest_index_changed"] for r in rows),
        "potentially_optimal_changed": sum(r["site_delta"]["potentially_optimal_changed"] for r in rows),
        "next_history_relation_changed": sum(r["site_delta"]["next_index_changed"] for r in rows),
        "n2_loss_improved": sum(r["downstream_after"]["n2_loss"] < r["downstream_before"]["n2_loss"] for r in rows),
        "n2_loss_worsened": sum(r["downstream_after"]["n2_loss"] > r["downstream_before"]["n2_loss"] for r in rows),
        "old_n2_endpoint_outside_later_box": sum(old_outside),
        "old_n2_last_parameter_positive": sum(r["downstream_before"]["n2_endpoint"][-1] > 0 for r in rows),
        "later_n2_last_parameter_negative": sum(r["downstream_after"]["n2_endpoint"][-1] < 0 for r in rows),
        "later_n2_last_parameter_within_0_25_of_minus_one": sum(
            abs(r["downstream_after"]["n2_endpoint"][-1] + 1.0) < .25 for r in rows),
        "n3_stop_before": dict(Counter(r["downstream_before"]["n3_stop_reason"] for r in rows)),
        "n3_stop_after": dict(Counter(r["downstream_after"]["n3_stop_reason"] for r in rows)),
        "model_fixed_site_score_increased": sum(r["model_selector"]["fixed_site_score_delta"] > 0 for r in rows),
        "model_fixed_site_rank_improved": sum(r["model_selector"]["fixed_site_rank_after"] < r["model_selector"]["fixed_site_rank_before"] for r in rows),
        "model_fixed_site_top_after": sum(r["model_selector"]["fixed_site_rank_after"] == 1 for r in rows),
        "box_volume_ratio": distribution([r["box_volume_ratio"] for r in rows]),
        "n2_endpoint_displacement": distribution([r["n2_endpoint_displacement"] for r in rows]),
        "absolute_cell_size_change": distribution([abs(r["site_delta"]["cell_size"]) for r in rows]),
        "absolute_surrogate_prediction_change": distribution([
            abs(r["site_delta"]["surrogate_prediction"]) for r in rows]),
        "absolute_nearest_distance_change": distribution([
            abs(r["site_delta"]["nearest_distance"] or 0.0) for r in rows]),
    }


def feature_vector(row: dict) -> np.ndarray:
    site = row["site_before"]
    return np.asarray([
        math.log1p(row["archived_loss"]),
        site["loss_rank_fraction"],
        site["nearest_distance"] if site["nearest_distance"] is not None else 2.0,
        site["cell_size"], site["owned_probe_fraction"],
        site["dominator_count"], float(site["potentially_optimal"]),
        site["surrogate_prediction"], site["loo_error"],
        float(site["is_incumbent"]),
    ], np.float64)


def main() -> None:
    state_path = DATA / "expanded_archive_states.jsonl"
    label_path = DATA / "expanded_archive_site_labels.jsonl"
    state_rows = load_jsonl(state_path)
    label_rows = load_jsonl(label_path)
    states = {(r["system"], int(r["case_id"]), int(r["evaluation_count"])): r
              for r in state_rows}
    geometry = {meta["state_id"]: state_geometry(meta) for meta in state_rows}

    site_rows_by_state = defaultdict(list)
    for row in label_rows:
        site_rows_by_state[row["state_id"]].append(row)
    archive_states = {meta["state_id"]: ArchiveState(meta, site_rows_by_state[meta["state_id"]])
                      for meta in state_rows}
    checkpoint = torch.load(
        "outputs/archive_handoff_relational_screen/archive_site_selector_screen_only.pt",
        map_location="cpu", weights_only=False)
    selector = ArchiveSiteSelector()
    selector.load_state_dict(checkpoint["model_state"])
    selector.eval()
    with torch.no_grad():
        model_scores = {state_id: np.asarray(selector(state).tolist(), np.float64)
                        for state_id, state in archive_states.items()}

    by_site = defaultdict(list)
    for row in label_rows:
        by_site[(row["system"], int(row["case_id"]), int(row["archive_index"]))].append(row)
    transitions, controls = [], []
    for key, rows in by_site.items():
        rows.sort(key=lambda row: int(row["current_evaluation_count"]))
        for before, after in zip(rows, rows[1:]):
            if not before["target_good"] and after["target_good"]:
                transitions.append(comparison("BAD_TO_GOOD", key, before, after,
                                              states, geometry))
            elif not before["target_good"] and not after["target_good"]:
                controls.append(comparison("BAD_TO_BAD", key, before, after,
                                           states, geometry))

    for row in transitions + controls:
        before_scores = model_scores[row["state_id_before"]]
        after_scores = model_scores[row["state_id_after"]]
        index = row["archive_index"]
        before_order = np.lexsort((np.arange(len(before_scores)), -before_scores))
        after_order = np.lexsort((np.arange(len(after_scores)), -after_scores))
        row["model_selector"] = {
            "fixed_site_score_before": float(before_scores[index]),
            "fixed_site_score_after": float(after_scores[index]),
            "fixed_site_score_delta": float(after_scores[index] - before_scores[index]),
            "fixed_site_rank_before": int(np.flatnonzero(before_order == index)[0] + 1),
            "fixed_site_rank_after": int(np.flatnonzero(after_order == index)[0] + 1),
            "top_site_before": int(before_order[0]),
            "top_site_after": int(after_order[0]),
        }

    # Match on exact count transition and archived-site insertion index where
    # possible, then on standardized pre-transition archive features.
    all_features = np.asarray([feature_vector(row) for row in transitions + controls])
    scale = np.std(all_features, axis=0)
    scale[scale == 0] = 1.0
    matched = []
    used: set[int] = set()
    for transition in transitions:
        candidates = [i for i, row in enumerate(controls)
                      if row["from_count"] == transition["from_count"]
                      and row["to_count"] == transition["to_count"]
                      and row["archive_index"] == transition["archive_index"]
                      and (row["case_id"] != transition["case_id"])]
        if not candidates:
            candidates = [i for i, row in enumerate(controls)
                          if row["from_count"] == transition["from_count"]
                          and row["to_count"] == transition["to_count"]
                          and row["case_id"] != transition["case_id"]]
        available = [i for i in candidates if i not in used]
        if available:
            candidates = available
        target = feature_vector(transition)
        chosen = min(candidates, key=lambda i: float(np.linalg.norm(
            (feature_vector(controls[i]) - target) / scale)))
        used.add(chosen)
        matched.append({
            "transition_id": f"{transition['system']}:{transition['case_id']}:{transition['archive_index']}:{transition['from_count']}->{transition['to_count']}",
            "control_id": f"{controls[chosen]['system']}:{controls[chosen]['case_id']}:{controls[chosen]['archive_index']}:{controls[chosen]['from_count']}->{controls[chosen]['to_count']}",
            "standardized_prestate_distance": float(np.linalg.norm(
                (feature_vector(controls[chosen]) - target) / scale)),
            "transition": transition,
            "control": controls[chosen],
        })

    transition_summary = summarize(transitions)
    control_summary = summarize(controls)
    matched_control_summary = summarize([row["control"] for row in matched])
    by_count = {}
    for count_pair in sorted(set((r["from_count"], r["to_count"]) for r in transitions)):
        key = f"{count_pair[0]}->{count_pair[1]}"
        by_count[key] = {
            "bad_to_good": summarize([r for r in transitions if
                                      (r["from_count"], r["to_count"]) == count_pair]),
            "bad_to_bad": summarize([r for r in controls if
                                     (r["from_count"], r["to_count"]) == count_pair]),
        }

    report = {
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING",
        "source_hashes": {str(state_path): sha(state_path), str(label_path): sha(label_path)},
        "selector_checkpoint_sha256": sha(Path(
            "outputs/archive_handoff_relational_screen/archive_site_selector_screen_only.pt")),
        "dataset": {"states": len(state_rows), "site_labels": len(label_rows),
                    "trajectories": len(set((r["system"], r["case_id"]) for r in state_rows))},
        "same_site_bad_to_good": transition_summary,
        "all_same_site_bad_to_bad_controls": control_summary,
        "matched_bad_to_bad_controls": matched_control_summary,
        "by_count_transition": by_count,
        "transition_profile": {
            "systems": dict(Counter(r["system"] for r in transitions)),
            "splits": dict(Counter(r["split"] for r in transitions)),
            "archive_indices": {str(k): v for k, v in sorted(Counter(
                r["archive_index"] for r in transitions).items())},
            "new_site_has_lower_loss_than_fixed_site": sum(
                r["new_sites"][0]["loss"] < r["archived_loss"] for r in transitions),
            "new_site_becomes_incumbent": sum(
                r["incumbent_after"]["index"] == r["new_sites"][0]["archive_index"]
                for r in transitions),
            "model_top_is_good_at_later_state": sum(
                r["model_selector"]["top_site_after"] in
                states[(r["system"], r["case_id"], r["to_count"])]["good_site_indices"]
                for r in transitions),
            "fixed_site_is_only_good_site": sum(
                states[(r["system"], r["case_id"], r["to_count"])]["good_site_indices"] ==
                [r["archive_index"]] for r in transitions),
            "model_selects_fixed_site_when_it_is_only_good": sum(
                states[(r["system"], r["case_id"], r["to_count"])]["good_site_indices"] ==
                [r["archive_index"]] and
                r["model_selector"]["top_site_after"] == r["archive_index"]
                for r in transitions),
        },
        "causal_path": {
            "fixed_inputs": ["archived theta", "archived objective loss", "N2 algorithm", "N3 algorithm"],
            "changed_downstream_input": "production N2 bounds derived from current evaluation count",
            "archive_geometry_used_by_n2_or_n3": False,
            "new_sites_used_by_n2_or_n3": False,
            "interpretation": "For a fixed site, BAD->GOOD is caused by the changed count-derived N2 box. New archive sites and geometry are correlated runtime evidence only.",
        },
        "transition_ids": [f"{r['system']}:{r['case_id']}:{r['archive_index']}:{r['from_count']}->{r['to_count']}" for r in transitions],
        "artifacts": {"transitions": str(OUT / "bad_to_good_transitions.jsonl"),
                      "controls": str(OUT / "bad_to_bad_controls.jsonl"),
                      "matched": str(OUT / "matched_pairs.jsonl")},
    }
    atomic_jsonl(OUT / "bad_to_good_transitions.jsonl", transitions)
    atomic_jsonl(OUT / "bad_to_bad_controls.jsonl", controls)
    atomic_jsonl(OUT / "matched_pairs.jsonl", matched)
    atomic_json(OUT / "report.json", report)
    print(json.dumps({
        "transitions": transition_summary,
        "controls": control_summary,
        "matched_controls": matched_control_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
