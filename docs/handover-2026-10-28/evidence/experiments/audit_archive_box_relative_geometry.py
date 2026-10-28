#!/usr/bin/env python3
"""Audit explicit site-relative N2-box geometry without fitting a model."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.audit_archive_same_site_transitions import load_jsonl
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse
from experiments.train_archive_handoff_screen import ArchiveSiteSelector, ArchiveState
from experiments.validate_lossless_edirect_representation import decode, replay


DATA = Path("outputs/chronological_archive_handoff_supervision_expanded")
PRIOR = Path("outputs/archive_same_site_transition_audit")
OUT = Path("outputs/archive_box_relative_geometry_audit")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def box_for_site(state, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical = state.lower + (state.upper - state.lower) * state.points[index]
    radius = (state.upper - state.lower) / len(state.losses) ** (1.0 / state.dimension)
    lower = np.maximum(state.lower, physical - radius)
    upper = np.minimum(state.upper, physical + radius)
    return lower, upper, physical


def relative_geometry(point: np.ndarray, target: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray) -> dict:
    width = upper - lower
    coordinate = (point - lower) / width
    direction = (point - target) / width
    below = np.maximum(-coordinate, 0.0)
    above = np.maximum(coordinate - 1.0, 0.0)
    return {
        "box_coordinate": coordinate.tolist(),
        "target_relative_direction": direction.tolist(),
        "lower_face_distance": coordinate.tolist(),
        "upper_face_distance": (1.0 - coordinate).tolist(),
        "outside_lower": below.tolist(),
        "outside_upper": above.tolist(),
        "inside_box": bool(np.all((coordinate >= 0.0) & (coordinate <= 1.0))),
        "normalized_distance_to_box": float(np.linalg.norm(below + above)),
    }


def pareto_counts(sizes: np.ndarray, losses: np.ndarray, index: int) -> tuple[int, int]:
    dominators = ((sizes >= sizes[index]) & (losses <= losses[index]) &
                  ((sizes > sizes[index]) | (losses < losses[index])))
    dominated = ((sizes <= sizes[index]) & (losses >= losses[index]) &
                 ((sizes < sizes[index]) | (losses > losses[index])))
    return int(dominators.sum()), int(dominated.sum())


def representation(meta: dict, target_index: int) -> dict:
    state = decode(Path(meta["lossless_state_binary"]).read_bytes())
    result = replay(state)
    data = result["data"]
    lower, upper, target = box_for_site(state, target_index)
    physical = state.lower + (state.upper - state.lower) * state.points
    losses = state.losses
    rank_order = np.argsort(losses, kind="stable")
    ranks = np.empty(len(losses), np.int64); ranks[rank_order] = np.arange(len(losses))
    selected = {int(index): position for position, index in enumerate(data.selected)}
    pairwise = np.linalg.norm(state.points[:, None] - state.points[None, :], axis=-1)
    masked = pairwise.copy(); np.fill_diagonal(masked, np.inf)
    surface = GaussianLocalResponse(state.points, losses)

    evaluated = []
    for index, point in enumerate(physical):
        dominators, dominated = pareto_counts(data.sizes, losses, index)
        rel = relative_geometry(point, target, lower, upper)
        evaluated.append({
            "archive_index": index,
            "is_target": bool(index == target_index),
            "physical_coordinate": point.tolist(),
            **rel,
            "loss": float(losses[index]),
            "encoded_loss": float(np.log10(max(losses[index], 1e-300))),
            "loss_rank": int(ranks[index]),
            "loss_rank_fraction": float(ranks[index] / max(1, len(losses) - 1)),
            "lower_loss_than_target": bool(losses[index] < losses[target_index]),
            "nearest_site_index": (int(np.argmin(masked[index])) if len(losses) > 1 else None),
            "nearest_site_distance_global": (float(np.min(masked[index])) if len(losses) > 1 else None),
            "cell_size": float(data.sizes[index]),
            "dominator_count": dominators,
            "dominated_count": dominated,
            "potentially_optimal": bool(index in selected),
            "hull_position": selected.get(index),
            "surrogate_prediction_at_site": float(surface.predict(state.points[index])[0]),
            "insertion_age": int(len(losses) - 1 - index),
        })

    proposals = []
    if len(result["proposals"]):
        predictions = surface.predict(result["proposals"])
        proposal_physical = state.lower + (state.upper - state.lower) * result["proposals"]
        for proposal_index, point in enumerate(proposal_physical):
            owner = int(result["proposal_owners"][proposal_index])
            proposals.append({
                "proposal_index": proposal_index,
                "owner_site_index": owner,
                "owner_is_target": bool(owner == target_index),
                "physical_coordinate": point.tolist(),
                **relative_geometry(point, target, lower, upper),
                "surrogate_prediction": float(predictions[proposal_index]),
                "predicted_better_than_target": bool(
                    predictions[proposal_index] < np.log10(max(losses[target_index], 1e-300))),
                "owner_loss_rank": int(ranks[owner]),
                "owner_potentially_optimal": bool(owner in selected),
            })
    return {
        "state_id": meta["state_id"],
        "evaluation_count": len(losses),
        "target_archive_index": target_index,
        "target_theta": target.tolist(),
        "target_loss": float(losses[target_index]),
        "box_lower": lower.tolist(), "box_upper": upper.tolist(),
        "box_width": (upper - lower).tolist(),
        "target_box_coordinate": ((target - lower) / (upper - lower)).tolist(),
        "evaluated_sites": evaluated,
        "reservoir_proposals": proposals,
        "continuous_surrogate_proposal_available": False,
        "continuous_surrogate_note": (
            "Chronological histories stop in the initial-design phase; no acted-on "
            "continuous-surrogate endpoint is stored. Reconstructible E-DIRECT cell "
            "reservoir proposals are included individually."),
    }


def evaluated_feature(token: dict, coordinate_only: bool = False) -> list[float]:
    values = ([1.0, float(token["is_target"])] + token["box_coordinate"] +
              token["target_relative_direction"] + token["lower_face_distance"] +
              token["upper_face_distance"] + token["outside_lower"] +
              token["outside_upper"] + [float(token["inside_box"])])
    if not coordinate_only:
        values += [
            math.copysign(math.log1p(abs(token["loss"])), token["loss"]),
            token["encoded_loss"], token["loss_rank_fraction"],
            float(token["lower_loss_than_target"]), token["cell_size"],
            float(token["dominator_count"]), float(token["dominated_count"]),
            float(token["potentially_optimal"]),
            token["surrogate_prediction_at_site"], float(token["insertion_age"]),
        ]
    return values


def proposal_feature(token: dict, coordinate_only: bool = False) -> list[float]:
    values = ([0.0, float(token["owner_is_target"])] + token["box_coordinate"] +
              token["target_relative_direction"] + token["lower_face_distance"] +
              token["upper_face_distance"] + token["outside_lower"] +
              token["outside_upper"] + [float(token["inside_box"])])
    if not coordinate_only:
        values += [token["surrogate_prediction"],
                   float(token["predicted_better_than_target"]),
                   float(token["owner_loss_rank"]),
                   float(token["owner_potentially_optimal"])] + [0.0] * 6
    return values


def padded_vectors(rows: list[dict], key: str, coordinate_only: bool) -> np.ndarray:
    # Canonical token order is retained. Padding carries an explicit mask bit.
    token_lists = []
    maximum = 0
    for row in rows:
        rep = row[key]
        tokens = [evaluated_feature(token, coordinate_only) for token in rep["evaluated_sites"]]
        tokens += [proposal_feature(token, coordinate_only) for token in rep["reservoir_proposals"]]
        token_lists.append(tokens); maximum = max(maximum, len(tokens))
    width = len(token_lists[0][0])
    output = np.zeros((len(rows), maximum, width + 1), np.float64)
    for row_index, tokens in enumerate(token_lists):
        for token_index, token in enumerate(tokens):
            output[row_index, token_index, 0] = 1.0
            output[row_index, token_index, 1:] = token
    return output.reshape(len(rows), -1)


def whiten(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def leave_one_out_1nn(values: np.ndarray, labels: np.ndarray,
                      strata: list[tuple]) -> dict:
    values = whiten(values)
    predictions = np.empty(len(values), bool)
    distances = np.empty(len(values), np.float64)
    neighbors = np.empty(len(values), np.int64)
    for index in range(len(values)):
        candidates = [j for j in range(len(values)) if j != index and strata[j] == strata[index]]
        delta = values[candidates] - values[index]
        local = int(np.argmin(np.linalg.norm(delta, axis=1)))
        neighbor = candidates[local]
        predictions[index] = labels[neighbor]
        distances[index] = float(np.linalg.norm(values[neighbor] - values[index]))
        neighbors[index] = neighbor
    correct = predictions == labels
    return {
        "examples": len(values), "accuracy": float(correct.mean()),
        "transition_correct": int(correct[labels].sum()),
        "transition_total": int(labels.sum()),
        "control_correct": int(correct[~labels].sum()),
        "control_total": int((~labels).sum()),
        "nearest_distance_median": float(np.median(distances)),
    }


def matched_overlap(values: np.ndarray, labels: np.ndarray,
                    strata: list[tuple], pair_ids: list[str]) -> dict:
    values = whiten(values)
    opposite_closer = 0; ratios = []
    for index in range(len(values)):
        opposite = next(j for j in range(len(values))
                        if j != index and pair_ids[j] == pair_ids[index])
        same = [j for j in range(len(values)) if j != index and
                labels[j] == labels[index] and strata[j] == strata[index]]
        if not same:
            continue
        opposite_distance = float(np.linalg.norm(values[index] - values[opposite]))
        same_distance = min(float(np.linalg.norm(values[index] - values[j])) for j in same)
        opposite_closer += int(opposite_distance < same_distance)
        ratios.append(opposite_distance / max(same_distance, 1e-300))
    return {
        "examples_with_same_class_comparator": len(ratios),
        "matched_opposite_closer_than_nearest_same_class": opposite_closer,
        "fraction": opposite_closer / max(1, len(ratios)),
        "matched_to_same_class_distance_ratio_median": float(np.median(ratios)),
    }


def frozen_representations(states: dict[str, ArchiveState]) -> dict[str, np.ndarray]:
    checkpoint = torch.load(
        "outputs/archive_handoff_relational_screen/archive_site_selector_screen_only.pt",
        map_location="cpu", weights_only=False)
    model = ArchiveSiteSelector(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    output = {}
    with torch.no_grad():
        for state_id, state in states.items():
            sites = model.encoder(state)
            archive_query = model.attention(model.query, sites)
            output[state_id] = model.norm(sites + model.context(archive_query)).numpy()
    return output


def probability_superiority(first, second) -> float:
    a = np.asarray(first, np.float64); b = np.asarray(second, np.float64)
    wins = sum(float(x > y) + .5 * float(x == y) for x in a for y in b)
    p = wins / (len(a) * len(b))
    return float(max(p, 1.0 - p))


def scalar_diagnostics(rows: list[dict], labels: np.ndarray) -> dict:
    names = {
        "new_site_loss_over_target": lambda r: (
            r["after"]["evaluated_sites"][-1]["loss"] / r["after"]["target_loss"]),
        "new_site_minus_target_encoded_loss": lambda r: (
            r["after"]["evaluated_sites"][-1]["encoded_loss"] -
            math.log10(max(r["after"]["target_loss"], 1e-300))),
        "target_cell_size": lambda r: r["after"]["evaluated_sites"][r["archive_index"]]["cell_size"],
        "target_surrogate_prediction": lambda r: r["after"]["evaluated_sites"][r["archive_index"]]["surrogate_prediction_at_site"],
        "target_dominator_count": lambda r: r["after"]["evaluated_sites"][r["archive_index"]]["dominator_count"],
    }
    result = {}
    for name, function in names.items():
        values = np.asarray([function(row) for row in rows], np.float64)
        result[name] = {
            "transition_median": float(np.median(values[labels])),
            "control_median": float(np.median(values[~labels])),
            "orientation_free_pairwise_auc": probability_superiority(values[labels], values[~labels]),
            "transition_range": [float(values[labels].min()), float(values[labels].max())],
            "control_range": [float(values[~labels].min()), float(values[~labels].max())],
        }
    return result


def evaluated_coordinate_signature(rep: dict) -> np.ndarray:
    values = [rep["box_lower"], rep["box_upper"], rep["box_width"],
              rep["target_box_coordinate"]]
    for token in rep["evaluated_sites"]:
        values.extend((token["box_coordinate"], token["target_relative_direction"],
                       token["lower_face_distance"], token["upper_face_distance"]))
    return np.concatenate([np.asarray(value, np.float64) for value in values])


def spatial_patterns(rows: list[dict], labels: np.ndarray) -> dict:
    output = {}
    for name, mask in (("bad_to_good", labels), ("bad_to_bad", ~labels)):
        selected = [row for row, use in zip(rows, mask) if use]
        sign_patterns = Counter()
        inside_hist = Counter(); lower_inside_hist = Counter()
        proposal_inside = Counter(); proposal_better = Counter()
        best_proposal_rho_sign = Counter()
        for row in selected:
            after = row["after"]; target_loss = after["target_loss"]
            newest = after["evaluated_sites"][-1]
            sign_patterns[str(tuple(int(np.sign(x)) for x in
                                    newest["target_relative_direction"]))] += 1
            inside_hist[sum(token["inside_box"] for token in after["evaluated_sites"])] += 1
            lower_inside_hist[sum(token["inside_box"] and token["loss"] < target_loss
                                  for token in after["evaluated_sites"])] += 1
            proposals = after["reservoir_proposals"]
            proposal_inside[sum(token["inside_box"] for token in proposals)] += 1
            proposal_better[sum(token["predicted_better_than_target"] for token in proposals)] += 1
            if proposals:
                best = min(proposals, key=lambda token: token["surrogate_prediction"])
                best_proposal_rho_sign[str(int(np.sign(
                    best["target_relative_direction"][-1])))] += 1
        output[name] = {
            "examples": len(selected),
            "new_site_direction_sign_patterns": dict(sign_patterns),
            "evaluated_sites_inside_box_histogram": {str(k): v for k, v in sorted(inside_hist.items())},
            "lower_loss_sites_inside_box_histogram": {str(k): v for k, v in sorted(lower_inside_hist.items())},
            "reservoir_proposals_inside_box_histogram": {str(k): v for k, v in sorted(proposal_inside.items())},
            "reservoir_proposals_predicted_better_histogram": {str(k): v for k, v in sorted(proposal_better.items())},
            "best_surrogate_proposal_rho_direction_sign": dict(best_proposal_rho_sign),
        }
    return output


def main() -> None:
    state_rows = load_jsonl(DATA / "expanded_archive_states.jsonl")
    site_rows = load_jsonl(DATA / "expanded_archive_site_labels.jsonl")
    state_meta = {row["state_id"]: row for row in state_rows}
    site_by_state = defaultdict(list)
    for row in site_rows: site_by_state[row["state_id"]].append(row)
    archive_states = {key: ArchiveState(meta, site_by_state[key])
                      for key, meta in state_meta.items()}
    frozen = frozen_representations(archive_states)

    matched = load_jsonl(PRIOR / "matched_pairs.jsonl")
    all_transitions = load_jsonl(PRIOR / "bad_to_good_transitions.jsonl")
    all_controls = load_jsonl(PRIOR / "bad_to_bad_controls.jsonl")
    cache = {}
    def rep(source: dict, which: str) -> dict:
        state_id = source[f"state_id_{which}"]
        key = (state_id, source["archive_index"])
        if key not in cache:
            cache[key] = representation(state_meta[state_id], source["archive_index"])
        return cache[key]
    def make_row(label: bool, source: dict, pair_id: str) -> dict:
        before = rep(source, "before"); after = rep(source, "after")
        current = frozen[source["state_id_after"]][source["archive_index"]]
        prior = frozen[source["state_id_before"]][source["archive_index"]]
        return {
            "label": "BAD_TO_GOOD" if label else "BAD_TO_BAD",
            "matched_pair_id": pair_id,
            "system": source["system"], "case_id": source["case_id"],
            "split": source["split"], "archive_index": source["archive_index"],
            "from_count": source["from_count"], "to_count": source["to_count"],
            "before": before, "after": after,
            "frozen_target_embedding_after": current.tolist(),
            "frozen_target_embedding_delta": (current - prior).tolist(),
        }

    rows = []
    for pair in matched:
        for label, source in ((True, pair["transition"]), (False, pair["control"])):
            rows.append(make_row(label, source, pair["transition_id"]))

    all_rows = ([make_row(True, source, f"transition:{index}")
                 for index, source in enumerate(all_transitions)] +
                [make_row(False, source, f"control:{index}")
                 for index, source in enumerate(all_controls)])

    labels = np.asarray([row["label"] == "BAD_TO_GOOD" for row in rows], bool)
    strata = [(row["from_count"], row["to_count"], row["archive_index"]) for row in rows]
    coordinate = padded_vectors(rows, "after", True)
    complete = padded_vectors(rows, "after", False)
    frozen_after = np.asarray([row["frozen_target_embedding_after"] for row in rows])
    frozen_delta = np.asarray([row["frozen_target_embedding_delta"] for row in rows])
    augmented = np.concatenate((frozen_after, complete), axis=1)

    exact_evaluated_coordinate_matches = 0
    for pair_index in range(len(matched)):
        transition = rows[2 * pair_index]
        control = rows[2 * pair_index + 1]
        exact_evaluated_coordinate_matches += int(np.array_equal(
            evaluated_coordinate_signature(transition["after"]),
            evaluated_coordinate_signature(control["after"])))
    pair_ids = [row["matched_pair_id"] for row in rows]

    report = {
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING",
        "examples": {"bad_to_good": int(labels.sum()),
                     "matched_bad_to_bad": int((~labels).sum())},
        "all_bad_to_bad_contractions": 1134,
        "representation": {
            "individual_evaluated_sites_preserved": True,
            "individual_reservoir_proposals_preserved": True,
            "canonical_token_order": "evaluated insertion order followed by deterministic reservoir proposal order",
            "global_pooling": "none",
            "continuous_surrogate_endpoint": "not present because all chronological states stop in initial-design phase",
        },
        "matched_pairs_with_bit_identical_evaluated_site_box_geometry": exact_evaluated_coordinate_matches,
        "nonparametric_separability": {
            "frozen_selector_target_embedding_after": leave_one_out_1nn(frozen_after, labels, strata),
            "frozen_selector_embedding_change": leave_one_out_1nn(frozen_delta, labels, strata),
            "box_relative_coordinates_only": leave_one_out_1nn(coordinate, labels, strata),
            "complete_box_relative_tokens": leave_one_out_1nn(complete, labels, strata),
            "frozen_embedding_plus_complete_box_tokens": leave_one_out_1nn(augmented, labels, strata),
            "method": "leave-one-out nearest exemplar within exact count/index stratum; descriptive only, no fitted parameters",
        },
        "matched_opposite_overlap": {
            "frozen_selector_target_embedding_after": matched_overlap(frozen_after, labels, strata, pair_ids),
            "frozen_selector_embedding_change": matched_overlap(frozen_delta, labels, strata, pair_ids),
            "box_relative_coordinates_only": matched_overlap(coordinate, labels, strata, pair_ids),
            "complete_box_relative_tokens": matched_overlap(complete, labels, strata, pair_ids),
            "frozen_embedding_plus_complete_box_tokens": matched_overlap(augmented, labels, strata, pair_ids),
        },
        "scalar_diagnostics": scalar_diagnostics(rows, labels),
        "all_control_scalar_diagnostics": scalar_diagnostics(
            all_rows, np.asarray([row["label"] == "BAD_TO_GOOD" for row in all_rows], bool)),
        "matched_spatial_patterns": spatial_patterns(rows, labels),
        "source_hashes": {
            str(DATA / "expanded_archive_states.jsonl"): sha(DATA / "expanded_archive_states.jsonl"),
            str(DATA / "expanded_archive_site_labels.jsonl"): sha(DATA / "expanded_archive_site_labels.jsonl"),
            str(PRIOR / "matched_pairs.jsonl"): sha(PRIOR / "matched_pairs.jsonl"),
            "selector_checkpoint": sha(Path("outputs/archive_handoff_relational_screen/archive_site_selector_screen_only.pt")),
        },
        "causal_interpretation": (
            "The explicit box is the only changed input consumed by N2/N3. "
            "Archive tokens can predict but cannot cause the transition. Box-relative "
            "sampled geometry is evaluated here for predictive separability."),
    }
    atomic_jsonl(OUT / "matched_box_relative_tokens.jsonl", rows)
    atomic_jsonl(OUT / "all_contractions_box_relative_tokens.jsonl", all_rows)
    atomic_json(OUT / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
