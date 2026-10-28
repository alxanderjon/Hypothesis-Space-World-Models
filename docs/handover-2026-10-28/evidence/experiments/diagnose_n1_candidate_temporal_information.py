#!/usr/bin/env python3
"""Read-only temporal diagnosis for persistent-Kepler candidate ranking.

Only ordered evaluated-site history stored in each lossless N1 snapshot is
used.  No model is trained and no production component is called or changed.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import (
    DATA, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_relational_architecture_screen import atomic_json


OUT = Path("outputs/n1_candidate_temporal_diagnosis")
CHECKPOINT = Path("outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
FRACTIONS = np.asarray([.125, .25, .375, .5, .625, .75, .875, 1.0])


def native(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_nearest(points, query, count=24):
    distances = np.linalg.norm(points - query[None], axis=1)
    ids = np.arange(len(points))
    order = np.lexsort((ids, distances))[:min(count, len(points))]
    return distances[order], order


def gaussian_prediction(distances, indices, encoded):
    bandwidth = max(float(distances[-1]), 1e-12)
    weights = np.exp(-9.0 * np.square(distances / bandwidth)) + 1e-15
    return float(np.sum(weights * encoded[indices]) / np.sum(weights))


def temporal_features(z, candidate_index):
    sites = np.asarray(z["sites"], np.float64)
    losses = np.asarray(z["site_losses"], np.float64)
    encoded = np.asarray(z["site_encoded_losses"], np.float64)
    query = np.asarray(z["reservoir"][candidate_index], np.float64)
    n = len(sites)
    cuts = np.unique(np.maximum(24, np.ceil(FRACTIONS * n).astype(int)))
    trajectory = []
    previous_owner = None; owner_switches = 0
    for cut in cuts:
        cut = min(int(cut), n)
        distances, indices = ordered_nearest(sites[:cut], query)
        owner = int(indices[0])
        if previous_owner is not None and owner != previous_owner:
            owner_switches += 1
        previous_owner = owner
        incumbent = int(np.argmin(losses[:cut]))
        trajectory.append(dict(
            evaluation_count=cut, fraction=float(cut / n), owner_site=owner,
            nearest_distance=float(distances[0]), radius24=float(distances[-1]),
            surrogate_prediction=gaussian_prediction(distances, indices, encoded),
            local_best_encoded_loss=float(encoded[indices].min()),
            local_median_encoded_loss=float(np.median(encoded[indices])),
            incumbent_distance=float(np.linalg.norm(sites[incumbent] - query)),
            incumbent_loss=float(losses[incumbent]),
        ))

    current_neighbors = np.asarray(z["candidate_neighbor_indices"][candidate_index], np.int64)
    current_distances = np.asarray(z["candidate_neighbor_distances"][candidate_index], np.float64)
    chronology = np.argsort(current_neighbors, kind="stable")
    chronological_sites = current_neighbors[chronology]
    chronological_values = encoded[chronological_sites]
    half = max(1, len(chronological_sites) // 2)
    correlation = spearmanr(chronological_sites, chronological_values).statistic
    if not np.isfinite(correlation): correlation = 0.0
    running = math.inf; records = 0
    for value in chronological_values:
        if value < running:
            running = float(value); records += 1
    best_position = int(chronological_sites[int(np.argmin(chronological_values))])
    owner = int(z["candidate_owner"][candidate_index])
    first, last = trajectory[0], trajectory[-1]
    return dict(
        owner_site=owner,
        owner_birth_fraction=float((owner + 1) / n),
        owner_age_fraction=float((n - owner - 1) / n),
        owner_switches_at_eighths=owner_switches,
        local_first_fraction=float((chronological_sites[0] + 1) / n),
        local_last_fraction=float((chronological_sites[-1] + 1) / n),
        local_recency_fraction=float((n - chronological_sites[-1] - 1) / n),
        local_time_span_fraction=float((chronological_sites[-1] - chronological_sites[0]) / n),
        local_neighbors_after_owner=int(np.sum(current_neighbors > owner)),
        local_neighbors_last_quarter=int(np.sum(current_neighbors >= int(.75 * n))),
        local_loss_time_spearman=float(correlation),
        local_late_minus_early_encoded_loss=float(
            np.median(chronological_values[-half:]) - np.median(chronological_values[:half])),
        local_record_improvements=records,
        local_best_recency_fraction=float((n - best_position - 1) / n),
        nearest_distance_initial=float(first["nearest_distance"]),
        nearest_distance_final=float(last["nearest_distance"]),
        nearest_distance_contraction=float(last["nearest_distance"] / max(first["nearest_distance"], 1e-300)),
        radius24_initial=float(first["radius24"]),
        radius24_final=float(last["radius24"]),
        radius24_contraction=float(last["radius24"] / max(first["radius24"], 1e-300)),
        surrogate_prediction_initial=float(first["surrogate_prediction"]),
        surrogate_prediction_final=float(last["surrogate_prediction"]),
        surrogate_prediction_change=float(last["surrogate_prediction"] - first["surrogate_prediction"]),
        surrogate_prediction_path_range=float(np.ptp([r["surrogate_prediction"] for r in trajectory])),
        local_best_encoded_loss_change=float(last["local_best_encoded_loss"] - first["local_best_encoded_loss"]),
        incumbent_distance_initial=float(first["incumbent_distance"]),
        incumbent_distance_final=float(last["incumbent_distance"]),
        incumbent_distance_change=float(last["incumbent_distance"] - first["incumbent_distance"]),
        current_neighbor_distance_min=float(current_distances[0]),
        current_neighbor_distance_max=float(current_distances[-1]),
        trajectory=trajectory,
    )


@torch.no_grad()
def model_scores(model, states, grouped):
    model.eval(); result = {}
    for state_id, examples in grouped.items():
        result[state_id] = np.asarray(model(states[state_id], examples).cpu(), np.float64)
    return result


TEMPORAL_FIELDS = [
    "owner_birth_fraction", "owner_age_fraction", "owner_switches_at_eighths",
    "local_first_fraction", "local_last_fraction", "local_recency_fraction",
    "local_time_span_fraction", "local_neighbors_after_owner", "local_neighbors_last_quarter",
    "local_loss_time_spearman", "local_late_minus_early_encoded_loss",
    "local_record_improvements", "local_best_recency_fraction",
    "nearest_distance_contraction", "radius24_contraction",
    "surrogate_prediction_change", "surrogate_prediction_path_range",
    "local_best_encoded_loss_change", "incumbent_distance_change",
]


def auc(values, labels):
    ranks = rankdata(values, method="average")
    labels = np.asarray(labels, bool); ng = int(labels.sum()); nb = len(labels) - ng
    raw = (float(ranks[labels].sum()) - ng * (ng + 1) / 2) / max(1, ng * nb)
    return raw, max(raw, 1.0 - raw)


def summarize_pairs(pairs):
    output = {}
    candidate_rows = [(p[label], label == "good") for p in pairs for label in ("good", "bad")]
    for field in TEMPORAL_FIELDS:
        deltas = np.asarray([p["good"][field] - p["bad"][field] for p in pairs], float)
        solved = np.asarray([p["model_top_good"] for p in pairs], bool)
        values = [row[field] for row, _ in candidate_rows]
        labels = [label for _, label in candidate_rows]
        raw_auc, oriented_auc = auc(values, labels)
        direction = "GOOD_HIGH" if raw_auc >= .5 else "GOOD_LOW"
        oriented_wins = int(np.sum(deltas > 0)) if direction == "GOOD_HIGH" else int(np.sum(deltas < 0))
        output[field] = dict(
            good_range=[float(min(values[i] for i, label in enumerate(labels) if label)),
                        float(max(values[i] for i, label in enumerate(labels) if label))],
            bad_range=[float(min(values[i] for i, label in enumerate(labels) if not label)),
                       float(max(values[i] for i, label in enumerate(labels) if not label))],
            raw_auc_good_high=float(raw_auc), oriented_auc=float(oriented_auc), orientation=direction,
            oriented_within_state_wins=oriented_wins, within_state_pairs=len(pairs),
            solved_delta_median=float(np.median(deltas[solved])),
            failed_delta_median=float(np.median(deltas[~solved])),
            solved_delta_range=[float(deltas[solved].min()), float(deltas[solved].max())],
            failed_delta_range=[float(deltas[~solved].min()), float(deltas[~solved].max())],
        )
    return output


def write_csv(path, pairs):
    fields = ["state_id", "case_id", "split", "evaluation_count", "model_top_good",
              "model_margin", "candidate_label", "candidate_index", "model_score"] + TEMPORAL_FIELDS
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for pair in pairs:
            for label in ("good", "bad"):
                candidate = pair[label]
                writer.writerow({"state_id": pair["state_id"], "case_id": pair["case_id"],
                    "split": pair["split"], "evaluation_count": pair["evaluation_count"],
                    "model_top_good": pair["model_top_good"], "model_margin": pair["model_margin"],
                    "candidate_label": label.upper(), "candidate_index": candidate["candidate_index"],
                    "model_score": candidate["model_score"],
                    **{field: candidate[field] for field in TEMPORAL_FIELDS}})


def main():
    OUT.mkdir(parents=True, exist_ok=True); before = hashes()
    rows = [r for r in json.loads((DATA / "dataset.json").read_text())
            if r["task"] == "candidate_selection" and r["system"] == "kepler5"
            and int(r["case_id"]) in PERSISTENT]
    grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    grouped = {s: e for s, e in grouped.items() if any(r["target_good"] for r in e)}
    metadata = {r["state_id"]: r for r in json.loads((DATA / "states.json").read_text())}
    states = {s: GraphState(metadata[s], e) for s, e in grouped.items()}
    model = DeepCandidateModel(32, 3)
    checkpoint = torch.load(CHECKPOINT, weights_only=False); model.load_state_dict(checkpoint["model_state"])
    scores = model_scores(model, states, grouped)
    pairs = []
    for state_id, examples in grouped.items():
        logits = scores[state_id]; good = np.asarray([bool(r["target_good"]) for r in examples])
        gi = int(np.flatnonzero(good)[np.argmax(logits[good])])
        bi = int(np.flatnonzero(~good)[np.argmax(logits[~good])])
        z = np.load(metadata[state_id]["state_npz"])
        def candidate(index):
            features = temporal_features(z, int(examples[index]["entity_index"]))
            saved_prediction = float(z["candidate_surrogate_prediction"][int(examples[index]["entity_index"])])
            if abs(features["surrogate_prediction_final"] - saved_prediction) > 2e-12:
                raise RuntimeError(f"surrogate reconstruction mismatch {state_id}")
            return {"candidate_index": int(examples[index]["entity_index"]),
                    "example_id": examples[index]["example_id"], "model_score": float(logits[index]),
                    "source": examples[index]["entity_source"], **features}
        pairs.append(dict(state_id=state_id, case_id=int(examples[0]["case_id"]),
                          split=examples[0]["split"], evaluation_count=int(examples[0]["evaluation_count"]),
                          model_top_good=bool(logits[gi] > logits[bi]),
                          model_margin=float(logits[gi] - logits[bi]),
                          good=candidate(gi), bad=candidate(bi)))
    pairs.sort(key=lambda p: (p["case_id"], p["evaluation_count"]))
    summary = summarize_pairs(pairs)
    ranked = sorted(summary.items(), key=lambda item: item[1]["oriented_auc"], reverse=True)
    cases_22_41 = [p for p in pairs if p["case_id"] in (22, 41)]
    after = hashes()
    report = native(dict(
        status="READ_ONLY_TEMPORAL_DIAGNOSIS",
        scope=dict(states=len(pairs), solved=sum(p["model_top_good"] for p in pairs),
                   failed=sum(not p["model_top_good"] for p in pairs), systems=["kepler5"]),
        reconstruction=dict(
            available="ordered evaluated sites/losses; exact prefix nearest-neighbor, local density, surrogate, owner, and incumbent trajectories",
            unavailable="prior reservoir recurrence and exact historical reservoir-derived Voronoi cell extents",
            final_surrogate_replay_tolerance=2e-12),
        architecture_history_access=dict(
            insertion_order_scalar_on_each_site=True,
            global_candidate_to_site_attention=True,
            ordered_top24_neighbor_attention=True,
            global_previous_next_site_edges=True,
            recurrent_relational_depth=3,
            explicit_local_temporal_sequence_encoder=False,
            explicit_running_trend_or_density_deltas=False),
        temporal_feature_summary=summary,
        strongest_by_univariate_auc=[{"feature": k, **v} for k, v in ranked],
        cases_22_41=cases_22_41,
        pairs=pairs,
        dataset_hashes={p.name: sha256(p) for p in (DATA/"dataset.json", DATA/"states.json", DATA/"hard_cases.jsonl")},
        checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
        protected_checkpoint_changes=[p for p in set(before)&set(after) if before[p] != after[p]],
        checkpoint_hashes_unchanged=before == after,
    ))
    atomic_json(OUT / "report.json", report); write_csv(OUT / "candidate_pair_temporal_features.csv", pairs)
    print(json.dumps({"scope": report["scope"], "strongest": report["strongest_by_univariate_auc"][:8],
                      "cases_22_41": [{"state_id": p["state_id"], "margin": p["model_margin"]}
                                      for p in cases_22_41],
                      "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"]}, indent=2))


if __name__ == "__main__": main()
