"""Diagnose downstream effects of the zero-parameter N1 exp assembly.

No search component is changed.  Production np.exp and the recovered
correctly-rounded assembly are evaluated side-by-side on the saved N1 corpus.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.sho import dimension_general_edirect as dimension_general
from experiments.sho.recover_n1_loss_encoding import FLOOR, corpus, normalize
from experiments.sho.recover_n1_scalar_exp import assembled_exp


OUT = Path("outputs/sho/stage1B/n1_scalar_exp_downstream")
WEIGHT_FLOOR = np.float64(1.0e-15)
RBF_NEIGHBORS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, np.float64).view(np.uint64) == np.asarray(right, np.float64).view(np.uint64)


def ordered_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, np.float64).view(np.uint64)
    sign = bits >> np.uint64(63)
    return np.where(sign != 0, ~bits, bits ^ (np.uint64(1) << np.uint64(63)))


def ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = ordered_bits(left), ordered_bits(right)
    return np.maximum(a, b) - np.minimum(a, b)


def weight_pair(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(distances, np.float64)
    bandwidth = np.maximum(distances[:, -1:], np.float64(1.0e-12))
    z = np.multiply(np.float64(-9.0), np.square(np.divide(distances, bandwidth)))
    production = np.add(np.exp(z), WEIGHT_FLOOR)
    recovered = np.add(assembled_exp(z.ravel()).reshape(z.shape), WEIGHT_FLOOR)
    return production, recovered


class PropagationAudit:
    def __init__(self):
        self.counts = Counter()
        self.by_kind = defaultdict(Counter)
        self.maxima = defaultdict(float)

    def rows(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        kind: str,
        targets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        distances = np.asarray(distances, np.float64)
        indices = np.asarray(indices, np.int64)
        values = np.asarray(values, np.float64)
        row_chunk = max(1, 500_000 // distances.shape[1])
        production_predictions = []
        recovered_predictions = []
        production_residuals = []
        recovered_residuals = []
        for start in range(0, len(distances), row_chunk):
            stop = start + row_chunk
            d, ids = distances[start:stop], indices[start:stop]
            production_weights, recovered_weights = weight_pair(d)
            local_values = values[ids]
            production_terms = np.multiply(production_weights, local_values)
            recovered_terms = np.multiply(recovered_weights, local_values)
            production_numerator = np.sum(production_terms, axis=1)
            recovered_numerator = np.sum(recovered_terms, axis=1)
            production_denominator = np.sum(production_weights, axis=1)
            recovered_denominator = np.sum(recovered_weights, axis=1)
            production_prediction = np.divide(production_numerator, production_denominator)
            recovered_prediction = np.divide(recovered_numerator, recovered_denominator)

            local = Counter(
                rows=len(d),
                weights=d.size,
                weight_differences=int(np.count_nonzero(~bits_equal(production_weights, recovered_weights))),
                weighted_term_differences=int(np.count_nonzero(~bits_equal(production_terms, recovered_terms))),
                numerator_differences=int(np.count_nonzero(~bits_equal(production_numerator, recovered_numerator))),
                denominator_differences=int(np.count_nonzero(~bits_equal(production_denominator, recovered_denominator))),
                prediction_differences=int(np.count_nonzero(~bits_equal(production_prediction, recovered_prediction))),
                prediction_numeric_differences=int(np.count_nonzero(production_prediction != recovered_prediction)),
            )
            if targets is not None:
                target = np.asarray(targets[start:stop], np.float64)
                production_residual = np.abs(np.subtract(target, production_prediction))
                recovered_residual = np.abs(np.subtract(target, recovered_prediction))
                local["loo_differences"] += int(np.count_nonzero(
                    ~bits_equal(production_residual, recovered_residual)
                ))
                production_residuals.append(production_residual)
                recovered_residuals.append(recovered_residual)
            self.counts.update(local)
            self.by_kind[kind].update(local)
            pred_ulp = ulp_distance(production_prediction, recovered_prediction)
            self.maxima["prediction_ulp"] = max(self.maxima["prediction_ulp"], float(pred_ulp.max(initial=0)))
            self.maxima["prediction_abs"] = max(
                self.maxima["prediction_abs"],
                float(np.max(np.abs(production_prediction - recovered_prediction), initial=0.0)),
            )
            production_predictions.append(production_prediction)
            recovered_predictions.append(recovered_prediction)
        return (
            np.concatenate(production_predictions),
            np.concatenate(recovered_predictions),
            np.concatenate(production_residuals) if production_residuals else None,
            np.concatenate(recovered_residuals) if recovered_residuals else None,
        )


def surface_pair(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    audit: PropagationAudit,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.atleast_2d(np.asarray(queries, np.float64))
    count = min(RBF_NEIGHBORS, len(points))
    distances, indices = cKDTree(points).query(queries, k=count)
    if count == 1:
        distances, indices = distances[:, None], indices[:, None]
    production, recovered, _, _ = audit.rows(distances, indices, values, kind)
    return production, recovered


def loo_pair(
    points: np.ndarray,
    values: np.ndarray,
    site_indices: np.ndarray,
    audit: PropagationAudit,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    site_indices = np.asarray(site_indices, np.int64)
    if len(points) <= 2:
        ones = np.ones(len(site_indices), np.float64)
        audit.counts["loo_short_circuit_rows"] += len(site_indices)
        audit.by_kind[kind]["loo_short_circuit_rows"] += len(site_indices)
        return ones, ones.copy()
    count = min(RBF_NEIGHBORS + 1, len(points))
    distances, indices = cKDTree(points).query(points[site_indices], k=count)
    if count == 1:
        distances, indices = distances[:, None], indices[:, None]
    grouped = defaultdict(list)
    for row, site in enumerate(site_indices):
        keep = indices[row] != site
        local_indices = indices[row, keep][:RBF_NEIGHBORS]
        local_distances = distances[row, keep][:RBF_NEIGHBORS]
        grouped[len(local_indices)].append((row, local_distances, local_indices))
    production = np.empty(len(site_indices), np.float64)
    recovered = np.empty(len(site_indices), np.float64)
    for length, entries in grouped.items():
        rows = np.asarray([entry[0] for entry in entries], np.int64)
        if length == 0:
            production[rows] = recovered[rows] = 0.0
            continue
        d = np.stack([entry[1] for entry in entries])
        ids = np.stack([entry[2] for entry in entries])
        target = values[site_indices[rows]]
        _, _, p_residual, r_residual = audit.rows(d, ids, values, kind, target)
        production[rows], recovered[rows] = p_residual, r_residual
    return production, recovered


def replay_cell_state(item: dict, audit: PropagationAudit) -> dict | None:
    if not item["events_path"].exists() or not item["continuous_path"].exists():
        return None
    events = json.loads(item["events_path"].read_text())
    traces = json.loads(item["continuous_path"].read_text())
    if not traces:
        return None
    iterations = sorted({int(event["iteration"]) for event in events if event["source"] == "cell"})
    if not iterations:
        return None
    iteration = iterations[-1]
    first = next(i for i, event in enumerate(events)
                 if event["source"] == "cell" and int(event["iteration"]) == iteration)
    points = np.asarray([event["point"] for event in events[:first]], np.float64)
    losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
    values = np.log10(np.maximum(losses, FLOOR))
    case_id = int(traces[0]["case_id"])
    api = dimension_general.composition(points.shape[1])
    data = api["_cell_data"](points, losses, case_id, iteration)
    selected_sizes = data.sizes[data.selected]
    size_weight = normalize(selected_sizes)
    p_loo, r_loo = loo_pair(points, values, data.selected, audit, "cell_selected_loo")
    p_loo_normalized, r_loo_normalized = normalize(p_loo), normalize(r_loo)
    p_mix = np.add(np.multiply(.5, size_weight), np.multiply(.5, p_loo_normalized))
    r_mix = np.add(np.multiply(.5, size_weight), np.multiply(.5, r_loo_normalized))
    largest_position = int(np.argmax(selected_sizes))
    p_mix[largest_position] = r_mix[largest_position] = 1.0

    p_proposals, r_proposals = [], []
    changed_cells = []
    candidate_prediction_differences = normalized_prediction_differences = 0
    score_differences = winner_differences = 0
    for selected_position, site_index in enumerate(data.selected):
        mask = data.owners == site_index
        candidates, distances = data.points[mask], data.distances[mask]
        if not len(candidates):
            continue
        if selected_position == largest_position:
            p_winner = r_winner = int(np.argmax(distances))
        else:
            distance_bar = normalize(distances)
            p_prediction, r_prediction = surface_pair(
                points, values, candidates, audit, "cell_candidate_prediction"
            )
            p_prediction_bar, r_prediction_bar = normalize(p_prediction), normalize(r_prediction)
            candidate_prediction_differences += int(np.count_nonzero(
                ~bits_equal(p_prediction, r_prediction)
            ))
            normalized_prediction_differences += int(np.count_nonzero(
                ~bits_equal(p_prediction_bar, r_prediction_bar)
            ))
            p_score = np.subtract(
                np.multiply(float(p_mix[selected_position]), distance_bar),
                np.multiply(1.0 - float(p_mix[selected_position]), p_prediction_bar),
            )
            r_score = np.subtract(
                np.multiply(float(r_mix[selected_position]), distance_bar),
                np.multiply(1.0 - float(r_mix[selected_position]), r_prediction_bar),
            )
            score_differences += int(np.count_nonzero(~bits_equal(p_score, r_score)))
            p_winner, r_winner = int(np.argmax(p_score)), int(np.argmax(r_score))
            winner_differences += int(p_winner != r_winner)
            if p_winner != r_winner:
                changed_cells.append(dict(site_index=int(site_index), selected_position=selected_position,
                                          production_winner=p_winner, recovered_winner=r_winner))
        for answer, winner in ((p_proposals, p_winner), (r_proposals, r_winner)):
            proposal = candidates[winner]
            if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
                answer.append(proposal)

    p_array = np.asarray(p_proposals, np.float64)
    r_array = np.asarray(r_proposals, np.float64)
    proposal_equal = p_array.shape == r_array.shape and bool(np.array_equal(p_array, r_array))
    return dict(
        state=str(item["state_path"]), family=item["family"], iteration=iteration,
        sites=len(points), selected_cells=len(data.selected),
        loo_differences=int(np.count_nonzero(~bits_equal(p_loo, r_loo))),
        normalized_loo_differences=int(np.count_nonzero(~bits_equal(p_loo_normalized, r_loo_normalized))),
        mixed_weight_differences=int(np.count_nonzero(~bits_equal(p_mix, r_mix))),
        candidate_prediction_differences=candidate_prediction_differences,
        normalized_prediction_differences=normalized_prediction_differences,
        score_differences=score_differences,
        winner_differences=winner_differences,
        production_proposals=len(p_array), recovered_proposals=len(r_array),
        proposal_decision_equal=proposal_equal, changed_cells=changed_cells,
    )


def run() -> None:
    started = time.monotonic()
    script_dependencies = {
        "production_surrogate": Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
        "dimension_general": Path("experiments/sho/dimension_general_edirect.py"),
        "continuous": Path("experiments/sho/continuous_surrogate_edirect.py"),
        "exp_assembly": Path("experiments/sho/recover_n1_scalar_exp.py"),
    }
    checkpoint_dependencies = {
        "multiply": Path("outputs/sho/stage1B/multiplication_operator_smoke/multiplication_before_evaluation.pt"),
        "subtract": Path("outputs/sho/stage1B/subtraction_operator_training/subtraction_before_evaluation.pt"),
        "add": Path("outputs/sho/stage1B/addition_operator_smoke/addition_before_evaluation.pt"),
    }
    dependencies = {**script_dependencies, **checkpoint_dependencies}
    before = {name: sha256(path) for name, path in dependencies.items()}
    records = corpus()
    audit = PropagationAudit()
    continuous = Counter()
    continuous_changed = []

    # Exact same two sources that formed the 15,055,875-value exp corpus:
    # every final-state LOO row and every saved continuous-start endpoint.
    for record_index, item in enumerate(records):
        points, losses = item["points"], item["losses"]
        values = np.log10(np.maximum(losses, FLOOR))
        p_loo, r_loo = loo_pair(
            points, values, np.arange(len(points), dtype=np.int64), audit, "final_state_loo"
        )
        continuous["final_loo_rows"] += len(p_loo)
        continuous["final_loo_differences"] += int(np.count_nonzero(~bits_equal(p_loo, r_loo)))
        path = item["continuous_path"]
        if path.exists():
            for trace_index, trace in enumerate(json.loads(path.read_text())):
                n = int(trace["site_count"])
                if n > len(points):
                    continue
                queries = np.asarray([entry["point"] for entry in trace["starts"]], np.float64)
                local_values = np.log10(np.maximum(losses[:n], FLOOR))
                p_prediction, r_prediction = surface_pair(
                    points[:n], local_values, queries, audit, "continuous_endpoint"
                )
                p_choice, r_choice = int(np.argmin(p_prediction)), int(np.argmin(r_prediction))
                stored_choice = int(trace["chosen_start"])
                stored_predictions = np.asarray(
                    [entry["prediction"] for entry in trace["starts"]], np.float64
                )
                continuous["states"] += 1
                continuous["endpoints"] += len(queries)
                continuous["endpoint_prediction_differences"] += int(np.count_nonzero(
                    ~bits_equal(p_prediction, r_prediction)
                ))
                continuous["choice_differences"] += int(p_choice != r_choice)
                continuous["production_saved_choice_differences"] += int(p_choice != stored_choice)
                continuous["production_saved_prediction_differences"] += int(np.count_nonzero(
                    ~bits_equal(p_prediction, stored_predictions)
                ))
                if p_choice != r_choice:
                    continuous_changed.append(dict(
                        state=str(item["state_path"]), family=item["family"], trace_index=trace_index,
                        iteration=int(trace["iteration"]), site_count=n,
                        production_choice=p_choice, recovered_choice=r_choice,
                        production_prediction=float(p_prediction[p_choice]),
                        recovered_prediction=float(r_prediction[r_choice]),
                    ))
        if record_index % 32 == 0:
            print(json.dumps(dict(record=record_index, weights=audit.counts["weights"],
                                  predictions=audit.counts["prediction_differences"])), flush=True)

    # Match the established bounded decision corpus: the last real cell round
    # in every saved case having both events and continuous traces.
    cell_rows = []
    for item in records:
        row = replay_cell_state(item, audit)
        if row is not None:
            cell_rows.append(row)
    cell_summary = Counter()
    for row in cell_rows:
        cell_summary["states"] += 1
        for key in ("selected_cells", "loo_differences", "normalized_loo_differences",
                    "mixed_weight_differences", "candidate_prediction_differences",
                    "normalized_prediction_differences", "score_differences",
                    "winner_differences", "production_proposals", "recovered_proposals"):
            cell_summary[key] += int(row[key])
        cell_summary["proposal_decision_differences"] += int(not row["proposal_decision_equal"])

    after = {name: sha256(path) for name, path in dependencies.items()}
    decision_differences = continuous["choice_differences"] + cell_summary["winner_differences"]
    result = dict(
        component="n1_scalar_exp_downstream_diagnosis",
        status="BEHAVIORALLY_EQUIVALENT_ON_SAVED_CORPUS" if decision_differences == 0 else "DECISION_DIFFERENCES_FOUND",
        branches={
            "A": "production np.exp",
            "B": "zero-parameter correctly-rounded exp assembly",
            "identical_later_operations": [
                "+1e-15", "ordered weighted multiplication/sum/division", "absolute LOO residual",
                "guarded min-max normalization", "ordinary score", "canonical first-maximum routing",
            ],
        },
        main_corpus=dict(
            records=len(records), expected_kernel_values=15_055_875,
            **dict(audit.counts), by_kind={key: dict(value) for key, value in audit.by_kind.items()},
            maximum_prediction_ulp=int(audit.maxima["prediction_ulp"]),
            maximum_prediction_absolute_difference=audit.maxima["prediction_abs"],
        ),
        continuous_endpoint_selection=dict(continuous, changed_states=continuous_changed),
        cell_proposal_decisions=dict(cell_summary, changed_states=[
            row for row in cell_rows if not row["proposal_decision_equal"] or row["winner_differences"]
        ]),
        first_downstream_difference=(
            "weighted terms" if audit.counts["weighted_term_differences"] else
            "weight-sum normalization" if audit.counts["denominator_differences"] else
            "none after kernel weights"
        ),
        first_decision_difference=(None if decision_differences == 0 else
                                   "continuous endpoint choice or ordinary-cell winner; see changed states"),
        closed_loop_run=bool(decision_differences != 0),
        closed_loop_note=(
            "Not run because no saved N1 endpoint/cell decision changed."
            if decision_differences == 0 else
            "Required: changed saved cases must be replayed closed-loop."
        ),
        dependencies={name: dict(path=str(path), sha256=before[name], unchanged=before[name] == after[name])
                      for name, path in dependencies.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cell_states.json").write_text(json.dumps(cell_rows, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if audit.by_kind["final_state_loo"]["weights"] + audit.by_kind["continuous_endpoint"]["weights"] != 15_055_875:
        raise SystemExit("main kernel corpus count changed")
    if before != after:
        raise SystemExit("dependency changed")


if __name__ == "__main__":
    run()
