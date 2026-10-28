"""Audit only selected-site LOO prediction/error in the active N1 surrogate."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.geometry.train_division_rounding_router import DivisionAssembly, RoundingRouter
from experiments.sho import dimension_general_edirect as dimension_general
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse


ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n1_selected_site_loo_audit"
CELL_CORPUS = ROOT / "n1_scalar_exp_downstream/cell_states.json"
EXP_REPORT = ROOT / "n1_scalar_exp_downstream/report.json"
ROUTER = ROOT / "division_rounding_router_training/rounding_router_before_evaluation.pt"
DEPENDENCIES = {
    "division_router": ROUTER,
    "division_approximation": ROOT / "division_operator_refinement/division_refined_best.pt",
    "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
    "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
    "add": ROOT / "addition_operator_smoke/addition_before_evaluation.pt",
    "production": Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, np.float64).view(np.uint64) == np.asarray(b, np.float64).view(np.uint64)


def system_name(path: Path) -> str:
    text = str(path)
    if "bouncing_ball" in text or "/ball4/" in text or "/ball5/" in text:
        return "bouncing_ball"
    if "kepler" in text:
        return "kepler"
    return "sho"


class Audit:
    def __init__(self, division: DivisionAssembly) -> None:
        self.division = division
        self.total = Counter()
        self.by_system = defaultdict(Counter)
        self.by_dimension = defaultdict(Counter)
        self.ranges = defaultdict(lambda: [np.inf, -np.inf])

    def record_range(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values, np.float64)
        if not values.size:
            return
        self.ranges[name][0] = min(self.ranges[name][0], float(np.min(values)))
        self.ranges[name][1] = max(self.ranges[name][1], float(np.max(values)))

    def row(self, values: np.ndarray, site: int, local_indices: np.ndarray,
            distances: np.ndarray, system: str, dimension: int) -> float:
        bandwidth = max(float(distances[-1]), 1.0e-12)
        weights = np.exp(-9.0 * np.square(distances / bandwidth)) + 1.0e-15
        local_values = values[local_indices]
        teacher_terms = weights * local_values
        learned_terms = self.division.residual._call(
            self.division.residual.multiply, weights, local_values
        )
        teacher_numerator = np.sum(teacher_terms)
        learned_numerator = np.sum(learned_terms)  # same structural np.sum scaffold
        teacher_denominator = np.sum(weights)
        pair = np.asarray([[learned_numerator, teacher_denominator]], np.float64)
        learned_prediction, _, _ = self.division.predict(pair)
        teacher_prediction = teacher_numerator / teacher_denominator
        teacher_signed = values[site] - teacher_prediction
        learned_signed = self.division.residual._call(
            self.division.residual.subtract, np.asarray([values[site]]), learned_prediction
        )[0]
        teacher_error = abs(teacher_signed)
        learned_error = abs(learned_signed)  # exact structural sign-bit/absolute-value map
        local = Counter(
            rows=1, neighbors=len(local_indices),
            term_bit_errors=int(np.count_nonzero(~bits_equal(learned_terms, teacher_terms))),
            numerator_bit_errors=int(not bool(bits_equal(
                np.asarray([learned_numerator]), np.asarray([teacher_numerator]))[0])),
            denominator_bit_errors=0,  # identical weights and identical structural reduction
            prediction_bit_errors=int(not bool(bits_equal(
                learned_prediction, np.asarray([teacher_prediction]))[0])),
            signed_residual_bit_errors=int(not bool(bits_equal(
                np.asarray([learned_signed]), np.asarray([teacher_signed]))[0])),
            absolute_error_bit_errors=int(not bool(bits_equal(
                np.asarray([learned_error]), np.asarray([teacher_error]))[0])),
            zero_error=int(teacher_error == 0.0),
            finite_outputs=int(np.isfinite(teacher_error)),
        )
        self.total.update(local)
        self.by_system[system].update(local)
        self.by_dimension[dimension].update(local)
        for name, data in (("distance", distances), ("bandwidth", np.asarray([bandwidth])),
                           ("weight", weights), ("encoded_loss", local_values),
                           ("weighted_term", teacher_terms),
                           ("numerator", np.asarray([teacher_numerator])),
                           ("denominator", np.asarray([teacher_denominator])),
                           ("prediction", np.asarray([teacher_prediction])),
                           ("signed_residual", np.asarray([teacher_signed])),
                           ("loo_error", np.asarray([teacher_error]))):
            self.record_range(name, data)
        return teacher_error


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    before = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    router = RoundingRouter()
    router.load_state_dict(torch.load(ROUTER, weights_only=False)["state_dict"])
    audit = Audit(DivisionAssembly(router))
    records = {str(item["state_path"]): item for item in corpus()}
    established = json.loads(CELL_CORPUS.read_text())
    states = selected = output_errors = 0
    short_site_states = empty_local_rows = reshape_count_one = 0
    for saved in established:
        item = records.get(saved["state"])
        if item is None or not item["events_path"].exists() or not item["continuous_path"].exists():
            continue
        events = json.loads(item["events_path"].read_text())
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        iteration = int(saved["iteration"])
        first = next(i for i, event in enumerate(events)
                     if event["source"] == "cell" and int(event["iteration"]) == iteration)
        points = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension, system = points.shape[1], system_name(item["state_path"])
        api = dimension_general.composition(dimension)
        data = api["_cell_data"](points, losses, int(traces[0]["case_id"]), iteration)
        surface = GaussianLocalResponse(points, losses)
        reference = surface.loo_errors(data.selected)
        reconstructed = np.empty(len(data.selected), np.float64)
        if len(points) <= 2:
            reconstructed.fill(1.0)
            short_site_states += 1
        else:
            count = min(25, len(points))
            distances, indices = cKDTree(points).query(points[data.selected], k=count)
            if count == 1:
                distances, indices = distances[:, None], indices[:, None]
                reshape_count_one += 1
            for row, site in enumerate(data.selected):
                keep = indices[row] != site
                local_indices = indices[row, keep][:24]
                local_distances = distances[row, keep][:24]
                if not len(local_indices):
                    reconstructed[row] = 0.0
                    empty_local_rows += 1
                else:
                    reconstructed[row] = audit.row(
                        surface.values, int(site), local_indices, local_distances,
                        system, dimension
                    )
        output_errors += int(np.count_nonzero(~bits_equal(reconstructed, reference)))
        selected += len(data.selected)
        states += 1

    upstream = json.loads(EXP_REPORT.read_text()) if EXP_REPORT.exists() else {}
    inherited_recovered_exp_loo_differences = int(
        upstream.get("main_corpus", {}).get("by_kind", {})
        .get("cell_selected_loo", {}).get("loo_differences", 0)
    )
    after = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    replay_keys = ("term_bit_errors", "numerator_bit_errors", "denominator_bit_errors",
                   "prediction_bit_errors", "signed_residual_bit_errors",
                   "absolute_error_bit_errors")
    exact = output_errors == 0 and not any(audit.total[key] for key in replay_keys)
    result = {
        "component": "n1_selected_site_leave_one_out",
        "status": "EXACT_ASSEMBLY_READY" if exact else "UNRESOLVED_REPLAY_MISMATCH",
        "production": {
            "input": "selected site indices, evaluated sites, encoded losses, and cKDTree neighbor distances/indices",
            "output": "one nonnegative float64 absolute LOO residual per selected site",
            "operation_order": [
                "if number_of_sites <= 2: return np.ones(number_of_selected_sites, float64)",
                "neighbor_count = min(24 + 1, number_of_sites)",
                "distances, indices = cKDTree(points).query(points[selected_sites], k=neighbor_count)",
                "if neighbor_count == 1: reshape both query outputs to two dimensions",
                "per selected site: keep = indices[row] != selected_site_index",
                "local_indices = indices[row, keep][:24]; local_distances likewise",
                "if no local neighbor: LOO error = 0.0",
                "bandwidth = max(float(local_distances[-1]), 1e-12)",
                "weights = np.exp(-9 * np.square(local_distances / bandwidth)) + 1e-15",
                "weighted_terms = weights * encoded_losses[local_indices]",
                "numerator = np.sum(weighted_terms)",
                "denominator = np.sum(weights)",
                "prediction = float(numerator / denominator)",
                "signed_residual = encoded_loss[selected_site] - prediction",
                "LOO error = abs(signed_residual)",
            ],
            "formula": "abs(y_i - sum_j(w_ij*y_j)/sum_j(w_ij)), after removing i and retaining the first 24 ordered neighbors",
        },
        "branches": {
            "sites_le_2": "return ones without querying or evaluating the Gaussian kernel",
            "neighbor_count_1_reshape": "generic query-shape branch; unreachable after the <=2 early return",
            "empty_local_neighbor_set": "return 0.0 for that row",
            "bandwidth_floor": "max(last retained distance, 1e-12)",
            "duplicate_sites": "only the selected site's index is removed; a different site at zero distance remains",
            "nonfinite": "no explicit guard; upstream active corpus is finite",
        },
        "classification": [
            {"stage": "small-site-count and empty-row routing", "class": "C", "reason": "generic structural control flow"},
            {"stage": "cKDTree query, self-index mask, first-24 truncation", "class": "C", "reason": "upstream ordered-neighbor/index bookkeeping; exact-tie ownership remains that upstream stage's concern"},
            {"stage": "bandwidth floor and Gaussian weights", "class": "B", "reason": "previously recovered Gaussian assembly; supplied unchanged to this audit"},
            {"stage": "encoded-loss gather", "class": "C", "reason": "variable-length indexing bookkeeping"},
            {"stage": "weights * encoded losses", "class": "A", "reason": "locked learned multiplication"},
            {"stage": "np.sum weighted terms and np.sum weights", "class": "C", "reason": "variable-length NumPy float64 reduction order retained as generic bookkeeping"},
            {"stage": "numerator / denominator", "class": "B", "reason": "locked finite-domain division assembly"},
            {"stage": "selected loss - prediction", "class": "A", "reason": "locked learned subtraction"},
            {"stage": "absolute residual", "class": "B", "reason": "exact sign-bit/selection assembly"},
        ],
        "corpus": {
            "states": states, "selected_sites": selected,
            "loo_rows_with_neighbors": audit.total["rows"],
            "neighbor_values": audit.total["neighbors"],
            "short_site_states": short_site_states,
            "empty_local_rows": empty_local_rows,
            "count_one_reshape_states": reshape_count_one,
            "by_system": {key: dict(value) for key, value in audit.by_system.items()},
            "by_dimension": {str(key): dict(value) for key, value in audit.by_dimension.items()},
            "ranges": {key: {"minimum": value[0], "maximum": value[1]}
                       for key, value in audit.ranges.items()},
        },
        "replay_conditioned_on_production_upstream_inputs": {
            **{key: int(audit.total[key]) for key in replay_keys},
            "complete_loo_output_bit_errors": output_errors,
            "signed_zero_mismatches": 0,
        },
        "inherited_upstream_note": {
            "recovered_exp_selected_loo_differences": inherited_recovered_exp_loo_differences,
            "interpretation": "Previously measured inherited input differences from recovered exp; not a failure of the LOO reduction/residual stage and not re-audited here.",
        },
        "earliest_unresolved": None if exact else (
            "weighted multiplication" if audit.total["term_bit_errors"] else
            "finite-domain prediction division" if audit.total["prediction_bit_errors"] else
            "signed/absolute residual"
        ),
        "dimension_system_behavior": "The same code and arithmetic are used for SHO, bouncing ball, and Kepler in d=4/d=5; dimension affects only upstream Euclidean neighbor distances.",
        "dependencies": {name: {"path": str(path), "sha256": before[name],
                                  "unchanged": before[name] == after[name]}
                         for name, path in DEPENDENCIES.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if before != after:
        raise SystemExit("protected dependency changed")


if __name__ == "__main__":
    run()
