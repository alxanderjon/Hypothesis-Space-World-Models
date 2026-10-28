"""Audit only the active N1 guarded min/max normalization.

This script is read-only with respect to learned checkpoints and E-DIRECT.  It
reconstructs the established saved cell-decision corpus, records every active
normalization input, and replays the scalar arithmetic with the already locked
subtraction, multiplication, and division assemblies.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_division_rounding_router import DivisionAssembly, RoundingRouter
from experiments.sho import dimension_general_edirect as dimension_general
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize


ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n1_guarded_normalization_audit"
CELL_CORPUS = ROOT / "n1_scalar_exp_downstream/cell_states.json"
ROUTER = ROOT / "division_rounding_router_training/rounding_router_before_evaluation.pt"
DEPENDENCIES = {
    "division_router": ROUTER,
    "division_approximation": ROOT / "division_operator_refinement/division_refined_best.pt",
    "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
    "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
    "production": Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, np.float64).view(np.uint64) == np.asarray(right, np.float64).view(np.uint64)


def system_name(path: Path, dimension: int) -> str:
    value = str(path)
    if "bouncing_ball" in value or "/ball4/" in value or "/ball5/" in value:
        return "bouncing_ball"
    if "kepler" in value:
        return "kepler"
    return "sho"


class Audit:
    def __init__(self) -> None:
        self.arrays: list[np.ndarray] = []
        self.kinds: list[str] = []
        self.systems: list[str] = []
        self.dimensions: list[int] = []
        self.counts = Counter()
        self.by_kind = defaultdict(Counter)
        self.by_system = defaultdict(Counter)
        self.by_dimension = defaultdict(Counter)
        self.ranges = defaultdict(lambda: [np.inf, -np.inf])

    def add(self, values: np.ndarray, kind: str, system: str, dimension: int) -> None:
        values = np.asarray(values, np.float64)
        if values.size == 0:
            raise AssertionError("production normalization never receives an empty array")
        minimum, maximum = float(np.min(values)), float(np.max(values))
        spread = maximum - minimum
        scale = max(1.0, abs(minimum), abs(maximum))
        tolerance = 1.0e-15 * scale
        guarded = spread <= tolerance
        local = Counter(calls=1, values=values.size, guarded_calls=int(guarded),
                        ordinary_calls=int(not guarded), exact_zero_range=int(spread == 0.0),
                        finite_values=int(np.count_nonzero(np.isfinite(values))),
                        nonfinite_values=int(np.count_nonzero(~np.isfinite(values))))
        self.counts.update(local)
        self.by_kind[kind].update(local)
        self.by_system[system].update(local)
        self.by_dimension[dimension].update(local)
        for name, value in (("input", values), ("minimum", np.asarray([minimum])),
                            ("maximum", np.asarray([maximum])), ("spread", np.asarray([spread])),
                            ("scale", np.asarray([scale])), ("tolerance", np.asarray([tolerance]))):
            self.ranges[name][0] = min(self.ranges[name][0], float(np.min(value)))
            self.ranges[name][1] = max(self.ranges[name][1], float(np.max(value)))
        self.arrays.append(values.copy())
        self.kinds.append(kind)
        self.systems.append(system)
        self.dimensions.append(dimension)


def reconstruct_inputs(audit: Audit) -> int:
    records = {str(item["state_path"]): item for item in corpus()}
    established = json.loads(CELL_CORPUS.read_text())
    replayed = 0
    for row in established:
        item = records.get(row["state"])
        if item is None or not item["events_path"].exists() or not item["continuous_path"].exists():
            continue
        events = json.loads(item["events_path"].read_text())
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        iteration = int(row["iteration"])
        first = next(i for i, event in enumerate(events)
                     if event["source"] == "cell" and int(event["iteration"]) == iteration)
        points = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension = points.shape[1]
        system = system_name(item["state_path"], dimension)
        api = dimension_general.composition(dimension)
        data = api["_cell_data"](points, losses, int(traces[0]["case_id"]), iteration)
        surface = GaussianLocalResponse(points, losses)
        selected_sizes = data.sizes[data.selected]
        audit.add(selected_sizes, "selected_cell_sizes", system, dimension)
        audit.add(surface.loo_errors(data.selected), "selected_cell_loo", system, dimension)
        largest_position = int(np.argmax(selected_sizes))
        for selected_position, site_index in enumerate(data.selected):
            mask = data.owners == site_index
            candidates, distances = data.points[mask], data.distances[mask]
            if not len(candidates) or selected_position == largest_position:
                continue
            audit.add(distances, "ordinary_candidate_distances", system, dimension)
            audit.add(surface.predict(candidates), "ordinary_candidate_predictions", system, dimension)
        replayed += 1
    return replayed


def replay(audit: Audit, division: DivisionAssembly) -> dict:
    calls = len(audit.arrays)
    minima = np.asarray([float(np.min(v)) for v in audit.arrays], np.float64)
    maxima = np.asarray([float(np.max(v)) for v in audit.arrays], np.float64)
    teacher_spread = maxima - minima
    spread = division.residual._call(division.residual.subtract, maxima, minima)
    scales = np.maximum(np.float64(1.0), np.maximum(np.abs(minima), np.abs(maxima)))
    tolerance = division.residual._call(
        division.residual.multiply, np.full(calls, np.float64(1.0e-15)), scales
    )
    teacher_tolerance = np.float64(1.0e-15) * scales
    teacher_guard = teacher_spread <= teacher_tolerance
    margin = division.residual._call(division.residual.subtract, spread, tolerance)
    with torch.no_grad():
        sign = division.threshold.decision(torch.from_numpy(margin)).numpy().astype(np.int8)
    assembled_guard = sign <= 0

    output_errors = numerator_errors = denominator_errors = division_errors = 0
    signed_zero_errors = 0
    values_replayed = 0
    max_ulp = 0
    # Keep memory bounded while retaining the exact per-call operation order.
    for values, minimum, maximum, guarded in zip(audit.arrays, minima, maxima, teacher_guard):
        teacher = _normalize(values)
        if guarded:
            assembled = np.full_like(values, 0.5, dtype=np.float64)
        else:
            numerator = division.residual._call(
                division.residual.subtract, values, np.full_like(values, minimum)
            )
            teacher_numerator = values - minimum
            denominator = division.residual._call(
                division.residual.subtract, np.asarray([maximum]), np.asarray([minimum])
            )[0]
            teacher_denominator = maximum - minimum
            numerator_errors += int(np.count_nonzero(~bits_equal(numerator, teacher_numerator)))
            denominator_errors += int(not bool(bits_equal(np.asarray([denominator]),
                                                        np.asarray([teacher_denominator]))[0]))
            pairs = np.column_stack((numerator, np.full_like(numerator, denominator)))
            assembled, _, _ = division.predict(pairs)
            teacher_division = teacher_numerator / teacher_denominator
            division_errors += int(np.count_nonzero(~bits_equal(assembled, teacher_division)))
        mismatch = ~bits_equal(assembled, teacher)
        output_errors += int(np.count_nonzero(mismatch))
        signed_zero_errors += int(np.count_nonzero(
            (assembled == 0.0) & (teacher == 0.0) & (np.signbit(assembled) != np.signbit(teacher))))
        if np.any(mismatch):
            # All expected errors, if any, are local division rounding errors.
            ai = assembled.view(np.uint64)
            ti = teacher.view(np.uint64)
            max_ulp = max(max_ulp, int(np.max(np.maximum(ai, ti) - np.minimum(ai, ti))))
        values_replayed += values.size
    return dict(
        calls=calls, values=values_replayed,
        spread_bit_errors=int(np.count_nonzero(~bits_equal(spread, teacher_spread))),
        tolerance_bit_errors=int(np.count_nonzero(~bits_equal(tolerance, teacher_tolerance))),
        guard_decision_errors=int(np.count_nonzero(assembled_guard != teacher_guard)),
        numerator_bit_errors=numerator_errors,
        denominator_bit_errors=denominator_errors,
        division_bit_errors=division_errors,
        final_output_bit_errors=output_errors,
        signed_zero_mismatches=signed_zero_errors,
        maximum_output_ulp_difference=max_ulp,
    )


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    before = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    audit = Audit()
    states = reconstruct_inputs(audit)
    router = RoundingRouter()
    router.load_state_dict(torch.load(ROUTER, weights_only=False)["state_dict"])
    division = DivisionAssembly(router)
    replay_result = replay(audit, division)
    after = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    exact = not any(replay_result[key] for key in (
        "spread_bit_errors", "tolerance_bit_errors", "guard_decision_errors",
        "numerator_bit_errors", "denominator_bit_errors", "division_bit_errors",
        "final_output_bit_errors", "signed_zero_mismatches"))
    result = {
        "component": "n1_guarded_normalization",
        "status": "LOCKED_ASSEMBLY_READY" if exact else "UNRESOLVED_REPLAY_MISMATCH",
        "production": {
            "input": "one nonempty float64 vector per active normalization call",
            "output": "same-shape float64 vector",
            "default": 0.5,
            "operation_order": [
                "minimum = float(np.min(values))",
                "maximum = float(np.max(values))",
                "spread_for_guard = maximum - minimum",
                "scale = max(1.0, abs(minimum), abs(maximum))",
                "tolerance = 1.0e-15 * scale",
                "if spread_for_guard <= tolerance: np.full_like(values, 0.5, dtype=np.float64)",
                "else numerator = values - minimum",
                "else denominator = maximum - minimum (evaluated again)",
                "else output = numerator / denominator",
            ],
            "nonfinite_behavior": "no explicit guard; np.min/max and arithmetic propagate nonfinite values",
            "empty_behavior": "np.min raises; active call sites exclude empty arrays",
        },
        "classification": [
            {"stage": "nonempty call-site routing", "class": "C", "reason": "generic control-flow bookkeeping"},
            {"stage": "ordered np.min/np.max reductions", "class": "C", "reason": "generic variable-length reduction/index bookkeeping; scalar comparisons are already available"},
            {"stage": "maximum - minimum (both evaluations)", "class": "A", "reason": "locked learned subtraction"},
            {"stage": "abs(minimum), abs(maximum)", "class": "B", "reason": "exact sign/selection assembly"},
            {"stage": "max(1, abs(min), abs(max))", "class": "B", "reason": "existing comparison plus structural selection"},
            {"stage": "1e-15 * scale", "class": "A", "reason": "locked learned multiplication"},
            {"stage": "spread <= tolerance", "class": "B", "reason": "locked subtraction and exact zero-sign router"},
            {"stage": "default 0.5 fill / branch selection", "class": "C", "reason": "shape fill and branch bookkeeping"},
            {"stage": "values - minimum", "class": "A", "reason": "locked learned subtraction"},
            {"stage": "numerator / denominator", "class": "B", "reason": "locked finite-domain division assembly"},
        ],
        "corpus": {
            "established_cell_states": states,
            "calls": audit.counts["calls"], "values": audit.counts["values"],
            "guarded_calls": audit.counts["guarded_calls"],
            "ordinary_calls": audit.counts["ordinary_calls"],
            "exact_zero_range_calls": audit.counts["exact_zero_range"],
            "nonfinite_values": audit.counts["nonfinite_values"],
            "by_kind": {key: dict(value) for key, value in audit.by_kind.items()},
            "by_system": {key: dict(value) for key, value in audit.by_system.items()},
            "by_dimension": {str(key): dict(value) for key, value in audit.by_dimension.items()},
            "ranges": {key: {"minimum": value[0], "maximum": value[1]}
                       for key, value in audit.ranges.items()},
        },
        "replay": replay_result,
        "earliest_unresolved": None if exact else (
            "finite-domain division rounding in (values-minimum)/(maximum-minimum)"
            if replay_result["division_bit_errors"] else "guard comparison or preceding arithmetic"
        ),
        "dimension_system_behavior": "one shared implementation for SHO, bouncing ball, and Kepler; d affects only vector contents/lengths, not normalization semantics",
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
