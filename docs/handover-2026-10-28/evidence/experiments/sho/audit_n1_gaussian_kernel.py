"""Audit only the active N1 Gaussian distance-to-weight calculation."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.geometry.train_division_rounding_router import (
    DivisionAssembly,
    RoundingRouter,
    ROOT,
)
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = ROOT / "n1_gaussian_kernel_audit"
ROUTER = ROOT / "division_rounding_router_training/rounding_router_before_evaluation.pt"
DEPENDENCIES = {
    "division_router": ROUTER,
    "division_approximation": ROOT / "division_operator_refinement/division_refined_best.pt",
    "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
    "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
    "add": ROOT / "addition_operator_smoke/addition_before_evaluation.pt",
    "sqrt": ROOT / "sqrt_operator_failure_refine/sqrt_refined_before_evaluation.pt",
}
FLOOR = np.float64(1.0e-12)
EXPONENT_SCALE = np.float64(-9.0)
WEIGHT_FLOOR = np.float64(1.0e-15)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, np.float64).view(np.uint64) == np.asarray(right, np.float64).view(np.uint64)


class Audit:
    def __init__(self, division: DivisionAssembly):
        self.division = division
        self.total = Counter()
        self.by_dimension = defaultdict(Counter)
        self.ranges = defaultdict(lambda: [np.inf, -np.inf])

    def record_range(self, name: str, values: np.ndarray):
        self.ranges[name][0] = min(self.ranges[name][0], float(np.min(values)))
        self.ranges[name][1] = max(self.ranges[name][1], float(np.max(values)))

    def consume(self, distances: np.ndarray, dimension: int):
        distances = np.asarray(distances, dtype=np.float64)
        assert distances.ndim == 2 and distances.shape[1] >= 1
        rows, columns = distances.shape
        batch_rows = max(1, 500_000 // columns)
        for start in range(0, rows, batch_rows):
            d = np.ascontiguousarray(distances[start:start + batch_rows])
            last = d[:, -1:]
            bandwidth = np.maximum(last, FLOOR)
            # Existing clamp assembly: locked subtraction/comparison plus
            # structural selection. Equality selects last; both bits coincide.
            margin = self.division.residual._call(self.division.residual.subtract, last, FLOOR)
            assembled_bandwidth = np.where(margin < 0.0, FLOOR, last)
            expanded_bandwidth = np.broadcast_to(assembled_bandwidth, d.shape)
            pairs = np.column_stack((d.ravel(), expanded_bandwidth.ravel()))
            ratio, _, _ = self.division.predict(pairs)
            teacher_ratio = np.divide(d, bandwidth)
            ratio = ratio.reshape(d.shape)
            square = self.division.residual._call(self.division.residual.multiply, ratio, ratio)
            teacher_square = np.square(teacher_ratio)
            scaled = self.division.residual._call(
                self.division.residual.multiply, np.full_like(square, EXPONENT_SCALE), square
            )
            teacher_scaled = np.multiply(EXPONENT_SCALE, teacher_square)

            values = d.size
            local = Counter(
                rows=len(d), values=values,
                bandwidth_clamps=int(np.count_nonzero(last < FLOOR)),
                bandwidth_equal_floor=int(np.count_nonzero(last == FLOOR)),
                bandwidth_bit_errors=int(np.count_nonzero(~bits_equal(assembled_bandwidth, bandwidth))),
                division_bit_errors=int(np.count_nonzero(~bits_equal(ratio, teacher_ratio))),
                square_bit_errors=int(np.count_nonzero(~bits_equal(square, teacher_square))),
                scale_bit_errors=int(np.count_nonzero(~bits_equal(scaled, teacher_scaled))),
                scale_numeric_errors=int(np.count_nonzero(scaled != teacher_scaled)),
                scale_signed_zero_mismatches=int(np.count_nonzero(
                    (scaled == 0.0) & (teacher_scaled == 0.0) &
                    (np.signbit(scaled) != np.signbit(teacher_scaled))
                )),
                zero_distances=int(np.count_nonzero(d == 0.0)),
                zero_ratios=int(np.count_nonzero(teacher_ratio == 0.0)),
                nonfinite_prefix=int(np.count_nonzero(
                    ~np.isfinite(bandwidth) | ~np.isfinite(ratio) |
                    ~np.isfinite(square) | ~np.isfinite(scaled)
                )),
            )
            self.total.update(local)
            self.by_dimension[dimension].update(local)
            for name, array in (
                ("distance", d), ("bandwidth", bandwidth), ("ratio", teacher_ratio),
                ("square", teacher_square), ("exp_input", teacher_scaled),
                ("exp_output", np.exp(teacher_scaled)),
                ("weight", np.exp(teacher_scaled) + WEIGHT_FLOOR),
            ):
                self.record_range(name, array)


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    before = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    router = RoundingRouter()
    router.load_state_dict(torch.load(ROUTER, weights_only=False)["state_dict"])
    division = DivisionAssembly(router)
    audit = Audit(division)
    states = corpus()
    source = Counter()

    for record_index, record in enumerate(states):
        points = record["points"]
        dimension = points.shape[1]
        count = min(25, len(points))
        distances, indices = cKDTree(points).query(points, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        loo = []
        for row in range(len(points)):
            keep = indices[row] != row
            local = distances[row, keep][:24]
            if len(local):
                loo.append(local)
        if loo:
            # Saved active states all have enough sites for 24 LOO neighbors.
            lengths = {len(row) for row in loo}
            for length in lengths:
                batch = np.stack([row for row in loo if len(row) == length])
                audit.consume(batch, dimension)
                source["loo_rows"] += len(batch)
                source["loo_values"] += batch.size

        path = record["continuous_path"]
        if path.exists():
            groups = defaultdict(list)
            for trace in json.loads(path.read_text()):
                n = int(trace["site_count"])
                queries = np.asarray([entry["point"] for entry in trace["starts"]], dtype=np.float64)
                if not len(queries):
                    continue
                k = min(24, n)
                distances = cKDTree(points[:n]).query(queries, k=k)[0]
                if k == 1:
                    distances = distances[:, None]
                groups[k].append(distances)
            for k, batches in groups.items():
                batch = np.concatenate(batches)
                audit.consume(batch, dimension)
                source["predict_rows"] += len(batch)
                source["predict_values"] += batch.size
        if record_index % 32 == 0:
            print(json.dumps(dict(record=record_index, values=audit.total["values"],
                                  prefix_errors=sum(audit.total[key] for key in
                                      ("bandwidth_bit_errors", "division_bit_errors",
                                       "square_bit_errors", "scale_bit_errors")))), flush=True)

    after = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    prefix_errors = {key: audit.total[key] for key in (
        "bandwidth_bit_errors", "division_bit_errors", "square_bit_errors",
        "scale_bit_errors", "scale_numeric_errors", "scale_signed_zero_mismatches",
        "nonfinite_prefix",
    )}
    exact_prefix = all(value == 0 for value in prefix_errors.values())
    result = dict(
        component="active_n1_gaussian_kernel",
        status="AUDITED_FIRST_UNRESOLVED_IS_EXP" if exact_prefix else "PREFIX_MISMATCH",
        production=dict(
            input="ordered neighbor distances per query row",
            output="one float64 Gaussian weight per neighbor distance",
            predict_neighbor_count="min(24, number_of_sites)",
            loo_neighbor_count="min(25, number_of_sites), remove queried site, retain first 24; <=2 sites bypasses kernel and returns LOO error 1",
            operation_order=[
                "bandwidth = np.maximum(distances[:, -1:], np.float64(1e-12))",
                "ratio = np.divide(distances, bandwidth)",
                "square = np.square(ratio)",
                "scaled = np.multiply(np.float64(-9.0), square)",
                "exponential = np.exp(scaled)",
                "weight = np.add(exponential, np.float64(1e-15))",
            ],
            formula="weight_i = exp(-9 * (distance_i / max(last_distance, 1e-12))^2) + 1e-15",
        ),
        classification=[
            dict(stage="last-neighbor bandwidth indexing/broadcast", classification="C", reason="deterministic array bookkeeping"),
            dict(stage="max(last_distance, 1e-12)", classification="B", reason="locked subtraction/comparison plus structural select"),
            dict(stage="distance / bandwidth", classification="B", reason="locked finite-domain division assembly"),
            dict(stage="ratio squared", classification="A", reason="existing learned multiplication primitive"),
            dict(stage="multiply square by -9.0", classification="A", reason="existing learned multiplication primitive"),
            dict(stage="np.exp(scaled)", classification="D", reason="no recovered scalar production-exp value operator exists"),
            dict(stage="add 1e-15", classification="A", reason="existing learned addition primitive; replay blocked by unresolved exp input"),
        ],
        corpus=dict(states=len(states), **source,
                    dimensions={str(d): dict(counts) for d, counts in sorted(audit.by_dimension.items())}),
        ranges={name: dict(minimum=bounds[0], maximum=bounds[1])
                for name, bounds in audit.ranges.items()},
        branch_counts=dict(
            bandwidth_clamps=audit.total["bandwidth_clamps"],
            bandwidth_equal_floor=audit.total["bandwidth_equal_floor"],
            zero_distances=audit.total["zero_distances"],
            zero_ratios=audit.total["zero_ratios"],
        ),
        replay=dict(values=audit.total["values"], exact_through_scaled_exp_input=exact_prefix,
                    **prefix_errors, exp_replayed_by_learned_assembly=False,
                    final_weight_replayed_by_learned_assembly=False),
        earliest_unresolved=dict(
            operation="scalar production np.exp(z)",
            exact_map="z -> np.exp(z)",
            actual_input_interval=audit.ranges["exp_input"],
            note="The later +1e-15 addition is already learned but cannot be composed exactly until exp is recovered.",
        ),
        dimension_behavior="Identical scalar/broadcast formula in d=4 and d=5; dimension affects only upstream distances.",
        dependencies={name: dict(path=str(path), sha256=before[name], unchanged=before[name] == after[name])
                      for name, path in DEPENDENCIES.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not exact_prefix or before != after:
        raise SystemExit("Gaussian prefix audit failed")


if __name__ == "__main__":
    run()
