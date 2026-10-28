"""Validate only the scalar sqrt correction on saved active N1 distances."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.geometry.n1_sqrt_rounding_assembly import (
    N1CorrectlyRoundedSqrt,
    ROOT,
    SQRT_CHECKPOINT,
)
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = ROOT / "n1_sqrt_assembly"
DEPENDENCIES = {
    "sqrt_power": SQRT_CHECKPOINT,
    "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
    "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
    "add": ROOT / "addition_operator_smoke/addition_before_evaluation.pt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = bits >> np.uint64(63)
    return np.where(sign != 0, ~bits, bits ^ (np.uint64(1) << np.uint64(63)))


def ulps(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = ordered_bits(left), ordered_bits(right)
    return np.maximum(a, b) - np.minimum(a, b)


def native_product_pair(a: np.ndarray, b: np.ndarray):
    splitter = np.float64(134217729.0)
    product = a * b
    ca = splitter * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = splitter * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    tail = (((a_hi * b_hi - product) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo
    return product, tail


class Audit:
    def __init__(self, assembly: N1CorrectlyRoundedSqrt):
        self.assembly = assembly
        self.counts = Counter()
        self.by_source = defaultdict(Counter)
        self.minimum = np.inf
        self.maximum = -np.inf
        self.min_positive = np.inf
        self.dimension_counts = Counter()

    def consume(self, squared: np.ndarray, production: np.ndarray, source: str,
                shape: tuple[int, int], dimension: int):
        squared = np.ascontiguousarray(squared, dtype=np.float64)
        production = np.ascontiguousarray(production, dtype=np.float64)
        rows, columns = shape
        assert squared.size == production.size == rows * columns
        rows_per_batch = max(1, 500_000 // columns)
        for row_begin in range(0, rows, rows_per_batch):
            row_end = min(rows, row_begin + rows_per_batch)
            begin, end = row_begin * columns, row_end * columns
            x = squared[begin:end]
            teacher = np.sqrt(x)
            actual = production[begin:end]
            result, trace = self.assembly(x, return_trace=True)
            baseline = trace["estimate"]
            baseline_ulp = ulps(baseline, teacher)
            result_ulp = ulps(result, teacher)
            distance_ulp = ulps(result, actual)
            n = len(x)
            local = Counter(
                scalar_values=n,
                baseline_exact=int(np.count_nonzero(baseline_ulp == 0)),
                baseline_one_ulp=int(np.count_nonzero(baseline_ulp == 1)),
                baseline_larger=int(np.count_nonzero(baseline_ulp > 1)),
                scalar_exact=int(np.count_nonzero(result_ulp == 0)),
                scalar_one_ulp=int(np.count_nonzero(result_ulp == 1)),
                scalar_larger=int(np.count_nonzero(result_ulp > 1)),
                distance_exact=int(np.count_nonzero(distance_ulp == 0)),
                distance_one_ulp=int(np.count_nonzero(distance_ulp == 1)),
                distance_larger=int(np.count_nonzero(distance_ulp > 1)),
                corrected_up=int(np.count_nonzero(result.view(np.uint64) > baseline.view(np.uint64))),
                corrected_down=int(np.count_nonzero(result.view(np.uint64) < baseline.view(np.uint64))),
                zero_inputs=int(np.count_nonzero(x == 0.0)),
                nonfinite=int(np.count_nonzero(~np.isfinite(result))),
                signed_zero_mismatch=int(np.count_nonzero(
                    (result == 0.0) & (teacher == 0.0) &
                    (np.signbit(result) != np.signbit(teacher))
                )),
            )
            # Verify the reused learned product-tail interface against its
            # native ordered graph at all three products used by the repair.
            for prefix, left, right, hi_name, lo_name in (
                ("estimate", baseline, baseline, "estimate_product_hi", "estimate_product_lo"),
                ("lower_square", trace["lower"], trace["lower"], "square_hi", "square_lo"),
                ("midpoint_cross", trace["lower"], trace["spacing"], "cross_hi", "cross_lo"),
            ):
                hi, lo = native_product_pair(left, right)
                local[prefix + "_product_hi_errors"] += int(np.count_nonzero(
                    hi.view(np.uint64) != trace[hi_name].view(np.uint64)))
                local[prefix + "_product_lo_errors"] += int(np.count_nonzero(
                    lo.view(np.uint64) != trace[lo_name].view(np.uint64)))
            self.counts.update(local)
            self.by_source[source].update(local)
            self.minimum = min(self.minimum, float(np.min(x)))
            self.maximum = max(self.maximum, float(np.max(x)))
            positive = x[x > 0.0]
            if len(positive):
                self.min_positive = min(self.min_positive, float(np.min(positive)))
            self.dimension_counts[dimension] += n
            if columns > 1:
                assembled = result.reshape(row_end - row_begin, columns)
                teacher_matrix = actual.reshape(row_end - row_begin, columns)
                strict = teacher_matrix[:, :-1] < teacher_matrix[:, 1:]
                self.counts["adjacent_pairs"] += int(strict.size)
                self.counts["strict_adjacent_pairs"] += int(np.count_nonzero(strict))
                self.counts["exact_tie_adjacent_pairs"] += int(np.count_nonzero(~strict))
                self.counts["strict_order_errors"] += int(np.count_nonzero(
                    strict & ~(assembled[:, :-1] < assembled[:, 1:])
                ))


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    before = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    records = corpus()
    assembly = N1CorrectlyRoundedSqrt()
    audit = Audit(assembly)
    query_counts = Counter()

    for record_index, record in enumerate(records):
        points = record["points"]
        dimension = points.shape[1]
        count = min(25, len(points))
        distances, indices = cKDTree(points).query(points, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        squared_rows, production_rows = [], []
        for row, site in enumerate(indices):
            keep = site != row
            selected = site[keep][:24]
            selected_distances = distances[row, keep][:24]
            delta = points[selected] - points[row]
            squared_rows.append(np.sum(delta * delta, axis=1))
            production_rows.append(selected_distances)
        if squared_rows:
            squared = np.concatenate(squared_rows)
            production = np.concatenate(production_rows)
            columns = len(squared_rows[0])
            audit.consume(squared, production, "loo", (len(squared_rows), columns), dimension)
            query_counts["loo_queries"] += len(squared_rows)

        path = record["continuous_path"]
        if path.exists():
            grouped_squared = defaultdict(list)
            grouped_production = defaultdict(list)
            for trace in json.loads(path.read_text()):
                site_count = int(trace["site_count"])
                sites = points[:site_count]
                queries = np.asarray([entry["point"] for entry in trace["starts"]], dtype=np.float64)
                if not len(queries):
                    continue
                neighbor_count = min(24, site_count)
                production, selected = cKDTree(sites).query(queries, k=neighbor_count)
                if neighbor_count == 1:
                    production, selected = production[:, None], selected[:, None]
                delta = queries[:, None, :] - sites[selected]
                squared = np.sum(delta * delta, axis=2)
                grouped_squared[neighbor_count].append(squared)
                grouped_production[neighbor_count].append(production)
                query_counts["continuous_endpoint_queries"] += len(queries)
            for neighbor_count in sorted(grouped_squared):
                squared = np.concatenate(grouped_squared[neighbor_count])
                production = np.concatenate(grouped_production[neighbor_count])
                audit.consume(squared.ravel(), production.ravel(), "continuous_endpoint",
                              squared.shape, dimension)
        if record_index % 32 == 0:
            print(json.dumps(dict(record=record_index, scalar_values=audit.counts["scalar_values"],
                                  scalar_errors=audit.counts["scalar_one_ulp"] + audit.counts["scalar_larger"])),
                  flush=True)

    after = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    exact = bool(
        audit.counts["scalar_exact"] == audit.counts["scalar_values"] and
        audit.counts["distance_exact"] == audit.counts["scalar_values"] and
        audit.counts["strict_order_errors"] == 0 and
        audit.counts["nonfinite"] == 0 and
        audit.counts["signed_zero_mismatch"] == 0 and
        all(value == 0 for key, value in audit.counts.items()
            if key.endswith("_product_hi_errors") or key.endswith("_product_lo_errors")) and
        before == after
    )
    result = dict(
        component="n1_sqrt_rounding",
        status="LOCKED_ASSEMBLY" if exact else "DIAGNOSTIC_ONLY",
        passed=exact,
        production_operation="np.sqrt(nonnegative squared Euclidean distance)",
        previous_candidate=dict(
            form="exp(b) * torch.pow(x, p)", parameters=2,
            checkpoint=str(SQRT_CHECKPOINT), sha256=before["sqrt_power"],
            cause="p is exactly 0.5 and exp(b) rounds to 1.0; residual errors are torch.pow kernel last-bit rounding",
        ),
        correction=dict(
            new_parameters=0,
            rule=[
                "use exact product expansion to bracket sqrt(x) with adjacent floats lower, upper",
                "compare exact-expanded x-lower^2 with lower*ulp + ulp^2/4",
                "select upper only when the first quantity is strictly larger",
            ],
            deterministic_scaffolding=["np.nextafter", "finite/nonnegative contract check", "comparisons", "graph wiring"],
            reused=["sqrt ScalarPower", "exact product-tail assembly", "locked multiply", "locked subtract", "locked add", "locked comparison semantics"],
        ),
        corpus=dict(
            states=len(records), dimensions={str(k): v for k, v in sorted(audit.dimension_counts.items())},
            squared_values=audit.counts["scalar_values"],
            minimum=audit.minimum, minimum_positive=audit.min_positive, maximum=audit.maximum,
            zero_inputs=audit.counts["zero_inputs"],
            **query_counts,
        ),
        baseline={
            "exact": audit.counts["baseline_exact"], "one_ulp": audit.counts["baseline_one_ulp"],
            "larger": audit.counts["baseline_larger"],
            "loo_one_ulp": audit.by_source["loo"]["baseline_one_ulp"],
            "continuous_endpoint_one_ulp": audit.by_source["continuous_endpoint"]["baseline_one_ulp"],
        },
        assembly={
            "scalar_bit_exact": audit.counts["scalar_exact"],
            "scalar_one_ulp_errors": audit.counts["scalar_one_ulp"],
            "scalar_larger_errors": audit.counts["scalar_larger"],
            "distance_bit_exact": audit.counts["distance_exact"],
            "distance_one_ulp_errors": audit.counts["distance_one_ulp"],
            "distance_larger_errors": audit.counts["distance_larger"],
            "corrected_up": audit.counts["corrected_up"],
            "corrected_down": audit.counts["corrected_down"],
            "signed_zero_mismatches": audit.counts["signed_zero_mismatch"],
            "nonfinite": audit.counts["nonfinite"],
            "product_expansion_errors": sum(value for key, value in audit.counts.items()
                                              if key.endswith("_product_hi_errors") or key.endswith("_product_lo_errors")),
        },
        ordering={
            "adjacent_pairs": audit.counts["adjacent_pairs"],
            "strict_pairs": audit.counts["strict_adjacent_pairs"],
            "exact_tie_pairs_deferred": audit.counts["exact_tie_adjacent_pairs"],
            "strict_order_errors": audit.counts["strict_order_errors"],
            "note":"KD-tree exact-tie identity/order remains outside this operation",
        },
        dependencies={name: dict(path=str(path), sha256=before[name], unchanged=before[name] == after[name])
                      for name, path in DEPENDENCIES.items()},
        parameter_count=dict(existing_sqrt=2, newly_added=0),
        generic_distance_status=("EXACT_LEARNED_ASSEMBLY_ON_ACTIVE_DOMAIN" if exact else "UNRESOLVED"),
        dimension_note="The scalar repair has no coordinate or dimension input; subtract/square/repeated-add remain the existing arbitrary-d reduction.",
        next_unresolved=("cKDTree exact-tie ordering (explicitly deferred)" if exact else
                         "adjacent-float sqrt midpoint decision"),
        elapsed_seconds=time.monotonic() - started,
    )
    print(json.dumps(result, indent=2), flush=True)
    if exact:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "n1_sqrt_locked_assembly.json").write_text(json.dumps(result, indent=2) + "\n")
        (OUT / "n1_sqrt_locked_assembly.sha256").write_text(
            sha256(OUT / "n1_sqrt_locked_assembly.json") + "\n"
        )
    else:
        raise SystemExit("sqrt assembly exactness gate failed")


if __name__ == "__main__":
    run()
