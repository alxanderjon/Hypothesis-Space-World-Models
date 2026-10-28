"""Recover only the scalar exponential used by the active N1 Gaussian kernel.

The candidate deliberately does not call exp.  It uses binary range reduction,
an error-free product/sum expansion, and a degree-24 Taylor polynomial evaluated
as a double-double expansion.  This is the smallest zero-parameter composition
supported by the already validated scalar arithmetic/product-tail machinery.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.geometry import train_division_operator as arithmetic_source
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_division_rounding_router import ROOT
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = ROOT / "n1_scalar_exp_recovery"
WEIGHT_FLOOR = np.float64(1.0e-15)
SPLITTER = np.float64(134217729.0)
mp.mp.dps = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_checkpoints() -> dict[str, Path]:
    return {
        "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
        "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
        "add": ROOT / "addition_operator_smoke/addition_before_evaluation.pt",
        "division_router": ROOT / "division_rounding_router_training/rounding_router_before_evaluation.pt",
        "division_approximation": ROOT / "division_operator_refinement/division_refined_best.pt",
        "sqrt": ROOT / "n1_sqrt_assembly/n1_sqrt_locked_assembly.json",
        "product_tail": ROOT / "product_tail_primitive_screen/results.json",
    }


def two_prod(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    product = a * b
    ca = SPLITTER * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = SPLITTER * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    tail = (((a_hi * b_hi - product) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo
    return product, tail


def two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = a + b
    b_virtual = total - a
    tail = (a - (total - b_virtual)) + (b - b_virtual)
    return total, tail


def quick_two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = a + b
    return total, b - (total - a)


def dd_add(
    a_hi: np.ndarray, a_lo: np.ndarray, b_hi: np.ndarray, b_lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    high, tail = two_sum(a_hi, b_hi)
    tail = tail + (a_lo + b_lo)
    return quick_two_sum(high, tail)


def dd_multiply(
    a_hi: np.ndarray, a_lo: np.ndarray, b_hi: np.ndarray, b_lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    high, tail = two_prod(a_hi, b_hi)
    tail = tail + a_hi * b_lo + a_lo * b_hi + a_lo * b_lo
    return quick_two_sum(high, tail)


def split_constant(value: mp.mpf) -> tuple[np.float64, np.float64]:
    high = np.float64(str(value))
    low = np.float64(str(value - mp.mpf(float(high))))
    return high, low


LN2_HIGH, LN2_LOW = split_constant(mp.log(2))
INV_LN2 = np.float64(str(1 / mp.log(2)))
COEFFICIENTS = tuple(split_constant(1 / mp.factorial(n)) for n in range(25))


def assembled_exp(
    values: np.ndarray, *, state: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a correctly-rounded exp candidate without calling exp."""
    x = np.asarray(values, dtype=np.float64)
    k = np.rint(x * INV_LN2).astype(np.int64)
    k_float = k.astype(np.float64)

    product_hi, product_lo = two_prod(k_float, np.full_like(x, LN2_HIGH))
    product_lo = product_lo + k_float * LN2_LOW
    product_hi, product_lo = quick_two_sum(product_hi, product_lo)
    reduced_hi, reduced_lo = two_sum(x, -product_hi)
    reduced_lo = reduced_lo - product_lo
    reduced_hi, reduced_lo = quick_two_sum(reduced_hi, reduced_lo)

    high = np.full_like(x, COEFFICIENTS[-1][0])
    low = np.full_like(x, COEFFICIENTS[-1][1])
    for coefficient_hi, coefficient_lo in COEFFICIENTS[-2::-1]:
        high, low = dd_multiply(high, low, reduced_hi, reduced_lo)
        high, low = dd_add(
            high, low,
            np.full_like(x, coefficient_hi), np.full_like(x, coefficient_lo),
        )
    high = np.ldexp(high, k)
    low = np.ldexp(low, k)
    rounded, tail = two_sum(high, low)
    if state:
        return rounded, tail, k, reduced_hi + reduced_lo
    return rounded


def signed_ulp_offset(candidate: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    # All required values are positive normal float64, so ordered uint64 bit
    # patterns are also ordered integer values.
    return teacher.view(np.uint64).astype(np.int64) - candidate.view(np.uint64).astype(np.int64)


def distance_batches():
    states = corpus()
    for record_index, record in enumerate(states):
        points = record["points"]
        dimension = points.shape[1]
        count = min(25, len(points))
        distances, indices = cKDTree(points).query(points, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        groups: dict[int, list[np.ndarray]] = defaultdict(list)
        for row in range(len(points)):
            local = distances[row, indices[row] != row][:24]
            if len(local):
                groups[len(local)].append(local)
        for rows in groups.values():
            yield record_index, dimension, "loo", np.stack(rows)

        path = record["continuous_path"]
        if path.exists():
            groups2: dict[int, list[np.ndarray]] = defaultdict(list)
            for trace in json.loads(path.read_text()):
                n = int(trace["site_count"])
                queries = np.asarray([entry["point"] for entry in trace["starts"]], np.float64)
                if not len(queries):
                    continue
                neighbor_count = min(24, n)
                batch = cKDTree(points[:n]).query(queries, k=neighbor_count)[0]
                if neighbor_count == 1:
                    batch = batch[:, None]
                groups2[neighbor_count].append(batch)
            for rows in groups2.values():
                yield record_index, dimension, "predict", np.concatenate(rows)


class Audit:
    def __init__(self, learned_add: Addition):
        self.learned_add = learned_add
        self.counts = Counter()
        self.by_dimension = defaultdict(Counter)
        self.offsets = Counter()
        self.weight_offsets = Counter()
        self.by_k = defaultdict(Counter)
        self.mismatch_inputs: list[float] = []
        self.minimum = math.inf
        self.maximum = -math.inf
        self.maximum_abs = 0.0
        self.maximum_weight_abs = 0.0
        self.gap_mismatch = [math.inf, -math.inf]
        self.gap_exact = [math.inf, -math.inf]

    def consume(self, distances: np.ndarray, dimension: int, source: str):
        distances = np.asarray(distances, np.float64)
        row_chunk = max(1, 500_000 // distances.shape[1])
        for start in range(0, len(distances), row_chunk):
            d = np.ascontiguousarray(distances[start:start + row_chunk])
            bandwidth = np.maximum(d[:, -1:], np.float64(1.0e-12))
            z = np.multiply(np.float64(-9.0), np.square(np.divide(d, bandwidth))).ravel()
            teacher = np.exp(z)
            candidate, tail, k, _ = assembled_exp(z, state=True)
            offsets = signed_ulp_offset(candidate, teacher)
            weight_teacher = np.add(teacher, WEIGHT_FLOOR)
            weight_candidate_np = np.add(candidate, WEIGHT_FLOOR)
            pairs = np.column_stack((candidate, np.full_like(candidate, WEIGHT_FLOOR)))
            with torch.no_grad():
                weight_candidate = self.learned_add(torch.from_numpy(pairs)).numpy()
            add_errors = weight_candidate.view(np.uint64) != weight_candidate_np.view(np.uint64)
            weight_offsets = signed_ulp_offset(weight_candidate, weight_teacher)
            mismatch = offsets != 0
            weight_mismatch = weight_offsets != 0

            ulp = np.nextafter(candidate, np.inf) - candidate
            midpoint_gap = np.float64(0.5) - np.abs(tail / ulp)
            exact = ~mismatch
            if mismatch.any():
                self.gap_mismatch[0] = min(self.gap_mismatch[0], float(midpoint_gap[mismatch].min()))
                self.gap_mismatch[1] = max(self.gap_mismatch[1], float(midpoint_gap[mismatch].max()))
                room = 4096 - len(self.mismatch_inputs)
                if room > 0:
                    self.mismatch_inputs.extend(z[mismatch][:room].tolist())
            if exact.any():
                self.gap_exact[0] = min(self.gap_exact[0], float(midpoint_gap[exact].min()))
                self.gap_exact[1] = max(self.gap_exact[1], float(midpoint_gap[exact].max()))

            local = Counter(
                values=z.size,
                scalar_exact=int(np.count_nonzero(~mismatch)),
                scalar_mismatches=int(np.count_nonzero(mismatch)),
                scalar_one_ulp=int(np.count_nonzero(np.abs(offsets) == 1)),
                scalar_larger=int(np.count_nonzero(np.abs(offsets) > 1)),
                weight_exact=int(np.count_nonzero(~weight_mismatch)),
                weight_mismatches=int(np.count_nonzero(weight_mismatch)),
                weight_one_ulp=int(np.count_nonzero(np.abs(weight_offsets) == 1)),
                weight_larger=int(np.count_nonzero(np.abs(weight_offsets) > 1)),
                learned_addition_bit_errors=int(np.count_nonzero(add_errors)),
                nonfinite_candidate=int(np.count_nonzero(~np.isfinite(candidate))),
                nonfinite_teacher=int(np.count_nonzero(~np.isfinite(teacher))),
                signed_zero_mismatches=int(np.count_nonzero(
                    (candidate == 0.0) & (teacher == 0.0) &
                    (np.signbit(candidate) != np.signbit(teacher))
                )),
            )
            self.counts.update(local)
            self.by_dimension[dimension].update(local)
            self.by_dimension[dimension][f"{source}_values"] += z.size
            self.offsets.update(map(int, offsets[mismatch]))
            self.weight_offsets.update(map(int, weight_offsets[weight_mismatch]))
            for kval in np.unique(k):
                selected = k == kval
                selected_mismatch = selected & mismatch
                self.by_k[int(kval)]["values"] += int(np.count_nonzero(selected))
                self.by_k[int(kval)]["mismatches"] += int(np.count_nonzero(selected_mismatch))
                self.by_k[int(kval)]["minus_one"] += int(np.count_nonzero(offsets[selected] == -1))
                self.by_k[int(kval)]["plus_one"] += int(np.count_nonzero(offsets[selected] == 1))
            self.minimum = min(self.minimum, float(z.min()))
            self.maximum = max(self.maximum, float(z.max()))
            self.maximum_abs = max(self.maximum_abs, float(np.max(np.abs(candidate - teacher))))
            self.maximum_weight_abs = max(
                self.maximum_weight_abs, float(np.max(np.abs(weight_candidate - weight_teacher)))
            )


def high_precision_check(values: list[float]) -> dict:
    candidate_exact = 0
    teacher_exact = 0
    for value in values:
        x = np.asarray([value], np.float64)
        candidate = assembled_exp(x)[0]
        teacher = np.exp(x)[0]
        reference = np.float64(mp.exp(mp.mpf(float(value))))
        candidate_exact += int(candidate.view(np.uint64) == reference.view(np.uint64))
        teacher_exact += int(teacher.view(np.uint64) == reference.view(np.uint64))
    return dict(
        values=len(values),
        candidate_matches_correctly_rounded_reference=candidate_exact,
        production_matches_correctly_rounded_reference=teacher_exact,
    )


def run() -> None:
    started = time.monotonic()
    protected = protected_checkpoints()
    before = {name: sha256(path) for name, path in protected.items()}
    torch.set_num_threads(1)
    learned_add, _ = arithmetic_source.load(Addition, "addition_operator_smoke")
    learned_add.eval().requires_grad_(False)
    audit = Audit(learned_add)
    last_record = -1
    for record_index, dimension, source, distances in distance_batches():
        audit.consume(distances, dimension, source)
        if record_index != last_record and record_index % 32 == 0:
            print(json.dumps(dict(record=record_index, values=audit.counts["values"],
                                  mismatches=audit.counts["scalar_mismatches"])), flush=True)
        last_record = record_index
    after = {name: sha256(path) for name, path in protected.items()}

    high_precision = high_precision_check(audit.mismatch_inputs)
    exact = audit.counts["scalar_mismatches"] == 0
    result = dict(
        component="n1_scalar_exp",
        status="FROZEN_CANDIDATE" if exact else "NOT_PROMOTED_FIRST_UNRESOLVED_PRODUCTION_ROUNDING_KERNEL",
        production=dict(
            operation="np.exp(z)",
            numpy_version=np.__version__,
            implementation="NumPy float64 baseline loop calls platform C exp through npy_exp",
            actual_domain=[audit.minimum, audit.maximum],
        ),
        prior_inventory=dict(
            reusable_variable_input_exp_checkpoint=False,
            scalar_power_internal_exp="static gain exp(b), not a variable-input exponential",
            variable_power_adjoint_exp="production torch.exp/pow scaffold, not recovered exponential",
        ),
        candidate=dict(
            architecture="binary range reduction + degree-24 double-double Taylor/Horner + exact exponent scaling",
            learned_parameters=0,
            constants=dict(
                ln2_high=float(LN2_HIGH), ln2_low=float(LN2_LOW),
                inverse_ln2=float(INV_LN2), taylor_coefficients=len(COEFFICIENTS),
            ),
            reused=[
                "locked scalar add/subtract/multiply",
                "validated exact product-tail splitter assembly",
                "existing comparison/routing semantics",
                "deterministic rint and binary exponent packing",
            ],
            does_not_call_exp=True,
        ),
        scalar_replay=dict(
            **dict(audit.counts),
            signed_ulp_offsets={str(key): value for key, value in sorted(audit.offsets.items())},
            maximum_absolute_error=audit.maximum_abs,
            exact_fraction=audit.counts["scalar_exact"] / audit.counts["values"],
            by_dimension={str(key): dict(value) for key, value in sorted(audit.by_dimension.items())},
        ),
        gaussian_weight_replay=dict(
            values=audit.counts["values"],
            exact=audit.counts["weight_exact"],
            mismatches=audit.counts["weight_mismatches"],
            one_ulp=audit.counts["weight_one_ulp"],
            larger=audit.counts["weight_larger"],
            learned_addition_bit_errors=audit.counts["learned_addition_bit_errors"],
            signed_ulp_offsets={str(key): value for key, value in sorted(audit.weight_offsets.items())},
            maximum_absolute_error=audit.maximum_weight_abs,
            operation="locked learned add(candidate_exp, np.float64(1e-15)); teacher is np.add",
        ),
        production_rounding_diagnosis=dict(
            high_precision_mismatch_sample=high_precision,
            correction_by_binary_range={str(key): dict(value) for key, value in sorted(audit.by_k.items())},
            candidate_midpoint_gap_on_mismatches=audit.gap_mismatch,
            candidate_midpoint_gap_on_exact_values=audit.gap_exact,
            simple_midpoint_threshold_separates_corrections=(
                audit.gap_mismatch[1] < audit.gap_exact[0]
            ),
            note=(
                "The zero-parameter assembly matches the correctly rounded mathematical exp on the "
                "high-precision mismatch sample.  Production C exp instead returns the adjacent float. "
                "The overlapping midpoint-gap ranges prevent an exact single-threshold correction."
            ),
        ),
        downstream=dict(
            kernel_weight_differences=audit.counts["weight_mismatches"],
            normalization_or_loo_replayed=False,
            reason="Stopped at the scalar exponential gate as requested; no downstream stage was modified.",
        ),
        earliest_unresolved=dict(
            operation="platform-production exp approximation/rounding correction",
            exact_map="(binary range k, reduced argument, polynomial/product-tail state) -> signed adjacent-ULP correction",
            observed_offsets=sorted(audit.offsets),
            reason="The reusable mathematical exp assembly is not bit-exact to production np.exp.",
        ),
        dependencies={name: dict(path=str(path), sha256=before[name], unchanged=before[name] == after[name])
                      for name, path in protected.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if before != after:
        raise SystemExit("protected checkpoint changed")
    if audit.counts["values"] != 15_055_875:
        raise SystemExit(f"unexpected corpus size: {audit.counts['values']}")


if __name__ == "__main__":
    run()
