#!/usr/bin/env python3
"""Resume the fail-closed hybrid-student replay after cKDTree distance repair.

This is deliberately a composition test, not another component audit.  The
repaired distance and every already-locked operation are invoked in frozen
teacher order.  Exact cKDTree minimum ties and the other inventory-D entries
remain explicit deterministic oracles.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import _sobol
from scipy._lib._util import rng_integers
import torch

from experiments.build_hybrid_student_baseline import representative_records, metric
from experiments.geometry.ckdtree_distance_assembly import CKDTreeEuclideanDistance
from experiments.geometry.n1_sqrt_rounding_assembly import N1CorrectlyRoundedSqrt
from experiments.geometry.n3_gradient_binding_assembly import (
    N3GradientBindingAssembly,
)
from experiments.geometry.n3_hessian_binding_assembly import (
    N3SymmetricHessianBindingAssembly,
)
from experiments.geometry.n3_eigenspectrum_condition_assembly import (
    N3EigenspectrumConditionAssembly,
)
from experiments.geometry.float64_unary_negation_assembly import (
    Float64UnaryNegationAssembly,
)
from experiments.geometry.prefix_residual_mse_assembly import (
    PrefixResidualMSEAssembly,
)
from experiments.resume_hybrid_student_at_curvature import load_locked_router
from experiments.sho import assemble_n3_gradient_operator as n3_arithmetic
from experiments.sho import run_stage1b_n3_gradient_polishing as n3_production
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.fast_nearest_argmin import FastNearestArgmin
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho import validate_stage1b_learned_cell_size as cells
from experiments.sho.validate_stage1b_distance_assembled import _learned_group_and_represent
from experiments.sho.run_stage1b_spacefill_generator_smoke import downstream
from experiments.sho.validate_largest_cell_router_tie_repair import load_model, repaired_choice
from experiments.sho.validate_cell_scoring_assemblies import ordinary_assembly
from experiments.sho.validate_sobol_batching_truncation import (
    LearnedSobol, schedule as sobol_schedule,
)
from experiments.sho.validate_shared_one_bit_xor import direction_column, update
from experiments.sho.validate_n1_boundary_merge_dedup import independent_unique
from experiments.sho.recover_n1_loss_encoding import assembled_clamp
from experiments.geometry.train_division_rounding_router import (
    DivisionAssembly, RoundingRouter,
)
from experiments.sho.validate_n1_continuous_powell_assembly import (
    NumpyDirectionController, SEARCH_SEED,
    compare_powell, powell_trace,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_after_ckdtree")
FREEZE = Path("outputs/deterministic_teacher_frozen")
INVENTORY = FREEZE / "operation_inventory.json"
CHECKPOINT_REFERENCE = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(a, b) -> bool:
    aa, bb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return aa.shape == bb.shape and bool(np.array_equal(aa.view(np.uint64), bb.view(np.uint64)))


class Log:
    def __init__(self):
        self.counts = Counter()
        self.implementation = {}

    def call(self, operation, category, implementation, count=1):
        self.counts[operation] += int(count)
        old = self.implementation.setdefault(operation, (category, implementation))
        if old != (category, implementation):
            raise RuntimeError(f"inconsistent implementation log for {operation}")


class StrictOwnerAssembly:
    """Exact distances plus learned strict comparisons; ties alone use cKDTree."""

    def __init__(self, frozen: FrozenProposals, log: Log):
        self.distance = CKDTreeEuclideanDistance()
        self.compare = FastNearestArgmin(None, frozen.margin, frozen.threshold, None)
        self.log = log

    def distance_matrix(self, query, sites):
        query = np.asarray(query, np.float64)
        sites = np.asarray(sites, np.float64)
        paired_q = np.repeat(query, len(sites), axis=0)
        paired_s = np.tile(sites, (len(query), 1))
        result = self.distance(paired_q, paired_s).reshape(len(query), len(sites))
        self.log.call("Euclidean distance", "A",
                      "CKDTreeEuclideanDistance exact FMA-lane assembly", result.size)
        return result

    def owners(self, query, sites):
        distances = self.distance_matrix(query, sites)
        teacher_two, teacher_index = cKDTree(sites).query(query, k=2)
        exact_tie = teacher_two[:, 0] == teacher_two[:, 1]
        owner = np.empty(len(query), np.int64)
        minimum = np.empty(len(query), np.float64)
        # The only ownership oracle permitted by this stage.
        if np.any(exact_tie):
            oracle_distance, oracle_owner = cKDTree(sites).query(query[exact_tie], k=1)
            owner[exact_tie] = oracle_owner
            minimum[exact_tie] = oracle_distance
            self.log.call("production cKDTree exact-tie ownership", "D",
                          "temporary cKDTree exact-minimum-tie oracle", np.count_nonzero(exact_tie))
        active = np.flatnonzero(~exact_tie)
        current_owner = np.zeros(len(active), np.int64)
        current = distances[active, 0]
        comparisons = errors = 0
        for candidate_index in range(1, len(sites)):
            candidate = distances[active, candidate_index]
            unequal = candidate != current
            replace = np.zeros(len(active), bool)
            if np.any(unequal):
                replace[unequal] = self.compare.logits(
                    current[unequal], candidate[unequal]) > 0
                errors += int(np.count_nonzero(
                    replace[unequal] != (candidate[unequal] < current[unequal])))
                comparisons += int(np.count_nonzero(unequal))
            current = np.where(replace, candidate, current)
            current_owner = np.where(replace, candidate_index, current_owner)
        owner[active] = current_owner
        minimum[active] = current
        self.log.call("nearest-owner routing away from exact ties", "B",
                      "locked subtraction/sign/comparison + FastNearestArgmin route",
                      comparisons)
        return owner, minimum, {
            "strict": int(len(active)), "ties": int(np.count_nonzero(exact_tie)),
            "comparison_errors": errors,
            "owner_errors": int(np.count_nonzero(owner != teacher_index[:, 0])),
            "distance_errors": int(np.count_nonzero(minimum != teacher_two[:, 0])),
        }

    def spacing(self, sites):
        matrix = self.distance_matrix(sites, sites)
        np.fill_diagonal(matrix, np.inf)
        reference, indices = cKDTree(sites).query(sites, k=3)
        tied = reference[:, 1] == reference[:, 2]
        output = np.empty(len(sites), np.float64)
        if np.any(tied):
            output[tied] = cKDTree(sites).query(sites[tied], k=2)[0][:, 1]
            self.log.call("production cKDTree exact-tie ownership", "D",
                          "temporary cKDTree exact-minimum-tie oracle", np.count_nonzero(tied))
        active = np.flatnonzero(~tied)
        first = np.where(active == 0, 1, 0)
        current = matrix[active, first]
        comparisons = errors = 0
        for candidate_index in range(len(sites)):
            candidate = matrix[active, candidate_index]
            unequal = np.isfinite(candidate) & (candidate != current)
            replace = np.zeros(len(active), bool)
            if np.any(unequal):
                replace[unequal] = self.compare.logits(
                    current[unequal], candidate[unequal]) > 0
                errors += int(np.count_nonzero(
                    replace[unequal] != (candidate[unequal] < current[unequal])))
                comparisons += int(np.count_nonzero(unequal))
            current = np.where(replace, candidate, current)
        output[active] = current
        self.log.call("nearest-other-site spacing", "B",
                      "exact distance + locked repeated strict-min route", comparisons)
        return output, {
            "strict": int(len(active)), "ties": int(np.count_nonzero(tied)),
            "comparison_errors": errors,
            "spacing_errors": int(np.count_nonzero(output != reference[:, 1])),
        }


_SOBOL_ASSEMBLY = None
_GAUSSIAN_DIVISION_ASSEMBLY = None
_FIRST_MAX_CONTROLLER = None
_NUMPY_EUCLIDEAN_SQRT = None
GAUSSIAN_BANDWIDTH_FLOOR = np.float64(1.0e-12)
FLOAT64_EXPONENT_MASK = np.uint64(0x7ff0_0000_0000_0000)


def scrambled_sobol_full_block(dimension, exponent, seed, bits=30):
    """Locked scrambled-Sobol assembly without a production Sobol engine.

    Direction-table initialization and RNG/state storage are deterministic
    scaffolding.  Scramble transform, shift packing, digital recurrence/XOR,
    and float64 scaling are the previously locked learned assemblies.
    """
    global _SOBOL_ASSEMBLY
    if _SOBOL_ASSEMBLY is None:
        _SOBOL_ASSEMBLY = LearnedSobol()
    assembly = _SOBOL_ASSEMBLY
    dtype = np.uint32 if bits <= 32 else np.uint64
    directions = np.zeros((dimension, bits), dtype=dtype)
    _sobol._initialize_v(directions, dim=dimension, bits=bits)

    # SciPy 1.13.0's transition-to-rng wrapper maps active integer seeds to a
    # fresh PCG64 Generator.  Draw order matches Sobol._scramble exactly:
    # digital shift first, then the LMS matrices.
    rng = np.random.default_rng(seed)
    raw_shift = rng_integers(
        rng, 2, size=(dimension, bits), dtype=dtype,
    )
    raw_matrix = rng_integers(
        rng, 2, size=(dimension, bits, bits), dtype=dtype,
    )
    scrambled_directions, _ = assembly.scramble.transform(
        directions, raw_matrix,
    )
    state = assembly.scramble.shift(raw_shift).astype(np.uint64)
    initial_state = state.copy()
    scale = np.float64(1.0 / 2 ** bits)
    generated = 2 ** exponent
    points = np.empty((generated, dimension), np.float64)
    points[0] = assembly.scaling(state, scale)
    for index in range(generated - 1):
        column = direction_column(index, bits, assembly.w, assembly.b)
        state = update(
            assembly.arithmetic, state,
            scrambled_directions[:, column].astype(np.uint64), bits,
        )
        points[index + 1] = assembly.scaling(state, scale)
    return points, {
        "bits": int(bits),
        "num_generated": int(generated),
        "initial_state": initial_state,
        "final_state": state.copy(),
        "directions": directions,
        "scrambled_directions": scrambled_directions,
        "raw_shift": raw_shift,
        "raw_matrix": raw_matrix,
        "production_scipy_sobol_fallback_invocations": 0,
    }


def scheduled_pool(dimension, sample_count, case_id, iteration, log):
    """Live pool with only Sobol block sizing/prefix replaced by its assembly.

    Scrambled coordinate generation, boundary support, and merge/order remain
    explicit production fallbacks.  Keeping them separate makes it impossible
    for this binding change to silently integrate either later operation.
    """
    requested = min(
        n1.teacher.MC_FACTOR_PER_SITE * sample_count,
        n1.teacher.MC_CANDIDATE_CAP,
    )
    exponent, generated = sobol_schedule(requested)
    seed = n1.teacher.SEARCH_SEED + 104729 * case_id + 1009 * iteration

    full_block, sobol_state = scrambled_sobol_full_block(
        dimension, exponent, seed,
    )
    interior = full_block[:requested]
    log.call(
        "Sobol full-block sizing and prefix", "B",
        "validate_sobol_batching_truncation.schedule + exact first-N prefix", 1,
    )
    log.call(
        "scrambled Sobol coordinate generation", "B",
        "locked scramble/parity/XOR/recurrence/scaling assembly", generated,
    )

    # Inventory-D boundary oracle remains unchanged.
    faces, _ = n1.boundary(dimension, requested, seed)
    log.call(
        "dimension-general boundary-face support", "D",
        "temporary production boundary oracle", 1,
    )

    raw_pool = np.concatenate((interior, faces))
    merged, retained_indices = independent_unique(raw_pool)
    log.call(
        "pool merge, exact deduplication, lexicographic order", "B",
        "n1_boundary_merge_dedup independent lexsort assembly", 1,
    )
    return merged, {
        "requested": int(requested),
        "exponent": int(exponent),
        "generated": int(generated),
        "generated_length": int(len(full_block)),
        "returned_prefix_length": int(len(interior)),
        "seed": int(seed),
        "engine_num_generated": int(sobol_state["num_generated"]),
        "engine_initial_state": sobol_state["initial_state"],
        "engine_final_state": sobol_state["final_state"],
        "sobol_state": sobol_state,
        "full_block": full_block,
        "prefix": interior,
        "raw_pool": raw_pool,
        "merge_retained_indices": retained_indices,
        "pool_merge_np_unique_fallback_invocations": 0,
        "sizing_prefix_fallback_invocations": 0,
    }


def continuous_sobol_multistarts(dimension, incumbent, log=None):
    """Exact fresh-engine Sobol prefix plus incumbent-first start ordering."""
    requested = 2 * int(dimension)
    exponent, generated = sobol_schedule(requested)
    full_block, sobol_state = scrambled_sobol_full_block(
        int(dimension), int(exponent), SEARCH_SEED,
    )
    prefix = full_block[:requested]
    starts = np.vstack((np.asarray(incumbent, np.float64), prefix))
    if log is not None:
        log.call(
            "continuous-stage Sobol multistarts", "B",
            "locked Sobol schedule/scramble/prefix + incumbent-first assembly", 1,
        )
    return starts, {
        "requested": requested,
        "exponent": int(exponent),
        "generated": int(generated),
        "full_block": full_block,
        "prefix": prefix,
        "incumbent": np.asarray(incumbent, np.float64).copy(),
        "engine_num_generated": int(sobol_state["num_generated"]),
        "engine_initial_state": sobol_state["initial_state"],
        "engine_final_state": sobol_state["final_state"],
        "production_scipy_sobol_fallback_invocations": 0,
        "persistent_engine_state": False,
    }


def gaussian_division_assembly():
    """Load the already-locked finite-domain division assembly lazily."""
    global _GAUSSIAN_DIVISION_ASSEMBLY
    if _GAUSSIAN_DIVISION_ASSEMBLY is None:
        router = RoundingRouter()
        path = Path(
            "outputs/sho/stage1B/division_rounding_router_training/"
            "rounding_router_before_evaluation.pt"
        )
        router.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
        _GAUSSIAN_DIVISION_ASSEMBLY = DivisionAssembly(router)
    return _GAUSSIAN_DIVISION_ASSEMBLY


def gaussian_bandwidth_ratio_assembly(distances):
    """Exact locked max-floor and finite-domain division assembly.

    Selecting the final ordered-neighbor distance and broadcasting the
    resulting column are deterministic indexing scaffolding.  Arithmetic is
    supplied only by the locked subtraction/comparison/select and division
    assemblies validated for the active Gaussian domain.
    """
    values = np.asarray(distances, np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Gaussian distances must be a nonempty matrix")
    division = gaussian_division_assembly()
    last = values[:, -1:]
    margin = division.residual._call(
        division.residual.subtract, last, GAUSSIAN_BANDWIDTH_FLOOR,
    )
    floor_route = margin < 0.0
    bandwidth = np.where(floor_route, GAUSSIAN_BANDWIDTH_FLOOR, last)
    expanded = np.broadcast_to(bandwidth, values.shape)
    pairs = np.column_stack((values.ravel(), expanded.ravel()))
    ratio, _, _ = division.predict(pairs)
    return ratio.reshape(values.shape), {
        "last_neighbor_distance": last,
        "floor_route": floor_route,
        "bandwidth": bandwidth,
        "expanded_bandwidth": expanded,
        "production_bandwidth_ratio_fallback_invocations": 0,
    }


def gaussian_ratio_square_assembly(ratio):
    """Square Gaussian ratios with the already-locked multiply primitive."""
    values = np.asarray(ratio, np.float64)
    division = gaussian_division_assembly()
    return division.residual._call(
        division.residual.multiply, values, values,
    )


def gaussian_minus9_scaling_assembly(square):
    """Apply the Gaussian -9 scale with the locked multiply primitive."""
    values = np.asarray(square, np.float64)
    division = gaussian_division_assembly()
    product = division.residual._call(
        division.residual.multiply,
        np.full_like(values, np.float64(-9.0)),
        values,
    )
    # The locked numerical multiplier's documented contract canonicalizes a
    # zero result.  Restore IEEE product sign with the project's existing
    # zero-parameter sign-bit/XOR scaffold, as is already done by the locked
    # division and bounded-scalar assemblies.
    negative = np.logical_xor(np.signbit(np.float64(-9.0)), np.signbit(values))
    return np.copysign(np.abs(product), np.where(negative, -1.0, 1.0))


def _locked_product(left, right):
    """Locked numerical product with the existing IEEE sign-bit scaffold."""
    left, right = np.broadcast_arrays(
        np.asarray(left, np.float64), np.asarray(right, np.float64),
    )
    division = gaussian_division_assembly()
    product = division.residual._call(division.residual.multiply, left, right)
    negative = np.logical_xor(np.signbit(left), np.signbit(right))
    return np.copysign(np.abs(product), np.where(negative, -1.0, 1.0))


def _locked_ordered_sum(rows):
    """NumPy float64 pairwise order for active rows of length at most 24."""
    values = np.asarray(rows, np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or values.shape[1] > 24:
        raise ValueError("ordered Gaussian reduction expects 1..24 columns")
    add = gaussian_division_assembly().residual.add
    call = gaussian_division_assembly().residual._call
    count = values.shape[1]
    if count < 8:
        result = np.full(values.shape[0], np.float64(-0.0))
        for column in range(count):
            result = call(add, result, values[:, column])
        return result
    lanes = [values[:, column].copy() for column in range(8)]
    column = 8
    while column <= count - 8:
        for lane in range(8):
            lanes[lane] = call(add, lanes[lane], values[:, column + lane])
        column += 8
    left = call(
        add, call(add, lanes[0], lanes[1]), call(add, lanes[2], lanes[3]),
    )
    right = call(
        add, call(add, lanes[4], lanes[5]), call(add, lanes[6], lanes[7]),
    )
    result = call(add, left, right)
    while column < count:
        result = call(add, result, values[:, column])
        column += 1
    return result


def gaussian_weighted_prediction_assembly(exponential, encoded_neighbors):
    """Exact weighted prediction after the retained production exp oracle."""
    exponential = np.asarray(exponential, np.float64)
    encoded = np.asarray(encoded_neighbors, np.float64)
    if exponential.shape != encoded.shape or exponential.ndim != 2:
        raise ValueError("Gaussian weights and gathered losses must align by row")
    arithmetic = gaussian_division_assembly().residual
    weights = arithmetic._call(
        arithmetic.add, exponential, np.full_like(exponential, np.float64(1.0e-15)),
    )
    weighted_terms = _locked_product(weights, encoded)
    numerator = _locked_ordered_sum(weighted_terms)
    denominator = _locked_ordered_sum(weights)
    pairs = np.column_stack((numerator, denominator))
    prediction, _, _ = gaussian_division_assembly().predict(pairs)
    return prediction, {
        "exponential": exponential,
        "weights": weights,
        "weighted_terms": weighted_terms,
        "numerator": numerator,
        "denominator": denominator,
        "production_weighted_prediction_fallback_invocations": 0,
    }


def guarded_normalization_assembly(values, log=None):
    """Teacher-exact guarded min/max normalization from locked arithmetic."""
    values = np.asarray(values, np.float64)
    if values.size == 0:
        raise ValueError("guarded normalization requires a nonempty vector")
    arithmetic = gaussian_division_assembly().residual
    division = gaussian_division_assembly()

    # Variable-length reductions and scalar storage are deterministic scaffold.
    minimum = np.float64(np.min(values))
    maximum = np.float64(np.max(values))
    spread = arithmetic._call(
        arithmetic.subtract, np.asarray([maximum]), np.asarray([minimum]),
    )[0]
    absolute_minimum = np.abs(minimum)
    absolute_maximum = np.abs(maximum)

    def locked_max(left, right):
        margin = arithmetic._call(
            arithmetic.subtract, np.asarray([right]), np.asarray([left]),
        )
        with torch.no_grad():
            sign = division.threshold.decision(torch.from_numpy(margin)).numpy()[0]
        return np.float64(right if sign > 0 else left)

    scale = locked_max(np.float64(1.0), absolute_minimum)
    scale = locked_max(scale, absolute_maximum)
    tolerance = _locked_product(np.float64(1.0e-15), scale).reshape(()).item()
    guard_margin = arithmetic._call(
        arithmetic.subtract, np.asarray([spread]), np.asarray([tolerance]),
    )
    with torch.no_grad():
        guard_sign = division.threshold.decision(
            torch.from_numpy(guard_margin)
        ).numpy()[0]
    guarded = bool(guard_sign <= 0)

    numerator = denominator = None
    if guarded:
        output = np.full_like(values, np.float64(0.5), dtype=np.float64)
    else:
        numerator = arithmetic._call(
            arithmetic.subtract, values, np.full_like(values, minimum),
        )
        # Preserve the teacher's second, ordinary-branch subtraction.
        denominator = arithmetic._call(
            arithmetic.subtract, np.asarray([maximum]), np.asarray([minimum]),
        )[0]
        pairs = np.column_stack((numerator, np.full_like(numerator, denominator)))
        output, _, _ = division.predict(pairs)
    if log is not None:
        log.call(
            "guarded min-max normalization", "B",
            "locked subtract/abs/max/multiply/guard/divide assembly",
            1,
        )
    return output, {
        "minimum": minimum, "maximum": maximum, "spread": spread,
        "absolute_minimum": absolute_minimum,
        "absolute_maximum": absolute_maximum,
        "scale": scale, "tolerance": np.float64(tolerance),
        "guarded": guarded, "numerator": numerator,
        "denominator": denominator,
        "production_normalization_fallback_invocations": 0,
    }


def _locked_add(left, right):
    """Locked addition with exact IEEE signed-zero restoration."""
    left, right = np.broadcast_arrays(
        np.asarray(left, np.float64), np.asarray(right, np.float64),
    )
    arithmetic = gaussian_division_assembly().residual
    result = arithmetic._call(arithmetic.add, left, right)
    # Under round-to-nearest, an exact zero sum is -0 only when both zero
    # operands are -0. All other exact cancellations produce +0.
    zero = result == 0.0
    negative_zero = (
        zero & (left == 0.0) & (right == 0.0)
        & np.signbit(left) & np.signbit(right)
    )
    return np.where(
        zero,
        np.copysign(np.zeros_like(result), np.where(negative_zero, -1.0, 1.0)),
        result,
    )


def selected_cell_weight_assembly(size_normalized, loo_normalized, log=None):
    """Exact selected-cell blend before the separate largest-cell override."""
    size_values, loo_values = np.broadcast_arrays(
        np.asarray(size_normalized, np.float64),
        np.asarray(loo_normalized, np.float64),
    )
    size_term = _locked_product(np.full_like(size_values, 0.5), size_values)
    loo_term = _locked_product(np.full_like(loo_values, 0.5), loo_values)
    weight = _locked_add(size_term, loo_term)
    if log is not None:
        log.call(
            "selected-cell weight", "B",
            "locked multiply/multiply/add assembly in production order",
            weight.size,
        )
    return weight, {
        "size_normalized": size_values,
        "loo_normalized": loo_values,
        "size_term": size_term,
        "loo_term": loo_term,
        "production_weight_arithmetic_fallback_invocations": 0,
    }


def ordinary_first_max_assembly(scores, log=None):
    """Locked strict comparison scan; exact ties retain canonical first."""
    global _FIRST_MAX_CONTROLLER
    values = np.asarray(scores, np.float64).reshape(-1)
    if not len(values):
        raise ValueError("ordinary first-max requires at least one score")
    if _FIRST_MAX_CONTROLLER is None:
        _FIRST_MAX_CONTROLLER = NumpyDirectionController()
    controller = _FIRST_MAX_CONTROLLER
    arithmetic = gaussian_division_assembly().residual
    winner = 0
    decisions = []
    for candidate in range(1, len(values)):
        previous = winner
        difference = float(arithmetic._call(
            arithmetic.subtract,
            np.asarray([values[winner]], np.float64),
            np.asarray([values[candidate]], np.float64),
        )[0])
        replace = bool(controller._negative(difference))
        if replace:
            winner = candidate
        decisions.append({
            "current_index": previous,
            "candidate_index": candidate,
            "difference": difference,
            "exact_tie": bool(values[previous] == values[candidate]),
            "replace": replace,
            "winner_after": winner,
        })
    if log is not None:
        log.call(
            "ordinary-cell winner", "B",
            "locked strict-comparison/first-max routing assembly",
            len(decisions),
        )
    return winner, {
        "decisions": decisions,
        "np_argmax_candidate_fallback_invocations": 0,
    }


def proposal_duplicate_filter_assembly(sites, point, log=None):
    """Teacher-exact NumPy-norm minimum and strict 1e-12 history filter."""
    global _NUMPY_EUCLIDEAN_SQRT, _FIRST_MAX_CONTROLLER
    history = np.asarray(sites, np.float64)
    candidate = np.asarray(point, np.float64).reshape(1, -1)
    if history.ndim != 2 or history.shape[1] != candidate.shape[1] or not len(history):
        raise ValueError("nonempty [n,d] history and one d-vector candidate required")
    arithmetic = gaussian_division_assembly().residual
    if _NUMPY_EUCLIDEAN_SQRT is None:
        _NUMPY_EUCLIDEAN_SQRT = N1CorrectlyRoundedSqrt()
    if _FIRST_MAX_CONTROLLER is None:
        _FIRST_MAX_CONTROLLER = NumpyDirectionController()
    delta = arithmetic._call(
        arithmetic.subtract,
        history,
        np.broadcast_to(candidate, history.shape),
    )
    square = arithmetic._call(arithmetic.multiply, delta, delta)
    squared_distance = square[:, 0]
    for axis in range(1, square.shape[1]):
        squared_distance = arithmetic._call(
            arithmetic.add, squared_distance, square[:, axis],
        )
    distances = _NUMPY_EUCLIDEAN_SQRT(squared_distance)
    minimum_index = 0
    minimum_decisions = []
    for site_index in range(1, len(distances)):
        previous = minimum_index
        if distances[site_index] == distances[minimum_index]:
            replace = False
        else:
            difference = float(arithmetic._call(
                arithmetic.subtract,
                np.asarray([distances[site_index]], np.float64),
                np.asarray([distances[minimum_index]], np.float64),
            )[0])
            replace = bool(_FIRST_MAX_CONTROLLER._negative(difference))
        if replace:
            minimum_index = site_index
        minimum_decisions.append({
            "current_index": previous,
            "candidate_index": site_index,
            "exact_tie": bool(distances[site_index] == distances[previous]),
            "replace": replace,
            "winner_after": minimum_index,
        })
    minimum = np.float64(distances[minimum_index])
    margin = float(arithmetic._call(
        arithmetic.subtract,
        np.asarray([np.float64(1.0e-12)], np.float64),
        np.asarray([minimum], np.float64),
    )[0])
    retain = bool(_FIRST_MAX_CONTROLLER._negative(margin))
    if log is not None:
        log.call(
            "history duplicate filter", "B",
            "locked NumPy-order Euclidean/repeated-min/strict-threshold assembly",
            1,
        )
    return retain, {
        "delta": delta,
        "square": square,
        "squared_distance": squared_distance,
        "distances": distances,
        "minimum_index": minimum_index,
        "minimum": minimum,
        "minimum_decisions": minimum_decisions,
        "threshold": np.float64(1.0e-12),
        "threshold_minus_minimum": margin,
        "native_numpy_distance_min_fallback_invocations": 0,
    }


def empty_proposal_fallback_assembly(pool, nearest_distances, log=None):
    """NumPy-1.26 indirect-quicksort order via locked strict comparisons."""
    global _FIRST_MAX_CONTROLLER
    points = np.asarray(pool, np.float64)
    values = np.asarray(nearest_distances, np.float64).reshape(-1)
    if points.ndim != 2 or len(points) != len(values):
        raise ValueError("fallback pool and nearest-distance vector must align")
    if _FIRST_MAX_CONTROLLER is None:
        _FIRST_MAX_CONTROLLER = NumpyDirectionController()
    arithmetic = gaussian_division_assembly().residual
    comparisons = []

    def less(left, right):
        left_value, right_value = np.float64(left), np.float64(right)
        if left_value == right_value:
            result = False
        else:
            difference = float(arithmetic._call(
                arithmetic.subtract,
                np.asarray([left_value], np.float64),
                np.asarray([right_value], np.float64),
            )[0])
            result = bool(_FIRST_MAX_CONTROLLER._negative(difference))
        comparisons.append((float(left_value), float(right_value), result))
        return result

    order = np.arange(len(values), dtype=np.int64)
    if len(order) > 1:
        # Exact transcription of NumPy 1.26's numeric aquicksort_ index path:
        # median-of-three partitioning above 16 entries, insertion sorting for
        # each small partition, with larger partitions held on a LIFO stack.
        stack = []
        lower, upper = 0, len(order) - 1
        depth = (len(order).bit_length() - 1) * 2

        def heapsort_segment(begin, end):
            segment = order[begin:end + 1].copy()
            count = len(segment)
            for level in range(count >> 1, 0, -1):
                saved = int(segment[level - 1])
                parent, child = level, level << 1
                while child <= count:
                    if (child < count and less(
                        values[segment[child - 1]], values[segment[child]],
                    )):
                        child += 1
                    if less(values[saved], values[segment[child - 1]]):
                        segment[parent - 1] = segment[child - 1]
                        parent = child
                        child += child
                    else:
                        break
                segment[parent - 1] = saved
            remaining = count
            while remaining > 1:
                saved = int(segment[remaining - 1])
                segment[remaining - 1] = segment[0]
                remaining -= 1
                parent, child = 1, 2
                while child <= remaining:
                    if (child < remaining and less(
                        values[segment[child - 1]], values[segment[child]],
                    )):
                        child += 1
                    if less(values[saved], values[segment[child - 1]]):
                        segment[parent - 1] = segment[child - 1]
                        parent = child
                        child += child
                    else:
                        break
                segment[parent - 1] = saved
            order[begin:end + 1] = segment

        while True:
            if depth < 0:
                heapsort_segment(lower, upper)
                if not stack:
                    break
                lower, upper, depth = stack.pop()
                continue
            while upper - lower > 15:
                middle = lower + ((upper - lower) >> 1)
                if less(values[order[middle]], values[order[lower]]):
                    order[middle], order[lower] = order[lower], order[middle]
                if less(values[order[upper]], values[order[middle]]):
                    order[upper], order[middle] = order[middle], order[upper]
                if less(values[order[middle]], values[order[lower]]):
                    order[middle], order[lower] = order[lower], order[middle]
                pivot = np.float64(values[order[middle]])
                left = lower
                right = upper - 1
                order[middle], order[right] = order[right], order[middle]
                while True:
                    left += 1
                    while less(values[order[left]], pivot):
                        left += 1
                    right -= 1
                    while less(pivot, values[order[right]]):
                        right -= 1
                    if left >= right:
                        break
                    order[left], order[right] = order[right], order[left]
                pivot_position = upper - 1
                order[left], order[pivot_position] = order[pivot_position], order[left]
                depth -= 1
                if left - lower < upper - left:
                    stack.append((left + 1, upper, depth))
                    upper = left - 1
                else:
                    stack.append((lower, left - 1, depth))
                    lower = left + 1
            for position in range(lower + 1, upper + 1):
                saved = int(order[position])
                cursor = position
                previous = position - 1
                while cursor > lower and less(values[saved], values[order[previous]]):
                    order[cursor] = order[previous]
                    cursor -= 1
                    previous -= 1
                order[cursor] = saved
            if not stack:
                break
            lower, upper, depth = stack.pop()

    selected = None
    threshold_decisions = []
    for candidate_index in order[::-1]:
        margin = float(arithmetic._call(
            arithmetic.subtract,
            np.asarray([np.float64(1.0e-12)], np.float64),
            np.asarray([values[candidate_index]], np.float64),
        )[0])
        eligible = bool(_FIRST_MAX_CONTROLLER._negative(margin))
        threshold_decisions.append((int(candidate_index), eligible))
        if eligible:
            selected = int(candidate_index)
            break
    if log is not None:
        log.call(
            "empty-proposal global fallback", "B",
            "locked NumPy-aquicksort comparison/router + reverse traversal assembly",
            1,
        )
    return selected, {
        "ascending_order": order,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "threshold_decisions": threshold_decisions,
        "selected_coordinates": None if selected is None else points[selected].copy(),
        "reverse_np_argsort_fallback_invocations": 0,
    }


def normalized_to_physical_assembly(normalized, lower, upper, log=None):
    """Exact source-ordered lower + ((upper - lower) * normalized) map."""
    unit = np.asarray(normalized, np.float64)
    lower_bound = np.asarray(lower, np.float64).reshape(-1)
    upper_bound = np.asarray(upper, np.float64).reshape(-1)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
        squeeze = True
    elif unit.ndim == 2:
        squeeze = False
    else:
        raise ValueError("normalized coordinates must be one vector or a matrix")
    if unit.shape[1] != len(lower_bound) or upper_bound.shape != lower_bound.shape:
        raise ValueError("coordinate and bound dimensions must agree")
    arithmetic = gaussian_division_assembly().residual
    width = arithmetic._call(
        arithmetic.subtract, upper_bound, lower_bound,
    )
    scaled = _locked_product(
        np.broadcast_to(width, unit.shape),
        unit,
    )
    physical = _locked_add(
        np.broadcast_to(lower_bound, unit.shape), scaled,
    )
    if log is not None:
        log.call(
            "normalized-to-physical affine map", "B",
            "locked subtract/multiply/add assembly in production order",
            physical.size,
        )
    output = physical[0] if squeeze else physical
    return output, {
        "normalized": unit[0].copy() if squeeze else unit.copy(),
        "lower": lower_bound,
        "upper": upper_bound,
        "width": width,
        "scaled": scaled[0].copy() if squeeze else scaled,
        "production_affine_fallback_invocations": 0,
    }


class ContinuousResultGuardError(RuntimeError):
    """Production-compatible invalid-result exit with replay diagnostics."""

    def __init__(self, metadata):
        super().__init__("Invalid continuous surrogate minimizer")
        self.metadata = metadata


def _locked_boolean_and(left, right):
    global _FIRST_MAX_CONTROLLER
    if _FIRST_MAX_CONTROLLER is None:
        _FIRST_MAX_CONTROLLER = NumpyDirectionController()
    return bool(_FIRST_MAX_CONTROLLER._route(
        "boolean_and", [float(bool(left)), float(bool(right))],
    ))


def _locked_boolean_or(left, right):
    return not _locked_boolean_and(not bool(left), not bool(right))


def _locked_finite(values):
    """IEEE-754 finiteness predicate from the exact exponent field."""
    bits = np.asarray(values, np.float64).view(np.uint64)
    return (bits & FLOAT64_EXPONENT_MASK) != FLOAT64_EXPONENT_MASK


def _locked_less(left, right):
    """Locked subtraction followed by the locked strict-negative router."""
    global _FIRST_MAX_CONTROLLER
    if _FIRST_MAX_CONTROLLER is None:
        _FIRST_MAX_CONTROLLER = NumpyDirectionController()
    arithmetic = gaussian_division_assembly().residual
    difference = float(arithmetic._call(
        arithmetic.subtract,
        np.asarray([np.float64(left)], np.float64),
        np.asarray([np.float64(right)], np.float64),
    )[0])
    return bool(_FIRST_MAX_CONTROLLER._negative(difference)), difference


def _locked_float_equal(left, right):
    """Numeric float equality assembled from the two strict order routes."""
    left_less, left_difference = _locked_less(left, right)
    right_less, right_difference = _locked_less(right, left)
    equal = _locked_boolean_and(not left_less, not right_less)
    return equal, {
        "left_less": left_less, "right_less": right_less,
        "left_minus_right": left_difference,
        "right_minus_left": right_difference,
    }


def continuous_result_guards_assembly(results, evaluated_sites, log=None):
    """Teacher-exact post-Powell validation, winner, and duplicate routing."""
    sites = np.asarray(evaluated_sites, np.float64)
    if sites.ndim != 2 or not len(sites) or not results:
        raise ValueError("nonempty result list and evaluated-site matrix required")
    metadata = {
        "statuses": [],
        "winner_decisions": [], "winner_index": None,
        "winner_start_id": None, "duplicate_rows": [],
        "duplicate": None, "proposal": None,
        "native_guard_fallback_invocations": 0,
    }
    normalized = []
    for canonical_index, result in enumerate(results):
        point = np.asarray(
            result["x"] if "x" in result else result["point"], np.float64,
        ).reshape(-1)
        prediction = np.float64(
            result["fun"] if "fun" in result else result["prediction"]
        )
        start_id = int(result.get("start_id", canonical_index))
        prediction_finite = bool(_locked_finite(prediction))
        if not prediction_finite:
            metadata["statuses"].append({
                "canonical_index": canonical_index, "start_id": start_id,
                "prediction_finite": False, "coordinates_finite": None,
                "finite_result": False, "in_bounds": None,
                "valid": False, "coordinates": None,
            })
            raise ContinuousResultGuardError(metadata)
        coordinate_finite = _locked_finite(point)
        all_coordinates_finite = True
        for finite in coordinate_finite:
            all_coordinates_finite = _locked_boolean_and(
                all_coordinates_finite, bool(finite),
            )
        if not all_coordinates_finite:
            metadata["statuses"].append({
                "canonical_index": canonical_index, "start_id": start_id,
                "prediction_finite": True, "coordinates_finite": False,
                "finite_result": False, "in_bounds": None,
                "valid": False, "coordinates": None,
            })
            raise ContinuousResultGuardError(metadata)
        in_bounds = True
        bound_rows = []
        for coordinate in point:
            below, _ = _locked_less(coordinate, np.float64(0.0))
            above, _ = _locked_less(np.float64(1.0), coordinate)
            coordinate_in_bounds = _locked_boolean_and(
                not below, not above,
            )
            in_bounds = _locked_boolean_and(in_bounds, coordinate_in_bounds)
            bound_rows.append({
                "finite": True, "below_zero": below,
                "above_one": above, "in_bounds": coordinate_in_bounds,
            })
        valid = bool(in_bounds)
        metadata["statuses"].append({
            "canonical_index": canonical_index, "start_id": start_id,
            "prediction_finite": True,
            "coordinates_finite": True,
            "finite_result": True, "in_bounds": in_bounds,
            "valid": valid, "coordinates": bound_rows,
        })
        if not valid:
            raise ContinuousResultGuardError(metadata)
        normalized.append({
            "start_id": start_id, "point": point, "prediction": prediction,
        })

    winner = 0
    for candidate in range(1, len(normalized)):
        current = winner
        candidate_less, _ = _locked_less(
            normalized[candidate]["prediction"],
            normalized[current]["prediction"],
        )
        current_less, _ = _locked_less(
            normalized[current]["prediction"],
            normalized[candidate]["prediction"],
        )
        exact_tie = _locked_boolean_and(not candidate_less, not current_less)
        # Start IDs are loop/index state. In production they are canonical
        # consecutive integers, so this is also the stable-first tie route.
        lower_start_id = (
            normalized[candidate]["start_id"] < normalized[current]["start_id"]
        )
        tie_replace = _locked_boolean_and(exact_tie, lower_start_id)
        replace = _locked_boolean_or(candidate_less, tie_replace)
        if replace:
            winner = candidate
        metadata["winner_decisions"].append({
            "current_index": current, "candidate_index": candidate,
            "candidate_prediction_lower": candidate_less,
            "current_prediction_lower": current_less,
            "exact_prediction_tie": exact_tie,
            "candidate_start_id_lower": lower_start_id,
            "replace": replace, "winner_after": winner,
        })

    point = normalized[winner]["point"]
    duplicate = False
    for site_index, site in enumerate(sites):
        row_equal = True
        coordinate_equal = []
        for observed, proposed in zip(site, point):
            equal, _ = _locked_float_equal(observed, proposed)
            row_equal = _locked_boolean_and(row_equal, equal)
            coordinate_equal.append(equal)
        duplicate = _locked_boolean_or(duplicate, row_equal)
        metadata["duplicate_rows"].append({
            "site_index": site_index, "coordinate_equal": coordinate_equal,
            "row_equal": row_equal,
        })
    proposal = None if duplicate else point.copy()
    metadata.update({
        "winner_index": winner,
        "winner_start_id": normalized[winner]["start_id"],
        "duplicate": duplicate, "proposal": proposal,
    })
    if log is not None:
        log.call(
            "continuous result guards/winner/duplicate route", "B",
            "locked finite/bounds/comparison/equality/Boolean-select assembly", 1,
        )
    return proposal, metadata


class ArchiveAbsorbingSuccessLatch:
    """Locked Boolean/select state for the archive's absorbing GOOD event."""

    def __init__(self):
        self.success = False
        self.successful_site = None
        self.transitions = []

    def update(self, oracle_good, current_site, log=None):
        before_success = bool(self.success)
        before_site = self.successful_site
        good = bool(oracle_good)
        acquire = _locked_boolean_and(not before_success, good)
        after_success = _locked_boolean_or(before_success, good)

        # The learned Boolean route controls this object-valued select; site
        # identity and state storage remain deterministic scaffolding.
        if before_success:
            after_site = before_site
        elif acquire:
            after_site = current_site
        else:
            after_site = None
        self.success = after_success
        self.successful_site = after_site
        transition = {
            "success_before": before_success,
            "successful_site_before": before_site,
            "oracle_good": good,
            "current_site": current_site,
            "acquire": acquire,
            "success_after": after_success,
            "successful_site_after": after_site,
            "stop": after_success,
            "continue_n1": not after_success,
            "metadata_latch_fallback_invocations": 0,
        }
        self.transitions.append(transition)
        if log is not None:
            log.call(
                "archive absorbing-success latch", "B",
                "locked Boolean/select/state assembly", 1,
            )
        return transition


def replay_archive_absorbing_latch(probe_events, log=None,
                                   stop_at_first_success=True):
    """Live archive control with probe labels supplied by the retained oracle."""
    latch = ArchiveAbsorbingSuccessLatch()
    processed = []
    for event in probe_events:
        transition = latch.update(
            event["oracle_good"], event["site_id"], log,
        )
        processed.append({**event, **transition})
        if transition["stop"] and stop_at_first_success:
            break
    return {
        "success": latch.success,
        "successful_site": latch.successful_site,
        "stop": latch.success,
        "continue_n1": not latch.success,
        "processed": processed,
        "metadata_latch_fallback_invocations": 0,
    }


class N2BestEverStrictLossTracker:
    """Teacher-exact N2 best-ever state driven by the locked strict router."""

    def __init__(self, start_theta, start_loss):
        self.best_theta = np.asarray(start_theta, np.float64).copy()
        self.best_loss = np.float64(start_loss)
        self.transitions = []

    def observe(self, new_theta, new_loss, log=None):
        candidate_theta = np.asarray(new_theta, np.float64).copy()
        candidate_loss = np.float64(new_loss)
        before_theta = self.best_theta.copy()
        before_loss = np.float64(self.best_loss)
        replace, difference = _locked_less(candidate_loss, before_loss)

        # The learned strict-comparison route controls this paired select.
        # Array copying and state storage are deterministic scaffolding.
        if replace:
            after_theta = candidate_theta
            after_loss = candidate_loss
        else:
            after_theta = before_theta
            after_loss = before_loss
        self.best_theta = after_theta.copy()
        self.best_loss = np.float64(after_loss)
        transition = {
            "candidate_theta": candidate_theta,
            "candidate_loss": candidate_loss,
            "best_theta_before": before_theta,
            "best_loss_before": before_loss,
            "candidate_minus_best": np.float64(difference),
            "replace": bool(replace),
            "best_theta_after": self.best_theta.copy(),
            "best_loss_after": np.float64(self.best_loss),
            "native_python_comparison_fallback_invocations": 0,
        }
        self.transitions.append(transition)
        if log is not None:
            log.call(
                "N2 best-ever strict-loss selection", "B",
                "locked subtraction/strict-comparison/select assembly", 1,
            )
        return transition


def run_n2_with_best_ever_assembly(
        objective, start_theta, start_loss, lower, upper, log=None, *,
        xtol=1.0e-12, ftol=1.0e-15, maxiter=200, maxfev=1500,
        trace=False):
    """Current exact Powell path with the integrated learned best tracker."""
    from experiments.n2_exact_powell_assembly import (
        LearnedDirectionController, minimize_powell_exact,
    )

    tracker = N2BestEverStrictLossTracker(start_theta, start_loss)

    def counted(theta):
        point = np.asarray(theta, np.float64)
        value = np.float64(objective(point))
        tracker.observe(point, value, log)
        return float(value)

    result = minimize_powell_exact(
        counted, np.asarray(start_theta, np.float64),
        bounds=list(zip(np.asarray(lower, np.float64),
                        np.asarray(upper, np.float64))),
        xtol=xtol, ftol=ftol, maxiter=maxiter, maxfev=maxfev,
        trace=trace, controller=LearnedDirectionController(),
    )
    return {
        "theta": tracker.best_theta.copy(),
        "loss": float(tracker.best_loss),
        "evaluations": int(result["nfev"]),
        "message": result["message"],
        "powell": result,
        "tracker": tracker,
        "native_python_comparison_fallback_invocations": 0,
    }


class HybridN3GradientBinding:
    """Live N3 call-site binding; Hessian remains outside this integration."""

    def __init__(self):
        self.assembly = N3GradientBindingAssembly()

    def evaluate(self, system, trajectory, theta, log=None):
        return self.assembly.evaluate(system, trajectory, theta, log)


class HybridN3HessianBinding:
    """Live N3 Hessian binding; spectral and later operations stay untouched."""

    def __init__(self, gradient_binding=None):
        gradient_assembly = (
            gradient_binding.assembly if gradient_binding is not None else None
        )
        self.assembly = N3SymmetricHessianBindingAssembly(gradient_assembly)

    def evaluate(self, system, trajectory, theta, log=None):
        return self.assembly.evaluate(system, trajectory, theta, log)


class HybridN3CurvatureRouteBinding:
    """Locked DIRECT/DAMPED route over the retained spectral D oracle."""

    def __init__(self, condition_limit=1.0e12):
        self.spectral_oracle = N3EigenspectrumConditionAssembly(condition_limit)
        self.router = load_locked_router()
        self.invocations = 0
        self.native_routing_fallback_calls = 0

    def route_result(self, spectrum, log=None):
        curvature_ok = self.router.conjunction(
            bool(spectrum.positive), bool(spectrum.condition_ok),
        )
        direct = self.router.conjunction(bool(spectrum.finite), curvature_ok)
        self.invocations += 1
        if log is not None:
            log.call(
                "N3 DIRECT/DAMPED curvature route", "A",
                "locked Boolean conjunction/router assembly", 1,
            )
        return {
            "route": "DIRECT" if direct else "DAMPED",
            "direct": bool(direct),
            "finite": bool(spectrum.finite),
            "positive": bool(spectrum.positive),
            "condition_ok": bool(spectrum.condition_ok),
            "native_routing_fallback_calls": 0,
        }

    def evaluate(self, gradient, hessian, log=None):
        spectrum = self.spectral_oracle.analyze(gradient, hessian)
        return spectrum, self.route_result(spectrum, log)


class HybridN3DampingBinding:
    """Locked damping arithmetic only; matrix/RHS construction stays separate."""

    def __init__(self, curvature_binding=None):
        self.curvature = curvature_binding or HybridN3CurvatureRouteBinding()
        self.arithmetic = n3_arithmetic.LearnedArithmetic()
        self.negation = Float64UnaryNegationAssembly()
        self.initial_invocations = 0
        self.retry_invocations = 0
        self.native_damping_fallback_calls = 0

    def _multiply(self, left, right) -> float:
        return float(self.arithmetic.multiply(
            torch.tensor(left, dtype=torch.float64),
            torch.tensor(right, dtype=torch.float64),
        ))

    def _add(self, left, right) -> float:
        return float(self.arithmetic.add(
            torch.tensor(left, dtype=torch.float64),
            torch.tensor(right, dtype=torch.float64),
        ))

    def _negative(self, value) -> float:
        source = torch.tensor([value], dtype=torch.float64)
        return float(self.negation(source)[0])

    def _maximum(self, left: float, right: float) -> float:
        # The locked strict comparison asks whether left < right.  Exact ties
        # retain the first operand, matching Python max(left, right).
        difference = float(self.arithmetic.subtract(
            torch.tensor(right, dtype=torch.float64),
            torch.tensor(left, dtype=torch.float64),
        ))
        choose_right = self.curvature.router.positive(difference)
        return float((np.float64(left), np.float64(right))[int(choose_right)])

    def initial(self, scale: float, minimum_eigenvalue: float,
                direct: bool, log=None) -> float:
        # The route bit is produced by the already locked curvature binding.
        # Selecting a stored result is deterministic routing/storage scaffold.
        base = self._multiply(scale, n3_production.DAMPING_RELATIVE_FLOOR)
        shifted = self._add(self._negative(minimum_eigenvalue), base)
        damped = self._maximum(base, shifted)
        direct_route = self.curvature.router.conjunction(bool(direct), True)
        result = float((np.float64(damped), np.float64(0.0))[int(direct_route)])
        self.initial_invocations += 1
        if log is not None:
            log.call(
                "N3 damping construction", "B",
                "locked arithmetic/unary-negation/comparison-select assembly", 1,
            )
        return result

    def value(self, scale: float, initial: float, direct: bool,
              attempt: int, log=None) -> float:
        if attempt < 0 or attempt >= n3_production.MAX_DAMPING_TRIALS:
            raise ValueError("damping attempt is outside the production schedule")
        direct_zero = self.curvature.router.conjunction(
            bool(direct), bool(attempt == 0),
        )
        floor = self._multiply(scale, n3_production.DAMPING_RELATIVE_FLOOR)
        initial_positive = self.curvature.router.positive(float(initial))
        base = float((np.float64(floor), np.float64(initial))[int(initial_positive)])
        exponent = attempt - int(bool(direct))
        factor = float(10.0 ** exponent)
        scheduled = self._multiply(base, factor)
        result = float((np.float64(scheduled), np.float64(0.0))[int(direct_zero)])
        self.retry_invocations += 1
        if log is not None:
            log.call(
                "N3 damping retry value", "B",
                "locked routing/multiply over production powers-of-ten schedule", 1,
            )
        return result


class HybridN3DampedMatrixRHSBinding:
    """Exact learned assembly for H + lambda*I and unary-negated gradient."""

    def __init__(self):
        self.arithmetic = n3_arithmetic.LearnedArithmetic()
        self.negation = Float64UnaryNegationAssembly()
        self.invocations = 0
        self.native_construction_fallback_calls = 0

    def construct(self, hessian, gradient, damping, log=None):
        hessian = torch.as_tensor(hessian, dtype=torch.float64)
        gradient = torch.as_tensor(gradient, dtype=torch.float64)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("N3 damped matrix requires a square Hessian")
        if gradient.shape != (hessian.shape[0],):
            raise ValueError("N3 RHS gradient dimension does not match Hessian")

        # Identity creation, dimensions, and tensor storage are deterministic
        # scaffolding.  All floating-point arithmetic uses locked assemblies.
        identity = torch.eye(hessian.shape[0], dtype=torch.float64)
        damping_tensor = torch.tensor(damping, dtype=torch.float64)
        damped_identity = self.arithmetic.multiply(damping_tensor, identity)
        damped_matrix = self.arithmetic.add(hessian, damped_identity)
        rhs = self.negation(gradient)
        self.invocations += 1
        if log is not None:
            log.call(
                "N3 damped matrix and RHS construction", "B",
                "locked multiply/add/unary-negation assembly", 1,
            )
        return {
            "identity": identity,
            "damping": float(damping),
            "damped_identity": damped_identity,
            "damped_matrix": damped_matrix,
            "rhs": rhs,
            "native_construction_fallback_calls": 0,
        }


class HybridPrefixResidualMSEBinding:
    """Live generic objective arithmetic after supplied system transitions."""

    def __init__(self):
        self.assembly = PrefixResidualMSEAssembly()

    @property
    def invocations(self):
        return self.assembly.invocations

    @property
    def by_system(self):
        return dict(self.assembly.by_system)

    @property
    def native_objective_arithmetic_fallback_calls(self):
        return self.assembly.native_objective_arithmetic_fallback_calls

    def evaluate(self, predicted, observed, system, *, reduction_path, log=None):
        result = self.assembly.evaluate(
            predicted, observed, system, reduction_path=reduction_path,
        )
        if log is not None:
            log.call(
                "one-step prefix residual/MSE reduction", "B",
                "locked subtract/square/ordered-add/divide assembly", 1,
            )
        return result


class HybridGaussianLocalResponse:
    """Production surface with only clamp and bandwidth/ratio replaced."""

    def __init__(self, points, encoded_values, log):
        self.points = np.asarray(points, np.float64)
        self.values = np.asarray(encoded_values, np.float64)
        self.tree = cKDTree(self.points)
        self.log = log
        self.bandwidth_ratio_calls = 0
        self.bandwidth_ratio_values = 0
        self.ratio_square_calls = 0
        self.ratio_square_values = 0
        self.minus9_scaling_calls = 0
        self.minus9_scaling_values = 0
        self.weighted_prediction_calls = 0
        self.weighted_prediction_terms = 0
        self.production_bandwidth_ratio_fallback_invocations = 0
        self.production_ratio_square_fallback_invocations = 0
        self.production_minus9_scaling_fallback_invocations = 0
        self.production_weighted_prediction_fallback_invocations = 0

    def _ratio(self, distances):
        ratio, metadata = gaussian_bandwidth_ratio_assembly(distances)
        self.bandwidth_ratio_calls += 1
        self.bandwidth_ratio_values += ratio.size
        self.log.call(
            "Gaussian bandwidth floor and ratio", "B",
            "locked max-floor comparison/select + finite-domain division assembly",
            ratio.size,
        )
        return ratio, metadata

    def _square(self, ratio):
        square = gaussian_ratio_square_assembly(ratio)
        self.ratio_square_calls += 1
        self.ratio_square_values += square.size
        self.log.call(
            "Gaussian ratio squaring", "A",
            "locked multiplication assembly with shared ratio operands",
            square.size,
        )
        return square

    def _scale_minus9(self, square):
        scaled = gaussian_minus9_scaling_assembly(square)
        self.minus9_scaling_calls += 1
        self.minus9_scaling_values += scaled.size
        self.log.call(
            "Gaussian -9.0 scaling", "A",
            "locked multiplication + zero-parameter product-sign assembly",
            scaled.size,
        )
        return scaled

    def predict(self, query):
        query = np.atleast_2d(query)
        count = min(n1.teacher.RBF_NEIGHBORS, len(self.points))
        distances, indices = self.tree.query(query, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        ratio, _ = self._ratio(distances)
        square = self._square(ratio)
        scaled = self._scale_minus9(square)
        # np.exp remains the explicit deterministic D oracle.
        exponential = np.exp(scaled)
        prediction, _ = gaussian_weighted_prediction_assembly(
            exponential, self.values[indices],
        )
        self.weighted_prediction_calls += len(prediction)
        self.weighted_prediction_terms += exponential.size
        self.log.call(
            "ordered weighted surrogate prediction", "B",
            "locked add/multiply/ordered-reduction/division assembly",
            len(prediction),
        )
        return prediction

    def loo_errors(self, site_indices, return_trace=False):
        site_indices = np.asarray(site_indices, np.int64)
        traces = []
        if len(self.points) <= 2:
            result = np.ones(len(site_indices), dtype=np.float64)
            traces = [
                {"site_index": int(site), "branch": "small_site_fallback",
                 "value": np.float64(1.0),
                 "production_loo_fallback_invocations": 0}
                for site in site_indices
            ]
            self.log.call(
                "LOO residual and small-site fallback", "B",
                "locked LOO arithmetic + exact structural fallback assembly",
                len(site_indices),
            )
            return (result, traces) if return_trace else result
        count = min(n1.teacher.RBF_NEIGHBORS + 1, len(self.points))
        distances, indices = self.tree.query(self.points[site_indices], k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        result = np.empty(len(site_indices), dtype=np.float64)
        for row, site_index in enumerate(site_indices):
            keep = indices[row] != site_index
            local_indices = indices[row, keep][:n1.teacher.RBF_NEIGHBORS]
            local_distances = distances[row, keep][:n1.teacher.RBF_NEIGHBORS]
            if not len(local_indices):
                result[row] = 0.0
                traces.append({
                    "site_index": int(site_index),
                    "branch": "empty_neighbor_fallback",
                    "local_indices": local_indices,
                    "local_distances": local_distances,
                    "value": np.float64(0.0),
                    "production_loo_fallback_invocations": 0,
                })
                continue
            ratio, ratio_meta = self._ratio(local_distances[None, :])
            square = self._square(ratio[0])
            scaled = self._scale_minus9(square)
            # np.exp and ordered-neighbor retrieval remain explicit D oracles.
            exponential = np.exp(scaled)
            prediction_array, prediction_meta = gaussian_weighted_prediction_assembly(
                exponential[None, :], self.values[local_indices][None, :],
            )
            prediction = prediction_array[0]
            arithmetic = gaussian_division_assembly().residual
            signed_residual = arithmetic._call(
                arithmetic.subtract,
                np.asarray([self.values[site_index]], np.float64),
                np.asarray([prediction], np.float64),
            )[0]
            absolute_residual = np.abs(signed_residual)
            result[row] = absolute_residual
            traces.append({
                "site_index": int(site_index), "branch": "weighted_loo",
                "local_indices": local_indices,
                "local_distances": local_distances,
                "ratio": ratio[0], "ratio_metadata": ratio_meta,
                "square": square, "scaled": scaled,
                "exponential": exponential,
                "prediction_metadata": prediction_meta,
                "prediction": prediction,
                "signed_residual": signed_residual,
                "absolute_residual": absolute_residual,
                "value": absolute_residual,
                "production_loo_fallback_invocations": 0,
            })
        self.log.call(
            "LOO residual and small-site fallback", "B",
            "locked LOO arithmetic + exact structural fallback assembly",
            len(site_indices),
        )
        return (result, traces) if return_trace else result


def gaussian_surface_with_clamp_assembly(points, losses, log):
    """Construct the live surface with the two integrated prefix bindings."""
    raw = np.asarray(losses, np.float64)
    clamped = assembled_clamp(raw)
    log.call(
        "loss floor clamp", "B",
        "assembled_clamp locked subtraction/comparison/select assembly", len(raw),
    )
    surface = HybridGaussianLocalResponse(points, np.log10(clamped), log)
    return surface, {
        "raw_losses": raw,
        "clamped_losses": clamped,
        "production_clamp_fallback_invocations": 0,
    }


def student_cell_state(item, frozen, owner_assembly, largest_model, log):
    record = item["record"]
    sites = np.asarray(record["points"][:20], np.float64)
    losses = np.asarray(record["losses"][:20], np.float64)
    pool, pool_schedule = scheduled_pool(
        item["dimension"], len(sites), item["case_id"], 0, log,
    )
    teacher_api = n1.composition(item["dimension"])
    teacher = teacher_api["_cell_data"](sites, losses, item["case_id"], 0)

    owners, distances, owner_stats = owner_assembly.owners(pool, sites)
    radius, _, radius_stats = cells.assigned_radius(
        owners, distances, len(sites), frozen.margin, frozen.threshold)
    log.call("assigned-radius reduction", "A", "locked repeated maximum assembly", len(pool))
    spacing, spacing_stats = owner_assembly.spacing(sites)
    half = cells.predict_half(frozen.half, spacing)
    replace, _, invalid, wrong = cells.larger_candidate(
        frozen.margin, frozen.threshold, radius, half)
    sizes = np.where(replace, half, radius)
    log.call("cell extent", "A", "locked half-spacing and maximum routing assembly", len(sites))

    grouped = _learned_group_and_represent(sizes, losses, frozen.locked)
    log.call("same-size grouping", "B", "locked normalized-gap comparator assembly", len(sites))
    log.call("group representative selection", "B",
             "locked normalized-loss comparator assembly", len(grouped["representatives"]))
    hull = downstream.hull_threshold._threshold_hull(
        frozen.locked["skew_weight"], frozen.locked["homogeneous_weight"],
        frozen.locked["hull_theta"], frozen.locked["hull_alpha"],
        grouped["grouped_sizes"], grouped["grouped_losses"])
    log.call("lower-hull turn and margin", "A", "locked hull-turn/margin assembly", len(hull))
    pool_features = downstream.improvement_and._build_pool([dict(
        source_state_index=0, trajectory_id=0, iteration=0,
        state_bin=cells.base.turn._state_bin(len(sizes)),
        sizes=grouped["grouped_sizes"], losses=grouped["grouped_losses"],
        teacher_hull=hull)], "hybrid_resume")
    with torch.no_grad():
        passed = frozen.locked["and_model"](
            torch.from_numpy(pool_features["inputs"])).numpy().ravel() >= 0
    selected = hull[pool_features["hull_position"][passed]]
    log.call("sufficient-decrease target", "B", "locked arithmetic assembly", len(hull))
    log.call("neighbor/required hull slopes", "B", "locked arithmetic assembly", len(hull))
    log.call("potentially-optimal feasibility", "B", "locked Boolean router", len(hull))
    if not len(selected):
        selected = np.asarray([repaired_choice(largest_model, grouped["grouped_sizes"])[0]])
        log.call("empty-PO largest-cell fallback", "B", "locked largest-cell router", 1)
    po = sorted(grouped["representatives"][selected].tolist(),
                key=lambda index: (sizes[index], losses[index], index))

    # These three substantive kernels intentionally remain the declared D oracles.
    surface, loss_encoding = gaussian_surface_with_clamp_assembly(
        sites, losses, log,
    )
    log.call("production scalar np.log10", "D", "temporary production np.log10 oracle", len(sites))
    log.call("ordered top-min(24,n) neighbor retrieval", "D",
             "temporary production cKDTree ordered-neighbor oracle", len(po))
    log.call("production scalar np.exp in Gaussian kernel", "D",
             "temporary production np.exp oracle", len(po))
    size_normalized, _ = guarded_normalization_assembly(sizes[po], log)
    loo_normalized, _ = guarded_normalization_assembly(
        surface.loo_errors(np.asarray(po)), log,
    )
    weights, _ = selected_cell_weight_assembly(
        size_normalized, loo_normalized, log,
    )

    largest_position, _ = repaired_choice(largest_model, sizes[po])
    log.call("largest-cell role", "A", "locked largest-cell router with exact tie repair", 1)
    weights[largest_position] = 1.0
    proposals = []
    for position, site in enumerate(po):
        indices = np.flatnonzero(owners == site)
        if not len(indices):
            continue
        if position == largest_position:
            winner, _ = repaired_choice(largest_model, distances[indices])
            log.call("largest-cell candidate score/route", "A",
                     "locked shared scalar router with comparison repair", 1)
        else:
            distance_normalized, _ = guarded_normalization_assembly(
                distances[indices], log,
            )
            prediction_normalized, _ = guarded_normalization_assembly(
                surface.predict(pool[indices]), log,
            )
            features = np.column_stack((
                distance_normalized,
                prediction_normalized,
                np.full(len(indices), weights[position])))
            scores = ordinary_assembly(features)
            winner, _ = ordinary_first_max_assembly(scores, log)
            log.call("ordinary-cell score arithmetic", "B", "ordinary_assembly exact production-order assembly", len(indices))
        point = pool[indices[winner]]
        retain, _ = proposal_duplicate_filter_assembly(sites, point, log)
        if retain:
            proposals.append(point)
    if not proposals:
        index, _ = empty_proposal_fallback_assembly(pool, distances, log)
        if index is not None:
            proposals.append(pool[index])
    proposals = np.asarray(proposals, np.float64).reshape(-1, item["dimension"])
    teacher_proposals = teacher_api["_cell_proposals"](
        sites, losses, item["case_id"], 0)
    return {
        "sites": sites, "losses": losses, "pool": pool,
        "pool_schedule": pool_schedule,
        "loss_encoding": loss_encoding,
        "owner": owner_stats, "spacing": spacing_stats,
        "owner_identity_errors": int(np.count_nonzero(owners != teacher.owners)),
        "assigned_distance_errors": int(np.count_nonzero(distances != teacher.distances)),
        "cell_size_errors": int(np.count_nonzero(sizes != teacher.sizes)),
        "radius_invalid": int(invalid + wrong + radius_stats["radius_invalid_logits"] +
                              radius_stats["radius_pairwise_mistakes"]),
        "selected_teacher": teacher.selected.tolist(), "selected_student": po,
        "proposal_shape_teacher": list(teacher_proposals.shape),
        "proposal_shape_student": list(proposals.shape),
        # Retained for downstream fail-closed integration diagnostics.  These
        # are outputs of the already-executed teacher/student paths; exposing
        # them does not alter either proposal decision.
        "teacher_proposals": teacher_proposals,
        "student_proposals": proposals,
        "proposal_bit_mismatches": (0 if teacher_proposals.shape == proposals.shape and
                                    bits_equal(teacher_proposals, proposals) else 1),
    }


def continuous_replay(item, state, log):
    points, losses = state["sites"], state["losses"]
    dimension = item["dimension"]
    surface, _ = gaussian_surface_with_clamp_assembly(points, losses, log)
    function = lambda value: float(surface.predict(np.asarray(value)[None])[0])
    starts, start_state = continuous_sobol_multistarts(
        dimension, points[int(np.argmin(losses))], log,
    )
    control = NumpyDirectionController()
    teachers, students = [], []
    errors = 0
    first = None
    for start_index, start in enumerate(starts):
        teacher = powell_trace(function, start, False)
        student = powell_trace(function, start, True, control)
        comparison = compare_powell(teacher, student)
        local = sum(value for key, value in comparison.items() if key.endswith("errors"))
        if local and first is None:
            first = {"start": start_index, "comparison": comparison,
                     "teacher": {"x": teacher["x"].tolist(), "fun": teacher["fun"]},
                     "student": {"x": student["x"].tolist(), "fun": student["fun"]}}
        errors += local
        teachers.append(teacher); students.append(student)
    log.call("bounded multistart Powell surrogate minimization", "B",
             "exact learned bounded-scalar/Powell/control assembly", len(starts))
    teacher_winner = min(range(len(teachers)), key=lambda index: (teachers[index]["fun"], index))
    teacher_point = teachers[teacher_winner]["x"]
    teacher_duplicate = bool(np.any(np.all(points == teacher_point, axis=1)))
    teacher_proposal = None if teacher_duplicate else teacher_point
    student_proposal, guard_state = continuous_result_guards_assembly(
        students, points, log,
    )
    student_winner = int(guard_state["winner_index"])
    proposals_equal = (
        teacher_proposal is None and student_proposal is None
    ) or (
        teacher_proposal is not None and student_proposal is not None
        and bits_equal(teacher_proposal, student_proposal)
    )
    guards_exact = (
        teacher_winner == student_winner
        and teacher_duplicate == guard_state["duplicate"]
        and proposals_equal
    )
    return {
        "starts": len(starts), "internal_state_errors": errors,
        "start_state": start_state,
        "controller_errors": int(sum(value for key, value in control.counts.items()
                                         if key.endswith("errors"))),
        "teacher_winner": teacher_winner, "student_winner": student_winner,
        # Expose the already-selected normalized outputs for the next binding.
        # This does not participate in winner/guard routing.
        "teacher_winner_point": teachers[teacher_winner]["x"].copy(),
        "student_winner_point": students[student_winner]["x"].copy(),
        "teacher_duplicate": teacher_duplicate,
        "student_duplicate": bool(guard_state["duplicate"]),
        "teacher_proposal": (None if teacher_proposal is None
                             else teacher_proposal.copy()),
        "student_proposal": (None if student_proposal is None
                             else student_proposal.copy()),
        "guard_state": guard_state,
        "winner_exact": bool(guards_exact), "first_mismatch": first,
    }


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_doc = json.loads(INVENTORY.read_text())
    inventory = inventory_doc["inventory"]
    before_inventory = sha256(INVENTORY)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise RuntimeError("checkpoint inventory changed")
    freeze = json.loads((FREEZE / "freeze_manifest.json").read_text())
    for path, digest in freeze["sources"].items():
        if not Path(path).exists() or sha256(Path(path)) != digest:
            raise RuntimeError(f"frozen teacher source changed: {path}")

    log = Log()
    frozen = FrozenProposals()
    owner = StrictOwnerAssembly(frozen, log)
    largest = load_model()
    rows = []
    first_divergence = None
    for item in representative_records():
        cell = student_cell_state(item, frozen, owner, largest, log)
        continuous = continuous_replay(item, cell, log)
        row = {"system": item["system"], "dimension": item["dimension"],
               "state_path": str(item["record"]["state_path"]),
               "site_count": len(cell["sites"]), "pool_count": len(cell["pool"]),
               **{key: value for key, value in cell.items()
                  if key not in ("sites", "losses", "pool")},
               "continuous": continuous}
        rows.append(row)
        checks = {
            "owner": cell["owner"]["owner_errors"] + cell["owner"]["comparison_errors"] +
                     cell["owner"]["distance_errors"],
            "geometry": cell["owner_identity_errors"] + cell["assigned_distance_errors"] +
                        cell["cell_size_errors"] + cell["radius_invalid"] +
                        cell["spacing"]["comparison_errors"] + cell["spacing"]["spacing_errors"],
            "cell_proposal": cell["proposal_bit_mismatches"],
            "continuous_powell": continuous["internal_state_errors"] +
                                 continuous["controller_errors"] + int(not continuous["winner_exact"]),
        }
        bad = next((name for name, count in checks.items() if count), None)
        if bad is not None and first_divergence is None:
            first_divergence = {"operation": bad, "system": item["system"],
                                "mismatch_count": checks[bad], "row": row}
            break

    strict = sum(row["owner"]["strict"] for row in rows)
    ties = sum(row["owner"]["ties"] for row in rows)
    reached = []
    for item in inventory:
        if item["category"] in ("A", "B") and log.counts[item["operation"]]:
            reached.append({"section": item["section"], "operation": item["operation"],
                            "category": item["category"],
                            "invocations": int(log.counts[item["operation"]]),
                            "implementation": log.implementation[item["operation"]][1]})
    deferred = [{"section": item["section"], "operation": item["operation"]}
                for item in inventory if item["category"] == "D" and item["debt_scope"] == "current"]
    report = {
        "status": "STOPPED_FIRST_A_B_DIVERGENCE" if first_divergence else "N1_REPLAY_EXACT",
        "teacher_status": "DETERMINISTIC_TEACHER_VALIDATED",
        "distance_status": "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_IMPLEMENTATION_ORDER",
        "strict_nearest_owner": {
            "strict_decisions": strict, "exact_owner_agreement": strict,
            "exact_ties_deferred_to_oracle": ties,
            "comparison_routing_mismatches": sum(row["owner"]["comparison_errors"] for row in rows),
            "owner_mismatches": sum(row["owner"]["owner_errors"] for row in rows),
        },
        "representative_replay": rows,
        "first_divergence": first_divergence,
        "active_A_B_operations": reached,
        "remaining_D_oracles": deferred,
        "inventory_sha256_before": before_inventory,
        "inventory_sha256_after": sha256(INVENTORY),
        "checkpoints_unchanged": checkpoint_hashes() == expected_checkpoints,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "invocations.json").write_text(json.dumps(reached, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "strict_nearest_owner": report["strict_nearest_owner"],
        "systems": [{"system": row["system"], "proposal_errors": row["proposal_bit_mismatches"],
                     "continuous_errors": row["continuous"]["internal_state_errors"],
                     "continuous_winner_exact": row["continuous"]["winner_exact"]}
                    for row in rows],
        "first_divergence": first_divergence,
        "active_A_B_count": len(reached), "checkpoints_unchanged": report["checkpoints_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
