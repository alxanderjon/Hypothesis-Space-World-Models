"""Post-baseline exact composition of the N1 boundary-support path."""

from __future__ import annotations

import numpy as np
import torch

from experiments.geometry.boundary_face_resolution_assembly import (
    BoundaryFaceResolutionAssembly,
)
from experiments.resume_hybrid_student_after_ckdtree import (
    normalized_to_physical_assembly,
    scrambled_sobol_full_block,
)
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.assemble_n3_gradient_operator import LearnedArithmetic
from experiments.sho.validate_n1_boundary_merge_dedup import independent_unique
from experiments.sho.validate_sobol_batching_truncation import schedule as sobol_schedule


class N1BoundarySupportAssembly:
    """Compose existing locked components without the production boundary helper."""

    def __init__(self) -> None:
        self.resolution = BoundaryFaceResolutionAssembly()
        self.arithmetic = LearnedArithmetic()

    def _d2_grid(self, count: int) -> tuple[np.ndarray, dict[str, int]]:
        indices = np.arange(count, dtype=np.float64)
        with torch.no_grad():
            step = self.arithmetic.divide(
                torch.as_tensor(1.0, dtype=torch.float64),
                torch.as_tensor(float(count - 1), dtype=torch.float64),
            )
            values = self.arithmetic.multiply(
                torch.from_numpy(indices), step.expand(count)
            ).numpy()
        # Production linspace promises the exact endpoint independently of
        # rounded step multiplication.
        values[-1] = 1.0
        return values[:, None], {
            "division_calls": 1,
            "multiplication_calls": count,
            "fallback_calls": 0,
        }

    def __call__(
        self,
        dimension: int,
        requested: int,
        seed: int,
        lower=None,
        upper=None,
        *,
        log=None,
    ):
        d = int(dimension)
        lower_bound = (
            np.zeros(d, dtype=np.float64)
            if lower is None else np.asarray(lower, dtype=np.float64)
        )
        upper_bound = (
            np.ones(d, dtype=np.float64)
            if upper is None else np.asarray(upper, dtype=np.float64)
        )
        if (
            d < 1
            or lower_bound.shape != (d,)
            or upper_bound.shape != (d,)
            or not np.all(upper_bound > lower_bound)
        ):
            raise ValueError("invalid dimension or bounds")

        count, resolution_trace = self.resolution(requested, return_trace=True)
        if log is not None:
            log.call(
                "boundary face resolution", "B",
                "locked multiply/add/compare/select plus deterministic loop", 1,
            )

        grid_state = None
        sobol_state = None
        if d == 1:
            unit = np.asarray([[0.0], [1.0]], dtype=np.float64)
            identities = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
            route = "d1_endpoints"
        else:
            if d == 2:
                free, grid_state = self._d2_grid(count)
                route = "d2_locked_grid"
                if log is not None:
                    log.call(
                        "boundary d2 grid arithmetic", "B",
                        "locked division/multiplication plus endpoint storage", count,
                    )
            else:
                exponent, generated = sobol_schedule(count)
                full_block, sobol_state = scrambled_sobol_full_block(
                    d - 1, exponent, int(seed)
                )
                free = full_block[:count]
                route = "dimension_general_locked_sobol_faces"
                if log is not None:
                    log.call(
                        "boundary scrambled Sobol block/prefix", "B",
                        "current locked scramble/XOR/recurrence/scaling assembly",
                        generated,
                    )

            blocks = []
            labels = []
            for axis in reversed(range(d)):
                for side in (0, 1):
                    block = np.empty((count, d), dtype=np.float64)
                    block[:, axis] = side
                    block[:, np.arange(d) != axis] = free
                    blocks.append(block)
                    labels.extend([(axis, side)] * count)
            unit = np.concatenate(blocks, axis=0)
            identities = np.asarray(labels, dtype=np.int64)

        points, affine_state = normalized_to_physical_assembly(
            unit, lower_bound, upper_bound, log
        )
        if log is not None:
            log.call(
                "dimension-general boundary-face support", "B",
                "exact learned assembly with deterministic control/storage", 1,
            )
        return points, identities, {
            "route": route,
            "k": count,
            "unit": unit,
            "resolution_trace": resolution_trace,
            "grid_state": grid_state,
            "sobol_state": sobol_state,
            "affine_state": affine_state,
            "face_resolution_binding_invocations": 1,
            "grid_binding_invocations": int(d == 2),
            "sobol_binding_invocations": int(d > 2),
            "affine_binding_invocations": int(unit.size),
            "old_boundary_oracle_calls": 0,
            "forbidden_fallback_calls": 0,
        }


def scheduled_pool_with_boundary_assembly(
    dimension: int,
    sample_count: int,
    case_id: int,
    iteration: int,
    log,
    boundary: N1BoundarySupportAssembly | None = None,
):
    """Post-baseline pool entry point with no production boundary call."""
    requested = min(
        n1.teacher.MC_FACTOR_PER_SITE * sample_count,
        n1.teacher.MC_CANDIDATE_CAP,
    )
    exponent, generated = sobol_schedule(requested)
    seed = n1.teacher.SEARCH_SEED + 104729 * case_id + 1009 * iteration
    full_block, interior_state = scrambled_sobol_full_block(
        dimension, exponent, seed
    )
    interior = full_block[:requested]
    log.call(
        "Sobol full-block sizing and prefix", "B",
        "current locked schedule and exact first-N prefix", 1,
    )
    log.call(
        "scrambled Sobol coordinate generation", "B",
        "current locked scramble/XOR/recurrence/scaling assembly", generated,
    )

    boundary_operator = boundary or N1BoundarySupportAssembly()
    faces, identities, boundary_state = boundary_operator(
        dimension, requested, seed, log=log
    )
    raw_pool = np.concatenate((interior, faces), axis=0)
    merged, retained_indices = independent_unique(raw_pool)
    log.call(
        "pool merge, exact deduplication, lexicographic order", "B",
        "current independent_unique exact assembly", 1,
    )
    source_identities = np.concatenate((
        np.full((len(interior), 2), -1, dtype=np.int64), identities
    ))
    return merged, {
        "requested": requested,
        "seed": seed,
        "exponent": exponent,
        "generated": generated,
        "interior": interior,
        "interior_state": interior_state,
        "faces": faces,
        "face_identities": identities,
        "boundary_state": boundary_state,
        "raw_pool": raw_pool,
        "raw_source_identities": source_identities,
        "merge_retained_indices": retained_indices,
        "merged_source_identities": source_identities[retained_indices],
        "pool_merge_binding_invocations": 1,
        "old_boundary_oracle_calls": 0,
        "forbidden_fallback_calls": 0,
    }
