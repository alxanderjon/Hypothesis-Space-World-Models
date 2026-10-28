"""Compose the fixed deterministic N1 interior-pool components at N=8600.

This is validation only.  It does not train, overwrite, or promote any input
operator.  The historical Sobol-recovery gates are copied by calling their
existing metric functions and using their saved baseline tables.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.sho.audit_online_quota_reservoir import (
    A,
    META,
    N,
    OUT as RESERVOIR_OUT,
    capacity,
    guarded_buffer,
    ids,
)
from experiments.sho.evaluate_progressive_spacefill_smoke import (
    metrics,
    overlap,
)
from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.progressive_spacefill_generator import (
    SCALES,
    boundary_faces,
    discrepancy_rows,
)
from experiments.sho.train_dyadic_spacefill_curriculum import health
from experiments.sho.train_geometry_allocation_curriculum import (
    GeometryAllocator,
    feature_tensor,
)
from experiments.sho.validate_geometry_preserving_quota_realizer import realize
from experiments.sho.validate_joint_quota_assignment_teacher import bounds, teacher


ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "complete_deterministic_probe_generator"
ALLOCATOR_CHECKPOINT = ROOT / "geometry_allocation_curriculum/stage_4_best.pt"
BASELINE = ROOT / "progressive_spacefill_smoke"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key, value in tuple(row.items()):
            if value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def allocator_record(boxes: list[tuple[np.ndarray, np.ndarray]], budget: int) -> dict:
    lower = np.asarray([a for a, _ in boxes])
    upper = np.asarray([b for _, b in boxes])
    volume = np.prod(upper - lower, axis=1)
    return dict(
        lower=lower,
        upper=upper,
        parent_lower=lower.min(0),
        parent_upper=upper.max(0),
        target=volume / volume.sum(),
        budget=int(budget),
        dimension=2,
        name="composed_quadtree",
        kind="validation",
    )


def regional_targets(model: GeometryAllocator) -> tuple[np.ndarray, list[dict]]:
    """Frozen allocator on a generic four-child hierarchy, four levels deep."""
    nodes = [(N, np.zeros(2), np.ones(2), ())]
    audit = []
    for level in range(4):
        following = []
        for budget, lo, hi, path in nodes:
            mid = (lo + hi) / 2
            children = []
            for bits in np.ndindex(2, 2):
                child_lo = np.where(bits, mid, lo)
                child_hi = np.where(bits, hi, mid)
                children.append((child_lo, child_hi))
            record = allocator_record(children, budget)
            features, mask, _ = feature_tensor([record])
            with torch.no_grad():
                fractions = model(features, mask).numpy()[0, :4]
            count = conserve(budget, fractions)
            assert int(count.sum()) == budget
            audit.append(
                dict(
                    level=level + 1,
                    path=list(path),
                    parent_budget=budget,
                    fractions=fractions.tolist(),
                    child_counts=count.tolist(),
                )
            )
            for index, ((child_lo, child_hi), child_count) in enumerate(zip(children, count)):
                following.append((int(child_count), child_lo, child_hi, path + (index,)))
        nodes = following
    # Convert quadtree child order to canonical 16x16 incidence-cell order.
    target = np.zeros(256, dtype=np.int64)
    for budget, lo, _, _ in nodes:
        ij = np.minimum((lo * 16).astype(np.int64), 15)
        target[np.ravel_multi_index(tuple(ij), (16, 16))] = budget
    assert target.sum() == N and np.all(target >= 0)
    return target, audit


def complete_pool(interior: np.ndarray) -> np.ndarray:
    boundary = boundary_faces(len(interior), interior.shape[1])
    complete = np.unique(np.concatenate((interior, boundary)), axis=0)
    assert np.array_equal(
        complete[np.any((complete == 0) | (complete == 1), axis=1)], boundary
    )
    return complete


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    protected_paths = list(ROOT.rglob("*.pt"))
    for folder in (
        ROOT / "joint_quota_assignment_teacher",
        ROOT / "geometry_preserving_quota_realizer",
        RESERVOIR_OUT,
    ):
        protected_paths.extend(path for path in folder.iterdir() if path.is_file())
    protected = {str(path): sha(path) for path in protected_paths}

    allocator = GeometryAllocator()
    allocator.load_state_dict(
        torch.load(ALLOCATOR_CHECKPOINT, weights_only=False)["state_dict"]
    )
    allocator.eval().requires_grad_(False)
    fine_target, allocation_trace = regional_targets(allocator)
    locally_conserved_counts = A @ fine_target

    # The joint teacher is the single global integer-realization stage.  Record
    # why the older independent per-parent conserve result cannot also be
    # imposed as a second target, but do not feed that diagnostic into runtime.
    historical_lower, historical_upper = bounds(META, N)
    conflict_ids = np.flatnonzero(
        (locally_conserved_counts < historical_lower)
        | (locally_conserved_counts > historical_upper)
    )
    compatibility_conflicts = []
    for row in conflict_ids:
        meta = META[int(row)]
        compatibility_conflicts.append(
            dict(
                constraint_id=int(row),
                family=meta["family"],
                cell=int(meta["cell"]),
                tier=int(meta["tier"]),
                historical_lower=int(historical_lower[row]),
                historical_upper=int(historical_upper[row]),
                regional_forced_count=int(locally_conserved_counts[row]),
                signed_error_from_nearest_bound=int(
                    locally_conserved_counts[row]
                    - np.clip(
                        locally_conserved_counts[row],
                        historical_lower[row],
                        historical_upper[row],
                    )
                ),
            )
        )
    if compatibility_conflicts:
        with (OUT / "regional_joint_quota_conflicts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=compatibility_conflicts[0].keys())
            writer.writeheader()
            writer.writerows(compatibility_conflicts)

    prior = json.loads((RESERVOIR_OUT / "results.json").read_text())
    captured = np.load(RESERVOIR_OUT / "captured_pools.npz")
    boundary = boundary_faces(N, 2)
    pools = {}
    records = []
    quota_traces = []
    for screen in prior["screens"]:
        condition = screen["condition"]
        base = captured[condition + "_base"]
        raw = captured[condition + "_raw"]
        reservoir = guarded_buffer(base, raw)
        class_ids = ids(reservoir)
        available = capacity(reservoir)
        quotas, certificate = teacher(
            A,
            available,
            np.zeros(len(META), dtype=np.int64),
            historical_lower,
            historical_upper,
            N,
        )
        if quotas is None:
            result = dict(
                status="STOPPED_REGIONAL_JOINT_QUOTA_INFEASIBLE",
                condition=condition,
                certificate=certificate,
            )
            (OUT / "results.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return
        trace_path = OUT / f"{condition}_geometry_trace.jsonl.gz"
        selected = realize(reservoir, class_ids, quotas, boundary, trace_path)
        interior = reservoir[selected]
        assert len(interior) == N
        assert len(np.unique(interior, axis=0)) == N
        assert np.isfinite(interior).all()
        assert np.array_equal(capacity(interior), quotas)
        realized_counts = A @ capacity(interior)
        assert np.all(realized_counts >= historical_lower)
        assert np.all(realized_counts <= historical_upper)
        pools[condition] = interior
        records.append(
            dict(
                condition=condition,
                archived=bool(screen["archived"]),
                reservoir_count=len(reservoir),
                interior_count=len(interior),
                quota_error=int(np.abs(capacity(interior) - quotas).max()),
                projection_error=int(
                    np.maximum(
                        historical_lower - realized_counts,
                        realized_counts - historical_upper,
                    ).max(initial=0)
                ),
                duplicate_count=N - len(np.unique(interior, axis=0)),
                finite=bool(np.isfinite(interior).all()),
                trace=str(trace_path),
            )
        )
        quota_traces.append(
            dict(
                condition=condition,
                available=available.tolist(),
                fixed_quota=quotas.tolist(),
                constraint_lower=historical_lower.tolist(),
                constraint_upper=historical_upper.tolist(),
            )
        )
        print(json.dumps(dict(event="pool", **records[-1])), flush=True)

    # Exact reproducibility of the full selection, including tie behavior.
    first = prior["screens"][0]["condition"]
    reservoir = guarded_buffer(captured[first + "_base"], captured[first + "_raw"])
    replay_path = OUT / "reproducibility_replay.jsonl.gz"
    replay_ids = realize(
        reservoir,
        ids(reservoir),
        np.asarray(quota_traces[0]["fixed_quota"], dtype=np.int64),
        boundary,
        replay_path,
    )
    deterministic = np.array_equal(reservoir[replay_ids], pools[first])
    assert deterministic

    # Existing evaluation reference, unchanged.
    grid = (np.arange(512) + 0.5) / 512
    reference = np.vstack(
        (
            np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2),
            np.random.default_rng(27093001).random((262144, 2)),
        )
    )
    geometry = []
    multiscale = []
    for condition, interior in pools.items():
        iteration = int(condition.removeprefix("iteration").split("_")[0])
        pool_type = condition.rsplit("_", 1)[-1]
        complete = complete_pool(interior)
        geometry.append(
            dict(
                N=N,
                generator="deterministic_composed",
                iteration=iteration,
                pool_type=pool_type,
                **metrics(complete, interior, reference, N),
            )
        )
        for row in discrepancy_rows(interior):
            multiscale.append(
                dict(
                    N=N,
                    generator="deterministic_composed",
                    iteration=iteration,
                    pool_type=pool_type,
                    **row,
                )
            )

    archived_names = [f"iteration{100 + index}_cell" for index in range(4)] + [
        "iteration100_greedy"
    ]
    fresh_names = ["iteration12000_cell", "iteration12001_cell", "iteration12000_greedy"]
    freshness = []
    for cohort, names, pairs in (
        ("archived", archived_names, ((0, 1), (1, 2), (2, 3), (0, 4))),
        ("fresh", fresh_names, ((0, 1), (0, 2))),
    ):
        for left, right in pairs:
            freshness.append(
                dict(
                    cohort=cohort,
                    left=names[left],
                    right=names[right],
                    **overlap(pools[names[left]], pools[names[right]], N),
                )
            )

    # Exact historical progressive smoke gate, restricted to its N=8600 cohort.
    baseline_geometry = read_csv(BASELINE / "progressive_spacefill_geometry.csv")
    baseline_multiscale = read_csv(BASELINE / "progressive_spacefill_multiscale.csv")
    baseline_freshness = read_csv(BASELINE / "progressive_spacefill_freshness_union.csv")
    archived_geometry = [row for row in geometry if int(row["iteration"]) in (100, 101, 102, 103)]
    archived_multiscale = [row for row in multiscale if int(row["iteration"]) in (100, 101, 102, 103)]
    baseline_geometry = [row for row in baseline_geometry if int(row["N"]) == N]
    baseline_multiscale = [row for row in baseline_multiscale if int(row["N"]) == N]
    baseline_freshness = [row for row in baseline_freshness if int(row["N"]) == N]

    scales = {}
    for generator in ("Sobol", "old_spacefill", "fresh_v1"):
        scales[generator] = {}
        for scale in SCALES:
            values = [
                float(row["grid_anchored_discrepancy"])
                for row in baseline_multiscale
                if row["generator"] == generator and int(row["bins_per_axis"]) == scale
            ]
            scales[generator][scale] = max(values)
    scales["deterministic_composed"] = {
        scale: max(
            row["grid_anchored_discrepancy"]
            for row in archived_multiscale
            if int(row["bins_per_axis"]) == scale
        )
        for scale in SCALES
    }

    def one_count_ratio(generator: str) -> float:
        return max(
            scales[generator][scale] / max(scales["Sobol"][scale], 1 / N)
            for scale in SCALES
        )

    sobol_geometry = [row for row in baseline_geometry if row["generator"] == "Sobol"]
    sobol_cover = max(float(row["complete_cover"]) for row in sobol_geometry)
    current_cover = max(float(row["complete_cover"]) for row in archived_geometry)
    paired_cover = []
    for row in archived_geometry:
        baseline_row = next(
            old
            for old in sobol_geometry
            if int(old["iteration"]) == int(row["iteration"])
            and old["pool_type"] == row["pool_type"]
        )
        paired_cover.append(float(row["complete_cover"]) / float(baseline_row["complete_cover"]))
    current_boundary = max(float(row["boundary_cover"]) for row in archived_geometry)
    sobol_boundary = max(float(row["boundary_cover"]) for row in sobol_geometry)
    current_holes = max(int(row["large_hole_count"]) for row in archived_geometry)
    sobol_holes = max(int(float(row["large_hole_count"])) for row in sobol_geometry)
    current_nn_cv = float(np.mean([float(row["nn_CV"]) for row in archived_geometry]))
    current_cluster = max(float(row["clustered_fraction"]) for row in archived_geometry)
    archived_overlap = [row for row in freshness if row["cohort"] == "archived"]

    current_union = np.unique(
        np.concatenate([complete_pool(pools[name]) for name in archived_names[:4]]), axis=0
    )
    current_union_cover = float(cKDTree(current_union).query(reference)[0].max())
    sobol_union = next(
        row
        for row in baseline_freshness
        if row["generator"] == "Sobol" and row["record_type"] == "union"
    )
    progressive_gate = dict(
        discrepancy_materially_better=all(
            one_count_ratio("deterministic_composed")
            <= 0.8 * one_count_ratio(generator)
            for generator in ("old_spacefill", "fresh_v1")
        ),
        no_scale_regression=all(
            scales["deterministic_composed"][scale]
            <= 1.1 * scales["fresh_v1"][scale]
            for scale in SCALES
        ),
        coverage=current_cover <= 1.10 * sobol_cover and max(paired_cover) <= 1.15,
        boundary=current_boundary == sobol_boundary,
        freshness=max(row["exact_overlap_fraction"] for row in archived_overlap) == 0
        and max(row["near_overlap_fraction_005_spacing"] for row in archived_overlap) <= 0.03,
        union=current_union_cover <= 1.25 * float(sobol_union["union_complete_cover"]),
        holes=current_holes <= sobol_holes + 0.001 * len(reference),
        not_rigid_or_clustered=current_nn_cv >= 0.08 and current_cluster <= 0.005,
    )
    progressive_gate["passed"] = all(progressive_gate.values())

    # Existing dyadic curriculum safeguard, separately for archived and fresh cohorts.
    prior_groups = {row["group"]: row for row in prior["historical_cohort_gates"]}
    cohort_health = []
    for cohort, names, pairs in (
        ("archived", archived_names, ((0, 1), (1, 2), (2, 3), (0, 4))),
        ("fresh", fresh_names, ((0, 1), (0, 2))),
    ):
        rows = [next(row for row in geometry if int(row["iteration"]) == int(name.removeprefix("iteration").split("_")[0]) and row["pool_type"] == name.rsplit("_", 1)[-1]) for name in names]
        pair_rows = [
            overlap(pools[names[left]], pools[names[right]], N) for left, right in pairs
        ]
        value = dict(
            cover=max(row["complete_cover"] for row in rows),
            hole_fraction=max(row["large_hole_fraction"] for row in rows),
            min_nn_CV=min(row["nn_CV"] for row in rows),
            nn_p01=min(
                float(np.quantile(cKDTree(pools[name]).query(pools[name], k=2)[0][:, 1], 0.01))
                for name in names
            ),
            clustering=max(row["clustered_fraction"] for row in rows),
            exact_overlap=max(row["exact_overlap_fraction"] for row in pair_rows),
            near_overlap=max(row["near_overlap_fraction_005_spacing"] for row in pair_rows),
        )
        entry = prior_groups[cohort]["entry"]
        cohort_health.append(dict(cohort=cohort, value=value, entry=entry, passed=health(value, entry)))

    first_failure = None
    if not progressive_gate["passed"]:
        first_failure = "historical_progressive_spacefill_gate"
    elif not all(row["passed"] for row in cohort_health):
        first_failure = "historical_dyadic_geometry_health"

    unchanged = all(sha(Path(path)) == digest for path, digest in protected.items())
    assert unchanged
    status = "VALIDATED_TEACHER" if first_failure is None else "STOPPED_HISTORICAL_GATE_FAILURE"
    result = dict(
        status=status,
        first_failed_stage=first_failure,
        pipeline=[
            "locked stage-4 proportional regional mass allocator",
            "unchanged conditioned proposal streams and guarded reservoirs",
            "fixed joint quota assignment teacher as the single global integer realizer",
            "fixed geometry-preserving quota realizer",
            "unchanged boundary support, np.unique deduplication and ordering",
        ],
        N=N,
        conditions=len(pools),
        allocation_trace=allocation_trace,
        local_conserve_target_diagnostic=fine_target.tolist(),
        local_conserve_joint_conflict_count=len(compatibility_conflicts),
        local_conserve_note=(
            "Diagnostic only: independent per-parent integer conservation is not imposed "
            "after the global joint integer-realization teacher"
        ),
        records=records,
        quota_traces=quota_traces,
        deterministic_replay=deterministic,
        geometry=geometry,
        multiscale=multiscale,
        freshness=freshness,
        anchored_scale_maxima=scales,
        worst_one_count_floor_ratio={
            generator: one_count_ratio(generator)
            for generator in ("Sobol", "old_spacefill", "fresh_v1", "deterministic_composed")
        },
        historical_progressive_gate=progressive_gate,
        cohort_health=cohort_health,
        downstream_n1_run=False,
        downstream_skip_reason=(
            None if first_failure is None else "Required stop at first failed historical probe-generator gate"
        ),
        protected_files=len(protected),
        protected_unchanged=unchanged,
        runtime_seconds=time.monotonic() - started,
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    (OUT / "regional_allocation_trace.json").write_text(json.dumps(allocation_trace, indent=2))
    (OUT / "quota_assignment_trace.json").write_text(json.dumps(quota_traces, indent=2))
    np.savez_compressed(OUT / "interior_pools.npz", **pools)
    (OUT / "protected_hashes.json").write_text(json.dumps(protected, indent=2))
    print(
        json.dumps(
            dict(
                status=status,
                first_failed_stage=first_failure,
                historical_progressive_gate=progressive_gate,
                cohort_health=[dict(cohort=row["cohort"], passed=row["passed"]) for row in cohort_health],
                runtime_seconds=result["runtime_seconds"],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
