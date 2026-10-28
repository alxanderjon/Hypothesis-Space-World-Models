#!/usr/bin/env python3
"""Learn half-spacing only; assemble two max reductions from frozen comparison."""
from __future__ import annotations

import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.validate_stage1b_nearest_argmin as nearest

base, margin = nearest.base, nearest.margin
OUT = Path("outputs/sho/stage1B/learned_cell_size_assembled")
SMOKE = OUT / "half_spacing_smoke.csv"
FULL = OUT / "cell_size_assembled_validation.csv"
HASHES = OUT / "cell_size_checkpoint_hash_summary.csv"
SEED = 7_071_067


def capture_nearest(sites, probes, distance, frozen, comparator, state):
    """Observe the locked scan's return frame; do not replace any computation."""
    captured = {}
    code = nearest.scan.__code__
    previous_profile = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is code:
            for name in ("winner", "winner_distance", "teacher_owner", "teacher_distance"):
                if name in frame.f_locals:
                    captured[name] = frame.f_locals[name].copy()

    try:
        sys.setprofile(observe)
        result = nearest.scan(sites, probes, distance, frozen, comparator, state)
    finally:
        sys.setprofile(previous_profile)
    assert len(captured) == 4
    scan_rows = [r for r in result[0] if r["record_type"] == "scan_state"]
    assert all(r["pairwise_mistakes"] == 0 and r["final_argmin_mistakes"] == 0
               and r["authoritative_tie_detector_mismatches"] == 0 for r in scan_rows)
    return captured, scan_rows


def predict_half(model, values):
    with torch.no_grad():
        return model(torch.from_numpy(values[:, None])).numpy()[:, 0]


def regression_metrics(model, values):
    target = 0.5*values  # Teacher only; inference always uses the learned weight.
    predicted = predict_half(model, values)
    error = predicted-target
    return dict(count=len(values), rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))), maximum_absolute_error=float(np.max(np.abs(error))),
        correlation=float(np.corrcoef(target, predicted)[0, 1]),
        exact_value_mismatches=int(np.sum(predicted != target)))


def train_half(training, validation):
    torch.manual_seed(SEED)
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    initial = float(model.weight.detach().item())
    assert initial != 0.5
    x = torch.from_numpy(training[:, None])
    y = 0.5*x
    hessian = 2*torch.mean(x*x)
    history = []
    for step in range(1, 13):
        model.zero_grad(set_to_none=True)
        loss = torch.mean((model(x)-y)**2)
        loss.backward()
        with torch.no_grad():
            model.weight.sub_(model.weight.grad/hessian)
        metrics = regression_metrics(model, training)
        history.append(dict(record_type="training_step", step=step,
            learned_w=float(model.weight.item()), training_mse=metrics["rmse"]**2))
        if metrics["maximum_absolute_error"] == 0.0:
            break
    model.eval().requires_grad_(False)
    rows = [dict(record_type="half_spacing_summary", learned_w=float(model.weight.item()), initial_w=initial,
        dtype="float64", bias=False, seed=SEED, steps=step,
        optimizer="full_batch_residual_Newton", objective="continuous_mean_squared_error",
        weight_projection=False)]
    for split, values in (("fitting", training), ("validation", validation)):
        rows.append(dict(record_type="half_spacing_metrics", split=split, **regression_metrics(model, values)))
    passed = rows[2]["maximum_absolute_error"] <= 1e-15 and rows[2]["rmse"] <= 1e-15
    rows[0]["status"] = "passed" if passed else "failed"
    return model, rows+history, passed


def larger_candidate(frozen, comparator, current, candidate):
    """Exact equality keeps current; no new learned comparison parameters."""
    tied = current == candidate
    replace = np.zeros(len(current), dtype=bool)
    active = ~tied
    scale = np.maximum(current[active], candidate[active])
    assert np.all(scale > 0)
    inputs = np.column_stack((current[active]/scale, candidate[active]/scale))
    m = margin.predict(frozen, inputs)
    logits = float(comparator.alpha())*(m-float(comparator.theta()))
    replace[active] = logits < 0.0
    invalid = int(np.sum((m == 0) | (logits == 0) | ~np.isfinite(logits)))
    # Teacher comparison is validation only, never a decision input.
    mistakes = int(np.sum(replace != (candidate > current)))
    return replace, int(tied.sum()), invalid, mistakes


def assigned_radius(owners, distances, n, frozen, comparator):
    counts = np.bincount(owners, minlength=n)
    order = np.argsort(owners, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    radius = np.zeros(n, dtype=np.float64)
    witness = np.full(n, -1, dtype=np.int64)
    nonempty = np.flatnonzero(counts > 0)
    first = order[offsets[nonempty]]
    radius[nonempty], witness[nonempty] = distances[first], first
    ties = mistakes = invalid = comparisons = 0
    for position in range(1, int(counts.max())):
        active = np.flatnonzero(counts > position)
        incoming = order[offsets[active]+position]
        replace, t, bad, wrong = larger_candidate(frozen, comparator, radius[active], distances[incoming])
        chosen = active[replace]
        radius[chosen], witness[chosen] = distances[incoming[replace]], incoming[replace]
        ties += t
        invalid += bad
        mistakes += wrong
        comparisons += len(active)
    return radius, witness, dict(radius_comparisons=comparisons, radius_equal_tie_fallbacks=ties,
        radius_pairwise_mistakes=mistakes, radius_invalid_logits=invalid, empty_sites=int(np.sum(counts == 0)))


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if SMOKE.exists() or FULL.exists():
        raise RuntimeError("Refusing to overwrite an existing cell-size run")
    with nearest.SUMMARY.open() as handle:
        protected = [Path(r["checkpoint_path"]) for r in csv.DictReader(handle) if r["checkpoint_path"]]
    operator = torch.load(nearest.OPERATOR, map_location="cpu", weights_only=False)
    protected += [Path(operator["implementation_path"]), nearest.THRESHOLD.with_suffix(".sha256"), nearest.OPERATOR.with_suffix(".sha256")]
    hashes = {str(p):base._sha256(p) for p in protected}
    assert operator["status"] == "LOCKED"
    assert hashes[str(Path(operator["implementation_path"]))] == operator["implementation_sha256"]
    for name in ("distance", "margin", "threshold"):
        assert base._sha256(Path(operator[name+"_checkpoint"])) == operator[name+"_sha256"]
    distance, _ = base._load_locked_distance()
    frozen = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    frozen.load_state_dict(torch.load(operator["margin_checkpoint"], weights_only=False)["state_dict"])
    frozen.eval().requires_grad_(False)
    comparator = nearest.threshold.NonzeroThreshold()
    comparator.load_state_dict(torch.load(operator["threshold_checkpoint"], weights_only=False)["state_dict"])
    comparator.eval().requires_grad_(False)
    payload = torch.load(base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    states = operator["held_out_state_indices"]
    fit, validation, test = states[:16], states[16:32], states[32:]
    assert not (set(fit)&set(validation) or set(fit)&set(test) or set(validation)&set(test))
    spacing_parts = {"fitting":[], "validation":[]}
    for split, subset in (("fitting", fit), ("validation", validation)):
        for pos, state in enumerate(subset, 1):
            sites, probes, _ = base.distance_metric._state_geometry(payload, trajectories, state)
            # Site-spacing queries are independent of the other query points.
            # One real probe keeps the unchanged scan's probe diagnostics valid.
            captured, _ = capture_nearest(sites, probes[:1], distance, frozen, comparator, state)
            spacing_parts[split].append(captured["winner_distance"][1:])
            if pos % 4 == 0:
                print(f"half-spacing data {split} {pos}/{len(subset)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    half, smoke_rows, passed = train_half(np.concatenate(spacing_parts["fitting"]), np.concatenate(spacing_parts["validation"]))
    smoke_rows[0].update(fitting_state_indices=json.dumps(fit), validation_state_indices=json.dumps(validation),
        untouched_final_test_state_indices=json.dumps(test), nearest_operator_sha256=base._sha256(nearest.OPERATOR))
    base._write_csv(SMOKE, smoke_rows)
    print(f"HALF SPACING: {smoke_rows[:3]}", flush=True)
    if not passed:
        base._write_csv(FULL, [dict(record_type="gate_stop", status="not_run", reason="half_spacing_not_at_numerical_precision")])
        assert all(base._sha256(Path(p)) == h for p,h in hashes.items())
        base._write_csv(HASHES, [dict(record_type="gate_stop", status="FAILED", new_checkpoints_saved=False)])
        return

    rows, downstream_states, spacing_by_split = [], [], {"fitting":[], "validation":[], "untouched_test":[]}
    for position, state in enumerate(states, 1):
        split = "fitting" if state in fit else "validation" if state in validation else "untouched_test"
        sites, probes, record = base.distance_metric._state_geometry(payload, trajectories, state)
        captured, upstream_rows = capture_nearest(sites, probes, distance, frozen, comparator, state)
        n, p = len(sites), len(probes)
        owner, learned_d = captured["winner"][:p], captured["winner_distance"][:p]
        spacing = captured["winner_distance"][p:]
        spacing_by_split[split].append(spacing)
        radius, witness, counts = assigned_radius(owner, learned_d, n, frozen, comparator)
        teacher_owner, teacher_d = captured["teacher_owner"][:p], captured["teacher_distance"][:p]
        teacher_radius = np.zeros(n, dtype=np.float64)
        np.maximum.at(teacher_radius, teacher_owner, teacher_d)
        teacher_witness = np.full(n, -1, dtype=np.int64)
        # Validation witness: first probe at the exact teacher maximum.
        for q in np.flatnonzero(teacher_d == teacher_radius[teacher_owner]):
            if teacher_witness[teacher_owner[q]] < 0:
                teacher_witness[teacher_owner[q]] = q
        half_values = predict_half(half, spacing)
        teacher_half = 0.5*captured["teacher_distance"][p:]
        replace, ties, invalid, mistakes = larger_candidate(frozen, comparator, radius, half_values)
        final = np.where(replace, half_values, radius)
        teacher_replace = teacher_half > teacher_radius  # equality keeps radius
        teacher_final = np.maximum(teacher_radius, teacher_half)
        row = dict(record_type="assembled_state", source_state_index=state, split=split, site_count=n, probe_count=p,
            **counts, upstream_comparisons=sum(r["comparisons"] for r in upstream_rows),
            upstream_owner_mistakes=sum(r["final_argmin_mistakes"] for r in upstream_rows),
            upstream_pairwise_mistakes=sum(r["pairwise_mistakes"] for r in upstream_rows),
            assigned_radius_mismatches=int(np.sum(radius != teacher_radius)),
            assigned_radius_max_error=float(np.max(np.abs(radius-teacher_radius))),
            farthest_witness_mismatches=int(np.sum(witness != teacher_witness)),
            half_spacing_mismatches=int(np.sum(half_values != teacher_half)),
            half_spacing_max_error=float(np.max(np.abs(half_values-teacher_half))),
            final_equal_tie_fallbacks=ties, final_invalid_logits=invalid, final_pairwise_mistakes=mistakes,
            final_max_winner_changes=int(np.sum(replace != teacher_replace)),
            final_cell_size_mismatches=int(np.sum(final != teacher_final)),
            final_cell_size_max_error=float(np.max(np.abs(final-teacher_final))))
        rows.append(row)
        losses = np.asarray(trajectories[int(record["trajectory_id"])]["evaluated_prefix_losses"][:n], dtype=np.float64)
        downstream_states.append(dict(source_state_index=state, trajectory_id=int(record["trajectory_id"]),
            iteration=int(record["iteration"]), state_bin=base.turn._state_bin(n), losses=losses,
            teacher_sizes=teacher_final, learned_sizes=final))
        base._write_csv(FULL, [dict(record_type="progress", status="running", completed_states=position)]+rows)
        print(f"cell-size {position}/{len(states)} state={state} radius_errors={row['assigned_radius_mismatches']} "
            f"half_errors={row['half_spacing_mismatches']} winner_errors={row['final_max_winner_changes']} "
            f"size_errors={row['final_cell_size_mismatches']} elapsed={time.monotonic()-started:.1f}s", flush=True)

    locked_downstream = nearest.downstream._load_downstream()
    downstream_rows, downstream_summary = nearest.downstream._downstream_validation(downstream_states, locked_downstream)
    for split, parts in spacing_by_split.items():
        smoke_rows.append(dict(record_type="full_pool_half_spacing_metrics", split=split,
                              **regression_metrics(half, np.concatenate(parts))))
    smoke_rows.append(dict(record_type="full_pool_half_spacing_metrics", split="all_80_upstream_held_out_states",
                          **regression_metrics(half, np.concatenate([v for parts in spacing_by_split.values() for v in parts]))))
    summary = dict(record_type="assembly_summary", state_count=len(states), untouched_test_state_count=len(test),
        tie_routing=nearest.SCAFFOLD, new_comparator_trained=False, learned_half_spacing_w=float(half.weight.item()))
    for key in rows[0]:
        if key not in ("record_type", "source_state_index", "split"):
            summary[key] = max(r[key] for r in rows) if key.endswith("max_error") else sum(r[key] for r in rows)
    gates = ("upstream_owner_mistakes", "upstream_pairwise_mistakes", "assigned_radius_mismatches",
             "farthest_witness_mismatches", "radius_pairwise_mistakes", "radius_invalid_logits",
             "half_spacing_mismatches", "final_invalid_logits", "final_pairwise_mistakes",
             "final_max_winner_changes", "final_cell_size_mismatches")
    passed = all(summary[k] == 0 for k in gates) and downstream_summary["exact_complete_downstream_state_count"] == len(states)
    summary["status"] = "LOCKED" if passed else "FAILED"
    summary["elapsed_seconds"] = time.monotonic()-started
    assert all(base._sha256(Path(p)) == h for p,h in hashes.items())
    summary["protected_files_verified_unchanged"] = len(hashes)
    hash_rows = [dict(record_type="existing_dependency", checkpoint_path=p, sha256=h, unchanged=True) for p,h in sorted(hashes.items())]
    if passed:
        implementation_hash = base._sha256(Path(__file__))
        dependencies = dict(nearest_operator_path=str(nearest.OPERATOR), nearest_operator_sha256=base._sha256(nearest.OPERATOR),
            comparator_margin_path=operator["margin_checkpoint"], comparator_margin_sha256=operator["margin_sha256"],
            comparator_threshold_path=operator["threshold_checkpoint"], comparator_threshold_sha256=operator["threshold_sha256"])
        new_artifacts = {}
        for component in ("half_spacing_operator", "assigned_radius_operator", "cell_size_operator"):
            path = OUT / (component+"_locked.pt")
            saved = dict(format_version=1, component=component, status="LOCKED", dtype="float64",
                tie_routing=nearest.SCAFFOLD, implementation_path=__file__, implementation_sha256=implementation_hash,
                dependencies=dependencies.copy(), validation_summary=summary, downstream_summary=downstream_summary,
                fitting_state_indices=fit, validation_state_indices=validation, untouched_final_test_state_indices=test)
            if component == "half_spacing_operator":
                saved.update(architecture="h_hat=w*s_nearest; no bias", state_dict=copy.deepcopy(half.state_dict()),
                             learned_w=float(half.weight.item()), training_summary=smoke_rows[0])
            elif component == "assigned_radius_operator":
                saved.update(algorithm="stable assigned-probe scan; locked comparator logit<0 replaces",
                             empty_radius=0.0, tie_rule="equal distances keep first probe witness")
            else:
                saved.update(algorithm="locked comparator logit<0 selects learned half-spacing over learned assigned radius",
                             tie_rule="exact equality keeps assigned-radius branch", component_checkpoints=new_artifacts.copy())
            torch.save(saved, path)
            digest = base._sha256(path)
            path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
            new_artifacts[component] = dict(path=str(path), sha256=digest)
            hash_rows.append(dict(record_type="new_checkpoint", component=component, status="LOCKED", checkpoint_path=str(path), sha256=digest))
        reloaded = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        reloaded.load_state_dict(torch.load(new_artifacts["half_spacing_operator"]["path"], weights_only=False)["state_dict"])
        values = np.concatenate(spacing_by_split["untouched_test"])
        assert np.array_equal(predict_half(reloaded, values), predict_half(half, values))
    base._write_csv(SMOKE, smoke_rows)
    base._write_csv(FULL, [summary]+rows+downstream_rows)
    base._write_csv(HASHES, [summary]+hash_rows)
    print(json.dumps(dict(assembly=summary, downstream=downstream_summary), indent=2), flush=True)


if __name__ == "__main__":
    run()
