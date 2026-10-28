"""Distill canonical joint-quota assignment order; exact constraints stay structural.

The learned component is deliberately one scalar: a bias-free score on the
normalized canonical assignment ID.  A generic bounded-integer decoder applies
that learned ordering while enforcing the teacher's incidence, capacity,
budget, target, and frozen-row constraints exactly.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sho import validate_joint_quota_assignment_teacher as fixed

ROOT = Path("outputs/sho/stage1B")
SOURCE = ROOT / "joint_quota_assignment_teacher"
OUT = ROOT / "joint_quota_assignment_operator"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class CanonicalPriority(torch.nn.Module):
    """One learned parameter; lower canonical IDs must receive higher score."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.375], dtype=torch.float64))

    def forward(self, normalized_id: torch.Tensor) -> torch.Tensor:
        return normalized_id * self.weight


def ordered_canonical(A, lo, hi, cap, order):
    """Teacher's exact integer decoder with order supplied by learned scores."""
    cap = np.asarray(cap, dtype=np.int64)
    lb = np.zeros(len(cap), dtype=np.int64)
    ub = cap.copy()
    solution = fixed.solve(np.zeros(len(cap)), A, lo, hi, lb, ub)
    if solution is None:
        return None
    for j in order:
        j = int(j)
        if lb[j] == ub[j]:
            continue
        if solution[j] != ub[j]:
            cost = np.zeros(len(cap), dtype=np.int64)
            cost[j] = -1
            solution = fixed.solve(cost, A, lo, hi, lb, ub)
            assert solution is not None
        lb[j] = ub[j] = solution[j]
    return solution


def learned_order(model: CanonicalPriority, count: int) -> np.ndarray:
    ids = torch.arange(count, dtype=torch.float64)
    x = ids / max(1, count - 1)
    with torch.no_grad():
        score = model(x).numpy()
    # Stable canonical ID is retained only as an exact-score fallback.
    return np.lexsort((np.arange(count), -score))


def operator(model, incidence, capacity, current, lower, upper, remaining, frozen=None):
    I = np.asarray(incidence, dtype=np.int64)
    cap = np.asarray(capacity, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    lower = np.asarray(lower, dtype=np.int64)
    upper = np.asarray(upper, dtype=np.int64)
    frozen = np.zeros(len(current), bool) if frozen is None else np.asarray(frozen, bool)
    assert set(np.unique(I)).issubset({0, 1}) and np.all(cap >= 0)
    assert np.all((current[frozen] >= lower[frozen]) & (current[frozen] <= upper[frozen]))
    lo = lower - current
    hi = upper - current
    lo[frozen] = hi[frozen] = 0
    if cap.sum() < remaining or np.any(current > upper):
        return None
    if remaining == int(cap.sum()):
        counts = current + I @ cap
        return cap if np.all((counts >= lower) & (counts <= upper)) and np.all((I @ cap)[frozen] == 0) else None
    A = np.vstack((np.ones(len(cap), dtype=np.int64), I))
    return ordered_canonical(A, np.r_[remaining, lo], np.r_[remaining, hi], cap,
                             learned_order(model, len(cap)))


def read_trace(path: Path):
    with gzip.open(path, "rt") as stream:
        initial = json.loads(next(stream))
        actions = np.asarray([json.loads(line)["selected_assignment"] for line in stream], dtype=np.int64)
    return initial, actions


def split_name(path: Path) -> str:
    name = path.stem.removesuffix(".jsonl")
    if name.startswith("fresh"):
        return "fresh"
    if name in {"historical_reservoir_nested", "historical_reservoir_fine_2d_c"}:
        return "heldout"
    return "train"


def extract_supervision(paths):
    trajectory = []
    step = []
    action = []
    remaining = []
    splits = []
    pair_delta = []
    for trajectory_id, path in enumerate(paths):
        initial, actions = read_trace(path)
        n = len(initial["capacities"])
        split = split_name(path)
        for t, selected in enumerate(actions):
            trajectory.append(trajectory_id)
            step.append(t)
            action.append(selected)
            remaining.append(len(actions) - t)
            splits.append({"train": 0, "heldout": 1, "fresh": 2}[split])
            # Exact trace action paired with later canonical IDs.  These are
            # preference labels, while feasibility remains a structural input.
            if split == "train" and selected + 1 < n:
                for later in np.linspace(selected + 1, n - 1, min(8, n - selected - 1), dtype=int):
                    pair_delta.append((float(selected) - float(later)) / max(1, n - 1))
    arrays = dict(trajectory=np.asarray(trajectory, np.int32), step=np.asarray(step, np.int32),
                  action=np.asarray(action, np.int32), remaining_budget=np.asarray(remaining, np.int32),
                  split=np.asarray(splits, np.int8), pair_delta=np.asarray(pair_delta, np.float64))
    return arrays


def train_model(deltas):
    torch.manual_seed(29093026)
    model = CanonicalPriority()
    # Re-randomize explicitly; the constructor value is only serialization shape.
    with torch.no_grad():
        model.weight.normal_(mean=.35, std=.05)
    initial = float(model.weight.detach())
    x = torch.from_numpy(deltas)
    optimizer = torch.optim.Adam(model.parameters(), lr=.08)
    epochs = 0
    for epoch in range(2000):
        optimizer.zero_grad()
        # Desired: score(earlier)-score(later) > 0.
        loss = torch.nn.functional.softplus(-(model.weight * x)).mean()
        loss.backward()
        optimizer.step()
        epochs = epoch + 1
        if float(model.weight) < -8 and epoch >= 100:
            break
    assert float(model.weight) < 0
    return model, initial, epochs, float(loss)


def replay(model, paths):
    rows = []
    for path in paths:
        initial, truth = read_trace(path)
        I = np.asarray(initial["incidence"], np.int64)
        cap = np.asarray(initial["capacities"], np.int64)
        current = np.asarray(initial["counts"], np.int64)
        target = np.asarray(initial["integer_targets"], np.int64)
        remaining = int(initial["remaining_budget"])
        frozen = (target == current) & (remaining > 0)
        z = operator(model, I, cap, current, target, target, remaining, frozen)
        assert z is not None
        predicted = np.repeat(np.arange(len(z), dtype=np.int64), z)
        action_errors = int(np.count_nonzero(predicted != truth)) if len(predicted) == len(truth) else max(len(predicted), len(truth))
        counts = current + I @ z
        rows.append(dict(trajectory=path.stem.removesuffix('.jsonl'), split=split_name(path),
                         dimension=max(len(v.get("family", [])) for v in initial["constraints"]),
                         levels=max(max(v.get("axis_levels", [0])) for v in initial["constraints"]),
                         events=len(truth), action_errors=action_errors,
                         budget_error=int(z.sum() - remaining), target_count_errors=int(np.count_nonzero(counts != target)),
                         frozen_increment_errors=int(np.count_nonzero((I @ z)[frozen])),
                         exact=bool(action_errors == 0 and z.sum() == remaining and np.array_equal(counts, target))))
    return rows


def local_screen(paths):
    errors = total = 0
    by = {}
    for path in paths:
        initial, truth = read_trace(path)
        I = np.asarray(initial["incidence"], np.int64)
        cap = np.asarray(initial["capacities"], np.int64)
        counts = np.asarray(initial["counts"], np.int64)
        target = np.asarray(initial["integer_targets"], np.int64)
        e = 0
        for selected in truth:
            deficit = target - counts
            valid = (cap > 0) & np.all((I == 0) | (deficit[:, None] > 0), axis=0)
            prediction = int(np.flatnonzero(valid)[0]) if valid.any() else -1
            e += prediction != selected
            cap[selected] -= 1
            counts += I[:, selected]
        errors += e
        total += len(truth)
        by[path.stem.removesuffix('.jsonl')] = dict(errors=int(e), events=len(truth))
    return dict(architecture="local earliest non-overshooting assignment", errors=int(errors), events=total,
                exact=False, failures={k: v for k, v in by.items() if v["errors"]})


def composition_check(model):
    from experiments.sho.audit_online_quota_reservoir import A, META, N, capacity
    from experiments.sho.validate_geometry_guarded_shift_realizer import realize
    from experiments.sho.validate_complete_deterministic_probe_generator import (
        ALLOCATOR_CHECKPOINT, GeometryAllocator, regional_targets,
    )
    source = json.loads((ROOT / "complete_deterministic_probe_generator/results.json").read_text())
    allocator = GeometryAllocator()
    allocator.load_state_dict(torch.load(ALLOCATOR_CHECKPOINT, weights_only=False)["state_dict"])
    allocator.eval().requires_grad_(False)
    regional_target, allocation_trace = regional_targets(allocator)
    assert allocation_trace == source["allocation_trace"]
    assert np.array_equal(regional_target, np.asarray(source["local_conserve_target_diagnostic"], np.int64))
    buffers = np.load(ROOT / "geometry_preserving_quota_realizer/input_reservoirs.npz")
    arrays = np.load(ROOT / "shift_aware_quota_realizer/state_arrays.npz")
    final = np.load(ROOT / "geometry_guarded_shift_aware_realizer/realized_pools.npz")
    lo, hi = fixed.bounds(META, N)
    conditions = []
    for quota_record in source["quota_traces"]:
        name = quota_record["condition"]
        cap = np.bincount(arrays[name + "_classes"], minlength=A.shape[1])
        learned = operator(model, A, cap, np.zeros(len(META), np.int64), lo, hi, N)
        teacher = np.asarray(quota_record["fixed_quota"], np.int64)
        assert np.array_equal(learned, teacher)
        conditions.append(dict(condition=name, quota_vector_exact=True,
                               count_vector_exact=bool(np.array_equal(A @ learned, A @ teacher))))
    # One actual wrapper replay through the existing guarded realizer.
    name = source["quota_traces"][0]["condition"]
    trace = OUT / "composition_geometry_replay.jsonl.gz"
    selected, stats = realize(buffers[name], arrays[name + "_classes"], arrays[name + "_initial_selected"],
                              json.loads((ROOT / "geometry_guarded_shift_aware_realizer/results.json").read_text())["limits"][name], trace)
    geometry_exact = bool(np.array_equal(buffers[name][selected], final[name]))
    assert geometry_exact and np.array_equal(capacity(buffers[name][selected]), np.asarray(source["quota_traces"][0]["fixed_quota"]))
    return dict(regional_allocator_checkpoint_unchanged=True, regional_allocation_trace_exact=True,
                regional_target_exact=True, regional_allocation_nodes=len(allocation_trace), conditions=conditions,
                replay_condition=name, geometry_realizer_exact=geometry_exact,
                quota_count_results_exact=True, geometry_events=stats["events"])


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    paths = sorted(p for p in SOURCE.glob("*.jsonl.gz") if "infeasible" not in p.name)
    protected_paths = list(ROOT.rglob("*.pt")) + [Path(fixed.__file__), SOURCE / "results.json", SOURCE / "manifest.json",
                                                   Path("experiments/sho/validate_geometry_guarded_shift_realizer.py")]
    protected = {str(p): sha(p) for p in protected_paths}
    supervision = extract_supervision(paths)
    np.savez_compressed(OUT / "extracted_state_action_pairs.npz", **supervision)
    split_manifest = {p.name: split_name(p) for p in paths}
    dump(OUT / "trajectory_splits.json", split_manifest)
    screen = local_screen(paths)
    model, initial_weight, epochs, loss = train_model(supervision["pair_delta"])
    rows = replay(model, paths)
    assert all(r["exact"] for r in rows)
    action_summary = {}
    for split in ("train", "heldout", "fresh"):
        rr = [r for r in rows if r["split"] == split]
        action_summary[split] = dict(trajectories=len(rr), events=sum(r["events"] for r in rr),
                                     errors=sum(r["action_errors"] for r in rr), accuracy=1.0)
    checkpoint = OUT / "joint_quota_assignment_operator.pt"
    torch.save(dict(state_dict=model.state_dict(), status="FROZEN_CANDIDATE",
                    architecture="bias-free scalar canonical-rank scorer + exact generic bounded-integer decoder",
                    parameter_count=1, decoder="ordered_canonical"), checkpoint)
    composition = composition_check(model)
    assert all(sha(Path(p)) == h for p, h in protected.items())
    result = dict(status="FROZEN_CANDIDATE", architecture="bias-free Linear(1,1) canonical-rank score; maximize learned score inside exact generic integer constrained decoder",
                  parameter_count=1, initial_weight=initial_weight, learned_weight=float(model.weight), epochs=epochs,
                  final_training_loss=loss, quick_screen=screen, smallest_sufficient_representation=dict(
                      learned_input="normalized canonical assignment ID",
                      structural_decoder_inputs=["binary candidate-constraint incidence", "integer capacities", "current/lower/upper counts", "remaining budget", "frozen coarse rows"],
                      output="canonical feasible integer plan, emitted as next smallest positive learned-priority assignment",
                      note="Global feasibility is indispensable; local deficit/capacity features fail."),
                  action_results=action_summary, trajectories=rows, total_action_errors=sum(r["action_errors"] for r in rows),
                  total_budget_errors=sum(r["budget_error"] != 0 for r in rows),
                  total_count_errors=sum(r["target_count_errors"] for r in rows),
                  total_frozen_increment_errors=sum(r["frozen_increment_errors"] for r in rows),
                  dimensions=sorted(set(r["dimension"] for r in rows)), levels=sorted(set(r["levels"] for r in rows)),
                  composition=composition, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  protected_files=len(protected), protected_unchanged=True, runtime_seconds=time.monotonic() - start)
    dump(OUT / "results.json", result)
    report = ["# Learned joint quota-assignment operator", "", "**joint_quota_assignment_operator = FROZEN_CANDIDATE**", "",
              "The teacher's generic integer constraints remain exact structural decoding. The only learned preference is its canonical tie policy: a one-parameter bias-free float64 score of normalized assignment ID. The parameter was randomly initialized positive (wrong ordering) and trained without snapping; its learned negative sign ranks lower canonical IDs first.", "",
              "A purely local earliest-nonovershooting screen made %d errors in %d events. This proves current deficits, incidence and capacity require a global residual-feasibility check. The final decoder accepts the complete incidence/capacity/count/budget/frozen state and applies the learned order while solving the same generic bounded integer constraints; there is no dimension or axis branch." % (screen["errors"], screen["events"]), "",
              "| Split | Complete trajectories | Actions | Errors |", "|---|---:|---:|---:|"]
    for split, value in action_summary.items():
        report.append(f'| {split} | {value["trajectories"]} | {value["events"]} | {value["errors"]} |')
    report += ["", f"Architecture: {result['architecture']}; 1 trainable parameter. Initial weight {initial_weight:.17g}; learned weight {float(model.weight):.17g}; {epochs} epochs.", "",
               f"Closed-loop trajectory replay: {len(rows)} trajectories, {sum(r['events'] for r in rows):,} actions, zero divergences, zero budget/count/frozen-row errors. Dimensions {result['dimensions']}; observed maximum hierarchy levels {result['levels']}. Fresh resumed trajectories include saturated coarse constraints.", "",
               "The learned scorer alone is not a feasibility model. Exact quota guarantees come from the generic constrained decoder. Calling this a standalone neural replacement for integer feasibility would be incorrect; this candidate distills the teacher preference while retaining deterministic feasibility scaffolding.", "",
               f"Composition: all {len(composition['conditions'])} online reservoir quota vectors and count vectors exactly match the deterministic path. One full geometry-guarded realization replay ({composition['replay_condition']}, {composition['geometry_events']:,} events) exactly matches saved candidate coordinates and quota counts.", "",
               f"Checkpoint SHA-256: `{result['checkpoint_sha256']}`. Protected files unchanged: {len(protected)}. Runtime: {result['runtime_seconds']:.2f} s. No downstream N1, N2, or N3 run."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    dump(OUT / "protected_hashes.json", protected)
    dump(OUT / "manifest.json", dict(joint_quota_assignment_operator="FROZEN_CANDIDATE", trained=True,
                                       checkpoint_sha256=sha(checkpoint), results_sha256=sha(OUT / "results.json"),
                                       implementation_sha256=sha(Path(__file__)), report_sha256=sha(OUT / "report.md")))
    print(json.dumps({k: result[k] for k in ("status", "parameter_count", "learned_weight", "action_results", "total_action_errors", "composition", "checkpoint_sha256", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
