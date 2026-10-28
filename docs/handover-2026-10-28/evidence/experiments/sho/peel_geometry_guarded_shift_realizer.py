"""Learn only the local geometry-margin decision; replay the exact exchange teacher."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.sho import validate_shift_aware_quota_realizer as cdf
from experiments.sho import validate_geometry_guarded_shift_realizer as teacher

ROOT = cdf.ROOT
SOURCE = ROOT / "geometry_guarded_shift_aware_realizer"
OUT = ROOT / "learned_geometry_guarded_shift_realizer"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class MarginThreshold(torch.nn.Module):
    """One bias-free float64 parameter. PASS iff learned score is nonnegative."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([-0.4], dtype=torch.float64))

    def forward(self, margin):
        return self.weight * margin

    def decide(self, margin):
        with torch.no_grad():
            score = self(torch.as_tensor(margin, dtype=torch.float64)).numpy()
        return score >= 0


def cohort(name):
    iteration = int(name.removeprefix("iteration").split("_")[0])
    if iteration >= 10000:
        return "fresh"
    return "heldout" if name in {"iteration103_cell", "iteration100_greedy"} else "train"


def extract():
    result = json.loads((SOURCE / "results.json").read_text())
    records = []
    for path in sorted(SOURCE.glob("iteration*.jsonl.gz")):
        name = path.name.removesuffix(".jsonl.gz")
        limit = result["limits"][name]
        with gzip.open(path, "rt") as stream:
            next(stream)
            for line in stream:
                event = json.loads(line)
                for attempt_index, attempt in enumerate(event["attempts"]):
                    coverage_margin = -float(attempt["uncovered_after"])
                    coverage_pass = attempt["reason"] != "coverage"
                    records.append(dict(condition=name, split=cohort(name), event=event["event"],
                                        attempt=attempt_index, safeguard="coverage", margin=coverage_margin,
                                        passed=coverage_pass, teacher_reason=attempt["reason"]))
                    if not coverage_pass:
                        continue
                    hole_margin = float(limit["holes"] - attempt["holes_after"])
                    hole_pass = attempt["reason"] != "holes"
                    records.append(dict(condition=name, split=cohort(name), event=event["event"],
                                        attempt=attempt_index, safeguard="holes", margin=hole_margin,
                                        passed=hole_pass, teacher_reason=attempt["reason"]))
                    if not hole_pass:
                        continue
                    # Only accepted attempts reached these calculations in the
                    # authoritative trace. Boundary controls supply negatives.
                    records.append(dict(condition=name, split=cohort(name), event=event["event"],
                                        attempt=attempt_index, safeguard="nn_cv",
                                        margin=float(attempt["nn_cv_after"] - .08), passed=True,
                                        teacher_reason=attempt["reason"]))
                    records.append(dict(condition=name, split=cohort(name), event=event["event"],
                                        attempt=attempt_index, safeguard="nn_p01",
                                        margin=float(attempt["nn_p01_after"] - limit["nn_p01"]), passed=True,
                                        teacher_reason=attempt["reason"]))
    return records


def train(records):
    torch.manual_seed(30093026)
    model = MarginThreshold()
    with torch.no_grad():
        model.weight.normal_(mean=-.4, std=.05)  # deliberately wrong sign
    initial = float(model.weight.detach())
    coverage = [r for r in records if r["safeguard"] == "coverage" and r["split"] == "train"]
    # Normalize magnitudes only for optimization; inference uses raw margins.
    negative = np.asarray([r["margin"] for r in coverage if not r["passed"]], np.float64)
    controls = np.array([-1., -np.finfo(float).eps, 0., np.finfo(float).eps, 1.], np.float64)
    labels = np.array([0., 0., 1., 1., 1.], np.float64)
    x = np.r_[negative / np.maximum(1., np.abs(negative)), controls]
    y = np.r_[np.zeros(len(negative)), labels]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=.08)
    loss = None
    for epoch in range(2000):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        if float(model.weight.detach()) > 8 and epoch >= 100:
            break
    assert float(model.weight.detach()) > 0
    return model, dict(initial_weight=initial, learned_weight=float(model.weight.detach()),
                       epochs=epoch + 1, loss=float(loss.detach()), training_examples=len(x),
                       coverage_trace_negatives=len(negative), boundary_controls=len(controls))


def isolated_validation(model, records):
    rows = []
    for safeguard in ("coverage", "holes", "nn_cv", "nn_p01"):
        for split in ("train", "heldout", "fresh"):
            rr = [r for r in records if r["safeguard"] == safeguard and r["split"] == split]
            pred = model.decide(np.asarray([r["margin"] for r in rr]))
            truth = np.asarray([r["passed"] for r in rr])
            rows.append(dict(safeguard=safeguard, split=split, cases=len(rr),
                             passes=int(truth.sum()), rejects=int((~truth).sum()),
                             errors=int(np.count_nonzero(pred != truth))))
    # Explicit equality and immediate floating-point neighbors.
    controls = np.array([-1., -np.finfo(float).eps, np.nextafter(0., -np.inf), 0.,
                         np.nextafter(0., np.inf), np.finfo(float).eps, 1.])
    truth = controls >= 0
    prediction = model.decide(controls)
    assert np.array_equal(prediction, truth)
    return rows, dict(values=controls.tolist(), errors=int(np.count_nonzero(prediction != truth)),
                      equality_routes="PASS", negative_routes="REJECT", positive_routes="PASS")


class LearnedGuard(teacher.GeometryGuard):
    def __init__(self, points, selected, cover, holes, nn_p01, threshold):
        super().__init__(points, selected, cover, holes, nn_p01)
        self.threshold = threshold

    def one(self, margin):
        return bool(self.threshold.decide(np.asarray([margin]))[0])

    def attempt(self, old, new):
        lost = np.setdiff1d(self.edges[old], self.edges[new], assume_unique=False)
        uncovered = int(np.sum(self.count[lost] == 1))
        hlost = np.setdiff1d(self.holes[old], self.holes[new])
        hgain = np.setdiff1d(self.holes[new], self.holes[old])
        holes = self.hole_count + int(np.sum(self.hcount[hlost] == 1)) - int(np.sum(self.hcount[hgain] == 0))
        info = dict(holes_before=self.hole_count, holes_after=holes, uncovered_after=uncovered)
        if not self.one(-float(uncovered)):
            return False, "coverage", info
        if not self.one(float(self.hole_limit - holes)):
            return False, "holes", info
        self.used[old] = False
        self.used[new] = True
        rows = np.union1d(self.reverse[old], self.reverse[new])
        previous = self.near[rows].copy()
        self.near[rows] = np.min(np.where(self.used[self.neighbors[rows]], self.distances[rows], np.inf), axis=1)
        assert np.isfinite(self.near).all()
        nn = self.near[self.used]
        cv = float(nn.std() / nn.mean())
        p01 = float(np.quantile(nn, .01))
        info.update(nn_cv_after=cv, nn_p01_after=p01)
        if not self.one(cv - .08):
            self.near[rows] = previous
            self.used[old] = True
            self.used[new] = False
            return False, "nn_cv", info
        if not self.one(p01 - self.p01):
            self.near[rows] = previous
            self.used[old] = True
            self.used[new] = False
            return False, "nn_p01", info
        self.count[self.edges[old]] -= 1
        self.count[self.edges[new]] += 1
        self.hcount[self.holes[old]] -= 1
        self.hcount[self.holes[new]] += 1
        self.hole_count = holes
        return True, "accepted", info


def realize(model, points, classes, selected, limits, teacher_trace, output_trace):
    selected = selected.copy()
    state = cdf.CDFState(points, selected)
    guard = LearnedGuard(points, selected, threshold=model, **limits)
    groups = {int(c): np.flatnonzero(classes == c) for c in np.unique(classes)}
    quotas = np.bincount(classes[selected], minlength=int(classes.max()) + 1)
    errors = attempts = accepted = 0
    with gzip.open(teacher_trace, "rt") as expected, gzip.open(output_trace, "wt") as output:
        teacher_header = json.loads(next(expected))
        output.write(json.dumps(dict(type="initial", selected=selected.tolist(), learned_threshold=True,
                                     teacher_trace_sha256=sha(teacher_trace))) + "\n")
        for sweep in range(cdf.MAX_SWEEPS):
            for slot in range(len(selected)):
                truth = json.loads(next(expected))
                old = int(selected[slot])
                group = groups[int(classes[old])]
                alternatives = group[(~guard.used[group]) | (group == old)]
                costs = state.effects(old, alternatives)
                order = np.lexsort((alternatives, costs))
                decisions = []
                new = old
                for index in order:
                    if costs[index] >= 0:
                        break
                    candidate = int(alternatives[index])
                    ok, reason, info = guard.attempt(old, candidate)
                    decisions.append(dict(candidate=candidate, cdf_scaled_change=int(costs[index]),
                                          accept=ok, reason=reason, **info))
                    attempts += 1
                    if ok:
                        new = candidate
                        break
                errors += int(old != truth["old"] or new != truth["selected"] or decisions != truth["attempts"])
                output.write(json.dumps(dict(event=truth["event"], sweep=sweep, slot=slot, old=old,
                                             attempts=decisions, selected=new), separators=(",", ":")) + "\n")
                if new != old:
                    state.swap(old, new)
                    selected[slot] = new
                    accepted += 1
            assert np.array_equal(np.bincount(classes[selected], minlength=len(quotas)), quotas)
            assert cdf.CDFState(points, selected).fingerprint() == state.fingerprint()
        assert expected.read() == ""
    return selected, dict(events=cdf.MAX_SWEEPS * len(selected), attempts=attempts,
                          accepted=accepted, trajectory_errors=errors)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    protected_paths = list(ROOT.rglob("*.pt")) + list(SOURCE.iterdir()) + [Path(teacher.__file__), Path(cdf.__file__)]
    protected = {str(p): sha(p) for p in protected_paths if p.is_file()}
    records = extract()
    write_csv(OUT / "local_decision_supervision.csv", records)
    model, training = train(records)
    isolated, controls = isolated_validation(model, records)
    assert all(r["errors"] == 0 for r in isolated) and controls["errors"] == 0
    checkpoint = OUT / "geometry_margin_threshold_operator.pt"
    torch.save(dict(state_dict=model.state_dict(), status="LOCKED", architecture="one-parameter bias-free scalar margin threshold",
                    semantics="PASS iff learned_weight * signed_margin >= 0; equality PASS"), checkpoint)

    inputs = np.load(ROOT / "geometry_preserving_quota_realizer/input_reservoirs.npz")
    arrays = np.load(ROOT / "shift_aware_quota_realizer/state_arrays.npz")
    teacher_pools = np.load(SOURCE / "realized_pools.npz")
    teacher_result = json.loads((SOURCE / "results.json").read_text())
    pools = {}
    selected_ids = {}
    trajectories = []
    for name in teacher_pools.files:
        chosen, stats = realize(model, inputs[name], arrays[name + "_classes"], arrays[name + "_initial_selected"],
                                teacher_result["limits"][name], SOURCE / (name + ".jsonl.gz"), OUT / (name + ".jsonl.gz"))
        pools[name] = inputs[name][chosen]
        selected_ids[name] = chosen
        stats.update(condition=name, split=cohort(name), coordinate_errors=int(np.count_nonzero(pools[name] != teacher_pools[name])),
                     quota_errors=int(np.count_nonzero(cdf.capacity(pools[name]) != arrays[name + "_quotas"])),
                     budget_error=len(pools[name]) - cdf.N)
        assert stats["trajectory_errors"] == stats["coordinate_errors"] == stats["quota_errors"] == stats["budget_error"] == 0
        trajectories.append(stats)
        print(json.dumps(stats), flush=True)
    np.savez_compressed(OUT / "realized_pools.npz", **pools)
    np.savez_compressed(OUT / "selected_candidate_ids.npz", **selected_ids)

    # Unchanged historical evaluation, written under the new output directory.
    old_out = cdf.OUT
    cdf.OUT = OUT
    source_result = json.loads((cdf.SOURCE / "results.json").read_text())
    evaluation = cdf.evaluate(pools, np.load(cdf.SOURCE / "interior_pools.npz"), source_result)
    fresh = cdf.fresh_baseline_check()
    cdf.OUT = old_out
    passed = bool(evaluation["passed"] and fresh["passed"] and all(r["trajectory_errors"] == 0 for r in trajectories))
    assert passed and all(sha(p) == h for p, h in protected.items())
    result = dict(status="VALIDATED_ASSEMBLY", margin_threshold_status="LOCKED", trained=True,
                  architecture="shared one-parameter bias-free float64 signed-margin threshold",
                  parameter_count=1, training=training, isolated_decisions=isolated,
                  boundary_controls=controls, trajectories=trajectories,
                  total_events=sum(r["events"] for r in trajectories), total_attempts=sum(r["attempts"] for r in trajectories),
                  total_trajectory_errors=sum(r["trajectory_errors"] for r in trajectories),
                  quota_errors=0, budget_errors=0, exact_teacher_pools=True,
                  historical_gate=evaluation["historical_gate"], cohort_health=evaluation["cohort_health"],
                  archived_one_count_ratio=evaluation["archived_one_count_ratio"],
                  fresh_discrepancy=fresh, scale_comparison=evaluation["scale_comparison"],
                  deterministic_scaffolding=["exact CDF residual/effect arithmetic", "CDF score then candidate-ID ordering",
                      "same-class compatibility and selected-mask filtering", "reference/nearest-neighbor count construction",
                      "sequential first-failed-safeguard routing", "accepted-swap state updates",
                      "clustering/separation invariant", "cross-pool freshness and historical gate evaluation"],
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), protected_files=len(protected),
                  protected_unchanged=True, runtime_seconds=time.monotonic() - started)
    dump(OUT / "results.json", result)
    write_csv(OUT / "isolated_decision_results.csv", isolated)
    decomposition = [
        dict(operation="CDF proposal ranking", classification="deterministic arithmetic/bookkeeping", detail="exact integer CDF effects; ascending score then candidate ID"),
        dict(operation="quota/class compatibility", classification="deterministic arithmetic/bookkeeping", detail="unused candidate in identical fixed incidence class"),
        dict(operation="coverage safeguard", classification="new learned local decision", detail="shared signed-margin threshold on -uncovered_after"),
        dict(operation="hole safeguard", classification="reuses newly locked operator", detail="signed margin hole_limit-holes_after"),
        dict(operation="NN-CV safeguard", classification="reuses newly locked operator", detail="signed margin nn_cv-.08; trace positives plus boundary controls"),
        dict(operation="NN-p01 safeguard", classification="reuses newly locked operator", detail="signed margin nn_p01-limit; trace positives plus boundary controls"),
        dict(operation="separation/clustering", classification="deterministic proven invariant", detail="guarded reservoir separation exceeds clustering cutoff"),
        dict(operation="freshness/overlap", classification="deterministic post-run validation", detail="not a local exchange decision"),
        dict(operation="ACCEPT/REJECT routing", classification="deterministic sequential control", detail="first failed safeguard rejects; all pass accepts"),
        dict(operation="state update", classification="deterministic bookkeeping", detail="sparse coverage/hole/NN/CDF state mutation"),
    ]
    write_csv(OUT / "operation_decomposition.csv", decomposition)
    report = ["# Geometry-guarded shift-aware realizer peelback", "", "**geometry_margin_threshold_operator = LOCKED**  ",
              "**learned_geometry_guarded_shift_aware_realizer = VALIDATED_ASSEMBLY**", "",
              "The first missing local operation is the inclusive zero-margin decision. One bias-free float64 scalar was trained from a deliberately wrong sign on coverage supervision, then reused only after exact isolated validation for holes, NN-CV, and NN-p01. Equality routes PASS. CDF search/order, compatibility, geometry arithmetic, sequential routing, and state updates remain deterministic.", "",
              "| Safeguard | Split | Cases | Rejects | Errors |", "|---|---|---:|---:|---:|"]
    for r in isolated:
        report.append(f'| {r["safeguard"]} | {r["split"]} | {r["cases"]} | {r["rejects"]} | {r["errors"]} |')
    report += ["", f"Architecture: one parameter. Initial weight {training['initial_weight']:.17g}; learned weight {training['learned_weight']:.17g}; no snapping. Checkpoint SHA-256 `{sha(checkpoint)}`.", "",
               f"Full replay: {result['total_events']:,} slot events and {result['total_attempts']:,} attempted exchanges across eight complete trajectories; zero decision, coordinate, quota, or budget divergences. The final pools are bit-identical to the deterministic teacher.", "",
               f"Historical gates: `{json.dumps(evaluation['historical_gate'])}`. Fresh discrepancy gates: `{json.dumps(fresh['gates'])}`. Archived ratio `{evaluation['archived_one_count_ratio']['archived']:.10g}`; fresh ratio `{fresh['worst_one_count_ratios']['shift_aware']:.10g}`.", "",
               "NN-CV and NN-p01 had no negative examples in the saved trajectories; their exact teacher comparisons were therefore checked with explicit negative/equality/positive floating-point boundary controls. This is threshold-semantic validation, not evidence of observed failures for those two safeguards.", "",
               "The assembly is ready for composition as a hybrid realizer with learned local safeguard decisions. It is not a fully learned exchange engine: CDF ranking/search and exact geometry/state bookkeeping remain deterministic by design. No downstream N1, N2, or N3 run was performed.", "",
               f"Protected files unchanged: {len(protected)}. Runtime: {result['runtime_seconds']:.2f} s."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    dump(OUT / "protected_hashes.json", protected)
    dump(OUT / "manifest.json", dict(geometry_margin_threshold_operator="LOCKED",
        learned_geometry_guarded_shift_aware_realizer="VALIDATED_ASSEMBLY", trained=True,
        checkpoint_sha256=sha(checkpoint), implementation_sha256=sha(__file__),
        results_sha256=sha(OUT / "results.json"), report_sha256=sha(OUT / "report.md"),
        pool_sha256=sha(OUT / "realized_pools.npz")))
    print(json.dumps(dict(status=result["status"], learned_weight=training["learned_weight"],
                          total_events=result["total_events"], total_attempts=result["total_attempts"],
                          historical_gate=result["historical_gate"], fresh_gates=fresh["gates"],
                          checkpoint_sha256=result["checkpoint_sha256"], runtime_seconds=result["runtime_seconds"]), indent=2))


if __name__ == "__main__":
    main()
