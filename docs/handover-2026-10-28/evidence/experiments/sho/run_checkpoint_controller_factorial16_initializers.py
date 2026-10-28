#!/usr/bin/env python3
"""Create the four frozen prospective N1+N2 histories for factorial16."""
from collections import Counter
from dataclasses import asdict, is_dataclass
import fcntl, hashlib, json, math, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

from experiments.sho import recover_affine_bias_targeted as parent
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.recover_generic_update_targeted import GenericArchiveN1
from experiments.resume_hybrid_student_after_ckdtree import Log, run_n2_with_best_ever_assembly

ROOT = Path("outputs/sho/autonomous_generic_sho_20261014")
PROTOCOL = ROOT / "protocols/generic_n2_n3_checkpoint_controller_factorial16_design.json"
ADDENDUM = ROOT / "reviews/generic_n2_n3_checkpoint_controller_factorial16_formula_contrast_review.md"
OUTROOT = ROOT / "initializers"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
PROTOCOL_SHA = "e2c55e0d79bdcc4b9999bac5997248c0373c31f57e060a8afdb64fdc153dc6d8"
ADDENDUM_SHA = "75f93d00957863cd1ba7c652d9704dacdb3bb330b26edcaf035deb6b52effdb1"


def clean(x):
    if is_dataclass(x): return clean(asdict(x))
    if isinstance(x, torch.Tensor): return clean(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(x) else str(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bits_equal(a, b):
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return x.shape == y.shape and np.array_equal(x.view(np.uint64), y.view(np.uint64))


class Objective:
    def __init__(self, adapter, trajectory):
        self.adapter, self.trajectory = adapter, trajectory
        self.phase = None; self.counts = Counter(); self.capture = None
    def __call__(self, phi):
        with torch.no_grad(): value = float(self.adapter.objective(phi, self.trajectory))
        self.counts[self.phase or "unspecified"] += 1
        if self.capture is not None:
            self.capture.append({"evaluation": len(self.capture) + 1,
                                 "phi": np.asarray(phi, np.float64).copy(), "loss": value})
        return value


def restore_runner(objective, adapter, case_id, events, log):
    runner = GenericArchiveN1(objective, adapter, case_id, log)
    runner.points = [np.asarray(row["point"], np.float64) for row in events]
    runner.losses = [float(row["loss"]) for row in events]
    runner.events = list(events)
    runner.iteration = int(events[-1]["iteration"]) if events else 0
    if not all(row["source"] == "initial" and int(row["iteration"]) == 0 for row in events):
        raise RuntimeError("TECHNICAL_UNAVAILABLE: predecessor is not a pure initial-design prefix")
    runner.phase = "initial"; runner.next_pending = len(events)
    runner.pending = np.empty((0, adapter.dimension), np.float64)
    if len(events) >= len(runner._design):
        raise RuntimeError("TECHNICAL_UNAVAILABLE: wrapper supports only frozen initial-design prefixes")
    for i, row in enumerate(events):
        if not bits_equal(runner._design[i], row["point"]):
            raise RuntimeError(f"ALERT: predecessor point {i + 1} differs from frozen initial design")
    return runner


def run_case(name, spec, corpus, global_inputs):
    run_id = spec["initializer_run_id"]
    out = OUTROOT / run_id
    if out.exists(): raise RuntimeError(f"TECHNICAL_UNAVAILABLE: fresh output exists: {out}")
    for base in (ROOT / "initializers", ROOT / "runs", ROOT / "screens"):
        if base.exists() and any(p.name.startswith(run_id) for p in base.rglob("*")):
            raise RuntimeError(f"TECHNICAL_UNAVAILABLE: declared identity already exists: {run_id}")
    prev = Path(spec["previous_n1_history"])
    if sha(prev) != spec["previous_n1_sha256"]:
        raise RuntimeError(f"ALERT: predecessor history changed: {prev}")
    events = json.loads(prev.read_text())
    n, i = map(int, spec["branch"])
    if len(events) != n - 1 or i != n - 1 or [int(x["evaluation"]) for x in events] != list(range(1, n)):
        raise RuntimeError("TECHNICAL_UNAVAILABLE: predecessor length/branch identity mismatch")
    trajectory = np.asarray(corpus[spec["trajectory_key"]], np.float64).copy()
    if hashlib.sha256(trajectory.tobytes()).hexdigest() != spec["trajectory_sha256"]:
        raise RuntimeError("ALERT: trajectory hash mismatch")
    adapter = GenericSHOUpdateNetworkAdapter(); objective = Objective(adapter, torch.from_numpy(trajectory))
    log = Log(); runner = restore_runner(objective, adapter, int(spec["case_id"]), events, log)
    out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    inputs = dict(global_inputs); inputs[str(prev.resolve())] = sha(prev)
    save(out / "setup.json", {"protocol_sha256": PROTOCOL_SHA, "formula_addendum_sha256": ADDENDUM_SHA,
        "case": name, "case_spec": spec, "input_sha256": inputs,
        "settings": {"new_N1_events": 1, "N2_maxfev": 1500, "maxiter": 200,
                     "xtol": 1e-12, "ftol": 1e-15, "trace": False},
        "no_N3": True, "fresh_output": True})
    objective.phase = "N1"; before_calls = sum(objective.counts.values()); runner.advance_one()
    if len(runner.events) != n or sum(objective.counts.values()) != before_calls + 1:
        raise RuntimeError("ALERT: prospective N1 did not add exactly one event")
    new_event = runner.events[-1]
    for old in events:
        if bits_equal(new_event["theta"], old["theta"]):
            raise RuntimeError("TECHNICAL_UNAVAILABLE: new N1 source repeats prior case-history variant")
    if int(new_event["evaluation"]) != n or int(new_event["best_index"]) < 0:
        raise RuntimeError("ALERT: new N1 event identity invalid")
    save(out / "n1_history.json", runner.events)
    source_phi = np.asarray(new_event["theta"], np.float64); source_loss = float(new_event["loss"])
    lower, upper = adapter.archive_box(source_phi, n)
    objective.phase = "N2"; objective.capture = []
    n2 = run_n2_with_best_ever_assembly(objective, source_phi, source_loss, lower, upper, log,
        xtol=1e-12, ftol=1e-15, maxiter=200, maxfev=1500, trace=False)
    history = objective.capture; objective.capture = None
    if len(history) != 1500 or int(n2["evaluations"]) != 1500:
        raise RuntimeError("TECHNICAL_UNAVAILABLE: N2 did not produce exactly 1500 evaluations")
    with (out / "objective_evaluations.jsonl").open("w") as stream:
        for row in history: stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
    best_phi, best_loss, best_eval = source_phi.copy(), source_loss, 0
    for row in history[:500]:
        loss = float(row["loss"])
        if loss <= best_loss:
            best_phi, best_loss, best_eval = np.asarray(row["phi"], np.float64), loss, int(row["evaluation"])
    endpoint_phi, endpoint_loss = np.asarray(n2["theta"], np.float64), float(n2["loss"])
    objective.phase = "endpoint_replay"; replay = float(objective(endpoint_phi))
    if not bits_equal(replay, endpoint_loss): raise RuntimeError("ALERT: N2 endpoint native replay mismatch")
    if not any(bits_equal(endpoint_phi, row["phi"]) and bits_equal(endpoint_loss, row["loss"]) for row in history):
        raise RuntimeError("ALERT: endpoint is absent from uninterrupted N2 history")
    checkpoints = {
        "n2_prefix500": {"theta": best_phi, "loss": best_loss, "selected_evaluation": best_eval,
                         "rule": "lowest native N2 loss among source evaluation 0 and evaluations 1..500; exact ties latest"},
        "n2_endpoint1500": {"theta": endpoint_phi, "loss": endpoint_loss, "selected_evaluation": None,
                            "rule": "unchanged strict best-ever endpoint after exactly 1500 evaluations"}}
    checkpoints["bitwise_identical"] = bits_equal(best_phi, endpoint_phi)
    save(out / "checkpoints.json", checkpoints)
    terminal = {k: v for k, v in n2.items() if k not in ("tracker", "powell")}
    save(out / "powell_terminal.json", terminal)
    result = {"status": "ENDPOINT_CREATED", "case": name, "branch": [n, i],
        "source_loss": source_loss, "prefix500_loss": best_loss, "prefix500_evaluation": best_eval,
        "endpoint1500_loss": endpoint_loss, "checkpoint_vectors_bitwise_identical": checkpoints["bitwise_identical"],
        "objective_counts": dict(objective.counts), "n2_evaluations": 1500,
        "stop_reason": str(n2["message"]), "cached_N1_events": n - 1,
        "elapsed_seconds": time.monotonic() - started, "adapter_calls": dict(adapter.calls)}
    save(out / "result.json", result)
    checks = {p: sha(p) == d for p, d in inputs.items()}
    save(out / "integrity.json", {"unchanged": all(checks.values()), "checks": checks})
    if not all(checks.values()): raise RuntimeError("ALERT: protected input changed during initializer")
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out / "artifact_manifest.json", {"sha256": files})
    print(f"INITIALIZER {name} branch={(n,i)} prefix500={best_loss:.17g}@{best_eval} endpoint={endpoint_loss:.17g}", flush=True)
    return result


def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    if sha(PROTOCOL) != PROTOCOL_SHA or sha(ADDENDUM) != ADDENDUM_SHA:
        raise RuntimeError("ALERT: frozen protocol or formula addendum hash mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "FROZEN_FOR_PARENT_SCIENTIFIC_REVIEW_DO_NOT_EXECUTE_YET":
        raise RuntimeError("ALERT: unexpected frozen protocol identity")
    lock_path = ROOT / "active.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("TECHNICAL_UNAVAILABLE: numerical lock is held")
    paths = [PROTOCOL, ADDENDUM, DATA,
        Path("experiments/sho/generic_update_network_adapter.py"),
        Path("experiments/sho/recover_generic_update_targeted.py"),
        Path("experiments/resume_hybrid_student_after_ckdtree.py"), Path(__file__)]
    global_inputs = {str(p.resolve()): sha(p) for p in paths}
    batch = ROOT / "screens/generic_n2_n3_checkpoint_controller_factorial16"
    if batch.exists(): raise RuntimeError(f"TECHNICAL_UNAVAILABLE: screen output exists: {batch}")
    batch.mkdir(parents=True, exist_ok=False)
    save(batch / "initializer_batch_started.json", {"status": "RUNNING", "protocol_sha256": PROTOCOL_SHA,
        "formula_addendum_sha256": ADDENDUM_SHA, "input_sha256": global_inputs, "sequential": True})
    rows = []
    try:
        with np.load(DATA) as corpus:
            for name, spec in protocol["source_eligibility"]["cases"].items():
                rows.append(run_case(name, spec, corpus, global_inputs))
                save(batch / "initializer_progress.json", {"completed": rows})
        save(batch / "initializer_batch_result.json", {"status": "COMPLETE", "completed": rows})
    except BaseException as exc:
        save(batch / "initializer_batch_failure.json", {"status": "FAILED", "type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "completed": rows,
            "no_source_replacement_or_retry": True})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


if __name__ == "__main__": main()
