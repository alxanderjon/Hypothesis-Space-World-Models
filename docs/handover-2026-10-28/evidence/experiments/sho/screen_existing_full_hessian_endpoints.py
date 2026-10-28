#!/usr/bin/env python3
"""Fresh isolated existing-full-Hessian screen; never resumes a long run.

The numerical loop follows audit_coordinated_generic_recovery.screen with
EXISTING_FULL_HESSIAN selected through the existing runtime hook. Persistence,
historical comparisons and provenance are diagnostic additions only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
BRANCHES = ("n00005_i00004", "n00004_i00003", "n00001_i00000")
MILESTONES = (0, 1, 10, 50, 100, 250, 500)
SOURCE_FILES = (
    "experiments/sho/audit_coordinated_generic_recovery.py",
    "experiments/sho/generic_update_network_adapter.py",
    "experiments/sho/generic_update_residual_n3_opt_in.py",
    "experiments/sho/resume_generic_update_targeted.py",
    "experiments/sho/run_stage1b_n3_gradient_polishing.py",
    "experiments/sho/assemble_n3_learned_polisher.py",
    "experiments/n3_box_qp.py",
    "experiments/geometry/prefix_residual_mse_assembly.py",
    "experiments/resume_hybrid_student_after_ckdtree.py",
    "experiments/sho/run_companion6_stabilized_residual_live.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(payload):
    nonfinite = []

    def clean(value, field="$"):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): clean(v, field + "." + str(k)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v, field + "[" + str(i) + "]") for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append({"field": field, "original_value": repr(value)})
            return None
        return value

    value = clean(payload)
    if nonfinite:
        value["nonfinite_values_replaced_by_null"] = nonfinite
    return value


def save(out, name, payload):
    temporary = out / (name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(serializable(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, out / name)


def append(stream, payload):
    stream.write(json.dumps(serializable(payload), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--arm", choices=("original", "extended"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; this driver never overwrites or resumes")
    if not (repo / SOURCE_FILES[0]).is_file():
        parser.error("--repo does not contain the expected recovery runtime")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    accepted = 0
    checkpoint = None
    try:
        import numpy as np
        import scipy
        import torch
        from experiments.resume_hybrid_student_after_ckdtree import Log
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import OptInResidualN3Runtime, EXISTING_FULL_HESSIAN
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        src = repo / "outputs/sho/generic_update_network_targeted_recovery/companion_6"
        batch = repo / "outputs/sho/n2_budget_discrimination_three_branch_20261013"
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        original_n2 = src / (args.branch + "_n2.json")
        endpoint_path = original_n2 if args.arm == "original" else batch / (args.branch + "_n2_extended.json")
        historical_paths = sorted(src.glob(args.branch + "_n3*.jsonl")) if args.arm == "original" else []
        if args.arm == "original" and not historical_paths:
            raise RuntimeError("ALERT: historical original full-Hessian trace missing")
        inputs = sorted(set([data, original_n2, endpoint_path, *historical_paths]))
        before = {str(p): sha256(p) for p in inputs}
        source_before = {name: sha256(repo / name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        endpoint_document = json.loads(endpoint_path.read_text())
        endpoint = endpoint_document["result"]
        expected_evaluations = 1500 if args.arm == "original" else 3000
        if endpoint["evaluations"] != expected_evaluations:
            raise RuntimeError("ALERT: endpoint evaluation-count identity mismatch")
        save(out, "manifest.json", {
            "branch": args.branch, "arm": args.arm, "direction_mode": EXISTING_FULL_HESSIAN,
            "repo": str(repo), "endpoint_source": str(endpoint_path),
            "endpoint_result": endpoint, "input_sha256": before,
            "direct_source_sha256": source_before, "script": str(Path(__file__).resolve()),
            "script_sha256": script_hash,
            "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                            "scipy": scipy.__version__, "torch_threads": torch.get_num_threads()},
            "protocol": "Existing full-Hessian runtime and audit.replay; unchanged audit screen stopping/counter logic; 500 accepted-step resource limit; strict loss <1e-20 handoff.",
            "checkpoint_note": "Fresh isolated diagnostic only. State is persisted for review or a separately verified exact restart; no resume entry point is implemented.",
            "count_note": "Trial scalar-loss calls exclude derivative/starting-objective calls. No optimizer tolerance, acceptance, bound, architecture or production setting changes.",
        })

        # Only states in this finite screen are needed. Every source file is
        # hashed in full; later portions are not parsed or used as initializers.
        historical = {}
        duplicate_records = []
        coverage = []
        for path in historical_paths:
            parsed = 0
            first_beyond = None
            previous_step = None
            with path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    step = int(record["diagnostics"]["accepted_before"])
                    if previous_step is not None and step < previous_step:
                        raise RuntimeError("ALERT: historical trace decreases in accepted index")
                    previous_step = step
                    if step > 500:
                        first_beyond = step
                        break
                    history, diagnostics = record["history"], record["diagnostics"]
                    if not bits_equal(history["theta"], diagnostics["phi"]) or not bits_equal(history["loss"], diagnostics["loss"]):
                        raise RuntimeError("ALERT: internal historical state disagreement")
                    parsed += 1
                    selected = {"theta": history["theta"], "loss": history["loss"],
                                "source": str(path), "line": line_number}
                    if step in historical:
                        prior = historical[step]
                        if not bits_equal(prior["theta"], selected["theta"]) or not bits_equal(prior["loss"], selected["loss"]):
                            raise RuntimeError("ALERT: conflicting duplicated historical state")
                        duplicate_records.append({"step": step, "source": str(path), "line": line_number})
                    else:
                        historical[step] = selected
            coverage.append({"source": str(path), "parsed_screen_records": parsed,
                             "first_record_beyond_screen": first_beyond})
        if historical and sorted(historical) != list(range(max(historical) + 1)):
            raise RuntimeError("ALERT: historical replay prefix has missing accepted steps")
        if args.arm == "original" and 0 not in historical:
            raise RuntimeError("ALERT: historical original start is missing")
        save(out, "historical_reference.json", {"coverage": coverage, "identical_duplicates": duplicate_records,
             "available_step_count": len(historical), "max_available_step": max(historical) if historical else None,
             "comparison": "history.theta and history.loss indexed by diagnostics.accepted_before; reporting-only route diagnostics are not compared."})

        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        rt = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(), trajectory, Log(), direction_mode=EXISTING_FULL_HESSIAN)
        theta = torch.tensor(endpoint["theta"], dtype=torch.float64)
        loss = float(endpoint["loss"])
        with torch.no_grad():
            start_loss = float(rt.adapter.objective(theta, trajectory))
        if not bits_equal(start_loss, loss):
            raise RuntimeError("ALERT: starting N2 objective loss differs bitwise")
        path_length, bt, counter = 0.0, 0, 0
        snapshots, contacts, historical_matches = {}, set(), set()
        stop = None
        started = time.monotonic()
        trace_path = out / "full_hessian.jsonl"

        def verify_historical():
            if accepted not in historical:
                return {"available": False}
            reference = historical[accepted]
            checks = {"theta_bits": bits_equal(theta, reference["theta"]),
                      "loss_bits": bits_equal(loss, reference["loss"])}
            if not all(checks.values()):
                save(out, "ALERT_historical_mismatch.json", {"accepted_steps": accepted,
                    "checks": checks, "generated_theta": theta, "generated_loss": loss,
                    "reference": reference})
                raise RuntimeError("ALERT: generated original trajectory differs from historical full-Hessian state")
            historical_matches.add(accepted)
            return {"available": True, **checks}

        def write_checkpoint():
            active = torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist()
            save(out, "diagnostic_state.json", {
                "branch": args.branch, "arm": args.arm, "theta": theta, "loss": loss,
                "accepted_steps": accepted, "stagnation_count": counter,
                "parameter_path_length": path_length, "backtracking_rejections": bt,
                "trial_objective_evaluations": rt.objective_evaluations,
                "derivative_calls": rt.binding_counters.derivative_calls,
                "stop_reason": stop, "active_bounds": active,
                "bound_coordinates_ever": sorted(contacts | set(active)), "snapshots": snapshots,
                "historical_matched_steps": sorted(historical_matches),
                "direction_mode": EXISTING_FULL_HESSIAN, "endpoint_source": str(endpoint_path),
                "input_sha256": before, "direct_source_sha256": source_before, "script_sha256": script_hash,
                "restart_note": "No automatic resume implemented. Any future resume must verify this state and replay counters exactly; never reuse a paused long-run checkpoint.",
            })

        checkpoint = write_checkpoint
        verify_historical()
        write_checkpoint()
        with trace_path.open("x") as stream:
            while True:
                historical_check = verify_historical()
                result, gradient, H, gn, pre, route, direction, qp = rt.derivatives_and_direction(theta)
                if not bits_equal(result.loss, loss):
                    raise RuntimeError("ALERT: derivative-state loss replay mismatch")
                pg = gradient.clone()
                pg[((theta == rt.lower) & (gradient >= 0)) | ((theta == rt.upper) & (gradient <= 0))] = 0
                active = torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist()
                contacts.update(active)
                row = {"step": accepted, "phi": theta, "loss": loss, "gradient_norm": gn,
                       "projected_gradient_norm": float(pg.norm()), "route": route,
                       "mu": None if qp is None else qp.get("damping"),
                       "direction_norm": None if direction is None else float(direction.norm()),
                       "direction": direction, "qp": qp, "active_bounds": active,
                       "path": path_length, "backtracking": bt,
                       "objective_evaluations": rt.objective_evaluations,
                       "stagnation_count": counter, "historical_replay": historical_check}
                if accepted in MILESTONES:
                    snapshots[str(accepted)] = serializable(row)
                if loss < 1e-20:
                    stop = "HANDOFF_THRESHOLD_REACHED"
                elif stop is None and pre != "CONTINUE":
                    stop = pre
                elif stop is None and accepted >= 500:
                    stop = "SCREEN_RESOURCE_LIMIT_500_ACCEPTED_STEPS"
                if stop:
                    row["stop_reason"] = stop
                    append(stream, row)
                    write_checkpoint()
                    break
                trial = replay(rt, theta, loss, gradient, direction)
                row["trial"] = trial
                if trial["accepted"]:
                    predicted = -float(qp["quadratic_value"])
                    counter = prod._updated_float64_stagnation_count(counter, loss, trial["loss"], predicted)
                    row["full_predicted_decrease"] = predicted
                    theta = torch.tensor(trial["phi"], dtype=torch.float64)
                    loss = trial["loss"]
                    path_length += trial["displacement"]
                    bt += trial["rejections"]
                    accepted += 1
                    if counter >= prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                        stop = prod.FLOAT64_STAGNATION_STOP_REASON
                else:
                    stop = trial["reason"]
                    bt += trial.get("rejections", 0)
                    row["stop_reason"] = stop
                append(stream, row)
                verify_historical()
                write_checkpoint()
                if not trial["accepted"]:
                    break
                if accepted in MILESTONES or stop or loss < 1e-20:
                    print(f"FULL_HESSIAN {args.branch} {args.arm} accepted={accepted} loss={loss:.17g} stop={stop}", flush=True)

        after = {str(p): sha256(p) for p in inputs}
        source_after = {name: sha256(repo / name) for name in SOURCE_FILES}
        unchanged = before == after and source_before == source_after and script_hash == sha256(Path(__file__))
        save(out, "integrity.json", {"direct_sources_inputs_script_unchanged": unchanged,
             "input_sha256_after": after, "direct_source_sha256_after": source_after,
             "note": "Full protected/checkpoint integrity is coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on source/input/script integrity changed")
        save(out, "result.json", {"branch": args.branch, "arm": args.arm,
             "accepted_steps": accepted, "loss": loss, "phi": theta, "stop_reason": stop,
             "handoff": loss < 1e-20, "parameter_path_length": path_length,
             "backtracking_rejections": bt, "new_objective_evaluations": rt.objective_evaluations,
             "derivative_calls": rt.binding_counters.derivative_calls,
             "elapsed_seconds": time.monotonic() - started, "snapshots": snapshots,
             "bound_coordinates_ever": sorted(contacts), "trace_sha256": sha256(trace_path),
             "historical_matched_states": len(historical_matches),
             "historical_matched_steps": sorted(historical_matches),
             "historical_available_through": max(historical) if historical else None,
             "direct_integrity_unchanged": unchanged, "reused": False})
        save(out, "artifact_manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.branch} {args.arm}: {accepted} steps, loss={loss:.17g}, {stop}", flush=True)
    except BaseException as exc:
        if checkpoint is not None:
            try:
                checkpoint()
            except BaseException:
                pass
        save(out, "failure.json", {"accepted_steps": accepted,
             "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
             "status": "ALERT_OR_TECHNICAL_BLOCKER_NO_FURTHER_OPTIMIZATION"})
        raise


if __name__ == "__main__":
    main()
