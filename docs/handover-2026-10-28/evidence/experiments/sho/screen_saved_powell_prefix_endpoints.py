#!/usr/bin/env python3
"""Fresh stabilized N3 screen of a predeclared saved Powell prefix minimum.

No Powell invocation, restart, interpolation, new N1 point or initializer.
Only --out is written. Numerical controls match the existing finite screen.
The sibling full-Hessian script supplies persistence/hash helpers only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
from screen_existing_full_hessian_endpoints import SOURCE_FILES, sha256, save, append, serializable

EXPECTED = {2000: (1990, 0.0004855535865235672),
            2500: (2499, 0.00027175828031855595)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=tuple(EXPECTED), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; no overwrite or automatic resume")
    if not (repo / SOURCE_FILES[0]).is_file():
        parser.error("--repo does not contain the expected runtime")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    accepted, checkpoint = 0, None
    try:
        import numpy as np
        import scipy
        import torch
        from experiments.resume_hybrid_student_after_ckdtree import Log
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import OptInResidualN3Runtime, STABILIZED_RESIDUAL
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        stem = "n00005_i00004"
        base = repo / "outputs/sho/n2_budget_discrimination_three_branch_20261013"
        trace_source = base / (stem + "_n2_evaluations.jsonl")
        extended_path = base / (stem + "_n2_extended.json")
        original_path = repo / "outputs/sho/generic_update_network_targeted_recovery/companion_6" / (stem + "_n2.json")
        artifact_manifest_path = base / "artifact_manifest.json"
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        helper = Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs = (trace_source, extended_path, original_path, artifact_manifest_path, data, helper)
        before = {str(p): sha256(p) for p in inputs}
        source_before = {name: sha256(repo / name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        extended = json.loads(extended_path.read_text())
        original = json.loads(original_path.read_text())
        artifact_manifest = json.loads(artifact_manifest_path.read_text())
        checks = {
            "trace_hash_matches_saved_record": before[str(trace_source)] == extended["trace_hash"],
            "original_hash_matches_saved_record": before[str(original_path)] == extended["source_hash"],
            "trace_matches_batch_manifest": before[str(trace_source)] == artifact_manifest.get(str(trace_source)),
            "endpoint_record_matches_batch_manifest": before[str(extended_path)] == artifact_manifest.get(str(extended_path)),
            "branch": extended["branch"] == stem,
            "single_Powell_invocation_recorded": extended["uninterrupted_single_invocation"] is True,
            "saved_exact_prefix_count": extended["exact_prefix_evaluations"] == 1500,
            "original_evaluation_count": original["result"]["evaluations"] == 1500,
            "extended_evaluation_count": extended["result"]["evaluations"] == 3000,
            "same_box_lower_bits": bits_equal(original["box_lower"], extended["box_lower"]),
            "same_box_upper_bits": bits_equal(original["box_upper"], extended["box_upper"]),
        }
        rows = [json.loads(line) for line in trace_source.read_text().splitlines()]
        checks["exact_evaluation_sequence"] = [r["evaluation"] for r in rows] == list(range(1, 3001))
        checks["original_history_length"] = len(original["history"]) == 1500
        if not all(checks.values()):
            save(out, "ALERT_provenance.json", {"checks": checks})
            raise RuntimeError("ALERT: source provenance or endpoint identity mismatch")
        lower, upper = np.asarray(extended["box_lower"]), np.asarray(extended["box_upper"])
        for record in rows:
            phi = np.asarray(record["phi"], dtype=np.float64)
            if phi.shape != (22,) or not np.isfinite(phi).all() or not math.isfinite(record["loss"]) or not ((phi >= lower) & (phi <= upper)).all():
                raise RuntimeError("ALERT: saved Powell point is nonfinite, malformed or outside its original box")
        for index, (saved, reference) in enumerate(zip(rows[:1500], original["history"]), 1):
            if not bits_equal(saved["phi"], reference["phi"]) or not bits_equal(saved["loss"], reference["loss"]):
                raise RuntimeError(f"ALERT: original saved Powell prefix mismatch at evaluation {index}")
        # Strict updates retain the earliest point if equal loss occurs.
        best = rows[0]
        best_at = {}
        for record in rows:
            if record["loss"] < best["loss"]:
                best = record
            if record["evaluation"] in (1500, args.budget, 3000):
                best_at[record["evaluation"]] = best
        for budget, reference in ((1500, original["result"]), (3000, extended["result"])):
            checks[f"best_{budget}_theta_bits"] = bits_equal(best_at[budget]["phi"], reference["theta"])
            checks[f"best_{budget}_loss_bits"] = bits_equal(best_at[budget]["loss"], reference["loss"])
        checks["original_endpoint_in_extended_record_bits"] = bits_equal(original["result"]["theta"], extended["original_endpoint"]["theta"]) and bits_equal(original["result"]["loss"], extended["original_endpoint"]["loss"])
        selected = best_at[args.budget]
        expected_evaluation, expected_loss = EXPECTED[args.budget]
        checks["predeclared_record_identity"] = selected["evaluation"] == expected_evaluation and bits_equal(selected["loss"], expected_loss)
        save(out, "endpoint_selection.json", {"branch": stem, "evaluation_budget": args.budget,
            "selected_record": selected, "cross_checked_best_endpoints": best_at,
            "checks": checks, "exact_saved_prefix_comparisons": 1500,
            "selection": "First strict minimum among saved evaluations 1..budget; no Powell execution or interpolation."})
        if not all(checks.values()):
            raise RuntimeError("ALERT: saved best-ever endpoint verification mismatch")
        lineage = {"branch": stem, "saved_Powell_evaluation_budget": args.budget,
                   "selected_best_evaluation": selected["evaluation"], "source_trace": str(trace_source),
                   "selected_loss": selected["loss"], "selected_phi": selected["phi"]}
        save(out, "manifest.json", {"lineage": lineage, "input_sha256": before,
            "direct_source_sha256": source_before, "script": str(Path(__file__).resolve()),
            "script_sha256": script_hash,
            "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                            "scipy": scipy.__version__, "torch_threads": torch.get_num_threads()},
            "protocol": "Existing stabilized direction/runtime and audit.replay; unchanged finite-screen 500 accepted-step cap, counters, bounds, backtracking, stopping and strict handoff.",
            "limits": "Single existing branch, dependent saved endpoints; no new Powell call and no independent seed. Full protected/checkpoint integrity is coordinator-owned."})
        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        rt = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(), trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
        theta = torch.tensor(selected["phi"], dtype=torch.float64)
        loss = selected["loss"]
        with torch.no_grad():
            initial_loss = float(rt.adapter.objective(theta, trajectory))
        if not bits_equal(initial_loss, loss):
            raise RuntimeError("ALERT: selected saved endpoint objective differs bitwise")
        path_length, bt, counter = 0.0, 0, 0
        snapshots, contacts = {}, set()
        stop = None
        started = time.monotonic()
        trace_path = out / "stabilized.jsonl"

        def write_checkpoint():
            active = set(torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist())
            save(out, "diagnostic_state.json", {"lineage": lineage, "theta": theta, "loss": loss,
                "accepted_steps": accepted, "stagnation_count": counter, "stop_reason": stop,
                "parameter_path_length": path_length, "backtracking_rejections": bt,
                "trial_objective_evaluations": rt.objective_evaluations,
                "derivative_calls": rt.binding_counters.derivative_calls,
                "bound_coordinates_ever": sorted(contacts | active), "snapshots": snapshots,
                "input_sha256": before, "direct_source_sha256": source_before, "script_sha256": script_hash,
                "restart_note": "No automatic resume; exact reconstruction must be verified before any future restart."})

        checkpoint = write_checkpoint
        write_checkpoint()
        with trace_path.open("x") as stream:
            while True:
                result, gradient, H, gn, pre, route, direction, qp = rt.derivatives_and_direction(theta)
                if not bits_equal(result.loss, loss):
                    raise RuntimeError("ALERT: derivative-state loss replay mismatch")
                projected = gradient.clone()
                projected[((theta == rt.lower) & (gradient >= 0)) | ((theta == rt.upper) & (gradient <= 0))] = 0
                active = torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist()
                contacts.update(active)
                row = {"step": accepted, "phi": theta, "loss": loss, "gradient_norm": gn,
                    "projected_gradient_norm": float(projected.norm()), "route": route,
                    "mu": None if qp is None else qp.get("damping"),
                    "direction_norm": None if direction is None else float(direction.norm()),
                    "direction": direction, "qp": qp, "active_bounds": active,
                    "path": path_length, "backtracking": bt, "objective_evaluations": rt.objective_evaluations,
                    "stagnation_count": counter}
                if accepted in (0, 1, 10, 50, 100, 250, 500):
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
                    prediction = -float(qp["quadratic_value"])
                    counter = prod._updated_float64_stagnation_count(counter, loss, trial["loss"], prediction)
                    row["full_predicted_decrease"] = prediction
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
                write_checkpoint()
                if not trial["accepted"]:
                    break
                if accepted in (1, 10, 50, 100, 250, 500) or stop or loss < 1e-20:
                    print(f"SAVED_PREFIX budget={args.budget} accepted={accepted} loss={loss:.17g} stop={stop}", flush=True)
        after = {str(p): sha256(p) for p in inputs}
        source_after = {name: sha256(repo / name) for name in SOURCE_FILES}
        unchanged = before == after and source_before == source_after and script_hash == sha256(Path(__file__))
        save(out, "integrity.json", {"direct_sources_inputs_script_unchanged": unchanged,
            "input_sha256_after": after, "direct_source_sha256_after": source_after,
            "note": "Full protected/checkpoint integrity is coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on source/input/script integrity changed")
        save(out, "result.json", {"lineage": lineage, "accepted_steps": accepted, "loss": loss,
            "phi": theta, "stop_reason": stop, "handoff": loss < 1e-20,
            "parameter_path_length": path_length, "backtracking_rejections": bt,
            "new_objective_evaluations": rt.objective_evaluations,
            "derivative_calls": rt.binding_counters.derivative_calls,
            "snapshots": snapshots, "bound_coordinates_ever": sorted(contacts),
            "elapsed_seconds": time.monotonic() - started, "trace_sha256": sha256(trace_path),
            "direct_integrity_unchanged": unchanged,
            "note": "A screen resource-limit stop is unfinished, not converged. Trial objective counts exclude derivative/start verification calls."})
        save(out, "artifact_manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE budget={args.budget}, accepted={accepted}, loss={loss:.17g}, {stop}", flush=True)
    except BaseException as exc:
        if checkpoint is not None:
            try:
                checkpoint()
            except BaseException:
                pass
        save(out, "failure.json", {"accepted_steps": accepted, "exception_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "status": "ALERT_OR_TECHNICAL_BLOCKER_NO_FURTHER_OPTIMIZATION"})
        raise


if __name__ == "__main__":
    main()
