#!/usr/bin/env python3
"""Isolated finite crossover from one approved budget-limited N3 screen.

Only --out is written. Never reads or resumes the paused long recovery.
Original-route validation is separate from the selected continuation runtime.
The original stagnation counter is inherited without alteration.
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
SOURCE_TRACES = {
    "stabilized": "outputs/sho/n2_budget_discrimination_three_branch_20261013/n00005_i00004_extended_stabilized.jsonl",
    "full_hessian": "outputs/sho/n2_n3_basin_local_discrimination_20261013/full_hessian_screens/n00005_i00004_extended/full_hessian.jsonl",
}
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
MILESTONES = (0, 1, 10, 50, 100, 250, 500)
RESOURCE_STOP = "SCREEN_RESOURCE_LIMIT_500_ACCEPTED_STEPS"


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

    result = clean(payload)
    if nonfinite:
        result["nonfinite_values_replaced_by_null"] = nonfinite
    return result


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
    parser.add_argument("--source-route", choices=tuple(SOURCE_TRACES), required=True)
    parser.add_argument("--next-route", choices=tuple(SOURCE_TRACES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; no restart or overwrite mode")
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
        from experiments.sho.generic_update_residual_n3_opt_in import (
            OptInResidualN3Runtime, EXISTING_FULL_HESSIAN, STABILIZED_RESIDUAL,
        )
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        modes = {"stabilized": STABILIZED_RESIDUAL, "full_hessian": EXISTING_FULL_HESSIAN}
        source_path = repo / SOURCE_TRACES[args.source_route]
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        inputs = (source_path, data)
        input_before = {str(p): sha256(p) for p in inputs}
        source_before = {name: sha256(repo / name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        source_contacts = set()
        source = None
        rows_count = 0
        with source_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"] != rows_count:
                    raise RuntimeError("ALERT: source trace is not contiguous and unique")
                if rows_count < 500 and row.get("stop_reason") is not None:
                    raise RuntimeError("ALERT: source trace includes an earlier stopped state")
                source_contacts.update(row["active_bounds"])
                source = row
                rows_count += 1
        if source is None or rows_count != 501 or source["step"] != 500:
            raise RuntimeError("ALERT: source is not exactly a completed 500-step screen")
        if source.get("stop_reason") != RESOURCE_STOP:
            raise RuntimeError("ALERT: source did not terminate solely at its screen resource limit")
        if not math.isfinite(source["loss"]) or source["loss"] < 1e-20:
            raise RuntimeError("ALERT: source already meets handoff or has nonfinite loss")
        if source["stagnation_count"] >= prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
            raise RuntimeError("ALERT: source has an already-triggered existing stagnation stop")
        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        theta = torch.tensor(source["phi"], dtype=torch.float64)
        loss = source["loss"]
        original_rt = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(), trajectory, Log(), direction_mode=modes[args.source_route])
        original = original_rt.derivatives_and_direction(theta)
        original_loss, original_g, original_H, original_gn, original_pre, original_route, original_direction, original_qp = original
        checks = {
            "original_pre_continue": original_pre == "CONTINUE",
            "loss_bits": bits_equal(original_loss.loss, loss),
            "gradient_norm_bits": bits_equal(original_gn, source["gradient_norm"]),
            "mu_bits": original_qp is not None and bits_equal(original_qp.get("damping"), source["mu"]),
            "direction_norm_bits": original_direction is not None and bits_equal(float(original_direction.norm()), source["direction_norm"]),
            "active_bounds": torch.where((theta == original_rt.lower) | (theta == original_rt.upper))[0].tolist() == source["active_bounds"],
            "within_original_bounds": bool(((theta >= original_rt.lower) & (theta <= original_rt.upper)).all()),
            "curvature_route": original_route == source["route"],
        }
        if args.source_route == "full_hessian":
            checks["full_Hessian_raw_direction_present"] = source.get("direction") is not None
            checks["full_Hessian_raw_direction_bits"] = original_direction is not None and source.get("direction") is not None and bits_equal(original_direction, source["direction"])
        save(out, "source_verification.json", {"source": str(source_path), "source_state": source,
             "checks": checks, "original_mode": modes[args.source_route],
             "original_direction": original_direction, "original_qp": original_qp,
             "original_gradient": original_g, "original_H": original_H,
             "validation_derivative_calls": original_rt.binding_counters.derivative_calls})
        if not all(checks.values()):
            raise RuntimeError("ALERT: original-route source replay differs or source is already stopped")

        lineage = {"branch": "n00005_i00004", "N2_endpoint": "extended_3000_evaluations",
                   "source_route": args.source_route, "next_route": args.next_route,
                   "source_trace": str(source_path), "source_accepted_steps": source["step"],
                   "source_loss": source["loss"], "source_stagnation_count": source["stagnation_count"],
                   "source_parameter_path": source["path"],
                   "source_backtracking_rejections": source["backtracking"],
                   "source_trial_objective_evaluations": source["objective_evaluations"]}
        save(out, "manifest.json", {"lineage": lineage, "source_state": source,
             "input_sha256": input_before, "direct_source_sha256": source_before,
             "script": str(Path(__file__).resolve()), "script_sha256": script_hash,
             "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                 "scipy": scipy.__version__, "torch_threads": torch.get_num_threads()},
             "protocol": "500 additional accepted steps maximum; original stagnation counter inherited; only direction_mode changes; unchanged audit.replay, acceptance, bounds, stopping and strict handoff.",
             "scope": "Finite continuation of a resource-limited screen, never continuation of a scientifically or numerically terminated state or a paused long run.",
             "counter_note": "Lifetime trial evaluations/path/backtracking add source metrics to new work. Original-route validation derivative is reported separately. No automatic resume mode."})

        rt = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(), trajectory, Log(), direction_mode=modes[args.next_route])
        counter = int(source["stagnation_count"])
        path_length, bt = 0.0, 0
        snapshots, contacts = {}, set()
        stop = None
        started = time.monotonic()
        trace_path = out / "crossover.jsonl"

        def counters():
            return {"additional_accepted_steps": accepted, "cumulative_accepted_steps": 500 + accepted,
                    "additional_parameter_path_length": path_length,
                    "cumulative_parameter_path_length": source["path"] + path_length,
                    "additional_backtracking_rejections": bt,
                    "cumulative_backtracking_rejections": source["backtracking"] + bt,
                    "additional_trial_objective_evaluations": rt.objective_evaluations,
                    "cumulative_trial_objective_evaluations": source["objective_evaluations"] + rt.objective_evaluations,
                    "additional_derivative_calls": rt.binding_counters.derivative_calls,
                    "source_validation_derivative_calls": original_rt.binding_counters.derivative_calls}

        def write_checkpoint():
            current_active = set(torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist())
            save(out, "diagnostic_state.json", {"lineage": lineage, "theta": theta, "loss": loss,
                 "stagnation_count": counter, "stop_reason": stop, **counters(),
                 "additional_bound_coordinates_ever": sorted(contacts | current_active),
                 "cumulative_bound_coordinates_ever": sorted(source_contacts | contacts | current_active),
                 "snapshots": snapshots, "input_sha256": input_before,
                 "direct_source_sha256": source_before, "script_sha256": script_hash,
                 "restart_note": "Persisted for exact reconstruction only; no automatic resume implementation or permission to resume a terminated state."})

        checkpoint = write_checkpoint
        write_checkpoint()
        with trace_path.open("x") as stream:
            while True:
                result, gradient, H, gn, pre, route, direction, qp = rt.derivatives_and_direction(theta)
                if not bits_equal(result.loss, loss):
                    raise RuntimeError("ALERT: selected-route state loss replay mismatch")
                if accepted == 0:
                    initial_checks = {"gradient_bits": bits_equal(gradient, original_g),
                        "H_bits": bits_equal(H, original_H), "loss_bits": bits_equal(result.loss, source["loss"]),
                        "mu_bits": qp is not None and bits_equal(qp.get("damping"), source["mu"])}
                    if args.source_route == args.next_route:
                        initial_checks["same_route_direction_bits"] = direction is not None and bits_equal(direction, original_direction)
                    save(out, "selected_route_start.json", {"checks": initial_checks,
                         "direction": direction, "qp": qp, "mode": modes[args.next_route], "pre": pre})
                    if not all(initial_checks.values()):
                        raise RuntimeError("ALERT: selected-route start differs beyond the direction hook")
                projected = gradient.clone()
                projected[((theta == rt.lower) & (gradient >= 0)) | ((theta == rt.upper) & (gradient <= 0))] = 0
                active = torch.where((theta == rt.lower) | (theta == rt.upper))[0].tolist()
                contacts.update(active)
                row = {"step": accepted, "cumulative_step": 500 + accepted,
                       "phi": theta, "loss": loss, "gradient_norm": gn,
                       "projected_gradient_norm": float(projected.norm()), "route": route,
                       "mu": None if qp is None else qp.get("damping"),
                       "direction_norm": None if direction is None else float(direction.norm()),
                       "direction": direction, "qp": qp, "active_bounds": active,
                       "path": path_length, "backtracking": bt,
                       "objective_evaluations": rt.objective_evaluations,
                       "stagnation_count": counter, "work": counters()}
                if accepted in MILESTONES:
                    snapshots[str(accepted)] = serializable(row)
                if loss < 1e-20:
                    stop = "HANDOFF_THRESHOLD_REACHED"
                elif stop is None and pre != "CONTINUE":
                    stop = pre
                elif stop is None and accepted >= 500:
                    stop = RESOURCE_STOP
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
                write_checkpoint()
                if not trial["accepted"]:
                    break
                if accepted in MILESTONES or stop or loss < 1e-20:
                    print(f"CROSSOVER {args.source_route}->{args.next_route} additional={accepted} cumulative={500+accepted} loss={loss:.17g} stop={stop}", flush=True)

        input_after = {str(p): sha256(p) for p in inputs}
        source_after = {name: sha256(repo / name) for name in SOURCE_FILES}
        unchanged = input_before == input_after and source_before == source_after and script_hash == sha256(Path(__file__))
        save(out, "integrity.json", {"direct_inputs_sources_script_unchanged": unchanged,
             "input_sha256_after": input_after, "direct_source_sha256_after": source_after,
             "note": "Full checkpoint/protected-file integrity remains coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on source/input/script integrity changed")
        save(out, "result.json", {"lineage": lineage, "phi": theta, "loss": loss,
             "stop_reason": stop, "handoff": loss < 1e-20, "stagnation_count": counter,
             **counters(), "snapshots": snapshots,
             "additional_bound_coordinates_ever": sorted(contacts),
             "cumulative_bound_coordinates_ever": sorted(source_contacts | contacts),
             "elapsed_seconds": time.monotonic() - started,
             "trace_sha256": sha256(trace_path), "direct_integrity_unchanged": unchanged,
             "resource_note": "SCREEN_RESOURCE_LIMIT_500_ACCEPTED_STEPS means 500 additional accepted steps, cumulative step1000; unfinished, not converged."})
        save(out, "artifact_manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.source_route}->{args.next_route}: additional={accepted}, loss={loss:.17g}, {stop}", flush=True)
    except BaseException as exc:
        if checkpoint is not None:
            try:
                checkpoint()
            except BaseException:
                pass
        save(out, "failure.json", {"additional_accepted_steps": accepted,
             "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
             "status": "ALERT_OR_TECHNICAL_BLOCKER_NO_FURTHER_OPTIMIZATION"})
        raise


if __name__ == "__main__":
    main()
