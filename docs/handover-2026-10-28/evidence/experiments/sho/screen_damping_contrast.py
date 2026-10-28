#!/usr/bin/env python3
"""One arm of the explicitly approved four-case, two-arm damping contrast.

Maximum 50 additional accepted steps. Production source is never modified.
Control is the existing stabilized runtime. Intervention changes only its
residual-model damping to the approved formula, using the existing BVLS solver.
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

CASES = {"original_step1": ("original", 1), "extended_step0": ("extended", 0),
         "original_step37": ("original", 37), "extended_step50": ("extended", 50)}
CAP = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--arm", choices=("control", "intervention"), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; no overwrite or resume mode")
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
            OptInResidualN3Runtime, STABILIZED_RESIDUAL, residual_and_jacobian,
            _bounded_residual_least_squares,
        )
        from experiments.sho.resume_generic_update_targeted import Runtime
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

        def tree_equal(a, b):
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            if hasattr(a, "tolist"):
                a = a.tolist()
            if hasattr(b, "detach"):
                b = b.detach().cpu().numpy()
            if hasattr(b, "tolist"):
                b = b.tolist()
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
            if isinstance(a, bool) or isinstance(b, bool):
                return type(a) is type(b) and a == b
            if isinstance(a, (int,float)) and isinstance(b, (int,float)):
                return bits_equal(a,b)
            return a == b

        def damping_metadata(H, mu_H, applied=None):
            # Exact user-approved formula. No floor is applied when mu_H == 0.
            eigenvalues = torch.linalg.eigvalsh(H)
            if not bool(torch.isfinite(eigenvalues).all()) or not math.isfinite(mu_H):
                raise RuntimeError("TECHNICAL_BLOCKER: nonfinite ordinary damping/spectrum; no substitute direction or damping")
            s = max(float(eigenvalues.abs().max()), torch.finfo(torch.float64).tiny)
            floor = 1e-12 * s
            mu_A = 0.0 if mu_H == 0 else max(mu_H / 10, floor)
            intervention_floor_active = bool(mu_H != 0 and floor >= mu_H / 10)
            return {"mu_H": mu_H, "approved_intervention_mu_A": mu_A,
                "mu_applied": mu_A if applied is None else applied,
                "spectral_scale_s": s, "H_eigenvalues": eigenvalues,
                "floor": floor, "floor_active": bool(applied is None and intervention_floor_active),
                "approved_intervention_floor_active": intervention_floor_active,
                "floor_strictly_exceeds_mu_H_over_10": bool(mu_H != 0 and floor > mu_H / 10),
                "zero_mu_exception": mu_H == 0}

        class InterventionRuntime(Runtime):
            def __init__(self, adapter, trajectory, log):
                super().__init__(adapter, trajectory, log)
                self.derivative_calls = 0
                self.last_contrast = None

            def derivatives_and_direction(self, theta, count_route=True):
                self.derivative_calls += 1
                self.last_contrast = None
                values = super().derivatives_and_direction(theta, count_route=count_route)
                result, gradient, H, gn, pre, route, old_direction, old_qp = values
                if pre != "CONTINUE":
                    return values
                if old_qp is None or old_qp.get("damping") is None:
                    raise RuntimeError("TECHNICAL_BLOCKER: ordinary CONTINUE route lacks damping; no fallback")
                meta = damping_metadata(H, float(old_qp["damping"]))
                mu_A = meta["approved_intervention_mu_A"]
                residual, J = residual_and_jacobian(self.adapter, self.trajectory, theta)
                M = residual.numel()
                regularization = torch.tensor(M / 2 * mu_A, dtype=theta.dtype)
                matrix = torch.cat((J, torch.sqrt(regularization) * torch.eye(theta.numel(), dtype=theta.dtype)))
                augmented_residual = torch.cat((residual, torch.zeros_like(theta)))
                direction, active_set = _bounded_residual_least_squares(matrix, augmented_residual, theta, self.lower, self.upper)
                jd = J @ direction
                residual_prediction = float(-(2 * residual @ jd + jd @ jd) / M)
                full_prediction = residual_prediction - mu_A / 2 * float(direction @ direction)
                qp = {**active_set, "route_status": "ok", "damping": mu_A,
                      "quadratic_value": -full_prediction}
                self.last_contrast = {**meta, "ordinary_full_Hessian_direction": old_direction,
                    "ordinary_full_Hessian_qp": old_qp, "residual_count": M,
                    "residual_regularization": float(regularization),
                    "residual_predicted_decrease": residual_prediction,
                    "damped_model_predicted_decrease": full_prediction}
                return result, gradient, H, gn, pre, route, direction, qp

        endpoint, source_step = CASES[args.case]
        base = repo / "outputs/sho/n2_n3_basin_local_discrimination_20261013/local_models"
        doc_path = base / f"n00005_i00004_{endpoint}_step{source_step:05d}.json"
        phase_manifest_path = base / "manifest.json"
        phase_artifact_manifest_path = base / "artifact_manifest.json"
        doc = json.loads(doc_path.read_text())
        phase_manifest = json.loads(phase_manifest_path.read_text())
        phase_artifact_manifest = json.loads(phase_artifact_manifest_path.read_text())["sha256"]
        trace_source = Path(doc["source_trace"])
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        helper = Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs = (doc_path, phase_manifest_path, phase_artifact_manifest_path, trace_source, data, helper)
        before = {str(p): sha256(p) for p in inputs}
        sources_before = {name: sha256(repo / name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        for path in (doc_path, phase_manifest_path):
            if before[str(path)] != phase_artifact_manifest.get(path.name):
                raise RuntimeError("ALERT: phaseA artifact hash mismatch")
        for path in (trace_source, data):
            if before[str(path)] != phase_manifest["input_sha256"].get(str(path)):
                raise RuntimeError("ALERT: source trace/data hash differs from phaseA provenance")
        expected_identity = {"stem":"n00005_i00004", "endpoint":endpoint, "saved_step":source_step}
        if doc["identity"] != expected_identity:
            raise RuntimeError("ALERT: predeclared source identity mismatch")
        history = {}
        with trace_source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if source_step <= row["step"] <= source_step + CAP:
                    if row["step"] in history:
                        raise RuntimeError("ALERT: duplicate source control state")
                    history[row["step"]] = row
        if sorted(history) != list(range(source_step, source_step+CAP+1)):
            raise RuntimeError("BLOCKER: all 51 historical control states are required")
        source = history[source_step]
        if not tree_equal(source,doc["saved_state"]):
            raise RuntimeError("ALERT: local-model source differs from saved trace record")
        if source.get("stop_reason") is not None or source["loss"] < 1e-20 or source["stagnation_count"] >= prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
            raise RuntimeError("ALERT: requested source has an existing true stop; no reset permitted")
        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        theta = torch.tensor(source["phi"],dtype=torch.float64)
        baseline = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(),trajectory,Log(),direction_mode=STABILIZED_RESIDUAL)
        original = baseline.derivatives_and_direction(theta)
        original_result, original_g, original_H, original_gn, original_pre, original_route, original_d, original_qp = original
        checks = {"pre_continue":original_pre=="CONTINUE",
            "loss_bits":bits_equal(original_result.loss,source["loss"]),
            "gradient_norm_bits":bits_equal(original_gn,source["gradient_norm"]),
            "mu_H_bits":original_qp is not None and bits_equal(original_qp.get("damping"),doc["mu"]),
            "source_mu_bits":original_qp is not None and bits_equal(original_qp.get("damping"),source["mu"]),
            "direction_norm_bits":original_d is not None and bits_equal(float(original_d.norm()),source["direction_norm"]),
            "direction_bits":tree_equal(original_d,doc["stabilized_direction"]),
            "qp_bits":tree_equal(original_qp,doc["stabilized_qp"]),
            "gradient_bits":bits_equal(original_g,doc["gradient"]), "H_bits":bits_equal(original_H,doc["H"]),
            "active_bounds":torch.where((theta==baseline.lower)|(theta==baseline.upper))[0].tolist()==source["active_bounds"],
            "route":original_route==source["route"]}
        if not all(checks.values()):
            save(out,"ALERT_source_replay.json",{"checks":checks,"qp":original_qp,"direction":original_d})
            raise RuntimeError("ALERT: unchanged baseline source replay failed")
        validation_trial = replay(baseline,theta,source["loss"],original_g,original_d)
        checks["actual_trial_local_model_bits"] = tree_equal(validation_trial,doc["route_comparisons"]["stabilized"]["trial"])
        checks["actual_trial_trace_bits"] = tree_equal(validation_trial,source["trial"])
        if not all(checks.values()):
            save(out,"ALERT_source_trial.json",{"checks":checks,"trial":validation_trial})
            raise RuntimeError("ALERT: unchanged baseline actual trial replay failed")
        source_mu = damping_metadata(original_H,float(original_qp["damping"]),float(original_qp["damping"]))
        save(out,"source_verification.json",{"checks":checks,"source_state":source,
            "baseline_qp":original_qp,"baseline_direction":original_d,"baseline_trial":validation_trial,
            "source_damping":source_mu,"baseline_validation_trial_evaluations":baseline.objective_evaluations})
        lineage = {"case":args.case,"arm":args.arm,"identity":expected_identity,
            "source_trace":str(trace_source),"source_local_model":str(doc_path),
            "source_stagnation_count":source["stagnation_count"]}
        save(out,"manifest.json",{"lineage":lineage,"input_sha256":before,
            "direct_source_sha256":sources_before,"script":str(Path(__file__).resolve()),"script_sha256":script_hash,
            "approved_formula":"mu_A=0 if mu_H==0 else max(mu_H/10, 1e-12*s); s=max(max(abs(eigvalsh(H))), float64.tiny)",
            "cap_additional_accepted_steps":CAP,"production_modified":False,
            "protocol":"Exactly four declared cases and two arms. Coordinator runs control then intervention, examines each outcome, and owns full integrity/ALERT handling. No automatic next arm.",
            "control_replay":"Compare all reached source-offset states, including inherited counter, and every executed trial. Cap permits 51 states and 50 trial transitions; no 51st trial is executed.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})
        rt = (OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(),trajectory,Log(),direction_mode=STABILIZED_RESIDUAL)
              if args.arm=="control" else InterventionRuntime(GenericSHOUpdateNetworkAdapter(),trajectory,Log()))
        loss = source["loss"]
        counter = int(source["stagnation_count"])
        path_length, cumulative_path, bt = 0.0,source["path"],0
        stop = None
        contacts,snapshots,control_states,control_trials = set(),{},[],[]
        started = time.monotonic()
        trace_path = out / "damping_contrast.jsonl"

        def calls():
            return rt.binding_counters.derivative_calls if args.arm=="control" else rt.derivative_calls

        def work():
            return {"additional_accepted_steps":accepted,"cumulative_accepted_steps":source_step+accepted,
                "additional_parameter_path":path_length,"cumulative_parameter_path":cumulative_path,
                "additional_backtracking_rejections":bt,"cumulative_backtracking_rejections":source["backtracking"]+bt,
                "additional_trial_objective_evaluations":rt.objective_evaluations,
                "cumulative_trial_objective_evaluations":source["objective_evaluations"]+rt.objective_evaluations,
                "derivative_calls":calls(),"source_validation_trial_evaluations":baseline.objective_evaluations,
                "source_validation_derivative_calls":baseline.binding_counters.derivative_calls}

        def write_checkpoint():
            active = set(torch.where((theta==rt.lower)|(theta==rt.upper))[0].tolist())
            save(out,"diagnostic_state.json",{"lineage":lineage,"theta":theta,"loss":loss,
                "stagnation_count":counter,"stop_reason":stop,"work":work(),
                "bound_coordinates_ever":sorted(contacts|active),"snapshots":snapshots,
                "control_matched_states":control_states,"control_matched_trials":control_trials,
                "input_sha256":before,"direct_source_sha256":sources_before,"script_sha256":script_hash,
                "restart_note":"No resume entry point. Never reset a true stop or inherit a different arm's state."})

        checkpoint = write_checkpoint
        write_checkpoint()
        with trace_path.open("x") as stream:
            while True:
                current_step = source_step+accepted
                reference = history[current_step]
                if args.arm=="control":
                    state_checks = {"phi_bits":bits_equal(theta,reference["phi"]),"loss_bits":bits_equal(loss,reference["loss"]),
                        "stagnation_count":counter==reference["stagnation_count"]}
                    if not all(state_checks.values()):
                        save(out,"ALERT_control_state.json",{"step":current_step,"checks":state_checks,"theta":theta,"loss":loss,"reference":reference})
                        raise RuntimeError("ALERT: control state/counter differs bitwise")
                result,g,H,gn,pre,route,d,qp = rt.derivatives_and_direction(theta)
                if not bits_equal(result.loss,loss):
                    raise RuntimeError("ALERT: derivative-state loss replay mismatch")
                if pre=="CONTINUE":
                    if qp is None or qp.get("damping") is None:
                        raise RuntimeError("TECHNICAL_BLOCKER: CONTINUE lacks residual QP; no fallback")
                    if args.arm=="control":
                        contrast = damping_metadata(H,float(qp["damping"]),float(qp["damping"]))
                        contrast.update({"residual_count":rt.last_residual_state["residual"].numel(),
                            "residual_regularization":rt.last_residual_state["regularization"],
                            "residual_predicted_decrease":rt.last_residual_state["residual_predicted_decrease"],
                            "damped_model_predicted_decrease":rt.last_residual_state["full_predicted_decrease"]})
                    else:
                        contrast = rt.last_contrast
                else:
                    contrast = None  # Preserve the ordinary pre-step stop.
                if accepted==0:
                    first_checks = {"gradient_bits":bits_equal(g,original_g),"H_bits":bits_equal(H,original_H),
                        "raw_mu_H_bits":contrast is not None and bits_equal(contrast["mu_H"],doc["mu"]),
                        "applied_mu_bits":contrast is not None and bits_equal(qp["damping"],source_mu["mu_H"] if args.arm=="control" else source_mu["approved_intervention_mu_A"])}
                    save(out,"selected_arm_start.json",{"checks":first_checks,"contrast":contrast,"qp":qp,"direction":d})
                    if not all(first_checks.values()):
                        raise RuntimeError("ALERT: initial selected-arm damping/derivative mismatch")
                active = torch.where((theta==rt.lower)|(theta==rt.upper))[0].tolist()
                contacts.update(active)
                if args.arm=="control":
                    derivative_checks = {"gradient_norm_bits":bits_equal(gn,reference["gradient_norm"]),
                        "mu_bits":qp is not None and bits_equal(qp.get("damping"),reference["mu"]),
                        "direction_norm_bits":d is not None and bits_equal(float(d.norm()),reference["direction_norm"]),
                        "active_bounds":active==reference["active_bounds"],"route":route==reference["route"]}
                    if not all(derivative_checks.values()):
                        save(out,"ALERT_control_derivative.json",{"step":current_step,"checks":derivative_checks})
                        raise RuntimeError("ALERT: control derivative diagnostics differ")
                    control_states.append(current_step)
                projected = g.clone()
                projected[((theta==rt.lower)&(g>=0))|((theta==rt.upper)&(g<=0))]=0
                row = {"step":accepted,"source_offset_step":current_step,"phi":theta,"loss":loss,
                    "gradient_norm":gn,"projected_gradient_norm":float(projected.norm()),"pre":pre,"route":route,
                    "direction":d,"direction_norm":None if d is None else float(d.norm()),"qp":qp,"contrast":contrast,
                    "active_bounds":active,"stagnation_count":counter,"work":work()}
                if accepted in (0,1,10,50):
                    snapshots[str(accepted)] = serializable(row)
                if loss<1e-20:
                    stop="HANDOFF_THRESHOLD_REACHED"
                elif stop is None and pre!="CONTINUE":
                    stop=pre
                elif stop is None and accepted>=CAP:
                    stop="SCREEN_RESOURCE_LIMIT_50_ADDITIONAL_ACCEPTED_STEPS"
                if stop:
                    row["stop_reason"]=stop
                    append(stream,row)
                    write_checkpoint()
                    break
                trial=replay(rt,theta,loss,g,d)
                row["trial"]=trial
                if args.arm=="control":
                    if not tree_equal(trial,reference.get("trial")):
                        save(out,"ALERT_control_trial.json",{"step":current_step,"trial":trial,"reference":reference.get("trial")})
                        raise RuntimeError("ALERT: control accepted/rejected trial differs bitwise")
                    if not bits_equal(-float(qp["quadratic_value"]),reference["full_predicted_decrease"]):
                        raise RuntimeError("ALERT: control model prediction differs bitwise")
                    control_trials.append(current_step)
                if trial["accepted"]:
                    prediction=-float(qp["quadratic_value"])
                    counter=prod._updated_float64_stagnation_count(counter,loss,trial["loss"],prediction)
                    row["full_predicted_decrease"]=prediction
                    theta=torch.tensor(trial["phi"],dtype=torch.float64)
                    loss=trial["loss"]
                    path_length+=trial["displacement"]
                    cumulative_path+=trial["displacement"]
                    bt+=trial["rejections"]
                    accepted+=1
                    if counter>=prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
                        stop=prod.FLOAT64_STAGNATION_STOP_REASON
                else:
                    stop=trial["reason"]
                    bt+=trial.get("rejections",0)
                    row["stop_reason"]=stop
                append(stream,row)
                write_checkpoint()
                if not trial["accepted"]:
                    break
                if accepted in (1,10,50) or stop or loss<1e-20:
                    print(f"DAMPING {args.case} {args.arm} additional={accepted} loss={loss:.17g} stop={stop}",flush=True)
        after={str(p):sha256(p) for p in inputs}
        sources_after={name:sha256(repo/name) for name in SOURCE_FILES}
        unchanged=before==after and sources_before==sources_after and script_hash==sha256(Path(__file__))
        save(out,"integrity.json",{"direct_inputs_sources_script_unchanged":unchanged,"input_sha256_after":after,
            "direct_source_sha256_after":sources_after,"note":"Full protected/checkpoint checks remain coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on inputs/source/script changed")
        save(out,"result.json",{"lineage":lineage,"phi":theta,"loss":loss,"stagnation_count":counter,
            "stop_reason":stop,"handoff":loss<1e-20,"work":work(),"snapshots":snapshots,
            "control_matched_states":control_states,"control_matched_trials":control_trials,
            "bound_coordinates_ever":sorted(contacts),"elapsed_seconds":time.monotonic()-started,
            "trace_sha256":sha256(trace_path),"direct_integrity_unchanged":unchanged})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.case} {args.arm}: additional={accepted}, loss={loss:.17g}, {stop}",flush=True)
    except BaseException as exc:
        if checkpoint is not None:
            try:
                checkpoint()
            except BaseException:
                pass
        save(out,"failure.json",{"additional_accepted_steps":accepted,"exception_type":type(exc).__name__,
            "error":str(exc),"traceback":traceback.format_exc(),"status":"ALERT_OR_TECHNICAL_BLOCKER_NO_FURTHER_OPTIMIZATION"})
        raise


if __name__=="__main__":
    main()
