#!/usr/bin/env python3
"""One saved-state conditional readout audit, with hidden parameters fixed.

One call to the existing bounded residual least-squares solver, no ridge,
rank cutoff, subsequent N3 continuation or alternating fit. Writes only --out.
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
from screen_existing_full_hessian_endpoints import SOURCE_FILES, sha256, save

CASES = {
    "54_original1": ("n00005_i00004", "original", 1),
    "54_extended0": ("n00005_i00004", "extended", 0),
    "54_original37": ("n00005_i00004", "original", 37),
    "54_extended50": ("n00005_i00004", "extended", 50),
    "43_original10": ("n00004_i00003", "original", 10),
    "43_extended10": ("n00004_i00003", "extended", 10),
}
LAYOUT = [
    {"name":"0.weight", "shape":[4,2], "start":0, "stop":8},
    {"name":"0.bias", "shape":[4], "start":8, "stop":12},
    {"name":"2.weight", "shape":[2,4], "start":12, "stop":20},
    {"name":"2.bias", "shape":[2], "start":20, "stop":22},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; no overwrite, batch loop or resume mode")
    if not (repo / SOURCE_FILES[0]).is_file():
        parser.error("--repo does not contain the expected runtime")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    started = time.monotonic()
    try:
        import numpy as np
        import scipy
        import torch
        from experiments.resume_hybrid_student_after_ckdtree import Log
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import (
            residual_and_jacobian, residual_vector, _bounded_residual_least_squares,
        )
        from experiments.sho.resume_generic_update_targeted import Runtime
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        stem, endpoint, step = CASES[args.case]
        base = repo / "outputs/sho/n2_n3_basin_local_discrimination_20261013/local_models"
        doc_path = base / f"{stem}_{endpoint}_step{step:05d}.json"
        manifest_path, artifact_manifest_path = base/"manifest.json", base/"artifact_manifest.json"
        doc = json.loads(doc_path.read_text())
        source_manifest = json.loads(manifest_path.read_text())
        source_artifact_manifest = json.loads(artifact_manifest_path.read_text())["sha256"]
        trace_source = Path(doc["source_trace"])
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        helper = Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs = (doc_path,manifest_path,artifact_manifest_path,trace_source,data,helper)
        before = {str(p):sha256(p) for p in inputs}
        sources_before = {name:sha256(repo/name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        for path in (doc_path,manifest_path):
            if before[str(path)] != source_artifact_manifest.get(path.name):
                raise RuntimeError("ALERT: phaseA artifact hash mismatch")
        for path in (trace_source,data):
            if before[str(path)] != source_manifest["input_sha256"].get(str(path)):
                raise RuntimeError("ALERT: phaseA source trace/data hash mismatch")
        expected_identity = {"stem":stem,"endpoint":endpoint,"saved_step":step}
        if doc["identity"] != expected_identity:
            raise RuntimeError("ALERT: predeclared source identity mismatch")
        source = None
        with trace_source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"]==step:
                    if source is not None:
                        raise RuntimeError("ALERT: duplicate source state")
                    source=row
        if source is None:
            raise RuntimeError("BLOCKER: required saved source state unavailable")
        if source != doc["saved_state"]:
            raise RuntimeError("ALERT: phaseA source state disagrees with trace")
        if source.get("stop_reason") is not None or source["loss"]<1e-20 or source["stagnation_count"]>=prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
            raise RuntimeError("ALERT: source already has an existing true stop; no reset")
        adapter = GenericSHOUpdateNetworkAdapter()
        if adapter.layout != LAYOUT or adapter.dimension != 22:
            raise RuntimeError("ALERT: adapter hidden/readout parameter layout changed")
        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        theta = torch.tensor(source["phi"],dtype=torch.float64)
        with torch.no_grad():
            source_loss=float(adapter.objective(theta,trajectory))
        r,J=residual_and_jacobian(adapter,trajectory,theta)

        def native_gradient(value):
            flat=value.detach().clone().requires_grad_(True)
            native_residual=adapter.forward(flat,trajectory[:-1])-trajectory[1:]
            native_loss=(native_residual*native_residual).sum()/native_residual.numel()
            gradient=torch.autograd.grad(native_loss,flat)[0]
            return float(native_loss.detach()),gradient.detach()

        native_source_loss,g=native_gradient(theta)
        checks={"source_loss_bits":bits_equal(source_loss,source["loss"]),
            "source_phi_bits":bits_equal(theta,doc["phi"]),
            "r_bits":bits_equal(r,doc["residual"]),"J_bits":bits_equal(J,doc["J"]),
            "native_loss_bits":bits_equal(native_source_loss,source_loss),
            "native_gradient_bits":bits_equal(g,doc["gradient"]),
            "gradient_norm_bits":bits_equal(float(torch.linalg.vector_norm(g)),source["gradient_norm"]),
            "global_bounds":bool(np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99))}
        save(out,"source_verification.json",{"identity":expected_identity,"checks":checks,
            "layout":adapter.layout,"source_state":source,"phi":theta,"residual":r,"J":J,"native_gradient":g})
        if not all(checks.values()):
            raise RuntimeError("ALERT: unchanged source replay/shape/bound identity failed")
        lineage={"case":args.case,"identity":expected_identity,"source_trace":str(trace_source),"source_local_model":str(doc_path)}
        save(out,"manifest.json",{"lineage":lineage,"input_sha256":before,"direct_source_sha256":sources_before,
            "script":str(Path(__file__).resolve()),"script_sha256":script_hash,
            "layout":adapter.layout,"fixed_hidden_indices":list(range(12)),"readout_indices":list(range(12,22)),
            "protocol":"One existing BVLS call on J[:,12:22] and r with existing readout parameter bounds; no regularization, truncation, new tolerance, full-parameter solve or continuation. Evaluate full candidate; optionally replay unchanged strict acceptance only if feasible and improving.",
            "limits":"Conditional readout candidate only; no certified global optimum for actual float64 computation, basin identification or broader recovery claim. Full integrity is coordinator-owned.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})
        rt=Runtime(adapter,trajectory,Log())
        # Runtime is used solely for unchanged candidate/acceptance/scalar-loss
        # operators. Its full-Hessian derivative/direction method is never called.
        A=J[:,12:22]
        readout=theta[12:22].clone()
        lower,upper=rt.lower[12:22],rt.upper[12:22]
        singular=torch.linalg.svdvals(A)
        save(out,"pre_solve.json",{"readout_matrix":A,"all_singular_values":singular,"residual":r,
            "theta":theta,"readout":readout,"lower":lower,"upper":upper,
            "step_lower":lower-readout,"step_upper":upper-readout,
            "solver":"existing _bounded_residual_least_squares without augmentation; all singular values retained"})
        delta,solver=_bounded_residual_least_squares(A,r,readout,lower,upper)
        d=torch.zeros_like(theta)
        d[12:22]=delta
        candidate=theta+d
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("TECHNICAL_BLOCKER: nonfinite conditional candidate; no fallback")
        if not bits_equal(candidate[:12],theta[:12]):
            save(out,"ALERT_hidden_change.json",{"source":theta,"candidate":candidate,"direction":d})
            raise RuntimeError("ALERT: conditional candidate altered fixed hidden parameter bits")
        candidate_in_bounds=bool(((candidate>=rt.lower)&(candidate<=rt.upper)).all())
        if not candidate_in_bounds:
            save(out,"infeasible_candidate.json",{
                "status":"TECHNICAL_BLOCKER_INFEASIBLE_CANDIDATE",
                "source_theta":theta,"candidate":candidate,"lifted_direction":d,
                "readout_direction":delta,"solver":solver,
                "lower":rt.lower,"upper":rt.upper,
                "candidate_lower_margins":candidate-rt.lower,
                "candidate_upper_margins":rt.upper-candidate,
                "below_lower_indices":torch.where(candidate<rt.lower)[0],
                "above_upper_indices":torch.where(candidate>rt.upper)[0],
                "candidate_objective_evaluated":False,
                "note":"Rounded full candidate violates the exact original box. No clipping, tolerance, fallback, handoff classification or continuation."})
            raise RuntimeError("TECHNICAL_BLOCKER_INFEASIBLE_CANDIDATE: rounded full candidate violates the exact box; no objective evaluation or fallback")
        model_residual=r+A@delta
        model_grad_half_sum=A.T@model_residual
        model_grad_mse=2/r.numel()*model_grad_half_sum
        active=torch.tensor(solver["active"],dtype=torch.int64)
        projected_model=model_grad_half_sum.clone()
        step_lower,step_upper=lower-readout,upper-readout
        projected_model[((delta==step_lower)&(model_grad_half_sum>=0))|((delta==step_upper)&(model_grad_half_sum<=0))]=0
        model_details={"linear_residual":model_residual,"linear_model_loss":float((model_residual*model_residual).sum()/r.numel()),
            "readout_model_gradient_half_sum_squares":model_grad_half_sum,
            "readout_model_gradient_MSE":model_grad_mse,
            "free_model_gradient_half_sum_squares":model_grad_half_sum[active==0],
            "free_model_gradient_norm":float(model_grad_half_sum[active==0].norm()),
            "projected_model_gradient_half_sum_squares":projected_model,
            "projected_model_gradient_norm":float(projected_model.norm()),
            "solver_lower_active_indices":torch.where(active==-1)[0],
            "solver_upper_active_indices":torch.where(active==1)[0],
            "lower_active_gradient_values":model_grad_half_sum[active==-1],
            "upper_active_gradient_values":model_grad_half_sum[active==1],
            "exact_lower_sign_violation_indices":torch.where((active==-1)&(model_grad_half_sum<0))[0],
            "exact_upper_sign_violation_indices":torch.where((active==1)&(model_grad_half_sum>0))[0],
            "note":"All signs/residuals are descriptive float64 solver diagnostics; no new KKT tolerance or numerical acceptance rule."}
        candidate_loss=rt.scalar_loss(candidate.numpy())
        if not math.isfinite(candidate_loss):
            save(out,"nonfinite_candidate.json",{"candidate":candidate,"candidate_loss":candidate_loss,
                "candidate_in_bounds":candidate_in_bounds,"direction":d,"solver":solver})
            raise RuntimeError("TECHNICAL_BLOCKER: nonfinite actual candidate objective; no fallback or continuation")
        if candidate_loss<1e-20:
            print(f"ALERT CONDITIONAL_READOUT_BELOW_HANDOFF {args.case} loss={candidate_loss:.17g} in_bounds={candidate_in_bounds}; no replay or continuation",flush=True)
        candidate_residual=residual_vector(adapter,trajectory,candidate).detach()
        affine_error=candidate_residual-model_residual
        details={"source_phi":theta,"source_loss":source_loss,"readout_direction":delta,
            "lifted_direction":d,"full_candidate_phi":candidate,"full_candidate_loss":candidate_loss,
            "exact_hidden_bits_unchanged":True,"candidate_in_bounds":candidate_in_bounds,
            "candidate_exact_active_bounds":torch.where((candidate==rt.lower)|(candidate==rt.upper))[0],
            "candidate_lower_margins":candidate-rt.lower,"candidate_upper_margins":rt.upper-candidate,
            "solver":solver,"all_readout_singular_values":singular,"model":model_details,
            "actual_candidate_residual":candidate_residual,"affine_residual_identity_error":affine_error,
            "affine_residual_identity_error_norm":float(affine_error.norm()),
            "actual_loss_minus_linear_model_loss":candidate_loss-model_details["linear_model_loss"],
            "note":"Affine identity differences are reported, not compared against a numerical tolerance or treated as bitwise replay failures."}
        if candidate_loss<1e-20:
            save(out,"candidate.json",details)
            result={"status":"ALERT_CONDITIONAL_READOUT_BELOW_HANDOFF","lineage":lineage,
                "candidate_loss":candidate_loss,"candidate_phi":candidate,"candidate_in_bounds":candidate_in_bounds,
                "handoff_eligible_by_loss_and_bounds":candidate_in_bounds,"acceptance_replay_executed":False,
                "reason":"Below existing loss threshold; immediate coordinator review required. No subsequent solve/recovery."}
        else:
            native_candidate_loss,candidate_gradient=native_gradient(candidate)
            if not bits_equal(native_candidate_loss,candidate_loss):
                raise RuntimeError("ALERT: native candidate MSE differs from assembled actual objective")
            actual_readout_gradient=candidate_gradient[12:22]
            actual_projected=actual_readout_gradient.clone()
            candidate_readout=candidate[12:22]
            actual_projected[((candidate_readout==lower)&(actual_readout_gradient>=0))|((candidate_readout==upper)&(actual_readout_gradient<=0))]=0
            details["actual_readout_gradient"]={"native_MSE_gradient":actual_readout_gradient,
                "projected_native_MSE_gradient":actual_projected,"projected_norm":float(actual_projected.norm()),
                "native_minus_linear_model_gradient":actual_readout_gradient-model_grad_mse,
                "source_directional_derivative_full_native_gradient":float(g@d),
                "method":"Torch autograd of native forward residual-square-sum/divide MSE; scalar verified bitwise against assembled objective. No Hessian/full-parameter solve."}
            improves=rt.controls.accepts(candidate_loss,source_loss)
            trial=None
            evaluations_before_replay=rt.objective_evaluations
            if candidate_in_bounds and improves:
                trial=replay(rt,theta,source_loss,g,d)
                if trial["accepted"]:
                    accepted_phi=torch.tensor(trial["phi"],dtype=torch.float64)
                    if not bits_equal(accepted_phi[:12],theta[:12]):
                        raise RuntimeError("ALERT: accepted replay altered fixed hidden parameter bits")
                    accepted_residual=residual_vector(adapter,trajectory,accepted_phi).detach()
                    details["accepted_replay_candidate"]={"phi":accepted_phi,"actual_residual":accepted_residual,
                        "actual_displacement":accepted_phi-theta,"same_as_full_candidate_bits":bits_equal(accepted_phi,candidate)}
                    if trial["loss"]<1e-20:
                        print(f"ALERT CONDITIONAL_READOUT_REPLAY_BELOW_HANDOFF {args.case} loss={trial['loss']:.17g}; no continuation",flush=True)
            details["strict_improvement_decision"]=improves
            details["unchanged_acceptance_replay"]=trial
            save(out,"candidate.json",details)
            replay_handoff=trial is not None and trial["accepted"] and trial["loss"]<1e-20
            result={"status":"ALERT_CONDITIONAL_READOUT_REPLAY_BELOW_HANDOFF" if replay_handoff else "CONDITIONAL_READOUT_AUDIT_COMPLETE",
                "lineage":lineage,"source_loss":source_loss,"full_candidate_loss":candidate_loss,
                "full_candidate_in_bounds":candidate_in_bounds,"strict_candidate_improvement":improves,
                "acceptance_replay_executed":trial is not None,"acceptance_replay":trial,
                "replay_trial_objective_evaluations":rt.objective_evaluations-evaluations_before_replay,
                "full_candidate_distinguished_from_replay":True,"no_subsequent_recovery":True}
        after={str(p):sha256(p) for p in inputs}
        sources_after={name:sha256(repo/name) for name in SOURCE_FILES}
        unchanged=before==after and sources_before==sources_after and script_hash==sha256(Path(__file__))
        save(out,"integrity.json",{"direct_sources_inputs_script_unchanged":unchanged,
            "input_sha256_after":after,"direct_source_sha256_after":sources_after,
            "note":"Full protected/checkpoint integrity remains coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on input/source/script integrity changed")
        result.update({"source_objective_verification_calls":1,
            "candidate_and_replay_scalar_objective_evaluations":rt.objective_evaluations,
            "direct_candidate_scalar_objective_evaluations":1,"BVLS_calls":1,
            "direct_integrity_unchanged":unchanged,"elapsed_seconds":time.monotonic()-started,
            "interpretation":"Finite-precision conditional candidate, not a certified global optimum or a new recovery policy."})
        save(out,"result.json",result)
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.case}: {result['status']}; full_candidate_loss={candidate_loss:.17g}",flush=True)
    except BaseException as exc:
        save(out,"failure.json",{"exception_type":type(exc).__name__,"error":str(exc),
            "traceback":traceback.format_exc(),"status":"TECHNICAL_BLOCKER_INFEASIBLE_CANDIDATE" if str(exc).startswith("TECHNICAL_BLOCKER_INFEASIBLE_CANDIDATE") else "ALERT_OR_TECHNICAL_BLOCKER_NO_FALLBACK_OR_CONTINUATION"})
        raise


if __name__=="__main__":
    main()
