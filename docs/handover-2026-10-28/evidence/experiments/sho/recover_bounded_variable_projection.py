#!/usr/bin/env python3
"""One original (5,4) N2 endpoint: reviewed preflight, then isolated reduced TRF.

No production changes, fixtures, resumed runs, second starts, or automatic rollout.
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

LAYOUT = [dict(name="0.weight", shape=[4,2], start=0, stop=8),
          dict(name="0.bias", shape=[4], start=8, stop=12),
          dict(name="2.weight", shape=[2,4], start=12, stop=20),
          dict(name="2.bias", shape=[2], start=20, stop=22)]
MILESTONES = (0, 1, 10, 50, 100, 250, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "optimize"), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--approved-preflight-sha256")
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists(): parser.error("--out must be fresh; no overwrite or resume")
    if args.mode == "optimize" and not (args.preflight and args.approved_preflight_sha256):
        parser.error("optimization requires reviewed --preflight and its --approved-preflight-sha256 artifact manifest")
    if args.mode == "preflight" and (args.preflight or args.approved_preflight_sha256):
        parser.error("approval arguments apply only to optimize")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    last_profile, state, checkpoint = None, None, None
    accepted, profile_calls, derivative_calls, source_objective_calls = 0, 0, 0, 0
    stop = None
    started = time.monotonic()
    try:
        import numpy as np
        import scipy
        import torch
        from scipy.linalg import svd
        from scipy.optimize._lsq import common as scipy_common
        from scipy.optimize._lsq import trf as scipy_trf
        from scipy.optimize._lsq.common import CL_scaling_vector, solve_lsq_trust_region, update_tr_radius, make_strictly_feasible
        from scipy.optimize._lsq.trf import select_step
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import residual_vector, residual_and_jacobian, _bounded_residual_least_squares
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        adapter = GenericSHOUpdateNetworkAdapter()
        if adapter.layout != LAYOUT or adapter.dimension != 22:
            raise RuntimeError("ALERT: adapter parameter layout mismatch")
        if not (np.all(adapter.lower == -1.99) and np.all(adapter.upper == 1.99)):
            raise RuntimeError("ALERT: original bounds changed")
        data = repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        n2_path = repo/"outputs/sho/generic_update_network_targeted_recovery/companion_6/n00005_i00004_n2.json"
        batch = repo/"outputs/sho/n2_budget_discrimination_three_branch_20261013"
        trace_path = batch/"n00005_i00004_original_stabilized.jsonl"
        batch_hash_path = batch/"artifact_manifest.json"
        helper = Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs = [data, n2_path, trace_path, batch_hash_path, helper]
        before = {str(p): sha256(p) for p in inputs}
        source_before = {name: sha256(repo/name) for name in SOURCE_FILES}
        library_before = {str(Path(module.__file__).resolve()): sha256(Path(module.__file__)) for module in (scipy_common, scipy_trf)}
        script_hash = sha256(Path(__file__))
        if before[str(trace_path)] != json.loads(batch_hash_path.read_text()).get(str(trace_path)):
            raise RuntimeError("ALERT: saved batch trace artifact hash mismatch")
        endpoint = json.loads(n2_path.read_text())["result"]
        if endpoint["evaluations"] != 1500: raise RuntimeError("ALERT: original N2 budget identity mismatch")
        controls = {}
        with trace_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"] in MILESTONES:
                    if str(row["step"]) in controls: raise RuntimeError("ALERT: duplicate saved control milestone")
                    controls[str(row["step"])] = row
                if row["step"] >= 500: break
        source = controls["0"]
        source_theta = torch.tensor(endpoint["theta"], dtype=torch.float64)
        if not (bits_equal(source_theta, source["phi"]) and bits_equal(endpoint["loss"], source["loss"])):
            raise RuntimeError("ALERT: N2 endpoint differs from saved original step0")
        if source.get("stop_reason") or source["loss"] < 1e-20:
            raise RuntimeError("ALERT: source has a stop or handoff; no continuation authorized")
        with np.load(data) as corpus: trajectory = torch.from_numpy(corpus["prefix_6"].copy())
        lower, upper = torch.from_numpy(adapter.lower.copy()), torch.from_numpy(adapter.upper.copy())
        with torch.no_grad(): source_loss = float(adapter.objective(source_theta, trajectory))
        source_objective_calls += 1
        if not bits_equal(source_loss, endpoint["loss"]): raise RuntimeError("ALERT: original source native objective mismatch")
        if not bool(((source_theta >= lower) & (source_theta <= upper)).all()): raise RuntimeError("ALERT: source infeasible")
        approval = None
        approved_initial = None
        if args.mode == "optimize":
            preflight = args.preflight.resolve(strict=True)
            approval_path = preflight/"artifact_manifest.json"
            if sha256(approval_path) != args.approved_preflight_sha256:
                raise RuntimeError("ALERT: approved preflight manifest hash mismatch")
            approved_hashes = json.loads(approval_path.read_text())["sha256"]
            for name, digest in approved_hashes.items():
                p = preflight/name
                if p.parent != preflight or sha256(p) != digest: raise RuntimeError("ALERT: approved preflight artifact changed")
            pre_result = json.loads((preflight/"result.json").read_text())
            pre_manifest = json.loads((preflight/"manifest.json").read_text())
            if pre_result["status"] != "PREFLIGHT_COMPLETE_REQUIRES_REVIEW" or not pre_result["integrity_unchanged"]:
                raise RuntimeError("ALERT: preflight did not complete")
            for key, actual in (("input_sha256", before), ("direct_source_sha256", source_before), ("library_source_sha256", library_before), ("script_sha256", script_hash)):
                if pre_manifest[key] != actual: raise RuntimeError("ALERT: preflight/current provenance disagreement: "+key)
            approved_initial = json.loads((preflight/"initial_profile.json").read_text())
            approval = {"preflight": str(preflight), "artifact_manifest_sha256": args.approved_preflight_sha256,
                        "review_authorization": "Coordinator explicitly supplied this reviewed preflight manifest digest."}
            for p in (approval_path, preflight/"manifest.json", preflight/"result.json", preflight/"initial_profile.json"):
                inputs.append(p)
            # Keep primary source keys identical to the approved preflight; hash review inputs separately.
        approval_before = {str(p):sha256(p) for p in inputs if str(p) not in before}
        save(out, "manifest.json", {"mode":args.mode, "endpoint":"original (5,4), original N2 1500, saved stabilized step0",
            "input_sha256":before, "approval_input_sha256":approval_before, "direct_source_sha256":source_before,
            "library_source_sha256":library_before, "script_sha256":script_hash, "script":str(Path(__file__).resolve()),
            "approval":approval, "layout":LAYOUT, "lower":lower, "upper":upper,
            "protocol":"One start only. Head10 eliminated by existing undamped untruncated BVLS from zero head at each hidden12 candidate. Native original MSE and exact full bounds. Fixed-face reduced residual Jacobian includes residual-weighted mixed term. SciPy reflective primitives; native strict-decrease acceptance. No locality/segment guard, restart, second endpoint, refit alternation, fixture, production change or automatic rollout.",
            "outer_controls":{"accepted_cap":500,"optimization_profile_call_cap":3000,"initial_radius":"norm(h/sqrt(ColemanLi_v)); inherited zero fallback1", "initial_alpha":0,
                "theta_backoff":"max(.995,1-||g*v||inf)","hidden_strict_feasibility":"SciPy make_strictly_feasible rstep0 nextafter, no head clipping",
                "tolerances":"No added outer ftol/gtol/xtol or rank truncation. Inherited solve_lsq_trust_region uses full_rank s_min > eps*m*s_max to gate GN trial, radius rtol=.01 and max_iter10; regularized solve retains all singular modes."},
            "preflight":"48 central profile probes at eps^(1/3)*max(1,abs(h_j)), then half this increment. Derivative comparisons only on same head face; raw errors descriptive and reviewed by coordinator, no invented tolerance gate.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()},
            "count_note":"Profile count includes initial refit and every outer candidate (cap3000), or initial+48 probes in preflight. Native source replay and derivative work separate; adapter call counters disclose internal forwards."})
        save(out, "source_verification.json", {"source_phi":source_theta,"source_loss":source_loss,"source_trace_row":source,
            "native_objective_bits":True,"endpoint_bits":True,"saved_controls":controls})

        class Handoff(Exception): pass

        def work():
            return {"profile_evaluations":profile_calls,"BVLS_calls":profile_calls,"accepted_hidden_steps":accepted,
                    "reduced_derivative_calls":derivative_calls,"source_objective_replays":source_objective_calls,
                    "adapter_calls":dict(adapter.calls),"initial_head_refit_is_separate":True}

        def profile(h, purpose):
            nonlocal profile_calls, last_profile
            h = torch.as_tensor(np.array(h, dtype=np.float64, copy=True))
            if not bool(torch.isfinite(h).all()) or not bool(((h>=lower[:12])&(h<=upper[:12])).all()):
                raise RuntimeError("TECHNICAL_BLOCKER: nonfinite or infeasible hidden candidate")
            profile_calls += 1
            zero = torch.cat((h, torch.zeros(10, dtype=torch.float64)))
            zero_r, zero_J = residual_and_jacobian(adapter, trajectory, zero)
            A = zero_J[:,12:22]
            anchor_r = -trajectory[1:].reshape(-1)
            singular = torch.linalg.svdvals(A)
            if not bool(torch.isfinite(singular).all()) or bool((singular<=0).any()):
                raise RuntimeError("TECHNICAL_BLOCKER: nonpositive/nonfinite head singular value; no truncation")
            head, solver = _bounded_residual_least_squares(A,anchor_r,torch.zeros(10),lower[12:22],upper[12:22])
            phi = torch.cat((h,head))
            row = {"profile_index":profile_calls,"purpose":purpose,"hidden":h,"phi":phi,"head_solver":solver,
                   "A":A,"head_singular_values":singular,"anchor_residual":anchor_r,
                   "native_zero_minus_anchor_residual":zero_r-anchor_r}
            if not bool(torch.isfinite(phi).all()) or not bool(((phi>=lower)&(phi<=upper)).all()):
                save(out,"infeasible_profile.json",row)
                raise RuntimeError("TECHNICAL_BLOCKER: exact full candidate box/finite check failed; no clipping")
            with torch.no_grad():
                native_loss = float(adapter.objective(phi,trajectory))
                r = residual_vector(adapter,trajectory,phi).detach()
            assembled = float((r*r).sum()/r.numel())
            if not math.isfinite(native_loss) or not bool(torch.isfinite(r).all()):
                save(out,"nonfinite_profile.json",{**row,"loss":native_loss,"residual":r})
                raise RuntimeError("TECHNICAL_BLOCKER: nonfinite native profile evaluation")
            if not bits_equal(native_loss,assembled): raise RuntimeError("ALERT: profile native/residual MSE assembly differs")
            model_r = anchor_r+A@head
            kkt, native_kkt = A.T@model_r, A.T@r
            active = torch.tensor(solver["active"])
            row.update(loss=native_loss,residual=r,linear_residual=model_r,native_minus_linear_residual=r-model_r,
                half_sum_model_head_gradient=kkt,half_sum_native_head_gradient=native_kkt,
                native_MSE_head_gradient=2/r.numel()*native_kkt,
                free_model_gradient=kkt[active==0],free_native_gradient=native_kkt[active==0],
                lower_active_native_gradient=native_kkt[active==-1],upper_active_native_gradient=native_kkt[active==1],
                active_bound_identity=bool(((active!=-1)|(head==lower[12:22])).all() and ((active!=1)|(head==upper[12:22])).all()),
                free_at_exact_bound=torch.where((active==0)&((head==lower[12:22])|(head==upper[12:22])))[0],
                active_zero_native_multiplier=torch.where((active!=0)&(native_kkt==0))[0],
                native_MSE_bits=True,feasible=True)
            if not row["active_bound_identity"]: raise RuntimeError("ALERT: solver active status does not match exact head bound")
            last_profile = row
            persisted = row if args.mode=="preflight" or purpose=="initial_zero_anchor_refit" else {
                key:value for key,value in row.items() if key not in ("A","anchor_residual","native_zero_minus_anchor_residual")}
            with (out/"profiles.jsonl").open("a") as stream: append(stream,persisted)
            if native_loss<1e-20:
                save(out,"ALERT_handoff_candidate.json",row)
                print(f"ALERT FEASIBLE_PROFILE_HANDOFF profile={profile_calls} loss={native_loss:.17g}; stop before further work",flush=True)
                raise Handoff("native feasible profile below unchanged handoff threshold")
            return row

        def differentiate(row, independent=False):
            nonlocal derivative_calls
            derivative_calls += 1
            phi, r0 = row["phi"],row["residual"]
            r,J = residual_and_jacobian(adapter,trajectory,phi)
            if not bits_equal(r,r0) or not bits_equal(J[:,12:22],row["A"]):
                raise RuntimeError("ALERT: native profile residual/head Jacobian replay failed")
            B,A = J[:,:12],J[:,12:22]
            active = torch.tensor(row["head_solver"]["active"])
            free = torch.where(active==0)[0]
            AF = A[:,free]
            fixed_r = r.detach()
            mixed = torch.autograd.functional.hessian(lambda value: fixed_r@residual_vector(adapter,trajectory,value),phi)
            C = mixed[12:22,:12][free]
            if len(free):
                U,s,Vh = torch.linalg.svd(AF,full_matrices=False)
                if bool((s<=0).any()) or not bool(torch.isfinite(s).all()):
                    raise RuntimeError("TECHNICAL_BLOCKER: nonpositive/nonfinite free-face singular value; no fallback")
                residual_term = U@((Vh@C)/s[:,None])
                Jr = B-U@(U.T@B)-residual_term
                head_derivative = -Vh.T@((U.T@B)/s[:,None]+(Vh@C)/(s[:,None]**2))
            else:
                s=torch.empty(0); residual_term=torch.zeros_like(B); Jr=B.clone(); head_derivative=torch.empty((0,12))
            if not bool(torch.isfinite(Jr).all()): raise RuntimeError("TECHNICAL_BLOCKER: nonfinite reduced Jacobian")
            g = Jr.T@r
            details={"full_native_J":J,"hidden_partial_J":B,"free_head_indices":free,"free_head_singular_values":s,
                "mixed_residual_C":C,"free_head_derivative":head_derivative,"reduced_J":Jr,
                "residual_weighted_reduced_term":residual_term,"reduced_J_singular_values":torch.linalg.svdvals(Jr),
                "implicit_stationarity_difference":AF.T@Jr+C,
                "J_reconstruction_difference":Jr-(B+AF@head_derivative),
                "half_sum_reduced_gradient":g,"MSE_reduced_gradient":2/r.numel()*g,
                "MSE_envelope_gradient":2/r.numel()*(B.T@r),
                "MSE_reduced_minus_envelope":2/r.numel()*(g-B.T@r),
                "qualification":"Derivative of the locally fixed active face with full free-column rank, assuming head stationarity; float64 solver/affine discrepancies are reported, not certified away. Rebuilt after every accepted candidate."}
            if independent:
                native = adapter.derivatives(phi,trajectory)
                if not bits_equal(native.loss,row["loss"]): raise RuntimeError("ALERT: native derivative scalar mismatch")
                details["native_hessian_mixed_identity_difference"] = native.raw_hessian[12:22,:12][free]*r.numel()/2-(AF.T@B+C)
                details["native_MSE_head_gradient"] = native.gradient[12:22]
                details["native_MSE_hidden_minus_envelope"] = native.gradient[:12]-2/r.numel()*(B.T@r)
            return details

        initial = profile(source_theta[:12].numpy(),"initial_zero_anchor_refit")
        if initial["loss"]>source_loss:
            save(out,"worse_initial_profile.json",{"source_loss":source_loss,"initial_profile":initial})
            raise RuntimeError("TECHNICAL_BLOCKER: initial native conditional refit is worse than source; no optimization")
        initial_derivative = differentiate(initial,independent=args.mode=="preflight")
        save(out,"initial_profile.json",initial)
        save(out,"initial_derivative.json",initial_derivative)
        if approved_initial is not None:
            checks={name:bits_equal(initial[name],approved_initial[name]) for name in ("phi","loss","residual","A")}
            checks["active_face"] = initial["head_solver"]["active"]==approved_initial["head_solver"]["active"]
            save(out,"approved_initial_replay.json",checks)
            if not all(checks.values()): raise RuntimeError("ALERT: approved preflight initial profile differs")

        if args.mode=="preflight":
            differences=[]
            h=source_theta[:12].numpy().copy()
            eps_step=np.finfo(np.float64).eps**(1/3)
            face=initial["head_solver"]["active"]
            for j in range(12):
                for factor in (1.0,0.5):
                    increment=eps_step*max(1.0,abs(h[j]))*factor
                    minus,plus=h.copy(),h.copy()
                    minus[j]-=increment; plus[j]+=increment
                    if minus[j]<adapter.lower[j] or plus[j]>adapter.upper[j]:
                        differences.append({"coordinate":j,"factor":factor,"status":"INFEASIBLE_PROBE_NOT_EVALUATED","increment":increment})
                        continue
                    lo=profile(minus,f"preflight_coordinate{j}_factor{factor}_minus")
                    hi=profile(plus,f"preflight_coordinate{j}_factor{factor}_plus")
                    sameface=lo["head_solver"]["active"]==face and hi["head_solver"]["active"]==face
                    entry={"coordinate":j,"factor":factor,"requested_increment":increment,
                           "actual_coordinate_difference":plus[j]-minus[j],"minus_profile":lo["profile_index"],"plus_profile":hi["profile_index"],
                           "minus_face":lo["head_solver"]["active"],"plus_face":hi["head_solver"]["active"],"same_face":sameface,
                           "status":"SAME_FACE_DESCRIPTIVE_COMPARISON" if sameface else "ACTIVE_FACE_CROSSING_NO_DERIVATIVE_COMPARISON"}
                    if sameface:
                        fd=(hi["residual"]-lo["residual"])/(plus[j]-minus[j])
                        analytic=initial_derivative["reduced_J"][:,j]
                        error=fd-analytic
                        entry.update(finite_difference=fd,analytic=analytic,difference=error,
                            error_norm=float(error.norm()),analytic_norm=float(analytic.norm()),finite_difference_norm=float(fd.norm()),
                            finite_difference_MSE=(hi["loss"]-lo["loss"])/(plus[j]-minus[j]),analytic_MSE=float(initial_derivative["MSE_reduced_gradient"][j]))
                    differences.append(entry)
                    save(out,"derivative_preflight.json",{"checks":differences,"no_numerical_pass_gate":True})
            state={"status":"PREFLIGHT_COMPLETE_REQUIRES_REVIEW","source_loss":source_loss,"profile_loss":initial["loss"],
                   "phi":initial["phi"],"work":work(),"comparisons":differences,
                   "limits":"No optimization run. Finite-difference and stationarity errors are descriptive; coordinator must review before passing the approval digest."}
        else:
            current, derivatives=initial,initial_derivative
            x=current["hidden"].numpy().copy()
            r=current["residual"].numpy().copy(); J=derivatives["reduced_J"].numpy().copy()
            m,n=J.shape
            lb,ub=adapter.lower[:12].copy(),adapter.upper[:12].copy()
            g=J.T@r
            v,_=CL_scaling_vector(x,g,lb,ub)
            if not np.all(np.isfinite(v)) or not np.all(v>0): raise RuntimeError("TECHNICAL_BLOCKER: degenerate initial reflective scaling")
            Delta=float(np.linalg.norm(x/np.sqrt(v)))
            if Delta==0: Delta=1.0
            alpha=0.0
            path_hidden,path_full=0.0,0.0
            snapshots={}
            trial_candidates=0
            def persist():
                save(out,"diagnostic_state.json",{"profile":current,"derivatives":derivatives,"accepted_steps":accepted,
                    "profile_calls":profile_calls,"trial_candidates":trial_candidates,"radius":Delta,"alpha":alpha,
                    "hidden_path":path_hidden,"full_parameter_path":path_full,"work":work(),"stop_reason":stop,
                    "approval":approval,"automatic_resume":False})
            checkpoint=persist
            def record_accepted(update=None):
                row={"step":accepted,"profile":current,"derivatives":derivatives,"update":update,"work":work(),
                     "radius":Delta,"alpha":alpha,"hidden_path":path_hidden,"full_parameter_path":path_full}
                if accepted in MILESTONES:
                    row["saved_stabilized_control"]=controls.get(str(accepted))
                    row["comparison_limit"]="Same counted hidden steps only, not equal work; every candidate includes a conditional head solve, and initial refit is separate."
                    snapshots[str(accepted)]={"loss":current["loss"],"profile_index":current["profile_index"],"work":work()}
                    print(f"VARPRO accepted={accepted} profiles={profile_calls} loss={current['loss']:.17g}",flush=True)
                with (out/"accepted.jsonl").open("a") as stream: append(stream,row)
                persist()
            record_accepted()
            while stop is None:
                if accepted>=500: stop="RESOURCE_LIMIT_500_ACCEPTED_HIDDEN_STEPS"; break
                if profile_calls>=3000: stop="RESOURCE_LIMIT_3000_PROFILE_EVALUATIONS"; break
                g=J.T@r
                if np.linalg.norm(g)==0: stop="ZERO_REDUCED_RESIDUAL_MODEL_GRADIENT"; break
                v,dv=CL_scaling_vector(x,g,lb,ub)
                if not np.all(np.isfinite(v)) or not np.all(v>0): stop="DEGENERATE_REFLECTIVE_SCALING"; break
                d=np.sqrt(v); diag_h=g*dv; gh=d*g; Jh=J*d
                augmented=np.vstack((Jh,np.diag(np.sqrt(diag_h))))
                U,s,Vt=svd(augmented,full_matrices=False)
                uf=U.T@np.concatenate((r,np.zeros(n)))
                theta_backoff=max(.995,1-float(np.linalg.norm(g*v,ord=np.inf)))
                seen=set()
                while stop is None:
                    if profile_calls>=3000: stop="RESOURCE_LIMIT_3000_PROFILE_EVALUATIONS"; break
                    ph,solved_alpha,nit=solve_lsq_trust_region(n,m,uf,s,Vt.T,Delta,initial_alpha=alpha)
                    p=d*ph
                    step,step_h,prediction=select_step(x,Jh,diag_h,gh,p,ph,d,Delta,lb,ub,theta_backoff)
                    candidate=make_strictly_feasible(x+step,lb,ub,rstep=0)
                    if not np.all(np.isfinite(candidate)): stop="NONFINITE_HIDDEN_TRIAL"; break
                    if np.array_equal(candidate,x) or candidate.tobytes() in seen:
                        stop="FLOAT64_UNCHANGED_OR_REPEATED_HIDDEN_TRIAL"; break
                    seen.add(candidate.tobytes())
                    trial_candidates+=1
                    trial=profile(candidate,f"outer_trial{trial_candidates}_accepted_before{accepted}")
                    reduction=(current["loss"]-trial["loss"])*m/2
                    next_delta,ratio=update_tr_radius(Delta,reduction,prediction,float(np.linalg.norm(step_h)),float(np.linalg.norm(step_h))>.95*Delta)
                    improves=trial["loss"]<current["loss"]
                    radius_valid=math.isfinite(next_delta) and next_delta>0
                    update={"trial":trial_candidates,"profile_index":trial["profile_index"],"accepted_before":accepted,
                        "accepted":improves and radius_valid,"native_strict_improvement":improves,
                        "radius_valid":radius_valid,"old_loss":current["loss"],"new_loss":trial["loss"],
                        "radius_before":Delta,"radius_after":next_delta,"prediction_half_sum_squares":prediction,
                        "prediction_MSE":2*prediction/m,"actual_MSE_decrease":current["loss"]-trial["loss"],"ratio":ratio,
                        "alpha_solved":solved_alpha,"root_iterations":nit,"hidden_displacement":candidate-x,
                        "head_face_before":current["head_solver"]["active"],"head_face_after":trial["head_solver"]["active"],
                        "head_face_changed":current["head_solver"]["active"]!=trial["head_solver"]["active"]}
                    with (out/"trials.jsonl").open("a") as stream: append(stream,update)
                    if not radius_valid:
                        stop="NUMERICAL_TRUST_RADIUS_EXHAUSTED"; break
                    alpha=solved_alpha*Delta/next_delta
                    Delta=next_delta
                    if improves:
                        path_hidden+=float(np.linalg.norm(candidate-x))
                        path_full+=float((trial["phi"]-current["phi"]).norm())
                        current=trial; x=candidate.copy(); accepted+=1
                        derivatives=differentiate(current)
                        r=current["residual"].numpy().copy(); J=derivatives["reduced_J"].numpy().copy()
                        record_accepted(update)
                        break
                    persist()
            persist()
            state={"status":"PILOT_COMPLETE","stop_reason":stop,"handoff":False,"phi":current["phi"],"loss":current["loss"],
                "source_loss":source_loss,"initial_profile_loss":initial["loss"],"initial_head_refit_displacement":float((initial["phi"]-source_theta).norm()),
                "work":work(),"trial_candidates":trial_candidates,"snapshots":snapshots,"saved_controls":controls,
                "hidden_path":path_hidden,"full_parameter_path_excluding_initial_refit":path_full,
                "limits":"Finite single-start diagnostic, no basin, constrained-stationarity or recovery impossibility certificate. New recovery method, not unchanged N3 or equal-compute comparison."}
        after={str(p):sha256(p) for p in inputs}
        unchanged=all(after[k]==v for k,v in {**before,**approval_before}.items()) and source_before=={name:sha256(repo/name) for name in SOURCE_FILES} and library_before=={p:sha256(Path(p)) for p in library_before} and script_hash==sha256(Path(__file__))
        save(out,"integrity.json",{"unchanged":unchanged,"input_sha256_after":after,"full_protected_guard":"coordinator-owned"})
        if not unchanged: raise RuntimeError("ALERT: relied-on source/input/script/library changed")
        state.update(integrity_unchanged=True,elapsed_seconds=time.monotonic()-started)
        save(out,"result.json",state)
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.mode}: {state['status']} profiles={profile_calls} accepted={accepted}",flush=True)
    except BaseException as exc:
        is_handoff=type(exc).__name__=="Handoff"
        status="ALERT_HANDOFF_CANDIDATE" if is_handoff else ("ALERT_SOLVER_CYCLE" if "cycle" in str(exc).lower() else ("ALERT_INTEGRITY_OR_REPLAY" if str(exc).startswith("ALERT") else "TECHNICAL_STOP"))
        if checkpoint is not None:
            try: checkpoint()
            except BaseException: pass
        save(out,"failure.json",{"status":status,"error":str(exc),"type":type(exc).__name__,"traceback":traceback.format_exc(),
            "last_completed_profile":last_profile,"accepted_steps":accepted,"profile_calls":profile_calls,
            "reduced_derivative_calls":derivative_calls,"source_objective_calls":source_objective_calls,
            "no_fallback_or_automatic_followup":True})
        integrity_checks={}
        integrity_errors={}
        # Read-only integrity bookkeeping also runs after a handoff candidate;
        # no additional model evaluation, derivative, solver, or rollout occurs.
        for group in ("before","approval_before","source_before","library_before"):
            expected=locals().get(group)
            if expected is None:
                integrity_errors[group]="Not initialized before stop; integrity coverage incomplete."
                continue
            for name,digest in expected.items():
                p=repo/name if group=="source_before" else Path(name)
                try: integrity_checks[str(p)]=sha256(p)==digest
                except BaseException as integrity_exc: integrity_errors[str(p)]=str(integrity_exc)
        if "script_hash" in locals():
            try: integrity_checks[str(Path(__file__).resolve())]=sha256(Path(__file__))==script_hash
            except BaseException as integrity_exc: integrity_errors["script"]=str(integrity_exc)
        else: integrity_errors["script"]="Baseline script hash unavailable."
        save(out,"integrity.json",{"unchanged":not integrity_errors and all(integrity_checks.values()),
            "checks":integrity_checks,"verification_errors":integrity_errors,"on_failure_or_handoff":True,
            "full_protected_guard":"coordinator-owned","handoff_scope":"Feasible candidate only; no rollout or complete-recovery certification."})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="artifact_manifest.json"}})
        print(f"{status}: {exc}",flush=True)
        raise


if __name__=="__main__":
    main()
