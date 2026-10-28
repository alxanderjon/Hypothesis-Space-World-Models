#!/usr/bin/env python3
"""Approved joint-model residual-correction pilot with unchanged head profiling.

New hidden trust-BOX method, not the preceding fixed-face TRF algorithm.
One fresh original-N2 pilot; reuse reviewed original preflight and cached repair.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode=True
from screen_existing_full_hessian_endpoints import sha256,save,append

LAYOUT=[dict(name="0.weight",shape=[4,2],start=0,stop=8),dict(name="0.bias",shape=[4],start=8,stop=12),
        dict(name="2.weight",shape=[2,4],start=12,stop=20),dict(name="2.bias",shape=[2],start=20,stop=22)]
OLD_DRIVER_SHA="37f0ecefa9e81ba5e9640add62f394aa3f64b1c52ede01cbfd98e7d755d30750"
ORIGINAL_JOINT_DRIVER_SHA="9e31920e8d617f9e3ebdb4abde9e081302f7adc9a8b7767d13a0f02e035f8380"
REFINED_SOLVER_SHA="edd972e83e4076dd8692d6ab8dfb0aea4df93875627d24d1d16de1ba3768d175"
MILESTONES=(0,1,10,50,100,250,500)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("optimize",),default="optimize")
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--approved-preflight-sha256",required=True)
    parser.add_argument("--repair-test",type=Path,required=True)
    parser.add_argument("--approved-repair-sha256",required=True)
    args=parser.parse_args()
    repo,out=args.repo.resolve(strict=True),args.out.resolve()
    if out.exists(): parser.error("--out must be fresh, never resume or overwrite")
    out.mkdir(parents=True,exist_ok=False);os.chdir(repo);sys.path.insert(0,str(repo))
    before={};coord_before={};baseline_complete=False;approval_before={}
    profiles=models=centers=accepted=source_objective_calls=0
    last_profile=last_model=None;checkpoint=None;stop=None
    script_hash=sha256(Path(__file__));started=time.monotonic()
    try:
        import numpy as np
        import scipy
        import torch
        from scipy.optimize._lsq import common as scipy_common
        from scipy.optimize._lsq.common import CL_scaling_vector,update_tr_radius
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import residual_vector,residual_and_jacobian,_bounded_residual_least_squares
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        oldroot=repo/"outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy"
        pilot,terminal=oldroot/"pilot",oldroot/"terminal_validation"
        # Preserve complete prior scientific artifacts and their provenance.
        for directory in (pilot,terminal):
            artifact_path=directory/"artifact_manifest.json"
            artifacts=json.loads(artifact_path.read_text())["sha256"]
            for name,digest in artifacts.items():
                path=directory/name
                if path.parent!=directory or sha256(path)!=digest: raise RuntimeError("ALERT: prior artifact hash mismatch: "+str(path))
                before[str(path)]=digest
            before[str(artifact_path)]=sha256(artifact_path)
            prior_manifest=json.loads((directory/"manifest.json").read_text())
            for key in ("input_sha256","approval_input_sha256","direct_source_sha256","library_source_sha256"):
                for name,digest in prior_manifest.get(key,{}).items():
                    path=repo/name if key=="direct_source_sha256" else Path(name)
                    if sha256(path)!=digest: raise RuntimeError("ALERT: prior relied-on file mismatch: "+str(path))
                    before[str(path)]=digest
            path=Path(prior_manifest["script"])
            if sha256(path)!=prior_manifest["script_sha256"]: raise RuntimeError("ALERT: prior driver changed")
            before[str(path)]=prior_manifest["script_sha256"]
        oldmanifest=json.loads((pilot/"manifest.json").read_text())
        if oldmanifest["script_sha256"]!=OLD_DRIVER_SHA: raise RuntimeError("ALERT: unexpected audited profile implementation")
        helper=Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        before[str(helper)]=sha256(helper)
        for path in (repo/"README.md",oldroot/"joint_model_plan.json"):
            before[str(path)]=sha256(path)
        common_path=Path(scipy_common.__file__).resolve();before[str(common_path)]=sha256(common_path)
        coord_before={str(path):sha256(path) for path in sorted((repo/"coordination").glob("*.md"))}
        oldresult=json.loads((pilot/"result.json").read_text());oldstate=json.loads((pilot/"diagnostic_state.json").read_text())
        oldinitial=json.loads((pilot/"initial_profile.json").read_text())
        terminalresult=json.loads((terminal/"result.json").read_text())
        terminalreplay=json.loads((terminal/"terminal_replay.json").read_text())
        if oldresult["status"]!="PILOT_COMPLETE" or not oldresult["integrity_unchanged"] or oldstate["accepted_steps"]!=156 or oldresult["stop_reason"]!="FLOAT64_UNCHANGED_OR_REPEATED_HIDDEN_TRIAL":
            raise RuntimeError("ALERT: prior pilot terminal identity mismatch")
        if terminalresult["status"]!="TERMINAL_PROFILE_AUDIT_COMPLETE" or not terminalresult["exact_terminal_replay"] or not terminalresult["integrity_unchanged"]:
            raise RuntimeError("ALERT: prior terminal validation incomplete")
        for key in ("phi","loss","residual","A"):
            if not bits_equal(oldstate["profile"][key],terminalreplay["profile"][key]): raise RuntimeError("ALERT: prior terminal profile disagreement")
        n2_path=repo/"outputs/sho/generic_update_network_targeted_recovery/companion_6/n00005_i00004_n2.json"
        trace_path=repo/"outputs/sho/n2_budget_discrimination_three_branch_20261013/n00005_i00004_original_stabilized.jsonl"
        data=repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        if any(str(path) not in before for path in (n2_path,trace_path,data)): raise RuntimeError("ALERT: source/data not covered by prior provenance")
        n2=json.loads(n2_path.read_text())["result"]
        controls={}
        with trace_path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row["step"] in MILESTONES:
                    if str(row["step"]) in controls: raise RuntimeError("ALERT: duplicate saved control milestone")
                    controls[str(row["step"])]=row
                if row["step"]>=500: break
        if n2["evaluations"]!=1500 or not bits_equal(n2["theta"],controls["0"]["phi"]) or not bits_equal(n2["loss"],controls["0"]["loss"]):
            raise RuntimeError("ALERT: original N2/source step0 mismatch")
        adapter=GenericSHOUpdateNetworkAdapter()
        if adapter.layout!=LAYOUT or adapter.dimension!=22 or not (np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99)):
            raise RuntimeError("ALERT: architecture/layout/original box changed")
        with np.load(data) as corpus: trajectory=torch.from_numpy(corpus["prefix_6"].copy())
        lower,upper=torch.from_numpy(adapter.lower.copy()),torch.from_numpy(adapter.upper.copy())
        source_theta=torch.tensor(n2["theta"],dtype=torch.float64)
        with torch.no_grad(): source_loss=float(adapter.objective(source_theta,trajectory))
        source_objective_calls+=1
        if not bits_equal(source_loss,n2["loss"]): raise RuntimeError("ALERT: N2 native scalar replay failed")
        approval=None;approved_initial=None
        if args.mode=="optimize":
            preflight=args.preflight.resolve(strict=True);artifact_path=preflight/"artifact_manifest.json"
            if sha256(artifact_path)!=args.approved_preflight_sha256: raise RuntimeError("ALERT: approval manifest digest differs")
            approval_before[str(artifact_path)]=args.approved_preflight_sha256
            for name,digest in json.loads(artifact_path.read_text())["sha256"].items():
                path=preflight/name
                if path.parent!=preflight or sha256(path)!=digest: raise RuntimeError("ALERT: reviewed preflight artifact differs")
                approval_before[str(path)]=digest
            pm=json.loads((preflight/"manifest.json").read_text());pr=json.loads((preflight/"result.json").read_text())
            if pr["status"]!="PREFLIGHT_COMPLETE_REQUIRES_REVIEW" or not pr["integrity_unchanged"] or pm["input_sha256"]!=before or pm["script_sha256"]!=ORIGINAL_JOINT_DRIVER_SHA:
                raise RuntimeError("ALERT: preflight provenance/completion differs")
            approved_initial=json.loads((preflight/"original_center.json").read_text())
            approval={"preflight":str(preflight),"manifest_sha256":args.approved_preflight_sha256,
                "coordination_at_preflight":pm["coordination_sha256"],"coordination_at_this_phase":coord_before,
                "note":"Coordinator supplied reviewed manifest digest; intentional coordination edits between phases are recorded, not silently ignored."}
        # Compare the original base input set above before adding any repair provenance.
        original_base_inputs=dict(before)
        baseline_complete=False
        original_joint_driver=repo/"experiments/sho/recover_joint_model_variable_projection.py"
        if sha256(original_joint_driver)!=ORIGINAL_JOINT_DRIVER_SHA or Path(pm["script"]).resolve()!=original_joint_driver:
            raise RuntimeError("ALERT: original joint driver identity differs")
        before[str(original_joint_driver)]=ORIGINAL_JOINT_DRIVER_SHA
        repair=args.repair_test.resolve(strict=True)
        if repair!=(oldroot/"joint_cycle_refinement").resolve(): raise RuntimeError("ALERT: unexpected approved repair directory")
        repair_artifact_path=repair/"artifact_manifest.json"
        if sha256(repair_artifact_path)!=args.approved_repair_sha256: raise RuntimeError("ALERT: approved repair artifact digest differs")
        for directory in (repair,oldroot/"joint_cycle_reference"):
            artifact_path=directory/"artifact_manifest.json"
            for name,digest in json.loads(artifact_path.read_text())["sha256"].items():
                path=directory/name
                if path.parent!=directory or sha256(path)!=digest: raise RuntimeError("ALERT: repair/reference artifact differs: "+str(path))
                before[str(path)]=digest
            before[str(artifact_path)]=sha256(artifact_path)
            for name,digest in json.loads((directory/"manifest.json").read_text())["input_sha256"].items():
                path=Path(name)
                if sha256(path)!=digest: raise RuntimeError("ALERT: repair/reference input differs: "+str(path))
                if name in before and before[name]!=digest: raise RuntimeError("ALERT: conflicting repair provenance")
                before[name]=digest
            if not json.loads((directory/"integrity.json").read_text())["unchanged"]: raise RuntimeError("ALERT: repair/reference integrity was not confirmed")
        repair_result=json.loads((repair/"result.json").read_text())
        if not (repair_result["status"]=="RETURNED_FOR_REVIEW" and repair_result["exception"] is None and
                repair_result["finite"] and repair_result["raw_step_feasible"] and repair_result["realized_feasible"] and
                repair_result["solver_calls"]==1 and repair_result["network_evaluations"]==0 and repair_result["correction_passes_per_free_solve"]==1):
            raise RuntimeError("ALERT: approved cached repair did not return the declared feasible result")
        for path in (oldroot/"approved_repair_authorization.json",oldroot/"cached_repair_review.json"):
            before[str(path)]=sha256(path)
        solver_path=repair/"refined_solver_source.py"
        solver_bytes=solver_path.read_bytes()
        import hashlib
        if hashlib.sha256(solver_bytes).hexdigest()!=REFINED_SOLVER_SHA or before[str(solver_path)]!=REFINED_SOLVER_SHA:
            raise RuntimeError("ALERT: exact approved residual-correction source differs")
        baseline_complete=True
        refinements=[]
        solver_namespace={"torch":torch,"refinements":refinements}
        exec(compile(solver_bytes,str(solver_path),"exec"),solver_namespace)
        refined_joint_bvls=solver_namespace["refined_bvls"]
        approval.update(repair_test=str(repair),repair_artifact_manifest_sha256=args.approved_repair_sha256,
            cached_repair_source_sha256=REFINED_SOLVER_SHA,original_joint_driver_sha256=ORIGINAL_JOINT_DRIVER_SHA,
            repair_scope="Coordinator approved one cached residual correction test and this conditional fresh pilot. No cached solve or additional preflight is executed here.")
        save(out,"manifest.json",{"mode":args.mode,"input_sha256":before,"coordination_sha256":coord_before,"approval_input_sha256":approval_before,
            "original_base_input_sha256":original_base_inputs,"original_joint_driver":str(original_joint_driver),"original_joint_driver_sha256":ORIGINAL_JOINT_DRIVER_SHA,
            "refined_joint_solver_source":str(solver_path),"refined_joint_solver_sha256":REFINED_SOLVER_SHA,
            "script":str(Path(__file__).resolve()),"script_sha256":script_hash,"approval":approval,"copied_profile_source":oldmanifest["script"],
            "copied_profile_source_sha256":OLD_DRIVER_SHA,"layout":LAYOUT,"lower":lower,"upper":upper,
            "protocol":"Same profiled joint Gauss-Newton hidden trust-BOX method with one approved residual correction per free SVD solve in JOINT models only. Original head box and hidden original-box intersection h±Delta*d. Exact zero-width hidden intervals eliminated, no proximity tolerance. Exact relative-bound equality reconstructs absolute endpoints; realized MODEL displacement determines prediction. Actual candidate uses unchanged original zero-anchor head BVLS and strict native original MSE acceptance.",
            "model_solver":"Exact saved refined_solver_source.py: one all-mode residual correction per free-face SVD solve. Exact bounds, blocking, KKT comparisons and every positive singular mode preserved. Finite wrappers and full/reduced/final-free singular records remain. No ridge, rank truncation, new tolerance, repeated correction, or fallback. Head profiling retains original BVLS.",
            "trust_rule":"d=sqrt(ColemanLi_v(h,J_hidden.T@r)); initial Delta=norm(h/sqrt(v)), inherited zero fallback1. d frozen through rejected trials, rebuilt after acceptance. SciPy update_tr_radius with realized hidden infinity norm max|p_h/d|, boundary hit >.95Delta. Zero d requires no movement. Native reduction scaled M/2; prediction uses model head, never reprofiled head.",
            "preflight":"Previously completed original joint-model preflight is required by digest. Initial original phi/J/radius replay only; no new preflight model/candidate or cached repair solve.",
            "pilot_limits":{"accepted_steps":500,"profile_evaluations_including_initial_refit":3000,"start":"only original (5,4) N2 hidden endpoint"},
            "stops":"Strict feasible native loss<1e-20 alerts immediately; no rollout. Nonpositive prediction, unchanged/repeated hidden candidate, radius failure, nonfinite arithmetic or BVLS failure stop numerically/technically, not as convergence certificates.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})
        save(out,"source_verification.json",{"phi":source_theta,"loss":source_loss,"N2_bits":True,"saved_controls":controls})
        setup={"comparison_mode":"single-start finite-budget recovery pilot; saved-step references are not compute matched",
            "model_type":"generic 22-parameter SHO update network","encoder_type":"Linear(2,4) -> SiLU -> Linear(4,2)",
            "activation":"SiLU","hidden_dim":4,"optimizer":"Joint Gauss-Newton profiled hidden trust box; one all-mode residual correction per free SVD solve in joint BVLS only, unchanged native head profiling",
            "learning_rate":None,"batch_size":len(trajectory)-1,"number_of_epochs_completed":None,"max_epochs":None,
            "epoch_note":"No epoch-based training; counts are accepted hidden steps, profile evaluations, and joint model solves.",
            "accepted_step_cap":500,"profile_evaluation_cap":3000,"new_preflight_states":0,"cached_repair_solves":0,
            "joint_model_corrections_per_free_SVD":1,"head_profile_corrections":0,
            "original_joint_driver_sha256":ORIGINAL_JOINT_DRIVER_SHA,"refined_joint_solver_sha256":REFINED_SOLVER_SHA,
            "approved_repair_manifest_sha256":args.approved_repair_sha256,"approved_original_preflight_manifest_sha256":args.approved_preflight_sha256,
            "stopping_criterion":"feasible native MSE<1e-20; finite resource cap or declared numerical/technical stop, never a convergence certificate",
            "stopping_monitor":"unchanged native transition MSE; repeated/unchanged hidden proposal, nonpositive model prediction, finite arithmetic and feasible boxes",
            "stopping_patience":None,"stopping_min_delta":None,"strict_acceptance":"candidate native MSE < center native MSE",
            "stopped_because":None,"hidden_scaling":"sqrt(ColemanLi_v), gradient=J_hidden.T@r; fixed through rejected proposals",
            "initial_radius":"norm(hidden/scaling); zero fallback1","trust_norm":"max(abs(realized model hidden step/scaling)); zero scales require zero movement",
            "radius_update":"SciPy update_tr_radius, boundary_hit > .95*radius; no added radius rule",
            "model_prediction":"0.5*(r.T@r-model_residual.T@model_residual), realized reconstructed MODEL step; actual refitted head excluded",
            "regularization":None,"rank_truncation":None,"fixed_coordinate_rule":"eliminate only exactly equal hidden lower/upper trust endpoints",
            "dataset":str(data),"dataset_sha256":before[str(data)],"trajectory_key":"prefix_6","residual_count":2*(len(trajectory)-1),
            "evaluation_settings":"CPU torch float64, single thread, original bounds±1.99 and native objective; no rollout or fixture"}
        save(out,"setup.json",setup)

        class Handoff(Exception): pass
        def work():
            return {"profile_evaluations":profiles,"head_BVLS_calls":profiles,"joint_model_BVLS_calls":models,
                    "native_center_derivative_evaluations":centers,"accepted_hidden_steps":accepted,
                    "source_objective_calls":source_objective_calls,"adapter_calls":dict(adapter.calls),"initial_head_refit_is_separate":True}

        def profile(h,purpose):
            nonlocal profiles,last_profile
            h=torch.as_tensor(np.array(h,dtype=np.float64,copy=True))
            if not bool(torch.isfinite(h).all()) or not bool(((h>=lower[:12])&(h<=upper[:12])).all()): raise RuntimeError("TECHNICAL_BLOCKER: hidden candidate invalid")
            profiles+=1
            zero=torch.cat((h,torch.zeros(10,dtype=torch.float64)))
            zero_r,zero_J=residual_and_jacobian(adapter,trajectory,zero)
            A=zero_J[:,12:22];anchor_r=-trajectory[1:].reshape(-1)
            if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(anchor_r).all()): raise RuntimeError("TECHNICAL_BLOCKER: nonfinite head BVLS input")
            singular=torch.linalg.svdvals(A)
            if not bool(torch.isfinite(singular).all()) or bool((singular<=0).any()): raise RuntimeError("TECHNICAL_BLOCKER: invalid head singular value; no truncation")
            save(out,"pending_head_profile.json",{"status":"pending","profile_index":profiles,"purpose":purpose,"hidden":h,
                "A":A,"anchor_residual":anchor_r,"head_lower":lower[12:22],"head_upper":upper[12:22],"head_singular_values":singular})
            head,solver=_bounded_residual_least_squares(A,anchor_r,torch.zeros(10),lower[12:22],upper[12:22])
            phi=torch.cat((h,head))
            row={"profile_index":profiles,"purpose":purpose,"hidden":h,"phi":phi,"head_solver":solver,"A":A,
                 "head_singular_values":singular,"anchor_residual":anchor_r,"native_zero_minus_anchor_residual":zero_r-anchor_r}
            if not bool(torch.isfinite(phi).all()) or not bool(((phi>=lower)&(phi<=upper)).all()):
                save(out,"invalid_profile.json",row);raise RuntimeError("TECHNICAL_BLOCKER: infeasible/nonfinite full profile; no clipping")
            with torch.no_grad(): loss=float(adapter.objective(phi,trajectory));r=residual_vector(adapter,trajectory,phi).detach()
            if not math.isfinite(loss) or not bool(torch.isfinite(r).all()):
                save(out,"invalid_profile.json",{**row,"loss":loss,"residual":r});raise RuntimeError("TECHNICAL_BLOCKER: nonfinite native profile")
            if not bits_equal(loss,float((r*r).sum()/r.numel())): raise RuntimeError("ALERT: native/residual assembly bits disagree")
            model_r=anchor_r+A@head;kkt=A.T@model_r;native_kkt=A.T@r;active=torch.tensor(solver["active"])
            row.update(loss=loss,residual=r,linear_residual=model_r,native_minus_linear_residual=r-model_r,
                half_sum_model_head_gradient=kkt,half_sum_native_head_gradient=native_kkt,native_MSE_head_gradient=2/r.numel()*native_kkt,
                free_model_gradient=kkt[active==0],free_native_gradient=native_kkt[active==0],
                lower_active_native_gradient=native_kkt[active==-1],upper_active_native_gradient=native_kkt[active==1],
                active_bound_identity=bool(((active!=-1)|(head==lower[12:22])).all() and ((active!=1)|(head==upper[12:22])).all()),feasible=True,native_MSE_bits=True)
            if not row["active_bound_identity"]: raise RuntimeError("ALERT: head active status/bound mismatch")
            last_profile=row
            save(out,"pending_head_profile.json",{"status":"completed","profile_index":profiles,"purpose":purpose,"profile":row})
            with (out/"profiles.jsonl").open("a") as stream: append(stream,{key:value for key,value in row.items() if key not in ("A","anchor_residual","native_zero_minus_anchor_residual")})
            if loss<1e-20:
                save(out,"ALERT_handoff_candidate.json",row);print(f"ALERT FEASIBLE_PROFILE_HANDOFF {loss:.17g}; stop",flush=True)
                raise Handoff("Feasible native profile below handoff; no more model work or rollout")
            return row

        def center_derivatives(row):
            nonlocal centers
            centers+=1
            phi=row["phi"];r,J=residual_and_jacobian(adapter,trajectory,phi)
            if not bits_equal(r,row["residual"]) or not bits_equal(J[:,12:22],row["A"]): raise RuntimeError("ALERT: center residual/head J replay failed")
            flat=phi.detach().clone().requires_grad_(True);native_r=residual_vector(adapter,trajectory,flat)
            native_loss=(native_r*native_r).sum()/native_r.numel()
            native_g=torch.autograd.grad(native_loss,flat)[0].detach()
            if not bits_equal(float(native_loss.detach()),row["loss"]): raise RuntimeError("ALERT: center native derivative scalar mismatch")
            if not bool(torch.isfinite(J).all()) or not bool(torch.isfinite(native_g).all()): raise RuntimeError("TECHNICAL_BLOCKER: nonfinite center derivatives")
            g=J[:,:12].T@r
            v,dv=CL_scaling_vector(phi[:12].numpy(),g.numpy(),adapter.lower[:12],adapter.upper[:12])
            if not np.all(np.isfinite(v)) or np.any(v<0): raise RuntimeError("NUMERICAL_STOP: invalid Coleman-Li scaling")
            d=np.sqrt(v)
            return {"r":r,"J":J,"native_MSE_gradient":native_g,"residual_J_MSE_gradient":2/r.numel()*(J.T@r),
                "half_sum_hidden_gradient":g,"ColemanLi_v":v,"ColemanLi_dv":dv,"d":d,"full_J_singular_values":torch.linalg.svdvals(J)}

        def initial_radius(row,derivative):
            h=row["hidden"].numpy();d=derivative["d"]
            if np.any(d==0): raise RuntimeError("NUMERICAL_STOP: zero initial Coleman-Li scale makes declared norm(h/d) undefined")
            delta=float(np.linalg.norm(h/d))
            if delta==0: delta=1.0
            if not math.isfinite(delta) or delta<=0: raise RuntimeError("NUMERICAL_STOP: invalid initial radius")
            return delta

        def joint_model(row,derivative,Delta,label):
            nonlocal models,last_model
            phi,r,J=row["phi"],derivative["r"],derivative["J"]
            d=torch.from_numpy(derivative["d"].copy())
            box_lower,box_upper=lower.clone(),upper.clone()
            box_lower[:12]=torch.maximum(lower[:12],phi[:12]-Delta*d)
            box_upper[:12]=torch.minimum(upper[:12],phi[:12]+Delta*d)
            fixed=torch.where(box_lower[:12]==box_upper[:12])[0]
            fixed_mask=torch.zeros(22,dtype=torch.bool);fixed_mask[fixed]=True;keep=torch.where(~fixed_mask)[0]
            metadata={"model_call_status":"pending","model_index":models+1,"label":label,"center_phi":phi,"center_loss":row["loss"],"r":r,"J":J,
                "radius":Delta,"d":d,"box_lower":box_lower,"box_upper":box_upper,"fixed_hidden_indices":fixed,
                "kept_indices":keep,"full_J_singular_values":derivative["full_J_singular_values"],
                "reduced_matrix_singular_values":torch.linalg.svdvals(J[:,keep]),
                "zero_step_feasible":bool(((phi>=box_lower)&(phi<=box_upper)).all()),
                "zero_model_half_sum_squares":float(.5*(r@r)),"native_center_half_sum_squares":row["loss"]*r.numel()/2}
            if not metadata["zero_step_feasible"] or not bool((phi[fixed]==box_lower[fixed]).all()):
                save(out,"invalid_model_input.json",metadata);raise RuntimeError("ALERT: zero model baseline/exact fixed interval infeasible")
            if not all(bool(torch.isfinite(value).all()) for value in (J,r,phi,box_lower,box_upper,d)):
                save(out,"invalid_model_input.json",metadata);raise RuntimeError("NUMERICAL_STOP: nonfinite model inputs")
            models+=1
            refinements.clear()
            metadata["joint_residual_corrections"]=[]
            metadata["joint_solver_source_sha256"]=REFINED_SOLVER_SHA
            save(out,"pending_model.json",metadata)
            try:
                step_reduced,solver=refined_joint_bvls(J[:,keep],r,phi[keep],box_lower[keep],box_upper[keep])
            finally:
                # Snapshot this call before the shared list is cleared for another model.
                metadata["joint_residual_corrections"]=list(refinements)
                save(out,"pending_model.json",metadata)
            raw_step=torch.zeros_like(phi);raw_step[keep]=step_reduced
            relative_lower,relative_upper=box_lower-phi,box_upper-phi
            model_phi=phi+raw_step
            lower_hits=raw_step==relative_lower;upper_hits=raw_step==relative_upper
            model_phi=torch.where(lower_hits,box_lower,model_phi)
            model_phi=torch.where(upper_hits,box_upper,model_phi)
            p=model_phi-phi
            if not bool(torch.isfinite(raw_step).all()) or not bool(torch.isfinite(model_phi).all()) or not bool(((model_phi>=box_lower)&(model_phi<=box_upper)).all()) or not bool(((model_phi>=lower)&(model_phi<=upper)).all()):
                save(out,"invalid_model_output.json",{**metadata,"raw_step":raw_step,"model_phi":model_phi,"solver":solver})
                raise RuntimeError("NUMERICAL_STOP: nonfinite/infeasible model output; no clipping or fallback")
            if bool((p[:12][d==0]!=0).any()) or bool((p[fixed]!=0).any()): raise RuntimeError("ALERT: exact fixed/zero-scale hidden coordinate moved")
            scaled=torch.zeros(12,dtype=torch.float64);positive=d>0;scaled[positive]=p[:12][positive].abs()/d[positive]
            step_norm=float(scaled.max());model_r=r+J@p;prediction=float(.5*((r@r)-(model_r@model_r)))
            active_reduced=torch.tensor(solver["active"]);free=keep[active_reduced==0]
            model_gradient=J.T@model_r
            status=torch.full((22,),2,dtype=torch.int64);status[keep]=active_reduced
            metadata.update(raw_solver_step=raw_step,solver=solver,model_phi=model_phi,realized_model_step=p,
                exact_lower_endpoint_reconstruction=lower_hits,exact_upper_endpoint_reconstruction=upper_hits,
                reconstructed_minus_raw_step=p-raw_step,model_residual=model_r,prediction_half_sum_squares=prediction,
                prediction_MSE=2*prediction/r.numel(),realized_hidden_infinity_norm=step_norm,
                scaled_hidden_displacement=scaled,boundary_hit=step_norm>.95*Delta,
                full_model_status=status,status_legend={"-1":"lower","0":"free","1":"upper","2":"exact fixed hidden eliminated"},
                final_free_indices=free,final_free_singular_values=torch.linalg.svdvals(J[:,free]) if len(free) else torch.empty(0),
                half_sum_model_gradient=model_gradient,free_model_gradient=model_gradient[free],
                lower_model_gradient=model_gradient[status==-1],upper_model_gradient=model_gradient[status==1],
                model_finite=bool(torch.isfinite(model_r).all()) and math.isfinite(prediction) and math.isfinite(step_norm),
                feasibility=True,actual_profile_head_used_in_prediction=False)
            last_model=metadata
            metadata["model_call_status"]="completed"
            save(out,"pending_model.json",metadata)
            with (out/"models.jsonl").open("a") as stream: append(stream,metadata)
            if not metadata["model_finite"]: raise RuntimeError("NUMERICAL_STOP: nonfinite realized model prediction")
            if prediction<=0: raise RuntimeError("NUMERICAL_STOP: nonpositive realized model prediction; not convergence")
            return metadata

        def verify_profile(row,reference,label):
            checks={key:bits_equal(row[key],reference[key]) for key in ("phi","loss","residual","A","head_singular_values","native_MSE_head_gradient")}
            checks["head_face"]=row["head_solver"]["active"]==reference["head_solver"]["active"]
            save(out,label+"_replay.json",checks)
            if not all(checks.values()): raise RuntimeError("ALERT: saved "+label+" profile exact replay failed")

        initial=profile(source_theta[:12].numpy(),"original_N2_zero_anchor_profile")
        verify_profile(initial,oldinitial,"original")
        if initial["loss"]>source_loss: raise RuntimeError("TECHNICAL_BLOCKER: initial refit worse than source")
        initial_derivative=center_derivatives(initial)
        initial_delta=initial_radius(initial,initial_derivative)
        save(out,"original_center.json",{"profile":initial,"derivatives":initial_derivative,"initial_radius":initial_delta})
        if approved_initial is not None:
            verify_profile(initial,approved_initial["profile"],"approved_original")
            if not bits_equal(initial_derivative["J"],approved_initial["derivatives"]["J"]) or not bits_equal(initial_delta,approved_initial["initial_radius"]): raise RuntimeError("ALERT: approved initial model center/radius replay failed")
        current,deriv,Delta=initial,initial_derivative,initial_delta
        hidden_path=full_path=0.0;snapshots={};trial_count=0
        def persist():
            save(out,"diagnostic_state.json",{"profile":current,"derivatives":deriv,"radius":Delta,"accepted_steps":accepted,
                "work":work(),"hidden_path":hidden_path,"full_parameter_path":full_path,"stop_reason":stop,"approval":approval,"automatic_resume":False})
        checkpoint=persist
        def record_accepted(update=None):
            row={"step":accepted,"profile":current,"derivatives":deriv,"radius":Delta,"work":work(),"update":update,
                 "hidden_path":hidden_path,"full_parameter_path":full_path}
            if accepted in MILESTONES:
                row["saved_stabilized_control"]=controls.get(str(accepted));row["comparison_note"]="Accepted-step context only, not compute matched."
                snapshots[str(accepted)]={"loss":current["loss"],"work":work()}
                print(f"JOINT_PROFILE accepted={accepted} profiles={profiles} models={models} loss={current['loss']:.17g}",flush=True)
            with (out/"accepted.jsonl").open("a") as stream: append(stream,row)
            persist()
        record_accepted()
        while stop is None:
            if accepted>=500: stop="RESOURCE_LIMIT_500_ACCEPTED_HIDDEN_STEPS";break
            if profiles>=3000: stop="RESOURCE_LIMIT_3000_PROFILE_EVALUATIONS";break
            seen=set()
            # The accepted-center r/J and Coleman-Li d remain frozen in this rejection loop.
            while stop is None:
                if profiles>=3000: stop="RESOURCE_LIMIT_3000_PROFILE_EVALUATIONS";break
                model=joint_model(current,deriv,Delta,f"accepted{accepted}_model{models+1}")
                hidden=model["model_phi"][:12].numpy().copy()
                if np.array_equal(hidden,current["hidden"].numpy()) or hidden.tobytes() in seen:
                    stop="FLOAT64_UNCHANGED_OR_REPEATED_HIDDEN_MODEL_CANDIDATE";break
                seen.add(hidden.tobytes());trial_count+=1
                trial=profile(hidden,f"outer_trial{trial_count}_accepted_before{accepted}")
                actual=(current["loss"]-trial["loss"])*current["residual"].numel()/2
                next_delta,ratio=update_tr_radius(Delta,actual,model["prediction_half_sum_squares"],model["realized_hidden_infinity_norm"],model["boundary_hit"])
                improves=trial["loss"]<current["loss"];radius_valid=math.isfinite(next_delta) and next_delta>0
                update={"trial":trial_count,"model_index":model["model_index"],"profile_index":trial["profile_index"],
                    "accepted_before":accepted,"accepted":improves and radius_valid,"native_strict_improvement":improves,
                    "old_loss":current["loss"],"new_loss":trial["loss"],"prediction_MSE":model["prediction_MSE"],
                    "actual_MSE_decrease":current["loss"]-trial["loss"],"ratio":ratio,"radius_before":Delta,"radius_after":next_delta,
                    "radius_valid":radius_valid,"actual_phi":trial["phi"],"model_phi":model["model_phi"],
                    "actual_minus_model_head":trial["phi"][12:22]-model["model_phi"][12:22]}
                with (out/"trials.jsonl").open("a") as stream: append(stream,update)
                if not radius_valid: stop="NUMERICAL_TRUST_RADIUS_EXHAUSTED";break
                Delta=next_delta
                if improves:
                    hidden_path+=float((trial["hidden"]-current["hidden"]).norm());full_path+=float((trial["phi"]-current["phi"]).norm())
                    current=trial;accepted+=1;deriv=center_derivatives(current);record_accepted(update);break
                persist()
        persist()
        answer={"status":"PILOT_COMPLETE","stop_reason":stop,"handoff":False,"phi":current["phi"],"loss":current["loss"],
            "source_loss":source_loss,"initial_profile_loss":initial["loss"],"initial_refit_displacement":float((initial["phi"]-source_theta).norm()),
            "work":work(),"trial_candidates":trial_count,"snapshots":snapshots,"hidden_path":hidden_path,"full_parameter_path_excluding_refit":full_path,
            "limits":"Single-start finite NEW joint Gauss-Newton profile trust-box diagnostic; numerical stop is not convergence, stationarity, basin disconnection or recovery impossibility. No restart or followup."}
        final_checks={name:sha256(Path(name))==digest for name,digest in {**before,**coord_before,**approval_before}.items()}
        final_checks[str(Path(__file__).resolve())]=sha256(Path(__file__))==script_hash
        unchanged=all(final_checks.values())
        save(out,"integrity.json",{"unchanged":unchanged,"checks":final_checks,"full_protected_guard":"coordinator-owned"})
        if not unchanged: raise RuntimeError("ALERT: relied-on artifacts/sources/coordination/script changed")
        answer.update(integrity_unchanged=True,elapsed_seconds=time.monotonic()-started)
        save(out,"result.json",answer)
        setup.update(stopped_because=answer.get("stop_reason",answer["status"]),accepted_steps_completed=accepted,
                     profile_evaluations_completed=profiles,joint_model_solves_completed=models)
        save(out,"setup.json",setup)
        save(out,"artifact_manifest.json",{"sha256":{path.name:sha256(path) for path in sorted(out.iterdir()) if path.is_file()}})
        print(f"COMPLETE {args.mode}: {answer['status']} profiles={profiles} models={models} accepted={accepted}",flush=True)
    except BaseException as exc:
        status="ALERT_HANDOFF_CANDIDATE" if type(exc).__name__=="Handoff" else ("TECHNICAL_SOLVER_CYCLE" if "cycle" in str(exc).lower() else ("ALERT_INTEGRITY_OR_REPLAY" if str(exc).startswith("ALERT") else ("NUMERICAL_STOP" if str(exc).startswith("NUMERICAL_STOP") else "TECHNICAL_STOP")))
        if checkpoint is not None:
            try: checkpoint()
            except BaseException: pass
        save(out,"failure.json",{"status":status,"error":str(exc),"type":type(exc).__name__,"traceback":traceback.format_exc(),
            "last_completed_profile":last_profile,"last_completed_model":last_model,"profile_calls":profiles,"model_calls":models,
            "accepted_steps":accepted,"center_derivative_calls":centers,"source_objective_calls":source_objective_calls,
            "no_fallback_or_automatic_followup":True,"handoff_scope":"Candidate only; no rollout or recovery certification."})
        checks={};errors={}
        for name,digest in {**before,**coord_before,**approval_before,str(Path(__file__).resolve()):script_hash}.items():
            try: checks[name]=sha256(Path(name))==digest
            except BaseException as check_exc: errors[name]=str(check_exc)
        save(out,"integrity.json",{"unchanged":baseline_complete and not errors and all(checks.values()),"baseline_coverage_complete":baseline_complete,
            "checks":checks,"verification_errors":errors,"on_failure_or_handoff":True,"full_protected_guard":"coordinator-owned"})
        if "setup" in locals():
            setup.update(stopped_because=status,accepted_steps_completed=accepted,profile_evaluations_completed=profiles,joint_model_solves_completed=models)
            save(out,"setup.json",setup)
        save(out,"artifact_manifest.json",{"sha256":{path.name:sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!="artifact_manifest.json"}})
        print(f"{status}: {exc}",flush=True);raise


if __name__=="__main__": main()
