"""Approved saved extended (5,4) validation with exact head and exact joint solves."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode=True
sys.set_int_max_str_digits(0)
from screen_existing_full_hessian_endpoints import sha256,save,append

LAYOUT=[dict(name="0.weight",shape=[4,2],start=0,stop=8),dict(name="0.bias",shape=[4],start=8,stop=12),
        dict(name="2.weight",shape=[2,4],start=12,stop=20),dict(name="2.bias",shape=[2],start=20,stop=22)]
OLD_DRIVER_SHA="37f0ecefa9e81ba5e9640add62f394aa3f64b1c52ede01cbfd98e7d755d30750"
ORIGINAL_JOINT_DRIVER_SHA="9e31920e8d617f9e3ebdb4abde9e081302f7adc9a8b7767d13a0f02e035f8380"
REFINED_DRIVER_SHA="16927ab9d6bf20103bd718352900ba44182bec88dd1d0c05710836ad9f8ba5c8"
EXACT_SOLVER_SHA="2f96012afadf59b1de867bae5757fdb5ef9faf99b36797e8f70c7f54433b946d"
EXACT_VALIDATION_SHA="4b7dffe338b1ea67f4f922fb7e041bd32a02aa889985be8cfaaaa2fc4d7692b5"
PROPOSAL_SHA="331af5b1b16f2888730f53c5f984e15534a661b26706410f80bb0122c5b931cd"
MILESTONES=(0,1,10,50,100,250,500)


def pack_exact(value):
    if isinstance(value,Fraction): return str(value.numerator)+"/"+str(value.denominator)
    if isinstance(value,dict): return {key:pack_exact(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [pack_exact(item) for item in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    args=parser.parse_args();args.mode="optimize"
    repo=args.repo.resolve(strict=True)
    oldroot=repo/"outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy"
    out=oldroot/"exact_both_validation_extended54"
    if out.exists(): parser.error("Output must be fresh; no automatic restart or continuation")
    out.mkdir();os.chdir(repo);sys.path.insert(0,str(repo))
    before={};coord_before={};approval_before={};baseline_complete=False
    profiles=models=centers=accepted=source_objective_calls=0
    head_calls=head_completed=0
    last_profile=last_model=None;checkpoint=None;stop=None
    script_hash=sha256(Path(__file__));started=time.monotonic()
    try:
        import numpy as np
        import scipy
        import torch
        from scipy.optimize._lsq.common import CL_scaling_vector,update_tr_radius
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import residual_vector,residual_and_jacobian
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        guard=oldroot.parent/"exact_both_extended54_integrity_before.json"
        baseline=json.loads(guard.read_text())["files"]
        before[str(guard)]=sha256(guard)
        for name,digest in baseline.items():
            if sha256(Path(name))!=digest: raise RuntimeError("ALERT: baseline changed before run: "+name)
        # The external guard covers all old artifacts; record every relied-on source plus data explicitly.
        for name,digest in baseline.items():
            path=Path(name)
            if path.suffix==".py" or path.parent==repo/"coordination": before[name]=digest
        prior=oldroot/"joint_exact_pilot"
        for name,digest in json.loads((prior/"artifact_manifest.json").read_text())["sha256"].items():
            path=prior/name
            if path.parent!=prior or sha256(path)!=digest: raise RuntimeError("ALERT: successful pilot artifact changed")
            before[str(path)]=digest
        prior_manifest=json.loads((prior/"manifest.json").read_text())
        for key in ("input_sha256","approval_input_sha256"):
            for name,digest in prior_manifest[key].items():
                if sha256(Path(name))!=digest: raise RuntimeError("ALERT: prior relied-on source/library/input changed: "+name)
                before[name]=digest
        if not json.loads((oldroot/"exact_handoff_rollout/result.json").read_text())["existing_contract_satisfied"]:
            raise RuntimeError("ALERT: preceding strategy recovery not validated")
        auth=oldroot/"exact_both_extended54_authorization.json"
        authorization=json.loads(auth.read_text());before[str(auth)]=sha256(auth)
        if authorization["direct_user_answer"]!="approve" or authorization["branch"]!=[5,4] or authorization["accepted_step_cap"]!=500 or authorization["profile_cap"]!=3000:
            raise RuntimeError("ALERT: validation protocol identity differs")
        protocol=oldroot/"exact_both_extended54_validation_proposal.md"
        if sha256(protocol)!=authorization["protocol_sha256"]: raise RuntimeError("ALERT: protocol changed")
        before[str(protocol)]=sha256(protocol)
        prior_driver=repo/"experiments/sho/recover_joint_model_variable_projection_exact.py"
        if sha256(prior_driver)!="86a80385672d293384d777f8df1efb53b8fe9b58b002f809bea978ef97f2d0f5": raise RuntimeError("ALERT: reviewed prior driver changed")
        solver_path=repo/"experiments/sho/exact_rational_bvls.py"
        solver_bytes=solver_path.read_bytes()
        if sha256(solver_path)!=EXACT_SOLVER_SHA: raise RuntimeError("ALERT: exact solver changed")
        solver_namespace={};exec(compile(solver_bytes,str(solver_path),"exec"),solver_namespace)
        exact_joint_bvls=solver_namespace["exact_bvls"]
        exact_head_bvls=exact_joint_bvls
        cached=oldroot/"head382_cached_diagnosis"
        artifact=cached/"artifact_manifest.json"
        if sha256(artifact)!=authorization["cached_head_artifact_sha256"]: raise RuntimeError("ALERT: cached head validation digest changed")
        before[str(artifact)]=sha256(artifact)
        for name,digest in json.loads(artifact.read_text())["sha256"].items():
            path=cached/name
            if path.parent!=cached or sha256(path)!=digest: raise RuntimeError("ALERT: cached head artifact changed")
            before[str(path)]=digest
        cr=json.loads((cached/"result.json").read_text())
        if not json.loads((cached/"integrity.json").read_text())["unchanged"] or cr["status"]!="HEAD_CYCLE_REPLAYED_AND_EXACT_REFERENCE_CERTIFIED" or cr["exact_solver_calls"]!=1 or cr["network_evaluations"]!=0 or not all(cr[k] for k in ("exact_feasible","exact_KKT","full_rank_exact","float_converted_head_feasible")):
            raise RuntimeError("ALERT: cached exact-head validation incomplete")
        previous43=repo/"experiments/sho/recover_exact_validation_original43.py"
        if sha256(previous43)!="30f2abd3cc3ee5939623df9a106cec2b36767a1404d074989b11cdee112fedf5": raise RuntimeError("ALERT: previous43 driver changed")
        n2_path=repo/"outputs/sho/n2_budget_discrimination_three_branch_20261013/n00005_i00004_n2_extended.json"
        trace_path=repo/"outputs/sho/n2_budget_discrimination_three_branch_20261013/n00005_i00004_extended_stabilized.jsonl"
        data=repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        for path in (n2_path,trace_path,data):
            if baseline.get(str(path))!=sha256(path): raise RuntimeError("ALERT: source/data not covered by baseline")
            before[str(path)]=sha256(path)
        if before[str(data)]!="d699b6b7e6de4a0d39ca25db48c417d5109b248862e32b33a57528c0ceffc774": raise RuntimeError("ALERT: data changed")
        n2=json.loads(n2_path.read_text())["result"];controls={}
        with trace_path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row["step"] in MILESTONES:
                    if str(row["step"]) in controls: raise RuntimeError("ALERT: duplicate control milestone")
                    controls[str(row["step"])]=row
                if row["step"]>=500:break
        if n2["evaluations"]!=3000 or not bits_equal(n2["theta"],controls["0"]["phi"]) or not bits_equal(n2["loss"],controls["0"]["loss"]): raise RuntimeError("ALERT: extended54 source/control mismatch")
        parent_driver=repo/"experiments/sho/recover_exact_both_original43.py"
        if sha256(parent_driver)!="416b2bac9926276fdbdd18e768c36b3c857929fffcee55590b753e8cc6f294b5": raise RuntimeError("ALERT: exact-both parent driver changed")
        validated_driver=repo/"experiments/sho/recover_exact_both_original54.py"
        if sha256(validated_driver)!="b7e1b4f3980a70f7b7f38e9b2f88327b42b92462742763c79b7d7af4bf2e3170": raise RuntimeError("ALERT: validated exact-both54 driver changed")
        if sha256(n2_path)!="1f8ddaa629a89f5a825f0e4ed083521cbaee8529daf32d6f460ba9e594ed0d5b" or not bits_equal(n2["loss"],0.00017808633909029335): raise RuntimeError("ALERT: approved extended54 source differs")
        if not json.loads((oldroot/"exact_both54_rollout/result.json").read_text())["existing_contract_satisfied"]: raise RuntimeError("ALERT: exact-both54 version not validated")
        if authorization.get("N2_evaluations")!=3000: raise RuntimeError("ALERT: extended source approval differs")
        adapter=GenericSHOUpdateNetworkAdapter()
        if adapter.layout!=LAYOUT or adapter.dimension!=22 or not (np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99)): raise RuntimeError("ALERT: architecture/box changed")
        with np.load(data) as corpus:trajectory=torch.from_numpy(corpus["prefix_6"].copy())
        lower,upper=torch.from_numpy(adapter.lower.copy()),torch.from_numpy(adapter.upper.copy())
        source_theta=torch.tensor(n2["theta"],dtype=torch.float64)
        with torch.no_grad():source_loss=float(adapter.objective(source_theta,trajectory))
        source_objective_calls+=1
        if not bits_equal(source_loss,n2["loss"]): raise RuntimeError("ALERT: saved N2 scalar replay failed")
        coord_before={str(path):sha256(path) for path in sorted((repo/"coordination").glob("*.md"))}
        approval=authorization;baseline_complete=True
        save(out,"manifest.json",{"mode":args.mode,"input_sha256":before,"coordination_sha256":coord_before,"approval_input_sha256":approval_before,
            "script":str(Path(__file__).resolve()),"script_sha256":script_hash,"approval":approval,
            "source_N2":str(n2_path),"source_branch":[5,4],"trajectory_key":"prefix_6",
            "layout":LAYOUT,"lower":lower,"upper":upper,"exact_joint_solver_sha256":EXACT_SOLVER_SHA,
            "arithmetic":"Exact rational head profiling added; existing joint model, derivatives, radius and outer acceptance loop unchanged. Rational certificates apply before float conversion.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})
        save(out,"source_verification.json",{"phi":source_theta,"loss":source_loss,"N2_bits":True,"saved_controls":controls})
        setup={"model_type":"original22parameter2->4->2SiLU","case":"companion_6","N2_branch":[5,4],
            "accepted_step_cap":500,"profile_evaluation_cap":3000,"joint_inner_iteration_cap":200,
            "optimizer":"Exact head and joint BVLS variable projection; unchanged outer rules",
            "head_inner_iteration_cap":200,
            "stopping_criterion":"native MSE<1e-20, declared numerical/technical stop or resource cap; no automatic continuation",
            "dataset":str(data),"dataset_sha256":before[str(data)],"trajectory_key":"prefix_6",
            "stopped_because":None,"limits":"Fixed exact-head+joint version validation on saved extended (5,4). No cross-trajectory reliability or compute-matched claim.",
            "evaluation_settings":"CPU float64 single thread; original box±1.99; no teacher/fixture initializer; rollout not part optimizer"}
        save(out,"setup.json",setup)
        class Handoff(Exception): pass
        def work():
            return {"profile_evaluations":profiles,"head_BVLS_calls":head_calls,"exact_head_calls":head_calls,"exact_head_completed":head_completed,"joint_model_BVLS_calls":models,
                    "native_center_derivative_evaluations":centers,"accepted_hidden_steps":accepted,
                    "source_objective_calls":source_objective_calls,"adapter_calls":dict(adapter.calls),"initial_head_refit_is_separate":True}

        def profile(h,purpose):
            nonlocal profiles,last_profile,head_calls,head_completed
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
            # Same cached zero-anchor head problem; exact arithmetic only inside BVLS.
            head_calls+=1
            exact_head=exact_head_bvls(
                [[Fraction(v) for v in row] for row in A.tolist()],
                [Fraction(v) for v in anchor_r.tolist()],
                [Fraction(v) for v in lower[12:22].tolist()],
                [Fraction(v) for v in upper[12:22].tolist()],max_iterations=200)
            head_certificate=f"exact_head_{profiles:05d}.json"
            save(out,head_certificate,{"profile_index":profiles,"head_call":head_calls,
                "solver_source_sha256":EXACT_SOLVER_SHA,"certificate":pack_exact(exact_head)})
            if not all(exact_head[k] is True for k in ("exact_feasible","exact_KKT","full_rank_exact")):
                raise RuntimeError("TECHNICAL_STOP: missing exact head certificate; no float conversion")
            head_completed+=1
            head=torch.tensor([float(v) for v in exact_head["direction"]],dtype=torch.float64)
            solver={k:exact_head[k] for k in ("active","iterations","exact_feasible","exact_KKT","full_rank_exact","face_factorizations")}
            solver.update(certificate_file=head_certificate,certificate_sha256=sha256(out/head_certificate),
                rounding_scope="Certificate before nearest-float64 conversion only; no rounded KKT claim.")
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
            # Subtraction is binary64 first, exactly as in approved cached validation.
            relative_lower,relative_upper=box_lower-phi,box_upper-phi
            metadata.update(joint_solver_source_sha256=EXACT_SOLVER_SHA,rounded_relative_lower=relative_lower,
                rounded_relative_upper=relative_upper,inner_iteration_cap=200,
                exact_certificate_scope="Rational model optimum before conversion only; no post-rounding KKT guarantee.")
            save(out,"pending_model.json",metadata)
            try:
                exact_solution=exact_joint_bvls(
                    [[Fraction(value) for value in values] for values in J[:,keep].tolist()],
                    [Fraction(value) for value in r.tolist()],
                    [Fraction(value) for value in relative_lower[keep].tolist()],
                    [Fraction(value) for value in relative_upper[keep].tolist()],max_iterations=200)
                certificate_name=f"exact_model_{models:05d}.json"
                save(out,certificate_name,{"model_index":models,"kept_indices":keep,"fixed_hidden_indices":fixed,
                    "solver_source_sha256":EXACT_SOLVER_SHA,"certificate":pack_exact(exact_solution)})
                metadata.update(exact_certificate=certificate_name,exact_certificate_sha256=sha256(out/certificate_name))
                save(out,"pending_model.json",metadata)
                if not all(exact_solution[key] is True for key in ("exact_feasible","exact_KKT","full_rank_exact")):
                    raise RuntimeError("TECHNICAL_STOP: missing exact rational certificate; no float conversion")
                step_reduced=torch.tensor([float(value) for value in exact_solution["direction"]],dtype=torch.float64)
                solver={key:exact_solution[key] for key in ("active","iterations","exact_feasible","exact_KKT","full_rank_exact","face_factorizations")}
                solver["certificate_file"]=certificate_name
                if not bool(torch.isfinite(step_reduced).all()) or not bool(((step_reduced>=relative_lower[keep])&(step_reduced<=relative_upper[keep])).all()):
                    metadata["invalid_converted_reduced_step"]=step_reduced
                    raise RuntimeError("NUMERICAL_STOP: nonfinite/infeasible raw exact-solver conversion; no fallback")
            except BaseException as solver_exc:
                metadata.update(model_call_status="failed",solver_error=str(solver_exc),solver_error_type=type(solver_exc).__name__)
                save(out,"pending_model.json",metadata)
                raise
            raw_step=torch.zeros_like(phi);raw_step[keep]=step_reduced
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

        initial=profile(source_theta[:12].numpy(),"saved_extended_N2_zero_anchor_profile")
        if initial["loss"]>source_loss: raise RuntimeError("TECHNICAL_BLOCKER: initial refit worse than source")
        initial_derivative=center_derivatives(initial)
        initial_delta=initial_radius(initial,initial_derivative)
        save(out,"original_center.json",{"profile":initial,"derivatives":initial_derivative,"initial_radius":initial_delta})
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
                     profile_evaluations_completed=profiles,joint_model_solves_completed=models,exact_head_calls=head_calls,exact_head_completed=head_completed)
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
            "exact_head_calls":head_calls,"exact_head_completed":head_completed,
            "no_fallback_or_automatic_followup":True,"handoff_scope":"Candidate only; no rollout or recovery certification."})
        checks={};errors={}
        for name,digest in {**before,**coord_before,**approval_before,str(Path(__file__).resolve()):script_hash}.items():
            try: checks[name]=sha256(Path(name))==digest
            except BaseException as check_exc: errors[name]=str(check_exc)
        save(out,"integrity.json",{"unchanged":baseline_complete and not errors and all(checks.values()),"baseline_coverage_complete":baseline_complete,
            "checks":checks,"verification_errors":errors,"on_failure_or_handoff":True,"full_protected_guard":"coordinator-owned"})
        if "setup" in locals():
            setup.update(stopped_because=status,accepted_steps_completed=accepted,profile_evaluations_completed=profiles,joint_model_solves_completed=models,exact_head_calls=head_calls,exact_head_completed=head_completed)
            save(out,"setup.json",setup)
        save(out,"artifact_manifest.json",{"sha256":{path.name:sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!="artifact_manifest.json"}})
        print(f"{status}: {exc}",flush=True);raise


if __name__=="__main__": main()
