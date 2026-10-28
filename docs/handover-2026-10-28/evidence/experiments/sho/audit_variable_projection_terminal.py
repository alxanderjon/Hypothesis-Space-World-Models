#!/usr/bin/env python3
"""Read-only terminal mathematics audit: 48 declared profile probes, no recovery.

Copied audited zero-anchor profile and fixed-face derivative arithmetic; never
imports, rewrites, or resumes the original optimization driver.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode=True
from screen_existing_full_hessian_endpoints import sha256,save,append

LAYOUT=[dict(name="0.weight",shape=[4,2],start=0,stop=8),dict(name="0.bias",shape=[4],start=8,stop=12),
        dict(name="2.weight",shape=[2,4],start=12,stop=20),dict(name="2.bias",shape=[2],start=20,stop=22)]
ORIGINAL_DRIVER_SHA="37f0ecefa9e81ba5e9640add62f394aa3f64b1c52ede01cbfd98e7d755d30750"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    repo,out=args.repo.resolve(strict=True),args.out.resolve()
    if out.exists(): parser.error("--out must be fresh; no overwrite, resume, or continuation")
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(repo);sys.path.insert(0,str(repo))
    before={};baseline_complete=False;profile_calls=0;derivative_calls=0;source_objective_calls=0;last_profile=None
    script_hash=sha256(Path(__file__))
    try:
        import numpy as np
        import scipy
        import torch
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import residual_vector,residual_and_jacobian,_bounded_residual_least_squares
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        pilot=repo/"outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy/pilot"
        pilot_hash_path=pilot/"artifact_manifest.json"
        pilot_manifest_path=pilot/"manifest.json"
        manifest=json.loads(pilot_manifest_path.read_text())
        expected=json.loads(pilot_hash_path.read_text())["sha256"]
        # Entire completed pilot and every recorded input/source are verified.
        for name,digest in expected.items():
            p=pilot/name
            if p.parent!=pilot or sha256(p)!=digest: raise RuntimeError("ALERT: pilot artifact mismatch: "+name)
            before[str(p)]=digest
        before[str(pilot_hash_path)]=sha256(pilot_hash_path)
        for key in ("input_sha256","approval_input_sha256","direct_source_sha256","library_source_sha256"):
            for name,digest in manifest[key].items():
                p=repo/name if key=="direct_source_sha256" else Path(name)
                if sha256(p)!=digest: raise RuntimeError("ALERT: original pilot input/source mismatch: "+str(p))
                before[str(p)]=digest
        original_driver=Path(manifest["script"])
        if manifest["script_sha256"]!=ORIGINAL_DRIVER_SHA or sha256(original_driver)!=ORIGINAL_DRIVER_SHA:
            raise RuntimeError("ALERT: audited original driver changed")
        before[str(original_driver)]=ORIGINAL_DRIVER_SHA
        helper=Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        before[str(helper)]=sha256(helper)
        baseline_complete=True
        diagnostic=json.loads((pilot/"diagnostic_state.json").read_text())
        result=json.loads((pilot/"result.json").read_text())
        saved=diagnostic["profile"];saved_derivative=diagnostic["derivatives"]
        last_accepted=None
        with (pilot/"accepted.jsonl").open() as stream:
            for line in stream: last_accepted=json.loads(line)
        if not (result["status"]=="PILOT_COMPLETE" and result["integrity_unchanged"] and
                result["stop_reason"]=="FLOAT64_UNCHANGED_OR_REPEATED_HIDDEN_TRIAL" and
                diagnostic["stop_reason"]==result["stop_reason"] and diagnostic["accepted_steps"]==156 and
                last_accepted["step"]==156 and result["work"]["accepted_hidden_steps"]==156):
            raise RuntimeError("ALERT: declared completed terminal source identity differs")
        for key in ("phi","loss","residual","A"):
            if not bits_equal(saved[key],last_accepted["profile"][key]): raise RuntimeError("ALERT: last accepted/diagnostic source mismatch: "+key)
        if not bits_equal(saved["phi"],result["phi"]) or not bits_equal(saved["loss"],result["loss"]):
            raise RuntimeError("ALERT: terminal result/source mismatch")
        if saved["head_solver"]["active"]!=last_accepted["profile"]["head_solver"]["active"] or not bits_equal(saved_derivative["reduced_J"],last_accepted["derivatives"]["reduced_J"]):
            raise RuntimeError("ALERT: saved terminal face/J mismatch")
        adapter=GenericSHOUpdateNetworkAdapter()
        if adapter.layout!=LAYOUT or adapter.dimension!=22 or not (np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99)):
            raise RuntimeError("ALERT: adapter layout or original bounds changed")
        data=repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        if str(data) not in before: raise RuntimeError("ALERT: data not covered by pilot provenance")
        with np.load(data) as corpus: trajectory=torch.from_numpy(corpus["prefix_6"].copy())
        lower,upper=torch.from_numpy(adapter.lower.copy()),torch.from_numpy(adapter.upper.copy())
        source_phi=torch.tensor(saved["phi"],dtype=torch.float64)
        with torch.no_grad(): source_loss=float(adapter.objective(source_phi,trajectory))
        source_objective_calls+=1
        if not bits_equal(source_loss,saved["loss"]): raise RuntimeError("ALERT: terminal native loss replay failed")
        save(out,"manifest.json",{"source":str(pilot),"source_accepted_step":156,"source_loss":source_loss,
            "source_stop_reason":result["stop_reason"],"source_radius":diagnostic["radius"],"input_sha256":before,
            "script_sha256":script_hash,"script":str(Path(__file__).resolve()),"copied_profile_math_source":str(original_driver),
            "copied_profile_math_sha256":ORIGINAL_DRIVER_SHA,"layout":LAYOUT,
            "protocol":"Exact terminal replay, then 48 independently zero-anchor-refitted hidden coordinate probes: full/half eps^(1/3)*max(1,abs(h_j)), both signs for all12. No optimizer, continuation, radius update, new damping/acceptance rule, alternative start, fixture, or rollout.",
            "interpretation":"Residual-J comparisons require same saved head face. One-sided native value slopes versus both reduced and envelope gradients are descriptive even across faces. No numerical tolerance/pass gate, no no-descent certificate.",
            "work_note":"49 profile calls include terminal replay plus48 probes; source native objective replay and one derivative replay are separate. All profile calls use existing BVLS exactly once.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})

        class Handoff(Exception): pass

        def profile(h,purpose):
            nonlocal profile_calls,last_profile
            h=torch.as_tensor(np.array(h,dtype=np.float64,copy=True))
            if not bool(torch.isfinite(h).all()) or not bool(((h>=lower[:12])&(h<=upper[:12])).all()):
                raise RuntimeError("TECHNICAL_BLOCKER: declared hidden probe nonfinite or infeasible")
            profile_calls+=1
            zero=torch.cat((h,torch.zeros(10,dtype=torch.float64)))
            zero_r,zero_J=residual_and_jacobian(adapter,trajectory,zero)
            A=zero_J[:,12:22]
            anchor_r=-trajectory[1:].reshape(-1)
            singular=torch.linalg.svdvals(A)
            if not bool(torch.isfinite(singular).all()) or bool((singular<=0).any()):
                raise RuntimeError("TECHNICAL_BLOCKER: nonpositive/nonfinite head singular value; no truncation")
            head,solver=_bounded_residual_least_squares(A,anchor_r,torch.zeros(10),lower[12:22],upper[12:22])
            phi=torch.cat((h,head))
            row={"profile_index":profile_calls,"purpose":purpose,"hidden":h,"phi":phi,"head_solver":solver,
                 "A":A,"head_singular_values":singular,"anchor_residual":anchor_r,"native_zero_minus_anchor_residual":zero_r-anchor_r}
            if not bool(torch.isfinite(phi).all()) or not bool(((phi>=lower)&(phi<=upper)).all()):
                save(out,"infeasible_profile.json",row)
                raise RuntimeError("TECHNICAL_BLOCKER: exact full candidate feasibility failed; no clipping")
            with torch.no_grad():
                loss=float(adapter.objective(phi,trajectory));r=residual_vector(adapter,trajectory,phi).detach()
            if not math.isfinite(loss) or not bool(torch.isfinite(r).all()):
                save(out,"nonfinite_profile.json",{**row,"loss":loss,"residual":r})
                raise RuntimeError("TECHNICAL_BLOCKER: nonfinite profile")
            if not bits_equal(loss,float((r*r).sum()/r.numel())): raise RuntimeError("ALERT: native/residual MSE bits differ")
            model_r=anchor_r+A@head
            kkt,native_kkt=A.T@model_r,A.T@r
            active=torch.tensor(solver["active"])
            row.update(loss=loss,residual=r,linear_residual=model_r,native_minus_linear_residual=r-model_r,
                half_sum_model_head_gradient=kkt,half_sum_native_head_gradient=native_kkt,native_MSE_head_gradient=2/r.numel()*native_kkt,
                free_model_gradient=kkt[active==0],free_native_gradient=native_kkt[active==0],
                lower_active_native_gradient=native_kkt[active==-1],upper_active_native_gradient=native_kkt[active==1],
                active_bound_identity=bool(((active!=-1)|(head==lower[12:22])).all() and ((active!=1)|(head==upper[12:22])).all()),
                free_at_exact_bound=torch.where((active==0)&((head==lower[12:22])|(head==upper[12:22])))[0],
                active_zero_native_multiplier=torch.where((active!=0)&(native_kkt==0))[0],
                lower_margin=head-lower[12:22],upper_margin=upper[12:22]-head,
                native_MSE_bits=True,feasible=True,strictly_better_than_terminal=loss<source_loss)
            if not row["active_bound_identity"]: raise RuntimeError("ALERT: head active status/bound mismatch")
            last_profile=row
            with (out/"profiles.jsonl").open("a") as stream:
                append(stream,{key:value for key,value in row.items() if key not in ("A","anchor_residual","native_zero_minus_anchor_residual")})
            if loss<1e-20:
                save(out,"ALERT_handoff_candidate.json",row)
                print(f"ALERT FEASIBLE_TERMINAL_PROBE_HANDOFF profile={profile_calls} loss={loss:.17g}; stop",flush=True)
                raise Handoff("Feasible native probe below handoff; no subsequent work or recovery certification")
            return row

        def derivative(row):
            nonlocal derivative_calls
            derivative_calls+=1
            phi,r0=row["phi"],row["residual"]
            r,J=residual_and_jacobian(adapter,trajectory,phi)
            if not bits_equal(r,r0) or not bits_equal(J[:,12:22],row["A"]): raise RuntimeError("ALERT: profile derivative residual/head J replay failed")
            B,A=J[:,:12],J[:,12:22]
            active=torch.tensor(row["head_solver"]["active"]);free=torch.where(active==0)[0];AF=A[:,free]
            fixed_r=r.detach()
            mixed=torch.autograd.functional.hessian(lambda value: fixed_r@residual_vector(adapter,trajectory,value),phi)
            C=mixed[12:22,:12][free]
            if len(free):
                U,s,Vh=torch.linalg.svd(AF,full_matrices=False)
                if bool((s<=0).any()) or not bool(torch.isfinite(s).all()): raise RuntimeError("TECHNICAL_BLOCKER: invalid free-face singular value")
                residual_term=U@((Vh@C)/s[:,None])
                Jr=B-U@(U.T@B)-residual_term
                head_derivative=-Vh.T@((U.T@B)/s[:,None]+(Vh@C)/(s[:,None]**2))
            else:
                s=torch.empty(0);residual_term=torch.zeros_like(B);Jr=B.clone();head_derivative=torch.empty((0,12))
            if not bool(torch.isfinite(Jr).all()): raise RuntimeError("TECHNICAL_BLOCKER: nonfinite reduced J")
            g=Jr.T@r
            return {"full_native_J":J,"hidden_partial_J":B,"free_head_indices":free,"free_head_singular_values":s,
                "mixed_residual_C":C,"free_head_derivative":head_derivative,"reduced_J":Jr,
                "residual_weighted_reduced_term":residual_term,"reduced_J_singular_values":torch.linalg.svdvals(Jr),
                "implicit_stationarity_difference":AF.T@Jr+C,"J_reconstruction_difference":Jr-(B+AF@head_derivative),
                "half_sum_reduced_gradient":g,"MSE_reduced_gradient":2/r.numel()*g,
                "MSE_envelope_gradient":2/r.numel()*(B.T@r),"MSE_reduced_minus_envelope":2/r.numel()*(g-B.T@r)}

        center=profile(source_phi[:12].numpy(),"terminal_exact_replay")
        live_derivative=derivative(center)
        checks={key:bits_equal(center[key],saved[key]) for key in ("phi","loss","residual","A","head_singular_values","half_sum_model_head_gradient","native_MSE_head_gradient")}
        checks["head_face"]=center["head_solver"]["active"]==saved["head_solver"]["active"]
        checks.update({"derivative_"+key:bits_equal(value,saved_derivative[key]) for key,value in live_derivative.items()})
        save(out,"terminal_replay.json",{"checks":checks,"profile":center,"derivatives":live_derivative})
        if not all(checks.values()): raise RuntimeError("ALERT: exact saved terminal profile/derivative replay failed")
        h=source_phi[:12].numpy().copy();face=center["head_solver"]["active"]
        eps_step=np.finfo(np.float64).eps**(1/3)
        comparisons=[];candidates=[]
        for j in range(12):
            for factor in (1.0,0.5):
                increment=eps_step*max(1.0,abs(h[j]))*factor
                minus,plus=h.copy(),h.copy();minus[j]-=increment;plus[j]+=increment
                if minus[j]<adapter.lower[j] or plus[j]>adapter.upper[j]:
                    raise RuntimeError("TECHNICAL_BLOCKER: predeclared central probe outside exact hidden bounds; no replacement")
                lo=profile(minus,f"coordinate{j}_factor{factor}_minus")
                hi=profile(plus,f"coordinate{j}_factor{factor}_plus")
                candidates.extend({"coordinate":j,"factor":factor,"sign":sign,"profile_index":row["profile_index"],
                    "loss":row["loss"],"phi":row["phi"],"strictly_better_than_terminal":row["loss"]<source_loss,
                    "head_face":row["head_solver"]["active"]} for sign,row in ((-1,lo),(1,hi)))
                sameface=lo["head_solver"]["active"]==face and hi["head_solver"]["active"]==face
                forward=(hi["loss"]-source_loss)/(plus[j]-h[j])
                backward=(source_loss-lo["loss"])/(h[j]-minus[j])
                reduced=float(live_derivative["MSE_reduced_gradient"][j]);envelope=float(live_derivative["MSE_envelope_gradient"][j])
                entry={"coordinate":j,"factor":factor,"requested_increment":increment,
                    "actual_positive_increment":plus[j]-h[j],"actual_negative_increment":h[j]-minus[j],
                    "minus_profile":lo["profile_index"],"plus_profile":hi["profile_index"],
                    "minus_face":lo["head_solver"]["active"],"plus_face":hi["head_solver"]["active"],"same_face":sameface,
                    "forward_face_matches_center":hi["head_solver"]["active"]==face,"backward_face_matches_center":lo["head_solver"]["active"]==face,
                    "forward_native_MSE_coordinate_slope":forward,"backward_native_MSE_coordinate_slope":backward,
                    "negative_direction_native_MSE_slope":-backward,"reduced_MSE_gradient":reduced,"envelope_MSE_gradient":envelope,
                    "forward_minus_reduced":forward-reduced,"backward_minus_reduced":backward-reduced,
                    "forward_minus_envelope":forward-envelope,"backward_minus_envelope":backward-envelope,
                    "central_native_MSE_coordinate_slope":(hi["loss"]-lo["loss"])/(plus[j]-minus[j]),
                    "minus_loss":lo["loss"],"plus_loss":hi["loss"],
                    "slope_note":"Backward coordinate slope is (f0-fminus)/positive_increment, compared with +gradient; negative-direction slope has opposite sign. Across-face finite slopes are descriptive, not a same-face residual derivative validation."}
                if sameface:
                    fd=(hi["residual"]-lo["residual"])/(plus[j]-minus[j]);analytic=live_derivative["reduced_J"][:,j];error=fd-analytic
                    entry.update(residual_derivative_comparison="SAME_FACE_DESCRIPTIVE",residual_finite_difference=fd,
                        residual_analytic=analytic,residual_difference=error,residual_error_norm=float(error.norm()),
                        residual_analytic_norm=float(analytic.norm()),residual_finite_difference_norm=float(fd.norm()))
                else: entry["residual_derivative_comparison"]="CROSS_FACE_NO_CENTRAL_RESIDUAL_J_COMPARISON"
                comparisons.append(entry)
                save(out,"probe_comparisons.json",{"comparisons":comparisons,"numerical_tolerance_gate":None})
                save(out,"candidate_summary.json",{"source_loss":source_loss,"candidates":candidates,
                    "minimum_probe":min(candidates,key=lambda row:row["loss"]),
                    "strictly_better_candidates":[row for row in candidates if row["strictly_better_than_terminal"]]})
        checks_after={name:sha256(Path(name))==digest for name,digest in before.items()}
        checks_after[str(Path(__file__).resolve())]=sha256(Path(__file__))==script_hash
        unchanged=all(checks_after.values())
        save(out,"integrity.json",{"unchanged":unchanged,"checks":checks_after,"full_protected_guard":"coordinator-owned"})
        if not unchanged: raise RuntimeError("ALERT: relied-on inputs/source/script changed")
        save(out,"result.json",{"status":"TERMINAL_PROFILE_AUDIT_COMPLETE","source_loss":source_loss,"source_accepted_step":156,
            "exact_terminal_replay":all(checks.values()),"profile_calls":profile_calls,"probe_calls":len(candidates),
            "BVLS_calls":profile_calls,"source_objective_calls":source_objective_calls,"derivative_calls":derivative_calls,
            "adapter_calls":dict(adapter.calls),"same_face_pairs":sum(row["same_face"] for row in comparisons),
            "cross_face_pairs":sum(not row["same_face"] for row in comparisons),"minimum_probe":min(candidates,key=lambda row:row["loss"]),
            "strictly_better_candidates":[row for row in candidates if row["strictly_better_than_terminal"]],
            "integrity_unchanged":unchanged,"no_optimization_or_continuation":True,
            "limits":"48 finite coordinate probes. A strict native improvement is only a feasible finite candidate witness; no sampled improvement does not certify no feasible descent. Piecewise residual face changes alone do not show a bug in a potentially differentiable profiled value. No handoff/recovery certificate."})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE terminal audit: profiles={profile_calls}, probes={len(candidates)}, minimum={min(row['loss'] for row in candidates):.17g}",flush=True)
    except BaseException as exc:
        status="ALERT_HANDOFF_CANDIDATE" if type(exc).__name__=="Handoff" else ("ALERT_SOLVER_CYCLE" if "cycle" in str(exc).lower() else ("ALERT_INTEGRITY_OR_REPLAY" if str(exc).startswith("ALERT") else "TECHNICAL_STOP"))
        save(out,"failure.json",{"status":status,"error":str(exc),"type":type(exc).__name__,"traceback":traceback.format_exc(),
            "last_profile":last_profile,"profile_calls":profile_calls,"derivative_calls":derivative_calls,"source_objective_calls":source_objective_calls,
            "no_fallback_or_further_research_calls":True})
        final_checks={};verification_errors={}
        for name,digest in {**before,str(Path(__file__).resolve()):script_hash}.items():
            try: final_checks[name]=sha256(Path(name))==digest
            except BaseException as check_exc: verification_errors[name]=str(check_exc)
        save(out,"integrity.json",{"unchanged":baseline_complete and not verification_errors and all(final_checks.values()),
            "baseline_coverage_complete":baseline_complete,
            "checks":final_checks,"verification_errors":verification_errors,"full_protected_guard":"coordinator-owned","on_failure_or_handoff":True})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="artifact_manifest.json"}})
        print(f"{status}: {exc}",flush=True)
        raise


if __name__=="__main__":
    main()
