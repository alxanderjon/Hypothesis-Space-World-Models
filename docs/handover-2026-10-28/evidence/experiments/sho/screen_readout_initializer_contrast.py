#!/usr/bin/env python3
"""Two declared saved readout initializers, one fresh 50-step screen per call.

No refit or control rerun. Existing stabilized N3, damping, bounds, acceptance
and stagnation update are reused unchanged. Writes only --out.
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
from screen_existing_full_hessian_endpoints import SOURCE_FILES,sha256,save,append,serializable

CASES={"54_original37":("original",37,"original_step37_control"),
       "54_extended50":("extended",50,"extended_step50_control")}
LAYOUT=[{"name":"0.weight","shape":[4,2],"start":0,"stop":8},
        {"name":"0.bias","shape":[4],"start":8,"stop":12},
        {"name":"2.weight","shape":[2,4],"start":12,"stop":20},
        {"name":"2.bias","shape":[2],"start":20,"stop":22}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=tuple(CASES),required=True)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    repo,out=args.repo.resolve(strict=True),args.out.resolve()
    if out.exists(): parser.error("--out must be fresh; no overwrite or resume")
    if not (repo/SOURCE_FILES[0]).is_file(): parser.error("--repo lacks expected runtime")
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    accepted,checkpoint=0,None
    try:
        import numpy as np
        import scipy
        import torch
        from experiments.resume_hybrid_student_after_ckdtree import Log
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import OptInResidualN3Runtime,STABILIZED_RESIDUAL,residual_vector
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
        from experiments.sho import run_stage1b_n3_gradient_polishing as prod
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        endpoint,source_step,control_name=CASES[args.case]
        run=repo/"outputs/sho/n2_n3_basin_local_discrimination_20261013"
        refit_dir=run/"conditional_readout"/args.case
        control_dir=run/"damping_contrast"/control_name
        candidate_path,refit_result_path,refit_manifest_path=refit_dir/"candidate.json",refit_dir/"result.json",refit_dir/"manifest.json"
        refit_hash_path=refit_dir/"artifact_manifest.json"
        candidate=json.loads(candidate_path.read_text())
        refit_result=json.loads(refit_result_path.read_text())
        refit_manifest=json.loads(refit_manifest_path.read_text())
        refit_hashes=json.loads(refit_hash_path.read_text())["sha256"]
        trace_source=Path(refit_result["lineage"]["source_trace"])
        doc_path=Path(refit_result["lineage"]["source_local_model"])
        doc=json.loads(doc_path.read_text())
        control_path,control_result_path,control_hash_path=control_dir/"damping_contrast.jsonl",control_dir/"result.json",control_dir/"artifact_manifest.json"
        control_result=json.loads(control_result_path.read_text())
        control_hashes=json.loads(control_hash_path.read_text())["sha256"]
        data=repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        helper=Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs=(candidate_path,refit_result_path,refit_manifest_path,refit_hash_path,trace_source,doc_path,
                control_path,control_result_path,control_hash_path,data,helper)
        before={str(p):sha256(p) for p in inputs}
        sources_before={name:sha256(repo/name) for name in SOURCE_FILES}
        script_hash=sha256(Path(__file__))
        for p in (candidate_path,refit_result_path,refit_manifest_path):
            if before[str(p)]!=refit_hashes.get(p.name): raise RuntimeError("ALERT: prior refit artifact hash mismatch")
        for p in (trace_source,doc_path,data):
            if before[str(p)]!=refit_manifest["input_sha256"].get(str(p)): raise RuntimeError("ALERT: prior refit input provenance mismatch")
        for p in (control_path,control_result_path):
            if before[str(p)]!=control_hashes.get(p.name): raise RuntimeError("ALERT: saved control artifact hash mismatch")
        identity={"stem":"n00005_i00004","endpoint":endpoint,"saved_step":source_step}
        if doc["identity"]!=identity or refit_result["lineage"]["identity"]!=identity or refit_manifest["lineage"]["case"]!=args.case:
            raise RuntimeError("ALERT: initializer source identity mismatch")
        history={}
        with trace_source.open() as f:
            for line in f:
                row=json.loads(line)
                if source_step<=row["step"]<=source_step+51:
                    if row["step"] in history: raise RuntimeError("ALERT: duplicated saved source/control state")
                    history[row["step"]]=row
        if sorted(history)!=list(range(source_step,source_step+52)): raise RuntimeError("BLOCKER: source through +51 control states unavailable")
        source=history[source_step]
        if source!=doc["saved_state"]: raise RuntimeError("ALERT: source state differs from phaseA")
        if source.get("stop_reason") is not None or source["loss"]<1e-20 or source["stagnation_count"]>=prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
            raise RuntimeError("ALERT: source has a true stop; no resetting allowed")
        matched_control=[]
        with control_path.open() as f:
            for line in f:
                row=json.loads(line)
                k=int(row["step"])
                if k!=len(matched_control) or not 0<=k<=50: raise RuntimeError("ALERT: saved control step sequence mismatch")
                ref=history[source_step+k]
                if row["source_offset_step"]!=source_step+k or not bits_equal(row["phi"],ref["phi"]) or not bits_equal(row["loss"],ref["loss"]) or row["stagnation_count"]!=ref["stagnation_count"]:
                    raise RuntimeError("ALERT: saved control differs from original batch trace")
                matched_control.append(k)
        if matched_control!=list(range(51)) or control_result["work"]["additional_accepted_steps"]!=50 or not bits_equal(control_result["loss"],history[source_step+50]["loss"]):
            raise RuntimeError("ALERT: completed 50-step control verification failed")
        prior_trial=candidate["unchanged_acceptance_replay"]
        accepted_candidate=candidate["accepted_replay_candidate"]
        theta=torch.tensor(accepted_candidate["phi"],dtype=torch.float64)
        initial_loss=float(prior_trial["loss"])
        source_theta=torch.tensor(source["phi"],dtype=torch.float64)
        checks={"prior_result_complete":refit_result["status"]=="CONDITIONAL_READOUT_AUDIT_COMPLETE",
            "prior_accepted":prior_trial["accepted"] is True,"prior_full_step":prior_trial["backtrack"]==0,
            "accepted_full_phi_bits":bits_equal(theta,candidate["full_candidate_phi"]),
            "accepted_trial_phi_bits":bits_equal(theta,prior_trial["phi"]),
            "result_trial_phi_bits":bits_equal(theta,refit_result["acceptance_replay"]["phi"]),
            "candidate_loss_bits":bits_equal(initial_loss,candidate["full_candidate_loss"]),
            "result_loss_bits":bits_equal(initial_loss,refit_result["full_candidate_loss"]),
            "source_phi_bits":bits_equal(source_theta,candidate["source_phi"]),
            "source_loss_bits":bits_equal(source["loss"],candidate["source_loss"]),
            "fixed_hidden_bits":bits_equal(theta[:12],source_theta[:12]),
            "prior_hidden_unchanged":candidate["exact_hidden_bits_unchanged"] is True,
            "prior_candidate_feasible":candidate["candidate_in_bounds"] is True,
            "prior_same_as_full_bits":accepted_candidate["same_as_full_candidate_bits"] is True}
        adapter=GenericSHOUpdateNetworkAdapter()
        checks["layout"]=adapter.layout==LAYOUT
        checks["bounds"]=bool(np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99))
        lower,upper=torch.from_numpy(adapter.lower.copy()),torch.from_numpy(adapter.upper.copy())
        checks["initializer_in_bounds"]=bool(((theta>=lower)&(theta<=upper)).all())
        if not all(checks.values()):
            save(out,"ALERT_initializer_identity.json",{"checks":checks})
            raise RuntimeError("ALERT: saved initializer identity, hidden parameters or bounds disagree")
        with np.load(data) as corpus: trajectory=torch.from_numpy(corpus["prefix_6"].copy())
        with torch.no_grad():
            live_loss=float(adapter.objective(theta,trajectory))
            live_residual=residual_vector(adapter,trajectory,theta).detach()
        checks["native_objective_bits"]=bits_equal(live_loss,initial_loss)
        checks["native_residual_MSE_bits"]=bits_equal(float((live_residual*live_residual).sum()/live_residual.numel()),live_loss)
        checks["candidate_residual_bits"]=bits_equal(live_residual,candidate["actual_candidate_residual"])
        checks["accepted_residual_bits"]=bits_equal(live_residual,accepted_candidate["actual_residual"])
        save(out,"initializer_verification.json",{"checks":checks,"theta":theta,"loss":live_loss,
            "residual":live_residual,"source_stagnation_count":source["stagnation_count"],"matched_control_states":matched_control})
        if not all(checks.values()): raise RuntimeError("ALERT: initial saved conditional candidate replay failed")
        initializer_displacement=theta-source_theta
        initializer_path=float(initializer_displacement.norm())
        prior_work={key:refit_result[key] for key in ("BVLS_calls","source_objective_verification_calls",
            "candidate_and_replay_scalar_objective_evaluations","replay_trial_objective_evaluations","direct_candidate_scalar_objective_evaluations")}
        lineage={"case":args.case,"source_identity":identity,"initializer_source":str(candidate_path),
            "source_loss":source["loss"],"initializer_loss":initial_loss,"initializer_displacement":initializer_displacement,
            "initializer_path_length":initializer_path,"initializer_is_prior_separate_move":True,
            "counter_policy":"Inherit the source stagnation counter unchanged. The saved conditional refit is a separate initializer intervention, not a fabricated N3 accepted step or stagnation update."}
        save(out,"manifest.json",{"lineage":lineage,"input_sha256":before,"direct_source_sha256":sources_before,
            "script":str(Path(__file__).resolve()),"script_sha256":script_hash,"prior_refit_recorded_work":prior_work,
            "setup_work":{"assembled_objective_verifications":1,"separate_residual_forward_verifications":1},
            "protocol":"Only two declared initializers. Unchanged existing stabilized N3, original damping/bounds/backtracking/strict acceptance/stagnation/stops, maximum 50 additional accepted steps. No new refit, control rerun or further arm.",
            "comparison":"Original saved source+50 is the paired equal-N3-step control; source+51 is descriptive context for the separate extra initializer move. Neither is an equal-compute comparison.",
            "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch_threads":torch.get_num_threads()}})

        def comparison(k):
            same,extra=history[source_step+k],history[source_step+k+1]
            return {"equal_additional_N3_steps":{"source_step":source_step+k,"loss":same["loss"]},
                "one_extra_saved_step_descriptive":{"source_step":source_step+k+1,"loss":extra["loss"]},
                "note":"Saved source trajectory; no control rerun. Extra-step context is not a compute-matched control."}

        save(out,"saved_controls.json",{"verified_damping_control_states":matched_control,
            "milestones":{str(k):comparison(k) for k in (0,1,10,50)},
            "source_plus_50":history[source_step+50],"source_plus_51":history[source_step+51]})
        rt=OptInResidualN3Runtime(adapter,trajectory,Log(),direction_mode=STABILIZED_RESIDUAL)
        loss=initial_loss
        counter=int(source["stagnation_count"])
        path_length,bt=0.0,0
        contacts,snapshots=set(),{}
        stop=None
        started=time.monotonic()
        trace_path=out/"stabilized.jsonl"

        def work():
            return {"N3_additional_accepted_steps":accepted,"source_N3_steps_plus_additional_excluding_refit":source_step+accepted,
                "separate_prior_initializer_moves":1,"N3_parameter_path":path_length,
                "path_including_initializer":initializer_path+path_length,"N3_backtracking_rejections":bt,
                "N3_trial_objective_evaluations":rt.objective_evaluations,
                "N3_derivative_calls":rt.binding_counters.derivative_calls,
                "setup_objective_verifications":1,"setup_separate_residual_forwards":1,
                "prior_refit_work_not_charged_as_new_N3":prior_work}

        def write_checkpoint():
            active=set(torch.where((theta==rt.lower)|(theta==rt.upper))[0].tolist())
            save(out,"diagnostic_state.json",{"lineage":lineage,"theta":theta,"loss":loss,
                "stagnation_count":counter,"stop_reason":stop,"work":work(),"snapshots":snapshots,
                "bound_coordinates_ever":sorted(contacts|active),"input_sha256":before,
                "direct_source_sha256":sources_before,"script_sha256":script_hash,"automatic_resume":False})

        checkpoint=write_checkpoint
        write_checkpoint()
        with trace_path.open("x") as stream:
            while True:
                if loss<1e-20:
                    stop="HANDOFF_THRESHOLD_REACHED"
                    print(f"ALERT READOUT_INITIALIZER_HANDOFF {args.case} N3_steps={accepted} loss={loss:.17g}; no further direction evaluation",flush=True)
                    append(stream,{"step":accepted,"phi":theta,"loss":loss,"stagnation_count":counter,
                        "stop_reason":stop,"work":work(),"derivatives_skipped_after_handoff":True})
                    write_checkpoint()
                    break
                result,g,H,gn,pre,route,d,qp=rt.derivatives_and_direction(theta)
                if not bits_equal(result.loss,loss): raise RuntimeError("ALERT: derivative-state loss replay mismatch")
                pg=g.clone()
                pg[((theta==rt.lower)&(g>=0))|((theta==rt.upper)&(g<=0))]=0
                active=torch.where((theta==rt.lower)|(theta==rt.upper))[0].tolist()
                contacts.update(active)
                row={"step":accepted,"phi":theta,"loss":loss,"gradient":g,"gradient_norm":gn,
                    "projected_gradient_norm":float(pg.norm()),"route":route,"pre":pre,
                    "mu":None if qp is None else qp.get("damping"),"direction":d,
                    "direction_norm":None if d is None else float(d.norm()),"qp":qp,"active_bounds":active,
                    "path":path_length,"backtracking":bt,"objective_evaluations":rt.objective_evaluations,
                    "stagnation_count":counter,"work":work(),"saved_control_comparison":comparison(accepted)}
                if accepted in (0,1,10,50): snapshots[str(accepted)]=serializable(row)
                if stop is None and pre!="CONTINUE": stop=pre
                elif stop is None and accepted>=50: stop="SCREEN_RESOURCE_LIMIT_50_ADDITIONAL_ACCEPTED_STEPS"
                if stop:
                    row["stop_reason"]=stop
                    append(stream,row)
                    write_checkpoint()
                    break
                trial=replay(rt,theta,loss,g,d)
                row["trial"]=trial
                if trial["accepted"]:
                    predicted=-float(qp["quadratic_value"])
                    counter=prod._updated_float64_stagnation_count(counter,loss,trial["loss"],predicted)
                    row["full_predicted_decrease"]=predicted
                    theta=torch.tensor(trial["phi"],dtype=torch.float64)
                    loss=trial["loss"]
                    path_length+=trial["displacement"]
                    bt+=trial["rejections"]
                    accepted+=1
                    if counter>=prod.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE: stop=prod.FLOAT64_STAGNATION_STOP_REASON
                else:
                    stop=trial["reason"]
                    bt+=trial.get("rejections",0)
                    row["stop_reason"]=stop
                append(stream,row)
                write_checkpoint()
                if not trial["accepted"]: break
                if accepted in (1,10,50) or stop: print(f"READOUT_INIT {args.case} N3_steps={accepted} loss={loss:.17g} stop={stop}",flush=True)
        after={str(p):sha256(p) for p in inputs}
        sources_after={name:sha256(repo/name) for name in SOURCE_FILES}
        unchanged=before==after and sources_before==sources_after and script_hash==sha256(Path(__file__))
        save(out,"integrity.json",{"direct_sources_inputs_script_unchanged":unchanged,"input_sha256_after":after,
            "direct_source_sha256_after":sources_after,"note":"Full protected/checkpoint integrity is coordinator-owned."})
        if not unchanged: raise RuntimeError("ALERT: relied-on inputs/source/script integrity changed")
        save(out,"result.json",{"lineage":lineage,"phi":theta,"loss":loss,"stagnation_count":counter,
            "stop_reason":stop,"handoff":loss<1e-20,"work":work(),"snapshots":snapshots,
            "matched_observed_horizon_controls":comparison(accepted),"requested_50_step_controls":comparison(50),
            "bound_coordinates_ever":sorted(contacts),"trace_sha256":sha256(trace_path),
            "direct_integrity_unchanged":unchanged,"elapsed_seconds":time.monotonic()-started,
            "limits":"Two dependent saved states; separate prior readout initializer costs extra work. No automatic primary-bottleneck claim or followup."})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.case}: N3_steps={accepted} loss={loss:.17g} {stop}",flush=True)
    except BaseException as exc:
        if checkpoint is not None:
            try: checkpoint()
            except BaseException: pass
        status="ALERT_SOLVER_CYCLE" if "cycle" in str(exc).lower() else "ALERT_OR_TECHNICAL_BLOCKER_NO_FOLLOWUP"
        print(f"{status}: {exc}",flush=True)
        save(out,"failure.json",{"status":status,"additional_N3_steps":accepted,"error":str(exc),
            "exception_type":type(exc).__name__,"traceback":traceback.format_exc()})
        raise


if __name__=="__main__": main()
