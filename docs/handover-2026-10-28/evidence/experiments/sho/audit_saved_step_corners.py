#!/usr/bin/env python3
"""Factorized 2x2 corners of one saved accepted stabilized N3 step.

Two counterfactual corner objective evaluations at most per case. No solver,
derivative, damping, acceptance or subsequent recovery call. Writes only --out.
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
from screen_existing_full_hessian_endpoints import sha256, save

CASES = {"54_original1": ("original",1), "54_extended0": ("extended",0)}
LAYOUT = [
    {"name":"0.weight","shape":[4,2],"start":0,"stop":8},
    {"name":"0.bias","shape":[4],"start":8,"stop":12},
    {"name":"2.weight","shape":[2,4],"start":12,"stop":20},
    {"name":"2.bias","shape":[2],"start":20,"stop":22},
]
SOURCES = (
    "experiments/sho/generic_update_network_adapter.py",
    "experiments/sho/generic_update_residual_n3_opt_in.py",
    "experiments/geometry/prefix_residual_mse_assembly.py",
    "src/hypothesis_space_world_models/models/sho_exact_solution_decoder.py",
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=tuple(CASES),required=True)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    repo,out=args.repo.resolve(strict=True),args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh; no overwrite, batch loop or resume")
    if not (repo/SOURCES[0]).is_file():
        parser.error("--repo lacks the expected adapter")
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    counts={"source_joint_objective_evaluations":0,"new_corner_objective_evaluations":0,
            "source_joint_residual_forward_evaluations":0,"new_corner_residual_forward_evaluations":0}
    completed_corners=[]
    started=time.monotonic()
    try:
        import numpy as np
        import torch
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import residual_vector

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        endpoint,step=CASES[args.case]
        base=repo/"outputs/sho/n2_n3_basin_local_discrimination_20261013/local_models"
        doc_path=base/f"n00005_i00004_{endpoint}_step{step:05d}.json"
        manifest_path,artifact_manifest_path=base/"manifest.json",base/"artifact_manifest.json"
        doc=json.loads(doc_path.read_text())
        source_manifest=json.loads(manifest_path.read_text())
        source_artifact_manifest=json.loads(artifact_manifest_path.read_text())["sha256"]
        trace_source=Path(doc["source_trace"])
        data=repo/"outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        helper=Path(__file__).resolve().with_name("screen_existing_full_hessian_endpoints.py")
        inputs=(doc_path,manifest_path,artifact_manifest_path,trace_source,data,helper)
        before={str(p):sha256(p) for p in inputs}
        sources_before={name:sha256(repo/name) for name in SOURCES}
        script_hash=sha256(Path(__file__))
        for p in (doc_path,manifest_path):
            if before[str(p)]!=source_artifact_manifest.get(p.name):
                raise RuntimeError("ALERT: phaseA artifact hash mismatch")
        for p in (trace_source,data):
            if before[str(p)]!=source_manifest["input_sha256"].get(str(p)):
                raise RuntimeError("ALERT: source trace/data hash mismatch")
        identity={"stem":"n00005_i00004","endpoint":endpoint,"saved_step":step}
        if doc["identity"]!=identity:
            raise RuntimeError("ALERT: source identity mismatch")
        rows={}
        with trace_source.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row["step"] in (step,step+1):
                    if row["step"] in rows:
                        raise RuntimeError("ALERT: duplicate relied-on trace state")
                    rows[row["step"]]=row
        if set(rows)!={step,step+1}:
            raise RuntimeError("BLOCKER: saved source/joint step unavailable")
        source,joint=rows[step],rows[step+1]
        if source!=doc["saved_state"] or not source.get("trial",{}).get("accepted"):
            raise RuntimeError("ALERT: local-model source or accepted-step identity mismatch")
        trial=doc["route_comparisons"]["stabilized"]["trial"]
        model=doc["route_comparisons"]["stabilized"]["accepted_model_comparison"]
        if not trial["accepted"] or not model["accepted_displacement_available"]:
            raise RuntimeError("BLOCKER: accepted saved joint step arrays unavailable")
        adapter=GenericSHOUpdateNetworkAdapter()
        if adapter.layout!=LAYOUT or adapter.dimension!=22:
            raise RuntimeError("ALERT: adapter fixed parameter layout changed")
        if not bool(np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99)):
            raise RuntimeError("ALERT: original box changed")
        lower,upper=torch.from_numpy(adapter.lower.copy()),torch.from_numpy(adapter.upper.copy())
        with np.load(data) as corpus:
            trajectory=torch.from_numpy(corpus["prefix_6"].copy())

        def equal(a,b):
            if isinstance(a,torch.Tensor):
                a=a.detach().numpy()
            if isinstance(b,torch.Tensor):
                b=b.detach().numpy()
            a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
            return a.shape==b.shape and np.array_equal(a.view(np.uint64),b.view(np.uint64))

        def feasible(phi):
            return bool(((phi>=lower)&(phi<=upper)).all())

        theta0=torch.tensor(source["phi"],dtype=torch.float64)
        theta_joint=torch.tensor(joint["phi"],dtype=torch.float64)
        saved_checks={"source_phi":equal(theta0,doc["phi"]),
            "source_loss":equal(source["loss"],doc["loss"]),
            "joint_phi_local_model_trial":equal(theta_joint,trial["phi"]),
            "joint_phi_trace_trial":equal(theta_joint,source["trial"]["phi"]),
            "joint_loss_local_model_trial":equal(joint["loss"],trial["loss"]),
            "joint_loss_trace_trial":equal(joint["loss"],source["trial"]["loss"]),
            "source_feasible":feasible(theta0),"joint_feasible":feasible(theta_joint)}
        if not all(saved_checks.values()):
            save(out,"ALERT_saved_identity.json",{"checks":saved_checks})
            raise RuntimeError("ALERT: saved source/joint identities or feasibility disagree")
        lineage={"case":args.case,"identity":identity,"source_trace":str(trace_source),
            "source_local_model":str(doc_path),"joint_saved_step":step+1}
        save(out,"manifest.json",{"lineage":lineage,"input_sha256":before,
            "direct_source_sha256":sources_before,"script":str(Path(__file__).resolve()),"script_sha256":script_hash,
            "layout":adapter.layout,"corner_order":["hidden_only","readout_only"],
            "maximum_new_corner_objective_evaluations":2,"source_joint_objective_replay_evaluations":2,
            "method":"Factorize exactly the saved accepted displacement. Coordinates are inherited from feasible saved endpoints; no solver, derivative, new step, damping, acceptance decision or recovery.",
            "count_note":"Each evaluated parameter vector uses one assembled objective forward and one separate residual-vector forward. Residual square/sum/divide uses that saved vector without another forward. All four count categories and adapter counters are reported.",
            "environment":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"torch_threads":torch.get_num_threads()},
            "limits":"These are counterfactual corner evaluations, not accepted optimization states. Decomposition identity differences are descriptive, with no numerical pass gate. Full protected/checkpoint integrity is coordinator-owned."})

        def evaluate(name,phi,is_corner):
            if not feasible(phi):
                save(out,"ALERT_infeasible_inherited_corner.json",{"name":name,"phi":phi,"lower":lower,"upper":upper})
                raise RuntimeError("ALERT: exact coordinate-inherited corner is infeasible")
            with torch.no_grad():
                loss=float(adapter.objective(phi,trajectory))
            category="new_corner" if is_corner else "source_joint"
            counts[category+"_objective_evaluations"]+=1
            if not math.isfinite(loss):
                raise RuntimeError("TECHNICAL_BLOCKER: nonfinite actual objective; no further corner evaluations")
            if is_corner and loss<1e-20:
                print(f"ALERT FEASIBLE_CORNER_BELOW_HANDOFF {args.case} {name} loss={loss:.17g}; skip remaining new corner",flush=True)
            with torch.no_grad():
                residual=residual_vector(adapter,trajectory,phi).detach()
            counts[category+"_residual_forward_evaluations"]+=1
            native_mse=float((residual*residual).sum()/residual.numel())
            if not equal(native_mse,loss):
                raise RuntimeError("ALERT: native residual MSE differs bitwise from unchanged assembled objective")
            payload={"name":name,"phi":phi,"loss":loss,"residual":residual,
                "residual_MSE_bits_match":True,"feasible":True,
                "exact_active_bounds":torch.where((phi==lower)|(phi==upper))[0],
                "counterfactual_not_accepted_state":is_corner}
            save(out,name+".json",payload)
            return payload

        start=evaluate("source",theta0,False)
        end=evaluate("saved_joint",theta_joint,False)
        replay_checks={"source_loss_bits":equal(start["loss"],source["loss"]),
            "joint_loss_bits":equal(end["loss"],joint["loss"]),
            "source_residual_bits":equal(start["residual"],doc["residual"]),
            "joint_residual_bits":equal(end["residual"],model["residual_new"]),
            "actual_displacement_bits":equal(theta_joint-theta0,model["actual_displacement"])}
        save(out,"source_joint_verification.json",{"saved_checks":saved_checks,"replay_checks":replay_checks})
        if not all(replay_checks.values()):
            raise RuntimeError("ALERT: exact saved source/joint objective, residual or displacement replay failed")
        corners={"hidden_only":torch.cat((theta_joint[:12],theta0[12:])),
                 "readout_only":torch.cat((theta0[:12],theta_joint[12:]))}
        evaluations={}
        alert=None
        for name,phi in corners.items():
            inherited=(equal(phi[:12],theta_joint[:12]) and equal(phi[12:],theta0[12:])) if name=="hidden_only" else (equal(phi[:12],theta0[:12]) and equal(phi[12:],theta_joint[12:]))
            if not inherited:
                raise RuntimeError("ALERT: coordinate inheritance changed parameter bits")
            value=evaluate(name,phi,True)
            evaluations[name]=value
            completed_corners.append(name)
            if value["loss"]<1e-20:
                alert={"status":"ALERT_FEASIBLE_CORNER_BELOW_HANDOFF","corner":name,
                    "loss":value["loss"],"phi":phi,"remaining_corner_skipped":len(completed_corners)<2,
                    "no_followup_recovery":True}
                break

        if alert is None:
            r0,r_joint=start["residual"],end["residual"]
            rh,ro=evaluations["hidden_only"]["residual"],evaluations["readout_only"]["residual"]
            J=torch.tensor(doc["J"],dtype=torch.float64)
            g=torch.tensor(doc["gradient"],dtype=torch.float64)
            delta=theta_joint-theta0
            dh,do=delta[:12],delta[12:]
            hidden_linear_change=J[:,:12]@dh
            readout_linear_change=J[:,12:]@do
            linear_joint=r0+J@delta
            E_h=rh-r0-hidden_linear_change
            E_o=ro-r0-readout_linear_change
            E_int=r_joint-rh-ro+r0
            E_total=E_h+E_o+E_int
            direct_joint_error=r_joint-linear_joint
            names=("linear_joint","E_h","E_o","E_int")
            vectors=(linear_joint,E_h,E_o,E_int)
            M=r0.numel()
            gram=torch.stack([torch.stack([a@b for b in vectors]) for a in vectors])
            diagonal={name:float(gram[i,i]/M) for i,name in enumerate(names)}
            cross={f"2*{names[i]}_dot_{names[j]}":float(2*gram[i,j]/M) for i in range(4) for j in range(i+1,4)}
            reconstructed=linear_joint+E_total
            decomposition={"lineage":lineage,"saved_J":J,"saved_gradient":g,"saved_actual_displacement":delta,
                "hidden_displacement":dh,"readout_displacement":do,
                "r_source":r0,"r_joint":r_joint,"r_hidden_only":rh,"r_readout_only":ro,
                "hidden_linear_change":hidden_linear_change,"readout_linear_change":readout_linear_change,
                "linear_joint":linear_joint,"E_h":E_h,"E_o":E_o,"E_int":E_int,"E_total":E_total,
                "direct_joint_linearization_error":direct_joint_error,
                "norms":{**{name:float(vector.norm()) for name,vector in zip(names,vectors)},
                    "r_source":float(r0.norm()),"r_joint":float(r_joint.norm()),
                    "r_hidden_only":float(rh.norm()),"r_readout_only":float(ro.norm()),
                    "hidden_linear_change":float(hidden_linear_change.norm()),
                    "readout_linear_change":float(readout_linear_change.norm())},
                "E_total_norm":float(E_total.norm()),"direct_joint_error_norm":float(direct_joint_error.norm()),
                "pairwise_dot_order":names,"pairwise_dot_matrix":gram,
                "pairwise_cross_energy_matrix":2*gram,
                "MSE_energy_terms":{"self_terms":diagonal,"distinct_cross_terms":cross,
                    "sum_all_terms":sum(diagonal.values())+sum(cross.values()),
                    "actual_joint_MSE":end["loss"],"reconstructed_joint_MSE":float((reconstructed*reconstructed).sum()/M)},
                "losses":{"source":start["loss"],"saved_joint":end["loss"],
                    "hidden_only":evaluations["hidden_only"]["loss"],"readout_only":evaluations["readout_only"]["loss"]},
                "loss_2x2_interaction":end["loss"]-evaluations["hidden_only"]["loss"]-evaluations["readout_only"]["loss"]+start["loss"],
                "saved_gradient_first_order_terms":{"hidden":float(g[:12]@dh),"readout":float(g[12:]@do)},
                "descriptive_identity_differences":{
                    "E_total_minus_direct_joint_error":E_total-direct_joint_error,
                    "E_total_minus_direct_joint_error_norm":float((E_total-direct_joint_error).norm()),
                    "joint_residual_minus_reconstruction":r_joint-reconstructed,
                    "joint_residual_minus_reconstruction_norm":float((r_joint-reconstructed).norm()),
                    "joint_linear_minus_split_linear":linear_joint-(r0+hidden_linear_change+readout_linear_change),
                    "direct_joint_error_minus_saved_error_norm":float((direct_joint_error-torch.tensor(model["residual_nonlinear_error"])).norm()),
                    "energy_terms_minus_actual_joint_MSE":sum(diagonal.values())+sum(cross.values())-end["loss"]},
                "interpretation":"E_h is hidden-block nonlinear residual, E_o output-block affine-rounding residual, E_int the exact sampled mixed residual interaction. Corner losses can worsen. Algebraic decomposition does not identify a primary recovery bottleneck or constitute accepted steps. No numerical identity tolerance/gate."}
            save(out,"decomposition.json",decomposition)
            status="SAVED_STEP_CORNER_AUDIT_COMPLETE"
        else:
            save(out,"ALERT.json",alert)
            status=alert["status"]
        after={str(p):sha256(p) for p in inputs}
        sources_after={name:sha256(repo/name) for name in SOURCES}
        unchanged=before==after and sources_before==sources_after and script_hash==sha256(Path(__file__))
        save(out,"integrity.json",{"direct_inputs_sources_script_unchanged":unchanged,"input_sha256_after":after,
            "direct_source_sha256_after":sources_after,"note":"Full protected/checkpoint integrity is coordinator-owned."})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on input/source/script integrity changed")
        counts["total_objective_evaluations"]=counts["source_joint_objective_evaluations"]+counts["new_corner_objective_evaluations"]
        counts["total_residual_forward_evaluations"]=counts["source_joint_residual_forward_evaluations"]+counts["new_corner_residual_forward_evaluations"]
        counts["expected_total_forward_calls"]=counts["total_objective_evaluations"]+counts["total_residual_forward_evaluations"]
        save(out,"result.json",{"status":status,"lineage":lineage,"completed_corners":completed_corners,
            "source_loss":start["loss"],"saved_joint_loss":end["loss"],
            "corner_losses":{name:value["loss"] for name,value in evaluations.items()},
            "evaluation_counts":counts,"adapter_counters":adapter.calls,"alert":alert,
            "no_solver_derivative_acceptance_or_recovery_calls":True,
            "direct_integrity_unchanged":unchanged,"elapsed_seconds":time.monotonic()-started})
        save(out,"artifact_manifest.json",{"sha256":{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(f"COMPLETE {args.case}: {status}; new_corner_objectives={counts['new_corner_objective_evaluations']}",flush=True)
    except BaseException as exc:
        save(out,"failure.json",{"error":str(exc),"exception_type":type(exc).__name__,
            "traceback":traceback.format_exc(),"evaluation_counts":counts,"completed_corners":completed_corners,
            "status":"ALERT_OR_TECHNICAL_BLOCKER_NO_FURTHER_CORNER_EVALUATIONS"})
        raise


if __name__=="__main__":
    main()
