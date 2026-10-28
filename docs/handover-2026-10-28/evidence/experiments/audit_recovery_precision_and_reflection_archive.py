#!/usr/bin/env python3
"""Focused recovery-precision and reflection-ball archive-handoff audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/recovery_precision_and_reflection_archive_audit")
BALL_CASES = OUT / "reflection_ball_cases"
BALL_CHECKS = OUT / "reflection_ball_checks"
BALL_HISTORY = Path("outputs/d5_production_pipeline_validation/cases/reflection_ball")
KEPLER_ARCHIVE = Path("outputs/kepler_archive_absorbing_handoff/cases")
IDENTIFIABILITY = Path("outputs/d5_handoff_characterization/reflection_identifiability.json")
STAGES = (20, 100, 200, 500, 1_000, 2_000, 5_000)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ulp_errors(values: np.ndarray, reference: np.ndarray) -> list[int]:
    """Exact distance in the monotonically ordered binary64 bit space."""
    mask=(1<<64)-1; sign=1<<63
    def ordered(value):
        bits=struct.unpack(">Q",struct.pack(">d",float(value)))[0]
        return ((~bits)&mask) if bits&sign else (bits|sign)
    return [abs(ordered(a)-ordered(b)) for a,b in
            zip(np.asarray(values,np.float64),np.asarray(reference,np.float64))]


def exact_rollout_diagnostic(system: str, trajectory: np.ndarray,
                             theta: np.ndarray) -> dict:
    if system == "kepler5":
        start = production.kepler_teacher.PREFIX_STATES - 1
    else:
        start = production.ball_teacher.PREFIX_STATES - 1
    truth = trajectory[start:]
    predicted = np.empty_like(truth); predicted[0] = truth[0]
    topology = 0
    for index in range(len(truth)-1):
        if system == "kepler5":
            predicted[index+1] = production.known.step(
                predicted[index:index+1], theta, np)[0]
        else:
            predicted[index+1] = ball.numpy_step(predicted[index], theta)
            topology += int(ball.event_signature(predicted[index], theta) !=
                            ball.event_signature(truth[index], ball.TEACHER))
    difference = predicted-truth
    return dict(
        rollout_values=int(truth.size),
        exact_rollout_values=int(np.count_nonzero(predicted == truth)),
        exact_rollout_fraction=float(np.count_nonzero(predicted == truth)/truth.size),
        rollout_max_error=float(np.max(np.abs(difference))),
        rollout_mse=float(np.mean(difference*difference)),
        topology_regressions=int(topology),
        rollout_finite=bool(np.isfinite(predicted).all()),
    )


def kepler_precision_audit() -> dict:
    rows = []
    for case_id in range(64):
        source = json.loads((KEPLER_ARCHIVE/f"case_{case_id:03d}.json").read_text())
        hit = source["successful_archived_handoff"]
        final = np.asarray(hit["final_theta"], np.float64)
        trajectory, _ = production._kepler_case(case_id)
        exact = exact_rollout_diagnostic("kepler5", trajectory, final)
        rows.append(dict(
            case_id=case_id, final_loss=float(hit["final_loss"]),
            loss_gate=bool(hit["final_loss"] < production.FINAL_GATE),
            final_theta=final.tolist(),
            parameter_error=(final-production.known.GT).tolist(),
            parameter_l2_error=float(np.linalg.norm(final-production.known.GT)),
            parameter_max_error=float(np.max(np.abs(final-production.known.GT))),
            exact_parameter_values=int(np.count_nonzero(final == production.known.GT)),
            parameter_ulp_errors=ulp_errors(final, production.known.GT),
            maximum_parameter_ulp_error=max(ulp_errors(final, production.known.GT)),
            **exact,
        ))
    parameter_order = sorted(rows, key=lambda row: row["parameter_l2_error"])
    rollout_order = sorted(rows, key=lambda row: row["rollout_max_error"])
    loss_order = sorted(rows, key=lambda row: row["final_loss"])

    def adjacent(values, key):
        answer = []
        for left, right in zip(values, values[1:]):
            a=float(left[key]); b=float(right[key])
            answer.append(dict(lower_case=left["case_id"], upper_case=right["case_id"],
                               lower=a, upper=b, ratio=(b/a if a else None)))
        return sorted(answer, key=lambda row: -math.inf if row["ratio"] is None
                      else row["ratio"], reverse=True)[:8]

    historical = {
        "sho_full_k_256": dict(
            source="outputs/sho/stage1B/checkpointed_resource_edirect/report.json",
            worst_parameter_error=8.717251743633099e-14,
            worst_rollout_error=1.3418155475619642e-12),
        "full_k_ball_50": dict(
            source="outputs/bouncing_ball/full_k_50_transfer/report.json",
            worst_parameter_error=7.218894946044636e-15,
            worst_rollout_error=1.1368683772161603e-13),
        "reflection_ball_8": dict(
            source="outputs/bouncing_ball/reflection_peelback/report.json",
            worst_parameter_error=6.675117091749783e-16,
            worst_rollout_error=7.771561172376096e-15),
        "four_parameter_kepler_8": dict(
            source="outputs/kepler/four_parameter_deterministic_peelback/report.json",
            worst_parameter_error=7.966422700303412e-14,
            worst_rollout_error=1.427469253911795e-13),
    }
    return dict(
        cases=64,
        coded_recovery_semantics=dict(
            n3_recovered="final_loss < 1e-20",
            full_pipeline_pass="n3_recovered AND finite rollout",
            independent_parameter_or_rollout_gate_exists=False,
            bit_exact_containment_is_used_for_teacher_adapters_but_not_recovered_parameters=True),
        rows=rows,
        parameter_error_adjacent_gaps=adjacent(parameter_order, "parameter_l2_error"),
        rollout_error_adjacent_gaps=adjacent(rollout_order, "rollout_max_error"),
        final_loss_adjacent_gaps=adjacent(loss_order, "final_loss"),
        historical_validated_recovery_precision=historical,
    )


class BallAudit:
    def __init__(self, case_id: int):
        self.case_id=case_id
        self.trajectory, self.initial_condition = production._ball_case(case_id)
        self.objective, self.torch_objective = production._objectives(
            "reflection_ball", self.trajectory)
        self.lower, self.upper, self.teacher = production._config("reflection_ball")
        self.width=self.upper-self.lower
        self.history_path=BALL_HISTORY/f"case_{case_id:03d}"/"n1_state.npz"
        saved=np.load(self.history_path)
        self.points=np.asarray(saved["points"],np.float64)
        self.losses=np.asarray(saved["losses"],np.float64)
        self.thetas=self.lower+self.width*self.points

    def endpoint(self,n,index):
        theta=self.thetas[index]; radius=self.width/float(n)**(1/len(theta))
        return dict(theta=theta,loss=float(self.losses[index]),evaluations=n,
                    lower=np.maximum(self.lower,theta-radius),
                    upper=np.minimum(self.upper,theta+radius),
                    stop_reason="reflection_archive_absorbing_audit")

    def check(self,n,index):
        path=BALL_CHECKS/f"case_{self.case_id:03d}"/f"n{n:05d}_i{index+1:05d}.json"
        if path.exists(): return json.loads(path.read_text())
        endpoint=self.endpoint(n,index)
        finish=production._finish("reflection_ball",self.trajectory,self.objective,
            self.torch_objective,endpoint,self.teacher,
            f"reflection_archive_case_{self.case_id}_n{n}_i{index+1}")
        final=np.asarray(finish["final_theta"],np.float64)
        metrics=dict(finish)
        metrics.update(exact_rollout_diagnostic("reflection_ball",self.trajectory,final))
        row=dict(case_id=self.case_id,current_evaluation=n,archive_index=index,
            insertion_evaluation=index+1,archived_theta=self.thetas[index].tolist(),
            archived_loss=float(self.losses[index]),
            final_loss_gate=bool(finish["recovered"]),
            existing_full_pipeline_pass=bool(finish["full_pipeline_pass"]),
            exact_parameter_values=int(np.count_nonzero(final==self.teacher)),
            parameter_ulp_errors=ulp_errors(final,self.teacher),
            maximum_parameter_ulp_error=max(ulp_errors(final,self.teacher)),
            **metrics)
        atomic_json(path,row); return row

    def checkpoint(self,n):
        for index in range(min(n,len(self.losses))):
            row=self.check(n,index)
            if row["existing_full_pipeline_pass"]: return row
        return None


def prior_ball_handoff(case_id: int) -> dict:
    return json.loads((BALL_HISTORY/f"case_{case_id:03d}"/"result.json").read_text())["production"]


def run_ball_case(case_id: int) -> dict:
    path=BALL_CASES/f"case_{case_id:03d}.json"
    if path.exists(): return json.loads(path.read_text())
    audit=BallAudit(case_id); started=time.monotonic(); hit=None; checks=[]
    # Exact early audit because count-derived viability is non-monotone.
    for n in range(1,min(20,len(audit.losses))+1):
        candidate=audit.checkpoint(n); checks.append(dict(evaluation=n,success=candidate is not None))
        if candidate is not None: hit=candidate; break
    if hit is None:
        for n in STAGES:
            if n<=20 or n>len(audit.losses): continue
            candidate=audit.checkpoint(n); checks.append(dict(evaluation=n,success=candidate is not None))
            if candidate is not None: hit=candidate; break
    previous=prior_ball_handoff(case_id)
    result=dict(case_id=case_id,initial_condition=audit.initial_condition,
        saved_evaluations=len(audit.losses),history_path=str(audit.history_path),
        history_sha256=file_hash(audit.history_path),checks=checks,
        successful_archived_handoff=None,elapsed_seconds=time.monotonic()-started)
    if hit is not None:
        n=int(hit["current_evaluation"]); index=int(hit["archive_index"])
        order=np.lexsort((np.arange(n),audit.losses[:n])); rank=int(np.flatnonzero(order==index)[0])+1
        incumbent=int(np.argmin(audit.losses[:n])); insertion=audit.check(index+1,index)
        result["successful_archived_handoff"]=dict(**hit,
            is_current_incumbent=bool(index==incumbent),
            archived_loss_rank=rank,current_incumbent_loss=float(audit.losses[incumbent]),
            archived_loss_over_incumbent=float(hit["archived_loss"]/audit.losses[incumbent]),
            archive_age_evaluations=n-index-1,
            viable_at_insertion=bool(insertion["existing_full_pipeline_pass"]),
            became_viable_later_due_to_box_scale=bool(
                not insertion["existing_full_pipeline_pass"] and n>index+1),
            prior_incumbent_handoff_evaluation=int(previous["n1_evaluations"]),
            handoff_lead_vs_production=int(previous["n1_evaluations"]-n))
    atomic_json(path,result)
    print(f"ball {case_id:02d}: "+("NO HIT" if hit is None else
          f"n={hit['current_evaluation']} insert={hit['insertion_evaluation']}"),flush=True)
    return result


def run_ball_incumbent_case(case_id: int) -> dict:
    """Offline comparator: earliest successful current-incumbent handoff."""
    path=BALL_CASES/f"case_{case_id:03d}_incumbent.json"
    if path.exists(): return json.loads(path.read_text())
    audit=BallAudit(case_id); hit=None; checks=[]
    for n in range(1,min(20,len(audit.losses))+1):
        index=int(np.argmin(audit.losses[:n])); row=audit.check(n,index)
        checks.append(dict(evaluation=n,archive_index=index,
                           success=bool(row["existing_full_pipeline_pass"])))
        if row["existing_full_pipeline_pass"]: hit=row; break
    if hit is None:
        for n in STAGES:
            if n<=20 or n>len(audit.losses): continue
            index=int(np.argmin(audit.losses[:n])); row=audit.check(n,index)
            checks.append(dict(evaluation=n,archive_index=index,
                               success=bool(row["existing_full_pipeline_pass"])))
            if row["existing_full_pipeline_pass"]: hit=row; break
    result=dict(case_id=case_id,checks=checks,successful_incumbent_handoff=hit)
    atomic_json(path,result)
    return result


def dist(values):
    a=np.asarray(values,np.float64)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),
                p90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(count=0)


def summarize_ball(rows):
    hits=[r["successful_archived_handoff"] for r in rows if r["successful_archived_handoff"]]
    return dict(cases=len(rows),recoveries=len(hits),loss_gate_recoveries=sum(h["final_loss_gate"] for h in hits),
        finite_rollout_recoveries=sum(h["existing_full_pipeline_pass"] for h in hits),
        nonincumbent_handoffs=sum(not h["is_current_incumbent"] for h in hits),
        became_viable_later=sum(h["became_viable_later_due_to_box_scale"] for h in hits),
        handoff_evaluations=dist([h["current_evaluation"] for h in hits]),
        insertion_evaluations=dist([h["insertion_evaluation"] for h in hits]),
        handoff_lead_vs_production=dist([h["handoff_lead_vs_production"] for h in hits]),
        parameter_l2_error=dist([h["final_parameter_l2_error"] for h in hits]),
        rollout_max_error=dist([h["rollout_max_error"] for h in hits]),
        rows=rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--ball-case",type=int)
    parser.add_argument("--ball-incumbent-case",type=int)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.ball_case is not None:
        run_ball_case(args.ball_case); return
    if args.ball_incumbent_case is not None:
        result=run_ball_incumbent_case(args.ball_incumbent_case)
        hit=result["successful_incumbent_handoff"]
        print(f"ball incumbent {args.ball_incumbent_case:02d}: "+
              ("none" if hit is None else
               f"n={hit['current_evaluation']} insert={hit['insertion_evaluation']}"),flush=True)
        return
    before=hashes(); precision=kepler_precision_audit(); atomic_json(OUT/"kepler_precision.json",precision)
    ident=json.loads(IDENTIFIABILITY.read_text())
    ids=[row["case_id"] for row in ident["rows"] if row["classification"]=="IDENTIFIABILITY_COMPLETE"]
    rows=[run_ball_case(case_id) for case_id in ids]
    ball_summary=summarize_ball(rows); after=hashes()
    report=dict(kepler_precision=precision,reflection_ball=ball_summary,
                checkpoint_hashes_unchanged=before==after)
    atomic_json(OUT/"report.json",report)
    print(json.dumps(dict(kepler_cases=precision["cases"],reflection_ball={k:v for k,v in ball_summary.items() if k!="rows"},checkpoint_hashes_unchanged=before==after),indent=2))


if __name__=="__main__": main()
