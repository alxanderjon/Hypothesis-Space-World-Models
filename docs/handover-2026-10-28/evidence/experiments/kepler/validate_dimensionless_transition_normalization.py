#!/usr/bin/env python3
"""Validate one shared transition-RMS normalization; stop on preservation failure."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.kepler import deterministic_teacher as kepler
from experiments.kepler import run_first_transfer as transfer
from experiments.sho import run_stage1b_fifo_vs_ranker as sho


OUT = Path("outputs/kepler/dimensionless_transition_normalization")


def transition_scale_squared(prefix: np.ndarray) -> float:
    """Pooled RMS one-step variation from observed states only."""
    observed = np.asarray(prefix, dtype=np.float64)
    differences = np.diff(observed, axis=0)
    scale_squared = float(np.mean(np.square(differences)))
    if scale_squared == 0.0:
        # Generic degenerate-prefix fallback; unused in all validated pools.
        scale_squared = float(np.mean(np.square(observed)))
    if not scale_squared > 0.0 or not np.isfinite(scale_squared):
        raise ValueError("Observed prefix has no finite nonzero scale")
    return scale_squared


def kepler_initial_screen() -> list[dict]:
    normalized = sho.edirect._initial_design()
    candidates = kepler.LOWER + normalized * (kepler.UPPER-kepler.LOWER)
    rows = []
    for case_id, spec in enumerate(transfer.FRESH_CASES):
        e, periapsis, orientation, phase = spec
        initial = kepler.initial_state(e, periapsis, orientation)
        phased = kepler.numpy_rollout(initial, steps=phase)[-1]
        trajectory = kepler.numpy_rollout(phased)
        raw = kepler.prefix_loss_numpy(candidates, trajectory)
        scale2 = transition_scale_squared(trajectory[:kepler.PREFIX_STATES])
        dimensionless = raw / scale2
        best = int(np.argmin(dimensionless))
        rows.append(dict(case_id=case_id, eccentricity=e, periapsis=periapsis,
                         orientation=orientation, phase_steps=phase,
                         scale_squared=scale2, raw_best=float(raw.min()),
                         dimensionless_best=float(dimensionless[best]),
                         best_gamma=float(candidates[best,0]), best_p=float(candidates[best,1]),
                         crossed_gate=int(dimensionless[best] <= sho.GATE),
                         ordering_equal=int(np.array_equal(np.argsort(raw,kind="stable"),
                                                           np.argsort(dimensionless,kind="stable")))))
    return rows


def sho_archived_trace_screen() -> dict:
    datasets, _ = sho.data_and_plan()
    source = Path("outputs/sho/stage1B/fifo_vs_ranker_closedloop/fifo_vs_ranker_closedloop.csv")
    with source.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    summaries = [row for row in all_rows if row["record_type"] == "trajectory" and row["arm"] == "B"]
    rows = []
    for summary in summaries:
        split, trajectory_id = summary["split"], int(summary["trajectory_id"])
        position = datasets[split]["p"][trajectory_id]
        scale2 = transition_scale_squared(position[:sho.edirect.P])
        events = [row for row in all_rows
                  if row["record_type"] == "global_evaluation" and row["arm"] == "B"
                  and row["split"] == split and int(row["trajectory_id"]) == trajectory_id]
        hits = [int(row["evaluation"]) for row in events
                if float(row["best_loss"]) / scale2 <= sho.GATE]
        rows.append(dict(split=split, trajectory_id=trajectory_id, scale_squared=scale2,
                         previous_handoff_evaluation=int(summary["evaluations_to_handoff"]),
                         normalized_handoff_on_same_trace=hits[0] if hits else None,
                         normalized_last_loss=float(events[-1]["best_loss"])/scale2,
                         preserved=int(bool(hits))))
    return dict(cases=len(rows), previous_handoffs=len(rows),
                normalized_handoffs_on_same_evaluated_traces=sum(row["preserved"] for row in rows),
                candidate_ordering="exact: each trajectory loss is divided by one positive scalar",
                live_replay_run=False,
                live_replay_reason="archived successful traces already fail the unchanged handoff gate",
                rows=rows)


def ball_screen() -> dict:
    raw_numpy = ball.prefix_loss_numpy

    def normalized_numpy(theta: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        return raw_numpy(theta, trajectory) / transition_scale_squared(
            trajectory[:ball.PREFIX_STATES])

    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=8, n_timesteps=ball.TOTAL_STEPS, dt=ball.DT, seed=20_261_001)
    with Path("outputs/bouncing_ball/second_broad_n1_n2_n3_transfer/per_trajectory.csv").open(newline="") as handle:
        previous = list(csv.DictReader(handle))
    frozen = ball.n1_current.FrozenProposals()
    try:
        ball.prefix_loss_numpy = normalized_numpy
        n1_results = [ball.run_n1(index, trajectory, frozen)
                      for index, trajectory in enumerate(trajectories)]
        n2_results = [ball.run_n2(index, trajectory, result)
                      for index, (trajectory, result) in enumerate(zip(trajectories, n1_results))]
    finally:
        ball.prefix_loss_numpy = raw_numpy
    rows = []
    for index, (old, one, two, trajectory) in enumerate(zip(previous, n1_results, n2_results, trajectories)):
        rows.append(dict(case_id=index,
            scale_squared=transition_scale_squared(trajectory[:ball.PREFIX_STATES]),
            previous_n1_evaluations=int(old["n1_evaluations"]),
            normalized_n1_evaluations=one.evaluations,
            previous_n1_loss=float(old["n1_loss"]), normalized_n1_loss=one.loss,
            previous_n2_success=int(old["n2_success"]), normalized_n2_success=int(two.success),
            normalized_n2_evaluations=two.evaluations,
            normalized_n2_loss=two.loss, normalized_n2_stop=two.stop_reason,
            normalized_n1_gamma=float(one.theta[0]), normalized_n1_p=float(one.theta[1]),
            learned_n1_decisions=one.learned_ranker_decisions))
    return dict(cases=len(rows), previous_n1_successes=sum(int(row["n1_success"]) for row in previous),
                normalized_n1_successes=sum(result.success for result in n1_results),
                previous_n2_successes=sum(int(row["n2_success"]) for row in previous),
                normalized_n2_successes=sum(result.success for result in n2_results),
                failed_cases=[row["case_id"] for row in rows if not row["normalized_n2_success"]],
                first_failure="N1_LOCAL_BOX_EXCLUDES_TEACHER" if any(
                    row["normalized_n2_stop"] == "N1_LOCAL_BOX_EXCLUDES_TEACHER" for row in rows) else "",
                candidate_ordering="exact for every trajectory by positive scalar division", rows=rows)


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    paths = transfer.checkpoint_paths(); before = transfer.manifest(paths)
    sho_result = sho_archived_trace_screen()
    ball_result = ball_screen()
    kepler_result = kepler_initial_screen()
    preservation = (sho_result["normalized_handoffs_on_same_evaluated_traces"] == sho_result["cases"]
                    and ball_result["normalized_n1_successes"] == ball_result["cases"]
                    and ball_result["normalized_n2_successes"] == ball_result["cases"])
    after = transfer.manifest(paths)
    report = dict(
        status="STOPPED_CROSS_SYSTEM_PRESERVATION_FAILURE" if not preservation else "PASSED",
        normalization=dict(
            formula="L_dimensionless = mean(residual^2) / mean((prefix[t+1]-prefix[t])^2)",
            scale_scope="one pooled scalar over all observed transitions and state components",
            observed_prefix_only=True, teacher_parameters_used=False, future_data_used=False,
            system_specific_constants_used=False,
            degenerate_fallback="prefix RMS only if transition RMS is exactly zero",
            ranking_property="one positive scalar per prefix preserves all candidate loss ordering exactly"),
        sho=sho_result, bouncing_ball=ball_result,
        kepler_initial_design=dict(cases=len(kepler_result),
            cases_crossing_gate=sum(row["crossed_gate"] for row in kepler_result), rows=kepler_result),
        preservation_passed=preservation,
        kepler_full_transfer_run=False,
        stop_reason=("SHO archived handoff behavior and bouncing-ball N2 locality regressed; "
                     "no alternative normalization was explored." if not preservation else ""),
        checkpoint_sha256_before=before, checkpoint_sha256_after=after,
        checkpoints_unchanged=before == after, elapsed_seconds=time.time()-started)
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
