#!/usr/bin/env python3
"""Offline archive-based absorbing handoff audit for saved Kepler N1 histories.

No N1 search is rerun.  For a saved current count n, every earlier evaluated
site is a possible handoff center.  Its production N2 box is formed using n,
then the unchanged N2/N3 pipeline is evaluated.  Results are cached per
(case, n, archive index), making the audit interruption-safe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/kepler_archive_absorbing_handoff")
CACHE = OUT / "checks"
CASES = OUT / "cases"
BASE = Path("outputs/d5_handoff_characterization/kepler_cases")
CONT = Path("outputs/d5_handoff_continuation_20000/cases")
PERSISTENT = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}
STAGES = (20, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_path(case_id: int) -> Path:
    continued = CONT / f"case_{case_id:03d}" / "n1_state.npz"
    base = BASE / f"case_{case_id:03d}" / "n1_state.npz"
    return continued if continued.exists() else base


def previous_incumbent_handoff(case_id: int) -> dict | None:
    continued = CONT / f"case_{case_id:03d}" / "result.json"
    base = BASE / f"case_{case_id:03d}" / "result.json"
    values = []
    for path in (base, continued):
        if path.exists():
            row = json.loads(path.read_text())
            hit = row.get("first_successful_handoff")
            if hit is not None:
                values.append(hit)
    def evaluation(row):
        return int(row["evaluation"] if "evaluation" in row else row["n1_evaluations"])
    return min(values, key=evaluation) if values else None


class CaseAudit:
    def __init__(self, case_id: int):
        self.case_id = int(case_id)
        self.trajectory, self.initial_condition = production._kepler_case(case_id)
        self.objective, self.torch_objective = production._objectives(
            "kepler5", self.trajectory)
        self.lower, self.upper, self.teacher = production._config("kepler5")
        self.width = self.upper - self.lower
        self.path = history_path(case_id)
        saved = np.load(self.path)
        self.points = np.asarray(saved["points"], np.float64)
        self.losses = np.asarray(saved["losses"], np.float64)
        self.thetas = self.lower + self.width * self.points
        if self.points.shape != (len(self.losses), len(self.lower)):
            raise RuntimeError(f"case {case_id}: malformed saved history")
        self.checks_run = 0
        self.checks_reused = 0

    def endpoint(self, n: int, index: int) -> dict:
        theta = self.thetas[index]
        radius = self.width / float(n) ** (1.0 / len(theta))
        return dict(theta=theta, loss=float(self.losses[index]),
                    evaluations=int(n), lower=np.maximum(self.lower, theta-radius),
                    upper=np.minimum(self.upper, theta+radius),
                    stop_reason="archive_absorbing_handoff_audit")

    def cache_path(self, n: int, index: int) -> Path:
        return CACHE / f"case_{self.case_id:03d}" / f"n{n:05d}_i{index+1:05d}.json"

    def check(self, n: int, index: int) -> dict:
        path = self.cache_path(n, index)
        if path.exists():
            self.checks_reused += 1
            return json.loads(path.read_text())
        started = time.monotonic()
        endpoint = self.endpoint(n, index)
        finish = production._finish(
            "kepler5", self.trajectory, self.objective, self.torch_objective,
            endpoint, self.teacher,
            f"archive_handoff_case_{self.case_id:03d}_n{n}_i{index+1}")
        # Preserve both repository outcomes explicitly.  `recovered` is the
        # existing final_loss < 1e-20 N3 criterion.  `full_pipeline_pass` adds
        # the existing finite-rollout requirement used by current validation.
        row = dict(
            case_id=self.case_id, current_evaluation=int(n),
            archive_index=int(index), insertion_evaluation=int(index+1),
            theta=self.thetas[index].tolist(), archived_loss=float(self.losses[index]),
            final_loss_gate=bool(finish["recovered"]),
            teacher_equivalent_recovery=bool(finish["full_pipeline_pass"]),
            elapsed_seconds=float(time.monotonic()-started), **finish)
        atomic_json(path, row)
        self.checks_run += 1
        return row

    def checkpoint(self, n: int) -> dict | None:
        # Canonical, no-future-information order: archive insertion order.
        for index in range(min(n, len(self.losses))):
            row = self.check(n, index)
            if row["teacher_equivalent_recovery"]:
                return row
        return None


def search_case(case_id: int) -> dict:
    result_path = CASES / f"case_{case_id:03d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    audit = CaseAudit(case_id)
    started = time.monotonic()
    length = len(audit.losses)
    # The count-derived N2 box can make viability non-monotone.  Audit every
    # count through the 20-point initial design before staged checkpoints so a
    # transient early success cannot disappear unnoticed at n=20.
    early_hit = None
    early_results = []
    for n in range(1, min(20, length) + 1):
        candidate = audit.checkpoint(n)
        early_results.append(dict(evaluation=n, success=candidate is not None,
                                  insertion_evaluation=(None if candidate is None else
                                                        candidate["insertion_evaluation"])))
        if candidate is not None:
            early_hit = candidate
            break
    stages = [n for n in STAGES if 20 < n <= length]
    if not stages or stages[-1] != length:
        stages.append(length)
    stage_results = early_results
    previous = min(20, length)
    hit = early_hit
    hit_stage = None
    if hit is None:
        for n in stages:
            candidate = audit.checkpoint(n)
            stage_results.append(dict(evaluation=n, success=candidate is not None,
                                      insertion_evaluation=(None if candidate is None else
                                                            candidate["insertion_evaluation"])))
            if candidate is not None:
                hit = candidate; hit_stage = n
                break
            previous = n

    observed_resolution = None
    if hit is not None:
        if early_hit is not None:
            observed_resolution = 1
        # Refine within the first successful staged interval.  Early states are
        # exact; beyond 100 evaluations use the established 100-evaluation
        # handoff diagnostic spacing to keep unchanged N2/N3 calls tractable.
        if hit_stage is None:
            counts = []
        elif hit_stage <= 100:
            counts = range(max(1, previous+1), hit_stage+1)
            observed_resolution = 1
        else:
            first = previous + (100 - previous % 100 if previous % 100 else 100)
            counts = list(range(first, hit_stage+1, 100))
            if hit_stage not in counts:
                counts.append(hit_stage)
            observed_resolution = 100
        for n in counts:
            candidate = audit.checkpoint(n)
            if candidate is not None:
                hit = candidate
                break

    previous_handoff = previous_incumbent_handoff(case_id)
    result = dict(
        case_id=case_id, initial_condition=audit.initial_condition,
        history_path=str(audit.path), history_sha256=file_hash(audit.path),
        saved_evaluations=length, staged_checks=stage_results,
        earliest_observed_resolution=observed_resolution,
        cache_checks_computed=audit.checks_run,
        cache_checks_reused=audit.checks_reused,
        previous_incumbent_handoff=previous_handoff,
        successful_archived_handoff=None,
        elapsed_seconds=float(time.monotonic()-started),
    )
    if hit is not None:
        n = int(hit["current_evaluation"]); index = int(hit["archive_index"])
        order = np.lexsort((np.arange(n), audit.losses[:n]))
        rank = int(np.flatnonzero(order == index)[0]) + 1
        incumbent_index = int(np.argmin(audit.losses[:n]))
        incumbent_loss = float(audit.losses[incumbent_index])
        insertion_check = audit.check(index+1, index)
        old_n = (None if previous_handoff is None else int(
            previous_handoff["evaluation"] if "evaluation" in previous_handoff
            else previous_handoff["n1_evaluations"]))
        result["successful_archived_handoff"] = dict(
            **hit,
            is_current_incumbent=bool(index == incumbent_index),
            current_incumbent_index=incumbent_index,
            current_incumbent_theta=audit.thetas[incumbent_index].tolist(),
            current_incumbent_loss=incumbent_loss,
            archived_loss_rank=rank,
            archived_loss_minus_incumbent=float(hit["archived_loss"]-incumbent_loss),
            archived_loss_over_incumbent=(float(hit["archived_loss"] / incumbent_loss)
                                           if incumbent_loss != 0.0 else None),
            archive_age_evaluations=int(n-(index+1)),
            viable_at_insertion_count=bool(
                insertion_check["teacher_equivalent_recovery"]),
            became_viable_later_due_to_current_box_scale=bool(
                not insertion_check["teacher_equivalent_recovery"] and n > index+1),
            prior_incumbent_handoff_evaluation=old_n,
            archive_handoff_lead_vs_incumbent=(None if old_n is None else old_n-n),
            insertion_lead_vs_incumbent=(None if old_n is None else old_n-(index+1)),
        )
    atomic_json(result_path, result)
    print(f"case {case_id:02d}: " + ("NO HIT" if hit is None else
          f"n={hit['current_evaluation']} insert={hit['insertion_evaluation']} "
          f"loss={hit['archived_loss']:.3e}"), flush=True)
    return result


def summarize(rows: list[dict], before_hashes: dict) -> dict:
    hits = [row for row in rows if row["successful_archived_handoff"] is not None]
    nonincumbent = [row for row in hits
                    if not row["successful_archived_handoff"]["is_current_incumbent"]]
    later = [row for row in hits if row["successful_archived_handoff"]
             ["became_viable_later_due_to_current_box_scale"]]
    persistent_rows = [row for row in rows if row["case_id"] in PERSISTENT]
    persistent_hits = [row for row in persistent_rows
                       if row["successful_archived_handoff"] is not None]
    handoff_counts = [row["successful_archived_handoff"]["current_evaluation"]
                      for row in hits]
    ages = [row["successful_archived_handoff"]["archive_age_evaluations"]
            for row in hits]
    leads = [row["successful_archived_handoff"]["archive_handoff_lead_vs_incumbent"]
             for row in hits if row["successful_archived_handoff"]
             ["archive_handoff_lead_vs_incumbent"] is not None]
    def dist(values):
        if not values:
            return dict(count=0)
        a = np.asarray(values, np.float64)
        return dict(count=len(values), minimum=float(a.min()), median=float(np.median(a)),
                    p90=float(np.quantile(a, .9)), maximum=float(a.max()))
    after_hashes = hashes()
    summary = dict(
        label_definition=dict(
            final_loss_gate="existing N3 recovered flag: final_loss < 1e-20",
            teacher_equivalent_recovery=(
                "existing full_pipeline_pass: N3 recovered and continuation rollout finite"),
            raw_parameter_and_rollout_errors_retained=True),
        cases=len(rows), archive_recoveries=len(hits),
        cases_without_evaluated_successful_handoff=len(rows)-len(hits),
        successful_nonincumbent_handoffs=len(nonincumbent),
        became_viable_only_at_later_box_scale=len(later),
        persistent_cases=len(persistent_rows),
        persistent_cases_repaired=len(persistent_hits),
        persistent_cases_unrepaired=[row["case_id"] for row in persistent_rows
                                     if row["successful_archived_handoff"] is None],
        evaluation_to_archive_handoff=dist(handoff_counts),
        successful_point_archive_age=dist(ages),
        archive_handoff_lead_vs_prior_incumbent=dist(leads),
        total_checks_computed=sum(row["cache_checks_computed"] for row in rows),
        total_checks_reused=sum(row["cache_checks_reused"] for row in rows),
        cached_unique_handoff_checks=sum(1 for _ in CACHE.rglob("*.json")),
        before_checkpoint_hashes=before_hashes,
        after_checkpoint_hashes=after_hashes,
        checkpoint_hashes_unchanged=before_hashes == after_hashes,
    )
    atomic_json(OUT / "summary.json", summary)
    fields = ["case_id", "saved_evaluations", "handoff_evaluation",
              "insertion_evaluation", "is_incumbent", "loss_rank", "archived_loss",
              "incumbent_loss", "archive_age", "viable_at_insertion",
              "became_viable_later", "n2_evaluations", "n3_steps", "final_loss",
              "final_parameter_l2_error", "continuation_max_error"]
    with (OUT / "case_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            hit = row["successful_archived_handoff"]
            writer.writerow(dict(case_id=row["case_id"],
                saved_evaluations=row["saved_evaluations"],
                handoff_evaluation=None if hit is None else hit["current_evaluation"],
                insertion_evaluation=None if hit is None else hit["insertion_evaluation"],
                is_incumbent=None if hit is None else hit["is_current_incumbent"],
                loss_rank=None if hit is None else hit["archived_loss_rank"],
                archived_loss=None if hit is None else hit["archived_loss"],
                incumbent_loss=None if hit is None else hit["current_incumbent_loss"],
                archive_age=None if hit is None else hit["archive_age_evaluations"],
                viable_at_insertion=None if hit is None else hit["viable_at_insertion_count"],
                became_viable_later=None if hit is None else hit["became_viable_later_due_to_current_box_scale"],
                n2_evaluations=None if hit is None else hit["n2_evaluations"],
                n3_steps=None if hit is None else hit["n3_accepted_steps"],
                final_loss=None if hit is None else hit["final_loss"],
                final_parameter_l2_error=None if hit is None else hit["final_parameter_l2_error"],
                continuation_max_error=None if hit is None else hit["continuation_max_error"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case is not None:
        search_case(args.case)
        return
    before = hashes()
    rows = [search_case(case_id) for case_id in range(64)]
    summary = summarize(rows, before)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
