#!/usr/bin/env python3
"""Full 822-case branch replay for the integrated hybrid student.

The frozen on-policy N1/archive history supplies each actual successful branch
input.  The live student re-executes exact N2 and corrected N3 from that input.
This avoids regenerating the already-frozen global search while still testing
the complete downstream closed-loop composition on every validated trajectory.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments import canonical_teacher_regression_corpus as canonical_corpus
from experiments.resume_hybrid_student_after_ckdtree import Log, run_n2_with_best_ever_assembly
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.validate_complete_hybrid_student_v2 import (
    HybridN3, HybridObjective, bits_equal, metric,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_complete_validation_v2")
ROWS = OUT / "stage2_rows.jsonl"
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)


def old_archive_successes():
    raise canonical_corpus.ProvenanceError(
        "historical archive outputs are disabled as regression targets; "
        "use load_canonical_corpus()"
    )


def records():
    # This fail-closed loader is the only regression target source.  Missing,
    # stale, or provenance-mismatched historical artifacts cannot enter here.
    result = []
    for case in canonical_corpus.load_canonical_corpus():
        branch = canonical_corpus.load_case_branch(
            case, case["successful_handoff_artifact_hash"]
        )
        result.append({
            "cohort": "canonical_frozen_teacher",
            "system": case["case_identity"]["system"],
            "case_id": case["case_identity"]["canonical_case_id"],
            "case": case, "branch": branch,
        })
    return result


def load_case(record):
    if record.get("cohort") != "canonical_frozen_teacher":
        raise canonical_corpus.ProvenanceError(
            "noncanonical case input rejected"
        )
    case = record["case"]; branch = record["branch"]
    trajectory, _, _, _, lower, upper, teacher = canonical_corpus.case_data(
        case["case_identity"]
    )
    return (trajectory, lower, upper, teacher,
            np.asarray(branch["archive_theta"], np.float64),
            float(branch["archive_loss"]),
            np.asarray(branch["n2_box_lower"], np.float64),
            np.asarray(branch["n2_box_upper"], np.float64))


def normalize_expected(record):
    if record.get("cohort") != "canonical_frozen_teacher":
        raise canonical_corpus.ProvenanceError(
            "noncanonical expected output rejected"
        )
    branch = record["branch"]
    return {
        "n2_endpoint": branch["n2"]["endpoint"],
        "n2_loss": branch["n2"]["loss"],
        "n2_evaluations": branch["n2"]["evaluations"],
        "n2_stop_reason": branch["n2"]["stop_reason"],
        "final_theta": branch["n3"]["final_theta"],
        "final_loss": branch["n3"]["final_loss"],
        "n3_accepted_steps": branch["n3"]["accepted_steps"],
        "n3_stop_reason": branch["n3"]["stop_reason"],
        "continuation_max_error": branch["offline"]["continuation_max_error"],
    }


def worker(record):
    if record.get("cohort") != "canonical_frozen_teacher":
        raise canonical_corpus.ProvenanceError(
            "noncanonical regression target rejected"
        )
    torch.set_num_threads(1)
    started = time.monotonic()
    trajectory, lower, upper, teacher, start, start_loss, box_lower, box_upper = load_case(record)
    expected = normalize_expected(record)
    log = Log()
    objective = HybridObjective(record["system"], trajectory, log)
    n2 = run_n2_with_best_ever_assembly(
        objective, start, start_loss, box_lower, box_upper, log,
    )
    n2_exact = bool(
        int(n2["evaluations"]) == int(expected["n2_evaluations"])
        and n2["message"] == expected["n2_stop_reason"]
        and bits_equal(n2["theta"], expected["n2_endpoint"])
        and bits_equal([n2["loss"]], [expected["n2_loss"]])
    )
    row = {
        "cohort": record["cohort"], "system": record["system"],
        "case_id": record["case_id"], "n2_exact": n2_exact,
        "n2_evaluations_teacher": int(expected["n2_evaluations"]),
        "n2_evaluations_student": int(n2["evaluations"]),
        "n2_stop_teacher": expected["n2_stop_reason"],
        "n2_stop_student": n2["message"],
        "n2_theta_metric": metric(expected["n2_endpoint"], n2["theta"]),
        "n2_loss_metric": metric([expected["n2_loss"]], [n2["loss"]]),
    }
    if not n2_exact:
        row.update({"exact": False, "first_divergence": "N2", "elapsed_seconds": time.monotonic()-started})
        return row
    n3 = HybridN3(record["system"], trajectory, lower, upper, log).run(
        n2["theta"], expected, crosscheck_gradient=False,
    )
    final = np.asarray(n3["final_theta"], np.float64)
    rollout = readiness.rollout(record["system"], trajectory, final, teacher)
    rollout_teacher = expected.get("continuation_max_error", expected.get("rollout_max_error"))
    rollout_exact = (
        rollout_teacher is None or
        bits_equal([rollout["continuation_max_error"]], [rollout_teacher])
    )
    exact = bool(
        n3["first_divergence"] is None
        and n3["accepted_steps"] == n3["expected_steps"]
        and n3["stop_reason"] == n3["expected_stop"]
        and n3["theta_metric"]["bit_mismatches"] == 0
        and n3["loss_metric"]["bit_mismatches"] == 0
        and rollout_exact
    )
    row.update({
        "exact": exact,
        "n3_accepted_steps_teacher": n3["expected_steps"],
        "n3_accepted_steps_student": n3["accepted_steps"],
        "n3_stop_teacher": n3["expected_stop"],
        "n3_stop_student": n3["stop_reason"],
        "final_theta_metric": n3["theta_metric"],
        "final_loss_metric": n3["loss_metric"],
        "rollout_max_teacher": rollout_teacher,
        "rollout_max_student": rollout["continuation_max_error"],
        "rollout_exact": rollout_exact,
        "binding_invocations": dict(log.counts),
        "first_divergence": n3["first_divergence"],
        "elapsed_seconds": time.monotonic() - started,
    })
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_hashes:
        raise RuntimeError("checkpoint inventory differs before Stage 2")
    all_records = records()
    if len(all_records) != 822:
        raise RuntimeError(f"expected 822 records, found {len(all_records)}")
    existing = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            row = json.loads(line)
            existing[(row["cohort"], row["system"], str(row["case_id"]))] = row
    pending = [row for row in all_records
               if (row["cohort"], row["system"], str(row["case_id"])) not in existing]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        future_map = {pool.submit(worker, row): row for row in pending}
        for index, future in enumerate(as_completed(future_map), 1):
            row = future.result()
            key = (row["cohort"], row["system"], str(row["case_id"]))
            existing[key] = row
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if not row["exact"]:
                for item in future_map:
                    item.cancel()
                print(json.dumps({"status": "STOPPED_FIRST_DIVERGENCE", "row": row}, indent=2), flush=True)
                return
            if index % 10 == 0 or index == len(pending):
                counts = Counter(value["system"] for value in existing.values())
                print(json.dumps({
                    "completed": len(existing), "pending_initial": len(pending),
                    "by_system": dict(counts),
                    "elapsed_seconds": time.monotonic() - started,
                }), flush=True)
    rows = list(existing.values())
    by_system = {}
    invocations = Counter()
    for system in ("full_k_sho", "reflection_ball", "kepler5"):
        subset = [row for row in rows if row["system"] == system]
        by_system[system] = {
            "cases": len(subset), "exact": sum(row["exact"] for row in subset),
            "n2_exact": sum(row["n2_exact"] for row in subset),
            "final_parameter_bit_exact": sum(
                row.get("final_theta_metric", {}).get("bit_mismatches", 1) == 0
                for row in subset
            ),
            "final_loss_bit_exact": sum(
                row.get("final_loss_metric", {}).get("bit_mismatches", 1) == 0
                for row in subset
            ),
            "rollout_exact": sum(row.get("rollout_exact", False) for row in subset),
        }
        for row in subset:
            invocations.update(row.get("binding_invocations", {}))
    report = {
        "status": "STAGE2_EXACT" if all(row["exact"] for row in rows) else "STAGE2_DIVERGENCE",
        "cases": len(rows), "by_system": by_system,
        "binding_invocations": dict(invocations),
        "checkpoint_hashes_unchanged": checkpoint_hashes() == expected_hashes,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "stage2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
