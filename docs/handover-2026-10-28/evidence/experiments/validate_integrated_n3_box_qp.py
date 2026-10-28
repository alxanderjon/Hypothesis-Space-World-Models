#!/usr/bin/env python3
"""Validate the integrated uncapped, box-constrained deterministic N3."""
from __future__ import annotations

import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import test_n3_active_set_repair as active
from experiments import validate_n3_accepted_step_cap as cap
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
SOURCE = ROOT / "handoff_promotion_unresolved.jsonl"
ACTIVE_REPORT = ROOT / "active_set_n3_diagnostic" / "report.json"
ORACLE = ROOT / "dimension_general_box_qp_validation" / "aliases"
OLD_CAP_REPORT = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = ROOT / "integrated_box_qp_n3_validation"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def result_record(result):
    return {
        "accepted_steps": int(result.accepted_steps),
        "stop_reason": result.stop_reason,
        "theta_history": [np.asarray(value, np.float64).tolist()
                          for value in result.theta_history],
        "loss_history": [float(value) for value in result.loss_history],
        "final_theta": np.asarray(result.theta_history[-1], np.float64).tolist(),
        "final_loss": float(result.loss_history[-1]),
    }


def exact_comparison(result, expected):
    return {
        "theta_history_exact": bool(np.array_equal(
            np.asarray(result["theta_history"]),
            np.asarray(expected["theta_history"]))),
        "loss_history_exact": result["loss_history"] == expected["loss_history"],
        "accepted_steps_exact": result["accepted_steps"] == expected["accepted_steps"],
        "stop_reason_exact": result["stop_reason"] == expected["stop_reason"],
        "final_theta_exact": result["final_theta"] == expected["final_theta"],
        "final_loss_exact": result["final_loss"] == expected["final_loss"],
    }


def validate_aliases():
    ids = {row["example_id"] for row in
           json.loads(ACTIVE_REPORT.read_text())["aliases"]}
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()
            if json.loads(line)["example_id"] in ids]
    output = []
    for row in rows:
        torch_objective, _, _, lower, upper, _, _ = active.alias_context(row)
        _, _, run_n3 = staged._make_n3(torch_objective, lower, upper)
        endpoint = polish.N2Endpoint(
            0, np.asarray(row["label_n2_endpoint"], np.float64),
            float(row["label_n2_loss"]), "integrated_box_qp_validation",
            "kepler_alias", False)
        result = result_record(run_n3(endpoint, torch.empty(0)))
        expected = json.loads((ORACLE / f'{row["example_id"]}.json').read_text())[
            "active_set"]
        comparison = exact_comparison(result, expected)
        output.append({"case_id": row["case_id"], "state_id": row["state_id"],
                       "example_id": row["example_id"], "result": result,
                       "comparison": comparison})
        print("alias", row["case_id"], all(comparison.values()), flush=True)
    return output


def validate_existing_controls():
    old = json.loads(OLD_CAP_REPORT.read_text())
    expected = {(row["system"], str(row["case_id"])): row["cap_suppressed"]
                for row in old["cases"]}
    current = []
    for case_id in cap.KEPLER_OUTLIERS + (12, 27, 45):
        current.append(cap.kepler_case(case_id))
    for group in (cap.sho_cases(), cap.ball4_cases(), cap.ball5_cases()):
        current.extend(group)
    output = []
    for row in current:
        key = (row["system"], str(row["case_id"]))
        result = row["production"]
        comparison = exact_comparison(result, expected[key])
        output.append({"system": row["system"], "case_id": row["case_id"],
                       "result": result, "expected": expected[key],
                       "comparison": comparison,
                       "formerly_step20_limited": (
                           row["system"] == "kepler5" and
                           int(row["case_id"]) in cap.KEPLER_OUTLIERS)})
        print("control", row["system"], row["case_id"],
              all(comparison.values()), flush=True)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    aliases = validate_aliases()
    controls = validate_existing_controls()
    after = hashes()
    six = [row for row in controls if row["formerly_step20_limited"]]
    report = {
        "aliases": aliases, "controls": controls,
        "summary": {
            "aliases": len(aliases),
            "aliases_exact": sum(all(row["comparison"].values()) for row in aliases),
            "controls": len(controls),
            "controls_exact": sum(all(row["comparison"].values()) for row in controls),
            "formerly_step20_limited": len(six),
            "formerly_step20_limited_exact": sum(
                all(row["comparison"].values()) for row in six),
            "formerly_step20_limited_steps": {
                str(row["case_id"]): row["result"]["accepted_steps"] for row in six},
        },
        "checkpoint_hashes_unchanged": before == after,
    }
    atomic_json(OUT / "report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
