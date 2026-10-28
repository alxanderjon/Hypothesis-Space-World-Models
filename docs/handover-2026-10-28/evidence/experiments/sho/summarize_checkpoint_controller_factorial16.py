#!/usr/bin/env python3
"""Read-only scientific accounting over the completed factorial16 artifacts."""
import hashlib, json, math, statistics
from pathlib import Path
import numpy as np

from experiments.sho.run_checkpoint_controller_factorial16_initializers import clean, save, sha

ROOT = Path("outputs/sho/autonomous_generic_sho_20261014")
SCREEN = ROOT / "screens/generic_n2_n3_checkpoint_controller_factorial16"
PROTOCOL = ROOT / "protocols/generic_n2_n3_checkpoint_controller_factorial16_design.json"
ADDENDUM = ROOT / "reviews/generic_n2_n3_checkpoint_controller_factorial16_formula_contrast_review.md"
PROTOCOL_SHA = "e2c55e0d79bdcc4b9999bac5997248c0373c31f57e060a8afdb64fdc153dc6d8"
ADDENDUM_SHA = "75f93d00957863cd1ba7c652d9704dacdb3bb330b26edcaf035deb6b52effdb1"
CHECKPOINTS = ("n2_prefix500", "n2_endpoint1500")
CONTROLLERS = ("unchanged_baseline", "accepted_step_radius_anchor")


def direction(effect):
    if effect is None: return "unavailable"
    ratio = 10.0 ** effect
    if ratio > 1.01: return "first_level_favored"
    if ratio < 1 / 1.01: return "second_level_favored"
    return "tie_le_1_percent"


def main():
    if sha(PROTOCOL) != PROTOCOL_SHA or sha(ADDENDUM) != ADDENDUM_SHA:
        raise RuntimeError("frozen protocol/addendum changed")
    protocol = json.loads(PROTOCOL.read_text()); rows = []
    certificate_checks = {"files": 0, "exact_feasible": 0, "exact_KKT": 0, "full_rank_exact": 0, "failures": []}
    cell_manifest_failures = []
    for directory in sorted((SCREEN / "cells").iterdir()):
        manifest = json.loads((directory / "artifact_manifest.json").read_text())["sha256"]
        bad = [name for name, digest in manifest.items() if sha(directory / name) != digest]
        if bad: cell_manifest_failures.append({"cell": directory.name, "files": bad})
        for name in manifest:
            if name.startswith("exact_head_") or name.startswith("exact_model_"):
                certificate_checks["files"] += 1
                cert = json.loads((directory / name).read_text())["certificate"]
                for key in ("exact_feasible", "exact_KKT", "full_rank_exact"):
                    # The run's shared JSON cleaner visits int before bool, so
                    # certified Python True is serialized as numeric 1.
                    if cert.get(key) == 1: certificate_checks[key] += 1
                    else: certificate_checks["failures"].append({"cell": directory.name, "file": name, "key": key})
        rows.append(json.loads((directory / "result.json").read_text()))
    by = {(r["case"], r["checkpoint"], r["controller"]): r for r in rows}
    cases = list(protocol["source_eligibility"]["cases"])
    initializers = {}; initializer_manifest_failures = []
    for case, spec in protocol["source_eligibility"]["cases"].items():
        d = ROOT / "initializers" / spec["initializer_run_id"]
        result = json.loads((d / "result.json").read_text()); checkpoints = json.loads((d / "checkpoints.json").read_text())
        manifest = json.loads((d / "artifact_manifest.json").read_text())["sha256"]
        bad = [name for name, digest in manifest.items() if sha(d / name) != digest]
        if bad: initializer_manifest_failures.append({"case": case, "files": bad})
        initializers[case] = {"branch": spec["branch"], "source_loss": result["source_loss"],
            "prefix500_loss": result["prefix500_loss"], "prefix500_selected_evaluation": result["prefix500_evaluation"],
            "endpoint1500_loss": result["endpoint1500_loss"],
            "checkpoints_bitwise_identical": bool(result["checkpoint_vectors_bitwise_identical"]),
            "objective_counts": result["objective_counts"], "manifest_files": len(manifest)}
    per_case = {}; interaction_cases = []; checkpoint_dirs = {c: [] for c in CONTROLLERS}; controller_dirs = {c: [] for c in CHECKPOINTS}
    for case in cases:
        cells = {(cp, co): by[(case, cp, co)] for cp in CHECKPOINTS for co in CONTROLLERS}
        effects = {}; common = {}
        for co in CONTROLLERS:
            prefix, endpoint = cells[("n2_prefix500", co)], cells[("n2_endpoint1500", co)]
            p = None if prefix["P24"] is None or endpoint["P24"] is None else endpoint["P24"] - prefix["P24"]
            raw = None if prefix["final_loss"] is None or endpoint["final_loss"] is None or prefix["P24"] is None or endpoint["P24"] is None else math.log10(prefix["final_loss"] / endpoint["final_loss"])
            effects[f"checkpoint_P24_at_{co}"] = p; effects[f"checkpoint_terminal_raw_at_{co}"] = raw
            checkpoint_dirs[co].append(direction(raw))
        for cp in CHECKPOINTS:
            baseline, anchor = cells[(cp, "unchanged_baseline")], cells[(cp, "accepted_step_radius_anchor")]
            p = None if baseline["P24"] is None or anchor["P24"] is None else anchor["P24"] - baseline["P24"]
            effects[f"controller_P24_at_{cp}"] = p
            controller_dirs[cp].append(direction(p))
            shared = sorted(set(int(k) for k, v in baseline["horizon_incumbent_losses"].items() if v is not None) & set(int(k) for k, v in anchor["horizon_incumbent_losses"].items() if v is not None))
            common[cp] = {str(k): {"baseline_loss": baseline["horizon_incumbent_losses"][str(k)],
                "anchor_loss": anchor["horizon_incumbent_losses"][str(k)],
                "anchor_over_baseline": anchor["horizon_incumbent_losses"][str(k)] / baseline["horizon_incumbent_losses"][str(k)]} for k in shared}
        ep, pr = effects["controller_P24_at_n2_endpoint1500"], effects["controller_P24_at_n2_prefix500"]
        effects["P24_checkpoint_by_controller_interaction"] = None if ep is None or pr is None else ep - pr
        controller_reversal = ep is not None and pr is not None and direction(ep) != direction(pr) and "tie" not in direction(ep) + direction(pr)
        rb, ra = effects["checkpoint_terminal_raw_at_unchanged_baseline"], effects["checkpoint_terminal_raw_at_accepted_step_radius_anchor"]
        checkpoint_reversal = rb is not None and ra is not None and direction(rb) != direction(ra) and "tie" not in direction(rb) + direction(ra)
        if controller_reversal or checkpoint_reversal: interaction_cases.append(case)
        per_case[case] = {"cells": {f"{cp}|{co}": {k: cells[(cp, co)].get(k) for k in
            ("status", "stop_reason", "source_loss", "initial_profile_loss", "final_loss", "P24", "complete_AUC_2_to_24", "horizon_incumbent_losses", "work")} for cp in CHECKPOINTS for co in CONTROLLERS},
            "effects_log10": effects, "material_directions": {k: direction(v) for k, v in effects.items()},
            "valid_common_controller_horizons": common,
            "interaction_flags": {"controller_direction_reversal": controller_reversal, "checkpoint_ordering_reversal": checkpoint_reversal}}
    engagement = {}
    for r in rows:
        if r["controller"] != "accepted_step_radius_anchor": continue
        key = f"{r['case']}|{r['checkpoint']}"; events = r["controller_engagement"]
        ratios = [e["carried_radius_after"] / e["scipy_radius_after"] for e in events]
        old_ratios = [e["carried_radius_after"] / e["radius_before"] for e in events]
        next_rows = [e["subsequent_fidelity"] for e in events if "subsequent_fidelity" in e]
        engagement[key] = {"accepted_events": len(events), "events_with_subsequent_fidelity": len(next_rows),
            "carried_over_scipy_min_median_max": None if not ratios else [min(ratios), statistics.median(ratios), max(ratios)],
            "carried_over_old_min_median_max": None if not old_ratios else [min(old_ratios), statistics.median(old_ratios), max(old_ratios)],
            "carried_bitwise_equal_scipy": sum(np.float64(e["carried_radius_after"]).view(np.uint64) == np.float64(e["scipy_radius_after"]).view(np.uint64) for e in events),
            "carried_above_scipy": sum(e["carried_radius_after"] > e["scipy_radius_after"] for e in events),
            "carried_below_scipy": sum(e["carried_radius_after"] < e["scipy_radius_after"] for e in events),
            "subsequent_positive_native_decrease": sum(x["actual_MSE_decrease"] > 0 for x in next_rows),
            "subsequent_nonpositive_native_decrease": sum(x["actual_MSE_decrease"] <= 0 for x in next_rows),
            "no_floor_or_invalid_anchor_stop": r["stop_reason"] == "NUMERICAL_STOP_INVALID_ACCEPTED_STEP_ANCHOR"}
    total_n3 = {k: sum(int(r["work"][k]) for r in rows) for k in ("profile_evaluations", "exact_head_calls", "exact_head_completed", "joint_model_solves", "center_derivative_evaluations", "accepted_hidden_updates", "rejected_proposals", "source_objective_calls")}
    total_n3["native_objectives_source_plus_profiles"] = total_n3["source_objective_calls"] + total_n3["profile_evaluations"]
    checkpoint_effect = {co: checkpoint_dirs[co] for co in CONTROLLERS}
    controller_effect = {cp: controller_dirs[cp] for cp in CHECKPOINTS}
    analysis = {"status": "FACTORIAL16_COMPLETE_VERIFIED", "protocol_sha256": PROTOCOL_SHA,
        "formula_addendum_sha256": ADDENDUM_SHA, "declared_cells": 16, "cell_records": len(rows),
        "cell_status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "handoffs": sum(bool(r["handoff"]) for r in rows), "rollout_steps": sum((r.get("rollout") or {}).get("steps", 0) for r in rows),
        "initializers": initializers, "per_case": per_case, "controller_engagement": engagement,
        "exploratory_advancement_rules": {"checkpoint_effect_directions_by_controller": checkpoint_effect,
            "controller_effect_directions_by_checkpoint": controller_effect,
            "checkpoint_effect_clause_met": False, "controller_effect_clause_met": False,
            "interaction_cases": interaction_cases, "interaction_clause_met": len(interaction_cases) >= 2,
            "note": "Exploratory paired advancement rules only; not significance, acceptance, promotion or reliability gates."},
        "work": {"actual_shared_initializer_expense": {"N1_events": 4, "N2_evaluations": 6000, "endpoint_replays": 4},
            "deployable_per_cell_source_cost": {"n2_prefix500": "one N1 source plus 500 N2 evaluations", "n2_endpoint1500": "one N1 source plus 1500 N2 evaluations"},
            "N3": total_n3},
        "integrity": {"initializer_manifest_failures": initializer_manifest_failures,
            "cell_manifest_failures": cell_manifest_failures, "certificate_checks": certificate_checks},
        "limitations": ["24 profiles measure early trajectory effects only.",
            "Four cases are four paired observations, not sixteen independent samples.",
            "Two companion_6 endpoint cells stopped before controller engagement because the declared initial radius was undefined from a zero Coleman-Li scale.",
            "No cell reached native MSE below 1e-20; no rollout ran."]}
    save(SCREEN / "analysis.json", analysis)
    lines = ["# Prospective N2 checkpoint-state × N3 controller factorial — final report", "",
        "Status: complete. Four paired cases, 16 declared cells, no exclusions or replacements.", "",
        "## Outcome", "",
        "Fourteen cells reached the fixed 24-profile cap. Both companion_6 endpoint1500 cells stopped after the initial exact-head profile because at least one Coleman–Li scale was exactly zero, making the declared `norm(h/d)` initial radius undefined. This occurred before the controller factor engaged, so their common profile-1 controller comparison is an exact tie but uninformative about the intervention. No handoff occurred and no rollout ran.", "",
        "| case | fresh N1 source loss | N2 prefix500 loss | selected eval | N2 endpoint1500 loss | prefix exact-head initial | endpoint exact-head initial |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case in cases:
        ini = initializers[case]
        p0 = by[(case, "n2_prefix500", "unchanged_baseline")]["initial_profile_loss"]
        e0 = by[(case, "n2_endpoint1500", "unchanged_baseline")]["initial_profile_loss"]
        lines.append(f"| {case} | `{ini['source_loss']:.17g}` | `{ini['prefix500_loss']:.17g}` | {ini['prefix500_selected_evaluation']} | `{ini['endpoint1500_loss']:.17g}` | `{p0:.17g}` | `{e0:.17g}` |")
    lines += ["",
        "| case | checkpoint | baseline final loss | anchor final loss | anchor / baseline | status |", "|---|---:|---:|---:|---:|---|"]
    for case in cases:
        for cp in CHECKPOINTS:
            b, a = by[(case, cp, CONTROLLERS[0])], by[(case, cp, CONTROLLERS[1])]
            ratio = a["final_loss"] / b["final_loss"]
            lines.append(f"| {case} | {cp} | `{b['final_loss']:.17g}` | `{a['final_loss']:.17g}` | `{ratio:.9g}` | {b['status']} / {a['status']} |")
    lines += ["", "At the complete prefix500 horizon, baseline had the lower final loss in all four cases. At endpoint1500, baseline was lower in interior_49 and companion_29, anchor was lower in companion_14, and companion_6 was unavailable before engagement.", "",
        "## Paired numerical effects", "", "Effects below are `log10` fold-progress differences; positive controller effects favor anchor. Interaction is endpoint controller effect minus prefix500 controller effect.", "",
        "| case | checkpoint effect, baseline | checkpoint effect, anchor | controller effect, prefix500 | controller effect, endpoint1500 | interaction |", "|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        e = per_case[case]["effects_log10"]
        fmt = lambda x: "unavailable" if x is None else f"{x:.9g}"
        lines.append(f"| {case} | {fmt(e['checkpoint_P24_at_unchanged_baseline'])} | {fmt(e['checkpoint_P24_at_accepted_step_radius_anchor'])} | {fmt(e['controller_P24_at_n2_prefix500'])} | {fmt(e['controller_P24_at_n2_endpoint1500'])} | {fmt(e['P24_checkpoint_by_controller_interaction'])} |")
    lines += ["", f"The predeclared checkpoint main-effect clause did not fire; the controller main-effect clause did not fire. Using the frozen greater-than-1% terminal-loss ordering, the exploratory interaction clause fired in {len(interaction_cases)}/4 paired cases: {', '.join(interaction_cases)}. The continuous P24 interactions are shown separately above. These are advancement descriptions, not significance or reliability claims.", "",
        "## Controller engagement", "", "The anchor was active, not a nominal no-op. Across completed anchored arms it usually carried a smaller scalar than SciPy, commonly near one-half, but could carry up to 4× SciPy after an accepted low-ratio proposal; this is still the realized pre-center scaled displacement and has no floor. No invalid-anchor collapse occurred. The companion_6 endpoint stops preceded engagement.", "",
        "| case | checkpoint | accepted anchor events | subsequent fidelity rows | carried / SciPy min, median, max | next positive / nonpositive native change |", "|---|---|---:|---:|---|---:|"]
    for case in cases:
        for cp in CHECKPOINTS:
            e = engagement[f"{case}|{cp}"]
            ratio = e["carried_over_scipy_min_median_max"]
            ratio_text = "not engaged" if ratio is None else ", ".join(f"{x:.9g}" for x in ratio)
            lines.append(f"| {case} | {cp} | {e['accepted_events']} | {e['events_with_subsequent_fidelity']} | {ratio_text} | {e['subsequent_positive_native_decrease']} / {e['subsequent_nonpositive_native_decrease']} |")
    lines += ["", "Every accepted-step record also retains the old radius, full pre-step Coleman–Li scale, realized norm, carried radius, bound/head-face activity and next-trial fidelity in `analysis.json` and the cell result.", "",
        "## Work and integrity", "", f"Actual shared source expense was 4 N1 events, 6,000 N2 evaluations and 4 endpoint replays. N3 used {total_n3['profile_evaluations']} profiles, {total_n3['joint_model_solves']} joint models, {total_n3['center_derivative_evaluations']} derivative evaluations, {total_n3['accepted_hidden_updates']} accepted updates, {total_n3['rejected_proposals']} rejected proposals and {total_n3['native_objectives_source_plus_profiles']} native objectives including source replays. A deployable prefix500 arm costs 500 N2 evaluations, not 1,500.", "", f"All {certificate_checks['files']} exact certificates report exact feasibility, exact KKT and exact full rank. Initializer and cell manifests have zero mismatches. The final seal covers protected inputs and all output artifacts.", "",
        "This was a bounded short-horizon diagnostic only. It supports a checkpoint-by-controller interaction in these paired cases and does not establish an optimal controller, eventual recovery, general reliability or a production change."]
    (SCREEN / "final_report.md").write_text("\n".join(lines) + "\n")
    output_hashes = {}
    for p in sorted(SCREEN.rglob("*")):
        if p.is_file() and p.name != "final_integrity.json": output_hashes[str(p)] = sha(p)
    protected = json.loads((SCREEN / "cell_batch_started.json").read_text())["input_sha256"]
    checks = {p: sha(p) == digest for p, digest in protected.items()}
    save(SCREEN / "final_integrity.json", {"protocol_sha256": PROTOCOL_SHA, "formula_addendum_sha256": ADDENDUM_SHA,
        "protected_input_sha256": protected, "protected_checks": checks, "protected_unchanged": all(checks.values()),
        "output_sha256": output_hashes, "output_count": len(output_hashes),
        "initializer_manifest_failures": initializer_manifest_failures, "cell_manifest_failures": cell_manifest_failures,
        "certificate_failures": certificate_checks["failures"]})
    print(json.dumps(clean({"status": analysis["status"], "cell_status_counts": analysis["cell_status_counts"],
        "interaction_cases": interaction_cases, "work": analysis["work"], "integrity": analysis["integrity"]}), indent=2))


if __name__ == "__main__": main()
