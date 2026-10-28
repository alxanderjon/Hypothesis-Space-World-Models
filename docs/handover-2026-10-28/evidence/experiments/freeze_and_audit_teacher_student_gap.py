#!/usr/bin/env python3
"""Freeze the validated deterministic teacher and inventory its student gap.

This is a read-only implementation audit apart from its output artifacts.  It
does not execute optimization, train a model, or change the teacher path.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.sho.deterministic_full_k_framework import hashes


OUT = Path("outputs/deterministic_teacher_frozen")
READINESS = Path("outputs/deterministic_teacher_readiness/report.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(section, operation, category, inputs, outputs, reuse, general,
         scaffold, behavior, debt="none"):
    return dict(section=section, operation=operation, category=category,
                input=inputs, output=outputs, reuse=reuse,
                dimension_general=general, scaffolding=scaffold,
                affects_behavior=behavior, debt_scope=debt)


def inventory():
    rows = []
    add = rows.append
    # N1: current continuous E-DIRECT path.
    add(item("N1", "dimension-general initial evaluated design", "D", "hypothesis dimension d", "ordered 5d normalized sites", "fixed modular-lattice design; no accepted learned replacement", True, False, True, "current"))
    add(item("N1", "ordered history snapshot", "C", "sites[n,d], losses[n]", "round state", "lossless ragged state", True, True, False))
    add(item("N1", "reservoir budget and deterministic seed", "C", "n, case_id, iteration", "N and Sobol seed", "fixed teacher policy", True, True, True))
    add(item("N1", "Sobol full-block sizing and prefix", "B", "N", "2^m block length and first-N route", "validated batching/truncation assembly", True, False, True))
    add(item("N1", "scrambled Sobol coordinate generation", "B", "d, seed, counter", "ordered normalized interior sites", "locked recurrence/XOR/scaling assemblies", True, False, True))
    add(item("N1", "dimension-general boundary-face support", "D", "N, d, face axis/side", "ordered face probes", "d=2 locked path only; d>2 Sobol-face rule remains teacher code", True, False, True, "current"))
    add(item("N1", "pool merge, exact deduplication, lexicographic order", "B", "interior and face rows", "canonical reservoir", "n1_boundary_merge_dedup assembly", True, False, True))
    add(item("N1", "cKDTree construction", "C", "ordered sites", "spatial index", "library data structure", True, True, False))
    add(item("N1", "Euclidean distance", "A", "two d-vectors", "float64 distance", "geometric_distance_metric_locked plus exact sqrt assembly", True, False, True))
    add(item("N1", "nearest-owner routing away from exact ties", "B", "candidate-site distances", "owner and nearest distance", "nearest argmin/comparison assemblies", True, False, True))
    add(item("N1", "production cKDTree exact-tie ownership", "D", "ordered equal-distance sites and query k", "production owner identity", "no learned k-dependent heap-tie equivalent", True, False, True, "current"))
    add(item("N1", "assigned-radius reduction", "A", "owners and distances", "maximum owned distance per cell", "assigned_radius_operator_locked", True, False, True))
    add(item("N1", "nearest-other-site spacing", "B", "site-site distances", "second returned distance", "distance plus nearest/repeated-min assembly", True, False, True))
    add(item("N1", "cell extent", "A", "assigned radius, 0.5 nearest spacing", "cell_size=max(inputs)", "half_spacing and cell_size locked operators", True, False, True))
    add(item("N1", "canonical cell sort", "C", "size, loss, site id", "ordered site ids", "stable lexicographic indexing", True, True, True))
    add(item("N1", "same-size grouping", "B", "ordered sizes", "size groups", "validated same-size grouping assembly", True, False, True))
    add(item("N1", "group representative selection", "B", "group losses and ids", "strict-lowest representative", "representative tie selector assembly", True, False, True))
    add(item("N1", "lower-hull turn and margin", "A", "three size/loss representatives", "KEEP/POP predicate", "locked hull turn/margin operators", True, False, True))
    add(item("N1", "lower-hull stack mutation", "C", "ordered representatives and predicates", "ordered hull", "variable-length stack/index state", True, True, True))
    add(item("N1", "sufficient-decrease target", "B", "minimum loss", "fmin-1e-4*abs(fmin)", "locked arithmetic assembly", True, False, True))
    add(item("N1", "neighbor/required hull slopes", "B", "adjacent hull vertices and target", "slope bounds", "locked subtraction/division plus endpoint routing", True, False, True))
    add(item("N1", "potentially-optimal feasibility", "B", "slope bounds", "selected-cell Boolean", "locked improvement-condition assembly", True, False, True))
    add(item("N1", "empty-PO largest-cell fallback", "B", "cell sizes and empty predicate", "fallback site id", "largest router and comparison assembly", True, False, True))
    add(item("N1", "selected-cell canonical order", "C", "selected ids, sizes, losses", "ordered selected cells", "index/order policy", True, True, True))
    add(item("N1", "loss floor clamp", "B", "raw loss", "max(loss,1e-300)", "comparison/select locked assembly", True, False, True))
    add(item("N1", "production scalar np.log10", "D", "positive clamped float64", "encoded loss", "decomposed baseline remains DIAGNOSTIC_ONLY due libm ULP residue", True, False, True, "current"))
    add(item("N1", "ordered top-min(24,n) neighbor retrieval", "D", "query, ordered sites, distances", "ordered ids/distances", "nearest-1 is solved; exact top-k/tie contract is not", True, False, True, "current"))
    add(item("N1", "Gaussian bandwidth floor and ratio", "B", "ordered distances", "distance/max(last,1e-12)", "comparison/select/division assemblies", True, False, True))
    add(item("N1", "production scalar np.exp in Gaussian kernel", "D", "z in [-9,0]", "production exp(z)", "zero-parameter correct-rounding assembly is behaviorally validated but not production-bit-exact", True, False, True, "current"))
    add(item("N1", "ordered weighted surrogate prediction", "B", "weights and encoded losses", "weighted mean", "locked add/multiply/divide with deterministic reduction order", True, False, True))
    add(item("N1", "LOO exact-site removal and truncation", "C", "ordered up-to-25 neighbors and selected id", "up-to-24 nonself neighbors", "variable-length indexing", True, True, True))
    add(item("N1", "LOO residual and small-site fallback", "B", "site encoding and neighbor prediction", "absolute residual or one", "selected-site LOO exact assembly", True, False, True))
    add(item("N1", "guarded min-max normalization", "B", "nonempty float64 vector", "normalized vector or 0.5 fill", "LOCKED_ASSEMBLY_READY; 0 replay errors", True, False, True))
    add(item("N1", "selected-cell weight", "B", "normalized extent and LOO", "u=0.5*s+0.5*loo", "locked arithmetic assembly", True, False, True))
    add(item("N1", "largest-cell role", "A", "selected extents", "first maximum identity", "largest_cell_router LOCKED", True, False, True))
    add(item("N1", "ownership mask and cell slices", "C", "owner ids and canonical pool", "per-cell ordered candidates", "indexing/storage", True, True, False))
    add(item("N1", "largest-cell candidate score/route", "A", "owned distances", "first farthest candidate", "largest-cell scoring/routing locked", True, False, True))
    add(item("N1", "ordinary-cell score arithmetic", "B", "u, normalized distance, normalized prediction", "candidate score", "ordinary_cell_scoring LOCKED_ASSEMBLY", True, False, True))
    add(item("N1", "ordinary-cell winner", "B", "ordered scores", "canonical first maximum", "comparison plus first-max router assembly", True, False, True))
    add(item("N1", "history duplicate filter", "B", "candidate and evaluated sites", "accept iff nearest distance>1e-12", "distance/comparison assembly", True, False, True))
    add(item("N1", "empty-proposal global fallback", "B", "all reservoir distances", "first valid reverse-distance candidate", "comparison/router plus deterministic traversal", True, False, True))
    add(item("N1", "FIFO proposal queue", "C", "ordered cell proposals", "next candidate", "container/order bookkeeping", True, True, True))
    add(item("N1", "normalized-to-physical affine map", "B", "u, lower, upper", "theta", "locked subtract/multiply/add assembly", True, False, False))
    add(item("N1", "physical objective callback invocation", "C", "theta and observed prefix", "scalar loss", "external system adapter", True, True, True))
    add(item("N1", "observation/archive insertion", "C", "theta, unit point, loss, ordinal", "extended ordered history", "storage", True, True, False))
    add(item("N1", "continuous-stage Sobol multistarts", "B", "d and fixed seed plus incumbent", "2d+1 ordered starts", "locked Sobol path plus structural prepend", True, False, True))
    add(item("N1", "bounded multistart Powell surrogate minimization", "B", "surrogate, starts, [0,1]^d", "per-start minima and winning proposal", "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_STATE", True, False, True))
    add(item("N1", "continuous result guards/winner/duplicate route", "B", "per-start results and history", "valid nonduplicate proposal or none", "finite/bounds/comparison/equality assemblies", True, False, True))
    add(item("N1", "round/cap/trace control", "C", "iteration, queue, counts", "next round or resource exit", "loops/counters/serialization", True, True, True))

    # Archive handoff orchestration.
    add(item("Archive", "chronological retention", "C", "each evaluated N1 site", "A_n={theta_1..theta_n}", "ordered storage", True, True, False))
    add(item("Archive", "retest old sites when current-count box changes", "D", "archive and current n", "site/count probe schedule", "validated deterministic schedule; no learned equivalent", True, False, True, "current"))
    add(item("Archive", "newest-first then older-site ordering", "D", "archive at count n", "ordered branch queue", "no promoted learned archive selector", True, False, True, "current"))
    add(item("Archive", "branch execution", "C", "site/count pair", "N2 then N3 result", "call/trace orchestration", True, True, True))
    add(item("Archive", "valid-handoff/recovered-law oracle", "D", "completed N2/N3 result and offline recovery evidence", "GOOD/BAD branch label", "teacher-only recovery judgement; no promoted runtime gate", True, False, True, "current"))
    add(item("Archive", "absorbing success latch", "B", "prior success and current GOOD", "stop/retain successful site", "locked Boolean/select semantics plus state bit", True, False, True))

    # N2.
    add(item("N2", "current-count local box construction", "D", "archive theta, global bounds, n, d", "clipped lower/upper via width/n^(1/d)", "generic arithmetic partly reusable; exact variable root/whole contract not recovered", True, False, True, "current"))
    add(item("N2", "bounded Powell mathematical state machine", "B", "objective, start, box, tolerances/caps", "Powell result", "EXACT_LEARNED_ASSEMBLY; SciPy-1.13.0 semantics", True, False, True))
    add(item("N2", "Powell loops/vector state/counters", "C", "mutable Powell state", "next state and termination", "deterministic loop/index/storage", True, True, False))
    add(item("N2", "best-ever strict-loss selection", "B", "objective evaluations in order", "best theta/loss", "locked strict comparison/select assembly", True, False, True))
    add(item("N2", "endpoint packaging to N3", "C", "best theta/loss and provenance", "N2Endpoint", "record construction", True, True, False))

    # N3.
    add(item("N3", "gradient operator", "A", "theta and objective graph", "float64 gradient", "n3_gradient_operator FROZEN_CANDIDATE", True, False, True))
    add(item("N3", "symmetric Hessian operator", "A", "frozen gradient graph", "symmetric Hessian", "n3_hessian_operator FROZEN_CANDIDATE", True, False, True))
    add(item("N3", "positive-base variable-power adjoints", "A", "base, exponent, incoming adjoints", "first/mixed/second adjoints", "n3_variable_power_adjoint FROZEN_CANDIDATE", True, False, True))
    add(item("N3", "Hessian eigenspectrum and condition", "D", "symmetric Hessian", "eigenvalues, min/max scale, condition", "torch.linalg.eigvalsh retained as deterministic oracle", True, False, True, "current"))
    add(item("N3", "DIRECT/DAMPED curvature route", "A", "min_eig>0 and condition<=1e12", "DIRECT or DAMPED", "n3_curvature_router LOCKED", True, False, True))
    add(item("N3", "damping arithmetic/retry values", "B", "eigen scale, min eigenvalue, trial", "lambda", "n3_damping_construction LOCKED_ASSEMBLY", True, False, True))
    add(item("N3", "damped matrix and RHS assembly", "B", "H, lambda, runtime d, gradient", "H+lambda*I and -g", "locked arithmetic with generic identity scaffold", True, False, True))
    add(item("N3", "dense linear solve", "D", "square matrix and RHS", "Newton direction", "temporary torch.linalg.solve scaffold; no accepted learned solve", True, False, True, "current"))
    add(item("N3", "dimension-general box-QP active-set direction", "D", "theta, bounds, g, H and solve callback", "feasible constrained quadratic direction", "validated deterministic active-set algorithm, 0 new parameters", True, False, True, "current"))
    add(item("N3", "active-set loop/index/state", "C", "working/free/active sets", "updated active set", "loop, masks, storage", True, True, False))
    add(item("N3", "descent validity", "B", "direction and gradient", "finite(d) AND g.dot(d)<0", "n3_descent_check LOCKED_ASSEMBLY", True, False, True))
    add(item("N3", "power-of-two backtracking schedule", "B", "direction and attempt k", "alpha=2^-k candidate", "locked arithmetic; loop counter retained", True, False, True))
    add(item("N3", "candidate bounds route", "B", "theta+alpha*d and bounds", "feasible/reject", "comparison/Boolean assembly", True, False, True))
    add(item("N3", "objective evaluation at candidate", "C", "candidate theta", "candidate loss", "external objective callback", True, True, True))
    add(item("N3", "strict step acceptance", "B", "candidate and current losses", "accept iff candidate<current", "n3_strict_step_acceptance LOCKED_ASSEMBLY", True, False, True))
    add(item("N3", "genuine numerical STOP/CONTINUE routes", "B", "gradient/finite/descent/backtrack/unchanged/predicted-decrease state", "stop reason or continue", "n3_stop_continue and backtracking control locked assemblies", True, False, True))
    add(item("N3", "FLOAT64_STAGNATION_TAIL stop", "D", "accepted-step loss/predicted-decrease ULPs and persistence", "tail stop after 3 qualifying accepted steps", "calibrated deterministic rule; no learned equivalent", True, False, True, "current"))
    add(item("N3", "iteration/history bookkeeping", "C", "accepted/rejected state and traces", "next iteration/final result", "loops/counters/storage", True, True, False))

    # Physical/hypothesis structure; deliberately separated as later peelback.
    add(item("Physics", "full-K SHO delay-state linear update", "D", "z_t and free 2x2 K", "z_{t+1}=K z_t", "teacher-supplied hypothesis structure", True, False, True, "later_hypothesis"))
    add(item("Physics", "reflection-ball K/rho event update", "D", "position/velocity, free K, rho, event mask", "transport/sample velocity after impact", "teacher-supplied known structure", True, False, True, "later_hypothesis"))
    add(item("Physics", "ball collision/event scaffold", "D", "state, transport velocity, box, dt", "hit time/wall, snap, ordering, remaining propagation", "exact deterministic physical scaffold", True, False, True, "later_hypothesis"))
    add(item("Physics", "five-parameter Kepler force law", "D", "state and (c1,c2,eta,kd,p)", "two-body accelerations", "teacher-supplied inverse-radius shared law", True, False, True, "later_hypothesis"))
    add(item("Physics", "velocity-Verlet update structure", "D", "state, accelerations, dt and coefficients", "next Kepler state", "exact production-order deterministic integrator", True, False, True, "later_hypothesis"))
    add(item("Physics", "one-step prefix residual/MSE reduction", "B", "candidate transitions and observed prefix", "scalar objective", "locked subtract/square/add/divide plus ordered loops", True, False, True))
    add(item("Physics", "state layouts, constants and metadata", "C", "system configuration", "typed transition inputs", "fixed structure/storage", True, True, False))
    add(item("Physics", "teacher rollout and event-topology diagnostics", "C", "recovered law and initial state", "continuation/error/signature", "offline validation scaffolding", True, True, False))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    readiness = json.loads(READINESS.read_text())
    if readiness["status"] != "DETERMINISTIC_TEACHER_VALIDATED":
        raise RuntimeError("teacher is not validated")
    checkpoint_hashes = hashes()
    before = json.loads(Path(
        "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
    ).read_text())
    if checkpoint_hashes != before:
        raise RuntimeError("checkpoint inventory changed since readiness validation")
    sources = [Path(value) for value in [
        "experiments/validate_deterministic_teacher_readiness.py",
        "experiments/expand_chronological_archive_handoff_dataset.py",
        "experiments/validate_d5_production_pipeline.py",
        "experiments/sho/dimension_general_edirect.py",
        "experiments/sho/run_full_k_transfer.py",
        "experiments/sho/run_stage1b_edirect_vs_direct.py",
        "experiments/sho/continuous_surrogate_edirect.py",
        "experiments/n2_exact_powell_assembly.py",
        "experiments/kepler/run_four_parameter_peelback.py",
        "experiments/n3_box_qp.py",
        "experiments/sho/run_stage1b_n3_gradient_polishing.py",
        "experiments/bouncing_ball/run_reflection_peelback.py",
        "experiments/bouncing_ball/run_first_transfer.py",
        "experiments/kepler/deterministic_teacher.py",
        "experiments/audit_known_structure.py",
    ]]
    rows = inventory()
    counts = dict(sorted(Counter(row["category"] for row in rows).items()))
    unresolved = [row for row in rows if row["category"] == "D"]
    dependency_graph = [
        "observed prefix + parameter bounds",
        "N1 initial design -> objective evaluations -> chronological archive",
        "N1 round: Sobol reservoir + boundary support -> Voronoi ownership/cell extents",
        "cell extents/losses -> grouping/representatives -> lower hull/potentially-optimal cells",
        "encoded losses + ordered neighbors + Gaussian weights -> surrogate predictions/LOO",
        "normalized cell features -> ordinary/largest scoring -> FIFO cell proposals",
        "updated archive -> exact assembled multistart Powell surrogate proposal -> archive insertion",
        "archive at count n -> newest-first site/count probes -> older-site retries",
        "archived site + current-count N2 box -> exact assembled bounded Powell -> strict best-ever endpoint",
        "N2 endpoint -> frozen gradient/Hessian -> eigenspectrum/curvature route/damping",
        "damped Hessian + RHS -> linear solve -> box-QP active-set feasible direction",
        "descent/bounds/backtracking/strict acceptance -> genuine stops or FLOAT64_STAGNATION_TAIL",
        "valid recovered-law branch -> absorbing archive success -> recovered parameters/rollout",
    ]
    integration_order = [
        "Pin the frozen source/checkpoint manifest and use the existing lossless ragged state; keep every D operation as the teacher oracle.",
        "Wire validated N1 A/B components in dependency order through cell selection and proposal scoring, preserving C ordering/storage exactly.",
        "Wire the exact N1 continuous-Powell assembly against the unchanged production surrogate callback.",
        "Retain chronological archive storage, newest-first/older retries, and direct branch probing; existing learned archive gate/selector screens are not promoted.",
        "Wire the current-count box interface and the exact N2 Powell assembly; preserve strict-best and endpoint provenance.",
        "Wire frozen N3 gradient/Hessian/variable-power and locked routing/control assemblies; retain eigvalsh, torch.linalg.solve, box-QP, and stagnation-tail D oracles.",
        "Run same-input component replay, then closed-loop teacher/student replay; only after parity isolate D operations one at a time in dependency order.",
    ]
    manifest = {
        "status": "FROZEN_VALIDATED_DETERMINISTIC_TEACHER",
        "teacher_readiness_status": readiness["status"],
        "teacher_readiness_report": str(READINESS),
        "teacher_readiness_sha256": digest(READINESS),
        "validated_trajectory_count": 822,
        "checkpoint_count": len(checkpoint_hashes),
        "checkpoint_manifest_sha256": digest(Path(
            "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
        )),
        "sources": {str(path): digest(path) for path in sources},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
        },
        "algorithm_changes": 0, "training_performed": False,
    }
    report = {
        "status": "TEACHER_FROZEN_STUDENT_GAP_AUDITED",
        "freeze_manifest": manifest,
        "classifications": {
            "A": "LEARNED / LOCKED",
            "B": "EXACT LEARNED ASSEMBLY",
            "C": "DETERMINISTIC SCAFFOLDING",
            "D": "SUBSTANTIVE DETERMINISTIC OPERATION STILL UNLEARNED",
        },
        "counts": {**counts, "total": len(rows)},
        "inventory": rows,
        "remaining_D": unresolved,
        "earliest_remaining_D": unresolved[0],
        "remaining_D_by_scope": {
            scope: [row["operation"] for row in unresolved
                    if row["debt_scope"] == scope]
            for scope in ("current", "later_hypothesis")
        },
        "dependency_graph": dependency_graph,
        "recommended_student_integration_order": integration_order,
        "checkpoint_hashes_unchanged": checkpoint_hashes == before,
    }
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "operation_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "operation_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen deterministic teacher: learned-student gap", "",
        f"Status: `{report['status']}`", "",
        f"Validated teacher: `{manifest['teacher_readiness_status']}` on 822/822 trajectories.", "",
        "## Counts", "",
        "| Class | Count |", "|---|---:|",
    ]
    for key in "ABCD":
        lines.append(f"| {key} | {counts.get(key, 0)} |")
    lines += [f"| Total | {len(rows)} |", "", "## Inventory", "",
              "| # | Stage | Operation | Class | Dimension-general | Affects behavior | Debt |",
              "|---:|---|---|:---:|:---:|:---:|---|"]
    for index, row in enumerate(rows, 1):
        lines.append(f"| {index} | {row['section']} | {row['operation']} | {row['category']} | "
                     f"{str(row['dimension_general']).lower()} | {str(row['affects_behavior']).lower()} | {row['debt_scope']} |")
    lines += ["", "## Remaining substantive deterministic operations", ""]
    for row in unresolved:
        lines.append(f"- **{row['section']} — {row['operation']}** ({row['debt_scope']}): "
                     f"`{row['input']}` → `{row['output']}`. {row['reuse']}")
    lines += ["", "## Dependency graph", ""]
    for index, value in enumerate(dependency_graph, 1):
        lines.append(f"{index}. {value}")
    lines += ["", "## Recommended student integration order", ""]
    for index, value in enumerate(integration_order, 1):
        lines.append(f"{index}. {value}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "earliest_D": report["earliest_remaining_D"]["operation"],
                      "D_current": len(report["remaining_D_by_scope"]["current"]),
                      "D_later": len(report["remaining_D_by_scope"]["later_hypothesis"]),
                      "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"]},
                     indent=2))


if __name__ == "__main__":
    main()
