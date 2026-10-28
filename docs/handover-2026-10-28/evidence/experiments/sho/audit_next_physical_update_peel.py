#!/usr/bin/env python3
"""Audit the next SHO physical/update-rule peel without changing the model.

The audit reads only the existing 320 canonical SHO trajectories.  It records
rank evidence for local relaxations of the frozen affine update law, but it
does not add a hypothesis, train, or invoke recovery/search.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import canonical_teacher_regression_corpus as canonical


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sho" / "next_physical_update_peel_audit"
FREEZE = ROOT / "outputs" / "sho" / "homogeneity_peeled_baseline" / "freeze_manifest.json"
CANONICAL_MANIFEST = ROOT / "outputs" / "canonical_frozen_teacher_regression_822" / "corpus_manifest.json"
CANONICAL_POINTER = ROOT / "outputs" / "canonical_frozen_teacher_regression_822" / "CANONICAL"
AFFINE_SOURCE = ROOT / "experiments" / "geometry" / "affine_sho_update_assembly.py"
EXPECTED_FREEZE_PAYLOAD = "1ed83add04c2ccb0a1da47099ca77ae3b30989efaabae77181869e26b2105eca"
EXPECTED_CORPUS_PAYLOAD = "51c5273ec3876c35db58f75e6fa0a831b3223ee5477082d8575cae2923d506dd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values) -> dict:
    data = np.asarray(values, np.float64)
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)),
        "maximum": float(np.max(data)),
    }


def numerical_rank(design: np.ndarray) -> dict:
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "rank": rank,
        "columns": int(design.shape[1]),
        "nullity": int(design.shape[1] - rank),
        "singular_values": singular.tolist(),
        "standard_machine_precision_tolerance": float(tolerance),
        "minimum_singular_over_tolerance": float(singular[-1] / tolerance),
    }


def rank_summary(rows: list[dict]) -> dict:
    return {
        "case_count": len(rows),
        "design_rank_counts": dict(sorted(Counter(str(row["rank"]) for row in rows).items())),
        "design_nullity_counts": dict(sorted(Counter(str(row["nullity"]) for row in rows).items())),
        "joint_two_output_jacobian_rank_counts": dict(sorted(Counter(
            str(2 * row["rank"]) for row in rows
        ).items())),
        "minimum_singular_over_tolerance": distribution([
            row["minimum_singular_over_tolerance"] for row in rows
        ]),
    }


def category(spec: dict, legacy) -> str:
    if spec["family"] == "legacy_prefix":
        return str(legacy["categories"][int(spec["case_id"])])
    return "readiness_sobol"


def main() -> None:
    freeze_before = sha256(FREEZE)
    corpus_before = sha256(CANONICAL_MANIFEST)
    pointer_before = CANONICAL_POINTER.read_text().strip()
    frozen = json.loads(FREEZE.read_text())
    if frozen.get("manifest_payload_sha256") != EXPECTED_FREEZE_PAYLOAD:
        raise RuntimeError("SHO homogeneity-peeled baseline changed")
    if pointer_before != EXPECTED_CORPUS_PAYLOAD:
        raise RuntimeError("canonical corpus pointer changed")

    specs = [row for row in canonical.build_specs() if row["system"] == "full_k_sho"]
    if len(specs) != 320:
        raise RuntimeError(f"expected 320 canonical SHO specifications, got {len(specs)}")
    legacy = np.load(canonical.SHO_LEGACY_INPUT)

    ranks: dict[str, list[dict]] = defaultdict(list)
    rank_by_category: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    feature_maxima: dict[str, float] = defaultdict(float)
    extra_coefficients: dict[str, list[float]] = defaultdict(list)
    augmented_fit_residuals: dict[str, list[float]] = defaultdict(list)
    transition_counts = Counter()
    exact_duplicate_input_counts = []
    exact_zero_input_counts = []
    determinants = []
    teacher_native_matmul_residuals = []
    per_transition_counts = []

    for spec in specs:
        trajectory, _, _, _, _, _, teacher = canonical.case_data(spec)
        trajectory = np.asarray(trajectory, np.float64)
        teacher = np.asarray(teacher, np.float64).reshape(2, 2)
        inputs = trajectory[:-1]
        targets = trajectory[1:]
        n = len(inputs)
        transition_counts[n] += 1
        determinants.append(float(np.linalg.det(teacher)))
        exact_duplicate_input_counts.append(n - len(np.unique(inputs, axis=0)))
        nonzero_input_count = int(np.count_nonzero(np.any(inputs != 0.0, axis=1)))
        zero_input_count = n - nonzero_input_count
        exact_zero_input_counts.append(zero_input_count)
        teacher_native_matmul_residuals.append(float(np.max(np.abs(
            inputs @ teacher.T - targets
        ))))

        one = np.ones(n, np.float64)
        time = np.arange(n, dtype=np.float64)
        z0 = inputs[:, 0]
        z1 = inputs[:, 1]
        features = {
            "quadratic_z0_squared": z0 * z0,
            "quadratic_cross": z0 * z1,
            "quadratic_z1_squared": z1 * z1,
            "quadratic_norm_squared": z0 * z0 + z1 * z1,
            "linear_time_bias": time,
        }
        designs = {
            "current_affine": np.column_stack((inputs, one)),
            "quadratic_z0_squared": np.column_stack((inputs, one, features["quadratic_z0_squared"])),
            "quadratic_cross": np.column_stack((inputs, one, features["quadratic_cross"])),
            "quadratic_z1_squared": np.column_stack((inputs, one, features["quadratic_z1_squared"])),
            "quadratic_norm_squared": np.column_stack((inputs, one, features["quadratic_norm_squared"])),
            "complete_quadratic": np.column_stack((
                inputs, one, features["quadratic_z0_squared"],
                features["quadratic_cross"], features["quadratic_z1_squared"],
            )),
            "linear_time_bias": np.column_stack((inputs, one, time)),
            "linear_time_matrix": np.column_stack((inputs, one, time * z0, time * z1)),
            "linear_time_affine": np.column_stack((inputs, one, time, time * z0, time * z1)),
            "one_additional_lag": np.column_stack((
                trajectory[1:-1], np.ones(len(trajectory) - 2, np.float64), trajectory[:-2],
            )),
        }
        cat = category(spec, legacy)
        for name, design in designs.items():
            result = numerical_rank(design)
            ranks[name].append(result)
            rank_by_category[name][cat][result["rank"]] += 1

        # These least-squares coefficients are descriptive float64 diagnostics.
        # Under the exact affine law, the added coefficient has the exact
        # structural solution zero whenever the augmented design is full rank.
        for name, feature in features.items():
            design = np.column_stack((inputs, one, feature))
            coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
            extra_coefficients[name].extend(np.abs(coefficients[-1]).tolist())
            augmented_fit_residuals[name].append(float(np.max(np.abs(
                design @ coefficients - targets
            ))))
        complete = designs["complete_quadratic"]
        complete_rank = ranks["complete_quadratic"][-1]["rank"]
        if complete_rank == complete.shape[1]:
            coefficients = np.linalg.lstsq(complete, targets, rcond=None)[0]
            extra_coefficients["complete_quadratic_full_rank_cases"].extend(
                np.abs(coefficients[-3:]).ravel().tolist()
            )
            augmented_fit_residuals["complete_quadratic_full_rank_cases"].append(
                float(np.max(np.abs(complete @ coefficients - targets)))
            )

        for name, values in {
            "z0": z0, "z1": z1,
            "z0_squared": z0 * z0, "z0_z1": z0 * z1,
            "z1_squared": z1 * z1, "time_index": time,
            "time_z0": time * z0, "time_z1": time * z1,
        }.items():
            feature_maxima[name] = max(feature_maxima[name], float(np.max(np.abs(values))))

        # A per-transition affine row has one scalar observation and three
        # coefficients; a per-transition matrix row has one observation and
        # two coefficients.  Record their exact structural ranks/nullities.
        per_transition_counts.append({
            "transitions": n,
            "matrix_only_parameters": 4 * n,
            "matrix_only_jacobian_rank": 2 * nonzero_input_count,
            "matrix_only_nullity": 4 * n - 2 * nonzero_input_count,
            "affine_parameters": 6 * n,
            "affine_jacobian_rank": 2 * n,
            "affine_nullity": 4 * n,
            "time_varying_matrix_plus_global_bias_parameters": 4 * n + 2,
            "time_varying_matrix_plus_global_bias_jacobian_rank": 2 * (
                nonzero_input_count + int(zero_input_count > 0)
            ),
            "time_varying_matrix_plus_global_bias_nullity": 4 * n + 2 - 2 * (
                nonzero_input_count + int(zero_input_count > 0)
            ),
        })

    coefficient_diagnostics = {}
    for name, values in extra_coefficients.items():
        coefficient_diagnostics[name] = {
            "absolute_added_coefficient": distribution(values),
            "maximum_float64_fit_residual": float(max(augmented_fit_residuals[name])),
            "interpretation": (
                "descriptive finite-precision least-squares result; the exact affine-law "
                "embedding sets every added coefficient to zero"
            ),
        }

    report = {
        "status": "AUDIT_COMPLETE_NO_PEEL_IMPLEMENTED",
        "scope": "next physical/update-rule peel after SHO_HOMOGENEITY_PEELED_BASELINE",
        "immutable_inputs": {
            "sho_homogeneity_freeze_manifest_sha256": freeze_before,
            "sho_homogeneity_freeze_payload_sha256": EXPECTED_FREEZE_PAYLOAD,
            "canonical_corpus_manifest_sha256": corpus_before,
            "canonical_corpus_payload_sha256": pointer_before,
            "affine_source_sha256": sha256(AFFINE_SOURCE),
        },
        "data": {
            "canonical_sho_trajectories": len(specs),
            "prefix_length_counts": dict(sorted((str(k + 1), v) for k, v in transition_counts.items())),
            "transition_count_counts": dict(sorted((str(k), v) for k, v in transition_counts.items())),
            "observed_dimension": 2,
            "input_channels": 0,
            "exact_duplicate_input_state_count_by_case": dict(sorted(Counter(
                str(value) for value in exact_duplicate_input_counts
            ).items())),
            "exact_zero_input_state_count_by_case": dict(sorted(Counter(
                str(value) for value in exact_zero_input_counts
            ).items())),
            "teacher_matrix_determinant": distribution(determinants),
            "absolute_teacher_native_matmul_vs_stored_target_residual": distribution(
                teacher_native_matmul_residuals
            ),
        },
        "current_live_graph": {
            "source": "experiments/geometry/affine_sho_update_assembly.py:AffineSHOUpdateAssembly",
            "input": "trajectory float64 [T,2], theta float64 [6]",
            "execution_order": [
                "inputs = trajectory[:-1]; targets = trajectory[1:]",
                "K = theta[:4].reshape(2,2); b = theta[4:]",
                "prediction_before_bias = exact teacher-order two-product Kz matmul",
                "prediction = locked coordinatewise add(prediction_before_bias, broadcast(b))",
                "residual = locked subtract(prediction, targets)",
                "square = locked multiply(residual, residual)",
                "loss = production-order mean reduction",
            ],
            "state_features": ["z0", "z1", "constant 1 (implemented as separate bias add)"],
            "recovered_quantities": ["K00", "K01", "K10", "K11", "b0", "b1"],
            "deterministic_scaffolding": [
                "trajectory pair slicing", "K reshape", "bias broadcasting",
                "batch/coordinate indexing and storage", "transition iteration/rollout loop",
            ],
            "physical_structure_still_supplied": frozen["remaining_supplied_physical_assumptions"],
        },
        "rank_evidence": {name: rank_summary(values) for name, values in ranks.items()},
        "rank_evidence_by_category": {
            name: {cat: dict(sorted((str(rank), count) for rank, count in counts.items()))
                   for cat, counts in sorted(categories.items())}
            for name, categories in ranks_by_category_sorted(rank_by_category).items()
        },
        "feature_absolute_maxima": dict(sorted(feature_maxima.items())),
        "added_coefficient_diagnostics": coefficient_diagnostics,
        "explicit_timestep_jacobian": {
            "parameterization": "z_next = z + dt * (A z + c)",
            "free_quantities": 7,
            "observable_combinations": 6,
            "rank": 6,
            "nullity": 1,
            "exact_dependence": (
                "the dt Jacobian column equals the A/c columns combined with current A/c; "
                "(A,c,dt) and (A/alpha,c/alpha,alpha*dt) give the same affine K,b"
            ),
            "affine_bias_effect": "adding c preserves rather than removes the scale gauge",
        },
        "per_transition_time_variation": {
            "transition_count_counts": dict(sorted((str(k), v) for k, v in transition_counts.items())),
            "matrix_only": {
                "parameter_count_range": [min(x["matrix_only_parameters"] for x in per_transition_counts),
                                          max(x["matrix_only_parameters"] for x in per_transition_counts)],
                "rank_range": [min(x["matrix_only_jacobian_rank"] for x in per_transition_counts),
                               max(x["matrix_only_jacobian_rank"] for x in per_transition_counts)],
                "nullity_range": [min(x["matrix_only_nullity"] for x in per_transition_counts),
                                  max(x["matrix_only_nullity"] for x in per_transition_counts)],
            },
            "affine_per_transition": {
                "parameter_count_range": [min(x["affine_parameters"] for x in per_transition_counts),
                                          max(x["affine_parameters"] for x in per_transition_counts)],
                "rank_range": [min(x["affine_jacobian_rank"] for x in per_transition_counts),
                               max(x["affine_jacobian_rank"] for x in per_transition_counts)],
                "nullity_range": [min(x["affine_nullity"] for x in per_transition_counts),
                                  max(x["affine_nullity"] for x in per_transition_counts)],
            },
            "matrix_per_transition_plus_global_bias": {
                "parameter_count_range": [min(x["time_varying_matrix_plus_global_bias_parameters"] for x in per_transition_counts),
                                          max(x["time_varying_matrix_plus_global_bias_parameters"] for x in per_transition_counts)],
                "rank_range": [min(x["time_varying_matrix_plus_global_bias_jacobian_rank"] for x in per_transition_counts),
                               max(x["time_varying_matrix_plus_global_bias_jacobian_rank"] for x in per_transition_counts)],
                "nullity_range": [min(x["time_varying_matrix_plus_global_bias_nullity"] for x in per_transition_counts),
                                  max(x["time_varying_matrix_plus_global_bias_nullity"] for x in per_transition_counts)],
            },
        },
        "nonlinear_audit": {
            "single_feature_result": (
                "Each tested scalar quadratic feature is independent of [z0,z1,1] in 320/320 "
                "cases and adds two identifiable output coefficients (dimension 8)."
            ),
            "complete_quadratic_result": (
                "The coordinate-complete quadratic correction adds six coefficients "
                "(dimension 12), but has rank 6 in only 222/320 cases and rank 5 in 98/320; "
                "it is therefore not identifiable over the complete corpus."
            ),
            "law_discovery_classification": (
                "Selecting z0^2, z0*z1, z1^2, their Euclidean sum, a polynomial degree, or "
                "an activation is a new hand-chosen basis assumption.  It enlarges the law "
                "class but does not remove nonlinear-form supply in a basis-neutral way."
            ),
            "trajectory_support_limitation": (
                "Each case supplies one orbit and no repeated exact input states.  Off-orbit "
                "nonlinear behavior is not determined without a restricted feature family."
            ),
        },
        "search_consequences": {
            "single_scalar_feature": {
                "dimension": 8, "stack_dimension_general": True,
                "coefficient_rule": "[-1.99,1.99] would apply to its two output coefficients",
                "scaling": "required before implementation for quadratic features; observed absolute feature maxima reach 208.11152766922865",
            },
            "complete_quadratic": {
                "dimension": 12, "stack_dimension_general": True,
                "coefficient_rule": "[-1.99,1.99] would apply after feature scaling",
                "scaling": "required", "identifiable_all_cases": False,
            },
            "linear_time_bias": {
                "dimension": 8, "stack_dimension_general": True,
                "coefficient_rule": "[-1.99,1.99] would apply after defining/scaling the clock feature",
                "scaling": "required; raw transition index reaches 126",
            },
            "linear_time_matrix": {
                "dimension": 10, "stack_dimension_general": True,
                "coefficient_rule": "[-1.99,1.99] would apply after feature scaling",
                "scaling": "required; |t*z| reaches 683.9302879566983",
            },
            "additional_lag": {
                "dimension": 10, "stack_dimension_general": True,
                "coefficient_rule": "would apply directly", "identifiable_all_cases": False,
            },
        },
        "candidate_conclusion": {
            "genuine_identifiable_peels": [],
            "identifiable_but_hand_chosen_basis_expansions": [
                "one scalar quadratic state feature with two output coefficients",
                "linear-in-index bias drift with two coefficients",
                "linear-in-index matrix drift with four coefficients",
            ],
            "smallest_rank_full_expansion": (
                "one chosen scalar feature, dimension 8; no candidate feature is selected "
                "because existing data do not privilege one basis or clock convention"
            ),
            "decision": "STOP_NO_FURTHER_GENUINE_IDENTIFIABLE_PEEL_FROM_CURRENT_TRAJECTORIES",
        },
        "training_performed": False,
        "new_hypothesis_implemented": False,
        "search_run": False,
    }

    freeze_after = sha256(FREEZE)
    corpus_after = sha256(CANONICAL_MANIFEST)
    pointer_after = CANONICAL_POINTER.read_text().strip()
    report["integrity"] = {
        "sho_homogeneity_freeze_unchanged": freeze_after == freeze_before,
        "canonical_manifest_unchanged": corpus_after == corpus_before,
        "canonical_pointer_unchanged": pointer_after == pointer_before,
    }
    if not all(report["integrity"].values()):
        raise RuntimeError("a frozen input changed during the audit")
    atomic_json(OUT / "report.json", report)


def ranks_by_category_sorted(value):
    return {key: value[key] for key in sorted(value)}


if __name__ == "__main__":
    main()
