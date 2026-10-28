#!/usr/bin/env python3
"""Pure saved-array algebra; no project/Torch imports or runtime calls.

Evaluate the existing damped residual model using every saved singular value.
All numerical identity differences are descriptive, never acceptance gates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
NAMES = (
    "n00005_i00004_extended_step00000.json",
    "n00005_i00004_extended_step00050.json",
    "n00005_i00004_original_step00001.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(out, name, payload):
    nonfinite = []

    def clean(value, field="$"):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): clean(v, field + "." + str(k)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v, field + "[" + str(i) + "]") for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append({"field": field, "original_value": repr(value)})
            return None
        return value

    result = clean(payload)
    result["serialization"] = {"nonfinite_values_replaced_by_null": nonfinite,
        "note": "Undefined ratios use NaN internally and annotated null in JSON; no epsilon or numerical cutoff."}
    temp = out / (name + ".tmp")
    with temp.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(strict=True), args.out.resolve()
    if out.exists():
        parser.error("--out must be fresh")
    out.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        import numpy as np
        base = repo / "outputs/sho/n2_n3_basin_local_discrimination_20261013/local_models"
        manifest_path = base / "manifest.json"
        artifact_manifest_path = base / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifact_manifest = json.loads(artifact_manifest_path.read_text())["sha256"]
        docs = {name: json.loads((base / name).read_text()) for name in NAMES}
        source_traces = {Path(doc["source_trace"]) for doc in docs.values()}
        paths = [manifest_path, artifact_manifest_path, *(base / n for n in NAMES), *sorted(source_traces)]
        before = {str(p): sha256(p) for p in paths}
        script_hash = sha256(Path(__file__))
        for path in (manifest_path, *(base / n for n in NAMES)):
            if before[str(path)] != artifact_manifest.get(path.name):
                raise RuntimeError("ALERT: local-model artifact differs from recorded manifest: " + str(path))
        for path in source_traces:
            if before[str(path)] != manifest["input_sha256"].get(str(path)):
                raise RuntimeError("ALERT: source trace differs from phaseA input hash: " + str(path))
        source_rows = {}
        for path in source_traces:
            wanted = {int(docs[n]["identity"]["saved_step"]) + offset
                      for n in NAMES if Path(docs[n]["source_trace"]) == path for offset in (0, 1)}
            found = {}
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["step"] in wanted:
                        if row["step"] in found:
                            raise RuntimeError("ALERT: duplicate relied-on source state")
                        found[row["step"]] = row
            if set(found) != wanted:
                raise RuntimeError("BLOCKER: required current/next saved trace states unavailable")
            source_rows[path] = found
        save(out, "manifest.json", {"inputs": before, "script": str(Path(__file__).resolve()),
            "script_sha256": script_hash, "selection": NAMES,
            "environment": {"python": sys.version, "numpy": np.__version__},
            "method": "Use saved r,J,U,s,Vh,mu,d,accepted displacement,r_new. No SVD recomputation, solve, objective evaluation, altered damping, rank truncation or recovery.",
            "identities": "lambda=(M/2)mu; c=U.T@r; retained_i=c_i*lambda/(sigma_i^2+lambda); orthogonal=r-Uc. All modes included.",
            "limits": "Numerical identity discrepancies use NumPy float64 on saved arrays originally produced with Torch. Report differences without tolerances or pass/fail classifications. Full integrity is coordinator-owned."})

        def bits_equal(a, b):
            left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            return left.shape == right.shape and np.array_equal(left.view(np.uint64), right.view(np.uint64))

        def norm(a):
            return float(np.linalg.norm(a))

        def energy(a):
            return float(a @ a)

        def cosine(a, b):
            denominator = norm(a) * norm(b)
            return float(a @ b) / denominator if denominator != 0 else float("nan")

        def fraction(numerator, denominator, denominator_label):
            return {"numerator_squared_energy": numerator,
                    "denominator_squared_energy": denominator,
                    "denominator": denominator_label,
                    "fraction": numerator/denominator if denominator != 0 else float("nan")}

        for name, doc in docs.items():
            trial = doc["route_comparisons"]["stabilized"]["trial"]
            model = doc["route_comparisons"]["stabilized"]["accepted_model_comparison"]
            qp = doc["stabilized_qp"]
            if trial["accepted"] is not True or trial["backtrack"] != 0 or any(qp["active"]) or qp["iterations"] != 1:
                raise RuntimeError("BLOCKER: selected state is not the predeclared unconstrained alpha-one case")
            if not model.get("accepted_displacement_available"):
                raise RuntimeError("BLOCKER: saved actual accepted displacement unavailable")
            keys = ("residual", "J", "J_left_singular_vectors", "J_singular_values",
                    "J_right_singular_vectors_transposed", "stabilized_direction")
            if any(key not in doc for key in keys) or "residual_new" not in model:
                raise RuntimeError("BLOCKER: saved raw arrays insufficient; no runtime reconstruction permitted")
            r, J, U, sigma, Vh, d = (np.asarray(doc[key], dtype=np.float64) for key in keys)
            displacement = np.asarray(model["actual_displacement"], dtype=np.float64)
            new_r = np.asarray(model["residual_new"], dtype=np.float64)
            M, N = J.shape
            if r.shape != (M,) or new_r.shape != (M,) or U.shape != (M,N) or sigma.shape != (N,) or Vh.shape != (N,N) or d.shape != (N,) or displacement.shape != (N,):
                raise RuntimeError("BLOCKER: saved array dimensions do not support full-mode accounting")
            if not all(np.isfinite(v).all() for v in (r,J,U,sigma,Vh,d,displacement,new_r)):
                raise RuntimeError("BLOCKER: saved accounting arrays contain nonfinite values")
            mu = float(doc["mu"])
            if not math.isfinite(mu):
                raise RuntimeError("BLOCKER: saved damping is nonfinite")
            identity = doc["identity"]
            rows = source_rows[Path(doc["source_trace"])]
            source_row, next_row = rows[identity["saved_step"]], rows[identity["saved_step"] + 1]
            source_checks = {"phi_bits": bits_equal(doc["phi"], source_row["phi"]),
                "loss_bits": bits_equal(doc["loss"], source_row["loss"]),
                "next_phi_bits": bits_equal(trial["phi"], next_row["phi"]),
                "next_loss_bits": bits_equal(trial["loss"], next_row["loss"])}
            if not all(source_checks.values()):
                raise RuntimeError("ALERT: raw accounting artifact disagrees with hashed source trajectory")
            lam = M / 2 * mu
            c = U.T @ r
            orthogonal = r - U @ c
            with np.errstate(divide="ignore", invalid="ignore"):
                retention = lam / (sigma * sigma + lam)
                retained = c * retention
                retained_energy_fraction = retained * retained / (c * c)
            modal_linear = orthogonal + U @ retained
            full_linear = r + J @ d
            accepted_linear = r + J @ displacement
            nonlinear_error = new_r - accepted_linear
            actual_c = U.T @ new_r
            nonlinear_c = U.T @ nonlinear_error
            nonlinear_orthogonal = nonlinear_error - U @ nonlinear_c
            accepted_c = U.T @ accepted_linear
            accepted_orthogonal = accepted_linear - U @ accepted_c
            start_energy, full_energy, accepted_energy = energy(r), energy(full_linear), energy(accepted_linear)
            nonlinear_energy, next_energy = energy(nonlinear_error), energy(new_r)
            cross_term = float(2 * accepted_linear @ nonlinear_error)
            modal_energy = energy(retained) + energy(orthogonal)
            description = {
                "identity": identity, "source_artifact": str(base / name),
                "source_trace": doc["source_trace"], "source_checks": source_checks,
                "case_structure": {"active_set": qp["active"], "active_set_iterations": qp["iterations"],
                    "accepted_backtrack": trial["backtrack"], "alpha": 2.0 ** (-trial["backtrack"]),
                    "exact_current_bound_coordinates": doc["bounds"]["exact_active"]},
                "M": M, "N": N, "mu": mu, "lambda": lam,
                "all_singular_values": sigma, "residual_coefficients_UTr": c,
                "retention_factors": retention, "retained_coefficients": retained,
                "removed_coefficients": c - retained,
                "mode_initial_squared_energy": c*c, "mode_retained_squared_energy": retained*retained,
                "mode_retained_energy_fraction": retained_energy_fraction,
                "orthogonal_residual": orthogonal, "orthogonal_residual_norm": norm(orthogonal),
                "modal_formula_linear_residual": modal_linear,
                "full_direction": d, "full_direction_linear_residual": full_linear,
                "actual_accepted_displacement": displacement, "accepted_linear_residual": accepted_linear,
                "actual_next_residual": new_r, "actual_nonlinear_residual_error": nonlinear_error,
                "actual_next_modal_coefficients": actual_c,
                "accepted_linear_modal_coefficients": accepted_c,
                "nonlinear_error_modal_coefficients": nonlinear_c,
                "nonlinear_error_orthogonal_component": nonlinear_orthogonal,
                "energy_accounting": {"start_squared_norm": start_energy,
                    "modal_initial_squared_norm_plus_orthogonal": energy(c)+energy(orthogonal),
                    "modal_retained_squared_norm_plus_orthogonal": modal_energy,
                    "modal_formula_vector_squared_norm": energy(modal_linear),
                    "full_direction_linear_squared_norm": full_energy,
                    "accepted_linear_squared_norm": accepted_energy,
                    "nonlinear_error_squared_norm": nonlinear_energy,
                    "twice_linear_dot_nonlinear": cross_term,
                    "actual_next_squared_norm": next_energy,
                    "actual_next_energy_minus_decomposed_energy": next_energy - (accepted_energy + cross_term + nonlinear_energy)},
                "energy_fractions": {
                    "damping_retained_modal_over_start": fraction(energy(retained), start_energy, "||r||^2"),
                    "orthogonal_over_start": fraction(energy(orthogonal), start_energy, "||r||^2"),
                    "damping_retained_modal_over_initial_modal": fraction(energy(retained), energy(c), "sum_i c_i^2, every saved singular component"),
                    "damping_retained_modal_share_of_modal_model": fraction(energy(retained), modal_energy, "sum_i retained_i^2 + ||r-U(U^T r)||^2"),
                    "orthogonal_share_of_modal_model": fraction(energy(orthogonal), modal_energy, "sum_i retained_i^2 + ||r-U(U^T r)||^2"),
                    "per_mode_retained_over_initial": {"numerator_squared_energy": retained*retained,
                        "denominator_squared_energy": c*c, "denominator": "c_i^2 for each saved singular component",
                        "fraction": retained_energy_fraction}},
                "loss_accounting": {"saved_start_loss": doc["loss"], "saved_next_loss": trial["loss"],
                    "array_start_MSE": start_energy/M, "array_next_MSE": next_energy/M,
                    "modal_formula_MSE": modal_energy/M,
                    "full_direction_linear_MSE": full_energy/M, "accepted_linear_MSE": accepted_energy/M,
                    "nonlinear_error_MSE_term": nonlinear_energy/M, "nonlinear_cross_MSE_term": cross_term/M,
                    "actual_saved_decrease": doc["loss"] - trial["loss"],
                    "accepted_linear_array_predicted_decrease": (start_energy-accepted_energy)/M,
                    "accepted_damped_array_predicted_decrease": (start_energy-accepted_energy)/M - mu/2*energy(displacement)},
                "alignment": {"nonlinear_error_dot_accepted_linear": float(nonlinear_error @ accepted_linear),
                    "nonlinear_error_cosine_accepted_linear": cosine(nonlinear_error, accepted_linear),
                    "nonlinear_error_cosine_modal_formula": cosine(nonlinear_error, modal_linear),
                    "nonlinear_error_cosine_original_orthogonal": cosine(nonlinear_error, orthogonal)},
                "descriptive_identity_differences": {
                    "lambda_minus_saved_regularization": lam - doc["residual_regularization"],
                    "coefficients_minus_saved_coefficients_norm": norm(c - np.asarray(doc["residual_singular_components"])),
                    "orthogonal_minus_saved_orthogonal_norm": norm(orthogonal - np.asarray(doc["orthogonal_residual"])),
                    "U_orthonormality_Frobenius": norm(U.T@U-np.eye(N)),
                    "V_orthonormality_Frobenius": norm(Vh@Vh.T-np.eye(N)),
                    "J_minus_saved_SVD_reconstruction_Frobenius": norm(J-(U*sigma)@Vh),
                    "full_linear_minus_modal_formula_norm": norm(full_linear-modal_linear),
                    "full_linear_modal_coefficients_minus_retained_norm": norm(U.T@full_linear-retained),
                    "accepted_displacement_minus_alpha_direction_norm": norm(displacement-(2.0**(-trial["backtrack"]))*d),
                    "accepted_linear_minus_full_linear_norm": norm(accepted_linear-full_linear),
                    "accepted_linear_minus_saved_linear_norm": norm(accepted_linear-np.asarray(model["residual_linear_model"])),
                    "nonlinear_error_minus_saved_error_norm": norm(nonlinear_error-np.asarray(model["residual_nonlinear_error"])),
                    "accepted_orthogonal_minus_original_orthogonal_norm": norm(accepted_orthogonal-orthogonal),
                    "actual_modal_minus_linear_plus_nonlinear_norm": norm(actual_c-(accepted_c+nonlinear_c)),
                    "array_start_MSE_minus_saved_loss": start_energy/M-doc["loss"],
                    "array_next_MSE_minus_saved_loss": next_energy/M-trial["loss"]},
                "interpretation_limit": "Algebraic description of one existing step at saved damping. No causal intervention, alternative damping test, convergence claim or bottleneck classification. No numeric identity difference is used as a pass/fail gate.",
            }
            save(out, name, description)
            completed.append(name)
            save(out, "progress.json", {"completed": completed})
            print("ACCOUNTED " + name, flush=True)
        after = {str(p): sha256(p) for p in paths}
        unchanged = before == after and script_hash == sha256(Path(__file__))
        save(out, "complete.json", {"completed": completed, "inputs_and_script_unchanged": unchanged,
            "input_sha256_after": after, "no_runtime_or_objective_calls": True})
        if not unchanged:
            raise RuntimeError("ALERT: relied-on saved inputs/script changed")
        save(out, "artifact_manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    except BaseException as exc:
        save(out, "failure.json", {"completed": completed, "error": str(exc),
            "exception_type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
