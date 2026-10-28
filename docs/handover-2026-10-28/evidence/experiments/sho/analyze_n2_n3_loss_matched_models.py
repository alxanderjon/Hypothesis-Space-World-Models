#!/usr/bin/env python3
"""Read saved N3 states; replay two existing directions without continuation.

Only --out is written. This diagnostic does not replace a repository-wide
integrity guard; the coordinator owns that guard and execution authorization.
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

SELECTION = [
    ("n00005_i00004", "original", 1),
    ("n00005_i00004", "original", 37),
    ("n00005_i00004", "extended", 0),
    ("n00005_i00004", "extended", 50),
    ("n00004_i00003", "original", 10),
    ("n00004_i00003", "extended", 10),
]
HORIZONS = (1, 10, 50, 100, 250)
SOURCE_FILES = (
    "experiments/sho/audit_coordinated_generic_recovery.py",
    "experiments/sho/generic_update_network_adapter.py",
    "experiments/sho/generic_update_residual_n3_opt_in.py",
    "experiments/sho/resume_generic_update_targeted.py",
    "experiments/sho/run_stage1b_n3_gradient_polishing.py",
    "experiments/sho/assemble_n3_learned_polisher.py",
    "experiments/n3_box_qp.py",
    "experiments/geometry/prefix_residual_mse_assembly.py",
    "experiments/resume_hybrid_student_after_ckdtree.py",
    "experiments/sho/run_companion6_stabilized_residual_live.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(out, name, payload):
    """Serialize nonfinite values explicitly without losing their locations."""
    nonfinite = []

    def clean(value, path="$" ):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): clean(v, path + "." + str(k)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v, path + "[" + str(i) + "]") for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append({"field": path, "original_value": repr(value)})
            return None
        return value

    content = clean(payload)
    content["serialization"] = {
        "nonfinite_values_replaced_by_null": nonfinite,
        "note": "Nulls listed here preserve nonfinite diagnostics; no clipping or numerical gate was applied.",
    }
    target = out / name
    temporary = out / (name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        parser.error("--out must be a fresh directory; refusing overwrite/resume")
    # The runtime's frozen primitive loaders resolve paths relative to the repo.
    if not (repo / SOURCE_FILES[0]).is_file():
        parser.error("--repo does not contain the expected recovery runtime")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    completed = []
    try:
        import numpy as np
        import torch
        import scipy
        from experiments.resume_hybrid_student_after_ckdtree import Log
        from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
        from experiments.sho.generic_update_residual_n3_opt_in import (
            OptInResidualN3Runtime, STABILIZED_RESIDUAL, residual_vector,
        )
        from experiments.sho.audit_coordinated_generic_recovery import replay
        from experiments.sho.run_companion6_stabilized_residual_live import bits_equal

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        base = repo / "outputs/sho/n2_budget_discrimination_three_branch_20261013"
        data = repo / "outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz"
        trace_paths = {
            (stem, arm): base / (stem + "_" + arm + "_stabilized.jsonl")
            for stem, arm, _ in SELECTION
        }
        inputs = [data, *trace_paths.values()]
        hashes_before = {str(p): sha256(p) for p in inputs}
        source_hashes = {name: sha256(repo / name) for name in SOURCE_FILES}
        script_hash = sha256(Path(__file__))
        save(out, "manifest.json", {
            "selection": SELECTION, "future_saved_horizons": HORIZONS,
            "repo": str(repo), "script": str(Path(__file__).resolve()),
            "script_sha256": script_hash, "input_sha256": hashes_before,
            "direct_source_sha256": source_hashes,
            "environment": {"python": sys.version, "torch": torch.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "torch_threads": torch.get_num_threads()},
            "method": "One unchanged stabilized and full-Hessian trial replay per selected saved state; no optimization continuation.",
            "limits": "Direct source/input checks supplement coordinator-owned full integrity checks. Thin SVD retains every singular component; no rank truncation, gate, or optimizer change.",
        })
        traces = {}
        for key, path in trace_paths.items():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if [row["step"] for row in rows] != list(range(len(rows))):
                raise RuntimeError("Noncontiguous or duplicated saved steps: " + str(path))
            traces[key] = {row["step"]: row for row in rows}
        with np.load(data) as corpus:
            trajectory = torch.from_numpy(corpus["prefix_6"].copy())

        def predictions(runtime, theta, loss, gradient, H, residual, J, mu, trial):
            if not trial["accepted"]:
                return {"accepted_displacement_available": False}
            candidate = torch.tensor(trial["phi"], dtype=torch.float64)
            displacement = candidate - theta  # actual rounded accepted move
            new_residual = residual_vector(runtime.adapter, trajectory, candidate).detach()
            linear_residual = residual + J @ displacement
            count = residual.numel()
            jd = J @ displacement
            residual_prediction = float(-(2 * residual @ jd + jd @ jd) / count)
            full_H_prediction = float(-(gradient @ displacement + 0.5 * displacement @ H @ displacement))
            penalty = float(mu / 2 * (displacement @ displacement))
            nonlinear_error = new_residual - linear_residual
            actual = loss - trial["loss"]
            return {
                "accepted_displacement_available": True,
                "actual_displacement": displacement, "displacement_norm": float(displacement.norm()),
                "actual_decrease": actual,
                "residual_quadratic_predicted_decrease": residual_prediction,
                "full_H_Taylor_predicted_decrease": full_H_prediction,
                "damping_penalty": penalty,
                "damped_residual_predicted_decrease": residual_prediction - penalty,
                "damped_full_H_predicted_decrease": full_H_prediction - penalty,
                "actual_minus_residual_prediction": actual - residual_prediction,
                "actual_minus_full_H_prediction": actual - full_H_prediction,
                "residual_nonlinear_error_norm": float(nonlinear_error.norm()),
                "residual_new": new_residual, "residual_linear_model": linear_residual,
                "residual_nonlinear_error": nonlinear_error,
                "actual_loss_reconstruction_bits": bits_equal(float((new_residual * new_residual).sum() / count), trial["loss"]),
            }

        for stem, arm, step in SELECTION:
            rows = traces[(stem, arm)]
            saved = rows[step]
            theta = torch.tensor(saved["phi"], dtype=torch.float64)
            runtime = OptInResidualN3Runtime(GenericSHOUpdateNetworkAdapter(), trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
            result, gradient, H, gradient_norm, pre, route, direction, qp = runtime.derivatives_and_direction(theta)
            if pre != "CONTINUE" or runtime.last_residual_state is None:
                raise RuntimeError(f"Saved selected state cannot replay CONTINUE: {stem}/{arm}/{step}: {pre}")
            state = runtime.last_residual_state
            mu = float(qp["damping"])
            checks = {
                "loss_bits": bits_equal(float(result.loss), saved["loss"]),
                "gradient_norm_bits": bits_equal(gradient_norm, saved["gradient_norm"]),
                "mu_bits": bits_equal(mu, saved["mu"]),
                "direction_norm_bits": bits_equal(float(direction.norm()), saved["direction_norm"]),
                "exact_active_bounds": torch.where((theta == runtime.lower) | (theta == runtime.upper))[0].tolist() == saved["active_bounds"],
                "curvature_route": route == saved["route"],
            }
            if not all(checks.values()):
                save(out, "replay_mismatch.json", {"identity": [stem, arm, step], "checks": checks})
                raise RuntimeError("Saved state replay mismatch")
            stabilized_trial = replay(runtime, theta, saved["loss"], gradient, direction)
            stabilized_calls = runtime.objective_evaluations
            if "trial" in saved:
                recorded = saved["trial"]
                for name in ("accepted", "backtrack", "rejections", "reason"):
                    if name in recorded:
                        checks["trial_" + name] = stabilized_trial.get(name) == recorded[name]
                for name in ("phi", "loss", "displacement", "decrease"):
                    if name in recorded:
                        checks["trial_" + name + "_bits"] = name in stabilized_trial and bits_equal(stabilized_trial[name], recorded[name])
            if "full_predicted_decrease" in saved:
                checks["raw_model_prediction_bits"] = bits_equal(-float(qp["quadratic_value"]), saved["full_predicted_decrease"])
            if stabilized_trial["accepted"] and step + 1 in rows:
                checks["next_saved_phi_bits"] = bits_equal(stabilized_trial["phi"], rows[step + 1]["phi"])
                checks["next_saved_loss_bits"] = bits_equal(stabilized_trial["loss"], rows[step + 1]["loss"])
            if not all(checks.values()):
                save(out, "replay_mismatch.json", {"identity": [stem, arm, step], "checks": checks, "trial": stabilized_trial})
                raise RuntimeError("Saved trial replay mismatch")
            full_trial = replay(runtime, theta, saved["loss"], gradient, state["existing_direction"])
            r, J = state["residual"], state["jacobian"]
            U, singular, Vh = torch.linalg.svd(J, full_matrices=False)
            components = U.T @ r
            orthogonal = r - U @ components
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            gn_eigenvalues = 2 / r.numel() * singular.square()
            residual_curvature = H - 2 / r.numel() * (J.T @ J)
            route_results = {}
            for name, trial in (("stabilized", stabilized_trial), ("existing_full_Hessian", full_trial)):
                metrics = predictions(runtime, theta, saved["loss"], gradient, H, r, J, mu, trial)
                if metrics.get("actual_loss_reconstruction_bits") is False:
                    raise RuntimeError("Candidate residual does not reconstruct accepted loss exactly")
                route_results[name] = {"trial": trial, "accepted_model_comparison": metrics}
            future = []
            for horizon in HORIZONS:
                future_row = rows.get(step + horizon)
                future.append({"additional_accepted_steps": horizon,
                               "available": future_row is not None,
                               "saved_step": step + horizon,
                               "loss": None if future_row is None else future_row["loss"]})
            record = {
                "identity": {"stem": stem, "endpoint": arm, "saved_step": step},
                "source_trace": str(trace_paths[(stem, arm)]), "saved_state": saved,
                "exact_checks": checks, "phi": theta, "loss": saved["loss"],
                "gradient": gradient, "H": H, "residual": r, "J": J,
                "H_eigenvalues": eigenvalues, "H_eigenvectors": eigenvectors,
                "H_exact_negative_eigenvalue_count": int((eigenvalues < 0).sum()),
                "J_singular_values": singular, "J_left_singular_vectors": U,
                "J_right_singular_vectors_transposed": Vh,
                "residual_singular_components": components,
                "residual_singular_component_squares": components.square(),
                "orthogonal_residual": orthogonal,
                "orthogonal_residual_norm": float(orthogonal.norm()),
                "orthogonal_projection_note": "Direct r-U(U^T r), retaining every thin-SVD column; this is not a thresholded numerical-rank estimate.",
                "residual_norm": float(r.norm()), "G_spectrum_from_J": gn_eigenvalues,
                "R": residual_curvature, "R_eigenvalues": torch.linalg.eigvalsh(residual_curvature),
                "mu": mu, "residual_regularization": state["regularization"],
                "mu_over_G_spectrum": mu / gn_eigenvalues,
                "mu_over_abs_H_eigenvalues": mu / eigenvalues.abs(),
                "mu_plus_H_eigenvalues": mu + eigenvalues,
                "mu_minus_negative_H_magnitude": mu - torch.clamp(-eigenvalues, min=0),
                "spectrum_note": "Signs/ratios are descriptive float64 values; no new tolerance, cutoff or gate.",
                "bounds": {"lower": runtime.lower, "upper": runtime.upper,
                           "lower_margins": theta - runtime.lower, "upper_margins": runtime.upper - theta,
                           "exact_active": saved["active_bounds"],
                           "all_in_bounds": bool(((theta >= runtime.lower) & (theta <= runtime.upper)).all())},
                "stabilized_direction": direction, "stabilized_qp": qp,
                "existing_full_Hessian_direction": state["existing_direction"],
                "existing_full_Hessian_qp": state["existing_qp"],
                "route_comparisons": route_results,
                "saved_stabilized_future": future,
                "future_note": "Observed saved stabilized trajectory only; no counterfactual continuation or new optimization.",
                "trial_objective_counts": {"stabilized": stabilized_calls,
                    "existing_full_Hessian": runtime.objective_evaluations - stabilized_calls},
                "derivative_calls": runtime.binding_counters.derivative_calls,
            }
            name = f"{stem}_{arm}_step{step:05d}.json"
            save(out, name, record)
            completed.append(name)
            save(out, "progress.json", {"completed": completed})
            print(f"REPLAYED {stem} {arm} step={step}; stabilized={stabilized_trial.get('loss')}; full_Hessian={full_trial.get('loss')}", flush=True)
            if any(trial.get("accepted") and trial["loss"] < 1e-20 for trial in (stabilized_trial, full_trial)):
                save(out, "ALERT.json", {"reason": "Existing handoff threshold reached in one-step replay; stop remaining probes for user review", "identity": [stem, arm, step], "evidence": name})
                break

        input_after = {str(p): sha256(p) for p in inputs}
        sources_after = {name: sha256(repo / name) for name in SOURCE_FILES}
        unchanged = hashes_before == input_after and source_hashes == sources_after and script_hash == sha256(Path(__file__))
        save(out, "complete.json", {"completed": completed, "selected_states": len(SELECTION),
             "source_and_input_hashes_unchanged": unchanged,
             "input_sha256_after": input_after, "direct_source_sha256_after": sources_after,
             "limits": "No global checkpoint rehash by this driver; coordinator owns full integrity verification. No new trajectory continuation was run."})
        if not unchanged:
            raise RuntimeError("Relied-on direct source/input/script integrity changed")
        save(out, "artifact_manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.glob("*.json"))}})
    except BaseException as exc:
        save(out, "failure.json", {"completed": completed, "exception_type": type(exc).__name__,
             "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
