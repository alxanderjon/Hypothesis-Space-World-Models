#!/usr/bin/env python3
"""Frozen eight-arm retained-versus-refit direct joint-path diagnostic."""
from fractions import Fraction
import fcntl
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize._lsq.common import CL_scaling_vector, update_tr_radius

from experiments.sho.exact_rational_bvls import exact_bvls
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import residual_and_jacobian
from experiments.sho.run_checkpoint_controller_factorial16_initializers import clean, save, sha, bits_equal


BASE = Path("outputs/sho/autonomous_generic_sho_20261014")
PROTOCOL = BASE / "protocols/generic_n2_n3_joint_head_path8_short.json"
DESIGN = BASE / "reviews/generic_n2_n3_joint_head_path8_short_design.md"
PREFLIGHT = BASE / "diagnostics/generic_n2_n3_head_state_persistence16"
FACTORIAL = BASE / "screens/generic_n2_n3_checkpoint_controller_factorial16"
SOURCE_PROTOCOL = BASE / "protocols/generic_n2_n3_checkpoint_controller_factorial16_design.json"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = BASE / "diagnostics/generic_n2_n3_joint_head_path8_short"
TRIAL_CAP = ACCEPT_CAP = MODEL_CAP = 11
INNER_CAP = 200
HANDOFF_GATE = 1e-20
BASELINE_P12 = {
    "companion_6": 8.717978797030247e-08,
    "interior_49": 1.2105972832136053e-16,
    "companion_14": 4.645076411857771e-10,
    "companion_29": 1.6784210663124046e-08,
}


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(clean(value), allow_nan=False) + "\n")


def pack_exact(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, dict):
        return {key: pack_exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack_exact(item) for item in value]
    return value


def artifact_ok(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text())["sha256"]
    return all(sha(directory / name) == digest for name, digest in manifest.items())


def first_profile(directory):
    with (directory / "profiles.jsonl").open() as stream:
        return json.loads(next(stream))


def rollout_verify(trajectory, teacher, phi, expected_loss):
    adapter = GenericSHOUpdateNetworkAdapter()
    phi_np = np.asarray(phi, np.float64)
    with torch.no_grad():
        replay = float(adapter.objective(phi_np, trajectory))
    native_bits = bits_equal(replay, expected_loss) and replay < HANDOFF_GATE
    predicted = adapter.rollout(phi_np, trajectory[-1], 4096).numpy()
    truth = np.empty_like(predicted)
    truth[0] = trajectory[-1].numpy()
    matrix = np.asarray(teacher, np.float64).reshape(2, 2)
    for index in range(4096):
        truth[index + 1] = matrix @ truth[index]
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted - truth
        return {
            "independent_native_replay": replay,
            "native_bits": native_bits,
            "steps": 4096,
            "continuation_max_error": float(np.max(np.abs(error))),
            "continuation_mse": float(np.mean(error * error)),
            "rollout_finite": bool(np.isfinite(predicted).all()),
            "bit_exact": bool(np.array_equal(predicted.view(np.uint64), truth.view(np.uint64))),
            "adapter_calls": dict(adapter.calls),
        }


class Arm:
    def __init__(self, out, case, head_state, source_phi, source_loss, trajectory, teacher, protected):
        self.out = out
        self.case = case
        self.head_state = head_state
        self.source_phi = torch.tensor(np.asarray(source_phi, np.float64), dtype=torch.float64)
        self.source_loss = float(source_loss)
        self.trajectory = trajectory
        self.teacher = teacher
        self.protected = protected
        self.adapter = GenericSHOUpdateNetworkAdapter()
        self.lower = torch.from_numpy(self.adapter.lower.copy())
        self.upper = torch.from_numpy(self.adapter.upper.copy())
        self.phi = None
        self.loss = None
        self.r = None
        self.J = None
        self.scale = None
        self.Delta = None
        self.models = 0
        self.trials = 0
        self.accepted = 0
        self.rejected = 0
        self.derivatives = 0
        self.direct_objectives = 0
        self.terminal_profiles = 0
        self.stop = None
        self.trial_rows = []
        self.handoff = None
        self.started = time.monotonic()

    def work(self):
        return {
            "source_native_replays": 1,
            "joint_trial_native_objectives": self.direct_objectives,
            "exact_joint_models": self.models,
            "center_derivative_evaluations": self.derivatives,
            "accepted_updates": self.accepted,
            "rejected_proposals": self.rejected,
            "terminal_exact_head_profiles": self.terminal_profiles,
            "rollouts": int(self.handoff is not None),
            "adapter_calls": dict(self.adapter.calls),
        }

    def center_derivatives(self):
        self.derivatives += 1
        r, J = residual_and_jacobian(self.adapter, self.trajectory, self.phi)
        replay = float((r * r).sum() / r.numel())
        if not bits_equal(replay, self.loss):
            raise RuntimeError("ALERT: center residual/native mismatch")
        if not bool(torch.isfinite(J).all() and torch.isfinite(r).all()):
            raise RuntimeError("NUMERICAL_STOP: nonfinite center derivatives")
        hidden_g = J[:, :12].T @ r
        v, dv = CL_scaling_vector(
            self.phi[:12].numpy(), hidden_g.numpy(), self.lower[:12].numpy(), self.upper[:12].numpy()
        )
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise RuntimeError("NUMERICAL_STOP: invalid Coleman-Li scaling")
        d = np.sqrt(v)
        self.r, self.J = r, J
        self.scale = {"hidden_gradient": hidden_g, "ColemanLi_v": v, "ColemanLi_dv": dv, "d": d}

    def initial_radius(self):
        d = self.scale["d"]
        if np.any(d == 0):
            raise RuntimeError("NUMERICAL_STOP: zero initial Coleman-Li scale")
        Delta = float(np.linalg.norm(self.phi[:12].numpy() / d))
        if Delta == 0:
            Delta = 1.0
        if not math.isfinite(Delta) or Delta <= 0:
            raise RuntimeError("NUMERICAL_STOP: invalid initial radius")
        return Delta

    def exact_joint_model(self):
        if self.models >= MODEL_CAP:
            raise RuntimeError("ALERT: exact-joint model cap entry attempted")
        d = torch.from_numpy(self.scale["d"].copy())
        box_lower, box_upper = self.lower.clone(), self.upper.clone()
        box_lower[:12] = torch.maximum(self.lower[:12], self.phi[:12] - self.Delta * d)
        box_upper[:12] = torch.minimum(self.upper[:12], self.phi[:12] + self.Delta * d)
        fixed_hidden = torch.where(box_lower[:12] == box_upper[:12])[0]
        fixed_mask = torch.zeros(22, dtype=torch.bool)
        fixed_mask[fixed_hidden] = True
        keep = torch.where(~fixed_mask)[0]
        if not bool(((self.phi >= box_lower) & (self.phi <= box_upper)).all()):
            raise RuntimeError("ALERT: joint-model zero baseline infeasible")
        relative_lower, relative_upper = box_lower - self.phi, box_upper - self.phi
        self.models += 1
        solution = exact_bvls(
            [[Fraction(value) for value in row] for row in self.J[:, keep].tolist()],
            [Fraction(value) for value in self.r.tolist()],
            [Fraction(value) for value in relative_lower[keep].tolist()],
            [Fraction(value) for value in relative_upper[keep].tolist()],
            max_iterations=INNER_CAP,
        )
        cert = f"exact_joint_{self.models:02d}.json"
        save(
            self.out / cert,
            {
                "model_index": self.models,
                "kept_indices": keep,
                "fixed_hidden_indices": fixed_hidden,
                "certificate": pack_exact(solution),
            },
        )
        if not all(solution[key] is True for key in ("exact_feasible", "exact_KKT", "full_rank_exact")):
            raise RuntimeError("TECHNICAL_STOP: exact joint certificate failed")
        raw = torch.zeros(22, dtype=torch.float64)
        raw[keep] = torch.tensor([float(value) for value in solution["direction"]], dtype=torch.float64)
        candidate = self.phi + raw
        candidate = torch.where(raw == relative_lower, box_lower, candidate)
        candidate = torch.where(raw == relative_upper, box_upper, candidate)
        step = candidate - self.phi
        if not bool(torch.isfinite(candidate).all()) or not bool(((candidate >= box_lower) & (candidate <= box_upper)).all()):
            raise RuntimeError("NUMERICAL_STOP: invalid realized joint candidate")
        if bool((step[fixed_hidden] != 0).any()):
            raise RuntimeError("ALERT: fixed hidden coordinate moved")
        scaled = torch.zeros(12, dtype=torch.float64)
        positive = d > 0
        if bool((step[:12][~positive] != 0).any()):
            raise RuntimeError("ALERT: zero-scale hidden coordinate moved")
        scaled[positive] = step[:12][positive].abs() / d[positive]
        step_norm = float(scaled.max())
        model_r = self.r + self.J @ step
        prediction = float(0.5 * ((self.r @ self.r) - (model_r @ model_r)))
        if not math.isfinite(prediction) or prediction <= 0 or not math.isfinite(step_norm):
            raise RuntimeError("NUMERICAL_STOP: nonpositive/nonfinite joint-model prediction")
        return {
            "model_index": self.models,
            "candidate": candidate,
            "step": step,
            "radius": self.Delta,
            "d": d,
            "box_lower": box_lower,
            "box_upper": box_upper,
            "scaled_hidden_displacement": scaled,
            "scaled_hidden_infinity_norm": step_norm,
            "boundary_hit": step_norm > 0.95 * self.Delta,
            "prediction_half_sum_squares": prediction,
            "prediction_MSE": 2 * prediction / self.r.numel(),
            "active": solution["active"],
            "certificate": cert,
            "certificate_sha256": sha(self.out / cert),
        }

    def exact_profile(self, hidden):
        self.terminal_profiles += 1
        h = torch.as_tensor(np.asarray(hidden, np.float64).copy())
        zero = torch.cat((h, torch.zeros(10, dtype=torch.float64)))
        _, J0 = residual_and_jacobian(self.adapter, self.trajectory, zero)
        A = J0[:, 12:]
        anchor = -self.trajectory[1:].reshape(-1)
        solution = exact_bvls(
            [[Fraction(value) for value in row] for row in A.tolist()],
            [Fraction(value) for value in anchor.tolist()],
            [Fraction(value) for value in self.lower[12:].tolist()],
            [Fraction(value) for value in self.upper[12:].tolist()],
            max_iterations=INNER_CAP,
        )
        cert = "exact_terminal_head.json"
        save(self.out / cert, {"certificate": pack_exact(solution)})
        if not all(solution[key] is True for key in ("exact_feasible", "exact_KKT", "full_rank_exact")):
            raise RuntimeError("TECHNICAL_STOP: terminal exact-head certificate failed")
        head = torch.tensor([float(value) for value in solution["direction"]], dtype=torch.float64)
        phi = torch.cat((h, head))
        with torch.no_grad():
            loss = float(self.adapter.objective(phi, self.trajectory))
        return {
            "phi": phi,
            "loss": loss,
            "head": head,
            "active": solution["active"],
            "certificate": cert,
            "certificate_sha256": sha(self.out / cert),
            "direct_head_to_profiled_head_norm": float(torch.linalg.vector_norm(self.phi[12:] - head)),
        }

    def verify_handoff(self, phi, loss, origin):
        verification = rollout_verify(self.trajectory, self.teacher, phi, loss)
        if not verification["native_bits"]:
            raise RuntimeError("ALERT: independent handoff verification failed")
        self.handoff = {"origin": origin, "loss": loss, "phi": phi, "verification": verification}
        save(self.out / "handoff_verification_and_rollout.json", self.handoff)

    def persist(self):
        save(
            self.out / "diagnostic_state.json",
            {
                "case": self.case,
                "head_state": self.head_state,
                "current_phi": self.phi,
                "current_loss": self.loss,
                "radius": self.Delta,
                "stop_reason": self.stop,
                "work": self.work(),
                "trials": self.trial_rows,
                "automatic_resume": False,
            },
        )

    def run(self):
        self.out.mkdir(parents=True, exist_ok=False)
        save(
            self.out / "setup.json",
            {
                "case": self.case,
                "head_state": self.head_state,
                "source_phi": self.source_phi,
                "source_loss": self.source_loss,
                "caps": {"joint_trials": TRIAL_CAP, "joint_models": MODEL_CAP, "accepted": ACCEPT_CAP, "inner": INNER_CAP},
                "input_sha256": self.protected,
                "no_head_profile_during_path": True,
            },
        )
        terminal = None
        try:
            with torch.no_grad():
                replay = float(self.adapter.objective(self.source_phi, self.trajectory))
            if not bits_equal(replay, self.source_loss):
                raise RuntimeError("ALERT: source native replay mismatch")
            self.phi, self.loss = self.source_phi.clone(), replay
            self.center_derivatives()
            self.Delta = self.initial_radius()
            save(self.out / "original_center.json", {"phi": self.phi, "loss": self.loss, "scale": self.scale, "initial_radius": self.Delta})
            self.persist()
            seen = set()
            while self.trials < TRIAL_CAP and self.accepted < ACCEPT_CAP and self.handoff is None:
                model = self.exact_joint_model()
                candidate = model["candidate"]
                key = candidate.numpy().tobytes()
                if bits_equal(candidate, self.phi) or key in seen:
                    self.stop = "FLOAT64_UNCHANGED_OR_REPEATED_FULL22_MODEL_CANDIDATE"
                    break
                seen.add(key)
                self.trials += 1
                with torch.no_grad():
                    candidate_loss = float(self.adapter.objective(candidate, self.trajectory))
                self.direct_objectives += 1
                actual = (self.loss - candidate_loss) * self.r.numel() / 2
                scipy_delta, ratio = update_tr_radius(
                    self.Delta,
                    actual,
                    model["prediction_half_sum_squares"],
                    model["scaled_hidden_infinity_norm"],
                    model["boundary_hit"],
                )
                scipy_valid = math.isfinite(scipy_delta) and scipy_delta > 0
                accepted = candidate_loss < self.loss and scipy_valid
                row = {
                    "trial": self.trials,
                    "model_index": model["model_index"],
                    "accepted": accepted,
                    "old_loss": self.loss,
                    "candidate_loss": candidate_loss,
                    "actual_MSE_decrease": self.loss - candidate_loss,
                    "prediction_MSE": model["prediction_MSE"],
                    "ratio": ratio,
                    "radius_before": self.Delta,
                    "radius_after": scipy_delta,
                    "radius_valid": scipy_valid,
                    "scaled_hidden_infinity_norm": model["scaled_hidden_infinity_norm"],
                    "boundary_hit": model["boundary_hit"],
                    "candidate": candidate,
                    "step": model["step"],
                    "active": model["active"],
                    "certificate": model["certificate"],
                    "certificate_sha256": model["certificate_sha256"],
                }
                self.trial_rows.append(row)
                append(self.out / "trials.jsonl", row)
                if not scipy_valid:
                    self.stop = "NUMERICAL_TRUST_RADIUS_EXHAUSTED"
                    break
                self.Delta = scipy_delta
                if accepted:
                    self.phi, self.loss = candidate, candidate_loss
                    self.accepted += 1
                    if self.loss < HANDOFF_GATE:
                        self.verify_handoff(self.phi, self.loss, "direct_joint_path")
                        self.stop = "VERIFIED_NATIVE_HANDOFF_DIRECT"
                        break
                    self.center_derivatives()
                    seen = set()
                else:
                    self.rejected += 1
                self.persist()
            if self.stop is None:
                self.stop = "RESOURCE_LIMIT_11_JOINT_TRIALS" if self.trials >= TRIAL_CAP else "RESOURCE_LIMIT_11_ACCEPTED_UPDATES"
            if self.handoff is None:
                terminal = self.exact_profile(self.phi[:12].numpy())
                if terminal["loss"] < HANDOFF_GATE:
                    self.verify_handoff(terminal["phi"], terminal["loss"], "terminal_exact_head_profile")
                    self.stop = "VERIFIED_NATIVE_HANDOFF_TERMINAL_PROFILE"
            status = "VERIFIED_HANDOFF" if self.handoff is not None else (
                "COMPLETE_RESOURCE_CAP" if self.stop.startswith("RESOURCE_LIMIT") else "NUMERICAL_STOP"
            )
        except BaseException as exc:
            classification = "ALERT" if str(exc).startswith("ALERT") else (
                "NUMERICAL_STOP" if str(exc).startswith("NUMERICAL_STOP") else "TECHNICAL_UNAVAILABLE"
            )
            self.stop = f"{classification}: {exc}"
            status = classification
            error = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        self.persist()
        result = {
            "status": status,
            "case": self.case,
            "head_state": self.head_state,
            "stop_reason": self.stop,
            "source_loss": self.source_loss,
            "final_direct_loss": self.loss,
            "terminal_profile": terminal,
            "handoff": self.handoff,
            "work": self.work(),
            "elapsed_seconds": time.monotonic() - self.started,
            "limits": "Fixed 11 direct full22 joint trials; terminal profile is audit-only and is not continued.",
        }
        if "error" in locals():
            result.update(error)
        checks = {path: sha(Path(path)) == digest for path, digest in self.protected.items()}
        save(self.out / "integrity.json", {"unchanged": all(checks.values()), "checks": checks})
        if not all(checks.values()):
            raise RuntimeError("ALERT: protected input changed during arm")
        save(self.out / "result.json", result)
        save(self.out / "artifact_manifest.json", {"sha256": {p.name: sha(p) for p in sorted(self.out.iterdir()) if p.is_file()}})
        print(
            f"ARM {self.case} {self.head_state} {status} trials={self.trials} accepted={self.accepted} "
            f"direct={self.loss} terminal={None if terminal is None else terminal['loss']}",
            flush=True,
        )
        return result


def source_specs():
    return json.loads(SOURCE_PROTOCOL.read_text())["source_eligibility"]["cases"]


def summarize(rows):
    grouped = {}
    retained_wins = 0
    baseline_wins = 0
    complete_pairs = 0
    for case in source_specs():
        retained = next(row for row in rows if row["case"] == case and row["head_state"] == "retained_N2_head")
        refit = next(row for row in rows if row["case"] == case and row["head_state"] == "global_exact_refit_head")
        a = retained["terminal_profile"]
        b = refit["terminal_profile"]
        if a is not None and b is not None:
            complete_pairs += 1
            ratio = a["loss"] / b["loss"]
            retained_material_win = ratio < 1 / 1.01
            baseline_tie_or_win = a["loss"] <= BASELINE_P12[case]
            retained_wins += int(retained_material_win)
            baseline_wins += int(baseline_tie_or_win)
        else:
            ratio = None
            retained_material_win = False
            baseline_tie_or_win = False
        grouped[case] = {
            "cached_unchanged_exact_profile_baseline_P12": BASELINE_P12[case],
            "retained": retained,
            "refit": refit,
            "terminal_profiled_loss_ratio_retained_over_refit": ratio,
            "retained_materially_beats_refit": retained_material_win,
            "retained_ties_or_beats_cached_baseline_P12": baseline_tie_or_win,
        }
    support = complete_pairs == 4 and retained_wins >= 3 and baseline_wins >= 3
    work_keys = (
        "source_native_replays",
        "joint_trial_native_objectives",
        "exact_joint_models",
        "center_derivative_evaluations",
        "accepted_updates",
        "rejected_proposals",
        "terminal_exact_head_profiles",
        "rollouts",
    )
    total = {key: sum(int(row["work"][key]) for row in rows) for key in work_keys}
    return {
        "status": "JOINT_HEAD_PATH8_SHORT_COMPLETE",
        "declared_arms": 8,
        "completed_records": len(rows),
        "complete_pairs": complete_pairs,
        "retained_material_wins_over_refit": retained_wins,
        "retained_ties_or_wins_vs_cached_baseline_P12": baseline_wins,
        "primary_support_gate_met": support,
        "decision": "SUPPORT_RETAINED_JOINT_PATH" if support else "CLOSE_RETAINED_HEAD_JOINT_CONTINUATION",
        "per_case": grouped,
        "total_work": total,
        "interpretation": "Short paired trajectory diagnostic only; terminal exact profiles are audit-only and were not fed back.",
    }


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if OUT.exists():
        raise RuntimeError("TECHNICAL_UNAVAILABLE: fresh output required; no resume/restart")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "FROZEN_PRE_EXECUTION":
        raise RuntimeError("ALERT: protocol is not frozen for execution")
    if not artifact_ok(PREFLIGHT):
        raise RuntimeError("ALERT: preflight artifact integrity failed")
    lock = (BASE / "active.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("TECHNICAL_UNAVAILABLE: numerical lock is held")
    base_paths = [
        PROTOCOL,
        DESIGN,
        SOURCE_PROTOCOL,
        DATA,
        Path(__file__),
        Path("experiments/sho/exact_rational_bvls.py"),
        Path("experiments/sho/generic_update_network_adapter.py"),
        Path("experiments/sho/generic_update_residual_n3_opt_in.py"),
        PREFLIGHT / "result.json",
        PREFLIGHT / "integrity.json",
        PREFLIGHT / "artifact_manifest.json",
        FACTORIAL / "final_integrity.json",
    ]
    protected = {str(path.resolve()): sha(path) for path in base_paths}
    for spec in source_specs().values():
        directory = BASE / "initializers" / spec["initializer_run_id"]
        if not artifact_ok(directory):
            raise RuntimeError(f"ALERT: initializer artifact integrity failed: {directory}")
        for path in (directory / "artifact_manifest.json", directory / "checkpoints.json", directory / "result.json"):
            protected[str(path.resolve())] = sha(path)
    for case in source_specs():
        baseline_dir = FACTORIAL / "cells" / f"{case}__n2_prefix500__unchanged_baseline"
        if not artifact_ok(baseline_dir):
            raise RuntimeError(f"ALERT: baseline cell artifact integrity failed: {baseline_dir}")
        profiles_path = baseline_dir / "profiles.jsonl"
        profiles = [json.loads(line) for line in profiles_path.read_text().splitlines()]
        cached_p12 = min(float(row["loss"]) for row in profiles[:12])
        if not bits_equal(cached_p12, BASELINE_P12[case]):
            raise RuntimeError(f"ALERT: cached profile-12 incumbent mismatch: {case}")
        for path in (baseline_dir / "artifact_manifest.json", baseline_dir / "result.json", profiles_path):
            protected[str(path.resolve())] = sha(path)
    rows = []
    try:
        OUT.mkdir(parents=True, exist_ok=False)
        save(
            OUT / "batch_started.json",
            {
                "status": "RUNNING",
                "protocol_sha256": sha(PROTOCOL),
                "design_sha256": sha(DESIGN),
                "input_sha256": protected,
                "sequential": True,
                "maximum_work": protocol["maximum_total_work"],
            },
        )
        with np.load(DATA) as corpus:
            for case, spec in source_specs().items():
                directory = BASE / "initializers" / spec["initializer_run_id"]
                checkpoints = json.loads((directory / "checkpoints.json").read_text())
                retained = checkpoints["n2_prefix500"]
                baseline_dir = FACTORIAL / "cells" / f"{case}__n2_prefix500__unchanged_baseline"
                refit = first_profile(baseline_dir)
                retained_phi = np.asarray(retained["theta"], np.float64)
                refit_phi = np.asarray(refit["phi"], np.float64)
                if not bits_equal(retained_phi[:12], refit_phi[:12]):
                    raise RuntimeError("ALERT: paired source hidden mismatch")
                trajectory = torch.from_numpy(corpus[spec["trajectory_key"]].copy())
                teacher = corpus["teachers"][int(spec["case_id"])]
                for head_state, phi, loss in (
                    ("retained_N2_head", retained_phi, retained["loss"]),
                    ("global_exact_refit_head", refit_phi, refit["loss"]),
                ):
                    arm_out = OUT / "arms" / f"{case}__{head_state}"
                    result = Arm(arm_out, case, head_state, phi, loss, trajectory, teacher, protected).run()
                    rows.append(result)
                    save(OUT / "progress.json", {"completed": rows})
        summary = summarize(rows)
        save(OUT / "result.json", summary)
        checks = {path: sha(Path(path)) == digest for path, digest in protected.items()}
        save(OUT / "batch_integrity.json", {"unchanged": all(checks.values()), "checks": checks})
        if not all(checks.values()):
            raise RuntimeError("ALERT: protected input changed during batch")
        output_hashes = {}
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "final_integrity.json":
                output_hashes[str(path)] = sha(path)
        save(
            OUT / "final_integrity.json",
            {
                "protocol_sha256": sha(PROTOCOL),
                "design_sha256": sha(DESIGN),
                "protected_input_sha256": protected,
                "output_sha256": output_hashes,
                "protected_unchanged": True,
                "output_count": len(output_hashes),
            },
        )
        print(json.dumps(clean({key: value for key, value in summary.items() if key != "per_case"}), indent=2), flush=True)
    except BaseException as exc:
        if OUT.exists():
            save(
                OUT / "batch_failure.json",
                {
                    "status": "FAILED",
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed": rows,
                    "no_retry_or_replacement": True,
                },
            )
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
