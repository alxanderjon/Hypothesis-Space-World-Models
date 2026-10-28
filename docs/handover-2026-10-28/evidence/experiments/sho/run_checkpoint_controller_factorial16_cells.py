#!/usr/bin/env python3
"""Execute the frozen 4x2x2 exact N3 checkpoint/controller screen."""
from fractions import Fraction
import fcntl, hashlib, json, math, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize._lsq.common import CL_scaling_vector, update_tr_radius

from experiments.sho.exact_rational_bvls import exact_bvls
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import residual_vector, residual_and_jacobian
from experiments.sho.run_checkpoint_controller_factorial16_initializers import clean, save, sha, bits_equal

ROOT = Path("outputs/sho/autonomous_generic_sho_20261014")
PROTOCOL = ROOT / "protocols/generic_n2_n3_checkpoint_controller_factorial16_design.json"
ADDENDUM = ROOT / "reviews/generic_n2_n3_checkpoint_controller_factorial16_formula_contrast_review.md"
SCREEN = ROOT / "screens/generic_n2_n3_checkpoint_controller_factorial16"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
PROTOCOL_SHA = "e2c55e0d79bdcc4b9999bac5997248c0373c31f57e060a8afdb64fdc153dc6d8"
ADDENDUM_SHA = "75f93d00957863cd1ba7c652d9704dacdb3bb330b26edcaf035deb6b52effdb1"
PROFILE_CAP, MODEL_CAP, ACCEPT_CAP, INNER_CAP = 24, 23, 23, 200
HANDOFF_GATE = 1e-20


def append(path, value):
    with path.open("a") as stream: stream.write(json.dumps(clean(value), allow_nan=False) + "\n")


def pack_exact(x):
    if isinstance(x, Fraction): return f"{x.numerator}/{x.denominator}"
    if isinstance(x, dict): return {k: pack_exact(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [pack_exact(v) for v in x]
    return x


def artifact_ok(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text())["sha256"]
    return all(sha(directory / name) == digest for name, digest in manifest.items())


def rollout_verify(trajectory, teacher, phi, expected_loss):
    adapter = GenericSHOUpdateNetworkAdapter()
    with torch.no_grad(): replay = float(adapter.objective(np.asarray(phi, np.float64), torch.from_numpy(trajectory.copy())))
    exact = bits_equal(replay, expected_loss) and replay < HANDOFF_GATE
    predicted = adapter.rollout(np.asarray(phi, np.float64), trajectory[-1], 4096).numpy()
    truth = np.empty_like(predicted); truth[0] = trajectory[-1]
    matrix = np.asarray(teacher, np.float64).reshape(2, 2)
    for i in range(4096): truth[i + 1] = matrix @ truth[i]
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted - truth
        return {"independent_native_replay": replay, "native_bits": exact, "steps": 4096,
            "continuation_max_error": float(np.max(np.abs(error))),
            "continuation_mse": float(np.mean(error * error)),
            "rollout_finite": bool(np.isfinite(predicted).all()),
            "bit_exact": bool(np.array_equal(predicted.view(np.uint64), truth.view(np.uint64))),
            "adapter_calls": dict(adapter.calls)}


class Handoff(Exception): pass


class Cell:
    def __init__(self, out, case, checkpoint, controller, source, trajectory, teacher, protected):
        self.out, self.case, self.checkpoint, self.controller = out, case, checkpoint, controller
        self.source, self.trajectory, self.teacher = source, trajectory, teacher
        self.protected = protected; self.started = time.monotonic()
        self.adapter = GenericSHOUpdateNetworkAdapter()
        self.lower = torch.from_numpy(self.adapter.lower.copy()); self.upper = torch.from_numpy(self.adapter.upper.copy())
        self.profiles = self.models = self.centers = self.accepted = self.source_calls = 0
        self.head_calls = self.head_completed = self.trial_count = self.rejected = 0
        self.stop = None; self.last_profile = self.last_model = None
        self.current = self.deriv = None; self.Delta = None
        self.incumbent = []; self.trials = []; self.engagement = []

    def work(self):
        return {"profile_evaluations": self.profiles, "exact_head_calls": self.head_calls,
            "exact_head_completed": self.head_completed, "joint_model_solves": self.models,
            "center_derivative_evaluations": self.centers, "accepted_hidden_updates": self.accepted,
            "rejected_proposals": self.rejected, "source_objective_calls": self.source_calls,
            "adapter_calls": dict(self.adapter.calls)}

    def profile(self, hidden, purpose):
        h = torch.as_tensor(np.asarray(hidden, np.float64).copy())
        if not bool(torch.isfinite(h).all()) or not bool(((h >= self.lower[:12]) & (h <= self.upper[:12])).all()):
            raise RuntimeError("TECHNICAL_STOP: invalid hidden candidate")
        if self.profiles >= PROFILE_CAP: raise RuntimeError("ALERT: profile cap entry attempted")
        self.profiles += 1
        zero = torch.cat((h, torch.zeros(10, dtype=torch.float64)))
        zero_r, zero_J = residual_and_jacobian(self.adapter, self.trajectory, zero)
        A = zero_J[:, 12:22]; anchor_r = -self.trajectory[1:].reshape(-1)
        singular = torch.linalg.svdvals(A)
        if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(anchor_r).all()) or not bool(torch.isfinite(singular).all()) or bool((singular <= 0).any()):
            raise RuntimeError("TECHNICAL_STOP: invalid exact-head input")
        self.head_calls += 1
        solution = exact_bvls([[Fraction(v) for v in row] for row in A.tolist()],
            [Fraction(v) for v in anchor_r.tolist()], [Fraction(v) for v in self.lower[12:].tolist()],
            [Fraction(v) for v in self.upper[12:].tolist()], max_iterations=INNER_CAP)
        cert = f"exact_head_{self.profiles:05d}.json"
        save(self.out / cert, {"profile_index": self.profiles, "solver_source_sha256": self.protected[str(Path('experiments/sho/exact_rational_bvls.py').resolve())], "certificate": pack_exact(solution)})
        if not all(solution[k] is True for k in ("exact_feasible", "exact_KKT", "full_rank_exact")):
            raise RuntimeError("TECHNICAL_STOP: missing exact head certificate")
        self.head_completed += 1
        head = torch.tensor([float(v) for v in solution["direction"]], dtype=torch.float64)
        phi = torch.cat((h, head)); active = torch.tensor(solution["active"])
        if not bool(torch.isfinite(phi).all()) or not bool(((phi >= self.lower) & (phi <= self.upper)).all()):
            raise RuntimeError("TECHNICAL_STOP: infeasible exact-head conversion")
        with torch.no_grad():
            loss = float(self.adapter.objective(phi, self.trajectory)); r = residual_vector(self.adapter, self.trajectory, phi).detach()
        if not math.isfinite(loss) or not bool(torch.isfinite(r).all()) or not bits_equal(loss, float((r * r).sum() / r.numel())):
            raise RuntimeError("ALERT: native profile/residual mismatch")
        model_r = anchor_r + A @ head; kkt = A.T @ model_r; native_kkt = A.T @ r
        row = {"profile_index": self.profiles, "purpose": purpose, "hidden": h, "phi": phi, "loss": loss,
            "residual": r, "A": A, "head_singular_values": singular, "head_active": active,
            "head_certificate": cert, "head_certificate_sha256": sha(self.out / cert),
            "linear_residual": model_r, "half_sum_model_head_gradient": kkt,
            "half_sum_native_head_gradient": native_kkt, "native_MSE_head_gradient": 2 / r.numel() * native_kkt,
            "active_bound_identity": bool(((active != -1) | (head == self.lower[12:])).all() and ((active != 1) | (head == self.upper[12:])).all())}
        if not row["active_bound_identity"]: raise RuntimeError("ALERT: head active/bound identity mismatch")
        self.last_profile = row
        append(self.out / "profiles.jsonl", {k: v for k, v in row.items() if k not in ("A", "residual", "linear_residual")})
        if loss < HANDOFF_GATE: raise Handoff("verified-profile candidate below native handoff gate")
        return row

    def derivatives(self, row):
        self.centers += 1
        phi, r = row["phi"], row["residual"]
        rr, J = residual_and_jacobian(self.adapter, self.trajectory, phi)
        if not bits_equal(rr, r) or not bits_equal(J[:, 12:22], row["A"]): raise RuntimeError("ALERT: center replay mismatch")
        flat = phi.detach().clone().requires_grad_(True); native_r = residual_vector(self.adapter, self.trajectory, flat)
        native_loss = (native_r * native_r).sum() / native_r.numel(); native_g = torch.autograd.grad(native_loss, flat)[0].detach()
        if not bits_equal(float(native_loss.detach()), row["loss"]) or not bool(torch.isfinite(J).all()) or not bool(torch.isfinite(native_g).all()):
            raise RuntimeError("ALERT: center derivative mismatch/nonfinite")
        g = J[:, :12].T @ r
        v, dv = CL_scaling_vector(phi[:12].numpy(), g.numpy(), self.adapter.lower[:12], self.adapter.upper[:12])
        if not np.all(np.isfinite(v)) or np.any(v < 0): raise RuntimeError("NUMERICAL_STOP: invalid Coleman-Li scaling")
        d = np.sqrt(v)
        return {"r": r, "J": J, "native_MSE_gradient": native_g,
            "residual_J_MSE_gradient": 2 / r.numel() * (J.T @ r), "half_sum_hidden_gradient": g,
            "ColemanLi_v": v, "ColemanLi_dv": dv, "d": d, "full_J_singular_values": torch.linalg.svdvals(J)}

    def initial_radius(self, row, deriv):
        d = deriv["d"]
        if np.any(d == 0): raise RuntimeError("NUMERICAL_STOP: zero initial Coleman-Li scale")
        delta = float(np.linalg.norm(row["hidden"].numpy() / d))
        if delta == 0: delta = 1.0
        if not math.isfinite(delta) or delta <= 0: raise RuntimeError("NUMERICAL_STOP: invalid initial radius")
        return delta

    def model(self):
        if self.models >= MODEL_CAP: raise RuntimeError("ALERT: joint model cap entry attempted")
        row, deriv, Delta = self.current, self.deriv, self.Delta
        phi, r, J = row["phi"], deriv["r"], deriv["J"]; d = torch.from_numpy(deriv["d"].copy())
        box_lower, box_upper = self.lower.clone(), self.upper.clone()
        box_lower[:12] = torch.maximum(self.lower[:12], phi[:12] - Delta * d)
        box_upper[:12] = torch.minimum(self.upper[:12], phi[:12] + Delta * d)
        fixed = torch.where(box_lower[:12] == box_upper[:12])[0]
        fixed_mask = torch.zeros(22, dtype=torch.bool); fixed_mask[fixed] = True; keep = torch.where(~fixed_mask)[0]
        if not bool(((phi >= box_lower) & (phi <= box_upper)).all()) or not bool((phi[fixed] == box_lower[fixed]).all()):
            raise RuntimeError("ALERT: zero model baseline infeasible")
        if not all(bool(torch.isfinite(x).all()) for x in (J, r, phi, box_lower, box_upper, d)):
            raise RuntimeError("NUMERICAL_STOP: nonfinite model inputs")
        self.models += 1; relative_lower, relative_upper = box_lower - phi, box_upper - phi
        solution = exact_bvls([[Fraction(v) for v in values] for values in J[:, keep].tolist()],
            [Fraction(v) for v in r.tolist()], [Fraction(v) for v in relative_lower[keep].tolist()],
            [Fraction(v) for v in relative_upper[keep].tolist()], max_iterations=INNER_CAP)
        cert = f"exact_model_{self.models:05d}.json"
        save(self.out / cert, {"model_index": self.models, "kept_indices": keep, "fixed_hidden_indices": fixed,
            "solver_source_sha256": self.protected[str(Path('experiments/sho/exact_rational_bvls.py').resolve())], "certificate": pack_exact(solution)})
        if not all(solution[k] is True for k in ("exact_feasible", "exact_KKT", "full_rank_exact")):
            raise RuntimeError("TECHNICAL_STOP: missing exact joint certificate")
        step_reduced = torch.tensor([float(v) for v in solution["direction"]], dtype=torch.float64)
        if not bool(torch.isfinite(step_reduced).all()) or not bool(((step_reduced >= relative_lower[keep]) & (step_reduced <= relative_upper[keep])).all()):
            raise RuntimeError("NUMERICAL_STOP: infeasible exact-joint conversion")
        raw = torch.zeros_like(phi); raw[keep] = step_reduced; model_phi = phi + raw
        lower_hits, upper_hits = raw == relative_lower, raw == relative_upper
        model_phi = torch.where(lower_hits, box_lower, model_phi); model_phi = torch.where(upper_hits, box_upper, model_phi)
        p = model_phi - phi
        if not bool(torch.isfinite(model_phi).all()) or not bool(((model_phi >= box_lower) & (model_phi <= box_upper)).all()):
            raise RuntimeError("NUMERICAL_STOP: invalid realized model output")
        if bool((p[:12][d == 0] != 0).any()) or bool((p[fixed] != 0).any()): raise RuntimeError("ALERT: fixed/zero-scale coordinate moved")
        scaled = torch.zeros(12, dtype=torch.float64); positive = d > 0; scaled[positive] = p[:12][positive].abs() / d[positive]
        step_norm = float(scaled.max()); model_r = r + J @ p; prediction = float(.5 * ((r @ r) - (model_r @ model_r)))
        status = torch.full((22,), 2, dtype=torch.int64); status[keep] = torch.tensor(solution["active"])
        meta = {"model_index": self.models, "center_loss": row["loss"], "radius": Delta, "d": d,
            "box_lower": box_lower, "box_upper": box_upper, "fixed_hidden_indices": fixed, "kept_indices": keep,
            "model_phi": model_phi, "realized_model_step": p, "scaled_hidden_displacement": scaled,
            "realized_hidden_infinity_norm": step_norm, "boundary_hit": step_norm > .95 * Delta,
            "prediction_half_sum_squares": prediction, "prediction_MSE": 2 * prediction / r.numel(),
            "full_model_status": status, "exact_certificate": cert, "exact_certificate_sha256": sha(self.out / cert),
            "model_finite": bool(torch.isfinite(model_r).all()) and math.isfinite(prediction) and math.isfinite(step_norm)}
        self.last_model = meta; append(self.out / "models.jsonl", meta)
        if not meta["model_finite"]: raise RuntimeError("NUMERICAL_STOP: nonfinite model prediction")
        if prediction <= 0: raise RuntimeError("NUMERICAL_STOP: nonpositive realized model prediction; not convergence")
        return meta

    def persist(self):
        save(self.out / "diagnostic_state.json", {"case": self.case, "checkpoint": self.checkpoint,
            "controller": self.controller, "current_phi": None if self.current is None else self.current["phi"],
            "current_loss": None if self.current is None else self.current["loss"], "radius": self.Delta,
            "stop_reason": self.stop, "work": self.work(), "incumbent_by_profile": self.incumbent,
            "controller_engagement": self.engagement, "automatic_resume": False})

    def run(self):
        self.out.mkdir(parents=True, exist_ok=False)
        save(self.out / "setup.json", {"case": self.case, "checkpoint": self.checkpoint, "controller": self.controller,
            "source": self.source, "protocol_sha256": PROTOCOL_SHA, "formula_addendum_sha256": ADDENDUM_SHA,
            "caps": {"profiles": PROFILE_CAP, "models": MODEL_CAP, "accepted": ACCEPT_CAP, "inner": INNER_CAP},
            "native_acceptance": "strict loss decrease and valid unchanged SciPy radius",
            "anchor": "after accepted trial only carry pre-center realized scaled hidden infinity norm; no floor/fallback",
            "input_sha256": self.protected})
        try:
            theta = np.asarray(self.source["theta"], np.float64)
            with torch.no_grad(): replay = float(self.adapter.objective(theta, self.trajectory))
            self.source_calls += 1
            if not bits_equal(replay, self.source["loss"]): raise RuntimeError("ALERT: checkpoint native replay mismatch")
            self.current = self.profile(theta[:12], "N2_checkpoint_zero_anchor_exact_head_profile")
            if self.current["loss"] > replay: raise RuntimeError("TECHNICAL_STOP: exact head refit worsened source")
            self.incumbent = [self.current["loss"]]
            self.deriv = self.derivatives(self.current); self.Delta = self.initial_radius(self.current, self.deriv)
            save(self.out / "original_center.json", {"profile": self.current, "derivatives": self.deriv, "initial_radius": self.Delta})
            self.persist(); seen = set(); pending_engagement = None
            while self.stop is None:
                if self.profiles >= PROFILE_CAP: self.stop = "RESOURCE_LIMIT_24_PROFILE_EVALUATIONS"; break
                if self.models >= MODEL_CAP: self.stop = "RESOURCE_LIMIT_23_JOINT_MODELS"; break
                if self.accepted >= ACCEPT_CAP: self.stop = "RESOURCE_LIMIT_23_ACCEPTED_HIDDEN_UPDATES"; break
                model = self.model(); hidden = model["model_phi"][:12].numpy().copy()
                if np.array_equal(hidden, self.current["hidden"].numpy()) or hidden.tobytes() in seen:
                    self.stop = "FLOAT64_UNCHANGED_OR_REPEATED_HIDDEN_MODEL_CANDIDATE"; break
                seen.add(hidden.tobytes()); self.trial_count += 1
                trial = self.profile(hidden, f"trial{self.trial_count}_accepted_before{self.accepted}")
                actual = (self.current["loss"] - trial["loss"]) * self.current["residual"].numel() / 2
                scipy_delta, ratio = update_tr_radius(self.Delta, actual, model["prediction_half_sum_squares"],
                    model["realized_hidden_infinity_norm"], model["boundary_hit"])
                improves = trial["loss"] < self.current["loss"]
                scipy_valid = math.isfinite(scipy_delta) and scipy_delta > 0
                accepted_trial = improves and scipy_valid
                carried = model["realized_hidden_infinity_norm"] if accepted_trial and self.controller == "accepted_step_radius_anchor" else scipy_delta
                anchor_valid = math.isfinite(carried) and carried > 0
                update = {"trial": self.trial_count, "model_index": model["model_index"], "profile_index": trial["profile_index"],
                    "accepted_before": self.accepted, "accepted": accepted_trial, "old_loss": self.current["loss"],
                    "new_loss": trial["loss"], "actual_MSE_decrease": self.current["loss"] - trial["loss"],
                    "prediction_MSE": model["prediction_MSE"], "ratio": ratio, "radius_before": self.Delta,
                    "realized_scaled_hidden_norm": model["realized_hidden_infinity_norm"],
                    "scipy_radius_after": scipy_delta, "carried_radius_after": carried,
                    "scipy_radius_valid": scipy_valid, "carried_radius_valid": anchor_valid,
                    "boundary_hit": model["boundary_hit"], "model_status": model["full_model_status"],
                    "center_head_active": self.current["head_active"], "trial_head_active": trial["head_active"]}
                if pending_engagement is not None and "subsequent_fidelity" not in pending_engagement:
                    pending_engagement["subsequent_fidelity"] = {"trial": self.trial_count, "prediction_MSE": model["prediction_MSE"],
                        "actual_MSE_decrease": update["actual_MSE_decrease"], "ratio": ratio, "accepted": accepted_trial}
                self.trials.append(update); append(self.out / "trials.jsonl", update)
                if not scipy_valid:
                    self.stop = "NUMERICAL_TRUST_RADIUS_EXHAUSTED"; self.incumbent.append(self.current["loss"]); break
                if accepted_trial:
                    old = self.current; self.current = trial; self.accepted += 1
                    engagement = {"accepted_step": self.accepted, "radius_before": self.Delta,
                        "pre_center_scale": self.deriv["d"], "realized_scaled_hidden_norm": model["realized_hidden_infinity_norm"],
                        "scipy_radius_after": scipy_delta, "carried_radius_after": carried,
                        "boundary_hit": model["boundary_hit"], "model_status": model["full_model_status"],
                        "center_head_active": old["head_active"], "accepted_head_active": trial["head_active"]}
                    self.engagement.append(engagement); pending_engagement = engagement
                    self.incumbent.append(self.current["loss"])
                    if not anchor_valid:
                        self.Delta = carried; self.stop = "NUMERICAL_STOP_INVALID_ACCEPTED_STEP_ANCHOR"; break
                    self.Delta = carried; self.deriv = self.derivatives(self.current); seen = set()
                else:
                    self.rejected += 1; self.Delta = scipy_delta; self.incumbent.append(self.current["loss"])
                self.persist()
            self.persist()
            status = "COMPLETE_RESOURCE_CAP" if self.stop and self.stop.startswith("RESOURCE_LIMIT") else "NUMERICAL_STOP"
            result = self.result(status)
        except Handoff as exc:
            handoff = self.last_profile
            verify = rollout_verify(self.trajectory.numpy(), self.teacher, handoff["phi"].numpy(), handoff["loss"])
            save(self.out / "handoff_verification_and_rollout.json", verify)
            if not verify["native_bits"]: raise RuntimeError("ALERT: independent handoff verification failed")
            self.stop = "VERIFIED_NATIVE_HANDOFF"; self.current = handoff
            if len(self.incumbent) < self.profiles: self.incumbent.append(handoff["loss"])
            self.persist(); result = self.result("VERIFIED_HANDOFF"); result["rollout"] = verify
        except BaseException as exc:
            classification = "ALERT" if str(exc).startswith("ALERT") else ("NUMERICAL_STOP" if str(exc).startswith("NUMERICAL_STOP") else "TECHNICAL_UNAVAILABLE")
            self.stop = f"{classification}: {exc}"
            self.persist(); result = self.result(classification)
            result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        checks = {p: sha(p) == d for p, d in self.protected.items()}
        save(self.out / "integrity.json", {"unchanged": all(checks.values()), "checks": checks})
        if not all(checks.values()): raise RuntimeError("ALERT: protected input changed during cell")
        save(self.out / "result.json", result)
        files = {p.name: sha(p) for p in sorted(self.out.iterdir()) if p.is_file()}
        save(self.out / "artifact_manifest.json", {"sha256": files})
        print(f"CELL {self.case} {self.checkpoint} {self.controller} {result['status']} profiles={self.profiles} accepted={self.accepted} loss={result['final_loss']}", flush=True)
        return result

    def result(self, status):
        initial = self.incumbent[0] if self.incumbent else None
        horizons = {str(k): self.incumbent[k - 1] if len(self.incumbent) >= k else None for k in (1, 8, 16, 24)}
        progress = [math.log10(initial / x) for x in self.incumbent] if initial and all(x > 0 for x in self.incumbent) else []
        return {"status": status, "case": self.case, "checkpoint": self.checkpoint, "controller": self.controller,
            "stop_reason": self.stop, "source_loss": self.source["loss"], "initial_profile_loss": initial,
            "final_loss": None if self.current is None else self.current["loss"], "handoff": status == "VERIFIED_HANDOFF",
            "work": self.work(), "horizon_incumbent_losses": horizons,
            "P24": progress[23] if len(progress) >= 24 else None,
            "observed_progress": progress, "observed_progress_auc_budgets_2_to_last": float(np.mean(progress[1:])) if len(progress) > 1 else None,
            "complete_AUC_2_to_24": float(np.mean(progress[1:24])) if len(progress) >= 24 else None,
            "controller_engagement": self.engagement, "elapsed_seconds": time.monotonic() - self.started,
            "limits": "24-profile short-horizon screen only; no reliability or eventual-recovery claim"}


def summarize(rows):
    by = {(r["case"], r["checkpoint"], r["controller"]): r for r in rows}
    cases = sorted({r["case"] for r in rows}); per_case = {}
    for case in cases:
        table = {f"{cp}|{co}": by[(case, cp, co)] for cp in ("n2_prefix500", "n2_endpoint1500") for co in ("unchanged_baseline", "accepted_step_radius_anchor")}
        effects = {}
        for co in ("unchanged_baseline", "accepted_step_radius_anchor"):
            a, b = table[f"n2_endpoint1500|{co}"], table[f"n2_prefix500|{co}"]
            effects[f"checkpoint_at_{co}"] = None if a["P24"] is None or b["P24"] is None else a["P24"] - b["P24"]
        for cp in ("n2_prefix500", "n2_endpoint1500"):
            a, b = table[f"{cp}|accepted_step_radius_anchor"], table[f"{cp}|unchanged_baseline"]
            effects[f"controller_at_{cp}"] = None if a["P24"] is None or b["P24"] is None else a["P24"] - b["P24"]
        x, y = effects["controller_at_n2_endpoint1500"], effects["controller_at_n2_prefix500"]
        effects["checkpoint_by_controller_interaction"] = None if x is None or y is None else x - y
        per_case[case] = {"cells": {k: {z: v.get(z) for z in ("status", "stop_reason", "source_loss", "initial_profile_loss", "final_loss", "P24", "work", "horizon_incumbent_losses")} for k, v in table.items()}, "P24_effects": effects}
    total = {k: sum(int(r["work"][k]) for r in rows) for k in ("profile_evaluations", "exact_head_calls", "joint_model_solves", "center_derivative_evaluations", "accepted_hidden_updates", "rejected_proposals", "source_objective_calls")}
    return {"status": "FACTORIAL16_COMPLETE", "declared_cells": 16, "completed_records": len(rows),
        "experimental_unit": "four paired cases, not sixteen independent samples", "per_case": per_case,
        "total_N3_work": total, "interpretation": "exploratory 24-profile short-horizon paired effects only"}


def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    if sha(PROTOCOL) != PROTOCOL_SHA or sha(ADDENDUM) != ADDENDUM_SHA: raise RuntimeError("ALERT: frozen design hash mismatch")
    protocol = json.loads(PROTOCOL.read_text()); cells_dir = SCREEN / "cells"
    if cells_dir.exists(): raise RuntimeError("TECHNICAL_UNAVAILABLE: cell output directory exists; no resume/restart")
    lock = (ROOT / "active.lock").open("a+")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("TECHNICAL_UNAVAILABLE: numerical lock is held")
    base_paths = [PROTOCOL, ADDENDUM, DATA, Path(__file__), Path("experiments/sho/exact_rational_bvls.py"),
        Path("experiments/sho/generic_update_network_adapter.py"), Path("experiments/sho/generic_update_residual_n3_opt_in.py")]
    protected = {str(p.resolve()): sha(p) for p in base_paths}
    for spec in protocol["source_eligibility"]["cases"].values():
        d = ROOT / "initializers" / spec["initializer_run_id"]
        if not artifact_ok(d): raise RuntimeError(f"ALERT: initializer artifact changed: {d}")
        for p in (d / "artifact_manifest.json", d / "checkpoints.json", d / "result.json"):
            protected[str(p.resolve())] = sha(p)
    cells_dir.mkdir(parents=True, exist_ok=False); rows = []
    save(SCREEN / "cell_batch_started.json", {"status": "RUNNING", "protocol_sha256": PROTOCOL_SHA,
        "formula_addendum_sha256": ADDENDUM_SHA, "input_sha256": protected, "sequential": True,
        "maximum_N3_work": protocol["expected_maximum_incremental_work"]})
    try:
        with np.load(DATA) as corpus:
            for case, spec in protocol["source_eligibility"]["cases"].items():
                init = ROOT / "initializers" / spec["initializer_run_id"]
                checkpoints = json.loads((init / "checkpoints.json").read_text())
                trajectory = torch.from_numpy(corpus[spec["trajectory_key"]].copy())
                teacher = corpus["teachers"][int(spec["case_id"])]
                for cp in ("n2_prefix500", "n2_endpoint1500"):
                    for controller in ("unchanged_baseline", "accepted_step_radius_anchor"):
                        out = cells_dir / f"{case}__{cp}__{controller}"
                        row = Cell(out, case, cp, controller, checkpoints[cp], trajectory, teacher, protected).run()
                        rows.append(row); save(SCREEN / "cell_progress.json", {"completed": rows})
        summary = summarize(rows); save(SCREEN / "summary.json", summary)
        checks = {p: sha(p) == d for p, d in protected.items()}
        save(SCREEN / "batch_integrity.json", {"unchanged": all(checks.values()), "checks": checks})
        if not all(checks.values()): raise RuntimeError("ALERT: batch protected input changed")
        output_hashes = {}
        for p in sorted(SCREEN.rglob("*")):
            if p.is_file() and p.name not in ("final_integrity.json",): output_hashes[str(p)] = sha(p)
        save(SCREEN / "final_integrity.json", {"protocol_sha256": PROTOCOL_SHA, "formula_addendum_sha256": ADDENDUM_SHA,
            "protected_input_sha256": protected, "output_sha256": output_hashes,
            "protected_unchanged": True, "output_count": len(output_hashes)})
        print(json.dumps(clean(summary), indent=2), flush=True)
    except BaseException as exc:
        save(SCREEN / "cell_batch_failure.json", {"status": "FAILED", "type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "completed": rows, "no_retry_or_replacement": True})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


if __name__ == "__main__": main()
