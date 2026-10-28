#!/usr/bin/env python3
"""Matched context-derived scaling discriminator for heterogeneous transfer."""
from __future__ import annotations

import hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from heterogeneous_entity_context_pilot_v1 import Model, aj, an, mse_system, rss, sha  # noqa: E402

PKG = ROOT / "outputs/heterogeneous_three_group_package_v1"
ROLL = ROOT / "outputs/heterogeneous_rollout5_labels_v1"
BASE = ROOT / "outputs/heterogeneous_entity_context_pilot_v1_recovery2"
OUT = ROOT / "outputs/heterogeneous_context_scaling_discriminator_v1"
GROUPS = ("sho", "ball", "kepler")
SEEDS = (2026102908, 2026102909)
UPDATES, BATCH_SYSTEMS, BATCH_QUERIES = 500, 8, 16
WALL, RSS_CAP, ART_CAP = 600.0, 2 << 30, 128 << 20


def scales_from_context(x, dx, cm):
    sx = torch.sqrt(torch.mean(x * x, dim=1)) + 1e-8
    sd = torch.sqrt(torch.mean(dx * dx, dim=1)) + 1e-8
    one = torch.ones_like(sx)
    return torch.where(cm, sx, one), torch.where(cm, sd, one)


def dt_norm(indices):
    logs = torch.log(DT[indices])
    return float(logs.mean()), float(logs.std(unbiased=False) + 1e-8)


def own_batch(indices, lm, ls):
    ii = torch.as_tensor(indices, dtype=torch.long)
    return (
        X[ii] / SX[ii, None],
        DX[ii] / SD[ii, None],
        QX[ii] / SX[ii, None],
        EM[ii], CM[ii], (torch.log(DT[ii]) - lm) / ls,
    )


def fit(train_indices, target_map, seed, name):
    torch.manual_seed(seed)
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lm, ls = dt_norm(train_indices)
    gen = torch.Generator().manual_seed(seed + 123)
    losses = []
    for _ in range(UPDATES):
        sel = torch.randperm(len(train_indices), generator=gen)[:BATCH_SYSTEMS].numpy()
        ids = [train_indices[int(j)] for j in sel]
        qi = torch.randperm(64, generator=gen)[:BATCH_QUERIES]
        cx, cd, qx, em, cm, dt = own_batch(ids, lm, ls)
        qy = torch.stack([torch.as_tensor(target_map[i][qi.numpy()]) / SD[i] for i in ids])
        pred = model.predict(qx[:, qi], em, cm, dt, model.context_code(cx, cd, em, cm, dt))
        mask = cm[:, None].to(pred.dtype)
        loss = ((pred - qy) ** 2 * mask).sum() / mask.sum().clamp_min(1.0) / BATCH_QUERIES
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    torch.save({"state_dict": model.state_dict(), "dt_normalizer": (lm, ls), "train_indices": train_indices}, OUT / f"{name}.pt")
    return model, (lm, ls), losses


def predict_modes(model, norm, train_indices, dev_indices):
    lm, ls = norm
    cx, cd, qx, em, cm, dt = own_batch(dev_indices, lm, ls)
    wrong_ids = list(np.roll(np.asarray(dev_indices), -1))
    wcx, wcd, _, wem, wcm, wdt = own_batch(wrong_ids, lm, ls)
    with torch.no_grad():
        code = model.context_code(cx, cd, em, cm, dt)
        wrong_code = model.context_code(wcx, wcd, wem, wcm, wdt)
        tcx, tcd, _, tem, tcm, tdt = own_batch(train_indices, lm, ls)
        mean_code = model.context_code(tcx, tcd, tem, tcm, tdt).mean(0, keepdim=True).expand(len(dev_indices), -1)
        correct = model.predict(qx, em, cm, dt, code) * SD[dev_indices, None]
        blind = model.predict(qx, em, cm, dt, mean_code) * SD[dev_indices, None]
        code_wrong = model.predict(qx, em, cm, dt, wrong_code) * SD[dev_indices, None]
        time_blind = model.predict(qx, em, cm, torch.zeros_like(dt), code) * SD[dev_indices, None]
        # A complete wrong-context transform uses the donor's scales as well as code.
        fqx = QX[dev_indices] / SX[wrong_ids, None]
        full_wrong = model.predict(fqx, em, cm, wdt, wrong_code) * SD[wrong_ids, None]
    return [v.numpy() for v in (correct, blind, code_wrong, full_wrong, time_blind)], code.numpy()


def rollout(model, norm, dev_indices, codes):
    lm, ls = norm
    em, cm = EM[dev_indices], CM[dev_indices]
    dt = (torch.log(DT[dev_indices]) - lm) / ls
    raw_state = X[dev_indices, -1] + DX[dev_indices, -1]
    code = torch.as_tensor(codes)
    rows = []
    with torch.no_grad():
        for _ in range(5):
            qx = raw_state[:, None] / SX[dev_indices, None]
            delta = model.predict(qx, em, cm, dt, code)[:, 0] * SD[dev_indices]
            raw_state = raw_state + delta
            rows.append(raw_state.numpy())
    return np.stack(rows, 1)


def main():
    global X, DX, QX, EM, CM, DT, SX, SD
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    manifest = json.load(open(PKG / "quarantined_split_manifest.json"))
    byidx = {int(r["learner_index"]): r for r in manifest["records"]}
    with np.load(PKG / "learner_inputs.npz", allow_pickle=False) as z:
        X = torch.as_tensor(z["context_x"]); DX = torch.as_tensor(z["context_dx"]); QX = torch.as_tensor(z["query_x"])
        EM = torch.as_tensor(z["entity_mask"]); CM = torch.as_tensor(z["channel_mask"]); DT = torch.as_tensor(z["delta_t"])
    SX, SD = scales_from_context(X, DX, CM)
    with np.load(PKG / "training_query_targets.npz", allow_pickle=False) as z:
        train_target = {int(i): np.asarray(y) for i, y in zip(z["learner_index"], z["query_dx"])}
    train_by = {g: [i for i, r in byidx.items() if r["group"] == g and r["role"] == "training"] for g in GROUPS}
    dev_by = {g: [i for i, r in byidx.items() if r["group"] == g and r["role"] == "opened_development"] for g in GROUPS}

    active = CM[:, None].expand_as(X)
    xerr = torch.max(torch.abs((X / SX[:, None]) * SX[:, None] - X)[active]).item()
    derr = torch.max(torch.abs((DX / SD[:, None]) * SD[:, None] - DX)[active]).item()
    if not math.isfinite(xerr + derr) or max(xerr, derr) > 1e-12 or torch.any(SX <= 0) or torch.any(SD <= 0):
        raise RuntimeError("SCALING_INVARIANT_FAILED")
    plan = {
        "status": "PRECHARGED_BEFORE_FITS", "fits": 12, "updates_per_fit": UPDATES, "total_updates": 6000,
        "training_context_rows_encoded": 1536000, "training_query_rows": 768000,
        "evaluation_query_rows": 18432, "rollout_prediction_steps": 480,
        "controls": ["correct", "blind", "code_wrong", "full_wrong", "time_blind", "scaled_scratch", "saved_unscaled_baseline"],
        "scaling": "per-system per-active-channel RMS from permitted context_x/context_dx only; no centering or clipping",
        "unchanged_forward_gate": {"shared_over_scaled_scratch_max": .8, "shared_over_blind_max": .8, "correct_wins_full_wrong_min": 6, "shared_rollout_over_scratch_max": 1.0},
        "ceilings": {"wall_seconds": WALL, "peak_rss_bytes": RSS_CAP, "artifact_bytes": ART_CAP},
        "synthetic_invariants": {"state_roundtrip_max_abs": xerr, "increment_roundtrip_max_abs": derr},
        "source_hashes": {"driver": sha(Path(__file__)), "proposal": sha(ROOT / "coordination/HETEROGENEOUS_CONTEXT_SCALING_DISCRIMINATOR_V1.md"), "package_barrier": sha(PKG / "prediction_barrier.json"), "baseline_result": sha(BASE / "result.json")},
        "qualification": "Already-opened development labels; no fresh-confirmation claim."
    }
    aj(OUT / "plan.json", plan)

    preds, rollpred, histories, metadata = {}, {}, {}, []
    for held in GROUPS:
        shared_train = sum((train_by[g] for g in GROUPS if g != held), [])
        scratch_train, dev = train_by[held], dev_by[held]
        for seed in SEEDS:
            sm, sn, sl = fit(shared_train, train_target, seed, f"shared_held_{held}_seed{seed}")
            modes, codes = predict_modes(sm, sn, shared_train, dev)
            for mode, val in zip(("correct", "blind", "code_wrong", "full_wrong", "time_blind"), modes):
                key = f"shared__{held}__{seed}__{mode}"; preds[key] = val
                metadata.append({"key": key, "arm": "shared", "held": held, "seed": seed, "mode": mode, "dev_indices": dev})
            rollpred[f"shared__{held}__{seed}"] = rollout(sm, sn, dev, codes); histories[f"shared__{held}__{seed}"] = sl
            tm, tn, tl = fit(scratch_train, train_target, seed + 1000, f"scratch_{held}_seed{seed}")
            modes2, codes2 = predict_modes(tm, tn, scratch_train, dev)
            key = f"scratch__{held}__{seed}__correct"; preds[key] = modes2[0]
            metadata.append({"key": key, "arm": "scratch", "held": held, "seed": seed, "mode": "correct", "dev_indices": dev})
            rollpred[f"scratch__{held}__{seed}"] = rollout(tm, tn, dev, codes2); histories[f"scratch__{held}__{seed}"] = tl
    an(OUT / "predictions_before_rescoring_opened_development.npz", **preds, **{f"rollout__{k}": v for k, v in rollpred.items()})
    aj(OUT / "prediction_barrier.json", {"status": "ALL_PREDICTIONS_PERSISTED_BEFORE_RESCORING_HISTORICALLY_OPENED_LABELS", "prediction_sha256": sha(OUT / "predictions_before_rescoring_opened_development.npz"), "metadata": metadata})

    with np.load(PKG / "protected_development_query_targets.npz", allow_pickle=False) as z:
        dev_target = {int(i): np.asarray(y) for i, y in zip(z["learner_index"], z["query_dx"])}
    with np.load(ROLL / "protected_rollout5_targets.npz", allow_pickle=False) as z:
        roll_target = {int(i): np.asarray(y) for i, y in zip(z["learner_index"], z["truth_x"])}
    base = json.load(open(BASE / "result.json"))
    base_cells = {(d["held_group"], int(d["seed"])): d for d in base["comparisons"]}
    comps, scores = [], []
    for held in GROUPS:
        dev = dev_by[held]; target = np.asarray([dev_target[i] for i in dev]); rt = np.asarray([roll_target[i] for i in dev]); mask = CM[dev].numpy()
        target_energy = float(np.mean([np.mean(t[np.broadcast_to(m, t.shape)] ** 2) for t, m in zip(target, mask)]))
        for seed in SEEDS:
            vals = {mode: mse_system(preds[f"shared__{held}__{seed}__{mode}"], target, mask) for mode in ("correct", "blind", "code_wrong", "full_wrong", "time_blind")}
            scratch = mse_system(preds[f"scratch__{held}__{seed}__correct"], target, mask)
            sr = mse_system(rollpred[f"shared__{held}__{seed}"], rt, mask); tr = mse_system(rollpred[f"scratch__{held}__{seed}"], rt, mask)
            bc = base_cells[(held, seed)]
            d = {
                "held_group": held, "seed": seed, "target_MSE_zero": target_energy,
                "scaled_shared_MSE": float(vals["correct"].mean()), "scaled_scratch_MSE": float(scratch.mean()),
                "blind_MSE": float(vals["blind"].mean()), "code_wrong_MSE": float(vals["code_wrong"].mean()), "full_wrong_MSE": float(vals["full_wrong"].mean()), "time_blind_MSE": float(vals["time_blind"].mean()),
                "scaled_shared_over_scaled_scratch": float(vals["correct"].sum() / scratch.sum()), "scaled_shared_over_blind": float(vals["correct"].sum() / vals["blind"].sum()),
                "correct_wins_full_wrong": int(np.sum(vals["correct"] < vals["full_wrong"])),
                "scaled_shared_rollout_MSE": float(sr.mean()), "scaled_scratch_rollout_MSE": float(tr.mean()), "scaled_shared_rollout_over_scratch": float(sr.sum() / tr.sum()),
                "scaled_shared_over_saved_unscaled_shared": float(vals["correct"].mean() / bc["shared_MSE"]),
                "scaled_scratch_over_saved_unscaled_scratch": float(scratch.mean() / bc["scratch_MSE"]),
            }
            d["passed_unchanged_forward_gate"] = bool(d["scaled_shared_over_scaled_scratch"] <= .8 and d["scaled_shared_over_blind"] <= .8 and d["correct_wins_full_wrong"] >= 6 and d["scaled_shared_rollout_over_scratch"] <= 1.0)
            comps.append(d)
            for j, i in enumerate(dev):
                scores.append({"held_group": held, "seed": seed, "learner_index": i, "correct_MSE": float(vals["correct"][j]), "blind_MSE": float(vals["blind"][j]), "code_wrong_MSE": float(vals["code_wrong"][j]), "full_wrong_MSE": float(vals["full_wrong"][j]), "scratch_MSE": float(scratch[j])})
    counts = {"fits": 12, "optimizer_updates": 6000, "training_context_rows_encoded": 6000 * 8 * 32, "training_query_rows": 6000 * 8 * 16, "evaluation_query_rows": 36 * 8 * 64, "rollout_prediction_steps": 12 * 8 * 5, "score_system_blocks": len(scores)}
    for k in ("training_context_rows_encoded", "training_query_rows", "evaluation_query_rows", "rollout_prediction_steps"):
        if counts[k] != plan[k]: raise RuntimeError(f"COUNT_{k}:{counts[k]}!={plan[k]}")
    elapsed = time.monotonic() - started; artifacts = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()); resources = {"wall_seconds": elapsed, "peak_rss_bytes": rss(), "artifact_bytes_before_result": artifacts}
    if elapsed > WALL or resources["peak_rss_bytes"] > RSS_CAP or artifacts > ART_CAP: raise RuntimeError(f"RESOURCE_CAP:{resources}")
    supported = all(d["passed_unchanged_forward_gate"] for d in comps)
    result = {
        "status": "HETEROGENEOUS_CONTEXT_SCALING_DISCRIMINATOR_V1_COMPLETE", "classification": "SUPPORTED_OPENED_DEVELOPMENT_ONLY" if supported else "NOT_SUPPORTED",
        "comparisons": comps, "scores": scores, "counts": counts, "resources": resources, "parameter_count": sum(p.numel() for p in Model().parameters()),
        "training_loss_endpoints": {k: v[-1] for k, v in histories.items()},
        "qualification": ["This isolates one context-derived coordinate scaling intervention in the frozen model; it is not a new architecture or corpus.", "Common tensor shape still supplies masks, entity ordering, position/velocity channels and timestep, not common physical semantics.", "Every evaluation cohort was already opened; this is development evidence only.", "Failure limits this scaling explanation for this learner and does not prove arbitrary mechanisms are unlearnable."]
    }
    aj(OUT / "result.json", result); aj(OUT / "completion.json", {"status": "COMPLETE", "classification": result["classification"], "result_sha256": sha(OUT / "result.json"), "counts": counts})
    print(json.dumps({"classification": result["classification"], "comparisons": comps, "counts": counts, "resources": resources}, indent=2))


if __name__ == "__main__":
    main()
