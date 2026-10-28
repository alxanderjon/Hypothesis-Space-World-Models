#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BASE = ROOT / "outputs/sho/autonomous_generic_sho_20261014"
SOURCE = BASE / "scripts/run_cross_mechanism_alpaca_pilot_v1.py"
META_DIR = BASE / "diagnostics/cross_mechanism_source_pretrained_meta_objective_v1"
OUT = BASE / "diagnostics/sequence_aware_plasticity_v1"
PLAN = BASE / "protocols/sequence_aware_plasticity_v1/plan.json"
LOCK = BASE / "active.lock"
SEEDS = (2026102751, 2026102752)
START = time.monotonic()


def load_source():
    spec = importlib.util.spec_from_file_location("cross_mech_v1", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v = load_source()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


class Guard:
    def __init__(self, caps):
        self.caps = caps
        self.counts = {key: 0 for key in caps if key not in {
            "wall_seconds", "peak_RSS_bytes", "artifact_bytes"
        }}

    def add(self, key, amount=1):
        future = self.counts[key] + amount
        if future > self.caps[key]:
            raise RuntimeError(f"PRECALL_CAP_{key}:{future}>{self.caps[key]}")
        self.counts[key] = future

    def resources(self):
        if time.monotonic() - START > self.caps["wall_seconds"]:
            raise RuntimeError("WALL_CAP")
        if peak_rss_bytes() > self.caps["peak_RSS_bytes"]:
            raise RuntimeError("RSS_CAP")


def params_of(model):
    return OrderedDict((name, parameter) for name, parameter in model.named_parameters())


def phi(model, params, x):
    return functional_call(model, params, (x,))


def fit_head(features, targets, guard, training):
    guard.add("training_ridge_solves" if training else "evaluation_ridge_solves")
    return v.ridge(features, targets)


def normalized_mse(prediction, target):
    return torch.mean((prediction - target) ** 2)


def main():
    if OUT.exists():
        raise FileExistsError("OUTPUT_NAMESPACE_EXISTS")
    source_doc = ROOT / "coordination/SEQUENCE_AWARE_PLASTICITY_V1.md"
    ordinary_paths = {
        seed: META_DIR / f"ordinary_excluding_pendulum_seed{seed}.pt" for seed in SEEDS
    }
    missing = [str(path) for path in [SOURCE, source_doc, *ordinary_paths.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    caps = {
        "fits": 4,
        "outer_updates": 1600,
        "differentiable_inner_steps": 4800,
        "training_ridge_solves": 9600,
        "training_feature_rows": 307200,
        "target_sequences": 8,
        "target_updates": 64,
        "evaluation_ridge_solves": 1024,
        "evaluation_prediction_rows": 61440,
        "one_step_simulations": 10752,
        "RK4_RHS_evaluations": 43008,
        "score_blocks": 960,
        "wall_seconds": 600,
        "peak_RSS_bytes": 1 << 30,
        "artifact_bytes": 128 << 20,
    }
    plan = {
        "status": "PROSPECTIVE_BEFORE_EXECUTION",
        "objective": "Test whether a representation meta-trained through a three-family sequence retains prior behavior while reducing later-family adaptation cost.",
        "arms": ["retention_aware", "current_only"],
        "seeds": list(SEEDS),
        "source_split_per_family": {"sequence_update": 16, "outer_eval": 8},
        "target_split": {"sequential_adaptation": 8, "never_adapted_evaluation": 8},
        "inner_update": {"context_fit_rows": "even16", "context_loss_rows": "odd16", "learning_rate": 0.01, "steps_per_law": 1},
        "outer_update": {"optimizer": "Adam", "learning_rate": 0.001, "iterations": 400, "family_order": "deterministic cyclic", "outer_pool_order": "deterministic cyclic"},
        "controls": ["current_only", "saved_ordinary_source_pretrained", "deterministic_scratch"],
        "decision": {
            "pendulum_stage8_over_stage0_max": 0.8,
            "retention_over_current_and_ordinary_max": 0.8,
            "retention_law_wins_min": 6,
            "source_after_over_before_max": 1.1,
            "source_laws_within_10pct_min": 20,
            "required_seeds": 2,
        },
        "ceilings": caps,
        "claim_limits": [
            "All systems and states are opened development evidence.",
            "The controlled state banks, known q'=v, dt=.05, RK4, oscillator families and 16-dimensional ridge head are supplied assumptions.",
            "Every target shared-weight update is supervised adaptation from permitted context responses; this is not zero-shot recognition.",
            "Passing does not establish passive-trajectory identification, arbitrary mechanisms, or million-parameter scalability.",
        ],
        "source_hashes": {"source_driver": sha(SOURCE), "design_document": sha(source_doc)},
        "saved_control_hashes": {str(seed): sha(path) for seed, path in ordinary_paths.items()},
    }

    with LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(PLAN, plan)
        OUT.mkdir(parents=True, exist_ok=False)
        guard = Guard(caps)

        rows = v.make_laws()
        source_train = [row for row in rows if row["role"] == "train"]
        source_dev = [row for row in rows if row["role"] == "represented_dev"]
        pendulum = [row for row in rows if row["role"] == "heldout_dev"]
        cx, qx = v.states(32, 2026102771), v.states(64, 2026102772)
        atomic_json(OUT / "systems_and_states_before_responses.json", {
            "systems": rows,
            "context_states": cx.tolist(),
            "query_states": qx.tolist(),
            "development_query_responses_generated": False,
            "source_training_responses_generated": False,
        })

        source_c = v.simulate(source_train, cx)
        source_q = v.simulate(source_train, qx)
        guard.add("one_step_simulations", len(source_train) * (len(cx) + len(qx)))
        guard.add("RK4_RHS_evaluations", 4 * len(source_train) * (len(cx) + len(qx)))
        ymean = source_q.reshape(-1, 2).mean(0)
        ystd = np.maximum(source_q.reshape(-1, 2).std(0), 1e-9)
        tc = torch.tensor(cx / np.array([1.5, 1.0]))
        tq = torch.tensor(qx / np.array([1.5, 1.0]))
        yc = torch.tensor((source_c - ymean) / ystd)
        yq = torch.tensor((source_q - ymean) / ystd)

        by_family = {}
        for index, row in enumerate(source_train):
            by_family.setdefault(row["family"], []).append(index)
        families = ("linear", "duffing", "vdp")
        inner_pools = {family: by_family[family][:16] for family in families}
        outer_pools = {family: by_family[family][16:24] for family in families}

        checkpoints = {}
        training_history = []
        for arm in ("retention_aware", "current_only"):
            for seed in SEEDS:
                guard.add("fits")
                torch.manual_seed(seed)
                model = v.Features()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                history = []
                for update in range(400):
                    guard.resources()
                    order = families[update % 3:] + families[:update % 3]
                    fast = params_of(model)
                    for family_position, family in enumerate(order):
                        law_index = inner_pools[family][(update + family_position) % 16]
                        even = torch.arange(0, 32, 2)
                        odd = torch.arange(1, 32, 2)
                        even_phi = phi(model, fast, tc[even])
                        odd_phi = phi(model, fast, tc[odd])
                        guard.add("training_feature_rows", 32)
                        head = fit_head(even_phi, yc[law_index, even], guard, True)
                        loss = normalized_mse(odd_phi @ head, yc[law_index, odd])
                        grads = torch.autograd.grad(loss, tuple(fast.values()), create_graph=True)
                        fast = OrderedDict((name, value - 0.01 * grad) for (name, value), grad in zip(fast.items(), grads))
                        guard.add("differentiable_inner_steps")

                    context_phi = phi(model, fast, tc)
                    query_phi = phi(model, fast, tq)
                    guard.add("training_feature_rows", 96)
                    outer_losses = []
                    for family_position, family in enumerate(order):
                        law_index = outer_pools[family][(update + family_position) % 8]
                        head = fit_head(context_phi, yc[law_index], guard, True)
                        outer_losses.append(normalized_mse(query_phi @ head, yq[law_index]))
                    if arm == "retention_aware":
                        outer_loss = torch.stack(outer_losses).mean()
                    else:
                        outer_loss = outer_losses[-1] + 0.0 * torch.stack(outer_losses[:-1]).sum()
                    optimizer.zero_grad()
                    outer_loss.backward()
                    optimizer.step()
                    guard.add("outer_updates")
                    if update in (0, 99, 199, 299, 399):
                        history.append({"update": update + 1, "losses": [float(x.detach()) for x in outer_losses], "optimized_loss": float(outer_loss.detach())})
                path = OUT / f"{arm}_seed{seed}.pt"
                torch.save(model.state_dict(), path)
                checkpoints[(arm, seed)] = path
                training_history.append({"arm": arm, "seed": seed, "history": history})

        # Only context responses are generated before target/development predictions.
        source_dev_c = v.simulate(source_dev, cx)
        pendulum_c = v.simulate(pendulum, cx)
        guard.add("one_step_simulations", (len(source_dev) + len(pendulum)) * len(cx))
        guard.add("RK4_RHS_evaluations", 4 * (len(source_dev) + len(pendulum)) * len(cx))
        source_dev_yc = torch.tensor((source_dev_c - ymean) / ystd)
        pendulum_yc = torch.tensor((pendulum_c - ymean) / ystd)

        model_specs = []
        for seed in SEEDS:
            model_specs.extend([
                ("retention_aware", seed, checkpoints[("retention_aware", seed)]),
                ("current_only", seed, checkpoints[("current_only", seed)]),
                ("ordinary", seed, ordinary_paths[seed]),
                ("scratch", seed, None),
            ])

        prediction_arrays = {}
        prediction_meta = []
        for arm, seed, checkpoint in model_specs:
            guard.add("target_sequences")
            torch.manual_seed(seed)
            model = v.Features()
            if checkpoint is not None:
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

            def evaluate(stage):
                with torch.no_grad():
                    pcontext = model(tc)
                    pquery = model(tq)
                    pend_pred = []
                    for local_index in range(8, 16):
                        head = fit_head(pcontext, pendulum_yc[local_index], guard, False)
                        pend_pred.append((pquery @ head).numpy() * ystd + ymean)
                    source_pred = []
                    if stage in (0, 8):
                        for local_index in range(24):
                            head = fit_head(pcontext, source_dev_yc[local_index], guard, False)
                            source_pred.append((pquery @ head).numpy() * ystd + ymean)
                pend_pred = np.asarray(pend_pred)
                guard.add("evaluation_prediction_rows", pend_pred.shape[0] * pend_pred.shape[1])
                prediction_arrays[f"pend_{arm}_{seed}_stage{stage}"] = pend_pred
                prediction_meta.append({"arm": arm, "seed": seed, "stage": stage, "role": "pendulum_never_adapted", "laws": list(range(8, 16))})
                if source_pred:
                    source_pred = np.asarray(source_pred)
                    guard.add("evaluation_prediction_rows", source_pred.shape[0] * source_pred.shape[1])
                    prediction_arrays[f"source_{arm}_{seed}_stage{stage}"] = source_pred
                    prediction_meta.append({"arm": arm, "seed": seed, "stage": stage, "role": "source_represented_development", "laws": list(range(24))})

            evaluate(0)
            for adaptation_index in range(8):
                even = torch.arange(0, 32, 2)
                odd = torch.arange(1, 32, 2)
                pe = model(tc[even])
                po = model(tc[odd])
                head = fit_head(pe, pendulum_yc[adaptation_index, even], guard, False)
                adaptation_loss = normalized_mse(po @ head, pendulum_yc[adaptation_index, odd])
                grads = torch.autograd.grad(adaptation_loss, tuple(model.parameters()))
                with torch.no_grad():
                    for parameter, grad in zip(model.parameters(), grads):
                        parameter.add_(grad, alpha=-0.01)
                guard.add("target_updates")
                evaluate(adaptation_index + 1)

        atomic_npz(OUT / "predictions_before_query_labels.npz", **prediction_arrays)
        atomic_json(OUT / "prediction_barrier.json", {
            "status": "ALL_TARGET_AND_SOURCE_DEVELOPMENT_PREDICTIONS_PERSISTED_BEFORE_QUERY_LABEL_GENERATION",
            "prediction_sha256": sha(OUT / "predictions_before_query_labels.npz"),
            "metadata": prediction_meta,
            "counts_so_far": guard.counts,
        })

        source_dev_q = v.simulate(source_dev, qx)
        pendulum_q = v.simulate(pendulum, qx)
        guard.add("one_step_simulations", (len(source_dev) + len(pendulum)) * len(qx))
        guard.add("RK4_RHS_evaluations", 4 * (len(source_dev) + len(pendulum)) * len(qx))

        scores = []
        for meta in prediction_meta:
            key = ("pend" if meta["role"].startswith("pendulum") else "source") + f"_{meta['arm']}_{meta['seed']}_stage{meta['stage']}"
            predictions = prediction_arrays[key]
            targets = pendulum_q[8:16] if meta["role"].startswith("pendulum") else source_dev_q
            for local_index in range(len(predictions)):
                scores.append({
                    **meta,
                    "law": int(meta["laws"][local_index]),
                    "MSE": float(np.mean((predictions[local_index] - targets[local_index]) ** 2)),
                })
                guard.add("score_blocks")

        def errors(arm, seed, role, stage):
            return np.asarray([entry["MSE"] for entry in scores if entry["arm"] == arm and entry["seed"] == seed and entry["role"] == role and entry["stage"] == stage])

        comparisons = []
        decisions = []
        for seed in SEEDS:
            retention0 = errors("retention_aware", seed, "pendulum_never_adapted", 0)
            retention8 = errors("retention_aware", seed, "pendulum_never_adapted", 8)
            current8 = errors("current_only", seed, "pendulum_never_adapted", 8)
            ordinary8 = errors("ordinary", seed, "pendulum_never_adapted", 8)
            source0 = errors("retention_aware", seed, "source_represented_development", 0)
            source8 = errors("retention_aware", seed, "source_represented_development", 8)
            entry = {
                "seed": seed,
                "pendulum_stage8_over_stage0": float(retention8.sum() / retention0.sum()),
                "retention_over_current_stage8": float(retention8.sum() / current8.sum()),
                "retention_over_ordinary_stage8": float(retention8.sum() / ordinary8.sum()),
                "retention_wins_current": int(np.sum(retention8 < current8)),
                "retention_wins_ordinary": int(np.sum(retention8 < ordinary8)),
                "source_after_over_before": float(source8.sum() / source0.sum()),
                "source_laws_within_10pct": int(np.sum(source8 <= 1.1 * source0)),
                "finite": bool(np.isfinite(np.concatenate([retention0, retention8, current8, ordinary8, source0, source8])).all()),
            }
            entry["passed"] = bool(
                entry["pendulum_stage8_over_stage0"] <= 0.8
                and entry["retention_over_current_stage8"] <= 0.8
                and entry["retention_over_ordinary_stage8"] <= 0.8
                and entry["retention_wins_current"] >= 6
                and entry["retention_wins_ordinary"] >= 6
                and entry["source_after_over_before"] <= 1.1
                and entry["source_laws_within_10pct"] >= 20
                and entry["finite"]
            )
            comparisons.append(entry)
            decisions.append(entry["passed"])

        for key, expected in caps.items():
            if key in guard.counts and guard.counts[key] != expected:
                raise RuntimeError(f"COUNT_{key}:{guard.counts[key]}!={expected}")
        elapsed = time.monotonic() - START
        artifact_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
        if elapsed > caps["wall_seconds"] or peak_rss_bytes() > caps["peak_RSS_bytes"] or artifact_bytes > caps["artifact_bytes"]:
            raise RuntimeError("RESOURCE_CAP")
        result = {
            "status": "SEQUENCE_AWARE_PLASTICITY_V1_COMPLETE",
            "classification": "SUPPORTED_DEVELOPMENT_ONLY" if all(decisions) else "NOT_SUPPORTED",
            "comparisons": comparisons,
            "training_history": training_history,
            "scores": scores,
            "counts": guard.counts,
            "resources": {"wall_seconds": elapsed, "peak_RSS_bytes": peak_rss_bytes(), "artifact_bytes_before_result": artifact_bytes},
            "source_hashes": {"driver": sha(HERE), "source_driver": sha(SOURCE), "plan": sha(PLAN)},
            "qualifications": plan["claim_limits"],
        }
        atomic_json(OUT / "result.json", result)
        atomic_json(OUT / "completion.json", {
            "status": "COMPLETE",
            "classification": result["classification"],
            "result_sha256": sha(OUT / "result.json"),
            "prediction_sha256": sha(OUT / "predictions_before_query_labels.npz"),
            "counts": guard.counts,
        })
        print(json.dumps({"classification": result["classification"], "comparisons": comparisons, "counts": guard.counts, "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
