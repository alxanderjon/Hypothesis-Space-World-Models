#!/usr/bin/env python3
"""Train only the nonzero nearest-distance REPLACE/KEEP threshold."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_nearest_margin as base
import experiments.sho.run_stage1b_nearest_margin_symmetry as margin

OUTPUT_DIR = Path("outputs/sho/stage1B/nearest_distance_threshold")
BOUNDARY_PATH = OUTPUT_DIR / "nearest_threshold_boundary.csv"
FULL_PATH = OUTPUT_DIR / "nearest_threshold_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "nearest_distance_threshold_frozen_candidate.pt"
MARGIN_SHA256 = "652a70e891517de39c3e20e7aad4830eb01f06717349cd470809f4db59b84b68"
MODEL_SEED = base.MODEL_SEED + 1_000_003
SCALE = 1.0e-18
MAX_EPOCHS = 5000
SYNTHETIC = np.array([1e-18, 1e-16, 1e-14, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0])
SYNTHETIC = np.concatenate((-SYNTHETIC[::-1], SYNTHETIC))


class NonzeroThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(SCALE, dtype=torch.float64))
        self.scaled_theta = nn.Parameter(torch.randn((), dtype=torch.float64))
        self.raw_alpha = nn.Parameter(torch.randn((), dtype=torch.float64))

    def theta(self):
        return self.scale * self.scaled_theta

    def alpha(self):
        return nn.functional.softplus(self.raw_alpha) / self.scale

    def forward(self, m):
        if torch.any(m == 0.0):
            raise ValueError("Zero margin must bypass the learned threshold")
        return self.alpha() * (m - self.theta())


def dense_grid():
    positive = np.unique(np.concatenate((np.geomspace(1e-24, 1.0, 4801),
        np.linspace(1e-24, 1e-20, 10001), np.linspace(1e-20, 1e-16, 10001))))
    return np.concatenate((-positive[::-1], positive))


def route(model, margins, tie_owners=None):
    """0 is a bookkeeping route; zero logits on nonzero inputs are failures."""
    values = np.asarray(margins, dtype=np.float64)
    nonzero = values != 0.0
    logits = np.full(len(values), np.nan)
    with torch.no_grad():
        if np.any(nonzero):
            logits[nonzero] = model(torch.from_numpy(values[nonzero])).numpy()
    decisions = np.full(len(values), "TIE_CKDTREE", dtype="U16")
    decisions[nonzero & (logits > 0.0)] = "REPLACE"
    decisions[nonzero & (logits < 0.0)] = "KEEP"
    decisions[nonzero & (logits == 0.0)] = "ZERO_LOGIT_ERROR"
    if np.any(~nonzero) and tie_owners is None:
        raise ValueError("Zero-margin routing requires deterministic cKDTree ownership")
    owners = None if tie_owners is None else np.where(~nonzero, tie_owners, -1)
    return logits, decisions, owners


def balanced_indices(pool, per_class, seed):
    rng = np.random.default_rng(seed)
    sides = [np.flatnonzero(pool.exact_margin < 0), np.flatnonzero(pool.exact_margin > 0)]
    count = min(per_class, *(len(s) for s in sides))
    selected = []
    for side in sides:
        required = side[pool.one_ulp[side]]
        assert len(required) <= count
        remaining = np.setdiff1d(side, required)
        # Keep the closest real examples as well as a broad random sample.
        close = remaining[np.argsort(np.abs(pool.exact_margin[remaining]))[:min(256, count-len(required))]]
        required = np.concatenate((required, close))
        remaining = np.setdiff1d(remaining, close)
        selected.append(np.concatenate((required, rng.choice(remaining, count-len(required), replace=False))))
    result = np.concatenate(selected)
    rng.shuffle(result)
    assert np.all(np.isin(np.flatnonzero(pool.one_ulp), result))
    return result


def metrics(model, values, truth, split, group):
    assert np.all(values != 0.0) and np.all(truth != 0.0)
    logits, decisions, _ = route(model, values)
    mistakes = (np.sign(logits) != np.sign(truth)) | ~np.isfinite(logits)
    correct_pos = (truth > 0) & ~mistakes
    correct_neg = (truth < 0) & ~mistakes
    row = dict(record_type="classification_summary", split=split, pair_group=group,
        count=len(values), REPLACE_count=int(np.sum(truth > 0)), KEEP_count=int(np.sum(truth < 0)),
        accuracy=float(np.mean(~mistakes)) if len(values) else "",
        mistakes=int(np.sum(mistakes)), false_zero_logits=int(np.sum(logits == 0)),
        closest_correct_positive=float(np.min(values[correct_pos])) if np.any(correct_pos) else "",
        closest_correct_negative=float(np.max(values[correct_neg])) if np.any(correct_neg) else "")
    return row, logits, decisions, mistakes


def train(values, truth):
    torch.manual_seed(MODEL_SEED)
    model = NonzeroThreshold()
    initial_theta, initial_alpha = float(model.theta().detach()), float(model.alpha().detach())
    assert initial_theta != 0 and initial_alpha > 0
    x, y = torch.from_numpy(values), torch.from_numpy(np.sign(truth))
    sx, sy = torch.from_numpy(SYNTHETIC), torch.from_numpy(np.sign(SYNTHETIC))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    history = []
    streak = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        # Numerically stable binary logistic loss, only on nonzero examples.
        loss = (nn.functional.softplus(-y * model(x)).mean()
                + nn.functional.softplus(-sy * model(sx)).mean()) / 2
        loss.backward()
        optimizer.step()
        theta, alpha = float(model.theta().detach()), float(model.alpha().detach())
        # Monotonicity makes the two closest dense-grid endpoints sufficient
        # during training; the entire grid is evaluated and saved afterward.
        exact = (abs(theta) < 1e-24 and alpha > 0
                 and np.all(np.sign(values - theta) == np.sign(truth)))
        streak = streak + 1 if exact else 0
        if epoch == 1 or epoch % 100 == 0 or streak >= 100:
            history.append(dict(record_type="training_checkpoint", epoch=epoch,
                loss=float(loss.detach()), learned_theta=theta, learned_alpha=alpha,
                dense_endpoint_gate=exact))
        if epoch == 1 or epoch % 500 == 0:
            print(f"threshold epoch={epoch} theta={theta:.17g} alpha={alpha:.6g} loss={float(loss.detach()):.3e}", flush=True)
        if streak >= 100:
            break
    return model, history, dict(initial_theta=initial_theta, initial_alpha=initial_alpha,
        learned_theta=theta, learned_alpha=alpha, epochs=epoch, model_seed=MODEL_SEED,
        optimizer="Adam", learning_rate=0.03, threshold_scale=SCALE,
        stopping_reason="100_consecutive_exact_checks" if streak >= 100 else "maximum_epochs",
        dtype="float64", hidden_layers=0, learned_zero_class=False,
        objective="equal_weight_real_and_synthetic_binary_logistic_loss")


def evaluate_pool(model, frozen, pool, split):
    values = margin.predict(frozen, pool.inputs)
    if np.any(values == 0):
        raise RuntimeError("Frozen margin erased a non-tied comparison")
    rows = []
    passed = True
    for orientation, v, truth in (("original", values, pool.exact_margin),
                                  ("swapped", margin.predict(frozen, pool.inputs[:, ::-1]), -pool.exact_margin)):
        row, logits, decisions, mistakes = metrics(model, v, truth, split, "all")
        row["orientation"] = orientation
        rows.append(row)
        passed &= row["mistakes"] == 0
        for name, mask in (("nearest_runner_up", pool.category >= 2), ("one_ulp", pool.one_ulp)):
            r, _, _, _ = metrics(model, v[mask], truth[mask], split, name)
            r["orientation"] = orientation
            rows.append(r)
        for i in np.flatnonzero(pool.one_ulp | mistakes):
            rows.append(dict(record_type="one_ulp_case" if pool.one_ulp[i] else "mistake",
                split=split, orientation=orientation, source_state_index=int(pool.source_state_index[i]),
                query_index=int(pool.query_index[i]), site1=int(pool.site1[i]), site2=int(pool.site2[i]),
                pair_group=base.CATEGORY_NAMES[int(pool.category[i])],
                exact_d1=float(pool.exact_d1[i]), exact_d2=float(pool.exact_d2[i]),
                margin=float(v[i]), logit=float(logits[i]), prediction=decisions[i],
                teacher="REPLACE" if truth[i] > 0 else "KEEP", correct=not bool(mistakes[i])))
    return rows, passed


def audit_ties(model, frozen, distance_model, payload, trajectories, states, ordinary_count, seed, split):
    rows = []
    for state in states:
        sites, probes, _ = base.distance_metric._state_geometry(payload, trajectories, state)
        tree = cKDTree(sites)
        pd, ps = tree.query(probes, k=2)
        sd, ss = tree.query(sites, k=3)
        probe_owner = tree.query(probes, k=1)[1]
        spacing_owner = tree.query(sites, k=2)[1][:, 1]
        rng = np.random.default_rng(seed + 1_000_003 * state)
        pq, p1, p2 = base._ordinary_probe_pairs(sites, probes, ordinary_count, rng)
        sq, s1, s2 = base._ordinary_spacing_pairs(len(sites), ordinary_count, rng)
        blocks = [(2, probes, np.arange(len(probes)), ps[:, 0], ps[:, 1], pd[:, 0], pd[:, 1], probe_owner),
                  (3, sites, np.arange(len(sites)), ss[:, 1], ss[:, 2], sd[:, 1], sd[:, 2], spacing_owner),
                  (0, probes, pq, p1, p2, np.linalg.norm(probes[pq]-sites[p1], axis=1), np.linalg.norm(probes[pq]-sites[p2], axis=1), probe_owner),
                  (1, sites, sq, s1, s2, np.linalg.norm(sites[sq]-sites[s1], axis=1), np.linalg.norm(sites[sq]-sites[s2], axis=1), spacing_owner)]
        for category, queries, q, s1, s2, d1, d2, owners in blocks:
            tied = d1 == d2
            if not np.any(tied):
                continue
            q, s1, s2 = q[tied], s1[tied], s2[tied]
            a = base.distance_metric._predict(distance_model, queries[q]-sites[s1])
            b = base.distance_metric._predict(distance_model, queries[q]-sites[s2])
            scale = np.maximum(a, b)
            inputs = np.zeros((len(q), 2), dtype=np.float64)
            np.divide(a, scale, out=inputs[:, 0], where=scale > 0)
            np.divide(b, scale, out=inputs[:, 1], where=scale > 0)
            values = margin.predict(frozen, inputs)
            _, decisions, routed = route(model, values, owners[q])
            for i in range(len(q)):
                rows.append(dict(record_type="exact_tie_case", split=split,
                    source_state_index=state, query_index=int(q[i]), pair_group=base.CATEGORY_NAMES[category],
                    site1=int(s1[i]), site2=int(s2[i]), margin=float(values[i]),
                    exact_d1=float(d1[tied][i]), exact_d2=float(d2[tied][i]),
                    locked_d1=float(a[i]), locked_d2=float(b[i]),
                    prediction=decisions[i], teacher_owner=int(owners[q[i]]), routed_owner=int(routed[i]),
                    threshold_bypassed=bool(values[i] == 0),
                    correct=bool(values[i] == 0 and routed[i] == owners[q[i]])))
    passed = all(r["correct"] for r in rows)
    return [dict(record_type="tie_routing_summary", split=split, count=len(rows),
                 mistakes=sum(not r["correct"] for r in rows), zero_training_examples=0)] + rows, passed


def run():
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert not any(p.exists() for p in (BOUNDARY_PATH, FULL_PATH, CANDIDATE_PATH))
    protected = [p for p in Path("outputs/sho/stage1B").rglob("*locked*") if p.is_file()]
    protected += [margin.CANDIDATE_PATH, margin.CANDIDATE_PATH.with_suffix(".sha256")]
    hashes = {str(p): base._sha256(p) for p in protected}
    assert hashes[str(margin.CANDIDATE_PATH)] == MARGIN_SHA256
    checkpoint = torch.load(margin.CANDIDATE_PATH, map_location="cpu", weights_only=False)
    assert checkpoint["status"] == "FROZEN_CANDIDATE"
    frozen = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    frozen.load_state_dict(checkpoint["state_dict"])
    frozen.eval().requires_grad_(False)
    distance_model, _ = base._load_locked_distance()
    payload = torch.load(base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    train_states, validation_states, held_out = [checkpoint[k] for k in
        ("training_state_indices", "validation_state_indices", "full_held_out_state_indices")]
    assert not set(train_states) & set(validation_states)
    assert not (set(train_states) | set(validation_states)) & set(held_out)
    training = base._build_pool(distance_model, payload, trajectories, train_states,
        base.TRAIN_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED, "threshold training")
    validation = base._build_pool(distance_model, payload, trajectories, validation_states,
        base.VALIDATION_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED+1, "threshold validation")
    ti = balanced_indices(training, 8192, MODEL_SEED)
    vi = balanced_indices(validation, 8192, MODEL_SEED+1)
    values = margin.predict(frozen, training.inputs[ti])
    model, history, summary = train(values, training.exact_margin[ti])
    summary.update(record_type="summary", component="nearest_distance_threshold",
        frozen_margin_sha256=MARGIN_SHA256, balanced_training_count=len(ti), balanced_validation_count=len(vi),
        training_state_indices=json.dumps(train_states), validation_state_indices=json.dumps(validation_states),
        full_held_out_state_indices=json.dumps(held_out), candidate_saved=False,
        complete_nearest_argmin_operator_locked=False)
    rows = []
    passed = True
    for split, v, truth in (("balanced_training", values, training.exact_margin[ti]),
        ("balanced_validation", margin.predict(frozen, validation.inputs[vi]), validation.exact_margin[vi]),
        ("synthetic_boundary", SYNTHETIC, SYNTHETIC), ("dense_nonzero_grid", dense_grid(), dense_grid())):
        row, logits, decisions, mistakes = metrics(model, v, truth, split, "all")
        rows.append(row)
        passed &= row["mistakes"] == 0
        if split in ("synthetic_boundary", "dense_nonzero_grid"):
            rows.extend(dict(record_type="scalar_case", split=split, margin=float(m), logit=float(l),
                prediction=p, teacher="REPLACE" if m > 0 else "KEEP", correct=not bool(e))
                for m, l, p, e in zip(v, logits, decisions, mistakes))
    for pool, states, n, seed, split in ((training, train_states, base.TRAIN_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED, "training"),
        (validation, validation_states, base.VALIDATION_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED+1, "validation")):
        result, ok = evaluate_pool(model, frozen, pool, split)
        ties, tie_ok = audit_ties(model, frozen, distance_model, payload, trajectories, states, n, seed, split)
        assert ties[0]["count"] == pool.exact_tie_exclusion_count
        rows.extend(result + ties)
        passed &= ok and tie_ok

    def verify():
        assert all(base._sha256(Path(p)) == digest for p, digest in hashes.items())
        assert not any(p.requires_grad for p in frozen.parameters())
        summary["protected_files_verified_unchanged"] = len(hashes)

    verify()
    summary["status"] = "passed" if passed else "failed"
    summary["tie_routing_mistakes"] = sum(r["mistakes"] for r in rows if r["record_type"] == "tie_routing_summary")
    summary["nonzero_classification_mistakes"] = sum(r["mistakes"] for r in rows if r["record_type"] == "classification_summary")
    summary["elapsed_seconds"] = time.monotonic()-started
    base._write_csv(BOUNDARY_PATH, [summary] + rows + history)
    if not passed:
        base._write_csv(FULL_PATH, [dict(record_type="gate_stop", status="not_run", reason="smoke_gate_failed",
            nonzero_classification_mistakes=summary["nonzero_classification_mistakes"],
            tie_routing_mistakes=summary["tie_routing_mistakes"], candidate_saved=False)])
        print("STOP: threshold smoke gate failed", flush=True)
        return
    print("Threshold smoke PASS; evaluating full held-out comparisons", flush=True)
    full = base._build_pool(distance_model, payload, trajectories, held_out,
        base.FULL_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED+2, "threshold full held-out")
    full_rows, ok = evaluate_pool(model, frozen, full, "full_held_out")
    ties, tie_ok = audit_ties(model, frozen, distance_model, payload, trajectories, held_out,
        base.FULL_ORDINARY_PER_TYPE_PER_STATE, base.DATA_SEED+2, "full_held_out")
    assert ties[0]["count"] == full.exact_tie_exclusion_count
    passed = ok and tie_ok
    full_summary = dict(record_type="summary", component="nearest_distance_threshold",
        status="FROZEN_CANDIDATE" if passed else "failed", learned_theta=float(model.theta().detach()),
        learned_alpha=float(model.alpha().detach()), held_out_state_count=len(held_out),
        non_tied_pair_count=len(full.target), exact_tie_count=ties[0]["count"],
        complete_nearest_argmin_operator_locked=False)
    verify()
    if passed:
        saved = dict(format_version=1, **full_summary, architecture="logit=alpha*(m-theta); alpha>0; no hidden layers",
            dtype="float64", state_dict=copy.deepcopy(model.state_dict()),
            parameterization="theta=scale*scaled_theta; alpha=softplus(raw_alpha)/scale",
            zero_margin_policy="bypass_threshold_and_preserve_deterministic_cKDTree_owner",
            frozen_margin_path=str(margin.CANDIDATE_PATH), frozen_margin_sha256=MARGIN_SHA256,
            training_summary=summary, training_state_indices=train_states,
            validation_state_indices=validation_states, full_held_out_state_indices=held_out)
        torch.save(saved, CANDIDATE_PATH)
        digest = base._sha256(CANDIDATE_PATH)
        CANDIDATE_PATH.with_suffix(".sha256").write_text(f"{digest}  {CANDIDATE_PATH.name}\n")
        reloaded = NonzeroThreshold()
        reloaded.load_state_dict(torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)["state_dict"])
        assert np.array_equal(route(model, dense_grid())[0], route(reloaded, dense_grid())[0])
        full_summary.update(checkpoint_sha256=digest, checkpoint_path=str(CANDIDATE_PATH))
        summary.update(candidate_saved=True, checkpoint_sha256=digest)
    summary["elapsed_seconds"] = time.monotonic()-started
    base._write_csv(FULL_PATH, [full_summary] + full_rows + ties)
    base._write_csv(BOUNDARY_PATH, [summary] + rows + history)
    print(f"FULL RESULT {full_summary}", flush=True)


if __name__ == "__main__":
    run()
