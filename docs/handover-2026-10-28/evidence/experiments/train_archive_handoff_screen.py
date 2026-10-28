#!/usr/bin/env python3
"""Offline screen for chronological archive handoff and site selection.

No search or optimizer is run.  Models consume the finalized lossless archive
states, retain every site as a token, and are saved as screen-only artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from torch import nn

from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

DATA = Path("outputs/chronological_archive_handoff_supervision_expanded")
OUT = Path("outputs/archive_handoff_relational_screen")
SPLITS = ("train", "held_out", "fresh_test")
SYSTEMS = ("reflection_ball", "kepler5")
SEED = 20261007
WIDTH = 32
DEPTH = 3


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_log(values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)


class ArchiveState:
    """Every evaluated site plus deterministic archive relations."""

    def __init__(self, meta: dict, site_rows: list[dict]):
        self.meta = meta
        self.site_rows = sorted(site_rows, key=lambda row: int(row["archive_index"]))
        z = np.load(meta["state_npz"])
        coordinates = np.asarray(z["normalized_sites"], np.float32)
        losses = np.asarray(z["site_losses"], np.float64)
        lower = np.asarray(z["lower"], np.float32)
        upper = np.asarray(z["upper"], np.float32)
        insertion = np.asarray(z["insertion_order"], np.int64)
        n, dimension = coordinates.shape
        if n != int(meta["site_count"]) or n != len(self.site_rows):
            raise ValueError((meta["state_id"], n, len(self.site_rows)))
        if not np.array_equal(insertion, np.arange(n, dtype=np.int64)):
            raise ValueError("archive order is not canonical insertion order")

        self.dimension = dimension
        self.site_count = n
        self.evaluation_count = int(meta["evaluation_count"])
        self.coordinates = torch.from_numpy(coordinates)
        self.lower = torch.from_numpy(lower)
        self.upper = torch.from_numpy(upper)

        loss_order = np.argsort(losses, kind="stable")
        loss_rank = np.empty(n, np.int64); loss_rank[loss_order] = np.arange(n)
        incumbent = int(meta["incumbent_site_index"])
        pairwise = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        if n == 1:
            nearest = np.asarray([1.0], np.float64)
            nearest_index = np.asarray([-1], np.int64)
        else:
            masked = pairwise.copy(); np.fill_diagonal(masked, np.inf)
            nearest_index = np.argmin(masked, axis=1).astype(np.int64)
            nearest = masked[np.arange(n), nearest_index]
        bound_margin = np.minimum(coordinates, 1.0 - coordinates).min(axis=1)
        encoded_loss = np.log10(np.maximum(losses, 1.0e-300))
        scalar = np.column_stack((
            signed_log(losses), encoded_loss.astype(np.float32),
            (loss_rank / max(1, n - 1)).astype(np.float32),
            (np.arange(n) / max(1, n - 1)).astype(np.float32),
            (np.arange(n) == 0).astype(np.float32),
            (np.arange(n) == n - 1).astype(np.float32),
            (np.arange(n) == incumbent).astype(np.float32),
            np.log1p(nearest).astype(np.float32),
            bound_margin.astype(np.float32),
            np.isfinite(losses).astype(np.float32),
        )).astype(np.float32)
        self.scalar = torch.from_numpy(scalar)
        self.previous = torch.arange(n, dtype=torch.long) - 1
        self.next = torch.arange(n, dtype=torch.long) + 1
        self.previous[0] = -1; self.next[-1] = -1
        self.nearest = torch.from_numpy(nearest_index)
        self.incumbent = incumbent
        self.pair_distance = torch.from_numpy(pairwise.astype(np.float32))


def relation_message(tokens: torch.Tensor, indices: torch.Tensor,
                     layer: nn.Module) -> torch.Tensor:
    valid = indices >= 0
    result = torch.zeros_like(tokens)
    if bool(valid.any()):
        result[valid] = layer(tokens[indices[valid]])
    return result


class SiteSelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.distance_bias = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(), nn.Linear(width, 1))
        self.scale = width ** -0.5

    def forward(self, tokens: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        logits = self.q(tokens) @ self.k(tokens).T * self.scale
        logits = logits + self.distance_bias(torch.log1p(distances)[..., None]).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        return self.o(weights @ self.v(tokens))


class QueryAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.scale = width ** -0.5

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.q(query) @ self.k(tokens).T * self.scale, dim=-1)
        return self.o(weights @ self.v(tokens))


class RelationalLayer(nn.Module):
    """Shared archive-token update; no token reduction is performed."""

    def __init__(self, width: int):
        super().__init__()
        self.self_attention = SiteSelfAttention(width)
        self.previous = nn.Linear(width, width, bias=False)
        self.next = nn.Linear(width, width, bias=False)
        self.nearest = nn.Linear(width, width, bias=False)
        self.incumbent = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                nn.Linear(width, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, state: ArchiveState, tokens: torch.Tensor) -> torch.Tensor:
        message = self.self_attention(tokens, state.pair_distance)
        message = message + relation_message(tokens, state.previous, self.previous)
        message = message + relation_message(tokens, state.next, self.next)
        message = message + relation_message(tokens, state.nearest, self.nearest)
        message = message + self.incumbent(tokens[state.incumbent])[None]
        tokens = self.norm1(tokens + message)
        return self.norm2(tokens + self.ff(tokens))


class ArchiveEncoder(nn.Module):
    def __init__(self, width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        # Per-coordinate input: site coordinate, lower/upper bound, ordinal,
        # and inverse dimension.  The GRU weights are shared for every site,
        # coordinate, dimension, and physical system.
        self.coordinate = nn.GRU(5, width, batch_first=True)
        self.scalar = nn.Linear(10, width)
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(RelationalLayer(width) for _ in range(depth))

    def forward(self, state: ArchiveState) -> torch.Tensor:
        count, dimension = state.coordinates.shape
        ordinal = torch.arange(dimension, dtype=torch.float32) / max(1, dimension - 1)
        inverse = torch.full_like(ordinal, 1.0 / dimension)
        coordinate = torch.stack((
            state.coordinates,
            state.lower[None].expand(count, -1),
            state.upper[None].expand(count, -1),
            ordinal[None].expand(count, -1),
            inverse[None].expand(count, -1),
        ), dim=-1)
        _, hidden = self.coordinate(coordinate)
        tokens = self.norm(hidden[-1] + self.scalar(state.scalar))
        for layer in self.layers:
            tokens = layer(state, tokens)
        return tokens


class ArchiveGate(nn.Module):
    def __init__(self, count_aware: bool):
        super().__init__()
        self.count_aware = count_aware
        self.encoder = ArchiveEncoder()
        self.query = nn.Parameter(torch.zeros(1, WIDTH))
        self.attention = QueryAttention(WIDTH)
        self.count = nn.Linear(1, WIDTH) if count_aware else None
        self.norm = nn.LayerNorm(WIDTH)
        self.output = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1))

    def forward(self, state: ArchiveState) -> torch.Tensor:
        sites = self.encoder(state)
        query = self.query
        if self.count_aware:
            value = torch.tensor([[math.log1p(state.evaluation_count)]], dtype=torch.float32)
            query = query + self.count(value)
        query = self.norm(query + self.attention(query, sites))
        return self.output(query).squeeze()


class ArchiveSiteSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ArchiveEncoder()
        self.query = nn.Parameter(torch.zeros(1, WIDTH))
        self.attention = QueryAttention(WIDTH)
        self.context = nn.Linear(WIDTH, WIDTH, bias=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.output = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1))

    def forward(self, state: ArchiveState) -> torch.Tensor:
        sites = self.encoder(state)
        archive_query = self.attention(self.query, sites)
        sites = self.norm(sites + self.context(archive_query))
        return self.output(sites).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def confusion(states: list[ArchiveState], predictions: dict[str, bool]) -> dict:
    count = Counter()
    for state in states:
        truth = bool(state.meta["target_any_good"])
        predicted = bool(predictions[state.meta["state_id"]])
        count["tp" if truth and predicted else "fn" if truth else
              "fp" if predicted else "tn"] += 1
    total = len(states)
    return {
        "states": total, "accuracy": (count["tp"] + count["tn"]) / max(1, total),
        "tp": count["tp"], "fp": count["fp"], "tn": count["tn"], "fn": count["fn"],
    }


@torch.no_grad()
def gate_probabilities(model: ArchiveGate, states: list[ArchiveState]) -> dict[str, float]:
    model.eval()
    return {state.meta["state_id"]: float(torch.sigmoid(model(state))) for state in states}


def gate_loss(model: ArchiveGate, states: list[ArchiveState]) -> float:
    model.eval(); values = []
    with torch.no_grad():
        for state in states:
            target = torch.tensor(float(state.meta["target_any_good"]))
            values.append(float(nn.functional.binary_cross_entropy_with_logits(
                model(state), target)))
    return float(np.mean(values))


def train_gate(model: ArchiveGate, states: list[ArchiveState], epochs: int,
               seed_offset: int) -> dict:
    train = [state for state in states if state.meta["split"] == "train"]
    held = [state for state in states if state.meta["split"] == "held_out"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    best = None; history = []; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); order = list(train); random.Random(SEED + seed_offset + epoch).shuffle(order)
        optimizer.zero_grad(); pending = 0; losses = []
        for state in order:
            target = torch.tensor(float(state.meta["target_any_good"]))
            loss = nn.functional.binary_cross_entropy_with_logits(model(state), target)
            (loss / 8.0).backward(); pending += 1; losses.append(float(loss.detach()))
            if pending == 8:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(); pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad()
        held_value = gate_loss(model, held)
        train_value = gate_loss(model, train)
        record = {"epoch": epoch, "train_loss": train_value, "held_loss": held_value}
        history.append(record)
        if best is None or held_value < best[0]:
            best = (held_value, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print("gate", model.count_aware, json.dumps(record), flush=True)
    model.load_state_dict(best[1])
    return {"best_epoch": best[2], "history": history,
            "seconds": time.monotonic() - started}


def top_list_loss(scores: torch.Tensor, targets: torch.Tensor):
    good = targets.bool(); bad = ~good
    if not bool(good.any()) or not bool(bad.any()):
        return None
    return nn.functional.softplus(scores[bad].max() - scores[good].max())


def selector_loss(model: ArchiveSiteSelector, states: list[ArchiveState]) -> float | None:
    model.eval(); values = []
    with torch.no_grad():
        for state in states:
            target = torch.tensor([row["target_good"] for row in state.site_rows], dtype=torch.bool)
            loss = top_list_loss(model(state), target)
            if loss is not None: values.append(float(loss))
    return float(np.mean(values)) if values else None


def train_selector(model: ArchiveSiteSelector, states: list[ArchiveState], epochs: int) -> dict:
    train = [s for s in states if s.meta["split"] == "train" and s.meta["target_any_good"]]
    held = [s for s in states if s.meta["split"] == "held_out" and s.meta["target_any_good"]]
    eligible = [s for s in train if s.meta["good_site_indices"] and s.meta["bad_site_indices"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    best = None; history = []; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); order = list(eligible); random.Random(SEED + 2000 + epoch).shuffle(order)
        optimizer.zero_grad(); pending = 0; losses = []
        for state in order:
            target = torch.tensor([row["target_good"] for row in state.site_rows], dtype=torch.bool)
            loss = top_list_loss(model(state), target)
            if loss is None: continue
            (loss / 8.0).backward(); pending += 1; losses.append(float(loss.detach()))
            if pending == 8:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(); pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad()
        held_value = selector_loss(model, held)
        train_value = selector_loss(model, train)
        record = {"epoch": epoch, "train_top_list_loss": train_value,
                  "held_top_list_loss": held_value}
        history.append(record)
        if best is None or held_value < best[0]:
            best = (held_value, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print("selector", json.dumps(record), flush=True)
    model.load_state_dict(best[1])
    return {"best_epoch": best[2], "eligible_train_states": len(eligible),
            "history": history, "seconds": time.monotonic() - started}


@torch.no_grad()
def selector_scores(model: ArchiveSiteSelector,
                    positive: list[ArchiveState]) -> dict[str, list[float]]:
    model.eval()
    return {s.meta["state_id"]: [float(x) for x in model(s)] for s in positive}


def selector_metrics(states: list[ArchiveState], scores: dict[str, list[float]]) -> dict:
    top_good = 0; pair_wins = pair_total = 0; ranks = []
    for state in states:
        values = np.asarray(scores[state.meta["state_id"]], np.float64)
        good = np.asarray([row["target_good"] for row in state.site_rows], bool)
        order = np.lexsort((np.arange(len(values)), -values))
        top_good += int(good[order[0]])
        rank = int(np.flatnonzero(good[order])[0] + 1); ranks.append(rank)
        for gi in np.flatnonzero(good):
            for bi in np.flatnonzero(~good):
                pair_wins += int(values[gi] > values[bi]); pair_total += 1
    return {
        "states": len(states), "top_good": top_good,
        "top_good_rate": top_good / max(1, len(states)),
        "best_good_rank_histogram": {str(k): v for k, v in sorted(Counter(ranks).items())},
        "best_good_rank_median": float(np.median(ranks)) if ranks else None,
        "pairwise_good_above_bad": pair_wins, "pairwise_pairs": pair_total,
        "pairwise_accuracy": pair_wins / max(1, pair_total),
    }


def gate_evaluation(model: ArchiveGate, states: list[ArchiveState],
                    baseline_wrong: set[str]) -> dict:
    probabilities = gate_probabilities(model, states)
    predictions = {key: value >= 0.5 for key, value in probabilities.items()}
    result = {}
    for split in (*SPLITS, "all"):
        selected = states if split == "all" else [s for s in states if s.meta["split"] == split]
        result[split] = confusion(selected, predictions)
        result[split]["by_system"] = {
            system: confusion([s for s in selected if s.meta["system"] == system], predictions)
            for system in SYSTEMS
        }
        errors = [s for s in selected if s.meta["state_id"] in baseline_wrong]
        result[split]["count_baseline_error_states"] = confusion(errors, predictions)
    return {"metrics": result, "probabilities": probabilities}


def count_baseline(states: list[ArchiveState]) -> dict:
    train = [s for s in states if s.meta["split"] == "train"]
    votes = defaultdict(Counter)
    for state in train:
        votes[state.evaluation_count][bool(state.meta["target_any_good"])] += 1
    lookup = {count: count_values[True] > count_values[False]
              for count, count_values in votes.items()}
    default = sum(s.meta["target_any_good"] for s in train) > len(train) / 2
    predictions = {s.meta["state_id"]: lookup.get(s.evaluation_count, default) for s in states}
    wrong = {s.meta["state_id"] for s in states
             if predictions[s.meta["state_id"]] != bool(s.meta["target_any_good"])}
    metrics = {}
    for split in (*SPLITS, "all"):
        selected = states if split == "all" else [s for s in states if s.meta["split"] == split]
        metrics[split] = confusion(selected, predictions)
    return {"train_majority_lookup": {str(k): v for k, v in sorted(lookup.items())},
            "default": bool(default), "metrics": metrics,
            "wrong_state_ids": sorted(wrong), "wrong_states": len(wrong)}


def hard_rows(hard_meta: list[dict], state_map: dict[str, ArchiveState],
              scores: dict[str, list[float]]) -> list[dict]:
    output = []
    for meta in hard_meta:
        state = state_map[meta["state_id"]]
        values = np.asarray(scores[meta["state_id"]], np.float64)
        good = np.asarray([row["target_good"] for row in state.site_rows], bool)
        top = int(np.lexsort((np.arange(len(values)), -values))[0])
        best_good = float(values[good].max())
        best_bad = float(values[~good].max())
        newest = state.site_count - 1; incumbent = state.incumbent
        output.append({
            "state_id": meta["state_id"], "system": meta["system"],
            "case_id": meta["case_id"], "split": meta["split"],
            "evaluation_count": meta["evaluation_count"],
            "good_site_indices": [int(x) for x in np.flatnonzero(good)],
            "newest_site_index": newest, "newest_is_good": bool(good[newest]),
            "incumbent_site_index": incumbent, "incumbent_is_good": bool(good[incumbent]),
            "learned_top_site_index": top, "learned_top_is_good": bool(good[top]),
            "good_minus_best_bad_margin": best_good - best_bad,
            "hard_tags": meta["hard_tags"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    before = hashes()
    source_paths = [DATA / "expanded_archive_states.jsonl",
                    DATA / "expanded_archive_site_labels.jsonl",
                    DATA / "expanded_hard_first_positive_states.jsonl"]
    source_hashes = {str(path): sha256(path) for path in source_paths}

    meta = [json.loads(line) for line in source_paths[0].read_text().splitlines()]
    site_rows = [json.loads(line) for line in source_paths[1].read_text().splitlines()]
    hard_meta = [json.loads(line) for line in source_paths[2].read_text().splitlines()]
    grouped = defaultdict(list)
    for row in site_rows: grouped[row["state_id"]].append(row)
    states = [ArchiveState(row, grouped[row["state_id"]]) for row in meta]
    state_map = {state.meta["state_id"]: state for state in states}
    positive = [state for state in states if state.meta["target_any_good"]]
    baseline = count_baseline(states)
    baseline_wrong = set(baseline["wrong_state_ids"])

    torch.manual_seed(SEED + 1)
    count_free = ArchiveGate(False); count_free_fit = train_gate(count_free, states, args.epochs, 100)
    torch.manual_seed(SEED + 2)
    count_aware = ArchiveGate(True); count_aware_fit = train_gate(count_aware, states, args.epochs, 200)
    torch.manual_seed(SEED + 3)
    selector = ArchiveSiteSelector(); selector_fit = train_selector(selector, states, args.epochs)

    count_free_eval = gate_evaluation(count_free, states, baseline_wrong)
    count_aware_eval = gate_evaluation(count_aware, states, baseline_wrong)
    site_scores = selector_scores(selector, positive)
    selector_eval = {}
    for split in (*SPLITS, "all"):
        selected = positive if split == "all" else [s for s in positive if s.meta["split"] == split]
        selector_eval[split] = selector_metrics(selected, site_scores)
        selector_eval[split]["by_system"] = {
            system: selector_metrics([s for s in selected if s.meta["system"] == system], site_scores)
            for system in SYSTEMS
        }
    detailed_hard = hard_rows(hard_meta, state_map, site_scores)
    selector_eval["hard"] = {
        split: selector_metrics(
            [state_map[row["state_id"]] for row in hard_meta if row["split"] == split], site_scores)
        for split in (*SPLITS, "all")
    }
    selector_eval["hard"]["all"] = selector_metrics(
        [state_map[row["state_id"]] for row in hard_meta], site_scores)

    newest_success = sum(bool(s.site_rows[-1]["target_good"]) for s in positive)
    incumbent_success = sum(bool(s.site_rows[s.incumbent]["target_good"]) for s in positive)
    hard_newest = sum(bool(state_map[r["state_id"]].site_rows[-1]["target_good"])
                      for r in hard_meta)
    hard_incumbent = sum(bool(state_map[r["state_id"]].site_rows[
        state_map[r["state_id"]].incumbent]["target_good"]) for r in hard_meta)

    torch.save({"status": "SCREEN_ONLY_NOT_PROMOTED", "task": "archive_gate_count_free",
                "width": WIDTH, "depth": DEPTH, "parameters": parameter_count(count_free),
                "model_state": count_free.state_dict()}, OUT / "archive_gate_count_free_screen_only.pt")
    torch.save({"status": "SCREEN_ONLY_NOT_PROMOTED", "task": "archive_gate_count_aware_diagnostic",
                "width": WIDTH, "depth": DEPTH, "parameters": parameter_count(count_aware),
                "model_state": count_aware.state_dict()}, OUT / "archive_gate_count_aware_diagnostic_only.pt")
    torch.save({"status": "SCREEN_ONLY_NOT_PROMOTED", "task": "archive_site_selector",
                "width": WIDTH, "depth": DEPTH, "parameters": parameter_count(selector),
                "model_state": selector.state_dict()}, OUT / "archive_site_selector_screen_only.pt")

    after = hashes()
    protected_changes = [path for path in set(before) & set(after)
                         if before[path] != after[path] and not path.startswith(str(OUT))]
    report = {
        "status": "SCREEN_COMPLETE_NOT_PROMOTED",
        "dataset": {"trajectories": 374, "states": len(states),
                    "continue_states": sum(not s.meta["target_any_good"] for s in states),
                    "handoff_states": len(positive), "site_labels": len(site_rows),
                    "good_site_labels": sum(r["target_good"] for r in site_rows),
                    "bad_site_labels": sum(not r["target_good"] for r in site_rows),
                    "hard_states": len(hard_meta), "source_hashes": source_hashes},
        "architecture": {
            "width": WIDTH, "relational_depth": DEPTH,
            "coordinate_encoder": "shared GRU over scalar coordinate/bound/ordinal tokens",
            "site_encoder": "shared site feature projection",
            "relations": ["all-site attention with pair-distance bias", "previous site",
                          "next site", "nearest site", "incumbent-to-site"],
            "gate_readout": "learned query token cross-attends contextualized sites",
            "selector_readout": "score from each contextualized site token after archive-query fusion",
            "pooling": "none; no mean/max pooling",
            "ground_truth_input": False, "system_identity_input": False,
            "fixed_coordinate_slots": False,
        },
        "count_only_baseline": baseline,
        "archive_gate_count_free": {
            "parameters": parameter_count(count_free), "fit": count_free_fit,
            "evaluation": count_free_eval,
            "raw_evaluation_or_site_count_input": False,
        },
        "archive_gate_count_aware_diagnostic": {
            "parameters": parameter_count(count_aware), "fit": count_aware_fit,
            "evaluation": count_aware_eval,
            "raw_count_input": "log1p(evaluation_count)",
        },
        "archive_site_selector": {
            "parameters": parameter_count(selector), "fit": selector_fit,
            "evaluation": selector_eval,
            "shortcut_baselines": {
                "newest": {"successes": newest_success, "states": len(positive),
                           "rate": newest_success / len(positive)},
                "lowest_loss_incumbent": {"successes": incumbent_success,
                                          "states": len(positive),
                                          "rate": incumbent_success / len(positive)},
                "hard_newest": {"successes": hard_newest, "states": len(hard_meta)},
                "hard_incumbent": {"successes": hard_incumbent, "states": len(hard_meta)},
            },
            "hard_state_details": detailed_hard,
            "supervision": "one state-balanced softplus(max_BAD - max_GOOD) term per eligible state",
        },
        "existing_checkpoint_hashes_unchanged": not protected_changes,
        "protected_checkpoint_changes": protected_changes,
        "screen_checkpoints_promoted": False,
    }
    atomic_json(OUT / "report.json", report)
    atomic_json(OUT / "checkpoint_hashes_before.json", before)
    atomic_json(OUT / "checkpoint_hashes_after.json", after)
    print(json.dumps({
        "count_baseline": baseline["metrics"],
        "count_free": count_free_eval["metrics"],
        "count_aware": count_aware_eval["metrics"],
        "selector": selector_eval,
        "selector_shortcuts": report["archive_site_selector"]["shortcut_baselines"],
        "checkpoint_integrity": not protected_changes,
    }, indent=2))


if __name__ == "__main__":
    main()
