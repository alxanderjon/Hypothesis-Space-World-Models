"""Validate a lossless ragged representation of dimension-general E-DIRECT state.

This does not fit a model.  Runtime state and teacher actions are deliberately
kept separate.  The representation is a canonical binary form of the minimal
state from which the deterministic implementation regenerates its reservoir,
Voronoi cells, hull, scoring operands, and surrogate proposal.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize


OUT = Path("outputs/lossless_variable_dimension_edirect_state")
MAGIC = b"EDRSTATE"
VERSION = 1
HEADER = struct.Struct("<8sIqqqq32s")
SHO = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases")
BALL4 = Path("outputs/bouncing_ball/full_k_50_transfer/cases")
BALL5 = Path("outputs/bouncing_ball/reflection_peelback/cases")
KEPLER_DIAG = Path("outputs/kepler/five_parameter_n1_diagnosis")
KEPLER_LONG = Path("outputs/kepler/five_parameter_n1_continuation")


def algorithm_fingerprint() -> bytes:
    paths = (
        Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
        Path("experiments/sho/dimension_general_edirect.py"),
        Path("experiments/sho/continuous_surrogate_edirect.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.digest()


FINGERPRINT = algorithm_fingerprint()


@dataclass(frozen=True)
class RoundState:
    dimension: int
    trajectory_id: int
    iteration: int
    lower: np.ndarray
    upper: np.ndarray
    points: np.ndarray
    losses: np.ndarray


def _little_f8(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def encode(state: RoundState) -> bytes:
    d = int(state.dimension)
    n = int(len(state.losses))
    if state.points.shape != (n, d):
        raise ValueError("site array does not match (site_count, dimension)")
    if state.lower.shape != (d,) or state.upper.shape != (d,):
        raise ValueError("bound array does not match dimension")
    # Canonical order is meaningful: coordinate index, then site insertion ID.
    return b"".join((
        HEADER.pack(MAGIC, VERSION, d, n, int(state.trajectory_id),
                    int(state.iteration), FINGERPRINT),
        _little_f8(state.lower).tobytes(order="C"),
        _little_f8(state.upper).tobytes(order="C"),
        _little_f8(state.points).tobytes(order="C"),
        _little_f8(state.losses).tobytes(order="C"),
    ))


def decode(payload: bytes) -> RoundState:
    header = HEADER.unpack_from(payload)
    magic, version, d, n, trajectory_id, iteration, fingerprint = header
    if magic != MAGIC or version != VERSION or fingerprint != FINGERPRINT:
        raise ValueError("state schema/algorithm fingerprint mismatch")
    offset = HEADER.size
    def take(count: int) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(payload, dtype="<f8", count=count, offset=offset).copy()
        offset += 8 * count
        return result
    lower = take(d)
    upper = take(d)
    points = take(n * d).reshape(n, d)
    losses = take(n)
    if offset != len(payload):
        raise ValueError("trailing bytes in state")
    return RoundState(d, trajectory_id, iteration, lower, upper, points, losses)


def bits_equal(first: np.ndarray, second: np.ndarray) -> bool:
    a = _little_f8(first)
    b = _little_f8(second)
    return a.shape == b.shape and np.array_equal(a.view("<u8"), b.view("<u8"))


def action_bytes(rows: list[dict]) -> bytes:
    """Canonical target bytes.  These are labels, never representation input."""
    output = bytearray()
    for row in rows:
        source = str(row.get("source", "unknown")).encode()
        point = _little_f8(row.get("point", ()))
        output.extend(struct.pack("<I", len(source)))
        output.extend(source)
        output.extend(struct.pack("<q", len(point)))
        output.extend(point.tobytes())
        if "loss" in row:
            output.extend(_little_f8([row["loss"]]).tobytes())
    return bytes(output)


def event_rounds(events: list[dict]) -> list[tuple[int, int, list[dict]]]:
    grouped: dict[int, list[dict]] = {}
    for row in events:
        if row["source"] != "initial":
            grouped.setdefault(int(row["iteration"]), []).append(row)
    result = []
    for iteration in sorted(grouped):
        rows = grouped[iteration]
        count = min(int(row["evaluation"]) for row in rows) - 1
        result.append((iteration, count, rows))
    return result


def source_records() -> Iterable[dict]:
    common4 = (np.full(4, -1.99), np.full(4, 1.99))
    common5 = (np.full(5, -1.99), np.full(5, 1.99))
    for path in sorted(SHO.glob("*/events.json")):
        root = path.parent
        result = json.loads((root / "result.json").read_text())
        saved = np.load(root / "stopping_state.npz")
        yield dict(system="sho_full_k", case=root.name,
                   trajectory_id=int(result["case_id"]), lower=common4[0], upper=common4[1],
                   points=saved["points"], losses=saved["losses"],
                   rounds=event_rounds(json.loads(path.read_text())))
    for path in sorted(BALL4.glob("*/events.json")):
        root = path.parent
        case = int(root.name.split("_")[-1])
        saved = np.load(root / "n1_state.npz")
        yield dict(system="ball_full_k", case=root.name,
                   trajectory_id=2_000_000 + case, lower=common4[0], upper=common4[1],
                   points=saved["points"], losses=saved["losses"],
                   rounds=event_rounds(json.loads(path.read_text())))
    for path in sorted(BALL5.glob("*/events.json")):
        root = path.parent
        case = int(root.name.split("_")[-1])
        saved = np.load(root / "n1_state.npz")
        yield dict(system="ball_reflection", case=root.name,
                   trajectory_id=3_000_000 + case, lower=common5[0], upper=common5[1],
                   points=saved["points"], losses=saved["losses"],
                   rounds=event_rounds(json.loads(path.read_text())))

    # The long Kepler artifact contains the diagnostic prefix, so use it once.
    saved = np.load(KEPLER_LONG / "resume.npz")
    lower = np.asarray([-.25, -1., -2., -.5, 1.5])
    upper = np.asarray([.5, 2., 3., 2.5, 4.5])
    rounds = []
    for row in json.loads((KEPLER_DIAG / "rounds.json").read_text()):
        actions = [dict(source="cell", point=value) for value in row["cell_proposals"]]
        if row["continuous_proposal"] is not None:
            actions.append(dict(source="continuous_surrogate",
                                point=((np.asarray(row["continuous_proposal"]) - lower) /
                                       (upper - lower)).tolist()))
        rounds.append((int(row["iteration"]), int(row["evaluations"]), actions))
    for row in json.loads((KEPLER_LONG / "round_summaries.json").read_text()):
        actions = []
        if row["continuous_theta"] is not None:
            actions.append(dict(source="continuous_surrogate",
                                point=((np.asarray(row["continuous_theta"]) - lower) /
                                       (upper - lower)).tolist()))
        rounds.append((int(row["iteration"]), int(row["evaluations_before"]), actions))
    yield dict(system="kepler5", case="failed_case_0_long", trajectory_id=91_000,
               lower=lower, upper=upper, points=saved["points"], losses=saved["losses"],
               rounds=sorted(rounds))


def round_state(record: dict, iteration: int, count: int) -> RoundState:
    d = int(record["points"].shape[1])
    return RoundState(d, int(record["trajectory_id"]), int(iteration),
                      np.asarray(record["lower"], dtype=np.float64),
                      np.asarray(record["upper"], dtype=np.float64),
                      np.asarray(record["points"][:count], dtype=np.float64),
                      np.asarray(record["losses"][:count], dtype=np.float64))


def exact_state_equal(first: RoundState, second: RoundState) -> bool:
    return (first.dimension == second.dimension
            and first.trajectory_id == second.trajectory_id
            and first.iteration == second.iteration
            and bits_equal(first.lower, second.lower)
            and bits_equal(first.upper, second.upper)
            and bits_equal(first.points, second.points)
            and bits_equal(first.losses, second.losses))


def replay(state: RoundState) -> dict:
    api = edirect.composition(state.dimension)
    data = api["_cell_data"](state.points, state.losses,
                             state.trajectory_id, state.iteration)
    surface = GaussianLocalResponse(state.points, state.losses)
    selected_sizes = data.sizes[data.selected]
    size_weight = _normalize(selected_sizes)
    loo = surface.loo_errors(data.selected)
    loo_weight = _normalize(loo)
    weights = .5 * size_weight + .5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    weights[largest_position] = 1.
    proposals = api["_cell_proposals"](state.points, state.losses,
                                       state.trajectory_id, state.iteration)
    proposal_owners = (edirect.cKDTree(state.points).query(proposals, k=1)[1]
                       if len(proposals) else np.empty(0, dtype=np.int64))
    return dict(data=data, surface_values=surface.values, loo=loo, weights=weights,
                largest_position=largest_position, proposals=proposals,
                proposal_owners=proposal_owners)


def exact_replay_equal(first: dict, second: dict) -> bool:
    a, b = first["data"], second["data"]
    return all((
        bits_equal(a.points, b.points),
        np.array_equal(a.owners, b.owners),
        bits_equal(a.distances, b.distances),
        bits_equal(a.sizes, b.sizes),
        np.array_equal(a.selected, b.selected),
        bits_equal(first["surface_values"], second["surface_values"]),
        bits_equal(first["loo"], second["loo"]),
        bits_equal(first["weights"], second["weights"]),
        first["largest_position"] == second["largest_position"],
        bits_equal(first["proposals"], second["proposals"]),
        np.array_equal(first["proposal_owners"], second["proposal_owners"]),
    ))


def choose_replay_rounds(record: dict) -> list[tuple[int, int, list[dict]]]:
    rows = record["rounds"]
    if not rows:
        return []
    targets = [0, len(rows) // 3, 2 * len(rows) // 3, len(rows) - 1]
    if record["system"] == "kepler5":
        # Include the missed-candidate/wrong-alias/continuous-recovery/long-run regions.
        for evaluation in (297, 1000, 1500, 5000, 10000):
            targets.append(min(range(len(rows)), key=lambda i: abs(rows[i][1] - evaluation)))
    return [rows[index] for index in sorted(set(targets))]


def synthetic_controls() -> list[dict]:
    rows = []
    for d in (1, 2, 3, 6, 8):
        for n in (5, 17, 41):
            rng = np.random.default_rng(700_000 + 101 * d + n)
            points = rng.random((n, d))
            losses = np.square(points - .37).sum(axis=1) + np.arange(n) * 1e-12
            state = RoundState(d, 800_000 + d, n, np.full(d, -2.), np.full(d, 3.),
                               points, losses)
            decoded = decode(encode(state))
            first, second = replay(state), replay(decoded)
            rows.append(dict(dimension=d, sites=n,
                             roundtrip_exact=exact_state_equal(state, decoded),
                             replay_exact=exact_replay_equal(first, second),
                             pool_count=len(first["data"].points),
                             selected_count=len(first["data"].selected),
                             proposal_count=len(first["proposals"])))
    return rows


def continuous_equal(first: list[dict], second: list[dict]) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        if (a["case_id"] != b["case_id"] or a["iteration"] != b["iteration"]
                or a["site_count"] != b["site_count"]
                or a["chosen_start"] != b["chosen_start"]
                or a["exact_duplicate_rejected"] != b["exact_duplicate_rejected"]
                or not bits_equal(a["point"], b["point"])
                or not bits_equal([a["prediction"]], [b["prediction"]])
                or len(a["starts"]) != len(b["starts"])):
            return False
        for x, y in zip(a["starts"], b["starts"]):
            if (x["start_id"] != y["start_id"] or x["nfev"] != y["nfev"]
                    or x["nit"] != y["nit"] or x["success"] != y["success"]
                    or x["message"] != y["message"]
                    or not bits_equal(x["start"], y["start"])
                    or not bits_equal(x["point"], y["point"])
                    or not bits_equal([x["prediction"]], [y["prediction"]])):
                return False
    return True


def continuous_controls(records: list[dict]) -> list[dict]:
    """Exercise the full-domain continuous extension on small real states."""
    result = []
    for system in ("sho_full_k", "ball_full_k", "ball_reflection", "kepler5"):
        record = next(row for row in records if row["system"] == system)
        candidates = record["rounds"]
        positions = [0]
        if system == "kepler5":
            positions.append(min(range(len(candidates)),
                                 key=lambda i: abs(candidates[i][1] - 297)))
        for position in positions:
            iteration, count, _ = candidates[position]
            state = round_state(record, iteration, count)
            decoded = decode(encode(state))
            first_trace: list[dict] = []
            second_trace: list[dict] = []
            first = continuous.proposal(state.points, state.losses, state.trajectory_id,
                                        state.iteration, first_trace)
            second = continuous.proposal(decoded.points, decoded.losses,
                                         decoded.trajectory_id, decoded.iteration,
                                         second_trace)
            exact = ((first is None and second is None)
                     or (first is not None and second is not None
                         and bits_equal(first, second)))
            trace_exact = continuous_equal(first_trace, second_trace)
            result.append(dict(system=system, case=record["case"], dimension=state.dimension,
                               iteration=iteration, sites=count, proposal_exact=exact,
                               optimization_trace_exact=trace_exact,
                               start_count=len(first_trace[0]["starts"]),
                               chosen_start=first_trace[0]["chosen_start"]))
    return result


def tie_controls() -> list[dict]:
    result = []
    teacher = edirect.teacher
    for d in (4, 5):
        sizes = np.asarray([.4, .4, .4, .7, .7], dtype=np.float64)
        losses = np.asarray([1., 1., 1., 2., 2.], dtype=np.float64)
        selected = teacher._potentially_optimal_cells(sizes, losses)
        # The same-size representative is the first exact-loss occurrence.
        result.append(dict(dimension=d, selected=selected.tolist(),
                           exact_equal_size_loss_first_retained=bool(0 in selected and
                                                                     1 not in selected and
                                                                     2 not in selected),
                           canonical_order=bool(np.array_equal(
                               selected,
                               np.asarray(sorted(selected,
                                                 key=lambda i: (sizes[i], losses[i], i)))))))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    records = list(source_records())
    state_digests: dict[str, str] = {}
    collisions = []
    systems: dict[str, dict] = {}
    total_coordinate_values = 0
    total_rounds = 0
    replay_rows = []

    for record in records:
        group = systems.setdefault(record["system"], dict(
            cases=0, rounds=0, dimensions=set(), minimum_sites=None, maximum_sites=0,
            exact_roundtrips=0, action_records=0,
        ))
        group["cases"] += 1
        group["dimensions"].add(int(record["points"].shape[1]))
        chosen = {(iteration, count) for iteration, count, _ in choose_replay_rounds(record)}
        for iteration, count, actions in record["rounds"]:
            state = round_state(record, iteration, count)
            payload = encode(state)
            decoded = decode(payload)
            if not exact_state_equal(state, decoded):
                raise AssertionError("non-lossless state round trip")
            digest = hashlib.sha256(payload).hexdigest()
            target_digest = hashlib.sha256(action_bytes(actions)).hexdigest()
            previous = state_digests.setdefault(digest, target_digest)
            if previous != target_digest:
                collisions.append(dict(system=record["system"], case=record["case"],
                                       iteration=iteration, state_digest=digest,
                                       first_action=previous, second_action=target_digest))
            d = state.dimension
            total_coordinate_values += count * d + count + 2 * d
            total_rounds += 1
            group["rounds"] += 1
            group["exact_roundtrips"] += 1
            group["action_records"] += int(bool(actions))
            group["minimum_sites"] = (count if group["minimum_sites"] is None
                                      else min(group["minimum_sites"], count))
            group["maximum_sites"] = max(group["maximum_sites"], count)
            if (iteration, count) in chosen:
                first = replay(state)
                second = replay(decoded)
                exact = exact_replay_equal(first, second)
                if not exact:
                    raise AssertionError("decoded decision replay diverged")
                expected_cells = [row for row in actions if row.get("source") == "cell"]
                saved_prefix_exact = True
                if expected_cells:
                    expected = np.asarray([row["point"] for row in expected_cells])
                    saved_prefix_exact = (len(first["proposals"]) >= len(expected)
                                          and bits_equal(first["proposals"][:len(expected)], expected))
                replay_rows.append(dict(
                    system=record["system"], case=record["case"], dimension=d,
                    iteration=iteration, sites=count, pool_count=len(first["data"].points),
                    selected_count=len(first["data"].selected),
                    proposal_count=len(first["proposals"]), exact=exact,
                    saved_cell_prefix_exact=saved_prefix_exact,
                ))

    synthetic = synthetic_controls()
    continuous_replays = continuous_controls(records)
    ties = tie_controls()
    if collisions or not all(row["roundtrip_exact"] and row["replay_exact"]
                             for row in synthetic) or not all(
                                 row["proposal_exact"] and row["optimization_trace_exact"]
                                 for row in continuous_replays) or not all(
                                     row["exact_equal_size_loss_first_retained"]
                                     and row["canonical_order"] for row in ties):
        raise AssertionError("lossless representation gate failed")
    after = hashes()
    serializable_systems = {}
    for name, row in systems.items():
        row = dict(row)
        row["dimensions"] = sorted(row["dimensions"])
        serializable_systems[name] = row

    schema = dict(
        name="canonical_typed_ragged_edirect_state_v1",
        runtime_fields=[
            "algorithm_fingerprint", "dimension", "site_count", "trajectory_id",
            "iteration", "lower[d]", "upper[d]", "sites[site_id,coordinate_id]",
            "losses[site_id]",
        ],
        materialized_decision_view=[
            "pool[proposal_id,coordinate_id]", "proposal_owner[proposal_id]",
            "proposal_distance[proposal_id]", "cell_size[site_id]",
            "surrogate_log_value[site_id]", "loo_error[selected_position]",
            "potentially_optimal_site_ids[ordered_position]",
            "selected_weight[selected_position]", "ordinary/largest role",
            "cell_proposal_coordinates[proposal_position,coordinate_id]",
            "continuous_start/results as an ordered trace",
        ],
        canonical_orders=dict(
            coordinates="hypothesis parameter order; preserved because identity is semantic",
            sites="evaluation/insertion order; site ID is its zero-based position",
            pool="production np.unique lexicographic order",
            selected_cells="production (cell_size, loss, site_id) order",
            proposals="selected-cell order; canonical first occurrence retained on ties",
        ),
        teacher_outputs_are_separate=True,
        forbidden_runtime_fields=["system identity", "teacher parameters",
                                  "distance to teacher", "Newton-correctability label"],
        coordinate_or_cell_pooling="none",
    )
    validation = dict(
        status="VALIDATED_LOSSLESS_REPRESENTATION",
        schema=schema["name"], algorithm_fingerprint=FINGERPRINT.hex(),
        systems=serializable_systems, total_cases=len(records), total_rounds=total_rounds,
        exact_roundtrips=total_rounds, total_float64_values_roundtripped=total_coordinate_values,
        state_hash_collisions_with_different_actions=len(collisions), collisions=collisions,
        exact_materialized_replays=len(replay_rows),
        materialized_replay_failures=sum(not row["exact"] for row in replay_rows),
        saved_cell_prefix_failures=sum(not row["saved_cell_prefix_exact"] for row in replay_rows),
        replay_rows=replay_rows, synthetic_controls=synthetic,
        continuous_surrogate_controls=continuous_replays,
        canonical_tie_controls=ties,
        dimensions=sorted(set(row["points"].shape[1] for row in records)),
        synthetic_dimensions=sorted(set(row["dimension"] for row in synthetic)),
        checkpoint_hashes_unchanged=before == after,
        protected_checkpoint_count=len(before), checkpoint_saved=False,
    )
    (OUT / "schema.json").write_text(json.dumps(schema, indent=2))
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps({key: validation[key] for key in (
        "status", "total_cases", "total_rounds", "exact_roundtrips",
        "total_float64_values_roundtripped", "state_hash_collisions_with_different_actions",
        "exact_materialized_replays", "materialized_replay_failures",
        "saved_cell_prefix_failures", "dimensions", "synthetic_dimensions",
        "checkpoint_hashes_unchanged", "protected_checkpoint_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
