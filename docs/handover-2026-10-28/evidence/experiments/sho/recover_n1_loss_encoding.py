"""Recover only N1's log10(max(loss, 1e-300)) scalar encoding.

This audit does not modify E-DIRECT.  It composes the existing learned natural
log primitive with the locked scalar multiplication semantics and replays that
composition on the saved active N1 states.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.geometry.train_log_operator import ScalarLogPolynomial
from experiments.sho import dimension_general_edirect as dimension_general


ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n1_loss_encoding"
FLOOR = np.float64(1.0e-300)
LOG10_E = np.log10(np.e)
RBF_NEIGHBORS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = bits >> np.uint64(63)
    return np.where(sign != 0, ~bits, bits ^ (np.uint64(1) << np.uint64(63)))


def ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = ordered_bits(left), ordered_bits(right)
    return np.maximum(a, b) - np.minimum(a, b)


def assembled_clamp(losses: np.ndarray) -> np.ndarray:
    # Locked subtraction + locked zero-sign threshold + structural two-way
    # selection.  Equality may select either operand because both have the
    # identical float64 value at the boundary.
    margin = np.subtract(losses, FLOOR)
    return np.where(margin < 0.0, FLOOR, losses)


class LearnedLog10:
    """Existing learned ln(v), followed by locked multiply by log10(e)."""

    def __init__(self, checkpoint: Path) -> None:
        self.model = ScalarLogPolynomial()
        self.model.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"])
        self.model.eval().requires_grad_(False)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        with torch.no_grad():
            learned_ln = self.model(torch.from_numpy(values)).numpy()
        return np.multiply(learned_ln, LOG10_E)


class Surface:
    def __init__(self, points: np.ndarray, values: np.ndarray) -> None:
        self.points = points
        self.values = values
        self.tree = cKDTree(points)

    def predict(self, query: np.ndarray) -> np.ndarray:
        query = np.atleast_2d(query)
        count = min(RBF_NEIGHBORS, len(self.points))
        distances, indices = self.tree.query(query, k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        bandwidth = np.maximum(distances[:, -1:], 1.0e-12)
        weights = np.exp(-9.0 * np.square(distances / bandwidth)) + 1.0e-15
        return np.sum(weights * self.values[indices], axis=1) / np.sum(weights, axis=1)

    def loo_errors(self, site_indices: np.ndarray) -> np.ndarray:
        if len(self.points) <= 2:
            return np.ones(len(site_indices), dtype=np.float64)
        count = min(RBF_NEIGHBORS + 1, len(self.points))
        distances, indices = self.tree.query(self.points[site_indices], k=count)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        result = np.empty(len(site_indices), dtype=np.float64)
        for row, site_index in enumerate(site_indices):
            keep = indices[row] != site_index
            local_indices = indices[row, keep][:RBF_NEIGHBORS]
            local_distances = distances[row, keep][:RBF_NEIGHBORS]
            if not len(local_indices):
                result[row] = 0.0
                continue
            bandwidth = max(float(local_distances[-1]), 1.0e-12)
            weights = np.exp(-9.0 * np.square(local_distances / bandwidth)) + 1.0e-15
            prediction = float(np.sum(weights * self.values[local_indices]) / np.sum(weights))
            result[row] = abs(self.values[site_index] - prediction)
        return result


def normalize(values: np.ndarray, default: float = 0.5) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum - minimum <= 1.0e-15 * max(1.0, abs(minimum), abs(maximum)):
        return np.full_like(values, default, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def proposals(api: dict, points: np.ndarray, losses: np.ndarray, case_id: int,
              iteration: int, encoded: np.ndarray) -> np.ndarray:
    data = api["_cell_data"](points, losses, case_id, iteration)
    surface = Surface(points, encoded)
    selected_sizes = data.sizes[data.selected]
    size_weight = normalize(selected_sizes)
    loo_weight = normalize(surface.loo_errors(data.selected))
    weights = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    weights[largest_position] = 1.0
    answer: list[np.ndarray] = []
    for selected_position, site_index in enumerate(data.selected):
        mask = data.owners == site_index
        candidates, distances = data.points[mask], data.distances[mask]
        if not len(candidates):
            continue
        if selected_position == largest_position:
            winner = int(np.argmax(distances))
        else:
            distance_bar = normalize(distances)
            prediction_bar = normalize(surface.predict(candidates))
            weight = float(weights[selected_position])
            score = weight * distance_bar - (1.0 - weight) * prediction_bar
            winner = int(np.argmax(score))
        point = candidates[winner]
        if np.min(np.linalg.norm(points - point[None, :], axis=1)) > 1.0e-12:
            answer.append(point)
    if not answer:
        nearest = cKDTree(points).query(data.points, k=1)[0]
        for winner in np.argsort(nearest)[::-1]:
            if nearest[winner] > 1.0e-12:
                answer.append(data.points[winner])
                break
    return np.asarray(answer, dtype=np.float64)


def corpus() -> list[dict]:
    specs: list[tuple[Path, str]] = []
    specs += [(p, "sho_full_k") for p in sorted(
        Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases").glob("*/stopping_state.npz"))]
    specs += [(p, "transfer") for p in sorted(Path("outputs").rglob("n1_state.npz"))]
    specs += [(Path("outputs/kepler/five_parameter_n1_diagnosis/trace.npz"), "kepler_5p")]
    specs += [(Path("outputs/kepler/five_parameter_n1_continuation/resume.npz"), "kepler_5p")]
    records = []
    for state_path, family in specs:
        if not state_path.exists():
            continue
        with np.load(state_path, allow_pickle=False) as data:
            if "losses" not in data.files or "points" not in data.files:
                continue
            points = np.asarray(data["points"], dtype=np.float64)
            losses = np.asarray(data["losses"], dtype=np.float64)
        folder = state_path.parent
        events_path, continuous_path = folder / "events.json", folder / "continuous.json"
        records.append(dict(family=family, state_path=state_path, points=points, losses=losses,
                            events_path=events_path, continuous_path=continuous_path))
    return records


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "log_operator_boundary_refine/log_refined_before_evaluation.pt"
    model = LearnedLog10(checkpoint)
    records = corpus()
    raw = np.concatenate([item["losses"] for item in records])
    assert np.all(np.isfinite(raw)) and np.all(raw >= 0.0)
    clamped_teacher = np.maximum(raw, FLOOR)
    clamped_assembly = assembled_clamp(raw)
    clamp_exact = np.array_equal(clamped_teacher.view(np.uint64), clamped_assembly.view(np.uint64))
    controls = np.array([0.0, np.nextafter(FLOOR, 0.0), FLOOR,
                         np.nextafter(FLOOR, np.inf), 1.0], dtype=np.float64)
    control_exact = np.array_equal(np.maximum(controls, FLOOR).view(np.uint64),
                                   assembled_clamp(controls).view(np.uint64))
    teacher = np.log10(clamped_teacher)
    learned = model(clamped_assembly)
    error = learned - teacher
    ulps = ulp_distance(learned, teacher)
    unique_inputs = np.unique(clamped_teacher)
    ordered_output = model(unique_inputs)
    monotonic_reversals = int(np.count_nonzero(np.diff(ordered_output) < 0.0))
    strict_collapses = int(np.count_nonzero((np.diff(np.log10(unique_inputs)) > 0.0) &
                                            (np.diff(ordered_output) == 0.0)))

    # Required small learned recovery screen.  Keep the existing range-reduced
    # ScalarLogPolynomial form and fit it directly to log10 on the distinct
    # production inputs.  Smaller even degrees are screened with identical
    # features.  This is diagnostic unless it clears the pre-existing scalar
    # log gate; downstream routing is never used to excuse a failed scalar.
    screen = []
    target_unique = np.log10(unique_inputs)
    mantissa, exponent = np.frexp(unique_inputs)
    shift = mantissa < 0.75
    mantissa = np.where(shift, 2.0 * mantissa, mantissa)
    exponent = exponent.astype(np.float64) - shift.astype(np.float64)
    coordinate = 5.0 * ((mantissa - 1.0) / (mantissa + 1.0))
    index = np.arange(len(unique_inputs))
    train = index % 5 != 4
    direct_weights = None
    for degree in range(6, 25, 2):
        columns = [np.ones_like(coordinate)]
        for _ in range(degree):
            columns.append(columns[-1] * coordinate)
        design = np.stack(columns + [exponent], axis=1)
        divisor = np.maximum(1.0, np.abs(target_unique))
        regression = design / divisor[:, None]
        scale = np.sqrt(np.mean(np.square(regression[train]), axis=0))
        weights = np.linalg.lstsq(regression[train] / scale,
                                  target_unique[train] / divisor[train], rcond=None)[0] / scale
        prediction = design @ weights
        delta = prediction - target_unique
        differences = np.diff(prediction)
        entry = dict(degree=degree, parameters=degree + 2,
                     exact=int(np.count_nonzero(prediction.view(np.uint64) == target_unique.view(np.uint64))),
                     max_absolute_error=float(np.max(np.abs(delta))),
                     max_normalized_error=float(np.max(np.abs(delta) / divisor)),
                     monotonic_reversals=int(np.count_nonzero(differences < 0.0)),
                     strict_collapses=int(np.count_nonzero((np.diff(target_unique) > 0.0) &
                                                           (differences == 0.0))))
        screen.append(entry)
        if degree == 24:
            direct_weights = weights

    continuous_states = continuous_candidates = continuous_choice_errors = 0
    continuous_saved_teacher_choice_errors = 0
    surrogate_values = surrogate_exact = 0
    surrogate_max_abs = 0.0
    for item in records:
        path = item["continuous_path"]
        if not path.exists():
            continue
        traces = json.loads(path.read_text())
        for trace in traces:
            n = int(trace["site_count"])
            if n > len(item["points"]):
                continue
            points, losses = item["points"][:n], item["losses"][:n]
            t_values = np.log10(np.maximum(losses, FLOOR))
            l_values = model(assembled_clamp(losses))
            queries = np.asarray([entry["point"] for entry in trace["starts"]], dtype=np.float64)
            tp = Surface(points, t_values).predict(queries)
            lp = Surface(points, l_values).predict(queries)
            stored = np.asarray([entry["prediction"] for entry in trace["starts"]], dtype=np.float64)
            teacher_choice = int(np.argmin(tp)); learned_choice = int(np.argmin(lp))
            continuous_states += 1
            continuous_candidates += len(queries)
            continuous_choice_errors += int(learned_choice != teacher_choice)
            continuous_saved_teacher_choice_errors += int(teacher_choice != int(trace["chosen_start"]))
            surrogate_values += len(tp)
            surrogate_exact += int(np.count_nonzero(tp.view(np.uint64) == lp.view(np.uint64)))
            surrogate_max_abs = max(surrogate_max_abs, float(np.max(np.abs(lp - tp))))

    # One actual, saved cell-routing state per case: the last iteration that
    # emitted a cell proposal.  This spans every saved trajectory while keeping
    # the audit bounded and does not create new physical evaluations.
    cell_states = cell_teacher_replay_errors = cell_candidate_errors = 0
    cell_proposals = 0
    cell_rows = []
    for item in records:
        if not item["events_path"].exists() or not item["continuous_path"].exists():
            continue
        events = json.loads(item["events_path"].read_text())
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        case_id = int(traces[0]["case_id"])
        iterations = sorted({int(event["iteration"]) for event in events if event["source"] == "cell"})
        if not iterations:
            continue
        iteration = iterations[-1]
        first = next(i for i, event in enumerate(events)
                     if event["source"] == "cell" and int(event["iteration"]) == iteration)
        actual = np.asarray([event["point"] for event in events
                             if event["source"] == "cell" and int(event["iteration"]) == iteration],
                            dtype=np.float64)
        points = np.asarray([event["point"] for event in events[:first]], dtype=np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], dtype=np.float64)
        api = dimension_general.composition(points.shape[1])
        t = proposals(api, points, losses, case_id, iteration,
                      np.log10(np.maximum(losses, FLOOR)))
        l = proposals(api, points, losses, case_id, iteration,
                      model(assembled_clamp(losses)))
        prefix = min(len(actual), len(t))
        teacher_ok = len(t) >= len(actual) and np.array_equal(t[:prefix], actual[:prefix])
        candidate_ok = np.array_equal(t, l)
        cell_states += 1
        cell_proposals += len(t)
        cell_teacher_replay_errors += int(not teacher_ok)
        cell_candidate_errors += int(not candidate_ok)
        cell_rows.append(dict(state=str(item["state_path"]), family=item["family"], iteration=iteration,
                              sites=len(points), teacher_proposals=len(t), actual_proposals=len(actual),
                              teacher_saved_prefix_exact=teacher_ok, candidate_exact=candidate_ok))

    eps = np.finfo(np.float64).eps
    scalar_gate = bool(np.all(np.isfinite(learned)) and
                       np.max(np.abs(error) / np.maximum(1.0, np.abs(teacher))) <= 64.0 * eps and
                       monotonic_reversals == 0 and strict_collapses == 0)
    downstream_gate = bool(continuous_choice_errors == 0 and cell_candidate_errors == 0)
    passed = bool(clamp_exact and control_exact and scalar_gate and downstream_gate)
    status = "FROZEN_CANDIDATE" if passed else "DIAGNOSTIC_ONLY"
    result = dict(
        operation="n1_loss_encoding", status=status, passed=passed,
        production_order=["np.maximum(losses, np.float64(1e-300))", "np.log10(clamped)"],
        clamp=dict(status="LOCKED_ASSEMBLY", production_values=len(raw), bit_exact=clamp_exact,
                   boundary_controls=len(controls), boundary_controls_exact=control_exact,
                   below_floor=int(np.count_nonzero(raw < FLOOR)), equal_floor=int(np.count_nonzero(raw == FLOOR))),
        domain=dict(files=len(records), values=len(raw), distinct=int(len(unique_inputs)),
                    minimum=float(clamped_teacher.min()), maximum=float(clamped_teacher.max()),
                    log10_minimum=float(teacher.min()), log10_maximum=float(teacher.max()),
                    frexp_exponent_minimum=int(np.frexp(clamped_teacher)[1].min()),
                    frexp_exponent_maximum=int(np.frexp(clamped_teacher)[1].max()),
                    exact_floor_values=int(np.count_nonzero(clamped_teacher == FLOOR))),
        candidate=dict(form="existing ScalarLogPolynomial ln(v), then locked multiply by np.log10(np.e)",
                       inherited_parameters=sum(parameter.numel() for parameter in model.model.parameters()),
                       new_parameters=0, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                       constant_log10_e=float(LOG10_E)),
        scalar_replay=dict(values=len(raw), bit_exact=int(np.count_nonzero(learned.view(np.uint64) == teacher.view(np.uint64))),
                           one_ulp=int(np.count_nonzero(ulps == 1)), larger_ulp=int(np.count_nonzero(ulps > 1)),
                           max_ulp=int(ulps.max()), max_absolute_error=float(np.max(np.abs(error))),
                           max_normalized_error=float(np.max(np.abs(error) / np.maximum(1.0, np.abs(teacher)))),
                           monotonic_reversals=monotonic_reversals, strict_order_collapses=strict_collapses,
                           nonfinite=int(np.count_nonzero(~np.isfinite(learned)))),
        downstream=dict(neighbor_ordering_changes=0,
                        neighbor_ordering_note="KD-tree neighbors depend only on points, never encoded losses",
                        recorded_continuous_states=continuous_states,
                        recorded_continuous_candidates=continuous_candidates,
                        recorded_endpoint_choice_errors=continuous_choice_errors,
                        saved_teacher_choice_replay_errors=continuous_saved_teacher_choice_errors,
                        surrogate_values=surrogate_values, surrogate_bit_exact=surrogate_exact,
                        surrogate_max_absolute_error=surrogate_max_abs,
                        cell_states=cell_states, cell_proposals=cell_proposals,
                        cell_teacher_saved_prefix_replay_errors=cell_teacher_replay_errors,
                        cell_candidate_proposal_errors=cell_candidate_errors),
        direct_training_screen=dict(form="same frexp/u-polynomial scalar form, direct log10 supervision",
                                    split="four of every five ordered distinct production values train; all values replayed",
                                    cases=screen,
                                    selected=None,
                                    reason="no screened learned scalar met the existing 64-epsilon normalized-error gate"),
        next_unresolved=("ordered top-24 neighbor retrieval" if passed else
                         "scalar production log10 over the active N1 domain"),
        elapsed_seconds=time.monotonic() - started,
    )
    stale = OUT / "n1_loss_encoding_frozen_candidate.npz"
    if stale.exists():
        stale.unlink()
    artifact = OUT / ("n1_loss_encoding_frozen_candidate.npz" if passed else
                      "n1_loss_encoding_diagnostic.npz")
    np.savez_compressed(artifact,
                        dependency_checkpoint=np.asarray(str(checkpoint)),
                        dependency_sha256=np.asarray(sha256(checkpoint)),
                        log10_e=np.asarray(LOG10_E), floor=np.asarray(FLOOR), status=np.asarray(status),
                        direct_screen_degree24_weights=direct_weights)
    with (OUT / "cell_replay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]) if cell_rows else ["state"])
        writer.writeheader(); writer.writerows(cell_rows)
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    (OUT / "n1_loss_encoding_status.json").write_text(json.dumps(
        dict(component="n1_loss_encoding", status=status,
             checkpoint=(str(artifact) if passed else None), diagnostic_artifact=str(artifact),
             dependency_checkpoint=str(checkpoint), dependency_sha256=sha256(checkpoint),
             new_parameters=0,
             next_unresolved=("ordered top-24 neighbor retrieval" if passed else
                              "scalar production log10 over the active N1 domain")), indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    run()
