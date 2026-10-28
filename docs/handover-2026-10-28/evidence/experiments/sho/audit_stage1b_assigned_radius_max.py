#!/usr/bin/env python3
"""Audit teacher assigned-probe max and read-only comparator reuse."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.audit_stage1b_voronoi_cellsize as prior
import experiments.sho.run_stage1b_nearest_margin as base
import experiments.sho.run_stage1b_nearest_margin_symmetry as margin
import experiments.sho.validate_stage1b_nearest_argmin as assembled

OUT = Path("outputs/sho/stage1B/assigned_radius_max_audit")
REPORT = OUT / "assigned_radius_max_audit.txt"
DISTRIBUTION = OUT / "assigned_radius_max_distribution.csv"
NEAR_GAPS = (1e-18, 1e-16, 1e-15, 1e-14, 1e-12, 1e-9, 1e-6)


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if REPORT.exists() or DISTRIBUTION.exists():
        raise RuntimeError("Refusing to overwrite an existing max audit")
    with assembled.SUMMARY.open() as handle:
        protected = [Path(r["checkpoint_path"]) for r in csv.DictReader(handle) if r["checkpoint_path"]]
    protected += [assembled.THRESHOLD.with_suffix(".sha256"), assembled.OPERATOR.with_suffix(".sha256")]
    hashes = {str(p): base._sha256(p) for p in protected}
    operator = torch.load(assembled.OPERATOR, map_location="cpu", weights_only=False)
    assert operator["status"] == "LOCKED"
    for name in ("distance", "margin", "threshold"):
        assert base._sha256(Path(operator[name+"_checkpoint"])) == operator[name+"_sha256"]
    frozen = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    frozen.load_state_dict(torch.load(operator["margin_checkpoint"], weights_only=False)["state_dict"])
    frozen.eval().requires_grad_(False)
    threshold = torch.load(operator["threshold_checkpoint"], weights_only=False)
    theta, alpha = float(threshold["learned_theta"]), float(threshold["learned_alpha"])
    assert alpha > 0

    payload = torch.load(base.turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    train = base.turn._stratified_state_indices(payload["states"], "train", base.turn.TRAIN_SOURCE_STATES, prior.smoke.DATA_SEED)
    held = base.turn._stratified_state_indices(payload["states"], "validation", base.turn.VALIDATION_SOURCE_STATES, prior.smoke.DATA_SEED+1)
    assert set(held) == set(operator["held_out_state_indices"])
    indexed = [(i, "training") for i in train] + [(i, "held_out") for i in held]
    site_rows, state_rows, error_rows = [], [], []
    for position, (state, split) in enumerate(indexed, 1):
        sites, probes, record = base.distance_metric._state_geometry(payload, trajectories, state)
        # Exact production inputs, used only for this teacher max audit. Do not
        # rerun the learned argmin operator or claim new assignment validation.
        distances, owners = cKDTree(sites).query(probes, k=1)
        assert distances.dtype == np.float64 and np.all(np.isfinite(distances)) and np.all(distances >= 0)
        n, p = len(sites), len(probes)
        teacher_radius = np.zeros(n, dtype=np.float64)
        np.maximum.at(teacher_radius, owners, distances)
        counts = np.bincount(owners, minlength=n)

        # Independent sequential strict-comparison scan. Initializing from the
        # first assigned probe handles both empty sets and zero distances.
        scan_radius = np.zeros(n, dtype=np.float64)
        scan_winner = np.full(n, -1, dtype=np.int64)
        comparison_current, comparison_candidate, comparison_probe = [], [], []
        exact_tie_comparisons = 0
        for probe_index, (owner, value) in enumerate(zip(owners, distances)):
            if scan_winner[owner] < 0:
                scan_winner[owner], scan_radius[owner] = probe_index, value
                continue
            current = scan_radius[owner]
            if value == current:
                exact_tie_comparisons += 1
                continue  # Keep the first maximum, exactly as np.argmax.
            comparison_current.append(current)
            comparison_candidate.append(value)
            comparison_probe.append(probe_index)
            if value > current:
                scan_winner[owner], scan_radius[owner] = probe_index, value

        current = np.asarray(comparison_current, dtype=np.float64)
        candidate = np.asarray(comparison_candidate, dtype=np.float64)
        scale = np.maximum(current, candidate)
        assert np.all(scale > 0)
        features = np.column_stack((current/scale, candidate/scale))
        predicted_margin = margin.predict(frozen, features)
        target_margin = features[:, 0]-features[:, 1]
        logits = alpha*(predicted_margin-theta)
        # Original nearest interpretation replaces on positive; max replaces
        # on negative, with unchanged features, weights, theta and alpha.
        predicted_replace = logits < 0
        true_replace = candidate > current
        mistakes = (predicted_replace != true_replace) | (logits == 0) | (predicted_margin == 0)
        ulp = base._ulp_separation(current, candidate) == 1
        margin_error = predicted_margin-target_margin
        for j in np.flatnonzero(mistakes):
            error_rows.append(dict(record_type="comparator_reuse_error", split=split, source_state_index=state,
                probe_index=comparison_probe[j], current_distance=float(current[j]), candidate_distance=float(candidate[j]),
                learned_margin=float(predicted_margin[j]), logit=float(logits[j]), one_ulp=bool(ulp[j])))

        order = np.argsort(owners, kind="stable")
        boundaries = np.concatenate(([0], np.cumsum(counts)))
        radius_mistakes, winner_mistakes, reverse_winner_changes = 0, 0, 0
        state_site_rows = []
        for site in range(n):
            indices = order[boundaries[site]:boundaries[site+1]]
            values = distances[indices]
            count = len(indices)
            row = dict(record_type="site", split=split, source_state_index=state,
                trajectory_id=int(record["trajectory_id"]), site_index=site, assigned_probe_count=count,
                r_probe=float(teacher_radius[site]), reconstructed_radius=float(scan_radius[site]),
                reconstructed_farthest_probe=int(scan_winner[site]),
                radius_exact=bool(scan_radius[site] == teacher_radius[site]))
            radius_mistakes += not row["radius_exact"]
            if count:
                argmax = int(np.argmax(values))
                teacher_winner = int(indices[argmax])
                max_count = int(np.sum(values == teacher_radius[site]))
                reverse_winner = int(indices[::-1][np.argmax(values[::-1])])
                assert float(values[::-1][np.argmax(values[::-1])]) == teacher_radius[site]
                reverse_winner_changes += reverse_winner != teacher_winner
                assert reverse_winner != teacher_winner if max_count > 1 else reverse_winner == teacher_winner
                row.update(teacher_farthest_probe=teacher_winner, maximum_multiplicity=max_count,
                    exact_max_tie=max_count > 1, winner_exact=bool(scan_winner[site] == teacher_winner),
                    reverse_order_farthest_probe=reverse_winner,
                    tied_farthest_probe_indices=";".join(map(str, indices[values == teacher_radius[site]])) if max_count > 1 else "")
            else:
                row.update(teacher_farthest_probe=-1, maximum_multiplicity=0, exact_max_tie=False,
                    winner_exact=bool(scan_winner[site] == -1), reverse_order_farthest_probe=-1)
            winner_mistakes += not row["winner_exact"]
            if count >= 2:
                # Second largest includes multiplicity; it equals the largest
                # whenever two or more distinct assigned probes share the max.
                top = np.partition(values, count-2)[-2:]
                largest, second = float(np.max(top)), float(np.min(top))
                gap = largest-second
                top_ulp = int(base._ulp_separation(np.array([largest]), np.array([second]))[0])
                row.update(largest_distance=largest, second_largest_distance=second, gap=gap,
                    normalized_gap=gap/largest if largest > 0 else 0.0,
                    top_two_ulp_separation=top_ulp, one_ulp_top_two=top_ulp == 1)
                for bound in NEAR_GAPS:
                    row[f"positive_gap_le_{bound:.0e}"] = 0 < gap <= bound
            state_site_rows.append(row)
        site_rows.extend(state_site_rows)
        state_rows.append(dict(record_type="state_summary", split=split, source_state_index=state,
            site_count=n, probe_count=p, occupied_site_count=int(np.sum(counts > 0)),
            zero_assigned_site_count=int(np.sum(counts == 0)),
            pairwise_scan_comparisons=p-int(np.sum(counts > 0)),
            exact_scan_tie_comparisons=exact_tie_comparisons,
            non_tied_comparator_audit_count=len(current), one_ulp_scan_comparisons=int(ulp.sum()),
            comparator_reuse_mistakes=int(mistakes.sum()), one_ulp_comparator_reuse_mistakes=int(np.sum(mistakes & ulp)),
            continuous_margin_maximum_error=float(np.max(np.abs(margin_error))) if len(current) else 0.0,
            radius_mistakes=int(radius_mistakes), farthest_probe_mistakes=int(winner_mistakes),
            reverse_order_winner_changes=reverse_winner_changes,
            zero_teacher_distances=int(np.sum(distances == 0)),
            complete_state_exact=radius_mistakes == 0 and winner_mistakes == 0))
        if position % 40 == 0:
            print(f"radius audit {position}/{len(indexed)} states, {len(site_rows)} sites, elapsed={time.monotonic()-started:.1f}s", flush=True)

    summaries, distributions = [], []
    for split in ("all", "training", "held_out"):
        selected = site_rows if split == "all" else [r for r in site_rows if r["split"] == split]
        states = state_rows if split == "all" else [r for r in state_rows if r["split"] == split]
        counts = np.asarray([r["assigned_probe_count"] for r in selected])
        radii = np.asarray([r["r_probe"] for r in selected])
        eligible = [r for r in selected if "gap" in r]
        gaps = np.asarray([r["gap"] for r in eligible])
        positive_gaps = gaps[gaps > 0]
        summary = dict(record_type="summary", split=split, state_count=len(states), site_count=len(selected),
            probe_count=sum(r["probe_count"] for r in states), occupied_site_count=int(np.sum(counts > 0)),
            zero_assigned_site_count=int(np.sum(counts == 0)), sites_with_at_least_two_probes=len(eligible),
            largest_assigned_set=int(counts.max()), assigned_count_mean=float(counts.mean()),
            r_probe_min=float(radii.min()), r_probe_max=float(radii.max()),
            closest_positive_gap=float(positive_gaps.min()) if len(positive_gaps) else "",
            exact_max_tie_sites=sum(r["exact_max_tie"] for r in selected),
            one_ulp_top_two_sites=sum(r["one_ulp_top_two"] for r in eligible),
            maximum_max_tie_multiplicity=max(r["maximum_multiplicity"] for r in selected))
        for key in ("pairwise_scan_comparisons", "exact_scan_tie_comparisons", "non_tied_comparator_audit_count",
                    "one_ulp_scan_comparisons", "comparator_reuse_mistakes", "one_ulp_comparator_reuse_mistakes",
                    "radius_mistakes", "farthest_probe_mistakes", "reverse_order_winner_changes", "zero_teacher_distances"):
            summary[key] = sum(r[key] for r in states)
        summary["exact_state_count"] = sum(r["complete_state_exact"] for r in states)
        for bound in NEAR_GAPS:
            summary[f"positive_top_two_gap_le_{bound:.0e}"] = int(np.sum((gaps > 0) & (gaps <= bound)))
        summaries.append(summary)
        for quantity, values in (("assigned_probe_count", counts), ("r_probe", radii),
                                 ("top_two_gap", gaps), ("positive_top_two_gap", positive_gaps)):
            distributions.extend(dict(split=split, **r) for r in prior._percentile_rows(quantity, values))
        for count, frequency in zip(*np.unique(counts, return_counts=True)):
            distributions.append(dict(record_type="assigned_count_histogram", split=split,
                assigned_probe_count=int(count), site_count=int(frequency)))
    assert all(base._sha256(Path(p)) == digest for p,digest in hashes.items())
    summary = summaries[0]
    assert summary["radius_mistakes"] == 0 and summary["farthest_probe_mistakes"] == 0
    assert summary["exact_state_count"] == len(indexed)
    base._write_csv(DISTRIBUTION, summaries+distributions+state_rows+site_rows+error_rows)
    closest = min((r for r in site_rows if r.get("gap", 0) > 0), key=lambda r:r["gap"])
    largest = max(site_rows, key=lambda r:r["assigned_probe_count"])
    lines = ["ASSIGNED-PROBE RADIUS MAX AUDIT", "", "Scope",
        "All 320 states in the existing Voronoi/cell-size audit pool: 240 training and 80 held-out. This is the existing stratified audit pool, not every state stored in the raw trace.",
        "Exact production probe ownership and returned float64 distances were replayed. Learned ownership/distance assembly was not rerun; its separate 80-state lock validation remains unchanged.",
        "No training, checkpoint writes, trajectory search, learned max assembly, or radius/max locking occurred.",
        f"Protected checkpoint/hash files verified unchanged: {len(hashes)}.", "", "Exact teacher behavior",
        "run_stage1b_edirect_vs_direct.py::_cell_data initializes sizes=np.zeros(number_of_sites), then np.maximum.at(sizes,owners,distances). This is r_probe before the separate 0.5-spacing floor and final maximum.",
        "Empty assigned set: r_probe=+0.0, no farthest probe (CSV sentinel -1). A nonempty all-zero set also has radius zero, but its first probe is a valid witness. All audited distances are finite and nonnegative.",
        "The radius reduction itself stores ONLY a value, no winning probe. Equal maxima leave the numerical radius unchanged regardless of probe order. No tolerance is used.",
        "When production explicitly needs a farthest probe, _cell_proposals uses np.argmax(distances) in the largest selected cell. Boolean masking retains original probe order, so the first equal maximum wins: lowest global probe index within that assigned set.",
        "The raw probe pool is np.unique(...,axis=0), hence lexicographically ordered before masking. Indices represent this pool order, not site-order tie ownership. The audit extends the same np.argmax witness convention to every nonempty set; production does not store such witnesses for every site.",
        "Other selected cells maximize a mixed utility score, not solely distance; proposal-duplication filtering and global rescue are outside this radius audit. Their 1e-12 exclusion threshold is not a radius-max tie tolerance.",
        "Reversing each set's probe order leaves every radius unchanged but changes the first-winner index precisely in the max-tied sets (counts below). Exact equality keeps the current winner in an original-order scan.",
        "", "Distributions and reconstruction"]
    for s in summaries:
        lines += [json.dumps(s, indent=2)]
    lines += ["Near-tie counts use strictly positive gap <= each displayed bound, excluding exact ties. These bounds are descriptive bins only, never tie-detection tolerances.",
        "Second-largest distance counts multiplicity: repeated maxima produce gap=0. One-ULP means adjacent nonnegative float64 bit patterns of the top two distances, excluding equal values.",
        "The CSV includes every site's largest/second-largest distance and gap for assigned counts >=2, plus exact tied winner indices, state summaries, percentiles, and the full assigned-count histogram.",
        f"Largest assigned set: state={largest['source_state_index']}, site={largest['site_index']}, count={largest['assigned_probe_count']}.",
        f"Closest positive top-two gap: state={closest['source_state_index']}, site={closest['site_index']}, largest={closest['largest_distance']:.17g}, second={closest['second_largest_distance']:.17g}, gap={closest['gap']:.17g}, ULPs={closest['top_two_ulp_separation']}.",
        "", "Reuse of the existing continuous comparator",
        f"Frozen independent weights: {frozen.weight.detach().numpy()[0].tolist()}; unchanged threshold theta={theta:.17g}, alpha={alpha:.17g}.",
        "For current d1 and candidate d2, keep the SAME S=max(d1,d2)>0, x1=d1/S, x2=d2/S and m_hat=w1*x1+w2*x2. Candidate is larger exactly when m<0; max REPLACE uses logit<0, max KEEP uses logit>0. This reverses winner interpretation only.",
        "No new continuous arithmetic, learned weights, or threshold fit is required for the tested non-tied distance comparisons. Retaining/selecting a distance and probe index is reduction bookkeeping.",
        "Exact ties are detected by equality of the provided teacher distances BEFORE normalization/learned comparison; keep the first probe. No cKDTree ownership tie rule is needed inside an already-assigned set. learned_margin==0 is not used to identify ties.",
        "Empty sets initialize to radius zero with no winner. The first assigned probe initializes the winner, bypassing comparison. Both-zero inputs are an exact tie, so no S=0 division occurs; one-zero/one-positive comparisons use the same primitive.",
        f"Unchanged comparator tested on {summary['non_tied_comparator_audit_count']} non-tied comparisons along the EXACT teacher scan, including {summary['one_ulp_scan_comparisons']} one-ULP comparisons: mistakes={summary['comparator_reuse_mistakes']}, one-ULP mistakes={summary['one_ulp_comparator_reuse_mistakes']}.",
        "This is a read-only primitive reuse audit on teacher inputs, not a new end-to-end learned max operator validation or lock.",
        "", "Independent full max reconstruction",
        "Scanned probes sequentially in original order, maintaining one current value/winner per site; initialized the first assigned probe and replaced only for strict candidate_distance>current_distance. Exact equal distances kept current. Compared against np.maximum.at values and independent per-set np.argmax witnesses.",
        f"Exact radius values: {summary['site_count']-summary['radius_mistakes']}/{summary['site_count']}; exact winner/sentinel: {summary['site_count']-summary['farthest_probe_mistakes']}/{summary['site_count']}; exact states: {summary['exact_state_count']}/{summary['state_count']}.",
        f"Elapsed seconds: {time.monotonic()-started:.3f}", "", "Protected input hashes"]
    lines += [f"{h}  {p}" for p,h in sorted(hashes.items())]
    REPORT.write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2), flush=True)
    print(f"Saved {REPORT} and {DISTRIBUTION}", flush=True)


if __name__ == "__main__":
    run()
