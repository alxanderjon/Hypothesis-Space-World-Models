#!/usr/bin/env python3
"""Read-only replay of archived teacher pairs and exact tie detectors."""
from __future__ import annotations

import csv
import itertools
import platform
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.audit_stage1b_nearest_pairwise as prior
import experiments.sho.run_stage1b_nearest_margin as base
import experiments.sho.run_stage1b_nearest_margin_symmetry as margin
import experiments.sho.run_stage1b_nearest_threshold as threshold

OUT = Path("outputs/sho/stage1B/nearest_tie_detection_audit")
REPORT = OUT / "nearest_tie_detection_audit.txt"
CASES = OUT / "nearest_tie_detection_cases.csv"


def arrays(rows, key, dtype):
    return np.asarray([r[key] for r in rows], dtype=dtype)


def direct_distances(delta):
    squared = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
    return squared, np.sqrt(squared)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in Path("outputs/sho/stage1B").rglob("*locked*") if p.is_file()]
    protected += [margin.CANDIDATE_PATH, margin.CANDIDATE_PATH.with_suffix(".sha256"),
                  threshold.BOUNDARY_PATH, threshold.FULL_PATH]
    hashes = {str(p): base._sha256(p) for p in protected}
    assert hashes[str(margin.CANDIDATE_PATH)] == threshold.MARGIN_SHA256
    with threshold.BOUNDARY_PATH.open() as handle:
        training_summary = next(csv.DictReader(handle))
    theta, alpha = float(training_summary["learned_theta"]), float(training_summary["learned_alpha"])
    assert alpha > 0.0
    frozen = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    frozen.load_state_dict(torch.load(margin.CANDIDATE_PATH, weights_only=False)["state_dict"])
    frozen.eval().requires_grad_(False)
    distance, _ = base._load_locked_distance()
    payload = torch.load(base.turn.RAW_TRACE_PATH, weights_only=False)
    trajectories = base.ordinary_data._trajectory_map(payload)
    rows_out, aggregates, source_hashes = [], {}, {}
    for kind, path, expected_count, expected_ties, expected_ulp in (
        ("probe", prior.PROBE_PATH, 1705952, 105, 1),
        ("spacing", prior.SPACING_PATH, 52777, 690, 8),
    ):
        source_hashes[str(path)] = base._sha256(path)
        total = dict(count=0, teacher_ties=0, one_ulp=0, route_mismatches=0,
            live_owner_changes=0, live_tie_mask_changes=0, live_pair_changes=0,
            live_reported_distance_changes=0, learned_zero_on_nontie=0,
            learned_nonzero_on_tie=0, tie_owner_not_lowest_pair_index=0,
            tie_owner_not_first_neighbor=0, spacing_distance_mismatches=0)
        method_totals = {m: dict(ties_recovered=0, ties_missed=0, non_tie_false_positives=0,
            one_ulp_false_positives=0, route_mismatches=0) for m in ("A_direct_sqrt", "B_squared", "C_ckdtree", "C_live_ckdtree")}
        with path.open() as handle:
            records = (r for r in csv.DictReader(handle) if r["record_type"] in ("comparison", "exact_tie"))
            for state_str, group in itertools.groupby(records, key=lambda r: r["source_state_index"]):
                archived = list(group)
                state = int(state_str)
                q = arrays(archived, "query_index", np.int64)
                s1 = arrays(archived, "current_site_index", np.int64)
                s2 = arrays(archived, "candidate_site_index", np.int64)
                d1 = arrays(archived, "d_current", np.float64)
                d2 = arrays(archived, "d_candidate", np.float64)
                teacher_owner = arrays(archived, "teacher_winner_site_index", np.int64)
                tied = d1 == d2
                ulp = base._ulp_separation(d1, d2) == 1
                assert np.array_equal(tied, arrays(archived, "exact_tie", str) == "True")
                sites, probes, _ = base.distance_metric._state_geometry(payload, trajectories, state)
                queries = probes if kind == "probe" else sites
                tree = cKDTree(sites)
                live_d, live_s = tree.query(queries, k=2 if kind == "probe" else 3)
                j = 0 if kind == "probe" else 1
                live_ties = live_d[q, j] == live_d[q, j+1]
                live_owner = tree.query(queries, k=1)[1][q] if kind == "probe" else tree.query(sites, k=2)[1][q, 1]
                lo, hi, ld1, ld2 = base._ordered_by_site(live_s[q, j], live_s[q, j+1], live_d[q, j], live_d[q, j+1])
                pair_changed = (lo != s1) | (hi != s2)
                delta1, delta2 = queries[q]-sites[s1], queries[q]-sites[s2]
                sq1, direct1 = direct_distances(delta1)
                sq2, direct2 = direct_distances(delta2)
                masks = dict(A_direct_sqrt=direct1 == direct2, B_squared=sq1 == sq2,
                             C_ckdtree=tied, C_live_ckdtree=live_ties)
                a, b = base.distance_metric._predict(distance, delta1), base.distance_metric._predict(distance, delta2)
                scale = np.maximum(a, b)
                x = np.zeros((len(q), 2), dtype=np.float64)
                np.divide(a, scale, out=x[:, 0], where=scale > 0)
                np.divide(b, scale, out=x[:, 1], where=scale > 0)
                learned = margin.predict(frozen, x)
                # Replay only the archived nearest/runner-up pair. No candidate-set reduction.
                logits = np.full(len(q), np.nan)
                logits[~tied] = alpha * (learned[~tied]-theta)
                reconstructed = live_owner.copy()
                reconstructed[~tied] = np.where(logits[~tied] > 0, s2[~tied], s1[~tied])
                invalid = (~tied) & ((learned == 0) | (logits == 0) | ~np.isfinite(logits))
                incorrect = (reconstructed != teacher_owner) | invalid
                # All methods are also compared under the same routing policy.
                for name, mask in masks.items():
                    t = method_totals[name]
                    t["ties_recovered"] += int(np.sum(mask & tied))
                    t["ties_missed"] += int(np.sum(~mask & tied))
                    t["non_tie_false_positives"] += int(np.sum(mask & ~tied))
                    t["one_ulp_false_positives"] += int(np.sum(mask & ulp))
                    alternate_logits = alpha * (learned[~mask]-theta)
                    alternate = live_owner.copy()
                    alternate[~mask] = np.where(alternate_logits > 0, s2[~mask], s1[~mask])
                    wrong = alternate != teacher_owner
                    wrong[~mask] |= (learned[~mask] == 0) | (alternate_logits == 0)
                    t["route_mismatches"] += int(np.sum(wrong))
                selected_delta = queries[q]-sites[reconstructed]
                _, selected_distance = direct_distances(selected_delta)
                spacing_mismatch = selected_distance != np.minimum(d1, d2)
                state_counts = dict(count=len(q), teacher_ties=int(tied.sum()), one_ulp=int(ulp.sum()),
                    route_mismatches=int(incorrect.sum()), live_owner_changes=int(np.sum(live_owner != teacher_owner)),
                    live_tie_mask_changes=int(np.sum(live_ties != tied)), live_pair_changes=int(pair_changed.sum()),
                    live_reported_distance_changes=int(np.sum((ld1 != d1) | (ld2 != d2))),
                    learned_zero_on_nontie=int(np.sum((learned == 0) & ~tied)),
                    learned_nonzero_on_tie=int(np.sum((learned != 0) & tied)),
                    tie_owner_not_lowest_pair_index=int(np.sum(tied & (live_owner != np.minimum(s1, s2)))),
                    tie_owner_not_first_neighbor=int(np.sum(tied & (live_owner != live_s[q, j]))),
                    spacing_distance_mismatches=int(spacing_mismatch.sum()) if kind == "spacing" else 0)
                for k, v in state_counts.items():
                    total[k] += v
                rows_out.append(dict(record_type="state_summary", kind=kind, source_state_index=state, **state_counts))
                disagreement = np.zeros(len(q), dtype=bool)
                for mask in masks.values():
                    disagreement |= mask != tied
                interesting = tied | ulp | disagreement | incorrect | pair_changed | (live_owner != teacher_owner)
                for i in np.flatnonzero(interesting):
                    rows_out.append(dict(record_type="case", kind=kind, source_state_index=state,
                        query_index=int(q[i]), current_site_index=int(s1[i]), candidate_site_index=int(s2[i]),
                        teacher_tie=bool(tied[i]), one_ulp=bool(ulp[i]),
                        query_x=float(queries[q[i], 0]), query_y=float(queries[q[i], 1]),
                        site1_x=float(sites[s1[i], 0]), site1_y=float(sites[s1[i], 1]),
                        site2_x=float(sites[s2[i], 0]), site2_y=float(sites[s2[i], 1]),
                        archived_d1=float(d1[i]), archived_d2=float(d2[i]),
                        direct_d1=float(direct1[i]), direct_d2=float(direct2[i]),
                        squared_d1=float(sq1[i]), squared_d2=float(sq2[i]),
                        squared_d1_hex=float(sq1[i]).hex(), squared_d2_hex=float(sq2[i]).hex(),
                        **{name: bool(mask[i]) for name, mask in masks.items()},
                        live_pair_changed=bool(pair_changed[i]), live_d1=float(ld1[i]), live_d2=float(ld2[i]),
                        learned_margin=float(learned[i]), threshold_logit="" if tied[i] else float(logits[i]),
                        learned_comparison_bypassed=bool(tied[i]),
                        route="TIE_CKDTREE" if tied[i] else "REPLACE" if logits[i] > 0 else "KEEP",
                        teacher_owner=int(teacher_owner[i]), live_bookkeeping_owner=int(live_owner[i]),
                        reconstructed_owner=int(reconstructed[i]), routing_exact=not bool(incorrect[i])))
                if len([r for r in rows_out if r["record_type"] == "state_summary"]) % 20 == 0:
                    print(f"tie audit {kind} state={state} cumulative={total['count']}", flush=True)
        assert (total["count"], total["teacher_ties"], total["one_ulp"]) == (expected_count, expected_ties, expected_ulp)
        aggregates[kind] = (total, method_totals)
        for method, counts in method_totals.items():
            rows_out.append(dict(record_type="method_summary", kind=kind, method=method, **counts))

    # Investigate the prior smoke failure separately; it is an ordinary pair,
    # not one of the 795 held-out decisive teacher ties.
    sites, _, _ = base.distance_metric._state_geometry(payload, trajectories, 40213)
    deltas = sites[13] - sites[[1, 6]]
    squared, direct = direct_distances(deltas)
    locked = base.distance_metric._predict(distance, deltas)
    prior_x = (locked / np.max(locked)).reshape(1, 2)
    prior_m = float(margin.predict(frozen, prior_x)[0])
    pair_tree_d, pair_tree_s = cKDTree(sites[[1, 6]]).query(sites[13], k=2)
    pair_tree_original = np.empty(2, dtype=np.float64)
    pair_tree_original[pair_tree_s] = pair_tree_d
    actual_spacing_d, actual_spacing_s = cKDTree(sites).query(sites[13], k=3)
    rows_out.append(dict(record_type="prior_smoke_failure", kind="ordinary_spacing", source_state_index=40213,
        query_index=13, current_site_index=1, candidate_site_index=6,
        A_direct_sqrt=bool(direct[0] == direct[1]), B_squared=bool(squared[0] == squared[1]),
        C_live_ckdtree=bool(pair_tree_d[0] == pair_tree_d[1]),
        live_d1=float(pair_tree_original[0]), live_d2=float(pair_tree_original[1]),
        direct_d1=float(direct[0]), direct_d2=float(direct[1]), squared_d1=float(squared[0]),
        squared_d2=float(squared[1]), learned_margin=prior_m,
        note="separate ordinary-pair smoke case; two-site tree only for distance-equality cross-check"))
    assert all(base._sha256(Path(p)) == digest for p, digest in hashes.items())
    base._write_csv(CASES, rows_out)

    lines = ["NEAREST EXACT-TIE DETECTION AUDIT", "", "Scope and provenance",
        "Read-only replay of the original 80-state archived decisive-pair audit.",
        "No training, checkpoint writes, complete-candidate argmin assembly, or trajectory search.",
        f"Python {platform.python_version()}; NumPy {np.__version__}; SciPy {scipy.__version__}; Torch {torch.__version__}.",
        f"Frozen margin SHA256: {threshold.MARGIN_SHA256}",
        f"Threshold effective parameters from prior boundary CSV: theta={theta:.17g}, alpha={alpha:.17g}.",
        "There is no frozen threshold checkpoint: its previous gate failed. Parameters were replayed unchanged from the saved report, without retraining or promoting a candidate.",
        f"Protected files checked unchanged: {len(hashes)}.", "", "Exact existing implementation",
        "Production: run_stage1b_edirect_vs_direct.py::_cell_data uses cKDTree(points).query(pool,k=1) for probe owners; query(points,k=2)[0][:,1] for nearest-other spacing distances.",
        "The original nearest_pairwise audit obtains probe nearest/runner-up distances with query(probes,k=2) columns 0,1; spacing alternatives with query(sites,k=3) columns 1,2 (column 0 is self).",
        "The audit orders pair members by original site index, then computes margins=d_current-d_candidate; ties=(margins==0.0), equivalent here to equality of the returned float64 distances.",
        "The 105 and 690 counts therefore refer to equality of rounded cKDTree-returned distances, not equality of squared distances or exact real-number distances.",
        "No tolerance, learned margin, or coordinate-equality predicate labels these ties. Coordinates enter cKDTree's metric. Original site order and query k determine the retained owner; sorting the pair only names current/candidate and does not define tie ownership.",
        "Tie ownership is preserved from the production probe k=1 query and spacing k=2 second neighbor; it is not replaced with lowest index or the first neighbor from the larger-k audit query.",
        "The older 320-state Voronoi tie-rule counterfactual is separate: its <=1e-12 screen only selected candidates; its squared-equality/lowest-index rule was not adopted. It did not produce these 105/690 counts.",
        "", "Compared predicates (float64; no tolerance)",
        "A: delta=query-site; q=delta_x*delta_x+delta_y*delta_y (separate products and addition); d=sqrt(q); compare d1==d2.",
        "B: compare the same pre-sqrt q1==q2.",
        "C: compare archived cKDTree-reported d1==d2. C_live independently regenerates the original query calls.", ""]
    for kind, (total, methods) in aggregates.items():
        lines.append(f"{kind}: {total}")
        lines.extend(f"  {name}: {counts}" for name, counts in methods.items())
    a_exact = all(m["A_direct_sqrt"]["ties_missed"] == 0 and m["A_direct_sqrt"]["non_tie_false_positives"] == 0 for _, m in aggregates.values())
    live_exact = all(t["live_tie_mask_changes"] == 0 for t, _ in aggregates.values())
    lines += ["", "Deterministic pre-check and routing",
        "Authoritative rule: compare the two float64 distances returned by the same cKDTree query contract with exact == BEFORE learned distance/margin/threshold comparison. A tie retains the original production cKDTree owner.",
        f"Direct arithmetic A exactly matches the archived teacher tie mask on every audited pair: {a_exact}.",
        f"Fresh cKDTree query equality exactly matches the archived tie mask: {live_exact}.",
        "This is an observed finite-pool equivalence for A, not a claim that different norm implementations, FMA choices, dimensions, SciPy versions, or tree/site ordering are interchangeable.",
        "Squared-equality misses are rounded-sqrt teacher ties with unequal pre-sqrt sums: 1 probe and 9 spacing cases. They must remain ties under the existing teacher contract. Detailed row identities and squared hex values are in the CSV.",
        "Both A and C recover 795/795 teacher ties with zero false positives on all 1,757,934 non-ties, including all 9 archived one-ULP cases. B recovers 785/795 with no false positives, but is not a complete tie detector.",
        "Ownership still depends on the original cKDTree tree/site order and query contract: 53 probe and 364 spacing ties do not choose the lower index of the audited pair; 104 probe and 676 spacing ties do not choose the first non-self neighbor from the larger-k audit query. Reusing that larger-k first neighbor is therefore not equivalent bookkeeping.",
        "Fresh replay changed some returned distance bits (counts above) but changed no pair identities, tie masks, or production owners. Do not claim bitwise distance reproduction across execution environments from this audit.",
        "For non-ties only: evaluate the unchanged locked metric, normalize by max(d1,d2), evaluate frozen independent-weight margin, then alpha*(margin-theta). Positive replaces, negative keeps. Zero learned margins/logits on non-ties are counted as errors.",
        "The replay uses only the original nearest/runner-up pairs already selected by the teacher. It does not reconstruct an argmin by reducing complete candidate sets.",
        "", "Prior smoke failure",
        f"State 40213, query site 13, candidate sites 1 and 6: A={direct[0]==direct[1]}, B={squared[0]==squared[1]}, two-site cKDTree equality={pair_tree_d[0]==pair_tree_d[1]}, frozen margin={prior_m:.17g}.",
        f"Direct distances: {direct[0]:.17g}, {direct[1]:.17g}; two-site cKDTree distances in original site order: {pair_tree_original[0]:.17g}, {pair_tree_original[1]:.17g}.",
        f"The actual full-state spacing query returns sites {actual_spacing_s.tolist()} (self then neighbors), distances {actual_spacing_d.tolist()}; sites 1 and 6 are not its decisive nearest/runner-up pair.",
        "The smoke pool used np.linalg.norm equality for ordinary pairs, but cKDTree-returned equality for decisive pairs. The earlier failure was therefore a direct-arithmetic teacher tie, not an identified tie in the 105/690 cKDTree audit. These are different numerical teacher contracts.",
        "A is exactly equivalent to C on the archived decisive pool, but this additional ordinary pair disproves universal equivalence. To preserve the Voronoi/spacing teacher, C is the authoritative pre-check. To reproduce the old ordinary-pair dataset, retain that dataset's direct-distance equality instead; do not silently treat the two definitions as interchangeable.",
        "In either contract, tie detection precedes learned comparison and learned_margin==0 is not the tie definition. This audit does not change the pipeline or choose a new cross-contract policy.",
        "", "Routing gate"]
    for kind, (total, _) in aggregates.items():
        lines.append(f"{kind}: {total['count']-total['route_mismatches']}/{total['count']} archived owner decisions exact; mismatches={total['route_mismatches']}.")
    lines.append("CSV contains every teacher tie, every archived non-tied one-ULP case, every predicate/owner/pair disagreement, state coverage summaries, method totals, and the prior smoke failure.")
    lines.append("No component is newly frozen or locked by this audit.")
    lines += ["", "Input hashes"] + [f"{digest}  {p}" for p, digest in {**hashes, **source_hashes}.items()]
    REPORT.write_text("\n".join(lines)+"\n")
    print("\n".join(lines[lines.index("Compared predicates (float64; no tolerance)"):lines.index("Input hashes")]), flush=True)


if __name__ == "__main__":
    run()
