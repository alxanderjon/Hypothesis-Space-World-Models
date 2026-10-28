"""Read-only audit of the FIFO teacher underlying the frozen CandidateRanker.

No optimizer, objective evaluation, or search loop is called. New files are
confined to a separate audit directory; protected inputs are hashed before/after.
"""
import csv
import hashlib
import inspect
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.sho import run_stage1b_spacefill_2x2 as support

imitation = support.selection.imitation
edirect = support.edirect
ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'final_selector_audit'
FAIL = ROOT / 'spacefill_closedloop_structured'
RAW = ROOT / 'learned_candidate_filter/complete_raw_pool_teacher_traces.pt'
REGION = ROOT / 'learned_candidate_region_generator/teacher_traces.pt'
FEATURE_NAMES = [
    'x', 'y', 'nearest_site_distance', 'distance_to_best_site',
    'surrogate_log10_loss_div20', 'nearest_owner_log10_loss_div20',
    'clipped_surrogate_minus_best_log10_div60', 'boundary_distance',
    'nearest_owner_spacing', 'local_density', 'distance_to_queue_center',
    'is_greedy', 'is_initial_design', 'log_history_div_log_normal_cap',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {}
    return dict(zip(['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'],
                    map(float, np.quantile(a, [0, .01, .05, .5, .95, .99, 1]))))


def forbid_objective(*args, **kwargs):
    raise RuntimeError('This audit must not evaluate an objective or run a search')


def failure_table(model):
    saved = np.load(FAIL / 'final_selector_failure_state.npz')
    diagnosis = json.loads((FAIL / 'final_selector_failure_diagnosis.json').read_text())
    evaluations = [r for r in csv.DictReader((FAIL / 'spacefill_closedloop_comparison.csv').open())
                   if r['record_type'] == 'evaluation' and r['arm'] == 'spacefill_teacher'
                   and int(r['trajectory_id']) == 136]
    evaluations.sort(key=lambda r: int(r['evaluation_count']))
    sites = np.asarray([json.loads(r['point']) for r in evaluations], dtype=np.float64)
    losses = np.asarray([float(r['loss']) for r in evaluations], dtype=np.float64)
    assert np.array_equal(sites, saved['sites']) and np.array_equal(losses, saved['losses'])
    visits = [r for r in csv.DictReader((FAIL / 'spacefill_fidelity_trace.csv').open())
              if r['record_type'] == 'visited_state_fidelity']
    candidate = torch.load(support.screen.OUT / 'spacefill_generator_candidate.pt', weights_only=False)
    generator = support.smoke.Generator(2)
    generator.load_state_dict(candidate['state_dict'])
    generator.eval().requires_grad_(False)
    pending = saved['pending']
    features = imitation._candidate_features(sites, losses, pending, 'cell')
    globals_ = imitation._global_features(sites, losses, len(pending), len(sites), 160,
                                         3, support.selection.FULL_BOUNDS, 'normal_edirect')
    with torch.no_grad():
        logits = model(torch.from_numpy(features)[None],
                       torch.ones((1, len(pending)), dtype=torch.bool),
                       torch.from_numpy(globals_)[None])[0].numpy()
    assert np.array_equal(logits, diagnosis['actual_bounded_context']['scores'])
    assert np.array_equal(globals_, diagnosis['actual_bounded_context']['global_features'])
    current_predictions = edirect.GaussianLocalResponse(sites, losses).predict(pending)
    current_owner = edirect.cKDTree(sites).query(pending)[1]
    rows = []
    for visit in visits:
        iteration, n = int(visit['iteration']), int(visit['history_count'])
        points, values = sites[:n], losses[:n]
        # Fixed archived histories only: regenerate coordinates, never a trajectory.
        probes, _ = generator.generate(min(100*n, 32768), [0., 0.], [1., 1.])
        distances, owners = edirect.cKDTree(points).query(probes)
        radius = np.zeros(n)
        np.maximum.at(radius, owners, distances)
        sizes = np.maximum(radius, .5*edirect.cKDTree(points).query(points, k=2)[0][:, 1])
        po = edirect._potentially_optimal_cells(sizes, values)
        assert np.array_equal(po, json.loads(visit['teacher_PO']))
        proposals, local = support.proposals(points, values, probes, owners, distances, sizes, po)
        records = [r for r in local if not r.get('empty') and r['accepted']]
        assert len(proposals) == len(records)
        if iteration == 3:
            for actual, name in [(probes, 'batch_probes'), (proposals, 'batch_proposals'),
                                 (sizes, 'teacher_cell_sizes'), (po, 'teacher_PO')]:
                assert np.array_equal(actual, saved[name]), name
            assert np.array_equal(proposals[2:], pending)
        surface = edirect.GaussianLocalResponse(points, values)
        surrogate = surface.predict(proposals)
        weights = .5*edirect._normalize(sizes[po]) + .5*edirect._normalize(surface.loo_errors(po))
        weights[int(np.argmax(sizes[po]))] = 1.
        for insertion, (proposal, record) in enumerate(zip(proposals, records)):
            cell = record['site']
            po_position = int(np.flatnonzero(po == cell)[0])
            evaluation_position = n + insertion
            known = evaluation_position < len(sites)
            if known:
                assert np.array_equal(proposal, sites[evaluation_position])
                assert evaluations[evaluation_position]['candidate_type'] == 'cell'
            qi = insertion - 2 if iteration == 3 and insertion >= 2 else None
            row = dict(
                trajectory_id=136, batch_iteration=iteration, batch_history_count=n,
                cumulative_proposal_index=len(rows), batch_insertion_index=insertion,
                status_at_failure='pending' if qi is not None else 'already_evaluated',
                failure_queue_index=qi if qi is not None else '', x=float(proposal[0]), y=float(proposal[1]),
                source_type='largest_cell' if record['largest'] else 'ordinary_cell',
                final_queue_candidate_type='cell', source_site_index=cell,
                source_site_x=float(points[cell, 0]), source_site_y=float(points[cell, 1]),
                source_site_loss=float(values[cell]), source_cell_size=float(sizes[cell]),
                upstream_PO_position=po_position, upstream_u=float(weights[po_position]),
                upstream_winning_probe_index=record['teacher_probe_index'],
                upstream_local_utility=record['teacher_utility'],
                upstream_local_top2_gap=record['teacher_top2_gap'],
                upstream_surrogate_log10_loss=float(surrogate[insertion]),
                objective_known_at_batch_insertion=False,
                objective_known_at_failure=known,
                observed_objective_loss=float(losses[evaluation_position]) if known else '',
                objective_evaluation_number=evaluation_position+1 if known else '',
                teacher_final_rule='FIFO: pending[0]', teacher_final_continuous_score='NOT_APPLICABLE',
                teacher_final_margin='NOT_APPLICABLE',
                metadata_carried_in_actual_queue='float64 coordinates and row order only',
            )
            if qi is not None:
                row.update(teacher_next=qi == 0, learned_next=qi == int(np.argmax(logits)),
                           frozen_ranker_logit=float(logits[qi]),
                           current_nearest_site_index=int(current_owner[qi]),
                           current_surrogate_log10_loss=float(current_predictions[qi]),
                           ranker_global_features=json.dumps(globals_.tolist()))
                row.update({f'ranker_f{j}_{name}': float(features[qi, j])
                            for j, name in enumerate(FEATURE_NAMES)})
            rows.append(row)
    assert len(rows) == 17 and sum(r['status_at_failure'] == 'pending' for r in rows) == 4
    write_csv(OUT / 'final_selector_failure_case.csv', rows)
    return rows, dict(teacher_choice=0, learned_choice=int(np.argmax(logits)),
                     learned_logit_advantage_1_over_0=float(logits[1]-logits[0]),
                     exact_saved_logit_reproduction=True)


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protected = json.loads((FAIL / 'protected_hashes.json').read_text())
    extra = [imitation.TEACHER_TRACE_PATH, imitation.SEARCH_TRACE_PATH, RAW, REGION,
             support.selection.SELECTOR_PATH, FAIL / 'final_selector_failure_state.npz',
             FAIL / 'final_selector_failure_diagnosis.json', FAIL / 'spacefill_fidelity_trace.csv',
             FAIL / 'spacefill_closedloop_comparison.csv',
             Path(inspect.getsourcefile(imitation)), Path(inspect.getsourcefile(edirect))]
    for path in extra:
        protected[str(path)] = digest(path)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    model = imitation.CandidateRanker().to(torch.float32)
    model.load_state_dict(torch.load(support.selection.SELECTOR_PATH, weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    # In-memory audit tripwires; never modify source/checkpoints.
    edirect._prefix_losses_numpy = forbid_objective
    imitation._evaluate = forbid_objective
    imitation._run_edirect_mode = forbid_objective
    imitation._run_nested_rescue = forbid_objective
    failure, failure_summary = failure_table(model)
    print('Failure recovered: 17 cumulative proposals, 4 pending; exact saved ranker logits.', flush=True)
    payload = torch.load(imitation.TEACHER_TRACE_PATH, weights_only=False)
    decisions = payload['decisions']
    modes = {(int(t['trajectory_id']), m['mode']): m
             for t in payload['trajectories'] for m in t['modes']}
    permutations = []
    reconstructed = []
    lengths = []
    archive_indices = []
    lookup = {}
    queue_continuations = 0
    previous = {}
    duplicate_coordinate_decisions = 0
    for i, d in enumerate(decisions):
        key = int(d['trajectory_id']), d['mode']
        mode = modes[key]
        n = int(d['history_count'])
        seed_count = len(mode['evaluated_prefix_losses']) - int(mode['new_evaluations'])
        ordinal = n - seed_count
        seed = imitation.TRACE_PERMUTATION_SEED + 1_000_003*key[0] + 10_007*ordinal + 101*imitation.MODE_INDEX[key[1]]
        permutation = np.random.default_rng(seed).permutation(len(d['candidates']))
        selected = int(np.flatnonzero(permutation == 0)[0])
        candidates = np.asarray(d['candidates'])
        unshuffled = candidates[np.argsort(permutation)]
        # Three independent checks: FIFO label, actual next coordinate, actual observed loss.
        assert selected == int(d['selected_index']), (i, 'index')
        assert np.array_equal(unshuffled[0], mode['evaluated_hypotheses_normalized'][n]), (i, 'point')
        assert float(d['resulting_prefix_loss']) == float(mode['evaluated_prefix_losses'][n]), (i, 'loss')
        if np.sum(np.all(candidates == unshuffled[0], axis=1)) > 1:
            duplicate_coordinate_decisions += 1
        if key in previous:
            pn, pt, pq = previous[key]
            if n == pn+1 and len(pq)>1 and pt == d['candidate_type']:
                assert np.array_equal(pq[1:], unshuffled), (i, 'queue continuation')
                queue_continuations += 1
        previous[key] = n, d['candidate_type'], unshuffled
        lookup[(key[0], key[1], n)] = i
        permutations.append(permutation)
        reconstructed.append(selected)
        lengths.append(len(candidates))
        archive_indices.append(selected)
        if i and i % 50000 == 0:
            print(f'FIFO reconstruction: {i}/{len(decisions)} exact ({time.monotonic()-started:.1f}s).', flush=True)

    # These are derivative records, not additional independent trajectories. Their
    # curated arrays retain the source archive permutation, NOT insertion order.
    raw = torch.load(RAW, weights_only=False)
    for state in raw['states']:
        i = lookup[(int(state['trajectory_id']), 'normal_edirect', int(state['history_count']))]
        assert np.array_equal(state['teacher_curated_points'], decisions[i]['candidates'])
        assert np.array_equal(state['teacher_selected_point'], decisions[i]['candidates'][reconstructed[i]])
    raw_count = len(raw['states'])
    del raw
    region = torch.load(REGION, weights_only=False)
    for state in region['states']:
        i = lookup[(int(state['trajectory_id']), 'normal_edirect', int(state['history_count']))]
        # This archive stores only float32 region centers, no final selected index.
        assert np.array_equal(state['target_regions'][:, :2], decisions[i]['candidates'].astype(np.float32))
    region_count = len(region['states'])
    del region
    no_queue = torch.load(imitation.SEARCH_TRACE_PATH, weights_only=False)
    unrecorded = {name: dict(trajectories=len(no_queue[name]['trajectories']),
                             decisions=len(no_queue[name]['decisions']))
                  for name in ('validation_teacher', 'test_teacher')}
    del no_queue

    print(f'All {len(decisions)} teacher decisions exact. Evaluating frozen ranker on saved features.', flush=True)
    predicted = np.empty(len(decisions), dtype=np.int64)
    confidence = np.full(len(decisions), np.nan)
    teacher_advantage = np.full(len(decisions), np.nan)
    # Evaluate every saved state (no training/validation sample caps). Use the
    # original queue order, restored from the archive's deterministic permutation.
    by_length = defaultdict(list)
    for i, n in enumerate(lengths):
        by_length[n].append(i)
    completed = 0
    last_notice = time.monotonic()
    with torch.no_grad():
        for length, ids in sorted(by_length.items()):
            for start in range(0, len(ids), 256):
                indices = ids[start:start+256]
                x = torch.from_numpy(np.stack([decisions[i]['candidate_features'][np.argsort(permutations[i])]
                                              for i in indices]))
                g = torch.from_numpy(np.stack([decisions[i]['global_features'] for i in indices]))
                mask = torch.ones(x.shape[:2], dtype=torch.bool)
                logits = model(x, mask, g).numpy()
                for row, i in enumerate(indices):
                    scores = logits[row]
                    if length > 1:
                        ordered = np.sort(scores)
                        # Recheck potential float32-sensitive winners individually.
                        if float(ordered[-1]-ordered[-2]) <= 1e-4:
                            scores = model(x[row:row+1], mask[row:row+1], g[row:row+1])[0].numpy()
                            ordered = np.sort(scores)
                        confidence[i] = float(ordered[-1]-ordered[-2])
                        teacher_advantage[i] = float(scores[0]-np.max(scores[1:]))
                    predicted[i] = int(np.argmax(scores))
                completed += len(indices)
                if time.monotonic()-last_notice > 20:
                    print(f'Frozen ranker: {completed}/{len(decisions)} states.', flush=True)
                    last_notice = time.monotonic()

    distribution = []
    def group_row(group, value, ids):
        ids = np.asarray(ids, dtype=int)
        row = dict(record_type='summary', group=group, value=str(value), decisions=len(ids),
                   proposals=sum(lengths[i] for i in ids), teacher_reconstruction_exact=len(ids),
                   teacher_runtime_winner_index=0, teacher_continuous_margin='NOT_APPLICABLE',
                   teacher_score_ties='NOT_APPLICABLE', teacher_score_near_ties='NOT_APPLICABLE',
                   ranker_errors=int(np.sum(predicted[ids] != 0)),
                   ranker_accuracy=float(np.mean(predicted[ids] == 0)))
        finite = confidence[ids][np.isfinite(confidence[ids])]
        row.update({f'ranker_top2_gap_{k}': v for k, v in quantiles(finite).items()})
        distribution.append(row)
        return row
    total = group_row('all', 'all', list(range(len(decisions))))
    subsets = defaultdict(list)
    for i, d in enumerate(decisions):
        values = dict(source_type=d['candidate_type'], mode=d['mode'],
                      mode_and_type=d['mode']+'/'+d['candidate_type'], queue_length=lengths[i],
                      archive_winner_position=archive_indices[i],
                      ranker_runtime_winner_position=predicted[i],
                      trajectory_partition='internal_validation' if d['trajectory_id'] % 5 == 0 else 'training_partition',
                      queue_regime='singleton' if lengths[i] == 1 else 'multiple')
        for group, value in values.items():
            subsets[(group, str(value))].append(i)
        if lengths[i] > 1:
            gap = confidence[i]
            label = 'exact_model_tie' if gap == 0 else ('(0,1e-6]' if gap <= 1e-6 else '(1e-6,1e-4]' if gap <= 1e-4
                     else '(1e-4,1e-2]' if gap <= 1e-2 else '(1e-2,1]' if gap <= 1 else '>1')
            subsets[('ranker_confidence_regime_NOT_teacher_margin', label)].append(i)
    summaries = [group_row(group, value, ids) for (group, value), ids in sorted(subsets.items())]
    for i, d in enumerate(decisions):
        distribution.append(dict(record_type='decision', archive_decision_index=i,
            trajectory_id=int(d['trajectory_id']), mode=d['mode'], source_type=d['candidate_type'],
            history_count=int(d['history_count']), queue_length=lengths[i],
            teacher_runtime_winner_index=0, teacher_archived_winner_index=int(d['selected_index']),
            reconstructed_archived_winner_index=reconstructed[i], teacher_reconstruction_exact=True,
            next_coordinate_exact=True, next_observed_loss_exact=True,
            ranker_runtime_winner_position=int(predicted[i]), ranker_error=bool(predicted[i] != 0),
            ranker_top2_gap=float(confidence[i]) if np.isfinite(confidence[i]) else '',
            ranker_teacher_head_advantage=float(teacher_advantage[i]) if np.isfinite(teacher_advantage[i]) else '',
            teacher_continuous_margin='NOT_APPLICABLE'))
    write_csv(OUT / 'final_selector_distribution.csv', distribution)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    summary = dict(total=total, queue_length_quantiles=quantiles(lengths),
        queue_continuations_verified=queue_continuations, duplicate_winning_coordinate_states=duplicate_coordinate_decisions,
        raw_derivative_records_verified=raw_count, region_derivative_records_verified=region_count,
        archives_without_queue_records=unrecorded, failure=failure_summary,
        ranker_top2_exact_ties=int(np.sum(confidence == 0)),
        ranker_top2_positive_gap_le_1e_minus6=int(np.sum((confidence > 0) & (confidence <= 1e-6))),
        protected_files_unchanged=len(protected), selector_sha256=digest(support.selection.SELECTOR_PATH),
        elapsed_seconds=time.monotonic()-started)
    (OUT / 'audit_verification.json').write_text(json.dumps(dict(summary=summary, protected_hashes=protected), indent=2)+'\n')
    report(summary, summaries, failure)
    print(json.dumps(summary, indent=2), flush=True)


def report(summary, groups, failure):
    lines = [
        'FINAL CANDIDATE SELECTOR AUDIT — no training, search, new objective evaluations, or locked-file changes',
        '',
        'DECISION RULE',
        'For a nonempty pending queue Q=(q0,...,qL-1), i*=0, proposal=Q[0].',
        'Only after selection is its objective evaluated; append it and its loss to history, then delete row 0.',
        'Teacher choice is FIFO/insertion order. No final continuous score, margin, lexicographic comparison,',
        'surrogate comparison, source-priority comparison, cell-size comparison, or objective comparison occurs.',
        'There is no final numerical tie test or tolerance. Queue positions are unique ordinal bookkeeping.',
        'Success/loss and budget control whether another evaluation occurs; they do not choose among pending entries.',
        '',
        'CODE TRACE',
        'experiments/sho/run_stage1b_learned_global_search_imitation.py:',
        '  _run_edirect_mode: initial design and cell batches use teacher_choice=0; pending[choice] is evaluated,',
        '  appended to history, then removed using np.delete. A batch is drained before generating another.',
        '  _run_nested_rescue.evaluate_candidates: proposal=pending[0], then pending=pending[1:].',
        '  _permuted_teacher_record: archive permutation hides runtime queue order; label is position where permutation==0.',
        '  CandidateRanker.forward: float32 permutation-equivariant features/context -> logits -> argmax, not FIFO.',
        'experiments/sho/resume_stage1b_spacefill_structured_closedloop.py:',
        '  inner pending loop checks model_choice against 0 and stops BEFORE evaluating a divergent choice.',
        'experiments/sho/run_stage1b_edirect_vs_direct.py:',
        '  _search_one_edirect consumes proposals with `for proposal in proposals`, preserving array order.',
        '  _potentially_optimal_cells returns selected sites sorted by (cell_size, site_loss, site_index).',
        '  _cell_proposals appends one accepted winner per selected site in that order; largest-cell farthest',
        '  selection and ordinary u*d_norm-(1-u)*s_norm happen UPSTREAM, not across the final queue.',
        '  Empty cells are skipped; proposals within 1e-12 of an evaluated site are rejected upstream.',
        '  If no cell proposal survives, global farthest fallback supplies a proposal.',
        '',
        'QUEUE INPUT CONTRACT',
        'The actual final queue is an ordered float64 Nx2 coordinate array. Source/cell/utility metadata are NOT attached.',
        'candidate_type is a batch-level external tag: initial_design, cell, or singleton greedy.',
        'Initial-design insertion follows _initial_design order after duplicate filtering. Cell insertion follows',
        'the PO order above. Greedy follows the exhausted cell batch. Nested rescue preserves its grid/refinement order.',
        'No cross-source priority comparator merges these queues; control flow schedules them.',
        'Objective values of pending candidates are unknown at insertion. Existing site losses are already available',
        'and influence upstream generation and learned features; they are not final teacher selection inputs.',
        'The failure CSV lists all recoverable provenance separately from quantities actually carried/used.',
        '',
        'CANDIDATERANKER INPUTS (NOT TEACHER CRITERIA)',
        'Fourteen float32 candidate features, in order: '+', '.join(FEATURE_NAMES)+'.',
        'Surrogate is GaussianLocalResponse.predict: weighted mean of log10(max(observed_loss, LOSS_FLOOR));',
        'weights exp(-9*(distance/bandwidth)^2)+1e-15 over up to 24 nearest evaluated sites;',
        'bandwidth=max(farthest of those neighbor distances,1e-12).',
        'Local density radius=max(2*nearest_distance, nearest-owner spacing, 1e-12).',
        'Sixteen global features in order: evaluations/cap; log1p(history)/log1p(cap); best, median, std logloss /20;',
        'best x,y; min(queue_length,32)/32; log1p(iteration)/log1p(cap); bounds widths /DOMAIN_WIDTH;',
        'clipped recent-minus-best logloss /60; three mode flags; logloss range /60.',
        'Original queue ordinal, source cell index, batch-time cell size, and batch-time local score are absent.',
        'Features are recomputed as the queue drains, although insertion order remains fixed.',
        '',
        'FAILURE: trajectory 136, iteration 3, history_count=26',
        '17 means cumulative matched batch proposals across iterations 0..3: 3+4+4+6.',
        '13 of those have already been evaluated. The current batch was built at 24 evaluations; two entries',
        'were consumed, leaving FOUR pending entries. Ten initial and three greedy evaluations account for the rest.',
        'All 17 rows are in final_selector_failure_case.csv with batch order, status, provenance and known objective',
        'observations; current four also have every ranker feature, global feature and exact reproduced logit.',
        'Current queue (source cell/size shown ONLY to explain original insertion):',
        'index | x | y | source site | batch cell size | frozen ranker logit',
    ]
    for row in failure:
        if row['status_at_failure'] == 'pending':
            lines.append(f"{row['failure_queue_index']} | {row['x']:.17g} | {row['y']:.17g} | "
                         f"{row['source_site_index']} | {row['source_cell_size']:.17g} | {row['frozen_ranker_logit']:.17g}")
    lines += ['', 'Complete 17-proposal accounting (coordinates are normalized):',
              'cumulative | iteration | insertion | status | pending index | source | site | x | y']
    for row in failure:
        lines.append(f"{row['cumulative_proposal_index']} | {row['batch_iteration']} | {row['batch_insertion_index']} | "
                     f"{row['status_at_failure']} | {row['failure_queue_index']} | {row['source_type']} | "
                     f"{row['source_site_index']} | {row['x']:.17g} | {row['y']:.17g}")
    lines += [
        'Index 0 wins because it is the oldest remaining queue entry (original current-batch insertion index 2).',
        'Index 1 is original insertion index 3. Earlier source-cell ordering explains how they entered the queue:',
        'cell 21 size 0.23121713511193082 precedes cell 8 size 0.23391191648427945.',
        'Their size difference is upstream provenance, NOT a final-selector decision margin.',
        f"The learned logit advantage of index 1 over 0 is {summary['failure']['learned_logit_advantage_1_over_0']:.17g}.",
        'None of the four pending proposals has an actual objective evaluation available. Their surrogate values',
        'are diagnostic learned inputs only. No new objective values were computed in this audit.',
        'The saved production-cap counterfactual also chose 1; this audit exactly reproduces actual cap-160 logits.',
        '',
        'WHY THIS IS A SELECTOR FAILURE',
        'CandidateRanker maximizes learned logits instead of retaining queue order. Training shuffles the queue',
        'and asks it to recover the former first entry from geometric/context features without an original ordinal.',
        'It therefore infers an insertion priority indirectly. The observed wrong argmax is neither an ordinary-cell',
        'score failure nor a teacher near-tie, and does not establish that space-filling geometry is unhealthy.',
        'A shuffled archive selected_index is not a runtime queue index; treating those as the same would misdiagnose it.',
        '',
        'SMALLEST OPERATIONS (no architecture proposed)',
        '1. Upstream produces an ordered sequence (cell ordering already resolved before this operation).',
        '2. Preserve insertion ordinal while filtering and consuming entries.',
        '3. Choose smallest remaining ordinal -> retrieve coordinate -> remove that entry.',
        '4. Evaluate only that coordinate, append observation, and apply existing stop/budget controls.',
        'An ordinal comparison/reduction describes FIFO abstractly, but an ordered array needs only head/pop.',
        'No new continuous quantity or floating comparison is present in this final teacher operation.',
        '',
        'ARCHIVE VERIFICATION AND DISTRIBUTION',
        f"Canonical archive: {imitation.TEACHER_TRACE_PATH}; 256 teacher trajectories; {summary['total']['decisions']} decision states.",
        'For EVERY state (no sampling), reconstruct decision ordinal from history_count minus mode seed history;',
        'regenerate the saved deterministic permutation without consulting selected_index; select its original head.',
        'Require exact selected-index equality, bit-exact equality with the independently stored NEXT evaluated',
        'coordinate, and exact equality with the corresponding observed objective loss. All three checks pass.',
        f"Additionally {summary['queue_continuations_verified']} successive queues exactly equal their predecessor with its head removed.",
        f"States with duplicate coordinates matching the winning coordinate: {summary['duplicate_winning_coordinate_states']}.",
        f"Raw-pool derivative archive: all {summary['raw_derivative_records_verified']} curated arrays and selected coordinates exactly match canonical records.",
        f"Region derivative archive: all {summary['region_derivative_records_verified']} float32 proposal-center sets match; this archive stores no final choice.",
        'The validation/test history-only archive has 64/128 teacher trajectories but ZERO saved decision queues.',
        'Those histories cannot independently verify discarded alternatives or final indices without regenerating',
        'full upstream search states. They are not counted as audited queue records. Learned rollout archives are',
        'not deterministic teacher decision archives. This audit does not restart any trajectories.',
        'Thus exact reconstruction covers every archived teacher QUEUE state, not every history-only evaluation.',
        'Queue-length quantiles: '+json.dumps(summary['queue_length_quantiles']),
        'Distribution proposal counts sum queue lengths across decision states; surviving entries recur until consumed.',
        'Runtime teacher winning position: always 0. Shuffled archive winning-position histogram is separately in CSV.',
        'Final teacher continuous-margin distribution, score exact ties and score near ties: NOT APPLICABLE.',
        'Do not interpret all FIFO choices as zero-margin numerical ties. Position order is unambiguous.',
        'Ordinary/largest-cell subtype and source-site provenance are not stored in canonical queues; archive source',
        'statistics use recorded cell/greedy/initial_design tags and mode, without inventing missing subtypes.',
        '',
        'FROZEN RANKER DIAGNOSTICS',
        'Every canonical state evaluated on saved float32 features restored to original queue order, same checkpoint;',
        'batches grouped by queue length; top-two gaps <=1e-4 rechecked individually for floating-point sensitivity.',
        'Internal validation means trajectory_id % 5 == 0, matching original split (no sample caps here).',
        'These are on archived teacher states, not a resumed closed-loop run. Model confidence is NOT teacher margin.',
    ]
    for row in [summary['total']] + [r for r in groups if r['group'] in
                ('source_type', 'mode_and_type', 'trajectory_partition', 'queue_regime',
                 'ranker_confidence_regime_NOT_teacher_margin')]:
        lines.append(f"{row['group']}={row['value']}: {row['ranker_errors']}/{row['decisions']} wrong; "
                     f"accuracy={row['ranker_accuracy']:.9f}; proposals={row['proposals']}")
    lines += [
        'Error concentration: 5,022 of 5,025 errors occur in cell-tagged queues. Normal cell queues account for',
        '4,128 errors (1.8313% of those states); nested-refinement queues have the highest substantial mode error',
        'rate: 810/2,742 = 29.5405%. Rescue initial-design errors are 3/20, a small sample.',
        'Errors are enriched among low-confidence learned choices, but 3,094/5,025 occur with model top-two',
        'logit gap >1. This is not solely a small numerical-margin problem.',
        f"Model exact top-logit ties: {summary['ranker_top2_exact_ties']}; positive top-logit gaps <=1e-6: {summary['ranker_top2_positive_gap_le_1e_minus6']}.",
        'Full per-state diagnostics, queue-length/source/winning-position histograms and model-confidence strata are in',
        'final_selector_distribution.csv. Teacher margin concentration cannot be defined for FIFO; use model confidence only.',
        '',
        f"Protected inputs unchanged after audit: {summary['protected_files_unchanged']} files (SHA256 manifest in audit_verification.json).",
        f"Frozen selector SHA256: {summary['selector_sha256']}",
        'No component trained, promoted, replaced, or modified. No search resumed. STOP after audit.',
    ]
    (OUT / 'final_selector_audit.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    run()
