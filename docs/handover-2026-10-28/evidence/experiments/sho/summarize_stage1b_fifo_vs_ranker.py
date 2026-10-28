"""Summarize the bounded selector ablation without launching any computation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT=Path('outputs/sho/stage1B/fifo_vs_ranker_closedloop')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_and_enrich(rows,plan,summary):
    """Validate persisted experiment records without evaluating any objective."""
    assert all(digest(p)==h for p,h in plan['protected_hashes'].items())
    pairs={}
    states={}
    for r in rows:
        if r['record_type']!='trajectory':continue
        key=r['split'],r['trajectory_id']
        pairs.setdefault(key,{})[r['arm']]=r
        run=torch.load(OUT/f"{key[0]}_{key[1]}_{r['arm']}.pt",weights_only=False)
        events=run['events'];states[key,r['arm']]=run
        assert len(events)==int(r['search_evaluations'])
        assert [e['evaluation'] for e in events]==list(range(1,len(events)+1))
        best=np.minimum.accumulate([e['loss'] for e in events])
        assert np.array_equal(best,[e['best_loss'] for e in events])
        hits=np.flatnonzero(best<=plan['gate'])
        assert bool(len(hits))==(r['global_handoff_success']=='True')
        if len(hits):assert int(hits[0])+1==int(r['evaluations_to_handoff'])
        normal=run['modes'][0]
        assert normal.new_evaluations<=plan['normal_cap']
        for mode in run['modes']:
            if mode.mode=='boundary_rescue':assert mode.new_evaluations<=plan['rescue_cap']
        last_improvement=0;longest=0;previous=float('inf')
        for e in events:
            if e['best_loss']<previous:last_improvement=e['evaluation']
            previous=e['best_loss']
            longest=max(longest,e['evaluation']-last_improvement)
        r.update(normal_evaluations=normal.new_evaluations,normal_best_loss=normal.best_loss,
                 maximum_evaluations_without_improvement=longest,
                 handoff_source_mode=events[int(hits[0])]['mode'] if len(hits) else '',
                 runtime_note='recorded execution seconds exclude discarded replay work and early pre-checkpoint intervals')
    matched_probe_states=0
    for key,pair in pairs.items():
        assert set(pair)=={'A','B'}
        a,b=states[key,'A'],states[key,'B']
        ea,eb=a['events'],b['events']
        assert np.array_equal([e['point'] for e in ea[:10]],[e['point'] for e in eb[:10]])
        assert np.array_equal([e['loss'] for e in ea[:10]],[e['loss'] for e in eb[:10]])
        pools=[]
        for events in (ea,eb):
            pool={};seen=set()
            for e in events:
                if e['kind']!='cell' or e['iteration'] in seen:continue
                seen.add(e['iteration'])
                pool[(e['iteration'],e['evaluation']-1)]=(e['probe_sha256'],e['probe_count'])
            pools.append(pool)
        common=set(pools[0])&set(pools[1])
        assert all(pools[0][k]==pools[1][k] for k in common)
        matched_probe_states+=len(common)
        info=dict(common_generation_states_with_identical_probe_hash=len(common),
                  generation_states_only_in_A=len(set(pools[0])-set(pools[1])),
                  generation_states_only_in_B=len(set(pools[1])-set(pools[0])))
        if a['success'] and b['success']:
            info['paired_handoff_evaluations_B_minus_A']=int(pair['B']['evaluations_to_handoff'])-int(pair['A']['evaluations_to_handoff'])
            aa=np.asarray([a['recovery']['final_a'],a['recovery']['final_b']])
            bb=np.asarray([b['recovery']['final_a'],b['recovery']['final_b']])
            info['paired_final_parameter_max_difference']=float(np.max(np.abs(aa-bb)))
            info['polished_parameters_exact_between_arms']=bool(np.array_equal(aa,bb))
        pair['A'].update(info);pair['B'].update(info)
    assert len(pairs)==summary['completed_matched_trajectories']
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'fifo_vs_ranker_closedloop.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    verification=dict(completed_pairs_verified=len(pairs),identical_initial_designs=True,
        recorded_observation_loss_curves_exact=True,search_budgets_respected=True,
        matched_probe_generation_states_verified=matched_probe_states,
        protected_files_unchanged=len(plan['protected_hashes']),
        harness_sha256=digest('experiments/sho/run_stage1b_fifo_vs_ranker.py'),
        report_implementation_sha256=digest(__file__))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return verification


def main():
    rows=list(csv.DictReader((OUT/'fifo_vs_ranker_closedloop.csv').open()))
    # The worker-start event was observed immediately before the coordinator
    # received the test:73 exception and terminated the remaining workers.
    for r in rows:
        if (r['record_type']=='not_run_hard_stop' and r['split']=='train'
                and r['trajectory_id']=='216' and r['arm']=='A'):
            r['record_type']='interrupted_before_first_snapshot'
            r['status']='START_OBSERVED_NO_DURABLE_ARM_STATE'
    diagnosis_path=OUT/'frozen_pipeline_stop_diagnosis.json'
    if diagnosis_path.exists():
        diagnosis=json.loads(diagnosis_path.read_text())
        rows=[r for r in rows if r['record_type']!='frozen_pipeline_failure']
        for edge in diagnosis.get('edges',[]):
            if edge['sign_mismatch']:
                rows.append(dict(record_type='frozen_pipeline_failure',**diagnosis['case'],
                    arm=diagnosis['arm'],evaluation=diagnosis['evaluated_sites'],
                    iteration=diagnosis['iteration'],search_best_loss=diagnosis['best_loss'],
                    status='UPSTREAM_FROZEN_FAILURE_NOT_RANKER_RECOVERY_LOSS',
                    reason=diagnosis['reproduced_error'],**edge))
    runs=[r for r in rows if r['record_type']=='trajectory']
    plan=json.loads((OUT/'plan.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    verification=verify_and_enrich(rows,plan,summary)
    pairs={}
    for r in runs:pairs.setdefault((r['split'],r['trajectory_id']),{})[r['arm']]=r
    yes=lambda r,k:r.get(k)=='True'
    lines=[
        'FIFO VERSUS FROZEN CANDIDATERANKER — SOBOL ONLY',
        f"Completed {len(pairs)}/{plan['planned_trajectories']} planned matched trajectories; stop={summary['stop_reason'] or 'screen completed'}.",
        'No retraining, checkpoint replacement, production Sobol modification, or space-filling generator use.',
        'Only normal-mode final cell-queue selection differs. Both arms start with the same evaluated ten-point design.',
        'Both use the same frozen distance/argmin, radius/half-spacing/max, groups, representatives, hull,',
        'improvement condition, old ordinary scorer, and unchanged largest-cell/greedy scaffolding.',
        'A is FIFO on that common frozen pipeline; it is not a separate all-teacher upstream geometry arm.',
        'Common deterministic strip/nested rescue is unchanged in both arms, including FIFO inside rescue.',
        f"Budgets: normal {plan['normal_cap']}, boundary rescue {plan['rescue_cap']}; existing finite nested schedule.",
        'Handoff stops search at loss <=1e-4. Best-loss-at-budget reports the retained incumbent after earlier handoff,',
        'or the actually observed budget-end loss for a failed arm; no further global evaluations follow handoff.',
        'Frozen learned Powell: existing 500-evaluation stage with existing continuation through at most 1500;',
        'same stochastic line seeds in paired arms. Existing safeguarded Newton: at most 20 accepted steps.',
        'Local boxes follow the existing nominal-spacing rule or the arm\'s rescue strip; hidden parameters are',
        'used only for case stratification and posthoc error/rollout reporting.',
        'Production Sobol uses identical generator ID/iteration/N schedules. Once histories differ, candidate',
        'ownership, later queue contents and possibly iteration/N can differ as consequences of scheduling.',
        '',
        'PER TRAJECTORY (A=FIFO, B=ranker; no failures omitted)',
        'split:id | stratum | handoff A/B | evaluations A/B | first divergence | final prefix loss A/B | max parameter error A/B',
    ]
    deltas=[];both_hits=0;global_losses=[];floor_losses=[]
    for key,pair in pairs.items():
        a,b=pair['A'],pair['B']
        sa,sb=yes(a,'global_handoff_success'),yes(b,'global_handoff_success')
        if sa and sb:
            deltas.append((int(b['evaluations_to_handoff'])-int(a['evaluations_to_handoff']),':'.join(key)))
            both_hits+=1
        if sa and not sb:global_losses.append(':'.join(key))
        if yes(a,'reaches_1e_minus24') and not yes(b,'reaches_1e_minus24'):floor_losses.append(':'.join(key))
        fmt=lambda r,k: f"{float(r[k]):.6g}" if r.get(k) else 'NOT_REACHED'
        lines.append(f"{':'.join(key)} | {a['stratum']} | {sa}/{sb} | "
                     f"{a['evaluations_to_handoff'] or 'FAIL'}/{b['evaluations_to_handoff'] or 'FAIL'} | "
                     f"{a['first_divergence_evaluation'] or 'none'} | "
                     f"{fmt(a,'final_prefix_loss')}/{fmt(b,'final_prefix_loss')} | "
                     f"{fmt(a,'final_parameter_max_error')}/{fmt(b,'final_parameter_max_error')}")
        lines.append(f"  best global loss A/B={float(a['search_best_loss']):.8g}/{float(b['search_best_loss']):.8g}; "
                     f"same observed-set return count={a['evaluated_set_reconvergence_count']}; "
                     f"later identical next points={a['later_identical_next_points']}; "
                     f"final handoff point exact={a['final_global_point_exact']}; "
                     f"ordered-history reconvergence={a['full_ordered_history_reconverged']}.")
        lines.append(f"  ranker non-FIFO choices={b['ranker_nonFIFO_choices']}; "
                     f"final evaluated-point sets exact={a['final_evaluated_sets_equal']}; "
                     f"longest no-improvement stretch A/B="
                     f"{a['maximum_evaluations_without_improvement']}/{b['maximum_evaluations_without_improvement']} evaluations.")
        if sa and sb:
            parameter_delta=max(abs(float(a[k])-float(b[k])) for k in ('final_a','final_b'))
            lines.append(f"  polished A/B parameter difference={parameter_delta:.8g}; "
                         f"analytic-reference prefix floor={float(a['reference_prefix_loss']):.8g}; "
                         f"1000-period MSE A/B={float(a['rollout_1000_period_mse']):.8g}/{float(b['rollout_1000_period_mse']):.8g}.")
    lines += ['', 'STRATA (numerators/denominators; unfinished cases are not failures or successes)']
    for stratum in ('interior','near_boundary','very_near_boundary'):
        for arm in ('A','B'):
            rs=[r for r in runs if r['stratum']==stratum and r['arm']==arm]
            success=[r for r in rs if yes(r,'global_handoff_success')]
            hits=[int(r['evaluations_to_handoff']) for r in success]
            lines.append(f"{stratum} {arm}: handoff {len(success)}/{len(rs)}, "
                         f"<=1e-29 {sum(yes(r,'reaches_1e_minus29') for r in rs)}/{len(rs)}, "
                         f"<=1e-28 {sum(yes(r,'reaches_1e_minus28') for r in rs)}/{len(rs)}, "
                         f"handoff evaluations={hits}; failures="+
                         str([r['split']+':'+r['trajectory_id'] for r in rs if not yes(r,'global_handoff_success')]))
    lines += ['', 'PRIMARY QUESTIONS']
    lines.append('Global reliability losses (FIFO succeeds, ranker fails): '+str(global_losses))
    lines.append('Final <=1e-24 existing polish-milestone losses (FIFO passes, ranker does not): '+str(floor_losses))
    if deltas:
        lines.append(f"Paired evaluations B-A among {both_hits} shared handoffs: median {np.median([x[0] for x in deltas]):g}; "
                     f"best {min(deltas)}; worst {max(deltas)}. Positive means ranker is slower.")
    lines.append('Scheduling changes can preserve the observed site set after a queue drains while changing history order;')
    lines.append('that is reported separately from exact trajectory reconvergence because indices and tie bookkeeping retain order.')
    if summary['stop_reason']:
        lines.append('HARD STOP: remaining cases were interrupted or not started. No larger test follows these results.')
        lines.append('UNFINISHED / UNPAIRED ARMS (censored, not scored as recovery failures):')
        for r in rows:
            if r['record_type'] in ('completed_arm_pair_interrupted','interrupted_partial_arm','not_run_hard_stop','interrupted_before_first_snapshot'):
                lines.append(f"  {r['split']}:{r['trajectory_id']} {r['arm']}: {r['record_type']}; "
                             f"evaluations={r.get('search_evaluations','')}; "
                             f"best loss={r.get('search_best_loss','')}.")
        diagnosis_path=OUT/'frozen_pipeline_stop_diagnosis.json'
        if diagnosis_path.exists():
            diagnosis=json.loads(diagnosis_path.read_text())
            lines.append('FROZEN PIPELINE DIAGNOSIS: '+json.dumps({k:v for k,v in diagnosis.items() if k!='edges'}))
            for edge in diagnosis.get('edges',[]):
                if edge['sign_mismatch']:lines.append('  Misclassified slope: '+json.dumps(edge))
            lines.append('This is an upstream FIFO-arm failure before ranker selection, not evidence of ranker-induced recovery loss.')
    elif all(yes(p['A'],'global_handoff_success')==yes(p['B'],'global_handoff_success') for p in pairs.values()):
        lines.append('Global recovery outcomes match on this small selected screen; this does not establish that FIFO can be discarded.')
    lines.append('Numerical-floor flags use the explicitly reported 1e-29 and existing 1e-28 milestone; raw losses,')
    lines.append('reference-floor losses and parameter errors remain visible rather than treating last-bit differences as exact equality.')
    lines.append('The existing N3 validation reaches up to 1.6633e-26 despite <=4.44e-16 parameter errors; 1e-28 is not a universal reliability gate.')
    lines.append('The pre-comparison reliability gate uses the existing 1e-24 polishing milestone, with finer thresholds reported separately.')
    lines.append('Rollout convention is the existing 10/100/500/1000-period analytic SHO validation from exact prefix-end state.')
    lines.append(f"Protected source/checkpoint hashes unchanged: {summary['protected_hashes_unchanged']}.")
    lines.append('Independent output checks: '+json.dumps(verification))
    (OUT/'fifo_vs_ranker_summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
