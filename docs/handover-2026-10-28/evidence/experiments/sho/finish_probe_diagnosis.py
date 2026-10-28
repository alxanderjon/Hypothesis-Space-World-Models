"""Supplement the static audit with first-score inputs and probe-refresh checks."""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho import run_stage1b_spacefill_generator_smoke as smoke

P=Path('outputs/sho/stage1B/probe_cellsize_accuracy')
rows=list(csv.DictReader((P/'probe_cellsize_accuracy_audit.csv').open()))
runs=torch.load('outputs/sho/stage1B/sobol_vs_spacefill_quick/quick_screen_states.pt',weights_only=False)['runs']
model=smoke.Generator(2);checkpoint=Path('outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt');before=src.sha(checkpoint)
model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
extras=[]
for row in rows:
    if row['record_type']!='initial_divergence':continue
    tid=int(row['trajectory_id']);a=next(r for r in runs if r['case']['trajectory_id']==tid and r['arm']=='A')
    sites,losses=a['sites'][:10],a['losses'][:10];surface=src.edirect.GaussianLocalResponse(sites,losses)
    extra=dict(record_type='first_score_diagnosis',state_id=row['state_id'],trajectory_id=tid)
    for name,arm in [('Sobol','A'),('spacefill','B')]:
        sr=[r for r in rows if r['record_type']=='site' and r['state_id']==row['state_id'] and r['generator']==name]
        sizes=np.asarray([float(r['final_cell_size']) for r in sr]);po=src.edirect._potentially_optimal_cells(sizes,losses)
        w=.5*src.edirect._normalize(sizes[po])+.5*src.edirect._normalize(surface.loo_errors(po));w[np.argmax(sizes[po])]=1
        site=int(row[f'{name}_own_first_cell']);point=np.asarray(json.loads(row[f'{name}_own_first_point']))
        hist=next(r for r in runs if r['case']['trajectory_id']==tid and r['arm']==arm)
        extra[f'{name}_first_weight_u']=float(w[np.flatnonzero(po==site)[0]])
        extra[f'{name}_first_surrogate_prediction']=float(surface.predict(point[None,:])[0])
        extra[f'{name}_archived_first_objective_loss']=hist['events'][10]['loss']
    extras.append(extra)
for budget in (1000,8600,32768):
    count=10 if budget==1000 else 86 if budget==8600 else 334
    a=src.edirect._candidate_pool(count,200136,0);a_next=src.edirect._candidate_pool(count,200136,1)
    a_greedy=src.edirect._candidate_pool(count,200136,1000000)
    b,_=model.generate(budget,[0.,0.],[1.,1.]);b_next,_=model.generate(budget,[0.,0.],[1.,1.])
    inside=lambda x:x[~np.any((x==0.)|(x==1.),axis=1)]
    for name,points,other,greedy in [('Sobol',a,a_next,a_greedy),('spacefill',b,b_next,b_next)]:
        interior=inside(points);other=inside(other);greedy=inside(greedy)
        base=set(map(tuple,interior));share=lambda x:len(base.intersection(map(tuple,x)))/len(base)
        extras.append(dict(record_type='probe_refresh',interior_probe_budget=budget,generator=name,
            exact_interior_overlap_next_iteration=share(other),exact_interior_overlap_greedy_pool=share(greedy),
            equal_points_next_iteration=bool(np.array_equal(interior,other))))
        for bins in (4,16,64,128):
            for shift in (.25,.5,.75):
                shifted=(interior+shift/bins)%1.
                counts=np.histogramdd(shifted,bins=(np.linspace(0,1,bins+1),)*2)[0]
                extras.append(dict(record_type='shifted_multiscale',interior_probe_budget=budget,generator=name,bins_per_axis=bins,
                    shift_fraction_of_bin=shift,count_variance=float(counts.var()),count_fano=float(counts.var()/counts.mean()),
                    empty_bin_fraction=float(np.mean(counts==0)),max_bin_count=int(counts.max())))
rows+=extras;src.write_csv(P/'probe_cellsize_accuracy_audit.csv',rows)
assert src.sha(checkpoint)==before
lines=['','DIAGNOSIS',
'The hypothesis that generally worse per-cell radius estimates explain spacefill underperformance is NOT supported by this sample.',
'Spacefill pooled MAE is about 1.8% lower and is lower in 17/30 states, but this small difference is below the mean half-to-full reference radius increment; do not overinterpret that ranking.',
'Boundary-cell and sparse-cell MAE are slightly worse for spacefill; reference-hull/PO errors are slightly better. Both show similar grouping/hull/PO disruptions.',
'Finite-reference caution: 201 cell-state instances have no interior reference hits, and reference PO membership remains unstable in 17/30 states after doubling the interior sample. These are approximate reference comparisons, not exact PO truth.',
'All six initial states have identical Sobol/spacefill/reference PO sets. Both choose the SAME first ordinary cell, and each original first point is reproduced exactly from the archive.',
'Replacing BOTH generators cell sizes and PO sets by the SAME reference leaves both first chosen points unchanged in all six cases. Therefore the observed FIRST divergence is candidate-location/scoring driven, not cell-size estimation driven.',
'Static shared-reference interventions do not prove the cause of later recovery differences; later histories, greedy proposals, and candidate-set score normalization can all interact.',
'The learned generator has no iteration/seed input. Once N saturates, it repeats an identical interior set every iteration and also for the greedy pool. Production Sobol refreshes its interior using iteration and greedy seed offsets.',
'This loss of new candidate locations is a concrete structural difference and a plausible contributor to stalled exploitation; no causal closed-loop ablation was run.',
'Global covering radius improves, but average nearest-neighbor CV is 0.0199 for spacefill versus 0.2802 for Sobol: the learned point set is highly regular.',
'At grid resolution 128, mean anchored discrepancy is 0.00650 versus 0.000750 (about 8.7x worse); at 64 it is 0.00468 versus 0.000643 (about 7.3x worse).',
'Origin-aligned dyadic local count variance is often identical, so that variance alone hides the difference. Shifted-bin diagnostics are also saved; these shifts are measurements, not changes to search probes.',
'No replacement architecture proposed, no training, no objective evaluations, and no search continuation. Candidate/checkpoints unchanged.',
'','FIRST-SCORE INPUTS']+[json.dumps(r) for r in extras if r['record_type']=='first_score_diagnosis']
lines+=['','PROBE REFRESH']+[json.dumps(r) for r in extras if r['record_type']=='probe_refresh']
with (P/'probe_generator_diagnosis.txt').open('a') as f:f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
