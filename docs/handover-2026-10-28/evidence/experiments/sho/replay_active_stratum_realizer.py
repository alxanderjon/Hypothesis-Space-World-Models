"""Compose existing CDF search/learned guard; bind only pool budget/context."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import validate_shift_aware_quota_realizer as cdf
from experiments.sho import validate_geometry_guarded_shift_realizer as original
from experiments.sho import peel_geometry_guarded_shift_realizer as learned
from experiments.sho.progressive_spacefill_generator import boundary_faces, discrepancy_rows, SCALES
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics
from experiments.sho.validate_joint_quota_assignment_teacher import metrics as count_metrics

ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'active_multiscale_stratum_teacher'


def main():
    start=time.monotonic();torch.set_num_threads(1)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protected={str(p):sha(p) for p in list(ROOT.rglob('*.pt'))+list(Path('experiments').rglob('*.py'))}
    result=json.loads((OUT/'results.json').read_text());assert result['status']=='READY_FOR_GUARDED_CDF_REPLAY'
    n=result['context']['N'];saved=np.load(OUT/'realized_initial_state.npz');joint=np.load(OUT/'frontier_trace.npz')
    points,selected,classes=saved['points'],saved['selected'],saved['classes']
    model=learned.MarginThreshold()
    model.load_state_dict(torch.load(ROOT/'learned_geometry_guarded_shift_realizer/geometry_margin_threshold_operator.pt',weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    # Load an isolated copy of the unchanged teacher driver. Its historical N
    # global is bound to this pool's N; no canonical module/source is modified.
    spec=importlib.util.spec_from_file_location('_active_stratum_guard_context',original.__file__)
    bound=importlib.util.module_from_spec(spec);spec.loader.exec_module(bound)
    bound.base=SimpleNamespace(**vars(cdf));bound.base.N=n
    ExistingGuard=bound.GeometryGuard
    class BoundLearnedGuard(ExistingGuard):
        def __init__(self,points,selected,cover,holes,nn_p01):
            super().__init__(points,selected,cover,holes,nn_p01)
            self.threshold=model
        one=learned.LearnedGuard.one
        attempt=learned.LearnedGuard.attempt
    bound.GeometryGuard=BoundLearnedGuard
    # N=1000 historical smoke has no NN-p01 acceptance threshold. Zero leaves
    # that nonnegative metric unrestricted; N=8600 later-cohort gates differ.
    limits=dict(result['existing_guard_limits'],nn_p01=0.)
    chosen,stats=bound.realize(points,classes,selected,limits,OUT/'cdf_exchange_trace.jsonl.gz')
    pool=points[chosen];boundary=boundary_faces(n,points.shape[1]);complete=np.unique(np.vstack((pool,boundary)),axis=0)
    geom=metrics(complete,pool,original.reference_points(),n);multi=discrepancy_rows(pool)
    q=np.bincount(classes[chosen],minlength=len(saved['plan']));counts=joint['incidence']@q
    assert np.array_equal(q,saved['plan']) and np.all((counts>=joint['lower'])&(counts<=joint['upper']))
    np.savez_compressed(OUT/'realized_cdf_pool.npz',selected=chosen,pool=pool)
    rows=list(csv.DictReader((ROOT/'progressive_spacefill_smoke/progressive_spacefill_multiscale.csv').open()))
    baseline={name:{k:max(float(r['grid_anchored_discrepancy']) for r in rows if int(float(r['N']))==n and r['generator']==name and int(float(r['bins_per_axis']))==k) for k in SCALES} for name in ('Sobol','old_spacefill','fresh_v1')}
    actual={k:max(r['grid_anchored_discrepancy'] for r in multi if r['bins_per_axis']==k) for k in SCALES}
    ratios={name:max(v[k]/max(baseline['Sobol'][k],1/n) for k in SCALES) for name,v in baseline.items()}
    ratio=max(actual[k]/max(baseline['Sobol'][k],1/n) for k in SCALES)
    gate=dict(discrepancy_materially_better_possible=all(ratio<=.8*ratios[name] for name in ('old_spacefill','fresh_v1')),
              no_scale_regression_possible=all(actual[k]<=1.1*baseline['fresh_v1'][k] for k in SCALES),
              coverage=geom['complete_cover']<=limits['cover'],holes=geom['large_hole_count']<=limits['holes'],
              boundary_support=np.array_equal(complete[np.any((complete==0)|(complete==1),axis=1)],boundary),
              clustering=geom['clustered_fraction']<=.005)
    passed=all(gate.values())
    # A single condition can disprove a max-over-cohort gate; passing here cannot
    # establish unseen-condition/freshness/union gates.
    summary=dict(status='SINGLE_CONDITION_PASSED_PENDING_COHORT' if passed else 'STOPPED_DOWNSTREAM_REALIZATION_HISTORICAL_GATE_FAILURE',
                 gate_scope='single-condition lower bound on historical cohort maxima; not full-cohort acceptance',
                 checks=gate,baseline_scale_maxima=baseline,actual_scale_maxima=actual,
                 worst_ratio=ratio,baseline_worst_ratios=ratios,final_geometry=geom,multiscale=multi,
                 quota_errors=0,budget_errors=0,regional_errors=0,
                 minimum_reservoir_separation=float(cKDTree(points).query(points,k=2)[0][:,1].min()),
                 stats=stats,limits=limits,budget_context_binding='isolated unchanged driver N=1000; no source/canonical-module changes',
                 boundary_cohort_gate='not evaluated: lower per-pool boundary-cover value can improve without changing boundary support',
                 protected_files=len(protected),protected_changes=[p for p,h in protected.items() if sha(p)!=h],
                 runtime_seconds=time.monotonic()-start)
    assert not summary['protected_changes']
    (OUT/'realization_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('multiscale','baseline_scale_maxima')},indent=2),flush=True)


if __name__=='__main__':main()
