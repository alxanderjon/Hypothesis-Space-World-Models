"""First production-budget screen, stopping at the first failed prerequisite."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments.sho.active_multiscale_stratum_teacher import read_tree, select_frontier
from experiments.sho.train_joint_quota_assignment_operator import CanonicalPriority
from experiments.sho.train_dyadic_spacefill_curriculum import PHASES
from experiments.sho.validate_geometry_preserving_quota_realizer import realize
from experiments.sho.progressive_spacefill_generator import boundary_faces, discrepancy_rows
from experiments.sho.evaluate_progressive_spacefill_smoke import metrics
from experiments.sho.validate_geometry_guarded_shift_realizer import reference_points

ROOT=Path('outputs/sho/stage1B');SOURCE=ROOT/'geometry_to_reservoir_interface'
OUT=ROOT/'active_multiscale_stratum_teacher'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    paths=list(ROOT.rglob('*.pt'))+list(Path('experiments').rglob('*.py'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protected={str(p):sha(p) for p in paths}
    priority=CanonicalPriority()
    priority.load_state_dict(torch.load(ROOT/'joint_quota_assignment_operator/joint_quota_assignment_operator.pt',weights_only=False)['state_dict'])
    priority.eval().requires_grad_(False)
    nodes=read_tree(json.loads((SOURCE/'N1000_splits.json').read_text()),json.loads((SOURCE/'N1000_leaves.json').read_text()))
    saved=np.load(SOURCE/'N1000_iteration100_cell_state.npz');points=saved['points']
    families=[(i,tuple(f)) for i,(_,_,fs) in enumerate(PHASES) for f in fs]
    context=dict(N=1000,dimension=2,iteration=100,pool='cell',source_state_sha256=sha(SOURCE/'N1000_iteration100_cell_state.npz'))
    state,plan,active,levels,trace=select_frontier(nodes,points,families,priority,context,OUT/'frontier_trace.jsonl')
    print('frontier',len(active),'levels',[(v['tier'],v['active_regions']) for v in levels],flush=True)
    counts={kind:sum(e['decision']==kind for e in trace) for kind in sorted({e['decision'] for e in trace})}
    replay=select_frontier(nodes,points,families,priority,context,OUT/'frontier_replay.jsonl')
    assert active==replay[2] and np.array_equal(plan,replay[1]) and trace==replay[4]
    assert (OUT/'frontier_trace.jsonl').read_bytes()==(OUT/'frontier_replay.jsonl').read_bytes().replace(b'frontier_replay.npz',b'frontier_trace.npz')
    boundary=boundary_faces(1000,2)
    selected=realize(points,state.class_ids,plan,boundary,OUT/'farthest_trace.jsonl.gz')
    pool=points[selected];complete=np.unique(np.vstack((pool,boundary)),axis=0)
    reference=reference_points();geom=metrics(complete,pool,reference,1000)
    assert np.array_equal(np.bincount(state.class_ids[selected],minlength=len(plan)),plan)
    assert len(pool)==1000 and len(np.unique(pool,axis=0))==1000 and np.isfinite(pool).all()
    np.savez_compressed(OUT/'realized_initial_state.npz',points=points,selected=selected,plan=plan,classes=state.class_ids,pool=pool)
    historical=list(csv.DictReader((ROOT/'progressive_spacefill_smoke/progressive_spacefill_geometry.csv').open()))
    sb=[r for r in historical if int(float(r['N']))==1000 and r['generator']=='Sobol']
    pair=next(r for r in sb if int(float(r['iteration']))==100 and r['pool_type']=='cell')
    criteria=json.loads((ROOT/'progressive_spacefill_smoke/evaluation_criteria.json').read_text())
    cover=min(criteria['max_complete_cover_ratio_to_Sobol']*max(float(r['complete_cover']) for r in sb),criteria['max_paired_complete_cover_ratio_to_Sobol']*float(pair['complete_cover']))
    holes=max(float(r['large_hole_count']) for r in sb)+criteria['max_large_hole_fraction_excess']*len(reference)
    checks=dict(coverage=geom['complete_cover']<=cover,holes=geom['large_hole_count']<=holes,
                boundary=geom['boundary_cover']==float(pair['boundary_cover']),
                clustering=geom['clustered_fraction']<=criteria['max_clustered_fraction'])
    # NN-CV is a historical cohort mean gate, not a new per-pool stopping gate.
    result=dict(status='READY_FOR_GUARDED_CDF_REPLAY' if all(checks.values()) else 'STOPPED_DOWNSTREAM_REALIZATION_INITIAL_GEOMETRY_FAILURE',
                context=context,active_regions=len(active),levels=levels,decisions=counts,
                final_active=active,targets=[nodes[p]['n'] for p in active],
                regional_checks=state.verify(plan),joint_feasible=True,deterministic_replay=True,
                final_leaf_constraints=0,geometry=geom,existing_guard_limits=dict(cover=cover,holes=holes),
                initial_geometry_checks=checks,initial_discrepancy=discrepancy_rows(pool),
                protected_files=len(protected),protected_changes=[p for p,h in protected.items() if sha(p)!=h],
                runtime_seconds=time.monotonic()-started,trained=False,downstream_n1_run=False,
                remaining_budgets=[8600,32768],fresh_and_higher_dimensional_generator_gates='not reached')
    assert not result['protected_changes']
    dump(OUT/'results.json',result);dump(OUT/'protected_hashes.json',protected)
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_discrepancy','levels','final_active','targets')},indent=2),flush=True)


if __name__=='__main__':main()
