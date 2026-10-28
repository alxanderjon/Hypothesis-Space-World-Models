"""Synthetic box geometry only: dimension-general supervised mass allocation.

No point placement, search, Sobol, physical parameters, or objective labels.
The structured network sees side lengths, not volumes or target fractions.
"""
import copy
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.smoke_artifacts import save_checkpoint


OUT = Path('outputs/sho/stage1B/geometry_allocation_curriculum')
SEED = 27162026
RESERVED = ((.13, .145), (.855, .87), (.29, .31), (.69, .71))
TRAIN_BUDGETS = (13, 128, 1000, 5000, 20000)
TEST_BUDGETS = (1, 7, 64, 997, 8600, 32768, 1000000)
FEATURES = ('log_length', 'length_minus_one', 'length_minus_one_squared',
            'log_length_squared', 'log_length_times_budget', 'log_length_times_budget_squared')
GATES = dict(rmse=1e-7, max_error=1e-6, min_child_relative_error=1e-3,
             conservation=2e-14, permutation=1e-12)


class GeometryAllocator(nn.Module):
    """Shared per-axis six-feature linear layer, axis sum, child softmax.

All six weights are learned from random initialization. No volume products
or fractions are computed in the forward path. Child and axis permutations
commute with inference up to floating-point reduction order.
"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, dtype=torch.float64) * .15)

    def forward(self, features, mask):
        scores = (features @ self.weight).masked_fill(~mask, -torch.inf)
        return torch.softmax(scores, dim=1)


def feature_tensor(records, budget_override=None):
    maximum = max(len(r['target']) for r in records)
    f = np.zeros((len(records), maximum, 6))
    target = np.zeros((len(records), maximum))
    mask = np.zeros((len(records), maximum), dtype=bool)
    for j, r in enumerate(records):
        # Geometry is normalized by parent SIDE LENGTH, never by volume.
        length = (r['upper'] - r['lower']) / (r['parent_upper'] - r['parent_lower'])
        assert np.all(length > 0) and np.all(length <= 1 + 1e-12)
        z = np.log(length)
        b = math.log1p(r['budget'] if budget_override is None else budget_override) / math.log1p(32768)
        per_axis = np.stack((z, length-1, (length-1)**2, z*z, z*b, z*b*b), axis=-1)
        k = len(length)
        f[j, :k] = per_axis.sum(axis=1)
        target[j, :k] = r['target']
        mask[j, :k] = True
    return torch.from_numpy(f), torch.from_numpy(mask), torch.from_numpy(target)


def record(boxes, budget, name, dimension, kind):
    lo = np.asarray([b[0] for b in boxes], dtype=np.float64)
    hi = np.asarray([b[1] for b in boxes], dtype=np.float64)
    target = np.prod(hi-lo, axis=1)
    assert abs(target.sum()-1) < 2e-14
    return dict(lower=lo, upper=hi, parent_lower=np.zeros(dimension),
                parent_upper=np.ones(dimension), target=target, budget=int(budget),
                dimension=dimension, name=name, kind=kind)


def random_partition(rng, dimension, children):
    boxes = [(np.zeros(dimension), np.ones(dimension))]
    for split in range(children-1):
        # Volume weighting only selects synthetic training geometries.
        volume = np.array([np.prod(hi-lo) for lo, hi in boxes])
        idx = rng.choice(len(boxes), p=volume/volume.sum())
        lo, hi = boxes.pop(idx)
        axis = split % dimension if split < dimension else int(rng.integers(dimension))
        q = rng.uniform(.025, .975)
        cut = lo[axis] + q*(hi[axis]-lo[axis])
        ah = hi.copy(); ah[axis] = cut
        bl = lo.copy(); bl[axis] = cut
        boxes.extend(((lo.copy(), ah), (bl, hi.copy())))
    return boxes


def make_data(stage, count, validation=False):
    rng = np.random.default_rng(SEED + 100*stage + int(validation))
    dimension = max(1, stage-1)
    records = []
    while len(records) < count:
        children = 2 if stage == 1 else int(rng.integers(3 if stage == 2 else 4, 9 if stage == 2 else 17))
        boxes = random_partition(rng, dimension, children)
        fractions = np.array([np.prod(b-a) for a, b in boxes])
        reserved = any(np.any((fractions >= a) & (fractions <= b)) for a,b in RESERVED)
        # Entire fraction intervals, not merely seeds, are disjoint.
        if reserved != validation:
            continue
        budgets = TEST_BUDGETS if validation else TRAIN_BUDGETS
        r = record(boxes, rng.choice(budgets), f's{stage}_{"test" if validation else "train"}_{len(records)}',
                   dimension, 'heldout_ratio' if validation else 'train')
        records.append(r)
    return records


def stress_data(stage):
    dimension = max(1, stage-1)
    records = []
    # Ratios never used in training, including extreme extrapolation.
    for q in (.137, .297, 1e-3, 1e-6, 1e-10):
        for axis in range(dimension):
            lo=np.zeros(dimension); hi=np.ones(dimension)
            ah=hi.copy(); ah[axis]=q
            bl=lo.copy(); bl[axis]=q
            records.append(record([(lo,ah),(bl,hi)], 8600, f'd{dimension}_q{q}_axis{axis}', dimension, 'explicit_ratio'))
    if stage >= 2:
        # Cartesian partitions are a held-out geometry family for d >= 2.
        for cuts in ((0,.137,.701,1), (0,.001,.137,.701,1)):
            boxes=[]
            for index in itertools.product(range(len(cuts)-1), repeat=dimension):
                boxes.append((np.array([cuts[i] for i in index]), np.array([cuts[i+1] for i in index])))
            records.append(record(boxes, 32768, f'd{dimension}_cartesian_{len(boxes)}', dimension, 'cartesian_unseen'))
    return records


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def metrics(pred, target, mask):
    e=(pred-target)[mask]
    smallest=target.masked_fill(~mask, torch.inf).argmin(1)
    largest=target.argmax(1)
    batch=torch.arange(len(target))
    sm=(pred-target)[batch,smallest]
    lg=(pred-target)[batch,largest]
    return dict(fraction_rmse=float(e.square().mean().sqrt()), fraction_mae=float(e.abs().mean()),
                fraction_max_error=float(e.abs().max()), conservation_max=float((pred.sum(1)-1).abs().max()),
                smallest_child_mae=float(sm.abs().mean()), smallest_child_max_error=float(sm.abs().max()),
                smallest_child_max_relative_error=float((sm/target[batch,smallest]).abs().max()),
                largest_child_mae=float(lg.abs().mean()), largest_child_max_error=float(lg.abs().max()))


def evaluate(model, records, stage, rows):
    f, mask, target=feature_tensor(records)
    with torch.no_grad():pred=model(f,mask)
    stats=metrics(pred,target,mask)
    rng=np.random.default_rng(SEED+stage)
    perm=np.stack([rng.permutation(f.shape[1]) for _ in records])
    ix=torch.from_numpy(perm)
    with torch.no_grad():pp=model(f.gather(1,ix[:,:,None].expand(-1,-1,6)),mask.gather(1,ix))
    stats['permutation_max_error']=float((pp-pred.gather(1,ix)).abs().max())
    groups=[('all','all',list(range(len(records))))]
    for field in ('dimension','kind'):
        groups += [(field,value,[i for i,r in enumerate(records) if r[field]==value]) for value in sorted({r[field] for r in records})]
    groups += [('children',k,[i for i,r in enumerate(records) if len(r['target'])==k]) for k in sorted({len(r['target']) for r in records})]
    for field,value,idx in groups:
        rows.append(dict(record_type='fraction_group',stage=stage,group=field,value=value,partitions=len(idx),
                         **metrics(pred[idx],target[idx],mask[idx])))
    for i,r in enumerate(records):
        if r['kind'] in ('explicit_ratio','cartesian_unseen'):
            for c in range(len(r['target'])):
                rows.append(dict(record_type='explicit_child',stage=stage,partition=r['name'],dimension=r['dimension'],
                                 child=c,children=len(r['target']),budget=r['budget'],target_fraction=float(target[i,c]),
                                 predicted_fraction=float(pred[i,c]),error=float(pred[i,c]-target[i,c])))
    budget_max=0.; count_worst=0;count_mismatches=0;ties=0;budget_metrics=[]
    for budget in TEST_BUDGETS:
        bf,bm,bt=feature_tensor(records,budget)
        with torch.no_grad():bp=model(bf,bm)
        budget_max=max(budget_max,float((bp-pred).abs().max()))
        errs=[];cons=[];mismatch_parts=0
        for i,r in enumerate(records):
            k=len(r['target']);p=bp[i,:k].numpy();t=bt[i,:k].numpy()
            pc=conserve(budget,p);tc=conserve(budget,t)
            ce=pc-tc;errs.extend(ce.tolist());cons.append(int(pc.sum())-budget)
            if np.any(ce):
                mismatch_parts+=1
                rem=(budget*(t/t.sum()))%1
                changed=np.flatnonzero(ce)
                remainder_gap=float(np.ptp(rem[changed]))
                near_tie=bool(remainder_gap<1e-8)
                ties+=int(near_tie)
                rows.append(dict(record_type='integer_mismatch',stage=stage,partition=r['name'],dimension=r['dimension'],
                                 budget=budget,target_counts=json.dumps(tc.tolist()),predicted_counts=json.dumps(pc.tolist()),
                                 count_max_error=int(abs(ce).max()),near_remainder_tie=near_tie,
                                 changed_child_indices=json.dumps(changed.tolist()),
                                 teacher_changed_child_fractions=json.dumps(t[changed].tolist()),
                                 teacher_remainder_gap_changed_children=remainder_gap))
        errs=np.asarray(errs);count_worst=max(count_worst,int(abs(errs).max()));count_mismatches+=mismatch_parts
        rows.append(dict(record_type='integer_budget',stage=stage,budget=budget,partitions=len(records),
                         count_mae=float(abs(errs).mean()),count_rmse=float(np.sqrt(np.mean(errs.astype(float)**2))),
                         count_max_error=int(abs(errs).max()),mismatched_partitions=mismatch_parts,
                         count_conservation_max=max(abs(v) for v in cons),**metrics(bp,bt,bm)))
        budget_metrics.append(metrics(bp,bt,bm))
    # Each budget has the identical partition/child population, so these
    # averages are exactly the aggregate metrics over all budget tests.
    for name in budget_metrics[0]:
        values=[b[name] for b in budget_metrics]
        stats[name]=(float(np.sqrt(np.mean(np.square(values)))) if name=='fraction_rmse'
                     else float(np.mean(values)) if name.endswith('_mae') else max(values))
    stats.update(budget_invariance_max=budget_max,count_max_error=count_worst,
                 integer_mismatched_partition_budget_pairs=count_mismatches,integer_near_tie_cases=ties)
    return stats


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    OUT.mkdir(parents=True,exist_ok=False)
    # Immutable existing checkpoints are hashed, never loaded into this model.
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path('outputs/sho/stage1B/hierarchical_allocation_poc').glob('*.pt')}
    config=dict(seed=SEED,dtype='float64',features=FEATURES,architecture='shared learned six-feature axis scoring; axis sum; child softmax',
                reserved_fraction_intervals=RESERVED,train_budgets=TRAIN_BUDGETS,test_budgets=TEST_BUDGETS,
                gates=GATES,training_partitions_per_stage=2048,validation_partitions_per_stage=512,
                optimizer='LBFGS lr=1, max_iter=20 per chunk, max 12 chunks, strong_wolfe',
                loss='1e6 * mean squared child fraction error; cumulative previous training stages',
                integer_rule='existing conserve(): largest remainder, child-index tie break; evaluation only',
                geometric_prior='additive shared axis score and logarithmic side-length features; not an unconstrained MLP',
                protected_hashes=protected)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    model=GeometryAllocator();train=[];validation=[];summary=[];generalization=[];history=[];checkpoints=[]
    previous=None
    for stage in range(1,5):
        stage_start=time.monotonic()
        if previous is not None:
            model.load_state_dict(torch.load(previous,weights_only=False)['state_dict'])
        start_weights=model.weight.detach().tolist()
        new_train=make_data(stage,2048);train+=new_train
        validation+=make_data(stage,512,True)+stress_data(stage)
        # Save geometry provenance: no downstream/search data enters the run.
        data_path=OUT/f'stage_{stage}_partitions.json'
        data_path.write_text(json.dumps(dict(training=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in new_train],
                                             heldout=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in validation])))
        f,mask,target=feature_tensor(train)
        vf,vm,vt=feature_tensor(validation)
        with torch.no_grad():before=metrics(model(vf,vm),vt,vm)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=20,max_eval=30,tolerance_grad=1e-13,
                                    tolerance_change=1e-18,history_size=50,line_search_fn='strong_wolfe')
        best=float('inf');best_state=None;closures=0;plateau=0
        for chunk in range(12):
            def closure():
                nonlocal closures
                optimizer.zero_grad();p=model(f,mask);loss=((p-target)[mask]**2).mean()*1e6
                loss.backward();closures+=1;return loss
            optimizer.step(closure)
            with torch.no_grad():p=model(f,mask);m=metrics(p,target,mask);objective=m['fraction_rmse']
            history.append(dict(stage=stage,chunk=chunk+1,closures=closures,**m))
            if objective<best:
                plateau=plateau+1 if best-objective<1e-13 else 0
                best=objective;best_state=copy.deepcopy(model.state_dict())
            else:plateau+=1
            print(json.dumps(dict(event='training',stage=stage,chunk=chunk+1,train_rmse=objective,train_max=m['fraction_max_error'])),flush=True)
            if m['fraction_max_error']<1e-10 or plateau>=3:break
        model.load_state_dict(best_state)
        path=OUT/f'stage_{stage}_best.pt'
        digest=save_checkpoint(path,dict(state_dict=model.state_dict(),stage=stage,config=config,
                                       previous_checkpoint=str(previous) if previous else None,initial_weights=start_weights,
                                       training_rmse=best,closures=closures,status='SAVED_BEFORE_HELDOUT_GATE'))
        # Evaluation uses the persisted weights, not an unsaved in-memory version.
        model.load_state_dict(torch.load(path,weights_only=False)['state_dict'])
        stats=evaluate(model,validation,stage,generalization)
        passed=(stats['fraction_rmse']<=GATES['rmse'] and stats['fraction_max_error']<=GATES['max_error']
                and stats['smallest_child_max_relative_error']<=GATES['min_child_relative_error']
                and stats['conservation_max']<=GATES['conservation'] and stats['permutation_max_error']<=GATES['permutation'])
        row=dict(stage=stage,dimension=max(1,stage-1),passed=passed,training_partitions=len(train),heldout_partitions=len(validation),
                 metric_scope='all heldout partition-child-budget combinations, seven budgets',
                 initial_heldout_rmse=before['fraction_rmse'],initial_heldout_max=before['fraction_max_error'],
                 training_rmse=best,optimizer_closures=closures,seconds=time.monotonic()-stage_start,
                 checkpoint=str(path),sha256=digest,learned_weights=json.dumps(model.weight.detach().tolist()),**stats)
        summary.append(row);checkpoints.append(dict(stage=stage,path=str(path),sha256=digest,passed=passed))
        write_csv(OUT/'allocation_curriculum_summary.csv',summary)
        write_csv(OUT/'allocation_generalization.csv',generalization)
        write_csv(OUT/'training_history.csv',history)
        (OUT/'checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))
        print(json.dumps(dict(event='stage_gate',**row)),flush=True)
        if not passed:break
        previous=path
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    result=dict(all_stages_passed=len(summary)==4 and all(r['passed'] for r in summary),
                elapsed_seconds=time.monotonic()-started,stages_completed=len(summary),checkpoints=checkpoints,
                protected_checkpoints_unchanged=True,final_weights=model.weight.detach().tolist())
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':run()
