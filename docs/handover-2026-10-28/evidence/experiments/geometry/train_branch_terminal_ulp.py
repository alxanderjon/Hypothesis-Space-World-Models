"""One small terminal-offset screen, using saved frozen-locator final states."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from experiments.geometry.train_exponent_log2_operator import ROOT,sha,write_csv
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.audit_brent_branch_offsets import read,bits
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'branch_switch_terminal_ulp_training'
SOURCE=ROOT/'terminal_ulp_correction_state_audit'
SEED=928202605
CLASSES=np.arange(-7,10,dtype=np.int64)


def features(rows):
    out=[]
    for r in rows:
        width=r['scan_right']-r['scan_left']
        x=[r['L'],r['R'],r['scan_left'],r['scan_right'],
           (r['L']-r['scan_left'])/width,np.log2(width)/64,*r['normalized_margins']]
        for v in r['geometry']:
            # d=1 has no other side: omin=inf is a structural sentinel.
            m,e=np.frexp(v if np.isfinite(v) else 0.)
            x.extend((float(m),float(e)/1024))
        n=sum(r['counts']);x.extend((r['dimension']/5,r['counts'][0]/n,r['counts'][1]/n,np.log2(n)/16))
        x.extend(float(r['switch_type']==typ) for typ in ('q_only','aspect_only','simultaneous'))
        terms=np.array(r['learned_terms']);scale=abs(terms).max();assert scale>0
        x.extend((terms/scale).tolist());x.append(np.log2(scale)/1024)
        out.append(x)
    return np.array(out,dtype=np.float64)


def geometry_key(parent):
    side=np.array(parent['upper'])-parent['lower'];scale=float(side.max())
    # Keep translation-equivalent and one-ULP-perturbed copies in the same
    # group. This grouping affects the split only, not inference features.
    return (parent['d'],parent['n'],format(scale,'.12g'),tuple(format(x/scale,'.12g') for x in side))


class TerminalULP(nn.Module):
    def __init__(self,inputs=30):
        super().__init__();self.net=nn.Sequential(nn.Linear(inputs,16,dtype=torch.float64),nn.Tanh(),nn.Linear(16,17,dtype=torch.float64))
    def forward(self,x):return self.net(x)


def shift(L,k):return np.array(bits(L)+int(k),dtype=np.uint64).view(np.float64).item()


def metric(rows,pred,ids,name):
    ids=np.asarray(ids,dtype=int);err=abs(pred[ids]-np.array([rows[i]['k'] for i in ids],dtype=int))
    return dict(group=name,cases=len(ids),exact=int((err==0).sum()),within_one_ULP=int((err<=1).sum()),
                exact_accuracy=float((err==0).mean()) if len(ids) else None,
                exactly_one_ULP=int((err==1).sum()),larger_errors=int((err>1).sum()),
                maximum_offset_error=int(err.max()) if len(ids) else 0)


def train():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*.json') if any(s in p.name for s in ('manifest','locked','validated'))})
    rows=json.loads((SOURCE/'terminal_state_cases.json').read_text())
    states=json.loads((ROOT/'branch_switch_locator_assembly/case_states.json').read_text())
    for p in (SOURCE/'terminal_state_cases.json',ROOT/'branch_switch_locator_assembly/case_states.json'):protected[str(p)]=sha(p)
    json_write(OUT/'protected_hashes.json',protected)
    assert len(rows)==818 and len(states)==818
    x=features(rows);assert x.shape==(818,30) and np.isfinite(x).all()
    y=np.array([r['k']+7 for r in rows]);assert y.min()==0 and y.max()==16
    keys=[geometry_key(s['parent']) for s in states]
    known_groups=sorted(set(keys[:774]));rng=np.random.default_rng(SEED)
    order=rng.permutation(len(known_groups));held=set(known_groups[i] for i in order[:int(np.ceil(.2*len(known_groups)))] )
    # The requested recursive fixture is an explicit unseen-geometry check.
    held.add(keys[765])
    fit=np.array([i for i in range(774) if keys[i] not in held],dtype=int)
    valid=np.array([i for i in range(774) if keys[i] in held],dtype=int)
    assert not set(keys[i] for i in fit)&set(keys[i] for i in valid)
    assert not set(keys[:774])&set(keys[774:])
    split=[dict(case_id=i,label=r['label'],geometry_group=str(keys[i]),
                role='fresh_holdout' if i>=774 else 'validation_geometry' if keys[i] in held else 'fit_geometry',k=r['k']) for i,r in enumerate(rows)]
    write_csv(OUT/'geometry_split.csv',split)
    plan=dict(seed=SEED,architecture='float64 Linear(30,16) -> Tanh -> Linear(16,17); argmax categorical k',
        classes=CLASSES.tolist(),fit_cases=len(fit),heldout_geometry_cases=len(valid),fresh_cases=44,
        fit_geometries=len(set(keys[i] for i in fit)),heldout_geometries=len(set(keys[i] for i in valid)),
        split='Whole dimension/budget/side-length geometry groups; 12-significant-digit key groups one-ULP copies together; translation ignored. Fixture geometry held out. Fresh 44 excluded from fitting and checkpoint selection.',
        input_features=['L','R','scan_left','scan_right','relative_L_in_scan','log2_scan_width/64','normalized_margin_L','normalized_margin_R',
                        'frexp mantissa/exponent of a,B,h,omin,omax (10 channels)','dimension/5','left_count/n','right_count/n','log2(n)/16',
                        'switch_type onehot (3 channels)','four learned terms normalized by common max','log2 common term scale/1024'],
        training_objective='unweighted categorical cross entropy',learning_rate=.01,maximum_updates=2000,
        validation_every=25,patience_updates=300,random_initialization=True,locator_replayed=False,
        training_class_counts={int(k):int(sum(rows[i]['k']==k for i in fit)) for k in CLASSES},
        no_automatic_model_enlargement=True)
    json_write(OUT/'training_plan.json',plan)
    model=TerminalULP();initial={k:v.detach().clone() for k,v in model.state_dict().items()};opt=torch.optim.Adam(model.parameters(),lr=.01)
    xt=torch.from_numpy(x);yt=torch.from_numpy(y);best=None;bestvalue=float('inf');beststep=0;history=[]
    for step in range(1,2001):
        opt.zero_grad();loss=nn.functional.cross_entropy(model(xt[fit]),yt[fit]);loss.backward();opt.step()
        if step%25:continue
        with torch.no_grad():value=float(nn.functional.cross_entropy(model(xt[valid]),yt[valid]))
        history.append(dict(update=step,fit_cross_entropy=float(loss.detach()),heldout_geometry_cross_entropy=value))
        if value<bestvalue-1e-7:bestvalue=value;beststep=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step-beststep>=300:break
    model.load_state_dict(best);model.eval().requires_grad_(False)
    cp=OUT/'terminal_ulp_diagnostic.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state=initial,plan=plan,
                                  fit_indices=fit,validation_indices=valid,best_update=beststep,stop_update=step))
    # Persist and reload before evaluating the held-out fresh cases or replay.
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict'])
    with torch.no_grad():pred=CLASSES[model(xt).argmax(1).numpy()]
    metrics=[metric(rows,pred,np.arange(818),'all_818'),metric(rows,pred,np.arange(774),'all_saved_774'),
             metric(rows,pred,fit,'fit_geometries'),metric(rows,pred,valid,'heldout_geometries'),
             metric(rows,pred,np.arange(774,818),'fresh_44')]
    for typ in ('q_only','aspect_only','simultaneous'):
        for name,ids in [('all',np.arange(818)),('heldout',valid),('fresh',np.arange(774,818))]:
            metrics.append(metric(rows,pred,[i for i in ids if rows[i]['switch_type']==typ],name+':'+typ))
    predictions=[];a=Audit.__new__(Audit)
    for i,(r,s) in enumerate(zip(rows,states)):
        c=shift(r['L'],pred[i]);assert bits(c)-bits(r['L'])==pred[i]
        a.setup(s['parent'],s['axis']);score=a.formula(np.array([r['teacher_cut'],c]),s['count'])[0]
        predictions.append(dict(case_id=i,label=r['label'],role=split[i]['role'],switch_type=r['switch_type'],
            teacher_k=r['k'],predicted_k=int(pred[i]),offset_error=int(abs(pred[i]-r['k'])),
            teacher_cut=r['teacher_cut'],candidate=c,exact_coordinate=c==r['teacher_cut'],
            outside_support=not int(pred[i]) in CLASSES,
            outside_scan=not r['scan_left']<=c<=r['scan_right'],teacher_score=score[0],candidate_teacher_score=score[1],
            score_outside_equivalence=bool(abs(score[1]-score[0])>1e-12*max(1.,score[0]))))
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv');groups=[]
    for g in range(21):
        ids=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==g})
        groups.append(metric(rows,pred,ids,str(g)))
    write_csv(OUT/'offset_metrics.csv',metrics);write_csv(OUT/'predictions.csv',predictions)
    write_csv(OUT/'conflicting_group_replay.csv',groups);write_csv(OUT/'training_history.csv',history)
    json_write(OUT/'learned_parameters.json',{k:v.tolist() for k,v in model.state_dict().items()})
    assert all(sha(Path(p))==v for p,v in protected.items())
    passed=all(g['exact']==g['cases'] for g in groups) and predictions[765]['offset_error']==0 and all(predictions[i]['offset_error']==0 for i in list(valid)+list(range(774,818))) and not any(p['outside_scan'] or p['score_outside_equivalence'] for p in predictions)
    result=dict(status='QUICK_SCREEN_PASSED_DOWNSTREAM_PENDING' if passed else 'QUICK_SCREEN_FAILED',quick_gate_passed=passed,
        checkpoint=str(cp),sha256=digest,architecture=plan['architecture'],parameters=sum(p.numel() for p in model.parameters()),
        best_update=beststep,stop_update=step,metrics=metrics,conflicting_group_results=groups,
        correct_conflicting_groups=sum(g['exact']==g['cases'] for g in groups),correct_conflicting_cases=sum(g['exact'] for g in groups),
        recursive_fixture=predictions[765],outside_support=sum(p['outside_support'] for p in predictions),
        outside_scan=sum(p['outside_scan'] for p in predictions),score_failures=sum(p['score_outside_equivalence'] for p in predictions),
        parent_geometry_overlap=0,fresh_geometry_overlap=0,protected_hashes_unchanged=True,protected_files=len(protected),elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'training_result.json',result)
    print((OUT/'training_result.json').read_text(),flush=True)


def replay():
    from experiments.geometry import train_branch_switch_location as harness
    torch.set_num_threads(1);result=json.loads((OUT/'training_result.json').read_text())
    state=json.loads((SOURCE/'terminal_state_cases.json').read_text())
    saved=json.loads((ROOT/'branch_switch_location_residual_training/location_cases.json').read_text())
    fresh=json.loads((ROOT/'branch_switch_ulp_offset_training/fresh_candidate_states.json').read_text())
    raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_all.csv')
    mapped=[state[int(raw[i]['unique_id'])] for i in range(550)]+state[774:];rows=saved+fresh
    xx=features(mapped);left=np.array([r['c_left'] for r in rows]);right=np.array([r['c_right'] for r in rows])
    class Adapter(TerminalULP):
        def locate(self,L,R,unused):
            assert np.array_equal(L,left) and np.array_equal(R,right)
            with torch.no_grad():pred=CLASSES[self(torch.from_numpy(xx)).argmax(1).numpy()]
            return np.array([shift(r['L'],k) for r,k in zip(mapped,pred)])
    model=Adapter();model.load_state_dict(torch.load(result['checkpoint'],weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    cuts=model.locate(left,right,None);boundary=np.clip(np.array([r['boundary'] for r in rows]),left,right)
    folder=OUT/'selector_replay';folder.mkdir(exist_ok=False)
    json_write(folder/'location_cases.json',rows);json_write(folder/'protected_hashes.json',json.loads((OUT/'protected_hashes.json').read_text()))
    mm=harness.location_metrics(rows,cuts,'terminal_ulp');bm=harness.location_metrics(rows,boundary,'boundary')
    json_write(folder/'quick_result.json',dict(checkpoint=result['checkpoint'],sha256=result['sha256'],updates=result['stop_update'],
                train_cases=next(m['cases'] for m in result['metrics'] if m['group']=='fit_geometries'),metrics=[mm,bm,bm,bm]))
    old_out,old_class=harness.OUT,harness.BranchLocation
    try:harness.OUT=folder;harness.BranchLocation=Adapter;harness.diagnostic_replay()
    finally:harness.OUT=old_out;harness.BranchLocation=old_class
    rr=json.loads((folder/'final_result.json').read_text())
    keep=('available_branch_cases','availability_mismatches','branch_candidate_identity_changes','all_branch_subpiece_winner_changes',
          'all_branch_subpiece_outside_equivalence','cached_branch_replacements','cached_branch_insertions','downstream','fixture',
          'all_protected_hashes_unchanged','protected_files','replay_seconds')
    downstream={k:rr[k] for k in keep}
    downstream['scope']='594 branch-positive subpieces including all 44 fresh; complete cached 106 archived + 8 existing fresh parents; original d2 N8 fixture. No locator rerun or recursion.'
    json_write(folder/'final_result.json',downstream)
    passed=result['quick_gate_passed'] and rr['all_branch_subpiece_winner_changes']==0 and rr['all_branch_subpiece_outside_equivalence']==0 and all(v['identity_changes']==0 and v['outside_equivalence']==0 for v in rr['downstream'].values()) and rr['downstream']['parent']['coordinate_changes']==0 and all(not r['identity_changed'] and not r['outside_equivalence'] for r in rr['fixture'])
    result.update(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',promoted=passed,downstream=downstream,
                  stop_reason=None if passed else 'Quick exact-offset/scan-bracket and/or downstream gate failed. No model enlargement or additional training.',
                  checkpoint_hash_unchanged=sha(Path(result['checkpoint']))==result['sha256'])
    json_write(OUT/'final_result.json',result)
    if passed:json_write(OUT/'branch_switch_terminal_ulp_candidate.json',result)
    print(json.dumps(dict(status=result['status'],downstream=rr['downstream'])),flush=True)


if __name__=='__main__':
    import sys
    replay() if '--replay' in sys.argv else train()
