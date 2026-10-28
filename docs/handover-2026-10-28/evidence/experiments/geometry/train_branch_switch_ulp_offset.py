"""One small discrete-offset model; frozen availability and arithmetic unchanged."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from experiments.geometry.audit_brent_branch_offsets import read, saved_geometries, bits
from experiments.geometry.train_branch_switch_location import json_write
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=ROOT/'branch_switch_ulp_offset_training'
SEED=928202604


def feature_row(parent,axis,count,left,right,b):
    a=Audit.__new__(Audit);a.setup(parent,axis)
    # Separate bounded mantissa/exponent channels retain absolute float scale
    # without unbounded inputs. No target-derived features are used.
    feats=[]
    for value in (a.a,a.B,a.h,a.omin,a.omax):
        # In d=1 there are no other sides: omin=inf is a structural sentinel,
        # not a continuous magnitude. Dimension identifies that absent field.
        if not np.isfinite(value):value=0.
        mant,exp=np.frexp(value);feats.extend((float(mant),float(exp)/1024))
    n=a.n;f=a.formula(np.array([left,right]),count)
    q=f[6];aspect=f[7];p=f[5][np.arange(2),f[4]]
    wm,we=np.frexp(right-left)
    feats.extend((a.d/5,count/n,(n-count)/n,np.log2(n)/16,
                  left,right,b,(b-left)/(right-left),float(wm),float(we)/1024))
    feats.extend((q/q.max()).ravel().tolist())
    feats.extend((aspect/(1+abs(aspect).max())).ravel().tolist())
    feats.extend(f[3].tolist());feats.extend(f[4].tolist());feats.extend(p.tolist())
    return feats


class OffsetModel(nn.Module):
    def __init__(self,inputs,classes):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inputs,16,dtype=torch.float64),nn.Tanh(),
                               nn.Linear(16,classes,dtype=torch.float64))
    def forward(self,x):return self.net(x)


def shift(b,k):
    return np.array(bits(b)+int(k),dtype=np.uint64).view(np.float64).item()


def metric(rows,pred,ids,name):
    ids=np.asarray(ids,dtype=int);kk=np.array([rows[i]['offset'] for i in ids]);pp=pred[ids];e=abs(pp-kk)
    return dict(group=name,cases=len(ids),exact_k=int((e==0).sum()),exact_coordinate=int((e==0).sum()),
                exact_accuracy=float((e==0).mean()) if len(ids) else None,
                within_one_ULP=int((e<=1).sum()),exactly_one_ULP=int((e==1).sum()),larger_errors=int((e>1).sum()),
                maximum_offset_error=int(e.max()) if len(ids) else 0)


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
    support=json.loads((OUT/'support_screen_result.json').read_text())
    assert support['status']=='SUPPORT_SCREEN_PASSED'
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    raw=read(ROOT/'brent_branch_switch_offset_audit/brent_offsets_unique.csv')
    saved=json.loads((ROOT/'branch_switch_location_residual_training/location_cases.json').read_text())
    geometry=saved_geometries();rows=[]
    for r in raw:
        i=int(r['index']);parent=saved[i]['parent'] if i<550 else geometry[r['call_label']]
        rows.append(dict(parent=parent,axis=int(r['axis']),count=int(r['count_left']),left=float(r['scan_left']),right=float(r['scan_right']),
                         b=float(r['projected_boundary']),teacher_cut=float(r['teacher_cut']),offset=int(r['signed_k']),
                         uid=int(r['unique_id']),source=r['pool'],label=r['label'],switch_type=r['switch_type']))
    known=len(rows)
    for r in json.loads((OUT/'fresh_candidate_states.json').read_text()):
        rows.append(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],
                         b=r['projected_boundary'],teacher_cut=r['teacher_cut'],offset=r['signed_k'],uid=None,
                         source='new_fresh_holdout',label=f"new:{r['batch']}:{r['case']}:{r['side']}",switch_type=r['switch_type']))
    x=np.array([feature_row(r['parent'],r['axis'],r['count'],r['left'],r['right'],r['b']) for r in rows])
    assert np.isfinite(x).all()
    classes=np.array(support['supported_offsets'],dtype=np.int64)
    y=np.array([int(np.flatnonzero(classes==r['offset'])[0]) for r in rows])
    keys=[json.dumps(r['parent'],sort_keys=True) for r in rows]
    assert not set(keys[:known])&set(keys[known:])
    # Parent-disjoint archive validation; preserve every known class in fitting.
    rng=np.random.default_rng(SEED);parents=sorted(set(keys[:known]));rng.shuffle(parents)
    held=set();fit=np.ones(known,dtype=bool)
    for key in parents:
        proposed=fit & np.array([v!=key for v in keys[:known]])
        if len(set(y[:known][proposed]))==len(classes):
            fit=proposed;held.add(key)
        if (~fit).sum()>=.2*known:break
    fitids=np.flatnonzero(fit);validids=np.flatnonzero(~fit)
    assert len(validids)>0 and set(y[fitids])==set(range(len(classes)))
    plan=dict(seed=SEED,architecture=f'float64 Linear({x.shape[1]},16), Tanh, Linear(16,{len(classes)}), categorical offset output',
              feature_description='frexp mantissa/exponent for a,B,h,omin,omax; normalized dimension/counts/log budget; exact scan coordinates/b; normalized scan position/width; normalized competing q/log-aspect values and identities',
              outputs=classes.tolist(),fit_cases=len(fitids),validation_cases=len(validids),fresh_cases=len(rows)-known,
              maximum_updates=2000,patience_updates=300,check_every=25,learning_rate=.01,
              training_objective='unweighted categorical cross entropy',no_model_enlargement=True,
              split='parent-disjoint known validation; independent new parents excluded from fitting and checkpoint selection')
    json_write(OUT/'training_plan.json',plan)
    model=OffsetModel(x.shape[1],len(classes));initial={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.01);xt=torch.from_numpy(x);yt=torch.from_numpy(y)
    best=None;bestloss=float('inf');beststep=0;history=[]
    for step in range(1,2001):
        opt.zero_grad();loss=nn.functional.cross_entropy(model(xt[fitids]),yt[fitids]);loss.backward();opt.step()
        if step%25:continue
        with torch.no_grad():vl=float(nn.functional.cross_entropy(model(xt[validids]),yt[validids]))
        history.append(dict(update=step,fit_cross_entropy=float(loss.detach()),validation_cross_entropy=vl))
        if vl<bestloss-1e-7:
            bestloss=vl;beststep=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step-beststep>=300:break
    model.load_state_dict(best);model.eval().requires_grad_(False)
    cp=OUT/'ulp_offset_diagnostic.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state=initial,plan=plan,
                                  fit_indices=fitids,validation_indices=validids,best_update=beststep,stop_update=step))
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict'])
    with torch.no_grad():pred=classes[model(xt).argmax(1).numpy()]
    # Persist before evaluation: predictions and full static states are audit artifacts.
    metrics=[metric(rows,pred,np.arange(known),'all_known_774'),metric(rows,pred,fitids,'archive_fit'),
             metric(rows,pred,validids,'heldout_known_geometries'),metric(rows,pred,np.arange(known,len(rows)),'new_fresh_holdout')]
    for typ in ('q_only','aspect_only','simultaneous'):
        for group,inds in [('known',range(known)),('new_fresh',range(known,len(rows)))]:
            metrics.append(metric(rows,pred,[i for i in inds if rows[i]['switch_type']==typ],group+':'+typ))
    a=Audit.__new__(Audit);predictions=[]
    for i,r in enumerate(rows):
        c=shift(r['b'],pred[i]);a.setup(r['parent'],r['axis'])
        ss=a.formula(np.array([r['teacher_cut'],c]),r['count'])[0];tol=1e-12*max(1.,ss[0])
        predictions.append(dict(uid=r['uid'],label=r['label'],source=r['source'],switch_type=r['switch_type'],
             teacher_k=r['offset'],predicted_k=int(pred[i]),offset_error=int(abs(pred[i]-r['offset'])),
             teacher_cut=r['teacher_cut'],predicted_cut=c,exact_coordinate=c==r['teacher_cut'],
             outside_pair=not r['left']<=c<=r['right'],teacher_score=ss[0],predicted_teacher_score=ss[1],
             outside_score_equivalence=abs(ss[1]-ss[0])>tol))
    conflicts=read(ROOT/'brent_branch_switch_offset_audit/exact_feature_conflicts.csv');groups=[]
    for group in range(21):
        ids=sorted({int(r['unique_id']) for r in conflicts if r['feature_set']=='local_plus_spacing_exponents' and int(r['conflict_group'])==group})
        groups.append(dict(original_group=group,**metric(rows,pred,ids,str(group))))
    write_csv(OUT/'offset_metrics.csv',metrics);write_csv(OUT/'offset_predictions.csv',predictions)
    write_csv(OUT/'conflicting_group_replay.csv',groups);write_csv(OUT/'training_history.csv',history)
    json_write(OUT/'training_states.json',rows)
    result=dict(status='QUICK_SCREEN_FAILED' if any(r['offset_error'] or r['outside_pair'] for r in predictions) else 'QUICK_SCREEN_PASSED',
                checkpoint=str(cp),sha256=digest,parameters=sum(p.numel() for p in model.parameters()),
                best_update=beststep,stop_update=step,metrics=metrics,
                conflicting_groups_all_correct=sum(r['exact_k']==r['cases'] for r in groups),
                conflicting_cases_correct=sum(r['exact_k'] for r in groups),conflicting_cases=117,
                recursive_k_minus6=predictions[765],outside_pair=sum(r['outside_pair'] for r in predictions),
                outside_score_equivalence=sum(r['outside_score_equivalence'] for r in predictions),
                fresh_geometries_overlap_known=0,elapsed_seconds=time.monotonic()-start)
    assert all(sha(Path(p))==v for p,v in protected.items())
    json_write(OUT/'training_result.json',result)
    print((OUT/'training_result.json').read_text(),flush=True)


if __name__=='__main__':run()
