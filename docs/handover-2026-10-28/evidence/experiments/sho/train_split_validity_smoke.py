"""Learn only integer-count validity; frozen learned min and complete finite score."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares

from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_characteristic_spacing_smoke import sha, write_csv
from experiments.sho.train_aspect_penalty_smoke import side_reduce
from experiments.sho.train_finite_split_score_smoke import load_frozen, upstream, FiniteSplitScore, predict
from experiments.sho.train_radius_quotient_smoke import rank_audit
from experiments.sho.audit_split_geometry_score import reconstruct

ROOT=Path('outputs/sho/stage1B'); OUT=ROOT/'split_validity_smoke'; SEED=27252026


class ValidityMargin(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(.2*torch.randn(2,dtype=torch.float64))

    def forward(self,n):
        return self.weights[0]*n+self.weights[1]


class ValidityThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_alpha=nn.Parameter(torch.randn((),dtype=torch.float64))
        self.theta=nn.Parameter(torch.randn((),dtype=torch.float64))

    def alpha(self):return nn.functional.softplus(self.raw_alpha)

    def forward(self,m):return self.alpha()*(m-self.theta)


def route(execution,margin,threshold,counts):
    n,_=side_reduce(execution,np.asarray(counts,dtype=np.float64),False)
    with torch.no_grad():
        m=margin(torch.from_numpy(n)); logits=threshold(m)
    return n,m.numpy(),logits.numpy(),logits.numpy()>=0


def synthetic(training):
    rng=np.random.default_rng(SEED+(1 if training else 2)); result=[]
    for k in (1,2,3,4,8,16,64):
        for i in range(256):
            c=rng.integers(1,65 if training else 1000001,k,dtype=np.int64)
            if i%2==0:
                z=min(k,1+(i//2)%k); c[rng.choice(k,z,replace=False)]=0
            else:
                minimum=int(rng.choice([1,2,3,7,16,31] if training else [1,2,5,17,127,997,32768]))
                c+=minimum; c[rng.integers(k)]=minimum
                if i%3==0:c[:max(1,k//2)]=minimum
            result.append(dict(source='synthetic_train' if training else 'synthetic_heldout',case=len(result),counts=c.tolist()))
    if not training:
        for n in (0,1,2,3,2**20,2**32,2**52,2**53-2):
            for k in (1,2,3,8,64):
                for pattern in ('tie','first','last'):
                    c=np.full(k,n if pattern=='tie' else n+1,dtype=np.int64)
                    c[0 if pattern!='last' else -1]=n
                    result.append(dict(source='integer_boundary_stress',case=len(result),counts=c.tolist()))
    return result


def archive_arrays(path):
    a=np.load(path)
    if 'scores' in a.files:
        k=14; pi=np.repeat(np.arange(len(a['budget'])),k)
        axes=np.tile(np.repeat(np.arange(2),7),len(a['budget']))
        cuts=np.tile(np.array([.2,.3,.4,.5,.6,.7,.8]*2),len(a['budget']))
        saved=a['scores'].reshape(-1); counts=a['counts'].reshape(-1,2)
    else:
        pi=a['parent_index']; axes=a['axis'][pi]; cuts=a['cut']; saved=a['y']; counts=a['counts']; k=len(pi)//len(a['budget'])
    return a,k,pi,axes,cuts,saved,counts


def count_audit(execution,records):
    groups=defaultdict(list); rows=[]
    for r in records:groups[len(r['counts'])].append(r)
    for k,rs in groups.items():
        counts=np.array([r['counts'] for r in rs],dtype=np.int64)
        assert np.all(counts>=0) and np.all(counts<=2**53)
        recovered,owner=side_reduce(execution,counts.astype(np.float64),False)
        truth=counts.min(1)
        for i,r in enumerate(rs):
            rows.append(dict(source=r['source'],case=r['case'],children=k,counts=json.dumps(r['counts']),
                teacher_n_min=int(truth[i]),learned_n_min=float(recovered[i]),min_exact=bool(recovered[i]==truth[i]),
                min_owner=int(owner[i]),zero_children=int(np.sum(counts[i]==0)),
                minimum_tied=bool(np.sum(counts[i]==truth[i])>1)))
    return rows


def run():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False)
    paths=sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz'))+sorted((ROOT/'split_quality_smoke').glob('*.npz'))
    additional=[ROOT/'child_qmax_assembly/validation_fixtures.json',
        ROOT/'finite_split_score_smoke/training_geometry.json',ROOT/'finite_split_score_smoke/heldout_geometry.json']
    protected={str(p):sha(p) for p in paths+additional+list(ROOT.glob('*/*.pt'))}
    for name in ('train_finite_split_score_smoke.py','train_aspect_penalty_smoke.py','fast_nearest_argmin.py',
                 'train_radius_quotient_smoke.py','validate_child_qmax_assembly.py'):
        p=Path('experiments/sho')/name;protected[str(p)]=sha(p)
    models,execution,manifests=load_frozen()
    finite_manifest=json.loads((ROOT/'finite_split_score_smoke/candidate_manifest.json').read_text())
    assert finite_manifest['passed'] and sha(finite_manifest['checkpoint'])==finite_manifest['sha256']
    finite=FiniteSplitScore();finite.load_state_dict(torch.load(finite_manifest['checkpoint'],weights_only=False)['state_dict'])
    finite.eval().requires_grad_(False)
    records=[]
    for path in paths:
        *_,counts=archive_arrays(path)
        source=path.parent.name+'/'+path.stem
        records.extend(dict(source=source,case=i,counts=c.tolist()) for i,c in enumerate(counts))
    fixture=json.loads(additional[0].read_text())
    records.extend(dict(source='qmax_fixtures/'+r['source'],case=i,counts=r['counts']) for i,r in enumerate(fixture))
    for p in additional[1:]:
        records.extend(dict(source='finite_geometry/'+p.stem,case=i,counts=r['counts']) for i,r in enumerate(json.loads(p.read_text())))
    train_syn=synthetic(True); held_syn=synthetic(False); records+=train_syn+held_syn
    (OUT/'synthetic_counts.json').write_text(json.dumps(train_syn+held_syn))
    audit=count_audit(execution,records)
    write_csv(OUT/'split_validity_min_audit.csv',audit)
    if not all(r['min_exact'] for r in audit):
        (OUT/'STOP.txt').write_text('Existing learned min failed exact recovery. No margin/threshold training performed.\n');return
    print(json.dumps(dict(event='min_reuse_passed',sets=len(audit),multiple_zero_sets=sum(r['zero_children']>1 for r in audit),
        tied_minimum_sets=sum(r['minimum_tied'] for r in audit))),flush=True)

    # Independent archived training pools + synthetic train count vectors.
    # Four equally weighted source/class strata; no held-out examples in training.
    rng=np.random.default_rng(SEED+10)
    archive_train=[r for r in audit if r['source'] in ('recursive_region_splitter_poc/training_partitions','split_quality_smoke/training_pool')]
    syn_train=[r for r in audit if r['source']=='synthetic_train']
    training=[]
    for pool in (archive_train,syn_train):
        for valid in (False,True):
            eligible=[r for r in pool if (r['teacher_n_min']>=1)==valid]
            selected=rng.choice(len(eligible),512,replace=False)
            training.extend(eligible[i] for i in selected)
    assert all(any(r['teacher_n_min']==n for r in training) for n in (0,1,2))
    n=np.array([r['learned_n_min'] for r in training]); target=np.array([r['teacher_n_min']-1 for r in training],dtype=np.float64)
    train_labels=np.array([r['teacher_n_min']>=1 for r in training],dtype=np.float64)
    write_csv(OUT/'training_examples.csv',training)
    config=dict(seed=SEED,dtype='float64',min_operator='unchanged learned comparison; side_reduce(maximum=False)',
        integer_domain='nonnegative child counts represented exactly in float64; audit up to 2**53-1',
        margin='w*n_min+b, continuous squared error, TRF max_nfev=60',
        threshold='softplus(raw_alpha)*(m-theta), BCE classification on equally weighted archive/synthetic valid/invalid strata',
        decision='logit >= 0 -> valid; invalid -> +inf before finite upstream evaluation',
        threshold_stopping='zero training errors and probability >= .99 for each boundary example n_min=0,1,2; cap 2000 updates',
        balanced_examples=len(training),per_source_per_class=512,protected_hashes=protected,
        teacher_equivalence='1e-12 * max(1, best)',no_recursion=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    margin=ValidityMargin();initial=margin.weights.detach().numpy().copy();hist=[];best=[np.inf,None]
    tx=torch.from_numpy(n)
    def residual(w):
        with torch.no_grad():margin.weights.copy_(torch.from_numpy(w));p=margin(tx).numpy()
        e=p-target;loss=float(np.mean(e*e));hist.append(dict(evaluation=len(hist)+1,w=float(w[0]),b=float(w[1]),rmse=float(np.sqrt(loss))))
        if loss<best[0]:best[:]=[loss,w.copy()]
        return e
    opt=least_squares(residual,initial,jac=lambda _:np.column_stack((n,np.ones(len(n)))),method='trf',max_nfev=60,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    with torch.no_grad():margin.weights.copy_(torch.from_numpy(best[1]))
    margin_fit=dict(initial=initial.tolist(),learned=best[1].tolist(),calls=len(hist),training_rmse=float(np.sqrt(best[0])),message=opt.message)
    mp=OUT/'validity_margin_before_evaluation.pt';mh=save_checkpoint(mp,dict(state_dict=margin.state_dict(),config=config,fit=margin_fit))
    write_csv(OUT/'margin_training.csv',hist)
    margin.load_state_dict(torch.load(mp,weights_only=False)['state_dict']);margin.eval().requires_grad_(False)
    alln=np.array([r['learned_n_min'] for r in audit]);alltarget=np.array([r['teacher_n_min']-1 for r in audit],dtype=np.float64)
    with torch.no_grad():allm=margin(torch.from_numpy(alln)).numpy();train_m=margin(tx)
    error=allm-alltarget
    if not np.array_equal(allm,alltarget):
        (OUT/'STOP.txt').write_text(f'Margin not exact: {int(np.sum(error!=0))} mistakes, max error {abs(error).max()}. No threshold training.\n');return
    threshold=ValidityThreshold();threshold_initial=dict(raw_alpha=float(threshold.raw_alpha.detach()),alpha=float(threshold.alpha().detach()),theta=float(threshold.theta.detach()))
    optimizer=torch.optim.Adam(threshold.parameters(),lr=.05);labels=torch.from_numpy(train_labels);thistory=[]
    with torch.no_grad():boundary_m=margin(torch.tensor([0.,1.,2.],dtype=torch.float64))
    boundary_labels=torch.tensor([0.,1.,1.],dtype=torch.float64)
    for epoch in range(1,2001):
        optimizer.zero_grad();logits=threshold(train_m);loss=nn.functional.binary_cross_entropy_with_logits(logits,labels)
        loss.backward();optimizer.step()
        with torch.no_grad():
            p=threshold(train_m);mistakes=int(((p>=0)!=(labels>=.5)).sum())
            b=threshold(boundary_m);conf=torch.sigmoid(b*(2*boundary_labels-1)).min().item()
        thistory.append(dict(update=epoch,loss=float(loss.detach()),mistakes=mistakes,boundary_min_confidence=conf,
            alpha=float(threshold.alpha().detach()),theta=float(threshold.theta.detach())))
        if mistakes==0 and conf>=.99:break
    threshold_fit=dict(initial=threshold_initial,updates=epoch,alpha=float(threshold.alpha().detach()),theta=float(threshold.theta.detach()),
        training_mistakes=mistakes,boundary_min_confidence=conf)
    combined=nn.ModuleDict(dict(margin=margin,threshold=threshold));cp=OUT/'split_validity_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=combined.state_dict(),config=config,margin_fit=margin_fit,threshold_fit=threshold_fit,status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'threshold_training.csv',thistory)
    combined.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);combined.eval().requires_grad_(False)
    print(json.dumps(dict(event='trained_saved',margin_fit=margin_fit,threshold_fit=threshold_fit,sha256=digest)),flush=True)
    with torch.no_grad():all_logits=threshold(torch.from_numpy(allm)).numpy()
    for i,r in enumerate(audit):
        r.update(teacher_margin=float(alltarget[i]),learned_margin=float(allm[i]),margin_exact=bool(allm[i]==alltarget[i]),
            logit=float(all_logits[i]),teacher_valid=bool(r['teacher_n_min']>=1),learned_valid=bool(all_logits[i]>=0),
            validity_correct=bool((all_logits[i]>=0)==(r['teacher_n_min']>=1)))
    write_csv(OUT/'split_validity_full_pool.csv',audit)
    smoke=[]
    for source in ['all']+sorted(set(r['source'] for r in audit)):
        rs=audit if source=='all' else [r for r in audit if r['source']==source]
        e=np.array([r['learned_margin']-r['teacher_margin'] for r in rs])
        smoke.append(dict(record_type='pool',source=source,examples=len(rs),min_mistakes=sum(not r['min_exact'] for r in rs),
            margin_rmse=float(np.sqrt(np.mean(e*e))),margin_mae=float(np.mean(abs(e))),margin_max_error=float(abs(e).max()),
            exact_margins=sum(r['margin_exact'] for r in rs),validity_mistakes=sum(not r['validity_correct'] for r in rs),
            invalid=sum(not r['teacher_valid'] for r in rs),valid=sum(r['teacher_valid'] for r in rs),
            multiple_zero_children=sum(r['zero_children']>1 for r in rs),tied_minima=sum(r['minimum_tied'] for r in rs)))
    for n0 in (0,1,2):
        with torch.no_grad():m=margin(torch.tensor([float(n0)],dtype=torch.float64));logit=threshold(m)
        smoke.append(dict(record_type='boundary',n_min=n0,margin=float(m[0]),logit=float(logit[0]),valid=bool(logit[0]>=0)))
    write_csv(OUT/'split_validity_smoke.csv',smoke)
    if any(not r['validity_correct'] for r in audit):
        (OUT/'STOP.txt').write_text('Validity errors present; no complete score lock or finite-score evaluation.\n');return

    ranks=[];changes=[];score_rows=[];accepted=0;rejected=0;max_relative=0.
    for path in paths:
        a,k,pi,axes,cuts,saved,counts=archive_arrays(path);source=path.parent.name+'/'+path.stem
        score,teacher_valid,values,_,children=reconstruct(a['lower'],a['upper'],a['budget'],pi,axes,cuts,counts)
        assert np.array_equal(score.view(np.uint64),saved.view(np.uint64))
        nm,mm,ll,valid=route(execution,margin,threshold,counts)
        assert np.array_equal(valid,teacher_valid)
        # Routing is controlled by the learned gate; invalid rows never reach a
        # logarithm or a count-root/quotient in the frozen finite-score branch.
        parents=(a['upper']-a['lower'])[pi[valid]]
        features=upstream(models,execution,parents,a['budget'][pi[valid]],children[valid],counts[valid])
        final=np.full(len(counts),np.inf);final[valid]=predict(finite,features)
        rel=abs((final[valid]-score[valid])/score[valid]);max_relative=max(max_relative,float(rel.max()))
        accepted+=int(valid.sum());rejected+=int((~valid).sum())
        for parent in range(len(a['budget'])):
            s=slice(parent*k,(parent+1)*k)
            row=rank_audit(score[s],final[s],source,parent,'complete_locked_split_score',cuts[s],axes[s],changes)
            ranks.append(row)
        score_rows.extend(dict(source=source,candidate=i,teacher_valid=bool(teacher_valid[i]),learned_valid=bool(valid[i]),
            teacher_score=float(score[i]),learned_score=float(final[i])) for i in range(len(score)))
        print(json.dumps(dict(event='full_score_validated',source=source,accepted=int(valid.sum()),rejected=int((~valid).sum()))),flush=True)
    write_csv(OUT/'split_score_full_pool.csv',score_rows);write_csv(OUT/'split_score_parent_rankings.csv',ranks)
    if changes:write_csv(OUT/'split_score_ranking_changes.csv',changes)
    keys=[k for k in ranks[0] if k not in ('source','parent','component')]
    rank_totals=dict(parents=len(ranks),**{k:sum(r[k] for r in ranks) for k in keys})
    unchanged=all(sha(p)==h for p,h in protected.items())
    passed=bool(accepted==113837 and rejected==3674 and unchanged and rank_totals['canonical_best_changed']==0
                and rank_totals['pair_changes_outside_equivalence']==0 and max_relative<=256*np.finfo(np.float64).eps)
    summary=dict(passed=passed,split_validity_operator='LOCKED' if passed else 'DIAGNOSTIC_ONLY',split_score_operator='LOCKED' if passed else 'UNLOCKED',
        margin_fit=margin_fit,threshold_fit=threshold_fit,all_count_sets=len(audit),min_mistakes=0,margin_rmse=0.,margin_mae=0.,margin_max_error=0.,
        exact_margin_recovery=True,validity_mistakes=sum(not r['validity_correct'] for r in audit),
        multiple_zero_sets=sum(r['zero_children']>1 for r in audit),tied_minimum_sets=sum(r['minimum_tied'] for r in audit),
        archived_valid_accepted=accepted,archived_invalid_rejected=rejected,ranking=rank_totals,complete_score_max_relative_error=max_relative,
        protected_unchanged=unchanged,checkpoint_before_evaluation=str(cp),checkpoint_before_evaluation_sha256=digest,
        domain='nonnegative integer child counts; audits up to 2**53-1. Threshold is not a continuous m>=0 classifier for arbitrary real m.',
        zero_logit_rule='VALID when logit >= 0; observed n_min=0,1,2 logits are strict',
        notes='Existing deterministic equivalence/canonical tie bookkeeping retained. No recursion or search.')
    if passed:
        vp=OUT/'split_validity_operator_locked.pt'
        vh=save_checkpoint(vp,dict(state_dict=combined.state_dict(),config=config,margin_fit=margin_fit,threshold_fit=threshold_fit,
            status='LOCKED',validation=summary))
        upstream_refs={k:dict(checkpoint=m['checkpoint'],sha256=m['sha256']) for k,m in manifests.items()}
        opath=ROOT/'nearest_argmin_assembled/nearest_argmin_operator_locked.pt'
        upstream_refs.update(finite_score=dict(checkpoint=finite_manifest['checkpoint'],sha256=finite_manifest['sha256']),
            validity=dict(checkpoint=str(vp),sha256=vh),comparison=dict(checkpoint=str(opath),sha256=sha(opath)))
        sp=OUT/'split_score_operator_locked.pt'
        sh=save_checkpoint(sp,dict(status='LOCKED',operator='split_score_operator',components=upstream_refs,
            execution='learned count min -> validity margin -> threshold; rejected rows +inf; accepted rows frozen finite chain',
            canonical_rule='first original candidate within 1e-12*max(1,best) of finite minimum',validation=summary))
        summary.update(validity_checkpoint=str(vp),validity_sha256=vh,split_score_checkpoint=str(sp),split_score_sha256=sh)
        (OUT/'locked_manifest.json').write_text(json.dumps(summary,indent=2))
    summary['elapsed_seconds']=time.monotonic()-start
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    report=['Split validity and complete split score: '+('LOCKED' if passed else 'GATE FAILED'),
        'Min uses the existing learned comparator, including exact-equality handling of zero/zero pairs.',
        'Margin learned independently from random initialization: '+json.dumps(margin_fit),
        'Monotone threshold learned from balanced archived/synthetic classes: '+json.dumps(threshold_fit),
        'The learned threshold separates integer margins -1 (invalid) and 0 (valid); theta=0 is neither forced nor needed.',
        f'Exact minima and exact margins on {len(audit)} count sets; validity mistakes: {summary["validity_mistakes"]}.',
        f'Multiple-zero sets: {summary["multiple_zero_sets"]}; tied-minimum sets: {summary["tied_minimum_sets"]}.',
        f'Archived accepted finite: {accepted}; rejected invalid: {rejected}.',
        'Full score ranking: '+json.dumps(rank_totals),
        'Frozen finite scorer recomputed only on learned-accepted rows; no retraining or invalid arithmetic.',
        'Protected hashes unchanged: '+str(unchanged),
        'Complete score lock uses teacher equivalence; raw floating-point tie/witness identities are not asserted.',
        'Checkpoint/hash summary: '+json.dumps({k:v for k,v in summary.items() if 'checkpoint' in k or 'sha256' in k}),
        f'Elapsed: {summary["elapsed_seconds"]:.3f} seconds.',summary['notes']]
    (OUT/'split_validity_summary.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
