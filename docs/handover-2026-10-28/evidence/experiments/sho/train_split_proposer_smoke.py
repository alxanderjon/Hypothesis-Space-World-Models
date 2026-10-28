"""Learn eight discrete split proposals with set supervision; frozen score/allocator."""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint
from experiments.sho.train_recursive_region_splitter_poc import LockedAllocation,ALLOC,ALLOC_SHA
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict
from experiments.sho.train_split_validity_smoke import ValidityMargin,ValidityThreshold,route

ROOT=Path('outputs/sho/stage1B');OUT=ROOT/'learned_split_proposer_smoke'
SEED=27282026;M=8;CUTS=np.arange(1,200,dtype=np.float64)/200;K=len(CUTS);STEPS=768
GATES=dict(overall_best_coverage=.995,small_budget_best_coverage=1.,worst_relative_penalty=1e-8)


class SplitProposer(nn.Module):
    """Shared per-axis network, eight joint axis/cut distributions.

    Fractions use an explicit fixed vocabulary, not a continuous cut regressor.
    No teacher score is computed by the proposer. Slot deduplication masks only
    already emitted tokens; it does not inspect geometry scores or validity.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(15,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,M*K)).double()
        self.register_buffer('cut_vocabulary',torch.from_numpy(CUTS.copy()))

    def forward(self,features,axis_mask):
        b,d,_=features.shape
        logits=self.net(features).reshape(b,d,M,K).permute(0,2,1,3)
        return logits.masked_fill(~axis_mask[:,None,:,None],-1e30).reshape(b,M,d*K)

    def propose(self,features,axis_mask):
        with torch.no_grad():logits=self(features,axis_mask).numpy()
        raw=logits.argmax(2);chosen=[]
        for i in range(len(logits)):
            used=[]
            for slot in range(M):
                v=logits[i,slot].copy();v[used]=-np.inf;used.append(int(np.argmax(v)))
            chosen.append(used)
        return np.array(chosen),raw


def regions(training):
    rng=np.random.default_rng(SEED+(0 if training else 1));rows=[]
    train_b=[24,32,48,64,96,128,192,256];test_b=[29,37,73,149,251]
    reserved=((1.3,1.5),(2.7,3.3),(6,8))
    for d in range(1,6):
        for i in range(256 if training else 128):
            n=2+i%19 if i%5 else int(rng.choice(train_b if training else test_b))
            while True:
                ratio=np.exp(rng.uniform(0,np.log(16 if training else 64)))
                if d==1 or not training or not any(a<ratio<b for a,b in reserved):break
            if not training and d>1 and i%2==0:
                a,b=reserved[i%3];ratio=rng.uniform(a,b)
            scale=10**rng.uniform(-2 if training else -6,2 if training else 6)
            z=np.zeros(d) if d==1 else rng.random(d)
            if d>1:z[0]=0;z[1]=1;rng.shuffle(z)
            lengths=scale*np.exp((z-z.mean())*np.log(ratio));lo=rng.uniform(-1,1,d)*scale
            hi=lo+lengths
            rows.append(dict(d=d,n=n,lower=lo.tolist(),upper=hi.tolist(),iteration=int(rng.integers(8) if training else rng.integers(16,24)),
                pool=int(rng.integers(2)),depth=int(rng.integers(12)),group='train' if training else 'heldout'))
    return rows


def features(records):
    f=np.zeros((len(records),5,15));mask=np.zeros((len(records),5),bool)
    for i,r in enumerate(records):
        d=r['d'];s=np.array(r['upper'])-r['lower'];log=np.log(s);z=log-log.mean();n=r['n']
        for a in range(d):
            f[i,a]=[z[a],np.exp(z[a]),z.min(),z.max(),z.std(),log.mean()/10,
                np.log1p(n)/np.log(257),1/n,n/256,d/5,a/d,np.sin(r['iteration']),np.cos(r['iteration']),r['pool'],r['depth']/12]
        mask[i,:d]=True
    return torch.from_numpy(f),torch.from_numpy(mask)


def budget_bucket(n):return '2-5' if n<=5 else '6-10' if n<=10 else '11-20' if n<=20 else '>20'


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in list(ROOT.glob('*/*.pt'))+list((ROOT/'recursive_split_assembly').iterdir()) if p.is_file()}
    lockpath=ROOT/'split_validity_smoke/split_score_operator_locked.pt'
    lm=json.loads((ROOT/'split_validity_smoke/locked_manifest.json').read_text());assert sha(lockpath)==lm['split_score_sha256']
    lock=torch.load(lockpath,weights_only=False);assert lock['status']=='LOCKED'
    for c in lock['components'].values():assert sha(c['checkpoint'])==c['sha256']
    models,execution,_=load_frozen();allocator=LockedAllocation();assert sha(ALLOC)==ALLOC_SHA
    finite=FiniteSplitScore();finite.load_state_dict(torch.load(lock['components']['finite_score']['checkpoint'],weights_only=False)['state_dict'])
    gate=nn.ModuleDict(dict(margin=ValidityMargin(),threshold=ValidityThreshold()))
    gate.load_state_dict(torch.load(lock['components']['validity']['checkpoint'],weights_only=False)['state_dict'])
    for model in (finite,gate):model.eval().requires_grad_(False)
    config=dict(seed=SEED,M=M,fraction_vocabulary=CUTS.tolist(),dimensions=list(range(1,6)),steps=STEPS,dtype='float64',
        representation='eight learned axis/cut categorical heads over 199 fractions; inference scores no splits itself',
        training='set membership, at-least-one optimum coverage, reverse set coverage, and slot collision penalty',
        small_budget_fraction='80% n=2..20',heldout='new geometries, reserved aspect bands, wider scales, larger unseen budgets, iteration codes',
        best_definition='minimum of locked scores across all axes and 199 fractions, not a certified continuous-cut optimum',
        good_set='up to 8 best/equivalent reference tokens, retaining multiple equivalents and mirrored choices',
        proposal_deduplication='mask exact already-emitted axis/cut tokens; no score or validity fallback',gates=GATES,protected_hashes=protected)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))

    def score_proposals(rs,axis,cut):
        # axis/cut are BxP; use exactly the same locked arithmetic for menus and learned proposals.
        d=rs[0]['d'];b,p=axis.shape
        lo=np.repeat(np.array([r['lower'] for r in rs]),p,axis=0);hi=np.repeat(np.array([r['upper'] for r in rs]),p,axis=0)
        n=np.repeat(np.array([r['n'] for r in rs]),p);a=axis.ravel();c=cut.ravel();idx=np.arange(len(a))
        cl=np.repeat(lo[:,None,:],2,1);ch=np.repeat(hi[:,None,:],2,1)
        coordinate=lo[idx,a]+c*(hi[idx,a]-lo[idx,a]);ch[idx,0,a]=coordinate;cl[idx,1,a]=coordinate
        lengths=ch-cl;parents=hi-lo
        counts,_,_=allocator.allocate(lengths/parents[:,None,:],n)
        _,_,_,valid=route(execution,gate['margin'],gate['threshold'],counts)
        score=np.full(len(a),np.inf)
        if valid.any():score[valid]=predict(finite,upstream(models,execution,parents[valid],n[valid],lengths[valid],counts[valid]))
        return score.reshape(b,p),counts.reshape(b,p,2)

    def label(rs,group):
        scores=np.full((len(rs),5*K),np.inf);target=np.zeros_like(scores,bool);optimal=np.zeros_like(scores,bool)
        for d in range(1,6):
            ids=np.array([i for i,r in enumerate(rs) if r['d']==d])
            for start in range(0,len(ids),16):
                ix=ids[start:start+16];batch=[rs[i] for i in ix]
                axes=np.tile(np.repeat(np.arange(d),K),(len(ix),1));cuts=np.tile(CUTS,d)[None,:].repeat(len(ix),0)
                s,_=score_proposals(batch,axes,cuts);scores[ix,:d*K]=s
                for j,i in enumerate(ix):
                    best=s[j].min();assert np.isfinite(best)
                    eq=np.flatnonzero(s[j]<=best+1e-12*max(1,float(best)));optimal[i,eq]=True
                    if len(eq)>=M:good=eq[np.linspace(0,len(eq)-1,M).round().astype(int)]
                    else:
                        good=eq.tolist()
                        for token in np.argsort(s[j],kind='stable'):
                            if token not in good and np.isfinite(s[j,token]):good.append(int(token))
                            if len(good)==M:break
                    target[i,good]=True
            print(json.dumps(dict(event='reference_labeled',group=group,dimension=d,regions=len(ids),menu_per_region=d*K)),flush=True)
        np.savez_compressed(OUT/(group+'_reference.npz'),scores=scores,good_set=target,optimal_set=optimal)
        (OUT/(group+'_regions.json')).write_text(json.dumps(rs))
        return scores,target,optimal

    train=regions(True);train_scores,good,optimum=label(train,'train')
    tx,tm=features(train);target=torch.from_numpy(good);best_target=torch.from_numpy(optimum)
    model=SplitProposer();initial=copy.deepcopy(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    history=[];best=[np.inf,None,0];rng=np.random.default_rng(SEED+5)
    def loss_for(index):
        logit=model(tx[index],tm[index]);prob=logit.softmax(2)
        support=target[index];eq=best_target[index]
        mass=(prob*support[:,None,:]).sum(2).clamp_min(1e-15)
        eqmass=(prob*eq[:,None,:]).sum(2)
        hit=(1-(1-eqmass).prod(1)).clamp_min(1e-15)
        reverse=-(torch.log(prob.mean(1).clamp_min(1e-15))*support).sum(1)/support.sum(1)
        pairs=torch.bmm(prob,prob.transpose(1,2));off=~torch.eye(M,dtype=torch.bool)
        collision=pairs[:,off].mean()
        return -torch.log(mass).mean()-2*torch.log(hit).mean()+.25*reverse.mean()+.5*collision
    def monitor(step):
        losses=[];choices=[]
        with torch.no_grad():
            for start in range(0,len(train),128):
                ids=np.arange(start,min(start+128,len(train)));losses.append(float(loss_for(ids)))
                choices.extend(model.propose(tx[ids],tm[ids])[0])
        selected=np.array(choices);best_scores=train_scores[np.arange(len(train))[:,None],selected].min(1)
        truth=train_scores.min(1);coverage=float(np.mean(best_scores<=truth+1e-12*np.maximum(1,truth)))
        loss=float(np.mean(losses));history.append(dict(step=step,loss=loss,training_coverage=coverage))
        if loss<best[0]:best[:]=[loss,copy.deepcopy(model.state_dict()),step]
        print(json.dumps(dict(event='training_monitor',**history[-1])),flush=True)
    monitor(0)
    for step in range(1,STEPS+1):
        ids=rng.integers(len(train),size=64);optimizer.zero_grad();loss=loss_for(ids);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%128==0:monitor(step)
    model.load_state_dict(best[1]);cp=OUT/'split_proposer_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=model.state_dict(),initial_state_dict=initial,config=config,selected_update=best[2],status='SAVED_BEFORE_EVALUATION'))
    write_csv(OUT/'split_proposer_training.csv',history)
    model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
    print(json.dumps(dict(event='checkpoint_saved',selected_update=best[2],sha256=digest)),flush=True)

    heldout=regions(False);reference,_,eq=label(heldout,'heldout');hx,hm=features(heldout)
    selected,raw=model.propose(hx,hm);proposal_scores=np.empty((len(heldout),M));proposal_counts=np.empty((len(heldout),M,2),dtype=np.int64)
    for d in range(1,6):
        ids=np.array([i for i,r in enumerate(heldout) if r['d']==d])
        s,c=score_proposals([heldout[i] for i in ids],selected[ids]//K,CUTS[selected[ids]%K]);proposal_scores[ids]=s;proposal_counts[ids]=c
    cached=reference[np.arange(len(heldout))[:,None],selected]
    assert np.array_equal(np.isfinite(cached),np.isfinite(proposal_scores))
    finite_mask=np.isfinite(cached)
    rescoring_max=float(np.max(abs(cached[finite_mask]-proposal_scores[finite_mask])))
    rows=[];details=[]
    for i,r in enumerate(heldout):
        best_true=float(reference[i].min());best_proposed=float(proposal_scores[i].min());tol=1e-12*max(1,best_true)
        ax=selected[i]//K;cuts=CUTS[selected[i]%K];pairs=[]
        for a in range(M):
            for b in range(a):pairs.append(float((ax[a]!=ax[b])+abs(cuts[a]-cuts[b])))
        rows.append(dict(case=i,dimension=r['d'],budget=r['n'],budget_bucket=budget_bucket(r['n']),iteration=r['iteration'],pool=r['pool'],
            teacher_best_score=best_true,learned_best_score=best_proposed,best_score_ratio=best_proposed/best_true,
            relative_penalty=best_proposed/best_true-1,absolute_penalty=best_proposed-best_true,
            teacher_equivalent=bool(best_proposed<=best_true+tol),invalid_proposals=int(np.sum(~np.isfinite(proposal_scores[i]))),
            all_invalid=bool(not np.isfinite(best_proposed)),unique_proposals=len(np.unique(selected[i])),
            raw_head_unique_proposals=len(np.unique(raw[i])),unique_axes=len(np.unique(ax)),unique_cuts=len(np.unique(cuts)),
            cut_span=float(cuts.max()-cuts.min()),mean_pair_diversity=float(np.mean(pairs)),minimum_pair_diversity=min(pairs)))
        for j in range(M):details.append(dict(case=i,slot=j,axis=int(ax[j]),cut=float(cuts[j]),score=float(proposal_scores[i,j]),
            counts=json.dumps(proposal_counts[i,j].tolist()),teacher_equivalent=bool(proposal_scores[i,j]<=best_true+tol)))
    write_csv(OUT/'split_proposer_cases.csv',rows);write_csv(OUT/'split_proposer_proposals.csv',details)
    summary=[]
    for by in ('all','dimension','budget_bucket'):
        for value in (['all'] if by=='all' else sorted(set(r[by] for r in rows),key=str)):
            rs=rows if by=='all' else [r for r in rows if r[by]==value]
            summary.append(dict(breakdown=by,value=value,regions=len(rs),best_split_coverage=float(np.mean([r['teacher_equivalent'] for r in rs])),
                mean_score_ratio=float(np.mean([r['best_score_ratio'] for r in rs])),worst_relative_penalty=max(r['relative_penalty'] for r in rs),
                worst_absolute_penalty=max(r['absolute_penalty'] for r in rs),invalid_proposal_rate=sum(r['invalid_proposals'] for r in rs)/(M*len(rs)),
                all_invalid_regions=sum(r['all_invalid'] for r in rs),mean_unique_proposals=float(np.mean([r['unique_proposals'] for r in rs])),
                mean_raw_head_unique=float(np.mean([r['raw_head_unique_proposals'] for r in rs])),
                mean_unique_axes=float(np.mean([r['unique_axes'] for r in rs])),mean_cut_span=float(np.mean([r['cut_span'] for r in rs])),
                mean_pair_diversity=float(np.mean([r['mean_pair_diversity'] for r in rs]))))
    write_csv(OUT/'split_proposer_smoke.csv',summary)
    small=[r for r in rows if r['budget']<=20];overall=summary[0]
    passed=overall['best_split_coverage']>=GATES['overall_best_coverage'] and all(r['teacher_equivalent'] for r in small) and overall['worst_relative_penalty']<=GATES['worst_relative_penalty']
    unchanged=all(sha(p)==h for p,h in protected.items());passed=bool(passed and unchanged)
    worst=max(rows,key=lambda r:r['relative_penalty'])
    failure=dict(region=heldout[worst['case']],result=worst,proposals=[r for r in details if r['case']==worst['case']])
    (OUT/'worst_case.json').write_text(json.dumps(failure,indent=2))
    result=dict(status='PROPOSAL_QUALITY_PASSED_PENDING_RECURSION' if passed else 'DIAGNOSTIC_ONLY',proposal_gate_passed=passed,
        checkpoint=str(cp),sha256=digest,selected_update=best[2],updates_run=STEPS,training_regions=len(train),heldout_regions=len(heldout),
        reference_candidates_train=int(sum(r['d']*K for r in train)),reference_candidates_heldout=int(sum(r['d']*K for r in heldout)),
        overall=overall,small_budget_coverage=float(np.mean([r['teacher_equivalent'] for r in small])),worst_case=worst,
        independently_rescored_max_error=rescoring_max,protected_hashes_unchanged=unchanged,allocator_stats=allocator.stats(),
        recursive_screen_run=False,elapsed_seconds=time.monotonic()-started,
        limitation='Discrete proposal-set learning over fixed 0.005-spaced cut vocabulary; no continuous-cut optimum claim.',
        stop_reason=None if passed else 'Proposal quality gate failed; no recursion, placement training, or search.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    lines=['Learned split proposer: '+result['status'],json.dumps(result,indent=2),
        'Teacher targets and evaluation use the locked allocator, validity gate, and full learned score.',
        'The proposer emits exactly 8 distinct axis/cut tokens. Deduplication masks only prior emitted tokens; no oracle filtering or fallback.',
        'Raw-head uniqueness is reported to expose slot collapse despite output deduplication.',
        'dimension/budget | group | regions | best coverage | mean score ratio | worst relative penalty | invalid proposal rate']
    for s in summary:lines.append(f"{s['breakdown']} | {s['value']} | {s['regions']} | {s['best_split_coverage']:.6g} | {s['mean_score_ratio']:.6g} | {s['worst_relative_penalty']:.6g} | {s['invalid_proposal_rate']:.6g}")
    (OUT/'split_proposer_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
