"""Continue only terminal-fraction training; frozen locator and offline minima."""
import ast
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from torch import nn
from experiments.geometry.train_stationary_terminal_candidate import TerminalCandidate,features,quality,OUT,ROOT,SEED
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.train_stationary_midpoint_operator import Midpoint
from experiments.geometry.train_stationary_query_scheduler import QueryScheduler,TABLE,MAIN,DLEFT,DRIGHT,NONE
from experiments.geometry.train_stationary_bracket_action_router import BracketActionRouter,LEFT,RIGHT
from experiments.geometry.train_stationary_stopping_operator import StoppingOperator
from experiments.geometry.validate_stationary_three_state_router import load_router,DEFER
from experiments.sho.train_subpiece_stationary_solver import roots_high_precision
from experiments.sho.audit_fixed_allocation_intervals import Audit
from experiments.sho.smoke_artifacts import save_checkpoint


class FrozenLocator:
    """The locked five-component locator, with no candidate enumeration."""
    def __init__(self):
        self.refs={}
        for name,folder,filename,cls in [
            ('midpoint','stationary_midpoint_operator_training','midpoint_before_evaluation.pt',Midpoint),
            ('scheduler','stationary_query_scheduler_training','query_scheduler_before_evaluation.pt',QueryScheduler),
            ('action','stationary_bracket_action_router_training','bracket_action_router_before_evaluation.pt',BracketActionRouter),
            ('stop','stationary_stopping_operator_training','stopping_operator_before_evaluation.pt',StoppingOperator)]:
            m=json.loads((ROOT/folder/'results.json').read_text());assert m['status']=='LOCKED'
            cp=ROOT/folder/filename;assert ref.sha(cp)==m.get('checkpoint_sha256',m.get('sha256'))
            model=cls();model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);model.eval().requires_grad_(False)
            setattr(self,name,model);self.refs[name]=dict(checkpoint=str(cp),sha256=ref.sha(cp))
        self.sign,self.refs['sign']=load_router()

    def stationary_candidates(self,coeff,initialL,initialR,terminal_model):
        """Candidate inference path: one prediction per bracket, no enumeration."""
        brackets=self.locate(coeff,initialL,initialR)
        x=features(self.sign,coeff,brackets['L'],brackets['R'],brackets['adjacent'])
        cuts=terminal_model.locate(brackets['L'],brackets['R'],x)
        assert np.all((cuts>=brackets['L'])&(cuts<=brackets['R']))
        return brackets,cuts

    def locate(self,coeff,initialL,initialR):
        L=initialL.copy();R=initialR.copy();n=len(L);oldL=L.copy();oldR=R.copy();dq=np.full(n,np.nan)
        qL=np.full(n,np.nan);qR=np.full(n,np.nan);flags=np.tile(TABLE[0],(n,1));active=np.ones(n,bool)
        reason=np.full(n,'',object);counts=np.zeros(n,int);iterations=np.zeros(n,int)
        def sign(ids,c):
            a=coeff[ids]
            with torch.no_grad():result=self.sign(torch.from_numpy(np.column_stack((a[:,:4],c,a[:,4:6])))).numpy()
            counts[ids]+=1;return result
        sl=sign(np.arange(n),L);sr=sign(np.arange(n),R)
        def invariant():assert np.all((L<R)&(sl!=sr)&(sl!=DEFER)&(sr!=DEFER))
        def schedule(ids):
            p=self.scheduler.predict(flags[ids]);take=p[:,0]!=NONE;out=np.full(len(ids),np.nan)
            if take.any():
                ii=ids[take];bank=np.column_stack((L[ii],R[ii],oldL[ii],oldR[ii],dq[ii],np.full(len(ii),np.nan)))
                pair=np.take_along_axis(bank,p[take,1:],axis=1);assert np.isfinite(pair).all()
                out[take]=self.midpoint.evaluate(*pair.T)
            return p[:,0],out
        def update(ids,c,signs):
            actions=self.action.actions(np.column_stack((sl[ids],sr[ids],signs)))
            left=actions==LEFT;right=actions==RIGHT
            L[ids[left]]=c[left];sl[ids[left]]=signs[left];R[ids[right]]=c[right];sr[ids[right]]=signs[right]
        invariant()
        while active.any():
            ids=np.flatnonzero(active);flags[ids]=TABLE[0];act,q=schedule(ids);assert np.all(act==MAIN)
            flags[ids,1]=1;wait,_=schedule(ids);assert np.all(wait==NONE)
            adj=np.nextafter(L[ids],R[ids])==R[ids];rounded=(q==L[ids])|(q==R[ids])
            done=self.stop.predict(np.column_stack((np.ones(len(ids)),adj,rounded,np.zeros((len(ids),2))))).astype(bool)
            stopids=ids[done];active[stopids]=False;flags[stopids,0]=0
            reason[stopids]=np.where(adj[done],'ADJACENT','ROUNDED')
            if len(stopids):assert np.all(schedule(stopids)[0]==NONE)
            ids=ids[~done];q=q[~done]
            if not len(ids):break
            iterations[ids]+=1;states=sign(ids,q);update(ids,q,states);invariant()
            flags[ids,2]=1;flags[ids,3]=states==DEFER;di=ids[states==DEFER]
            oldL[di]=L[di];oldR[di]=R[di];dq[di]=q[states==DEFER]
            pending=ids
            while len(pending):
                act,c=schedule(pending);assert not np.any(act==MAIN)
                left=act==DLEFT;right=act==DRIGHT
                qL[pending[left]]=c[left];flags[pending[left],4]=1
                qR[pending[right]]=c[right];flags[pending[right],5]=1
                pending=pending[act!=NONE]
            if len(di):
                left=sign(di,qL[di]);right=sign(di,qR[di]);update(di,qL[di],left);invariant();update(di,qR[di],right);invariant()
                assert np.all(L[di]>=oldL[di]) and np.all(R[di]<=oldR[di])
                stop=self.stop.predict(np.column_stack((np.zeros((len(di),3)),L[di]>oldL[di],R[di]<oldR[di]))).astype(bool)
                stopped=di[stop];active[stopped]=False;flags[stopped,0]=0;reason[stopped]='NO_PROGRESS'
                if len(stopped):assert np.all(schedule(stopped)[0]==NONE)
        invariant();assert np.array_equal(sign(np.arange(n),L),sl) and np.array_equal(sign(np.arange(n),R),sr)
        return dict(L=L,R=R,adjacent=reason=='ADJACENT',reason=reason,queries=counts,iterations=iterations)


def coeff_record(a):return dict(zip(('A','B','a','beta','t','v'),a[:6]))


def generic_score(a,c):
    c=np.asarray(c);A,B,aa,beta,ot,ov,C=a;t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
    return A*np.sqrt(B+aa**2*t*t)+beta*np.log(v)+C


def original_parents():
    rows=[]
    for role in ('train','heldout'):rows+=json.loads((ROOT/f'learned_split_proposer_smoke/{role}_regions.json').read_text())
    old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz')
    rows.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
    return rows


def high_root(a,l,r):
    return [root for root in roots_high_precision(coeff_record(a)) if mp.mpf(float(l))<root<mp.mpf(float(r))]


def fresh_data():
    rng=np.random.default_rng(SEED+19);pieces=[];entries=[];parents={};audit=Audit()
    # Complete interval/axis candidate pools for eight fresh geometric parents.
    for pi in range(8):
        d=2+pi%3;n=(3,5,7,11)[pi%4];s=10**rng.uniform(-.55,.55,d)*10**rng.uniform(-3,3)
        parent=dict(d=d,n=n,lower=np.zeros(d).tolist(),upper=s.tolist());parents[pi]=parent
        for axis in range(d):
            audit.setup(parent,axis);edge=1e-14;scan=np.linspace(edge,1-edge,max(2049,n*32+1));cc=audit.counts(scan)[:,0]
            assert np.all(np.diff(cc)>=0) and cc[0]==0 and cc[-1]==n
            lo=np.array([scan[np.flatnonzero(cc<k)[-1]] for k in range(1,n+1)])
            hi=np.array([scan[np.flatnonzero(cc>=k)[0]] for k in range(1,n+1)])
            for _ in range(60):
                mid=(lo+hi)/2;mask=audit.counts(mid)[:,0]<np.arange(1,n+1);lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
            for k in range(1,n):
                br=audit.breaks(hi[k-1],lo[k],k)
                for j,(l,r) in enumerate(zip(br[:-1],br[1:])):
                    case=f'{pi}:{axis}:{k}:{j}';f=audit.formula(np.array([(l+r)/2]),k)
                    iq=int(f[3][0]);ia=int(f[4][0]);power=int(f[5][0,ia])
                    A=1/(audit.h*np.power(float(k if iq==0 else n-k),1/d));beta=.05*power
                    C=.05*np.log(audit.a/audit.omin) if power==1 else .05*np.log(audit.omax/audit.a) if power==-1 else .05*np.log(audit.omax/audit.omin)
                    a=np.array([A,audit.B,audit.a,beta,iq,ia,C]);roots=high_root(a,l,r)
                    assert len(roots)<=1
                    p=dict(scope='fresh_geometric',case=case,parent=pi,axis=axis,k=k,lo=float(l),hi=float(r),coeff=a,root=roots[0] if roots else None)
                    pieces.append(p)
                    for kind,c in [('left',l),('right',r)]:
                        entries.append(dict(parent=pi,axis=axis,k=k,case=case,identity=case+':'+kind,kind=kind,teacher_cut=float(c),learned_cut=float(c)))
                    if roots:entries.append(dict(parent=pi,axis=axis,k=k,case=case,identity=case+':stationary',kind='stationary',teacher_cut=float(roots[0]),learned_cut=None))
    # Broad coefficient/orientation coverage, completely independent of fitting.
    for i in range(192):
        A=10**rng.uniform(-6,6);a=10**rng.uniform(-6,6);B=a*a*10**rng.uniform(-3,3)
        ot=i%2;ov=(i//2)%2;c=float(rng.uniform(.025,.975));t=c if ot==0 else 1-c;v=c if ov==0 else 1-c
        st=1 if ot==0 else -1;sv=1 if ov==0 else -1
        beta=-A*a*a*t*st*v/(sv*np.sqrt(B+a*a*t*t));aa=np.array([A,B,a,beta,ot,ov,0.])
        roots=roots_high_precision(coeff_record(aa));idx=int(np.argmin([abs(float(root)-c) for root in roots]));root=roots[idx]
        left=mp.mpf(0) if idx==0 else roots[idx-1];right=mp.mpf(1) if idx==len(roots)-1 else roots[idx+1]
        l=float((left+root)/2);r=float((right+root)/2)
        pieces.append(dict(scope='fresh_coefficients',case='generic:'+str(i),parent=None,axis=0,k=0,lo=l,hi=r,coeff=aa,root=root))
    return pieces,entries,parents


def selection_checks(entries,score_fn,group):
    """Unchanged teacher canonical rule; isolate terminal-coordinate effect."""
    for e in entries:
        e['teacher_score']=float(score_fn(e,e['teacher_cut']));e['learned_score']=float(score_fn(e,e['learned_cut']))
    def canonical(pool,arm):
        order=sorted(pool,key=lambda e:(e['axis'],e[arm+'_cut'],e['identity']))
        best=min(e[arm+'_score'] for e in order)
        chosen=next(e for e in order if e[arm+'_score']<=best+1e-12*max(1,best))
        return chosen,best
    levels=[];results={}
    for level,keys in [('subpiece',('parent','axis','k','case')),('interval',('parent','axis','k')),('parent',('parent',))]:
        pools=defaultdict(list)
        for e in entries:pools[tuple(e[k] for k in keys)].append(e)
        for key,pool in pools.items():
            t,best=canonical(pool,'teacher');l,_=canonical(pool,'learned');tol=1e-12*max(1,best)
            levels.append(dict(group=group,level=level,case=str(key),teacher_identity=t['identity'],learned_identity=l['identity'],
                identity_changed=t['identity']!=l['identity'],teacher_cut=t['teacher_cut'],learned_cut=l['learned_cut'],
                coordinate_changed=t['axis']!=l['axis'] or t['teacher_cut']!=l['learned_cut'],
                score_penalty=max(0.,l['learned_score']-best),outside_equivalence=l['learned_score']>best+tol))
        current=[r for r in levels if r['level']==level]
        results[level]=dict(pools=len(current),identity_changes=sum(r['identity_changed'] for r in current),
            raw_coordinate_changes=sum(r['coordinate_changed'] for r in current),outside_equivalence=sum(r['outside_equivalence'] for r in current))
    ref.write_csv(OUT/(group+'_minimum_selection.csv'),levels)
    return results


def run():
    start=time.monotonic();torch.set_num_threads(1);mp.mp.dps=90
    screen=json.loads((OUT/'quick_screen_results.json').read_text());assert screen['screen_passed']
    assert not (OUT/'full_results.json').exists()
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    data=np.load(OUT/'archive_terminal_features.npz');xx=data['features'];y=data['target_fraction'];L=data['L'];R=data['R'];coeff=data['coeff']
    roots=[mp.mpf(s) for s in data['root_decimal']];adj=data['adjacent'];rng=np.random.default_rng(SEED+21)
    # Reserve quick validation plus additional unseen archived cases. Fresh
    # coefficient/geometry brackets are generated only after model persistence.
    heldout=np.zeros(len(y),bool);heldout[data['validation']]=True
    extra=np.setdiff1d(rng.permutation(len(y))[:1000],data['train']);heldout[extra]=True
    train=np.flatnonzero(~heldout);valid=np.flatnonzero(heldout)
    model=TerminalCandidate();model.load_state_dict(torch.load(OUT/'quick_screen_before_evaluation.pt',weights_only=False)['state_dict'])
    opt=torch.optim.Adam(model.parameters(),lr=.005);xt=torch.from_numpy(xx[train]);yt=torch.from_numpy(y[train])
    history=[];best_loss=float('inf');best=None
    for step in range(1,1001):
        opt.zero_grad();loss=((model(xt)-yt)**2).mean();loss.backward();opt.step()
        with torch.no_grad():value=float(((model(xt)-yt)**2).mean())
        if value<best_loss:best_loss=value;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step%50==0:history.append(dict(update=step,training_MSE=value))
    model.load_state_dict(best);model.eval().requires_grad_(False)
    checkpoint=OUT/'terminal_candidate_before_evaluation.pt'
    digest=save_checkpoint(checkpoint,dict(state_dict=model.state_dict(),initial_checkpoint=screen['checkpoint'],initial_sha256=screen['checkpoint_sha256'],
        seed=SEED,updates=1000,prior_updates=600,training_indices=train.tolist(),heldout_indices=valid.tolist(),
        architecture='5 local features -> affine -> sigmoid; predicts bracket fraction',target='high-precision mathematical root fraction',
        trained_only_terminal_fraction=True))
    ref.write_csv(OUT/'full_training_history.csv',history)
    print(json.dumps(dict(stage='full_model_saved_before_evaluation',train=len(train),heldout=len(valid),sha256=digest)),flush=True)
    locator=FrozenLocator()
    archived=list(csv.DictReader((ROOT/'defer_aware_stationary_locator_audit/defer_aware_root_recovery.csv').open()))
    located=locator.locate(coeff,np.array([float(r['initial_L']) for r in archived]),np.array([float(r['initial_R']) for r in archived]))
    assert np.array_equal(located['L'],L) and np.array_equal(located['R'],R), 'Frozen locator regression.'
    assert np.array_equal(located['iterations'],[int(r['iterations']) for r in archived])
    pred=model.predict_fraction(xx);cuts=L+pred*(R-L);mid=locator.midpoint.evaluate(L,R)
    metric_rows=[]
    def add_metrics(group,ids,L,R,roots,lam,target,adj):
        for stop in ('all','ADJACENT','NO_PROGRESS'):
            select=ids if stop=='all' else ids[adj[ids]==(stop=='ADJACENT')]
            if not len(select):continue
            for arm,frac in [('learned',lam[select]),('midpoint',np.full(len(select),.5))]:
                metric_rows.append(dict(group=group,stopping_type=stop,arm=arm,
                    **quality(L[select],R[select],[roots[i] for i in select],frac,target[select])))
    add_metrics('all_archived',np.arange(len(y)),L,R,roots,pred,y,adj)
    add_metrics('heldout_archived',valid,L,R,roots,pred,y,adj)
    parents=original_parents();audit=Audit.__new__(Audit)
    amin=list(csv.DictReader((ROOT/'stationary_float64_minimizer_full_audit/stationary_float64_minimizers.csv').open()))
    index={(r['scope'],r['case']):i for i,r in enumerate(archived)}
    comparisons=[]
    for i,r in enumerate(archived):
        cs=np.array([float(roots[i]),cuts[i],mid[i],float(amin[i]['c_learned_min'])])
        if r['scope']=='geometric':
            parent,axis,k,piece=ast.literal_eval(r['case']);audit.setup(parents[parent],axis);ss=audit.formula(cs,k)[0]
            endpoint=audit.formula(np.array([float(r['initial_L']),float(r['initial_R'])]),k)[0]
        else:
            ss=generic_score(coeff[i],cs);endpoint=generic_score(coeff[i],np.array([float(r['initial_L']),float(r['initial_R'])]))
        tol=1e-12*max(1.,ss[0]);kinds=['left','stationary','right'];oldpool=[(float(r['initial_L']),endpoint[0],'left'),(cs[0],ss[0],'stationary'),(float(r['initial_R']),endpoint[1],'right')]
        newpool=[oldpool[0],(cs[1],ss[1],'stationary'),oldpool[2]]
        def canonical(pool):
            best=min(v[1] for v in pool)
            return next(v[2] for v in sorted(pool,key=lambda z:z[0]) if v[1]<=best+1e-12*max(1.,best))
        comparisons.append(dict(group='archive',scope=r['scope'],case=r['case'],c_teacher=cs[0],c_hat=cs[1],c_midpoint=cs[2],c_enumerated=cs[3],
            teacher_score=ss[0],learned_location_teacher_score=ss[1],midpoint_teacher_score=ss[2],enumerated_teacher_score=ss[3],
            score_difference=abs(ss[1]-ss[0]),outside_equivalence=bool(abs(ss[1]-ss[0])>tol),
            subpiece_identity_changed=canonical(oldpool)!=canonical(newpool),enumerated_coordinate_match=cs[1]==cs[3],
            enumerated_score_difference=abs(ss[1]-ss[3]),enumerated_outside_equivalence=bool(abs(ss[1]-ss[3])>tol)))
    entries=[]
    for r in csv.DictReader((ROOT/'fixed_allocation_interval_minimizer_audit/complete_candidate_sets.csv').open()):
        e={k:int(r[k]) for k in ('parent','axis')};e.update(k=int(r['interval']),case=r['identity'].rsplit(':',1)[0],identity=r['identity'],kind=r['kind'],teacher_cut=float(r['teacher_cut']),learned_cut=float(r['teacher_cut']))
        if r['kind']=='stationary':
            i=index[('geometric',e['case'])];e['teacher_cut']=float(roots[i]);e['learned_cut']=float(cuts[i])
        entries.append(e)
    def archive_score(e,c):
        audit.setup(parents[e['parent']],e['axis']);return audit.formula(np.array([c]),e['k'])[0][0]
    archive_selection=selection_checks(entries,archive_score,'archived')
    ref.write_csv(OUT/'archived_candidate_sets.csv',entries)
    print(json.dumps(dict(stage='archived_validation',selection=archive_selection,score_failures=sum(r['outside_equivalence'] for r in comparisons))),flush=True)
    # Fresh holdout: no fitting, selection or tuning from this point onward.
    pieces,fresh_entries,freshparents=fresh_data();bracketed=[p for p in pieces if p['root'] is not None]
    cf=np.array([p['coeff'] for p in bracketed]);lo=np.array([p['lo'] for p in bracketed]);hi=np.array([p['hi'] for p in bracketed])
    fresh=locator.locate(cf,lo,hi);fl=fresh['L'];fr=fresh['R'];fro=[p['root'] for p in bracketed]
    ftarget=np.array([float((root-mp.mpf(float(l)))/(mp.mpf(float(r))-mp.mpf(float(l)))) for root,l,r in zip(fro,fl,fr)])
    fx=features(locator.sign,cf,fl,fr,fresh['adjacent']);fp=model.predict_fraction(fx);fc=fl+fp*(fr-fl)
    containment=np.array([mp.mpf(float(l))<=root<=mp.mpf(float(r)) for root,l,r in zip(fro,fl,fr)])
    add_metrics('fresh_holdout',np.arange(len(fc)),fl,fr,fro,fp,ftarget,fresh['adjacent'])
    flookup={p['case']:i for i,p in enumerate(bracketed)}
    fresh_scores=[]
    for i,p in enumerate(bracketed):
        cs=np.array([float(p['root']),fc[i],(fl[i]+fr[i])/2,p['lo'],p['hi']])
        if p['scope']=='fresh_geometric':
            audit.setup(freshparents[p['parent']],p['axis']);ss=audit.formula(cs,p['k'])[0]
        else:ss=generic_score(p['coeff'],cs)
        def pick(c,score):
            pool=[(p['lo'],ss[3],'left'),(c,score,'stationary'),(p['hi'],ss[4],'right')];best=min(v[1] for v in pool)
            return next(v[2] for v in sorted(pool,key=lambda z:z[0]) if v[1]<=best+1e-12*max(1.,best))
        fresh_scores.append(dict(group='fresh',scope=p['scope'],case=p['case'],c_teacher=cs[0],c_hat=fc[i],L=fl[i],R=fr[i],teacher_root_contained=bool(containment[i]),
            teacher_score=ss[0],learned_location_teacher_score=ss[1],midpoint_teacher_score=ss[2],score_difference=abs(ss[1]-ss[0]),
            outside_equivalence=bool(abs(ss[1]-ss[0])>1e-12*max(1.,ss[0])),subpiece_identity_changed=pick(cs[0],ss[0])!=pick(cs[1],ss[1])))
    for e in fresh_entries:
        if e['kind']=='stationary':e['learned_cut']=float(fc[flookup[e['case']]])
    def fresh_score(e,c):
        audit.setup(freshparents[e['parent']],e['axis']);return audit.formula(np.array([c]),e['k'])[0][0]
    fresh_selection=selection_checks(fresh_entries,fresh_score,'fresh')
    ref.write_csv(OUT/'fresh_candidate_sets.csv',fresh_entries)
    ref.write_csv(OUT/'terminal_location_quality.csv',metric_rows)
    ref.write_csv(OUT/'archived_minimum_quality.csv',comparisons)
    ref.write_csv(OUT/'fresh_minimum_quality.csv',fresh_scores)
    np.savez_compressed(OUT/'fresh_terminal_brackets.npz',coeff=cf,initial_L=lo,initial_R=hi,L=fl,R=fr,features=fx,
        root_decimal=np.array([str(r) for r in fro]),lambda_teacher=ftarget,lambda_hat=fp,candidate=fc,stopping_type=fresh['reason'])
    (OUT/'fresh_parents.json').write_text(json.dumps(freshparents,indent=2))
    failures=[r for r in comparisons+fresh_scores if r['outside_equivalence'] or r['subpiece_identity_changed']]
    out_of_bracket=int(np.sum((cuts<L)|(cuts>R)))+int(np.sum((fc<fl)|(fc>fr)))
    passed=not failures and out_of_bracket==0 and all(v['outside_equivalence']==v['identity_changes']==0 for s in (archive_selection,fresh_selection) for v in s.values())
    passed=passed and archive_selection['parent']['raw_coordinate_changes']==fresh_selection['parent']['raw_coordinate_changes']==0
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    result=dict(operator='stationary_terminal_candidate_operator',status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture='5 local features -> learned affine -> sigmoid -> bracket fraction; 6 float64 parameters',
        parameters={k:v.tolist() for k,v in model.state_dict().items()},training_cases=len(train),heldout_archived_cases=len(valid),additional_updates=1000,
        quick_screen=screen,location_metrics=metric_rows,predictions_outside_bracket=out_of_bracket,
        frozen_archive_locator_final_bracket_changes=0,fresh_bracketed_subpieces=len(fc),fresh_geometric_parents=len(freshparents),
        fresh_geometric_stationary_cases=sum(p['scope']=='fresh_geometric' for p in bracketed),fresh_generic_stationary_cases=sum(p['scope']=='fresh_coefficients' for p in bracketed),
        fresh_teacher_roots_contained=int(containment.sum()),
        archived_score_max_difference=max(r['score_difference'] for r in comparisons),fresh_score_max_difference=max(r['score_difference'] for r in fresh_scores),
        archived_score_outside_equivalence=sum(r['outside_equivalence'] for r in comparisons),fresh_score_outside_equivalence=sum(r['outside_equivalence'] for r in fresh_scores),
        archived_subpiece_minimum_identity_changes=sum(r['subpiece_identity_changed'] for r in comparisons),fresh_subpiece_minimum_identity_changes=sum(r['subpiece_identity_changed'] for r in fresh_scores),
        archived_selection=archive_selection,fresh_selection=fresh_selection,
        enumerated_float64_coordinate_matches=sum(r['enumerated_coordinate_match'] for r in comparisons),
        enumerated_score_max_difference=max(r['enumerated_score_difference'] for r in comparisons),enumerated_score_outside_equivalence=sum(r['enumerated_outside_equivalence'] for r in comparisons),
        downstream_method='Original direct teacher score held fixed; existing canonical ordering and equivalence semantics. Only the stationary candidate coordinate changes. No interval-selector training.',
        checkpoint=str(checkpoint),checkpoint_sha256=digest,frozen_locator_refs=locator.refs,all_existing_protected_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-start,quick_screen_seconds=screen['elapsed_seconds'],no_recursion=True,no_global_search=True)
    result=json.loads(json.dumps(result,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    if passed:
        manifest=OUT/'candidate_manifest.json';manifest.write_text(json.dumps(result,indent=2));h=ref.sha(manifest)
        manifest.with_suffix('.sha256').write_text(h+'  '+manifest.name+'\n');result['candidate_manifest_sha256']=h
        # This new path supersedes enumeration for subsequent candidate tests;
        # existing validated experiment files/checkpoints remain untouched.
        (OUT/'stationary_candidate_path.json').write_text(json.dumps(dict(status='CANDIDATE_TEST_PATH',
            locator='validate_terminal_candidate_full.FrozenLocator',location_model='train_stationary_terminal_candidate.TerminalCandidate',
            checkpoint=str(checkpoint),sha256=digest,features='train_stationary_terminal_candidate.features',
            inference='frozen locator -> terminal features -> sigmoid fraction -> L+fraction*(R-L)',float64_enumeration=False,
            interval_selection='unchanged deterministic teacher canonical rule'),indent=2))
    if failures:ref.write_csv(OUT/'meaningful_minimum_failures.csv',failures)
    (OUT/'full_results.json').write_text(json.dumps(result,indent=2))
    (OUT/'terminal_candidate_summary.txt').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('quick_screen','frozen_locator_refs')},indent=2),flush=True)


if __name__=='__main__':run()
