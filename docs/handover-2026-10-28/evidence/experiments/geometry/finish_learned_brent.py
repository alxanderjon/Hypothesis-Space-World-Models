"""Gated incremental completion. Stop on the first strict validation failure."""
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry import train_division_operator as hashes
from experiments.geometry.learned_brent_completion import LearnedBrent, CategoricalHead
from experiments.geometry import learned_brent_completion as implementation
from experiments.geometry.validate_locked_brent_linear import read, write, bits

ROOT=Path('outputs/sho/stage1B')
OUT=ROOT/'learned_brent_completion'
SEED=928202641


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


class GateFailure(RuntimeError):
    pass


class Run:
    def __init__(self):
        OUT.mkdir(exist_ok=False)
        self.start=time.monotonic();self.stage='A';self.status=[];self.trained=[];self.replays=[]
        self.protected={str(p):hashes.sha(p) for p in ROOT.rglob('*.pt')}
        self.protected.update({str(p):hashes.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('locked','manifest','validated'))})
        self.protected.update({str(p):hashes.sha(p) for p in Path('experiments/geometry').glob('*.py')})
        dump(OUT/'protected_hashes.json',self.protected)
        self.engine=LearnedBrent();self.teacher_cache={};self.last_detail={}
        self.cases=json.loads((ROOT/'exact_branch_switch_brent_audit/solve_inputs.json').read_text())
        self.splits={int(r['case_id']):r['group'] for r in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
        for folder,name in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),
                            ('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),
                            ('subtraction_operator_training','fresh_candidates.json')]:
            for r in json.loads((ROOT/folder/name).read_text()):
                self.cases.append(dict(parent=r['parent'],axis=r['axis'],count=r['k'],left=r['c_left'],right=r['c_right'],teacher_cut=r['teacher_cut'],label=r['case']))
        assert len(self.cases)==875
        self.audit=authority.Audit.__new__(authority.Audit)

    def gate(self,condition,detail):
        if not condition:
            self.last_detail=detail;dump(OUT/'first_failure.json',dict(stage=self.stage,**detail))
            raise GateFailure(detail.get('reason','Strict gate failed'))

    def train(self,name,x,y,outputs):
        torch.manual_seed(SEED+len(self.trained))
        model=CategoricalHead(x.shape[1],outputs)
        initial={k:v.detach().tolist() for k,v in model.state_dict().items()}
        opt=torch.optim.Adam(model.parameters(),lr=.06)
        xt=torch.from_numpy(x);yt=torch.from_numpy(y)
        history=[]
        for update in range(1,2001):
            opt.zero_grad();logits=model.linear(xt)
            loss=nn.functional.binary_cross_entropy_with_logits(logits[:,0],yt.double()) if outputs==1 else nn.functional.cross_entropy(logits,yt)
            loss.backward();opt.step()
            with torch.no_grad():
                pred=model.predict(x);errors=int((pred!=y).sum())
                v=model.linear(xt)
                if outputs==1:margin=float((v[:,0]*(2*yt-1)).min())
                else:
                    good=v.gather(1,yt[:,None])[:,0];v[torch.arange(len(y)),yt]=-torch.inf
                    margin=float((good-v.max(1).values).min())
            history.append(dict(update=update,loss=float(loss.detach()),errors=errors,minimum_logit_margin=margin))
            if errors==0 and margin>=3:break
        self.gate(errors==0,dict(reason='Tiny categorical head failed truth table',component=name,errors=errors))
        model.eval().requires_grad_(False)
        checkpoint=OUT/(name+'_checkpoint.pt')
        torch.save(dict(state_dict=model.state_dict(),initial=initial,seed=SEED+len(self.trained),updates=update,
                        truth_inputs=x,truth_targets=y,optimizer_state=opt.state_dict()),checkpoint)
        model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
        write(OUT/(name+'_training.csv'),history)
        info=dict(component=name,architecture=f'float64 Linear({x.shape[1]},{outputs}) with bias',
                  parameter_count=sum(p.numel() for p in model.parameters()),checkpoint=str(checkpoint),sha256=hashes.sha(checkpoint),
                  initial_parameters=initial,learned_parameters={k:v.detach().tolist() for k,v in model.state_dict().items()},
                  updates=update,truth_table_errors=errors,truth_table_cases=len(y),heldout_fit=False)
        self.trained.append(info);dump(OUT/(name+'_parameters.json'),info)
        return model

    def lock(self,name,previous,metrics,notes):
        manifest=dict(operator=name,status='LOCKED',previous_status=previous,validation=metrics,notes=notes,
                      implementation=str(Path(implementation.__file__)),implementation_sha256=hashes.sha(Path(implementation.__file__)),
                      authority=str(Path(authority.__file__)),authority_sha256=hashes.sha(Path(authority.__file__)),
                      newly_trained=[r for r in self.trained if r['component']==name])
        p=OUT/(name+'_locked.json');dump(p,manifest);p.with_suffix('.sha256').write_text(hashes.sha(p)+'\n')
        self.status.append(dict(component=name,previous_status=previous,new_status='LOCKED',validation=metrics,manifest=str(p)))
        print(json.dumps(dict(stage=self.stage,locked=name,validation=metrics)),flush=True)

    def callback(self,case):
        self.audit.setup(case['parent'],case['axis']);k=case['count']
        def evaluate(c):
            f=self.audit.formula(float(c),k);ia=int(f[4][0])
            return float(f[1][0]),dict(q_active=int(f[3][0]),aspect_active=ia,aspect_power=int(f[5][0,ia]),
                                     competing_q=f[6][0].tolist(),competing_log_aspect=f[7][0].tolist())
        return evaluate

    def replay(self,level,label):
        fn,code=self.engine.build(level);(OUT/(label+'_interpreter.py')).write_text(code)
        rows=[];self.engine.counts={}
        for i,c in enumerate(self.cases):
            evaluate=self.callback(c)
            if i not in self.teacher_cache:
                calls=[]
                def tcall(x):
                    f,b=evaluate(x);calls.append((float(x).hex(),f.hex(),b));return f,b
                root,trace,n,reason=authority.traced_brent(tcall,c['left'],c['right'],i)
                self.gate(root.hex()==float(c['teacher_cut']).hex(),dict(reason='Authority differs from archived root',case_id=i))
                self.teacher_cache[i]=(root,trace,n,reason,calls)
            root,trace,n,reason,calls=self.teacher_cache[i];lc=[]
            def lcall(x):
                f,b=evaluate(x);lc.append((float(x).hex(),f.hex(),b));return f,b
            got,lt,ln,lr=fn(lcall,c['left'],c['right'],i)
            bad=next((j for j,(a,b) in enumerate(zip(calls,lc)) if a!=b),None)
            if bad is None and len(calls)!=len(lc):bad=min(len(calls),len(lc))
            row=dict(case_id=i,split=self.splits.get(i,'additional_fresh'),queries=ln,
                     exact_queries=sum(a[0]==b[0] for a,b in zip(calls,lc)),state_trace_exact=trace==lt,
                     exact_return=got.hex()==root.hex(),first_query_divergence=bad)
            rows.append(row)
            if bad is not None or trace!=lt or got.hex()!=root.hex():
                write(OUT/(label+'_replay.csv'),rows)
                self.gate(False,dict(reason='Brent replay divergence',component=label,case_id=i,parent=c,
                    first_query_divergence=bad,teacher_trace=trace,learned_trace=lt,teacher_calls=calls,learned_calls=lc,
                    teacher_return=root.hex(),learned_return=got.hex()))
            if (i+1)%100==0:print(json.dumps(dict(stage=label,solves=i+1,seconds=round(time.monotonic()-self.start,1))),flush=True)
        write(OUT/(label+'_replay.csv'),rows)
        summaries=[]
        for group in ['all']+sorted({r['split'] for r in rows}):
            rr=[r for r in rows if group=='all' or r['split']==group]
            summaries.append(dict(split=group,solves=len(rr),queries=sum(r['queries'] for r in rr),
                exact_queries=sum(r['exact_queries'] for r in rr),exact_states=sum(r['state_trace_exact'] for r in rr),
                exact_returns=sum(r['exact_return'] for r in rr),divergences=0))
        write(OUT/(label+'_replay_summary.csv'),summaries)
        self.replays.append(dict(stage=label,metrics=summaries,hook_counts=self.engine.counts.copy()))
        print(json.dumps(dict(stage=label,replay='PASS',summary=summaries[0])),flush=True)
        return summaries[0]

    def stage_a(self):
        self.stage='A'
        x=np.array(list(itertools.product([0.,1.],repeat=2)))
        y=np.array([2,1,0,0],dtype=np.int64)
        self.engine.step_head=self.train('brent_step_application',x,y,3)
        metrics=[]
        for name,file in [('saved','step_application_events.csv'),('boundary','boundary_controls.csv')]:
            rows=read(ROOT/'brent_step_application_isolation_audit'/file)
            a={k:np.array([float(r[k]) for r in rows]) for k in ('xcur','scur','sbis','delta')}
            query,applied,action=self.engine.step(*(a[k] for k in ('xcur','scur','sbis','delta')))
            for key,got in [('teacher_next_x',query),('teacher_applied_step',applied)]:
                truth=np.array([float(r[key]) for r in rows]);stat=hashes.stats(truth,got,name+'/'+key);metrics.append(stat)
                self.gate(stat['errors']==0,dict(reason='Step application isolated mismatch',metric=stat))
            target=np.array([{'USE_SCUR':0,'USE_POSITIVE_DELTA':1,'USE_NEGATIVE_DELTA':2}[r['action']] for r in rows])
            self.gate(np.array_equal(action,target),dict(reason='Step action mismatch',dataset=name))
        write(OUT/'step_validation.csv',metrics)
        self.lock('brent_step_application','AUDITED',dict(saved=29062,boundary=6880,truth_table=4,errors=0),
                  '9-parameter categorical head; frozen strict comparisons/addition; operand selection only.')
        self.replay(1,'A_step')

    def stage_b(self):
        self.stage='B'
        text='''Exact state operation order (audit_exact_branch_brent.traced_brent):
At proposal acceptance: spre=old scur; scur=trial. At fallback: spre=sbis; scur=sbis.
Before query application: xpre=old xcur; fpre=old fcur; pre_branches=old cur_branches.
Apply learned step to xcur; callback replaces fcur and cur_branches.
Next loop: reset iff fpre!=0 AND fcur!=0 AND opposite sign bits.
If reset: xblk=xpre; fblk=fpre; blk_branches=pre_branches;
scur=subtract(xcur,xpre); spre=scur.
Then promote iff abs(fblk)<abs(fcur), with equality retaining state.
SEQUENTIAL copies: xpre=xcur; xcur=xblk; xblk=xpre (new xpre),
fpre=fcur; fcur=fblk; fblk=fpre (new fpre), and branch metadata likewise.
Thus this is not a three-way cyclic simultaneous swap. Old xpre/fpre are discarded.
Reset uses the existing frozen AND head twice on nonzero and sign-bit predicates;
zero and signed-zero never reset. Promotion uses frozen subtraction + threshold.
All copies and snapshot storage remain deterministic scaffolding. No new router needed.
'''
        (OUT/'state_update_audit.txt').write_text(text)
        fp=[];fc=[];fb=[];metrics=[]
        for root,tr,n,reason,calls in self.teacher_cache.values():
            for r in tr:
                s=r['entering'];fp.append(s['fpre']);fc.append(s['fcur'])
                s=r['after_refresh'];fb.append(s['fblk'])
        fp=np.array(fp);fc=np.array(fc);fb=np.array(fb)
        reset=self.engine.refresh(fp,fc);truth=(fp!=0)&(fc!=0)&(np.signbit(fp)!=np.signbit(fc))
        promote=self.engine.promote(fb,fc);target=abs(fb)<abs(fc)
        self.gate(np.array_equal(reset,truth) and np.array_equal(promote,target),dict(reason='State predicates mismatch'))
        values=np.array([-1.,-np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.),1.])
        pairs=np.array(list(itertools.product(values,repeat=2)))
        r=self.engine.refresh(pairs[:,0],pairs[:,1]);t=(pairs[:,0]!=0)&(pairs[:,1]!=0)&(np.signbit(pairs[:,0])!=np.signbit(pairs[:,1]))
        self.gate(np.array_equal(r,t),dict(reason='Reset signed-zero boundary mismatch'))
        pairs2=np.array([(a,b) for a in values for b in [a,np.nextafter(a,-np.inf),np.nextafter(a,np.inf)]])
        self.gate(np.array_equal(self.engine.promote(pairs2[:,0],pairs2[:,1]),abs(pairs2[:,0])<abs(pairs2[:,1])),dict(reason='Promote equality boundary mismatch'))
        # Independent complete state transition including sequential copy semantics.
        checked=0
        for cid,(root,tr,n,reason,calls) in self.teacher_cache.items():
            for row in tr:
                s=dict(row['entering']);xp,xc,xb,fp0,fc0,fb0,sp,sc=(s[k] for k in ('xpre','xcur','xblk','fpre','fcur','fblk','spre','scur'))
                ep,ec,eb=s['pre_branches'],s['cur_branches'],s['blk_branches']
                if bool(self.engine.refresh(fp0,fc0)):
                    xb=xp;fb0=fp0;eb=ep;sc=float(self.engine.ops.subtract(xc,xp));sp=sc
                def check(target):
                    actual=dict(xpre=xp,xcur=xc,xblk=xb,fpre=fp0,fcur=fc0,fblk=fb0,spre=sp,scur=sc)
                    self.gate(all(float(actual[k]).hex()==float(target[k]).hex() for k in actual) and
                              (ep,ec,eb)==(target['pre_branches'],target['cur_branches'],target['blk_branches']),
                              dict(reason='Independent state transition mismatch',case_id=cid,iteration=row['iteration'],actual=actual,target=target))
                check(row['after_refresh'])
                if bool(self.engine.promote(fb0,fc0)):
                    xp=xc;ep=ec;xc=xb;ec=eb;xb=xp;eb=ep
                    fp0=fc0;fc0=fb0;fb0=fp0
                check(row['normalized_state']);checked+=1
        self.lock('brent_state_update','NOT_ASSEMBLED',dict(iteration_states=checked,reset_boundary_cases=len(pairs),promotion_boundary_cases=len(pairs2),errors=0),text)
        self.replay(2,'B_state')

    def stage_c(self):
        self.stage='C'
        audit='Initial left zero returns left with priority; otherwise right zero returns right. Loop STOP iff fcur==0 OR abs(sbis)<delta. Width equality CONTINUES. Adjacent floats have no separate rule. maxiter=100 is a fixed failure guard, not learned convergence. Signed zeros compare equal to zero.'
        (OUT/'stopping_audit.txt').write_text(audit)
        x=np.array(list(itertools.product([0.,1.],repeat=2)));y=np.array([0,1,1,1],np.int64)
        self.engine.stop_head=self.train('brent_stopping_operator',x,y,1)
        rows=[r for _,tr,_,_,_ in self.teacher_cache.values() for r in tr]
        fc=np.array([r['normalized_state']['fcur'] for r in rows]);sb=np.array([r['sbis'] for r in rows]);d=np.array([r['delta'] for r in rows])
        z,w,decision=self.engine.stopping(fc,sb,d);truth=(fc==0)|(abs(sb)<d)
        self.gate(np.array_equal(decision,truth) and np.array_equal(z,fc==0) and np.array_equal(w,abs(sb)<d),dict(reason='Stopping saved-state mismatch'))
        vals=np.unique(np.r_[d[::max(1,len(d)//64)],np.ldexp(1.,[-1074,-1022,-52,0,600])])
        controls=np.array([(f,s*mag,tol) for tol in vals for mag in (np.nextafter(tol,0.),tol,np.nextafter(tol,np.inf)) for s in (1.,-1.) for f in (0.,-0.,1.,-1.)])
        f,sb,tol=controls.T;cz,cw,cd=self.engine.stopping(f,sb,tol)
        self.gate(np.array_equal(cd,(f==0)|(abs(sb)<tol)),dict(reason='Stopping boundary mismatch'))
        write(OUT/'stopping_boundary.csv',[dict(fcur=f,sbis=b,delta=t,stop=bool(v)) for (f,b,t),v in zip(controls,cd)])
        self.lock('brent_stopping_operator','NOT_ASSEMBLED',dict(states=len(rows),boundary=len(controls),truth_table=4,errors=0),audit)
        self.replay(3,'C_stopping')

    def stage_d(self):
        self.stage='D'
        audit='Initial fp==0 returns xp, taking priority over fc==0. Initial fc==0 returns xc. Every normal loop stop returns xc, without endpoint comparison, score tie or canonical operation. Exhausting maxiter raises; it returns no coordinate. One bit initial_left_zero selects xp versus xc; loop stops supply False.'
        (OUT/'final_return_audit.txt').write_text(audit)
        self.engine.return_head=self.train('brent_final_return',np.array([[0.],[1.]]),np.array([0,1],np.int64),1)
        controls=[]
        for xp in (0.,-0.,.125,-1.,np.nextafter(0.,1.)):
            for xc in (0.,-0.,.875,1.,np.nextafter(1.,0.)):
                for flag in (False,True):
                    got=self.engine.returned(xp,xc,flag);target=xp if flag else xc
                    self.gate(bits(got)==bits(target),dict(reason='Return signed-zero control failed'))
                    controls.append(dict(xpre=xp,xcur=xc,initial_left_zero=flag,teacher_hex=target.hex(),learned_hex=got.hex()))
        write(OUT/'return_controls.csv',controls)
        self.lock('brent_final_return','NOT_ASSEMBLED',dict(truth_table=2,boundary=len(controls),errors=0),audit)
        summary=self.replay(4,'D_final_return')
        self.stage='E'
        # Complete replay at level 4 above already includes every learned decision.
        self.lock('learned_branch_switch_brent','PARTIAL',summary,
                  'Complete level-4 replay: all proposal, acceptance, step, reset, promotion, stopping and returned-coordinate decisions learned/reused. Copies, sign-bit/abs extraction, fixed constants/loop guard, dynamic geometry callback remain deterministic scaffold. Native computations retained in trace fields are observation-only, not control decisions.')

    def finish_report(self,failure=None):
        unchanged=all(Path(p).exists() and hashes.sha(Path(p))==h for p,h in self.protected.items())
        result=dict(status='STOPPED_AT_FIRST_FAILURE' if failure else 'COMPLETED',failed_stage=self.stage if failure else None,
                    failure=str(failure) if failure else None,components=self.status,newly_trained=self.trained,replays=self.replays,
                    protected_files=len(self.protected),protected_hashes_unchanged=unchanged,elapsed_seconds=time.monotonic()-self.start)
        dump(OUT/'results.json',result)
        lines=['# Consolidated learned Brent completion',f"Status: {result['status']}. " + (f'Stopped in {self.stage}: {failure}' if failure else 'All requested gates completed.'),'',
               '| Component | Previous status | New status | Exact validation |','|---|---|---|---|']
        for r in self.status:lines.append(f"| {r['component']} | {r['previous_status']} | {r['new_status']} | {json.dumps(r['validation'])} |")
        lines+=['','## Newly trained heads',json.dumps(self.trained,indent=2),'','## Full replay results',json.dumps(self.replays,indent=2),'',
                f'Protected artifacts unchanged: {unchanged} ({len(self.protected)} files).',
                'All checkpoints and individual stage audits are in this directory. No previously locked component was retrained.',
                'Function callbacks dynamically reselect geometric branches using the existing audited geometry scaffold. Learning that geometry computation is outside this run.']
        if failure:lines+=['',f'Later gates after {self.stage} were NOT RUN. No failure was tolerated or repaired downstream.', 'See first_failure.json for exact inputs and traces.']
        (OUT/'consolidated_report.md').write_text('\n'.join(lines)+'\n')
        assert unchanged
        return result


def main():
    torch.set_num_threads(1);r=Run()
    try:
        r.stage_a();r.stage_b();r.stage_c();r.stage_d()
        from experiments.geometry.finish_brent_integration import integrate
        integrate(r)
    except Exception as exc:
        if not (OUT/'first_failure.json').exists():dump(OUT/'first_failure.json',dict(stage=r.stage,reason=repr(exc)))
        r.finish_report(exc);raise
    r.finish_report()


if __name__=='__main__':main()
