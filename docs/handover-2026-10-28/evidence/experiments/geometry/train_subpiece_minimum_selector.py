"""Only learn canonical subpiece candidate routing; all arithmetic frozen."""
import csv
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.geometry import refine_sqrt_failure_cases as ref
from experiments.geometry.validate_terminal_candidate_full import original_parents, generic_score
from experiments.geometry.train_stationary_terminal_candidate import TerminalCandidate
from experiments.geometry.train_mantissa_log import MonotoneMantissaLog
from experiments.geometry.train_exponent_log2_operator import ExponentLog2
from experiments.sho.train_finite_split_score_smoke import load_frozen,FiniteSplitScore,upstream,predict
from experiments.sho.smoke_artifacts import save_checkpoint

ROOT=ref.ROOT;SOURCE=ROOT/'stationary_terminal_candidate_training';OUT=ROOT/'subpiece_minimum_selector_training'
SEED=28322026
NAMES=['SELECT_LEFT','SELECT_RIGHT','SELECT_STATIONARY'];PRIORITY=[0,2,1]


def canonical(rows,field='teacher_score',cut='cut'):
    ordered=sorted(rows,key=lambda r:(int(r['axis']),r[cut],r['identity']))
    best=min(r[field] for r in ordered)
    return next(r for r in ordered if r[field]<=best+1e-12*max(1.,best))


class SubpieceSelector(nn.Module):
    """Raw interface: normalized scores + validity only.

    Frozen comparisons build the best value and equivalence eligibility; only
    the final three-way canonical routing is newly learned. No argmin over
    scores or hand-coded first-eligible selection occurs in inference.
    """
    def __init__(self,execution):
        super().__init__();self.route=nn.Linear(3,3,dtype=torch.float64);self.execution=execution

    def less(self,a,b):
        # Same frozen margin/threshold, positive homogeneous normalization also
        # permits signed generic subpiece scores. Exact input ties are bookkeeping.
        a=np.asarray(a);b=np.asarray(b);scale=np.maximum(abs(a),abs(b));active=a!=b
        result=np.zeros(a.shape,bool)
        if active.any():
            s=scale[active];w=self.execution.w
            margin=w[0]*(a[active]/s)+w[1]*(b[active]/s)
            result[active]=self.execution.alpha*(margin-self.execution.theta)>0
        return result  # b < a

    def minimum_value(self,scores,valid):
        assert np.all(valid.any(1))
        best=np.zeros(len(scores));have=np.zeros(len(scores),bool)
        for j in range(3):
            take=valid[:,j]&(~have | self.less(best,scores[:,j]))
            best[take]=scores[take,j];have|=valid[:,j]
        return best

    def normalize(self,scores,valid):
        # Binary scaling preserves float64 comparison/bound evaluation order.
        # When best < 1, retain unit scale: max(1,best) is an absolute tolerance
        # floor and cannot be removed by a scale-invariant representation.
        best=self.minimum_value(scores,valid);exponent=np.zeros(len(scores),int)
        high=best>=1
        exponent[high]=np.frexp(best[high])[1]-1
        return np.ldexp(scores,-exponent[:,None])

    def eligibility(self,normalized_scores,valid):
        best=self.minimum_value(normalized_scores,valid)
        bound=best+1e-12*np.maximum(1.,best)
        return valid & ~self.less(normalized_scores,np.broadcast_to(bound[:,None],normalized_scores.shape))

    def select(self,normalized_scores,valid):
        eligible=self.eligibility(normalized_scores,valid)
        assert np.all(eligible.any(1))
        with torch.no_grad():return self.route(torch.from_numpy(eligible.astype(float))).argmax(1).numpy()


def teacher_select(scores,valid):
    # Label/audit only; not called by learned selector actuation.
    masked=np.where(valid,scores,np.inf);best=masked.min(1);eligible=valid&(scores<=best[:,None]+1e-12*np.maximum(1.,best[:,None]))
    out=np.full(len(scores),-1)
    for j in PRIORITY:out[(out<0)&eligible[:,j]]=j
    assert np.all(out>=0);return out


def metrics(scores,valid,pred,labels=None):
    truth=teacher_select(scores,valid) if labels is None else labels
    bad=pred!=truth;best=np.where(valid,scores,np.inf).min(1);chosen=scores[np.arange(len(scores)),pred]
    meaningful=chosen>best+1e-12*np.maximum(1.,best)
    ties=np.zeros(len(scores),bool);ulp=np.zeros(len(scores),bool)
    for a,b in itertools.combinations(range(3),2):
        ok=valid[:,a]&valid[:,b];ties|=ok&(scores[:,a]==scores[:,b])
        ulp|=ok&((np.nextafter(scores[:,a],np.inf)==scores[:,b])|(np.nextafter(scores[:,b],np.inf)==scores[:,a]))
    return dict(examples=len(scores),selection_errors=int(bad.sum()),meaningful_errors=int(meaningful.sum()),
        invalid_candidate_selections=int(np.sum(~valid[np.arange(len(scores)),pred])),
        exact_tie_examples=int(ties.sum()),exact_tie_errors=int(np.sum(bad&ties)),
        one_ULP_examples=int(ulp.sum()),one_ULP_errors=int(np.sum(bad&ulp)),
        **{name+'_errors':int(np.sum(bad&(truth==i))) for i,name in enumerate(NAMES)})


def synthetics(seed,n=16000):
    rng=np.random.default_rng(seed);scores=10**rng.uniform(-10,10,(n,3));valid=rng.random((n,3))>.25;valid[~valid.any(1),0]=True
    groups=np.full(n,'ordinary',object)
    for i in range(n):
        mode=i%5
        if mode==1:scores[i]=scores[i,0];groups[i]='exact_tie'
        elif mode==2:
            b=scores[i,0];scores[i]=[b,np.nextafter(b,np.inf),np.nextafter(b,0.)];groups[i]='one_ULP'
        elif mode==3:
            b=scores[i,0];scores[i]=b+1e-12*max(1,b)*rng.uniform(-2,2,3);groups[i]='near_equivalence'
        elif mode==4:scores[i]*=rng.choice([-1.,1.],3);groups[i]='signed_generic'
    return scores,valid,groups


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):ref.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):ref.sha(p) for p in ROOT.rglob('*.json') if any(k in p.name for k in ('locked','validated','manifest'))})
    terminal_meta=json.loads((SOURCE/'candidate_manifest.json').read_text());assert terminal_meta['status']=='FROZEN_CANDIDATE'
    assert ref.sha(Path(terminal_meta['checkpoint']))==terminal_meta['checkpoint_sha256']
    terminal=TerminalCandidate();terminal.load_state_dict(torch.load(terminal_meta['checkpoint'],weights_only=False)['state_dict']);terminal.eval().requires_grad_(False)
    terminal_data=np.load(SOURCE/'archive_terminal_features.npz')
    terminal_cut=terminal.locate(terminal_data['L'],terminal_data['R'],terminal_data['features'])
    roots=list(csv.DictReader((ROOT/'defer_aware_stationary_locator_audit/defer_aware_root_recovery.csv').open()))
    rootmap={(r['scope'],r['case']):i for i,r in enumerate(roots)}
    models,execution,score_refs=load_frozen();selector=SubpieceSelector(execution)
    fm=json.loads((ROOT/'finite_split_score_smoke/candidate_manifest.json').read_text());finite=FiniteSplitScore()
    assert ref.sha(Path(fm['checkpoint']))==fm['sha256']
    finite.load_state_dict(torch.load(fm['checkpoint'],weights_only=False)['state_dict']);finite.eval().requires_grad_(False)
    parents=original_parents();freshparents={int(k):v for k,v in json.loads((SOURCE/'fresh_parents.json').read_text()).items()}
    entries=[];legacy=0
    for group,file,parent_data in [('archive_geometry','archived_candidate_sets.csv',parents),('fresh_geometry','fresh_candidate_sets.csv',freshparents)]:
        local=[]
        for r in csv.DictReader((SOURCE/file).open()):
            e=dict(group=group,parent=int(r['parent']),axis=int(r['axis']),k=int(r['k']),case=r['case'],identity=r['identity'],kind=r['kind'],cut=float(r['learned_cut']),teacher_score=float(r['learned_score']))
            if group=='archive_geometry' and e['kind']=='stationary':
                assert e['cut']==terminal_cut[rootmap[('geometric',e['case'])]]
            e['dimension']=parent_data[e['parent']]['d'];local.append(e)
        for d in sorted(set(e['dimension'] for e in local)):
            rows=[e for e in local if e['dimension']==d]
            sides=np.array([np.array(parent_data[e['parent']]['upper'])-parent_data[e['parent']]['lower'] for e in rows])
            budgets=np.array([parent_data[e['parent']]['n'] for e in rows]);counts=np.array([[e['k'],n-e['k']] for e,n in zip(rows,budgets)])
            children=np.tile(sides[:,None,:],(1,2,1));ii=np.arange(len(rows));axes=np.array([e['axis'] for e in rows]);cuts=np.array([e['cut'] for e in rows])
            children[ii,0,axes]=sides[ii,axes]*cuts;children[ii,1,axes]=sides[ii,axes]*(1-cuts)
            scores=predict(finite,upstream(models,execution,sides,budgets,children,counts))
            for e,s in zip(rows,scores):e['score']=float(s)
        entries.extend(local)
    # Generic archived single-stationary subpieces and monotone fixtures use the
    # same previously assembled generic learned score, including signed cases.
    base,_=ref.load_terms();parts=base.components
    for name,folder,cls in [('sqrt','sqrt_operator_failure_refine',ref.ScalarPower),('mantissa','mantissa_log_operator_support_refine',MonotoneMantissaLog),('exponent','exponent_log2_operator_smoke',ExponentLog2)]:
        m=json.loads((ROOT/folder/'candidate_manifest.json').read_text());cp=Path(m['checkpoint']);assert ref.sha(cp)==m['sha256']
        net=cls();net.load_state_dict(torch.load(cp,weights_only=False)['state_dict']);parts[name]=net.eval().requires_grad_(False)
    def learned_generic(a,c):
        z=torch.from_numpy(np.column_stack((a[:,:4],c,a[:,4:])));A,B,aa,beta,cc,ot,ov,C=z.unbind(1)
        with torch.no_grad():
            def mul(x,y):return parts['multiply'](torch.stack((x,y),1))
            def add(x,y):return parts['addition'](torch.stack((x,y),1))
            t=parts['orientation'](torch.stack((cc,ot),1));v=parts['orientation'](torch.stack((cc,ov),1))
            q=parts['quadratic'](torch.stack((B,aa,t),1));first=mul(A,parts['sqrt'](q))
            mant,ex=np.frexp(v.numpy());mant*=2;ex-=1
            logv=add(parts['mantissa'](torch.from_numpy(mant)),parts['exponent'](torch.from_numpy(ex.astype(float))))
            return add(add(first,mul(beta,logv)),C).numpy()
    generic=[];excluded=[];folder=ROOT/'split_subpiece_stationary_solver_smoke'
    paths=sorted(folder.glob('stage_*_train.json'))+sorted(folder.glob('stage_*_selection.json'))+sorted(folder.glob('stage_*_heldout.json'))+[folder/'interval_location_holdouts.json']
    for path in paths:
        for i,r in enumerate(json.loads(path.read_text())):
            case=path.stem+':'+str(i)
            if len(r['roots'])>1:
                excluded.append(dict(case=case,reason='Generic interval contains two stationary roots; not a single-stationary fixed-active subpiece',roots=json.dumps(r['roots'])));continue
            cs=[(r['lower'],'left'),(r['upper'],'right')]
            if r['roots']:cs.append((float(terminal_cut[rootmap[('generic_coefficients',case)]]),'stationary'))
            a=np.array([r['A'],r['B'],r['a'],r['beta'],r['t'],r['v'],r.get('C',0.)])
            for c,kind in cs:generic.append(dict(group='archive_generic',parent=-1,axis=0,k=0,case=case,identity=case+':'+kind,kind=kind,cut=float(c),coeff=a,teacher_score=float(generic_score(a,c))))
    fresh_data=np.load(SOURCE/'fresh_terminal_brackets.npz');freshrows=list(csv.DictReader((SOURCE/'fresh_minimum_quality.csv').open()))
    for i,r in enumerate(freshrows):
        if r['scope']!='fresh_coefficients':continue
        a=fresh_data['coeff'][i];cstar=terminal.locate(fresh_data['L'][i:i+1],fresh_data['R'][i:i+1],fresh_data['features'][i:i+1])[0]
        assert cstar==float(r['c_hat'])
        for c,kind in [(fresh_data['initial_L'][i],'left'),(fresh_data['initial_R'][i],'right'),(cstar,'stationary')]:
            generic.append(dict(group='fresh_generic',parent=-1,axis=0,k=0,case=r['case'],identity=r['case']+':'+kind,kind=kind,cut=float(c),coeff=a,teacher_score=float(generic_score(a,c))))
    for start in range(0,len(generic),8192):
        chunk=generic[start:start+8192];ss=learned_generic(np.array([e['coeff'] for e in chunk]),np.array([e['cut'] for e in chunk]))
        for e,s in zip(chunk,ss):e['score']=float(s);e.pop('coeff')
    entries.extend(generic);pools=defaultdict(list)
    for e in entries:pools[(e['group'],e['case'])].append(e)
    sets=[]
    for (group,case),rows in pools.items():
        rows=sorted(rows,key=lambda e:(e['cut'],e['identity']));assert len(rows) in (2,3)
        assert len(set(e['cut'] for e in rows))==len(rows)
        slots=[rows[0],rows[-1],rows[1] if len(rows)==3 else None]
        if group=='archive_geometry' and len(rows)==3 and rows[1]['kind'].startswith('saved'):legacy+=1
        sets.append(dict(group=group,case=case,slots=slots,teacher=canonical(rows)))
    ss=np.array([[e['score'] if e else 0 for e in r['slots']] for r in sets]);valid=np.array([[e is not None for e in r['slots']] for r in sets])
    ty=np.array([[e['teacher_score'] if e else 0 for e in r['slots']] for r in sets])
    labels=np.array([next(j for j,e in enumerate(r['slots']) if e is r['teacher']) for r in sets])
    assert np.isfinite(ss).all()
    ns=selector.normalize(ss,valid);eligible=selector.eligibility(ns,valid)
    audit=dict(candidate_order='Canonical order (axis, cut, identity); strict interiors imply LEFT, STATIONARY, RIGHT regardless of insertion order.',
        candidate_validity='Endpoints always valid in nonempty subpieces; third interior candidate only when supplied by the existing stationary machinery. No coordinate/score training.',
        duplicate_handling='Saved-root subpieces deduplicate exact cuts by sorted(set(...)); bracketed subpieces use endpoints plus terminal candidate. No remaining within-subpiece duplicate cuts in this pool. Shared endpoints across subpieces remain separate identities.',
        exact_ties='First ascending cut, then identity lexicographic; not arbitrary argmin witness.',
        equivalence='Find strict best, then score <= best + 1e-12*max(1,best). Select first eligible candidate in canonical order.',
        normalization='Use frozen minimum to choose a power-of-two divisor when best >= 1; otherwise divisor 1 preserves the absolute teacher tolerance floor. A wholly scale-invariant representation would erase required information for small/negative generic scores.',
        model='Externally receives only normalized learned scores and validity. Internally reuses frozen margin/threshold for value reduction and eligibility. New random float64 Linear(3,3) learns final routing from eligibility; no hard-coded winner/argmin after training.',
        legacy_saved_interior_roots=legacy,excluded_generic_multi_root_intervals=len(excluded),
        exclusion_reason='These 46 general-purpose intervals contain two stationary points; they are not three-candidate subpieces. No interval splitting or interval-level selector is learned here.',
        archive_geometry_sets=sum(r['group']=='archive_geometry' for r in sets),archive_generic_sets=sum(r['group']=='archive_generic' for r in sets),
        fresh_geometry_sets=sum(r['group']=='fresh_geometry' for r in sets),fresh_generic_sets=sum(r['group']=='fresh_generic' for r in sets),
        source_hashes={str(p):ref.sha(p) for p in [Path('experiments/geometry/validate_terminal_candidate_full.py'),Path('experiments/geometry/validate_fixed_allocation_interval_minimizer.py')]})
    (OUT/'subpiece_teacher_audit.json').write_text(json.dumps(audit,indent=2));(OUT/'subpiece_teacher_audit.txt').write_text('\n\n'.join(k+': '+str(v) for k,v in audit.items()))
    ref.write_csv(OUT/'excluded_multi_root_intervals.csv',excluded)
    ref.write_csv(OUT/'frozen_candidate_scores.csv',entries)
    # Complete valid/eligible finite supervision and score-derived supervised data.
    table=[];tablevalid=[];tablelabels=[]
    for v in itertools.product((False,True),repeat=3):
        for e in itertools.product((False,True),repeat=3):
            if any(e) and all(not e[j] or v[j] for j in range(3)):
                table.append(e);tablevalid.append(v);tablelabels.append(next(j for j in PRIORITY if e[j]))
    table=np.array(table);tablelabels=np.array(tablelabels)
    train_syn,train_valid,_=synthetics(SEED+1);train_norm=selector.normalize(train_syn,train_valid)
    train_elig=selector.eligibility(train_norm,train_valid);train_y=teacher_select(train_syn,train_valid)
    archive_mask=np.array([r['group'].startswith('archive') for r in sets]);hold=np.array([i%5==0 for i in range(len(sets))])
    training=archive_mask&~hold
    x=np.concatenate((table,eligible[training],train_elig)).astype(float);y=np.concatenate((tablelabels,labels[training],train_y))
    # Loss aggregation by identical binary features/labels preserves every example.
    unique,inverse=np.unique(np.column_stack((x,y)),axis=0,return_inverse=True);count=np.bincount(inverse).astype(float)
    weights=count/count.sum();weights=.5*weights+.5/len(unique)
    xt=torch.from_numpy(unique[:,:3]);yt=torch.from_numpy(unique[:,3].astype(int));wt=torch.from_numpy(weights)
    initial={k:v.detach().tolist() for k,v in selector.route.state_dict().items()}
    opt=torch.optim.Adam(selector.route.parameters(),lr=.06);history=[]
    for step in range(1,2001):
        opt.zero_grad();logits=selector.route(xt);loss=(nn.functional.cross_entropy(logits,yt,reduction='none')*wt).sum();loss.backward();opt.step()
        with torch.no_grad():
            logits=selector.route(xt);val=float((nn.functional.cross_entropy(logits,yt,reduction='none')*wt).sum());err=int((logits.argmax(1)!=yt).sum())
        if step%10==0:history.append(dict(update=step,loss=val,feature_label_errors=err))
        if err==0 and val<.01:break
    selector.eval().requires_grad_(False);cp=OUT/'subpiece_selector_before_evaluation.pt'
    digest=save_checkpoint(cp,dict(state_dict=selector.route.state_dict(),optimizer_state=opt.state_dict(),initial_parameters=initial,seed=SEED,updates=step,
        architecture='Frozen scalar comparisons + equivalence eligibility -> learned Linear(3,3) -> categorical candidate',
        external_inputs='normalized learned scores L,R,S and validity flags',actions=NAMES,normalization=audit['normalization'],
        frozen_margin_weights=execution.w.tolist(),frozen_threshold_theta=execution.theta,frozen_threshold_alpha=execution.alpha))
    ref.write_csv(OUT/'training_history.csv',history)
    print(json.dumps(dict(stage='saved_before_evaluation',updates=step,loss=val,sets=len(sets),sha256=digest)),flush=True)
    pred=selector.select(ns,valid)
    with torch.no_grad():tp=selector.route(torch.from_numpy(table.astype(float))).argmax(1).numpy()
    freshsyn,freshvalid,kinds=synthetics(SEED+2,24000);npred=selector.select(selector.normalize(freshsyn,freshvalid),freshvalid)
    reports={'finite_routing_rule':dict(examples=len(table),selection_errors=int(np.sum(tp!=tablelabels))),
        'archive':metrics(ty[archive_mask],valid[archive_mask],pred[archive_mask],labels[archive_mask]),
        'heldout_archive':metrics(ty[archive_mask&hold],valid[archive_mask&hold],pred[archive_mask&hold],labels[archive_mask&hold]),
        'fresh_supplied_subpieces':metrics(ty[~archive_mask],valid[~archive_mask],pred[~archive_mask],labels[~archive_mask]),
        'fresh_synthetic':metrics(freshsyn,freshvalid,npred)}
    for name in np.unique(kinds):
        ix=kinds==name;reports['synthetic_'+name]=metrics(freshsyn[ix],freshvalid[ix],npred[ix])
    # Reassign score/validity contents to roles and recompute the canonical label.
    # Ties are intentionally role-prioritized, not permutation equivariant.
    ps=[];pv=[]
    for perm in itertools.permutations(range(3)):
        ps.append(freshsyn[:2048,perm]);pv.append(freshvalid[:2048,perm])
    ps=np.concatenate(ps);pv=np.concatenate(pv);pp=selector.select(selector.normalize(ps,pv),pv)
    reports['permutation_style_role_checks']=metrics(ps,pv,pp)
    ref.write_csv(OUT/'subpiece_selector_accuracy.csv',[dict(group=k,**v) for k,v in reports.items()])
    ref.write_csv(OUT/'finite_routing_truth_table.csv',[dict(valid=str(v.tolist()),eligible=str(e.tolist()),teacher=NAMES[t],learned=NAMES[p]) for v,e,t,p in zip(np.array(tablevalid),table,tablelabels,tp)])
    selected=[];details=[]
    for i,r in enumerate(sets):
        e=r['slots'][pred[i]];t=r['teacher'];best=min(x['teacher_score'] for x in r['slots'] if x)
        details.append(dict(group=r['group'],case=r['case'],teacher_identity=t['identity'],learned_identity=e['identity'],teacher_action=NAMES[labels[i]],learned_action=NAMES[pred[i]],
            exact_winner_match=e['identity']==t['identity'],winner_identity_changed=e['identity']!=t['identity'],coordinate_changed=e['cut']!=t['cut'],
            score_penalty=max(0.,e['teacher_score']-best),outside_equivalence=e['teacher_score']>best+1e-12*max(1.,best)))
        selected.append(e)
    ref.write_csv(OUT/'subpiece_selector_replay.csv',details)
    # Only evaluation: unchanged deterministic canonical interval/parent rules.
    downstream=[]
    for group in ('archive_geometry','fresh_geometry'):
        full=defaultdict(list);reduced=defaultdict(list)
        for e in entries:
            if e['group']==group:full[(e['parent'],e['axis'],e['k'])].append(e)
        for e in selected:
            if e['group']==group:reduced[(e['parent'],e['axis'],e['k'])].append(e)
        teacher_parents=defaultdict(list);learned_parents=defaultdict(list)
        for key,pool in full.items():
            t=canonical(pool);l=canonical(reduced[key],'score');best=min(e['teacher_score'] for e in pool)
            outside=l['teacher_score']>best+1e-12*max(1.,best)
            downstream.append(dict(group=group,level='interval',case=str(key),teacher_identity=t['identity'],learned_identity=l['identity'],
                identity_changed=t['identity']!=l['identity'],coordinate_changed=t['cut']!=l['cut'],outside_equivalence=outside,score_penalty=max(0.,l['teacher_score']-best)))
            teacher_parents[key[0]].extend(pool);learned_parents[key[0]].append(l)
        for parent,pool in teacher_parents.items():
            t=canonical(pool);l=canonical(learned_parents[parent],'score');best=min(e['teacher_score'] for e in pool)
            downstream.append(dict(group=group,level='parent',case=str(parent),teacher_identity=t['identity'],learned_identity=l['identity'],
                identity_changed=t['identity']!=l['identity'],coordinate_changed=t['axis']!=l['axis'] or t['cut']!=l['cut'],
                outside_equivalence=l['teacher_score']>best+1e-12*max(1.,best),score_penalty=max(0.,l['teacher_score']-best)))
    ref.write_csv(OUT/'interval_parent_validation.csv',downstream)
    passed=(np.all(tp==tablelabels) and all(v.get('invalid_candidate_selections',0)==0 and v.get('meaningful_errors',0)==0 for v in reports.values())
        and not any(r['outside_equivalence'] for r in details+downstream)
        and not any(r['coordinate_changed'] or r['identity_changed'] for r in downstream if r['level']=='parent'))
    assert all(ref.sha(Path(p))==h for p,h in protected.items())
    summary=dict(operator='subpiece_minimum_selector',status='LOCKED' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,
        architecture=audit['model'],learned_parameters={k:v.tolist() for k,v in selector.route.state_dict().items()},parameter_count=12,updates=step,seed=SEED,
        audit=audit,metrics=reports,archived_subpieces=int(archive_mask.sum()),heldout_archived=int(np.sum(archive_mask&hold)),fresh_subpieces=int((~archive_mask).sum()),
        exact_winner_matches=sum(r['exact_winner_match'] for r in details),winner_identity_changes=sum(r['winner_identity_changed'] for r in details),
        teacher_equivalent_coordinate_differences=sum(r['coordinate_changed'] and not r['outside_equivalence'] for r in details),
        subpiece_score_outside_equivalence=sum(r['outside_equivalence'] for r in details),
        interval_winner_changes=sum(r['identity_changed'] for r in downstream if r['level']=='interval'),
        interval_winner_changes_outside_equivalence=sum(r['identity_changed'] and r['outside_equivalence'] for r in downstream if r['level']=='interval'),
        canonical_best_cut_changes=sum(r['coordinate_changed'] or r['identity_changed'] for r in downstream if r['level']=='parent'),
        parent_score_outside_equivalence=sum(r['outside_equivalence'] for r in downstream if r['level']=='parent'),
        checkpoint=str(cp),checkpoint_sha256=digest,all_existing_protected_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-started,implementation_sha256=ref.sha(Path(__file__)),no_interval_training=True,no_parent_training=True,no_recursion=True,no_global_search=True)
    summary=json.loads(json.dumps(summary,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    if passed:
        lock=OUT/'subpiece_minimum_selector_locked.json';lock.write_text(json.dumps(summary,indent=2));h=ref.sha(lock)
        lock.with_suffix('.sha256').write_text(h+'  '+lock.name+'\n');summary['lock_manifest_sha256']=h
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'subpiece_selector_summary.txt').write_text(json.dumps(summary,indent=2))
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
