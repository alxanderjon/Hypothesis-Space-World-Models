"""Promote acceptance, then audit saved fallback events. Never execute Brent."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import train_division_operator as source
from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry import brent_interpolation_acceptance as implementation
from experiments.geometry.brent_interpolation_acceptance import BrentInterpolationAcceptance
from experiments.geometry.validate_locked_brent_linear import read,write,dump,bits

ROOT=Path('outputs/sho/stage1B')
ACCEPT=ROOT/'brent_interpolation_acceptance_assembly'
OUT=ROOT/'brent_fallback_isolation_audit'
PRIOR=ROOT/'brent_acceptance_isolation_audit'
STATE=('xpre','xcur','xblk','fpre','fcur','fblk','spre','scur')


def run():
    torch.set_num_threads(1);ACCEPT.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    protected={str(p):source.sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):source.sha(p) for p in ROOT.rglob('*.json') if any(t in p.name for t in ('locked','manifest','validated'))})
    protected.update({str(p):source.sha(p) for p in Path('experiments/geometry').glob('*.py')})
    protected.update({str(p):source.sha(p) for p in PRIOR.iterdir() if p.is_file()})
    dump(OUT/'protected_hashes.json',protected)
    prior=json.loads((PRIOR/'results.json').read_text());assert prior['all_reuse_checks_passed']
    model=BrentInterpolationAcceptance()
    acceptance=read(PRIOR/'acceptance_decisions.csv')
    ar={k:np.array([float(r[k]) for r in acceptance]) for k in STATE+('stry','delta','sbis')}
    context=model.context(ar['xcur'],ar['xblk'])
    decisions,parts=model.evaluate(ar['stry'],ar['spre'],context['sbis'],context['delta'])
    metrics=[]
    for key,value in dict(**context,**parts).items():
        if value.dtype==bool:
            target=np.array([r[key]=='True' for r in acceptance]);assert np.array_equal(target,value)
        else:
            target=np.array([float(r[key]) for r in acceptance])
            metrics.append(source.stats(target,value,key))
    truth=np.array([r['label']=='ACCEPT' for r in acceptance]);errors=int((decisions!=truth).sum())
    boundaries=np.load(PRIOR/'acceptance_boundary_controls.npz');v=boundaries['triples']
    bp,_=model.compare_limits(v[:,0],v[:,1],v[:,2]);boundary_errors=int((bp!=boundaries['teacher']).sum())
    table=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    with torch.no_grad():tp=model.eligibility.head(torch.from_numpy(table)).argmax(1).numpy()
    assert errors==boundary_errors==0 and np.array_equal(tp,[0,0,0,1]) and all(r['errors']==0 for r in metrics)
    write(ACCEPT/'arithmetic_validation.csv',metrics)
    manifest=dict(operator='brent_interpolation_acceptance',status='LOCKED_ASSEMBLY',
        module='experiments.geometry.brent_interpolation_acceptance',class_name='BrentInterpolationAcceptance',
        implementation_sha256=source.sha(Path(implementation.__file__)),inputs=['stry','spre','sbis','delta'],
        optional_context_inputs=['xcur','xblk','xtol','rtol'],output='ACCEPT boolean; False is REJECT',
        operation_order=['lhs=multiply(2,abs(stry))','lim1=abs(spre)','triple=multiply(3,abs(sbis))','lim2=subtract(triple,delta)',
                         'strict c1=threshold(subtract(lhs,lim1))<0','strict c2=threshold(subtract(lhs,lim2))<0','frozen eligibility.head([c1,c2])'],
        equality='REJECT',arithmetic_dependencies=model.arithmetic.manifests,division_dependency=model.foundation.manifest,
        comparison_dependency=model.division.threshold_meta,AND_head_dependency=model.head_manifest,
        validation_cases=len(acceptance),decision_errors=errors,boundary_cases=len(v),boundary_errors=boundary_errors,
        truth_table_errors=0,new_parameters=0,training=False,new_model_checkpoint=False,
        prior_audit=str(PRIOR/'results.json'),prior_audit_sha256=source.sha(PRIOR/'results.json'))
    cp=ACCEPT/'brent_interpolation_acceptance_locked.json';dump(cp,manifest)
    (ACCEPT/'assembly_checkpoint.sha256').write_text(source.sha(cp)+'\n')
    print(json.dumps(dict(event='acceptance_LOCKED_ASSEMBLY',cases=len(acceptance),boundary_cases=len(v),errors=0)),flush=True)

    # Only saved state extraction. No call to traced_brent or step application.
    splits={int(r['case_id']):r['group'] for r in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    events=[]
    for raw in read(ROOT/'exact_branch_switch_brent_audit/brent_iterations.csv'):
        if raw['stop']=='True' or raw['effective_method']!='BISECTION':continue
        state=json.loads(raw['normalized_state']);post=json.loads(raw['after_proposal'])
        eligible=raw['eligible']=='True';accepted=raw['interpolation_accepted']=='True' if eligible else None
        events.append(dict(case_id=int(raw['case_id']),iteration=int(raw['iteration']),split=splits[int(raw['case_id'])],
            provenance='saved complete trace',**{k:state[k] for k in STATE},
            delta=float(raw['delta']),sbis=float(raw['sbis']),eligible=eligible,accepted=accepted,
            reason='INTERPOLATION_REJECTED' if eligible else 'NOT_ELIGIBLE',
            method=raw['attempt_method'],stry=float(raw['trial_step']) if eligible else None,
            fallback_step=float(raw['fallback_step']),raw_proposal=float(raw['fallback_query']),
            selected_step=float(raw['selected_step']),recorded_spre_after_proposal=post['spre'],recorded_scur_after_proposal=post['scur'],
            later_minimum_step_recorded=raw['minimum_step']=='True',
            comparison_previous_step=raw['previous_step_large']=='True',comparison_residual_decreasing=raw['residual_decreasing']=='True'))
    # Every rejected additional-fresh interpolation is present in the acceptance
    # CSV, with exact pre-decision state and sbis. Materialize teacher fallback
    # labels from the audited rule; this is not a solver state update.
    for r in acceptance:
        if r['split']!='additional_fresh' or r['label']!='REJECT':continue
        s={k:float(r[k]) for k in STATE};sbis=float(r['sbis']);delta=float(r['delta'])
        events.append(dict(case_id=int(r['solve_id']),iteration=int(r['iteration']),split=r['split'],
            provenance='saved acceptance state; raw fallback labels from audited teacher rule',**s,
            delta=delta,sbis=sbis,eligible=True,accepted=False,reason='INTERPOLATION_REJECTED',method=r['interpolation_branch'],stry=float(r['stry']),
            fallback_step=sbis,raw_proposal=s['xcur']+sbis,selected_step=sbis,
            recorded_spre_after_proposal=None,recorded_scur_after_proposal=None,later_minimum_step_recorded=None,
            comparison_previous_step=abs(s['spre'])>delta,comparison_residual_decreasing=abs(s['fcur'])<abs(s['fpre'])))
    # Recover the sole additional-fresh NOT_ELIGIBLE record from saved operands.
    # The previous rejected record identifies unchanged xblk/fblk and the stored
    # historical steps. No query is calculated/applied to obtain the missing state.
    replay=read(ROOT/'brent_inverse_quadratic_assembly_validation/solve_replay.csv')
    missing=[]
    for row in replay:
        if row['group']!='additional_fresh':continue
        cid=int(row['case_id']);executed=int(row['iteration_queries'])
        available={int(r['iteration']) for r in acceptance if int(r['solve_id'])==cid}
        gap=sorted(set(range(1,executed+1))-available)
        assert len(gap)==int(row['eligibility'])-int(row['branch'])
        missing.extend((cid,i) for i in gap)
    assert missing==[(818,41)]
    with np.load(ROOT/'subtraction_operator_training/Brent_subtraction_operands.npz') as data:
        operands=data['operands'];assembly_operands=data['assembly_operands']
    om=read(ROOT/'subtraction_operator_training/operand_metadata.csv');am=read(ROOT/'subtraction_operator_training/assembly_metadata.csv')
    op_map={(int(r['case_id']),int(r['iteration']),r['operation']):operands[i] for i,r in enumerate(om)}
    asm_map={(int(r['case_id']),int(r['iteration']),r['operation']):assembly_operands[i] for i,r in enumerate(am)}
    recovery=[]
    for cid,it in missing:
        prev=next(r for r in acceptance if int(r['solve_id'])==cid and int(r['iteration'])==it-1)
        assert prev['label']=='REJECT'
        xc,xp=asm_map[(cid,it,'dx')];fc,fp=asm_map[(cid,it,'df')]
        xb,xc_check=op_map[(cid,it,'sbis_difference')]
        assert bits(xc)==bits(xc_check) and bits(xp)==bits(float(prev['xcur'])) and bits(fp)==bits(float(prev['fcur']))
        assert bits(xb)==bits(float(prev['xblk'])) and (cid,it,'refresh_step') not in op_map
        state=dict(xpre=float(xp),xcur=float(xc),xblk=float(xb),fpre=float(fp),fcur=float(fc),fblk=float(prev['fblk']),
                   spre=float(prev['sbis']),scur=float(prev['sbis']))
        delta=(authority.XTOL+authority.RTOL*abs(xc))/2;sbis=(xb-xc)/2
        assert not (abs(state['spre'])>delta and abs(fc)<abs(fp))
        events.append(dict(case_id=cid,iteration=it,split='additional_fresh',provenance='saved subtraction operands + preceding saved rejected-proposal state',
            **state,delta=float(delta),sbis=float(sbis),eligible=False,accepted=None,reason='NOT_ELIGIBLE',method='NONE',stry=None,
            fallback_step=float(sbis),raw_proposal=float(xc+sbis),selected_step=float(sbis),
            recorded_spre_after_proposal=None,recorded_scur_after_proposal=None,later_minimum_step_recorded=None,
            comparison_previous_step=abs(state['spre'])>delta,comparison_residual_decreasing=abs(fc)<abs(fp)))
        recovery.append(dict(case_id=cid,iteration=it,state=state,delta=float(delta),sbis=float(sbis),
            evidence=['saved dx/df operands','saved sbis operands','no saved refresh_step at this iteration','same block coordinate as preceding record','preceding rejected proposal supplies both historical step values']))
    dump(OUT/'missing_record_reconstruction.json',recovery)
    events.sort(key=lambda r:(r['case_id'],r['iteration']))
    get=lambda key:np.array([r[key] for r in events],dtype=float)
    xcur=get('xcur');xblk=get('xblk');n=len(events)
    context=model.context(xcur,xblk)
    teacher=dict(abs_xcur=abs(xcur),relative_tolerance=authority.RTOL*abs(xcur),
        tolerance_sum=authority.XTOL+authority.RTOL*abs(xcur),delta=get('delta'),bracket_difference=xblk-xcur,sbis=get('sbis'))
    metrics=[]
    for key,value in context.items():metrics.append(source.stats(teacher[key],value,key))
    # Raw diagnostic proposal only; never apply minimum-step logic or mutate xcur.
    raw=model.add(xcur,context['sbis'])
    metrics.append(source.stats(get('fallback_step'),context['sbis'],'raw_fallback_step'))
    metrics.append(source.stats(get('selected_step'),context['sbis'],'selected_fallback_step'))
    metrics.append(source.stats(get('raw_proposal'),raw,'raw_fallback_proposal_coordinate'))
    archived=np.array([r['case_id']<818 for r in events])
    for key in ('recorded_spre_after_proposal','recorded_scur_after_proposal'):
        metrics.append(source.stats(np.array([r[key] for r in events if r['case_id']<818]),context['sbis'][archived],key+'_read_only_consistency'))
    assert all(r['errors']==0 for r in metrics)
    ei=np.column_stack((get('spre'),get('delta'),get('fcur'),get('fpre')))
    comparisons,elig=model.eligibility.predict(ei)
    eligibility_errors=int((elig!=np.array([r['eligible'] for r in events],int)).sum())
    assert eligibility_errors==0
    comparison_errors=int((comparisons[:,0]!=np.array([r['comparison_previous_step'] for r in events])).sum()+
                          (comparisons[:,1]!=np.array([r['comparison_residual_decreasing'] for r in events])).sum())
    valid=elig.astype(bool);rej,accept_parts=model.evaluate(get('stry')[valid],get('spre')[valid],get('sbis')[valid],get('delta')[valid])
    assert not rej.any() and comparison_errors==0
    # All previously saved solve summaries determine exact fallback counts:
    # noneligible iterations + rejected acceptance records. Verify coverage per solve.
    for sr in replay:
        cid=int(sr['case_id']);ac=[r for r in acceptance if int(r['solve_id'])==cid]
        expected=int(sr['eligibility'])-int(sr['branch'])+sum(r['label']=='REJECT' for r in ac)
        assert sum(r['case_id']==cid for r in events)==expected
    accepted_map={(int(r['solve_id']),int(r['iteration'])):r for r in acceptance}
    dataset=[]
    for i,r in enumerate(events):
        row=dict(r,xtol=authority.XTOL,rtol=authority.RTOL,learned_sbis=float(context['sbis'][i]),learned_raw_proposal=float(raw[i]),
                 sbis_positive=bool(r['sbis']>0),abs_sbis=float(abs(r['sbis'])),
                 later_raw_step_exceeds_delta=bool(abs(r['sbis'])>r['delta']),
                 later_minimum_step_boundary_equal=bool(abs(r['sbis'])==r['delta']))
        row.update({k:float(v[i]) for k,v in context.items() if k not in ('delta','sbis')})
        ac=accepted_map.get((r['case_id'],r['iteration']))
        for key in ('lhs','lim1','lim2','selected_limit','lhs_lt_lim1','lhs_lt_lim2','lim1_lt_lim2'):
            row[key]=ac[key] if ac else None
        numbers={k:v for k,v in row.items() if isinstance(v,(float,np.floating))}
        row['hex_values']=json.dumps({k:float(v).hex() for k,v in numbers.items()})
        dataset.append(row)
    write(OUT/'fallback_events.csv',dataset);write(OUT/'arithmetic_validation.csv',metrics)
    counts=[]
    for group in ['all']+sorted({r['split'] for r in events}):
        rr=[r for r in events if group=='all' or r['split']==group]
        counts.append(dict(split=group,events=len(rr),NOT_ELIGIBLE=sum(r['reason']=='NOT_ELIGIBLE' for r in rr),
            INTERPOLATION_REJECTED=sum(r['reason']=='INTERPOLATION_REJECTED' for r in rr),
            rejected_linear=sum(r['reason']=='INTERPOLATION_REJECTED' and r['method'] in ('LINEAR','LINEAR_INTERPOLATION') for r in rr),
            rejected_IQ=sum(r['reason']=='INTERPOLATION_REJECTED' and r['method'] not in ('LINEAR','LINEAR_INTERPOLATION') for r in rr),
            positive_sbis=sum(r['sbis']>0 for r in rr),negative_sbis=sum(r['sbis']<0 for r in rr),
            raw_step_at_tolerance=sum(abs(r['sbis'])==r['delta'] for r in rr),
            archived_later_minimum_step_recorded=sum(r['later_minimum_step_recorded'] is True for r in rr)))
    write(OUT/'fallback_counts.csv',counts)
    unchanged=all(source.sha(Path(p))==h for p,h in protected.items());assert unchanged
    result=dict(status='FALLBACK_AUDIT_ONLY_COMPLETE',acceptance_status='LOCKED_ASSEMBLY',
        acceptance_manifest=str(cp),acceptance_manifest_sha256=source.sha(cp),source_solves=len(replay),fallback_events=n,
        counts=counts,arithmetic_metrics=metrics,eligibility_errors=eligibility_errors,comparison_errors=comparison_errors,
        rejected_acceptance_errors=int(rej.sum()),first_unresolved_arithmetic=None,new_learned_fallback_primitive_needed=False,
        raw_fallback='sbis=(xblk-xcur)/2; teacher copies sbis into both spre and scur',
        raw_coordinate='xcur+sbis is diagnostic proposal only; no step applied',
        next_scope='minimum-step application and state updates remain outside this audit',
        reconstructed_saved_records=recovery,Brent_executed=False,query_step_application=False,state_update_implementation=False,
        training=False,recursion=False,protected_files=len(protected),protected_hashes_unchanged=unchanged)
    dump(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('arithmetic_metrics','reconstructed_saved_records')},indent=2),flush=True)


if __name__=='__main__':run()
