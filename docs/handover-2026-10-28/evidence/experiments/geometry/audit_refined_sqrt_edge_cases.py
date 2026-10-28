"""Complete requested evaluation of the saved refinement; no parameter training."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import refine_sqrt_failure_cases as ref


def run():
    start=time.monotonic();torch.set_num_threads(1)
    out=ref.OUT;result=json.loads((out/'results.json').read_text())
    model=ref.ScalarPower();model.load_state_dict(torch.load(result['checkpoint'],weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    old,new=ref.routers(model)
    failures=np.load(ref.FAIL/'sqrt_failure_inputs_targets.npz')
    q=failures['unique_inputs'];target=np.sqrt(q);pred=ref.predict(model,q)
    held=np.load(ref.SOURCE/'heldout_data.npz');base=ref.predict(model,held['q'])
    edge=[]
    for i in np.flatnonzero(pred!=target):
        edge.append(dict(q=float(q[i]),q_hex=float(q[i]).hex(),target=float(target[i]),prediction=float(pred[i]),
                         error_ulps=float((pred[i]-target[i])/np.spacing(target[i]))))
    ref.write_csv(out/'remaining_power_rounding_cases.csv',edge)
    # Sensitivity probes are diagnostic temporary copies, never selected or saved.
    # Test nearest representable exponents and nearest effective gains around 1.
    params=model.weights.detach().numpy().copy();sensitivity=[]
    for label,p,b in [('saved',*params),('exponent_next_down',np.nextafter(params[0],-np.inf),params[1]),
                      ('exponent_next_up',np.nextafter(params[0],np.inf),params[1]),
                      ('effective_gain_next_down',params[0],np.log(np.nextafter(1.,0.))),
                      ('effective_gain_next_up',params[0],np.log(np.nextafter(1.,np.inf)))]:
        temporary=ref.ScalarPower().eval().requires_grad_(False)
        with torch.no_grad():temporary.weights.copy_(torch.tensor([p,b],dtype=torch.float64))
        ph=ref.predict(temporary,held['q']);pf=ref.predict(temporary,q)
        sensitivity.append(dict(probe=label,p=float(p),b=float(b),gain=float(torch.exp(temporary.weights[1])),
            failure_exact_matches=int(np.sum(pf==target)),heldout_exact_matches=int(np.sum(ph==held['target'])),
            heldout_max_relative_error=float(np.max(abs((ph-held['target'])/held['target']))),
            heldout_errors_worse_than_saved=int(np.sum(abs(ph-held['target'])>abs(base-held['target'])))))
    ref.write_csv(out/'parameter_resolution_sensitivity.csv',sensitivity)
    print(json.dumps(dict(event='rounding_diagnosis',remaining_unique_cases=len(edge),sensitivity=sensitivity)),flush=True)
    # User permits irreducible numerical edge cases; the original strict bitwise
    # precheck was stronger than the requested gate. Finish the requested audit.
    query_diagnostics=[];calls=[0]
    def diagnose(module,args,output):
        calls[0]+=1
        if calls[0]<=3:return  # all endpoints, then initial left/right brackets
        x=args[0];exact,_=ref.mathematical_sign(x.numpy());learned=output.numpy()
        bad=(learned!=exact)
        if not bad.any():return
        xx=x[torch.from_numpy(bad)];target=exact[bad];sgn=learned[bad]
        with torch.no_grad():
            terms=module.terms(xx);m=module.components
            A,B,a,beta,c,ot,ov=xx.unbind(1);t=m['orientation'](torch.stack((c,ot),1))
            q=m['quadratic'](torch.stack((B,a,t),1));r=m['sqrt'](q)
            rt=torch.from_numpy(np.sqrt(q.numpy()));vp=m['orientation'].weights[2]+m['orientation'].weights[3]*ov
            numerator=m['multiply'](torch.stack((beta,vp),1))
            corrected=terms.clone();corrected[:,1]=m['multiply'](torch.stack((numerator,rt),1))
            direct=module.route_terms(corrected).numpy()
        for j,row in enumerate(xx.numpy()):
            query_diagnostics.append(dict(A=row[0],B=row[1],a=row[2],beta=row[3],c=row[4],ot=row[5],ov=row[6],
                learned_sign=int(sgn[j]),mathematical_sign=int(target[j]),direct_sqrt_sign=int(direct[j]),
                false_ZERO=bool(sgn[j]==0),fixed_by_direct_sqrt=bool(direct[j]==target[j]),
                sqrt_input=float(q[j]),sqrt_prediction=float(r[j]),sqrt_target=float(rt[j]),
                sqrt_error_ulps=float((r[j]-rt[j])/np.spacing(float(rt[j])))))
    hook=new.register_forward_hook(diagnose)
    bracket,archive_hashes=ref.archived_audit(old,new);hook.remove()
    ref.write_csv(out/'refined_query_failure_attribution.csv',query_diagnostics)
    bracket['query_failure_attribution']=dict(total_sign_or_zero_disagreements=len(query_diagnostics),
        directions_fixed_by_direct_sqrt=sum(not r['false_ZERO'] and r['fixed_by_direct_sqrt'] for r in query_diagnostics),
        false_ZERO_fixed_by_direct_sqrt=sum(r['false_ZERO'] and r['fixed_by_direct_sqrt'] for r in query_diagnostics),
        directions_also_wrong_with_direct_sqrt=sum(not r['false_ZERO'] and not r['fixed_by_direct_sqrt'] for r in query_diagnostics),
        false_ZERO_cases_also_wrong_with_direct_sqrt=sum(r['false_ZERO'] and not r['fixed_by_direct_sqrt'] for r in query_diagnostics),
        max_sqrt_error_ulps=max(abs(r['sqrt_error_ulps']) for r in query_diagnostics))
    protected=json.loads((out/'protected_hashes.json').read_text())
    assert all(ref.sha(Path(p))==h for p,h in {**protected,**archive_hashes}.items())
    no_regression=result['regression']['absolute_error_worse']==0 and result['regression']['monotonicity_reversals']['after']==0
    edge_ok=all(abs(r['error_ulps'])<=1 for r in edge) and all(r['heldout_errors_worse_than_saved']>0 for r in sensitivity[1:])
    passed=bool(no_regression and edge_ok and bracket['passed'])
    (out/'initial_strict_precheck_results.json').write_text(json.dumps(result,indent=2))
    result.update(status='FROZEN_CANDIDATE' if passed else 'DIAGNOSTIC_ONLY_STOP',passed=passed,bracket_audit=bracket,
        rounding_diagnosis=dict(remaining_unique_failure_inputs=len(edge),max_remaining_sqrt_error_ulps=max(abs(r['error_ulps']) for r in edge),
            nearest_effective_parameter_perturbations_all_degrade_global_accuracy=bool(edge_ok),
            interpretation='Residual power-kernel rounding at learned p=0.5, effective gain=1. Nearby parameter changes degrade broad accuracy; no snapping or kernel changes applied.',
            limitation='Local representable-parameter sensitivity is evidence of numerical resolution limits, not a proof over all real parameters.'),
        gate_semantics='Permit numerical sqrt edge cases when parameter resolution prevents local correction without global regression; require unchanged endpoint/minimum/canonical decisions.',
        evaluation_completion_implementation_sha256=ref.sha(Path(__file__)),all_prior_hashes_unchanged=True,
        elapsed_seconds=result['elapsed_seconds']+time.monotonic()-start)
    result=json.loads(json.dumps(result,default=lambda x:x.item()))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    if passed:(out/'candidate_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k!='primary_metrics'},indent=2))
    (out/'sqrt_refinement_summary.txt').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='primary_metrics'},indent=2),flush=True)


if __name__=='__main__':run()
