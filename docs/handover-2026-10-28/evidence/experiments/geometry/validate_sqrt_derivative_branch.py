"""Frozen primitive assembly only: square-root derivative branch, float64."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.audit_reciprocal_required_domain import Collector
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_orientation_transform import OrientationTransform
from experiments.geometry.train_inner_quadratic_operator import PolynomialArithmetic
from experiments.geometry.train_sqrt_operator import ScalarPower
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower

OUT=ROOT/'sqrt_derivative_branch_assembly'
EPS=np.finfo(np.float64).eps
BLOCK=8192
COMPONENTS={
    'orientation':('orientation_transform_endpoint_refine/orientation_transform_operator_locked.json',OrientationTransform),
    'multiply':('multiplication_operator_smoke/candidate_manifest.json',QuadraticArithmetic),
    'quadratic':('inner_quadratic_operator_smoke/candidate_manifest.json',PolynomialArithmetic),
    'sqrt':('sqrt_operator_smoke/candidate_manifest.json',ScalarPower),
    'reciprocal':('reciprocal_required_domain_audit/candidate_manifest.json',PositivePower),
}


class SqrtDerivativeBranch(nn.Module):
    """Inputs [A,B,a,c,orientation]; only frozen primitives plus weight derivative."""
    def __init__(self,components):
        super().__init__(); self.components=nn.ModuleDict(components)

    def forward(self,inputs,details=False):
        A,B,a,c,o=inputs.unbind(1); models=self.components
        t=models['orientation'](torch.stack((c,o),1))
        w=models['orientation'].weights
        t_prime=w[2]+w[3]*o
        q=models['quadratic'](torch.stack((B,a,t),1))
        r=models['sqrt'](q)
        r_inv=models['reciprocal'](r)
        def multiply(x,y): return models['multiply'](torch.stack((x,y),1))
        a2=multiply(a,a)
        n1=multiply(A,a2)
        n2=multiply(n1,t)
        numerator=multiply(n2,t_prime)
        output=multiply(numerator,r_inv)
        if details: return torch.stack((output,t,t_prime,q,r,r_inv,numerator),1)
        return output


class DerivativeInputs(Collector):
    def __init__(self):
        super().__init__(); self.cache={}; self.parents=[]; self.generic={}

    def read(self,path):
        key=str(path)
        if key not in self.cache: self.cache[key]=super().read(path)
        return self.cache[key]

    def collect(self):
        for role in ('train','heldout'): self.parents+=self.read(ROOT/f'learned_split_proposer_smoke/{role}_regions.json')
        old=self.read(ROOT/'split_quality_smoke/previous_n4_failure.npz')
        self.parents.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
        base=ROOT/'split_subpiece_stationary_solver_smoke'
        paths=sorted(base.glob('stage_*_train.json'))+sorted(base.glob('stage_*_selection.json'))+sorted(base.glob('stage_*_heldout.json'))+[base/'interval_location_holdouts.json']
        for path in paths:
            for i,r in enumerate(self.read(path)): self.generic[path.stem+':'+str(i)]=r
        super().collect()

    def geometry(self,r,axis,k,c,scope,case):
        self.geometry_context=(r,k)
        super().geometry(r,axis,k,c,scope,case)

    def add(self,scope,case,c,a,B,tcode,vcode,order='generic'):
        super().add(scope,case,c,a,B,tcode,vcode,order)
        batch=self.batches[-1]; o=np.broadcast_to(np.asarray(tcode,dtype=np.float64),batch['c'].shape).copy()
        if scope=='archived_generic_subpieces':
            A=np.full(len(c),self.generic[str(case)]['A'])
        else:
            if scope=='archived_subpiece_grids':
                parent,axis,k,piece=case; r=self.parents[parent]
            else: r,k=self.geometry_context
            sides=np.asarray(r['upper'])-r['lower']
            h=float(np.exp((np.log(np.prod(sides))-np.log(r['n']))/r['d']))
            # Reconstruct archived INPUT coefficient A; this is not derivative inference.
            masses=np.where(o==0,k,r['n']-k)
            A=1/(h*np.power(masses.astype(float),1/r['d']))
        batch['A']=A; batch['o']=o


def metrics(y,p):
    err=p-y; finite=np.isfinite(p); normal=abs(y)>=np.finfo(float).tiny
    largest=float(np.max(abs(err)))
    relative=err[normal]/y[normal]
    return dict(cases=len(y),rmse=float(largest*np.sqrt(np.mean((err/largest)**2))) if largest else 0.,
                mae=float(np.mean(abs(err))),max_absolute_error=largest,
                relative_cases=int(normal.sum()),max_relative_error=float(abs(relative).max()) if len(relative) else 0.,
                relative_rmse=float(np.sqrt(np.mean(relative*relative))) if len(relative) else 0.,
                sign_mismatches=int(np.sum(np.sign(y)!=np.sign(p))),nonfinite_outputs=int((~finite).sum()),
                zero_teacher_cases=int(np.sum(y==0)),zero_failures=int(np.sum((y==0)&(p!=0))),
                exact_matches=int(np.sum(y==p)))


def run():
    started=time.monotonic(); torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    components={}; manifest_components={}
    for name,(file,cls) in COMPONENTS.items():
        manifest=json.loads((ROOT/file).read_text()); cp=Path(manifest['checkpoint']); assert sha(cp)==manifest['sha256']
        model=cls(); model.load_state_dict(torch.load(cp,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
        components[name]=model; manifest_components[name]=dict(checkpoint=str(cp),sha256=manifest['sha256'],manifest=str(ROOT/file))
    assembly=SqrtDerivativeBranch(components).eval().requires_grad_(False)
    source=DerivativeInputs(); source.collect()
    batches=source.batches
    data=np.concatenate([np.column_stack((b['A'],np.full(len(b['c']),b['B']),np.full(len(b['c']),b['a']),b['c'],b['o'])) for b in batches])
    archived_count=len(data); rng=np.random.default_rng(27472026)
    n=4096; synth=np.column_stack((10**rng.uniform(-10,10,n),10**rng.uniform(-30,14,n),10**rng.uniform(-8,7,n),rng.uniform(0,1,n),rng.integers(0,2,n)))
    # Explicit endpoints and zero/near-zero numerators, with positive denominators.
    special=[]
    for A in (0.,1e-240,1e-120,1.,1e8):
        for a in (0.,1e-12,1.,1e6):
            for B in (1e-30,1e-8,1.,1e12):
                for c in (0.,np.nextafter(0.,1.),1e-12,.5,np.nextafter(1.,0.),1.):
                    for o in (0.,1.): special.append([A,B,a,c,o])
    synth=np.vstack((synth,np.array(special)))
    st=np.where(synth[:,4]==0,synth[:,3],1-synth[:,3])
    sr=np.sqrt(synth[:,1]+(synth[:,2]*synth[:,2])*(st*st))
    # Keep compact synthetic coverage within previously observed reciprocal input range.
    domain=json.loads((ROOT/'reciprocal_required_domain_audit/candidate_manifest.json').read_text())['validated_domain']['x_sqrt']
    synth=synth[(sr>=domain['minimum'])&(sr<=domain['maximum'])]
    data=np.concatenate((data,synth)); total=len(data)
    print(json.dumps(dict(event='inputs_ready',archived=archived_count,synthetic=len(synth),block_size=BLOCK)),flush=True)
    A,B,a,c,o=data.T; t=np.where(o==0,c,1-c)
    # Teacher orientations below are validation labels only; inference uses learned weights.
    tp=np.where(o==0,1.,-1.)
    q=B+(a*a)*(t*t); r=np.sqrt(q)
    numerator=((A*(a*a))*t)*tp
    teacher=numerator/r
    assert np.all(np.isfinite(teacher)) and np.all(r>0)
    output=np.empty((total,7),dtype=np.float64)
    with torch.no_grad():
        for begin in range(0,total,BLOCK):
            end=min(begin+BLOCK,total)
            output[begin:end]=assembly(torch.from_numpy(data[begin:end]),details=True).numpy()
    pred=output[:,0]
    orientation=[]
    for code in (0,1):
        mask=o==code
        orientation.append(dict(orientation='forward' if code==0 else 'reverse',cases=int(mask.sum()),
            learned_derivative_min=float(output[mask,2].min()),learned_derivative_max=float(output[mask,2].max()),
            expected_derivative=1. if code==0 else -1.,derivative_mismatches=int(np.sum(output[mask,2]!=tp[mask])),
            transform_mismatches=int(np.sum(output[mask,1]!=t[mask]))))
    rows=[dict(group='all',**metrics(teacher,pred)),dict(group='archived',**metrics(teacher[:archived_count],pred[:archived_count])),
          dict(group='compact_synthetic',**metrics(teacher[archived_count:],pred[archived_count:]))]
    for name,mask in [('small_denominator_below_1e-8',r<1e-8),('large_reciprocal_above_1e8',1/r>1e8),
                      ('near_zero_nonzero_numerator', (abs(numerator)>0)&(abs(numerator)<1e-12)),('zero_numerator',numerator==0)]:
        if mask.any(): rows.append(dict(group=name,**metrics(teacher[mask],pred[mask])))
    scopes=sorted(set(b['scope'] for b in batches))
    for scope in scopes:
        ids=np.concatenate([np.arange(b['start'],b['end']) for b in batches if b['scope']==scope])
        rows.append(dict(group=scope,**metrics(teacher[ids],pred[ids])))
    write_csv(OUT/'sqrt_derivative_validation.csv',rows)
    write_csv(OUT/'orientation_derivative_validation.csv',orientation)
    abs_error=abs(pred-teacher); relative=np.zeros(total); nonzero=teacher!=0; relative[nonzero]=abs_error[nonzero]/abs(teacher[nonzero])
    worst=[]; ends=np.array([b['end'] for b in batches])
    for ordering,errors in [('relative',relative),('absolute',abs_error)]:
        ids=np.argsort(-errors,kind='stable')[:20]
        for rank,i in enumerate(ids,1):
            context=batches[int(np.searchsorted(ends,i,side='right'))] if i<archived_count else dict(scope='compact_synthetic',case=str(i-archived_count))
            worst.append(dict(ordering=ordering,rank=rank,scope=context['scope'],case=context['case'],
                A=float(A[i]),B=float(B[i]),a=float(a[i]),c=float(c[i]),orientation=int(o[i]),t=float(output[i,1]),t_prime=float(output[i,2]),
                teacher_q=float(q[i]),learned_q=float(output[i,3]),teacher_denominator=float(r[i]),learned_denominator=float(output[i,4]),
                learned_reciprocal=float(output[i,5]),teacher_numerator=float(numerator[i]),learned_numerator=float(output[i,6]),
                teacher=float(teacher[i]),learned=float(pred[i]),absolute_error=float(abs_error[i]),relative_error=float(relative[i])))
    write_csv(OUT/'sqrt_derivative_worst_cases.csv',worst)
    with np.errstate(divide='ignore',invalid='ignore'):
        qmask=q!=0
        stage=dict(q_max_relative_error=float(np.max(abs((output[qmask,3]-q[qmask])/q[qmask]))),
            sqrt_max_relative_error=float(np.max(abs((output[:,4]-r)/r))),
            reciprocal_max_relative_error_at_actual_learned_input=float(np.max(abs((output[:,5]-1/output[:,4])/(1/output[:,4])))),
            learned_denominator_min=float(output[:,4].min()),learned_denominator_max=float(output[:,4].max()),
            learned_reciprocal_max=float(output[:,5].max()),
            learned_denominators_outside_previous_observed_range=int(np.sum((output[:,4]<domain['minimum'])|(output[:,4]>domain['maximum']))),
            numerator_max_relative_error=float(np.max(abs((output[numerator!=0,6]-numerator[numerator!=0])/numerator[numerator!=0]))),
            denominator_nonpositive=int(np.sum(output[:,4]<=0)),intermediate_nonfinite=int(np.sum(~np.isfinite(output))))
    finite_relative=relative[np.isfinite(relative)]
    overall=rows[0]
    passed=bool(overall['max_relative_error']<=32*EPS and overall['nonfinite_outputs']==0 and overall['sign_mismatches']==0
                and overall['zero_failures']==0 and stage['intermediate_nonfinite']==0 and stage['denominator_nonpositive']==0
                and all(r['derivative_mismatches']==0 and r['transform_mismatches']==0 for r in orientation))
    unchanged=all(sha(Path(p))==h for p,h in protected.items()); assert unchanged
    assert all(sha(Path(p))==h for p,h in source.inputs.items())
    result=dict(operator='sqrt_derivative_branch',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_ONLY',gate_passed=passed,
        metrics=rows,orientation=orientation,intermediates=stage,relative_gate=32*EPS,block_size=BLOCK,
        components=manifest_components,protected_checkpoints=len(protected),all_prior_checkpoint_hashes_unchanged=unchanged,
        data_scope='Archived fixed interval/subpiece grids, saved targets and queries, plus compact synthetic coverage. No new root solving or derivative minimization.',
        no_log_derivative=True,no_full_derivative=True,elapsed_seconds=time.monotonic()-started)
    if passed:
        manifest=dict(operator='sqrt_derivative_branch',status='VALIDATED_ASSEMBLY',components=manifest_components,
            module='experiments.geometry.validate_sqrt_derivative_branch',model='SqrtDerivativeBranch',
            implementation_sha256=sha(Path(__file__)),inputs=['A','B','a','c','orientation'],
            steps=['t=orientation(c,o)','t_prime=orientation.weights[2]+orientation.weights[3]*o',
                'q=quadratic(B,a,t)','r=sqrt(q)','r_inv=reciprocal(r)','a2=multiply(a,a)',
                'n1=multiply(A,a2)','n2=multiply(n1,t)','numerator=multiply(n2,t_prime)','D=multiply(numerator,r_inv)'],
            inference_uses_no_teacher=True,validation=str(OUT/'results.json'))
        path=OUT/'sqrt_derivative_branch_validated.json'; path.write_text(json.dumps(manifest,indent=2))
        digest=sha(path); path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n')
        result.update(assembly_manifest=str(path),assembly_sha256=digest)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[f"sqrt_derivative_branch = {result['status']}",f'Overall: {json.dumps(overall)}',
        f'Orientation derivative extraction: {json.dumps(orientation)}',f'Intermediate checks: {json.dumps(stage)}',
        'Inference follows the prescribed learned operations. Teacher arithmetic is isolated to validation.',
        'Relative error is evaluated on nonzero normal teacher outputs. Exact zero and sign failures are gated separately.',
        'The compact synthetic set includes endpoint, zero and very small numerators with positive denominators.',
        'No checkpoint modification, training, log derivative, full derivative, root finding, minimization, recursion, or search.',
        f'Assembly manifest SHA256: {result.get("assembly_sha256","not promoted")}',
        f'All {len(protected)} prior checkpoint hashes unchanged. Execution: {result["elapsed_seconds"]:.3f} seconds.']
    (OUT/'sqrt_derivative_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
