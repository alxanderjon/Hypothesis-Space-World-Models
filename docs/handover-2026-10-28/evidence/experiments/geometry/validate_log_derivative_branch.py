"""Assemble frozen beta*v'/v primitives only; no full derivative or training."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.geometry.audit_reciprocal_required_domain import Collector
from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_orientation_transform import OrientationTransform
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_reciprocal_operator import PositivePower

OUT=ROOT/'log_derivative_branch_assembly'
EPS=np.finfo(np.float64).eps
BLOCK=16384
COMPONENTS={
    'orientation':('orientation_transform_endpoint_refine/orientation_transform_operator_locked.json',OrientationTransform),
    'reciprocal':('reciprocal_required_domain_audit/candidate_manifest.json',PositivePower),
    'multiply':('multiplication_operator_smoke/candidate_manifest.json',QuadraticArithmetic),
}


class LogDerivativeBranch(nn.Module):
    """Raw [beta,c,orientation] -> learned beta*v'/v, with v' from locked weights."""
    def __init__(self,components):
        super().__init__(); self.components=nn.ModuleDict(components)

    def forward(self,inputs,details=False):
        beta,c,o=inputs.unbind(1); models=self.components
        v=models['orientation'](torch.stack((c,o),1))
        w=models['orientation'].weights
        v_prime=w[2]+w[3]*o
        v_inv=models['reciprocal'](v)
        n1=models['multiply'](torch.stack((beta,v_prime),1))
        output=models['multiply'](torch.stack((n1,v_inv),1))
        if details: return torch.stack((output,v,v_prime,v_inv,n1),1)
        return output


class LogInputs(Collector):
    def __init__(self):
        super().__init__(); self.cache={}; self.generic={}; self.piece_sign={}

    def read(self,path):
        key=str(path)
        if key not in self.cache: self.cache[key]=super().read(path)
        return self.cache[key]

    def collect(self):
        for r in self.read(ROOT/'fixed_allocation_interval_audit/fixed_interval_score_structure.csv'):
            key=tuple(int(r[k]) for k in ('parent','axis','interval','piece'))
            self.piece_sign[key]=int(r['aspect_log_power'])
        base=ROOT/'split_subpiece_stationary_solver_smoke'
        paths=sorted(base.glob('stage_*_train.json'))+sorted(base.glob('stage_*_selection.json'))+sorted(base.glob('stage_*_heldout.json'))+[base/'interval_location_holdouts.json']
        for path in paths:
            for i,r in enumerate(self.read(path)): self.generic[path.stem+':'+str(i)]=r
        super().collect()

    def geometry(self,r,axis,k,c,scope,case):
        self.geometry_context=(r,axis)
        super().geometry(r,axis,k,c,scope,case)

    def add(self,scope,case,c,a,B,tcode,vcode,order='generic'):
        super().add(scope,case,c,a,B,tcode,vcode,order)
        batch=self.batches[-1]; o=np.broadcast_to(np.asarray(vcode,dtype=np.float64),batch['c'].shape).copy()
        if scope=='archived_generic_subpieces':
            beta=np.full(len(c),self.generic[str(case)]['beta'])
        elif scope=='archived_subpiece_grids':
            beta=np.full(len(c),.05*self.piece_sign[tuple(case)])
        else:
            # Reconstruct archived coefficient beta from active aspect geometry;
            # none of these teacher comparisons occur in branch inference.
            r,axis=self.geometry_context; sides=np.asarray(r['upper'])-r['lower']; other=np.delete(sides,axis)
            if len(other):
                side=a*batch['v']
                power=np.where(side>other.max(),1.,np.where(side<other.min(),-1.,0.))
                beta=.05*power
            else: beta=np.zeros(len(c))
        batch['beta']=beta; batch['o']=o


def metrics(y,p):
    err=p-y; normal=abs(y)>=np.finfo(float).tiny; relative=err[normal]/y[normal]
    largest=float(np.max(abs(err)))
    return dict(cases=len(y),rmse=float(largest*np.sqrt(np.mean((err/largest)**2))) if largest else 0.,
        mae=float(np.mean(abs(err))),max_absolute_error=largest,relative_cases=int(normal.sum()),
        max_relative_error=float(abs(relative).max()) if len(relative) else 0.,
        relative_rmse=float(np.sqrt(np.mean(relative*relative))) if len(relative) else 0.,
        sign_mismatches=int(np.sum(np.sign(y)!=np.sign(p))),zero_teacher_cases=int(np.sum(y==0)),
        zero_output_mismatches=int(np.sum((y==0)&(p!=0))),nonfinite_outputs=int(np.sum(~np.isfinite(p))),
        exact_matches=int(np.sum(y==p)))


def run():
    started=time.monotonic(); torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    components={}; refs={}
    for name,(file,cls) in COMPONENTS.items():
        manifest=json.loads((ROOT/file).read_text()); checkpoint=Path(manifest['checkpoint'])
        assert sha(checkpoint)==manifest['sha256']
        model=cls(); model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
        components[name]=model; refs[name]=dict(checkpoint=str(checkpoint),sha256=manifest['sha256'],manifest=str(ROOT/file))
    assembly=LogDerivativeBranch(components).eval().requires_grad_(False)
    source=LogInputs(); source.collect(); batches=source.batches
    data=np.concatenate([np.column_stack((b['beta'],b['c'],b['o'])) for b in batches])
    archived_count=len(data)
    prior=json.loads((ROOT/'reciprocal_required_domain_audit/candidate_manifest.json').read_text())['validated_domain']['x_v']
    vmin=prior['minimum']; rng=np.random.default_rng(27482026)
    n=4096; vv=10**rng.uniform(np.log10(vmin),0,n); orientation=rng.integers(0,2,n)
    beta=10**rng.uniform(-12,8,n)*rng.choice([-1.,1.],n)
    beta[::13]=0.; beta[1::13]=rng.choice([-1.,1.],len(beta[1::13]))*1e-240
    synth=np.column_stack((beta,np.where(orientation==0,vv,1-vv),orientation))
    special=[]
    for b in (0.,-0.,-1e-300,1e-300,-1e-240,1e-240,-1e-120,1e-120,-1e-12,1e-12,-.05,.05,-1.,1.,-1e8,1e8):
        for v in (vmin,np.nextafter(vmin,np.inf),1e-12,1e-8,.5,np.nextafter(1.,0.),1.):
            for o in (0,1): special.append((b,v if o==0 else 1-v,o))
    synth=np.vstack((synth,np.array(special)))
    real_v=np.where(synth[:,2]==0,synth[:,1],1-synth[:,1])
    synth=synth[(real_v>=vmin)&(real_v<=1.)]
    data=np.concatenate((data,synth)); total=len(data)
    print(json.dumps(dict(event='inputs_ready',archived=archived_count,synthetic=len(synth),block_size=BLOCK)),flush=True)
    beta,c,o=data.T
    v=np.where(o==0,c,1-c); vp=np.where(o==0,1.,-1.)  # Teacher labels only.
    numerator=beta*vp; inv=1/v; teacher=numerator/v
    assert np.all(v>0) and np.all(np.isfinite(teacher))
    output=np.empty((total,5),dtype=np.float64)
    with torch.no_grad():
        for begin in range(0,total,BLOCK):
            end=min(begin+BLOCK,total)
            output[begin:end]=assembly(torch.from_numpy(data[begin:end]),details=True).numpy()
    pred=output[:,0]
    orientation_rows=[]
    for code in (0,1):
        mask=o==code
        orientation_rows.append(dict(orientation='forward' if code==0 else 'reverse',cases=int(mask.sum()),
            expected_derivative=1. if code==0 else -1.,learned_min=float(output[mask,2].min()),learned_max=float(output[mask,2].max()),
            derivative_mismatches=int(np.sum(output[mask,2]!=vp[mask])),orientation_value_mismatches=int(np.sum(output[mask,1]!=v[mask]))))
    groups=[('all',np.ones(total,bool)),('archived',np.arange(total)<archived_count),('compact_synthetic',np.arange(total)>=archived_count),
        ('positive_beta',beta>0),('negative_beta',beta<0),('zero_beta',beta==0),
        ('near_zero_nonzero_beta',(abs(beta)>0)&(abs(beta)<1e-12)),('near_one_v',v>.99),
        ('smallest_v_region',v<=max(10*vmin,1e-12))]
    rows=[dict(group=name,**metrics(teacher[mask],pred[mask])) for name,mask in groups if mask.any()]
    for scope in sorted(set(b['scope'] for b in batches)):
        ids=np.concatenate([np.arange(b['start'],b['end']) for b in batches if b['scope']==scope])
        rows.append(dict(group=scope,**metrics(teacher[ids],pred[ids])))
    # Attribute learned-operator error separately from division-vs-reciprocal
    # multiplication rounding, using controls on the exact same operands.
    exact_reciprocal_product=numerator*inv
    learned_reciprocal_product=numerator*output[:,3]
    raw_final_product=output[:,4]*output[:,3]
    attribution=[dict(component='reciprocal_on_v',**metrics(inv,output[:,3])),
        dict(component='first_multiplication_vs_exact_same_operands',**metrics(beta*output[:,2],output[:,4])),
        dict(component='last_multiplication_vs_exact_same_operands',**metrics(raw_final_product,pred)),
        dict(component='division_vs_exact_reciprocal_then_product',**metrics(teacher,exact_reciprocal_product)),
        dict(component='propagated_reciprocal_approximation',**metrics(exact_reciprocal_product,learned_reciprocal_product))]
    write_csv(OUT/'log_derivative_validation.csv',rows)
    write_csv(OUT/'log_derivative_error_attribution.csv',attribution)
    write_csv(OUT/'orientation_derivative_validation.csv',orientation_rows)
    abs_error=abs(pred-teacher); relative=np.zeros(total); nz=teacher!=0; relative[nz]=abs_error[nz]/abs(teacher[nz])
    worst=[]; ends=np.array([b['end'] for b in batches])
    for ordering,error in [('relative',relative),('absolute',abs_error)]:
        for rank,i in enumerate(np.argsort(-error,kind='stable')[:20],1):
            context=batches[int(np.searchsorted(ends,i,side='right'))] if i<archived_count else dict(scope='compact_synthetic',case=str(i-archived_count))
            worst.append(dict(ordering=ordering,rank=rank,scope=context['scope'],case=context['case'],
                beta=float(beta[i]),c=float(c[i]),orientation=int(o[i]),v=float(v[i]),v_prime=float(output[i,2]),
                teacher_reciprocal=float(inv[i]),learned_reciprocal=float(output[i,3]),teacher_numerator=float(numerator[i]),learned_numerator=float(output[i,4]),
                teacher=float(teacher[i]),learned=float(pred[i]),absolute_error=float(abs_error[i]),relative_error=float(relative[i]),
                division_to_reciprocal_product_error=float(exact_reciprocal_product[i]-teacher[i]),
                propagated_reciprocal_error=float(learned_reciprocal_product[i]-exact_reciprocal_product[i]),
                learned_multiplication_error=float(pred[i]-raw_final_product[i])))
    write_csv(OUT/'log_derivative_worst_cases.csv',worst)
    overall=rows[0]
    intermediate_nonfinite=int(np.sum(~np.isfinite(output)))
    passed=bool(overall['max_relative_error']<=32*EPS and overall['sign_mismatches']==0 and overall['zero_output_mismatches']==0
        and overall['nonfinite_outputs']==0 and intermediate_nonfinite==0 and np.all(output[:,1]>0)
        and all(r['derivative_mismatches']==0 and r['orientation_value_mismatches']==0 for r in orientation_rows))
    unchanged=all(sha(Path(p))==h for p,h in protected.items()); assert unchanged
    assert all(sha(Path(p))==h for p,h in source.inputs.items())
    result=dict(operator='log_derivative_branch',status='VALIDATED_ASSEMBLY' if passed else 'DIAGNOSTIC_ONLY',gate_passed=passed,
        metrics=rows,orientation=orientation_rows,error_attribution=attribution,v_min=float(v.min()),v_max=float(v.max()),
        learned_reciprocal_max=float(output[:,3].max()),intermediate_nonfinite=intermediate_nonfinite,
        relative_gate=32*EPS,block_size=BLOCK,components=refs,protected_checkpoints=len(protected),all_prior_checkpoint_hashes_unchanged=unchanged,
        data_scope='All persisted fixed-interval/subpiece inputs, saved cuts and queries, reproducible archived grids, plus compact synthetic coverage.',
        no_full_derivative=True,no_training=True,elapsed_seconds=time.monotonic()-started)
    if passed:
        manifest=dict(operator='log_derivative_branch',status='VALIDATED_ASSEMBLY',components=refs,
            module='experiments.geometry.validate_log_derivative_branch',model='LogDerivativeBranch',implementation_sha256=sha(Path(__file__)),
            inputs=['beta','c','orientation'],steps=['v=orientation(c,o)','v_prime=orientation.weights[2]+orientation.weights[3]*o',
                'v_inv=reciprocal(v)','n1=multiply(beta,v_prime)','D=multiply(n1,v_inv)'],
            inference_uses_no_teacher=True,validation=str(OUT/'results.json'))
        path=OUT/'log_derivative_branch_validated.json'; path.write_text(json.dumps(manifest,indent=2))
        digest=sha(path); path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n')
        result.update(assembly_manifest=str(path),assembly_sha256=digest)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(source.inputs,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[f"log_derivative_branch = {result['status']}",f'Overall: {json.dumps(overall)}',
        f'v derivative from learned orientation weights: {json.dumps(orientation_rows)}',f'Error attribution: {json.dumps(attribution)}',
        'Inputs include positive, negative and zero beta, near-zero beta, smallest archived v, near-one v and v=1.',
        'The two learned multiplication calls are compared separately against native multiplication on identical operands.',
        'Division versus reciprocal-then-multiply rounding is a distinct control; it is not attributed to learned multiplication error.',
        'Teacher arithmetic appears only in archive reconstruction and evaluation; inference uses only the three frozen operators and the exact derivative of learned orientation weights.',
        'No full derivative, training, root finding, interval minimization, recursion or search.',
        f'Assembly SHA256: {result.get("assembly_sha256","not promoted")}',
        f'All {len(protected)} prior checkpoints unchanged. Execution: {result["elapsed_seconds"]:.3f} seconds.']
    (OUT/'log_derivative_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
