"""Small isolated adapter validation. No fitting, optimizer, or recovery call."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from hypothesis_space_world_models.models.sho_exact_solution_decoder import _make_mlp_encoder
from experiments.sho.deterministic_full_k_framework import hashes

OUT = Path('outputs/sho/generic_update_network_adapter')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    a, b = a.detach().contiguous(), b.detach().contiguous()
    if a.shape != b.shape or a.dtype != torch.float64 or b.dtype != torch.float64:
        raise RuntimeError('shape/dtype mismatch')
    return dict(values=a.numel(), bit_mismatches=int((a.view(torch.int64)!=b.view(torch.int64)).sum()),
                max_abs_difference=float((a-b).abs().max()) if a.numel() else 0.0,
                sign_mismatches=int((torch.signbit(a)!=torch.signbit(b)).sum()))


def ordinary(adapter, flat):
    with torch.random.fork_rng():
        model = _make_mlp_encoder(hidden_dims=[4], activation='silu').to(torch.float64)
    with torch.no_grad():
        for entry, p in zip(adapter.layout, model.parameters()):
            p.copy_(flat[entry['start']:entry['stop']].reshape(entry['shape']))
    return model


def reference_derivatives(model, states):
    prediction = model(states[:-1])
    residual = prediction-states[1:]
    loss = (residual*residual).sum()/residual.numel()
    params = tuple(model.parameters())
    grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params, create_graph=True)])
    rows = []
    for value in grad:
        derivatives = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                               for p,g in zip(params, derivatives)]))
    raw = torch.stack(rows)
    return prediction, loss, grad, raw, 0.5*(raw+raw.T)


def protected():
    paths = [Path('outputs/sho/homogeneity_peeled_baseline/freeze_manifest.json'),
             Path('outputs/hybrid_student_baseline_frozen/freeze_manifest.json'),
             Path('outputs/deterministic_teacher_frozen/freeze_manifest.json'),
             Path('outputs/canonical_frozen_teacher_regression_822/corpus_manifest.json')]
    manifest = json.loads(paths[0].read_text())
    paths.extend(Path(p) for p in manifest['affine_source_hashes'])
    return {str(p):sha(p) for p in paths}


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    before, checkpoints_before = protected(), hashes()
    parent = json.loads(Path('outputs/sho/homogeneity_peeled_baseline/freeze_manifest.json').read_text())
    for p, expected in parent['affine_source_hashes'].items():
        if sha(p) != expected: raise RuntimeError('frozen parent source mismatch')
    forbidden = Counter()
    loaded = []
    original_load = torch.load
    allowed_arithmetic = ('addition_operator_smoke/', 'multiplication_operator_smoke/',
                          'subtraction_operator_training/', 'division_rounding_router_training/',
                          'stationary_magnitude_threshold_smoke/', 'reciprocal_',
                          'division_operator_refinement/')

    def checked_load(path, *args, **kwargs):
        name = str(path)
        loaded.append(name)
        if not any(part in name for part in allowed_arithmetic):
            forbidden['old_non_arithmetic_weights'] += 1
            raise RuntimeError('forbidden learned update/controller checkpoint')
        return original_load(path, *args, **kwargs)

    def profile(frame, event, arg):
        if event != 'call': return
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        disallowed = ('affine_sho_update_assembly.py', 'full_k_sho_hessian_assembly.py',
                      'torch_matmul2_fma_assembly.py', 'torch_mm_backward_dot_assembly.py',
                      'recover_affine_bias_targeted.py')
        if filename.endswith(disallowed):
            forbidden[Path(filename).name+':'+name] += 1
            raise RuntimeError('forbidden physical affine path')
        obj = frame.f_locals.get('self')
        if obj is not None and type(obj).__name__ == 'SHOExactSolutionDecoder':
            forbidden['analytic_decoder'] += 1
            raise RuntimeError('analytic decoder invoked')

    report = dict(status='VALIDATING', training=False, recovery_run=False)
    try:
        sys.setprofile(profile)
        with patch('torch.load', checked_load):
            adapter = GenericSHOUpdateNetworkAdapter()
            rng = torch.random.get_rng_state().clone()
            second = GenericSHOUpdateNetworkAdapter()
            assert torch.equal(rng, torch.random.get_rng_state())
            assert all(p.is_meta and not p.requires_grad for p in adapter.template.parameters())
            assert [tuple(p.shape) for p in adapter.template.parameters()] == [(4,2),(4,),(2,4),(2,)]
            vectors = [torch.zeros(22), torch.linspace(-.7,.8,22),
                       .35*torch.sin(torch.arange(22,dtype=torch.float64)+.3),
                       torch.linspace(-1.99,1.99,22)]
            corpus = np.load('outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz')
            replay = []
            for case in (0,32,200):
                states = torch.from_numpy(corpus[f'prefix_{case}'].copy())
                for index, phi in enumerate(vectors):
                    model = ordinary(adapter, phi)
                    pred, loss, gradient, raw, symmetric = reference_derivatives(model, states)
                    actual = adapter.derivatives(phi, states)
                    assembled = adapter.objective_terms(phi, states)
                    row = dict(case=case, transitions=len(states)-1, parameter_fixture=index,
                        prediction=compare(adapter.forward(phi,states[:-1]),pred),
                        residual=compare(assembled.residual,pred-states[1:]),
                        square=compare(assembled.square,(pred-states[1:])*(pred-states[1:])),
                        loss=compare(assembled.loss,loss),
                        gradient=compare(actual.gradient,gradient),
                        raw_hessian=compare(actual.raw_hessian,raw),
                        symmetric_hessian=compare(actual.symmetric_hessian,symmetric))
                    replay.append(row)
                    for key in ('prediction','residual','square','loss','gradient','raw_hessian','symmetric_hessian'):
                        if row[key]['bit_mismatches']:
                            raise RuntimeError(f'first replay mismatch {case}/{index}/{key}: {row[key]}')
            # Independence/state mutation: a,b,a gives identical a outputs.
            x = torch.from_numpy(corpus['prefix_32'][:-1].copy())
            a = adapter.forward(vectors[1],x).clone()
            adapter.forward(vectors[2],x)
            assert compare(a,adapter.forward(vectors[1],x))['bit_mismatches']==0
            # Per-coordinate perturbations exercise every independent flat slot.
            changed=[]
            anchor=vectors[2].clone(); base=adapter.forward(anchor,x)
            for j in range(adapter.dimension):
                perturbed=anchor.clone(); perturbed[j]+=.01
                changed.append(bool(torch.any(adapter.forward(perturbed,x)!=base)))
            assert all(changed)
            containment=[]
            # Test-only affine embedding; no fixture is provided by the adapter.
            maps=[(torch.eye(2),torch.zeros(2)),
                  (torch.tensor([[.3,-.7],[.8,.2]]),torch.tensor([.1,-.2]))]
            for number,(matrix,bias) in enumerate(maps):
                phi=torch.zeros(22)
                phi[:8]=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]).reshape(-1)
                phi[12:20]=torch.stack((matrix[:,0],-matrix[:,0],matrix[:,1],-matrix[:,1]),dim=1).reshape(-1)
                phi[20:]=bias
                containment.append(dict(fixture=number, comparison=compare(adapter.forward(phi,x),x@matrix.T+bias)))
            rollout=[]
            for number,phi in enumerate(vectors[:3]):
                initial=torch.from_numpy(corpus['prefix_32'][-1].copy())
                model=ordinary(adapter,phi)
                expected=[initial.clone()]
                with torch.no_grad():
                    for _ in range(4096): expected.append(model(expected[-1][None,:])[0])
                actual=adapter.rollout(phi,initial)
                result=compare(actual,torch.stack(expected))
                assert result['bit_mismatches']==0 and torch.isfinite(actual).all()
                rollout.append(dict(fixture=number,shape=list(actual.shape),comparison=result))
            boxes=[]
            for n in (1,2,30,110):
                lo,hi=adapter.archive_box(vectors[1],n)
                radius=3.98/float(n)**(1.0/len(vectors[1]))
                assert np.array_equal(lo,np.maximum(-1.99,vectors[1].numpy()-radius))
                assert np.array_equal(hi,np.minimum(1.99,vectors[1].numpy()+radius))
                boxes.append(dict(evaluation_count=n,dimension=len(lo),radius=radius))
            # Demonstrate true network second derivatives: record full-H minus GN.
            phi=vectors[2].clone().requires_grad_(True)
            states=torch.from_numpy(corpus['prefix_32'].copy())
            jac=torch.autograd.functional.jacobian(lambda v:adapter.forward(v,states[:-1]).reshape(-1),phi)
            gn=(2.0/jac.shape[0])*(jac.T@jac)
            full=adapter.derivatives(phi,states).raw_hessian
            gn_difference=float((full-gn).abs().max())
            assert gn_difference>0
            report.update(status='GENERIC_SHO_UPDATE_NETWORK_ADAPTER_READY',
                architecture='2 -> 4 SiLU -> 2; ordinary biases; no skip',
                free_parameters=adapter.dimension,layout=adapter.layout,
                bounds=dict(lower=adapter.lower.tolist(),upper=adapter.upper.tolist()),
                derivative_oracle=adapter.derivative_oracle,recovery_status=adapter.recovery_status,
                physical_affine_graph='removed',replay=replay,affine_containment=containment,
                rollout=rollout,archive_box_controls=boxes,
                full_hessian_minus_gauss_newton_max_abs=gn_difference,
                independently_exercised_parameters=sum(changed),
                template_is_meta_and_frozen=True,rng_unchanged_by_template_creation=True,
                invocation_counts=adapter.calls,arithmetic_checkpoint_reads=loaded,
                forbidden_affine_path_calls=dict(forbidden),forbidden_affine_path_call_total=sum(forbidden.values()))
    except Exception as error:
        report.update(status='FAIL_CLOSED',error=str(error))
        raise
    finally:
        sys.setprofile(None)
        report['integrity']=dict(frozen_metadata_and_affine_sources_unchanged=before==protected(),
                                 checkpoint_count=len(checkpoints_before),
                                 checkpoint_hashes_unchanged=checkpoints_before==hashes())
        report['protected_hashes']=before
        report['source_hashes']={str(p):sha(p) for p in [Path(__file__),Path('experiments/sho/generic_update_network_adapter.py')]}
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not all(v for k,v in report['integrity'].items() if k!='checkpoint_count'):
        raise RuntimeError('integrity failure')
    print(json.dumps({k:report[k] for k in ('status','free_parameters','integrity','invocation_counts')},indent=2))


if __name__=='__main__': main()
