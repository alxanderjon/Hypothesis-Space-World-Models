"""Read-only reciprocal input audit using archived cuts, grids and coefficients."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_exponent_log2_operator import ROOT, sha, write_csv
from experiments.geometry.train_reciprocal_operator import PositivePower, predict

OUT = ROOT/'reciprocal_required_domain_audit'
SOURCE = ROOT/'reciprocal_operator_training_refine'
EPS = np.finfo(np.float64).eps


class Collector:
    def __init__(self):
        self.inputs = {}; self.batches = []; self.offset = 0

    def read(self,path):
        self.inputs[str(path)] = sha(path)
        if path.suffix=='.json': return json.loads(path.read_text())
        if path.suffix=='.csv': return list(csv.DictReader(path.open()))
        return np.load(path)

    def add(self,scope,case,c,a,B,tcode,vcode,order='generic'):
        c=np.asarray(c,dtype=np.float64); t=np.where(np.asarray(tcode)==0,c,1-c); v=np.where(np.asarray(vcode)==0,c,1-c)
        # Match each archived implementation's actual arithmetic order.
        q=B+(a*t)**2 if order=='geometric' else B+a**2*t*t
        root=np.sqrt(q)
        assert np.all((root>0)&np.isfinite(root)) and np.all((v>0)&np.isfinite(v))
        self.batches.append(dict(scope=scope,case=str(case),start=self.offset,end=self.offset+len(c),c=c,
                                 a=float(a),B=float(B),t=t,v=v,x_sqrt=root,x_v=v,order=order))
        self.offset += len(c)

    def geometry(self,r,axis,k,c,scope,case):
        c=np.asarray(c); sides=np.asarray(r['upper'])-r['lower']; a=sides[axis]; other=np.delete(sides,axis); B=float(np.sum(other**2))
        t=np.column_stack((c,1-c)); diagonal=np.sqrt(B+(a*t)**2)
        q=diagonal/np.power(np.array([k,r['n']-k],dtype=np.float64),1/r['d'])
        tc=q.argmax(1)
        if len(other):
            sides_axis=a*t
            logs=np.log(np.maximum(sides_axis,other.max())/np.minimum(sides_axis,other.min()))
            vc=logs.argmax(1)
        else: vc=np.zeros(len(c),dtype=int)
        self.add(scope,case,c,a,B,tc,vc,'geometric')

    def collect(self):
        parents=[]
        for role in ('train','heldout'): parents += self.read(ROOT/f'learned_split_proposer_smoke/{role}_regions.json')
        old=self.read(ROOT/'split_quality_smoke/previous_n4_failure.npz')
        parents.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
        base=ROOT/'fixed_allocation_interval_audit'
        intervals=self.read(base/'fixed_allocation_interval_audit.csv')
        pieces=self.read(base/'fixed_interval_score_structure.csv')
        saved=self.read(ROOT/'split_subpiece_stationary_solver_smoke/audited_subpiece_assembly.csv')
        predicted={(int(r['parent']),int(r['axis']),int(r['interval']),int(r['piece'])):r for r in saved}
        for r in intervals:
            if r['valid']!='True': continue
            parent,axis,k=[int(r[x]) for x in ('parent','axis','interval')]
            cuts=np.unique(np.r_[np.linspace(float(r['lower']),float(r['upper']),513),
                                 json.loads(r['analytic_breakpoints']),json.loads(r['stationary_points']),float(r['sampled_min_cut'])])
            self.geometry(parents[parent],axis,k,cuts,'archived_interval_grids',f'{parent}:{axis}:{k}')
        for r in pieces:
            parent,axis,k,piece=[int(r[x]) for x in ('parent','axis','interval','piece')]
            l,u=float(r['lower']),float(r['upper']); key=(parent,axis,k,piece)
            # This is the saved audit's deterministic grid, not a new search.
            cuts=np.r_[l,u,np.linspace(np.nextafter(l,u),np.nextafter(u,l),int(r['derivative_samples'])),
                       json.loads(r['stationary_points']),json.loads(r['quartic_stationary_roots'])]
            if key in predicted and predicted[key]['predicted_root']=='True': cuts=np.r_[cuts,float(predicted[key]['root_cut'])]
            sides=np.asarray(parents[parent]['upper'])-parents[parent]['lower']; other=np.delete(sides,axis)
            self.add('archived_subpiece_grids',key,np.unique(cuts),sides[axis],float(np.sum(other**2)),
                     int(r['radius_active_child']),int(r['aspect_active_child']),'geometric')
        for folder in ('fixed_interval_cut_optimizer_smoke','fixed_interval_iterative_optimizer_smoke'):
            for path in sorted((ROOT/folder).glob('*intervals.json')):
                rows=self.read(path); role='heldout' if 'heldout' in path.stem else 'train' if 'train' in path.stem else 'selection'
                predictions=defaultdict(list)
                if role!='selection':
                    files=([ROOT/folder/f'stage_1_{role}_predictions.csv'] if folder.startswith('fixed_interval_cut_') else
                           [ROOT/folder/f'queries_2_{role}.csv',ROOT/f'fixed_interval_iterative_optimizer_continued/two_query_best_{role}.csv'])
                    for pp in files:
                        if pp.exists():
                            for p in self.read(pp): predictions[(p['parent'],int(p['axis']),int(p['left_count']))].append(p)
                for i,r in enumerate(rows):
                    cuts=[r['interval_lower'],r['interval_upper'],r['target_cut']]
                    for p in predictions.get((str(r['parent']),r['axis'],r['left_count']),[]):
                        cuts.append(float(p['predicted_cut']))
                        if p.get('query_z'):
                            cuts += [r['interval_lower']+z*(r['interval_upper']-r['interval_lower']) for z in json.loads(p['query_z'])]
                    self.geometry(r,r['axis'],r['left_count'],np.unique(cuts),'archived_optimizer_queries',folder+'/'+path.stem+':'+str(i))
        base=ROOT/'split_subpiece_stationary_solver_smoke'
        paths=sorted(base.glob('stage_*_train.json'))+sorted(base.glob('stage_*_selection.json'))+sorted(base.glob('stage_*_heldout.json'))+[base/'interval_location_holdouts.json']
        for path in paths:
            rows=self.read(path); pp=path.with_name(path.stem+'_predictions.csv') if path.stem!='interval_location_holdouts' else path.with_suffix('.csv')
            predictions={int(p['case']):p for p in self.read(pp)} if pp.exists() else {}
            for i,r in enumerate(rows):
                cuts=[r['lower'],r['upper']]+r['roots']
                if r.get('c_star') is not None: cuts.append(r['c_star'])
                if i in predictions:
                    p=predictions[i]
                    if p['predicted_exists']=='True': cuts.append(float(p['predicted_cut']))
                    cuts.append(float(p['selected_cut']))
                self.add('archived_generic_subpieces',path.stem+':'+str(i),np.unique(cuts),r['a'],r['B'],r['t'],r['v'])


def summarize(x,truth,pred,branch):
    err=pred-truth; rel=err/truth; largest=float(abs(err).max())
    identity=x*pred-1
    order=np.argsort(x,kind='stable'); xx=x[order]; yy=pred[order]; strictly=np.diff(xx)>0
    quantiles=np.quantile(x,[0,.001,.01,.25,.5,.75,.99,.999,1])
    result=dict(branch=branch,count=len(x),unique_inputs=int(np.unique(x).size),minimum=float(x.min()),maximum=float(x.max()),
                log10_min=float(np.log10(x.min())),log10_max=float(np.log10(x.max())),
                input_subnormals=int(np.sum(x<np.finfo(float).tiny)),output_subnormals=int(np.sum((pred>0)&(pred<np.finfo(float).tiny))),
                input_within_16_binary_orders_of_limit=int(np.sum((x<np.ldexp(np.finfo(float).tiny,16))|(x>np.ldexp(np.finfo(float).max,-16)))),
                output_within_16_binary_orders_of_limit=int(np.sum((truth<np.ldexp(np.finfo(float).tiny,16))|(truth>np.ldexp(np.finfo(float).max,-16)))),
                exact_matches=int(np.sum(pred==truth)),rmse=float(largest*np.sqrt(np.mean((err/largest)**2))) if largest else 0.,
                max_absolute_error=largest,relative_rmse=float(np.sqrt(np.mean(rel*rel))),max_relative_error=float(abs(rel).max()),
                monotonicity_reversals=int(np.sum(strictly&(np.diff(yy)>0))),underflows=int(np.sum(pred==0)),overflows=int(np.isinf(pred).sum()),
                nonfinite_outputs=int(np.sum(~np.isfinite(pred))),identity_rmse=float(np.sqrt(np.mean(identity*identity))),identity_max_error=float(abs(identity).max()))
    result.update({name:float(v) for name,v in zip(['q0','q001','q01','q25','q50','q75','q99','q999','q100'],quantiles)})
    return result


def run():
    started=time.monotonic(); torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    prior=json.loads((SOURCE/'results.json').read_text()); checkpoint=Path(prior['checkpoint'])
    assert sha(checkpoint)==prior['sha256']
    model=PositivePower(); model.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']); model.eval().requires_grad_(False)
    assert float(model.weights[0])==-1. and float(torch.exp(model.weights[1]))==1.
    c=Collector(); c.collect()
    print(json.dumps(dict(event='archived_inputs_collected',batches=len(c.batches),inputs_per_branch=c.offset)),flush=True)
    arrays={branch:np.concatenate([b[branch] for b in c.batches]) for branch in ('x_sqrt','x_v')}
    rows=[]; worst=[]; predictions={}; ends=np.array([b['end'] for b in c.batches])
    for branch,x in arrays.items():
        truth=np.divide(1.,x); pred=predict(model,x); predictions[branch]=pred
        rows.append(summarize(x,truth,pred,branch))
        for ranking,error in [('relative',abs((pred-truth)/truth)),('absolute',abs(pred-truth))]:
            ids=np.argsort(-error,kind='stable')[:20]
            for rank,i in enumerate(ids,1):
                batch=c.batches[int(np.searchsorted(ends,i,side='right'))]; local=i-batch['start']
                worst.append(dict(branch=branch,ranking=ranking,rank=rank,scope=batch['scope'],case=batch['case'],
                    cut=float(batch['c'][local]),a=batch['a'],B=batch['B'],t=float(batch['t'][local]),v=float(batch['v'][local]),
                    input=float(x[i]),input_hex=float(x[i]).hex(),teacher=float(truth[i]),learned=float(pred[i]),
                    absolute_error=float(abs(pred[i]-truth[i])),relative_error=float(abs((pred[i]-truth[i])/truth[i])),
                    identity_error=float(x[i]*pred[i]-1)))
    combined=np.concatenate(list(arrays.values())); combined_prediction=np.concatenate(list(predictions.values()))
    rows.append(summarize(combined,np.divide(1.,combined),combined_prediction,'combined'))
    write_csv(OUT/'reciprocal_required_domain_audit.csv',rows)
    write_csv(OUT/'reciprocal_required_domain_worst_cases.csv',worst)
    counts=defaultdict(lambda:dict(batches=0,inputs_per_branch=0))
    for b in c.batches:
        counts[b['scope']]['batches']+=1; counts[b['scope']]['inputs_per_branch']+=len(b['c'])
    write_csv(OUT/'archive_counts.csv',[dict(scope=k,**v) for k,v in counts.items()])
    np.savez_compressed(OUT/'required_inputs_and_predictions.npz',**arrays,**{k+'_prediction':v for k,v in predictions.items()})
    allr=rows[-1]
    passed=bool(allr['nonfinite_outputs']==0 and allr['underflows']==0 and allr['monotonicity_reversals']==0
                and allr['max_relative_error']<=16*EPS and allr['identity_max_error']<=16*EPS)
    unchanged=all(sha(Path(p))==h for p,h in protected.items()); assert unchanged
    assert all(sha(Path(p))==h for p,h in c.inputs.items())
    limitation=json.loads((SOURCE/'double_reciprocal_failure_cases.json').read_text())
    result=dict(operator='reciprocal_operator',status='FROZEN_CANDIDATE_ON_REQUIRED_DOMAIN' if passed else 'DIAGNOSTIC_ONLY',
                gate_passed=passed,learned_p=-1.,learned_s=1.,sha256=prior['sha256'],original_checkpoint=str(checkpoint),
                metrics=rows,archive_counts=dict(counts),known_out_of_domain_double_reciprocal_failure=limitation,
                double_reciprocal_test_executed=False,full_derivative_assembled=False,
                scope='All persisted fixed-interval and subpiece geometries, saved optima/queries, and reproducible archived 513/129 grids; not a claim over unsaved historical solver queries or all future geometries.',
                safeguards='16 float64 eps relative reciprocal and product-identity limits; zero nonfinite/underflow/reversal failures.',
                all_prior_checkpoint_hashes_unchanged=unchanged,protected_checkpoints=len(protected),elapsed_seconds=time.monotonic()-started)
    if passed:
        saved=OUT/'reciprocal_required_domain.pt'
        with saved.open('xb') as handle: handle.write(checkpoint.read_bytes())
        assert sha(saved)==prior['sha256']
        saved.with_suffix('.sha256').write_text(f'{prior["sha256"]}  {saved.name}\n')
        manifest=dict(operator='reciprocal_operator',status=result['status'],checkpoint=str(saved),sha256=prior['sha256'],
                      module='experiments.geometry.train_reciprocal_operator',model='PositivePower',learned_p=-1.,learned_s=1.,
                      validated_domain={r['branch']:dict(minimum=r['minimum'],maximum=r['maximum'],samples=r['count']) for r in rows[:2]},
                      validation_scope=result['scope'],known_limitation=limitation,evaluation=str(OUT/'results.json'),
                      checkpoint_copy='Byte-identical to existing diagnostic; no parameter or payload modification')
        (OUT/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2))
        result['promoted_checkpoint']=str(saved)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    (OUT/'input_archive_hashes.json').write_text(json.dumps(c.inputs,indent=2))
    (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=[f"reciprocal_operator = {result['status']}",
           'Only reciprocal inputs/outputs and x*R(x) evaluated. No double reciprocal or derivative assembled.',
           'Input occurrences include overlapping archived datasets/grids; distinct input counts are reported separately.',
           'Saved grid points are reconstructed from archived bounds; no allocation discovery, roots or optimizer queries are run.',
           'For geometry records, active branches are identified using exact existing q/aspect comparisons. Both branch inputs are retained even when their derivative coefficient is zero (conservative coverage).',
           'Quantile columns q001/q01/q25/q50/q75/q99/q999 correspond to 0.1/1/25/50/75/99/99.9 percent.',
           'Near-limit diagnostic means within 16 binary orders of a normal-range limit; subnormals are reported independently.']
    for r in rows: lines.append(json.dumps(r))
    lines += ['Known limitation retained: at x=1.7976931348623151e308, the prior double-reciprocal test overflowed despite a finite teacher result. This input is outside the archived required domain; double reciprocal is not required here.',
              result['scope'],f'Checkpoint SHA256 unchanged: {prior["sha256"]}.',
              f'All {len(protected)} prior checkpoints and all input archives unchanged. Execution: {result["elapsed_seconds"]:.3f} seconds.']
    (OUT/'reciprocal_required_domain_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
