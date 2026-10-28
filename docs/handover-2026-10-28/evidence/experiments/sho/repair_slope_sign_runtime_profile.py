"""Bounded threshold-only repair and static profile; never advances a search."""
import argparse, copy, csv, hashlib, json, math, signal, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho import run_stage1b_fifo_vs_ranker as run
from experiments.sho import run_stage1b_hull_slope_sign as sign
from experiments.sho import run_stage1b_k_lower_margin as lower

OUT=Path('outputs/sho/stage1B/slope_sign_repair_runtime')
ROWS=[]; DEADLINE=0

class TimeLimit(RuntimeError): pass
def alarm(*args): raise TimeLimit('component or total wall-clock limit')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def arm(seconds): signal.setitimer(signal.ITIMER_REAL,max(.001,min(seconds,DEADLINE-time.time())))

def repair(payload):
    old=torch.load(sign.SIGN_CHECKPOINT_PATH,weights_only=False)
    scale=float(old['training_scale_S'])
    full=[r for r in csv.DictReader(sign.ASSEMBLED_PATH.open()) if r['record_type']=='finite_edge']
    assert len(full)==int(old['full_pool_summary']['event_count'])
    df=np.array([float(r['delta_f']) for r in full]); valid=np.arcsinh(df/scale)
    indices=sign.turn._stratified_state_indices(payload['states'],'train',8,sign.smoke.DATA_SEED)
    states=sign.turn._build_states(payload,indices,1,'repair real training')
    pool=sign._build_edge_pool(states,'train')
    assert not(set(indices)&{int(r['source_state_index']) for r in full})
    real=np.arcsinh(pool['delta_f']/scale)
    synthetic=np.array([1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-4,1e-2])
    # Every real magnitude and every boundary magnitude receives equal + / - mass.
    mags=np.r_[np.abs(real),synthetic]
    x=np.r_[mags,-mags]; y=np.sign(x)
    model=sign.MonotoneThreshold(1e-8,float(old['state_dict']['alpha_scale']))
    with torch.no_grad():
        model.theta_scaled.fill_(float(old['learned_theta'])/1e-8)
        model.raw_alpha.copy_(old['state_dict']['raw_alpha'])
    xt=torch.from_numpy(x/1e-8); yt=torch.from_numpy(y)
    optimizer=torch.optim.LBFGS([model.theta_scaled],lr=.5,max_iter=1,
        tolerance_grad=1e-15,tolerance_change=1e-20,line_search_fn='strong_wolfe')
    alpha_opt=torch.optim.Adam([model.raw_alpha],lr=.01)
    failure=json.loads((run.OUT/'frozen_pipeline_stop_diagnosis.json').read_text())
    fx=np.array([e['compressed_delta_f'] for e in failure['edges']])
    checks=np.r_[valid,synthetic,-synthetic,fx]
    start=time.monotonic();arm(119)
    history=[]
    for step in range(1,1001):
        def closure():
            optimizer.zero_grad()
            loss=1e-5*torch.logsumexp(-yt*(xt-model.theta_scaled)/1e-5,dim=0)
            loss.backward();return loss
        optimizer.step(closure)
        alpha_opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x)),torch.from_numpy((y>0).astype(float)))
        loss.backward();alpha_opt.step()
        with torch.no_grad(): pred=model(torch.from_numpy(checks)).numpy()
        mistakes=int(np.sum(np.sign(pred)!=np.sign(checks)))
        history.append(dict(record_type='training',step=step,theta=float(model.theta()),alpha=float(model.alpha()),mistakes=mistakes))
        if mistakes==0:break
        if time.monotonic()-start>115:break
    passed=mistakes==0
    rows=[dict(record_type='summary',passed=passed,training_seconds=time.monotonic()-start,
        updates=step,training_real_edges=len(real),training_states=json.dumps(indices),held_out_states=len(set(r['source_state_index'] for r in full)),
        held_out_edges=len(valid),held_out_mistakes=int(np.sum(np.sign(pred[:len(valid)])!=np.sign(valid))),
        synthetic_cases=16,synthetic_mistakes=int(np.sum(np.sign(pred[len(valid):len(valid)+16])!=np.sign(checks[len(valid):len(valid)+16]))),
        failure_state_mistakes=int(np.sum(np.sign(pred[-len(fx):])!=np.sign(fx))),
        initial_theta=float(old['learned_theta']),learned_theta=float(model.theta()),learned_alpha=float(model.alpha()),training_scale_S=scale,
        symmetry='equal mass for each +/- magnitude; zero not a training example',theta_hard_coded=False)]+history
    for i,(a,b) in enumerate(zip(checks,pred)):
        rows.append(dict(record_type='sign_case',pool='held_out' if i<len(valid) else 'synthetic' if i<len(valid)+16 else 'failure_state',x=float(a),logit=float(b),correct=bool(np.sign(a)==np.sign(b))))
    if passed:
        saved=copy.deepcopy(old);saved.update(state_dict=copy.deepcopy(model.state_dict()),
            learned_theta=model.theta().detach(),learned_alpha=model.alpha().detach(),status='REPAIRED_THRESHOLD_VALIDATED',
            repair_summary=rows[0],previous_checkpoint_sha256=sha(sign.SIGN_CHECKPOINT_PATH))
        path=OUT/'hull_slope_sign_repaired.pt';torch.save(saved,path)
        digest=sha(path);path.with_suffix('.sha256').write_text(digest+'  '+path.name+'\n')
        rows[0].update(checkpoint=str(path),sha256=digest)
    write(OUT/'slope_sign_repair.csv',rows)
    print('REPAIR '+json.dumps(rows[0]),flush=True)
    arm(540)
    if not passed:raise RuntimeError('repair failed: entire run stopped')
    return path,rows[0]

class Meter:
    def __init__(self):
        self.times=defaultdict(float);self.calls=defaultdict(int);self.stack=[]
        self.neural_seconds=0.;self.neural_calls=0;self.neural_depth=0
        self.distance_batches=0;self.distance_values=0;self.margin_values=0
    def wrap(self,fn,name):
        def measured(*args,**kwargs):
            start=time.perf_counter();entry=[name,0.];self.stack.append(entry)
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-start;self.stack.pop()
                self.times[name]+=elapsed-entry[1];self.calls[name]+=1
                if self.stack:self.stack[-1][1]+=elapsed
        return measured
    def neural(self,fn):
        def measured(*a,**kw):
            outer=self.neural_depth==0;self.neural_depth+=1;t=time.perf_counter()
            try:return fn(*a,**kw)
            finally:
                self.neural_depth-=1
                if outer:self.neural_seconds+=time.perf_counter()-t;self.neural_calls+=1
        return measured

def profile(payload,repaired):
    original_load=torch.load
    def replaced_load(path,*a,**kw):
        if isinstance(path,(str,Path)) and Path(path).resolve()==sign.SIGN_CHECKPOINT_PATH.resolve():path=repaired
        return original_load(path,*a,**kw)
    torch.load=replaced_load
    frozen=run.FrozenProposals()
    trajectories=run.cells.base.ordinary_data._trajectory_map(payload)
    for budget in (1000,8600,32768):
        if DEADLINE-time.time()<80:raise TimeLimit('insufficient remaining wall-clock budget')
        archive=np.load(f'outputs/sho/stage1B/spacefill_downstream_screen/spacefill_probes_N{budget}.npz')
        state=int(archive['source_state_index']);record=payload['states'][state];n=int(record['history_count'])
        trajectory=trajectories[int(record['trajectory_id'])]
        sites=np.asarray(trajectory['evaluated_hypotheses_normalized'][:n],dtype=np.float64)
        losses=np.asarray(trajectory['evaluated_prefix_losses'][:n],dtype=np.float64)
        case=dict(generator_id=int(record['generator_id']),trajectory_id=int(record['trajectory_id']))
        meter=Meter();patches=[];neural_original=torch.nn.Module._call_impl
        def patch(obj,name,value):patches.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
        def timed(obj,name,label):patch(obj,name,meter.wrap(getattr(obj,name),label))
        # Observe the exact assembled implementation, including its reference audit.
        timed(run.edirect,'_candidate_pool','probe_generation')
        original_predict=run.nearest.base.distance_metric._predict
        def distance(*a,**kw):
            value=original_predict(*a,**kw);meter.distance_batches+=1;meter.distance_values+=len(value);return value
        patch(run.nearest.base.distance_metric,'_predict',meter.wrap(distance,'learned_distance'))
        # capture_nearest observes scan's original code frame; preserve that function.
        timed(run.cells,'capture_nearest','argmin_reduction_and_validation')
        original_margin=run.nearest.margin.predict
        def margin(*a,**kw):
            meter.margin_values+=len(a[1]);return original_margin(*a,**kw)
        patch(run.nearest.margin,'predict',meter.neural(margin))
        tree_original=run.nearest.cKDTree
        class ObservedTree:
            def __init__(self,*a,**kw):self.tree=meter.wrap(tree_original,'tie_handling_and_reference_audit')(*a,**kw)
            def query(self,*a,**kw):return meter.wrap(self.tree.query,'tie_handling_and_reference_audit')(*a,**kw)
        patch(run.nearest,'cKDTree',ObservedTree)
        timed(run.cells,'assigned_radius','assigned_radius_max')
        timed(run.cells,'predict_half','half_spacing')
        larger_original=run.cells.larger_candidate
        def larger(*a,**kw):
            if any(x[0]=='assigned_radius_max' for x in meter.stack):return larger_original(*a,**kw)
            return meter.wrap(larger_original,'final_cell_size_max')(*a,**kw)
        patch(run.cells,'larger_candidate',larger)
        timed(run.down,'_learned_group_and_represent','groups_and_representatives')
        timed(run.down.hull_threshold,'_threshold_hull','hull')
        timed(run.down.improvement_and,'_build_pool','improvement_condition')
        for name in ('__init__','loo_errors','predict'):
            timed(run.edirect.GaussianLocalResponse,name,'surrogate_and_LOO')
        timed(frozen.score,'forward','candidate_scoring')
        timed(frozen.locked['and_model'],'forward','improvement_condition')
        torch.nn.Module._call_impl=meter.neural(neural_original)
        # Additional frozen NumPy linear forward helper, without double counting torch.
        timed_neural=lower._checkpoint_linear_predict
        patch(lower,'_checkpoint_linear_predict',meter.neural(timed_neural))
        started=time.perf_counter();status='complete';error='';result=None
        # A stricter 59-second cap on the entire static pass bounds every component.
        arm(59)
        try:result=meter.wrap(frozen,'proposal_queue_and_bookkeeping')(sites,losses,case,int(record['iteration']))
        except TimeLimit as exc:status='aborted_59s';error=str(exc)
        except Exception as exc:status='pipeline_error';error=traceback.format_exc()
        finally:
            elapsed=time.perf_counter()-started
            torch.nn.Module._call_impl=neural_original
            for obj,name,value in reversed(patches):setattr(obj,name,value)
            arm(540)
        probes=run.edirect._candidate_pool(n,case['generator_id'],int(record['iteration']))
        t=time.perf_counter();tree=cKDTree(sites);tree.query(probes,k=1);tree.query(sites,k=2)
        deterministic=time.perf_counter()-t
        p=len(probes);comparisons=p*(n-1)+n*(n-2)
        completed_candidates=meter.distance_batches
        factor=n/max(completed_candidates,1)
        scan_total=sum(meter.times[k] for k in ('learned_distance','argmin_reduction_and_validation','tie_handling_and_reference_audit'))
        common=dict(N=budget,source_state_index=state,site_count=n,probe_count=p,status=status,error=error,
            static_total_seconds=elapsed,full_argmin_comparisons=comparisons,completed_candidate_scans=completed_candidates,
            expected_candidate_scans=n,distance_values=meter.distance_values,margin_pairwise_values=meter.margin_values,
            neural_calls=meter.neural_calls,neural_seconds=meter.neural_seconds,
            seconds_per_million_argmin_comparisons=scan_total*factor/comparisons*1e6,
            learned_argmin_inclusive_seconds=scan_total,estimated_full_argmin_seconds=scan_total*factor,
            ckdtree_seconds=deterministic,argmin_to_ckdtree_ratio=scan_total*factor/deterministic,
            timing_includes_validation_audit=True,spacing_query_count=n,
            spacing_note='nearest-site scan shares distance/argmin/tie categories; half-spacing separately timed',
            proposals=len(result[0]) if result is not None else '')
        for name,value in meter.times.items():
            ROWS.append(dict(record_type='component',**common,component=name,seconds=value,
                percent_total=100*value/elapsed,calls=meter.calls[name],estimated_seconds=value*factor if status=='aborted_59s' else value))
        ROWS.append(dict(record_type='overlap_metric',**common,component='all_observed_neural_forwards',seconds=meter.neural_seconds,
            percent_total=100*meter.neural_seconds/elapsed,overlaps_other_rows=True))
        ROWS.append(dict(record_type='comparison',**common,component='deterministic_ckdtree',seconds=deterministic))
        write(OUT/'pipeline_runtime_profile.csv',ROWS)
        print('PROFILE '+json.dumps(common),flush=True)
        if status!='complete':break
    torch.load=original_load

def main():
    global DEADLINE
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args();DEADLINE=args.deadline
    OUT.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGALRM,alarm);arm(540)
    start=time.time();summary=None;error=None
    protected=json.loads((run.OUT/'plan.json').read_text())['protected_hashes']
    try:
        torch.set_num_threads(1)
        payload=torch.load(sign.turn.RAW_TRACE_PATH,weights_only=False)
        repaired=OUT/'hull_slope_sign_repaired.pt'
        if repaired.exists():
            saved=torch.load(repaired,weights_only=False);summary=saved['repair_summary']
            assert summary['passed'] and sha(repaired)==repaired.with_suffix('.sha256').read_text().split()[0]
            print('Reusing saved repaired threshold; no additional training.',flush=True)
        else:repaired,summary=repair(payload)
        profile(payload,repaired)
    except Exception as exc:
        error=repr(exc);print('STOP '+error,flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if not (OUT/'slope_sign_repair.csv').exists():write(OUT/'slope_sign_repair.csv',[dict(status='stopped',reason=error)])
        if not ROWS:write(OUT/'pipeline_runtime_profile.csv',[dict(status='not_run',reason=error)])
        unchanged=all(sha(p)==h for p,h in protected.items())
        lines=['SLOPE SIGN REPAIR AND STATIC RUNTIME PROFILE',json.dumps(summary),
            f'Process elapsed seconds: {time.time()-start:.3f}; stop: {error}; protected hashes unchanged: {unchanged}',
            'No search iterations, objective evaluations, rescue, or other network training.',
            'Repaired threshold saved separately and injected into this static profile only; original locks preserved.',
            'CPU single-thread; one instrumented pass per saved state, no warm-up. Timings include instrumentation and existing validation.',
            'Distance, reduction, and tie/audit rows partition the combined probe + site-spacing scan.',
            'Neural timing overlaps component rows; counts include outer torch module calls, NumPy margin forwards and slope linear helpers.',
            'Other inline learned arithmetic is included in its parent component; no claim of isolated hardware-kernel timing.',
            'component | small seconds | medium seconds | large seconds | % large total']
        names=list(dict.fromkeys(r['component'] for r in ROWS))
        for name in names:
            rr={r['N']:r for r in ROWS if r['component']==name}
            values=[f"{rr[n]['seconds']:.6f}" if n in rr else 'not measured' for n in (1000,8600,32768)]
            pct=rr.get(32768,{}).get('percent_total')
            lines.append(name+' | '+' | '.join(values)+' | '+(f'{pct:.2f}%' if pct is not None else 'comparison'))
        for r in ROWS:
            if r['record_type']=='comparison':lines.append('STATE '+json.dumps(r))
        (OUT/'pipeline_runtime_summary.txt').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
