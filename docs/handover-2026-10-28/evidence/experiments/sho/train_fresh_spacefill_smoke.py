"""One bounded geometry-only training run; persist before independent evaluation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.fresh_spacefill_generator import FreshGenerator,NOISE_SEED
from experiments.sho import run_stage1b_spacefill_generator_smoke as old
from experiments.sho import run_stage1b_fifo_vs_ranker as src
from experiments.sho.smoke_artifacts import save_checkpoint

OUT=Path('outputs/sho/stage1B/fresh_spacefill_smoke')
SEED=27092027;BUDGETS=(1000,8600,32768)

def distances(points,reference):
    ids=cKDTree(points.detach().numpy()).query(reference.numpy())[1]
    return torch.linalg.vector_norm(reference-points[ids],dim=1)

def discrepancy(points,n):
    # Differentiable periodic cloud-in-cell counts; exact hard histograms are
    # reserved for validation. Off-grid shifts prevent dyadic-grid aliasing.
    terms=[]
    for k in (4,8,16,32,64,128):
        for shift in (.25,.75):
            z=torch.remainder(points*k+shift-.5,k);base=torch.floor(z).long();frac=z-base
            hist=torch.zeros(k*k,dtype=torch.float64)
            for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
                weight=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
                ids=((base[:,0]+dx)%k)*k+(base[:,1]+dy)%k
                hist=hist.scatter_add(0,ids,weight)
            cdf=hist.reshape(k,k).cumsum(0).cumsum(1)/n
            edge=torch.arange(1,k+1,dtype=torch.float64)/k
            error=cdf-edge[:,None]*edge[None,:]
            terms.append(math.sqrt(n)*(error.abs().max()+error.square().mean().sqrt()))
    return torch.stack(terms).mean()

def objective(model,n,iteration,reference):
    points=[model.interior(n,iteration+j,'cell') for j in range(4)]
    greedy=model.interior(n,iteration,'greedy');boundary=torch.from_numpy(old.faces(n,2)[0]);scale=math.sqrt(n)
    covering=[];cluster=[];disc=[]
    for x in (points[0],greedy):
        d=distances(torch.cat((x,boundary)),reference)*scale
        covering.append(d.max()+.1*torch.topk(d,32).values.mean())
        ids=cKDTree(x.detach().numpy()).query(x.detach().numpy(),k=2)[1][:,1]
        separation=torch.linalg.vector_norm(x-x[ids],dim=1)*scale
        cluster.append(torch.relu(.4-separation).square().mean())
        disc.append(discrepancy(x,n))
    fresh=[]
    for a,b in ((points[0],points[1]),(points[1],points[2]),(points[2],points[3]),(points[0],greedy)):
        ids=cKDTree(b.detach().numpy()).query(a.detach().numpy())[1]
        delta=torch.linalg.vector_norm(a-b[ids],dim=1)*scale
        fresh.append(torch.exp(-(delta/.15).square()).mean())
    union=distances(torch.cat((*points,boundary)),reference)*math.sqrt(4*n)
    parts=dict(cover=torch.stack(covering).mean(),discrepancy=torch.stack(disc).mean(),
        clustering=torch.stack(cluster).mean(),near_overlap=torch.stack(fresh).mean(),union_cover=union.max()+.1*torch.topk(union,32).values.mean())
    loss=parts['cover']+2*parts['discrepancy']+.2*parts['clustering']+.5*parts['near_overlap']+.5*parts['union_cover']
    return loss,{k:float(v.detach()) for k,v in parts.items()}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'fresh_spacefill_trained_pending_evaluation.pt'
    if path.exists():raise FileExistsError('No automatic repeat or extension of smoke training')
    torch.set_num_threads(1);torch.manual_seed(SEED);started=time.time()
    protected=json.loads(Path('outputs/sho/stage1B/grid_branch_bound/grid_branch_bound_candidate.json').read_text())['protected_hashes']
    oldpath='outputs/sho/stage1B/spacefill_downstream_screen/spacefill_generator_candidate.pt';protected[oldpath]=src.sha(oldpath)
    model=FreshGenerator(2);optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    references={n:torch.from_numpy(np.random.default_rng(SEED+n).random((32768,2))) for n in BUDGETS}
    history=[]
    def monitor(step):
        losses=[]
        with torch.no_grad():
            for n in BUDGETS:
                loss,parts=objective(model,n,7000,references[n]);losses.append(float(loss))
                history.append(dict(record_type='fixed_training_monitor',step=step,N=n,loss=float(loss),**parts))
        return float(np.mean(losses))
    initial=monitor(0);last=initial;stop_reason='60_update_cap';updates=0
    for step in range(60):
        n=BUDGETS[step%3];loss,parts=objective(model,n,5000+4*step,references[n])
        if not torch.isfinite(loss):stop_reason='nonfinite_loss';break
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();updates=step+1
        history.append(dict(record_type='training',step=updates,N=n,iteration=5000+4*step,loss=float(loss.detach()),**parts))
        if updates%15==0:
            last=monitor(updates);src.write_csv(OUT/'training_history.csv',history)
            print(f'update={updates}/60 fixed_train_loss={last:.6g} initial={initial:.6g} elapsed={time.time()-started:.1f}s',flush=True)
        if updates==30 and last>=initial*.995:stop_reason='no_clear_fixed_monitor_improvement_at_30';break
        if time.time()-started>180:stop_reason='three_minute_training_cap';break
    model.eval().requires_grad_(False)
    digest=save_checkpoint(path,dict(status='TRAINED_PENDING_EVALUATION',state_dict=model.state_dict(),dimension=2,
        initialization_seed=SEED,noise_seed=NOISE_SEED,training_budgets=BUDGETS,training_updates=updates,optimizer='Adam',learning_rate=.005,
        objective_weights=dict(cover=1,discrepancy=2,clustering=.2,near_overlap=.5,union_cover=.5),
        reference_seeds={n:SEED+n for n in BUDGETS},training_iteration_rule='5000+4*step; cell t..t+3 and greedy t; fixed monitor t=7000',
        training_history=history,stop_reason=stop_reason,implementation='experiments/sho/fresh_spacefill_generator.py',
        implementation_sha256=src.sha('experiments/sho/fresh_spacefill_generator.py')))
    src.write_csv(OUT/'training_history.csv',history)
    summary=dict(updates=updates,stop_reason=stop_reason,initial_fixed_loss=initial,final_fixed_loss=last,
        training_seconds=time.time()-started,checkpoint=str(path),sha256=digest,protected_hashes=protected,
        protected_unchanged=all(src.sha(p)==h for p,h in protected.items()),
        evaluation_allowed=updates>=30 and last<initial*.995)
    (OUT/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='protected_hashes'},indent=2),flush=True)

if __name__=='__main__':run()
