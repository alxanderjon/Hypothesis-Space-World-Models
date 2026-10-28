"""Resume ONLY the existing two-query optimizer until its recorded plateau or pass gate.

No update cap, architecture change, interval construction change, or 4-query run.
"""
import copy,csv,json,math,time
from pathlib import Path
import numpy as np
import torch
from experiments.sho.train_fixed_interval_iterative_optimizer import (
 Controller,FrozenScore,geometry,rollout,authoritative,passed,SEED,CHECK_EVERY,GATES)
from experiments.sho.train_fixed_interval_cut_optimizer import features,score_predictions,metrics
from experiments.sho.audit_fixed_allocation_intervals import Audit,ROOT
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv
from experiments.sho.smoke_artifacts import save_checkpoint

SOURCE=ROOT/'fixed_interval_iterative_optimizer_smoke'
OUT=ROOT/'fixed_interval_iterative_optimizer_continued'
CHECKPOINT=SOURCE/'queries_2_best.pt'


def run():
 started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(SEED)
 OUT.mkdir(parents=True,exist_ok=False)
 old_manifest=json.loads((SOURCE/'candidate_manifest.json').read_text())
 assert sha(CHECKPOINT)==old_manifest['checkpoints'][0]['sha256']
 checkpoint=torch.load(CHECKPOINT,weights_only=False);assert checkpoint['queries']==2
 start_step=int(checkpoint['best_step']);assert start_step==int(checkpoint['steps'])
 protected={str(p):sha(p) for p in ROOT.glob('*/*.pt')}
 for name in ('train_fixed_interval_iterative_optimizer.py','audit_fixed_allocation_intervals.py','train_fixed_interval_cut_optimizer.py','validate_recursive_split_assembly.py','train_recursive_region_splitter_poc.py','train_finite_split_score_smoke.py'):
  path=Path('experiments/sho')/name;protected[str(path)]=sha(path)
 for p in SOURCE.glob('*_intervals.json'):protected[str(p)]=sha(p)
 data={g:json.loads((SOURCE/(g+'_intervals.json')).read_text()) for g in ('train','selection','heldout')}
 train,selection,test=[data[g] for g in ('train','selection','heldout')]
 tx=features(train);vx=features(selection);testx=features(test);td=geometry(train);vd=geometry(selection)
 audit=Audit();scorer=FrozenScore(audit);model=Controller();model.load_state_dict(checkpoint['state_dict'])
 reference=torch.tensor([r['minimum_score'] for r in train],dtype=torch.float64)
 counts={n:sum(r['n']==n for r in train) for n in range(2,21)}
 prob=np.array([1/counts[r['n']] for r in train]);prob/=prob.sum()
 # The old cosine schedule ended at its floor. Continue at that terminal LR;
 # do not restart or periodically cycle the old schedule.
 terminal_lr=3e-5+(.0015-3e-5)*.5*(1+math.cos(math.pi*(start_step-1)/6000))
 optimizer=torch.optim.Adam(model.parameters(),lr=terminal_lr)
 restored_optimizer=False
 if 'optimizer_state_dict' in checkpoint:
  optimizer.load_state_dict(checkpoint['optimizer_state_dict']);restored_optimizer=True
 rng=np.random.default_rng(SEED)
 if 'numpy_rng_state' in checkpoint:rng.bit_generator.state=checkpoint['numpy_rng_state'];rng_mode='restored'
 else:
  # Reproduce only cheap batch-index draws, not old training: one random validation
  # vector preceded the prior run's budget-balanced minibatch sampling.
  rng.random(len(selection))
  for _ in range(start_step):rng.choice(len(train),256,replace=True,p=prob)
  rng_mode='reconstructed original NumPy sampling stream at update 6000'
 old_history=list(csv.DictReader((SOURCE/'iterative_optimizer_training_history.csv').open()))
 bestkey=math.inf;anchor=math.inf;last_progress=0
 for r in old_history:
  if int(r['step'])>start_step:break
  bestkey=min(bestkey,float(r['selection_mean_relative_penalty']))
  if bestkey<anchor*.99:anchor=bestkey;last_progress=int(r['step'])
 best_step=start_step;best=copy.deepcopy(model.state_dict());best_path=CHECKPOINT;best_hash=sha(best_path)
 config=dict(source_checkpoint=str(CHECKPOINT),source_sha256=best_hash,start_update=start_step,queries=2,
  architecture_changed=False,optimizer_restored=restored_optimizer,optimizer_note='Original checkpoint has no optimizer state; Adam moments initialized fresh at previous terminal learning rate.' if not restored_optimizer else 'Restored saved Adam state.',
  learning_rate=terminal_lr,rng=rng_mode,objective=checkpoint['config']['loss'],training_sampling='Same budget-balanced probabilities, batch size 256, n=2..20.',
  max_additional_updates=None,plateau='Existing rule: 1200 total updates without >=1% cumulative improvement in best selection mean relative score penalty; minimum total update 2400.',
  inherited_best_mean_penalty=bestkey,inherited_plateau_anchor=anchor,inherited_last_progress=last_progress,validation_every=CHECK_EVERY,
  checkpoint_policy='Immediately save/hash every newly observed validation best, including optimizer/RNG/plateau state; no overwrite.',gates=GATES,protected_hashes=protected)
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 # Establish restoration fidelity and feedback execution equivalence before training.
 baseline,bz,bv=authoritative(model,audit,selection,vx,2)
 with torch.no_grad():dz,dv=rollout(model,vx,vd,scorer,2)
 assert np.max(abs(dz.numpy()-bz))<1e-13
 score_error=float(np.max(abs(dv.numpy()-bv)/np.maximum(1,abs(bv))));assert score_error<256*np.finfo(float).eps
 initial=metrics(baseline);assert abs(initial['mean_relative_penalty']-bestkey)<1e-13
 write_csv(OUT/'resume_validation.csv',[dict(**initial,source_mean_penalty=bestkey,forward_scaled_score_error=score_error,source_hash_verified=True)])
 manifest=[];history=[];total=start_step;stop_reason=None;latest_validation=baseline
 print(json.dumps(dict(event='resumed',update=total,initial_mean_penalty=bestkey,last_progress=last_progress,optimizer_restored=restored_optimizer,lr=terminal_lr)),flush=True)
 while True:
  total+=1
  ix=torch.from_numpy(rng.choice(len(train),256,replace=True,p=prob))
  zs,scores=rollout(model,tuple(x[ix] for x in tx),tuple(x[ix] for x in td),scorer,2)
  rel=(scores-reference[ix,None])/reference[ix,None];loss=rel.min(1).values.mean()+.05*rel.mean()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
  if total%CHECK_EVERY:continue
  latest_validation,_,_=authoritative(model,audit,selection,vx,2)
  m=metrics(latest_validation);key=m['mean_relative_penalty'];new_best=key<bestkey
  if new_best:bestkey=key;best_step=total;best=copy.deepcopy(model.state_dict())
  if bestkey<anchor*.99:anchor=bestkey;last_progress=total
  if new_best:
   best_path=OUT/f'best_update_{total:08d}.pt'
   payload=dict(state_dict=best,optimizer_state_dict=optimizer.state_dict(),numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),queries=2,
    status='UNPROMOTED_BEST_VALIDATION_CHECKPOINT',steps=total,best_step=best_step,additional_updates=total-start_step,source_checkpoint=str(CHECKPOINT),source_sha256=config['source_sha256'],
    plateau_state=dict(best_mean_penalty=bestkey,anchor=anchor,last_progress=last_progress),validation_metrics=m,config=config)
   best_hash=save_checkpoint(best_path,payload)
   manifest.append(dict(update=total,additional_updates=total-start_step,mean_relative_penalty=bestkey,checkpoint=str(best_path),sha256=best_hash))
   (OUT/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
  row=dict(update=total,additional_updates=total-start_step,loss=float(loss.detach()),new_best=new_best,best_update=best_step,best_mean_penalty=bestkey,plateau_anchor=anchor,last_progress_update=last_progress,updates_without_1pct_progress=total-last_progress,seconds=time.monotonic()-started,**m)
  history.append(row);write_csv(OUT/'two_query_continuation_history.csv',history)
  print(json.dumps(dict(event='validation',update=total,new_best=new_best,recovery=m['teacher_equivalent_recovery'],mean_penalty=key,worst_penalty=m['worst_relative_penalty'],last_progress=last_progress,seconds=row['seconds'])),flush=True)
  if passed(latest_validation):stop_reason='SELECTION_GATE_PASSED';break
  if total>=2400 and total-last_progress>=1200:stop_reason='VALIDATION_PLATEAU';break
 # Preserve the actual stopping state as well as the separately saved best state.
 stop_path=OUT/f'stop_update_{total:08d}.pt'
 stop_hash=save_checkpoint(stop_path,dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),queries=2,status='STOP_STATE',steps=total,best_step=best_step,best_checkpoint=str(best_path),best_sha256=best_hash,stop_reason=stop_reason,plateau_state=dict(best_mean_penalty=bestkey,anchor=anchor,last_progress=last_progress),config=config))
 model.load_state_dict(torch.load(best_path,weights_only=False)['state_dict'])
 results,zmat,vmat=authoritative(model,audit,test,testx,2);write_csv(OUT/'two_query_best_heldout.csv',results);np.savez_compressed(OUT/'two_query_best_history.npz',z=zmat,scores=vmat)
 selection_results,_,_=authoritative(model,audit,selection,vx,2)
 train_results,_,_=authoritative(model,audit,train,tx,2)
 success=bool(passed(results) and passed(selection_results))
 write_csv(OUT/'two_query_best_train.csv',train_results)
 breakdown=[]
 for field in ('dimension','budget','minimum_location','interval_kind'):
  for value in sorted(set(r[field] for r in results),key=str):breakdown.append(dict(split='heldout',grouping=field,group=value,**metrics([r for r in results if r[field]==value])))
 breakdown+=[dict(split='train',grouping='all',group='all',**metrics(train_results)),dict(split='selection',grouping='all',group='all',**metrics(selection_results))]
 write_csv(OUT/'two_query_best_breakdown.csv',breakdown)
 write_csv(OUT/'two_query_previous_failures.csv',[r for r in results if r['reason']])
 write_csv(OUT/'two_query_worst_cases.csv',sorted(results,key=lambda r:r['relative_penalty'],reverse=True)[:20])
 containment=[]
 for j in range(2):
  rr=score_predictions(audit,test,zmat[:,j]);assert np.all((zmat[:,j]>=0)&(zmat[:,j]<=1))
  assert all(r['allocation_preserved'] and r['inside_interval'] for r in rr)
  containment.append(dict(query=j+1,examples=len(rr),allocation_changes=0,outside_interval=0))
 write_csv(OUT/'all_query_containment.csv',containment)
 assert all(sha(p)==h for p,h in protected.items())
 result=dict(operator='fixed_interval_iterative_optimizer',query_stage=2,status='FROZEN_CANDIDATE' if success else 'DIAGNOSTIC_ONLY',passed=success,stop_reason=stop_reason,
  additional_updates=total-start_step,start_update=start_step,best_update=best_step,stop_update=total,last_progress_update=last_progress,plateau_anchor=anchor,updates_since_1pct_progress=total-last_progress,
  original_selection_mean_penalty=config['inherited_best_mean_penalty'],best_selection_mean_penalty=bestkey,optimizer_restored=restored_optimizer,optimizer_note=config['optimizer_note'],
  best_heldout=metrics(results),best_checkpoint=str(best_path),best_sha256=best_hash,stop_checkpoint=str(stop_path),stop_sha256=stop_hash,best_checkpoints_saved=len(manifest),protected_hashes_unchanged=True,
  four_query_training_run=False,recursion_run=False,sho_search_run=False,seconds=time.monotonic()-started)
 (OUT/'candidate_manifest.json').write_text(json.dumps(result,indent=2));(OUT/'two_query_continuation_summary.txt').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
