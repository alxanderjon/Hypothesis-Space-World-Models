"""Final reporting/label and reference checks; no training or new search."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import mpmath as mp
from experiments.sho.train_subpiece_stationary_solver import OUT,ROOT,SOURCE,StationarySolver,infer,root_metrics,evaluate_generic,derivative,Audit
from experiments.sho.train_characteristic_spacing_smoke import sha,write_csv

def run():
 torch.set_num_threads(1);mp.mp.dps=60
 m=json.loads((OUT/'candidate_manifest.json').read_text());model=StationarySolver();cp=m['checkpoints'][-1];model.load_state_dict(torch.load(cp['checkpoint'],weights_only=False)['state_dict']);model.eval();audit=Audit()
 for c in m['checkpoints']:assert sha(c['checkpoint'])==c['sha256']
 rows=json.loads((OUT/'stage_4_train.json').read_text());p,z=infer(model,rows);train=root_metrics(rows,p,z)
 retention=[]
 for stage in (1,2):
  rs=json.loads((OUT/f'stage_{stage}_heldout.json').read_text());p,z=infer(model,rs);retention.append(dict(source_stage=stage,**root_metrics(rs,p,z)))
 write_csv(OUT/'beta_zero_retention.csv',retention)
 labelcheck=[]
 for stage in (3,4):
  for group in ('train','selection','heldout'):
   rs=json.loads((OUT/f'stage_{stage}_{group}.json').read_text());res=[];multiple=0
   for r in rs:
    multiple+=r['root_count']>1
    if not r['exists']:continue
    c=mp.mpf(r['root_decimal']);t=c if r['t']==0 else 1-c;v=c if r['v']==0 else 1-c
    A,a,B,beta=[mp.mpf(r[k]) for k in ('A','a','B','beta')];st=1 if r['t']==0 else -1;sv=1 if r['v']==0 else -1
    d=st*A*a*a*t/mp.sqrt(B+a*a*t*t)+sv*beta/v;res.append(float(abs(d)/max(mp.mpf(1),A*a,abs(beta))))
   labelcheck.append(dict(stage=stage,group=group,root_examples=len(res),multi_root_examples=multiple,max_high_precision_scaled_residual=max(res,default=0.)))
 assert max(r['max_high_precision_scaled_residual'] for r in labelcheck)<1e-35
 write_csv(OUT/'teacher_label_verification.csv',labelcheck)
 # Independent reconstruction using the original audit's endpoint/stationary candidates.
 ap=ROOT/'fixed_allocation_interval_audit';parents=[]
 for group in ('train','heldout'):parents+=json.loads((SOURCE/(group+'_regions.json')).read_text())
 old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz');parents.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
 ir=[r for r in csv.DictReader((ap/'fixed_allocation_interval_audit.csv').open()) if r['valid']=='True'];got={(r['parent'],r['axis'],r['interval']):r for r in csv.DictReader((OUT/'audited_interval_assembly.csv').open())};verification=[]
 byaxis={}
 for r in ir:byaxis.setdefault((int(r['parent']),int(r['axis'])),[]).append(r)
 for (ri,axis),rr in byaxis.items():
  audit.setup(parents[ri],axis);cuts=[];spans=[]
  for r in rr:
   ca=np.unique([float(r['lower']),float(r['upper'])]+json.loads(r['analytic_breakpoints'])+json.loads(r['stationary_points']));start=len(cuts);cuts.extend(ca);spans.append((start,len(cuts)))
  values,_=audit.scores(cuts)
  for r,(a,b) in zip(rr,spans):
   best=float(values[a:b].min());observed=float(got[(r['parent'],r['axis'],r['interval'])]['minimum_score']);err=abs(best-observed)
   verification.append(dict(parent=ri,axis=axis,interval=r['interval'],original_solver_best=best,reconstructed_best=observed,error=err,equivalent=bool(err<=1e-12*max(1,best))))
 assert all(r['equivalent'] for r in verification)
 write_csv(OUT/'deterministic_reference_verification.csv',verification)
 breakdown=[]
 assembled=list(csv.DictReader((OUT/'audited_interval_assembly.csv').open()))
 for field in ('dimension','budget'):
  for value in sorted(set(r[field] for r in assembled),key=int):
   rr=[r for r in assembled if r[field]==value];breakdown.append(dict(grouping=field,group=value,intervals=len(rr),recovery=float(np.mean([r['minimum_recovered']=='True' for r in rr])),worst_relative_penalty=max(float(r['relative_penalty']) for r in rr)))
 write_csv(OUT/'audited_interval_breakdown.csv',breakdown)
 m['label_verification']=labelcheck;m['final_training_metrics']=train;m['final_beta_zero_retention']=retention;m['analytic_reference_reconstruction']=dict(intervals=len(verification),equivalent=len(verification),max_error=max(r['error'] for r in verification))
 (OUT/'candidate_manifest.json').write_text(json.dumps(m,indent=2))
 breakdown_root=list(csv.DictReader((OUT/'stationary_solver_breakdown.csv').open()))
 lines=['Supervised generic stationary solver: DIAGNOSTIC_ONLY; hard gate FAILED.',
 'Training consumed only newly generated synthetic coefficients/orientations/intervals. No SHO trajectories, parameters, dynamics, losses, or search labels were read for training. The assembled test uses only the previously audited synthetic box geometry.',
 'All four curriculum stages warm-started from the preceding saved checkpoint. Checkpoints were saved and hashed before final evaluation. No locked weights or deterministic discovery code changed. No iterative scoring queries occur in solver inference: one two-head neural forward supplies existence and location.',
 'Important: beta=0 with A>0,a>0,B>=0 has no strict interior stationary point. Stages 1 and 2 validate no-root classification, not root-location learning. Stage 3 trained same-orientation t/v cases; opposite orientations were held out. Stage 4 trained all four orientations; complete coefficient bands, wider absolute scales and narrow intervals were held out.',
 'Root-location loss is applied only to true-root labels. Training: BCE existence plus five times normalized root MSE. Float64 shared two-layer 64-unit tanh body with an existence logit and sigmoid location head. Inputs are coefficients (including dimensionless transforms), orientations and bounds only.',
 'Generic functions can have two stationary roots, unlike the observed audited subpieces. Example A=a=B=1,beta=0.1,t=c,v=1-c has roots 0.113531904559840499 and 0.845071854302510128. Label is the lowest-score interior stationary root, choosing leftmost in an exact tie; endpoints remain explicit candidates. Multi-root counts are reported.',
 'Targets use 60-decimal analytic/high-precision calculation. Independent high-precision derivative residual checks are in teacher_label_verification.csv. Final root error/residual metrics include every true-root example, including existence false negatives; no filtering to easy correctly detected roots.',
 'Arbitrary coefficients need not map to valid child boxes. Generic minimum tests evaluate their two formula terms with the locked finite-sum module and locked comparison selector (common translation for signed scores). Full audited geometric tests use the complete unchanged locked split-score operator. These two checks are reported separately.',
 '', 'CURRICULUM:']
 for s in m['stages']:lines.append(json.dumps(s))
 lines+=['','FINAL MODEL TRAINING FIT:',json.dumps(train),'','FINAL MODEL BETA=0 RETENTION:']
 for r in retention:lines.append(json.dumps(r))
 lines+=['','FINAL GENERALIZATION BREAKDOWN:']
 for r in breakdown_root:
  if r['stage']=='4':lines.append(json.dumps(r))
 lines+=['','ASSEMBLED FIXED-INTERVAL TEST:',json.dumps(m['assembly']),json.dumps(m['analytic_reference_reconstruction'])]
 for r in breakdown:lines.append(json.dumps(r))
 lines+=['','DIAGNOSIS:',
 'Both existence classification and root localization remain inadequate. Broad-range heldout root errors and derivative residuals are materially above numerical precision. Endpoint candidates rescue many cases, so minimum recovery alone overstates stationary-solver accuracy.',
 'The complete fixed-interval assembly misses 370 of 4356 minima, with worst relative geometry penalty 0.162492%. Therefore the learned solver is not promoted and deterministic active-subpiece discovery/optimization remain unchanged.',
 'This was a finite supervised smoke curriculum; it does not establish impossibility for other stationary solvers. No automatic training extension, architecture replacement, recursion or search follows.',
 '', 'CHECKPOINTS/HASHES:']
 for c in m['checkpoints']:lines.append(json.dumps(c))
 (OUT/'stationary_solver_summary.txt').write_text('\n'.join(lines)+'\n')
 print('training',json.dumps(train));print('retention',json.dumps(retention));print('labels',json.dumps(labelcheck));print('reference',m['analytic_reference_reconstruction']);print('dimensions',[r for r in breakdown if r['grouping']=='dimension'])
 print('endpoint',[r for r in breakdown_root if r['stage']=='4' and r['group']=='endpoint_near'])
if __name__=='__main__':run()
