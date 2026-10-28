"""Approved fixed-array accounting of saved model380; no solver or network."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys,struct,traceback
sys.dont_write_bytecode=True;sys.set_int_max_str_digits(0)
R=Path('/Users/alexanderrichardson/Documents/hypthesis-space-world-models');B=R/'outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy';P=B/'exact_both_validation_original43';OUT=B/'model380_prediction_accounting'
MODEL_SHA='b6f5cc1fd6ee8f16c54a6ff5f57c0338e4571e2fefc5d50d5f2916e51de690ee';CERT_SHA='94aa54a2ed6b71696cdfbd443d03c81b4c78782b6dd5de7de5abc627045d6dba'
if OUT.exists():raise RuntimeError('Fresh output required; no rerun')
OUT.mkdir();base={};complete=False;float_replays=0

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protect(p,h=None):
 actual=sha(p)
 if h is not None and actual!=h:raise RuntimeError('ALERT provenance mismatch: '+str(p))
 base[str(p)]=actual

def pack(x):
 if isinstance(x,F):return str(x.numerator)+'/'+str(x.denominator)
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x

def dump(name,v):
 with (OUT/name).open('x') as f:json.dump(pack(v),f,indent=2,allow_nan=False)

def halfsum(v):return sum((x*x for x in v),F(0))/2

def formed(A,r,p):return [r[i]+sum((x*y for x,y in zip(row,p)),F(0)) for i,row in enumerate(A)]

def bit_scalar(x,y):return struct.pack('d',x)==struct.pack('d',y)

try:
 protect(Path(__file__).resolve());guard=B.parent/'model380_accounting_integrity_before.json';protect(guard)
 for name,h in json.loads(guard.read_text())['files'].items():
  if sha(Path(name))!=h:raise RuntimeError('ALERT protected baseline differs: '+name)
 protect(P/'artifact_manifest.json')
 for name,h in json.loads((P/'artifact_manifest.json').read_text())['sha256'].items():
  p=P/name
  if p.parent!=P:raise RuntimeError('ALERT artifact path')
  protect(p,h)
 if not json.loads((P/'integrity.json').read_text())['unchanged']:raise RuntimeError('ALERT prior integrity')
 for name,h in json.loads((B/'joint_exact_pilot/manifest.json').read_text())['input_sha256'].items():protect(Path(name),h)
 for p in (R/'coordination').glob('*.md'):protect(p)
 auth=B/'model380_accounting_authorization.json';protect(auth);au=json.loads(auth.read_text());proposal=B/'exact_both43_prediction_accounting_proposal.md';protect(proposal,au['proposal_sha256'])
 if au['direct_user_answer']!='approve' or au['model_sha256']!=MODEL_SHA or au['certificate_sha256']!=CERT_SHA:raise RuntimeError('ALERT approval identity')
 modelpath=P/'pending_model.json';certpath=P/'exact_model_00380.json';protect(modelpath,MODEL_SHA);protect(certpath,CERT_SHA)
 m=json.loads(modelpath.read_text());record=json.loads(certpath.read_text());cert=record['certificate'];failure=json.loads((P/'failure.json').read_text())
 if m['model_index']!=380 or m['model_call_status']!='completed' or m['kept_indices']!=list(range(22)) or m['fixed_hidden_indices'] or record['kept_indices']!=m['kept_indices'] or record['model_index']!=380 or m['exact_certificate_sha256']!=CERT_SHA or m['prediction_half_sum_squares']!=0.0 or failure['status']!='NUMERICAL_STOP':raise RuntimeError('ALERT saved terminal identity')
 if not all(cert[k] is True for k in ['exact_feasible','exact_KKT','full_rank_exact']) or not all(F(x)>0 for x in cert['full_LDL_pivots']):raise RuntimeError('ALERT saved certificate flags/rank')
 A=[[F(v) for v in row] for row in m['J']];r=[F(v) for v in m['r']];x=[F(v) for v in cert['direction']];lo=[F(v) for v in m['rounded_relative_lower']];hi=[F(v) for v in m['rounded_relative_upper']]
 if len(A)!=252 or any(len(row)!=22 for row in A) or len(r)!=252 or len(x)!=22:raise RuntimeError('ALERT shape')
 initial=halfsum(r);opt_res=formed(A,r,x);optcost=halfsum(opt_res)
 exactgradient=[sum((A[k][j]*opt_res[k] for k in range(len(r))),F(0)) for j in range(22)]
 kkt=[True if lo[j]==hi[j] else exactgradient[j]>=0 if x[j]==lo[j] else exactgradient[j]<=0 if x[j]==hi[j] else exactgradient[j]==0 for j in range(22)]
 if optcost!=F(cert['half_sum']) or exactgradient!=[F(v) for v in cert['gradient']] or not all(kkt) or not all(lo[j]<=x[j]<=hi[j] for j in range(22)):raise RuntimeError('ALERT direct residual certificate disagreement')
 raw=[F(v) for v in m['raw_solver_step']];realized=[F(v) for v in m['realized_model_step']];stored_res=[F(v) for v in m['model_residual']]
 rawres=formed(A,r,raw);realres=formed(A,r,realized)
 costs={'exact_rational_optimum':optcost,'saved_nearest_float_raw_step':halfsum(rawres),'saved_reconstructed_realized_step':halfsum(realres),'saved_rounded_model_residual':halfsum(stored_res)}
 predictions={k:initial-v for k,v in costs.items()}
 # One unchanged float64 replay of the saved reconstruction/residual/prediction.
 import torch
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 tensor=lambda k:torch.tensor(m[k],dtype=torch.float64)
 jt,rt,phi,L,U=tensor('J'),tensor('r'),tensor('center_phi'),tensor('box_lower'),tensor('box_upper')
 step=torch.tensor([float(v) for v in x],dtype=torch.float64);relL=L-phi;relU=U-phi
 eq=lambda a,b:bool(torch.equal(a.view(torch.int64),b.view(torch.int64)))
 if not eq(step,tensor('raw_solver_step')) or not eq(relL,tensor('rounded_relative_lower')) or not eq(relU,tensor('rounded_relative_upper')):raise RuntimeError('ALERT saved rounding/relative-bound replay differs')
 lower_hits=step==relL;upper_hits=step==relU
 model_phi=phi+step;model_phi=torch.where(lower_hits,L,model_phi);model_phi=torch.where(upper_hits,U,model_phi);p=model_phi-phi
 if not eq(model_phi,tensor('model_phi')) or not eq(p,tensor('realized_model_step')) or lower_hits.tolist()!=m['exact_lower_endpoint_reconstruction'] or upper_hits.tolist()!=m['exact_upper_endpoint_reconstruction']:raise RuntimeError('ALERT reconstruction replay differs')
 if not bool(((model_phi>=L)&(model_phi<=U)&(model_phi>=-1.99)&(model_phi<=1.99)).all()) or not bool(((step>=relL)&(step<=relU)).all()):raise RuntimeError('ALERT saved step feasibility differs')
 float_replays+=1;model_r=rt+jt@p;float_base=float(.5*(rt@rt));float_final=float(.5*(model_r@model_r));prediction=float(.5*((rt@rt)-(model_r@model_r)))
 if not eq(model_r,tensor('model_residual')) or not bit_scalar(float_base,m['zero_model_half_sum_squares']) or not bit_scalar(prediction,m['prediction_half_sum_squares']):raise RuntimeError('ALERT float residual/prediction replay differs')
 predictions['saved_float_prediction']=F(prediction)
 differences={}
 keys=list(predictions)
 for a,z in zip(keys,keys[1:]):differences[a+' -> '+z]=predictions[z]-predictions[a]
 complete=True
 dump('accounting_exact.json',{'baseline_half_sum':initial,'stage_half_sums':costs,'stage_predictions':predictions,'stage_prediction_changes':differences,'MSE_scale_predictions':{k:2*v/len(r) for k,v in predictions.items()},'certificate_direct_residual_checks':{'objective_equal':True,'gradient_equal':True,'feasible':True,'KKT':True,'saved_LDL_pivots_positive':True},'raw_step_feasible_in_rounded_relative_box':all(lo[j]<=raw[j]<=hi[j] for j in range(22)),'realized_step_relative_feasibility_descriptive':all(lo[j]<=realized[j]<=hi[j] for j in range(22)),'reconstructed_absolute_box_feasible':True,'float_residual_and_prediction_bits_match':True,'float_baseline_half_sum':float_base,'float_final_half_sum':float_final,'float_baseline_rounding_error':F(float_base)-initial,'float_final_rounding_error':F(float_final)-costs['saved_rounded_model_residual']})
 result={'status':'SAVED_MODEL_ACCOUNTING_COMPLETE','stages_half_sum_prediction_as_float':{k:float(v) for k,v in predictions.items()},'stage_prediction_signs':{k:('positive' if v>0 else 'negative' if v<0 else 'zero') for k,v in predictions.items()},'stage_prediction_changes_as_float':{k:float(v) for k,v in differences.items()},'baseline_half_sum_as_float':float(initial),'float_baseline_half_sum':float_base,'float_final_half_sum':float_final,'exact_certificate_recomputed_from_saved_residuals':True,'float_replay_bit_exact':True,'float_expression_replays':float_replays,'solver_calls':0,'network_evaluations':0,'derivative_evaluations':0,'recovery_runs':0,'rollout_steps':0,'limits':'Arithmetic on fixed saved arrays only; exact linear model gain does not establish native descent, convergence, success or validity of a changed acceptance/prediction rule.'}
 dump('result.json',result);print(json.dumps(result,indent=2),flush=True)
except BaseException as exc:
 dump('failure.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'float_expression_replays':float_replays,'solver_calls':0,'network_evaluations':0,'no_rerun_or_continuation':True});raise
finally:
 checks={};errors={}
 for name,h in base.items():
  try:checks[name]=sha(Path(name))==h
  except BaseException as exc:errors[name]=str(exc)
 dump('integrity.json',{'unchanged':complete and not errors and all(checks.values()),'checks':checks,'errors':errors})
 dump('manifest.json',{'input_sha256':base,'protocol':'Direct exact arithmetic on saved arrays and one unchanged float reconstruction/residual/prediction replay; zero solver or network calls.','float_expression_replays':float_replays,'solver_calls':0,'network_evaluations':0})
 dump('artifact_manifest.json',{'sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 if errors or not all(checks.values()):raise RuntimeError('ALERT integrity mismatch')
