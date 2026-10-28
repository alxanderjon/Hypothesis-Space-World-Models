"""One approved saved-hidden candidate head fit and native comparison; no recovery."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,math,traceback
sys.dont_write_bytecode=True;sys.set_int_max_str_digits(0)
ROOT=Path('/Users/alexanderrichardson/Documents/hypthesis-space-world-models');sys.path.insert(0,str(ROOT))
B=ROOT/'outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy';P=B/'exact_both_validation_original43';OUT=B/'model380_native_candidate'
MODEL='b6f5cc1fd6ee8f16c54a6ff5f57c0338e4571e2fefc5d50d5f2916e51de690ee';CERT='94aa54a2ed6b71696cdfbd443d03c81b4c78782b6dd5de7de5abc627045d6dba';SOLVER='2f96012afadf59b1de867bae5757fdb5ef9faf99b36797e8f70c7f54433b946d';DRIVER='416b2bac9926276fdbdd18e768c36b3c857929fffcee55590b753e8cc6f294b5'
if OUT.exists():raise RuntimeError('Fresh output required; no rerun')
OUT.mkdir();inputs={};done=False;adapter=None
counts=dict(center_objective=0,center_residual=0,head_basis_jacobian=0,exact_head_solver=0,candidate_objective=0,candidate_residual=0,center_derivatives=0,joint_solver=0,trust_updates=0,recovery_runs=0,rollout_steps=0)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protect(p,h=None):
 actual=sha(p)
 if h is not None and actual!=h:raise RuntimeError('ALERT hash mismatch: '+str(p))
 inputs[str(p)]=actual

def pack(v):
 if isinstance(v,F):return str(v.numerator)+'/'+str(v.denominator)
 if isinstance(v,dict):return {k:pack(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [pack(x) for x in v]
 if hasattr(v,'tolist'):return v.tolist()
 return v

def save(n,v):
 with (OUT/n).open('x') as f:json.dump(pack(v),f,indent=2,allow_nan=False)

try:
 protect(Path(__file__).resolve());guard=B.parent/'model380_native_integrity_before.json';protect(guard)
 for name,h in json.loads(guard.read_text())['files'].items():
  if sha(Path(name))!=h:raise RuntimeError('ALERT before baseline differs: '+name)
 protect(P/'artifact_manifest.json')
 for n,h in json.loads((P/'artifact_manifest.json').read_text())['sha256'].items():
  p=P/n
  if p.parent!=P:raise RuntimeError('ALERT artifact path')
  protect(p,h)
 for name,h in json.loads((B/'joint_exact_pilot/manifest.json').read_text())['input_sha256'].items():protect(Path(name),h)
 for p in (ROOT/'coordination').glob('*.md'):protect(p)
 auth=B/'model380_native_authorization.json';protect(auth);au=json.loads(auth.read_text());protect(B/'model380_native_candidate_proposal.md',au['proposal_sha256'])
 if au['direct_user_answer']!='approve' or au['model_sha256']!=MODEL or au['certificate_sha256']!=CERT:raise RuntimeError('ALERT authorization identity')
 protect(P/'pending_model.json',MODEL);protect(P/'exact_model_00380.json',CERT);protect(ROOT/'experiments/sho/recover_exact_both_original43.py',DRIVER)
 solver_path=ROOT/'experiments/sho/exact_rational_bvls.py';protect(solver_path,SOLVER)
 m=json.loads((P/'pending_model.json').read_text());state=json.loads((P/'diagnostic_state.json').read_text());profile=state['profile']
 if m['model_index']!=380 or m['model_call_status']!='completed' or m['prediction_half_sum_squares']!=0.0 or json.loads((P/'failure.json').read_text())['status']!='NUMERICAL_STOP':raise RuntimeError('ALERT saved terminal identity')
 if not json.loads((B/'model380_prediction_accounting/integrity.json').read_text())['unchanged']:raise RuntimeError('ALERT prior accounting integrity')
 import numpy as np
 import scipy
 import torch
 from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
 from experiments.sho.generic_update_residual_n3_opt_in import residual_vector,residual_and_jacobian
 from experiments.sho.run_companion6_stabilized_residual_live import bits_equal
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 env={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'torch_threads':torch.get_num_threads()}
 if env!=json.loads((P/'manifest.json').read_text())['environment']:raise RuntimeError('ALERT numerical environment differs')
 adapter=GenericSHOUpdateNetworkAdapter();lower=torch.from_numpy(adapter.lower.copy());upper=torch.from_numpy(adapter.upper.copy())
 layout=[dict(name='0.weight',shape=[4,2],start=0,stop=8),dict(name='0.bias',shape=[4],start=8,stop=12),dict(name='2.weight',shape=[2,4],start=12,stop=20),dict(name='2.bias',shape=[2],start=20,stop=22)]
 if adapter.layout!=layout or adapter.dimension!=22 or not (np.all(adapter.lower==-1.99) and np.all(adapter.upper==1.99)):raise RuntimeError('ALERT architecture/bounds differ')
 data=ROOT/'outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz';protect(data,'d699b6b7e6de4a0d39ca25db48c417d5109b248862e32b33a57528c0ceffc774')
 with np.load(data) as corpus:trajectory=torch.from_numpy(corpus['prefix_6'].copy())
 center=torch.tensor(m['center_phi']);h=torch.tensor(m['model_phi'][:12]);expected_r=torch.tensor(m['r'])
 if not bits_equal(center,profile['phi']) or not bits_equal(expected_r,profile['residual']) or not bits_equal(m['center_loss'],profile['loss']) or not bits_equal(m['center_loss'],5.522072828717873e-15):raise RuntimeError('ALERT center identity')
 if not bool(torch.isfinite(center).all() and torch.isfinite(h).all()) or not bool(((center>=lower)&(center<=upper)).all()) or not bool(((h>=lower[:12])&(h<=upper[:12])).all()):raise RuntimeError('ALERT saved feasibility')
 with torch.no_grad():
  counts['center_objective']+=1;center_loss=float(adapter.objective(center,trajectory))
  counts['center_residual']+=1;center_r=residual_vector(adapter,trajectory,center).detach()
 if not bits_equal(center_loss,m['center_loss']) or not bits_equal(center_r,expected_r) or not bits_equal(center_loss,float((center_r*center_r).sum()/center_r.numel())):raise RuntimeError('ALERT center native replay differs')
 save('center_replay.json',{'phi':center,'loss':center_loss,'residual':center_r,'bit_exact':True})
 zero=torch.cat((h,torch.zeros(10,dtype=torch.float64)))
 counts['head_basis_jacobian']+=1;zero_r,zero_J=residual_and_jacobian(adapter,trajectory,zero)
 A=zero_J[:,12:22];anchor_r=-trajectory[1:].reshape(-1)
 if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(anchor_r).all()):raise RuntimeError('TECHNICAL_BLOCKER nonfinite head input')
 singular=torch.linalg.svdvals(A)
 if not bool(torch.isfinite(singular).all()) or bool((singular<=0).any()):raise RuntimeError('TECHNICAL_BLOCKER invalid head singular value; no truncation')
 save('head_problem.json',{'hidden':h,'A':A,'anchor_residual':anchor_r,'lower':lower[12:22],'upper':upper[12:22],'singular_values':singular,'zero_residual':zero_r})
 namespace={};exec(compile(solver_path.read_bytes(),str(solver_path),'exec'),namespace)
 counts['exact_head_solver']+=1
 exact=namespace['exact_bvls']([[F(v) for v in row] for row in A.tolist()],[F(v) for v in anchor_r.tolist()],[F(v) for v in lower[12:22].tolist()],[F(v) for v in upper[12:22].tolist()],max_iterations=200)
 save('head_certificate.json',{'certificate':exact,'solver_sha256':SOLVER,'certificate_scope':'Exact rational problem before nearest binary64 head conversion.'})
 if not all(exact[k] is True for k in ('exact_feasible','exact_KKT','full_rank_exact')):raise RuntimeError('TECHNICAL_STOP missing certificate')
 head=torch.tensor([float(v) for v in exact['direction']],dtype=torch.float64);phi=torch.cat((h,head));active=torch.tensor(exact['active'])
 if not bool(torch.isfinite(phi).all()) or not bool(((phi>=lower)&(phi<=upper)).all()):raise RuntimeError('TECHNICAL_BLOCKER infeasible/nonfinite profile; no clipping')
 if not bits_equal(phi[:12],m['model_phi'][:12]):raise RuntimeError('ALERT hidden candidate changed')
 with torch.no_grad():
  counts['candidate_objective']+=1;loss=float(adapter.objective(phi,trajectory))
  counts['candidate_residual']+=1;r=residual_vector(adapter,trajectory,phi).detach()
 if not math.isfinite(loss) or not bool(torch.isfinite(r).all()):raise RuntimeError('TECHNICAL_BLOCKER nonfinite native profile')
 if not bits_equal(loss,float((r*r).sum()/r.numel())):raise RuntimeError('ALERT native/residual assembly differs')
 if not bool(((active!=-1)|(head==lower[12:22])).all() and ((active!=1)|(head==upper[12:22])).all()):raise RuntimeError('ALERT head active/bound mismatch')
 save('candidate_profile.json',{'phi':phi,'hidden':h,'loss':loss,'residual':r,'feasible':True,'native_MSE_bits':True,'hidden_proposal_bits':True,'active_bound_identity':True,'head_iterations':exact['iterations']})
 comparison='strictly_lower' if loss<center_loss else 'equal' if loss==center_loss else 'higher'
 result={'status':'ALERT_HANDOFF' if loss<1e-20 else 'FIXED_CANDIDATE_NATIVE_CHECK_COMPLETE','center_native_MSE':center_loss,'candidate_native_MSE':loss,'native_comparison':comparison,'actual_MSE_decrease':center_loss-loss,'actual_half_sum_decrease':(center_loss-loss)*center_r.numel()/2,'original_recorded_prediction':m['prediction_half_sum_squares'],'head_iterations':exact['iterations'],'head_exact_feasible':True,'head_exact_KKT':True,'head_full_rank':True,'candidate_feasible':True,'hidden_proposal_bits':True,'center_replay_bits':True,'handoff':loss<1e-20,'accepted_into_recovery':False,'work':counts,'adapter_calls':dict(adapter.calls),'environment':env,'limits':'One saved hidden proposal; strict native comparison only. No trust update, new joint model, recovery or rollout. Certificate before float conversion.'}
 save('result.json',result);done=True;print(json.dumps(result,indent=2),flush=True)
except BaseException as exc:
 save('failure.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'work':counts,'adapter_calls':dict(adapter.calls) if adapter is not None else None,'no_retry':True});raise
finally:
 checks={};errors={}
 for name,h in inputs.items():
  try:checks[name]=sha(Path(name))==h
  except BaseException as exc:errors[name]=str(exc)
 save('integrity.json',{'unchanged':not errors and all(checks.values()),'protocol_completed':done,'checks':checks,'errors':errors})
 save('manifest.json',{'input_sha256':inputs,'work':counts,'direct_approval':'approve','scope':'Single fixed-hidden exact head profile and native center/candidate comparison.'})
 save('artifact_manifest.json',{'sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 if errors or not all(checks.values()):raise RuntimeError('ALERT protected input changed')
