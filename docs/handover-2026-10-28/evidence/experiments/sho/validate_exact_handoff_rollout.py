"""One approved unchanged rollout of the saved exact-joint handoff candidate."""
from pathlib import Path
import hashlib,json,sys,struct,traceback
sys.dont_write_bytecode=True
ROOT=Path('/Users/alexanderrichardson/Documents/hypthesis-space-world-models');sys.path.insert(0,str(ROOT))
N=ROOT/'outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy';P=N/'joint_exact_pilot';OUT=N/'exact_handoff_rollout'
CANDIDATE_SHA='9e8336787866a1d554d37694a1cec62a8bc983534453aa3b0f0652630b203865'
PROPOSAL_SHA='bc5d700c2f1b2418b5b9124c8f35691d570143ac8615ffd2f596dadab3e20f70'
if OUT.exists():raise RuntimeError('fresh output required; no rerun')
OUT.mkdir();base={};complete=False;rollout_calls=0;objective_calls=0

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,h=None):
 actual=sha(p)
 if h is not None and actual!=h:raise RuntimeError('ALERT provenance mismatch '+str(p))
 base[str(p)]=actual

def dump(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def finite_json(value):
 import math
 if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':str(value)}
 if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
 return value

try:
 record(Path(__file__).resolve());record(N/'exact_rollout_authorization.json')
 auth=json.loads((N/'exact_rollout_authorization.json').read_text());candidate=P/'ALERT_handoff_candidate.json';proposal=N/'exact_handoff_rollout_proposal.md'
 if auth['direct_user_answer']!='approve' or auth['candidate_sha256']!=CANDIDATE_SHA or auth['proposal_sha256']!=PROPOSAL_SHA or Path(auth['candidate'])!=candidate or Path(auth['approved_proposal'])!=proposal:raise RuntimeError('ALERT authorization differs')
 record(candidate,CANDIDATE_SHA);record(proposal,PROPOSAL_SHA)
 record(P/'artifact_manifest.json')
 for name,h in json.loads((P/'artifact_manifest.json').read_text())['sha256'].items():
  q=P/name
  if q.parent!=P:raise RuntimeError('ALERT artifact path')
  record(q,h)
 manifest=json.loads((P/'manifest.json').read_text())
 for key in ('input_sha256','approval_input_sha256'):
  for name,h in manifest[key].items():record(Path(name),h)
 if not json.loads((P/'integrity.json').read_text())['unchanged']:raise RuntimeError('ALERT pilot integrity not established')
 guard=ROOT/'outputs/sho/n2_n3_basin_local_discrimination_20261013/exact_rollout_integrity_before.json';record(guard);guard_files=json.loads(guard.read_text())['files']
 for name in ('experiments/sho/recover_generic_update_targeted.py','experiments/sho/generic_update_network_adapter.py'):
  q=ROOT/name;record(q,guard_files[str(q)])
 for q in (ROOT/'coordination').glob('*.md'):record(q,guard_files[str(q)])
 data=ROOT/'outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz';record(data,manifest['input_sha256'][str(data)])
 saved=json.loads(candidate.read_text());failure=json.loads((P/'failure.json').read_text())
 if failure['status']!='ALERT_HANDOFF_CANDIDATE' or saved!=failure['last_completed_profile']:raise RuntimeError('ALERT candidate identity differs')
 complete=True
 import numpy as np
 import torch
 from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
 from experiments.sho.recover_generic_update_targeted import rollout_diagnostic
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 class CapturingAdapter(GenericSHOUpdateNetworkAdapter):
  def rollout(self,phi,initial_state,steps=4096):
   global rollout_calls
   if rollout_calls!=0 or steps!=4096:raise RuntimeError('ALERT extra or changed rollout request')
   rollout_calls+=1
   predicted=super().rollout(phi,initial_state,steps)
   self.saved_prediction=predicted.detach().numpy().copy()
   return predicted
 adapter=CapturingAdapter();phi=np.asarray(saved['phi'],dtype=np.float64);phi_before=phi.copy()
 if adapter.dimension!=22 or adapter.layout!=manifest['layout'] or not np.all(adapter.lower==-1.99) or not np.all(adapter.upper==1.99):raise RuntimeError('ALERT architecture/bounds differ')
 if not np.isfinite(phi).all() or not ((phi>=adapter.lower)&(phi<=adapter.upper)).all():raise RuntimeError('ALERT candidate not finite/in bounds')
 with np.load(data) as corpus:trajectory=corpus['prefix_6'].copy();teacher=corpus['teachers'][6].copy()
 with torch.no_grad():objective_calls+=1;loss=float(adapter.objective(phi,trajectory))
 exact_replay=struct.pack('d',loss)==struct.pack('d',saved['loss'])
 dump('native_replay.json',{'native_transition_MSE':loss,'saved_MSE':saved['loss'],'bit_exact_replay':exact_replay,'feasible':True,'phi':phi.tolist(),'objective_calls':objective_calls})
 if not exact_replay or not loss<1e-20:raise RuntimeError('ALERT native handoff replay differs; rollout not run')
 rollout=rollout_diagnostic(adapter,trajectory,teacher,phi)
 if not np.array_equal(phi.view(np.uint64),phi_before.view(np.uint64)):raise RuntimeError('ALERT parameters mutated')
 if rollout_calls!=1 or adapter.calls['rollout_steps']!=4096 or adapter.calls['objective']!=1 or adapter.calls['derivative_oracle']!=0:raise RuntimeError('ALERT unexpected validation call counts')
 predicted=adapter.saved_prediction
 if predicted.shape!=(4097,2) or not np.array_equal(predicted[0].view(np.uint64),trajectory[-1].view(np.uint64)):raise RuntimeError('ALERT rollout length/start differs')
 np.savez(OUT/'rollout_states.npz',predicted=predicted,initial_state=trajectory[-1],phi=phi_before)
 result={'status':'EXPERIMENTAL_CASE_RECOVERY_VALIDATED' if rollout['rollout_finite'] else 'ROLLOUT_NONFINITE_FAILURE','native_transition_MSE':loss,'handoff':True,'rollout':rollout,'existing_contract_satisfied':bool(rollout['rollout_finite']),'model':'original22parameter2→4→2SiLU','case':'companion_6','N2_branch':[5,4],'strategy':'experimental joint-model variable projection with exact rational inner solves','parameters_bitwise_unchanged':True,'rollout_calls':rollout_calls,'native_objective_replays':objective_calls,'adapter_calls':dict(adapter.calls),'initial_state':'prefix_6[-1]','teacher_role':'teachers[6] used only by existing offline continuation-error calculation; no optimizer/refit','success_contract':'native MSE<1e-20 AND finite4096-step rollout; errors/bit equality have no additional threshold','limitations':'One saved candidate/trajectory/start. No general reliability claim, production promotion or extra optimization.'}
 dump('result.json',finite_json(result));print(json.dumps(finite_json(result),indent=2),flush=True)
except BaseException as exc:
 dump('failure.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'objective_calls':objective_calls,'rollout_calls':rollout_calls,'no_retry_or_continuation':True});raise
finally:
 checks={};errors={}
 for name,h in base.items():
  try:checks[name]=sha(Path(name))==h
  except BaseException as exc:errors[name]=str(exc)
 unchanged=complete and not errors and all(checks.values())
 dump('integrity.json',{'unchanged':unchanged,'baseline_coverage_complete':complete,'checks':checks,'errors':errors})
 dump('manifest.json',{'input_sha256':base,'protocol':'One exact native scalar replay, then one existing rollout_diagnostic; subclass only captures the returned states from the unchanged parent rollout. No arithmetic/acceptance change or second rollout.','objective_calls':objective_calls,'rollout_calls':rollout_calls,'full_protected_guard':'coordinator-owned'})
 dump('artifact_manifest.json',{'sha256':{q.name:sha(q) for q in OUT.iterdir() if q.is_file()}})
 if errors or not all(checks.values()):raise RuntimeError('ALERT integrity changed')
