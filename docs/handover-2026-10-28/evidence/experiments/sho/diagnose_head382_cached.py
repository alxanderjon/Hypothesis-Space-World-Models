"""Approved cached-only one float head replay and one exact rational reference."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,sys,traceback,time
sys.dont_write_bytecode=True;sys.set_int_max_str_digits(0)
ROOT=Path('/Users/alexanderrichardson/Documents/hypthesis-space-world-models');B=ROOT/'outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy';P=B/'joint_exact_validation_original43';OUT=B/'head382_cached_diagnosis'
HEAD_SHA='00d31d03ae87fff7bcd8fefb1c4be77cfb8718caf79878d1a5b87b3507737852'
FLOAT_SHA='3aa1f04390420bb83b19cc8d22961589df7c643c3bb920298298a289c73c8281'
EXACT_SHA='2f96012afadf59b1de867bae5757fdb5ef9faf99b36797e8f70c7f54433b946d'
if OUT.exists():raise RuntimeError('Fresh output required; no rerun')
OUT.mkdir();base={};complete=False;float_calls=exact_calls=0;started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protect(p,h=None):
 actual=sha(p)
 if h is not None and actual!=h:raise RuntimeError('ALERT provenance mismatch: '+str(p))
 base[str(p)]=actual

def pack(v):
 if isinstance(v,F):return str(v.numerator)+'/'+str(v.denominator)
 if hasattr(v,'detach'):return v.detach().tolist()
 if isinstance(v,dict):return {k:pack(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [pack(x) for x in v]
 return v

def dump(name,v):
 with (OUT/name).open('x') as f:json.dump(pack(v),f,indent=2,allow_nan=False)

def metrics(A,r,x,lo,hi):
 residual=[r[k]+sum((A[k][j]*x[j] for j in range(len(x))),F(0)) for k in range(len(r))]
 g=[sum((A[k][j]*residual[k] for k in range(len(r))),F(0)) for j in range(len(x))]
 feasible=all(lo[j]<=x[j]<=hi[j] for j in range(len(x)))
 valid=[g[j]>=0 if x[j]==lo[j] else g[j]<=0 if x[j]==hi[j] else g[j]==0 for j in range(len(x))]
 return {'gradient':g,'half_sum':sum((v*v for v in residual),F(0))/2,'feasible':feasible,'KKT':feasible and all(valid),'KKT_invalid_indices':[j for j,v in enumerate(valid) if not v]}
try:
 protect(Path(__file__).resolve())
 guard=B.parent/'head382_diagnosis_integrity_before.json';protect(guard)
 for name,h in json.loads(guard.read_text())['files'].items():
  if sha(Path(name))!=h:raise RuntimeError('ALERT full before baseline mismatch: '+name)
 protect(P/'artifact_manifest.json')
 for name,h in json.loads((P/'artifact_manifest.json').read_text())['sha256'].items():
  p=P/name
  if p.parent!=P:raise RuntimeError('ALERT unsafe artifact path')
  protect(p,h)
 if not json.loads((P/'integrity.json').read_text())['unchanged']:raise RuntimeError('ALERT saved run integrity')
 for name,h in json.loads((B/'joint_exact_pilot/manifest.json').read_text())['input_sha256'].items():protect(Path(name),h)
 for p in (ROOT/'coordination').glob('*.md'):protect(p)
 auth=B/'head382_diagnosis_authorization.json';protect(auth);authorization=json.loads(auth.read_text())
 proposal=B/'exact_head382_diagnosis_proposal.md';protect(proposal,authorization['proposal_sha256'])
 if authorization['direct_user_answer']!='approve' or authorization['head_sha256']!=HEAD_SHA:raise RuntimeError('ALERT authorization mismatch')
 headpath=P/'pending_head_profile.json';protect(headpath,HEAD_SHA);head=json.loads(headpath.read_text());failure=json.loads((P/'failure.json').read_text())
 if head['status']!='pending' or head['profile_index']!=382 or failure['status']!='TECHNICAL_SOLVER_CYCLE' or '_bounded_residual_least_squares' not in failure['traceback']:raise RuntimeError('ALERT cached failure identity')
 if len(head['A'])!=252 or any(len(row)!=10 for row in head['A']) or len(head['anchor_residual'])!=252 or head['head_lower']!=[-1.99]*10 or head['head_upper']!=[1.99]*10:raise RuntimeError('ALERT saved shape/bounds')
 import torch
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 floatpath=ROOT/'experiments/sho/generic_update_residual_n3_opt_in.py';protect(floatpath,FLOAT_SHA);source=floatpath.read_text()
 tree=ast.parse(source,filename=str(floatpath));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bounded_residual_least_squares')
 # Compile the original function AST unchanged; no imports of network/runtime code.
 ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(floatpath),'exec'),ns);float_solver=ns[node.name]
 exactpath=ROOT/'experiments/sho/exact_rational_bvls.py';protect(exactpath,EXACT_SHA);exactsource=exactpath.read_text();ens={};exec(compile(exactsource,str(exactpath),'exec'),ens);exact_solver=ens['exact_bvls']
 dump('solver_sources.json',{'float_sha256':FLOAT_SHA,'float_function_ast':ast.dump(node,include_attributes=False),'exact_sha256':EXACT_SHA,'exact_source':exactsource})
 matrix=torch.tensor(head['A']);residual=torch.tensor(head['anchor_residual']);lower=torch.tensor(head['head_lower']);upper=torch.tensor(head['head_upper']);zero=torch.zeros(10)
 inputs=[x.clone() for x in (matrix,residual,lower,upper,zero)]
 complete=True
 # Map observations to pinned original source statements.
 labels={}
 markers={'key = (tuple(status.tolist()), direction.numpy().tobytes())':'iteration_enter','raise RuntimeError("exact active-set cycle")':'repeat_detected','delta = trial - direction':'face_solved','if blockers:':'blockers_ready','if violations:':'violations_ready','status[coordinate] = 0':'release','if not bool(((direction >= step_lower) & (direction <= step_upper)).all()):':'return_checks'}
 for i,line in enumerate(source.splitlines(),1):
  if node.lineno<=i<=node.end_lineno and line.strip() in markers:labels[i]=markers[line.strip()]
 if len(labels)!=len(markers):raise RuntimeError('ALERT float trace markers differ')
 keys=('iterations','status','direction','trial','step_lower','step_upper','free','lower_active','upper_active','singular','blockers','alpha','projected_model_gradient','violations','coordinate')
 events=[]
 def trace_float(frame,event,arg):
  if frame.f_code is not float_solver.__code__:return None
  if event=='line' and frame.f_lineno in labels:
   events.append({'event':labels[frame.f_lineno],'line':frame.f_lineno,'locals':{k:pack(frame.f_locals[k]) for k in keys if k in frame.f_locals}})
  return trace_float
 caught=None
 try:
  sys.settrace(trace_float);float_calls+=1;float_solver(matrix,residual,zero,lower,upper)
 except RuntimeError as exc:caught={'type':type(exc).__name__,'error':str(exc)}
 finally:sys.settrace(None)
 dump('float_trace.json',{'events':events,'source_sha256':FLOAT_SHA,'line_labels':labels,'exception':caught,'solver_calls':float_calls})
 if caught!={'type':'RuntimeError','error':'exact active-set cycle'}:raise RuntimeError('ALERT original head cycle did not replay exactly')
 if not all(torch.equal(a,b) for a,b in zip(inputs,(matrix,residual,lower,upper,zero))):raise RuntimeError('ALERT input mutated')
 print('HEAD_CYCLE_REPRODUCED events='+str(len(events)),flush=True)
 A=[[F(v) for v in row] for row in head['A']];rr=[F(v) for v in head['anchor_residual']];lo=[F(v) for v in head['head_lower']];hi=[F(v) for v in head['head_upper']]
 exact_events=[];exactlabels={}
 for i,line in enumerate(exactsource.splitlines(),1):
  if line.strip().startswith('delta=[trial[j]-x[j]'):exactlabels[i]='face_solved'
  if line.strip()=='if violations:':exactlabels[i]='violations_ready'
 def trace_exact(frame,event,arg):
  if frame.f_code is not exact_solver.__code__:return None
  if event=='line' and frame.f_lineno in exactlabels:exact_events.append({'event':exactlabels[frame.f_lineno],'line':frame.f_lineno,'locals':{k:pack(frame.f_locals[k]) for k in ('iteration','status','x','trial','free','active','g','violations') if k in frame.f_locals}})
  return trace_exact
 try:
  sys.settrace(trace_exact);exact_calls+=1;sol=exact_solver(A,rr,lo,hi,max_iterations=200)
 finally:
  sys.settrace(None);dump('exact_trace.json',{'events':exact_events,'solver_calls':exact_calls})
 dump('exact_solution.json',sol)
 if not all(sol[k] is True for k in ('exact_feasible','exact_KKT','full_rank_exact')):raise RuntimeError('ALERT missing exact certificate')
 raw=torch.tensor([float(x) for x in sol['direction']]);rounded=metrics(A,rr,[F(v) for v in raw.tolist()],lo,hi)
 if not bool(torch.isfinite(raw).all()) or not rounded['feasible']:raise RuntimeError('ALERT exact solution float conversion infeasible')
 fr=residual+matrix@raw
 dump('float_conversion.json',{'head':raw.tolist(),'exact_metrics_after_rounding':rounded,'float64_linear_half_sum':float(.5*(fr@fr)),'float64_linear_MSE':float((fr*fr).mean()),'native_network_evaluations':0,'rounding_policy':'Direct nearest-binary64 conversion; zero anchor, original head box; no clipping or tolerance.'})
 # Compare only faces visited by the two authorized solves; no extra face solve.
 faces=[e['locals'] for e in events if e['event']=='face_solved'];exactfaces=[e['locals'] for e in exact_events if e['event']=='face_solved'];matches=[]
 for index,fl in enumerate(faces):
  same=[x for x in exactfaces if x['status']==fl['status']]
  for ex in same:
   xf=[F(v) for v in fl['trial']];xe=[F(v) for v in ex['trial']];fm=metrics(A,rr,xf,lo,hi);em=metrics(A,rr,xe,lo,hi)
   matches.append({'float_face_event_index':index,'float_iteration':fl['iterations'],'exact_iteration':ex['iteration'],'status':fl['status'],'float_trial':fl['trial'],'exact_trial':ex['trial'],'float_trial_exact_metrics':fm,'exact_trial_metrics':em,'max_trial_difference_as_float':float(max(abs(a-b) for a,b in zip(xf,xe)))})
 dump('matching_face_comparison.json',{'matches':matches,'additional_face_solves':0,'note':'Only traces from the two declared solver invocations. Metrics are exact residual/gradient evaluation of saved face trials, not new solves.'})
 result={'status':'HEAD_CYCLE_REPLAYED_AND_EXACT_REFERENCE_CERTIFIED','float_solver_calls':float_calls,'exact_solver_calls':exact_calls,'network_evaluations':0,'recovery_runs':0,'rollout_steps':0,'exact_iterations':sol['iterations'],'exact_feasible':True,'exact_KKT':True,'full_rank_exact':True,'exact_active':sol['active'],'exact_linear_half_sum_as_float':float(sol['half_sum']),'float_converted_head_feasible':rounded['feasible'],'rounded_exact_KKT':rounded['KKT'],'float64_linear_MSE':float((fr*fr).mean()),'matching_face_pairs':len(matches),'float_trace_events':len(events),'elapsed_seconds':time.monotonic()-started,'limitations':'Cached linear head problem only. No native neural loss, handoff, root-cause attribution or recovery claim without evidence review; exact certificate is pre-rounding.'}
 dump('result.json',result);print(json.dumps(result,indent=2),flush=True)
except BaseException as exc:
 sys.settrace(None);dump('failure.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'float_calls':float_calls,'exact_calls':exact_calls,'no_retry_or_fallback':True});raise
finally:
 checks={};errors={}
 for name,h in base.items():
  try:checks[name]=sha(Path(name))==h
  except BaseException as exc:errors[name]=str(exc)
 unchanged=complete and not errors and all(checks.values())
 dump('integrity.json',{'unchanged':unchanged,'checks':checks,'errors':errors,'full_protected_guard':'coordinator-owned'})
 dump('manifest.json',{'input_sha256':base,'float_solver_calls':float_calls,'exact_solver_calls':exact_calls,'network_evaluations':0,'protocol':'One unchanged cached float64 head trace and one pinned exact rational reference; metrics on existing face trials only, no further solve.'})
 dump('artifact_manifest.json',{'sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 if errors or not all(checks.values()):raise RuntimeError('ALERT final integrity mismatch')
