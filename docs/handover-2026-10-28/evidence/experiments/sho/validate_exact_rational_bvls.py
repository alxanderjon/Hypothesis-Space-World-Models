from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys,time,traceback,math
sys.dont_write_bytecode=True;sys.set_int_max_str_digits(0)
root=Path('/Users/alexanderrichardson/Documents/hypthesis-space-world-models');basepath=root/'outputs/sho/n2_n3_basin_local_discrimination_20261013/variable_projection_strategy';out=basepath/'exact_bvls_cached_validation'
if out.exists():raise RuntimeError('fresh output required')
out.mkdir();base={};records=[];started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(x):
 if isinstance(x,F):return str(x.numerator)+'/'+str(x.denominator)
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x
def dump(name,obj):
 with (out/name).open('x') as f:json.dump(pack(obj),f,indent=2,allow_nan=False)
def protect(p):base[str(p)]=sha(p)
protect(Path(__file__).resolve())
source=Path(__file__).with_name('exact_rational_bvls.py');protect(source)
if base[str(source)]!='2f96012afadf59b1de867bae5757fdb5ef9faf99b36797e8f70c7f54433b946d':raise RuntimeError('ALERT kernel hash differs')
namespace={};exec(compile(source.read_bytes(),str(source),'exec'),namespace);solver=namespace['exact_bvls'];dump('solver_source.json',{'sha256':sha(source),'source':source.read_text()})
import fractions,torch
protect(Path(fractions.__file__));torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
def verify_folder(folder):
 manifest=folder/'artifact_manifest.json';protect(manifest)
 for name,h in json.loads(manifest.read_text())['sha256'].items():
  p=folder/name
  if p.parent!=folder or sha(p)!=h:raise RuntimeError('ALERT input artifact mismatch '+str(p))
  base[str(p)]=h
 if not json.loads((folder/'integrity.json').read_text())['unchanged']:raise RuntimeError('ALERT saved integrity not verified')
for name in ('joint_pilot','joint_cycle_trace','joint_cycle_reference','joint_refined_pilot','joint358_cycle_trace','joint358_cycle_reference'):verify_folder(basepath/name)
for p in (root/'coordination').glob('*.md'):protect(p)
protect(basepath/'exact_bvls_cached_protocol.json')
def exact_metrics(A,r,x,lo,hi):
 residual=[r[k]+sum((A[k][j]*x[j] for j in range(len(x))),F(0)) for k in range(len(r))]
 g=[sum((A[k][j]*residual[k] for k in range(len(r))),F(0)) for j in range(len(x))]
 feasible=all(lo[j]<=x[j]<=hi[j] for j in range(len(x)))
 kkt=[True if lo[j]==hi[j] else g[j]>=0 if x[j]==lo[j] else g[j]<=0 if x[j]==hi[j] else g[j]==0 for j in range(len(x))]
 return {'half_sum':sum((v*v for v in residual),F(0))/2,'gradient':g,'feasible_relative_box':feasible,'KKT_valid':feasible and all(kkt),'KKT_invalid_indices':[j for j,z in enumerate(kkt) if not z]}
try:
 for model,pilot,trace in ((353,'joint_pilot','joint_cycle_trace'),(358,'joint_refined_pilot','joint358_cycle_trace')):
  begin=time.monotonic();meta=json.loads((basepath/pilot/'pending_model.json').read_text());events=json.loads((basepath/trace/'trace.json').read_text())['events']
  events=[e.get('locals',e) for e in events]
  bounds=next(x for x in events if 'step_lower' in x and 'step_upper' in x)
  lo0=bounds['step_lower'];hi0=bounds['step_upper'];assert all(e['step_lower']==lo0 and e['step_upper']==hi0 for e in events if 'step_lower' in e)
  assert meta['model_index']==model and not meta['fixed_hidden_indices'] and meta['kept_indices']==list(range(22))
  A=[[F(v) for v in row] for row in meta['J']];r=[F(v) for v in meta['r']];lo=[F(v) for v in lo0];hi=[F(v) for v in hi0]
  print('EXACT_START',model,flush=True)
  solution=solver(A,r,lo,hi,max_iterations=200)
  dump(f'model{model}_exact_solution.json',solution)
  # Same binary64 rounding and exact-bound-endpoint reconstruction as existing caller.
  J=torch.tensor(meta['J']);rt=torch.tensor(meta['r']);phi=torch.tensor(meta['center_phi']);L=torch.tensor(meta['box_lower']);U=torch.tensor(meta['box_upper']);raw=torch.tensor([float(v) for v in solution['direction']]);relL=L-phi;relU=U-phi
  assert relL.tolist()==lo0 and relU.tolist()==hi0
  candidate=phi+raw;candidate=torch.where(raw==relL,L,candidate);candidate=torch.where(raw==relU,U,candidate);realized=candidate-phi
  finite=all(bool(torch.isfinite(v).all()) for v in (raw,candidate,realized));absolute_feasible=bool(((candidate>=L)&(candidate<=U)&(candidate>=-1.99)&(candidate<=1.99)).all())
  raw_metrics=exact_metrics(A,r,[F(v) for v in raw.tolist()],lo,hi);real_metrics=exact_metrics(A,r,[F(v) for v in realized.tolist()],lo,hi)
  zero=sum((v*v for v in r),F(0))/2;float_res=rt+J@realized;float_initial=float(.5*(rt@rt));float_final=float(.5*(float_res@float_res))
  conversion={'raw_step':raw.tolist(),'model_phi':candidate.tolist(),'realized_model_step':realized.tolist(),'finite':finite,'absolute_local_and_original_box_feasible':absolute_feasible,'raw_exact_metrics':raw_metrics,'realized_exact_metrics':real_metrics,'float64_initial_half_sum':float_initial,'float64_realized_half_sum':float_final,'float64_prediction_half_sum':float_initial-float_final,'native_network_evaluations':0,'rounding_policy':'Existing exact relative-bound equality reconstruction; no clipping or tolerance.'}
  dump(f'model{model}_float_conversion.json',conversion)
  summary={'model':model,'iterations':solution['iterations'],'exact_feasible':solution['exact_feasible'],'exact_KKT':solution['exact_KKT'],'full_rank_exact':solution['full_rank_exact'],'exact_initial_half_sum_as_float':float(zero),'exact_optimum_half_sum_as_float':float(solution['half_sum']),'raw_and_candidate_finite':finite,'raw_feasible':raw_metrics['feasible_relative_box'],'realized_relative_box_feasible':real_metrics['feasible_relative_box'],'absolute_candidate_feasible':absolute_feasible,'raw_KKT_after_rounding':raw_metrics['KKT_valid'],'realized_KKT_after_rounding':real_metrics['KKT_valid'],'float64_prediction_half_sum':float_initial-float_final,'active':solution['active'],'elapsed_seconds':time.monotonic()-begin}
  records.append(summary);print('EXACT_RESULT',json.dumps(summary),flush=True)
  if not finite or not absolute_feasible or not raw_metrics['feasible_relative_box'] or not math.isfinite(float_initial-float_final) or not float_initial-float_final>0:raise RuntimeError('CACHED_FLOAT_INTERFACE_NOT_USABLE')
 dump('result.json',{'status':'TWO_CACHED_PROBLEMS_EXACTLY_CERTIFIED','cases':records,'solver_calls':2,'network_evaluations':0,'recovery_runs':0,'elapsed_seconds':time.monotonic()-started,'limits':'Exact certificates apply only to rational optima of two saved rounded-displacement linear subproblems. Float outputs are checked separately; no nonlinear acceptance or handoff is tested.'})
except BaseException as exc:
 dump('failure.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'completed_cases':records,'no_fallback_or_rerun':True});raise
finally:
 checks={s:sha(Path(s))==h for s,h in base.items()}
 dump('integrity.json',{'unchanged':all(checks.values()),'checks':checks})
 dump('manifest.json',{'input_sha256':base,'python':sys.version,'torch':torch.__version__,'arithmetic':'Python Fraction exact rational LDL and active-set arithmetic; residual-based exact KKT independently recomputed without Gram reuse. Binary64-rounded displacement bounds preserved.','max_iterations_per_problem':200,'network_evaluations':0})
 dump('artifact_manifest.json',{'sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 if not all(checks.values()):raise RuntimeError('ALERT integrity mismatch')
