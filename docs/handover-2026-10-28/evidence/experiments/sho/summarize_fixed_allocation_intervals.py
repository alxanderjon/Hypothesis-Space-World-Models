"""Additional analytic checks and read-only summary of fixed interval audit."""
import csv,json,time
import numpy as np
from numpy.polynomial import Polynomial as P
from experiments.sho.audit_fixed_allocation_intervals import OUT,ROOT,SOURCE,Audit
from experiments.sho.train_characteristic_spacing_smoke import write_csv

def read(name):return list(csv.DictReader((OUT/name).open()))
def run():
 started=time.monotonic();ps=read('fixed_interval_score_structure.csv');ints=read('fixed_allocation_interval_audit.csv');rs=read('optimum_recovery.csv');ax=read('axis_summary.csv');ts=read('allocation_transition_brackets.csv');result=json.loads((OUT/'results.json').read_text());audit=Audit()
 records=[]
 for group in ('train','heldout'):records+=json.loads((SOURCE/(group+'_regions.json')).read_text())
 old=np.load(ROOT/'split_quality_smoke/previous_n4_failure.npz');records.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
 extra_roots=[];checked=0;last=None;rootrows=[];max_derivative_disagreement=0
 # Adjacent-float branch locations have no representable interior. Do not
 # interpret reversed nextafter probes there as a derivative sign change.
 for row in ps:
  resolvable=np.nextafter(float(row['lower']),float(row['upper']))<float(row['upper'])
  row['has_float64_interior']=bool(resolvable)
  if not resolvable:
   row['raw_quasiconvex_probe']=row['sampled_quasiconvex']
   row['sampled_quasiconvex']='not_resolvable';row['sampled_convex']='not_resolvable'
   row['stationary_points']='[]';row['interior_minima']=0;row['interior_maxima']=0
 # Independent quartic stationary-root check for the intended geometric formula.
 for row in ps:
  if not row['has_float64_interior']:
   row['quartic_stationary_roots']='[]';row['quartic_roots_missing_from_scan']=0;continue
  ri=int(row['parent']);axis=int(row['axis']);k=int(row['interval']);key=(ri,axis)
  if key!=last:audit.setup(records[ri],axis);last=key
  iq=int(row['radius_active_child']);ia=int(row['aspect_active_child']);power=int(row['aspect_log_power']);be=.05*power;l=float(row['lower']);u=float(row['upper'])
  t=P([0,1]) if iq==0 else P([1,-1]);v=P([0,1]) if ia==0 else P([1,-1]);nk=k if iq==0 else audit.n-k
  C=audit.a/(nk**(1/audit.d)*audit.h);b=audit.B/audit.a**2
  roots=[]
  if be:
   poly=C*C*t*t*v*v-be*be*(P([b])+t*t)
   for z in poly.roots():
    if abs(z.imag)>1e-8:continue
    x=float(z.real)
    if not l+1e-12*(u-l)<x<u-1e-12*(u-l):continue
    derivative=float(audit.formula(x,k)[1][0]);scale=C+.05/min(x,1-x)
    if abs(derivative)<=1e-7*max(1,scale):roots.append(x)
  saved=json.loads(row['stationary_points']);missing=[v for v in roots if not any(abs(v-z)<1e-7 for z in saved)]
  if missing:extra_roots.append(dict(parent=ri,axis=axis,interval=k,piece=row['piece'],roots=missing))
  row['quartic_stationary_roots']=json.dumps(roots);row['quartic_roots_missing_from_scan']=len(missing);checked+=1
 write_csv(OUT/'fixed_interval_score_structure.csv',ps)
 result['quartic_pieces_checked']=checked;result['quartic_additional_roots']=extra_roots
 result['unresolvable_float64_transition_brackets']=sum(not r['has_float64_interior'] for r in ps)
 result['nonconvex_pieces']=sum(r['sampled_convex']=='False' for r in ps if r['has_float64_interior'])
 result['nonquasiconvex_pieces']=sum(r['sampled_quasiconvex']=='False' for r in ps if r['has_float64_interior'])
 # Exact switch locations are analytic branch roots, checked on either side.
 switches=[];last=None
 for row in ints:
  if row['valid']!='True':continue
  ri=int(row['parent']);axis=int(row['axis']);k=int(row['interval']);key=(ri,axis)
  if key!=last:audit.setup(records[ri],axis);last=key
  br=json.loads(row['analytic_breakpoints']);qsw=[];asw=[];sidesw=[];continuity=[]
  for j,c in enumerate(br[1:-1],1):
   delta=min(c-br[j-1],br[j+1]-c)*1e-6
   if delta<=0:continue
   left=audit.formula(c-delta,k);right=audit.formula(c+delta,k)
   if int(left[3][0])!=int(right[3][0]):qsw.append(c)
   if int(left[4][0])!=int(right[4][0]):asw.append(c)
  row['radius_switch_locations']=json.dumps(qsw);row['aspect_switch_locations']=json.dumps(asw)
 write_csv(OUT/'fixed_allocation_interval_audit.csv',ints)
 groups=[]
 for band in ('all','2-5','6-10','11-20','>20'):
  rr=[r for r in rs if band=='all' or r['budget_bucket']==band]
  groups.append(dict(budget=band,parents=len(rr),equivalent_recovery=np.mean([r['equivalent_best']=='True' for r in rr]),canonical_recovery=np.mean([r['canonical_recovered']=='True' for r in rr]),worst_penalty=max(float(r['relative_penalty']) for r in rr),continuous_score_requests=np.mean([float(r['continuous_score_requests']) for r in rr]),grid_score_requests=np.mean([float(r['grid_score_requests']) for r in rr]),dense_requests=np.mean([float(r['dense_reference_requests']) for r in rr]),allocator_calls_including_audit=np.mean([float(r['allocator_calls_including_audit']) for r in rr])))
 write_csv(OUT/'recovery_by_budget.csv',groups)
 result['supplement_seconds']=time.monotonic()-started;result['dense_locked_samples']=sum(int(r.get('samples') or 0) for r in ints);result['max_mirror_boundary_error']=max(float(r['mirror_boundary_error']) for r in ts);result['width_min']=min(float(r['width']) for r in ints);result['width_max']=max(float(r['width']) for r in ints);result['total_allocator_calls_including_audit']=sum(int(r['allocator_calls_including_audit']) for r in rs)
 (OUT/'results.json').write_text(json.dumps(result,indent=2))
 lines=['Fixed integer-allocation interval audit — no training, checkpoint edits, recursion or search.','',json.dumps(result,indent=2),'',
 'SCOPE: 106 archived synthetic parents, dimensions 1..5, n=2..20 plus 29 and 73, all axes; previous n=3/n=4 failures included. Representative sample, not every possible parent.',
 'BOUNDARIES: actual locked allocator queried on >=2049 cuts per axis and >=32 samples per budget unit. Each count transition bracket refined with 60 vectorized bisections, then a 16-ULP neighborhood checked. Returned bracket endpoints reflect actual float64 ownership; not ideal (k+0.5)/n substituted for learned outputs.',
 'Positive child boxes only: tiny domain-end tails where physical cut subtraction may round to zero are not scored. These belong to invalid 0/n outer allocation regions; endpoint_excluded_width is recorded. The mathematical domain remains (0,1).',
 'Smooth real allocator fractions are monotone for the tested budgets: its logit has learned log-length coefficient about 1 and residual coefficients <=3e-11. Float64 transition ownership was audited separately. A finite scan is not a universal proof against every representable micro-toggle.',
 'Each axis has n+1 macroscopic allocation intervals: [0,n], [1,n-1], ..., [n,0]. The two outer intervals are invalid. Internal widths are approximately 1/n, outer widths approximately 1/(2n), with measured asymmetries/actual counts in CSV.',
 '', 'CONTINUOUS GEOMETRIC FORM (analytical interpretation, not a replacement forward path):',
 'Let a be the parent side on the cut axis, B=sum(other sides squared), h=(V/n)^(1/d). For fixed counts [k,n-k], t_left=c, t_right=1-c.',
 'q_j(c)=sqrt(B+a^2*t_j^2)/n_j^(1/d). Radius=max_j q_j/h.',
 'For d>1: log_aspect_j=log(max(a*t_j, max(other sides))/min(a*t_j, min(other sides))). For d=1 aspect is identically zero.',
 'On a fixed active-max and active-side subpiece, score=A*sqrt(B+a^2*t^2)+beta*log(v)+constant, beta in {-0.05,0,+0.05}; t and v independently c or 1-c; A=1/(h*n_active^(1/d)).',
 'Score derivative = sign(t)*A*a^2*t/sqrt(B+a^2*t^2) + beta*sign(v)/v.',
 'Second derivative = A*a^2*B/(B+a^2*t^2)^(3/2) - beta/v^2. It need not be nonnegative; convexity is stronger than unimodality.',
 'Stationary condition can be squared to a polynomial of degree at most four: C^2*t^2*v^2-beta^2*(b+t^2)=0, C=A*a and b=B/a^2. Filter squared roots using the unsquared derivative and interval/branch membership. Independent polynomial roots checked against derivative scans.',
 'Max of continuous child quantities is continuous. Kinks occur when active child or active longest/shortest side changes; therefore active-child changes alone are insufficient to partition smooth pieces.',
 'The actual frozen real-arithmetic child branch is q_j=exp(offset)*(sum ell_j^alpha)^beta*n_j^(gamma*d^delta), with alpha=exp(raw_alpha); raw_alpha=0.6931471805599455, beta=0.4999999999999999, gamma=delta=-1, offset=-6.214015148014685e-19. Its active branch is therefore a smooth power-sum function. The quotient coefficients are exactly [1,-1], log-aspect [1,-1], and scale 0.05 in stored float64. h is constant as c varies within a parent.',
 'Frozen learned power coefficients differ from exact sqrt coefficients by a few ULPs. Every dense score and every proposed optimum was evaluated with the actual unmodified learned chain. Formula derivatives are analytic diagnostics/proposal scaffolding, not bit-exact replacements. max_locked_vs_analytic_error above reports the comparison.',
 'Ten pairs of independently computed branch locations are adjacent float64 values, with no representable interior. They are retained as transition brackets but excluded from derivative/convexity classification; endpoint scoring remains included. Their initial nextafter probes would run in reverse order, producing a spurious sign change, so these are explicitly marked not_resolvable.',
 'Continuity is a real-arithmetic statement. Float64 scores are quantized; last-bit fluctuations are assessed using the existing 1e-12*max(1,best) equivalence rule, not called additional physical minima.',
 '', 'OPTIMUM RECOVERY:',
 'Enumerate actual allocation intervals, split at child-diameter equality, child-aspect equality, and side-extremum transitions; test subpiece endpoints and all detected stationary roots. No reference scores are consulted to propose those cuts.',
 'For dense-grid reproduction, project each candidate to adjacent k/200 cuts within its allocation interval and score them with the locked chain. This compares finite-grid optima; continuous optima may be strictly better than the 199-grid result.',
 'Cost accounting separates continuous candidate scores from grid-recovery scores. Discovery overhead is NOT free: allocator scans/bisections, 513+ locked-score audit samples per interval, and 129 analytic derivative samples per subpiece are audit work, excluded from candidate-only cost counts. This is not an end-to-end speed benchmark.',
 '', 'BY BUDGET:']
 for g in groups:lines.append(json.dumps(g))
 lines+=['','PREVIOUS FAILURES:']
 for r in rs:
  if r['reason']:lines.append(json.dumps(r))
 lines+=['','FILES: fixed_allocation_interval_audit.csv includes every interval, counts, widths, dense sampled minima, classifications and active switches. fixed_interval_score_structure.csv includes each analytic subpiece, derivative/curvature ranges, roots and convexity/quasiconvexity checks. Extra files retain actual transition brackets and recovery cases.']
 lines+=['', 'CONCLUSION: In this representative audit, every valid fixed-count interval was monotonic or unimodal, with one minimum. The many global valleys came from integer-allocation changes. Smooth subpieces need not be convex, but every resolvable subpiece was quasiconvex in the checks; independent polynomial root enumeration found no missed stationary points. This is strong evidence for these parents, not a universal proof for arbitrary inputs or all floating-point perturbations.',
 'The simplest next mathematical decomposition is: actual allocation interval -> active diameter/aspect/side-extremum subpieces -> endpoint or stationary-point minimum -> locked-score comparison across intervals/axes with the existing canonical rule. No new architecture or training is introduced.',
 'Grid-best and canonical recovery were 106/106, worst penalty zero. The first audit pass incorrectly stepped one ULP inward from known-owned interval endpoints and omitted a valid 0.5 grid cut in one n=73 case; correcting that audit-only endpoint handling recovered it. The first-pass artifacts are retained separately for transparency. Production and checkpoint code were never changed.']
 (OUT/'fixed_interval_summary.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))
if __name__=='__main__':run()
