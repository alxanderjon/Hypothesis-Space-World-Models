"""Summarize the fixed continuous-extension experiment without rerunning search."""
import json,statistics
from pathlib import Path
import numpy as np
from experiments.sho.validate_continuous_surrogate_edirect import OUT,DATA

def main():
 r=json.loads((OUT/'report.json').read_text());rows=r['cases'];data=np.load(DATA)
 valid=[x for x in rows if 'matrix_error' in x];table=[]
 for category in np.unique(data['categories']):
  a=[x for x in rows if x['category']==category]
  table.append(f"| {category} | {np.sum(data['categories']==category)} | {len(a)} | {sum(x['n1_success'] for x in a)} | {sum(x.get('final_loss',float('inf'))<1e-20 for x in a)} |")
 case=next(x for x in rows if x['case']=='companion_6');calls=sum(x['continuous_calls'] for x in rows)
 summary=dict(status=r['status'],tested=len(rows),n1_pass=sum(x['n1_success'] for x in rows),
   n2_pass=sum(x.get('n2_gate',False) for x in rows),n3_pass=sum(x.get('final_loss',float('inf'))<1e-20 for x in rows),
   n1_evaluations_total=sum(x['n1_evaluations'] for x in rows),n1_evaluations_median=statistics.median(x['n1_evaluations'] for x in rows),
   n2_evaluations_total=sum(x.get('n2_evaluations',0) for x in rows),n3_evaluations_total=sum(x.get('n3_evaluations',0) for x in rows),
   continuous_calls=calls,continuous_physical_evaluations=sum(x['continuous_evaluations'] for x in rows),
   duplicate_rejections=sum(x['duplicate_rejections'] for x in rows),physical_duplicates=sum(x['duplicate_physical_evaluations'] for x in rows),
   surrogate_evaluations=sum(x['surrogate_evaluations'] for x in rows),optimizer_starts=9*calls,
   optimizer_nonconverged_starts=sum(x['optimizer_nonconverged_starts'] for x in rows),
   worst_matrix_error=max(x['matrix_error'] for x in valid),worst_rollout_error=max(x['rollout_max_error'] for x in valid))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 text=f'''# Continuous-surrogate E-DIRECT extension

Status: **{r['status']}**. No learned component was trained or modified.

## Scope and surrogate

`continuous_surrogate_edirect.py` reuses the original cell functions and substitutes only `_greedy_proposal` through private function globals. Boundary pools, cell ownership, radius-based sizes, grouping/hull/improvement, ordinary/largest-cell scoring, initial design, physical objective, gate and N2/N3 policies remain unchanged. The old circular/spherical finite-radius candidate stage is never called by the extension.

Input to the surrogate is a normalized point in `[0,1]^d`. Its fitted state is the evaluated normalized sites, their `log10(max(loss,1e-300))` values, and a cKDTree. Predictions use up to 24 nearest sites with bandwidth equal to the farthest queried distance, floored at `1e-12`; weights are `exp(-9*(distance/bandwidth)^2)+1e-15`. Output is their weighted-average log loss. There is no new fit, normalization, or objective. Changing neighbors/bandwidth makes this a piecewise-smooth predictor rather than a globally differentiable model.

## Optimizer and integration

Each greedy update uses SciPy bounded Powell from the current best site plus 2d fixed scrambled-Sobol starts. Every start searches the full `[0,1]^d` domain, corresponding to physical `[-1.99,1.99]^4` in these tests. Settings fixed before validation: `xtol=1e-10`, `ftol=1e-12`, `maxiter=100`, `maxfev=2000` per start. No teacher K or physical objective is supplied to this optimizer.

Choose the lowest returned surrogate value, breaking ties by start index. Reject an exact evaluated-site duplicate; otherwise the existing N1 loop evaluates the physical objective once. No radius restriction or finite-stencil fallback is used. This is multistart local minimization, **not certified global minimization**. Budget-terminated starts remain explicitly recorded, with no hidden retry or second optimizer.

## Targeted checks

Five unit tests passed, including exact-duplicate rejection and acceptance of an immediately adjacent float. Inherited 2D boundary/cell/scoring outputs remain bit-exact; the continuous greedy proposal intentionally differs from the old substitute. Repeated actual surrogate minimization controls in d=2 and d=4 produce identical traces and off-grid, finite, in-bounds proposals.

Case 7: {case['n1_evaluations']} physical N1 evaluations versus 5,000 previously; {case['continuous_calls']} continuous calls; handoff loss {case['n1_loss']:.17g} versus 0.0003099082254362297 previously. Teacher contained: {case['teacher_contained']}. N2 calls {case.get('n2_evaluations')}; N3 calls {case.get('n3_evaluations')}; final loss {case.get('final_loss')}; matrix Frobenius error {case.get('matrix_error')}; 4,096-step rollout max error {case.get('rollout_max_error')}. Exact full case replay: {case.get('full_case_replay_exact',False)}.

## Prepared suite and completed results

| Category | Prepared | Tested | N1 passes | Final loss passes |
|---|---:|---:|---:|---:|
'''+'\n'.join(table)+f'''

Statistics for primary runs (excluding the extra 408-evaluation case-7 deterministic replay):

```json
{json.dumps(summary,indent=2)}
```

The prepared data contain stable identifiable non-companion matrices and the original SHO companion subset. Synthetic observations expose both state components; all use the existing mean-squared one-step transition residual without rescaling. Per-case entrywise errors, evaluation counts and rollout errors are in `report.json`. Every completed search has sample arrays, physical-evaluation events and continuous-call traces including all starts, returned endpoints, scores, convergence flags and duplicate decisions. Fitted states reconstruct from the saved sample history and each call's `site_count`; internal Powell line-search query sequences are not claimed as recorded.

First failure: {r['first_failure']}. Unexecuted cases are not counted as passes. Validation runtime including controls and case replay: {r['seconds']:.2f} seconds. All {r['checkpoint_count']} checkpoint SHA-256 hashes are unchanged.

## Remaining deterministic debt

Multistart Powell is temporary optimization scaffolding. Its finite start set and evaluation caps provide no global guarantee. The local Gaussian smoother, capped probe budget, exact duplicate policy, search bookkeeping and existing handoff/local-box construction remain deterministic. N2/N3 are the previously used deterministic finish adapter, unmodified here. This experiment does not claim that these stages are newly learned or that the surrogate matches the paper's kriging model.
'''
 (OUT/'REPORT.md').write_text(text);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
