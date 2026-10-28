"""Resume unexecuted gates after restoring omitted synthetic fixture geometry.

No operator is trained, modified or unlocked. Completed numerical gates stand.
"""
import json
import time
from pathlib import Path
import torch
from experiments.geometry.finish_learned_brent import Run,ROOT,OUT,dump
from experiments.geometry.finish_brent_integration import integrate
from experiments.geometry.load_completed_brent import load_completed_brent
from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry.train_division_operator import sha
from experiments.geometry.validate_locked_brent_linear import read


def main():
    torch.set_num_threads(1)
    previous=json.loads((OUT/'results.json').read_text())
    issue=json.loads((OUT/'first_failure.json').read_text())
    assert previous['failed_stage']=='F' and issue['case_id']==408 and issue['matching_requests']==[]
    issue.update(diagnosis='Harness dropped declared +4 ULP subpiece-boundary perturbation. Parent geometry alone cannot regenerate this deliberately synthetic input. No learned numerical component disagreed.',
                 correction='Regenerate base subpiece geometry, apply saved fixture boundary_shift_ULPs, reconstruct 129-point scan and rerun locked availability. No cached teacher cut is supplied to generation.',
                 authority='experiments/geometry/audit_branch_availability_false_positives.py:133-141',
                 original_subpiece_upper=0.33333333333333326,perturbed_subpiece_upper=0.3333333333333335)
    dump(OUT/'fixture_reconstruction_issue.json',issue)
    (OUT/'first_failure.json').rename(OUT/'initial_harness_failure.json')
    (OUT/'results.json').rename(OUT/'initial_run_results.json')
    (OUT/'consolidated_report.md').rename(OUT/'initial_run_report.md')
    r=Run.__new__(Run);r.start=time.monotonic()-previous['elapsed_seconds'];r.stage='F';r.last_detail={}
    r.status=previous['components'];r.trained=previous['newly_trained'];r.replays=previous['replays'];r.teacher_cache={}
    r.protected=json.loads((OUT/'protected_hashes.json').read_text())
    assert all(sha(Path(p))==h for p,h in r.protected.items())
    for p in OUT.glob('*_locked.json'):r.protected[str(p)]=sha(p)
    for p in OUT.glob('*.pt'):r.protected[str(p)]=sha(p)
    dump(OUT/'resume_protected_hashes.json',r.protected)
    r.engine,_=load_completed_brent()
    r.audit=authority.Audit.__new__(authority.Audit)
    r.cases=json.loads((ROOT/'exact_branch_switch_brent_audit/solve_inputs.json').read_text())
    r.splits={int(x['case_id']):x['group'] for x in read(ROOT/'brent_interpolation_eligibility_training/solve_splits.csv')}
    for folder,name in [('brent_interpolation_eligibility_training','new_fresh_candidates.json'),
                        ('brent_interpolation_branch_selector_training','new_fresh_candidates.json'),
                        ('subtraction_operator_training','fresh_candidates.json')]:
        for c in json.loads((ROOT/folder/name).read_text()):
            r.cases.append(dict(parent=c['parent'],axis=c['axis'],count=c['k'],left=c['c_left'],right=c['c_right'],teacher_cut=c['teacher_cut'],label=c['case']))
    assert len(r.cases)==875
    try:
        integrate(r,resume_regeneration=True)
    except Exception as exc:
        if not (OUT/'first_failure.json').exists():dump(OUT/'first_failure.json',dict(stage=r.stage,reason=repr(exc)))
        r.finish_report(exc);raise
    r.finish_report()


if __name__=='__main__':main()
