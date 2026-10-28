"""Requested read-only selector quick check of the failed small offset model."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import train_branch_switch_location as harness
from experiments.geometry.train_branch_switch_ulp_offset import OUT, OffsetModel, feature_row, shift
from experiments.geometry.train_exponent_log2_operator import ROOT,sha


def run():
    torch.set_num_threads(1)
    result=json.loads((OUT/'training_result.json').read_text())
    checkpoint=torch.load(result['checkpoint'],weights_only=False);plan=checkpoint['plan']
    saved=json.loads((ROOT/'branch_switch_location_residual_training/location_cases.json').read_text())
    fresh=json.loads((OUT/'fresh_candidate_states.json').read_text());rows=saved+fresh
    xx=np.array([feature_row(r['parent'],r['axis'],r['k'],r['c_left'],r['c_right'],
                            float(np.clip(r['boundary'],r['c_left'],r['c_right']))) for r in rows])
    classes=np.array(plan['outputs']);left=np.array([r['c_left'] for r in rows]);right=np.array([r['c_right'] for r in rows])
    boundary=np.clip(np.array([r['boundary'] for r in rows]),left,right)
    class Adapter(OffsetModel):
        def __init__(self):super().__init__(xx.shape[1],len(classes))
        def locate(self,L,R,unused):
            assert np.array_equal(L,left) and np.array_equal(R,right)
            with torch.no_grad():offsets=classes[self(torch.from_numpy(xx)).argmax(1).numpy()]
            return np.array([shift(b,k) for b,k in zip(boundary,offsets)])
    model=Adapter();model.load_state_dict(checkpoint['state_dict']);model.eval()
    cuts=model.locate(left,right,None)
    folder=OUT/'selector_replay';folder.mkdir(exist_ok=False)
    harness.json_write(folder/'location_cases.json',rows)
    harness.json_write(folder/'protected_hashes.json',json.loads((OUT/'protected_hashes.json').read_text()))
    measured=harness.location_metrics(rows,cuts,'ulp_predictions')
    baseline=harness.location_metrics(rows,boundary,'boundary')
    harness.json_write(folder/'quick_result.json',dict(checkpoint=result['checkpoint'],sha256=result['sha256'],
        updates=result['stop_update'],train_cases=plan['fit_cases'],metrics=[measured,baseline,baseline,baseline]))
    old_folder,old_class=harness.OUT,harness.BranchLocation
    try:
        harness.OUT=folder;harness.BranchLocation=Adapter
        harness.diagnostic_replay()
    finally:
        harness.OUT=old_folder;harness.BranchLocation=old_class
    replay=json.loads((folder/'final_result.json').read_text())
    keep=('available_branch_cases','availability_mismatches','branch_candidate_identity_changes',
          'all_branch_subpiece_winner_changes','all_branch_subpiece_outside_equivalence',
          'cached_branch_replacements','cached_branch_insertions','downstream','fixture',
          'all_protected_hashes_unchanged','protected_files','replay_seconds')
    summary={k:replay[k] for k in keep}
    summary.update(status='DIAGNOSTIC_REPLAY_ONLY',
        scope='594 saved/new branch-positive subpieces; cached 106 archived + 8 existing fresh parent candidate pools; explicit original d2 N8 fixture. Fresh 44 evaluated at subpiece level. No recursion, no regeneration of new full parent pools.',
        invalid_predicted_scan_locations=int(((cuts<left)|(cuts>right)).sum()),
        no_clamping_or_repair=True)
    harness.json_write(folder/'final_result.json',summary)
    result.update(status='DIAGNOSTIC_ONLY_STOP',promoted=False,downstream_quick_check=summary,
                  stop_reason='Small classifier fails exact-offset/conflict-group/fresh gates. Training stopped without model enlargement.',
                  original_checkpoint_hash_still_matches=sha(Path(result['checkpoint']))==result['sha256'])
    harness.json_write(OUT/'final_result.json',result)
    print((OUT/'final_result.json').read_text(),flush=True)


if __name__=='__main__':run()
