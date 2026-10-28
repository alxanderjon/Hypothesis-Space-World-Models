"""Exactly one authorized 60-update replay, followed by reproduction checks."""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from experiments.sho.smoke_artifacts import save_checkpoint

PRIOR = Path('outputs/sho/stage1B/spacefill_generator_smoke')
OUT = Path('outputs/sho/stage1B/spacefill_downstream_screen')


def run():
    torch.set_num_threads(1)
    summary = json.loads((PRIOR/'summary.json').read_text())
    source = Path('experiments/sho/run_stage1b_spacefill_generator_smoke.py')
    archived = OUT/'original_spacefill_smoke.py'
    assert not archived.exists(), 'Replay already started; do not rerun training'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == summary['implementation_sha256']
    shutil.copyfile(source, archived)
    spec = importlib.util.spec_from_file_location('original_spacefill_smoke', archived)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for path, digest in summary['protected_hashes'].items():
        assert original.base._sha256(Path(path)) == digest
    model, history = original.train([1000,8600,32768])
    # Persist immediately, before either replay validation or downstream work.
    pending = OUT/'spacefill_replay_pending_verification.pt'
    digest = save_checkpoint(pending, dict(status='TRAINED_PENDING_REPLAY_VERIFICATION',
        state_dict=model.state_dict(), seed=original.SEED, dimension=2, training_budgets=[1000,8600,32768],
        training_updates=60, optimizer='Adam',lr=.006, training_history=history,
        implementation=str(archived), implementation_sha256=original.base._sha256(archived)))
    original.base._write_csv(OUT/'replayed_training_history.csv',history)
    saved_history = list(csv.DictReader((PRIOR/'training_history.csv').open()))
    comparisons = []
    for i, (expected, actual) in enumerate(zip(saved_history, history)):
        for key in ('step','N','estimated_training_cover','normalized_loss','anti_cluster'):
            a, b = float(expected[key]), float(actual[key])
            comparisons.append(dict(kind='training',row=i,field=key,expected=a,actual=b,
                absolute_error=abs(a-b),exact=a==b,passed=bool(np.isclose(a,b,rtol=1e-12,atol=1e-15))))
    assert len(history)==len(saved_history)==60
    t = (np.arange(1000)+.5)/1000
    reference = np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),
        np.random.default_rng(original.SEED+1000000).random((262144,2))))
    saved_geometry = list(csv.DictReader((PRIOR/'spacefill_generator_geometry.csv').open()))
    arrays, geometry = {}, []
    for expected in saved_geometry:
        if expected['generator'] != 'learned':
            continue
        n, state = int(expected['N']), int(expected['source_state_index'])
        points, interior = model.generate(n,[0.,0.],[1.,1.])
        arrays[f'points_{n}'], arrays[f'interior_{n}'] = points, interior
        # Each generated set is durable before its evaluation begins.
        coord_path=OUT/f'spacefill_probes_N{n}.npz'
        assert not coord_path.exists()
        np.savez_compressed(coord_path,points=points,interior=interior,source_state_index=state)
        coord_path.with_suffix('.sha256').write_text(original.base._sha256(coord_path)+'  '+coord_path.name+'\n')
        actual, _ = original.metrics(points,interior,reference,n,'learned',state)
        geometry.append(actual)
        for key, value in actual.items():
            if key in ('record_type','generator'):
                assert expected[key]==value
                continue
            pairs = zip(json.loads(expected[key]),json.loads(value)) if key=='face_radii' else [(float(expected[key]),float(value))]
            for a,b in pairs:
                comparisons.append(dict(kind='geometry',row=n,field=key,expected=a,actual=b,
                    absolute_error=abs(a-b),exact=a==b,passed=bool(np.isclose(a,b,rtol=1e-12,atol=1e-15))))
        print(f'Replay geometry verified N={n}',flush=True)
    original.base._write_csv(OUT/'replay_verification.csv',comparisons)
    result=dict(passed=all(r['passed'] for r in comparisons),all_metrics_exact=all(r['exact'] for r in comparisons),
        compared_values=len(comparisons),max_absolute_error=max(r['absolute_error'] for r in comparisons),
        original_weights_byte_identity_verifiable=False, pending_checkpoint_sha256=digest)
    (OUT/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError('Replay mismatch; downstream prohibited; see replay_verification.csv')
    recovered=torch.load(pending,weights_only=False)
    recovered.update(status='RECOVERED_VERIFIED_PENDING_DOWNSTREAM',replay_verification=result)
    path=OUT/'spacefill_generator_recovered.pt'
    result['recovered_checkpoint_sha256']=save_checkpoint(path,recovered)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
