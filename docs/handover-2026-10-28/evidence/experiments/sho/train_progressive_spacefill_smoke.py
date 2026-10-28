"""Small standalone training smoke. Geometry only, no archived search data."""
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator, SCALES, discrepancy_rows, SEED
from experiments.sho.smoke_artifacts import save_checkpoint

OUT = Path('outputs/sho/stage1B/progressive_spacefill_smoke')
BUDGETS = (1000, 8600, 32768)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def monitor(model, reference, cycle):
    rows = []
    with torch.no_grad():
        for n in BUDGETS:
            complete, points = model.generate(n, 7000, 'cell')
            ds = discrepancy_rows(points)
            scales = []
            for k in SCALES:
                v = [r for r in ds if r['bins_per_axis'] == k]
                mu = n/k**2
                # Remove the unavoidable integer-count floor, not a Sobol target.
                floor = (mu % 1)*(1-mu % 1)/mu
                value = np.mean([r['count_fano']-floor+.5*n*r['grid_anchored_RMSE']**2 for r in v])
                scales.append(float(value))
            rows.append(dict(cycle=cycle, N=n, primary_monitor=max(scales),
                scale_objectives=json.dumps(scales),
                complete_cover=float(cKDTree(complete).query(reference)[0].max())))
    return rows


def run():
    started = time.monotonic(); torch.set_num_threads(1); torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path('outputs/sho/stage1B/grid_branch_bound/grid_branch_bound_candidate.json').read_text())
    paths = list(manifest['protected_hashes'])
    paths += [str(p) for p in Path('outputs/sho/stage1B').glob('*spacefill*/*.pt')]
    paths += [str(p) for p in Path('experiments/sho').glob('*spacefill*.py') if 'progressive' not in p.name]
    protected = {p: sha(p) for p in dict.fromkeys(paths) if Path(p).is_file()}
    config = dict(seed=SEED, budgets=BUDGETS, max_cycles=3, optimizer='Adam', learning_rate=.003,
        training_seconds_cap=180, scales=SCALES, occupancy_shifts=[0., .5],
        candidate_oversampling=8, blocks={'1000':32,'8600':128,'32768':256},
        primary='smooth maximum across scale/shift of expected block count + anchored-CDF squared-error increment',
        secondary='hole proxy; gradient projected not to oppose primary and capped at 0.1 of primary norm',
        inputs='N, iteration, pool; current multiscale occupancy, anchored-count residual fields, nearest-prefix distance',
        proposals='SHA256-context seeded PCG64 IID uniform only; no lattice/digital-net coordinates',
        minimum_separation='.4/sqrt(N); geometric anti-collapse scaffold, not a point sequence',
        checkpoint_selection='lowest mean across budgets of worst-scale training monitor; reject >10% monitor coverage regression',
        early_stop='two consecutive cycles without primary improvement, or 180 seconds at cycle boundary',
        supervision='geometry only; no Sobol, downstream labels, objective values, parameters, archived candidate coordinates')
    (OUT/'training_config.json').write_text(json.dumps(config, indent=2)+'\n')
    model = ProgressiveGenerator()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    # IID geometry reference; generated here, unrelated to any search/objective.
    reference = np.random.default_rng(SEED+1).random((16384, 2))
    monitors = monitor(model, reference, 0)
    initial = np.mean([r['primary_monitor'] for r in monitors])
    initial_cover = {r['N']:r['complete_cover'] for r in monitors}
    best = initial; best_cycle = 0; best_state = initial_state
    history = []; stale = 0; reason = 'modest three-cycle cap'
    print(json.dumps(dict(event='initial_monitor', primary=initial, rows=monitors)), flush=True)
    train_started = time.monotonic()
    for cycle in range(1, 4):
        for j, n in enumerate(BUDGETS):
            model.generate(n, 5000+cycle*10+j, 'greedy' if cycle%2 == 0 else 'cell', optimizer, history)
        measured = monitor(model, reference, cycle); monitors.extend(measured)
        value = float(np.mean([r['primary_monitor'] for r in measured]))
        feasible = all(r['complete_cover'] <= 1.1*initial_cover[r['N']] for r in measured)
        if value < best*(1-1e-4) and feasible:
            best = value; best_cycle = cycle; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        print(json.dumps(dict(event='cycle', cycle=cycle, primary=value, feasible=feasible,
                             seconds=time.monotonic()-train_started, rows=measured)), flush=True)
        if stale >= 2 or time.monotonic()-train_started > 180:
            reason = 'primary plateau' if stale >= 2 else 'short training time cap'; break
    model.load_state_dict(best_state)
    # Persist immediately, before any held-out evaluation/baseline comparison.
    checkpoint = OUT/'progressive_best_before_evaluation.pt'
    digest = save_checkpoint(checkpoint, dict(state_dict=model.state_dict(), config=config,
        best_cycle=best_cycle, primary_initial=initial, primary_best=best,
        history=history, monitors=monitors, status='TRAINED_PENDING_GEOMETRY_SCREEN'))
    write_csv(OUT/'progressive_training_history.csv', history)
    write_csv(OUT/'progressive_training_monitor.csv', monitors)
    summary = dict(checkpoint=str(checkpoint), sha256=digest, best_cycle=best_cycle,
        primary_initial=initial, primary_best=best, primary_improvement_fraction=1-best/initial,
        updates=len(history), stop_reason=reason, elapsed_seconds=time.monotonic()-started,
        protected_hashes=protected, protected_unchanged=all(sha(p)==h for p,h in protected.items()),
        source_sha256=sha('experiments/sho/progressive_spacefill_generator.py'),
        no_closed_loop=True, no_objective_or_downstream_labels=True)
    (OUT/'training_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='protected_hashes'}), flush=True)


if __name__ == '__main__': run()
