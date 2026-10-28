"""Read-only frozen-pipeline diagnosis at the last saved failing-case state."""
import json
import numpy as np
import torch

from experiments.sho import run_stage1b_fifo_vs_ranker as screen
from experiments.sho import run_stage1b_k_lower_margin as lower


def main():
    torch.set_num_threads(1)
    saved = torch.load(screen.OUT / 'test_73_A.partial.pt', weights_only=False)
    captured = []
    original = screen.down.improvement_and._build_pool

    def capture(states, split):
        captured.extend(states)
        return original(states, split)

    screen.down.improvement_and._build_pool = capture
    result = dict(case=saved['case'], arm=saved['arm'],
                  evaluated_sites=len(saved['sites']), iteration=saved['iteration'],
                  best_loss=float(np.min(saved['losses'])),
                  scope='One frozen-pipeline evaluation of saved state; no search steps or objective evaluations.')
    try:
        frozen = screen.FrozenProposals()
        frozen(saved['sites'], saved['losses'], saved['case'], saved['iteration'])
        result['reproduced_error'] = None
    except RuntimeError as exc:
        result['reproduced_error'] = str(exc)
    finally:
        screen.down.improvement_and._build_pool = original

    if captured:
        torch.save(dict(saved_state=saved, grouped_states=captured),
                   screen.OUT / 'frozen_slope_sign_failure_state.pt')
        checkpoint = torch.load(lower.SLOPE_SIGN_PATH, weights_only=False)
        scale = float(checkpoint['training_scale_S'])
        theta = float(checkpoint['learned_theta'])
        alpha = float(checkpoint['learned_alpha'])
        result.update(slope_sign_checkpoint=str(lower.SLOPE_SIGN_PATH),
                      slope_sign_checkpoint_sha256=screen.sha(lower.SLOPE_SIGN_PATH),
                      training_scale_S=scale, learned_theta=theta, learned_alpha=alpha)
        edges = []
        for state in captured:
            hull = np.asarray(state['teacher_hull'])
            for i, (left, right) in enumerate(zip(hull[:-1], hull[1:])):
                df = float(state['losses'][right]-state['losses'][left])
                dd = float(state['sizes'][right]-state['sizes'][left])
                compressed = float(np.arcsinh(df / scale))
                logit = alpha * (compressed-theta)
                learned_positive = logit >= 0
                edges.append(dict(hull_edge_position=i, left_group=int(left), right_group=int(right),
                    left_size=float(state['sizes'][left]), right_size=float(state['sizes'][right]),
                    left_loss=float(state['losses'][left]), right_loss=float(state['losses'][right]),
                    delta_f=df, delta_d=dd, exact_slope=df/dd, compressed_delta_f=compressed,
                    learned_sign_logit=logit, teacher_positive=df>0,
                    learned_positive=learned_positive, sign_mismatch=bool((df>0)!=learned_positive)))
        result['edges'] = edges
        result['sign_mismatch_count'] = sum(e['sign_mismatch'] for e in edges)
    plan = json.loads((screen.OUT/'plan.json').read_text())
    result['protected_hashes_unchanged'] = all(screen.sha(p)==h for p,h in plan['protected_hashes'].items())
    (screen.OUT/'frozen_pipeline_stop_diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
