"""Load the independently gated, immutable learned Brent components."""
import json
from pathlib import Path
import torch
from experiments.geometry.learned_brent_completion import LearnedBrent, CategoricalHead
from experiments.geometry import learned_brent_completion as implementation
from experiments.geometry.train_division_operator import sha


def load_completed_brent(folder=Path('outputs/sho/stage1B/learned_brent_completion')):
    complete=json.loads((folder/'learned_branch_switch_brent_locked.json').read_text())
    assert complete['status']=='LOCKED'
    assert sha(Path(implementation.__file__))==complete['implementation_sha256']
    engine=LearnedBrent()
    for name,attribute,inputs,outputs in (
        ('brent_step_application','step_head',2,3),
        ('brent_stopping_operator','stop_head',2,1),
        ('brent_final_return','return_head',1,1),
    ):
        manifest=json.loads((folder/(name+'_locked.json')).read_text())
        assert manifest['status']=='LOCKED'
        cp=manifest['newly_trained'][0]
        assert sha(Path(cp['checkpoint']))==cp['sha256']
        model=CategoricalHead(inputs,outputs)
        model.load_state_dict(torch.load(cp['checkpoint'],weights_only=False)['state_dict'])
        model.eval().requires_grad_(False)
        setattr(engine,attribute,model)
    state=json.loads((folder/'brent_state_update_locked.json').read_text())
    assert state['status']=='LOCKED' and state['implementation_sha256']==complete['implementation_sha256']
    solver,_=engine.build(4)
    return engine,solver
