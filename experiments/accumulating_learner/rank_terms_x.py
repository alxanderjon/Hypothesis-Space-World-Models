"""Root rankings of the development flows systems' terms by saved networks (the slowdown diagnosis, 10 November; Stage X's
bump report): at the start of each equation's search, every term of the system's vocabulary is ranked by the search's
probability of adding it (movenet_g2.NetworkPrior, the 2,048-term shortlist). Scoring only: no training, no search.
Development systems only. Usage: rank_terms_x.py <workers> <nets folder> <out.json> <network ...>"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYS = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"


def job(args):
    rec, nets_dir, names = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import movenet_g2 as MN
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    nets = {n: (MN.load_net(str(Path(nets_dir) / n) + ".pt").eval(), {4: np.load(str(Path(nets_dir) / n) + "_d4.npy")}) for n in names}
    out = []
    for i, eq in enumerate(rec["structure"]):
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        true = [vocab.index[k] for k in eq["keys"] if k in vocab.index]
        for n, (net, emb) in nets.items():
            p = MN.NetworkPrior(net, emb, shortlist=2048)(se.root, se.ctx).p_add
            order = np.argsort(-p)
            rank = np.empty(len(p), dtype=np.int64)
            rank[order] = np.arange(1, len(p) + 1)
            out.append({"id": rec["id"], "eq": i, "net": n, "true_ranks": [int(rank[t]) for t in true],
                        "true_keys": [vocab.keys[t] for t in true], "top": [vocab.keys[int(t)] for t in order[:20]], "V": int(vocab.V)})
    return out


if __name__ == "__main__":
    workers, nets_dir, out_path, names = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    recs = [r for r in json.loads((SYS / "dev_flows_L1.json").read_text()) if r["status"] == "accepted"]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = [x for part in ex.map(job, [(r, nets_dir, names) for r in recs]) for x in part]
    Path(out_path).write_text(json.dumps(res))
    print(f"ranked {len(res)} (network, equation) pairs for {len(recs)} development systems")
