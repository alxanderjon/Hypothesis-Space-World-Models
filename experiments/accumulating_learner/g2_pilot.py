"""Stage 2a of the general learner: the development pilot (research log, 7 November). Development systems of the new
domains only. (1) Budget check: uniform search at 300 structures and 60 thorough fits per equation; if it solves more
than 50% of a domain's development equations, that domain's budget drops to 100 structures and 20 thorough fits.
(2) Timing of uniform and network-guided searches (an untrained network, for its cost only) at that budget.
Usage: g2_pilot.py <workers> <out.json>"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYS = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"


def job(args):
    rec, budget, cap, guided = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import movenet_g2 as MN
    import search_g1 as SG
    if guided:
        torch.manual_seed(0)
        net = MN.MoveNetL2().eval()
        emb = {d: MN.encode_vocabulary(net, MN.token_table(SG.vocabulary_order(d), d)) for d in (2, 3)}
        prior = MN.NetworkPrior(net, emb)
    else:
        prior = SG.uniform_prior
    t0 = time.time()
    out = SG.search_system(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap,
                           probes=[eq["keys"] for eq in rec["structure"]])
    return {"id": rec["id"], "domain": rec["domain"], "budget": budget, "guided": guided, "seconds": time.time() - t0,
            "solved": [e["effort"]["first_exact"] is not None for e in out["equations"]],
            "network_seconds": getattr(prior, "seconds", 0.0), "probe": getattr(prior, "probe", None)}


if __name__ == "__main__":
    workers, path = int(sys.argv[1]), sys.argv[2]
    recs = [r for d in ("chemistry", "chaotic") for r in json.loads((SYS / f"dev_{d}.json").read_text())]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, [(r, 300, 60, False) for r in recs]))
    budgets = {}
    for d in ("chemistry", "chaotic"):
        rs = [r for r in res if r["domain"] == d]
        n, s = sum(len(r["solved"]) for r in rs), sum(sum(r["solved"]) for r in rs)
        budgets[d] = 300 if s <= 0.5 * n else 100
        print(f"budget check, {d}: uniform at 300 solves {s} of {n} development equations -> budget {budgets[d]}", flush=True)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res2 = list(ex.map(job, [(r, budgets[r["domain"]], budgets[r["domain"]] // 5, g) for r in recs for g in (False, True)]))
    for d in ("chemistry", "chaotic"):
        for g in (False, True):
            rs = [r for r in res2 if r["domain"] == d and r["guided"] == g]
            secs = sorted(r["seconds"] for r in rs)
            print(f"timing, {d}, {'guided (untrained)' if g else 'uniform'} at {budgets[d]}: system seconds median "
                  f"{secs[len(secs) // 2]:.0f}, max {secs[-1]:.0f}, mean {sum(secs) / len(secs):.0f}; network seconds mean "
                  f"{sum(r['network_seconds'] for r in rs) / len(rs):.0f}", flush=True)
    Path(path).write_text(json.dumps({"budgets": budgets, "check": res, "timing": res2}))
