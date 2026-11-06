"""Stage 1 of the general learner: the development timing pilot (research log): uniform search at the full budget
(1,000 structures and 200 thorough fits per equation, no time limit) on the development systems, to set time limits and
project Stage 1's compute. Development systems only. Usage: g1_pilot.py <workers> <out.json>"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYS = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"


def job(rec):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    t0 = time.time()
    out = SG.search_system(np.array(rec["bursts"]), SG.uniform_prior, budget=1000, thorough_cap=200)
    eqs = [{"eq": e["eq"], "solved": e["effort"]["first_exact"] is not None, "first": e["effort"]["first_exact"],
            "true_found": bool(e.get("solution")) and sorted(e["solution"]["keys"]) == sorted(t["keys"]),
            **{k: e["effort"][k] for k in ("structures", "seconds", "thorough", "fit_seconds", "thorough_seconds", "prior_seconds")}}
           for e, t in zip(out["equations"], rec["structure"])]
    return {"id": rec["id"], "domain": rec["domain"], "setup": out["setup_seconds"], "seconds": time.time() - t0, "equations": eqs}


if __name__ == "__main__":
    workers, path = int(sys.argv[1]), sys.argv[2]
    recs = [r for d in ("mechanics", "circuits", "ecology") for r in json.loads((SYS / f"dev_{d}.json").read_text())]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, recs))
    Path(path).write_text(json.dumps(res))
    for d in ("mechanics", "circuits", "ecology"):
        rs = [r for r in res if r["domain"] == d]
        eqs = [e for r in rs for e in r["equations"]]
        s = sorted(r["seconds"] for r in rs)
        es = sorted(e["seconds"] for e in eqs)
        print(f"{d}: equations solved {sum(e['solved'] for e in eqs)} of {len(eqs)} (true structure {sum(e['true_found'] for e in eqs)}); "
              f"system seconds median {s[len(s)//2]:.0f}, max {s[-1]:.0f}; equation seconds median {es[len(es)//2]:.0f}, "
              f"p90 {es[int(0.9*len(es))]:.0f}, max {es[-1]:.0f}", flush=True)
