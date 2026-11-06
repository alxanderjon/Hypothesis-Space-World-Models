"""Stage 1 of the general learner, bar (a) (research log, 6 November 20:30): the generic fitter on each stored system's
true structures (fitter_g1.evaluate_structure per equation), with a time limit per system. Checks each system's burst
hash. Writes per-system records (no structures or constants) and prints counts only. Usage: g1_bar_a.py <workers> <set>"""
import hashlib
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYS = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"
OUT = HERE.parents[1] / "outputs/accumulating_learner/g1/bar_a"
LIMIT = 600.0                     # seconds per system (pilot's slowest known-structure fit: 0.6 s)


class Limit(Exception):
    pass


def _alarm(signum, frame):
    raise Limit()


def job(rec):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    out = {"id": rec["id"], "domain": rec["domain"], "equations": []}
    b = np.ascontiguousarray(np.array(rec["bursts"], dtype=np.float64))
    out["hash_ok"] = hashlib.sha256(b.tobytes()).hexdigest() == rec["hash"]
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(LIMIT))
    try:
        keys = sorted({k for eq in rec["structure"] for k in eq["keys"]})
        ctx = FG.system_context(b, seed=1, keys=keys)
        fit = FG.make_fitter(ctx, seed=1)
        for i, eq in enumerate(rec["structure"]):
            node = FG.evaluate_structure(eq["keys"], ctx, i, fit)
            out["equations"].append({"eq": i, "terms": len(eq["keys"]), "with_constants": sum(FG.info(k).k > 0 for k in eq["keys"]),
                                     "exact": node["exact"], "rel_t": node["rel_t"], "rel_v": node["rel_v"], "tau": node["tau"],
                                     "degenerate": node["degenerate"], "gate": node["gate"], "thorough": node["thorough"],
                                     "seconds": node["seconds"]})
        out["status"] = "done"
    except Limit:
        out["status"] = "time limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = type(ex).__name__
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    out["fitter_stats"] = dict(fit.stats) if "fit" in locals() else {}
    return out


if __name__ == "__main__":
    workers, which = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    doms = ("mechanics", "circuits", "ecology")
    recs = [r for d in doms for r in json.loads((SYS / f"{which}_{d}.json").read_text()) if r["status"] == "accepted"]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, recs))
    (OUT / f"{which}.json").write_text(json.dumps(res, default=float))
    for d in doms:
        rs = [r for r in res if r["domain"] == d]
        eqs = [e for r in rs for e in r["equations"]]
        secs = sorted(r["seconds"] for r in rs)
        print(f"{which} {d}: equations exact {sum(e['exact'] for e in eqs)} of {sum(1 for r in rs for _ in r['equations'])} "
              f"(with constants inside terms: {sum(e['exact'] for e in eqs if e['with_constants'])} of "
              f"{sum(1 for e in eqs if e['with_constants'])}); systems with both equations exact "
              f"{sum(1 for r in rs if r['status'] == 'done' and all(e['exact'] for e in r['equations']))} of {len(rs)}; "
              f"time per system median {secs[len(secs) // 2]:.1f}s, max {secs[-1]:.1f}s; time-limit hits "
              f"{sum(r['status'] == 'time limit' for r in rs)}; errors {sum(r['status'] == 'error' for r in rs)}; "
              f"hash mismatches {sum(not r['hash_ok'] for r in rs)}", flush=True)
