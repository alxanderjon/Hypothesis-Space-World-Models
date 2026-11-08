"""Stage 2a'' of the general learner (research log, 8 November): draw and store the harder domains' systems at one
difficulty level (bursts at M = 8, seed 1, with a hash per system). Seeds "g2h-{set}-{domain}-k" (the domain name
carries the level, e.g. reactions_L1), with replacement seeds in order when a draw is rejected. Flows also get the
largest Lyapunov exponent (reported only). Test systems are drawn only after the ladder has settled. Prints counts
only. Usage: g2h_draw.py <workers> <level> <set ...> (sets: pilot, train, dev, test)"""
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"
SIZES = {"pilot": 24, "train": 100, "dev": 40, "test": 40}
KINDS = ("reactions", "flows")


def job(args):
    which, dom, k = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import domains_g1 as DG
    t0 = time.time()
    syn, reasons = DG.draw_accepted(f"g2h-{which}-{dom}", k, dom)
    rec = {"id": f"g2h-{which}-{dom}-{k}", "set": which, "domain": dom, "k": k, "rejections": len(reasons),
           "reasons": dict(collections.Counter(r.split(":")[0] for r in reasons)), "seconds": time.time() - t0}
    if syn is None:
        rec["status"] = "none accepted"
        return rec
    b = np.ascontiguousarray(syn.bursts(8), dtype=np.float64)
    rec.update(status="accepted", seed=syn.desc["seed"], desc=syn.desc, structure=syn.structure, bursts=b.tolist(),
               hash=hashlib.sha256(b.tobytes()).hexdigest(), build_seconds=syn.build_seconds,
               ref={"dt": syn.ref.sys.dt, "nsub": int(syn.ref.nsub), "refine": int(syn.ref.refine)})
    try:
        rec["lyapunov"] = DG.largest_lyapunov(syn) if dom.startswith("flows") else None
    except Exception:
        rec["lyapunov"] = None
    return rec


if __name__ == "__main__":
    workers, level, sets = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    sys.path.insert(0, str(HERE))
    import fitter_g1 as FG
    OUT.mkdir(parents=True, exist_ok=True)
    doms = [f"{k}_L{level}" for k in KINDS]
    jobs = [(w, d, k) for w in sets for d in doms for k in range(SIZES[w])]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        recs = list(ex.map(job, jobs, chunksize=1))
    for w in sets:
        for d in doms:
            rs = sorted([r for r in recs if r["set"] == w and r["domain"] == d], key=lambda r: r["k"])
            (OUT / f"{w}_{d}.json").write_text(json.dumps(rs))
            reasons = collections.Counter()
            for r in rs:
                reasons.update(r["reasons"])
            acc = [r for r in rs if r["status"] == "accepted"]
            terms = collections.Counter(len(eq["keys"]) for r in acc for eq in r["structure"])
            cb_sys = collections.Counter(len({k for eq in r["structure"] for k in eq["keys"] if FG.info(k).k > 0}) for r in acc)
            cb_eq = collections.Counter(sum(FG.info(k).k > 0 for k in eq["keys"]) for r in acc for eq in r["structure"])
            lyap = [r["lyapunov"] for r in acc if r.get("lyapunov") is not None]
            secs = sorted(r["seconds"] for r in rs)
            print(f"{w:>5} {d:>12}: accepted {len(acc)} of {len(rs)}; rejected draws {sum(r['rejections'] for r in rs)} "
                  f"({dict(reasons)}); terms per equation {dict(sorted(terms.items()))}; distinct constant-bearing keys per "
                  f"system {dict(sorted(cb_sys.items()))}; per equation {dict(sorted(cb_eq.items()))}; largest Lyapunov "
                  f"exponent > 0.01: {sum(1 for v in lyap if v == v and v > 0.01)} of {len(lyap)}; seconds per system median "
                  f"{secs[len(secs) // 2]:.0f}, max {secs[-1]:.0f}", flush=True)
    print(f"total {time.time() - t0:.0f}s")
