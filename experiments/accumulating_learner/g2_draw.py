"""Stage 2a of the general learner (research log, 7 November): draw and store the new domains' training tasks (the first
cycle), development and test systems once (bursts at M = 8, seed 1, with a hash per system). Seeds "g2-{set}-{domain}-k"
with replacement seeds in order when a draw is rejected. Chaotic-flow systems also get the largest Lyapunov exponent
(reported only). Prints counts only. Usage: g2_draw.py <workers> [set ...]"""
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
SIZES = {"train": 100, "dev": 10, "test": 40}
DOMAINS = ("chemistry", "chaotic")


def job(args):
    which, dom, k = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import domains_g1 as DG
    t0 = time.time()
    syn, reasons = DG.draw_accepted(f"g2-{which}-{dom}", k, dom)
    rec = {"id": f"g2-{which}-{dom}-{k}", "set": which, "domain": dom, "k": k, "rejections": len(reasons),
           "reasons": dict(collections.Counter(r.split(":")[0] for r in reasons)), "seconds": time.time() - t0}
    if syn is None:
        rec["status"] = "none accepted"
        return rec
    b = np.ascontiguousarray(syn.bursts(8), dtype=np.float64)
    rec.update(status="accepted", seed=syn.desc["seed"], desc=syn.desc, structure=syn.structure, bursts=b.tolist(),
               hash=hashlib.sha256(b.tobytes()).hexdigest(), build_seconds=syn.build_seconds,
               ref={"dt": syn.ref.sys.dt, "nsub": int(syn.ref.nsub), "refine": int(syn.ref.refine)})
    try:
        rec["lyapunov"] = DG.largest_lyapunov(syn)
    except Exception:
        rec["lyapunov"] = None
    return rec


if __name__ == "__main__":
    workers = int(sys.argv[1])
    sets = sys.argv[2:] or list(SIZES)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, d, k) for w in sets for d in DOMAINS for k in range(SIZES[w])]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        recs = list(ex.map(job, jobs, chunksize=1))
    for w in sets:
        for d in DOMAINS:
            rs = sorted([r for r in recs if r["set"] == w and r["domain"] == d], key=lambda r: r["k"])
            (OUT / f"{w}_{d}.json").write_text(json.dumps(rs))
            reasons = collections.Counter()
            for r in rs:
                reasons.update(r["reasons"])
            acc = [r for r in rs if r["status"] == "accepted"]
            lyap = [r["lyapunov"] for r in acc if r.get("lyapunov") is not None]
            chaotic = sum(1 for v in lyap if v == v and v > 0.01)
            dims = collections.Counter(r["desc"]["d"] for r in acc)
            print(f"{w:>5} {d:>9}: accepted {len(acc)} of {len(rs)} (dimensions {dict(dims)}); rejected draws "
                  f"{sum(r['rejections'] for r in rs)} ({dict(reasons)}); largest Lyapunov exponent > 0.01: {chaotic} of "
                  f"{len(lyap)}", flush=True)
