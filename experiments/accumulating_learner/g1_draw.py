"""Stage 1 of the general learner (research log, Stage 1 rules, section 8): draw and store each domain's training tasks,
development systems and test systems once (bursts at M = 8, seed 1, with a hash per system), so every arm sees identical
systems. Prints counts only. Usage: g1_draw.py <workers> [set ...]"""
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
SIZES = {"train": 150, "dev": 10, "test": 40}


def job(args):
    which, dom, k = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import domains_g1 as DG
    t0 = time.time()
    syn, reasons = DG.draw_accepted(f"g1-{which}-{dom}", k, dom)
    rec = {"id": f"g1-{which}-{dom}-{k}", "set": which, "domain": dom, "k": k, "rejections": len(reasons),
           "reasons": dict(collections.Counter(r.split(":")[0] for r in reasons)), "seconds": time.time() - t0}
    if syn is None:
        rec["status"] = "none accepted"
        return rec
    b = np.ascontiguousarray(syn.bursts(8), dtype=np.float64)
    rec.update(status="accepted", seed=syn.desc["seed"], desc=syn.desc, structure=syn.structure, bursts=b.tolist(),
               hash=hashlib.sha256(b.tobytes()).hexdigest(), build_seconds=syn.build_seconds,
               ref={"dt": syn.ref.sys.dt, "nsub": int(syn.ref.nsub), "refine": int(syn.ref.refine)})
    return rec


if __name__ == "__main__":
    workers = int(sys.argv[1])
    sets = sys.argv[2:] or list(SIZES)
    sys.path.insert(0, str(HERE))
    import domains_g1 as DG
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, d, k) for w in sets for d in DG.DOMAINS for k in range(SIZES[w])]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        recs = list(ex.map(job, jobs, chunksize=2))
    for w in sets:
        for d in DG.DOMAINS:
            rs = sorted([r for r in recs if r["set"] == w and r["domain"] == d], key=lambda r: r["k"])
            (OUT / f"{w}_{d}.json").write_text(json.dumps(rs))
            reasons = collections.Counter()
            for r in rs:
                reasons.update(r["reasons"])
            print(f"{w:>5} {d:>9}: accepted {sum(r['status'] == 'accepted' for r in rs)} of {len(rs)}; "
                  f"rejected draws {sum(r['rejections'] for r in rs)} ({dict(reasons)})", flush=True)
