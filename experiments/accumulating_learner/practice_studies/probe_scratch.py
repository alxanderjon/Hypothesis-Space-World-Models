# DEVELOPMENT half only: the current from-scratch learner at small observation budgets (strong-scratch baseline).
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
sys.path.insert(0, SP)
from probe_accum import job

if __name__ == "__main__":
    tag = sys.argv[1]
    names = sys.argv[2].split(",")
    Ms = [int(m) for m in sys.argv[3].split(",")]
    workers = int(sys.argv[4])
    out = f"{SP}/{tag}"
    os.makedirs(out, exist_ok=True)
    specs = [{"name": n, "M": M, "arm": "scratch", "pos": -1, "library": [], "out": f"{out}/{n}_M{M}_scratch.json"} for M in Ms for n in names]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(job, s) for s in specs]):
            r = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {r['name']} M{r['M']}: {r.get('status')} exact={r.get('exact')} val={r.get('val')} {r.get('wall', 0):.0f}s", flush=True)
