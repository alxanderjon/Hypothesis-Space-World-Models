# Trial 27 build, DEVELOPMENT half only: accumulation probe with static libraries.
# Each system (fixed random order) gets the library of new pieces accepted on the systems before it (their M = 64 scratch
# runs from the latest development rounds); library vs scratch (and decoy, second half) at small observation budgets.
import json, os, sys, time, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import numpy as np

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
OUT = f"{SP}/probe"
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")


def source_record(name):
    for d in ("d27reg3", "d27reg2", "d27reg1", "d27full", "d27v18", "d27v17", "d27v15", "d27", "d27s2"):
        p = f"{SP}/{d}/{name}_M64.json"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def job(spec):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(2)
    import harness27 as H27, discovery_learner as DL, compile_artifact as CA
    name, M, arm, lib_rows, out = spec["name"], spec["M"], spec["arm"], spec["library"], spec["out"]
    if os.path.exists(out):
        return json.load(open(out))
    assert H27.split_half(name) == "dev"
    rec = {k: spec[k] for k in ("name", "M", "arm", "pos")}
    t0 = time.time()
    try:
        ref = H27.Reference27(name, seed=1)
        b = H27.bursts_from(ref, 1, M)
        lib = DL.Library.from_json(lib_rows)
        res = DL.learn(b, seed=1, time_limit=300 if M <= 8 else 900, library=lib, discover_pieces=True)
        rec.update(status=res["status"], val=res.get("val_err_scaled"), seconds=res["seconds"], library_size=len(lib.entries),
                   reused=[p.get("library_id") for p in res.get("pieces", []) if p.get("origin") == "reused"])
        rec["exact"] = False
        if res["status"] == "accepted":
            m = res["model"]
            rec.update(model=m.describe(), mu=m.mu.tolist(), sig=m.sig.tolist(), nsub=res["nsub"], pieces=res.get("pieces", []))
            roll = CA.build(m.describe(), m.mu.tolist(), m.sig.tolist(), ref.sys.dt)
            ev = H27.evaluate(ref, None, res["nsub"], rollout_hat=lambda x0, steps: roll.adaptive(x0, steps, nsub0=max(1, res["nsub"] // 4)))
            rec["exact"], rec["one"] = bool(ev["exact"]), ev["one_step_max_scaled_error"]
    except Exception as e:
        rec.update(status="error", exact=False, error=repr(e)[:200])
    rec["wall"] = time.time() - t0
    json.dump(rec, open(out, "w"), default=float)
    return rec


def main():
    import harness27 as H27, discovery_learner as DL
    os.makedirs(OUT, exist_ok=True)
    names = sorted(n for n in H27.pool_records() if H27.split_half(n) == "dev")
    names = [n for n in names if n not in ("BIOMD0000000741", "BIOMD0000000802")]      # reference cannot be built
    order = [names[k] for k in np.random.default_rng([2727, 99, 0]).permutation(len(names))]
    lib = DL.Library()
    specs = []
    for k, name in enumerate(order):
        rows = lib.to_json()
        for M in (8, 16, 32):
            specs.append({"name": name, "M": M, "arm": "scratch", "pos": k, "library": [], "out": f"{OUT}/{name}_M{M}_scratch.json"})
            if rows:
                specs.append({"name": name, "M": M, "arm": "library", "pos": k, "library": rows, "out": f"{OUT}/{name}_M{M}_library.json"})
                if k >= len(order) // 2:
                    dec = DL.decoy_library(lib, seed=500 + k).to_json()
                    specs.append({"name": name, "M": M, "arm": "decoy", "pos": k, "library": dec, "out": f"{OUT}/{name}_M{M}_decoy.json"})
        src = source_record(name)
        if src and src.get("status") == "accepted":
            for pc in src.get("pieces") or []:
                if pc.get("origin") == "new":
                    lib.add(DL.as_library_piece(pc), source=name)
    json.dump({"order": order, "final_library": lib.to_json()}, open(f"{OUT}/plan.json", "w"), default=float)
    print(len(specs), "jobs; library grows to", len(lib.entries), "entries", flush=True)
    specs.sort(key=lambda s: (s["M"] > 8, s["pos"]))                  # cheap budgets first
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 18, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for f in as_completed(futs):
            r = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] pos {r.get('pos')} {r['name']} M{r['M']} {r['arm']}: {r.get('status')} exact={r.get('exact')} "
                  f"reused={r.get('reused')} lib={r.get('library_size')} {r.get('wall', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
