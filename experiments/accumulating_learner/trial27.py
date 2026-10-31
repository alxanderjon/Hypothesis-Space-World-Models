"""Trial 27 driver: discovery of missing pieces from data, and reuse of discovered pieces on later systems.

Streams (3 orders x 2 data draws, fixed in advance): each evaluation system is learned in turn with the library of pieces
discovered on the systems before it, at every observation budget; then the new pieces its own M = 64 run discovered and
accepted are added. Controls: scratch (same learner, empty library), decoy library (same templates, random shapes, second
half of each stream), copy-the-closest earlier system, removal check (separate stage). See docs/research-log.md (Trial 27).
Everything is simulated in memory; only small JSON records are written.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/accumulating_learner/trial27"
MS = (2, 4, 8, 16, 32, 64)
ORDERS = (0, 1, 2)
SEEDS = (1, 2)
L = 13


def systems(half):
    import harness27 as H27
    recs = H27.pool_records()
    return sorted(i for i in recs if H27.split_half(i) == half)


def stream_order(names, seed, order):
    return [names[k] for k in np.random.default_rng([2727, seed, order]).permutation(len(names))]


def time_limit(M):
    return TIME_LIMITS["small"] if M <= 8 else TIME_LIMITS["large"]


TIME_LIMITS = {"small": 300, "large": 900}       # fixed during development; recorded in the log before evaluation


def setup_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


# ---------------- one learning run + evaluation ----------------

def run_one(job):
    """job: dict(name, seed, M, mode, library (json rows), out (path), copy_from (optional))."""
    setup_worker()
    import harness27 as H27
    import discovery_learner as DL
    import compile_artifact as CA
    out = Path(job["out"])
    if out.exists():
        return job, json.loads(out.read_text())
    t0 = time.time()
    rec = {k: job[k] for k in ("name", "seed", "M", "mode") if k in job}
    rec.update({k: job.get(k) for k in ("order", "pos")})
    try:
        ref = H27.Reference27(job["name"], seed=job["seed"])
        b = H27.bursts_from(ref, job["seed"], job["M"])
        lib = DL.Library.from_json(job.get("library") or [])
        rec["library_size"] = len(lib.entries)
        if job["mode"] == "copy_closest":
            res = copy_closest(b, job["copy_from"], job["seed"])
        else:
            res = DL.learn(b, seed=job["seed"], time_limit=time_limit(job["M"]), library=lib, discover_pieces=True)
        rec.update({k: res.get(k) for k in ("status", "val_err_scaled", "n_terms", "seconds", "discovery_used", "copied")})
        rec["pieces"] = res.get("pieces", [])
        if job["M"] == 64:
            rec["bursts"] = b.tolist()
        if "model" in res:
            m = res["model"]
            rec.update({"model": m.describe(), "mu": m.mu.tolist(), "sig": m.sig.tolist(), "nsub": res["nsub"]})
            roll = CA.build(rec["model"], rec["mu"], rec["sig"], ref.sys.dt)
            # the artifact = learned equations + their own adaptive 8th-order integrator (no simulator)
            art = lambda x0, steps: roll.adaptive(x0, steps, nsub0=max(1, res["nsub"] // 4))
            rec["exactness"] = H27.evaluate(ref, None, res["nsub"], rollout_hat=art)
            rec["exact"] = bool(rec["exactness"]["exact"])
            if rec["pieces"] and rec["exact"]:   # owner addition: exactness outside the region the pieces were learned on
                rec["out_of_range"] = H27.evaluate_out_of_range(ref, lambda x0, steps, k: roll.adaptive(x0, steps, nsub0=1))
        else:
            rec["exact"] = False
        rec["sampling_refinements"] = int(getattr(ref, "refine", 0))
    except Exception as e:
        rec.update({"status": "error", "exact": False, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
    rec["wall"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, default=float))
    return job, rec


def copy_closest(bursts, earlier, seed):
    """Baseline: the whole learned model of the earlier system most similar in data (same dimension, all variable
    relabelings), constants refitted on this system's bursts."""
    import itertools
    import torch
    import discovery_learner as DL
    import guide as GD
    b = np.asarray(bursts)
    d = b.shape[-1]
    fb = GD.burst_features(b).mean(0).numpy()
    best = None
    for r in earlier:
        if r.get("status") != "accepted" or "model" not in r or len(r["model"]) != d:
            continue
        fh = GD.burst_features(np.asarray(r["bursts"])).mean(0).numpy()
        for pm in itertools.permutations(range(d)):
            dist = float(np.sum((fb - fh[list(pm)]) ** 2))
            if best is None or dist < best[0]:
                best = (dist, r, pm)
    if best is None:
        return {"status": "no_model", "seconds": 0.0}
    _, r, pm = best
    inv = {old: new for new, old in enumerate(pm)}
    desc = []
    for new_i in range(d):
        comp = []
        for t in r["model"][pm[new_i]]:
            facs = []
            for f in t["factors"]:
                var = [inv[v] for v in f[1]] if isinstance(f[1], list) else inv[f[1]]
                facs.append([f[0], var, f[2], f[3] if len(f) > 3 else []])
            comp.append({"mono": [t["mono"][pm[k]] for k in range(d)], "factors": facs, "level": t["level"], "coef": t["coef"]})
        desc.append(comp)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    M = b.shape[0]
    nval = max(1, int(round(0.25 * M)))
    perm = rng.permutation(M)
    Bt = torch.as_tensor(b)
    allx = Bt.reshape(-1, d)
    mu, sig = allx.mean(0), allx.std(0) + 1e-300
    mu = torch.where(mu.abs() > DL.CENTER_RATIO * sig, mu, torch.zeros_like(mu))
    Ztr, Zva = (Bt[perm[nval:] if M > 1 else perm] - mu) / sig, (Bt[perm[:nval]] - mu) / sig
    model = DL.model_from_description(desc, mu, sig)
    z0, z1 = Ztr[:, :-1].reshape(-1, d), Ztr[:, 1:].reshape(-1, d)
    zv0, zv1 = Zva[:, :-1].reshape(-1, d), Zva[:, 1:].reshape(-1, d)
    try:
        model, nsub, err, _ = DL.polish(model, z0, z1, iters=40)
        val, _, nsub_v = DL.converged_val(model, z0, zv0, zv1)
    except (RuntimeError, ValueError):
        return {"status": "no_model", "seconds": time.time() - t0, "copied": r["name"]}
    return {"status": "accepted" if val <= DL.VAL_TOL else "rejected", "model": model, "nsub": nsub_v or nsub,
            "val_err_scaled": val, "seconds": time.time() - t0, "copied": r["name"], "n_terms": model.n_terms()}


# ---------------- scheduler: streams + scratch on one shared pool ----------------

def run_trial(names, seeds, orders, workers, include_scratch=True, include_decoy=True, include_copy=True, tag="eval"):
    import discovery_learner as DL
    base = OUT / tag
    state = {}
    for s in seeds:
        for o in orders:
            state[(s, o)] = {"order": stream_order(names, s, o), "pos": 0, "lib": DL.Library(), "earlier": []}
    queue = []                                               # (priority, job)

    def enqueue_position(s, o):
        st = state[(s, o)]
        k = st["pos"]
        if k >= len(st["order"]):
            return
        name = st["order"][k]
        lib_rows = st["lib"].to_json()
        d = base / f"seed{s}" / f"order{o}"
        (d / f"library_pos{k:02d}.json").parent.mkdir(parents=True, exist_ok=True)
        (d / f"library_pos{k:02d}.json").write_text(json.dumps(lib_rows, default=float))
        for M in MS:
            queue.append((0 if M == 64 else 1, {"name": name, "seed": s, "M": M, "mode": "library", "library": lib_rows,
                                                "order": o, "pos": k, "out": str(d / f"pos{k:02d}_{name}_M{M}_library.json")}))
        if include_decoy and k >= len(st["order"]) // 2 and st["lib"].entries:
            dec = DL.decoy_library(st["lib"], seed=1000 * s + 100 * o + k).to_json()
            for M in MS:
                queue.append((2, {"name": name, "seed": s, "M": M, "mode": "decoy", "library": dec, "order": o, "pos": k,
                                  "out": str(d / f"pos{k:02d}_{name}_M{M}_decoy.json")}))
        if include_copy and st["earlier"]:
            for M in MS:
                queue.append((2, {"name": name, "seed": s, "M": M, "mode": "copy_closest", "copy_from": st["earlier"],
                                  "order": o, "pos": k, "out": str(d / f"pos{k:02d}_{name}_M{M}_copy.json")}))

    for key in state:
        enqueue_position(*key)
    if include_scratch:
        for s in seeds:
            for name in names:
                for M in MS:
                    queue.append((3, {"name": name, "seed": s, "M": M, "mode": "scratch",
                                      "out": str(base / f"seed{s}" / "scratch" / f"{name}_M{M}.json")}))
    ctx = get_context("spawn")
    running = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        while queue or running:
            queue.sort(key=lambda e: e[0])
            while queue and len(running) < workers:
                _, job = queue.pop(0)
                running[ex.submit(run_one, job)] = job
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                job = running.pop(fut)
                try:
                    _, rec = fut.result()
                except Exception as e:                       # a crashed worker: record and continue
                    rec = {"status": "worker_error", "exact": False, "error": repr(e)[:300]}
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] {done} {job['mode']} s{job['seed']} o{job.get('order')} "
                      f"p{job.get('pos')} {job['name']} M{job['M']}: {rec.get('status')} exact={rec.get('exact')} "
                      f"pieces={len(rec.get('pieces') or [])} lib={rec.get('library_size')}", flush=True)
                if job["mode"] == "library" and job["M"] == 64:      # advance the stream
                    st = state[(job["seed"], job["order"])]
                    if rec.get("status") == "accepted":
                        for pc in rec.get("pieces") or []:
                            if pc.get("origin") == "new":
                                st["lib"].add(DL.as_library_piece(pc), source=job["name"])
                        st["earlier"].append({k: rec[k] for k in ("name", "status", "model", "bursts") if k in rec})
                    st["pos"] += 1
                    enqueue_position(job["seed"], job["order"])
    for (s, o), st in state.items():
        d = base / f"seed{s}" / f"order{o}"
        (d / "library_final.json").write_text(json.dumps(st["lib"].to_json(), default=float))


def removal_jobs(tag, seeds):
    """Removal check: every stream run where the library learner was exact, scratch was not (same system, same M), and
    library pieces were reused, is rerun with those pieces removed from that position's library."""
    import discovery_learner as DL
    jobs = []
    for s in seeds:
        base = OUT / tag / f"seed{s}"
        for p in sorted(base.glob("order*/pos*_library.json")):
            r = json.loads(p.read_text())
            if not r.get("exact"):
                continue
            sp = base / "scratch" / f"{r['name']}_M{r['M']}.json"
            if sp.exists() and json.loads(sp.read_text()).get("exact"):
                continue
            used = sorted({pc["library_id"] for pc in r.get("pieces") or [] if pc.get("origin") == "reused"})
            if not used:
                continue
            lib = DL.Library.from_json(json.loads((p.parent / f"library_pos{r['pos']:02d}.json").read_text()))
            jobs.append({"name": r["name"], "seed": s, "M": r["M"], "mode": "removal", "order": r["order"], "pos": r["pos"],
                         "library": lib.without(used).to_json(), "removed": used,
                         "out": str(base / "removal" / f"o{r['order']}_p{r['pos']:02d}_{r['name']}_M{r['M']}.json")})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "dev", "lists", "removal"])
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--orders", default="0,1,2")
    ap.add_argument("--systems", default="")
    ap.add_argument("--ms", default="", help="observation budgets (default: all); rehearsal only")
    a = ap.parse_args()
    if a.ms:
        global MS
        MS = tuple(int(x) for x in a.ms.split(","))
    if a.stage == "lists":
        print("dev", len(systems("dev")), "eval", len(systems("eval")))
        return
    if a.stage == "removal":
        jobs = removal_jobs(a.tag, [int(x) for x in a.seeds.split(",")])
        print(len(jobs), "removal jobs", flush=True)
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            for job, rec in ex.map(run_one, jobs):
                print(job["name"], job["M"], "removed", job["removed"], "->", rec.get("status"), rec.get("exact"), flush=True)
        return
    names = a.systems.split(",") if a.systems else systems(a.stage)
    if a.stage == "eval":
        assert all(n in systems("eval") for n in names)
    run_trial(names, [int(x) for x in a.seeds.split(",")], [int(x) for x in a.orders.split(",")], a.workers, tag=a.stage)


if __name__ == "__main__":
    main()
