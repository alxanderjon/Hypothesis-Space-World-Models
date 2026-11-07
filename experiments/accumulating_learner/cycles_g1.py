"""Stage 1 of the general learner (research log, 6 November): the wake-dream cycles, the fresh networks and the test
evaluations, in a fixed order. Resumable: every phase writes its outputs under outputs/accumulating_learner/g1/run1/ and
is skipped when they exist. Counts only are printed for test systems.

Lineages: "cont" (one network through mechanics, circuits, ecology), "freshB" (circuits only), "freshC" (ecology only).
Each domain: wake 1 (uniform for a new lineage, else the lineage's latest network) -> dream 1 -> wake 2 -> dream 2.
A wake searches all 150 training tasks (1,000 structures and 200 thorough fits per equation); a dream trains the
lineage's network further (2,000 steps of 32) on the examples of every verified solution of the lineage so far plus 500
new fantasies. Bar (c) is judged after mechanics; the run stops there if it fails (owner, 6 November).
Usage: cycles_g1.py <workers> [last phase]"""

import hashlib
import json
import math
import pickle
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYS = ROOT / "outputs/accumulating_learner/g1/systems"
RUN = ROOT / "outputs/accumulating_learner/g1/run1"
BUDGET, CAP = 1000, 200
TIME_CAP = 3600.0                 # seconds per system (set from the development pilot; see the research log)
N_FANT, STEPS, BATCH, LR = 500, 2000, 32, 3e-4
NAME = {"A": "mechanics", "B": "circuits", "C": "ecology"}


# ---------------------------------------------------------------------------------------------------------------------
# worker jobs
# ---------------------------------------------------------------------------------------------------------------------

_NET = {}


def _prior(path):
    sys.path.insert(0, str(HERE))
    import search_g1 as SG
    if path == "uniform":
        return SG.uniform_prior
    if path not in _NET:
        import numpy as np
        import torch
        import movenet_g1 as MN
        net = MN.MoveNetL1()
        net.load_state_dict(torch.load(path + ".pt", map_location="cpu"))
        _NET[path] = (net.eval(), np.load(path + ".npy"))
    import movenet_g1 as MN
    net, emb = _NET[path]
    return MN.NetworkPrior(net, emb)


class _Limit(Exception):
    pass


def _alarm(signum, frame):
    raise _Limit()


def search_job(args):
    rec, prior_path, seed = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "domain": rec["domain"], "status": "done"}
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(2 * TIME_CAP + 600))                       # safety net only: the search stops itself at TIME_CAP
    try:
        prior = _prior(prior_path)
        res = SG.search_system(np.array(rec["bursts"]), prior, budget=BUDGET, thorough_cap=CAP, time_cap=TIME_CAP, seed=seed)
        out["equations"] = []
        for e in res["equations"]:
            ef = e["effort"]
            out["equations"].append({"eq": e["eq"], "first_exact": ef["first_exact"], "structures": ef["structures"],
                                     "seconds": ef["seconds"], "thorough": ef["thorough"], "time_capped": ef["time_capped"],
                                     "n_exact": e["n_exact"], "solution": e.get("solution"),
                                     "prior_seconds": ef["prior_seconds"], "fit_errors": ef.get("fit_errors", 0)})
        out["network_seconds"] = getattr(prior, "seconds", 0.0)
        out["setup_seconds"] = res["setup_seconds"]
    except _Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    return out


def examples_job(args):
    bursts, structures, seed = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g1 as DR
    try:
        return DR.make_examples(np.array(bursts), structures, seed)
    except Exception:
        return []


def fantasy_job(args):
    prefix, k, probs = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g1 as DR
    import search_g1 as SG
    order = SG.vocabulary_order()
    syn, reasons = DR.accepted_fantasy(prefix, k, order, np.asarray(probs))
    if syn is None:
        return {"k": k, "examples": [], "reasons": reasons, "accepted": False}
    try:
        ex = DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
    except Exception:
        ex = []
    return {"k": k, "examples": ex, "reasons": reasons, "accepted": True, "structure": [eq["keys"] for eq in syn.structure]}


def train_job(args):
    """Train in a fresh process (so its GPU memory is released when it exits): examples from pickles, the previous
    network (or a fresh one, seed 1), STEPS steps; saves the network (.pt) and the vocabulary's encodings (.npy)."""
    paths, prev, out, gi, seed = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import movenet_g1 as MN
    import search_g1 as SG
    examples = []
    for p in paths:
        examples += pickle.loads(Path(p).read_bytes())
    torch.manual_seed(1)
    net = MN.MoveNetL1()
    if prev is not None:
        net.load_state_dict(torch.load(prev + ".pt", map_location="cpu"))
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(8)
    table = MN.token_table(SG.vocabulary_order())
    tl = MN.train(net, examples, table, STEPS, batch=BATCH, lr=LR, device=device, seed=seed)
    emb = MN.encode_vocabulary(net, table, device=device)
    net = net.cpu()
    torch.save(net.state_dict(), out + ".pt")
    np.save(out + ".npy", emb)
    return tl


# ---------------------------------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------------------------------

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN / "progress.log", "a") as f:
        f.write(line + "\n")


def records(which, letter):
    return [r for r in json.loads((SYS / f"{which}_{NAME[letter]}.json").read_text()) if r["status"] == "accepted"]


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


class Driver:
    def __init__(self, workers):
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
        for sub in ("wake", "test", "nets", "examples/replay", "examples/fantasy", "dream"):
            (RUN / sub).mkdir(parents=True, exist_ok=True)

    # -- searches --
    def search(self, path, recs, prior_path, label):
        if path.exists():
            return json.loads(path.read_text())
        log(f"{label}: searching {len(recs)} systems with prior {Path(prior_path).name}")
        t0 = time.time()
        res = list(self.pool.map(search_job, [(r, prior_path, 1) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        log(f"{label}: done in {time.time() - t0:.0f}s; " + self.summary(res, recs, label.startswith("test")))
        return res

    def summary(self, res, recs, counts_only=True):
        truth = {r["id"]: r["structure"] for r in recs}
        eqs = [(r, e) for r in res if r["status"] == "done" for e in r["equations"]]
        solved = sum(e["first_exact"] is not None for _, e in eqs)
        true_found = sum(1 for r, e in eqs if e["solution"] and sorted(e["solution"]["keys"]) == sorted(truth[r["id"]][e["eq"]]["keys"]))
        systems = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
        capped = sum(e["time_capped"] for _, e in eqs)
        bad = sum(r["status"] != "done" for r in res)
        return (f"equations solved {solved} of {sum(len(truth[r['id']]) for r in res)} (true structure {true_found}); "
                f"systems with both solved {systems} of {len(res)}; time-capped equations {capped}; failed systems {bad}")

    # -- replays, fantasies, training --
    def replays(self, lineage):
        """(task id, eq) -> keys of the simplest verified solution, over the lineage's wakes so far."""
        p = RUN / f"replays_{lineage}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def add_replays(self, lineage, wake):
        rep = self.replays(lineage)
        import fitter_g1 as FG
        for r in wake:
            if r["status"] != "done":
                continue
            for e in r["equations"]:
                if e["solution"]:
                    k = f"{r['id']}|{e['eq']}"
                    new = sorted(e["solution"]["keys"])
                    old = rep.get(k)
                    cost = lambda ks: (len(ks), sum(FG.info(x).k > 0 for x in ks))
                    if old is None or cost(new) < cost(old):
                        rep[k] = new
        (RUN / f"replays_{lineage}.json").write_text(json.dumps(rep))
        return rep

    def replay_examples(self, rep):
        """Examples of every verified solution in rep (cached per task and structure)."""
        tasks = {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            tasks.setdefault(tid, {})[int(eq)] = keys
        recs = {r["id"]: r for L in "ABC" for r in records("train", L)}
        jobs, paths = [], []
        for tid, eqs in sorted(tasks.items()):
            st = [eqs.get(i) for i in range(2)]
            p = RUN / "examples/replay" / f"{tid}__{h(st)}.pkl"
            paths.append(p)
            if not p.exists():
                jobs.append((p, (recs[tid]["bursts"], st, int(h(tid), 16) % (1 << 31))))
        if jobs:
            for (p, _), ex in zip(jobs, self.pool.map(examples_job, [a for _, a in jobs])):
                p.write_bytes(pickle.dumps(ex))
        out = []
        for p in paths:
            out += pickle.loads(p.read_bytes())
        return out

    def replay_paths(self, rep):
        tasks = {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            tasks.setdefault(tid, {})[int(eq)] = keys
        return [str(RUN / "examples/replay" / f"{tid}__{h([eqs.get(i) for i in range(2)])}.pkl") for tid, eqs in sorted(tasks.items())]

    def usage(self, rep):
        import dream_g1 as DR
        u = {}
        for k, keys in rep.items():
            eq = int(k.rsplit("|", 1)[1])
            for key in keys:
                rk = DR.role_key(key, eq)
                if rk is not None:
                    u[rk] = u.get(rk, 0) + 1
        return u

    def fantasies(self, lineage, tag, rep):
        p = RUN / "examples/fantasy" / f"{lineage}_{tag}.pkl"
        if p.exists():
            return pickle.loads(p.read_bytes())
        import dream_g1 as DR
        import search_g1 as SG
        order = SG.vocabulary_order()
        Vinfo = json.loads(SG.VOCAB_PATH.read_text())["terms"]
        probs = DR.term_distribution(order, Vinfo, self.usage(rep)).tolist()
        t0 = time.time()
        res = list(self.pool.map(fantasy_job, [(f"g1-fantasy-{lineage}-{tag}", k, probs) for k in range(N_FANT)]))
        ex = [e for r in res for e in r["examples"]]
        reasons = {}
        for r in res:
            for x in r["reasons"]:
                reasons[x] = reasons.get(x, 0) + 1
        (RUN / "dream" / f"{lineage}_{tag}_fantasies.json").write_text(json.dumps(
            {"accepted": sum(r["accepted"] for r in res), "rejected_draws": reasons, "examples": len(ex),
             "seconds": time.time() - t0, "structures": [r.get("structure") for r in res]}))
        log(f"dream {lineage} {tag}: fantasies accepted {sum(r['accepted'] for r in res)} of {N_FANT}, examples {len(ex)}, "
            f"{time.time() - t0:.0f}s")
        p.write_bytes(pickle.dumps(ex))
        return ex

    def gpu(self):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            import os
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            vis = [int(v) for v in vis.split(",") if v.strip()] if vis else None
            for line in q:
                i, u, m = [int(x) for x in line.split(",")]
                if u < 10 and m < 1000 and (vis is None or i in vis):
                    return (i if vis is None else vis.index(i)), f"GPU {i} (utilization {u}%, {m} MiB used at the check)"
        except Exception:
            pass
        return None, "CPU (no free GPU at the check)"

    def dream(self, lineage, tag, prev, wake):
        """Train the lineage's network further; returns the network path (without suffix)."""
        out = RUN / "nets" / f"{lineage}_{tag}"
        if Path(str(out) + ".npy").exists():
            return str(out)
        import numpy as np
        import torch
        import movenet_g1 as MN
        import search_g1 as SG
        rep = self.add_replays(lineage, wake)
        rex = self.replay_examples(rep)
        fex = self.fantasies(lineage, tag, rep)
        examples = rex + fex
        paths = self.replay_paths(rep) + [str(RUN / "examples/fantasy" / f"{lineage}_{tag}.pkl")]
        gi, gdesc = self.gpu()
        t0 = time.time()
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:   # a fresh process: its GPU memory is
            tl = one.submit(train_job, (paths, prev, str(out), gi, int(h(lineage + tag), 16) % (1 << 31))).result()   # freed on exit
        (RUN / "dream" / f"{lineage}_{tag}_train.json").write_text(json.dumps(
            {"replay_solutions": len(rep), "replay_examples": len(rex), "fantasy_examples": len(fex), "device": gdesc,
             "seconds": time.time() - t0, "log": tl}))
        log(f"dream {lineage} {tag}: trained on {len(rex)} replay + {len(fex)} fantasy examples ({len(rep)} verified "
            f"solutions) on {gdesc} in {time.time() - t0:.0f}s; last loss {tl[-1]['loss']:.3f}, top-5 add {tl[-1]['top5_add']:.2f}")
        return str(out)

    def domain(self, lineage, L, start):
        """Two wake-dream cycles of lineage on domain L from network `start` ("uniform" for a new lineage)."""
        tasks = records("train", L)
        w1 = self.search(RUN / "wake" / f"{lineage}_{L}1.json", tasks, start, f"wake {lineage} {L} 1")
        n1 = self.dream(lineage, f"{L}1", None if start == "uniform" else start, w1)
        w2 = self.search(RUN / "wake" / f"{lineage}_{L}2.json", tasks, n1, f"wake {lineage} {L} 2")
        n2 = self.dream(lineage, f"{L}2", n1, w2)
        return n1, n2

    def test(self, net, L, label):
        return self.search(RUN / "test" / f"{label}_{L}.json", records("test", L), net, f"test {label} on {L}")


def median_first(res):
    v = sorted((e["first_exact"] if e["first_exact"] is not None else math.inf) for r in res if r["status"] == "done"
               for e in r["equations"])
    return v[len(v) // 2] if v else math.inf


def bar_c(guided, uniform):
    n = sum(len(r["equations"]) for r in uniform if r["status"] == "done")
    sg = sum(e["first_exact"] is not None for r in guided if r["status"] == "done" for e in r["equations"])
    su = sum(e["first_exact"] is not None for r in uniform if r["status"] == "done" for e in r["equations"])
    pts = 100.0 * (sg - su) / max(n, 1)
    mg, mu = median_first(guided), median_first(uniform)
    met = pts >= 20 or (math.isfinite(mg) and mg <= 0.5 * mu)
    return {"guided_solved": sg, "uniform_solved": su, "equations": n, "points": pts, "median_first_guided": mg,
            "median_first_uniform": mu, "met": bool(met)}


def main():
    workers = int(sys.argv[1])
    last = sys.argv[2] if len(sys.argv) > 2 else "end"
    RUN.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    D = Driver(workers)
    log(f"driver started: workers {workers}, budget {BUDGET}, thorough cap {CAP}, time limit {TIME_CAP:.0f}s per system")
    # mechanics: two cycles of the continued network, then bar (c)
    a1, a2 = D.domain("cont", "A", "uniform")
    tu_A = D.test("uniform", "A", "uniform")
    tg_A = D.test(a2, "A", "cont_A2")
    bc = bar_c(tg_A, tu_A)
    (RUN / "bar_c.json").write_text(json.dumps(bc, default=float))
    log(f"bar (c): {'MET' if bc['met'] else 'NOT MET'}: guided {bc['guided_solved']} vs uniform {bc['uniform_solved']} of "
        f"{bc['equations']} test equations ({bc['points']:+.1f} points); median structures to the first plausible model "
        f"{bc['median_first_guided']} vs {bc['median_first_uniform']}")
    if not bc["met"] or last == "bar_c":
        log("stopping after bar (c)" + ("" if bc["met"] else " (not met): owner's instruction, before circuits"))
        return
    # circuits: baselines and zero-shot, continued and fresh networks
    D.test("uniform", "B", "uniform")
    D.test("uniform", "C", "uniform")
    D.test(a2, "B", "cont_A2")
    b1, b2 = D.domain("cont", "B", a2)
    D.test(b1, "B", "cont_B1")
    D.test(b2, "B", "cont_B2")
    D.test(b2, "C", "cont_B2")
    fb1, fb2 = D.domain("freshB", "B", "uniform")
    D.test(fb1, "B", "freshB_B1")
    D.test(fb2, "B", "freshB_B2")
    # ecology
    c1, c2 = D.domain("cont", "C", b2)
    D.test(c1, "C", "cont_C1")
    D.test(c2, "C", "cont_C2")
    D.test(c2, "A", "cont_C2")
    D.test(c2, "B", "cont_C2")
    fc1, fc2 = D.domain("freshC", "C", "uniform")
    D.test(fc1, "C", "freshC_C1")
    D.test(fc2, "C", "freshC_C2")
    log("all phases done")


if __name__ == "__main__":
    main()
