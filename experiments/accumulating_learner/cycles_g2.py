"""Stage 2a of the general learner (research log, 7 November): histories, made-up-systems controls, learning curves in
the new domains, retention. Resumable: every phase writes its outputs under outputs/accumulating_learner/g2/run1/ and is
skipped when they exist. Counts only are printed for test systems.

Orders: 1: mechanics, circuits, ecology, chemistry, chaotic (quadratic flows); 2: ecology, circuits, mechanics, chaotic,
chemistry. Histories (the first three domains) follow Stage 1's rules; the new domains are learned in a first cycle of
100 tasks with dreams after 10, 30 and 100, and tests on the domain's 40 test systems before any task and after each
dream, at the pilot's budgets. Arms: the continued network (its own order), the made-up-systems control (the same dream
schedule, fantasies only, drawn from the language with every term size equally likely), the fresh network.
Usage: cycles_g2.py <workers> [order ...]"""

import hashlib
import json
import math
import os
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
RUN = ROOT / "outputs/accumulating_learner/g2/run1"
ORDERS = {1: ("mechanics", "circuits", "ecology", "chemistry", "chaotic"), 2: ("ecology", "circuits", "mechanics", "chaotic", "chemistry")}
HIST = {"budget": 1000, "cap": 200, "fantasies": 500, "steps": 2000}
NEWB = {"chemistry": (300, 60), "chaotic": (100, 20)}           # the development pilot's budgets (research log)
NEW = {"fantasies": 100, "steps": 500}
POINTS = (10, 30, 100)
TIME_CAP = 3600.0
BATCH, LR = 32, 3e-4


# ---------------------------------------------------------------------------------------------------------------------
# worker jobs
# ---------------------------------------------------------------------------------------------------------------------

_NET = {}


def _prior(path):
    sys.path.insert(0, str(HERE))
    import search_g1 as SG
    if path == "uniform":
        return SG.uniform_prior
    import numpy as np
    import torch
    import movenet_g2 as MN
    if path not in _NET:
        net = MN.MoveNetL2()
        net.load_state_dict(torch.load(path + ".pt", map_location="cpu", weights_only=True))
        _NET[path] = (net.eval(), {d: np.load(f"{path}_d{d}.npy") for d in (2, 3)})
    net, emb = _NET[path]
    return MN.NetworkPrior(net, emb)


class _Limit(Exception):
    pass


def _alarm(signum, frame):
    raise _Limit()


def search_job(args):
    rec, prior_path, seed, budget, cap = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "domain": rec["domain"], "status": "done", "budget": budget}
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(2 * TIME_CAP + 600))
    try:
        prior = _prior(prior_path)
        res = SG.search_system(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap, time_cap=TIME_CAP, seed=seed,
                               probes=[eq["keys"] for eq in rec["structure"]])
        out["equations"] = []
        for e in res["equations"]:
            ef = e["effort"]
            out["equations"].append({"eq": e["eq"], "first_exact": ef["first_exact"], "structures": ef["structures"],
                                     "seconds": ef["seconds"], "thorough": ef["thorough"], "time_capped": ef["time_capped"],
                                     "n_exact": e["n_exact"], "solution": e.get("solution"), "prior_seconds": ef["prior_seconds"],
                                     "fit_errors": ef.get("fit_errors", 0)})
        out["network_seconds"] = getattr(prior, "seconds", 0.0)
        out["probe"] = getattr(prior, "probe", None)
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
    import dream_g2 as DR
    try:
        return DR.make_examples(np.array(bursts), structures, seed)
    except Exception:
        return []


def fantasy_job(args):
    prefix, k, probs, d = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import search_g1 as SG
    syn, reasons = DR.accepted_fantasy(prefix, k, SG.vocabulary_order(d), np.asarray(probs), d)
    if syn is None:
        return {"k": k, "d": d, "examples": [], "reasons": reasons, "accepted": False}
    try:
        ex = DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
    except Exception:
        ex = []
    return {"k": k, "d": d, "examples": ex, "reasons": reasons, "accepted": True}


def train_job(args):
    """Train in a fresh process (GPU memory freed on exit); saves the network and the search vocabulary's encodings for
    two and three variables."""
    paths, prev, out, gi, seed, steps = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import movenet_g2 as MN
    import search_g1 as SG
    examples = []
    for p in paths:
        examples += pickle.loads(Path(p).read_bytes())
    torch.manual_seed(1)
    net = MN.MoveNetL2()
    if prev is not None:
        net.load_state_dict(torch.load(prev + ".pt", map_location="cpu", weights_only=True))
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(8)
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in (2, 3)}
    tl = MN.train(net, examples, tables, steps, batch=BATCH, lr=LR, device=device, seed=seed)
    for d in (2, 3):
        np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    counts = {}
    for e in examples:
        counts[str(e["d"])] = counts.get(str(e["d"]), 0) + 1
    return tl, counts


# ---------------------------------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------------------------------

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN / "progress.log", "a") as f:
        f.write(line + "\n")


def records(which, dom):
    rs = [r for r in json.loads((SYS / f"{which}_{dom}.json").read_text()) if r["status"] == "accepted"]
    return sorted(rs, key=lambda r: r["k"])


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


class Driver:
    def __init__(self, workers):
        self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
        for sub in ("wake", "test", "nets", "examples/replay", "examples/fantasy", "dream"):
            (RUN / sub).mkdir(parents=True, exist_ok=True)
        self.recs = {r["id"]: r for dom in ORDERS[1] for r in records("train", dom)}

    def search(self, path, recs, prior, label, budget, cap):
        if path.exists():
            return json.loads(path.read_text())
        log(f"{label}: searching {len(recs)} systems with prior {Path(prior).name} (budget {budget}, thorough cap {cap})")
        t0 = time.time()
        res = list(self.pool.map(search_job, [(r, prior, 1, budget, cap) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        truth = {r["id"]: r["structure"] for r in recs}
        eqs = [(r, e) for r in res if r["status"] == "done" for e in r["equations"]]
        solved = sum(e["first_exact"] is not None for _, e in eqs)
        true = sum(1 for r, e in eqs if e["solution"] and sorted(e["solution"]["keys"]) == sorted(truth[r["id"]][e["eq"]]["keys"]))
        systems = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
        pr = [r["probe"] for r in res if r.get("probe")]
        cut = f"; shortlist cut a missing true term at {sum(p['cut'] for p in pr)} of {sum(p['with_missing_true'] for p in pr)} " \
              f"expansions" if pr else ""
        log(f"{label}: done in {time.time() - t0:.0f}s; equations solved {solved} of {sum(len(truth[r['id']]) for r in res)} "
            f"(true structure {true}); systems {systems} of {len(res)}; time-capped {sum(e['time_capped'] for _, e in eqs)}; "
            f"failed {sum(r['status'] != 'done' for r in res)}{cut}")
        return res

    # -- replays and fantasies --
    def replays(self, lineage):
        p = RUN / f"replays_{lineage}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def add_replays(self, lineage, wake):
        sys.path.insert(0, str(HERE))
        import fitter_g1 as FG
        rep = self.replays(lineage)
        cost = lambda ks: (len(ks), sum(FG.info(x).k > 0 for x in ks))
        for r in wake:
            if r["status"] != "done":
                continue
            for e in r["equations"]:
                if e["solution"]:
                    k = f"{r['id']}|{e['eq']}"
                    new = sorted(e["solution"]["keys"])
                    if k not in rep or cost(new) < cost(rep[k]):
                        rep[k] = new
        (RUN / f"replays_{lineage}.json").write_text(json.dumps(rep))
        return rep

    def replay_paths(self, rep):
        tasks = {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            tasks.setdefault(tid, {})[int(eq)] = keys
        jobs, paths = [], []
        for tid, eqs in sorted(tasks.items()):
            d = self.recs[tid]["desc"]["d"]
            st = [eqs.get(i) for i in range(d)]
            p = RUN / "examples/replay" / f"{tid}__{h(st)}.pkl"
            paths.append(str(p))
            if not p.exists():
                jobs.append((p, (self.recs[tid]["bursts"], st, int(h(tid), 16) % (1 << 31))))
        if jobs:
            for (p, _), ex in zip(jobs, self.pool.map(examples_job, [a for _, a in jobs])):
                p.write_bytes(pickle.dumps(ex))
        return paths

    def usage(self, rep):
        import dream_g2 as DR
        u = {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            d = self.recs[tid]["desc"]["d"]
            for key in keys:
                rk = DR.role_key(key, int(eq), d)
                if rk is not None:
                    u.setdefault(d, {})
                    u[d][rk] = u[d].get(rk, 0) + 1
        return u

    def probs(self, d, usage):
        import dream_g1 as D1
        import search_g1 as SG
        V = json.loads((SG.G1_DIR / f"vocab_d{d}.json").read_text())["terms"]
        return D1.term_distribution(SG.vocabulary_order(d), V, usage.get(d, {}) if usage is not None else {}).tolist()

    def fantasies(self, prefix, n_by_d, usage):
        """Fantasy examples: n_by_d[d] fantasies of d variables. usage None -> the size prior only (control)."""
        p = RUN / "examples/fantasy" / f"{prefix}.pkl"
        if p.exists():
            return str(p)
        t0 = time.time()
        jobs = []
        for d, n in sorted(n_by_d.items()):
            pr = self.probs(d, usage)
            jobs += [(f"{prefix}-d{d}", k, pr, d) for k in range(n)]
        res = list(self.pool.map(fantasy_job, jobs))
        ex = [e for r in res for e in r["examples"]]
        p.write_bytes(pickle.dumps(ex))
        log(f"fantasies {prefix}: accepted {sum(r['accepted'] for r in res)} of {len(res)}, examples {len(ex)}, "
            f"{time.time() - t0:.0f}s")
        return str(p)

    def control_fantasies(self, prefix, target_by_d):
        """Size-prior fantasies for a control dream: unused pool batches in order (generating more when short) until
        each dimension's example count reaches its target; extra examples of the last batch are dropped."""
        p = RUN / "examples/fantasy" / f"{prefix}.pkl"
        if p.exists():
            return str(p)
        t0 = time.time()
        (RUN / "pool").mkdir(parents=True, exist_ok=True)
        got, taken = {d: [] for d in target_by_d}, []
        for d, t in sorted(target_by_d.items()):
            while len(got[d]) < t:
                led = self._ledger()
                used = {x for v in led["consumed"].values() for x in v} | set(taken)
                free = [q for q in self._pool_batches(d) if q.name not in used]
                if not free:
                    self.gen_pool_batch(d)
                    continue
                got[d] += pickle.loads(free[0].read_bytes())
                taken.append(free[0].name)
        led = self._ledger()
        led["consumed"][prefix] = taken
        (RUN / "pool" / "ledger.json").write_text(json.dumps(led))
        ex = [e for d, t in sorted(target_by_d.items()) for e in got[d][:t]]
        p.write_bytes(pickle.dumps(ex))
        log(f"control fantasies {prefix}: examples {len(ex)} for targets {target_by_d} from {len(taken)} pool batches, "
            f"{time.time() - t0:.0f}s")
        return str(p)

    def gpu(self, poll=60):
        """A free GPU, or else one that can be shared under the owner's rule (not fully utilized and with free memory;
        never interfering with other users' jobs): over one minute, mean utilization <= 60%, never above 80%, and at
        least 6 GB free throughout. No CPU fallback (owner, 7 November: the design stays as approved): wait and re-check
        every `poll` seconds. Returns (index, description)."""
        waited, t0 = False, time.time()
        while True:
            samples = []
            for _ in range(6):
                try:
                    q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
                    samples.append([[int(x) for x in line.split(",")] for line in q.strip().splitlines()])
                except Exception:
                    pass
                if samples and len(samples) == 1:
                    for i, u, m, tot in samples[0]:
                        if u < 10 and m < 1000:
                            note = f" after waiting {(time.time() - t0) / 60:.0f} min" if waited else ""
                            return i, f"GPU {i} (free: utilization {u}%, {m} MiB used at the check){note}"
                    if os.environ.get("G2_ALLOW_CPU") == "1":      # smoke tests only (tiny dreams); never set for the run
                        return None, "CPU (smoke test: no free GPU at the check)"
                time.sleep(10)
            if samples:
                for g in range(len(samples[0])):
                    us = [s_[g][1] for s_ in samples]
                    free = [s_[g][3] - s_[g][2] for s_ in samples]
                    if sum(us) / len(us) <= 60 and max(us) <= 80 and min(free) >= 6000:
                        note = f" after waiting {(time.time() - t0) / 60:.0f} min" if waited else ""
                        return g, (f"GPU {g} (shared: utilization mean {sum(us) / len(us):.0f}%, max {max(us)}%, at least "
                                   f"{min(free)} MiB free over a minute){note}")
            if not waited:
                log("no free or shareable GPU: waiting (checking every minute; idle workers fill the control pool)")
                waited = True
            if not self.fill_pool():                           # the wait overlaps generating the control's fantasies
                time.sleep(poll)

    # -- the control's fantasies come from a pool (size prior only, independent of anything learned), filled while the
    #    driver waits for a GPU and topped up on demand; each control dream consumes fresh, unused batches --
    POOL_TARGET = {2: 50000, 3: 3000}

    def _ledger(self):
        p = RUN / "pool" / "ledger.json"
        return json.loads(p.read_text()) if p.exists() else {"next": {"2": 0, "3": 0}, "consumed": {}}

    def _pool_batches(self, d):
        return sorted((RUN / "pool").glob(f"d{d}_b*.pkl"), key=lambda q: int(q.stem.split("_b")[1].split("__n")[0]))

    def pool_size(self, d, ledger):
        used = {x for v in ledger["consumed"].values() for x in v}
        return sum(int(q.name.split("__n")[1].split(".")[0]) if "__n" in q.name else 0
                   for q in self._pool_batches(d) if q.name not in used)

    def gen_pool_batch(self, d, n_fant=48):
        (RUN / "pool").mkdir(parents=True, exist_ok=True)
        led = self._ledger()
        b = led["next"][str(d)]
        led["next"][str(d)] = b + 1
        (RUN / "pool" / "ledger.json").write_text(json.dumps(led))
        res = list(self.pool.map(fantasy_job, [(f"g2-pool-d{d}-b{b}", k, self.probs(d, None), d) for k in range(n_fant)]))
        ex = [e for r in res for e in r["examples"]]
        tmp = RUN / "pool" / f".tmp_d{d}_b{b}.pkl"
        tmp.write_bytes(pickle.dumps(ex))
        tmp.rename(RUN / "pool" / f"d{d}_b{b}__n{len(ex)}.pkl")
        return len(ex)

    def fill_pool(self):
        led = self._ledger()
        for d, target in self.POOL_TARGET.items():
            if self.pool_size(d, led) < target:
                self.gen_pool_batch(d)
                return True
        return False

    def train(self, out, paths, prev, steps, label, info):
        gi, gdesc = self.gpu()
        t0 = time.time()
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
            tl, counts = one.submit(train_job, (paths, prev, str(out), gi, int(h(label), 16) % (1 << 31), steps)).result()
        rec = dict(info, device=gdesc, seconds=time.time() - t0, examples_by_d=counts, steps=steps, log=tl)
        (RUN / "dream" / f"{label}_train.json").write_text(json.dumps(rec))
        log(f"dream {label}: trained {steps} steps on {sum(counts.values())} examples {counts} on {gdesc} in "
            f"{time.time() - t0:.0f}s; last loss {tl[-1]['loss']:.3f}; skipped steps {tl[-1]['skipped']}")
        return rec

    def dream(self, lineage, tag, prev, wake, n_fant, steps, dims):
        """Train the lineage's network further on every verified solution of the lineage plus new fantasies (dimension
        drawn per fantasy from `dims`, {d: weight})."""
        out = RUN / "nets" / f"{lineage}_{tag}"
        if Path(f"{out}_d3.npy").exists():
            return str(out)
        rep = self.add_replays(lineage, wake)
        paths = self.replay_paths(rep)
        tot = sum(dims.values())
        n_by_d, left = {}, n_fant
        for j, (d, w) in enumerate(sorted(dims.items())):
            n_by_d[d] = left if j == len(dims) - 1 else int(round(n_fant * w / tot))
            left -= n_by_d[d]
        paths.append(self.fantasies(f"g2-fantasy-{lineage}-{tag}", n_by_d, self.usage(rep)))
        self.train(out, paths, prev, steps, f"{lineage}_{tag}", {"replay_solutions": len(rep), "fantasies": n_by_d})
        return str(out)

    def control_dream(self, clineage, tag, prev, mirror_label, steps):
        """A control dream mirroring another dream (its example counts per dimension and its steps): fantasies only."""
        out = RUN / "nets" / f"{clineage}_{tag}"
        if Path(f"{out}_d3.npy").exists():
            return str(out)
        target = {int(d): n for d, n in json.loads((RUN / "dream" / f"{mirror_label}_train.json").read_text())["examples_by_d"].items()}
        path = self.control_fantasies(f"g2-control-{clineage}-{tag}", target)
        self.train(out, [path], prev, steps, f"{clineage}_{tag}", {"mirrors": mirror_label, "targets": target})
        return str(out)

    # -- phases --
    def history(self, o):
        lineage, net = f"o{o}", "uniform"
        for dom in ORDERS[o][:3]:
            tasks = records("train", dom)
            for c in (1, 2):
                w = self.search(RUN / "wake" / f"{lineage}_{dom}{c}.json", tasks, net, f"wake {lineage} {dom} {c}",
                                HIST["budget"], HIST["cap"])
                net = self.dream(lineage, f"{dom}{c}", None if net == "uniform" else net, w, HIST["fantasies"], HIST["steps"], {2: 1})
        return net

    def control_base(self, o):
        net = None
        for dom in ORDERS[o][:3]:
            for c in (1, 2):
                net = self.control_dream(f"c{o}", f"{dom}{c}", net, f"o{o}_{dom}{c}", HIST["steps"])
        return net

    def control_extend(self, o, dom4, base):
        net = base
        for k in POINTS:
            net = self.control_dream(f"c{o}x", f"{dom4}{k}", net, f"o{o}_{dom4}_d{k}", NEW["steps"])
        return net

    def test(self, net, dom, label, budget, cap):
        return self.search(RUN / "test" / f"{label}_{dom}.json", records("test", dom), net, f"test {label} on {dom}", budget, cap)

    def curve(self, lineage, start, dom, label):
        """The first cycle in a new domain: tests before any task and after the dreams at 10, 30 and 100 tasks."""
        tasks = records("train", dom)
        B, cap = NEWB[dom]
        net = start
        self.test(net, dom, f"{label}_t0", B, cap)
        prev = 0
        for k in POINTS:
            w = self.search(RUN / "wake" / f"{label}_w{k}.json", tasks[prev:k], net, f"wake {label} tasks {prev + 1}-{k}", B, cap)
            dims = {}
            for r in tasks[:k]:
                dims[r["desc"]["d"]] = dims.get(r["desc"]["d"], 0) + 1
            tag = f"{dom}_d{k}"
            net = self.dream(lineage, tag, None if net == "uniform" else net, w, NEW["fantasies"], NEW["steps"], dims)
            self.test(net, dom, f"{label}_t{k}", B, cap)
            prev = k
        return net

    def retention(self, o, net, when):
        for dom in ORDERS[o][:3]:
            self.test(net, dom, f"o{o}_{when}", HIST["budget"], HIST["cap"])


def main():
    workers, orders = int(sys.argv[1]), [int(x) for x in sys.argv[2:]] or [1, 2]
    RUN.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    D = Driver(workers)
    log(f"driver started: workers {workers}; orders {orders}; history budget {HIST}; new-domain budgets {NEWB}, {NEW}; "
        f"points {POINTS}; fantasy pre-screen on")
    # phases that need no network first (identical results whenever they run)
    for o in orders:
        dom = ORDERS[o][0]
        D.search(RUN / "wake" / f"o{o}_{dom}1.json", records("train", dom), "uniform", f"wake o{o} {dom} 1", HIST["budget"], HIST["cap"])
    for dom in ("chemistry", "chaotic"):
        D.test("uniform", dom, f"fresh_{dom}_t0", *NEWB[dom])
        D.search(RUN / "wake" / f"fresh_{dom}_w10.json", records("train", dom)[:10], "uniform", f"wake fresh_{dom} tasks 1-10", *NEWB[dom])
    for o in orders:
        d4, d5 = ORDERS[o][3], ORDERS[o][4]
        hist = D.history(o)
        D.retention(o, hist, "before")
        cbase = D.control_base(o)
        for dom in (d4, d5):                                     # the fresh networks do not depend on the order
            D.curve(f"fresh_{dom}", "uniform", dom, f"fresh_{dom}")
        c4 = D.curve(f"o{o}", hist, d4, f"o{o}_{d4}")
        D.curve(f"c{o}_{d4}", cbase, d4, f"c{o}_{d4}")
        cext = D.control_extend(o, d4, cbase)
        c5 = D.curve(f"o{o}", c4, d5, f"o{o}_{d5}")
        D.curve(f"c{o}_{d5}", cext, d5, f"c{o}_{d5}")
        D.retention(o, c5, "after")
        D.test(c5, d4, f"o{o}_after", *NEWB[d4])
        log(f"order {o}: all phases done")
    log("all requested orders done")


if __name__ == "__main__":
    main()
