"""Stage 2a'' of the general learner (research log, 8 November): harder new domains with four variables (reaction
networks, nonlinear flows) at one difficulty level. Resumable: every phase writes its outputs under
outputs/accumulating_learner/g2h/<run>/ and is skipped when they exist. Counts only are printed for test systems.

Reused from Stage 2a (outputs/accumulating_learner/g2/run1): the two histories (order 1: mechanics, circuits, ecology;
order 2: ecology, circuits, mechanics), their verified solutions, and the two made-up-systems control bases with their
made-up examples. Reused networks get the fourth variable role as the mean of "other1" and "other2" (movenet_g2.load_net).

Fairness fixes (owner, 8 November):
  - balanced dreams: for the continued network and the control, half of each dream's training batches come from the
    newest domain's examples (its verified solutions so far and this dream's fantasies), half from everything earlier;
  - rehearsal-matched control: in every dream after its base (the extension and the new-domain dreams) the control
    replays its own earlier made-up examples, as many per dimension as the continued network replays from earlier
    domains in the dream it mirrors (a fixed-seed sample of its earlier made-up example files);
  - the fresh network has nothing earlier and is unchanged.

Modes:
  sat <level>            the saturation check: the fresh network's uniform first wake on training tasks 1-10, a dream,
                         a wake on tasks 11-30, a dream, then a test on the 40 development systems (both domains)
  run <level> [order ..] the claim run: learning curves at 0, 10, 30 and 100 tasks for the fresh network, the
                         continued network and the rehearsal-matched control in each order, the control extension,
                         retention at the end (the fresh network's first 30 tasks are those of the saturation check)
Usage: cycles_g2h.py <workers> sat <level> | cycles_g2h.py <workers> run <level> [order ...]
Environment: G2H_RUN (run folder name, default run1), G2H_SMOKE=1 (tiny budgets on development systems, CPU allowed;
smoke tests only)."""

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
RUN = ROOT / "outputs/accumulating_learner/g2h" / os.environ.get("G2H_RUN", "run1")
REUSE = ROOT / "outputs/accumulating_learner/g2/run1"
SMOKE = os.environ.get("G2H_SMOKE") == "1"
HISTORY = ("mechanics", "circuits", "ecology")
HIST = {"budget": 1000, "cap": 200}                             # history domains' retention tests, as in Stage 2a
NEWB = (300, 60) if not SMOKE else (20, 4)                      # Stage 2a's new-domain budget, every arm and both domains
NEW = {"fantasies": 100, "steps": 500} if not SMOKE else {"fantasies": 4, "steps": 10}
POINTS = (10, 30, 100) if not SMOKE else (2, 4, 6)
TIME_CAP = float(os.environ.get("G2H_TIME_CAP", 3600.0))     # seconds per equation (raised only for the shortlist check's reruns)
BATCH, LR = 32, 3e-4
DIMS = (2, 3, 4)
HIST_NET = {1: "o1_ecology2", 2: "o2_mechanics2"}             # Stage 2a's history networks and control bases
CBASE_NET = {1: "c1_ecology2", 2: "c2_mechanics2"}


def domains(level):
    return f"reactions_L{level}", f"flows_L{level}"


def orders(level):
    R, F = domains(level)
    return {1: ("mechanics", "circuits", "ecology", R, F), 2: ("ecology", "circuits", "mechanics", F, R)}


def cbase_files(o):
    """The control base's made-up example files (Stage 2a, one per base dream)."""
    hist = orders(1)[o][:3]
    return [str(REUSE / "examples/fantasy" / f"g2-control-c{o}-{dom}{c}.pkl") for dom in hist for c in (1, 2)]


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
    import movenet_g2 as MN
    if path not in _NET:
        net = MN.load_net(path + ".pt")
        _NET[path] = (net.eval(), {d: np.load(f"{path}_d{d}.npy") for d in DIMS if Path(f"{path}_d{d}.npy").exists()})
    net, emb = _NET[path]
    return MN.NetworkPrior(net, emb, shortlist=os.environ.get("G2H_SHORTLIST"))


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
        out["probe_eq"] = getattr(prior, "probe_eq", None)
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


def _load_examples(sources):
    """sources: [(path, group, indices or None)] -> (examples, groups)."""
    import numpy as np
    ex, groups = [], []
    for path, g, idx in sources:
        part = pickle.loads(Path(path).read_bytes())
        if idx is not None:
            part = [part[int(i)] for i in idx]
        ex += part
        groups += [g] * len(part)
    return ex, np.asarray(groups, dtype=np.int64)


def train_job(args):
    """Train in a fresh process (GPU memory freed on exit); saves the network and the search vocabulary's encodings for
    two, three and four variables. steps == 0: only load (with the fourth role) and encode (a reused network)."""
    sources, prev, out, gi, seed, steps, balanced = args
    sys.path.insert(0, str(HERE))
    import torch
    import numpy as np
    import movenet_g2 as MN
    import search_g1 as SG
    torch.manual_seed(1)
    net = MN.load_net(prev + ".pt") if prev is not None else MN.MoveNetL2()
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(8)
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in DIMS}
    tl, counts = [], {}
    if steps:
        examples, groups = _load_examples(sources)
        tl = MN.train(net, examples, tables, steps, batch=BATCH, lr=LR, device=device, seed=seed,
                      groups=groups if balanced else None)
        for e, g in zip(examples, groups):
            c = counts.setdefault(str(int(g)), {})
            c[str(e["d"])] = c.get(str(e["d"]), 0) + 1
    for d in DIMS:
        np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
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


def dom_of(tid):
    """The domain of a training task id (g1-train-mechanics-3, g2h-train-reactions_L1-7)."""
    return tid.split("-")[2]


class Driver:
    def __init__(self, workers, level):
        self.level = level
        self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
        for sub in ("wake", "test", "nets", "examples/replay", "examples/fantasy", "dream"):
            (RUN / sub).mkdir(parents=True, exist_ok=True)
        self.recs = {r["id"]: r for dom in HISTORY + domains(level) for r in records("train", dom)}

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
        secs = sorted(r["seconds"] for r in res)
        log(f"{label}: done in {time.time() - t0:.0f}s; equations solved {solved} of {sum(len(truth[r['id']]) for r in res)} "
            f"(true structure {true}); systems {systems} of {len(res)}; time-capped {sum(e['time_capped'] for _, e in eqs)}; "
            f"failed {sum(r['status'] != 'done' for r in res)}; seconds per system median {secs[len(secs) // 2]:.0f}, max "
            f"{secs[-1]:.0f}{cut}")
        return res

    # -- replays, fantasies, made-up examples, rehearsal --
    def replays(self, lineage):
        p = RUN / f"replays_{lineage}.json"
        if p.exists():
            return json.loads(p.read_text())
        rep = {}
        if lineage in ("o1", "o2"):                             # the continued network starts with its history's solutions
            rep = {k: v for k, v in json.loads((REUSE / f"replays_{lineage}.json").read_text()).items()
                   if dom_of(k.rsplit("|", 1)[0]) in HISTORY}
            p.write_text(json.dumps(rep))
        return rep

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

    def replay_sources(self, rep, newest):
        """[(path, group, None)] for every verified solution's examples: group 1 for the newest domain's tasks."""
        tasks = {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            tasks.setdefault(tid, {})[int(eq)] = keys
        jobs, out = [], []
        for tid, eqs in sorted(tasks.items()):
            d = self.recs[tid]["desc"]["d"]
            st = [eqs.get(i) for i in range(d)]
            name = f"{tid}__{h(st)}.pkl"
            p = REUSE / "examples/replay" / name
            if not p.exists():
                p = RUN / "examples/replay" / name
                if not p.exists():
                    jobs.append((p, (self.recs[tid]["bursts"], st, int(h(tid), 16) % (1 << 31))))
            out.append((str(p), 1 if dom_of(tid) == newest else 0, None))
        if jobs:
            for (p, _), ex in zip(jobs, self.pool.map(examples_job, [a for _, a in jobs])):
                p.write_bytes(pickle.dumps(ex))
        return out

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
        """Fantasy examples: n_by_d[d] fantasies of d variables; usage None -> the size prior only (made-up systems)."""
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

    def madeup(self, prefix, target_by_d):
        """New made-up examples (size prior only) for a control extension dream: batches of 48 fantasies per dimension
        until the example count reaches the target; extra examples of the last batch are dropped."""
        p = RUN / "examples/fantasy" / f"{prefix}.pkl"
        if p.exists():
            return str(p)
        t0 = time.time()
        got, b = {d: [] for d in target_by_d}, 0
        for d, t in sorted(target_by_d.items()):
            pr = self.probs(d, None)
            while len(got[d]) < t:
                res = list(self.pool.map(fantasy_job, [(f"{prefix}-d{d}-b{b}", k, pr, d) for k in range(48)]))
                got[d] += [e for r in res for e in r["examples"]]
                b += 1
        ex = [e for d, t in sorted(target_by_d.items()) for e in got[d][:t]]
        p.write_bytes(pickle.dumps(ex))
        log(f"made-up examples {prefix}: {len(ex)} for targets {target_by_d} from {b} batches, {time.time() - t0:.0f}s")
        return str(p)

    def rehearsal(self, label, files, target_by_d):
        """A fixed-seed sample of the control's earlier made-up examples: target_by_d[d] examples of each dimension,
        without replacement (all of them, logged, if fewer exist). Returns [(path, 0, indices)]."""
        import numpy as np
        p = RUN / "dream" / f"{label}_rehearsal.json"
        if p.exists():
            return [tuple(x) for x in json.loads(p.read_text())]
        index = []
        for f in files:
            for i, e in enumerate(pickle.loads(Path(f).read_bytes())):
                index.append((f, i, int(e["d"])))
        rng = np.random.default_rng(int(h(label), 16) % (1 << 31))
        out, short = [], {}
        for d, t in sorted(target_by_d.items()):
            cand = [(f, i) for f, i, dd in index if dd == d]
            pick = rng.choice(len(cand), size=min(t, len(cand)), replace=False) if cand else []
            if len(cand) < t:
                short[d] = (len(cand), t)
            by_f = {}
            for j in sorted(int(x) for x in pick):
                f, i = cand[j]
                by_f.setdefault(f, []).append(i)
            out += [(f, 0, idx) for f, idx in sorted(by_f.items())]
        p.write_text(json.dumps(out))
        log(f"rehearsal {label}: {sum(len(x[2]) for x in out)} earlier made-up examples for targets {target_by_d}"
            + (f"; fewer available {short}" if short else ""))
        return out

    # -- GPU and training --
    def gpu(self, poll=60):
        """A free GPU, or else one that can be shared under the owner's rule (not fully utilized and with free memory;
        never interfering with other users' jobs): over one minute, mean utilization <= 60%, never above 80%, and at
        least 6 GB free throughout. No CPU fallback: wait and re-check every `poll` seconds. Returns (index, text)."""
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
                    if SMOKE:
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
                log("no free or shareable GPU: waiting (checking every minute)")
                waited = True
            time.sleep(poll)

    def train(self, out, sources, prev, steps, label, info, balanced):
        gi, gdesc = self.gpu()
        t0 = time.time()
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
            tl, counts = one.submit(train_job, (sources, prev, str(out), gi, int(h(label), 16) % (1 << 31), steps,
                                                balanced)).result()
        rec = dict(info, device=gdesc, seconds=time.time() - t0, examples_by_group_d=counts, steps=steps, balanced=balanced,
                   log=tl)
        (RUN / "dream" / f"{label}_train.json").write_text(json.dumps(rec))
        n = sum(sum(c.values()) for c in counts.values())
        log(f"dream {label}: trained {steps} steps on {n} examples (by group, then dimension: {counts}; group 1 is the newest "
            f"domain){' with balanced batches' if balanced else ''} on {gdesc} in {time.time() - t0:.0f}s; last loss "
            f"{tl[-1]['loss']:.3f}; skipped steps {tl[-1]['skipped']}" if tl else f"prepared {label} on {gdesc}")
        return rec

    def prepared(self, name):
        """A Stage 2a network, saved here with the fourth variable role and encodings for two to four variables."""
        out = RUN / "nets" / name
        if not Path(f"{out}_d4.npy").exists():
            self.train(out, [], str(REUSE / "nets" / name), 0, f"prepare_{name}", {"source": str(REUSE / "nets" / name)}, False)
        return str(out)

    def counts_of(self, label):
        """Example counts by group and dimension of a stored dream ({group: {d: n}}, ints)."""
        rec = json.loads((RUN / "dream" / f"{label}_train.json").read_text())
        return {int(g): {int(d): n for d, n in c.items()} for g, c in rec["examples_by_group_d"].items()}

    def dream(self, lineage, tag, prev, wake, n_fant, steps, dims, newest, balanced):
        """Train the lineage's network further on every verified solution of the lineage plus new fantasies (dimension
        drawn per fantasy from `dims`, {d: weight}); balanced: half the batches from the newest domain's examples."""
        out = RUN / "nets" / f"{lineage}_{tag}"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        rep = self.add_replays(lineage, wake)
        sources = self.replay_sources(rep, newest)
        sources.append((self.fantasies(f"g2h-fantasy-{lineage}-{tag}", self._split(n_fant, dims), self.usage(rep)), 1, None))
        self.train(out, sources, prev, steps, f"{lineage}_{tag}", {"replay_solutions": len(rep)}, balanced)
        return str(out)

    @staticmethod
    def _split(n, dims):
        tot, out, left = sum(dims.values()), {}, n
        for j, (d, w) in enumerate(sorted(dims.items())):
            out[d] = left if j == len(dims) - 1 else int(round(n * w / tot))
            left -= out[d]
        return out

    def control_dream(self, clineage, tag, prev, wake, n_fant, steps, dims, newest, mirror, earlier_files):
        """The rehearsal-matched control's new-domain dream: its own verified solutions and new fantasies (group 1), plus
        its earlier made-up examples as many per dimension as the mirrored continued dream's group 0; balanced."""
        out = RUN / "nets" / f"{clineage}_{tag}"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        rep = self.add_replays(clineage, wake)
        sources = self.replay_sources(rep, newest)
        sources.append((self.fantasies(f"g2h-fantasy-{clineage}-{tag}", self._split(n_fant, dims), self.usage(rep)), 1, None))
        sources += self.rehearsal(f"{clineage}_{tag}", earlier_files, self.counts_of(mirror).get(0, {}))
        self.train(out, sources, prev, steps, f"{clineage}_{tag}", {"replay_solutions": len(rep), "mirrors": mirror}, True)
        return str(out)

    def control_extend(self, o, dom4, base):
        """The control's counterpart of the continued network's cycle in the 4th domain: for each of its dreams, new
        made-up examples as many per dimension as that dream's group 1, plus earlier made-up examples as many as its
        group 0; balanced. Returns (network, the made-up files it added)."""
        net, files = base, []
        for k in POINTS:
            mirror = f"o{o}_{dom4}_d{k}"
            c = self.counts_of(mirror)
            out = RUN / "nets" / f"c{o}x_{dom4}{k}"
            new = self.madeup(f"g2h-control-c{o}x-{dom4}{k}", c.get(1, {}))
            if not Path(f"{out}_d4.npy").exists():
                sources = [(new, 1, None)] + self.rehearsal(f"c{o}x_{dom4}{k}", cbase_files(o) + files, c.get(0, {}))
                self.train(out, sources, net, NEW["steps"], f"c{o}x_{dom4}{k}", {"mirrors": mirror}, True)
            files.append(new)
            net = str(out)
        return net, files

    def test(self, net, dom, label, budget, cap, which="test"):
        return self.search(RUN / "test" / f"{label}_{dom}.json", records(which, dom), net, f"test {label} on {dom}", budget, cap)

    def curve(self, kind, lineage, start, dom, label, o=None, earlier_files=None):
        """The first cycle in a new domain: tests before any task and after the dreams at each point. kind: fresh,
        continued or control (the control mirrors the continued network of the same order and domain)."""
        tasks = records("train", dom)
        B, cap = NEWB
        net = start
        if not SMOKE:
            self.test(net, dom, f"{label}_t0", B, cap)
        prev = 0
        for k in POINTS:
            w = self.search(RUN / "wake" / f"{label}_w{k}.json", tasks[prev:k], net, f"wake {label} tasks {prev + 1}-{k}", B, cap)
            dims = {}
            for r in tasks[:k]:
                dims[r["desc"]["d"]] = dims.get(r["desc"]["d"], 0) + 1
            tag = f"{dom}_d{k}"
            p = None if net == "uniform" else net
            if kind == "control":
                net = self.control_dream(lineage, tag, p, w, NEW["fantasies"], NEW["steps"], dims, dom, f"o{o}_{tag}", earlier_files)
            else:
                net = self.dream(lineage, tag, p, w, NEW["fantasies"], NEW["steps"], dims, dom, kind == "continued")
            self.test(net, dom, f"{label}_t{k}", B, cap, which="dev" if SMOKE else "test")
            prev = k
        return net

    def retention(self, o, net, when):
        for dom in orders(self.level)[o][:3]:
            if SMOKE:
                return
            self.test(net, dom, f"o{o}_{when}", HIST["budget"], HIST["cap"])


def saturation(D, level):
    """The pre-registered saturation check (research log, Stage 2a'' proposal): the fresh network after 30 tasks on the
    development systems; under 60% of new-form development equations (new-form relative to mechanics, circuits and
    ecology) passes; under 15% calls for a check-in."""
    shared = json.loads((SYS / f"shared_forms_g2h_L{level}.json").read_text())
    for dom in domains(level):
        tasks = records("train", dom)
        net, prev = "uniform", 0
        for k in (10, 30):
            w = D.search(RUN / "wake" / f"fresh_{dom}_w{k}.json", tasks[prev:k], net, f"wake fresh_{dom} tasks {prev + 1}-{k}",
                         *NEWB)
            net = D.dream(f"fresh_{dom}", f"{dom}_d{k}", None if net == "uniform" else net, w, NEW["fantasies"], NEW["steps"],
                          {4: 1}, dom, False)
            prev = k
        res = D.test(net, dom, f"fresh_{dom}_t30dev", *NEWB, which="dev")
        flags = shared[f"mce_dev_{dom}"]
        new = [(r["id"], i) for r in records("dev", dom) for i in range(len(r["structure"])) if not flags[r["id"]][i]]
        solved = {(r["id"], e["eq"]) for r in res if r["status"] == "done" for e in r["equations"] if e["first_exact"] is not None}
        n = sum(1 for k in new if k in solved)
        share = n / max(len(new), 1)
        verdict = "PASSES (under 60%)" if share < 0.60 else "SATURATED (60% or more): next level"
        if share < 0.15:
            verdict = "UNDER 15%: check in"
        log(f"saturation check {dom}: the fresh network after 30 tasks solves {n} of {len(new)} new-form development "
            f"equations ({100 * share:.1f}%) -> {verdict}; all equations "
            f"{len(solved)} of {sum(len(r['structure']) for r in records('dev', dom))}")


def main():
    workers, mode, level = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    RUN.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    D = Driver(workers, level)
    log(f"driver started: mode {mode}; level {level}; workers {workers}; new-domain budget {NEWB}, {NEW}; points {POINTS}; "
        f"time cap {TIME_CAP:.0f}s per system{'; SMOKE TEST' if SMOKE else ''}")
    if mode == "sat":
        saturation(D, level)
        log("saturation check done")
        return
    os_ = [int(x) for x in sys.argv[4:]] or [1, 2]
    for dom in domains(level):                                  # the fresh network does not depend on the order
        D.curve("fresh", f"fresh_{dom}", "uniform", dom, f"fresh_{dom}")
    for o in os_:
        d4, d5 = orders(level)[o][3], orders(level)[o][4]
        hist = D.prepared(HIST_NET[o])
        cbase = D.prepared(CBASE_NET[o])
        D.retention(o, hist, "before")
        c4 = D.curve("continued", f"o{o}", hist, d4, f"o{o}_{d4}")
        D.curve("control", f"c{o}_{d4}", cbase, d4, f"c{o}_{d4}", o=o, earlier_files=cbase_files(o))
        cext, ext_files = D.control_extend(o, d4, cbase)
        c5 = D.curve("continued", f"o{o}", c4, d5, f"o{o}_{d5}")
        D.curve("control", f"c{o}_{d5}", cext, d5, f"c{o}_{d5}", o=o, earlier_files=cbase_files(o) + ext_files)
        D.retention(o, c5, "after")
        if not SMOKE:
            D.test(c5, d4, f"o{o}_after", *NEWB)
        log(f"order {o}: all phases done")
    log("all requested orders done")


if __name__ == "__main__":
    main()
