"""Step 2 of the harder fields (owner, 14 November; research log "Step 2 of the harder fields: the short search test
with the confidence-weighted blend", where the design and pass rule were set before the run). Reaction networks
(learned 4th after order 1's history) and flows (4th after order 2's), from claim run 2's history networks and control
bases; arms: the confidence-weighted blend, the network alone, the made-up control (its own library, the same
confidence blend) and a fresh network; 2 seeds (each its own 30 of the 100 training tasks, in its own order); wakes on
tasks 1-10 and 11-30, each followed by a dream, a library rebuild, Reanalyse (the standing rule: unsolved training
tasks searched again with the newest prior at 100 structures per equation, 20 thorough fits) and a test on 40 new test
systems per field ("test3") at 300 structures per equation.
Usage: harder_step2.py <workers> <draw|run|systems|report>   (environment: G2H_SMOKE=1 for the smoke test)"""

import hashlib
import json
import math
import os
import pickle
import random
import sys
import threading
import time
from pathlib import Path

SMOKE = os.environ.get("G2H_SMOKE") == "1"
os.environ.setdefault("G2H_RUN", "step2_smoke" if SMOKE else "step2")
os.environ.setdefault("G2H_SHORTLIST", "2048")
os.environ.setdefault("G2H_TEST_SET", "test3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402  (claim run 2's driver; RUN = g2h/step2)
import harder_check as H                                         # noqa: E402

RUN, SYS, RUN2 = C.RUN, C.SYS, H.RUN2
FIELDS = {"reactions_L1": 1, "flows_L1": 2}                     # field -> the order whose history it follows (4th)
ARMS = ("blend", "net", "ctrl", "fresh")
ARM_NAMES = {"blend": "confidence blend", "net": "network alone", "ctrl": "made-up control", "fresh": "fresh network"}
SEEDS = (1, 2) if not SMOKE else (1,)
POINTS = (10, 30) if not SMOKE else (2, 4)
N_TEST = 40 if not SMOKE else 2
REANALYSE = (100, 20) if not SMOKE else (10, 2)
B, CAP = C.NEWB
_LOCK = threading.Lock()


def log(msg):
    with _LOCK:
        C.log(msg)


# ---------------------------------------------------------------------------------------------------------------------
# the prior: network, uniform, or the confidence-weighted blend of a network and a library table
# ---------------------------------------------------------------------------------------------------------------------

class ConfidencePrior:
    """(1 - u) * p_network + u * p_blend over the term to add (and the added term of a swap), u = 1 - the network's top
    probability over the legal additions; p_blend the plain top-100 blend (curriculum_blend.BlendPrior); removals stay
    the network's (quick check 1's rule)."""

    def __init__(self, base, table):
        self.base, self.table = base, table

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, node, ctx):
        import numpy as np
        import muzero_checks as MZ
        pr = self.base(node, ctx)
        legal = pr.legal_add
        if not legal.any():
            return pr
        lib = self.table[ctx["vocab"].gid, ctx["i"]]
        pn = np.where(legal, pr.p_swap_add, 0.0)
        pn = pn / pn.sum()
        pb = MZ.blend_dist(lib, pn, legal)
        u = 1.0 - float(pn.max())
        mix = (1.0 - u) * pn + u * pb
        share = pr.p_add.sum()
        if share > 0:
            pr.p_add = share * mix
        pr.p_swap_add = mix
        return pr


_TAB = {}


def make_prior(spec):
    import numpy as np
    if spec[0] == "uniform":
        return C._prior("uniform")
    base = C._prior(spec[1])
    if spec[0] == "net":
        return base
    if spec[2] not in _TAB:
        _TAB[spec[2]] = np.load(spec[2])
    return ConfidencePrior(base, _TAB[spec[2]])


def search_job(args):
    """cycles_g2h.search_job with a prior spec: ("uniform",), ("net", path) or ("confidence", net path, table path)."""
    rec, spec, seed, budget, cap = args
    import signal
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "domain": rec["domain"], "status": "done", "budget": budget}
    t0 = time.time()
    signal.signal(signal.SIGALRM, C._alarm)
    signal.alarm(int(2 * C.TIME_CAP + 600))
    try:
        prior = make_prior(spec)
        res = SG.search_system(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed,
                               probes=[eq["keys"] for eq in rec["structure"]])
        out["equations"] = []
        for e in res["equations"]:
            ef = e["effort"]
            out["equations"].append({"eq": e["eq"], "first_exact": ef["first_exact"], "structures": ef["structures"],
                                     "seconds": ef["seconds"], "thorough": ef["thorough"], "time_capped": ef["time_capped"],
                                     "n_exact": e["n_exact"], "solution": e.get("solution"), "prior_seconds": ef["prior_seconds"],
                                     "fit_errors": ef.get("fit_errors", 0)})
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the test draw
# ---------------------------------------------------------------------------------------------------------------------

def step_draw(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import g2h_draw as GD
    jobs = [("test3", d, k) for d in FIELDS for k in range(N_TEST)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        recs = list(ex.map(GD.job, jobs, chunksize=1))
    old = set()
    for d in FIELDS:
        for which in ("train", "dev", "pilot", "test", "test2"):
            p = SYS / f"{which}_{d}.json"
            if p.exists():
                old |= {r["hash"] for r in json.loads(p.read_text()) if r.get("hash")}
    for d in FIELDS:
        rs = sorted([r for r in recs if r["domain"] == d], key=lambda r: r["k"])
        acc = [r for r in rs if r["status"] == "accepted"]
        clash = sum(r["hash"] in old for r in acc)
        print(f"test3 {d}: accepted {len(acc)} of {len(rs)}; rejected draws {sum(r['rejections'] for r in rs)}; identical to an "
              f"earlier system {clash}")
        if clash:
            raise SystemExit("test3 is not new: not writing it")
        (SYS / f"test3_{d}.json").write_text(json.dumps(rs))
    print(f"draw: {time.time() - t0:.0f}s")


def test_records(field):
    return C.records("dev", field)[:N_TEST] if SMOKE else C.records("test3", field)


# ---------------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------------

class Step2(C.Driver):
    def __init__(self, workers):
        super().__init__(workers, 1)
        for sub in ("reanalyse", "tables"):
            (RUN / sub).mkdir(parents=True, exist_ok=True)
        self.canon_path = RUN / "canon.json"
        self.canon = json.loads(self.canon_path.read_text()) if self.canon_path.exists() else {}
        self.canon_lock = threading.Lock()
        self.made = {o: H.madeup_equations(o, 3, len(self.history(o)))[0] for o in set(FIELDS.values())}
        self._init_gpus()

    @staticmethod
    def history(o):
        return {k: v for k, v in json.loads((C.REUSE / f"replays_o{o}.json").read_text()).items()
                if C.dom_of(k.rsplit("|", 1)[0]) in C.HISTORY}

    def replays(self, lineage):
        """Every lineage's verified solutions; the confidence blend and the network alone start with their history's."""
        p = RUN / f"replays_{lineage}.json"
        with _LOCK:
            if p.exists():
                return json.loads(p.read_text())
            arm, field = lineage.split("-")[0], lineage.split("-")[1]
            rep = self.history(FIELDS[field]) if arm in ("blend", "net") else {}
            p.write_text(json.dumps(rep))
            return rep

    # -- GPUs: the GPUs free at the start, at most 2 of this run's trainings on each; nvidia-smi before each training --
    @staticmethod
    def _gpu_rows():
        import subprocess
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        return {int(r[0]): [int(x) for x in r] for r in (line.split(",") for line in q.strip().splitlines())}

    def _init_gpus(self):
        rows = self._gpu_rows()
        free = [g for g, (_, u, m, tot) in sorted(rows.items()) if u < 10 and m < 1000][:4]
        self.slots = {g: threading.Semaphore(1) for g in free}         # one training peaks near 10 GB: one per GPU
        self._tl = threading.local()
        log(f"step 2 GPUs: {free} (free at the start: utilization under 10%, under 1,000 MiB used); one training at a time each")

    def train(self, out, sources, prev, steps, label, info, balanced):
        if not getattr(self, "slots", None):
            return C.Driver.train(self, out, sources, prev, steps, label, info, balanced)
        while True:
            rows = self._gpu_rows()
            for g, sem in self.slots.items():
                _, u, m, tot = rows[g]
                if tot - m >= 11000 and u <= 90 and sem.acquire(blocking=False):
                    self._tl.gpu = g
                    try:
                        return C.Driver.train(self, out, sources, prev, steps, label, info, balanced)
                    finally:
                        self._tl.gpu = None
                        sem.release()
            time.sleep(15)

    def gpu(self, poll=60):
        g = getattr(getattr(self, "_tl", None), "gpu", None)
        if g is None:
            return C.Driver.gpu(self, poll)
        return g, f"GPU {g} (step 2 slot: one of this run's trainings per GPU; at least 11 GB free at the check)"

    def search2(self, path, recs, spec, label, budget, cap):
        if path.exists():
            return json.loads(path.read_text())
        t0 = time.time()
        res = list(self.pool.map(search_job, [(r, spec, 1, budget, cap) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        eqs = [e for r in res if r["status"] == "done" for e in r["equations"]]
        log(f"{label}: {len(recs)} systems in {time.time() - t0:.0f}s; equations solved {sum(e['first_exact'] is not None for e in eqs)} "
            f"of {sum(len(r['structure']) for r in recs)}; time-capped {sum(e['time_capped'] for e in eqs)}; failed "
            f"{sum(r['status'] != 'done' for r in res)}")
        return res

    def tasks(self, field, seed):
        allt = C.records("train", field)
        if SMOKE:
            return allt[:POINTS[-1]]
        return random.Random(f"step2-{field}-s{seed}").sample(allt, POINTS[-1])

    def fixed(self, sid, eq, keys):
        return self.canon.get(H.canon_key(sid, eq, keys), keys)

    def fix_new(self, rep):
        """The extended equivalence fix on every new-field solution of a replay set not fixed before."""
        import canon_check as CN
        todo = {}
        for k, keys in rep.items():
            sid, eq = k.rsplit("|", 1)
            if sid.startswith("g2h-") and H.canon_key(sid, int(eq), keys) not in self.canon:
                todo.setdefault(sid, {})[eq] = keys
        if not todo:
            return
        futs = {sid: self.pool.submit(CN.canon_job, (self.recs[sid], d)) for sid, d in todo.items()}
        with self.canon_lock:
            for sid, f in futs.items():
                for eq, e in f.result()["equations"].items():
                    self.canon[H.canon_key(sid, int(eq), e["raw"])] = e["canon_plus"]
            self.canon_path.write_text(json.dumps(self.canon))

    def library(self, lineage, arm, field, tag):
        """The lineage's library table: the blend's own solutions (history raw, new-field fixed), or the control's
        made-up equations plus its own new-field solutions (fixed)."""
        path = RUN / "tables" / f"{lineage}_{tag}.npy"
        if path.exists():
            return str(path)
        rep = self.replays(lineage)
        self.fix_new(rep)
        eqs = [(int(k.rsplit("|", 1)[1]), self.fixed(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1]), v) if k.startswith("g2h-") else v)
               for k, v in rep.items()]
        if arm == "ctrl":
            eqs = list(self.made[FIELDS[field]]) + eqs
        info = self.pool.submit(H.table_job, (eqs, str(path))).result()
        log(f"library {lineage} {tag}: {len(eqs)} equations, {len(info['pieces'])} pieces")
        return str(path)

    def prior(self, arm, lineage, net, field, tag):
        if net == "uniform":
            return ("uniform",)
        if arm in ("net", "fresh"):
            return ("net", net)
        return ("confidence", net, self.library(lineage, arm, field, tag))

    def reanalyse(self, lineage, tasks, spec, k):
        """The standing rule: every training task with an unsolved equation, searched again with the newest prior."""
        path = RUN / "reanalyse" / f"{lineage}_r{k}.json"
        ypath = RUN / "reanalyse" / f"{lineage}_r{k}_yield.json"
        if ypath.exists():
            return json.loads(ypath.read_text())
        rep = self.replays(lineage)
        unsolved = [r for r in tasks if any(f"{r['id']}|{i}" not in rep for i in range(len(r["structure"])))]
        before = {f"{r['id']}|{i}" for r in unsolved for i in range(len(r["structure"])) if f"{r['id']}|{i}" in rep}
        res = self.search2(path, unsolved, spec, f"reanalyse {lineage} after {k} tasks", *REANALYSE) if unsolved else []
        rep = self.add_replays(lineage, res)
        new_eq = sum(1 for r in unsolved for i in range(len(r["structure"])) if f"{r['id']}|{i}" in rep and f"{r['id']}|{i}" not in before)
        new_sys = sum(1 for r in unsolved if all(f"{r['id']}|{i}" in rep for i in range(len(r["structure"]))))
        y = {"lineage": lineage, "after_tasks": k, "re_searched": len(unsolved), "new_equations": new_eq, "new_systems": new_sys,
             "structures": sum(e["structures"] for r in res if r["status"] == "done" for e in r["equations"]),
             "time_capped": sum(e["time_capped"] for r in res if r["status"] == "done" for e in r["equations"])}
        ypath.write_text(json.dumps(y))
        log(f"reanalyse {lineage} after {k} tasks: re-searched {len(unsolved)} systems; newly solved {new_eq} equations and "
            f"{new_sys} whole systems; {y['structures']} structures")
        return y

    def lineage(self, arm, field, seed):
        o = FIELDS[field]
        name = f"{arm}-{field}-s{seed}"
        tasks = self.tasks(field, seed)
        tests = test_records(field)
        net = {"blend": str(RUN2 / "nets" / C.HIST_NET[o]), "net": str(RUN2 / "nets" / C.HIST_NET[o]),
               "ctrl": str(RUN2 / "nets" / C.CBASE_NET[o]), "fresh": "uniform"}[arm]
        spec = self.prior(arm, name, net, field, "d0")
        prev = 0
        for k in POINTS:
            w = self.search2(RUN / "wake" / f"{name}_w{k}.json", tasks[prev:k], spec, f"wake {name} tasks {prev + 1}-{k}", B, CAP)
            tag = f"{field}_d{k}"
            dims = {4: k}
            if arm == "ctrl":
                mirror = f"net-{field}-s{seed}_{tag}"
                while not (RUN / "dream" / f"{mirror}_train.json").exists():
                    time.sleep(30)
                net = self.control_dream(name, tag, net, w, C.NEW["fantasies"], C.NEW["steps"], dims, field, mirror, C.cbase_files(o))
            else:
                net = self.dream(name, tag, None if net == "uniform" else net, w, C.NEW["fantasies"], C.NEW["steps"], dims, field,
                                 arm != "fresh")
            spec = self.prior(arm, name, net, field, tag)
            self.reanalyse(name, tasks[:k], spec, k)
            self.search2(RUN / "test" / f"{name}_t{k}_{field}.json", tests, spec, f"test {name} after {k} tasks", B, CAP)
            prev = k
        log(f"{name}: done")


def step_run(workers):
    D = Step2(workers)
    jobs = [(a, f, s) for f in FIELDS for s in SEEDS for a in ARMS]
    log(f"step 2: {len(jobs)} lineages; points {POINTS}; budgets {C.NEWB}; Reanalyse {REANALYSE}; workers {workers}")
    threads = [threading.Thread(target=D.lineage, args=j, name="-".join(map(str, j))) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("step 2: all lineages done")


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def newform_test(field):
    cache = {}
    seen = {H.rep_form(eq["keys"], cache) for d in C.HISTORY for r in H.records("train", d) for eq in r["structure"]}
    return {(r["id"], i) for r in test_records(field) for i, eq in enumerate(r["structure"]) if H.rep_form(eq["keys"], cache) not in seen}


def solved(res):
    return {(r["id"], e["eq"]) for r in res if r["status"] == "done" for e in r["equations"] if e["first_exact"] is not None}


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(a, b) + 1)) / 2 ** n) if n else 1.0


def spearman(x, y):
    import numpy as np
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def step_report():
    lines = ["== Step 2 of the harder fields: new-form test equations solved (40 test systems per field)"]
    ok_all, lin_rows = True, []
    for field in FIELDS:
        new = newform_test(field)
        lines.append(f"   {field} (new-form test equations: {len(new)}):")
        S = {}
        for arm in ARMS:
            for s in SEEDS:
                for k in POINTS:
                    p = RUN / "test" / f"{arm}-{field}-s{s}_t{k}_{field}.json"
                    S[(arm, s, k)] = solved(json.loads(p.read_text())) & new if p.exists() else None
                    sp = RUN / "test" / f"{arm}-{field}-s{s}_t{k}_{field}_systems.json"
                    if sp.exists():
                        S[(arm, s, k, "exact")] = sum(x.get("exact", False) for x in json.loads(sp.read_text()))
        for k in POINTS:
            lines.append(f"      after {k} tasks: " + "; ".join(
                f"{ARM_NAMES[a]} " + " / ".join(str(len(S[(a, s, k)])) if S[(a, s, k)] is not None else "-" for s in SEEDS)
                + (f" (whole systems exact {' / '.join(str(S.get((a, s, k, 'exact'), '-')) for s in SEEDS)})" if any((a, s, k, "exact") in S for s in SEEDS) else "")
                for a in ARMS))
        k = POINTS[-1]
        if any(S[(a, s, k)] is None for a in ARMS for s in SEEDS):
            ok_all = False
            lines.append("      incomplete")
            continue
        ok = True
        for other in ("net", "ctrl"):
            each = all(len(S[("blend", s, k)]) > len(S[(other, s, k)]) for s in SEEDS)
            a_ = sum(len(S[("blend", s, k)] - S[(other, s, k)]) for s in SEEDS)
            b_ = sum(len(S[(other, s, k)] - S[("blend", s, k)]) for s in SEEDS)
            p = sign_p(a_, b_)
            ok &= each and a_ > b_ and p < 0.05
            lines.append(f"      vs {ARM_NAMES[other]}: more in each seed {each}; only the blend {a_}, only {ARM_NAMES[other]} {b_}, p = {p:.2g}"
                         + (" (between 0.01 and 0.05: to be confirmed by a rerun)" if 0.01 <= p < 0.05 else ""))
        nf = all(len(S[("blend", s, k)]) >= len(S[("fresh", s, k)]) for s in SEEDS)
        ok &= nf
        lines.append(f"      no fewer than the fresh network in each seed: {nf}")
        ok_all &= ok
        lines.append(f"      {field}: {'meets' if ok else 'does not meet'} the rule")
        for arm in ARMS:
            for s in SEEDS:
                name = f"{arm}-{field}-s{s}"
                ys = [json.loads((RUN / "reanalyse" / f"{name}_r{kk}_yield.json").read_text()) for kk in POINTS
                      if (RUN / "reanalyse" / f"{name}_r{kk}_yield.json").exists()]
                gain = len(S[(arm, s, POINTS[-1])]) - len(S[(arm, s, POINTS[0])])
                retried = ys[0]["new_equations"] if ys else 0           # only these can shape the learning between the tests
                lin_rows.append((name, retried, gain, ys))
    lines.append("== Reanalyse's yield (per lineage: after 10 / after 30 tasks: systems re-searched, equations newly solved, "
                 "whole systems newly solved, structures)")
    for name, retried, gain, ys in lin_rows:
        lines.append(f"   {name}: " + " | ".join(f"{y['re_searched']} re-searched, {y['new_equations']} equations, {y['new_systems']} "
                                              f"systems, {y['structures']} structures" for y in ys)
                     + f"; retried solutions in time for the second dream {retried}; gain from {POINTS[0]} to {POINTS[-1]} tasks {gain:+d}")
    if len(lin_rows) > 2:
        lines.append(f"   rank correlation across lineages, retried solutions vs gain: {spearman([r[1] for r in lin_rows], [r[2] for r in lin_rows]):.2f} "
                     "(better learners also solve more on retry: a correlation does not show cause)")
    lines.append(f"== Pass rule (both fields): {'PASSES' if ok_all else 'does not pass'}")
    text = "\n".join(lines)
    (RUN / "report.txt").write_text(text)
    print(text)


def step_systems(workers):
    import subprocess
    files = sorted(str(p) for p in (RUN / "test").glob("*_t*_*.json") if not p.stem.endswith("_systems"))
    env = dict(os.environ, G2H_TEST_SET="dev" if SMOKE else "test3")
    subprocess.run([sys.executable, str(HERE / "g1_systems.py"), str(workers)] + files, env=env, check=True)


def main():
    import shutil
    workers, step = int(sys.argv[1]), sys.argv[2]
    RUN.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RUN.resolve()).free / 2 ** 30
    if free < 2:
        raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {RUN.resolve()} (owner's crash guard)")
    {"draw": lambda: step_draw(workers), "run": lambda: step_run(workers), "systems": lambda: step_systems(workers),
     "report": step_report}[step]()


if __name__ == "__main__":
    main()
