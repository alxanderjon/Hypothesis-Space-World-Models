"""A short search test of arm 3's top-100 blend (owner, 13 November; research log "Owner: arm 3's quick-check rule
overridden; a short search test of the top-100 blend"): the curriculum pilot's network lineage, unchanged except for the
search's prior. After each round the network is trained as in the network arm, and the library is rebuilt from the
lineage's verified solutions rewritten by the extended equivalence fix (canon_check.canon_job: the simplest exact form
on each system's data), context-free form, compression plus counting. The prior over the term to add: the library's 100
most probable addable terms share 95% in proportion to the geometric mean of the library's and the network's
probabilities; the other 5% is spread over all other addable terms in proportion to the library's; removals stay the
network's. The pilot's systems, seeds 1 and 2; compared with the pilot's network-alone lineages.
Usage: curriculum_blend.py <workers> <run|report> [smoke]"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
os.environ.setdefault("G2H_RUN", "cblend_smoke" if SMOKE else "cblend")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_pilot as CP                                    # noqa: E402
import canon_check as CN                                         # noqa: E402

C, D = CP.C, CP.D
OUT = C.RUN
PILOT = C.ROOT / "outputs/accumulating_learner/g2h/cpilot"
TOP, SHARE_TOP = 100, 0.95


class BlendPrior:
    """A network prior whose distribution over the term to add (and the added term of a swap) is the top-100 blend."""

    def __init__(self, base, table):
        self.base, self.table = base, table

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, node, ctx):
        import numpy as np
        pr = self.base(node, ctx)
        legal = pr.legal_add
        n = int(legal.sum())
        if not n:
            return pr
        lib = self.table[ctx["vocab"].gid, ctx["i"]]
        order = np.lexsort((np.arange(len(lib)), -np.where(legal, lib, -np.inf)))
        top = order[:min(TOP, n)]
        g = 0.5 * lib[top] + 0.5 * np.log(np.maximum(pr.p_swap_add[top], 1e-300))
        mix = np.zeros(len(lib))
        mix[top] = np.exp(g - g.max())
        mix[top] *= (SHARE_TOP if n > len(top) else 1.0) / mix[top].sum()
        rest = legal.copy()
        rest[top] = False
        if rest.any():
            r = np.where(rest, lib, -np.inf)
            w = np.exp(r - r[rest].max())
            mix += (1.0 - SHARE_TOP) * w / w.sum()
        share = pr.p_add.sum()
        if share > 0:
            pr.p_add = share * mix
        pr.p_swap_add = mix
        return pr


def search_job(args):
    """curriculum_pilot.search_job with the blend prior ("blend", network, library table) available."""
    rec, prior, seed, budget, cap = args
    if prior[0] != "blend":
        return CP.search_job(args)
    import signal
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "level": rec["level"], "status": "done", "budget": budget}
    t0 = time.time()
    signal.signal(signal.SIGALRM, C._alarm)
    signal.alarm(int(2 * C.TIME_CAP + 600))
    try:
        p = BlendPrior(C._prior(prior[1]), np.load(prior[2]))
        res = SG.search_system(np.array(rec["bursts"]), p, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed)
        out["equations"] = [{"eq": e["eq"], "first_exact": e["effort"]["first_exact"], "structures": e["effort"]["structures"],
                             "time_capped": e["effort"]["time_capped"], "solution": e["solution"]["keys"] if e.get("solution") else None,
                             "true": bool(e.get("solution")) and sorted(e["solution"]["keys"]) == sorted(true)}
                            for e, true in zip(res["equations"], rec["structure"])]
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["solved"] = out["status"] == "done" and all(e["first_exact"] is not None for e in out["equations"])
    out["seconds"] = time.time() - t0
    return out


class BlendLineage(CP.Lineage):
    """The pilot's network lineage ("net" schedule and training) with the top-100 blend as its search prior."""

    def __init__(self, seed, pool, gpus, systems, tokens, order):
        super().__init__("net", seed, pool, gpus, systems, [], tokens, order)
        self.name = f"net-blend-s{seed}"
        old, self.dir = self.dir, OUT / "lineages" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prior = ("grammar", str(self.dir / "size.npy"))
        CP.grammar_table([], self.prior[1], tokens, order)
        if (old / "size.npy").exists():                          # the parent class's folder under the old name
            (old / "size.npy").unlink()
        if old.exists() and not any(old.iterdir()):
            old.rmdir()
        self.canon = {}                                          # (system id, eq, found keys) -> canonical keys
        self.recs = {r["id"]: r for lv in CP.LEVELS for r in systems[lv]["train"]}
        self.rec["kind"] = "net-blend"

    def search(self, recs, budget, cap):
        futs = [self.pool.submit(search_job, (r, self.prior, self.seed, budget, cap)) for r in recs]
        return [f.result() for f in futs]

    def extra_state(self):
        return {"canon": self.canon}

    def set_extra_state(self, extra):
        self.canon = extra.get("canon", {})

    def learn(self, level=None, j=0):
        info = super().learn(level, j)                           # the network, exactly as in the network arm
        todo = {}
        for sid, d in self.sols.items():
            for eq, keys in d.items():
                if (sid, int(eq), tuple(keys)) not in self.canon:
                    todo.setdefault(sid, {})[str(eq)] = keys
        futs = {sid: self.pool.submit(CN.canon_job, (self.recs[sid], d)) for sid, d in todo.items()}
        for sid, f in futs.items():
            for eq, e in f.result()["equations"].items():
                self.canon[(sid, int(eq), tuple(e["raw"]))] = e["canon_plus"]
        eqs = [(int(eq), self.canon[(sid, int(eq), tuple(keys))]) for sid, d in self.sols.items() for eq, keys in d.items()]
        path = str(self.dir / f"library{self.n_learn}.npy")
        info["library_pieces"] = CP.grammar_table(eqs, path, self.tokens, self.order)
        info["library_equations"] = len(eqs)
        self.prior = ("blend", self.net, path)
        return info


def step_run(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import grammar_ops as GO
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    tokens = GO.alphabet(order, d=D)
    gpus = queue.Queue()
    free = CP.CC.free_gpus()
    if not free:
        raise SystemExit("no free GPU now; the network does not train on the CPU (owner's rule)")
    for g in free[:2]:
        gpus.put(g)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        lineages = []
        for s in CP.SEEDS:
            if (OUT / "lineages" / f"net-blend-s{s}.json").exists():
                continue
            systems = {int(k): v for k, v in json.loads((PILOT / "gen" / f"systems-s{s}.json").read_text()).items()
                       if int(k) in CP.LEVELS}
            if SMOKE:
                systems = {k: {sp: v[sp][:CP.N_TRAIN if sp == "train" else CP.N_TEST] for sp in v} for k, v in systems.items()}
            lineages.append(BlendLineage(s, pool, gpus, systems, tokens, order))
        CP.log(f"top-100 blend test: {len(lineages)} lineages; levels {CP.LEVELS}; budgets {CP.BUDGETS}; GPUs {free[:2]}; workers {workers}")
        threads = [threading.Thread(target=x.run, name=x.name) for x in lineages]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    CP.log("top-100 blend test: done")


def step_report():
    import numpy as np
    lines = ["== Top-100 blend against the network alone (the pilot), final test: held-out systems solved per level (of 10)"]
    ok_all = True
    for s in CP.SEEDS:
        b = json.loads((OUT / "lineages" / f"net-blend-s{s}.json").read_text())
        n = json.loads((PILOT / "lineages" / f"net-s{s}.json").read_text())
        per = {name: {lv: sum(y["solved"] for y in rec["final"] if y["level"] == lv) for lv in CP.LEVELS}
               for name, rec in (("blend", b), ("network", n))}
        hi = {k: sum(v.get(lv, 0) for lv in (6, 7)) for k, v in per.items()}
        lo = {k: sum(v.get(lv, 0) for lv in (1, 2, 3, 5)) for k, v in per.items()}
        ok = hi["blend"] > hi["network"] and lo["blend"] >= lo["network"] - 1
        ok_all &= ok
        rounds = {name: " ".join(f"L{L_['level']} {len(L_['rounds']) if L_['mastered'] else 'x'}" for L_ in rec["levels"])
                  for name, rec in (("blend", b), ("network", n))}
        fe = {name: float(np.median([e["first_exact"] for y in [y for L_ in rec["levels"] for r in L_["rounds"] for y in r["wake"]]
                                     if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None]))
              for name, rec in (("blend", b), ("network", n))}
        for name in ("blend", "network"):
            lines.append(f"   seed {s} {name:>8}: " + "; ".join(f"L{lv} {per[name][lv]}" for lv in CP.LEVELS)
                         + f" | levels 6-7 {hi[name]} of 20, levels 1-5 {lo[name]} of 40 | rounds to mastery: {rounds[name]} | "
                         f"median structures before a solution (training) {fe[name]:.0f}")
        lines.append(f"   seed {s}: levels 6-7 more with the blend {hi['blend'] > hi['network']}; levels 1-5 no loss beyond 1 "
                     f"{lo['blend'] >= lo['network'] - 1} -> {'meets' if ok else 'does not meet'} the rule")
    lines.append(f"== Pass rule (both seeds): {'PASSES' if ok_all else 'does not pass'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers = int(ARGS[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ARGS[2:] if s != "smoke"]:
        free = shutil.disk_usage(OUT.resolve()).free / 2 ** 30
        if free < 2:
            raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {OUT.resolve()} (owner's crash guard)")
        {"run": lambda: step_run(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
