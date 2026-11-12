"""K1 of the compositional plan, the same-network test (owner's approval, 11 November; research log "Proposed, awaiting
approval: ... K1" and "Owner: skip the replication; the all-parses check, then K1 with ordering C"): the grammar's order
inside today's search. Test only (no training), on the development systems, budget 300, thorough cap 60, the 2,048-term
shortlist, search_g1's search otherwise (the f6 rules), one new search seed (4) for every test.

Orderings (the distribution over the term to add, and the added term of a swap; removals stay the network's):
  A  the network alone;
  B  the network mixed half and half with K0's operation grammar fitted on the network's own lineage at 30 tasks (K0b's
     "ctx" table of that lineage point);
  C  the same with the context-free grammar and its library, fitted over all parses (K0b's all-parses check:
     "uni_lib_all").
Networks: the 30-task networks, flows: continued and fresh f6 seeds 1-3, controls n and ns; reaction networks:
continued and fresh f6 seed 1 and versions n, ns, controls n and ns. The f6 networks' earlier development tests (seeds
1-3) give the A-versus-A noise. Development systems only; counts only.
Usage: k1_check.py <workers> [smoke] (smoke: one network per domain, two development systems, budget 20, written to
k1_smoke; prints only statuses and timings)"""

import json
import os
import signal
import sys
import time
from pathlib import Path

SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
os.environ.setdefault("G2H_RUN", "k1_smoke" if SMOKE else "k1")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402

X_RUN = C.ROOT / "outputs/accumulating_learner/g2h/stageX"
PILOT = C.ROOT / "outputs/accumulating_learner/g2h/pilot"
DOMS = {"flows": "flows_L1", "reactions": "reactions_L1"}
NETS = {"flows": {"continued": [("f6 seed 1", PILOT, "p6_o2", 1), ("f6 seed 2", X_RUN, "xf6r2_o2", 2),
                                ("f6 seed 3", X_RUN, "xf6r3_o2", 3)],
                  "fresh": [("f6 seed 1", PILOT, "p6fresh_2", 1), ("f6 seed 2", X_RUN, "xfreshf6r2_2", 2),
                            ("f6 seed 3", X_RUN, "xfreshf6r3_2", 3)],
                  "control": [("n", X_RUN, "xcn_2", None), ("ns", X_RUN, "xcns_2", None)]},
        "reactions": {"continued": [("f6 seed 1", PILOT, "q6_o1", 1), ("n", X_RUN, "xn_o1", None), ("ns", X_RUN, "xns_o1", None)],
                      "fresh": [("f6 seed 1", PILOT, "q6fresh_1", 1), ("n", X_RUN, "xfreshn_1", None),
                                ("ns", X_RUN, "xfreshns_1", None)],
                      "control": [("n", X_RUN, "xcn_1", None), ("ns", X_RUN, "xcns_1", None)]}}
ORDERINGS = {"A": None, "B": ("k0b", "ctx"), "C": ("k0b2", "uni_lib_all")}
SEED, K = 4, 30
BUDGET = (300, 60) if not SMOKE else (20, 4)
OUT = C.RUN / "test"


class MixedPrior:
    """A network prior whose distribution over the term to add (and the added term of a swap) is mixed with a grammar's,
    renormalized over the terms not in the structure: (1 - w) network + w grammar. table: log P per vocabulary term
    (search_g1.vocabulary_order(4)) and equation."""

    def __init__(self, base, table, w=0.5):
        self.base, self.table, self.w = base, table, w
        self._rows = {}

    def __getattr__(self, name):                                 # seconds, probe, probe_eq of the network prior
        return getattr(self.base, name)

    def __call__(self, node, ctx):
        import numpy as np
        pr = self.base(node, ctx)
        legal = pr.legal_add
        if not legal.any():
            return pr
        key = (id(ctx["vocab"]), ctx["i"])
        if key not in self._rows:
            self._rows[key] = self.table[ctx["vocab"].gid, ctx["i"]]
        g = np.where(legal, self._rows[key], -np.inf)
        pg = np.exp(g - g[legal].max())
        pg /= pg.sum()
        mix = (1.0 - self.w) * pr.p_swap_add + self.w * pg
        share = pr.p_add.sum()
        if share > 0:
            pr.p_add = share * mix
        pr.p_swap_add = mix
        return pr


def job(args):
    """cycles_g2h.search_job with the ordering's prior."""
    rec, net, table, seed, budget, cap = args
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
        prior = C._prior(net)
        if table is not None:
            path, key = table
            prior = MixedPrior(prior, np.load(path)[key])
        res = SG.search_system(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed,
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
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    return out


def table_of(which, arm, run, ordering):
    """(npz path, key) of the ordering's grammar for the lineage point (arm, run, 30 tasks), or None for A."""
    if ORDERINGS[ordering] is None:
        return None
    folder, key = ORDERINGS[ordering]
    name = "libraries" if folder == "k0b" else "fits"
    pts = json.loads((X_RUN / folder / f"{name}_{which}.json").read_text())
    j = next(j for j, p in enumerate(pts) if p["arm"] == arm and p["run"] == run and p["k"] == K)
    return str(X_RUN / folder / f"tables_{which}_{j}.npz"), key


def tests():
    """Every (domain, arm, run, network, ordering), interleaved: the r-th network of each domain and arm, A, B, C."""
    out = []
    for r in range(3):
        for which, arms in NETS.items():
            for arm, nets in arms.items():
                if r < len(nets):
                    run, folder, lin, _ = nets[r]
                    for o in ORDERINGS:
                        out.append((which, arm, run, str(folder / "nets" / f"{lin}_{DOMS[which]}_d{K}"), lin, o))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    todo = tests()
    if SMOKE:
        todo = [t for t in todo if t[1] == "continued" and t[2] == "f6 seed 1"]
    C.log(f"K1 same-network test started: {len(todo)} tests; seed {SEED}; budget {BUDGET}; shortlist "
          f"{os.environ['G2H_SHORTLIST']}; time cap {C.TIME_CAP:.0f}s; workers {workers}")
    recs = {w: C.records("dev", d)[:2] if SMOKE else C.records("dev", d) for w, d in DOMS.items()}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        queued = []                                              # every test's systems queued at once, in test order,
        for which, arm, run, net, lin, o in todo:                # so no worker waits for a test's slowest system
            path = OUT / f"{lin}_{o}_s{SEED}_{DOMS[which]}.json"
            if not path.exists():
                tab = table_of(which, arm, run, o)
                queued.append(((which, arm, run, o), path, [pool.submit(job, (r, net, tab, SEED, *BUDGET)) for r in recs[which]]))
        t0 = time.time()
        for (which, arm, run, o), path, futs in queued:
            res = [f.result() for f in futs]
            path.write_text(json.dumps(res, default=float))
            eqs = [e for r in res if r["status"] == "done" for e in r["equations"]]
            msg = (f"{which} {arm} {run} ordering {o}: done at {time.time() - t0:.0f}s; failed "
                   f"{sum(r['status'] != 'done' for r in res)}; time-capped {sum(e['time_capped'] for e in eqs)}")
            if not SMOKE:
                msg += f"; equations solved {sum(e['first_exact'] is not None for e in eqs)} of {len(eqs)}"
            C.log(msg)
    C.log("K1 same-network test done")


if __name__ == "__main__":
    main()
