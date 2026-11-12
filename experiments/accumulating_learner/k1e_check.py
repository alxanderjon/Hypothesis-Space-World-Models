"""Build-and-check inside today's search, the same-network test (owner, 12 November; research log "Owner: the flows
same-network test of build-and-check; reaction networks if it passes"). Ordering E: at every node the choice of the term
to add (and the added term of a swap) gives half its weight to the network's distribution and half, evenly, to the 10
terms with the most evidence (search_g1.evidence, the data's score) among the network's top 1,024 addable terms;
removals stay the network's. Otherwise K1's test (k1_check): development systems, budget 300, thorough cap 60, the
2,048-term shortlist, search seed 4, test only. Compared with K1's ordering-A runs on the same networks.
Usage: k1e_check.py <workers> <flows|reactions|report> [smoke]"""

import json
import os
import sys
import time
from pathlib import Path

SMOKE = "smoke" in sys.argv[2:]
os.environ.setdefault("G2H_RUN", "k1_smoke" if SMOKE else "k1")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1_check as K                                             # noqa: E402

C = K.C
TOP, PICK = 1024, 10
NETS = {"flows": [(arm, run, lin) for arm in ("continued", "fresh") for run, _, lin, _ in K.NETS["flows"][arm]],
        "reactions": [(arm, run, lin) for arm in ("continued", "fresh") for run, _, lin, _ in K.NETS["reactions"][arm]]}
FOLDER = {lin: folder for w in K.NETS for arm in K.NETS[w] for _, folder, lin, _ in K.NETS[w][arm]}


class ScreenedPrior:
    """A network prior whose distribution over the term to add is (1 - w) network + w evenly over the PICK terms with
    the most evidence among the network's TOP addable terms."""

    def __init__(self, base, w=0.5):
        self.base, self.w = base, w

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, node, ctx):
        import numpy as np
        import search_g1 as SG
        pr = self.base(node, ctx)
        legal = pr.legal_add
        n_legal = int(legal.sum())
        if not n_legal:
            return pr
        pa = pr.p_swap_add
        cand = np.lexsort((np.arange(len(pa)), -np.where(legal, pa, -np.inf)))[:min(TOP, n_legal)]
        ev = SG.evidence(node, ctx)[0]
        pick = cand[np.lexsort((cand, -ev[cand]))[:PICK]]
        pe = np.zeros(len(pa))
        pe[pick] = 1.0 / len(pick)
        mix = (1.0 - self.w) * pa + self.w * pe
        share = pr.p_add.sum()
        if share > 0:
            pr.p_add = share * mix
        pr.p_swap_add = mix
        return pr


def job(args):
    """k1_check.job with the screened prior."""
    rec, net, seed, budget, cap = args
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
        prior = ScreenedPrior(C._prior(net))
        res = SG.search_system(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed,
                               probes=[eq["keys"] for eq in rec["structure"]])
        out["equations"] = [{"eq": e["eq"], "first_exact": e["effort"]["first_exact"], "structures": e["effort"]["structures"],
                             "seconds": e["effort"]["seconds"], "thorough": e["effort"]["thorough"],
                             "time_capped": e["effort"]["time_capped"], "n_exact": e["n_exact"], "solution": e.get("solution"),
                             "prior_seconds": e["effort"]["prior_seconds"], "fit_errors": e["effort"].get("fit_errors", 0)}
                            for e in res["equations"]]
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    return out


def run(which, workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    K.OUT.mkdir(parents=True, exist_ok=True)
    dom = K.DOMS[which]
    nets = NETS[which][:1] if SMOKE else NETS[which]
    recs = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
    budget = (20, 4) if SMOKE else (300, 60)
    C.log(f"build-and-check (ordering E) on {which}: {len(nets)} networks; seed {K.SEED}; budget {budget}; workers {workers}")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        queued = []
        for arm, run_, lin in nets:
            path = K.OUT / f"{lin}_E_s{K.SEED}_{dom}.json"
            if not path.exists():
                net = str(FOLDER[lin] / "nets" / f"{lin}_{dom}_d{K.K}")
                queued.append(((arm, run_), path, [pool.submit(job, (r, net, K.SEED, *budget)) for r in recs]))
        t0 = time.time()
        for (arm, run_), path, futs in queued:
            res = [f.result() for f in futs]
            path.write_text(json.dumps(res, default=float))
            eqs = [e for r in res if r["status"] == "done" for e in r["equations"]]
            msg = (f"{which} {arm} {run_} ordering E: done at {time.time() - t0:.0f}s; failed {sum(r['status'] != 'done' for r in res)}; "
                   f"time-capped {sum(e['time_capped'] for e in eqs)}")
            if not SMOKE:
                msg += f"; equations solved {sum(e['first_exact'] is not None for e in eqs)} of {len(eqs)}"
            C.log(msg)


def report(which):
    """E against A on the same networks, and the pass rule."""
    import numpy as np
    import k1_report as KR
    dom = K.DOMS[which]
    recs = C.records("dev", dom)
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())[f"mce_dev_{dom}"]
    lines, d_new, sh_a, sh_e = [f"== Build-and-check (E) against the network alone (A), {which}, seed {K.SEED}"], [], 0, 0
    for arm, run_, lin in NETS[which]:
        ea, ta = KR.outcome(json.loads((K.OUT / f"{lin}_A_s{K.SEED}_{dom}.json").read_text()), recs, shared)
        ee, te = KR.outcome(json.loads((K.OUT / f"{lin}_E_s{K.SEED}_{dom}.json").read_text()), recs, shared)
        keys = [k for k in ea if k in ee and not shared[k[0]][k[1]]]
        only_e = sum(ee[k][0] and not ea[k][0] for k in keys)
        only_a = sum(ea[k][0] and not ee[k][0] for k in keys)
        d_new.append(te["new"] - ta["new"])
        sh_a, sh_e = sh_a + ta["shared"], sh_e + te["shared"]
        lines.append(f"   {arm:>9} {run_:>9}: new-form A {ta['new']}, E {te['new']} ({te['new'] - ta['new']:+d}; only E {only_e}, only A "
                     f"{only_a}, p = {KR.binom_p(only_e, only_a):.2g}); shared-form A {ta['shared']}, E {te['shared']}; true structure A "
                     f"{ta['true']}, E {te['true']}; time-capped E {te['time_capped']}; median seconds per system A "
                     f"{np.median(ta['seconds']):.0f}, E {np.median(te['seconds']):.0f}")
    wins, med = sum(x > 0 for x in d_new), float(np.median(d_new))
    ok = wins >= 5 and med > 1.5 and sh_e >= sh_a
    lines.append(f"   E better in {wins} of {len(d_new)} networks; median gain {med:+.1f}; shared-form total A {sh_a}, E {sh_e} -> "
                 + ("PASSES" if ok else "does not pass"))
    text = "\n".join(lines)
    (C.RUN / f"report_E_{which}.txt").write_text(text)
    print(text)
    return ok


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    if step == "report":
        report(sys.argv[3])
        return
    run(step, workers)


if __name__ == "__main__":
    main()
