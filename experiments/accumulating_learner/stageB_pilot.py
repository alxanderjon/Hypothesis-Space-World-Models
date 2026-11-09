"""Stage B of the redesign: dreams from each lineage's grammar, the development pilot (research log "Stage B: dreams from
each lineage's grammar, and its pilot's design"). Development systems only; nothing is run before the owner approves.

Stage A's pilot (stageA_pilot.py) with one change: every review round's 100 fantasies are drawn from the lineage's
grammar at that point (dream_g2b: each fantasy whole from the grammar, or with probability 1/2 from the open language
as before), instead of half from single-term usage and half from the size prior. The grammar is the one the review
round trains with (fitted to the lineage's verified solutions after the wake; for the control, its made-up systems'
structures as its earlier experience). Arms b_o<o> (continued), bc_<o> (rehearsal-matched control), bfresh_<o>
(fresh). The first wakes on tasks 1-10 are Stage A's (identical network, grammar, tasks, budget and seed), reused.
Baselines: Stage A's lineages (a_o<o>, ac_<o>, afresh_<o>). Usage: stageB_pilot.py <workers> <step ...> (flows,
reactions, report)"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageB")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402
import g2h_pilot as P                                            # noqa: E402
import stageA_pilot as A                                         # noqa: E402

STAGEA = C.ROOT / "outputs/accumulating_learner/g2h/stageA"
SETUPS = P.SETUPS
POINTS = A.POINTS
_SIZE = {}


def fantasy_job_b(args):
    """cycles_g2h.fantasy_job with dream_g2b's fantasies."""
    prefix, k, gp, gn, d, share = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g1 as D1
    import dream_g2 as DR
    import dream_g2b as DB
    import search_g1 as SG
    order = SG.vocabulary_order(d)
    if d not in _SIZE:                                           # the size prior, once per worker
        _SIZE[d] = D1.size_prior(order, json.loads((SG.G1_DIR / f"vocab_d{d}.json").read_text())["terms"])
    syn, reasons, branch = DB.accepted_fantasy(prefix, k, order, np.asarray(gp), np.asarray(gn), _SIZE[d], d, share)
    if syn is None:
        return {"k": k, "d": d, "examples": [], "reasons": reasons, "accepted": False, "branch": branch}
    try:
        ex = DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
    except Exception:
        ex = []
    return {"k": k, "d": d, "examples": ex, "reasons": reasons, "accepted": True, "branch": branch,
            "terms": [len(eq["keys"]) for eq in syn.structure]}


class PilotB(A.PilotA):
    def grammar_for(self, lineage, dom, o):
        """PilotA.grammar_for for Stage B's lineage names (the control: bc_<o>)."""
        import grammar_g2 as GR
        earlier, groups, newest = [], [], []
        for k, keys in self.replays(lineage).items():
            tid, eq = k.rsplit("|", 1)
            d, dm = self.recs[tid]["desc"]["d"], C.dom_of(tid)
            if dm == dom:
                newest.append((d, int(eq), keys))
            else:
                earlier.append((d, int(eq), keys))
                groups.append(dm)
        if lineage.startswith("bc_"):
            p = C.RUN / f"madeup_structures_c{o}.json"
            if not p.exists():
                p.write_text(json.dumps(A.madeup_structures(C.cbase_files(o))))
            earlier, groups = [tuple(x) for x in json.loads(p.read_text())], None
        return GR.Grammar.fit(earlier, newest, earlier_groups=groups or None)

    def fantasies_b(self, prefix, n_by_d, grammar):
        """cycles_g2h.Driver.fantasies drawn from the grammar (dream_g2b); acceptance and terms per branch logged."""
        import numpy as np
        import dream_g2b as DB
        p = C.RUN / "examples/fantasy" / f"{prefix}.pkl"
        if p.exists():
            return str(p)
        t0 = time.time()
        jobs = []
        for d, n in sorted(n_by_d.items()):
            gp = np.exp(grammar.log_ratio(d)[:, 0].astype(np.float64))
            gp /= gp.sum()
            gn = np.exp(grammar.nterms_logp(d))
            gn /= gn.sum()
            jobs += [(f"{prefix}-d{d}", k, gp.tolist(), gn.tolist(), d, DB.SHARE) for k in range(n)]
        res = list(self.pool.map(fantasy_job_b, jobs))
        ex = [e for r in res for e in r["examples"]]
        p.write_bytes(pickle.dumps(ex))
        stats = {}
        for r in res:
            s = stats.setdefault(r["branch"], {"fantasies": 0, "accepted": 0, "rejected draws": 0, "terms": []})
            s["fantasies"] += 1
            s["accepted"] += r["accepted"]
            s["rejected draws"] += len(r["reasons"])
            s["terms"] += r.get("terms", [])
        Path(f"{p}.stats.json").write_text(json.dumps(stats))
        C.log(f"fantasies {prefix} (from the grammar, open-language share {DB.SHARE}): " + "; ".join(
            f"{b}: accepted {s['accepted']} of {s['fantasies']} ({s['rejected draws']} rejected draws), mean terms per "
            f"equation {np.mean(s['terms']) if s['terms'] else float('nan'):.2f}" for b, s in sorted(stats.items()))
            + f"; examples {len(ex)}, {time.time() - t0:.0f}s")
        return str(p)

    def new_dream(self, lineage, tag, prev, wake, dom, mode, steps, opts=None, earlier_files=None, mirror=None, o=None):
        """PilotA.new_dream with the review round's fantasies drawn from the grammar it trains with."""
        import grammar_g2 as GR
        out = C.RUN / "nets" / f"{lineage}_{tag}"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        rep = self.add_replays(lineage, wake)
        grammar = self.grammar_for(lineage, dom, o)
        src = self.replay_sources(rep, dom)
        src.append((self.fantasies_b(f"g2h-fantasy-{lineage}-{tag}", {4: C.NEW["fantasies"]}, grammar), 1, None))
        if earlier_files is not None and mode == "balanced":    # the control in the balanced schedule
            src += self.rehearsal(f"{lineage}_{tag}", earlier_files, self.counts_of(mirror).get(0, {}))
        self.train(out, src, None if prev == "uniform" else prev, steps, f"{lineage}_{tag}", {"mode": mode},
                   mode == "balanced", {"grammar": grammar})
        GR.save(out, grammar, trained=True)
        return str(out)


def run(D, which):
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    cont, ctrl, fresh = f"b_o{o}", f"bc_{o}", f"bfresh_{o}"
    reuse = (lambda lab: None) if C.SMOKE else (lambda lab: STAGEA / "wake" / f"{lab}.json")
    hist = D.start(cont, setup["hist"], dom, o)
    cbase = D.start(ctrl, setup["cbase"], dom, o)
    D.curve_a(cont, hist, dom, "balanced", o, reuse_w10=reuse(f"a_o{o}_w10"))
    D.curve_a(ctrl, cbase, dom, "balanced", o, reuse_w10=reuse(f"ac_{o}_w10"), earlier_files=C.cbase_files(o),
              mirror_prefix=cont)
    D.curve_a(fresh, "uniform", dom, "fresh", o, reuse_w10=None if C.SMOKE else P.RUN1 / "wake" / f"fresh_{dom}_w10.json")


def report():
    import numpy as np
    import grammar_g2 as GR
    import g2h_report as R
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    for which, setup in SETUPS.items():
        dom, o = setup["dom"], setup["o"]
        flags = shared[f"mce_dev_{dom}"]
        dev = C.records("dev", dom)
        new = [(r["id"], i) for r in dev for i in range(len(r["structure"])) if not flags[r["id"]][i]]
        eqs = [(r["desc"]["d"], i, eq["keys"]) for r in dev for i, eq in enumerate(r["structure"]) if not flags[r["id"]][i]]

        def ld(folder, lab):
            q = folder / "test" / f"{lab}_{dom}.json"
            return json.loads(q.read_text()) if q.exists() else None
        arms = {"continued": (f"b_o{o}", f"a_o{o}"), "control": (f"bc_{o}", f"ac_{o}"), "fresh": (f"bfresh_{o}", f"afresh_{o}")}
        print(f"== {which} ({dom} after order {o}'s history; {len(new)} new-form development equations)")
        res = {}
        for arm, (lin, base) in arms.items():
            for k in POINTS:
                b, a = ld(C.RUN, f"{lin}_t{k}dev"), ld(STAGEA, f"{base}_t{k}dev")
                res[(arm, k)] = b
                if b is None:
                    continue
                line = f"   {arm:>9} at {k:>2} tasks: with grammar dreams {R.counts(b, new)}"
                if a is not None:
                    x, y = R.paired(b, a, new)
                    line += f", Stage A {R.counts(a, new)} (only with grammar dreams {x}, only Stage A {y}, p = {R.sign_p(x, y):.2g})"
                capped = sum(e["time_capped"] for r in b if r["status"] == "done" for e in r["equations"])
                print(line + f"; time-capped {capped}")
        for k in POINTS:
            c, k_, f = res.get(("continued", k)), res.get(("control", k)), res.get(("fresh", k))
            if c is not None and k_ is not None:
                met, pts, a_, b_ = R.bar_b(c, k_, new)
                print(f"   continued vs control at {k} tasks: {pts:+.1f} points, only continued {a_}, only control {b_}, "
                      f"p = {R.sign_p(a_, b_):.2g}" + (" -> bar (B) level on development" if met and k == 10 else ""))
            if c is not None and f is not None:
                x, y = R.paired(c, f, new)
                print(f"   continued vs fresh at {k} tasks: only continued {x}, only fresh {y}, p = {R.sign_p(x, y):.2g}")
        rows = [("uniform prior", np.mean([GR.Grammar([]).bits(d, i, k) for d, i, k in eqs]))]
        for arm, lin in (("continued", f"b_o{o}"), ("control", f"bc_{o}"), ("fresh", f"bfresh_{o}")):
            for k in POINTS:
                g, _ = GR.load(C.RUN / "nets" / f"{lin}_{dom}_d{k}")
                if g is not None:
                    rows.append((f"{arm}, after {k} tasks", np.mean([g.bits(d, i, k_) for d, i, k_ in eqs])))
        print(f"   description length of the {len(eqs)} true new-form equations (mean bits): " +
              "; ".join(f"{lab} {v:.1f}" for lab, v in rows))
        for lin in (f"b_o{o}", f"bc_{o}", f"bfresh_{o}"):
            for k in POINTS:
                q = C.RUN / "examples/fantasy" / f"g2h-fantasy-{lin}-{dom}_d{k}.pkl.stats.json"
                if q.exists():
                    st = json.loads(q.read_text())
                    print(f"   dreams {lin} after {k} tasks: " + "; ".join(
                        f"{b} accepted {s['accepted']} of {s['fantasies']}, mean terms {np.mean(s['terms']) if s['terms'] else float('nan'):.2f}"
                        for b, s in sorted(st.items())))


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    if steps_ == ["report"]:
        report()
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    D = PilotB(workers, 1)
    C.log(f"Stage B pilot started: steps {steps_}; shortlist {os.environ['G2H_SHORTLIST']}")
    for step in steps_:
        if step in SETUPS:
            run(D, step)
        elif step == "report":
            report()
    C.log("Stage B pilot steps done")


if __name__ == "__main__":
    main()
