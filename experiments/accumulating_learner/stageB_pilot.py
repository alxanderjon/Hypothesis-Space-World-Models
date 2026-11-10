"""Stage B of the redesign: dreams from each lineage's grammar, on top of Stage A' (owner, 10 November: the A' pilot is
skipped), the development pilot (research log "Stage B on A': the pilot's design"). Development systems only; nothing
is run before the owner approves.

The slowdown pilot's f6 recipe (the 2,048-term shortlist in training and search, balanced review rounds, wakes on
training tasks 1-10 and 11-30, development tests after each review round) with Stage A' and Stage B together:
  - A' (stageA2_pilot.py): each lineage's grammar (grammar_g2) is fitted after every wake and used in the search only
    to share out the rest slot's probability over the terms outside the network's shortlist; the network's scores,
    shortlist and training are the f6 recipe's;
  - B (dream_g2b.py): every review round's 100 fantasies are drawn from that grammar (each fantasy whole from the
    grammar, or with probability 1/2 from the open language), instead of half from single-term usage and half from
    the size prior.
Arms b<tag>_o<o> (continued), bc<tag>_<o> (rehearsal-matched control: its made-up systems' structures as its earlier
experience), bfresh<tag>_<o> (fresh); tag "" for the pilot, "r2" for a rerun with other seeds (owner, 10 November: any
decision with 0.01 <= p <= 0.05 is confirmed by a rerun before it counts; the rerun's fantasies and training differ
through its lineage names, its searches use seed 2, and its first wakes are searched again). The pilot's first wakes
on tasks 1-10 are Stage A's (the same starting network, the grammar only in the rest slot, the same tasks, budget and
seed), reused. Baselines: the slowdown pilot's f6 lineages (no grammar); Stage A's lineages for context.
Usage: stageB_pilot.py <workers> <step ...> (flows, reactions, flows:r2[:continued,control,fresh], ..., report)"""

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
import stageA2_pilot as A2                                       # noqa: E402

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


class PilotB(A2.PilotA2):
    def grammar_for(self, lineage, dom, o):
        """The lineage's grammar (PilotA.grammar_for) for Stage B's names: the control (bc<tag>_<o>) has its made-up
        systems' structures as its earlier experience."""
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
        if lineage.startswith("bc"):
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
            s = stats.setdefault(r["branch"], {"fantasies": 0, "accepted": 0, "rejected draws": 0, "terms": [], "examples": 0})
            s["fantasies"] += 1
            s["accepted"] += r["accepted"]
            s["rejected draws"] += len(r["reasons"])
            s["terms"] += r.get("terms", [])
            s["examples"] += len(r["examples"])
        Path(f"{p}.stats.json").write_text(json.dumps(stats))
        C.log(f"fantasies {prefix} (from the grammar, open-language share {DB.SHARE}): " + "; ".join(
            f"{b}: accepted {s['accepted']} of {s['fantasies']} ({s['rejected draws']} rejected draws), examples "
            f"{s['examples']}, mean terms per equation {np.mean(s['terms']) if s['terms'] else float('nan'):.2f}"
            for b, s in sorted(stats.items())) + f"; {time.time() - t0:.0f}s")
        return str(p)

    def new_dream(self, lineage, tag, prev, wake, dom, mode, steps, opts=None, earlier_files=None, mirror=None, o=None):
        """PilotA2.new_dream (training as before Stage A; the grammar saved for the rest slot only) with the review
        round's fantasies drawn from the grammar fitted after the wake."""
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
        self.train(out, src, None if prev == "uniform" else prev, steps, f"{lineage}_{tag}", {"mode": mode}, mode == "balanced")
        GR.save(out, grammar, trained=False)
        return str(out)


def run(D, step):
    """step: <testbed> (the pilot) or <testbed>:r2[:arm,...] (a rerun with other seeds of the arms listed, default all)."""
    parts = step.split(":")
    which, tag = parts[0], parts[1] if len(parts) > 1 else ""
    arms = parts[2].split(",") if len(parts) > 2 else ["continued", "control", "fresh"]
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    D.search_seed = 2 if tag else 1
    cont, ctrl, fresh = f"b{tag}_o{o}", f"bc{tag}_{o}", f"bfresh{tag}_{o}"
    first = (lambda lab: None) if (C.SMOKE or tag) else (lambda lab: STAGEA / "wake" / f"{lab}.json")
    if "continued" in arms or "control" in arms:                 # the control's rehearsal mirrors this continued run
        hist = D.start(cont, setup["hist"], dom, o)
        D.curve_a(cont, hist, dom, "balanced", o, reuse_w10=first(f"a_o{o}_w10"))
    if "control" in arms:
        cbase = D.start(ctrl, setup["cbase"], dom, o)
        D.curve_a(ctrl, cbase, dom, "balanced", o, reuse_w10=first(f"ac_{o}_w10"), earlier_files=C.cbase_files(o),
                  mirror_prefix=cont)
    if "fresh" in arms:
        reuse = None if (C.SMOKE or tag) else P.RUN1 / "wake" / f"fresh_{dom}_w10.json"
        D.curve_a(fresh, "uniform", dom, "fresh", o, reuse_w10=reuse)


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
        pre = "p" if which == "flows" else "q"
        print(f"== {which} ({dom} after order {o}'s history; {len(new)} new-form development equations)")
        for tag in ("", "r2"):
            arms = {"continued": (f"b{tag}_o{o}", f"{pre}6_o{o}", f"a_o{o}"), "control": (f"bc{tag}_{o}", None, f"ac_{o}"),
                    "fresh": (f"bfresh{tag}_{o}", f"{pre}6fresh_{o}", f"afresh_{o}")}
            res = {}
            for arm, (lin, base, stage_a) in arms.items():
                for k in POINTS:
                    x_ = ld(C.RUN, f"{lin}_t{k}dev")
                    res[(arm, k)] = x_
                    if x_ is None:
                        continue
                    line = [f"   {arm:>9}{' (rerun)' if tag else ''} at {k:>2} tasks: A'+B {R.counts(x_, new)}"]
                    for lab_, folder, b_lab in (("without the grammar", A.PILOT, base), ("Stage A", STAGEA, stage_a)):
                        y_ = ld(folder, f"{b_lab}_t{k}dev") if b_lab else None
                        if y_ is not None:
                            a_, b_ = R.paired(x_, y_, new)
                            line.append(f"{lab_} {R.counts(y_, new)} (only A'+B {a_}, only {lab_} {b_}, p = {R.sign_p(a_, b_):.2g})")
                    pr = [r["probe"] for r in x_ if r.get("probe")]
                    cut = sum(p_["cut"] for p_ in pr) / max(1, sum(p_["with_missing_true"] for p_ in pr))
                    capped = sum(e["time_capped"] for r in x_ if r["status"] == "done" for e in r["equations"])
                    print("; ".join(line) + f"; cut rate {100 * cut:.1f}%; time-capped {capped}")
            for k in POINTS:
                c, k_, f = res.get(("continued", k)), res.get(("control", k)), res.get(("fresh", k))
                if c is not None and k_ is not None:
                    met, pts, a_, b_ = R.bar_b(c, k_, new)
                    print(f"   continued vs control at {k} tasks{' (rerun)' if tag else ''}: {pts:+.1f} points, only continued {a_}, "
                          f"only control {b_}, p = {R.sign_p(a_, b_):.2g}")
                if c is not None and f is not None:
                    a_, b_ = R.paired(c, f, new)
                    print(f"   continued vs fresh at {k} tasks{' (rerun)' if tag else ''}: only continued {a_}, only fresh {b_}, "
                          f"p = {R.sign_p(a_, b_):.2g}")
            rows = [("uniform prior", np.mean([GR.Grammar([]).bits(d, i, k) for d, i, k in eqs]))] if not tag else []
            for arm, lin in (("continued", f"b{tag}_o{o}"), ("control", f"bc{tag}_{o}"), ("fresh", f"bfresh{tag}_{o}")):
                for k in POINTS:
                    g, _ = GR.load(C.RUN / "nets" / f"{lin}_{dom}_d{k}")
                    if g is not None:
                        rows.append((f"{arm}, after {k} tasks", np.mean([g.bits(d, i, k_) for d, i, k_ in eqs])))
            if rows:
                print(f"   bits check{' (rerun)' if tag else ''}: description length of the {len(eqs)} true new-form equations "
                      f"(mean bits): " + "; ".join(f"{lab} {v:.1f}" for lab, v in rows))
            for lin in (f"b{tag}_o{o}", f"bc{tag}_{o}", f"bfresh{tag}_{o}"):
                for k in POINTS:
                    q = C.RUN / "examples/fantasy" / f"g2h-fantasy-{lin}-{dom}_d{k}.pkl.stats.json"
                    if q.exists():
                        st = json.loads(q.read_text())
                        print(f"   dreams {lin} after {k} tasks: " + "; ".join(
                            f"{b} accepted {s['accepted']} of {s['fantasies']}, examples {s.get('examples', 'n/a')}, mean terms "
                            f"{np.mean(s['terms']) if s['terms'] else float('nan'):.2f}" for b, s in sorted(st.items())))


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    if steps_ == ["report"]:
        report()
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    D = PilotB(workers, 1)
    C.log(f"Stage B (on A') pilot started: steps {steps_}; shortlist {os.environ['G2H_SHORTLIST']}")
    for step in steps_:
        if step.split(":")[0] in SETUPS:
            run(D, step)
        elif step == "report":
            report()
    C.log("Stage B (on A') pilot steps done")


if __name__ == "__main__":
    main()
