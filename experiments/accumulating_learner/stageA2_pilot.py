"""Stage A' (proposed after the Stage A pilot's stop rule, 10 November; research log "Stage A pilot: results, diagnosis and
a proposed fix"): the grammar only shares out the probability of the terms outside the network's shortlist. The
network's scores, its shortlist and its training are as before Stage A (the slowdown pilot's f6 recipe), so the grammar
can no longer push a term out of the attended set or lower an attended term's score; it decides only which of the
unattended terms get the rest slot's probability (movenet_g2a.GrammarPrior with trained=False). Grammars are fitted as
in Stage A and saved next to every network for the search, for Stage B's dreams and for the bits check.
Arms a2_o<o> (continued) and a2fresh_<o> (fresh), or the control ac2_<o> with steps "<testbed>-control". Development
systems only; nothing is run before the owner approves. Usage: stageA2_pilot.py <workers> <step ...> (flows, reactions,
flows-control, reactions-control, report)"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageA2")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402
import g2h_pilot as P                                            # noqa: E402
import stageA_pilot as A                                         # noqa: E402

STAGEA = C.ROOT / "outputs/accumulating_learner/g2h/stageA"
SETUPS = P.SETUPS
POINTS = A.POINTS


class PilotA2(A.PilotA):
    def grammar_for(self, lineage, dom, o):
        """PilotA.grammar_for for Stage A' names: the control (ac2_<o>) has its made-up systems as earlier experience."""
        if not lineage.startswith("ac2_"):
            return A.PilotA.grammar_for(self, lineage, dom, o)
        import grammar_g2 as GR
        newest = []
        for k, keys in self.replays(lineage).items():
            tid, eq = k.rsplit("|", 1)
            if C.dom_of(tid) == dom:
                newest.append((self.recs[tid]["desc"]["d"], int(eq), keys))
        p = C.RUN / f"madeup_structures_c{o}.json"
        if not p.exists():
            p.write_text(json.dumps(A.madeup_structures(C.cbase_files(o))))
        return GR.Grammar.fit([tuple(x) for x in json.loads(p.read_text())], newest, earlier_groups=None)

    def train(self, out, sources, prev, steps, label, info, balanced, opts=None):
        """The f6 recipe's training (no grammar): g2h_pilot.Pilot.train with the 2,048-term shortlist."""
        return P.Pilot.train(self, out, sources, prev, steps, label, info, balanced, {"shortlist": int(os.environ["G2H_SHORTLIST"])})

    def new_dream(self, lineage, tag, prev, wake, dom, mode, steps, opts=None, earlier_files=None, mirror=None, o=None):
        """PilotA.new_dream with training as before Stage A; the grammar fitted after the wake is saved next to the
        network as not trained with it, so the search uses it only for the rest slot."""
        import grammar_g2 as GR
        out = C.RUN / "nets" / f"{lineage}_{tag}"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        rep = self.add_replays(lineage, wake)
        grammar = self.grammar_for(lineage, dom, o)
        src = self.replay_sources(rep, dom)
        src.append((self.fantasies(f"g2h-fantasy-{lineage}-{tag}", {4: C.NEW["fantasies"]},
                                   self.usage({k: v for k, v in rep.items() if C.dom_of(k.rsplit("|", 1)[0]) == dom})), 1, None))
        if earlier_files is not None and mode == "balanced":    # the control in the balanced schedule
            src += self.rehearsal(f"{lineage}_{tag}", earlier_files, self.counts_of(mirror).get(0, {}))
        self.train(out, src, None if prev == "uniform" else prev, steps, f"{lineage}_{tag}", {"mode": mode}, mode == "balanced")
        GR.save(out, grammar, trained=False)
        return str(out)


def run(D, step):
    which, control = step.split("-")[0], step.endswith("-control")
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    cont = f"a2_o{o}"
    if control:
        cbase = D.start(f"ac2_{o}", setup["cbase"], dom, o)
        D.curve_a(f"ac2_{o}", cbase, dom, "balanced", o, reuse_w10=None if C.SMOKE else STAGEA / "wake" / f"ac_{o}_w10.json",
                  earlier_files=C.cbase_files(o), mirror_prefix=cont)
        return
    hist = D.start(cont, setup["hist"], dom, o)
    D.curve_a(cont, hist, dom, "balanced", o, reuse_w10=None if C.SMOKE else STAGEA / "wake" / f"a_o{o}_w10.json")
    D.curve_a(f"a2fresh_{o}", "uniform", dom, "fresh", o,
              reuse_w10=None if C.SMOKE else P.RUN1 / "wake" / f"fresh_{dom}_w10.json")


def report():
    import numpy as np
    import grammar_g2 as GR
    import g2h_report as R
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    PILOT = A.PILOT
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
        arms = {"continued": (f"a2_o{o}", f"{pre}6_o{o}", f"a_o{o}"), "control": (f"ac2_{o}", None, f"ac_{o}"),
                "fresh": (f"a2fresh_{o}", f"{pre}6fresh_{o}", f"afresh_{o}")}
        print(f"== {which} ({dom} after order {o}'s history; {len(new)} new-form development equations)")
        res = {}
        for arm, (lin, base, stage_a) in arms.items():
            for k in POINTS:
                x_ = ld(C.RUN, f"{lin}_t{k}dev")
                res[(arm, k)] = x_
                if x_ is None:
                    continue
                parts = [f"   {arm:>9} at {k:>2} tasks: Stage A' {R.counts(x_, new)}"]
                for tag, folder, lab in (("without the grammar", PILOT, base), ("Stage A", STAGEA, stage_a)):
                    y_ = ld(folder, f"{lab}_t{k}dev") if lab else None
                    if y_ is not None:
                        a_, b_ = R.paired(x_, y_, new)
                        parts.append(f"{tag} {R.counts(y_, new)} (only Stage A' {a_}, only {tag} {b_}, p = {R.sign_p(a_, b_):.2g})")
                pr = [r["probe"] for r in x_ if r.get("probe")]
                cut = sum(p["cut"] for p in pr) / max(1, sum(p["with_missing_true"] for p in pr))
                capped = sum(e["time_capped"] for r in x_ if r["status"] == "done" for e in r["equations"])
                print("; ".join(parts) + f"; cut rate {100 * cut:.1f}%; time-capped {capped}")
        for k in POINTS:
            c, k_, f = res.get(("continued", k)), res.get(("control", k)), res.get(("fresh", k))
            if c is not None and k_ is not None:
                met, pts, a_, b_ = R.bar_b(c, k_, new)
                print(f"   continued vs control at {k} tasks: {pts:+.1f} points, only continued {a_}, only control {b_}, p = {R.sign_p(a_, b_):.2g}")
            if c is not None and f is not None:
                a_, b_ = R.paired(c, f, new)
                print(f"   continued vs fresh at {k} tasks: only continued {a_}, only fresh {b_}, p = {R.sign_p(a_, b_):.2g}")
        rows = [("uniform prior", np.mean([GR.Grammar([]).bits(d, i, k) for d, i, k in eqs]))]
        for arm, lin in (("continued", f"a2_o{o}"), ("control", f"ac2_{o}"), ("fresh", f"a2fresh_{o}")):
            for k in POINTS:
                g, _ = GR.load(C.RUN / "nets" / f"{lin}_{dom}_d{k}")
                if g is not None:
                    rows.append((f"{arm}, after {k} tasks", np.mean([g.bits(d, i, k_) for d, i, k_ in eqs])))
        print(f"   description length of the {len(eqs)} true new-form equations (mean bits): " +
              "; ".join(f"{lab} {v:.1f}" for lab, v in rows))


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    if steps_ == ["report"]:
        report()
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    D = PilotA2(workers, 1)
    C.log(f"Stage A' pilot started: steps {steps_}; shortlist {os.environ['G2H_SHORTLIST']}")
    for step in steps_:
        if step.split("-")[0] in SETUPS:
            run(D, step)
        elif step == "report":
            report()
    C.log("Stage A' pilot steps done")


if __name__ == "__main__":
    main()
