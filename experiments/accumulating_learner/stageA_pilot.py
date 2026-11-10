"""Stage A of the redesign: the learned grammar's development pilot (owner's approval, 9 November; research log "Stage A:
the learned grammar, and its pilot's design"). Development systems only; nothing is run before the owner approves.

The slowdown pilot's f6 recipe (g2h_pilot.py: the 2,048-term shortlist in training and search, balanced review rounds,
the first wakes on training tasks 1-10 and 11-30, development tests after each review round) with one change: every
lineage has a learned grammar (grammar_g2) as its search's base prior and its network's training base
(movenet_g2a). Testbeds as in the slowdown pilot: nonlinear flows after order 2's history (ecology, circuits,
mechanics) and reaction networks after order 1's history (mechanics, circuits, ecology). Arms:
  a_o<o>     the continued network: grammar from its history's verified solutions (earlier) and its new-domain ones;
  ac_<o>     the rehearsal-matched control: grammar from its made-up systems' structures (earlier) and its own
             new-domain solutions; its review rounds replay its made-up examples as many per dimension as a_o<o>'s;
  afresh_<o> the fresh network: grammar from its own new-domain solutions only (empty, so uniform, at its first wake,
             which is the claim run's uniform first wake, reused).
Baselines without the grammar: the slowdown pilot's f6 lineages (p6_o2, p6fresh_2 on flows; q6_o1, q6fresh_1 on
reaction networks). Usage: stageA_pilot.py <workers> <step ...> (flows, reactions, report)"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageA")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402
import g2h_pilot as P                                            # noqa: E402

PILOT = C.ROOT / "outputs/accumulating_learner/g2h/pilot"
SETUPS = P.SETUPS
POINTS = (10, 30) if not C.SMOKE else (2, 4)


# ---------------------------------------------------------------------------------------------------------------------
# worker jobs: the search with the grammar prior, training with the grammar
# ---------------------------------------------------------------------------------------------------------------------

_NET = {}


def _prior_g(path):
    """cycles_g2h._prior with the network's grammar (the file saved next to it) as the base prior."""
    import search_g1 as SG
    if path == "uniform":
        return SG.uniform_prior
    import numpy as np
    import grammar_g2 as GR
    import movenet_g2 as MN
    import movenet_g2a as MA
    if path not in _NET:
        net = MN.load_net(path + ".pt")
        emb = {d: np.load(f"{path}_d{d}.npy") for d in C.DIMS if Path(f"{path}_d{d}.npy").exists()}
        _NET[path] = (net.eval(), emb, *GR.load(path))
    net, emb, grammar, trained = _NET[path]
    return MA.GrammarPrior(net, emb, shortlist=os.environ.get("G2H_SHORTLIST"), grammar=grammar, trained=trained)


def search_job_g(args):
    C._prior = _prior_g                                          # this worker process only
    return C.search_job(args)


def train_job_g(args):
    """cycles_g2h.train_job with the grammar as the training base (movenet_g2a.train)."""
    sources, prev, out, gi, seed, steps, balanced, grammar_json = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import grammar_g2 as GR
    import movenet_g2 as MN
    import movenet_g2a as MA
    import search_g1 as SG
    MN.SHORTLIST = int(os.environ["G2H_SHORTLIST"])
    torch.manual_seed(1)
    net = MN.load_net(prev + ".pt") if prev is not None else MN.MoveNetL2()
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(8)
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in C.DIMS}
    grammar = GR.Grammar.from_json(grammar_json)
    examples, groups = C._load_examples(sources)
    tl = MA.train(net, examples, tables, steps, batch=C.BATCH, lr=C.LR, device=device, seed=seed,
                  groups=groups if balanced else None, grammar_tables={d: grammar.log_ratio(d) for d in C.DIMS})
    counts = {}
    for e, g in zip(examples, groups):
        c = counts.setdefault(str(int(g)), {})
        c[str(e["d"])] = c.get(str(e["d"]), 0) + 1
    for d in C.DIMS:
        np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    return tl, counts


# ---------------------------------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------------------------------

def madeup_structures(files):
    """The made-up systems' equation structures [(d, own, keys)] from a control's made-up example files: every example
    holds its equation's true term set (the state's terms not to be removed, plus the terms to add); consecutive
    examples with the same dimension, equation and term set are one equation."""
    import search_g1 as SG
    out, prev = [], None
    for f in files:
        for e in pickle.loads(Path(f).read_bytes()):
            d, eq = int(e["d"]), int(e["eq"])
            T = tuple(sorted(int(g) for g in e["gids"][(e["inS"] & ~e["rem"]) | e["add"]]))
            if (d, eq, T) != prev:
                order = SG.vocabulary_order(d)
                out.append((d, eq, [order[g] for g in T]))
            prev = (d, eq, T)
    return out


class PilotA(P.Pilot):
    def search(self, path, recs, prior, label, budget, cap):
        """cycles_g2h.Driver.search with the grammar prior."""
        if path.exists():
            return json.loads(path.read_text())
        C.log(f"{label}: searching {len(recs)} systems with prior {Path(prior).name} (budget {budget}, thorough cap {cap})")
        t0 = time.time()
        seed = getattr(self, "search_seed", 1)                  # reruns confirming borderline p use another seed
        res = list(self.pool.map(search_job_g, [(r, prior, seed, budget, cap) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        truth = {r["id"]: r["structure"] for r in recs}
        eqs = [(r, e) for r in res if r["status"] == "done" for e in r["equations"]]
        solved = sum(e["first_exact"] is not None for _, e in eqs)
        systems = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
        pr = [r["probe"] for r in res if r.get("probe")]
        cut = f"; shortlist cut a missing true term at {sum(p['cut'] for p in pr)} of {sum(p['with_missing_true'] for p in pr)} " \
              f"expansions" if pr else ""
        secs = sorted(r["seconds"] for r in res)
        C.log(f"{label}: done in {time.time() - t0:.0f}s; equations solved {solved} of {sum(len(truth[r['id']]) for r in res)}; "
              f"systems {systems} of {len(res)}; time-capped {sum(e['time_capped'] for _, e in eqs)}; failed "
              f"{sum(r['status'] != 'done' for r in res)}; seconds per system median {secs[len(secs) // 2]:.0f}, max "
              f"{secs[-1]:.0f}{cut}")
        return res

    def grammar_for(self, lineage, dom, o):
        """The lineage's grammar now: earlier = its history's verified solutions (each with its task's domain) or, for
        the control, its made-up systems' structures; newest = its verified solutions in the new domain."""
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
        if lineage.startswith("ac_"):
            p = C.RUN / f"madeup_structures_c{o}.json"
            if not p.exists():
                p.write_text(json.dumps(madeup_structures(C.cbase_files(o))))
            earlier, groups = [tuple(x) for x in json.loads(p.read_text())], None
        return GR.Grammar.fit(earlier, newest, earlier_groups=groups or None)

    def train(self, out, sources, prev, steps, label, info, balanced, opts=None):
        """Pilot.train with the grammar (opts["grammar"]: the grammar to train with)."""
        gi, gdesc = self.gpu()
        t0 = time.time()
        grammar = opts["grammar"]
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
            tl, counts = one.submit(train_job_g, (sources, prev, str(out), gi, int(C.h(label), 16) % (1 << 31), steps,
                                                  balanced, grammar.to_json())).result()
        alphas = [c.alpha for c in grammar.components]
        rec = dict(info, device=gdesc, seconds=time.time() - t0, examples_by_group_d=counts, steps=steps, balanced=balanced,
                   grammar_alphas=alphas, log=tl)
        (C.RUN / "dream" / f"{label}_train.json").write_text(json.dumps(rec))
        n = sum(sum(c.values()) for c in counts.values())
        C.log(f"dream {label}: trained {steps} steps on {n} examples ({counts}){' balanced' if balanced else ''} with the "
              f"grammar (smoothing {alphas}) on {gdesc} in {time.time() - t0:.0f}s; last loss {tl[-1]['loss']:.3f}; "
              f"skipped {tl[-1]['skipped']}")
        return rec

    def new_dream(self, lineage, tag, prev, wake, dom, mode, steps, opts=None, earlier_files=None, mirror=None, o=None):
        """Pilot.new_dream with the lineage's grammar fitted after the wake, trained with and saved next to the network."""
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
        self.train(out, src, None if prev == "uniform" else prev, steps, f"{lineage}_{tag}", {"mode": mode},
                   mode == "balanced", {"grammar": grammar})
        GR.save(out, grammar, trained=True)
        return str(out)

    def curve_a(self, lineage, start, dom, mode, o, reuse_w10=None, earlier_files=None, mirror_prefix=None):
        """Pilot.curve with the grammar."""
        tasks = C.records("train", dom)
        net, prev = start, 0
        for k in POINTS:
            w = self.wake(lineage, f"{lineage}_w{k}", tasks[prev:k], net, reuse=reuse_w10 if k == POINTS[0] else None)
            mir = f"{mirror_prefix}_{dom}_d{k}" if mirror_prefix else None
            net = self.new_dream(lineage, f"{dom}_d{k}", net, w, dom, mode, C.NEW["steps"], None, earlier_files, mir, o)
            self.test(net, dom, f"{lineage}_t{k}dev", *C.NEWB, which="dev")
            prev = k
        return net

    def test(self, net, dom, label, budget, cap, which="test"):
        if not C.SMOKE:
            return super().test(net, dom, label, budget, cap, which)
        return self.search(C.RUN / "test" / f"{label}_{dom}.json", C.records(which, dom)[:3], net, f"test {label} on {dom}",
                           budget, cap)                          # smoke tests: three development systems

    def start(self, lineage, name, dom, o):
        """A starting network (copied from claim run 1, as in the slowdown pilot) with the lineage's grammar before the
        new domain, saved as not trained with it."""
        import grammar_g2 as GR
        net = P.copy_net(P.RUN1 / "nets" / name, C.RUN / "nets" / name)
        if not Path(f"{net}_grammar.json").exists():
            GR.save(net, self.grammar_for(lineage, dom, o), trained=False)
        return net


def run(D, which):
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    cont, ctrl, fresh = f"a_o{o}", f"ac_{o}", f"afresh_{o}"
    hist = D.start(cont, setup["hist"], dom, o)
    cbase = D.start(ctrl, setup["cbase"], dom, o)
    D.curve_a(cont, hist, dom, "balanced", o)
    D.curve_a(ctrl, cbase, dom, "balanced", o, earlier_files=C.cbase_files(o), mirror_prefix=cont)
    D.curve_a(fresh, "uniform", dom, "fresh", o, reuse_w10=None if C.SMOKE else P.RUN1 / "wake" / f"fresh_{dom}_w10.json")


def bits_table(which):
    """The diagnostic that needs no search: the mean description length in bits of the development systems' true
    new-form equations under each arm's grammar before the new domain and after each review round (the uniform
    prior's for comparison). Development systems only; counts and means only."""
    import grammar_g2 as GR
    import numpy as np
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    flags = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())[f"mce_dev_{dom}"]
    eqs = [(r["desc"]["d"], i, eq["keys"]) for r in C.records("dev", dom) for i, eq in enumerate(r["structure"])
           if not flags[r["id"]][i]]
    uni = GR.Grammar([])
    rows = [("uniform prior", np.mean([uni.bits(d, i, k) for d, i, k in eqs]))]
    for lab, name in ((f"continued, before", setup["hist"]), ("control, before", setup["cbase"])):
        g, _ = GR.load(C.RUN / "nets" / name)
        if g is not None:
            rows.append((lab, np.mean([g.bits(d, i, k) for d, i, k in eqs])))
    for arm, lin in (("continued", f"a_o{o}"), ("control", f"ac_{o}"), ("fresh", f"afresh_{o}")):
        for k in POINTS:
            g, _ = GR.load(C.RUN / "nets" / f"{lin}_{dom}_d{k}")
            if g is not None:
                rows.append((f"{arm}, after {k} tasks", np.mean([g.bits(d, i, k_) for d, i, k_ in eqs])))
    return len(eqs), rows


def report():
    import g2h_report as R
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    for which, setup in SETUPS.items():
        dom, o = setup["dom"], setup["o"]
        flags = shared[f"mce_dev_{dom}"]
        new = [(r["id"], i) for r in C.records("dev", dom) for i in range(len(r["structure"])) if not flags[r["id"]][i]]

        def ld(folder, lab):
            q = folder / "test" / f"{lab}_{dom}.json"
            return json.loads(q.read_text()) if q.exists() else None
        pre = "p" if which == "flows" else "q"
        arms = {"continued": (f"a_o{o}", f"{pre}6_o{o}"), "control": (f"ac_{o}", None), "fresh": (f"afresh_{o}", f"{pre}6fresh_{o}")}
        print(f"== {which} ({dom} after order {o}'s history; {len(new)} new-form development equations)")
        res = {}
        for arm, (lin, base) in arms.items():
            for k in POINTS:
                a = ld(C.RUN, f"{lin}_t{k}dev")
                b = ld(PILOT, f"{base}_t{k}dev") if base else None
                res[(arm, k)] = a
                if a is None:
                    continue
                line = f"   {arm:>9} at {k:>2} tasks: with the grammar {R.counts(a, new)}"
                if b is not None:
                    x, y = R.paired(a, b, new)
                    line += f", without {R.counts(b, new)} (only with {x}, only without {y}, p = {R.sign_p(x, y):.2g})"
                capped = sum(e["time_capped"] for r in a if r["status"] == "done" for e in r["equations"])
                pr = [r["probe"] for r in a if r.get("probe")]
                root = sum(p["root_cut"] for p in pr), sum(p["root_with_missing_true"] for p in pr)
                line += f"; time-capped {capped}; root expansions with a missing true term outside the attended set {root[0]} of {root[1]}"
                print(line)
        for k in POINTS:
            c, k_, f = res.get(("continued", k)), res.get(("control", k)), res.get(("fresh", k))
            if c is not None and k_ is not None:
                met, pts, a_, b_ = R.bar_b(c, k_, new)
                print(f"   continued vs control at {k} tasks (with the grammar): {pts:+.1f} points, only continued {a_}, only "
                      f"control {b_}, p = {R.sign_p(a_, b_):.2g}" + (" -> bar (B) level on development" if met and k == 10 else ""))
            if c is not None and f is not None:
                x, y = R.paired(c, f, new)
                print(f"   continued vs fresh at {k} tasks (with the grammar): only continued {x}, only fresh {y}, p = {R.sign_p(x, y):.2g}")
        n, rows = bits_table(which)
        print(f"   description length of the {n} true new-form equations (mean bits): " +
              "; ".join(f"{lab} {v:.1f}" for lab, v in rows))


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    if steps_ == ["report"]:
        report()
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    D = PilotA(workers, 1)
    C.log(f"Stage A pilot started: steps {steps_}; shortlist {os.environ['G2H_SHORTLIST']}")
    for step in steps_:
        if step in SETUPS:
            run(D, step)
        elif step == "report":
            report()
    C.log("Stage A pilot steps done")


if __name__ == "__main__":
    main()
