"""K0b of the compositional plan (owner's approval, 11 November; research log "Owner: K0b ..."): pieces with holes learned
by compression (library_ops) from each lineage's own verified solutions, and where the development systems' true
new-form terms would rank under the operation grammar with that library, against K0's operation grammar without it, the
lineage's network at the root and the simplest-first order; with the bits check. No search, no training; development
systems only; counts only.

Lineages as in K0 (the continued network: its history's verified solutions, then its new-domain wakes; the control: its
made-up systems' structures, then its wakes; the fresh network: its wakes), now with every Stage X run of each arm (at
least 2-3 seeds per arm, owner 11 November): versions n and ns and the f6 recipe (seeds 1-3 on flows; seed 1 is the
slowdown pilot's p6/q6 baseline; no f6 control was run). Before the new domain and after 10 and 30 tasks. A lineage's
library is learned from all its solutions so far (earlier and newest pooled); both of its grammar's components use it.
Grammars (library_ops): ctx (K0's operation grammar) and ctx_lib (with the library it learns), uni and uni_lib (the
context-free form of DreamCoder's prior, without and with the library it learns). Term sets: all true terms of the new-form equations; those unfamiliar to the lineage (K0's
definition: kind of term not among its solutions so far); those new to both prior experiences (kind absent from both the
history's solutions and the made-up structures, the same set for every arm); the bump; the saturating term.
Usage: k0b_check.py <workers> [smoke] (smoke: two development systems per domain and two runs per arm, written to
k0b_smoke; prints only checks that reveal no results)"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageX")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402

X_RUN = C.ROOT / "outputs/accumulating_learner/g2h/stageX"
PILOT = C.ROOT / "outputs/accumulating_learner/g2h/pilot"
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
OUT = X_RUN / ("k0b_smoke" if SMOKE else "k0b")
SETUPS = {"flows": {"dom": "flows_L1", "o": 2, "hist": "o2_mechanics2", "cbase": "c2_mechanics2"},
          "reactions": {"dom": "reactions_L1", "o": 1, "hist": "o1_ecology2", "cbase": "c1_ecology2"}}
RUNS = {"flows": {"continued": [("n", X_RUN, "xn_o2"), ("ns", X_RUN, "xns_o2"), ("f6 seed 1", PILOT, "p6_o2"),
                                ("f6 seed 2", X_RUN, "xf6r2_o2"), ("f6 seed 3", X_RUN, "xf6r3_o2")],
                  "control": [("n", X_RUN, "xcn_2"), ("ns", X_RUN, "xcns_2")],
                  "fresh": [("n", X_RUN, "xfreshn_2"), ("ns", X_RUN, "xfreshns_2"), ("f6 seed 1", PILOT, "p6fresh_2"),
                            ("f6 seed 2", X_RUN, "xfreshf6r2_2"), ("f6 seed 3", X_RUN, "xfreshf6r3_2")]},
        "reactions": {"continued": [("n", X_RUN, "xn_o1"), ("ns", X_RUN, "xns_o1"), ("f6 seed 1", PILOT, "q6_o1")],
                      "control": [("n", X_RUN, "xcn_1"), ("ns", X_RUN, "xcns_1")],
                      "fresh": [("n", X_RUN, "xfreshn_1"), ("ns", X_RUN, "xfreshns_1"), ("f6 seed 1", PILOT, "q6fresh_1")]}}
ARMS = ("continued", "control", "fresh")
GRAMMARS = ("ctx", "ctx_lib", "uni", "uni_lib")
COLUMNS = (("net", "network"), ("ctx", "K0 grammar"), ("ctx_lib", "+ library"), ("uni", "context-free"),
           ("uni_lib", "+ library"), ("simplest", "simplest-first"))


def points(which):
    """Every lineage at every point: {arm, run, dir, lineage, k, net}; one 'before' point per arm (shared by its runs)."""
    setup = SETUPS[which]
    out = []
    for arm in ARMS:
        start = {"continued": setup["hist"], "control": setup["cbase"], "fresh": None}[arm]
        out.append({"arm": arm, "run": "-", "dir": None, "lineage": None, "k": 0,
                    "net": str(X_RUN / "nets" / start) if start else None})
        for label, d, lin in RUNS[which][arm][:2] if SMOKE else RUNS[which][arm]:
            for k in (10, 30):
                out.append({"arm": arm, "run": label, "dir": str(d), "lineage": lin, "k": k,
                            "net": str(d / "nets" / f"{lin}_{setup['dom']}_d{k}")})
    return out


def experience(pt, o):
    """k0_check.experience for one lineage point: (earlier [(d, own, keys)], their groups or None, newest)."""
    earlier, groups = [], None
    if pt["arm"] == "continued":
        rep = json.loads((C.REUSE / f"replays_o{o}.json").read_text())
        hist = [(k_.rsplit("|", 1), v) for k_, v in rep.items() if C.dom_of(k_.rsplit("|", 1)[0]) in C.HISTORY]
        earlier = [(2, int(e), v) for (tid, e), v in hist]
        groups = [C.dom_of(tid) for (tid, e), v in hist]
    elif pt["arm"] == "control":
        earlier = [tuple(x) for x in json.loads((X_RUN / "k0" / f"madeup_structures_c{o}.json").read_text())]
    newest = []
    for w in [w for w in (10, 30) if w <= pt["k"]]:
        res = json.loads((Path(pt["dir"]) / "wake" / f"{pt['lineage']}_w{w}.json").read_text())
        newest += [(4, e["eq"], e["solution"]["keys"]) for r in res if r["status"] == "done" for e in r["equations"]
                   if e["solution"]]
    return earlier, groups, newest


def grammar_job(args):
    which, i, pt, order, tokens, new_eqs, out_dir = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import grammar_g2 as GR
    import library_ops as L
    t0 = time.time()
    earlier, groups, newest = experience(pt, SETUPS[which]["o"])
    terms = Counter(L.tree(k, int(own)) for d, own, keys in earlier + newest for k in keys)
    libs = {form: L.learn(terms, tokens, form) if terms else ([], []) for form in ("ctx", "uni")}
    t_learn = time.time() - t0
    E, N = [(own, keys) for d, own, keys in earlier], [(own, keys) for d, own, keys in newest]
    tables, bits, alphas = {}, {}, {}
    for g in GRAMMARS:
        form = g.split("_")[0]
        gr = L.LibGrammar.fit(E, N, tokens, libs[form][0] if g.endswith("_lib") else (), earlier_groups=groups, form=form)
        tables[g] = np.array([[gr.logp_term(key, own) for own in range(4)] for key in order])
        bits[g] = float(np.mean([gr.bits(own, keys) for own, keys in new_eqs]))
        alphas[g] = [c.alpha for c in gr.components]
    np.savez(Path(out_dir) / f"tables_{which}_{i}.npz", **tables)
    seen = {s for d, own, keys in earlier + newest for s in GR.equation_shapes(d, own, keys)}
    return {"i": i, "bits": bits, "alphas": alphas, "seen": seen, "seconds": [t_learn, time.time() - t0],
            "library": {form: [{"piece": L.show(x["piece"]), "saved": x["saved"], "uses": x["uses"]} for x in log]
                        for form, (_, log) in libs.items()},
            "sizes": {"equations": len(earlier) + len(newest), "newest equations": len(newest),
                      "terms": sum(terms.values()), "distinct": len(terms)}}


def rank_job(args):
    which, rec, pts, out_dir = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import movenet_g2 as MN
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    gid = vocab.gid

    def ranks(score):                                            # 1 = first; ties by vocabulary order (simplest first)
        order = np.lexsort((np.arange(len(score)), -score))
        r = np.empty(len(score), dtype=np.int64)
        r[order] = np.arange(1, len(score) + 1)
        return r
    simplest = ranks(-np.arange(vocab.V, dtype=np.float64))
    trues = [[vocab.index[k] for k in eq["keys"] if k in vocab.index] for eq in rec["structure"]]
    roots = [SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
             for i in range(len(rec["structure"]))]
    out = []
    for j, pt in enumerate(pts):
        npz = np.load(Path(out_dir) / f"tables_{which}_{j}.npz")
        tab = {g: npz[g] for g in GRAMMARS}
        net_r = None
        if pt["net"]:
            net = MN.load_net(pt["net"] + ".pt").eval()
            emb = {4: np.load(pt["net"] + "_d4.npy")}
            net_r = [ranks(MN.NetworkPrior(net, emb, shortlist=2048)(se.root, se.ctx).p_add) for se in roots]
            del net, emb
        for i, true in enumerate(trues):
            gr = {g: ranks(tab[g][gid, i]) for g in GRAMMARS}
            for t in true:
                out.append(dict({g: int(gr[g][t]) for g in GRAMMARS}, id=rec["id"], eq=i, pt=j, key=vocab.keys[t],
                                net=int(net_r[i][t]) if net_r else None, simplest=int(simplest[t]), V=int(vocab.V)))
    return out


def summary(vals):
    """Median over runs [range]."""
    import numpy as np
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.0f}"
    return f"{np.median(vals):.0f} [{min(vals):.0f}-{max(vals):.0f}]"


def report(which, pts, gres, rows):
    import numpy as np
    import grammar_g2 as GR
    import stageX_report as XR
    first = {arm: next(j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == 0) for arm in ARMS}
    prior_seen = gres[first["continued"]]["seen"] | gres[first["control"]]["seen"]
    shape = {}
    for x in rows:
        s = shape.setdefault((x["key"], x["eq"]), GR.equation_shapes(4, x["eq"], [x["key"]])[0])
        x["unfamiliar"] = s not in gres[x["pt"]]["seen"]
        x["new_to_both"] = s not in prior_seen
        x["kind"] = XR.kind(x["key"], x["eq"]) or "other"
    by_pt = {}
    for x in rows:
        by_pt.setdefault(x["pt"], []).append(x)
    n_eq = len({(x["id"], x["eq"]) for x in rows})
    runs = {arm: sorted({p["run"] for p in pts if p["arm"] == arm and p["k"]}) for arm in ARMS}
    lines = [f"== {which}: {n_eq} new-form development equations in {len({x['id'] for x in rows})} systems; runs per arm: "
             + "; ".join(f"{arm} {len(runs[arm])} ({', '.join(runs[arm])})" for arm in ARMS)]
    lines.append("   Experience and library learned (equations, term occurrences, distinct terms; pieces with their uses and "
                 "the bits each saved when added):")
    for j, p in enumerate(pts):
        g = gres[j]
        lines.append(f"     {p['arm']} {'before' if p['k'] == 0 else p['run'] + ' after ' + str(p['k'])}: "
                     f"{g['sizes']['equations']} eq ({g['sizes']['newest equations']} in the new domain), "
                     f"{g['sizes']['terms']} terms, {g['sizes']['distinct']} distinct")
        for form, title in (("ctx", "contextual"), ("uni", "context-free")):
            lib = "; ".join(f"{x['piece']} [{x['uses']:.0f} uses, {x['saved']:.0f} bits]" for x in g["library"][form])
            lines.append(f"        {title}: {len(g['library'][form])} pieces{': ' + lib if lib else ''}")
    sets = (("unfamiliar to the lineage", lambda x: x["unfamiliar"]), ("new to both prior experiences", lambda x: x["new_to_both"]),
            ("bump", lambda x: x["kind"] == "bump"), ("saturating", lambda x: x["kind"] == "saturating"),
            ("all", lambda x: True))
    V = rows[0]["V"] if rows else 0
    lines.append(f"   Median rank of true terms (of about {V} per system); per arm and point, the median over runs [range] of each "
                 f"run's median. Columns: " + " | ".join(t for _, t in COLUMNS) + "; then the share of terms each form's "
                 "library ranks better / worse than the same form without it (mean over runs)")
    for title, sel in sets:
        lines.append(f"     {title}:")
        for arm in ARMS:
            for k in (0, 10, 30):
                js = [j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == k]
                per = [[x for x in by_pt.get(j, []) if sel(x)] for j in js]
                per = [r for r in per if r]
                if not per:
                    continue
                cells = []
                for f, _ in COLUMNS:
                    meds = [np.median([x[f] for x in r if x[f] is not None]) if any(x[f] is not None for x in r) else None
                            for r in per]
                    cells.append(summary(meds))
                bw = []
                for form in ("ctx", "uni"):
                    better = np.mean([np.mean([x[form + "_lib"] < x[form] for x in r]) for r in per])
                    worse = np.mean([np.mean([x[form + "_lib"] > x[form] for x in r]) for r in per])
                    bw.append(f"{100 * better:.0f}/{100 * worse:.0f}%")
                lines.append(f"       {arm:>9} {'before' if k == 0 else f'after {k}':>8} ({summary([len(r) for r in per])} terms): "
                             + " | ".join(cells) + f"; library better/worse: contextual {bw[0]}, context-free {bw[1]}")
    lines.append("   Bits check (mean bits per new-form equation; median over runs [range]): uniform "
                 f"{gres[first['fresh']]['bits']['ctx']:.1f}. Columns: K0 grammar | + library | context-free | + library")
    for arm in ARMS:
        for k in (0, 10, 30):
            js = [j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == k]
            if js:
                lines.append(f"       {arm:>9} {'before' if k == 0 else f'after {k}':>8}: "
                             + " | ".join(_bits(gres, js, g) for g in GRAMMARS))
    return lines


def _bits(gres, js, g):
    import numpy as np
    v = [gres[j]["bits"][g] for j in js]
    return f"{np.median(v):.1f}" + (f" [{min(v):.1f}-{max(v):.1f}]" if len(v) > 1 else "")


def main():
    import numpy as np
    import grammar_ops as GO
    import search_g1 as SG
    workers = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    order = SG.vocabulary_order(4)
    tokens = GO.alphabet(order + SG.vocabulary_order(2))
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    lines, checks = [], []
    for which, setup in SETUPS.items():
        dom = setup["dom"]
        dev = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        new_eqs = [(i, eq["keys"]) for r in dev for i, eq in enumerate(r["structure"]) if not flags[r["id"]][i]]
        pts = points(which)
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            gres = list(ex.map(grammar_job, [(which, i, pt, order, tokens, new_eqs, str(OUT)) for i, pt in enumerate(pts)]))
            t1 = time.time()
            rows = [x for part in ex.map(rank_job, [(which, r, pts, str(OUT)) for r in dev]) for x in part]
        flagd = {(r["id"], i): not flags[r["id"]][i] for r in dev for i in range(len(r["structure"]))}
        rows = [x for x in rows if flagd[(x["id"], x["eq"])]]          # new-form equations only
        (OUT / f"ranks_{which}.json").write_text(json.dumps(rows))
        (OUT / f"libraries_{which}.json").write_text(json.dumps(
            [dict(p, **{k: v for k, v in g.items() if k != "seen"}) for p, g in zip(pts, gres)], default=str))
        checks.append(f"{which}: {len(pts)} lineage points; grammars and libraries {t1 - t0:.0f}s (slowest library "
                      f"{max(g['seconds'][0] for g in gres):.0f}s); ranks {time.time() - t1:.0f}s; rows {len(rows)}")
        if SMOKE:                                                # the grammar without a library is K0's, exactly
            k0 = np.load(X_RUN / "k0" / f"tables_{which}.npz")
            for j, p in enumerate(pts):
                if p["run"] in ("-", "n") and f"{p['arm']}_{p['k']}_op" in k0.files:
                    t = np.load(OUT / f"tables_{which}_{j}.npz")["ctx"]
                    checks.append(f"   {p['arm']} {p['k']}: no-library grammar equals K0's: "
                                  f"{bool(np.allclose(t, k0[p['arm'] + '_' + str(p['k']) + '_op'], rtol=0, atol=1e-9))}")
        lines += report(which, pts, gres, rows)
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print("\n".join(checks))
    if not SMOKE:
        print(text)


if __name__ == "__main__":
    main()
