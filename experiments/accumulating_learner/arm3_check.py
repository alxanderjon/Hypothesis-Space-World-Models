"""Arm 3's quick check (owner, 13 November; research log "Arm 3's quick check: design and pass rule"): would a network that
adjusts the library's weights for each system from its data rank true terms better than the library alone and the
network alone? A train-free stand-in: at each node the network's distribution over addable terms is spread onto the
library's building choices (context-free form; each choice gets the mass of the terms whose parse uses it, normalized
within its slot type), averaged in log with the library's own weights and renormalized; a term's score is the sum of its
parse's log weights. Reference: the product (library log probability plus the network's). The pilot's network lineages
(network after level 5; library from their own solutions at levels 1, 2, 3, 5 with the extended equivalence fix), on the
follow-up check's held-out systems. No search, no training. Usage: arm3_check.py <workers> <prepare|rank|report>"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "cpilot")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canon_check as CN                                         # noqa: E402

C, D, CP = CN.C, CN.D, CN.CP
OUT = C.ROOT / "outputs/accumulating_learner/g2h/arm3"
LINEAGES = ("net-s1", "net-s2")
COMBOS = CN.COMBOS
SLOTS = ("expression", "exponent")


def network_after(lineage, level=5):
    """The network after the lineage's last round at the given level (one training per round)."""
    rec = json.loads((CN.PILOT / "lineages" / f"{lineage}.json").read_text())
    n = sum(len(L_["rounds"]) for L_ in rec["levels"] if L_["level"] <= level)
    return str(CN.PILOT / "lineages" / lineage / f"learn{n}")


def step_prepare(workers):
    """Canonical solutions of the network lineages (extended fix), their library, and its choice weights."""
    import numpy as np
    import grammar_ops as GO
    import library_ops as L
    import search_g1 as SG
    from collections import Counter
    OUT.mkdir(parents=True, exist_ok=True)
    CN.step_canon(workers, list(LINEAGES))
    order = SG.vocabulary_order(D)
    tokens = GO.alphabet(order, d=D)
    info = {}
    for lin in LINEAGES:
        _, level = CN.solutions(lin)
        res = json.loads((CN.OUT / f"canon_{lin}.json").read_text())
        eqs = [(int(eq), e["canon_plus"]) for r in res if level[r["id"]] in CN.MASTERED for eq, e in r["equations"].items()]
        terms = Counter(L.tree(k, own) for own, keys in eqs for k in keys)
        lib = L.learn(terms, tokens, "uni")[0] if terms else []
        g = L.LibGrammar.fit(eqs, [], tokens, lib, form="uni")
        comp = g.components[0]
        weights = {f"{slot}|{sym}": comp.lp(slot, sym, comp.alpha) for slot in SLOTS for sym in list(tokens) + [n for n, _ in lib]}
        tab = np.array([[g.logp_term(k, own) for own in range(D)] for k in order])
        np.savez(OUT / f"library-{lin}.npz", t=tab)
        (OUT / f"library-{lin}.pkl").write_bytes(pickle.dumps({"library": lib, "weights": weights}))
        info[lin] = {"equations": len(eqs), "pieces": [L.show(p) for _, p in lib], "network": network_after(lin)}
    (OUT / "prepare.json").write_text(json.dumps(info, indent=1))
    print("prepare: " + "; ".join(f"{k}: {v['equations']} equations, {len(v['pieces'])} pieces, network {Path(v['network']).name}"
                                  for k, v in info.items()))


def rank_job(args):
    rec, lin, net_path, tab_path, lib_path, new_pats, familiar_levels = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import curriculum_check as CC
    import curriculum_check2 as C2
    import fitter_g1 as FG
    import library_ops as L
    import movenet_g2 as MN
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    net = MN.NetworkPrior(MN.load_net(net_path + ".pt").eval(), {D: np.load(net_path + f"_d{D}.npy")}, shortlist=2048)
    tab = np.load(tab_path)["t"][vocab.gid]
    lw = pickle.loads(Path(lib_path).read_bytes())
    lib, weights = L.ordered(lw["library"]), lw["weights"]
    key_fn = L.context("uni", lib)
    choice_id, inc = {}, {}

    def incidence(i):
        """(term index, choice index) pairs of every term's parse for equation i, and each choice's slot."""
        if i not in inc:
            pt, pc = [], []
            for u, k in enumerate(vocab.keys):
                for p, a, ch in L.parse(L.tree(k, i), lib)[0]:
                    c = (key_fn(p, a), ch)
                    if c not in choice_id:
                        choice_id[c] = len(choice_id)
                    pt.append(u)
                    pc.append(choice_id[c])
            inc[i] = (np.array(pt), np.array(pc))
        return inc[i]

    def rank(score, legal, t):
        s = np.where(legal, score, -np.inf)
        return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)
    out = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        kinds = {}
        for t in true:
            if C2.is_new(vocab.keys[t], i, new_pats):
                kinds[t] = "new"
            elif CC.level_of(vocab.keys[t], i) in familiar_levels:
                kinds[t] = "familiar"
        if not kinds:
            continue
        pt, pc = incidence(i)
        cl = sorted(choice_id.items(), key=lambda x: x[1])
        slot_of = np.array([SLOTS.index(c[0]) if c[0] in SLOTS else 0 for c, _ in cl])
        wlib = np.array([weights.get(f"{c[0]}|{c[1]}", -30.0) for c, _ in cl])
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        memo = {}
        for t, kind in kinds.items():
            others = [u for u in true if u != t]
            points = [("empty start", ()), ("one short", tuple(others))]
            if len(true) >= 3:
                points.append(("two short", tuple(others[1:])))
            for point, S in points:
                if S not in memo:
                    nd = se.root if not S else se.evaluate(S, None)
                    legal = np.ones(vocab.V, dtype=bool)
                    legal[list(S)] = False
                    pn = np.where(legal, net(nd, se.ctx).p_add, 0.0)
                    pn = pn / pn.sum()
                    m = np.zeros(len(cl))
                    np.add.at(m, pc, pn[pt])
                    q = np.zeros(len(cl))
                    for sl in range(len(SLOTS)):
                        idx = slot_of == sl
                        q[idx] = (m[idx] + 1e-6) / (m[idx] + 1e-6).sum()
                    adj = 0.5 * wlib + 0.5 * np.log(q)
                    for sl in range(len(SLOTS)):
                        idx = slot_of == sl
                        adj[idx] = adj[idx] - np.logaddexp.reduce(adj[idx])
                    score = np.zeros(vocab.V)
                    np.add.at(score, pt, adj[pc])
                    with np.errstate(divide="ignore"):
                        logn = np.log(np.maximum(pn, 1e-300))
                    prod = tab[:, i] + logn
                    sc = {"network": pn, "library": tab[:, i], "adjusted": score, "product": prod}
                    lib_order = np.lexsort((np.arange(vocab.V), -np.where(legal, tab[:, i], -np.inf)))
                    for n_top in (100, 1000):                    # the library proposes, the network rescores whole terms
                        top = lib_order[:min(n_top, int(legal.sum()))]
                        blend = tab[:, i].copy()
                        blend[top] = 1e6 + 0.5 * tab[top, i] + 0.5 * logn[top]
                        sc[f"blend{n_top}"] = blend
                    memo[S] = (legal, sc)
                legal, sc = memo[S]
                row = {"lineage": lin, "eq": i, "key": vocab.keys[t], "kind": kind, "point": point}
                row.update({n: rank(v, legal, t) for n, v in sc.items()})
                out.append(row)
    return out


def step_rank(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import curriculum_check2 as C2
    info = json.loads((OUT / "prepare.json").read_text())
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futs = []
        for name in COMBOS:
            _, _, _, pats = C2.combos()[name]
            dev = pickle.loads((CN.CHECK2 / "gen" / f"dev-{name}.pkl").read_bytes())
            for lin in LINEAGES:
                for r in dev:
                    futs.append((name, ex.submit(rank_job, (r, lin, info[lin]["network"], str(OUT / f"library-{lin}.npz"),
                                                            str(OUT / f"library-{lin}.pkl"), pats, CN.MASTERED))))
        for name, f in futs:
            rows += [dict(x, combo=name) for x in f.result()]
    (OUT / "ranks.json").write_text(json.dumps(rows))
    print(f"rank: {len(rows)} rows in {time.time() - t0:.0f}s")


def step_report():
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    methods = ("network", "library", "adjusted", "product", "blend100", "blend1000")
    lines = ["== Arm 3's quick check: median rank (of about 1,100) at the empty start / one short / two short; seed 1 | seed 2",
             "   columns per term set: " + ", ".join(methods)]
    ok = {m: True for m in ("adjusted", "product", "blend100", "blend1000")}
    for kind in ("new", "familiar"):
        lines.append(f"   {kind} terms:")
        for m in methods:
            cells = []
            for lin in LINEAGES:
                v = []
                for pt in ("empty start", "one short", "two short"):
                    R = [x[m] for x in rows if x["lineage"] == lin and x["kind"] == kind and x["point"] == pt]
                    v.append(float(np.median(R)) if R else float("nan"))
                cells.append(" / ".join(f"{x:.0f}" for x in v))
            lines.append(f"      {m:>9}: " + " | ".join(cells))
    for m in ("adjusted", "product", "blend100", "blend1000"):
        for lin in LINEAGES:
            for pt in ("empty start", "one short", "two short"):
                med = {k: {mm: float(np.median([x[mm] for x in rows if x["lineage"] == lin and x["kind"] == k and x["point"] == pt]))
                           for mm in methods} for k in ("new", "familiar")}
                ok[m] &= (med["new"][m] < med["new"]["library"] and med["new"][m] < med["new"]["network"]
                          and med["familiar"][m] <= 2 * med["familiar"]["network"])
        lines.append(f"== {m}: {'PASSES' if ok[m] else 'does not pass'} (new terms below both the library and the network, familiar "
                     "terms within twice the network, both seeds, all three points)")
    n = {k: len([x for x in rows if x["kind"] == k]) for k in ("new", "familiar")}
    lines.append(f"   cases: new {n['new']}, familiar {n['familiar']} (both seeds, all points)")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    {"prepare": lambda: step_prepare(workers), "rank": lambda: step_rank(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
