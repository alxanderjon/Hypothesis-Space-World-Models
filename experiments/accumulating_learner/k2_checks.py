"""The six quick checks of the K2 and Stage C design (owner's approval, 12 November; research log "Owner: run the six
quick checks; their exact pass rules, set before any of them runs"). No search, no training; development systems' new-form
equations at the evidence probe's nodes (one and two terms short), K1's 8 networks per domain; counts only.

Steps: nodes (one pass for checks 1, 2(ii), 3 and 4: evidence, each network's and each lineage's K0 grammar's order,
the choice-level data weights with no library, each lineage's learned contextual library and a decoy library, the
screened hits, and the proposals across equations); headroom (check 2(i)); pairs (check 5); constants (check 6); report.
Usage: k2_checks.py <workers> <step ...> [smoke] (smoke: two development systems per domain, written to k2checks_smoke,
prints only timings)"""

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
os.environ.setdefault("G2H_RUN", "k1")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1_check as K                                             # noqa: E402

C = K.C
OUT = C.ROOT / "outputs/accumulating_learner/g2h" / ("k2checks_smoke" if SMOKE else "k2checks")
NS = (64, 256, 1024)
EPS = 1e-4
NEW_DOMS = ("flows_L1", "reactions_L1")


def networks(which):
    return [(arm, run, str(folder / "nets" / f"{lin}_{K.DOMS[which]}_d{K.K}"), lin)
            for arm, nets in K.NETS[which].items() for run, folder, lin, _ in nets]


def rank(score, legal, t):
    import numpy as np
    s = np.where(legal, score, -np.inf)
    return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)


def screened_hit(score, ev, legal, t, n):
    """The missing term is among the source's top n and fewer than 10 of those n have more evidence than it."""
    import numpy as np
    s = np.where(legal, score, -np.inf)
    top = np.lexsort((np.arange(len(s)), -s))[:n]
    if t not in set(top.tolist()):
        return False
    e = ev[top]
    return int(((e > ev[t]) | ((e == ev[t]) & (top < t))).sum()) < 10


class Choices:
    """The choice-level data weights over a vocabulary's terms for one equation, with or without a library."""

    def __init__(self, keys, own, library):
        import numpy as np
        import library_ops as L
        lib = L.ordered(library) if library else []
        ids, ctxs, pt, pc, cctx = {}, {}, [], [], []
        for u, key in enumerate(keys):
            for p, a, ch in L.parse(L.tree(key, own), lib)[0]:
                cid = ids.get((p, a, ch))
                if cid is None:
                    cid = ids[(p, a, ch)] = len(ids)
                    cctx.append(ctxs.setdefault((p, a), len(ctxs)))
                pt.append(u)
                pc.append(cid)
        self.pt, self.pc, self.cctx = np.array(pt), np.array(pc), np.array(cctx)
        self.nC, self.nX, self.V = len(ids), len(ctxs), len(keys)

    def score(self, ev, legal):
        import numpy as np
        sel = legal[self.pt]
        pt, pc = self.pt[sel], self.pc[sel]
        w = np.zeros(self.nC)
        np.maximum.at(w, pc, np.maximum(ev[pt], 0.0))
        present = np.zeros(self.nC, dtype=bool)
        present[pc] = True
        num = np.where(present, w + EPS, 0.0)
        den = np.zeros(self.nX)
        np.add.at(den, self.cctx, num)
        d = den[self.cctx]
        logp = np.full(self.nC, -np.inf)
        logp[present] = np.log(num[present] / d[present])
        out = np.zeros(self.V)
        np.add.at(out, pt, logp[pc])
        return np.where(legal, out, -np.inf)


def job_nodes(args):
    which, rec, flags, nets, tables, libs, sols = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import search_g1 as SG
    t0 = time.time()
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    gid, keys = vocab.gid, vocab.keys
    priors = {lin: C._prior(path) for _, _, path, lin in nets}
    k0 = {lin: np.load(p)[k][gid] for lin, (p, k) in tables.items()}
    choices = {}

    def ch(tag, own):
        if (tag, own) not in choices:
            choices[(tag, own)] = Choices(keys, own, None if tag is None else libs[tag[0]][tag[1]])
        return choices[(tag, own)]
    trues = [sorted({vocab.index[k] for k in eq["keys"] if k in vocab.index}) for eq in rec["structure"]]
    out = []
    for i, eq in enumerate(rec["structure"]):
        if flags[i]:
            continue
        true = trues[i]
        others = {u for j, tj in enumerate(trues) if j != i for u in tj}
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        probes = []
        for j, t in enumerate(true):
            probes.append(("one short", t, tuple(u for u in true if u != t)))
            if len(true) >= 2:
                t2 = true[(j + 1) % len(true)]
                probes.append(("two short", t, tuple(u for u in true if u not in (t, t2))))
        memo = {}
        for kind, t, S in probes:
            if S not in memo:
                nd = se.root if not S else se.evaluate(S, None)
                ev = SG.evidence(nd, se.ctx)[0]
                legal = np.ones(vocab.V, dtype=bool)
                legal[list(S)] = False
                net = {lin: np.where(legal, pr(nd, se.ctx).p_add, -np.inf) for lin, pr in priors.items()}
                memo[S] = (ev, legal, net, ch(None, i).score(ev, legal),
                           {lin: (ch((lin, "learned"), i).score(ev, legal), ch((lin, "decoy"), i).score(ev, legal)) for lin in priors})
            ev, legal, net, cl0, cl = memo[S]
            row = {"id": rec["id"], "eq": i, "key": keys[t], "node": kind, "ev": rank(ev, legal, t), "cl0": rank(cl0, legal, t),
                   "other_eq": bool(t in others)}
            for n in NS:
                row[f"hit:cl0:{n}"] = screened_hit(cl0, ev, legal, t, n)
            for lin in priors:
                nr = rank(net[lin], legal, t)
                row[f"net:{lin}"] = nr
                row[f"k0:{lin}"] = rank(k0[lin][:, i], legal, t)
                row[f"lib:{lin}"] = rank(cl[lin][0], legal, t)
                row[f"dec:{lin}"] = rank(cl[lin][1], legal, t)
                for n in NS:
                    row[f"hit:net:{n}:{lin}"] = screened_hit(net[lin], ev, legal, t, n)
                    row[f"hit:k0:{n}:{lin}"] = screened_hit(k0[lin][:, i], ev, legal, t, n)
                if kind == "one short":                          # check 4: proposals across equations
                    found = Counter(u for j, ks in sols.get(lin, {}).items() if int(j) != i
                                    for u in {vocab.index[k] for k in ks if k in vocab.index})
                    prop = [u for u, _ in sorted(found.items(), key=lambda x: (-x[1], x[0])) if legal[u]]
                    row[f"c1:{lin}"] = prop.index(t) + 1 if t in prop else len(prop) + nr
            out.append(row)
    return {"rows": out, "seconds": time.time() - t0}


def job_headroom(args):
    which, rec, flags, tables, tokens = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import grammar_ops as GO
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    vocab = ctx["vocab"]
    gid, keys = vocab.gid, vocab.keys
    k0 = {lin: np.load(p)[k][gid] for lin, (p, k) in tables.items()}
    tk = [[k for k in eq["keys"] if k in vocab.index] for eq in rec["structure"]]
    out = []
    for i, eq in enumerate(rec["structure"]):
        if flags[i]:
            continue
        for key in tk[i]:
            t = vocab.index[key]
            eqs = [(j, [k for k in ks if not (j == i and k == key)]) for j, ks in enumerate(tk)]
            comp = GO.OpComponent([(j, ks) for j, ks in eqs if ks], tokens, alpha=1.0)
            ls = np.array([comp.logp_term(k, i) for k in keys])
            legal = np.ones(vocab.V, dtype=bool)
            legal[[vocab.index[k] for k in tk[i] if k != key]] = False
            row = {"id": rec["id"], "eq": i, "key": key}
            for lin, tab in k0.items():
                row[f"head:{lin}"] = rank(np.logaddexp(tab[:, i], ls) - math.log(2), legal, t)
                row[f"k0:{lin}"] = rank(tab[:, i], legal, t)
            out.append(row)
    return {"rows": out}


def job_pairs(args):
    which, rec, flags = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import fitter_g1 as FG
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    trues = [sorted({vocab.index[k] for k in eq["keys"] if k in vocab.index}) for eq in rec["structure"]]
    ses = {}

    def at(i, S):
        if i not in ses:
            ses[i] = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        se = ses[i]
        nd = se.root if not S else se.evaluate(S, None)
        return SG.evidence(nd, se.ctx)[0]
    out = []
    d = len(trues)
    for i in range(d):
        for j in range(i + 1, d):
            if flags[i] and flags[j]:
                continue
            for t in sorted(set(trues[i]) & set(trues[j])):
                Si = tuple(u for u in trues[i] if u != t)
                Sj = tuple(u for u in trues[j] if u != t)
                ei, ej = at(i, Si), at(j, Sj)
                legal = np.ones(vocab.V, dtype=bool)
                legal[list(set(Si) | set(Sj))] = False
                out.append({"id": rec["id"], "eqs": [i, j], "key": vocab.keys[t], "one_i": rank(ei, legal, t),
                            "one_j": rank(ej, legal, t), "joint": rank(ei + ej, legal, t)})
    return {"rows": out}


def _terms(rhs, params):
    """[(coefficient parameter or None, numeric factor, monomial, inner parameters)] of an equation's description."""
    import sympy as sp
    e = sp.sympify(rhs)
    out = []
    for term in sp.Add.make_args(e):
        num, coef, rest, inner = 1, None, [], set()
        for f in sp.Mul.make_args(term):
            if f.is_Number:
                num *= f
            elif f.is_Symbol and str(f) in params and coef is None:
                coef = str(f)
            else:
                rest.append(f)
                inner |= {str(s) for s in f.free_symbols if str(s) in params}
        out.append((coef, num, sp.Mul(*rest), inner))
    return out


def constants_training():
    """Check 6(i): per domain, the share of training equations with a parameter that also appears in another equation."""
    out = {}
    for dom in C.HISTORY + NEW_DOMS:
        n = shared_any = shared_coef = shared_inner = 0
        for r in C.records("train", dom):
            params = r["desc"]["params"]
            eqs = [_terms(rhs, params) for rhs in r["desc"]["rhs"]]
            sets = [({c for c, _, _, _ in ts if c} | {s for _, _, _, inn in ts for s in inn}) for ts in eqs]
            for i, ts in enumerate(eqs):
                others = set().union(*[s for j, s in enumerate(sets) if j != i])
                coefs = {c for c, _, _, _ in ts if c}
                inner = {s for _, _, _, inn in ts for s in inn}
                n += 1
                shared_any += bool(sets[i] & others)
                shared_coef += bool(coefs & others)
                shared_inner += bool(inner & others)
        out[dom] = {"equations": n, "any": shared_any / max(n, 1), "coefficient": shared_coef / max(n, 1),
                    "inner": shared_inner / max(n, 1)}
    return out


def job_constants(args):
    """Check 6(ii): development pairs of plain-product terms whose coefficient is one parameter in two equations: separate
    fits against fits with that coefficient tied (each equation's rows scaled by its target's size)."""
    which, rec = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import sympy as sp
    import torch
    import fitter_g1 as FG
    import search_g1 as SG
    params = rec["desc"]["params"]
    d = len(rec["structure"])
    ys = [sp.Symbol(f"y{k}", real=True) for k in range(d)]
    xs = [sp.Symbol(f"x{k + 1}", real=True) for k in range(d)]
    eqs = [_terms(rhs, params) for rhs in rec["desc"]["rhs"]]
    occ = {}
    for i, ts in enumerate(eqs):
        for coef, num, mono, inner in ts:
            if coef and not inner and mono.free_symbols <= set(ys):
                key = sp.srepr(mono.subs(dict(zip(ys, xs)), simultaneous=True))
                occ.setdefault(coef, []).append((i, key, float(num), mono))
    pairs = [(a, b) for v in occ.values() for x, a in enumerate(v) for b in v[x + 1:] if a[0] != b[0]]
    if not pairs:
        return {"rows": []}
    ctx = SG.system_context(np.array(rec["bursts"]))
    sig = np.asarray(ctx["view"]["sig"], dtype=float)           # the learner's coordinates are x / sig, its targets dx/dt / sig

    def factor(i, num, mono):
        """The shared parameter's multiplier in equation i in the learner's coordinates."""
        return num * float(np.prod([sig[ys.index(v)] ** int(p_) for v, p_ in mono.as_powers_dict().items()])) / sig[i]
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    fits = {}

    def fit(i):
        if i not in fits:
            se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
            S = tuple(sorted({vocab.index[k] for k in rec["structure"][i]["keys"] if k in vocab.index}))
            nd = se.evaluate(S, None)
            Phi = torch.stack([FG.value(t, ctx["W"]) for t in nd["terms"]], 1).double().numpy()
            fits[i] = (list(nd["keys"]), np.nan_to_num(Phi), se.y.double().numpy(), float(se.yn))
        return fits[i]
    out = []
    for (i, ki, ni_, mi), (j, kj, nj_, mj) in pairs:
        fi, fj = factor(i, ni_, mi), factor(j, nj_, mj)
        Ki, Pi, yi, ni = fit(i)
        Kj, Pj, yj, nj = fit(j)
        if ki not in Ki or kj not in Kj:
            continue
        a, b = Ki.index(ki), Kj.index(kj)
        ci = np.linalg.lstsq(Pi, yi, rcond=None)[0]
        cj = np.linalg.lstsq(Pj, yj, rcond=None)[0]
        sep = (float(np.linalg.norm(yi - Pi @ ci) / ni), float(np.linalg.norm(yj - Pj @ cj) / nj))
        oi, oj = [k for k in range(Pi.shape[1]) if k != a], [k for k in range(Pj.shape[1]) if k != b]
        A = np.zeros((len(yi) + len(yj), len(oi) + len(oj) + 1))
        A[:len(yi), :len(oi)] = Pi[:, oi] / ni
        A[len(yi):, len(oi):len(oi) + len(oj)] = Pj[:, oj] / nj
        A[:len(yi), -1] = fi * Pi[:, a] / ni
        A[len(yi):, -1] = fj * Pj[:, b] / nj
        z = np.linalg.lstsq(A, np.concatenate([yi / ni, yj / nj]), rcond=None)[0]
        ri = yi - Pi[:, oi] @ z[:len(oi)] - fi * z[-1] * Pi[:, a]
        rj = yj - Pj[:, oj] @ z[len(oi):len(oi) + len(oj)] - fj * z[-1] * Pj[:, b]
        tied = (float(np.linalg.norm(ri) / ni), float(np.linalg.norm(rj) / nj))
        out.append({"id": rec["id"], "eqs": [i, j], "separate": sep, "tied": tied})
    return {"rows": out}


def prepare(which, tokens):
    """Per lineage: its K0 grammar table, its learned contextual library at 30 tasks and a decoy of the same sizes, and
    K1's ordering-A solutions per system."""
    import numpy as np
    import k0b_check as KB
    import library_ops as L
    import search_g1 as SG
    nets = networks(which)
    tables = {lin: K.table_of(which, arm, run, "B") for arm, run, _, lin in nets}
    libs, sols = {}, {}
    order = SG.vocabulary_order(4)
    pts = KB.points(which)
    for x, (arm, run, _, lin) in enumerate(nets):
        pt = next(p for p in pts if p["arm"] == arm and p["run"] == run and p["k"] == K.K)
        e, g, n = KB.experience(pt, KB.SETUPS[which]["o"])
        terms = Counter(L.tree(k, int(own)) for d, own, keys in e + n for k in keys)
        learned = L.learn(terms, tokens, "ctx")[0] if terms else []
        rng = np.random.default_rng([7, x])
        taken, decoy = {p for _, p in learned}, []
        for _, p in learned:
            s, h = L.size(p), L.holes(p)
            for _ in range(2000):
                tr = L.tree(order[int(rng.integers(len(order)))], int(rng.integers(4)))
                cands = [a for u in L.subtrees(tr) for a in L.abstractions(u) if L.size(a) == s and L.holes(a) == h and a not in taken]
                if cands:
                    a = cands[int(rng.integers(len(cands)))]
                    decoy.append((f"D{len(decoy)}", a))
                    taken.add(a)
                    break
        libs[lin] = {"learned": learned, "decoy": decoy}
        res = json.loads((K.OUT / f"{lin}_A_s{K.SEED}_{K.DOMS[which]}.json").read_text())
        sols[lin] = {r["id"]: {str(e_["eq"]): e_["solution"]["keys"] for e_ in r.get("equations", [])
                               if e_.get("first_exact") is not None and e_.get("solution")} for r in res if r["status"] == "done"}
    return nets, tables, libs, sols


def run_steps(steps, workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import grammar_ops as GO
    import search_g1 as SG
    OUT.mkdir(parents=True, exist_ok=True)
    order = SG.vocabulary_order(4)
    tokens = GO.alphabet(order + SG.vocabulary_order(2))
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    checks = []
    for which, dom in K.DOMS.items():
        recs = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        prep = prepare(which, tokens) if {"nodes", "headroom"} & set(steps) else None
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            for step in steps:
                t0 = time.time()
                if step == "nodes":
                    nets, tables, libs, sols = prep
                    res = list(ex.map(job_nodes, [(which, r, flags[r["id"]], nets, tables, libs,
                                                   {lin: sols[lin].get(r["id"], {}) for lin in sols}) for r in recs]))
                    (OUT / f"nodes_{which}.json").write_text(json.dumps(
                        {"rows": [x for r in res for x in r["rows"]],
                         "libraries": {lin: {k: len(v) for k, v in d.items()} for lin, d in libs.items()}}))
                elif step == "headroom":
                    nets, tables, _, _ = prep
                    res = list(ex.map(job_headroom, [(which, r, flags[r["id"]], tables, tokens) for r in recs]))
                    (OUT / f"headroom_{which}.json").write_text(json.dumps([x for r in res for x in r["rows"]]))
                elif step == "pairs":
                    res = list(ex.map(job_pairs, [(which, r, flags[r["id"]]) for r in recs]))
                    (OUT / f"pairs_{which}.json").write_text(json.dumps([x for r in res for x in r["rows"]]))
                elif step == "constants":
                    res = list(ex.map(job_constants, [(which, r) for r in recs]))
                    (OUT / f"constants_{which}.json").write_text(json.dumps([x for r in res for x in r["rows"]]))
                else:
                    continue
                checks.append(f"{which} {step}: {time.time() - t0:.0f}s")
    if "constants" in steps:
        (OUT / "constants_training.json").write_text(json.dumps(constants_training()))
        checks.append("constants (training systems) done")
    print("\n".join(checks))


def main():
    workers = int(ARGS[1])
    steps = [s for s in ARGS[2:] if s != "smoke"]
    if steps == ["report"]:
        import k2_checks_report
        k2_checks_report.report()
        return
    run_steps(steps, workers)


if __name__ == "__main__":
    main()
