"""Stage 2a of the general learner: the report (research log, Stage 2a rules, section 7). Counts only.
Per order and new domain: the learning curves of the continued network, the made-up-systems control and the fresh
network (tests before any task and after 10, 30 and 100 tasks), split into new-form and shared-form equations (claims of
accumulation rest on new forms and must beat the control); the validity check (the fresh network at most 70% of new-form
equations after 30 tasks); bar (B) at 10 tasks (continued vs control, new-form, +10 points and a paired sign test
p < 0.05), zero-shot as the second measure, (C) against the fresh network, retention (E) per system, the shortlist probe,
time-limit hits, search effort, and the provenance of new-form equations solved only by the continued network. With both
orders: (D), the continued network in order 1 against order 2 on the equations that are new-form in both orders, and the
gate to Stage 2b ((B) met in at least one new domain, in both orders). Usage: g2_report.py [order ...]"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYS = ROOT / "outputs/accumulating_learner/g1/systems"
RUN = ROOT / "outputs/accumulating_learner/g2/run1"
ORDERS = {1: ("mechanics", "circuits", "ecology", "chemistry", "chaotic"), 2: ("ecology", "circuits", "mechanics", "chaotic", "chemistry")}
NAME = {"chemistry": "chemistry", "chaotic": "quadratic flows (mostly non-chaotic)", "mechanics": "mechanics",
        "circuits": "circuits", "ecology": "ecology"}
POINTS = (0, 10, 30, 100)


def load(label, dom):
    p = RUN / "test" / f"{label}_{dom}.json"
    return json.loads(p.read_text()) if p.exists() else None


def solved_map(res):
    return {(r["id"], e["eq"]): (r["status"] == "done" and e["first_exact"] is not None)
            for r in res for e in r.get("equations", [])}


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(a, b) + 1)) / 2 ** n) if n else 1.0


def counts(res, keys):
    s = solved_map(res)
    return sum(s.get(k, False) for k in keys)


def paired(a, b, keys):
    sa, sb = solved_map(a), solved_map(b)
    return (sum(1 for k in keys if sa.get(k, False) and not sb.get(k, False)),
            sum(1 for k in keys if sb.get(k, False) and not sa.get(k, False)))


def only(a, b, keys):
    sa, sb = solved_map(a), solved_map(b)
    return [k for k in keys if sa.get(k, False) and not sb.get(k, False)]


def bar_b(c, k, new):
    a_, b_ = paired(c, k, new)
    pts = 100.0 * (counts(c, new) - counts(k, new)) / max(len(new), 1)
    return pts >= 10 and a_ > b_ and sign_p(a_, b_) < 0.05


def probe(res):
    p = [r["probe"] for r in res if r.get("probe")]
    if not p:
        return "no network"
    w, c = sum(x["with_missing_true"] for x in p), sum(x["cut"] for x in p)
    rw, rc = sum(x["root_with_missing_true"] for x in p), sum(x["root_cut"] for x in p)
    return f"cut at {c} of {w} expansions with a missing true term ({100.0 * c / max(w, 1):.1f}%); at the root {rc} of {rw}"


def effort(res, keys):
    """Median structures to the first plausible model over the given equations (unsolved count as not reached), and
    the search compute of the whole test (structures, thorough fits, CPU seconds)."""
    first = {(r["id"], e["eq"]): (e["first_exact"] if r["status"] == "done" and e["first_exact"] is not None else math.inf)
             for r in res for e in r.get("equations", [])}
    f = sorted(first.get(k, math.inf) for k in keys)
    med = f[len(f) // 2] if f else math.inf
    eqs = [e for r in res for e in r.get("equations", [])]
    return (f"median structures to the first plausible model (new-form) {med if math.isfinite(med) else 'not reached'}; "
            f"structures {sum(e['structures'] for e in eqs)}, thorough fits {sum(e['thorough'] for e in eqs)}, "
            f"CPU {sum(r['seconds'] for r in res) / 3600:.2f} h")


class Provenance:
    """Which earlier-domain verified solutions of the lineage share a true term with an equation. Terms are compared
    relative to their equation (own variable first, the other variables in either order) as three-variable vocabulary
    representatives, so that two- and three-variable systems are comparable."""

    def __init__(self):
        sys.path.insert(0, str(HERE))
        import domains_g1 as DG
        self.DG, self.cache = DG, {}
        self.dim = {r["id"]: r["desc"]["d"] for dom in ORDERS[1]
                    for r in json.loads((SYS / f"train_{dom}.json").read_text()) if r["status"] == "accepted"}

    def canon(self, key, own, d):
        ck = (key, own, d)
        if ck not in self.cache:
            DG = self.DG
            others = [j for j in range(d) if j != own]
            perm = [0] * d
            for r, j in enumerate(others):
                perm[j] = r + 1
            r = DG.relabelled(key, perm)
            out = None
            if r is not None:
                cands = [DG.representative(x, 3) for x in (DG.relabelled(r, p) for p in ([0, 1, 2], [0, 2, 1])) if x is not None]
                out = min(cands) if cands else None
            self.cache[ck] = out
        return self.cache[ck]

    def earlier_terms(self, lineage, domains):
        p = RUN / f"replays_{lineage}.json"
        rep = json.loads(p.read_text()) if p.exists() else {}
        terms = set()
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            if tid.split("-")[2] in domains:
                terms |= {self.canon(x, int(eq), self.dim[tid]) for x in keys}
        terms.discard(None)
        return terms

    def summary(self, wins, truth, dims, terms):
        n_any = sum(1 for sid, i in wins if any(self.canon(x, i, dims[sid]) in terms for x in truth[sid][i]["keys"]))
        n_all = sum(1 for sid, i in wins if all(self.canon(x, i, dims[sid]) in terms for x in truth[sid][i]["keys"]))
        return f"{len(wins)} equations; {n_any} with a true term in an earlier domain's verified solutions, {n_all} with all"


def main():
    orders = [int(x) for x in sys.argv[1:]] or [1, 2]
    shared = json.loads((SYS / "shared_forms_g2.json").read_text())
    prov = Provenance()
    verdict, new_of = {}, {}
    for o in orders:
        for pos, dom in ((4, ORDERS[o][3]), (5, ORDERS[o][4])):
            tests = [r for r in json.loads((SYS / f"test_{dom}.json").read_text()) if r["status"] == "accepted"]
            truth = {r["id"]: r["structure"] for r in tests}
            dims = {r["id"]: r["desc"]["d"] for r in tests}
            flags = shared[f"order{o}_{dom}"]
            keys = [(r["id"], i) for r in tests for i in range(len(r["structure"]))]
            new = [k for k in keys if not flags[k[0]][k[1]]]
            old = [k for k in keys if flags[k[0]][k[1]]]
            new_of[(o, dom)] = set(new)
            print(f"== order {o}, {NAME[dom]} (position {pos}): {len(keys)} test equations, {len(new)} new-form, {len(old)} shared-form")
            arms = {"continued": f"o{o}_{dom}", "control": f"c{o}_{dom}", "fresh": f"fresh_{dom}"}
            R = {a: {t: load(f"{lab}_t{t}", dom) for t in POINTS} for a, lab in arms.items()}
            for a in arms:
                row = []
                for t in POINTS:
                    res = R[a][t]
                    if res is None:
                        row.append(f"{t}: missing")
                        continue
                    sys_ok = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
                    capped = sum(e["time_capped"] for r in res for e in r.get("equations", []))
                    row.append(f"{t}: {counts(res, keys)} ({counts(res, new)} new-form) eq., {sys_ok} sys."
                               + (f", {capped} capped" if capped else ""))
                print(f"   {a:>9}: " + "; ".join(row))
            f30 = R["fresh"][30]
            if f30 is not None and new:
                share = counts(f30, new) / len(new)
                print(f"   validity: the fresh network after 30 tasks solves {100 * share:.0f}% of new-form equations -> "
                      f"{'saturated from 30 on (reported only)' if share > 0.70 else 'valid'}")
            for t in POINTS:
                c, k = R["continued"][t], R["control"][t]
                if c is None or k is None:
                    continue
                for name, ks in (("new-form", new), ("all", keys)):
                    a_, b_ = paired(c, k, ks)
                    pts = 100.0 * (counts(c, ks) - counts(k, ks)) / max(len(ks), 1)
                    tag = ""
                    if t == 10 and name == "new-form":
                        verdict[(o, dom)] = bar_b(c, k, new)
                        tag = "  <- bar (B), this order: " + ("MET" if verdict[(o, dom)] else "NOT MET")
                    if t == 0 and name == "new-form":
                        tag = "  <- zero-shot, the second measure"
                    print(f"   continued vs control at {t:>3}, {name:>8}: {pts:+.1f} points; only continued {a_}, only control "
                          f"{b_}, p = {sign_p(a_, b_):.2g}{tag}")
                f = R["fresh"][t]
                if f is not None:
                    a_, b_ = paired(c, f, new)
                    print(f"   continued vs fresh   at {t:>3}, new-form: {100.0 * (counts(c, new) - counts(f, new)) / max(len(new), 1):+.1f} "
                          f"points; only continued {a_}, only fresh {b_}, p = {sign_p(a_, b_):.2g}")
            for a in arms:
                res = R[a][10]
                if res is not None:
                    print(f"   shortlist probe, {a} at 10 tasks: {probe(res)}")
            for a in arms:
                for t in (0, 10):
                    if R[a][t] is not None:
                        print(f"   effort, {a} at {t:>2} tasks: {effort(R[a][t], new)}")
            terms = prov.earlier_terms(f"o{o}", ORDERS[o][:pos - 1])
            for t in POINTS:
                c, k = R["continued"][t], R["control"][t]
                if c is not None and k is not None:
                    print(f"   provenance at {t:>3} tasks, new-form solved only by the continued network (vs the control): "
                          f"{prov.summary(only(c, k, new), truth, dims, terms)}")
            c, f = R["continued"][10], R["fresh"][10]
            if c is not None and f is not None:
                print(f"   provenance at  10 tasks, new-form solved only by the continued network (vs the fresh network): "
                      f"{prov.summary(only(c, f, new), truth, dims, terms)}")
            print(f"   provenance base rate, all new-form test equations: {prov.summary(new, truth, dims, terms)}")
        # retention
        for dom in ORDERS[o][:3]:
            b, a = load(f"o{o}_before", dom), load(f"o{o}_after", dom)
            if b and a:
                sb = {r["id"] for r in b if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
                sa = {r["id"] for r in a if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
                print(f"== retention, order {o}, {NAME[dom]}: systems {len(sb)} before the new domains, {len(sa)} after; "
                      f"lost {len(sb - sa)} -> {'MET' if len(sb - sa) <= 2 else 'NOT MET'}")
        d4 = ORDERS[o][3]
        b, a = load(f"o{o}_{d4}_t100", d4), load(f"o{o}_after", d4)
        if b and a:
            sb = {r["id"] for r in b if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            sa = {r["id"] for r in a if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            print(f"== retention, order {o}, {NAME[d4]} (after the 5th domain vs right after its own cycle): systems {len(sb)} -> "
                  f"{len(sa)}; lost {len(sb - sa)} -> {'MET' if len(sb - sa) <= 2 else 'NOT MET'}")
    if set(orders) >= {1, 2}:
        # (D): the continued network in order 1 against order 2, on the equations new-form in both orders
        for dom in ("chemistry", "chaotic"):
            both = sorted(new_of[(1, dom)] & new_of[(2, dom)])
            for t in POINTS:
                a, b = load(f"o1_{dom}_t{t}", dom), load(f"o2_{dom}_t{t}", dom)
                if a is None or b is None:
                    continue
                a_, b_ = paired(a, b, both)
                print(f"== (D) {NAME[dom]} at {t:>3} tasks, {len(both)} equations new-form in both orders: order 1 {counts(a, both)}, "
                      f"order 2 {counts(b, both)}; only order 1 {a_}, only order 2 {b_}, p = {sign_p(a_, b_):.2g}")
        both_b = [dom for dom in ("chemistry", "chaotic") if verdict.get((1, dom)) and verdict.get((2, dom))]
        if all((o, dom) in verdict for o in (1, 2) for dom in ("chemistry", "chaotic")):
            print(f"== bar (B) in both orders: " + "; ".join(f"{NAME[dom]} {'MET' if dom in both_b else 'NOT MET'}"
                                                           for dom in ("chemistry", "chaotic")))
            print(f"== the gate to Stage 2b ((B) met in at least one new domain, in both orders): "
                  f"{'OPEN' if both_b else 'NOT OPEN'}")


if __name__ == "__main__":
    main()
