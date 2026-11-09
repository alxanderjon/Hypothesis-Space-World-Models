"""Stage 2a'' of the general learner: the report (research log, Stage 2a'' proposal, section 6; Stage 2a's bars).
Counts only. Per order and harder domain: learning curves of the continued network, the rehearsal-matched control and
the fresh network (tests before any task and after 10, 30 and 100 tasks), new-form and shared-form equations; validity
(the fresh network at most 70% of new-form test equations after 30 tasks); bar (B) at 10 tasks (continued vs control,
new-form, +10 points and a paired sign test p < 0.05), zero-shot as the second measure, (C) against the fresh network,
retention (E), whole systems exact, effort, provenance, time-limit hits. The shortlist's cut rate for each arm, domain,
order and point (owner, 8 November), and the material for judging whether shortlist cuts could explain bar (B):
(B) on the equations where neither arm ever had a true term cut, the discordant equations' cut shares, and within each
arm the solved share of equations with and without cuts. With both orders: (D) and the gate to Stage 2b.
Usage: g2h_report.py <level> [order ...]"""

import itertools
import json
import os
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYS = ROOT / "outputs/accumulating_learner/g1/systems"
RUN = ROOT / "outputs/accumulating_learner/g2h" / os.environ.get("G2H_RUN", "run1")
TEST_SET = os.environ.get("G2H_TEST_SET", "test")
POINTS = (0, 10, 30, 100)
HISTORY = ("mechanics", "circuits", "ecology")


def orders(level):
    R, F = f"reactions_L{level}", f"flows_L{level}"
    return {1: ("mechanics", "circuits", "ecology", R, F), 2: ("ecology", "circuits", "mechanics", F, R)}


def name(dom):
    return {"reactions": "reaction networks", "flows": "nonlinear flows"}.get(dom.split("_")[0], dom)


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


def bar_b(c, k, keys):
    a_, b_ = paired(c, k, keys)
    pts = 100.0 * (counts(c, keys) - counts(k, keys)) / max(len(keys), 1)
    return pts >= 10 and a_ > b_ and sign_p(a_, b_) < 0.05, pts, a_, b_


def systems_solved(res):
    return sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))


def probe_text(res):
    p = [r["probe"] for r in res if r.get("probe")]
    if not p:
        return "no network (uniform)"
    w, c = sum(x["with_missing_true"] for x in p), sum(x["cut"] for x in p)
    rw, rc = sum(x["root_with_missing_true"] for x in p), sum(x["root_cut"] for x in p)
    return f"{100.0 * c / max(w, 1):.1f}% ({c} of {w} expansions with a missing true term; root {rc} of {rw})"


def eq_probe(res):
    """(system id, eq) -> (expansions with a missing true term, cuts) from the per-equation probe."""
    out = {}
    for r in res:
        for i, v in (r.get("probe_eq") or {}).items():
            out[(r["id"], int(i))] = (v["with_missing_true"], v["cut"])
    return out


def effort(res, keys):
    first = {(r["id"], e["eq"]): (e["first_exact"] if r["status"] == "done" and e["first_exact"] is not None else math.inf)
             for r in res for e in r.get("equations", [])}
    f = sorted(first.get(k, math.inf) for k in keys)
    med = f[len(f) // 2] if f else math.inf
    eqs = [e for r in res for e in r.get("equations", [])]
    return (f"median structures to the first plausible model (new-form) {med if math.isfinite(med) else 'not reached'}; "
            f"structures {sum(e['structures'] for e in eqs)}, thorough fits {sum(e['thorough'] for e in eqs)}, "
            f"CPU {sum(r['seconds'] for r in res) / 3600:.2f} h")


class Provenance:
    """Earlier-domain verified solutions of the lineage sharing a true term with an equation; terms compared relative
    to their equation (own variable first, the others in any order) as four-variable vocabulary representatives."""

    def __init__(self, level):
        sys.path.insert(0, str(HERE))
        import domains_g1 as DG
        self.DG, self.cache = DG, {}
        self.dim = {r["id"]: r["desc"]["d"] for dom in HISTORY + tuple(orders(level)[1][3:])
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
                cands = []
                for p in itertools.permutations((1, 2, 3)):
                    x = DG.relabelled(r, [0] + list(p))
                    if x is not None:
                        cands.append(DG.representative(x, 4))
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


def shortlist_assessment(c, k, new):
    """Could shortlist cuts explain the continued network's result against the control at this point?"""
    pc, pk = eq_probe(c), eq_probe(k)
    sc, sk = solved_map(c), solved_map(k)
    lines = []
    clean = [q for q in new if pc.get(q, (0, 0))[1] == 0 and pk.get(q, (0, 0))[1] == 0]
    met, pts, a_, b_ = bar_b(c, k, clean)
    lines.append(f"(B) on the {len(clean)} new-form equations where neither arm ever had a true term cut: {pts:+.1f} points; "
                 f"only continued {a_}, only control {b_}, p = {sign_p(a_, b_):.2g}")

    def share(p, qs):
        w = sum(p.get(q, (0, 0))[0] for q in qs)
        return f"{100.0 * sum(p.get(q, (0, 0))[1] for q in qs) / max(w, 1):.1f}% of {w}"
    oc = [q for q in new if sc.get(q, False) and not sk.get(q, False)]
    ok = [q for q in new if sk.get(q, False) and not sc.get(q, False)]
    lines.append(f"cut share on equations solved only by the continued network ({len(oc)}): continued {share(pc, oc)}, "
                 f"control {share(pk, oc)}; only by the control ({len(ok)}): continued {share(pc, ok)}, control {share(pk, ok)}")
    for arm, p, s in (("continued", pc, sc), ("control", pk, sk)):
        cut = [q for q in new if p.get(q, (0, 0))[1] > 0]
        free = [q for q in new if p.get(q, (0, 0))[1] == 0]
        lines.append(f"{arm}: solved {sum(s.get(q, False) for q in free)} of {len(free)} new-form equations never cut, "
                     f"{sum(s.get(q, False) for q in cut)} of {len(cut)} with at least one cut")
    return lines


def main():
    level = int(sys.argv[1])
    os_ = [int(x) for x in sys.argv[2:]] or [1, 2]
    O = orders(level)
    shared = json.loads((SYS / (f"shared_forms_g2h_L{level}.json" if TEST_SET == "test" else f"shared_forms_g2h_L{level}_{TEST_SET}.json")).read_text())
    sysx = {}
    prov = Provenance(level)
    verdict, new_of = {}, {}
    for o in os_:
        for pos, dom in ((4, O[o][3]), (5, O[o][4])):
            tests = [r for r in json.loads((SYS / f"{TEST_SET}_{dom}.json").read_text()) if r["status"] == "accepted"]
            truth = {r["id"]: r["structure"] for r in tests}
            dims = {r["id"]: r["desc"]["d"] for r in tests}
            flags = shared[f"order{o}_{dom}"]
            keys = [(r["id"], i) for r in tests for i in range(len(r["structure"]))]
            new = [q for q in keys if not flags[q[0]][q[1]]]
            new_of[(o, dom)] = set(new)
            print(f"== order {o}, {name(dom)} (position {pos}): {len(keys)} test equations, {len(new)} new-form, "
                  f"{len(keys) - len(new)} shared-form")
            arms = {"continued": f"o{o}_{dom}", "control": f"c{o}_{dom}", "fresh": f"fresh_{dom}"}
            R = {a: {t: load(f"{lab}_t{t}", dom) for t in POINTS} for a, lab in arms.items()}
            for a in arms:
                row = []
                for t in POINTS:
                    res = R[a][t]
                    if res is None:
                        row.append(f"{t}: missing")
                        continue
                    capped = sum(e["time_capped"] for r in res for e in r.get("equations", []))
                    failed = sum(r["status"] != "done" for r in res)
                    row.append(f"{t}: {counts(res, keys)} ({counts(res, new)} new-form) eq., {systems_solved(res)} sys."
                               + (f", {capped} capped" if capped else "") + (f", {failed} failed" if failed else ""))
                print(f"   {a:>9}: " + "; ".join(row))
            f30 = R["fresh"][30]
            if f30 is not None and new:
                share = counts(f30, new) / len(new)
                print(f"   validity: the fresh network after 30 tasks solves {100 * share:.0f}% of new-form test equations -> "
                      f"{'saturated from 30 on (reported only)' if share > 0.70 else 'valid'}")
            for t in POINTS:
                c, k = R["continued"][t], R["control"][t]
                if c is None or k is None:
                    continue
                for nm, ks in (("new-form", new), ("all", keys)):
                    met, pts, a_, b_ = bar_b(c, k, ks)
                    tag = ""
                    if t == 10 and nm == "new-form":
                        verdict[(o, dom)] = met
                        tag = "  <- bar (B), this order: " + ("MET" if met else "NOT MET")
                    if t == 0 and nm == "new-form":
                        tag = "  <- zero-shot, the second measure"
                    print(f"   continued vs control at {t:>3}, {nm:>8}: {pts:+.1f} points; only continued {a_}, only control "
                          f"{b_}, p = {sign_p(a_, b_):.2g}{tag}")
                f = R["fresh"][t]
                if f is not None:
                    a_, b_ = paired(c, f, new)
                    print(f"   continued vs fresh   at {t:>3}, new-form: {100.0 * (counts(c, new) - counts(f, new)) / max(len(new), 1):+.1f} "
                          f"points; only continued {a_}, only fresh {b_}, p = {sign_p(a_, b_):.2g}")
            print("   shortlist cut rate (share of expansions with a missing true term where one was outside the attended set):")
            for a in arms:
                print(f"     {a:>9}: " + "; ".join(f"{t}: {probe_text(R[a][t])}" for t in POINTS if R[a][t] is not None))
            c, k = R["continued"][10], R["control"][10]
            if c is not None and k is not None:
                print("   shortlist assessment at 10 tasks:")
                for line in shortlist_assessment(c, k, new):
                    print(f"     {line}")
            for a in arms:
                for t in (0, 10):
                    if R[a][t] is not None:
                        print(f"   effort, {a} at {t:>2} tasks: {effort(R[a][t], new)}")
                        sp = RUN / "test" / f"{arms[a]}_t{t}_{dom}_systems.json"
                        if sp.exists():
                            sysx[(o, dom, a, t)] = sum(x.get("exact", False) for x in json.loads(sp.read_text()))
            if sysx:
                print("   whole systems exact (one-step <= 1e-12 and the multi-step envelope): " + "; ".join(
                    f"{a} at {t}: {sysx[(o, dom, a, t)]}" for a in arms for t in (0, 10) if (o, dom, a, t) in sysx))
            terms = prov.earlier_terms(f"o{o}", O[o][:pos - 1])
            for t in POINTS:
                c, k = R["continued"][t], R["control"][t]
                if c is not None and k is not None:
                    print(f"   provenance at {t:>3} tasks, new-form solved only by the continued network (vs the control): "
                          f"{prov.summary(only(c, k, new), truth, dims, terms)}")
            print(f"   provenance base rate, all new-form test equations: {prov.summary(new, truth, dims, terms)}")
        for dom in O[o][:3]:
            b, a = load(f"o{o}_before", dom), load(f"o{o}_after", dom)
            if b and a:
                sb = {r["id"] for r in b if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
                sa = {r["id"] for r in a if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
                print(f"== retention, order {o}, {dom}: systems {len(sb)} before the new domains, {len(sa)} after; lost "
                      f"{len(sb - sa)} -> {'MET' if len(sb - sa) <= 2 else 'NOT MET'}")
        d4 = O[o][3]
        b, a = load(f"o{o}_{d4}_t100", d4), load(f"o{o}_after", d4)
        if b and a:
            sb = {r["id"] for r in b if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            sa = {r["id"] for r in a if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            print(f"== retention, order {o}, {name(d4)} (after the 5th domain vs right after its own cycle): systems "
                  f"{len(sb)} -> {len(sa)}; lost {len(sb - sa)} -> {'MET' if len(sb - sa) <= 2 else 'NOT MET'}")
    if set(os_) >= {1, 2}:
        for dom in (O[1][3], O[1][4]):
            both = sorted(new_of[(1, dom)] & new_of[(2, dom)])
            for t in POINTS:
                a, b = load(f"o1_{dom}_t{t}", dom), load(f"o2_{dom}_t{t}", dom)
                if a is None or b is None:
                    continue
                a_, b_ = paired(a, b, both)
                print(f"== (D) {name(dom)} at {t:>3} tasks, {len(both)} equations new-form in both orders: order 1 "
                      f"{counts(a, both)}, order 2 {counts(b, both)}; only order 1 {a_}, only order 2 {b_}, p = {sign_p(a_, b_):.2g}")
        doms = (O[1][3], O[1][4])
        if all((o, dom) in verdict for o in (1, 2) for dom in doms):
            both_b = [dom for dom in doms if verdict[(1, dom)] and verdict[(2, dom)]]
            print("== bar (B) in both orders: " + "; ".join(f"{name(dom)} {'MET' if dom in both_b else 'NOT MET'}" for dom in doms))
            print(f"== the gate to Stage 2b ((B) met in at least one harder domain, in both orders): "
                  f"{'OPEN' if both_b else 'NOT OPEN'}")


if __name__ == "__main__":
    main()
