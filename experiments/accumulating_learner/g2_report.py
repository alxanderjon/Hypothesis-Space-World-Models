"""Stage 2a of the general learner: the report (research log, Stage 2a rules, section 7). Counts only.
Per order and new domain: the learning curves of the continued network, the made-up-systems control and the fresh
network (tests before any task and after 10, 30 and 100 tasks), split into new-form and shared-form equations (claims of
accumulation rest on new forms and must beat the control); the validity check (the fresh network at most 70% of new-form
equations after 30 tasks); bar (B) at 10 tasks (continued vs control, new-form, +10 points and a paired sign test
p < 0.05), zero-shot as the second measure, (C) against the fresh network, retention (E) per system, the shortlist probe,
time-limit hits. Usage: g2_report.py [order ...]"""

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


def probe(res):
    p = [r["probe"] for r in res if r.get("probe")]
    if not p:
        return "no network"
    w, c = sum(x["with_missing_true"] for x in p), sum(x["cut"] for x in p)
    rw, rc = sum(x["root_with_missing_true"] for x in p), sum(x["root_cut"] for x in p)
    return f"cut at {c} of {w} expansions with a missing true term ({100.0 * c / max(w, 1):.1f}%); at the root {rc} of {rw}"


def main():
    orders = [int(x) for x in sys.argv[1:]] or [1, 2]
    shared = json.loads((SYS / "shared_forms_g2.json").read_text())
    for o in orders:
        for pos, dom in ((4, ORDERS[o][3]), (5, ORDERS[o][4])):
            tests = [r for r in json.loads((SYS / f"test_{dom}.json").read_text()) if r["status"] == "accepted"]
            flags = shared[f"order{o}_{dom}"]
            keys = [(r["id"], i) for r in tests for i in range(len(r["structure"]))]
            new = [k for k in keys if not flags[k[0]][k[1]]]
            old = [k for k in keys if flags[k[0]][k[1]]]
            print(f"== order {o}, {NAME[dom]} (position {pos}): {len(keys)} test equations, {len(new)} new-form, {len(old)} shared-form")
            arms = {"continued": f"o{o}_{dom}", "control": f"c{o}_{dom}", "fresh": f"fresh_{dom}"}
            R = {a: {t: load(f"{lab}_t{t}", dom) for t in (0, 10, 30, 100)} for a, lab in arms.items()}
            for a in arms:
                row = []
                for t in (0, 10, 30, 100):
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
            for t in (0, 10, 30, 100):
                c, k = R["continued"][t], R["control"][t]
                if c is None or k is None:
                    continue
                for name, ks in (("new-form", new), ("all", keys)):
                    a_, b_ = paired(c, k, ks)
                    pts = 100.0 * (counts(c, ks) - counts(k, ks)) / max(len(ks), 1)
                    tag = ""
                    if t == 10 and name == "new-form":
                        tag = "  <- bar (B), this order: " + ("MET" if pts >= 10 and a_ > b_ and sign_p(a_, b_) < 0.05 else "NOT MET")
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


if __name__ == "__main__":
    main()
