"""The report of the six quick checks (k2_checks.py), with each pass rule as set before the run (research log "Owner:
run the six quick checks; their exact pass rules, set before any of them runs"). Usage: via k2_checks.py <w> report"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k2_checks as KC                                           # noqa: E402

K, C, OUT, NS = KC.K, KC.C, KC.OUT, KC.NS
CELLS = [(w, k) for w in K.DOMS for k in ("one short", "two short")]


def report():
    import numpy as np
    import grammar_g2 as GR
    import k1_report as KR
    lines, verdicts = [], {}
    data, seen, shape = {}, {}, {}
    for which in K.DOMS:
        nets = KC.networks(which)
        data[which] = json.loads((OUT / f"nodes_{which}.json").read_text())["rows"]
        seen[which] = {lin: KR.seen_shapes(which, arm, run) for arm, run, _, lin in nets}
    lins = {w: [lin for _, _, _, lin in KC.networks(w)] for w in K.DOMS}

    def unf(which, lin, key, eq):
        s = shape.setdefault((key, eq), GR.equation_shapes(4, eq, [key])[0])
        return s not in seen[which][lin]

    def rows(which, kind, lin):
        return [x for x in data[which] if x["node"] == kind and unf(which, lin, x["key"], x["eq"])]

    def over(which, kind, fn):
        """Median over the lineages that have missing unfamiliar terms in the cell of fn(their rows, lineage)."""
        vals = [fn(r, lin) for lin in lins[which] for r in [rows(which, kind, lin)] if r]
        return float(np.median(vals)) if vals else float("nan")
    # check 1
    lines.append("== Check 1 (build and check): share of missing unfamiliar terms in the screened top 10 (median over lineages); "
                 "baseline = the network's own top 10")
    best = None
    for src in ("net", "k0", "cl0"):
        for n in NS:
            ok = True
            parts = []
            for which, kind in CELLS:
                base = over(which, kind, lambda R, lin: np.mean([x[f"net:{lin}"] <= 10 for x in R]))
                f = (lambda x, lin: x[f"hit:cl0:{n}"]) if src == "cl0" else (lambda x, lin: x[f"hit:{src}:{n}:{lin}"])
                rate = over(which, kind, lambda R, lin: np.mean([f(x, lin) for x in R]))
                ok &= rate > base
                parts.append(f"{which} {kind} {100 * rate:.0f}% vs {100 * base:.0f}%")
            lines.append(f"   source {src}, top {n}: " + "; ".join(parts) + (" <- better in all four" if ok else ""))
            if ok and best is None:
                best = (src, n)
    verdicts[1] = best is not None
    lines.append(f"   -> check 1 {'PASSES' if best else 'FAILS'}" + (f" (first passing: source {best[0]}, top {best[1]})" if best else ""))
    # check 2(i)
    lines.append("== Check 2(i) (headroom): median rank of held-out unfamiliar terms, weights half the lineage's grammar and half the "
                 "system's other true terms (K0 grammar alone in brackets)")
    ok = True
    for which in K.DOMS:
        H = json.loads((OUT / f"headroom_{which}.json").read_text())
        hu = {lin: [x for x in H if unf(which, lin, x["key"], x["eq"])] for lin in lins[which]}
        m = np.median([np.median([x[f"head:{lin}"] for x in v]) for lin, v in hu.items() if v])
        m0 = np.median([np.median([x[f"k0:{lin}"] for x in v]) for lin, v in hu.items() if v])
        ok &= m <= 64
        lines.append(f"   {which}: {m:.0f} ({m0:.0f})")
    verdicts["2i"] = ok
    lines.append(f"   -> check 2(i) {'PASSES' if ok else 'FAILS'}")
    # check 2(ii)
    lines.append("== Check 2(ii) (choice-level data signal): median rank of missing unfamiliar terms: choice-level | evidence | network")
    ok = True
    for which, kind in CELLS:
        cl = over(which, kind, lambda R, lin: np.median([x["cl0"] for x in R]))
        ev = over(which, kind, lambda R, lin: np.median([x["ev"] for x in R]))
        nt = over(which, kind, lambda R, lin: np.median([x[f"net:{lin}"] for x in R]))
        ok &= cl < ev and cl < nt
        lines.append(f"   {which} {kind}: {cl:.0f} | {ev:.0f} | {nt:.0f}")
    verdicts["2ii"] = ok
    lines.append(f"   -> check 2(ii) {'PASSES' if ok else 'FAILS'}")
    # check 4
    lines.append("== Check 4 (C1, proposals across equations), one term short: share of missing terms found in another equation's "
                 "true structure; median rank with the other equations' found terms proposed first (all missing terms; unfamiliar)")
    for which in K.DOMS:
        R = [x for x in data[which] if x["node"] == "one short"]
        share = np.mean([x["other_eq"] for x in R])
        m = np.median([np.median([x[f"c1:{lin}"] for x in R]) for lin in lins[which]])
        mu = over(which, "one short", lambda R, lin: np.median([x[f"c1:{lin}"] for x in R]))
        mn = np.median([np.median([x[f"net:{lin}"] for x in R]) for lin in lins[which]])
        lines.append(f"   {which}: in another equation {100 * share:.0f}%; rank {m:.0f} (network alone {mn:.0f}); unfamiliar {mu:.0f}")
        if which == "reactions":
            verdicts[4] = share >= 0.5 and m <= 10
    lines.append(f"   -> check 4 {'PASSES' if verdicts[4] else 'FAILS'}")
    # check 5
    lines.append("== Check 5 (C2, paired moves): terms shared by two equations, missing from both: median one-equation rank | joint rank")
    for which in K.DOMS:
        P = json.loads((OUT / f"pairs_{which}.json").read_text())
        one = np.median([x["one_i"] for x in P] + [x["one_j"] for x in P]) if P else float("nan")
        jo = np.median([x["joint"] for x in P]) if P else float("nan")
        lines.append(f"   {which}: {len(P)} shared terms; {one:.0f} | {jo:.0f}")
        if which == "reactions":
            verdicts[5] = bool(P) and jo <= 64 and one > 64
    lines.append(f"   -> check 5 {'PASSES' if verdicts[5] else 'FAILS'}")
    # check 6
    T = json.loads((OUT / "constants_training.json").read_text())
    lines.append("== Check 6 (C3, shared constants): (i) training equations with a parameter also in another equation (any / as a "
                 "coefficient / inside a term):")
    for dom, v in T.items():
        lines.append(f"   {dom}: {100 * v['any']:.0f}% / {100 * v['coefficient']:.0f}% / {100 * v['inner']:.0f}% of {v['equations']}")
    part1 = max(v["any"] for v in T.values()) >= 0.25
    pr = [x for w in K.DOMS for x in json.loads((OUT / f"constants_{w}.json").read_text())]
    ex = [x for x in pr if max(x["separate"]) <= 1e-6]
    kept = [x for x in ex if all(t <= max(2 * s, 1e-9) for t, s in zip(x["tied"], x["separate"]))]
    lines.append(f"   (ii) development pairs with a shared coefficient: {len(pr)}; separate fits exact-level: {len(ex)}; tying keeps "
                 f"exactness in {len(kept)} ({100 * len(kept) / max(len(ex), 1):.0f}%)")
    verdicts[6] = part1 and bool(ex) and len(kept) / len(ex) >= 0.95
    lines.append(f"   -> check 6 {'PASSES' if verdicts[6] else 'FAILS'}")
    # check 3
    lines.append("== Check 3 (Stage C, library with data-set weights): median rank of missing unfamiliar terms: learned library | "
                 "no library | decoy")
    ok = True
    for which, kind in CELLS:
        lib = over(which, kind, lambda R, lin: np.median([x[f"lib:{lin}"] for x in R]))
        no = over(which, kind, lambda R, lin: np.median([x["cl0"] for x in R]))
        dec = over(which, kind, lambda R, lin: np.median([x[f"dec:{lin}"] for x in R]))
        ok &= lib < no and lib < dec
        lines.append(f"   {which} {kind}: {lib:.0f} | {no:.0f} | {dec:.0f}")
    verdicts[3] = ok
    lines.append(f"   -> check 3 {'PASSES' if ok else 'FAILS'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)
