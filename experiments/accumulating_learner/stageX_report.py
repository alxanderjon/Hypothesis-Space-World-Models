"""Stage X's report (research log "Stage X: exploration as a safety net, the pilot's design" and the owner's additions of
10 November). Development systems only; counts only. Per testbed, version (n, ns) and arm: new-form equations solved at
10 and 30 tasks against the f6 baseline; continued against control and fresh (and the f6 lineages' continued-fresh gap);
exploration counts (low-rated terms tried per equation, the deepest position in the simplest-first order, their size,
structures searched with even weights, solutions with a term the network rated below even at the root); kinds of term
found for the first time in each wake; the bump 1/(K + x^2) and the saturating x/(K + x^2) on flows (root ranks from
ranks_flows.json, and solutions containing them in each lineage's wakes); the bits check; time-limit hits; the time
split (for the quick-fit question); the winner under the pre-registered rule; the budget check if run."""

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("G2H_RUN", "stageX")
os.environ.setdefault("G2H_SHORTLIST", "2048")
import cycles_g2h as C                                           # noqa: E402
import g2h_report as R                                          # noqa: E402
import grammar_g2 as GR                                          # noqa: E402
import language_g1 as LG                                         # noqa: E402
import stageA_pilot as A                                         # noqa: E402
import stageX_pilot as X                                         # noqa: E402

PILOT = X.PILOT
F6 = {"flows": ("p6_o2", "p6fresh_2"), "reactions": ("q6_o1", "q6fresh_1")}
_KIND = {}


def kind(key, own, d=4):
    """'bump' for 1/(K + x^2), 'saturating' for x/(K + x^2) (in any variable), else None."""
    s = GR.equation_shapes(d, own, [key])[0]
    if s not in _KIND:
        shapes, _, _ = GR.tables()
        e = LG.expr_of(shapes[s])

        def is_bump(f):
            if not (f.is_Pow and f.exp == -1 and f.base.is_Add and len(f.base.args) == 2):
                return None
            a, b = f.base.args
            k, q = (a, b) if a.is_Symbol and str(a).startswith("K") else (b, a)
            if str(k).startswith("K") and q.is_Pow and q.exp == 2 and q.base.is_Symbol:
                return q.base
            return None
        out = None
        if is_bump(e) is not None:
            out = "bump"
        elif e.is_Mul and len(e.args) == 2:
            x_, f_ = (e.args[0], e.args[1]) if e.args[0].is_Symbol else (e.args[1], e.args[0])
            if x_.is_Symbol and is_bump(f_) == x_:
                out = "saturating"
        _KIND[s] = out
    return _KIND[s]


def ld(folder, lab, dom):
    q = folder / "test" / f"{lab}_{dom}.json"
    return json.loads(q.read_text()) if q.exists() else None


def wake_of(folder, lab):
    q = folder / "wake" / f"{lab}.json"
    return json.loads(q.read_text()) if q.exists() else None


def solutions(wake):
    return [(r["id"], e["eq"], e["solution"]["keys"]) for r in wake if r["status"] == "done" for e in r["equations"] if e["solution"]]


def history(lineage):
    """The continued lineage's history solutions [(d, own, keys, domain)] (Stage 2a's run)."""
    if not lineage.endswith(("_o1", "_o2")):
        return []
    rep = json.loads((C.REUSE / f"replays_{lineage[-2:]}.json").read_text())
    out = []
    for k, keys in rep.items():
        tid, eq = k.rsplit("|", 1)
        if C.dom_of(tid) in C.HISTORY:
            out.append((2, int(eq), keys, C.dom_of(tid)))
    return out


def report():
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    madeup = {}
    totals, eligible, notes, f6_totals = {}, {}, [], {}
    for which, setup in X.SETUPS.items():
        dom, o = setup["dom"], setup["o"]
        dev = C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        new = [(r["id"], i) for r in dev for i in range(len(r["structure"])) if not flags[r["id"]][i]]
        eqs = [(r["desc"]["d"], i, eq["keys"]) for r in dev for i, eq in enumerate(r["structure"]) if not flags[r["id"]][i]]
        f6c, f6f = F6[which]
        print(f"== {which} ({dom} after order {o}'s history; {len(new)} new-form development equations)")
        base = {("continued", k): ld(PILOT, f"{f6c}_t{k}dev", dom) for k in X.POINTS}
        base.update({("fresh", k): ld(PILOT, f"{f6f}_t{k}dev", dom) for k in X.POINTS})
        f6_totals[which] = sum(R.counts(base[(a, X.POINTS[-1])], new) for a in ("continued", "fresh")
                               if base[(a, X.POINTS[-1])] is not None)
        for k in X.POINTS:
            c, f = base[("continued", k)], base[("fresh", k)]
            if c is not None and f is not None:
                a_, b_ = R.paired(c, f, new)
                print(f"   f6 (no exploration): continued {R.counts(c, new)}, fresh {R.counts(f, new)} at {k} tasks; only continued {a_}, "
                      f"only fresh {b_}, p = {R.sign_p(a_, b_):.2g}")
        for v in ("n", "ns"):
            for tag in ("", "r2"):
                cont, ctrl, fresh = X.names(v, tag, o)
                res = {}
                for arm, lin in (("continued", cont), ("control", ctrl), ("fresh", fresh)):
                    for k in X.POINTS:
                        x_ = ld(C.RUN, f"{lin}_t{k}dev", dom)
                        res[(arm, k)] = x_
                        if x_ is None:
                            continue
                        b_lab = None if arm == "control" else (f6c if arm == "continued" else f6f)
                        if tag:                                  # the two-sided rerun's other side
                            b_lab = X.names("f6", tag, o)[("continued", "control", "fresh").index(arm)]
                        y_ = ld(C.RUN if tag else PILOT, f"{b_lab}_t{k}dev", dom) if b_lab else None
                        line = f"   version {v}{' (rerun)' if tag else ''}, {arm:>9} at {k:>2} tasks: {R.counts(x_, new)}"
                        if y_ is not None:
                            a_, b_ = R.paired(x_, y_, new)
                            p = R.sign_p(a_, b_)
                            line += f"; f6 {R.counts(y_, new)} (only version {a_}, only f6 {b_}, p = {p:.2g})"
                            if k == X.POINTS[-1] and arm in ("continued", "fresh") and not tag and b_ > a_:
                                if p < 0.01:
                                    eligible.setdefault(v, []).append(f"{which} {arm}: worse, p = {p:.2g}")
                                elif p <= 0.05:
                                    notes.append(f"version {v}, {which}, {arm}: worse at p = {p:.2g} -> two-sided rerun "
                                                 f"({which}:{v}:r2:{arm} and {which}:f6:r2:{arm})")
                        eq_ = [e for r in x_ if r["status"] == "done" for e in r["equations"]]
                        low = [e["low_tried"] for e in eq_ if e.get("low_tried") is not None]
                        dep = [e["low_depth"] for e in eq_ if e.get("low_depth") is not None]
                        mxn = [e["low_max_nodes"] for e in eq_ if e.get("low_max_nodes") is not None]
                        even = sum(e.get("even_structures") or 0 for e in eq_) / max(1, sum(e["structures"] for e in eq_))
                        solved_low = sum(1 for e in eq_ if e["first_exact"] is not None and (e.get("solution_low_terms") or 0) > 0)
                        capped = sum(e["time_capped"] for e in eq_)
                        line += (f"; low-rated terms tried per equation median {np.median(low) if low else 0:.0f} (max {max(low) if low else 0}), "
                                 f"deepest simplest-first position median {np.median(dep) if dep else 0:.0f} (max {max(dep) if dep else 0}), "
                                 f"largest such term {max(mxn) if mxn else 0} nodes; even-weights share of structures {100 * even:.0f}%; "
                                 f"solved with a term rated below even at the root {solved_low}; time-capped {capped}")
                        print(line)
                        if not tag and k == X.POINTS[-1]:
                            totals[(v, which, arm)] = R.counts(x_, new)
                for k in X.POINTS:
                    c, k_, f = res.get(("continued", k)), res.get(("control", k)), res.get(("fresh", k))
                    if c is not None and k_ is not None:
                        met, pts, a_, b_ = R.bar_b(c, k_, new)
                        print(f"   version {v}{' (rerun)' if tag else ''}: continued vs control at {k} tasks {pts:+.1f} points (only "
                              f"continued {a_}, only control {b_}, p = {R.sign_p(a_, b_):.2g})")
                    if c is not None and f is not None:
                        a_, b_ = R.paired(c, f, new)
                        print(f"   version {v}{' (rerun)' if tag else ''}: continued vs fresh at {k} tasks: continued {R.counts(c, new)}, "
                              f"fresh {R.counts(f, new)}; only continued {a_}, only fresh {b_}, p = {R.sign_p(a_, b_):.2g}")
        # discovery: kinds of term found for the first time in each wake; bump and saturating solutions (flows)
        lineages = [(f"f6 {a}", PILOT, lab) for a, lab in (("continued", f6c), ("fresh", f6f))]
        lineages += [(f"version {v} {a}", C.RUN, lab) for v in ("n", "ns") for a, lab in zip(("continued", "control", "fresh"), X.names(v, "", o))]
        for title, folder, lab in lineages:
            seen = {s for d, own, keys, _ in history(lab) for s in GR.equation_shapes(d, own, keys)}
            parts = []
            for k in X.POINTS:
                w = wake_of(folder, f"{lab}_w{k}")
                if w is None:
                    continue
                sol = solutions(w)
                shp = {s for _, i, keys in sol for s in GR.equation_shapes(4, i, keys)}
                first = shp - seen
                seen |= shp
                part = f"tasks to {k}: {len(sol)} solved, {len(first)} kinds found for the first time"
                if which == "flows":
                    nb = sum(any(kind(t, i) == "bump" for t in keys) for _, i, keys in sol)
                    ns_ = sum(any(kind(t, i) == "saturating" for t in keys) for _, i, keys in sol)
                    part += f", {nb} with a bump, {ns_} with a saturating term"
                parts.append(part)
            if parts:
                print(f"   discovery, {title}: " + " | ".join(parts))
        # the bits check
        rows = [("uniform prior", np.mean([GR.Grammar([]).bits(d, i, k) for d, i, k in eqs]))]
        for title, folder, lab in lineages:
            hist = history(lab)
            if lab.startswith("xc"):
                if o not in madeup:
                    madeup[o] = A.madeup_structures(C.cbase_files(o))
                earlier, groups = madeup[o], None
            else:
                earlier, groups = [(d, own, keys) for d, own, keys, _ in hist], [g for *_, g in hist] or None
            newest = []
            for k in X.POINTS:
                w = wake_of(folder, f"{lab}_w{k}")
                if w is None:
                    break
                newest += [(4, i, keys) for _, i, keys in solutions(w)]
                g = GR.Grammar.fit(earlier, newest, earlier_groups=groups)
                rows.append((f"{title}, {k} tasks", np.mean([g.bits(d, i, k_) for d, i, k_ in eqs])))
        print("   bits check (mean bits per true new-form equation): " + "; ".join(f"{t} {v_:.1f}" for t, v_ in rows))
        # time split (the quick-fit question)
        for v in ("n", "ns"):
            eq_ = [e for lin in X.names(v, "", o) for k in X.POINTS for r in (ld(C.RUN, f"{lin}_t{k}dev", dom) or [])
                   if r["status"] == "done" for e in r["equations"]]
            if eq_:
                tot = lambda f: sum(e.get(f) or 0.0 for e in eq_)
                s = tot("seconds")
                print(f"   time split, version {v} tests: per equation {s / len(eq_):.0f}s; ordinary fits {100 * tot('fit_seconds') / s:.0f}%, "
                      f"thorough fits {100 * tot('thorough_seconds') / s:.0f}%, precise checks {100 * tot('precise_seconds') / s:.0f}%, "
                      f"prior (network and evidence) {100 * tot('prior_seconds') / s:.0f}%; structures per equation "
                      f"{tot('structures') / len(eq_):.0f}, thorough fits per equation {tot('thorough') / len(eq_):.1f}")
    # the bump's root ranks (flows)
    rk = C.RUN / "ranks_flows.json"
    if rk.exists():
        res = json.loads(rk.read_text())
        by = {}
        for r in res:
            for key, rank in zip(r["true_keys"], r["true_ranks"]):
                kd = kind(key, r["eq"])
                if kd:
                    by.setdefault((r["net"], kd), []).append(rank)
        print("== root ranks on the development flows systems (median among about 4,000 terms; the diagnosis: fresh f6 17, "
              "earlier-trained 670-2,100 for the bump)")
        for net in sorted({r["net"] for r in res}):
            print(f"   {net}: bump {np.median(by.get((net, 'bump'), [np.nan])):.0f}, saturating {np.median(by.get((net, 'saturating'), [np.nan])):.0f}")
    # the winner
    print("== the winner (pre-registered rule)")
    for v in ("n", "ns"):
        tot = sum(totals.get((v, w, a), 0) for w in X.SETUPS for a in ("continued", "control", "fresh"))
        tot_cf = sum(totals.get((v, w, a), 0) for w in X.SETUPS for a in ("continued", "fresh"))
        f6_cf = sum(f6_totals.values())
        print(f"   version {v}: new-form solved at {X.POINTS[-1]} tasks, three arms and both testbeds {tot}; continued and fresh "
              f"{tot_cf} against f6's {f6_cf}; {'ineligible: ' + '; '.join(eligible[v]) if v in eligible else 'eligible'}")
    for n_ in notes:
        print(f"   pending: {n_}")
    # the budget check
    for v in ("n", "ns"):
        setup = X.SETUPS["flows"]
        dom, o = setup["dom"], setup["o"]
        flags = shared[f"mce_dev_{dom}"]
        new = [(r["id"], i) for r in C.records("dev", dom) for i in range(len(r["structure"])) if not flags[r["id"]][i]]
        for lin in (X.names(v, "", o)[0], X.names(v, "", o)[2]):
            b = ld(C.RUN, f"{lin}_t{X.POINTS[-1]}dev_b{X.BUDGET_CHECK[0]}", dom)
            a = ld(C.RUN, f"{lin}_t{X.POINTS[-1]}dev", dom)
            if b is None or a is None:
                continue
            x_, y_ = R.paired(b, a, new)
            nb = lambda res_: sum(1 for r in res_ if r["status"] == "done" for e in r["equations"] if e["solution"] and
                                  any(kind(t, e["eq"]) == "bump" for t in e["solution"]["keys"]))
            low = [e["low_tried"] for r in b if r["status"] == "done" for e in r["equations"]]
            eq_ = [e for r in b if r["status"] == "done" for e in r["equations"]]
            s = sum(e["seconds"] for e in eq_)
            print(f"== budget check, version {v}, {lin}: {X.BUDGET_CHECK[0]} structures {R.counts(b, new)}, 300 {R.counts(a, new)} new-form "
                  f"(only {X.BUDGET_CHECK[0]} {x_}, only 300 {y_}, p = {R.sign_p(x_, y_):.2g}); solutions with a bump {nb(b)} vs {nb(a)}; "
                  f"low-rated terms tried per equation median {np.median(low):.0f}; time per equation {s / max(1, len(eq_)):.0f}s, of which "
                  f"thorough fits {100 * sum(e.get('thorough_seconds') or 0 for e in eq_) / s:.0f}%; time-capped "
                  f"{sum(e['time_capped'] for e in eq_)}")


if __name__ == "__main__":
    report()
