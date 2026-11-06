"""Stage 1 of the general learner: the report (research log, Stage 1 rules, section 10, and the owner's notes). Counts only.
  (c) after mechanics, guided vs uniform on mechanics' tests: +20 points of equations solved, or at most half the median
      structures to the first plausible model;
  (d) forward transfer on circuits and ecology: continued vs fresh network after the first cycle (+10 points and a paired
      sign test p < 0.05), and after the second cycle (the fresh network not significantly better); split into new-form
      and shared-form equations (claims of accumulation rest on new forms);
  (e) retention after ecology, per system: at most 2 systems lost per earlier domain (a system counts as solved when all
      its equations are);
  also: uniform baselines, zero-shot transfer, true structure among solved, provenance, time-limit hits per arm.
Usage: g1_report.py"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
G1 = HERE.parents[1] / "outputs/accumulating_learner/g1"
RUN, SYS = G1 / "run1", G1 / "systems"
NAME = {"A": "mechanics", "B": "circuits", "C": "ecology"}


def load(label, L):
    p = RUN / "test" / f"{label}_{L}.json"
    return json.loads(p.read_text()) if p.exists() else None


def truth(L):
    return {r["id"]: r["structure"] for r in json.loads((SYS / f"test_{NAME[L]}.json").read_text()) if r["status"] == "accepted"}


def eq_table(res):
    """(system id, eq) -> (solved, first exact or inf, solution keys)."""
    out = {}
    for r in res:
        for e in r.get("equations", []):
            out[(r["id"], e["eq"])] = (e["first_exact"] is not None and r["status"] == "done",
                                       e["first_exact"] if e["first_exact"] is not None else math.inf,
                                       sorted(e["solution"]["keys"]) if e.get("solution") else None)
    return out


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(a, b) + 1)) / 2 ** n) if n else 1.0


def describe(label, L, res, T, keys=None):
    t = eq_table(res)
    ks = keys if keys is not None else list(t)
    solved = sum(t[k][0] for k in ks if k in t)
    true = sum(1 for k in ks if k in t and t[k][0] and t[k][2] == sorted(T[k[0]][k[1]]["keys"]))
    firsts = sorted(t[k][1] for k in ks if k in t)
    med = firsts[len(firsts) // 2] if firsts else math.inf
    secs = sorted(r["seconds"] for r in res)
    capped = sum(e["time_capped"] for r in res for e in r.get("equations", []))
    sys_solved = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
    return {"label": label, "domain": NAME[L], "equations": len(ks), "solved": solved, "true_structure": true,
            "median_first": med, "systems_solved": sys_solved, "systems": len(res), "time_median": secs[len(secs) // 2],
            "time_max": secs[-1], "time_capped": capped, "failed": sum(r["status"] != "done" for r in res)}


def paired(a, b, keys):
    """Discordant counts: solved by a only, by b only."""
    ta, tb = eq_table(a), eq_table(b)
    ao = sum(1 for k in keys if ta.get(k, (False,))[0] and not tb.get(k, (False,))[0])
    bo = sum(1 for k in keys if tb.get(k, (False,))[0] and not ta.get(k, (False,))[0])
    return ao, bo


def fmt(d):
    m = d["median_first"]
    return (f"{d['label']} on {d['domain']}: solved {d['solved']} of {d['equations']} equations (true structure "
            f"{d['true_structure']}); systems {d['systems_solved']} of {d['systems']}; median structures to the first "
            f"plausible model {m if math.isfinite(m) else 'not reached'}; time per system median {d['time_median']:.0f}s, max "
            f"{d['time_max']:.0f}s; time-capped equations {d['time_capped']}; failed systems {d['failed']}")


def main():
    shared = json.loads((SYS / "shared_forms.json").read_text())
    print("== evaluations")
    for L, labels in (("A", ["uniform", "cont_A2", "cont_C2"]), ("B", ["uniform", "cont_A2", "cont_B1", "cont_B2", "freshB_B1",
                                                                      "freshB_B2", "cont_C2"]),
                      ("C", ["uniform", "cont_B2", "cont_C1", "cont_C2", "freshC_C1", "freshC_C2"])):
        T = truth(L)
        for lab in labels:
            res = load(lab, L)
            if res is not None:
                print("  " + fmt(describe(lab, L, res, T)))
    # (c)
    T = truth("A")
    u, g = load("uniform", "A"), load("cont_A2", "A")
    if u and g:
        du, dg = describe("uniform", "A", u, T), describe("cont_A2", "A", g, T)
        pts = 100.0 * (dg["solved"] - du["solved"]) / du["equations"]
        met = pts >= 20 or (math.isfinite(dg["median_first"]) and dg["median_first"] <= 0.5 * du["median_first"])
        ao, bo = paired(g, u, [k for k in eq_table(u)])
        print(f"== bar (c): {'MET' if met else 'NOT MET'}: {pts:+.1f} points; median first plausible model "
              f"{dg['median_first']} vs {du['median_first']}; solved only guided {ao}, only uniform {bo} (p = {sign_p(ao, bo):.2g})")
    # (d)
    for L, fr in (("B", "freshB"), ("C", "freshC")):
        T = truth(L)
        keys = [(sid, i) for sid, st in T.items() for i in range(len(st))]
        novel = [k for k in keys if not (shared.get(k[0]) or [False] * 2)[k[1]]]
        sh = [k for k in keys if k not in set(novel)]
        for cyc in (1, 2):
            c, f = load(f"cont_{L}{cyc}", L), load(f"{fr}_{L}{cyc}", L)
            if not (c and f):
                continue
            lines = []
            for name, ks in (("all", keys), ("new-form", novel), ("shared-form", sh)):
                if not ks:
                    continue
                tc, tf = eq_table(c), eq_table(f)
                sc, sf = sum(tc.get(k, (False,))[0] for k in ks), sum(tf.get(k, (False,))[0] for k in ks)
                ao, bo = paired(c, f, ks)
                lines.append(f"{name} {len(ks)}: continued {sc}, fresh {sf} ({100.0 * (sc - sf) / len(ks):+.1f} points; "
                             f"only continued {ao}, only fresh {bo}, p = {sign_p(ao, bo):.2g})")
            tc, tf = eq_table(c), eq_table(f)
            sc, sf = sum(tc.get(k, (False,))[0] for k in keys), sum(tf.get(k, (False,))[0] for k in keys)
            ao, bo = paired(c, f, keys)
            pts = 100.0 * (sc - sf) / len(keys)
            if cyc == 1:
                verdict = "MET" if pts >= 10 and ao > bo and sign_p(ao, bo) < 0.05 else "NOT MET"
            else:
                verdict = "MET" if not (bo > ao and sign_p(ao, bo) < 0.05) else "NOT MET"
            print(f"== bar (d) {NAME[L]} after cycle {cyc}: {verdict}; " + "; ".join(lines))
            rp = RUN / "replays_cont.json"
            if rp.exists():                                    # provenance of every equation solved only by the continued
                earlier = ("mechanics",) if L == "B" else ("mechanics", "circuits")   # network: earlier-domain verified
                used = set()                                   # solutions sharing any of its true terms
                for k, v in json.loads(rp.read_text()).items():
                    if any(f"-{d}-" in k for d in earlier):
                        used |= set(v)
                only = [k for k in keys if tc.get(k, (False,))[0] and not tf.get(k, (False,))[0]]
                with_shared_term = sum(1 for k in only if set(T[k[0]][k[1]]["keys"]) & used)
                print(f"   provenance: {len(only)} equations solved only by the continued network; {with_shared_term} of "
                      f"them have a true term that appears in an earlier domain's verified solutions; "
                      f"{sum(1 for k in only if k in set(novel))} of them are new-form")
        z, u = load("cont_A2" if L == "B" else "cont_B2", L), load("uniform", L)
        if z and u:
            ao, bo = paired(z, u, keys)
            print(f"   zero-shot on {NAME[L]}: earlier-domain network vs uniform: only network {ao}, only uniform {bo} "
                  f"(p = {sign_p(ao, bo):.2g})")
    # (e)
    for L, before in (("A", "cont_A2"), ("B", "cont_B2")):
        b, a = load(before, L), load("cont_C2", L)
        if b and a:
            sb = {r["id"] for r in b if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            sa = {r["id"] for r in a if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"])}
            lost, gained = len(sb - sa), len(sa - sb)
            print(f"== bar (e) {NAME[L]}: {'MET' if lost <= 2 else 'NOT MET'}: systems solved right after the domain "
                  f"{len(sb)}, after ecology {len(sa)}; lost {lost}, gained {gained}")


if __name__ == "__main__":
    main()
