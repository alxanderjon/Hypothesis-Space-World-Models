"""K1's same-network test: the report (research log "K1's same-network test: launched"). Per network and ordering:
equations solved (new-form, shared-form, all; true structure; whole systems), the bump found, time; each new ordering
against A on the same network (per-equation counts within the network); the A-versus-A noise from the f6 networks'
earlier development tests; and the winner rule set before the run. Works on partial results (missing tests are listed).
Usage: k1_report.py"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "k1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1_check as K                                             # noqa: E402

C = K.C


def binom_p(a, b):
    """Two-sided exact sign test of a against b discordant pairs."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def outcome(res, recs, flags):
    """{(id, eq): (solved, true structure)} and per-test totals."""
    truth = {r["id"]: r["structure"] for r in recs}
    eqs, out = {}, {"failed": 0, "time_capped": 0, "systems": 0, "seconds": []}
    for r in res:
        if r["status"] != "done":
            out["failed"] += 1
            continue
        out["seconds"].append(r["seconds"])
        ok = True
        for e in r["equations"]:
            s = e["first_exact"] is not None
            t = bool(e["solution"]) and sorted(e["solution"]["keys"]) == sorted(truth[r["id"]][e["eq"]]["keys"])
            eqs[(r["id"], e["eq"])] = (s, t, e["solution"]["keys"] if e["solution"] else None)
            out["time_capped"] += bool(e["time_capped"])
            ok &= s
        out["systems"] += ok
    new = [k for k in eqs if not flags[k[0]][k[1]]]
    shared = [k for k in eqs if flags[k[0]][k[1]]]
    out.update(new=sum(eqs[k][0] for k in new), shared=sum(eqs[k][0] for k in shared), all=sum(v[0] for v in eqs.values()),
               true=sum(v[1] for v in eqs.values()), n_new=len(new), n_shared=len(shared))
    return eqs, out


def seen_shapes(which, arm, run):
    """Kinds of term among the lineage's experience at 30 tasks (k0b_check's lineage point)."""
    sys.argv = sys.argv[:1]                                      # k0b_check reads its own arguments
    import grammar_g2 as GR
    import k0b_check as KB
    pt = next(p for p in KB.points(which) if p["arm"] == arm and p["run"] == run and p["k"] == K.K)
    e, g, n = KB.experience(pt, KB.SETUPS[which]["o"])
    return {s_ for d, own, keys in e + n for s_ in GR.equation_shapes(d, own, keys)}


def report():
    import numpy as np
    import grammar_g2 as GR
    import stageX_report as XR
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    lines, rows, missing = [], [], []
    for which, arms in K.NETS.items():
        dom = K.DOMS[which]
        recs = C.records("dev", dom)
        flags = {rid: v for rid, v in shared[f"mce_dev_{dom}"].items()}
        for arm, nets in arms.items():
            for run, folder, lin, seed in nets:
                got = {}
                for o in K.ORDERINGS:
                    p = K.OUT / f"{lin}_{o}_s{K.SEED}_{dom}.json"
                    if p.exists():
                        got[o] = outcome(json.loads(p.read_text()), recs, flags)
                    else:
                        missing.append(f"{which} {arm} {run} {o}")
                old = folder / "test" / f"{lin}_t{K.K}dev_{dom}.json"
                orig = outcome(json.loads(old.read_text()), recs, flags) if seed is not None and old.exists() else None
                bump = {o: sum(1 for (rid, i), (s, t, keys) in g[0].items() if s and keys and
                               any(XR.kind(k, i) == "bump" for k in keys)) for o, g in got.items()}
                seen = seen_shapes(which, arm, run)
                unf = {o: sum(1 for (rid, i), (s, t, keys) in g[0].items() if s and keys and
                              any(GR.equation_shapes(4, i, [k])[0] not in seen for k in keys)) for o, g in got.items()}
                rows.append({"which": which, "arm": arm, "run": run, "lin": lin, "got": got, "orig": orig, "orig_seed": seed,
                             "bump": bump, "unfamiliar": unf})
    lines.append(f"== K1 same-network test: {sum(len(r['got']) for r in rows)} of {len(rows) * len(K.ORDERINGS)} tests done"
                 + (f"; missing: {', '.join(missing)}" if missing else ""))
    lines.append("   Per network (development systems; new-form of 159 on flows, 155 on reaction networks): ordering: new-form / "
                 "shared-form / all solved, true structure, whole systems, bump solutions, solutions using a kind of term new to the "
                 "lineage, time-capped, median seconds per system; "
                 "then each new ordering against A on new-form equations (solved only with it / only with A, sign test)")
    for r in rows:
        head = f"     {r['which']:>9} {r['arm']:>9} {r['run']:>9}:"
        parts = []
        for o, (eqs, t) in r["got"].items():
            parts.append(f"{o} {t['new']}/{t['shared']}/{t['all']}, true {t['true']}, sys {t['systems']}, bump {r['bump'][o]}, "
                         f"new kind {r['unfamiliar'][o]}, capped "
                         f"{t['time_capped']}, {np.median(t['seconds']) if t['seconds'] else 0:.0f}s")
        if r["orig"] is not None:
            t = r["orig"][1]
            parts.append(f"A with its own seed {r['orig_seed']}: {t['new']}/{t['shared']}/{t['all']}")
        lines.append(head + " " + "; ".join(parts))
        if "A" in r["got"]:
            ea = r["got"]["A"][0]
            for o in ("B", "C"):
                if o in r["got"]:
                    eo = r["got"][o][0]
                    keys = [k for k in ea if k in eo and not shared[f"mce_dev_{K.DOMS[r['which']]}"][k[0]][k[1]]]
                    a = sum(eo[k][0] and not ea[k][0] for k in keys)
                    b = sum(ea[k][0] and not eo[k][0] for k in keys)
                    lines.append(f"{'':>40}{o} against A: only {o} {a}, only A {b}, p = {binom_p(a, b):.2g}")
    # the winner rule
    noise = [abs(r["got"]["A"][1]["new"] - r["orig"][1]["new"]) for r in rows if "A" in r["got"] and r["orig"] is not None]
    noise_sh = [abs(r["got"]["A"][1]["shared"] - r["orig"][1]["shared"]) for r in rows if "A" in r["got"] and r["orig"] is not None]
    lines.append(f"== A against A (the f6 networks, seed 4 against their own seed), new-form: differences {noise}; median "
                 f"{np.median(noise) if noise else float('nan'):.1f}; shared-form: {noise_sh}")
    verdict = {}
    for o in ("B", "C"):
        both = [r for r in rows if "A" in r["got"] and o in r["got"]]
        if not both:
            continue
        d = [r["got"][o][1]["new"] - r["got"]["A"][1]["new"] for r in both]
        dsh = [r["got"][o][1]["shared"] - r["got"]["A"][1]["shared"] for r in both]
        wins = sum(x > 0 for x in d)
        per = {w: sum(x > 0 for x, r in zip(d, both) if r["which"] == w) for w in K.NETS}
        nper = {w: sum(1 for r in both if r["which"] == w) for w in K.NETS}
        med, med_sh = float(np.median(d)), float(np.median(dsh))
        nz = float(np.median(noise)) if noise else float("nan")
        ok = (len(both) == 16 and wins >= 14 and all(per[w] >= 6 for w in K.NETS) and med > nz and med_sh >= -nz)
        verdict[o] = (ok, med)
        by_arm = {a: float(np.median([x for x, r in zip(d, both) if r["arm"] == a])) for a in ("continued", "control", "fresh")
                  if any(r["arm"] == a for r in both)}
        lines.append(f"== {o} against A on new-form equations: differences {d}; beats A in {wins} of {len(both)} networks ("
                     + ", ".join(f"{w} {per[w]} of {nper[w]}" for w in K.NETS) + f"); median gain {med:+.1f} (by arm: "
                     + ", ".join(f"{a} {v:+.1f}" for a, v in by_arm.items()) + f"); shared-form median change {med_sh:+.1f} -> "
                     + ("PASSES" if ok else "does not pass") + (" (incomplete)" if len(both) < 16 else ""))
    passing = [o for o, (ok, m) in verdict.items() if ok]
    if len(verdict) == 2 and all(len([r for r in rows if "A" in r["got"] and o in r["got"]]) == 16 for o in verdict):
        win = max(passing, key=lambda o: verdict[o][1]) if passing else None
        lines.append(f"== Winner (rule set before the run): {win if win else 'none -- no learning curves; check in'}")
    text = "\n".join(lines)
    (C.RUN / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    report()
