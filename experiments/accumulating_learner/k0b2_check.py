"""K0b's all-parses check (owner, 11 November; research log "Owner: skip the replication; the all-parses check, ..."):
K0b's lineage points again, each library grammar fitted two ways: one parse per solution, the largest pieces first (as
in K0b), and all parses weighted by their probability (library_ops counting="all"). The libraries are learned again
with ties broken in a fixed order, then compared with K0b's; the grammars without a library, the networks' ranks and the
simplest-first ranks are K0b's (unchanged by the counting). No search, no training; development systems only; counts only.
Usage: k0b2_check.py <workers> [smoke] (smoke: two development systems per domain and two runs per arm, written to
k0b2_smoke; prints only checks that reveal no results)"""

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k0b_check as KB                                           # noqa: E402

C = KB.C
SMOKE = KB.SMOKE
OUT = KB.X_RUN / ("k0b2_smoke" if SMOKE else "k0b2")
K0B = KB.X_RUN / "k0b"
ARMS = KB.ARMS
LIB = ("ctx_lib", "ctx_lib_all", "uni_lib", "uni_lib_all")
COLUMNS = (("ctx", "K0 grammar"), ("ctx_lib", "+ library, one parse"), ("ctx_lib_all", "+ library, all parses"),
           ("uni", "context-free"), ("uni_lib", "+ library, one parse"), ("uni_lib_all", "+ library, all parses"))


def ident(p):
    return p["arm"], p["run"], p["k"]


def grammar_job(args):
    which, i, pt, order, tokens, new_eqs, out_dir = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import library_ops as L
    t0 = time.time()
    earlier, groups, newest = KB.experience(pt, KB.SETUPS[which]["o"])
    terms = Counter(L.tree(k, int(own)) for d, own, keys in earlier + newest for k in keys)
    libs = {form: L.learn(terms, tokens, form) if terms else ([], []) for form in ("ctx", "uni")}
    E, N = [(own, keys) for d, own, keys in earlier], [(own, keys) for d, own, keys in newest]
    tables, bits, alphas, iters = {}, {}, {}, {}
    for g in LIB:
        form, counting = g.split("_")[0], "all" if g.endswith("_all") else "one"
        gr = L.LibGrammar.fit(E, N, tokens, libs[form][0], earlier_groups=groups, form=form, counting=counting)
        tables[g] = np.array([[gr.logp_term(key, own) for own in range(4)] for key in order])
        bits[g] = float(np.mean([gr.bits(own, keys) for own, keys in new_eqs]))
        alphas[g] = [c.alpha for c in gr.components]
        iters[g] = [getattr(c, "em_iterations", None) for c in gr.components]
    np.savez(Path(out_dir) / f"tables_{which}_{i}.npz", **tables)
    return {"i": i, "bits": bits, "alphas": alphas, "em_iterations": iters, "seconds": time.time() - t0,
            "library": {form: [L.show(x["piece"]) for x in log] for form, (_, log) in libs.items()}}


def rank_job(args):
    which, rec, n_pts, out_dir = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    vocab = ctx["vocab"]
    gid = vocab.gid

    def ranks(score):                                            # 1 = first; ties by vocabulary order (simplest first)
        order = np.lexsort((np.arange(len(score)), -score))
        r = np.empty(len(score), dtype=np.int64)
        r[order] = np.arange(1, len(score) + 1)
        return r
    trues = [[vocab.index[k] for k in eq["keys"] if k in vocab.index] for eq in rec["structure"]]
    out = []
    for j in range(n_pts):
        npz = np.load(Path(out_dir) / f"tables_{which}_{j}.npz")
        tab = {g: npz[g] for g in LIB}
        for i, true in enumerate(trues):
            gr = {g: ranks(tab[g][gid, i]) for g in LIB}
            for t in true:
                out.append(dict({g: int(gr[g][t]) for g in LIB}, id=rec["id"], eq=i, pt=j, key=vocab.keys[t]))
    return out


def report(which, pts, gres, rows, k0b_pts, k0b_rows):
    import numpy as np
    import grammar_g2 as GR
    import stageX_report as XR
    old = {ident(p): j for j, p in enumerate(k0b_pts)}
    prev = {(x["pt"], x["id"], x["eq"], x["key"]): x for x in k0b_rows}
    for x in rows:                                               # K0b's ranks of the grammars without a library
        y = prev[(old[ident(pts[x["pt"]])], x["id"], x["eq"], x["key"])]
        x.update(ctx=y["ctx"], uni=y["uni"], net=y["net"], k0b_ctx_lib=y["ctx_lib"], k0b_uni_lib=y["uni_lib"])
    seen = []
    for p in pts:
        e, g, n = KB.experience(p, KB.SETUPS[which]["o"])
        seen.append({s for d, own, keys in e + n for s in GR.equation_shapes(d, own, keys)})
    first = {arm: next(j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == 0) for arm in ARMS}
    prior_seen = seen[first["continued"]] | seen[first["control"]]
    shape = {}
    for x in rows:
        s = shape.setdefault((x["key"], x["eq"]), GR.equation_shapes(4, x["eq"], [x["key"]])[0])
        x["unfamiliar"], x["new_to_both"] = s not in seen[x["pt"]], s not in prior_seen
        x["kind"] = XR.kind(x["key"], x["eq"]) or "other"
    by_pt = {}
    for x in rows:
        by_pt.setdefault(x["pt"], []).append(x)
    k0b_libs = {ident(q): {f: [x["piece"] for x in q["library"][f]] for f in ("ctx", "uni")} for q in k0b_pts}
    same_lib = sum(gres[j]["library"] == k0b_libs[ident(p)] for j, p in enumerate(pts))
    same_rank = sum(all(x["ctx_lib"] == x["k0b_ctx_lib"] and x["uni_lib"] == x["k0b_uni_lib"] for x in by_pt.get(j, []))
                    for j in range(len(pts)))
    lines = [f"== {which}: {len(pts)} lineage points; libraries identical to K0b's at {same_lib}; one-parse ranks identical "
             f"to K0b's at {same_rank}; all-parses fits converged in a median of "
             f"{np.median([v for g in gres for k in ('ctx_lib_all', 'uni_lib_all') for v in g['em_iterations'][k] if v]):.0f} "
             f"iterations (max {max([v for g in gres for k in ('ctx_lib_all', 'uni_lib_all') for v in g['em_iterations'][k] if v] or [0])})"]
    sets = (("unfamiliar to the lineage", lambda x: x["unfamiliar"]), ("new to both prior experiences", lambda x: x["new_to_both"]),
            ("bump", lambda x: x["kind"] == "bump"), ("saturating", lambda x: x["kind"] == "saturating"),
            ("all", lambda x: True))
    lines.append("   Median rank of true terms; per arm and point, the median over runs [range] of each run's median. Columns: "
                 + " | ".join(t for _, t in COLUMNS))
    cells = {}
    for title, sel in sets:
        lines.append(f"     {title}:")
        for arm in ARMS:
            for k in (0, 10, 30):
                js = [j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == k]
                per = [r for r in ([x for x in by_pt.get(j, []) if sel(x)] for j in js) if r]
                if not per:
                    continue
                meds = {f: [float(np.median([x[f] for x in r])) for r in per] for f, _ in COLUMNS}
                cells[(title, arm, k)] = {f: float(np.median(v)) for f, v in meds.items()}
                cells[(title, arm, k)]["has_library"] = any(gres[j]["library"]["ctx"] or gres[j]["library"]["uni"] for j in js)
                lines.append(f"       {arm:>9} {'before' if k == 0 else f'after {k}':>8} ({KB.summary([len(r) for r in per])} terms): "
                             + " | ".join(KB.summary(meds[f]) for f, _ in COLUMNS))
    per_pt = {j: {"new_to_both": float(np.median([x["uni_lib_all"] for x in by_pt.get(j, []) if x["new_to_both"]]))
                  if any(x["new_to_both"] for x in by_pt.get(j, [])) else None,
                  "new_to_both_ctx": float(np.median([x["ctx_lib_all"] for x in by_pt.get(j, []) if x["new_to_both"]]))
                  if any(x["new_to_both"] for x in by_pt.get(j, [])) else None,
                  "bits": gres[j]["bits"], "bits_plain": k0b_pts[old[ident(p)]]["bits"], "arm": p["arm"], "k": p["k"]}
              for j, p in enumerate(pts)}
    lines.append("   Bits check (mean bits per new-form equation; median over runs): columns one parse | all parses, contextual then "
                 "context-free")
    for arm in ARMS:
        for k in (0, 10, 30):
            js = [j for j, p in enumerate(pts) if p["arm"] == arm and p["k"] == k]
            if js:
                lines.append(f"       {arm:>9} {'before' if k == 0 else f'after {k}':>8}: "
                             + " | ".join(KB._bits(gres, js, g) for g in LIB))
    return lines, cells, per_pt


def verdicts(cells_by_domain, per_pt_by_domain):
    """The reading set before the run: (1) harm removed; (2) K0b's readings for the all-parses fit; (3) the fit
    ordering C uses."""
    out, harm, fixed, c_all, c_one, tie = [], 0, 0, 0, 0, 0
    for form, title in (("ctx", "contextual"), ("uni", "context-free")):
        ranks_ok = all(v[f"{form}_lib_all"] < v[form] for cells in cells_by_domain.values()
                       for (t, arm, k), v in cells.items() if t == "unfamiliar to the lineage" and v["has_library"])
        bits_ok = all(q["bits"][f"{form}_lib_all"] < q["bits_plain"][form] for per in per_pt_by_domain.values()
                      for q in per.values() if q["k"] or q["arm"] != "fresh")
        out.append(f"(2) K0b reading 1 for the all-parses {title} library: lower median rank of unfamiliar true terms in every cell "
                   f"with a library {ranks_ok}; lower bits at every point {bits_ok} -> {'MET' if ranks_ok and bits_ok else 'not met'}")
    for form, title, key in (("ctx", "contextual", "new_to_both_ctx"), ("uni", "context-free", "new_to_both")):
        ok, parts = True, []
        for which, per in per_pt_by_domain.items():
            for k in (0, 10, 30):
                cont = [q for q in per.values() if q["arm"] == "continued" and q["k"] == k]
                ctrl = [q for q in per.values() if q["arm"] == "control" and q["k"] == k]
                r = sum(a[key] < b[key] for a in cont for b in ctrl if a[key] is not None and b[key] is not None)
                b_ = sum(a["bits"][f"{form}_lib_all"] < b["bits"][f"{form}_lib_all"] for a in cont for b in ctrl)
                n = len(cont) * len(ctrl)
                ok &= r == n and b_ == n
                parts.append(f"{which} {k}: ranks {r}/{n}, bits {b_}/{n}")
        out.append(f"(2) K0b reading 2 for the all-parses {title} library (continued better than the control, run pairings): "
                   + "; ".join(parts) + f" -> {'MET' if ok else 'not met'}")
    for which, cells in cells_by_domain.items():
        for (title, arm, k), v in cells.items():
            if title != "unfamiliar to the lineage" or not v["has_library"]:
                continue
            for form in ("ctx", "uni"):
                if v[f"{form}_lib"] > v[form]:
                    harm += 1
                    fixed += v[f"{form}_lib_all"] <= v[form]
            if v["uni_lib_all"] < v["uni_lib"]:
                c_all += 1
            elif v["uni_lib_all"] > v["uni_lib"]:
                c_one += 1
            else:
                tie += 1
    out.append(f"(1) cells where the one-parse library ranked unfamiliar true terms worse than the grammar without it: {harm}; "
               f"of these, the all-parses library no worse than without: {fixed} -> harm {'REMOVED' if harm and fixed == harm else 'NOT removed'}")
    out.append(f"(3) context-free grammar with its library, unfamiliar true terms, cells where all parses rank better / one parse "
               f"better / equal: {c_all} / {c_one} / {tie}")
    return out, c_all, c_one


def main():
    import numpy as np
    import grammar_ops as GO
    import search_g1 as SG
    workers = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    order = SG.vocabulary_order(4)
    tokens = GO.alphabet(order + SG.vocabulary_order(2))
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    lines, checks, cells_by_domain, bits_by_domain = [], [], {}, {}
    for which, setup in KB.SETUPS.items():
        dom = setup["dom"]
        dev = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        new_eqs = [(i, eq["keys"]) for r in dev for i, eq in enumerate(r["structure"]) if not flags[r["id"]][i]]
        pts = KB.points(which)
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            gres = list(ex.map(grammar_job, [(which, i, pt, order, tokens, new_eqs, str(OUT)) for i, pt in enumerate(pts)]))
            t1 = time.time()
            rows = [x for part in ex.map(rank_job, [(which, r, len(pts), str(OUT)) for r in dev]) for x in part]
        flagd = {(r["id"], i): not flags[r["id"]][i] for r in dev for i in range(len(r["structure"]))}
        rows = [x for x in rows if flagd[(x["id"], x["eq"])]]
        k0b_pts = json.loads((K0B / f"libraries_{which}.json").read_text())
        k0b_rows = json.loads((K0B / f"ranks_{which}.json").read_text())
        part, cells, per_pt = report(which, pts, gres, rows, k0b_pts, k0b_rows)
        cells_by_domain[which], bits_by_domain[which] = cells, per_pt
        (OUT / f"ranks_{which}.json").write_text(json.dumps(rows))
        (OUT / f"fits_{which}.json").write_text(json.dumps([dict(p, **g) for p, g in zip(pts, gres)], default=str))
        checks.append(f"{which}: grammars {t1 - t0:.0f}s (slowest point {max(g['seconds'] for g in gres):.0f}s); ranks "
                      f"{time.time() - t1:.0f}s; rows {len(rows)}; " + part[0].split(": ", 1)[1].split("; all-parses")[0])
        lines += part
    v, _, _ = verdicts(cells_by_domain, bits_by_domain)
    lines += ["== Reading set before the run:"] + ["   " + x for x in v]
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print("\n".join(checks))
    if not SMOKE:
        print(text)


if __name__ == "__main__":
    main()
