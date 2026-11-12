"""The evidence probe (owner, 12 November; research log "Owner: quick checks first; the evidence probe now; D only if it
passes"): a minutes-scale check without search of whether the data's own score puts missing true terms, unfamiliar ones
and the bump especially, near the front. At the root, the node one term short of each true structure and the node two
terms short (each fitted as the search fits a structure), the missing term's rank among the terms that can be added by
the evidence (search_g1.evidence: each term's best partial R^2 for the residual), by each of K1's 16 networks, and by
the K0 grammar and the context-free grammar with its library of each network's lineage. Development systems, new-form
equations; counts only. Usage: probe_evidence.py <workers> [smoke] (smoke: two systems per domain, written to
probe_smoke, prints only timings)"""

import json
import os
import sys
import time
from pathlib import Path

SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
os.environ.setdefault("G2H_RUN", "k1")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1_check as K                                             # noqa: E402

C = K.C
OUT = C.ROOT / "outputs/accumulating_learner/g2h" / ("probe_smoke" if SMOKE else "probe")
WINDOW = 64


def networks(which):
    """[(arm, run, network path, lineage)] of K1's networks for the domain."""
    return [(arm, run, str(folder / "nets" / f"{lin}_{K.DOMS[which]}_d{K.K}"), lin)
            for arm, nets in K.NETS[which].items() for run, folder, lin, _ in nets]


def job(args):
    which, rec, flags, nets, tables = args
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
    gid = vocab.gid
    priors = {lin: C._prior(path) for _, _, path, lin in nets}
    tabs = {(lin, g): np.load(p)[k][gid] for (lin, g), (p, k) in tables.items()}
    secs = {"fit": 0.0, "evidence": 0.0, "network": 0.0, "network_calls": 0}

    def rank(score, legal, t):                                   # 1 = first among legal adds; ties simplest first
        s = np.where(legal, score, -np.inf)
        better = (s > s[t]).sum()
        ties = ((s == s[t]) & (np.arange(len(s)) < t)).sum()
        return int(better + ties + 1)
    out = []
    for i, eq in enumerate(rec["structure"]):
        if flags[i]:                                             # shared-form: not probed
            continue
        true = sorted({vocab.index[k] for k in eq["keys"] if k in vocab.index})
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        cache = {(): se.root}

        def node(S):
            if S not in cache:
                tf = time.time()
                cache[S] = se.evaluate(S, None)
                secs["fit"] += time.time() - tf
            return cache[S]
        scored = {}

        def scores(S):
            if S not in scored:
                nd = node(S)
                te = time.time()
                ev = SG.evidence(nd, se.ctx)[0]
                secs["evidence"] += time.time() - te
                net = {}
                for lin, pr in priors.items():
                    tn = time.time()
                    net[lin] = pr(nd, se.ctx).p_add
                    secs["network"] += time.time() - tn
                    secs["network_calls"] += 1
                legal = np.ones(vocab.V, dtype=bool)
                legal[list(S)] = False
                scored[S] = (ev, net, legal)
            return scored[S]
        probes = [("root", t, ()) for t in true]
        for j, t in enumerate(true):
            probes.append(("one short", t, tuple(u for u in true if u != t)))
            if len(true) >= 2:
                t2 = true[(j + 1) % len(true)]
                probes.append(("two short", t, tuple(u for u in true if u not in (t, t2))))
        for kind, t, S in probes:
            ev, net, legal = scores(S)
            row = {"id": rec["id"], "eq": i, "key": vocab.keys[t], "node": kind, "V": int(legal.sum()),
                   "evidence": rank(ev, legal, t), "evidence_gain": float(ev[t])}
            for lin in priors:
                row[f"net:{lin}"] = rank(net[lin], legal, t)
            for (lin, g), tab in tabs.items():
                row[f"{g}:{lin}"] = rank(tab[:, i], legal, t)
            out.append(row)
    return {"rows": out, "seconds": time.time() - t0, "secs": secs}


def summarize(vals):
    import numpy as np
    if not vals:
        return "-"
    v = np.asarray(vals)
    return f"{np.median(v):.0f} [{100 * np.mean(v <= WINDOW):.0f}% in the top {WINDOW}, {100 * np.mean(v <= 10):.0f}% in the top 10]"


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    import grammar_g2 as GR
    import stageX_report as XR
    import k1_report as KR
    workers = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    lines, checks, verdict = [], [], {}
    for which, dom in K.DOMS.items():
        nets = networks(which)
        tables = {(lin, g): K.table_of(which, arm, run, o) for arm, run, _, lin in nets for o, g in (("B", "ctx"), ("C", "uni"))}
        recs = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(job, [(which, r, flags[r["id"]], nets, tables) for r in recs]))
        rows = [x for r in res for x in r["rows"]]
        (OUT / f"ranks_{which}.json").write_text(json.dumps(rows))
        tot = {k: sum(r["secs"][k] for r in res) for k in res[0]["secs"]}
        checks.append(f"{which}: {len(rows)} probes in {time.time() - t0:.0f}s; fits {tot['fit']:.0f}s, evidence {tot['evidence']:.0f}s, "
                      f"network calls {tot['network_calls']} in {tot['network']:.0f}s "
                      f"({tot['network'] / max(tot['network_calls'], 1):.2f}s each)")
        if SMOKE:
            continue
        seen = {lin: KR.seen_shapes(which, arm, run) for arm, run, _, lin in nets}
        before = {p["arm"]: p for p in __import__("k0b_check").points(which) if p["k"] == 0}
        prior_seen = set()
        for arm in ("continued", "control"):
            e, g, n = __import__("k0b_check").experience(before[arm], __import__("k0b_check").SETUPS[which]["o"])
            prior_seen |= {s for d, own, keys in e + n for s in GR.equation_shapes(d, own, keys)}
        for x in rows:
            x["shape"] = GR.equation_shapes(4, x["eq"], [x["key"]])[0]
            x["kind"] = XR.kind(x["key"], x["eq"]) or "other"
        lines.append(f"== {which}: {len({(x['id'], x['eq']) for x in rows})} new-form development equations, "
                     f"{len({(x['id'], x['eq'], x['key']) for x in rows})} true terms; of about {rows[0]['V']} terms. Median rank "
                     f"of the missing true term [share in the top {WINDOW}, top 10]: evidence | networks (median over the "
                     f"{len(nets)} networks' medians) | K0 grammar | context-free grammar with library (both: median over lineages)")
        for kind in ("one short", "two short", "root"):
            lines.append(f"   node {kind}:")
            R = [x for x in rows if x["node"] == kind]
            sets = (("all", lambda x, lin: True), ("new to both prior experiences", lambda x, lin: x["shape"] not in prior_seen),
                    ("unfamiliar to the lineage", lambda x, lin: x["shape"] not in seen[lin]),
                    ("bump", lambda x, lin: x["kind"] == "bump"), ("saturating", lambda x, lin: x["kind"] == "saturating"))
            for title, sel in sets:
                per_lin = {lin: [x for x in R if sel(x, lin)] for _, _, _, lin in nets}
                if not any(per_lin.values()):
                    continue
                if title == "unfamiliar to the lineage":
                    ev_meds = [np.median([x["evidence"] for x in v]) for v in per_lin.values() if v]
                    ev = f"{np.median(ev_meds):.0f} (lineages {min(ev_meds):.0f}-{max(ev_meds):.0f})"
                    n_terms = f"{np.median([len(v) for v in per_lin.values()]):.0f} per lineage"
                    if kind == "one short":
                        verdict[(which, "unfamiliar")] = float(np.median(ev_meds))
                else:
                    sel_rows = next(v for v in per_lin.values())
                    ev = summarize([x["evidence"] for x in sel_rows])
                    n_terms = str(len(sel_rows))
                    if kind == "one short" and title in ("new to both prior experiences", "bump"):
                        verdict[(which, title)] = float(np.median([x["evidence"] for x in sel_rows])) if sel_rows else None
                nets_m = [np.median([x[f"net:{lin}"] for x in v]) for lin, v in per_lin.items() if v]
                g_m = {g: [np.median([x[f"{g}:{lin}"] for x in v]) for lin, v in per_lin.items() if v] for g in ("ctx", "uni")}
                lines.append(f"     {title} ({n_terms}): {ev} | {np.median(nets_m):.0f} (networks {min(nets_m):.0f}-{max(nets_m):.0f}) | "
                             f"{np.median(g_m['ctx']):.0f} | {np.median(g_m['uni']):.0f}")
    if not SMOKE:
        lines.append(f"== D's condition (set before the run): at the node one term short, in both domains, median evidence rank "
                     f"<= {WINDOW} for the bump, the terms new to both prior experiences, and each lineage's unfamiliar terms:")
        ok = True
        for which in K.DOMS:
            parts = []
            for key in ("bump", "new to both prior experiences", "unfamiliar"):
                v = verdict.get((which, key))
                good = v is not None and v <= WINDOW
                ok &= good
                parts.append(f"{key} {v if v is None else round(v)} {'yes' if good else 'NO'}")
            lines.append(f"   {which}: " + "; ".join(parts))
        lines.append(f"   -> {'MET: run ordering D' if ok else 'NOT MET: D does not run'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print("\n".join(checks))
    if not SMOKE:
        print(text)


if __name__ == "__main__":
    main()
