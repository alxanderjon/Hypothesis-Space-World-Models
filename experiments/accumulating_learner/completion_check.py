"""Quick check B (owner, 14 November; research log "Owner: two quick checks, a confidence-weighted blend and exact
completion", where the design and pass rule were set before the run): exact completion one term from done. At a node
the trigger fires when the best one-term evidence (the share of the remaining residual the single best addition
explains) is at least tau (0.99 and 0.999); it would then fit the 30 highest-evidence candidates and accept any that
makes the equation exact. Measured without search: the firing rate per kind of node (on-path empty start, one short,
two short; off-path partners), and at one-short on-path nodes whether the needed term is among the 30 candidates.
Settings: the curriculum combination systems (first 20 per set; pilot networks after level 5, both seeds) and the
harder fields' development systems (first 20 per field; step 1's continued networks at 30 tasks, four configurations).
Usage: completion_check.py <workers> <curriculum|harder|report>"""

import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("G2H_RUN", "completion")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harder_check as H                                         # noqa: E402

C = H.C
OUT = C.RUN
TAUS = (0.99, 0.999)
N_CAND = 30
FIRST = 20


def node_row(eq, nd, X, vocab, MZ):
    """The best one-term evidence at a node over its legal additions, and the evidence vector."""
    import numpy as np
    import search_g1 as SG
    gain = SG.evidence(nd, eq.se.ctx)[0]
    legal = MZ.legal_adds(vocab.V, X)
    return float(np.max(np.where(legal, gain, -1.0))), gain, legal


def equation_rows(eq, true, kinds, net, vocab, MZ, partners_all):
    """Rows for one equation: on-path nodes (empty start, one short, two short) and off-path partners."""
    import numpy as np
    rows = []
    on = {(): "empty start"}
    for t in true:
        others = tuple(x for x in true if x != t)
        on[others] = "one short"
        if len(true) >= 3:
            for u in others:
                on.setdefault(tuple(x for x in others if x != u), "two short")
    for X, kind_of_node in on.items():
        nd = eq.node(X)
        if not math.isfinite(nd["res"]):
            continue
        g, gain, legal = node_row(eq, nd, X, vocab, MZ)
        row = {"node": kind_of_node, "size": len(X), "best_gain": g}
        if kind_of_node == "one short":
            t = [x for x in true if x not in X][0]
            pn, _ = MZ.net_dist(net, nd, eq.se.ctx, legal)
            row.update({"kind": kinds.get(t), "evidence_rank": MZ.rank(np.where(legal, gain, -1.0), legal, t),
                        "network_rank": MZ.rank(np.log(np.maximum(pn, 1e-300)), legal, t), "key": vocab.keys[t]})
        rows.append(row)
    # off-path partners: (S - {a}) + {w}, w the network's top addition outside the true structure at S - {a}
    dropped = 0
    sources = [X for X, k in on.items() if X and (partners_all or k == "one short")]
    for S in sources:
        for a in (S if partners_all else S[:1]):
            Sp = tuple(x for x in S if x != a)
            ndp = eq.node(Sp)
            legal = MZ.legal_adds(vocab.V, Sp)
            pn, _ = MZ.net_dist(net, ndp, eq.se.ctx, legal)
            legal_w = legal.copy()
            legal_w[list(true)] = False
            w = int(np.argmax(np.where(legal_w, pn, -1.0)))
            if eq.node(tuple(sorted((set(true) - {a}) | {w})))["exact"]:
                dropped += 1
                continue
            P = tuple(sorted(Sp + (w,)))
            ndq = eq.node(P, ndp)
            if not math.isfinite(ndq["res"]):
                continue
            g, _, _ = node_row(eq, ndq, P, vocab, MZ)
            rows.append({"node": "off-path", "size": len(P), "best_gain": g, "from": on[S]})
    return rows, dropped


def curriculum_job(args):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import muzero_checks as MZ
    combo, rec, lin, net_path, pats = args
    ctx, fitter = MZ.system(rec)
    vocab = ctx["vocab"]
    net = MZ.load_net(net_path)
    rows, dropped = [], 0
    for i in range(len(rec["structure"])):
        true, cases = MZ.heldout_cases(rec, i, vocab, pats)
        if len(true) < 2:
            continue
        kinds = {t: k for t, k, _, _ in cases}
        r, d = equation_rows(MZ.Eq(ctx, fitter, i), true, kinds, net, vocab, MZ, partners_all=True)
        rows += [dict(x, replicate=lin, combo=combo, eq=i) for x in r]
        dropped += d
    return {"rows": rows, "dropped": dropped, "replicate": lin}


def harder_job(args):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import library_ops as L
    import movenet_g2 as MN
    import search_g1 as SG
    import muzero_checks as MZ
    rec, cfg, net_path, newform, familiar = args
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    net = MN.NetworkPrior(MN.load_net(net_path + ".pt").eval(), {4: np.load(net_path + "_d4.npy")}, shortlist=2048)
    rows, dropped = [], 0
    rep = f"{cfg[2]} {cfg[1] + 1}th (order {cfg[0]})"
    for i, eqrec in enumerate(rec["structure"]):
        if not newform[i]:
            continue
        true = sorted({vocab.index[k] for k in eqrec["keys"] if k in vocab.index})
        if len(true) < 2:
            continue
        kinds = {t: ("familiar" if L.tree(vocab.keys[t], i) in familiar else "new") for t in true}
        r, d = equation_rows(MZ.Eq(ctx, fitter, i), true, kinds, net, vocab, MZ, partners_all=False)
        rows += [dict(x, replicate=rep, id=rec["id"], eq=i) for x in r]
        dropped += d
    return {"rows": rows, "dropped": dropped, "replicate": rep}


def step_curriculum(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import muzero_checks as MZ
    import curriculum_check2 as C2
    import pickle
    setup = MZ.setup_info()
    jobs = []
    for combo in MZ.COMBOS:
        _, _, _, pats = C2.combos()[combo]
        dev = pickle.loads((MZ.CN.CHECK2 / "gen" / f"dev-{combo}.pkl").read_bytes())[:FIRST]
        jobs += [(combo, rec, lin, setup[lin][0], pats) for rec in dev for lin in MZ.LINEAGES]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(curriculum_job, jobs))
    (OUT / "curriculum.json").write_text(json.dumps({"rows": [x for r in res for x in r["rows"]],
                                                     "dropped": {lin: sum(r["dropped"] for r in res if r["replicate"] == lin) for lin in MZ.LINEAGES}}))
    print(f"curriculum: {sum(len(r['rows']) for r in res)} nodes from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def step_harder(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import library_ops as L
    jobs = []
    for o, pos, field in H.configs():
        recs = H.records("dev", field)[:FIRST]
        flags = H.newform_flags(o, pos, field, recs)
        familiar = {L.tree(k, eq) for (sid, eq), keys in H.history_solutions(o, pos).items() for k in keys}
        net = str(H.RUN2 / "nets" / f"o{o}_{field}_d30")
        jobs += [(r, (o, pos, field), net, flags[r["id"]], familiar) for r in recs]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(harder_job, jobs))
    reps = sorted({r["replicate"] for r in res})
    (OUT / "harder.json").write_text(json.dumps({"rows": [x for r in res for x in r["rows"]],
                                                 "dropped": {rp: sum(r["dropped"] for r in res if r["replicate"] == rp) for rp in reps}}))
    print(f"harder: {sum(len(r['rows']) for r in res)} nodes from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def step_report():
    import numpy as np
    lines = [f"== Quick check B, exact completion one term from done (trigger: best one-term evidence >= tau; {N_CAND} "
             "candidates by evidence)"]
    passing = {tau: True for tau in TAUS}
    for name in ("curriculum", "harder"):
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d["rows"]
        for rep in sorted({x["replicate"] for x in rows}):
            R = [x for x in rows if x["replicate"] == rep]
            lines.append(f"   {name}, {rep}: partners dropped as stand-ins {d['dropped'].get(rep, 0)}")
            for tau in TAUS:
                def rate(sel):
                    S = [x for x in R if sel(x)]
                    return (float(np.mean([x["best_gain"] >= tau for x in S])) if S else float("nan")), len(S)
                fire = {k: rate(lambda x, k=k: x["node"] == k) for k in ("empty start", "one short", "two short", "off-path")}
                hit = {}
                for kind in ("new", "familiar"):
                    S = [x for x in R if x["node"] == "one short" and x.get("kind") == kind and x["best_gain"] >= tau]
                    hit[kind] = (float(np.mean([x["evidence_rank"] <= N_CAND for x in S])) if S else float("nan"), len(S))
                    extra = [x for x in S if x["evidence_rank"] <= N_CAND and x["network_rank"] > N_CAND]
                    hit[kind + " beyond network top 30"] = (len(extra), len(S))
                fired = [x for x in R if x["best_gain"] >= tau]
                wasted = float(np.mean([x["node"] != "one short" for x in fired])) if fired else float("nan")
                ok = fire["one short"][0] >= 0.70                # nan compares False: no data, no pass
                ok &= hit["new"][0] >= 0.80
                for k in ("two short", "off-path"):
                    if not math.isnan(fire[k][0]):
                        ok &= fire[k][0] <= 0.20
                passing[tau] &= ok
                lines.append(f"      tau {tau}: fires at " + ", ".join(f"{k} {v[0]:.0%} of {v[1]}" for k, v in fire.items())
                             + f"; needed term among the {N_CAND} when firing one short: new {hit['new'][0]:.0%} of {hit['new'][1]}, "
                             f"familiar {hit['familiar'][0]:.0%} of {hit['familiar'][1]}; hits beyond the network's top 30: new "
                             f"{hit['new beyond network top 30'][0]}, familiar {hit['familiar beyond network top 30'][0]}; firings not one "
                             f"short {wasted:.0%}; cost {N_CAND} fits per firing -> {'meets' if ok else 'does not meet'} the rule")
    ok_any = [tau for tau, v in passing.items() if v]
    lines.append(f"== Check B: {'PASSES with tau ' + ', '.join(map(str, ok_any)) if ok_any else 'does not pass'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers, step = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT.resolve()).free / 2 ** 30
    if free < 2:
        raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {OUT.resolve()} (owner's crash guard)")
    {"curriculum": lambda: step_curriculum(workers), "harder": lambda: step_harder(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
