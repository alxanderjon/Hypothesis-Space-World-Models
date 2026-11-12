"""The late-search quick check (owner, 12 November; research log "Owner: the late-search quick check, then the Stage C and
K2 design"): would mixing the data's score into the network's choice only where the current structure already explains
most of the data help unfamiliar terms without diluting familiar completions? At the evidence probe's nodes (root, one
term short, two terms short) of the development systems' new-form equations: the fit's unexplained share (residual over
the target's size); the missing term's rank by the evidence; and, where that share is at most 0.5, its rank by three
networks per domain and by each network mixed half and half with the evidence's top 64 terms (in proportion to their
explained share). No search, no training; counts only. Usage: probe_late.py <workers> [smoke]"""

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
OUT = C.ROOT / "outputs/accumulating_learner/g2h" / ("late_smoke" if SMOKE else "late")
GATES = (0.5, 0.3, 0.1, 0.03)
WINDOW = 64


def networks(which):
    """The first network of each arm in K1's list: continued and fresh f6 seed 1, control n."""
    return [(arm, nets[0][0], str(nets[0][1] / "nets" / f"{nets[0][2]}_{K.DOMS[which]}_d{K.K}"), nets[0][2])
            for arm, nets in K.NETS[which].items()]


def job(args):
    which, rec, flags, nets = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    priors = {lin: C._prior(path) for _, _, path, lin in nets}
    calls = 0

    def rank(score, legal, t):
        s = np.where(legal, score, -np.inf)
        return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)
    out = []
    for i, eq in enumerate(rec["structure"]):
        if flags[i]:
            continue
        true = sorted({vocab.index[k] for k in eq["keys"] if k in vocab.index})
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        memo = {}

        def at(S):
            if S not in memo:
                nd = se.root if not S else se.evaluate(S, None)
                rel = float(nd["res"] / se.yn) if np.isfinite(nd["res"]) else float("inf")
                ev = SG.evidence(nd, se.ctx)[0]
                legal = np.ones(vocab.V, dtype=bool)
                legal[list(S)] = False
                mix = {}
                if rel <= max(GATES):
                    g = np.where(legal, np.maximum(ev, 0.0), -1.0)
                    top = np.argsort(-g, kind="stable")[:WINDOW]
                    pe = np.zeros(vocab.V)
                    w = np.maximum(g[top], 0.0)
                    pe[top] = w / w.sum() if w.sum() > 0 else 1.0 / len(top)
                    for lin, pr in priors.items():
                        pa = np.where(legal, pr(nd, se.ctx).p_add, 0.0)
                        if pa.sum() > 0:
                            pa = pa / pa.sum()
                            mix[lin] = (pa, 0.5 * pa + 0.5 * pe)
                memo[S] = (rel, ev, legal, mix)
            return memo[S]
        probes = [("root", t, ()) for t in true]
        for j, t in enumerate(true):
            probes.append(("one short", t, tuple(u for u in true if u != t)))
            if len(true) >= 2:
                t2 = true[(j + 1) % len(true)]
                probes.append(("two short", t, tuple(u for u in true if u not in (t, t2))))
        for kind, t, S in probes:
            rel, ev, legal, mix = at(S)
            row = {"id": rec["id"], "eq": i, "key": vocab.keys[t], "node": kind, "rel": rel, "evidence": rank(ev, legal, t)}
            for lin, (pa, pm) in mix.items():
                row[f"net:{lin}"] = rank(pa, legal, t)
                row[f"mix:{lin}"] = rank(pm, legal, t)
            out.append(row)
        calls += sum(len(m[3]) for m in memo.values())
    return {"rows": out, "calls": calls}


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
        recs = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(job, [(which, r, flags[r["id"]], nets) for r in recs]))
        rows = [x for r in res for x in r["rows"]]
        (OUT / f"ranks_{which}.json").write_text(json.dumps(rows))
        checks.append(f"{which}: {len(rows)} probes, {sum(r['calls'] for r in res)} network calls, {time.time() - t0:.0f}s")
        if SMOKE:
            continue
        seen = {lin: KR.seen_shapes(which, arm, run) for arm, run, _, lin in nets}
        for x in rows:
            x["shape"] = GR.equation_shapes(4, x["eq"], [x["key"]])[0]
            x["kind"] = XR.kind(x["key"], x["eq"]) or "other"
        lines.append(f"== {which}: {len(rows)} probes. Gate open at nodes one short / two short / root; then at the open nodes, "
                     f"the missing term's median rank: evidence | network | network + evidence (medians over the "
                     f"{len(nets)} networks' lineages)")
        for g in GATES:
            rate = {k: np.mean([x["rel"] <= g for x in rows if x["node"] == k]) for k in ("one short", "two short", "root")}
            G = [x for x in rows if x["rel"] <= g]
            lines.append(f"   gate {g}: open at {100 * rate['one short']:.0f}% / {100 * rate['two short']:.0f}% / "
                         f"{100 * rate['root']:.0f}% of nodes ({len(G)} probes)")
            res_g = {}
            for title, sel in (("unfamiliar", lambda x, lin: x["shape"] not in seen[lin]),
                               ("familiar", lambda x, lin: x["shape"] in seen[lin]),
                               ("bump", lambda x, lin: x["kind"] == "bump"), ("all", lambda x, lin: True)):
                ev, net, mix, n = [], [], [], []
                for _, _, _, lin in nets:
                    sel_rows = [x for x in G if sel(x, lin) and f"net:{lin}" in x]
                    if sel_rows:
                        ev.append(np.median([x["evidence"] for x in sel_rows]))
                        net.append(np.median([x[f"net:{lin}"] for x in sel_rows]))
                        mix.append(np.median([x[f"mix:{lin}"] for x in sel_rows]))
                        n.append(len(sel_rows))
                if not ev:
                    continue
                res_g[title] = (float(np.median(ev)), float(np.median(net)), float(np.median(mix)))
                lines.append(f"      {title} ({np.median(n):.0f} probes per lineage): {np.median(ev):.0f} | {np.median(net):.0f} | "
                             f"{np.median(mix):.0f}")
            ok = (rate["one short"] >= 0.25 and "unfamiliar" in res_g and "bump" in res_g and "familiar" in res_g
                  and res_g["unfamiliar"][0] <= WINDOW and res_g["bump"][0] <= WINDOW
                  and res_g["unfamiliar"][2] < res_g["unfamiliar"][1] and res_g["familiar"][2] <= 5)
            verdict[(which, g)] = ok
            lines.append(f"      -> conditions met at this gate: {'yes' if ok else 'no'}")
    if not SMOKE:
        good = [g for g in GATES if all(verdict.get((w, g)) for w in K.DOMS)]
        lines.append(f"== Reading set before the run: a gate meeting all four conditions in both domains: "
                     f"{good if good else 'none'} -> {'PASSES' if good else 'does not pass'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print("\n".join(checks))
    if not SMOKE:
        print(text)


if __name__ == "__main__":
    main()
