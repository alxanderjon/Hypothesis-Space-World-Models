"""Quick check 1 (owner, 14 November; research log "Owner: two quick checks, a confidence-weighted blend and exact
completion", where the design and pass rule were set before the run): a blend that leans on the library only where the
network is unsure. At a node, u = 1 - the network's largest probability over the legal additions, and the confidence
blend is (1 - u) * p_network + u * p_blend (the plain top-100 blend); reported too, u from the network's normalized
entropy. No search, no training: step 1's networks and libraries at 30 tasks (four field configurations), and the
curriculum combination systems with the pilot networks after level 5 (reported, not ruled).
Usage: confidence_check.py <workers> <harder|curriculum|report>"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "confidence")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harder_check as H                                         # noqa: E402

C = H.C
OUT = C.RUN
HARDER = C.ROOT / "outputs/accumulating_learner/g2h/harder"
POINTS = H.POINTS
T = 30


def uncertainty(p, n_legal):
    """1 - the largest probability, and the entropy normalized by its maximum (log of the number of legal additions), of
    a distribution over the legal additions."""
    import numpy as np
    q = p[p > 0]
    return float(1.0 - p.max()), float(-(q * np.log(q)).sum() / math.log(max(n_legal, 2)))


def scores_at(nd, ectx, legal, nets, libs, MZ):
    """The arms' log-probabilities over the legal additions at a node (nets/libs: 'continued' and optionally 'control')."""
    import numpy as np
    out, us = {}, {}
    for arm, lib_key in (("continued", "blend"), ("control", "control")):
        if arm not in nets:
            continue
        pn, _ = MZ.net_dist(nets[arm], nd, ectx, legal)
        pb = MZ.blend_dist(libs[lib_key], pn, legal)
        u, uh = uncertainty(np.where(legal, pn, 0.0), int(legal.sum()))
        name = "" if arm == "continued" else "made-up control "
        out[f"{name}network alone" if arm == "continued" else "control network"] = np.log(np.maximum(pn, 1e-300))
        out[f"{name}plain blend" if arm == "continued" else "made-up control (plain)"] = np.log(np.maximum(pb, 1e-300))
        out[f"{name}confidence blend" if arm == "continued" else "made-up control (confidence)"] = np.log(np.maximum((1 - u) * pn + u * pb, 1e-300))
        if arm == "continued":
            out["confidence blend (entropy)"] = np.log(np.maximum((1 - uh) * pn + uh * pb, 1e-300))
        us[arm] = (u, uh)
    return out, us


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
    rec, cfg, nets, tabs, newform, familiar = args
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    NP = {a: MN.NetworkPrior(MN.load_net(p + ".pt").eval(), {4: np.load(p + "_d4.npy")}, shortlist=2048) for a, p in nets.items()}
    TB = {a: np.load(p)[vocab.gid] for a, p in tabs.items()}
    rows = []
    for i, eqrec in enumerate(rec["structure"]):
        if not newform[i]:
            continue
        true = sorted({vocab.index[k] for k in eqrec["keys"] if k in vocab.index})
        if not true:
            continue
        eq = MZ.Eq(ctx, fitter, i)
        memo = {}
        for t in true:
            kind = "familiar" if L.tree(vocab.keys[t], i) in familiar else "new"
            others = [x for x in true if x != t]
            pts = [("empty start", ()), ("one short", tuple(others))] + ([("two short", tuple(others[1:]))] if len(true) >= 3 else [])
            for point, S in pts:
                if S not in memo:
                    nd = eq.node(S)
                    legal = MZ.legal_adds(vocab.V, S)
                    memo[S] = (legal,) + scores_at(nd, eq.se.ctx, legal, NP, {a: TB[a][:, i] for a in TB}, MZ)
                legal, sc, us = memo[S]
                row = {"order": cfg[0], "position": cfg[1] + 1, "field": cfg[2], "id": rec["id"], "eq": i, "key": vocab.keys[t],
                       "kind": kind, "point": point, "u": us["continued"][0], "u_entropy": us["continued"][1]}
                row.update({k: MZ.rank(v, legal, t) for k, v in sc.items()})
                rows.append(row)
    return rows


def step_harder(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import library_ops as L
    jobs = []
    for o, pos, field in H.configs():
        recs = H.records("dev", field)
        flags = H.newform_flags(o, pos, field, recs)
        familiar = {L.tree(k, eq) for (sid, eq), keys in H.history_solutions(o, pos).items() for k in keys}
        nets = {"continued": str(H.RUN2 / "nets" / f"o{o}_{field}_d{T}"), "control": str(H.RUN2 / "nets" / f"c{o}_{field}_{field}_d{T}")}
        tabs = {a: str(HARDER / "tables" / f"{a}-o{o}-{field}-T{T}.npy") for a in ("blend", "control")}
        jobs += [(r, (o, pos, field), nets, tabs, flags[r["id"]], familiar) for r in recs]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(harder_job, jobs) for x in part]
    (OUT / "harder.json").write_text(json.dumps(rows))
    print(f"harder: {len(rows)} rows from {len(jobs)} development systems in {time.time() - t0:.0f}s")


def curriculum_job(args):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import muzero_checks as MZ
    combo, rec, lin, net_path, tab_path, pats = args
    ctx, fitter = MZ.system(rec)
    vocab = ctx["vocab"]
    net, tab = MZ.load_net(net_path), MZ.load_tab(tab_path, vocab)
    rows = []
    for i in range(len(rec["structure"])):
        _, cases = MZ.heldout_cases(rec, i, vocab, pats)
        if not cases:
            continue
        eq = MZ.Eq(ctx, fitter, i)
        memo = {}
        for t, kind, point, S in cases:
            if S not in memo:
                nd = eq.node(S)
                legal = MZ.legal_adds(vocab.V, S)
                memo[S] = (legal,) + scores_at(nd, eq.se.ctx, legal, {"continued": net}, {"blend": tab[:, i]}, MZ)
            legal, sc, us = memo[S]
            row = {"lineage": lin, "combo": combo, "eq": i, "key": vocab.keys[t], "kind": kind, "point": point,
                   "u": us["continued"][0], "u_entropy": us["continued"][1]}
            row.update({k: MZ.rank(v, legal, t) for k, v in sc.items()})
            rows.append(row)
    return rows


def step_curriculum(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import muzero_checks as MZ
    import curriculum_check2 as C2
    setup = MZ.setup_info()
    jobs = []
    for combo, rec in MZ.heldout_jobs():
        _, _, _, pats = C2.combos()[combo]
        jobs += [(combo, rec, lin, setup[lin][0], setup[lin][1], pats) for lin in MZ.LINEAGES]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(curriculum_job, jobs) for x in part]
    (OUT / "curriculum.json").write_text(json.dumps(rows))
    print(f"curriculum: {len(rows)} rows from {len(jobs)} jobs in {time.time() - t0:.0f}s")


ARMS = ("network alone", "plain blend", "confidence blend", "confidence blend (entropy)", "made-up control (plain)",
        "made-up control (confidence)")
GAIN_CELLS = {("reactions_L1", 1): (1, 2), ("flows_L1", 1): (0, 2)}      # (field, order): point indices of the gains


def med(R, col, pts=POINTS):
    import numpy as np
    return [float(np.median([x[col] for x in R if x["point"] == pt])) if any(x["point"] == pt for x in R) else float("nan") for pt in pts]


def step_report():
    import numpy as np
    lines = ["== Quick check 1, the confidence-weighted blend: median rank (empty start / one short / two short), 30 tasks"]
    ok_gain = ok_loss = ok_fam = True
    if (OUT / "harder.json").exists():
        rows = json.loads((OUT / "harder.json").read_text())
        for o, pos, field in H.configs():
            R = [x for x in rows if x["order"] == o and x["field"] == field]
            lines.append(f"   {field}, {pos + 1}th (order {o}):")
            for kind in ("new", "familiar"):
                S = [x for x in R if x["kind"] == kind]
                if not S:
                    continue
                lines.append(f"      {kind:>8} ({len(S)}): " + "; ".join(f"{a} {' / '.join(f'{v:.0f}' for v in med(S, a))}" for a in ARMS))
                lines.append(f"      {kind:>8}: mean uncertainty u {' / '.join(f'{v:.2f}' for v in [float(np.mean([x['u'] for x in S if x['point'] == pt])) if any(x['point'] == pt for x in S) else float('nan') for pt in POINTS])}")
            new = [x for x in R if x["kind"] == "new"]
            fam = [x for x in R if x["kind"] == "familiar"]
            mc, mn, mb = med(new, "confidence blend"), med(new, "network alone"), med(new, "plain blend")
            if (field, o) in GAIN_CELLS and pos == (3 if field == "reactions_L1" else 4):
                g = all(mc[j] < mn[j] and mc[j] <= 2 * mb[j] for j in GAIN_CELLS[(field, o)])
                ok_gain &= g
                lines.append(f"      gains kept (points {[POINTS[j] for j in GAIN_CELLS[(field, o)]]}): {g}")
            if field == "reactions_L1" and pos == 4:
                lo = all(mc[j] <= mn[j] for j in range(3) if not math.isnan(mc[j]))
                ok_loss &= lo
                lines.append(f"      loss removed (no worse than the network alone at all points): {lo}")
            fc, fn = med(fam, "confidence blend"), med(fam, "network alone")
            f_ = all(fc[j] <= fn[j] for j in range(3) if not math.isnan(fc[j]))
            ok_fam &= f_
            lines.append(f"      familiar no worse than the network alone: {f_}")
    if (OUT / "curriculum.json").exists():
        rows = json.loads((OUT / "curriculum.json").read_text())
        lines.append("   curriculum combination systems (reported, not ruled):")
        for lin in ("net-s1", "net-s2"):
            for kind in ("new", "familiar"):
                S = [x for x in rows if x["lineage"] == lin and x["kind"] == kind]
                lines.append(f"      {lin} {kind:>8} ({len(S)}): " + "; ".join(f"{a} {' / '.join(f'{v:.0f}' for v in med(S, a))}"
                                                                         for a in ARMS[:4]))
    ok = ok_gain and ok_loss and ok_fam
    lines.append(f"== Check 1: {'PASSES' if ok else 'does not pass'} (gains kept {ok_gain}; loss removed {ok_loss}; familiar no cost {ok_fam})")
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
    {"harder": lambda: step_harder(workers), "curriculum": lambda: step_curriculum(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
