"""AlphaDev quick checks (owner, 16 November; research log "Owner: rung c, and a review of AlphaDev for refinements",
where the four checks and their pass rules were set before the run). CPU only, no search, no training; run beside rung c
only on spare cores (each job is submitted only while the machine has spare cores), never on a GPU.

A per-burst evidence and B pair gain: on new rung-c-style development systems (3 variables, 3 terms) and the a2 dream
check's development systems (2 variables, 3 terms), the true terms the end-of-rung-b networks rank worse than 10 (the
tail) at the empty start, one short and two short, re-ranked by the pooled gain (what the network sees now), by per-burst
gains (min / median / max over bursts) and by the best pair gain.
C pruning: at those nodes plus 20 random nodes of 1-3 terms per equation, the share of legal adds and swaps that cannot
change the fit (columns in the span of the current terms, or an alias of an earlier term), and whether a true term is
ever among them.
D library pieces by search saved: re-rank the blend's library by usage x search cost (structures to the first solution in
the rung's wakes) and compare top-100 coverage of rung c's held-out true terms (counts only) and of curriculum held-out
true terms with the compression ranking.
Usage: alphadev_checks.py <workers> <gen|stats|ranks|prune|libd|report>"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "alphadev")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402  (read only)

C, G2H, OUT = LD.C, LD.G2H, LD.OUT
SEEDS, ARMS = (1, 2, 3), ("blend", "net")
N_DEV, N_RANDOM = 20, 20
SPARE = int(os.environ.get("ALPHADEV_SPARE", "4"))              # submit a job only while at least this many cores are idle
DEV2 = G2H / "dreamcheck_a2" / "gen"                             # the a2 dream check's development systems (2 variables)
CUR = {1: "cpilot/gen/systems-s1.json", 2: "cpilot/gen/systems-s2.json", 3: "cseed3/gen/systems-s3.json"}


def spare_cores():
    return os.cpu_count() - os.getloadavg()[0]


def gated_map(fn, jobs, workers, label):
    """ProcessPool map that submits a job only while the machine has spare cores (rung c comes first)."""
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from multiprocessing import get_context
    out, t0, waited = [None] * len(jobs), time.time(), 0.0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        pend, nxt = {}, 0
        while nxt < len(jobs) or pend:
            while nxt < len(jobs) and len(pend) < workers:
                if spare_cores() < SPARE and pend:
                    break
                if spare_cores() < SPARE:
                    time.sleep(30)
                    waited += 30
                    continue
                pend[ex.submit(fn, jobs[nxt])] = nxt
                nxt += 1
            done, _ = wait(list(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                out[pend.pop(f)] = f.result()
    print(f"{label}: {len(jobs)} jobs in {time.time() - t0:.0f}s ({waited:.0f}s waiting for spare cores)", flush=True)
    return out


def b_state(arm, s):
    st = json.loads((G2H / "ladder" / "b" / f"{arm}-s{s}.state.json").read_text())
    return st["net"], {int(k): v for k, v in st["tables"].items()}


def dev_sets():
    """{(setting, seed): records}: setting 'd3' (new, rung-c style) and 'd2' (the a2 check's)."""
    out = {}
    for s in SEEDS:
        out[("d3", s)] = json.loads((OUT / "gen" / f"dev3-s{s}.json").read_text())
        out[("d2", s)] = json.loads((DEV2 / f"dev-s{s}.json").read_text())
    return out


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    if all((OUT / "gen" / f"dev3-s{s}.json").exists() for s in SEEDS):
        print("dev systems: kept from an earlier attempt")
        return
    keys = LD.piece_keys(3)
    allowed = {lv: keys[lv] for lv in (1, 2, 3, 5, 6, 7)}
    with ProcessPoolExecutor(min(workers, 4), mp_context=get_context("spawn")) as ex:
        dev = {s: [r for r in ex.map(LD.gen_job, [(f"alphadev-dev3-s{s}", k, allowed, (3, 3), 3) for k in range(N_DEV)]) if r["accepted"]]
               for s in SEEDS}
    old = LD.earlier_keys("alphadev")
    for d in (G2H / "ladder" / "gen", G2H / "dreamcheck" / "gen", DEV2):
        for p in d.glob("*.json"):
            obj = json.loads(p.read_text())
            recs = obj if isinstance(obj, list) else [r for v in obj.values() for r in (v if isinstance(v, list) else [])]
            old |= {LD.data_key(r["bursts"]) for r in recs if isinstance(r, dict) and r.get("bursts") is not None}
    clash = sum(LD.data_key(r["bursts"]) in old for s in SEEDS for r in dev[s])
    print(f"dev systems (3 variables): {sum(len(v) for v in dev.values())} accepted; identical to {len(old)} earlier systems: {clash}")
    if clash:
        raise SystemExit("development systems not new")
    for s in SEEDS:
        (OUT / "gen" / f"dev3-s{s}.json").write_text(json.dumps([dict(r, id=f"alphadev-dev3-s{s}-{r['k']}") for r in dev[s]]))


# ---------------------------------------------------------------------------------------------------------------------
# per-system statistics (data only; shared by every arm and seed)
# ---------------------------------------------------------------------------------------------------------------------

def node_points(rec, vocab):
    """(eq, term index t, point, S) for the true terms, as in the dream checks."""
    pts = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        for t in true:
            others = [u for u in true if u != t]
            pts += [(i, t, "empty start", ())] + [(i, t, "one short", tuple(others))] + ([(i, t, "two short", tuple(others[1:]))] if len(true) >= 3 else [])
    return pts


def projected(node, ctx, A):
    import torch
    import fitter_g1 as FG
    y, W = ctx["y"], ctx["W"]
    if node["keys"]:
        Phi = torch.nan_to_num(torch.stack([FG.value(t, W) for t in node["terms"]], 1), nan=0.0, posinf=0.0, neginf=0.0)
        Q = torch.linalg.qr(Phi)[0]
        r = y - Q @ (Q.T @ y)
        Cm = A - Q @ (Q.T @ A)
    else:
        r, Cm = y, A
    return r, Cm


def stats_job(args):
    """For each (eq, t, point): pooled gain, per-burst gain min/median/max and best pair gain of every legal term, as ranks
    of t; plus check C's prunable counts at the node (and at random nodes)."""
    rec, setting = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import muzero_checks as MZ
    t0 = time.time()
    ctx, fitter = MZ.system(rec)
    v = ctx["vocab"]
    A, owner, V = v.A, v.owner, v.V
    L = ctx["view"]["Ztr"].shape[1] - 8
    n = A.shape[0]
    nb = n // L
    # aliases: a term whose normalized, sign-fixed columns equal an earlier term's on this data
    first = {}
    alias = np.zeros(V, dtype=bool)
    for k in range(V):
        Ck = A[:, owner == k].numpy()
        sg = np.sign(Ck[np.argmax(np.abs(Ck), 0), np.arange(Ck.shape[1])])
        h = np.round(Ck * sg, 9).tobytes()
        if h in first:
            alias[k] = True
        else:
            first[h] = k
    rows, prune = [], []
    eqs = {}

    def eq_of(i):
        if i not in eqs:
            eqs[i] = MZ.Eq(ctx, fitter, i)
        return eqs[i]

    def per_term_max(g):
        return torch.zeros(V, dtype=g.dtype).scatter_reduce(0, owner, g, reduce="amax")

    def node_stats(i, S):
        eq = eq_of(i)
        nd = eq.node(S)
        r, Cm = projected(nd, eq.se.ctx, A)
        cn = (Cm * Cm).sum(0)
        rr = float(r @ r) + 1e-300
        g = torch.where(cn > 1e-20, (Cm.T @ r) ** 2 / (cn * rr), torch.zeros_like(cn)).clamp(0, 1)
        pooled = per_term_max(g).numpy()
        # per burst (rows burst-major): the burst-local correlation of the projected columns with the residual
        Rb, Cb = r[: nb * L].reshape(nb, L), Cm[: nb * L].reshape(nb, L, -1)
        num = torch.einsum("bl,blc->bc", Rb, Cb) ** 2
        den = (Cb * Cb).sum(1) * (Rb * Rb).sum(1, keepdim=True) + 1e-300
        gb = torch.where((Cb * Cb).sum(1) > 1e-20, num / den, torch.zeros_like(num)).clamp(0, 1)
        pb = torch.stack([per_term_max(gb[b]) for b in range(nb)]).numpy()
        # best pair gain: each term's best column, then R^2 of the residual on two columns
        gn, on = g.numpy(), owner.numpy()
        srt = np.lexsort((gn, on))                               # by term, then gain: each term's last entry is its best column
        last = np.r_[on[srt][1:] != on[srt][:-1], True]
        col = np.zeros(V, dtype=np.int64)
        col[on[srt][last]] = srt[last]
        col = torch.as_tensor(col)
        B = Cm[:, col]
        bn = B.norm(dim=0).clamp_min(1e-30)
        U = B / bn
        ok = (cn[col] > 1e-20).numpy()
        ra = (U.T @ r).numpy() / math.sqrt(rr)
        rho = (U.T @ U).numpy()
        rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
        pair = (ra[:, None] ** 2 + ra[None, :] ** 2 - 2 * rho * ra[:, None] * ra[None, :]) / (1 - rho ** 2)
        pair = np.where(ok[:, None] & ok[None, :], pair, 0.0)
        legal = np.ones(V, dtype=bool)
        legal[list(S)] = False
        np.fill_diagonal(pair, 0.0)
        pair[:, ~legal] = 0.0
        pairbest = np.clip(pair.max(1), 0, 1)
        # check C: adds that cannot change the fit (all columns in the span of S) or are aliases
        inspan = per_term_max((cn < 1e-10).double()).numpy() > 0.5     # some column in span ...
        allspan = -per_term_max(-(cn < 1e-10).double()).numpy() > 0.5  # ... every column in span
        p_add = legal & (allspan | alias)
        n_swap, p_swap = 0, 0
        for s in S:
            S2 = tuple(x for x in S if x != s)
            nd2 = eq.node(S2)
            _, C2 = projected(nd2, eq.se.ctx, A)
            cn2 = (C2 * C2).sum(0)
            all2 = -per_term_max(-(cn2 < 1e-10).double()).numpy() > 0.5
            n_swap += int(legal.sum())
            p_swap += int((legal & (all2 | alias)).sum())
        return legal, pooled, pb, pairbest, p_add, n_swap, p_swap

    cache = {}
    for (i, t, point, S) in node_points(rec, v):
        key = (i, tuple(sorted(S)))
        if key not in cache:
            cache[key] = node_stats(i, key[1])
        legal, pooled, pb, pairbest, p_add, n_swap, p_swap = cache[key]
        R = lambda sc: MZ.rank(np.where(legal, sc, -np.inf), legal, t)
        rows.append({"id": rec["id"], "setting": setting, "eq": i, "t": t, "key": v.keys[t], "point": point, "S": list(key[1]),
                     "pooled": R(pooled), "burst_min": R(pb.min(0)), "burst_median": R(np.median(pb, 0)),
                     "burst_max": R(pb.max(0)), "pair": R(pairbest), "pruned_true": bool(p_add[t])})
    rng = np.random.default_rng(LD.stable(rec["id"], "alphadev-random"))
    for (i, S), x in list(cache.items()):
        legal, _, _, _, p_add, n_swap, p_swap = x
        prune.append({"kind": "true path", "eq": i, "n_add": int(legal.sum()), "p_add": int(p_add.sum()), "n_swap": n_swap, "p_swap": p_swap})
    for i in range(len(rec["structure"])):
        true = {v.index[k] for k in rec["structure"][i] if k in v.index}
        for _ in range(N_RANDOM):
            S = tuple(sorted(rng.choice(V, size=int(rng.integers(1, 4)), replace=False).tolist()))
            try:
                legal, _, _, _, p_add, n_swap, p_swap = node_stats(i, S)
            except Exception:
                continue
            prune.append({"kind": "random", "eq": i, "n_add": int(legal.sum()), "p_add": int(p_add.sum()), "n_swap": n_swap, "p_swap": p_swap,
                          "true_pruned": int(sum(p_add[u] for u in true if u not in S))})
    return {"id": rec["id"], "setting": setting, "rows": rows, "prune": prune, "V": V, "aliases": int(alias.sum()), "seconds": time.time() - t0}


def step_stats(workers):
    jobs = [(r, st) for (st, s), recs in dev_sets().items() for r in recs]
    res = gated_map(stats_job, jobs, workers, "stats")
    (OUT / "stats.json").write_text(json.dumps(res, default=float))


def net_rank_job(args):
    """The network's (or confidence blend's) rank of each true term at each point, as the dream checks rank."""
    rec, spec, tag = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import muzero_checks as MZ
    ctx, fitter = MZ.system(rec)
    v = ctx["vocab"]
    prior = LD.make_prior(spec)
    out, eqs = [], {}
    for (i, t, point, S) in node_points(rec, v):
        eq = eqs.setdefault(i, MZ.Eq(ctx, fitter, i))
        nd = eq.node(S)
        legal = MZ.legal_adds(v.V, S)
        pr = prior(nd, eq.se.ctx)
        out.append({"tag": tag, "id": rec["id"], "eq": i, "t": t, "point": point,
                    "rank": MZ.rank(np.log(np.maximum(np.where(legal, pr.p_swap_add, 0), 1e-300)), legal, t)})
    return out


def step_ranks(workers):
    jobs = []
    sets = dev_sets()
    for a in ARMS:
        for s in SEEDS:
            net, tables = b_state(a, s)
            spec = ("confidence", net, tables) if a == "blend" else ("net", net)
            for st in ("d3", "d2"):
                jobs += [(r, spec, f"{a}-s{s}-{st}") for r in sets[(st, s)]]
    res = gated_map(net_rank_job, jobs, workers, "network ranks")
    (OUT / "ranks.json").write_text(json.dumps([x for part in res for x in part]))


# ---------------------------------------------------------------------------------------------------------------------
# check D: library pieces by search saved
# ---------------------------------------------------------------------------------------------------------------------

def step_libd(workers):
    """Usage x search cost from each blend lineage's wakes on rungs a2 (2 variables) and b (3 variables); top-100 coverage
    of held-out true terms (rung c's at 3 variables, counts only; the curriculum's at 2) under the compression ranking
    (the end-of-b library table, empty-context column) and under the search-cost ranking (terms with a search cost first,
    by usage x cost, then the rest in compression order)."""
    import numpy as np
    import dream_g2 as DR
    import search_g1 as SG
    out = {}
    for s in SEEDS:
        _, tables = b_state("blend", s)
        for d, rung, held in ((3, "b", "c"), (2, "a2", "cur")):
            order = SG.vocabulary_order(d)
            gid = {k: g for g, k in enumerate(order)}
            T = np.load(tables[d])
            comp = list(np.argsort(-T[:, 0], kind="stable"))
            rec = json.loads((G2H / "ladder" / rung / f"blend-s{s}.json").read_text())
            systems = json.loads((G2H / "ladder" / "gen" / f"{rung}-s{s}.json").read_text())
            struct = {r_["id"]: r_["structure"] for r_ in systems["train"]}
            usage, cost = {}, {}
            for rnd in rec["rounds"]:
                for y in rnd["wake"]:
                    if y["status"] != "done" or y["id"] not in struct:
                        continue
                    for e in y["equations"]:
                        i = int(e["eq"])
                        c = e["first_exact"] if e["first_exact"] is not None else y["budget"]
                        for k in struct[y["id"]][i]:
                            rk = DR.role_key(k, i, d)
                            if rk in gid:
                                cost.setdefault(gid[rk], []).append(c)
                                if e["first_exact"] is not None:
                                    usage[gid[rk]] = usage.get(gid[rk], 0) + 1
            score = {g: usage.get(g, 0) * float(np.mean(c)) for g, c in cost.items() if usage.get(g, 0)}
            sc = sorted(score, key=lambda g: -score[g]) + [g for g in comp if g not in score]
            if held == "c":
                hs = json.loads((G2H / "ladder" / "gen" / f"c-s{s}.json").read_text())["test"]
            else:
                cur = json.loads((G2H / CUR[s]).read_text())
                hs = [r_ for lv in (1, 2, 3, 5, 6, 7) for r_ in cur[str(lv)]["test"]]
            true = [gid[rk] for r_ in hs for i, ks in enumerate(r_["structure"]) for k in ks if (rk := DR.role_key(k, i, d)) in gid]
            cov = lambda ranking: float(np.mean([g in set(ranking[:LD.TOP]) for g in true]))
            out[f"s{s}-d{d}"] = {"held-out": "rung c held-out" if held == "c" else "curriculum held-out", "true terms": len(true),
                                 "compression": cov(comp), "search cost": cov(sc), "terms with a cost": len(score)}
    (OUT / "libd.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def step_report():
    import numpy as np
    stats = json.loads((OUT / "stats.json").read_text())
    ranks = json.loads((OUT / "ranks.json").read_text())
    S = {(r["id"], r["eq"], r["t"], r["point"]): r for x in stats for r in x["rows"]}
    lines = ["== AlphaDev quick checks (pass rules logged before the run)"]
    verdict = {}
    for check, statnames in (("A per-burst evidence", ("burst_min", "burst_median", "burst_max")), ("B pair gain", ("pair",))):
        lines.append(f"-- {check}: share of tail true terms (network rank > 10) moved into the top 10 that pooled gain leaves out; "
                     f"non-tail mean log2 rank, statistic vs pooled gain")
        ok_all = True
        for st in ("d3", "d2"):
            wins = 0
            for a in ARMS:
                for s in SEEDS:
                    tag = f"{a}-s{s}-{st}"
                    R = [(x, S[(x["id"], x["eq"], x["t"], x["point"])]) for x in ranks if x["tag"] == tag]
                    tail = [y for x, y in R if x["rank"] > 10]
                    rest = [y for x, y in R if x["rank"] <= 10]
                    best, txt = False, []
                    for nm in statnames:
                        moved = float(np.mean([y[nm] <= 10 and y["pooled"] > 10 for y in tail])) if tail else 0.0
                        nt = (float(np.mean([math.log2(y[nm]) for y in rest])), float(np.mean([math.log2(y["pooled"]) for y in rest]))) if rest else (0, 0)
                        p = moved >= 0.30 and nt[0] <= nt[1]
                        best |= p
                        txt.append(f"{nm} {moved:.0%} ({nt[0]:.2f} vs {nt[1]:.2f})")
                    wins += best
                    lines.append(f"   {tag}: tail {len(tail)} of {len(R)}; " + "; ".join(txt) + (" PASS" if best else ""))
            ok_all &= wins >= 5
            lines.append(f"   {st}: passes in {wins} of 6")
        verdict[check] = ok_all
        lines.append(f"== {check}: {'PASSES' if ok_all else 'does not pass'} (needs 5 of 6 in each setting)")
    # C
    P = [p for x in stats for p in x["prune"]]
    for kind in ("true path", "random"):
        Q = [p for p in P if p["kind"] == kind]
        share = (sum(p["p_add"] + p["p_swap"] for p in Q) / max(1, sum(p["n_add"] + p["n_swap"] for p in Q)))
        lines.append(f"-- C pruning at {kind} nodes ({len(Q)}): {share:.1%} of legal adds and swaps prunable "
                     f"(adds {sum(p['p_add'] for p in Q) / max(1, sum(p['n_add'] for p in Q)):.1%})")
    allshare = sum(p["p_add"] + p["p_swap"] for p in P) / max(1, sum(p["n_add"] + p["n_swap"] for p in P))
    true_pruned = sum(r["pruned_true"] for x in stats for r in x["rows"]) + sum(p.get("true_pruned", 0) for p in P)
    aliases = [x["aliases"] / x["V"] for x in stats]
    verdict["C pruning"] = allshare >= 0.10 and true_pruned == 0
    lines.append(f"   aliases per system {np.mean(aliases):.1%} of the vocabulary; true terms ever prunable {true_pruned}")
    lines.append(f"== C pruning: {allshare:.1%} of moves prunable -> {'PASSES' if verdict['C pruning'] else 'does not pass'} (needs 10% and no true term)")
    # D
    D = json.loads((OUT / "libd.json").read_text())
    wins = 0
    for s in SEEDS:
        c3, c2 = D[f"s{s}-d3"], D[f"s{s}-d2"]
        w = c3["search cost"] - c3["compression"] >= 0.05 and c2["search cost"] >= c2["compression"]
        wins += w
        lines.append(f"   s{s}: rung c held-out top-100 coverage {c3['compression']:.0%} -> {c3['search cost']:.0%}; curriculum "
                     f"{c2['compression']:.0%} -> {c2['search cost']:.0%}{' PASS' if w else ''}")
    verdict["D library by search saved"] = wins >= 2
    lines.append(f"== D library by search saved: {wins} of 3 seeds -> {'PASSES' if wins >= 2 else 'does not pass'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    {"gen": lambda: step_gen(workers), "stats": lambda: step_stats(workers), "ranks": lambda: step_ranks(workers),
     "libd": lambda: step_libd(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
