"""The MuZero-family quick checks (owner, 14 November; research log "Owner: a third seed of the blend test, and the
MuZero-family quick checks alongside it", where the designs and pass rules were set before the runs). No search; the
exact fit stays the judge of every structure that is fitted.
  check4  Reanalyse: needed terms of training systems left unsolved at mastered levels, ranked by the prior that last
          tried them and by the lineage's final prior.
  check2  sampled exploration: the chance that the search's exploration draws include a needed term, uniform vs proposals.
  check3  the Gumbel root rule: 16 candidates drawn without repeats, one fit each, 2 finalists by g + logit + sigma(q).
  check1  a learned fit predictor, trained on the lineage's own fits; candidates ranked by the blend plus predicted fit.
  check5  the value of a partial equation: on-path vs off-path partial equations, by fit quality, evidence, learned value.
  time    a timing slice (fits only, no outcomes) for choosing the held-out subsample of checks 3, 1 and 5.
Usage: muzero_checks.py <workers> <step> [limit=N] [gpu=G] (N: held-out systems per combination set)"""

import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "muzero")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm3_check as A3                                          # noqa: E402

CN, CP, C, D = A3.CN, A3.CP, A3.C, A3.D
OUT = C.RUN
G2H = C.ROOT / "outputs/accumulating_learner/g2h"
PILOT, BLEND = G2H / "cpilot", G2H / "cblend"
LINEAGES = A3.LINEAGES                                           # the pilot's network lineages, seeds 1 and 2
COMBOS = CN.COMBOS
FAMILIAR = CN.MASTERED
POINTS = ("empty start", "one short", "two short")
TOP, SHARE_TOP = 100, 0.95                                       # the top-100 blend, as searched
OPTS = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
LIMIT = int(OPTS.get("limit", 0)) or None


# ---------------------------------------------------------------------------------------------------------------------
# shared pieces (worker side)
# ---------------------------------------------------------------------------------------------------------------------

def load_tab(path, vocab):
    import numpy as np
    t = np.load(path)
    t = t["t"] if path.endswith(".npz") else t
    return t[vocab.gid]


def load_net(path):
    import numpy as np
    import movenet_g2 as MN
    return MN.NetworkPrior(MN.load_net(path + ".pt").eval(), {D: np.load(path + f"_d{D}.npy")}, shortlist=2048)


def softmax_legal(logp, legal):
    import numpy as np
    s = np.where(legal, logp, -np.inf)
    w = np.exp(s - s[legal].max())
    return w / w.sum()


def blend_dist(lib, pn, legal):
    """The top-100 blend's distribution over legal additions (curriculum_blend.BlendPrior)."""
    import numpy as np
    n = int(legal.sum())
    order = np.lexsort((np.arange(len(lib)), -np.where(legal, lib, -np.inf)))
    top = order[:min(TOP, n)]
    g = 0.5 * lib[top] + 0.5 * np.log(np.maximum(pn[top], 1e-300))
    mix = np.zeros(len(lib))
    mix[top] = np.exp(g - g.max())
    mix[top] *= (SHARE_TOP if n > len(top) else 1.0) / mix[top].sum()
    rest = legal.copy()
    rest[top] = False
    if rest.any():
        r = np.where(rest, lib, -np.inf)
        w = np.exp(r - r[rest].max())
        mix += (1.0 - SHARE_TOP) * w / w.sum()
    return mix


def net_dist(net, nd, sctx, legal):
    """The network's distribution over legal additions at a node, and its Prior (for removal probabilities)."""
    import numpy as np
    pr = net(nd, sctx)
    pn = np.where(legal, pr.p_swap_add, 0.0)
    return pn / pn.sum(), pr


def rank(score, legal, t):
    import numpy as np
    s = np.where(legal, score, -np.inf)
    return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)


def legal_adds(V, S):
    import numpy as np
    legal = np.ones(V, dtype=bool)
    legal[list(S)] = False
    return legal


def heldout_cases(rec, i, vocab, pats):
    """(term, kind, point, S) for equation i of a held-out system, as in arm 3's check."""
    import curriculum_check as CC
    import curriculum_check2 as C2
    true = sorted({vocab.index[k] for k in rec["structure"][i] if k in vocab.index})
    out = []
    for t in true:
        if C2.is_new(vocab.keys[t], i, pats):
            kind = "new"
        elif CC.level_of(vocab.keys[t], i) in FAMILIAR:
            kind = "familiar"
        else:
            continue
        others = [u for u in true if u != t]
        pts = [("empty start", ()), ("one short", tuple(others))]
        if len(true) >= 3:
            pts.append(("two short", tuple(others[1:])))
        out += [(t, kind, p, S) for p, S in pts]
    return true, out


def path_cases(keys_i, vocab):
    """(term, point, S) along a structure (a verified solution or a true structure): empty start, one short, two short."""
    true = sorted({vocab.index[k] for k in keys_i if k in vocab.index})
    out = []
    for t in true:
        others = [u for u in true if u != t]
        pts = [("empty start", ()), ("one short", tuple(others))]
        if len(true) >= 3:
            pts.append(("two short", tuple(others[1:])))
        out += [(t, p, S) for p, S in pts]
    return true, out


class Eq:
    """One equation's node evaluations (the search's own evaluate, memoized), with fit timing."""

    def __init__(self, ctx, fitter, i):
        import numpy as np
        import search_g1 as SG
        self.se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        self.fit_seconds, self.fits = 0.0, 0

    def node(self, S, parent=None):
        S = tuple(sorted(S))
        if not S:
            return self.se.root
        if S not in self.se.table:
            t0 = time.perf_counter()
            self.se.evaluate(S, parent)
            self.fit_seconds += time.perf_counter() - t0
            self.fits += 1
        return self.se.table[S]


def system(rec):
    import numpy as np
    import fitter_g1 as FG
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    return ctx, FG.make_fitter(ctx, seed=1)


def stable_seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _worker_init():
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)


def heldout_jobs():
    """[(combo, record)] of the follow-up check's combination systems (the first LIMIT of each set if a limit is set)."""
    out = []
    for name in COMBOS:
        dev = pickle.loads((CN.CHECK2 / "gen" / f"dev-{name}.pkl").read_bytes())
        out += [(name, r) for r in (dev[:LIMIT] if LIMIT else dev)]
    return out


def setup_info():
    info = json.loads((A3.OUT / "prepare.json").read_text())
    return {lin: (info[lin]["network"], str(A3.OUT / f"library-{lin}.npz")) for lin in LINEAGES}


def pool(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    return ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_worker_init)


# ---------------------------------------------------------------------------------------------------------------------
# check 4: Reanalyse
# ---------------------------------------------------------------------------------------------------------------------

def prior_logp(spec, nd, sctx, legal, vocab, i, cache):
    """log probabilities of the legal additions under a lineage prior: ("grammar", table), ("net", network) or
    ("blend", network, table)."""
    import numpy as np
    if spec[0] == "grammar":
        return np.log(np.maximum(softmax_legal(load_tab(spec[1], vocab)[:, i], legal), 1e-300))
    if spec[1] not in cache:
        cache[spec[1]] = load_net(spec[1])
    pn, _ = net_dist(cache[spec[1]], nd, sctx, legal)
    if spec[0] == "net":
        return np.log(np.maximum(pn, 1e-300))
    lib = load_tab(spec[2], vocab)[:, i]
    return np.log(np.maximum(blend_dist(lib, pn, legal), 1e-300))


def check4_job(args):
    lin, rec, specs = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    cache, rows = {}, []
    for i, keys in enumerate(rec["structure"]):
        if not keys:
            continue
        eq = Eq(ctx, fitter, i)
        _, cases = path_cases(keys, vocab)
        for t, point, S in cases:
            nd = eq.node(S)
            legal = legal_adds(vocab.V, S)
            row = {"lineage": lin, "id": rec["id"], "level": rec["level"], "eq": i, "key": vocab.keys[t], "point": point}
            for role, spec in specs.items():
                row[role] = rank(prior_logp(spec, nd, eq.se.ctx, legal, vocab, i, cache), legal, t)
            rows.append(row)
    return rows


def lineage_history(lin):
    """The lineage's record, its directory, the prior spec after n learning steps, and per training system the number
    of learning steps before its last attempt and before the round that solved it."""
    base = BLEND if lin.startswith("net-blend") else PILOT
    rec = json.loads((base / "lineages" / f"{lin}.json").read_text())
    d = base / "lineages" / lin
    n, last, solved_at = 0, {}, {}
    for L in rec["levels"]:
        for rd in L["rounds"]:
            for y in rd["wake"]:
                last[y["id"]] = n
                if y["solved"] and y["id"] not in solved_at:
                    solved_at[y["id"]] = n
            n += 1

    def spec(k):
        if k == 0:
            return ("grammar", str(d / "size.npy"))
        if lin.startswith("net-blend"):
            return ("blend", str(d / f"learn{k}"), str(d / f"library{k}.npy"))
        return ("net", str(d / f"learn{k}"))
    mastered = {L["level"] for L in rec["levels"] if L["mastered"]}
    return rec, spec, n, last, solved_at, mastered


def step_check4(workers):
    import numpy as np
    jobs, info = [], {}
    for lin in ("net-s1", "net-s2", "net-blend-s1", "net-blend-s2"):
        s = int(lin.rsplit("-s", 1)[1])
        systems = {r["id"]: r for lv, d in json.loads((PILOT / "gen" / f"systems-s{s}.json").read_text()).items() for r in d["train"]}
        rec, spec, N, last, solved_at, mastered = lineage_history(lin)
        unsolved = [sid for sid in last if sid not in solved_at and systems[sid]["level"] in mastered]
        solved = [sid for sid in solved_at if systems[sid]["level"] in mastered]
        info[lin] = {"learning steps": N, "unsolved at mastered levels": len(unsolved), "solved at mastered levels": len(solved),
                     "unsolved by level": {lv: sum(systems[x]["level"] == lv for x in unsolved) for lv in sorted(mastered)}}
        jobs += [(lin, systems[sid], {"searching": spec(last[sid]), "final": spec(N)}) for sid in unsolved]
        jobs += [(lin, systems[sid], {"solving": spec(solved_at[sid])}) for sid in solved]
    t0 = time.time()
    with pool(workers) as ex:
        rows = [x for part in ex.map(check4_job, jobs) for x in part]
    (OUT / "check4.json").write_text(json.dumps({"rows": rows, "info": info}))
    print(f"check4: {len(rows)} rows from {len(jobs)} systems in {time.time() - t0:.0f}s; " + json.dumps(info))


def report4():
    import numpy as np
    d = json.loads((OUT / "check4.json").read_text())
    rows, info = d["rows"], d["info"]
    lines = ["== Check 4 (Reanalyse): training systems left unsolved at mastered levels; needed-term ranks under the prior that "
             "last tried them vs the lineage's final prior"]
    for lin, v in info.items():
        lines.append(f"   {lin}: {v}")
    reach = {}
    for lin in info:
        R = [x["solving"] for x in rows if x["lineage"] == lin and "solving" in x and x["point"] == "one short"]
        reach[lin] = float(np.percentile(R, 90)) if R else float("nan")
    ok = True
    for s in (1, 2):
        U = [x for x in rows if x["lineage"].endswith(f"-s{s}") and "final" in x]
        if not U:
            lines.append(f"   seed {s}: no unsolved systems")
            ok = False
            continue
        better = float(np.mean([x["final"] < x["searching"] for x in U]))
        one = [x for x in U if x["point"] == "one short"]
        within_f = float(np.mean([x["final"] <= reach[x["lineage"]] for x in one])) if one else float("nan")
        within_s = float(np.mean([x["searching"] <= reach[x["lineage"]] for x in one])) if one else float("nan")
        a, b = better >= 2 / 3, within_f >= 0.5 and within_f > within_s
        ok &= a and b
        med = {role: [float(np.median([x[role] for x in U if x["point"] == pt])) if any(x["point"] == pt for x in U) else float("nan")
                      for pt in POINTS] for role in ("searching", "final")}
        lines.append(f"   seed {s}: {len(U)} cases ({len(one)} at one short); final prior strictly better in {better:.0%} (rule: >= 67%) "
                     f"{a}; within reach at one short: final {within_f:.0%} vs searching {within_s:.0%} (rule: >= 50% and more) {b}; "
                     f"median ranks searching {' / '.join(f'{m:.0f}' for m in med['searching'])} -> final "
                     f"{' / '.join(f'{m:.0f}' for m in med['final'])} (empty / one short / two short); reach "
                     + ", ".join(f"{k} {v:.0f}" for k, v in reach.items() if k.endswith(f"-s{s}")))
    lines.append(f"== Check 4: {'PASSES' if ok else 'does not pass'}")
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# check 2: sampled exploration
# ---------------------------------------------------------------------------------------------------------------------

M_DRAWS = {"empty start": 45, "one short": 10, "two short": 10}
N_MC = 2000


def inclusion(w, t, w_other, n_other, m, rng):
    """P(move 'add t' is among m draws without repeats) when additions have weights w (0 = illegal) and n_other other
    moves weigh w_other each: Gumbel-top-m, N_MC draws; the other moves above t are counted by a binomial draw."""
    import numpy as np
    if not w[t] > 0:
        return 0.0
    mask = w > 0
    lw = np.log(w[mask])
    pos = int(np.flatnonzero(mask).tolist().index(t))
    keys = lw[None, :] + rng.gumbel(size=(N_MC, len(lw)))
    kt = keys[:, pos]
    above = (keys > kt[:, None]).sum(1)
    if n_other and w_other > 0:
        p = -np.expm1(-np.exp(-(kt - math.log(w_other))))
        above = above + rng.binomial(n_other, np.clip(p, 0.0, 1.0))
    return float((above < m).mean())


def check2_job(args):
    import numpy as np
    import search_g1 as SG
    combo, rec, lin, net_path, tab_path, pats = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab = load_net(net_path), load_tab(tab_path, vocab)
    rng = np.random.default_rng(stable_seed(rec["id"], lin, "exploration"))
    rows = []
    for i in range(len(rec["structure"])):
        _, cases = heldout_cases(rec, i, vocab, pats)
        if not cases:
            continue
        eq = Eq(ctx, fitter, i)
        memo = {}
        for t, kind, point, S in cases:
            if S not in memo:
                nd = eq.node(S)
                legal = legal_adds(vocab.V, S)
                pn, _ = net_dist(net, nd, eq.se.ctx, legal)
                lib = softmax_legal(tab[:, i], legal)
                flat = np.where(legal, np.sqrt(pn), 0.0)
                memo[S] = (legal, pn, {"library": lib, "flattened network": flat / flat.sum(),
                                       "blend": blend_dist(tab[:, i], pn, legal)})
            legal, pn, props = memo[S]
            nS, n_out = len(S), int(legal.sum())
            n_moves = n_out + nS + nS * n_out
            m = M_DRAWS[point]
            w_add = 1.0 if nS == 0 else SG.WEIGHTS["add"] / sum(SG.WEIGHTS.values())
            n_other = nS + nS * n_out
            w_other = (1.0 - w_add) / n_other if n_other else 0.0
            top20 = np.argsort(-np.where(legal, pn, -1.0))[:20]
            row = {"lineage": lin, "combo": combo, "eq": i, "key": vocab.keys[t], "kind": kind, "point": point, "m": m,
                   "uniform": min(1.0, m / n_moves)}
            for name, beta in props.items():
                row[name] = inclusion(w_add * beta, t, w_other, n_other, m, rng)
                row[f"{name} mass on network top 20"] = float(beta[top20].sum())
            rows.append(row)
    return rows


def step_check2(workers):
    setup = setup_info()
    jobs = []
    for combo, rec in heldout_jobs():
        _, _, _, pats = __import__("curriculum_check2").combos()[combo]
        for lin in LINEAGES:
            jobs.append((combo, rec, lin, setup[lin][0], setup[lin][1], pats))
    t0 = time.time()
    with pool(workers) as ex:
        rows = [x for part in ex.map(check2_job, jobs) for x in part]
    (OUT / "check2.json").write_text(json.dumps(rows))
    print(f"check2: {len(rows)} rows from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def report2():
    import numpy as np
    rows = json.loads((OUT / "check2.json").read_text())
    props = ("library", "flattened network", "blend")
    lines = ["== Check 2 (sampled exploration): mean chance that the exploration draws include the needed term "
             "(m = 45 at the empty start, 10 at one and two short); empty start / one short / two short"]
    passing = {p: True for p in props}
    for s, lin in zip((1, 2), LINEAGES):
        for kind in ("new", "familiar"):
            cells = {}
            for col in ("uniform",) + props:
                cells[col] = [float(np.mean([x[col] for x in rows if x["lineage"] == lin and x["kind"] == kind and x["point"] == pt]))
                              if any(x["lineage"] == lin and x["kind"] == kind and x["point"] == pt for x in rows) else float("nan")
                              for pt in POINTS]
            lines.append(f"   seed {s}, {kind}: " + "; ".join(f"{c} {' / '.join(f'{v:.0%}' for v in cells[c])}" for c in cells))
            if kind == "new":
                for p in props:
                    passing[p] &= all(cells[p][j] - cells["uniform"][j] >= 0.20 for j in range(3) if not math.isnan(cells[p][j]))
        dup = {p: float(np.mean([x[f"{p} mass on network top 20"] for x in rows if x["lineage"] == lin])) for p in props}
        lines.append(f"   seed {s}: mean share of each proposal's mass on the network's top 20: "
                     + ", ".join(f"{k} {v:.0%}" for k, v in dup.items()))
    ok = any(passing.values())
    lines.append(f"== Check 2: {'PASSES' if ok else 'does not pass'} (a proposal >= 20 points above uniform on new terms at all "
                 f"three points, both seeds): " + ", ".join(f"{p} {v}" for p, v in passing.items()))
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# check 3: the Gumbel root rule
# ---------------------------------------------------------------------------------------------------------------------

M_CAND, N_GUMBEL, C_VISIT = 16, 20, 50
C_SCALES = (0.1, 1.0)


def check3_job(args):
    import numpy as np
    combo, rec, lin, net_path, tab_path, pats = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab = load_net(net_path), load_tab(tab_path, vocab)
    rng = np.random.default_rng(stable_seed(rec["id"], lin, "gumbel"))
    rows, fits, fit_s = [], 0, 0.0
    for i in range(len(rec["structure"])):
        _, cases = heldout_cases(rec, i, vocab, pats)
        if not cases:
            continue
        eq = Eq(ctx, fitter, i)
        memo = {}
        for t, kind, point, S in cases:
            if S not in memo:
                nd = eq.node(S)
                legal = legal_adds(vocab.V, S)
                pn, _ = net_dist(net, nd, eq.se.ctx, legal)
                per = {}
                for pname, dist in (("blend", blend_dist(tab[:, i], pn, legal)), ("network", pn)):
                    logit = np.where(legal, np.log(np.maximum(dist, 1e-300)), -np.inf)
                    top2 = set(np.argsort(-logit)[:2].tolist())
                    draws = []
                    for _ in range(N_GUMBEL):
                        g = rng.gumbel(size=vocab.V)
                        cand = np.argsort(-(g + logit))[:M_CAND]
                        q = np.array([eq.node(S + (int(c),), nd)["value"] for c in cand])
                        span = q.max() - q.min()
                        qh = (q - q.min()) / span if span > 0 else np.zeros_like(q)
                        fin = {}
                        for cs in C_SCALES:
                            sc = g[cand] + logit[cand] + (C_VISIT + 1) * cs * qh
                            fin[cs] = set(cand[np.argsort(-sc)[:2]].tolist())
                        best = set(cand[np.lexsort((-logit[cand], -q))[:2]].tolist())
                        draws.append((set(cand.tolist()), fin, best))
                    per[pname] = (top2, draws)
                memo[S] = per
            row = {"lineage": lin, "combo": combo, "eq": i, "key": vocab.keys[t], "kind": kind, "point": point}
            for pname, (top2, draws) in memo[S].items():
                row[f"{pname} top2"] = float(t in top2)
                row[f"{pname} sampled"] = float(np.mean([t in c for c, _, _ in draws]))
                row[f"{pname} best2 fits"] = float(np.mean([t in b for _, _, b in draws]))
                for cs in C_SCALES:
                    row[f"{pname} finalists c{cs}"] = float(np.mean([t in f[cs] for _, f, _ in draws]))
            rows.append(row)
        fits += eq.fits
        fit_s += eq.fit_seconds
    return {"rows": rows, "fits": fits, "fit_seconds": fit_s}


def step_check3(workers):
    setup = setup_info()
    jobs = []
    for combo, rec in heldout_jobs():
        _, _, _, pats = __import__("curriculum_check2").combos()[combo]
        for lin in LINEAGES:
            jobs.append((combo, rec, lin, setup[lin][0], setup[lin][1], pats))
    t0 = time.time()
    with pool(workers) as ex:
        res = list(ex.map(check3_job, jobs))
    rows = [x for r in res for x in r["rows"]]
    fits, fs = sum(r["fits"] for r in res), sum(r["fit_seconds"] for r in res)
    (OUT / "check3.json").write_text(json.dumps({"rows": rows, "fits": fits, "fit_seconds": fs, "systems": len(jobs) // 2}))
    print(f"check3: {len(rows)} rows, {fits} fits ({fs / max(fits, 1):.2f} s each) from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def report3():
    import numpy as np
    d = json.loads((OUT / "check3.json").read_text())
    rows = d["rows"]
    lines = [f"== Check 3 (the Gumbel root rule; {d['systems']} held-out systems per lineage; {d['fits']} fits): how often the "
             "needed term is among the 2 finalists vs the prior's top 2 and the 2 best fits; empty start / one short / two short"]

    def m(lin, kind, col):
        return [float(np.mean([x[col] for x in rows if x["lineage"] == lin and x["kind"] == kind and x["point"] == pt]))
                if any(x["lineage"] == lin and x["kind"] == kind and x["point"] == pt for x in rows) else float("nan") for pt in POINTS]
    ok = True
    for pname in ("blend", "network"):
        for s, lin in zip((1, 2), LINEAGES):
            for kind in ("new", "familiar"):
                cols = [f"{pname} top2", f"{pname} sampled", f"{pname} best2 fits"] + [f"{pname} finalists c{cs}" for cs in C_SCALES]
                lines.append(f"   {pname}, seed {s}, {kind}: " + "; ".join(
                    f"{c.replace(pname + ' ', '')} {' / '.join(f'{v:.0%}' for v in m(lin, kind, c))}" for c in cols))
                if pname == "blend":
                    fin, top = m(lin, kind, "blend finalists c0.1"), m(lin, kind, "blend top2")
                    if kind == "new":
                        ok &= all(fin[j] > top[j] for j in (0, 2) if not math.isnan(fin[j]))
                    else:
                        ok &= all(fin[j] >= top[j] - 0.05 for j in range(3) if not math.isnan(fin[j]))
    lines.append(f"== Check 3: {'PASSES' if ok else 'does not pass'} (primary: blend prior, c_scale 0.1; new terms more often "
                 "among the finalists than in the top 2 at the empty start and two short; familiar within 5 points at all points)")
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# check 1: a learned fit predictor
# ---------------------------------------------------------------------------------------------------------------------

N_TOP1 = 30
FEATURES = ("gain", "analytic", "v_node", "n_terms", "size", "n_const", "net_logp", "lib_logp")


def term_shape(vocab):
    import numpy as np
    import fitter_g1 as FG
    import library_ops as L

    def size(tr):
        return 1 + sum(size(c) for c in tr[1:]) if isinstance(tr, tuple) else 1
    sz = np.array([size(L.tree(k, 0)) for k in vocab.keys], dtype=float)
    nk = np.array([FG.info(k).k for k in vocab.keys], dtype=float)
    return sz, nk


def node_features(eq, nd, S, legal, pn, lib_logp, shape):
    """Cheap features of every legal addition at a node (no fits): evidence gain, the analytic fit estimate, the node's
    own fit quality and size, term size and constants, network and library log-probabilities."""
    import numpy as np
    import search_g1 as SG
    t0 = time.perf_counter()
    gain = SG.evidence(nd, eq.se.ctx)[0]
    rel = (nd["res"] / eq.se.yn) if math.isfinite(nd["res"]) else 1.0
    after = rel * np.sqrt(np.clip(1.0 - gain, 1e-28, 1.0))
    analytic = np.clip(np.log10(1.0 / np.maximum(after, 1e-14)) / 14.0, 0.0, 1.0)
    X = np.stack([gain, analytic, np.full(len(gain), nd["value"]), np.full(len(gain), len(S)), shape[0], shape[1],
                  np.log(np.maximum(pn, 1e-300)), lib_logp], 1).astype(np.float32)
    return X, gain, time.perf_counter() - t0


def check1_train_job(args):
    """Fits at the points along the lineage's verified solutions of one training system: candidates = the network's top 30,
    the evidence's top 30 and the solution's missing terms; plus every legal addition's cheap features."""
    import numpy as np
    lin, rec, sols, net_path, tab_path = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab, shape = load_net(net_path), load_tab(tab_path, vocab), term_shape(vocab)
    nodes, feat_s = [], 0.0
    fits, fit_s = 0, 0.0
    for eqs, keys in sols.items():
        i = int(eqs)
        true, cases = path_cases(keys, vocab)
        if not true:
            continue
        eq = Eq(ctx, fitter, i)
        for S in sorted({S for _, _, S in cases}, key=len):
            nd = eq.node(S)
            if not math.isfinite(nd["res"]):
                continue
            legal = legal_adds(vocab.V, S)
            pn, _ = net_dist(net, nd, eq.se.ctx, legal)
            lib_logp = np.log(np.maximum(softmax_legal(tab[:, i], legal), 1e-300))
            X, gain, fs = node_features(eq, nd, S, legal, pn, lib_logp, shape)
            feat_s += fs
            need = [u for u in true if u not in S]
            cand = set(np.argsort(-np.where(legal, pn, -1.0))[:N_TOP1].tolist()) | set(np.argsort(-np.where(legal, gain, -1.0))[:N_TOP1].tolist()) | set(need)
            lab = {int(c): float(eq.node(S + (int(c),), nd)["value"]) for c in sorted(cand)}
            blend = np.log(np.maximum(blend_dist(tab[:, i], pn, legal), 1e-300))
            nodes.append({"X": X, "legal": legal, "lab": lab, "need": need, "blend": blend.astype(np.float32)})
        fits += eq.fits
        fit_s += eq.fit_seconds
    return {"lineage": lin, "nodes": nodes, "fits": fits, "fit_seconds": fit_s, "feature_seconds": feat_s,
            "V": vocab.V}


def check1_test_job(args):
    import numpy as np
    combo, rec, lin, net_path, tab_path, pats = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab, shape = load_net(net_path), load_tab(tab_path, vocab), term_shape(vocab)
    out = []
    for i in range(len(rec["structure"])):
        _, cases = heldout_cases(rec, i, vocab, pats)
        if not cases:
            continue
        eq = Eq(ctx, fitter, i)
        memo = {}
        for t, kind, point, S in cases:
            if S not in memo:
                nd = eq.node(S)
                legal = legal_adds(vocab.V, S)
                pn, _ = net_dist(net, nd, eq.se.ctx, legal)
                lib_logp = np.log(np.maximum(softmax_legal(tab[:, i], legal), 1e-300))
                X, gain, _ = node_features(eq, nd, S, legal, pn, lib_logp, shape)
                blend = np.log(np.maximum(blend_dist(tab[:, i], pn, legal), 1e-300))
                memo[S] = {"X": X, "legal": legal, "blend": blend.astype(np.float32), "cases": []}
            memo[S]["cases"].append((int(t), kind, point, vocab.keys[t]))
        out += [dict(v, lineage=lin, combo=combo, eq=i) for v in memo.values()]
    return out


def gpu_device():
    import torch
    g = OPTS.get("gpu")
    return torch.device(f"cuda:{g}") if g is not None and torch.cuda.is_available() else None


def fit_mlp(X, y, device, classify=False, steps=3000, seed=1):
    """A small two-layer network (standardized inputs), trained full-batch with Adam on the given GPU."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        z = net(Xt)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) if classify else ((z - yt) ** 2).mean()
        loss.backward()
        opt.step()
    net = net.cpu().eval()

    def predict(Z):
        with torch.no_grad():
            return net(torch.tensor((Z - mu) / sd, dtype=torch.float32))[:, 0].numpy()
    return predict, float(loss)


W_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def step_check1(workers):
    import numpy as np
    setup = setup_info()
    device = gpu_device()
    if device is None:
        raise SystemExit("check1 needs a free GPU (gpu=G): small models are not trained on the CPU (owner's rule)")
    jobs = []
    for lin in LINEAGES:
        s = int(lin.rsplit("-s", 1)[1])
        systems = {r["id"]: r for lv, d in json.loads((PILOT / "gen" / f"systems-s{s}.json").read_text()).items() for r in d["train"]}
        sols, level = CN.solutions(lin)
        for sid, d in sols.items():
            if level[sid] in FAMILIAR:
                jobs.append((lin, systems[sid], d, setup[lin][0], setup[lin][1]))
    t0 = time.time()
    with pool(workers) as ex:
        train = list(ex.map(check1_train_job, jobs))
        tjobs = []
        for combo, rec in heldout_jobs():
            _, _, _, pats = __import__("curriculum_check2").combos()[combo]
            tjobs += [(combo, rec, lin, setup[lin][0], setup[lin][1], pats) for lin in LINEAGES]
        test = [x for part in ex.map(check1_test_job, tjobs) for x in part]
    fits, fit_s = sum(r["fits"] for r in train), sum(r["fit_seconds"] for r in train)
    feat_s = sum(r["feature_seconds"] for r in train)
    n_feat_nodes = sum(len(r["nodes"]) for r in train)
    V = train[0]["V"]
    res = {"fits": fits, "seconds_per_fit": fit_s / max(fits, 1), "feature_seconds_per_candidate": feat_s / max(n_feat_nodes * V, 1),
           "lineages": {}}
    for lin in LINEAGES:
        nodes = [n for r in train if r["lineage"] == lin for n in r["nodes"]]
        X = np.concatenate([n["X"][list(n["lab"])] for n in nodes])
        y = np.concatenate([np.array(list(n["lab"].values()), dtype=np.float32) for n in nodes])
        predict, loss = fit_mlp(X, y, device)
        tp = time.perf_counter()
        _ = predict(nodes[0]["X"])
        mlp_s = (time.perf_counter() - tp) / len(nodes[0]["X"])

        pred = [predict(n["X"]) for n in nodes]

        def logrank(score_fn):
            r = [math.log(rank(score_fn(j, n), n["legal"], t)) for j, n in enumerate(nodes) for t in n["need"]]
            return float(np.mean(r)) if r else float("inf")
        w_l = min(W_GRID, key=lambda w: logrank(lambda j, n: n["blend"] + w * pred[j]))
        w_a = min(W_GRID, key=lambda w: logrank(lambda j, n: n["blend"] + w * n["X"][:, 1]))
        rows = []
        for n in [x for x in test if x["lineage"] == lin]:
            X, legal = n["X"], n["legal"]
            scores = {"network": X[:, 6], "library": X[:, 7], "evidence": X[:, 0], "blend": n["blend"],
                      "blend + learned fit": n["blend"] + w_l * predict(X), "blend + analytic fit": n["blend"] + w_a * X[:, 1]}
            for t, kind, point, key in n["cases"]:
                rows.append(dict({k: rank(v, legal, t) for k, v in scores.items()}, kind=kind, point=point, key=key,
                                 combo=n["combo"], eq=n["eq"]))
        res["lineages"][lin] = {"rows": rows, "w_learned": w_l, "w_analytic": w_a, "train_loss": loss,
                                "train_points": len(nodes), "train_labels": int(len(y)), "mlp_seconds_per_candidate": mlp_s}
    (OUT / "check1.json").write_text(json.dumps(res))
    print(f"check1: {fits} training fits ({res['seconds_per_fit']:.2f} s each); {len(test)} held-out nodes; {time.time() - t0:.0f}s")


def report1():
    import numpy as np
    d = json.loads((OUT / "check1.json").read_text())
    cols = ("network", "library", "evidence", "blend", "blend + learned fit", "blend + analytic fit")
    lines = [f"== Check 1 (a learned fit predictor; {d['fits']} training fits at {d['seconds_per_fit']:.2f} s each): median rank "
             "of the needed term (of about 1,100), empty start / one short / two short"]
    ok = True
    for s, lin in zip((1, 2), LINEAGES):
        L_ = d["lineages"][lin]
        rows = L_["rows"]
        med = {k: {c: [float(np.median([x[c] for x in rows if x["kind"] == k and x["point"] == pt]))
                       if any(x["kind"] == k and x["point"] == pt for x in rows) else float("nan") for pt in POINTS] for c in cols}
               for k in ("new", "familiar")}
        for k in ("new", "familiar"):
            lines.append(f"   seed {s}, {k}: " + "; ".join(f"{c} {' / '.join(f'{v:.0f}' for v in med[k][c])}" for c in cols))
        lp = med["new"]["blend + learned fit"]
        better = all(lp[j] < min(med["new"][c][j] for c in ("network", "library", "evidence", "blend")) for j in (1, 2)
                     if not math.isnan(lp[j]))
        empty = lp[0] <= med["new"]["blend"][0]
        fam = all(med["familiar"]["blend + learned fit"][j] <= 2 * med["familiar"]["network"][j] for j in range(3)
                  if not math.isnan(med["familiar"]["network"][j]))
        cost = (d["feature_seconds_per_candidate"] + L_["mlp_seconds_per_candidate"]) / d["seconds_per_fit"]
        ok &= better and empty and fam and cost < 0.01
        lines.append(f"   seed {s}: weights chosen on training points: learned {L_['w_learned']}, analytic {L_['w_analytic']}; "
                     f"{L_['train_labels']} labels at {L_['train_points']} training points; new terms better than every ordering at "
                     f"one and two short {better}; no worse than the blend at the empty start {empty}; familiar within 2x the network "
                     f"{fam}; cost per candidate {cost:.2%} of one fit (rule: < 1%)")
    lines.append(f"== Check 1: {'PASSES' if ok else 'does not pass'}")
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# check 5: the value of a partial equation
# ---------------------------------------------------------------------------------------------------------------------

VFEATURES = ("v", "gain1", "gain2", "gain3", "rem_min", "rem_max", "n_terms", "net_add_max", "net_rem_min", "net_rem_max",
             "lib_sum", "lib_min")


def value_features(eq, nd, S, net, tab_i, V):
    import numpy as np
    import search_g1 as SG
    gain, rem = SG.evidence(nd, eq.se.ctx)
    legal = legal_adds(V, S)
    g = np.sort(np.where(legal, gain, 0.0))[::-1][:3]
    pn, pr = net_dist(net, nd, eq.se.ctx, legal)
    prem = pr.p_rem / pr.p_rem.sum() if len(S) and pr.p_rem.sum() > 0 else np.zeros(max(len(S), 1))
    lib = np.array([tab_i[x] for x in S]) if S else np.zeros(1)
    rem = np.log10(np.maximum(rem, 1e-30)) if len(S) else np.zeros(1)
    return [float(nd["value"]), float(g[0]), float(g[1]), float(g[2]), float(rem.min()), float(rem.max()), float(len(S)),
            float(pn.max()), float(prem.min()), float(prem.max()), float(lib.sum()), float(lib.min())], float(g[0])


def value_nodes(eq, true, net, tab_i, vocab):
    """On-path nodes (the structure minus one or two terms, at least one term left) and their off-path partners:
    (S - {a}) + {w}, w the network's highest-ranked addition outside the structure at S - {a}; partners whose w can stand
    in for a structure term ((true - {b}) + {w} fits exactly) are dropped and counted."""
    import numpy as np
    from itertools import combinations
    rows, dropped = [], 0
    ons = {tuple(sorted(set(true) - set(M))) for k in (1, 2) if len(true) - k >= 1 for M in combinations(true, k)}
    for S in sorted(ons):
        nd = eq.node(S)
        if not math.isfinite(nd["res"]):
            continue
        f, best = value_features(eq, nd, S, net, tab_i, vocab.V)
        rows.append({"S": [vocab.keys[x] for x in S], "label": 1, "x": f, "v": f[0], "best_gain": best})
        for a in S:
            Sp = tuple(x for x in S if x != a)
            ndp = eq.node(Sp)
            legal = legal_adds(vocab.V, Sp)
            legal[list(true)] = False
            pn, _ = net_dist(net, ndp, eq.se.ctx, legal_adds(vocab.V, Sp))
            w = int(np.argmax(np.where(legal, pn, -1.0)))
            if any(eq.node(tuple(sorted((set(true) - {b}) | {w})))["exact"] for b in true):
                dropped += 1
                continue
            P = tuple(sorted(Sp + (w,)))
            ndq = eq.node(P, ndp)
            if not math.isfinite(ndq["res"]):
                continue
            f, best = value_features(eq, ndq, P, net, tab_i, vocab.V)
            rows.append({"S": [vocab.keys[x] for x in P], "label": 0, "x": f, "v": f[0], "best_gain": best})
    return rows, dropped


def check5_train_job(args):
    lin, rec, sols, net_path, tab_path = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab = load_net(net_path), load_tab(tab_path, vocab)
    rows, dropped = [], 0
    for eqs, keys in sols.items():
        i = int(eqs)
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        if len(true) < 2:
            continue
        r, dr = value_nodes(Eq(ctx, fitter, i), true, net, tab[:, i], vocab)
        rows += r
        dropped += dr
    return {"lineage": lin, "rows": rows, "dropped": dropped}


def check5_test_job(args):
    import curriculum_check as CC
    import curriculum_check2 as C2
    combo, rec, lin, net_path, tab_path, pats = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net, tab = load_net(net_path), load_tab(tab_path, vocab)
    rows, dropped = [], 0
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        if len(true) < 2:
            continue
        has_new = any(C2.is_new(vocab.keys[t], i, pats) for t in true)
        familiar_only = all(CC.level_of(vocab.keys[t], i) in FAMILIAR for t in true)
        group = "new" if has_new else ("familiar" if familiar_only else None)
        if group is None:
            continue
        r, dr = value_nodes(Eq(ctx, fitter, i), true, net, tab[:, i], vocab)
        rows += [dict(x, group=group, combo=combo, eq=i, lineage=lin) for x in r]
        dropped += dr
    return {"lineage": lin, "rows": rows, "dropped": dropped}


def auc(scores, labels):
    """Probability that a random on-path node scores above a random off-path one (ties count half)."""
    import numpy as np
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def step_check5(workers):
    import numpy as np
    setup = setup_info()
    device = gpu_device()
    if device is None:
        raise SystemExit("check5 needs a free GPU (gpu=G): small models are not trained on the CPU (owner's rule)")
    jobs = []
    for lin in LINEAGES:
        s = int(lin.rsplit("-s", 1)[1])
        systems = {r["id"]: r for lv, d in json.loads((PILOT / "gen" / f"systems-s{s}.json").read_text()).items() for r in d["train"]}
        sols, level = CN.solutions(lin)
        jobs += [(lin, systems[sid], d, setup[lin][0], setup[lin][1]) for sid, d in sols.items() if level[sid] in FAMILIAR]
    t0 = time.time()
    with pool(workers) as ex:
        train = list(ex.map(check5_train_job, jobs))
        tjobs = []
        for combo, rec in heldout_jobs():
            _, _, _, pats = __import__("curriculum_check2").combos()[combo]
            tjobs += [(combo, rec, lin, setup[lin][0], setup[lin][1], pats) for lin in LINEAGES]
        test = list(ex.map(check5_test_job, tjobs))
    res = {"lineages": {}}
    for lin in LINEAGES:
        tr = [x for r in train if r["lineage"] == lin for x in r["rows"]]
        te = [x for r in test if r["lineage"] == lin for x in r["rows"]]
        X = np.array([x["x"] for x in tr], dtype=np.float32)
        y = np.array([x["label"] for x in tr], dtype=np.float32)
        predict, loss = fit_mlp(X, y, device, classify=True)
        out = {"train_nodes": len(tr), "train_on_path": int(y.sum()), "train_dropped": sum(r["dropped"] for r in train if r["lineage"] == lin),
               "test_dropped": sum(r["dropped"] for r in test if r["lineage"] == lin), "groups": {}}
        for g in ("new", "familiar"):
            G_ = [x for x in te if x["group"] == g]
            if not G_:
                continue
            lab = [x["label"] for x in G_]
            pv = predict(np.array([x["x"] for x in G_], dtype=np.float32))
            out["groups"][g] = {"nodes": len(G_), "on_path": int(sum(lab)), "auc_fit_quality": auc([x["v"] for x in G_], lab),
                                "auc_best_evidence": auc([x["best_gain"] for x in G_], lab), "auc_learned": auc(pv, lab)}
        res["lineages"][lin] = out
    (OUT / "check5.json").write_text(json.dumps(res))
    print(f"check5: {len(jobs)} training systems, {len(tjobs)} held-out jobs in {time.time() - t0:.0f}s")


def report5():
    d = json.loads((OUT / "check5.json").read_text())
    lines = ["== Check 5 (the value of a partial equation): AUC of on-path vs off-path partial equations"]
    ok = True
    for s, lin in zip((1, 2), LINEAGES):
        L_ = d["lineages"][lin]
        for g, v in L_["groups"].items():
            lines.append(f"   seed {s}, {g} equations: {v['nodes']} nodes ({v['on_path']} on-path); fit quality {v['auc_fit_quality']:.2f}, "
                         f"best one-term evidence {v['auc_best_evidence']:.2f}, learned value {v['auc_learned']:.2f}")
        n, f = L_["groups"].get("new"), L_["groups"].get("familiar")
        a = n is not None and n["auc_learned"] >= 0.75 and n["auc_learned"] >= n["auc_fit_quality"] + 0.10
        b = f is None or f["auc_learned"] >= f["auc_fit_quality"]
        ok &= a and b
        lines.append(f"   seed {s}: training nodes {L_['train_nodes']} ({L_['train_on_path']} on-path); partners dropped as stand-ins: "
                     f"training {L_['train_dropped']}, held-out {L_['test_dropped']}; new-term rule {a}; familiar rule {b}")
    lines.append(f"== Check 5: {'PASSES' if ok else 'does not pass'}")
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# timing slice (no outcomes) and the report
# ---------------------------------------------------------------------------------------------------------------------

def time_job(args):
    import numpy as np
    combo, rec, net_path, tab_path = args
    ctx, fitter = system(rec)
    vocab = ctx["vocab"]
    net = load_net(net_path)
    fits, fs = 0, 0.0
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        if not true:
            continue
        eq = Eq(ctx, fitter, i)
        for S in ((), tuple(true[1:])):
            nd = eq.node(S)
            legal = legal_adds(vocab.V, S)
            pn, _ = net_dist(net, nd, eq.se.ctx, legal)
            for c in np.argsort(-np.where(legal, pn, -1.0))[:12]:
                eq.node(S + (int(c),), nd)
        fits += eq.fits
        fs += eq.fit_seconds
    return fits, fs


def step_time(workers):
    setup = setup_info()
    lin = LINEAGES[0]
    jobs = [(combo, rec, setup[lin][0], setup[lin][1]) for combo, rec in heldout_jobs()[::40][:workers]]
    t0 = time.time()
    with pool(workers) as ex:
        res = list(ex.map(time_job, jobs))
    fits, fs = sum(r[0] for r in res), sum(r[1] for r in res)
    wall = time.time() - t0
    print(f"time: {fits} fits in {len(jobs)} parallel jobs; {fs / max(fits, 1):.3f} s of fitting per fit; wall {wall:.0f}s")
    (OUT / "timing.json").write_text(json.dumps({"fits": fits, "seconds_per_fit": fs / max(fits, 1), "workers": workers, "wall": wall}))


def step_report():
    lines = []
    for name, fn in (("check4", report4), ("check2", report2), ("check3", report3), ("check1", report1), ("check5", report5)):
        if (OUT / f"{name}.json").exists():
            lines += fn()
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
    {"check4": lambda: step_check4(workers), "check2": lambda: step_check2(workers), "check3": lambda: step_check3(workers),
     "check1": lambda: step_check1(workers), "check5": lambda: step_check5(workers), "time": lambda: step_time(workers),
     "report": step_report}[step]()


if __name__ == "__main__":
    main()
