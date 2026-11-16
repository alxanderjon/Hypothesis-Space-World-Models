"""The dream quick check (owner, 15 November; research log "Owner: fix the dreams and the wake pool before rerunning a1",
where the design and pass rule were set before the run). From each learner arm's end-of-curriculum state (seeds 1-3),
one learning round with: a. the current dreams; b. +1,000 rung-shaped dreams from the rung's generator (3 terms, x, x*y,
x^2); c. +1,000 rung-shaped dreams of 3 terms drawn from the pieces of the learner's own solutions, weighted by its own
library grammar. Real solutions held at 30% of the examples; steps scaled to keep each example's views. Then the ranks of
the true terms of new a1-style development systems, and of curriculum held-out terms, at the empty start, one short and
two short. No search. Usage: dream_check.py <workers> <gen|train|rank|report>"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

RUNG = os.environ.get("DREAMCHECK_RUNG", "a1")                   # a1: from the curriculum; a2: from the a1w state
os.environ.setdefault("G2H_RUN", "dreamcheck" if RUNG == "a1" else f"dreamcheck_{RUNG}")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

C, G2H, OUT = LD.C, LD.G2H, LD.OUT
SEEDS, ARMS = (1, 2, 3), ("blend", "net")
N_DEV, N_DREAM, N_GEN, SHARE_REAL, BASE_STEPS = 20, 1000, 50, 0.30, 500
A1_LEVELS = (1, 2, 3) if RUNG == "a1" else (1, 2, 3, 5, 6, 7)
CUR = {1: "cpilot/gen/systems-s1.json", 2: "cpilot/gen/systems-s2.json", 3: "cseed3/gen/systems-s3.json"}


def pool(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    return ProcessPoolExecutor(workers, mp_context=get_context("spawn"))


def dream_job(args):
    """One accepted rung-shaped dream and its examples: ("gen", allowed) from the rung's generator, or ("lib", keys,
    weights) for 3 distinct terms drawn from the learner's own pieces by its grammar weights."""
    prefix, k, how = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        if how[0] == "gen":
            desc = LD.draw(seed, how[1], (3, 3), 2)
        elif how[0] == "bal":
            # piece-balanced (owner, 16 Nov): a piece the learner has used, uniformly, then a term of it, uniformly
            groups = how[1]
            rng = np.random.default_rng(LD.stable(seed, "bal"))
            picks_per_eq = []
            for _ in range(2):
                picks = []
                while len(picks) < min(3, sum(len(g) for g in groups)):
                    g = groups[int(rng.integers(len(groups)))]
                    key = g[int(rng.integers(len(g)))]
                    if key not in picks:
                        picks.append(key)
                picks_per_eq.append(picks)
            desc = draw_fixed(seed, picks_per_eq)
        else:
            keys, w = how[1], np.asarray(how[2], dtype=float)
            rng = np.random.default_rng(LD.stable(seed, "lib"))
            allowed = {0: []}
            picks_per_eq = []
            for _ in range(2):
                idx = rng.choice(len(keys), size=min(3, len(keys)), replace=False, p=w / w.sum())
                picks_per_eq.append([keys[j] for j in idx])
            desc = draw_fixed(seed, picks_per_eq)
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
        except Exception:
            continue
        try:
            return DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
        except Exception:
            return []
    return []


def draw_fixed(seed, picks_per_eq):
    """ladder.draw with each equation's (role-relative) terms given."""
    import numpy as np
    import sympy as sp
    import torch
    import domains_g1 as DG
    import fitter_g1 as FG
    import language_g1 as LG
    d = 2
    rng = DG.rng_of(seed)
    Y, X = DG.Y4, LG.variables(4)
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    for i in range(d):
        actual = [i] + [j for j in range(d) if j != i]
        e = 0
        for t, key in enumerate(picks_per_eq[i]):
            expr = LG.expr_of(key).xreplace({X[r]: Y[actual[r]] for r in range(d)})
            inf = FG.info(key)
            sub = {}
            if inf.k:
                Wp = torch.tensor(np.asarray(ic) * rng.uniform(0.5, 1.5, (64, d)))
                th = FG.natural_thetas(inf, rng.random((1, inf.k)), Wp[:, actual])[0]
                if not np.all(np.isfinite(th)):
                    th = np.ones(inf.k)
                for a in range(inf.k):
                    name = f"c{i}_{t}_{a}"
                    params[name] = float(th[a])
                    sub[sp.Symbol(f"K{a}", real=True)] = sp.Symbol(name)
            cname = f"a{i}_{t}"
            params[cname] = float(rng.choice([-1.0, 1.0]) * math.exp(rng.uniform(math.log(0.1), math.log(2.0))))
            e = e + sp.Symbol(cname) * expr.xreplace(sub)
        rhs.append(e)
    return {"seed": seed, "domain": "fantasy", "form": "fantasy", "d": d, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False}


def starts(ex):
    """{(arm, seed): Lineage at its end-of-curriculum state} (no rung systems added)."""
    gpus = None
    return {(a, s): LD.Lineage(a, s, RUNG, ex, gpus) for a in ARMS for s in SEEDS}


def make_dev(ex, allowed):
    dev = {s: [r for r in ex.map(LD.gen_job, [(f"dreamcheck-{RUNG}-dev-s{s}", k, allowed, (3, 3), 2) for k in range(N_DEV)]) if r["accepted"]] for s in SEEDS}
    old = LD.earlier_keys("dreamcheck") | {LD.data_key(r["bursts"]) for p in (G2H / "ladder/gen").glob("*.json")
                                            for sp in json.loads(p.read_text()).values() for r in sp}
    clash = sum(LD.data_key(r["bursts"]) in old for s in SEEDS for r in dev[s])
    print(f"dev systems: {sum(len(v) for v in dev.values())} accepted; identical to earlier systems {clash}")
    if clash:
        raise SystemExit("development systems not new")
    for s in SEEDS:
        (OUT / "gen" / f"dev-s{s}.json").write_text(json.dumps([dict(r, id=f"dev-s{s}-{r['k']}") for r in dev[s]]))


def step_gen(workers):
    import numpy as np
    import dream_g1 as D1
    import dream_g2 as DR
    import search_g1 as SG
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    keys = LD.piece_keys(2)
    allowed = {lv: keys[lv] for lv in A1_LEVELS}
    order = SG.vocabulary_order(2)
    V = json.loads((SG.G1_DIR / "vocab_d2.json").read_text())["terms"]
    t0 = time.time()
    with pool(workers) as ex:
        # development systems (a1-style), checked against every earlier set (kept if an earlier attempt already drew them)
        if all((OUT / "gen" / f"dev-s{s}.json").exists() for s in SEEDS):
            print("dev systems: kept from the earlier attempt")
        else:
            make_dev(ex, allowed)
        # generator dreams (b), shared by both arms of a seed
        for s in SEEDS:
            p = OUT / "gen" / f"dreams-gen-s{s}.pkl"
            if not p.exists():
                exs = list(ex.map(dream_job, [(f"dreamcheck-{RUNG}-gen-s{s}", k, ("gen", allowed)) for k in range(N_DREAM)], chunksize=4))
                p.write_bytes(pickle.dumps([e for x in exs for e in x]))
                print(f"generator dreams s{s}: {sum(bool(x) for x in exs)} of {N_DREAM} with examples; {sum(len(x) for x in exs)} examples; {time.time() - t0:.0f}s")
        L = starts(ex)
        for (a, s), lin in L.items():
            name = f"{a}-s{s}"
            # replay examples of the lineage's verified solutions
            p = OUT / "gen" / f"replay-{name}.pkl"
            if not p.exists():
                rep = list(ex.map(LD.examples_job, [(lin.recs[sid]["bursts"], [lin.sols[sid].get(i) for i in range(len(lin.recs[sid]["structure"]))],
                                                    LD.stable(sid, "examples")) for sid in lin.sols]))
                p.write_bytes(pickle.dumps([e for x in rep for e in x]))
            # general dreams, as now (usage half, size prior half)
            p = OUT / "gen" / f"general-{name}.pkl"
            usage = {}
            for sid, eqs in lin.sols.items():
                for eq, ks in eqs.items():
                    for k in ks:
                        rk = DR.role_key(k, int(eq), 2)
                        if rk is not None:
                            usage[rk] = usage.get(rk, 0) + 1
            if not p.exists():
                probs = D1.term_distribution(order, V, usage).tolist()
                g = list(ex.map(LD.fantasy_job, [(f"dreamcheck-{RUNG}-{name}-general", k, probs, 2) for k in range(N_GEN)]))
                p.write_bytes(pickle.dumps([e for x in g for e in x]))
            # library dreams (c): the pieces of its own solutions, weighted by its own library grammar
            p = OUT / "gen" / f"dreams-lib-{name}.pkl"
            if not p.exists():
                tab = lin.tables.get(2)
                if tab is None:
                    eqs = [(eq, ks) for sid, d in lin.sols.items() for eq, ks in d.items()]
                    tab = str(OUT / "gen" / f"library-{name}.npy")
                    ex.submit(LD.table_job, (eqs, tab, 2)).result()
                T = np.load(tab)
                gid = {k: g for g, k in enumerate(order)}
                pieces = sorted({rk for rk in usage if rk in gid})
                w = [float(np.exp(T[gid[k], 0])) for k in pieces]
                exs = list(ex.map(dream_job, [(f"dreamcheck-{RUNG}-lib-{name}", k, ("lib", pieces, w)) for k in range(N_DREAM)], chunksize=4))
                p.write_bytes(pickle.dumps([e for x in exs for e in x]))
                print(f"library dreams {name}: {len(pieces)} pieces; {sum(bool(x) for x in exs)} of {N_DREAM} with examples; "
                      f"{sum(len(x) for x in exs)} examples; {time.time() - t0:.0f}s")


def step_train(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    gpus = LD.GPUs()
    info = {}
    with pool(workers) as ex:
        L = starts(ex)
    for (a, s), lin in L.items():
        name = f"{a}-s{s}"
        load = lambda f: pickle.loads((OUT / "gen" / f).read_bytes())
        replay, general = load(f"replay-{name}.pkl"), load(f"general-{name}.pkl")
        n_base = len(replay) + len(general)
        for ver, extra in (("a", []), ("b", load(f"dreams-gen-s{s}.pkl")), ("c", load(f"dreams-lib-{name}.pkl"))):
            out = OUT / "nets" / f"{name}-{ver}"
            out.parent.mkdir(parents=True, exist_ok=True)
            if ver == "a":
                exs = replay + general
            else:
                others = general + extra
                m = max(1, round(SHARE_REAL * len(others) / ((1 - SHARE_REAL) * max(len(replay), 1))))
                exs = replay * m + others
            steps = round(BASE_STEPS * len(exs) / n_base)
            if not Path(f"{out}.pt").exists():
                p = OUT / "nets" / f"{name}-{ver}_examples.pkl"
                p.write_bytes(pickle.dumps(exs))
                t0 = time.time()

                def run(g):
                    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
                        return one.submit(LD.train_net_job, ([str(p)], lin.net, str(out), g, LD.stable(name, ver), steps)).result(), g
                res, g = gpus.run(run)
                p.unlink()
                info[f"{name}-{ver}"] = {"examples": len(exs), "real share": len(replay) * (m if ver != "a" else 1) / len(exs),
                                         "steps": steps, "seconds": time.time() - t0, "gpu": g}
                LD.log(f"dreamcheck {name}-{ver}: {steps} steps on {len(exs)} examples on GPU {g} in {time.time() - t0:.0f}s")
    p = OUT / "train.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps(dict(old, **info), indent=1))


def rank_job(args):
    rec, spec, tag, kinds = args
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
    rows = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({v.index[k] for k in keys if k in v.index})
        eq = MZ.Eq(ctx, fitter, i)
        for t in true:
            kind = kinds(v.keys[t], i) if callable(kinds) else kinds
            if kind is None:
                continue
            others = [u for u in true if u != t]
            for point, S in [("empty start", ()), ("one short", tuple(others))] + ([("two short", tuple(others[1:]))] if len(true) >= 3 else []):
                nd = eq.node(S)
                legal = MZ.legal_adds(v.V, S)
                pr = prior(nd, eq.se.ctx)
                rows.append({"tag": tag, "kind": kind, "point": point, "rank": MZ.rank(np.log(np.maximum(np.where(legal, pr.p_swap_add, 0), 1e-300)), legal, t)})
    return rows


def step_rank(workers):
    jobs = []
    with pool(workers) as ex:
        L = starts(ex)
        for (a, s), lin in L.items():
            name = f"{a}-s{s}"
            dev = json.loads((OUT / "gen" / f"dev-s{s}.json").read_text())
            cur = json.loads((G2H / CUR[s]).read_text())
            cur = [r for lv in (1, 2, 3, 5, 6, 7) for r in cur[str(lv)]["test"][:3]]
            for ver in ("a", "b", "c"):
                net = str(OUT / "nets" / f"{name}-{ver}")
                spec = ("confidence", net, lin.tables) if a == "blend" else ("net", net)
                jobs += [(r, spec, f"{name}-{ver}", "dev") for r in dev] + [(r, spec, f"{name}-{ver}", "curriculum") for r in cur]
        t0 = time.time()
        rows = [x for part in ex.map(rank_job, jobs) for x in part]
    (OUT / "ranks.json").write_text(json.dumps(rows))
    print(f"rank: {len(rows)} rows from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def step_report_tail():
    """The a2 rule: badly ranked tail (share ranked worse than 10) and curriculum mean log2 rank (no ceiling medians)."""
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    pts = ("empty start", "one short", "two short")
    lines = ["== Dream quick check (" + RUNG + "): share of true terms ranked worse than 10 (empty / one short / two short); curriculum mean log2 rank"]
    wins = {"b": 0, "c": 0}
    bc = 0
    for a in ARMS:
        for s in SEEDS:
            name, m = f"{a}-s{s}", {}
            for v in "abc":
                R = [x for x in rows if x["tag"] == f"{name}-{v}"]
                dev = [x for x in R if x["kind"] == "dev"]
                tail = [float(np.mean([x["rank"] > 10 for x in dev if x["point"] == p])) for p in pts]
                cur = float(np.mean([np.log2(x["rank"]) for x in R if x["kind"] == "curriculum"]))
                m[v] = (tail, cur)
            judged = [0, 2] + ([1] if m["a"][0][1] >= 0.05 else [])
            for v in "bc":
                wins[v] += all(m[v][0][j] < m["a"][0][j] for j in judged) and m[v][1] <= m["a"][1] + 0.25
            bc += all(m["b"][0][j] < m["c"][0][j] - 0.05 for j in (0, 2))
            lines.append(f"   {name}: " + "; ".join(f"{v} tail {' / '.join(f'{x:.0%}' for x in m[v][0])}, curriculum {m[v][1]:.2f}" for v in "abc"))
    ok = {v: wins[v] >= 5 for v in "bc"}
    use = "c" if ok["c"] and (not ok["b"] or bc <= 3) else ("b" if ok["b"] else "neither")
    lines.append(f"== b passes in {wins['b']} of 6 ({'PASSES' if ok['b'] else 'does not pass'}); c in {wins['c']} of 6 ({'PASSES' if ok['c'] else 'does not pass'}); "
                 f"b clearly ahead of c in {bc} of 6 -> use {use}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def step_report():
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    tr = json.loads((OUT / "train.json").read_text())
    pts = ("empty start", "one short", "two short")

    def med(tag, kind):
        return [float(np.median([x["rank"] for x in rows if x["tag"] == tag and x["kind"] == kind and x["point"] == p])) for p in pts]
    lines = ["== Dream quick check: median rank of true terms (empty start / one short / two short); a baseline, b generator dreams, c library dreams"]
    ok = {"b": True, "c": True}
    closer = 0
    for a in ARMS:
        for s in SEEDS:
            name = f"{a}-s{s}"
            m = {v: (med(f"{name}-{v}", "dev"), med(f"{name}-{v}", "curriculum")) for v in "abc"}
            lines.append(f"   {name}: development " + "; ".join(f"{v} {' / '.join(f'{x:.0f}' for x in m[v][0])}" for v in "abc")
                         + " | curriculum " + "; ".join(f"{v} {' / '.join(f'{x:.0f}' for x in m[v][1])}" for v in "abc")
                         + " | steps " + ", ".join(f"{v} {tr[f'{name}-{v}']['steps']} ({tr[f'{name}-{v}']['seconds']:.0f}s)" for v in "abc"))
            for v in "bc":
                ok[v] &= all(m[v][0][j] < m["a"][0][j] for j in (1, 2)) and all(m[v][1][j] <= m["a"][1][j] + 1 for j in range(3))
            closer += any(m["b"][0][j] < 0.75 * m["c"][0][j] for j in (1, 2))
    lines.append(f"== b {'PASSES' if ok['b'] else 'does not pass'}; c {'PASSES' if ok['c'] else 'does not pass'}; b better than c by more than a "
                 f"quarter in {closer} of 6 cases -> use {('c' if ok['c'] and (not ok['b'] or closer <= 3) else 'b' if ok['b'] else 'neither')}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


# ---------------------------------------------------------------------------------------------------------------------
# the learner's own dreams, rebalanced (owner, 16 November; research log "Owner: two clarifications, the learner's own
# dreams, and rung b"): c1 piece-balanced, c2 the grammar weights flattened (power 1/3). Trained on the CPU, from the
# same start, with the same replay, general dreams, real share and steps as version c of the a2 check.
# ---------------------------------------------------------------------------------------------------------------------

FLAT = 3.0
CPU_THREADS = int(os.environ.get("DREAMCHECK_CPU_THREADS", "4"))


def own_pieces(lin):
    import dream_g2 as DR
    usage = {}
    for sid, eqs in lin.sols.items():
        for eq, ks in eqs.items():
            for k in ks:
                rk = DR.role_key(k, int(eq), 2)
                if rk is not None:
                    usage[rk] = usage.get(rk, 0) + 1
    return usage


def piece_groups(pieces):
    """The learner's used terms grouped by curriculum piece type (terms of no curriculum piece form one group)."""
    keys = LD.piece_keys(2)
    of = {k: lv for lv, ks in keys.items() for k in ks}
    groups = {}
    for k in pieces:
        groups.setdefault(of.get(k, "other"), []).append(k)
    return {str(g): v for g, v in sorted(groups.items(), key=lambda x: str(x[0]))}


def step_gen2(workers):
    import numpy as np
    import search_g1 as SG
    order = SG.vocabulary_order(2)
    gid = {k: g for g, k in enumerate(order)}
    info = {}
    t0 = time.time()
    with pool(workers) as ex:
        L = starts(ex)
        for (a, s), lin in L.items():
            name = f"{a}-s{s}"
            usage = own_pieces(lin)
            pieces = sorted({rk for rk in usage if rk in gid})
            groups = piece_groups(pieces)
            tab = lin.tables.get(2) or str(OUT / "gen" / f"library-{name}.npy")
            T = np.load(tab)
            w = np.exp(np.array([T[gid[k], 0] for k in pieces]) / FLAT)
            info[name] = {"pieces": len(pieces), "groups": {g: len(v) for g, v in groups.items()}}
            for ver, how in (("c1", ("bal", list(groups.values()))), ("c2", ("lib", pieces, w.tolist()))):
                p = OUT / "gen" / f"dreams-{ver}-{name}.pkl"
                if not p.exists():
                    exs = list(ex.map(dream_job, [(f"dreamcheck-{RUNG}-{ver}-{name}", k, how) for k in range(N_DREAM)], chunksize=4))
                    p.write_bytes(pickle.dumps([e for x in exs for e in x]))
                    print(f"{ver} dreams {name}: {len(pieces)} pieces in {len(groups)} piece types; {sum(bool(x) for x in exs)} of {N_DREAM} "
                          f"with examples; {sum(len(x) for x in exs)} examples; {time.time() - t0:.0f}s", flush=True)
    (OUT / "gen2.json").write_text(json.dumps(info, indent=1))


def train_cpu_job(args):
    """ladder.train_net_job on the CPU (owner: this check is CPU only)."""
    paths, prev, out, seed, steps, threads = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    import movenet_g2 as MN
    import search_g1 as SG
    MN.SHORTLIST = int(os.environ["G2H_SHORTLIST"])
    torch.manual_seed(seed)
    net = MN.load_net(prev + ".pt") if prev else MN.MoveNetL2()
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in LD.DIMS}
    ex = [e for p in paths for e in pickle.loads(Path(p).read_bytes())]
    t0 = time.time()
    tl = MN.train(net, ex, tables, steps, batch=C.BATCH, lr=C.LR, device="cpu", seed=seed, groups=None)
    secs = time.time() - t0
    if out:
        for d in LD.DIMS:
            np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device="cpu"))
        torch.save(net.state_dict(), out + ".pt")
    return {"examples": len(ex), "steps": steps, "seconds": secs, "last_loss": tl[-1]["loss"] if tl else None}


def examples_for(name, s, ver):
    load = lambda f: pickle.loads((OUT / "gen" / f).read_bytes())
    replay, general, extra = load(f"replay-{name}.pkl"), load(f"general-{name}.pkl"), load(f"dreams-{ver}-{name}.pkl")
    others = general + extra
    m = max(1, round(SHARE_REAL * len(others) / ((1 - SHARE_REAL) * max(len(replay), 1))))
    return replay * m + others, len(replay) * m


def step_train2(workers, timing=False):
    """workers = parallel trainings; each uses CPU_THREADS threads. Steps = version c's in the a2 check."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    tr = json.loads((OUT / "train.json").read_text())
    with pool(4) as ex:
        L = starts(ex)
    jobs, meta = [], {}
    for (a, s), lin in L.items():
        name = f"{a}-s{s}"
        for ver in ("c1", "c2"):
            out = OUT / "nets" / f"{name}-{ver}"
            if Path(f"{out}.pt").exists() and not timing:
                continue
            exs, n_real = examples_for(name, s, ver)
            p = OUT / "nets" / f"{name}-{ver}_examples.pkl"
            p.write_bytes(pickle.dumps(exs))
            steps = 50 if timing else tr[f"{name}-c"]["steps"]
            meta[f"{name}-{ver}"] = {"examples": len(exs), "real share": n_real / len(exs), "steps": steps, "path": p}
            jobs.append((f"{name}-{ver}", ([str(p)], lin.net, None if timing else str(out), LD.stable(name, ver), steps, CPU_THREADS)))
            if timing:
                break
        if timing:
            break
    info = {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as one:
        futs = {tag: one.submit(train_cpu_job, j) for tag, j in jobs}
        for tag, f in futs.items():
            r = f.result()
            meta[tag]["path"].unlink()
            info[tag] = {k: v for k, v in meta[tag].items() if k != "path"} | {"seconds": r["seconds"], "device": f"cpu x{CPU_THREADS}"}
            LD.log(f"dreamcheck {tag}: {r['steps']} steps on {r['examples']} examples on the CPU ({CPU_THREADS} threads) in {r['seconds']:.0f}s")
    if timing:
        print(json.dumps(info, indent=1))
        return
    p = OUT / "train.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), **info), indent=1))


def step_rank2(workers):
    jobs = []
    with pool(workers) as ex:
        L = starts(ex)
        for (a, s), lin in L.items():
            name = f"{a}-s{s}"
            dev = json.loads((OUT / "gen" / f"dev-s{s}.json").read_text())
            cur = json.loads((G2H / CUR[s]).read_text())
            cur = [r for lv in (1, 2, 3, 5, 6, 7) for r in cur[str(lv)]["test"][:3]]
            for ver in ("c1", "c2"):
                net = str(OUT / "nets" / f"{name}-{ver}")
                spec = ("confidence", net, lin.tables) if a == "blend" else ("net", net)
                jobs += [(r, spec, f"{name}-{ver}", "dev") for r in dev] + [(r, spec, f"{name}-{ver}", "curriculum") for r in cur]
        t0 = time.time()
        rows = [x for part in ex.map(rank_job, jobs) for x in part]
    (OUT / "ranks2.json").write_text(json.dumps(rows))
    print(f"rank: {len(rows)} rows from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def step_report2():
    """The a2 rule, for c1 and c2 against the baseline a (b and c shown for comparison)."""
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text()) + json.loads((OUT / "ranks2.json").read_text())
    pts = ("empty start", "one short", "two short")
    vers = ("a", "b", "c", "c1", "c2")
    lines = ["== Own-dream quick check (" + RUNG + "): share of development true terms ranked worse than 10 (empty / one short / two short); "
             "curriculum mean log2 rank. a baseline, b teacher dreams, c library dreams, c1 piece-balanced, c2 flattened"]
    wins = {"c1": 0, "c2": 0}
    for a in ARMS:
        for s in SEEDS:
            name, m = f"{a}-s{s}", {}
            for v in vers:
                R = [x for x in rows if x["tag"] == f"{name}-{v}"]
                dev = [x for x in R if x["kind"] == "dev"]
                tail = [float(np.mean([x["rank"] > 10 for x in dev if x["point"] == p])) for p in pts]
                cur = float(np.mean([np.log2(x["rank"]) for x in R if x["kind"] == "curriculum"]))
                m[v] = (tail, cur)
            judged = [0, 2] + ([1] if m["a"][0][1] >= 0.05 else [])
            for v in wins:
                wins[v] += all(m[v][0][j] < m["a"][0][j] for j in judged) and m[v][1] <= m["a"][1] + 0.25
            lines.append(f"   {name}: " + "; ".join(f"{v} {' / '.join(f'{x:.0%}' for x in m[v][0])}, cur {m[v][1]:.2f}" for v in vers))
    lines.append("== " + "; ".join(f"{v} passes in {wins[v]} of 6 ({'PASSES' if wins[v] >= 5 else 'does not pass'})" for v in wins))
    text = "\n".join(lines)
    (OUT / "report2.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen": lambda: step_gen(workers), "train": lambda: step_train(workers), "rank": lambda: step_rank(workers),
     "report": step_report if RUNG == "a1" else step_report_tail,
     "gen2": lambda: step_gen2(workers), "train2": lambda: step_train2(workers), "time2": lambda: step_train2(1, timing=True),
     "rank2": lambda: step_rank2(workers), "report2": step_report2}[step]()


if __name__ == "__main__":
    main()
