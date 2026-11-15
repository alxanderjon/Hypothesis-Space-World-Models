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

os.environ.setdefault("G2H_RUN", "dreamcheck")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

C, G2H, OUT = LD.C, LD.G2H, LD.OUT
SEEDS, ARMS = (1, 2, 3), ("blend", "net")
N_DEV, N_DREAM, N_GEN, SHARE_REAL, BASE_STEPS = 20, 1000, 50, 0.30, 500
A1_LEVELS = (1, 2, 3)
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
    return {(a, s): LD.Lineage(a, s, "a1", ex, gpus) for a in ARMS for s in SEEDS}


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
        # development systems (a1-style), checked against every earlier set
        dev = {s: [r for r in ex.map(LD.gen_job, [(f"dreamcheck-dev-s{s}", k, allowed, (3, 3), 2) for k in range(N_DEV)]) if r["accepted"]] for s in SEEDS}
        old = LD.earlier_keys("dreamcheck") | {LD.data_key(r["bursts"]) for p in (G2H / "ladder/gen").glob("*.json")
                                                for sp in json.loads(p.read_text()).values() for r in sp}
        clash = sum(LD.data_key(r["bursts"]) in old for s in SEEDS for r in dev[s])
        print(f"dev systems: {sum(len(v) for v in dev.values())} accepted; identical to earlier systems {clash}")
        if clash:
            raise SystemExit("development systems not new")
        for s in SEEDS:
            (OUT / "gen" / f"dev-s{s}.json").write_text(json.dumps([dict(r, id=f"dev-s{s}-{r['k']}") for r in dev[s]]))
        # generator dreams (b), shared by both arms of a seed
        for s in SEEDS:
            p = OUT / "gen" / f"dreams-gen-s{s}.pkl"
            if not p.exists():
                exs = list(ex.map(dream_job, [(f"dreamcheck-gen-s{s}", k, ("gen", allowed)) for k in range(N_DREAM)], chunksize=4))
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
                g = list(ex.map(LD.fantasy_job, [(f"dreamcheck-{name}-general", k, probs, 2) for k in range(N_GEN)]))
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
                exs = list(ex.map(dream_job, [(f"dreamcheck-lib-{name}", k, ("lib", pieces, w)) for k in range(N_DREAM)], chunksize=4))
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


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen": lambda: step_gen(workers), "train": lambda: step_train(workers), "rank": lambda: step_rank(workers),
     "report": step_report}[step]()


if __name__ == "__main__":
    main()
