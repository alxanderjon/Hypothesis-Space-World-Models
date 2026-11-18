"""Emphasized teacher dreams, quick check (owner, 18 November; research log "Owner: the emphasized-dream quick check",
where the versions and pass rule were set before the run). From each learner lineage's state at the start of e3 (the end
of e12), one learning round of 4,500 steps with:
  a. e3's round-1 teacher dreams (the set every e3 lineage trained on; every dream holds at least one x/(K+x^2) term);
  b. phase 1 of the two-phase schedule (owner, 18 Nov): the same teacher dreams as ~90% of the dream examples, own
     piece-balanced dreams built only from pieces already learned (no x/(K+x^2)) as ~10%, and replay as ~10% of all
     examples, drawn first from solved systems holding x/(K+x^2);
version a also has the lineage's own 1,000 piece-balanced dreams and its replayed solutions at 30% of the examples. Then, no
search: the ranks of the true terms of new e3-style development systems at the empty start, one short and two short, and
the injection check (the best-ranked x/(K+x^2) term on equations without one). Also the share of all training examples
whose equation holds the new piece, by source. Usage: emph_check.py <workers> <gen|train|rank|report>"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "emph_check")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402  (read only)

C, G2H, OUT = LD.C, LD.G2H, LD.OUT
L_DIR = G2H / "ladder"
SEEDS, ARMS, RUNG, PREV = (1, 2, 3), ("blend", "net"), "e3", "e12"
N_DEV, N_TEACH, N_EMPH, N_OWN, STEPS = 20, 1000, 900, 1000, 4500
N_OWN1, OWN1_SHARE, REPLAY1_SHARE = 130, 0.10, 0.10              # phase 1: own dreams ~10% of dream examples; replay ~10% of all
D = 3


def pool(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    return ProcessPoolExecutor(workers, mp_context=get_context("spawn"))


def pieces():
    keys = LD.piece_keys(D)
    R = LD.RUNGS[RUNG]
    allowed = {lv: keys[lv] for lv in R["levels"]}
    new = LD.new_keys(RUNG, keys)
    return keys, allowed, new


def draw_forced(seed, allowed, new, n_terms, d):
    """ladder.draw with each equation's first term drawn from the new piece (force=True)."""
    import numpy as np
    import sympy as sp
    import torch
    import domains_g1 as DG
    import fitter_g1 as FG
    import language_g1 as LG
    rng = DG.rng_of(seed)
    Y, X = DG.Y4, LG.variables(4)
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    for i in range(d):
        n = int(rng.integers(n_terms[0], n_terms[1] + 1))
        picks = [new[int(rng.integers(len(new)))]]
        while len(picks) < n:
            lv = int(rng.choice(sorted(allowed)))
            key = allowed[lv][int(rng.integers(len(allowed[lv])))]
            if key not in picks:
                picks.append(key)
        actual = [i] + [j for j in range(d) if j != i]
        e = 0
        for t, key in enumerate(picks):
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
    return {"seed": seed, "domain": "curriculum", "form": "curriculum", "d": d, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False}


def emph_job(args):
    """One accepted teacher dream: kind "every" (an x/(K+x^2) term in every equation) or "none" (no x/(K+x^2) term)."""
    prefix, k, allowed, new, kind = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    newset = set(new)
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        if kind == "every":
            desc = draw_forced(seed, allowed, new, (3, 3), D)
        else:
            desc = LD.draw(seed, {lv: v for lv, v in allowed.items() if lv != 8}, (3, 3), D)
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
        except Exception:
            continue
        st = [eq["keys"] for eq in syn.structure]
        has = [any(DR.role_key(x, i, D) in newset for x in ks) for i, ks in enumerate(st)]
        if (kind == "every" and not all(has)) or (kind == "none" and any(has)):
            continue
        try:
            return DR.make_examples(syn.bursts(8), st, seed=10_000 + k)
        except Exception:
            continue
    return []


def recycled_map(fn, jobs, workers, chunk=100):
    """Fresh pools every `chunk` jobs (compiled libraries are never unloaded: ladder.dream_map's lesson)."""
    out = []
    for c in range(0, len(jobs), chunk):
        with pool(workers) as ex:
            out += list(ex.map(fn, jobs[c:c + chunk]))
    return out


def lineage(a, s):
    """The lineage at the start of e3: e12's end state, plus its e3 round-1 wake solutions (as e3's first learning saw)."""
    st = json.loads((L_DIR / PREV / f"{a}-s{s}.state.json").read_text())
    sols = {k: {int(e): v for e, v in d.items()} for k, d in st["sols"].items()}
    for y in json.loads((L_DIR / RUNG / f"{a}-s{s}" / "r1_wake.json").read_text()):
        if y["status"] == "done":
            for e in y["equations"]:
                if e["first_exact"] is not None and e["solution"]:
                    sols.setdefault(y["id"], {})[int(e["eq"])] = e["solution"]
    return st, sols


def step_gen(workers):
    import dream_g2 as DR
    import search_g1 as SG
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    keys, allowed, new = pieces()
    t0 = time.time()
    # development systems (e3-style), checked against every earlier set
    if not all((OUT / "gen" / f"dev-s{s}.json").exists() for s in SEEDS):
        with pool(workers) as ex:
            dev = {s: [r for r in ex.map(LD.gen_job, [(f"emph-dev-s{s}", k, allowed, (3, 3), D, new) for k in range(N_DEV)]) if r["accepted"]]
                   for s in SEEDS}
        old = LD.earlier_keys("emph")
        for p in (L_DIR / "gen").glob("*.json"):
            old |= {LD.data_key(r["bursts"]) for sp in json.loads(p.read_text()).values() for r in sp}
        clash = sum(LD.data_key(r["bursts"]) in old for s in SEEDS for r in dev[s])
        print(f"dev systems: {sum(len(v) for v in dev.values())} accepted; identical to {len(old)} earlier systems: {clash}", flush=True)
        if clash:
            raise SystemExit("development systems not new")
        for s in SEEDS:
            (OUT / "gen" / f"dev-s{s}.json").write_text(json.dumps([dict(r, id=f"emph-dev-s{s}-{r['k']}") for r in dev[s]]))
    # per lineage: own piece-balanced dreams (as e3's learn1 drew them) and the replay examples
    of = {k: lv for lv, ks in keys.items() for k in ks}
    order = set(SG.vocabulary_order(D))
    for a in ARMS:
        for s in SEEDS:
            name = f"{a}-s{s}"
            st, sols = lineage(a, s)
            p = OUT / "gen" / f"own-{name}.pkl"
            if not p.exists():
                recs = {}
                for f in st["record_files"] + [str(L_DIR / "gen" / f"{RUNG}-s{s}.json")]:
                    obj = json.loads(Path(f).read_text())
                    for part in ([obj] if "train" in obj else obj.values()):
                        for r in part["train"]:
                            recs[r["id"]] = r
                groups = {}
                for sid, eqs in sols.items():
                    if len(recs[sid]["structure"]) != D:
                        continue
                    for eq, ks in eqs.items():
                        for k in ks:
                            rk = DR.role_key(k, int(eq), D)
                            if rk is not None and rk in order and rk not in groups.get(of.get(rk, LD.OTHER), []):
                                groups.setdefault(of.get(rk, LD.OTHER), []).append(rk)
                groups = {g: sorted(v) for g, v in sorted(groups.items())}
                res = recycled_map(LD.rung_dream_job, [(f"emph-own-{name}", k, groups, (3, 3), D) for k in range(N_OWN)], workers)
                ex_ = [e for x in res for e in x]
                if len(ex_) < 2 * N_OWN:
                    raise SystemExit(f"own dreams {name}: {len(ex_)} examples: broken batch")
                p.write_bytes(pickle.dumps(ex_))
                print(f"own dreams {name}: {sum(len(v) for v in groups.values())} used terms in {len(groups)} piece types; "
                      f"{sum(bool(x) for x in res)} of {N_OWN} with examples; {len(ex_)} examples; {time.time() - t0:.0f}s", flush=True)
                (OUT / "gen" / f"groups-{name}.json").write_text(json.dumps({str(g): v for g, v in groups.items()}))
            p = OUT / "gen" / f"own1-{name}.pkl"
            if not p.exists():
                g_all = {int(g): v for g, v in json.loads((OUT / "gen" / f"groups-{name}.json").read_text()).items()}
                g1 = {g: [k for k in v if k not in set(new)] for g, v in g_all.items()}
                g1 = {g: v for g, v in g1.items() if v}
                res = recycled_map(LD.rung_dream_job, [(f"emph-own1-{name}", k, g1, (3, 3), D) for k in range(N_OWN1)], workers)
                ex_ = [e for x in res for e in x]
                if len(ex_) < 2 * N_OWN1:
                    raise SystemExit(f"phase-1 own dreams {name}: {len(ex_)} examples: broken batch")
                p.write_bytes(pickle.dumps(ex_))
                print(f"phase-1 own dreams {name}: {sum(len(v) for v in g1.values())} learned terms in {len(g1)} piece types (no new piece); "
                      f"{sum(bool(x) for x in res)} of {N_OWN1} with examples; {len(ex_)} examples; {time.time() - t0:.0f}s", flush=True)
            p = OUT / "gen" / f"replay-{name}.pkl"
            if not p.exists():
                ex_all = pickle.loads((L_DIR / RUNG / name / "examples.pkl").read_bytes())
                rep = [e for sid in sols if sid in ex_all for e in ex_all[sid]["ex"]]
                p.write_bytes(pickle.dumps(rep))
                print(f"replay {name}: {len(rep)} examples from {sum(sid in ex_all for sid in sols)} of {len(sols)} solved systems", flush=True)


def mix(name, ver):
    import random
    import search_g1 as SG
    load = lambda f: pickle.loads((OUT / "gen" / f).read_bytes())
    teacher = pickle.loads((L_DIR / RUNG / "teacher_round1.pkl").read_bytes())
    replay = load(f"replay-{name}.pkl")
    if ver == "a":                                               # e3's own round-1 mix
        own = load(f"own-{name}.pkl")
        others = own + teacher
        m = max(1, round(LD.SHARE_REAL * len(others) / ((1 - LD.SHARE_REAL) * max(len(replay), 1))))
        return {"replay": replay * m, "own": own, "teacher": teacher}
    # phase 1: own dreams ~10% of dream examples, replay ~10% of all, new-piece systems first
    _, _, new = pieces()
    newset, order = set(new), SG.vocabulary_order(D)
    rng = random.Random(LD.stable(name, "phase1"))
    own_all = load(f"own1-{name}.pkl")
    n_own = round(OWN1_SHARE / (1 - OWN1_SHARE) * len(teacher))
    own = rng.sample(own_all, min(n_own, len(own_all)))
    n_rep = round(REPLAY1_SHARE / (1 - REPLAY1_SHARE) * (len(teacher) + len(own)))
    first = [e for e in replay if holds_new(e, newset, order)]
    rest = [e for e in replay if not holds_new(e, newset, order)]
    rng.shuffle(first)
    rng.shuffle(rest)
    rep_ = (first + rest)[:n_rep]
    while len(rep_) < n_rep:                                     # fewer replay examples than wanted: repeat
        rep_ += (first + rest)[:n_rep - len(rep_)]
    return {"replay": rep_, "own": own, "teacher": teacher}


def holds_new(e, newset, order):
    import dream_g2 as DR
    import numpy as np
    true = set(np.asarray(e["gids"])[np.asarray(e["add"]) | (np.asarray(e["inS"]) & ~np.asarray(e["rem"]))].tolist())
    return any(DR.role_key(order[int(g)], int(e["eq"]), int(e["d"])) in newset for g in true)


def step_train(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import search_g1 as SG
    _, _, new = pieces()
    newset, order = set(new), SG.vocabulary_order(D)
    gpus = LD.GPUs()
    info = {}
    import threading
    lock = threading.Lock()

    def one(a, s, ver):
        name = f"{a}-s{s}"
        st, _ = lineage(a, s)
        M = mix(name, ver)
        share = {src: (sum(holds_new(e, newset, order) for e in ex) / max(1, len(ex)), len(ex)) for src, ex in M.items()}
        exs = M["replay"] + M["own"] + M["teacher"]
        total = sum(sh * n for sh, n in share.values()) / len(exs)
        out = OUT / "nets" / f"{name}-{ver}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not Path(f"{out}.pt").exists():
            p = OUT / "nets" / f"{name}-{ver}_examples.pkl"
            p.write_bytes(pickle.dumps(exs))
            t0 = time.time()

            def run(g):
                with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex1:
                    return ex1.submit(LD.train_net_job, ([str(p)], st["net"], str(out), g, LD.stable(name, ver, "emph"), STEPS)).result(), g
            _, g = gpus.run(run)
            p.unlink()
            LD.log(f"emph {name}-{ver}: {STEPS} steps on {len(exs)} examples on GPU {g} in {time.time() - t0:.0f}s")
        with lock:
            info[f"{name}-{ver}"] = {"examples": len(exs), "new-piece share": total,
                                     "by source": {k: {"share": v[0], "examples": v[1]} for k, v in share.items()}}
    threads = [threading.Thread(target=one, args=(a, s, v)) for a in ARMS for s in SEEDS for v in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (OUT / "train.json").write_text(json.dumps(info, indent=1))


def rank_job(args):
    rec, spec, tag, new = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import muzero_checks as MZ
    ctx, fitter = MZ.system(rec)
    v = ctx["vocab"]
    prior = LD.make_prior(spec)
    newset = set(new)
    rows = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({v.index[k] for k in keys if k in v.index})
        isnew = {t: DR.role_key(v.keys[t], i, D) in newset for t in true}
        eq = MZ.Eq(ctx, fitter, i)
        legal0 = MZ.legal_adds(v.V, ())
        pr0 = prior(eq.se.root, eq.se.ctx)
        sc0 = np.log(np.maximum(np.where(legal0, pr0.p_swap_add, 0), 1e-300))
        if not any(isnew.values()):
            cand = [t for t in range(v.V) if DR.role_key(v.keys[t], i, D) in newset]
            if cand:
                rows.append({"tag": tag, "kind": "injection", "rank": min(MZ.rank(sc0, legal0, t) for t in cand)})
        for t in true:
            others = [u for u in true if u != t]
            for point, S in [("empty start", ()), ("one short", tuple(others))] + ([("two short", tuple(others[1:]))] if len(true) >= 3 else []):
                if S:
                    nd = eq.node(S)
                    legal = MZ.legal_adds(v.V, S)
                    pr = prior(nd, eq.se.ctx)
                    sc = np.log(np.maximum(np.where(legal, pr.p_swap_add, 0), 1e-300))
                else:
                    legal, sc = legal0, sc0
                rows.append({"tag": tag, "kind": "new" if isnew[t] else "other", "point": point, "rank": MZ.rank(sc, legal, t)})
    return rows


def step_rank(workers):
    _, _, new = pieces()
    jobs = []
    for a in ARMS:
        for s in SEEDS:
            st, _ = lineage(a, s)
            tables = {int(k): v for k, v in st["tables"].items()}
            dev = json.loads((OUT / "gen" / f"dev-s{s}.json").read_text())
            for ver in "ab":
                net = str(OUT / "nets" / f"{a}-s{s}-{ver}")
                spec = ("confidence", net, tables) if a == "blend" else ("net", net)
                jobs += [(r, spec, f"{a}-s{s}-{ver}", new) for r in dev]
    t0 = time.time()
    with pool(workers) as ex:
        rows = [x for part in ex.map(rank_job, jobs) for x in part]
    (OUT / "ranks.json").write_text(json.dumps(rows))
    print(f"rank: {len(rows)} rows from {len(jobs)} jobs in {time.time() - t0:.0f}s")


def step_report():
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    tr = json.loads((OUT / "train.json").read_text())
    lines = ["== Phase-1 quick check (e3; a e3 round-1 mix, b phase 1): new piece's top-10 share at the empty start; other true terms' mean log2 rank "
             "(all points); injection: best x/(K+x^2) rank on equations without one (mean log2); training share of new-piece examples"]
    wins = 0
    for a in ARMS:
        for s in SEEDS:
            name, m = f"{a}-s{s}", {}
            for v in "ab":
                R = [x for x in rows if x["tag"] == f"{name}-{v}"]
                top = float(np.mean([x["rank"] <= 10 for x in R if x["kind"] == "new" and x["point"] == "empty start"]))
                oth = float(np.mean([math.log2(x["rank"]) for x in R if x["kind"] == "other"]))
                inj = float(np.mean([math.log2(x["rank"]) for x in R if x["kind"] == "injection"]))
                m[v] = (top, oth, inj, tr[f"{name}-{v}"]["new-piece share"])
            ok = m["b"][0] - m["a"][0] >= 0.20 and m["b"][1] <= m["a"][1] + 0.25 and m["b"][2] >= m["a"][2] - 0.25
            wins += ok
            lines.append(f"   {name}: " + "; ".join(f"{v} top-10 {m[v][0]:.0%}, other {m[v][1]:.2f}, injection {m[v][2]:.2f}, "
                                                     f"training share {m[v][3]:.0%}" for v in "ab") + (" PASS" if ok else ""))
    for k, v in sorted(tr.items()):
        lines.append(f"   mix {k}: {v['examples']} examples, new-piece share {v['new-piece share']:.0%} (" +
                     ", ".join(f"{src} {x['share']:.0%} of {x['examples']}" for src, x in v["by source"].items()) + ")")
    lines.append(f"== phase 1 passes in {wins} of 6 ({'PASSES' if wins >= 4 else 'does not pass'})")
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
