"""The curriculum's quick check (owner's approval, 12 November; research log "Owner: the curriculum quick check approved,
with two additions"): after brief training on the levels so far, does the next level's new piece rank much higher than
for an untrained network, the simplest-first order, and a network trained on made-up systems? No search.

Level generator: dream_g2's made-up-system machinery (coefficients, internal constants, initial states, pre-screen, the
full build's acceptance) with each equation's terms drawn only from the level's allowed pieces (each allowed piece
equally likely, then one of its terms); two variables; the level's new piece in every system. Training systems: 1-2
terms per equation; held-out systems of a tested level: 2-3 terms per equation (so nodes two terms short exist).
Steps: gen (systems and their supervised examples), train (from scratch, 500 steps, seeds 1 and 2), rank, report.
Usage: curriculum_check.py <workers> <step ...> [smoke]"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
os.environ.setdefault("G2H_RUN", "curriculum_smoke" if SMOKE else "curriculum")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402

OUT = C.RUN
D = 2
LEVELS = range(1, 9)
TRANSITIONS = ((2, 3), (5, 6), (6, 7))                           # (levels mastered, level tested)
SEEDS = (1, 2)
N_TRAIN = 50 if not SMOKE else 3                                 # training systems per level and seed
N_DEV = 40 if not SMOKE else 3                                   # held-out systems per tested level
STEPS = 500 if not SMOKE else 10
TRAIN_TERMS, DEV_TERMS = (1, 2), (2, 3)


def _pow(b, n):
    return ("Pow", b, (f"n{n}",))


def _rec(b):
    return _pow(("Add", ("K",), b), -1)


def patterns(level):
    """The level's piece as trees of the equation-relative reading (library_ops.tree with own = 0), own and other."""
    out = []
    for v in (("own",), ("other",)):
        out.append({1: v, 2: ("Mul", ("own",), ("other",)), 3: _pow(v, 2), 4: _pow(v, 3), 5: _rec(v),
                    6: ("Mul", v, _rec(v)), 7: _rec(_pow(v, 2)), 8: ("Mul", v, _rec(_pow(v, 2)))}[level])
    return out


def canon(t):
    """A tree with the arguments of Add and Mul flattened and sorted."""
    if len(t) == 1:
        return t
    if t[0] in ("Add", "Mul"):
        args = []

        def flat(u):
            if u[0] == t[0] and len(u) == 3:
                flat(u[1])
                flat(u[2])
            else:
                args.append(canon(u))
        flat(t)
        return (t[0],) + tuple(sorted(args, key=repr))
    return (t[0],) + tuple(canon(c) for c in t[1:])


def piece_keys():
    """{level: [role-relative keys of the 2-variable vocabulary]} (x1 the equation's own variable)."""
    import library_ops as L
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    pats = {lv: {canon(p) for p in patterns(lv)} for lv in LEVELS}
    out = {lv: [] for lv in LEVELS}
    for key in order:
        c = canon(L.tree(key, 0))
        for lv in LEVELS:
            if c in pats[lv]:
                out[lv].append(key)
    return out


def level_of(key, own):
    """The level whose piece the term (absolute key of the 2-variable vocabulary, for equation own) is, or None."""
    import library_ops as L
    c = canon(L.tree(key, own))
    for lv in LEVELS:
        if c in {canon(p) for p in patterns(lv)}:
            return lv
    return None


def draw(seed, allowed, new, n_terms, made_up_probs=None):
    """dream_g2.draw_fantasy with each equation's terms drawn from the allowed pieces ({level: keys}; the new piece's
    level forced into one random equation), or, for the made-up arm, from the whole vocabulary by the size prior."""
    import numpy as np
    import sympy as sp
    import torch
    import domains_g1 as DG
    import fitter_g1 as FG
    import language_g1 as LG
    import search_g1 as SG
    rng = DG.rng_of(seed)
    Y, X = DG.Y4, LG.variables(4)
    order = SG.vocabulary_order(D)
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(D)]
    forced = int(rng.integers(D))
    for i in range(D):
        n = int(rng.integers(n_terms[0], n_terms[1] + 1))
        picks = []
        while len(picks) < n:
            if made_up_probs is not None:
                key = order[int(rng.choice(len(order), p=made_up_probs))]
            else:
                lv = new if (i == forced and not picks) else int(rng.choice(sorted(allowed)))
                key = allowed[lv][int(rng.integers(len(allowed[lv])))]
            if key not in picks:
                picks.append(key)
        actual = [i] + [j for j in range(D) if j != i]
        e = 0
        for t, key in enumerate(picks):
            expr = LG.expr_of(key).xreplace({X[r]: Y[actual[r]] for r in range(D)})
            inf = FG.info(key)
            sub = {}
            if inf.k:
                Wp = torch.tensor(np.asarray(ic) * rng.uniform(0.5, 1.5, (64, D)))
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
    return {"seed": seed, "domain": "curriculum", "form": "curriculum", "d": D, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False}


def gen_job(args):
    """One accepted system: its bursts, true structure and (training systems) supervised examples."""
    prefix, k, allowed, new, n_terms, probs, examples = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    reasons = []
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw(seed, allowed, new, n_terms, None if probs is None else np.asarray(probs))
        why = DR.prescreen(desc)
        if why is not None:
            reasons.append(why)
            continue
        try:
            syn = DG.build(desc)
        except DG.Rejected as ex:
            reasons.append(str(ex).split(":")[0])
            continue
        except Exception as ex:
            reasons.append(type(ex).__name__)
            continue
        structure = [eq["keys"] for eq in syn.structure]
        out = {"k": k, "accepted": True, "tries": r + 1, "structure": structure, "bursts": syn.bursts(8).tolist()}
        if examples:
            try:
                out["examples"] = DR.make_examples(syn.bursts(8), structure, seed=10_000 + k)
            except Exception:
                out["examples"] = []
        return out
    return {"k": k, "accepted": False, "reasons": reasons[-5:]}


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import dream_g1 as D1
    import search_g1 as SG
    keys = piece_keys()
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    (OUT / "piece_keys.json").write_text(json.dumps({lv: keys[lv] for lv in LEVELS}))
    order = SG.vocabulary_order(D)
    V = json.loads((SG.G1_DIR / f"vocab_d{D}.json").read_text())["terms"]
    probs = D1.term_distribution(order, V, {}).tolist()
    jobs = []
    for s in SEEDS:                                              # curriculum training systems, every level up to 6
        for lv in range(1, 7):
            allowed = {l_: keys[l_] for l_ in range(1, lv + 1)}
            jobs += [(("train", s, lv), (f"cur-s{s}-L{lv}", k, allowed, lv, TRAIN_TERMS, None, True)) for k in range(N_TRAIN)]
        n_made = 6 * N_TRAIN                                     # made-up systems (taken in order to match examples)
        jobs += [(("madeup", s, 0), (f"made-s{s}", k, None, None, TRAIN_TERMS, probs, True)) for k in range(n_made + 2 * N_TRAIN)]
    for _, lv in TRANSITIONS:                                    # held-out systems of the tested levels
        allowed = {l_: keys[l_] for l_ in range(1, lv + 1)}
        jobs += [(("dev", 0, lv), (f"dev-L{lv}", k, allowed, lv, DEV_TERMS, None, False)) for k in range(N_DEV)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(gen_job, [a for _, a in jobs]))
    groups = {}
    for (tag, _), r in zip(jobs, res):
        groups.setdefault(tag, []).append(r)
    stats = {}
    for (kind, s, lv), rs in groups.items():
        acc = [r for r in rs if r["accepted"]]
        stats[f"{kind}-s{s}-L{lv}"] = {"accepted": len(acc), "of": len(rs), "examples": sum(len(r.get("examples", [])) for r in acc)}
        (OUT / "gen" / f"{kind}-s{s}-L{lv}.pkl").write_bytes(pickle.dumps(acc))
    (OUT / "gen_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"gen: {len(jobs)} systems in {time.time() - t0:.0f}s; accepted per group: "
          + "; ".join(f"{k} {v['accepted']}/{v['of']} ({v['examples']} examples)" for k, v in sorted(stats.items())))


def train_job(args):
    """cycles_g2h.train_job with the seed also setting the network's initial weights (steps 0: untrained); examples read
    straight from the generated files ([(path, number to take or None)]); only the 2-variable encodings are saved."""
    sources, out, gi, seed, steps = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import movenet_g2 as MN
    import search_g1 as SG
    MN.SHORTLIST = int(os.environ["G2H_SHORTLIST"])
    torch.manual_seed(seed)
    net = MN.MoveNetL2()
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(4)
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in C.DIMS}
    log = []
    if steps:
        ex = []
        for path, take in sources:
            part = [e for r in pickle.loads(Path(path).read_bytes()) for e in r.get("examples", [])]
            ex += part if take is None else part[:take]
        log = MN.train(net, ex, tables, steps, batch=C.BATCH, lr=C.LR, device=device, seed=seed, groups=None)
    np.save(f"{out}_d{D}.npy", MN.encode_vocabulary(net, tables[D], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    return {"out": out, "last_loss": log[-1]["loss"] if log else None, "n": None}


def free_gpus():
    """GPUs free now (utilization under 10% and under 1,000 MiB used); the sharing rule is not needed for minutes."""
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30).stdout
    return [int(i) for i, u, m in (line.split(",") for line in q.strip().splitlines()) if int(u) < 10 and int(m) < 1000]


def step_train():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    (OUT / "nets").mkdir(parents=True, exist_ok=True)
    stats = json.loads((OUT / "gen_stats.json").read_text())
    jobs = []
    for s in SEEDS:
        for mastered, _ in TRANSITIONS:
            files = [str(OUT / "gen" / f"train-s{s}-L{lv}.pkl") for lv in range(1, mastered + 1)]
            n = sum(stats[f"train-s{s}-L{lv}"]["examples"] for lv in range(1, mastered + 1))
            jobs.append(([(f, None) for f in files], str(OUT / "nets" / f"cur-s{s}-upto{mastered}"), s, STEPS, n))
            jobs.append(([(str(OUT / "gen" / f"madeup-s{s}-L0.pkl"), n)], str(OUT / "nets" / f"made-s{s}-upto{mastered}"), s, STEPS,
                         min(n, stats[f"madeup-s{s}-L0"]["examples"])))
        jobs.append(([], str(OUT / "nets" / f"untrained-s{s}"), s, 0, 0))
    gpus = free_gpus()
    if not gpus:
        raise SystemExit("no free GPU now; not training on the CPU (owner's rule)")
    t0 = time.time()
    with ProcessPoolExecutor(len(gpus), mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(train_job, (p, out, gpus[j % len(gpus)], s, steps)) for j, (p, out, s, steps, n) in enumerate(jobs)]
        res = [f.result() for f in futs]
    print(f"train: {len(jobs)} networks on GPUs {gpus} in {time.time() - t0:.0f}s; examples per network "
          + ", ".join(f"{Path(o).name} {n}" for p, o, s, st, n in jobs))
    (OUT / "train_stats.json").write_text(json.dumps([dict(r, n=n) for r, (p, o, s, st, n) in zip(res, jobs)], indent=1))


def rank_job(args):
    rec, nets, new = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import movenet_g2 as MN
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    priors = {name: MN.NetworkPrior(MN.load_net(p + ".pt").eval(), {D: np.load(p + f"_d{D}.npy")}, shortlist=2048)
              for name, p in nets.items()}

    def rank(score, legal, t):
        s = np.where(legal, score, -np.inf)
        return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)
    simplest = -np.arange(vocab.V, dtype=np.float64)
    out = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        targets = [t for t in true if level_of(vocab.keys[t], i) == new]
        if not targets:
            continue
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        memo = {}
        for t in targets:
            others = [u for u in true if u != t]
            points = [("empty start", ()), ("one short", tuple(others))]
            if len(true) >= 3:
                points.append(("two short", tuple(others[1:])))
            for kind, S in points:
                if S not in memo:
                    nd = se.root if not S else se.evaluate(S, None)
                    legal = np.ones(vocab.V, dtype=bool)
                    legal[list(S)] = False
                    memo[S] = (legal, {n: pr(nd, se.ctx).p_add for n, pr in priors.items()})
                legal, sc = memo[S]
                row = {"eq": i, "key": vocab.keys[t], "point": kind, "simplest": rank(simplest, legal, t)}
                row.update({n: rank(v, legal, t) for n, v in sc.items()})
                out.append(row)
    return out


def step_rank(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    rows = {}
    for mastered, new in TRANSITIONS:
        nets = {f"{arm}-s{s}": str(OUT / "nets" / f"{arm}-s{s}-upto{mastered}") for arm in ("cur", "made") for s in SEEDS}
        nets.update({f"untrained-s{s}": str(OUT / "nets" / f"untrained-s{s}") for s in SEEDS})
        dev = pickle.loads((OUT / "gen" / f"dev-s0-L{new}.pkl").read_bytes())
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            rows[new] = [x for part in ex.map(rank_job, [(r, nets, new) for r in dev]) for x in part]
        print(f"rank: level {new}: {len(dev)} systems, {len(rows[new])} rows, {time.time() - t0:.0f}s")
    (OUT / "ranks.json").write_text(json.dumps(rows))


def step_report():
    import numpy as np
    rows = {int(k): v for k, v in json.loads((OUT / "ranks.json").read_text()).items()}
    lines = ["== The curriculum's quick check: median rank of the next level's new piece (of the 2-variable language's terms)",
             "   columns: curriculum (seeds 1, 2) | made-up (1, 2) | untrained (1, 2) | simplest-first"]
    passing = 0
    names = {3: "squares (x^2), after levels 1-2", 6: "x/(K + x), after levels 1-5", 7: "the bump 1/(K + x^2), after levels 1-6"}
    for mastered, new in TRANSITIONS:
        lines.append(f"   level {new}, {names[new]}:")
        ok_all = True
        for kind in ("empty start", "one short", "two short"):
            R = [x for x in rows[new] if x["point"] == kind]
            if not R:
                lines.append(f"      {kind}: no cases")
                ok_all = False
                continue
            m = {n: float(np.median([x[n] for x in R])) for n in R[0] if n not in ("eq", "key", "point")}
            ok = all(m[f"cur-s{s}"] <= 0.5 * m[f"untrained-s{s}"] and m[f"cur-s{s}"] <= 0.5 * m["simplest"]
                     and m[f"cur-s{s}"] < m[f"made-s{s}"] for s in SEEDS)
            ok_all &= ok
            lines.append(f"      {kind} ({len(R)} cases): {m['cur-s1']:.0f}, {m['cur-s2']:.0f} | {m['made-s1']:.0f}, {m['made-s2']:.0f} | "
                         f"{m['untrained-s1']:.0f}, {m['untrained-s2']:.0f} | {m['simplest']:.0f}" + ("  <- meets the rule" if ok else ""))
        passing += ok_all
        lines.append(f"      -> {'meets' if ok_all else 'does not meet'} the rule at all three points")
    lines.append(f"== Pass rule (at least 2 of 3 transitions): {passing} of 3 -> {'PASSES' if passing >= 2 else 'FAILS'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers = int(ARGS[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ARGS[2:] if s != "smoke"]:
        free = shutil.disk_usage("/home").free / 2 ** 30
        if free < 5:
            raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 5 GB (owner's crash guard)")
        {"gen": lambda: step_gen(workers), "train": step_train, "rank": lambda: step_rank(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
