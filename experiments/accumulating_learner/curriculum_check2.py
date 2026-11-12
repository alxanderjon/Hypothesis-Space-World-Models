"""The curriculum's follow-up quick check (owner, 12 November; research log "Owner: more combinations of each kind, and a
no-network library comparison"): six combinations, three of a piece inside a frame and three of a frame multiplied by a
variable, ranked by the curriculum network, the made-up network, an untrained network, a DreamCoder-style library learned
from the mastered levels' true structures (compression plus counting, context-free form, no network), its made-up
counterpart, counting alone, and simplest-first; at the empty start, one term short and two terms short. No search.
Reuses curriculum_check's generator, systems and networks. Usage: curriculum_check2.py <workers> <step ...> [smoke]"""

import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_check as CC                                    # noqa: E402

C, D, SEEDS = CC.C, CC.D, CC.SEEDS
SRC = C.ROOT / "outputs/accumulating_learner/g2h/curriculum"     # the first check's systems and networks
OUT = C.ROOT / "outputs/accumulating_learner/g2h" / ("curriculum2_smoke" if SMOKE else "curriculum2")
N_DEV = 40 if not SMOKE else 3
_V = (("own",), ("other",))


def combos():
    """{name: (kind, description, mastered levels, patterns for own = 0)}."""
    rec, pw = CC._rec, CC._pow
    out = {"A1": ("inside", "1/(K + x^2)", 5, [rec(pw(v, 2)) for v in _V]),
            "A2": ("inside", "1/(K + x^3)", 5, [rec(pw(v, 3)) for v in _V]),
            "A3": ("inside", "1/(K + x*y)", 5, [rec(("Mul", ("own",), ("other",)))]),
            "B1": ("times", "x/(K + x)", 5, [("Mul", v, rec(v)) for v in _V]),
            "B2": ("times", "x/(K + x^2)", 7, [("Mul", v, rec(pw(v, 2))) for v in _V]),
            "B3": ("times", "y/(K + x)", 5, [("Mul", _V[1], rec(_V[0])), ("Mul", _V[0], rec(_V[1]))])}
    return {k: v for k, v in out.items() if v[2] == 5} if SMOKE else out   # smoke: never write into the first check's files


def keys_of(pats):
    import library_ops as L
    import search_g1 as SG
    want = {CC.canon(p) for p in pats}
    return [k for k in SG.vocabulary_order(D) if CC.canon(L.tree(k, 0)) in want]


def is_new(key, own, pats):
    import library_ops as L
    return CC.canon(L.tree(key, own)) in {CC.canon(p) for p in pats}


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    keys = CC.piece_keys()
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (_, _, mastered, pats) in combos().items():
        nk = keys_of(pats)
        if not nk:
            raise SystemExit(f"{name}: no term of the 2-variable language matches")
        allowed = {lv: keys[lv] for lv in range(1, mastered + 1)}
        allowed[100] = nk
        jobs += [(("dev", name), (f"dev2-{name}", k, allowed, 100, CC.DEV_TERMS, None, False)) for k in range(N_DEV)]
    if not SMOKE and not all((SRC / "gen" / f"train-s{s}-L7.pkl").exists() for s in SEEDS):
        for s in SEEDS:                                          # the curriculum's level 7 (the bump), as before
            allowed = {lv: keys[lv] for lv in range(1, 8)}
            jobs += [(("train", s), (f"cur-s{s}-L7", k, allowed, 7, CC.TRAIN_TERMS, None, True)) for k in range(CC.N_TRAIN)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(CC.gen_job, [a for _, a in jobs]))
    groups = {}
    for (tag, _), r in zip(jobs, res):
        groups.setdefault(tag, []).append(r)
    stats = json.loads((SRC / "gen_stats.json").read_text())
    for tag, rs in groups.items():
        acc = [r for r in rs if r["accepted"]]
        if tag[0] == "dev":
            (OUT / "gen" / f"dev-{tag[1]}.pkl").write_bytes(pickle.dumps(acc))
        else:
            (SRC / "gen" / f"train-s{tag[1]}-L7.pkl").write_bytes(pickle.dumps(acc))
            stats[f"train-s{tag[1]}-L7"] = {"accepted": len(acc), "of": len(rs), "examples": sum(len(r.get("examples", [])) for r in acc)}
    if not SMOKE:
        (SRC / "gen_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"gen: {len(jobs)} systems in {time.time() - t0:.0f}s; accepted: "
          + "; ".join(f"{t} {sum(r['accepted'] for r in rs)}/{len(rs)}" for t, rs in groups.items()))


def step_train():
    """The curriculum and made-up networks for levels 1-7 (levels 1-5's exist from the first check)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    if SMOKE:
        return
    stats = json.loads((SRC / "gen_stats.json").read_text())
    jobs = []
    for s in SEEDS:
        if (SRC / "nets" / f"cur-s{s}-upto7.pt").exists():
            continue
        files = [str(SRC / "gen" / f"train-s{s}-L{lv}.pkl") for lv in range(1, 8)]
        n = sum(stats[f"train-s{s}-L{lv}"]["examples"] for lv in range(1, 8))
        jobs.append(([(f, None) for f in files], str(SRC / "nets" / f"cur-s{s}-upto7"), s, CC.STEPS, n))
        jobs.append(([(str(SRC / "gen" / f"madeup-s{s}-L0.pkl"), n)], str(SRC / "nets" / f"made-s{s}-upto7"), s, CC.STEPS, n))
    if not jobs:
        return
    gpus = CC.free_gpus()
    if not gpus:
        raise SystemExit("no free GPU now; not training on the CPU (owner's rule)")
    t0 = time.time()
    with ProcessPoolExecutor(len(gpus), mp_context=get_context("spawn")) as ex:
        res = [f.result() for f in [ex.submit(CC.train_job, (src, out, gpus[j % len(gpus)], s, st))
                                    for j, (src, out, s, st, n) in enumerate(jobs)]]
    print(f"train: {len(jobs)} networks on GPUs {gpus} in {time.time() - t0:.0f}s; examples "
          + ", ".join(f"{Path(o).name} {n}" for _, o, _, _, n in jobs))


def systems_for(s, mastered, arm):
    """The training systems behind a network: the curriculum's levels 1..mastered, or the made-up systems whose examples
    the matched made-up network used (in order, up to the same number of examples)."""
    stats = json.loads((SRC / "gen_stats.json").read_text())
    if arm == "cur":
        return [r for lv in range(1, mastered + 1) for r in pickle.loads((SRC / "gen" / f"train-s{s}-L{lv}.pkl").read_bytes())]
    n = sum(stats[f"train-s{s}-L{lv}"]["examples"] for lv in range(1, mastered + 1))
    out, got = [], 0
    for r in pickle.loads((SRC / "gen" / f"madeup-s{s}-L0.pkl").read_bytes()):
        if got >= n:
            break
        out.append(r)
        got += len(r.get("examples", []))
    return out


def step_grammars():
    """Per seed, mastered set and arm: the context-free library grammar's and counting-alone's log P of every term of
    the 2-variable language, for both equations ([V, 2] tables)."""
    import numpy as np
    import grammar_ops as GO
    import library_ops as L
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    tokens = GO.alphabet(order, d=D)
    (OUT / "grammars").mkdir(parents=True, exist_ok=True)
    info = {}
    for s in SEEDS:
        for mastered in ((5,) if SMOKE else (5, 7)):
            for arm in ("cur", "made"):
                eqs = [(i, keys) for r in systems_for(s, mastered, arm) for i, keys in enumerate(r["structure"]) if keys]
                terms = Counter(L.tree(k, own) for own, keys in eqs for k in keys)
                lib, log = L.learn(terms, tokens, "uni")
                tabs = {}
                for name, library in (("lib", lib), ("count", [])):
                    g = L.LibGrammar.fit(eqs, [], tokens, library, form="uni")
                    tabs[name] = np.array([[g.logp_term(k, own) for own in range(D)] for k in order])
                np.savez(OUT / "grammars" / f"{arm}-s{s}-upto{mastered}.npz", **tabs)
                info[f"{arm}-s{s}-upto{mastered}"] = {"systems": len(eqs) // D, "terms": sum(terms.values()),
                                                      "pieces": [L.show(x["piece"]) for x in log]}
    (OUT / "grammars" / "info.json").write_text(json.dumps(info, indent=1))
    print("grammars: " + "; ".join(f"{k}: {v['systems']} systems, {len(v['pieces'])} pieces" for k, v in info.items()))


def rank_job(args):
    rec, nets, tabs, pats = args
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
    gid = vocab.gid
    priors = {n: MN.NetworkPrior(MN.load_net(p + ".pt").eval(), {D: np.load(p + f"_d{D}.npy")}, shortlist=2048)
              for n, p in nets.items()}
    gram = {n: np.load(p)[k][gid] for n, (p, k) in tabs.items()}

    def rank(score, legal, t):
        s = np.where(legal, score, -np.inf)
        return int((s > s[t]).sum() + ((s == s[t]) & (np.arange(len(s)) < t)).sum() + 1)
    simplest = -np.arange(vocab.V, dtype=np.float64)
    out = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({vocab.index[k] for k in keys if k in vocab.index})
        targets = [t for t in true if is_new(vocab.keys[t], i, pats)]
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
                row.update({n: rank(g[:, i], legal, t) for n, g in gram.items()})
                out.append(row)
    return out


def step_rank(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    rows = {}
    for name, (_, _, mastered, pats) in combos().items():
        nets = {f"{arm}-s{s}": str(SRC / "nets" / f"{arm}-s{s}-upto{mastered}") for arm in ("cur", "made") for s in SEEDS}
        nets.update({f"untrained-s{s}": str(SRC / "nets" / f"untrained-s{s}") for s in SEEDS})
        tabs = {f"{g}-{arm}-s{s}": (str(OUT / "grammars" / f"{arm}-s{s}-upto{mastered}.npz"), g)
                for g in ("lib", "count") for arm in ("cur", "made") for s in SEEDS}
        dev = pickle.loads((OUT / "gen" / f"dev-{name}.pkl").read_bytes())
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            rows[name] = [x for part in ex.map(rank_job, [(r, nets, tabs, pats) for r in dev]) for x in part]
        print(f"rank {name}: {len(dev)} systems, {len(rows[name])} rows, {time.time() - t0:.0f}s")
    (OUT / "ranks.json").write_text(json.dumps(rows))


def step_report():
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    lines = ["== Which combinations transfer: median rank of the new term at the empty start / one short / two short",
             "   per method: seed 1, seed 2; 'transfers' as set before the run"]
    verdict = {}
    for name, (kind, desc, mastered, _) in combos().items():
        R = rows[name]
        lines.append(f"   {name} ({'piece inside a frame' if kind == 'inside' else 'frame times a variable'}), {desc}, after levels 1-{mastered}:")
        med = {}
        for pt in ("empty start", "one short", "two short"):
            P = [x for x in R if x["point"] == pt]
            med[pt] = {n: float(np.median([x[n] for x in P])) for n in P[0] if n not in ("eq", "key", "point")} if P else None
        ok = {"network": True, "library": True}
        for pt, m in med.items():
            if m is None:
                ok = {k: False for k in ok}
                continue
            for s in SEEDS:
                ok["network"] &= (m[f"cur-s{s}"] <= 0.5 * m[f"untrained-s{s}"] and m[f"cur-s{s}"] <= 0.5 * m["simplest"]
                                  and m[f"cur-s{s}"] < m[f"made-s{s}"])
                ok["library"] &= m[f"lib-cur-s{s}"] <= 0.5 * m["simplest"] and m[f"lib-cur-s{s}"] < m[f"lib-made-s{s}"]
        for title, cols in (("curriculum network", "cur"), ("made-up network", "made"), ("untrained network", "untrained"),
                            ("library (curriculum)", "lib-cur"), ("library (made-up)", "lib-made"), ("counting alone (curriculum)", "count-cur")):
            lines.append(f"      {title:>28}: " + " / ".join(
                ("-" if med[pt] is None else f"{med[pt][cols + '-s1']:.0f}, {med[pt][cols + '-s2']:.0f}") for pt in med))
        lines.append(f"      {'simplest-first':>28}: " + " / ".join(("-" if med[pt] is None else f"{med[pt]['simplest']:.0f}") for pt in med))
        n_cases = " / ".join(str(len([x for x in R if x["point"] == pt])) for pt in med)
        lines.append(f"      cases {n_cases}; transfers under the network: {'YES' if ok['network'] else 'no'}; under the library: "
                     f"{'YES' if ok['library'] else 'no'}")
        verdict[name] = ok
    for kind, title in (("inside", "piece inside a frame"), ("times", "frame times a variable")):
        names = [n for n, c in combos().items() if c[0] == kind]
        lines.append(f"== {title}: network {sum(verdict[n]['network'] for n in names)} of {len(names)}, library "
                     f"{sum(verdict[n]['library'] for n in names)} of {len(names)}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers = int(ARGS[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ARGS[2:] if s != "smoke"]:
        free = shutil.disk_usage("/home").free / 2 ** 30
        if free < 2:
            raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB (owner's crash guard)")
        {"gen": lambda: step_gen(workers), "train": step_train, "grammars": step_grammars,
         "rank": lambda: step_rank(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
