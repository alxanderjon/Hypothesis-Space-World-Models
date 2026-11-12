"""The short curriculum pilot (owner, 12 November; research log "Owner: a short curriculum pilot, the library ordering the
search's candidates"): the curriculum's first levels, with the search's candidate terms ordered by a DreamCoder-style
library (compression plus counting, context-free form, from the lineage's own verified solutions) and the data checking
each structure by fitting; against the network as now, and four controls of the library arm (shuffled order, a fresh
start at each level, a made-up library, retry-only). Levels 1, 2, 3, 5, 6, 7 of the curriculum (two variables, 1-2 terms
per equation; 10 training and 10 held-out systems per level and seed). Usage: curriculum_pilot.py <workers> <gen|run|report>
[smoke]"""

import json
import os
import pickle
import queue
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
os.environ.setdefault("G2H_RUN", "cpilot_smoke" if SMOKE else "cpilot")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_check as CC                                    # noqa: E402

C, D = CC.C, CC.D
OUT = C.RUN
LEVELS = (1, 2, 3, 5, 6, 7) if not SMOKE else (1, 2)
N_TRAIN, N_TEST = (10, 10) if not SMOKE else (2, 2)
BUDGETS = ((300, 60), (1000, 200), (3000, 600)) if not SMOKE else ((20, 4), (30, 6), (40, 8))
TEST_BUDGET = BUDGETS[0]
MASTERY = 8 if not SMOKE else 1
SEEDS = (1, 2)
N_FANT, STEPS = (50, 500) if not SMOKE else (2, 10)
KINDS = ("lib", "net", "lib-shuffled", "lib-fresh", "lib-madeup", "lib-retry")
_LOCK = threading.Lock()


def log(msg):
    with _LOCK:
        C.log(msg)


class GrammarPrior:
    """The search's prior from a grammar table (log P per term of the 2-variable language and equation): the term to
    add (and the added term of a swap) in proportion to the grammar's probability; removals even."""

    def __init__(self, table):
        self.table = table

    def __call__(self, node, ctx):
        import numpy as np
        import search_g1 as SG
        vocab, i = ctx["vocab"], ctx["i"]
        return SG.Prior(vocab.V, node["idx"], self.table[vocab.gid, i], np.zeros(len(node["idx"])))


def search_job(args):
    rec, prior, seed, budget, cap = args
    import signal
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "level": rec["level"], "status": "done", "budget": budget}
    t0 = time.time()
    signal.signal(signal.SIGALRM, C._alarm)
    signal.alarm(int(2 * C.TIME_CAP + 600))
    try:
        p = C._prior(prior[1]) if prior[0] == "net" else GrammarPrior(np.load(prior[1]))
        res = SG.search_system(np.array(rec["bursts"]), p, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed)
        eqs = []
        for e, true in zip(res["equations"], rec["structure"]):
            sol = e.get("solution")
            eqs.append({"eq": e["eq"], "first_exact": e["effort"]["first_exact"], "structures": e["effort"]["structures"],
                        "time_capped": e["effort"]["time_capped"], "solution": sol["keys"] if sol else None,
                        "true": bool(sol) and sorted(sol["keys"]) == sorted(true)})
        out["equations"] = eqs
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["solved"] = out["status"] == "done" and all(e["first_exact"] is not None for e in out["equations"])
    out["seconds"] = time.time() - t0
    return out


def examples_job(args):
    """Training examples from one verified system (its found structure; unsolved equations skipped)."""
    bursts, structures, seed = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    try:
        return DR.make_examples(np.array(bursts), structures, seed)
    except Exception:
        return []


def fantasy_job(args):
    prefix, k, probs = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import search_g1 as SG
    syn, _ = DR.accepted_fantasy(prefix, k, SG.vocabulary_order(D), np.asarray(probs), D)
    if syn is None:
        return []
    try:
        return DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
    except Exception:
        return []


def train_net_job(args):
    """cycles_g2h.train_job for the network arm: further training from the previous network (or a new one), saving the
    2-variable encodings."""
    paths, prev, out, gi, seed, steps = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import movenet_g2 as MN
    import search_g1 as SG
    MN.SHORTLIST = int(os.environ["G2H_SHORTLIST"])
    torch.manual_seed(seed)
    net = MN.load_net(prev + ".pt") if prev else MN.MoveNetL2()
    device = f"cuda:{gi}"
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in C.DIMS}
    ex = [e for p in paths for e in pickle.loads(Path(p).read_bytes())]
    tl = MN.train(net, ex, tables, steps, batch=C.BATCH, lr=C.LR, device=device, seed=seed, groups=None)
    np.save(f"{out}_d{D}.npy", MN.encode_vocabulary(net, tables[D], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    return {"examples": len(ex), "last_loss": tl[-1]["loss"] if tl else None}


def grammar_table(equations, path, tokens, order):
    """The context-free library grammar (pieces by compression, weights by counting, one parse) from [(own, keys)],
    as log P of every term for both equations; no equations: the language's size prior. Returns the number of pieces."""
    import numpy as np
    import library_ops as L
    terms = Counter(L.tree(k, own) for own, keys in equations for k in keys)
    lib = L.learn(terms, tokens, "uni")[0] if terms else []
    g = L.LibGrammar.fit(equations, [], tokens, lib, form="uni") if equations else L.LibGrammar([], tokens)
    np.save(path, np.array([[g.logp_term(k, own) for own in range(D)] for k in order]))
    return len(lib)


class Lineage:
    def __init__(self, kind, seed, pool, gpus, systems, madeup, tokens, order):
        self.kind, self.seed, self.pool, self.gpus = kind, seed, pool, gpus
        self.systems, self.madeup, self.tokens, self.order = systems, madeup, tokens, order
        self.name = f"{kind}-s{seed}"
        self.dir = OUT / "lineages" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sols = {}                                           # system id -> {eq: keys}, verified solutions
        self.level_of = {}
        self.examples = {}                                       # system id -> replay examples (network arm)
        self.n_learn, self.net = 0, None
        self.prior = ("grammar", str(self.dir / "size.npy"))
        grammar_table([], self.prior[1], tokens, order)
        self.rec = {"kind": kind, "seed": seed, "levels": [], "blocks": [], "rounds": [], "final": None}

    # -- searching --
    def search(self, recs, budget, cap):
        futs = [self.pool.submit(search_job, (r, self.prior, self.seed, budget, cap)) for r in recs]
        return [f.result() for f in futs]

    def wake(self, recs, budget, cap):
        res = self.search(recs, budget, cap)
        for r, x in zip(recs, res):
            self.level_of[r["id"]] = r["level"]
            if x["status"] == "done":
                found = {e["eq"]: e["solution"] for e in x["equations"] if e["first_exact"] is not None and e["solution"]}
                if found:
                    self.sols.setdefault(r["id"], {}).update(found)
        return res

    # -- learning --
    def equations(self, levels=None):
        return [(int(eq), keys) for sid, d in self.sols.items() if levels is None or self.level_of[sid] in levels
                for eq, keys in d.items()]

    def learn(self, level=None, j=0):
        self.n_learn += 1
        tag = f"learn{self.n_learn}"
        if self.kind == "net":
            return self.learn_net(tag)
        if self.kind == "lib-fresh":
            eqs = self.equations({level})
        elif self.kind == "lib-madeup":
            made = [(i, keys) for r in self.madeup[:N_TRAIN * j] for i, keys in enumerate(r["structure"]) if keys]
            eqs = made + self.equations({level})
        else:
            eqs = self.equations()
        path = str(self.dir / f"{tag}.npy")
        n = grammar_table(eqs, path, self.tokens, self.order)
        self.prior = ("grammar", path)
        return {"equations": len(eqs), "pieces": n}

    def learn_net(self, tag):
        import dream_g2 as DR
        import dream_g1 as D1
        import search_g1 as SG
        new = [sid for sid in self.sols if sid not in self.examples]
        recs = {r["id"]: r for lv in LEVELS for split in ("train",) for r in self.systems[lv][split]}
        futs = {sid: self.pool.submit(examples_job, (recs[sid]["bursts"], [self.sols[sid].get(i) for i in range(D)], 1000 + j))
                for j, sid in enumerate(new)}
        for sid, f in futs.items():
            self.examples[sid] = f.result()
        usage = {}
        for sid, d in self.sols.items():
            for eq, keys in d.items():
                for k in keys:
                    rk = DR.role_key(k, int(eq), D)
                    if rk is not None:
                        usage[rk] = usage.get(rk, 0) + 1
        V = json.loads((SG.G1_DIR / f"vocab_d{D}.json").read_text())["terms"]
        probs = D1.term_distribution(self.order, V, usage).tolist()
        fant = [e for f in [self.pool.submit(fantasy_job, (f"pilot-{self.name}-{tag}", k, probs)) for k in range(N_FANT)]
                for e in f.result()]
        replay = [e for sid in self.sols for e in self.examples[sid]]
        p = self.dir / f"{tag}_examples.pkl"
        p.write_bytes(pickle.dumps(replay + fant))
        out = str(self.dir / tag)
        gi = self.gpus.get()
        try:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
                info = one.submit(train_net_job, ([str(p)], self.net, out, gi, self.seed, STEPS)).result()
        finally:
            self.gpus.put(gi)
        p.unlink()                                               # the examples are rebuilt from the solutions if needed
        self.net = out
        self.prior = ("net", out)
        return dict(info, replay=len(replay), fantasies=len(fant), gpu=gi)

    # -- the schedules --
    def run(self):
        t0 = time.time()
        if self.kind in ("lib", "net", "lib-fresh", "lib-madeup"):
            for j, lv in enumerate(LEVELS):
                unsolved, mastered, rounds = list(self.systems[lv]["train"]), False, []
                for r, (budget, cap) in enumerate(BUDGETS):
                    res = self.wake(unsolved, budget, cap)
                    unsolved = [x for x, y in zip(unsolved, res) if not y["solved"]]
                    info = self.learn(lv, j)
                    test = self.search(self.systems[lv]["test"], *TEST_BUDGET)
                    ok = sum(y["solved"] for y in test)
                    rounds.append({"round": r + 1, "budget": budget, "wake": res, "learn": info, "test": test, "test_solved": ok})
                    log(f"{self.name} level {lv} round {r + 1}: training solved {sum(y['solved'] for y in res)} of {len(res)} "
                        f"tried; held-out solved {ok} of {len(test)}")
                    if ok >= MASTERY:
                        mastered = True
                        break
                self.rec["levels"].append({"level": lv, "mastered": mastered, "rounds": rounds})
                if not mastered:
                    log(f"{self.name}: level {lv} not mastered in {len(BUDGETS)} rounds; the lineage stops here")
                    break
        else:
            pool_ = [r for lv in LEVELS for r in self.systems[lv]["train"]]
            if self.kind == "lib-shuffled":
                random.Random(f"shuffle-{self.seed}").shuffle(pool_)
                blocks = [pool_[b * N_TRAIN:(b + 1) * N_TRAIN] for b in range(len(LEVELS))]
            else:
                blocks = [pool_]
            for b, block in enumerate(blocks):
                unsolved, rounds = list(block), []
                for r, (budget, cap) in enumerate(BUDGETS):
                    if not unsolved:
                        break
                    res = self.wake(unsolved, budget, cap)
                    unsolved = [x for x, y in zip(unsolved, res) if not y["solved"]]
                    info = self.learn()
                    rounds.append({"round": r + 1, "budget": budget, "wake": res, "learn": info})
                    log(f"{self.name} block {b + 1} round {r + 1}: solved {sum(y['solved'] for y in res)} of {len(res)} tried")
                self.rec["blocks"].append({"block": b + 1, "rounds": rounds})
        held = [r for lv in LEVELS for r in self.systems[lv]["test"]]
        final = self.search(held, *TEST_BUDGET)
        self.rec["final"] = final
        self.rec["seconds"] = time.time() - t0
        (OUT / "lineages" / f"{self.name}.json").write_text(json.dumps(self.rec, default=float))
        log(f"{self.name}: done in {time.time() - t0:.0f}s; final held-out solved {sum(y['solved'] for y in final)} of {len(final)}")


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    keys = CC.piece_keys()
    jobs = []
    for s in SEEDS:
        for lv in LEVELS:
            allowed = {l_: keys[l_] for l_ in LEVELS if l_ <= lv}
            for split, n in (("train", N_TRAIN), ("test", N_TEST)):
                jobs += [((s, lv, split, k), (f"pilot-s{s}-L{lv}-{split}", k, allowed, lv, CC.TRAIN_TERMS, None, False)) for k in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(CC.gen_job, [a for _, a in jobs]))
    systems = {s: {lv: {"train": [], "test": []} for lv in LEVELS} for s in SEEDS}
    for ((s, lv, split, k), _), r in zip(jobs, res):
        if r["accepted"]:
            systems[s][lv][split].append({"id": f"s{s}-L{lv}-{split}-{k}", "level": lv, "structure": r["structure"],
                                          "bursts": r["bursts"]})
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        (OUT / "gen" / f"systems-s{s}.json").write_text(json.dumps(systems[s]))
    print(f"gen: {sum(r['accepted'] for r in res)} of {len(jobs)} systems accepted in {time.time() - t0:.0f}s")


def step_run(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import grammar_ops as GO
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    tokens = GO.alphabet(order, d=D)
    gpus = queue.Queue()
    free = CC.free_gpus()
    if not free:
        raise SystemExit("no free GPU now; the network arm does not train on the CPU (owner's rule)")
    for g in free[:2]:
        gpus.put(g)
    src = C.ROOT / "outputs/accumulating_learner/g2h/curriculum/gen"
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        lineages = []
        for s in SEEDS:
            systems = {int(k): v for k, v in json.loads((OUT / "gen" / f"systems-s{s}.json").read_text()).items()}
            madeup = pickle.loads((src / f"madeup-s{s}-L0.pkl").read_bytes())
            for kind in KINDS:
                if not (OUT / "lineages" / f"{kind}-s{s}.json").exists():
                    lineages.append(Lineage(kind, s, pool, gpus, systems, madeup, tokens, order))
        log(f"curriculum pilot: {len(lineages)} lineages; levels {LEVELS}; budgets {BUDGETS}; GPUs {free[:2]}; workers {workers}")
        threads = [threading.Thread(target=x.run, name=x.name) for x in lineages]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    log("curriculum pilot: all lineages done")


def step_report():
    import numpy as np
    lines, finals = [], {}
    for s in SEEDS:
        for kind in KINDS:
            p = OUT / "lineages" / f"{kind}-s{s}.json"
            if p.exists():
                finals[(kind, s)] = json.loads(p.read_text())
    lines.append("== Curriculum pilot. Final test: systems solved exactly per level (of 10), seed 1 / seed 2")
    for kind in KINDS:
        cells = []
        for lv in LEVELS:
            cells.append(f"L{lv} " + "/".join(str(sum(y["solved"] for y in finals[(kind, s)]["final"] if y["level"] == lv))
                                              if (kind, s) in finals else "-" for s in SEEDS))
        lines.append(f"   {kind:>13}: " + "; ".join(cells))
    lines.append("== Curriculum order: rounds to mastery per level (x = not mastered, the lineage stopped), seed 1 / seed 2")
    for kind in ("lib", "net", "lib-fresh", "lib-madeup"):
        cells = []
        for lv in LEVELS:
            v = []
            for s in SEEDS:
                L_ = next((x for x in finals.get((kind, s), {}).get("levels", []) if x["level"] == lv), None)
                v.append("-" if L_ is None else (str(len(L_["rounds"])) if L_["mastered"] else "x"))
            cells.append(f"L{lv} " + "/".join(v))
        lines.append(f"   {kind:>13}: " + "; ".join(cells))
    lines.append("== Structures tried before each solution (median over solved equations, all searches), and total structures spent")
    for kind in KINDS:
        parts = []
        for s in SEEDS:
            rec = finals.get((kind, s))
            if rec is None:
                parts.append("-")
                continue
            searches = [y for L_ in rec["levels"] for r in L_["rounds"] for y in r["wake"] + r["test"]] + \
                       [y for B in rec["blocks"] for r in B["rounds"] for y in r["wake"]] + rec["final"]
            fe = [e["first_exact"] for y in searches if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None]
            tot = sum(e["structures"] for y in searches if y["status"] == "done" for e in y["equations"])
            parts.append(f"median {np.median(fe):.0f}, total {tot}" if fe else f"none solved, total {tot}")
        lines.append(f"   {kind:>13}: " + " | ".join(parts))
    comb = {k: [sum(y["solved"] for y in rec["final"] if y["level"] in (6, 7)) for s in SEEDS for rec in [finals.get((k, s))] if rec]
            for k in KINDS}
    mastered_all = all(all(x["mastered"] for x in finals[("lib", s)]["levels"]) and len(finals[("lib", s)]["levels"]) == len(LEVELS)
                       for s in SEEDS if ("lib", s) in finals)
    margin = all(len(comb[k]) == len(SEEDS) and all(comb["lib"][i] >= comb[k][i] + 2 for i in range(len(SEEDS)))
                 for k in KINDS if k != "lib")
    lines.append(f"== Levels 6-7 held-out solved in the final test (of 20), per seed: " + "; ".join(f"{k} {comb[k]}" for k in KINDS))
    lines.append(f"== Pass rule: library masters all levels in both seeds {mastered_all}; beats the network and every control by "
                 f">= 2 on levels 6-7 in both seeds {margin} -> {'PASSES' if mastered_all and margin else 'does not pass'}")
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
        {"gen": lambda: step_gen(workers), "run": lambda: step_run(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
