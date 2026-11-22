"""The ladder (owner, 15 November; research log "Owner: build up slowly (a ladder, one change per rung)", and each rung's
design and mastery criterion, set before its run): rungs that each change one thing, starting from the mastered
curriculum. Arms: the confidence-weighted blend and the network alone, each continuing that seed's lineage (from the
curriculum, then from the previous rung), and a fresh network; 3 seeds; up to 3 rounds per rung (300, then retries at
1,000 and 3,000 structures per equation); after each round: learning, the held-out test, then Reanalyse (the standing
rule). Searches and networks are cached, so a restart resumes.
Usage: ladder.py <workers> <rung> <gen|run|report>   (environment: LADDER_SMOKE=1 for the smoke test)"""

import hashlib
import json
import math
import os
import pickle
import sys
import threading
import time
from collections import Counter
from pathlib import Path

SMOKE = os.environ.get("LADDER_SMOKE") == "1"
os.environ.setdefault("G2H_RUN", "ladder_smoke" if SMOKE else "ladder")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_check as CC                                    # noqa: E402  (the curriculum's pieces)
import curriculum_pilot as CP                                    # noqa: E402  (GrammarPrior)

C = CC.C
G2H = C.ROOT / "outputs/accumulating_learner/g2h"
OUT = C.RUN
RUNGS = {"a": {"d": 2, "terms": (3, 3), "levels": (1, 2, 3, 5, 6, 7), "change": "3 terms per equation", "after": None},
         "a1": {"d": 2, "terms": (3, 3), "levels": (1, 2, 3), "change": "3 terms per equation, only x, x*y, x^2", "after": None},
         "a1w": {"d": 2, "terms": (3, 3), "levels": (1, 2, 3), "change": "3 terms, only x, x*y, x^2; wake pool of 50", "after": None, "n_train": 50},
         "a2": {"d": 2, "terms": (3, 3), "levels": (1, 2, 3, 5, 6, 7), "change": "the rational pieces back (after a1w)", "after": "a1w",
                "n_train": 50, "rung_dreams": 1000},
         "b": {"d": 3, "terms": (1, 2), "levels": (1, 2, 3, 5, 6, 7), "change": "3 variables, 1-2 terms per equation (after a2)", "after": "a2",
               "n_train": 50, "rung_dreams": 1000, "steps": 4500},       # owner, 16 Nov: the same training updates for every arm
         "c": {"d": 3, "terms": (3, 3), "levels": (1, 2, 3, 5, 6, 7), "change": "3 variables, 3 terms per equation (after b)", "after": "b",
               "n_train": 50, "rung_dreams": 1000, "steps": 4500, "own_dreams": 1000}}   # owner, 16 Nov: piece-balanced own dreams replace the general ones
CONSTANT, HILL = 9, 10                                           # piece groups beyond the curriculum's levels 1-8
_E = (1, 2, 3, 5, 6, 7)
RUNGS.update({                                                   # owner, 17 Nov: the harder fields' pieces, one sub-rung each (e1+e2 combined)
    "e12": {"d": 3, "terms": (3, 3), "levels": _E + (4, CONSTANT), "new": (4, CONSTANT), "change": "the constant and x^3 (after c)",
            "after": "c", "n_train": 50, "rung_dreams": 1000, "steps": 4500, "own_dreams": 1000},
    "e3": {"d": 3, "terms": (3, 3), "levels": _E + (4, CONSTANT, 8), "new": (8,), "change": "x/(K+x^2) (after e12)",
           "after": "e12", "n_train": 50, "rung_dreams": 1000, "steps": 4500, "own_dreams": 1000},
    "e4": {"d": 3, "terms": (3, 3), "levels": _E + (4, CONSTANT, 8, HILL), "new": (HILL,), "change": "Hill activation x^2/(K+x^2) (after e3)",
           "after": "e3", "n_train": 50, "rung_dreams": 1000, "steps": 4500, "own_dreams": 1000}})
_ALL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
_R = {"n_train": 50, "rung_dreams": 1000, "steps": 4500, "own_dreams": 1000}
RUNGS.update({                                                   # owner, 20 Nov: new pieces first at 2 variables, 1-2 terms; then climb
    "p8": dict(_R, d=2, terms=(1, 2), levels=(1, 2, 3, 4, 5, 6, 7, 9, 8), new=(8,), change="x/(K+x^2) at 2 variables, 1-2 terms (after e12)", after="e12"),
    "p10": dict(_R, d=2, terms=(1, 2), levels=_ALL, new=(HILL,), change="Hill x^2/(K+x^2) at 2 variables, 1-2 terms (after p8)", after="p8"),
    "q23": dict(_R, d=2, terms=(3, 3), levels=_ALL, change="all ten pieces, 2 variables, 3 terms (after p10)", after="p10"),
    "q31": dict(_R, d=3, terms=(1, 2), levels=_ALL, change="all ten pieces, 3 variables, 1-2 terms (after q23)", after="q23"),
    "q33": dict(_R, d=3, terms=(3, 3), levels=_ALL, change="all ten pieces, 3 variables, 3 terms (after q31)", after="q31"),
    "q41": dict(_R, d=4, terms=(1, 2), levels=_ALL, change="all ten pieces, 4 variables, 1-2 terms (after q33)", after="q33"),
    "q43": dict(_R, d=4, terms=(3, 3), levels=_ALL, change="all ten pieces, 4 variables, 3 terms (after q41)", after="q41")})
_GB = {"budgets": ((3000, 600), (3000, 600), (3000, 600)), "test_budget": (3000, 600), "reanalyse": (1000, 200)}
RUNGS["q41"].update(_GB, after="q33g", change="all ten pieces, 4 variables, 1-2 terms, generous budgets (after q33g)")   # owner, 21 Nov
RUNGS["q43"].update(_GB, change="all ten pieces, 4 variables, 3 terms, generous budgets (after q41)")
_G = {"budgets": ((3000, 600), (3000, 600), (3000, 600)), "test_budget": (3000, 600), "reanalyse": (1000, 200)}   # owner, 20 Nov: generous budgets
RUNGS.update({
    "q33g": dict(_R, **_G, d=3, terms=(3, 3), levels=_ALL, change="all ten pieces, 3 variables, 3 terms, generous budgets (after q31)", after="q31"),
    "q33a": dict(_R, **_G, d=3, terms=(3, 3), levels=_ALL, max_k=1, change="3 variables, 3 terms, at most one K term per equation, generous budgets (after q31)", after="q31"),
    "q33h": dict(_R, **_G, d=3, terms=(3, 3), levels=_ALL, change="all ten pieces, 3 variables, 3 terms, generous budgets (after q33a)", after="q33a")})
OTHER = 99                                                       # piece type of the learner's terms that are no curriculum piece
SHARE_REAL = 0.30                                                # real solutions' share of training when rung dreams are used
SEEDS = (1, 2, 3) if not SMOKE else (1,)
ARMS = ("blend", "net", "fresh")
ARM_NAMES = {"blend": "confidence blend", "net": "network alone", "fresh": "fresh network"}
N_TRAIN, N_TEST = (10, 10) if not SMOKE else (2, 2)
BUDGETS = ((300, 60), (1000, 200), (3000, 600)) if not SMOKE else ((20, 4), (30, 6), (40, 8))
TEST_BUDGET = BUDGETS[0]
REANALYSE = (100, 20) if not SMOKE else (10, 2)
MASTERY = 8 if not SMOKE else 1
N_FANT, STEPS = (50, 500) if not SMOKE else (2, 10)
DIMS = (2, 3, 4)
TOP, SHARE_TOP = 100, 0.95
CURRIC = {("blend", 1): ("cblend", "net-blend-s1", "cpilot/gen/systems-s1.json"),
          ("blend", 2): ("cblend", "net-blend-s2", "cpilot/gen/systems-s2.json"),
          ("blend", 3): ("cseed3", "net-blend-s3", "cseed3/gen/systems-s3.json"),
          ("net", 1): ("cpilot", "net-s1", "cpilot/gen/systems-s1.json"),
          ("net", 2): ("cpilot", "net-s2", "cpilot/gen/systems-s2.json"),
          ("net", 3): ("cseed3", "net-s3", "cseed3/gen/systems-s3.json")}
EARLIER = ["cpilot/gen", "curriculum/gen", "curriculum2/gen", "cseed3/gen"]
_LOCK = threading.Lock()
# Dreams are built in short-lived pools (rung c, 17 Nov): every simulated system loads a compiled library (compile_artifact)
# that is never unloaded, and long-lived workers hit the kernel's 65,530 memory-map limit, after which every build failed
# silently. A pool is replaced every DREAM_CHUNK dreams; at most DREAM_POOLS pools run at once.
DREAM_WORKERS = int(os.environ.get("LADDER_DREAM_WORKERS", "8"))   # 22 Nov: sized to what is free
DREAM_POOLS = int(os.environ.get("LADDER_DREAM_POOLS", "3"))
DREAM_CHUNK = 100
TMP_TRAIN = os.environ.get("LADDER_TMP")                        # 22 Nov: training files on another disk (owner: use both disks)
_DREAM_SEM = threading.Semaphore(DREAM_POOLS)
MIN_EXAMPLES_PER_DREAM = 2.0                                     # below this a dream batch is treated as broken: stop loudly
SHARE_TEACHER_FROM = ("d", "e", "e12", "e3", "e4", "f", "g", "p8", "p10", "q23", "q31", "q33", "q41", "q43")                        # owner, 17 Nov: one teacher-dream set per round, shared by every lineage
_TEACHER, _TEACHER_LOCKS = {}, {}


def log(msg):
    with _LOCK:
        C.log(msg)


def shared_teacher(rung, rnd, allowed, terms, d, n, new=None, max_k=None):
    """The round's teacher dreams: built once by the first lineage to need them, then given to every lineage in that
    round (and saved, so a restarted run reuses them). A new set every round."""
    with _LOCK:
        lock = _TEACHER_LOCKS.setdefault((rung, rnd), threading.Lock())
    with lock:
        if (rung, rnd) not in _TEACHER:
            p = OUT / rung / f"teacher_round{rnd}.pkl"
            if p.exists():
                _TEACHER[(rung, rnd)] = pickle.loads(p.read_bytes())
                log(f"{rung} round {rnd}: shared teacher dreams loaded ({len(_TEACHER[(rung, rnd)])} examples)")
            else:
                t0 = time.time()
                ex, ok = dream_map([(f"ladder-{rung}-round{rnd}-teacher", k, allowed, terms, d, new, False, max_k) for k in range(n)],
                                   f"{rung} round {rnd} shared teacher dreams")
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_bytes(pickle.dumps(ex))
                tmp.replace(p)
                _TEACHER[(rung, rnd)] = ex
                log(f"{rung} round {rnd}: {n} shared teacher dreams built ({ok} with examples; {len(ex)} examples) in {time.time() - t0:.0f}s")
    return _TEACHER[(rung, rnd)]


def dream_map(jobs, what):
    """rung_dream_job over jobs in fresh pools of DREAM_WORKERS, replaced every DREAM_CHUNK jobs; checks the yield."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    out = []
    for c in range(0, len(jobs), DREAM_CHUNK):
        with _DREAM_SEM:
            with ProcessPoolExecutor(DREAM_WORKERS, mp_context=get_context("spawn")) as ex:
                out += list(ex.map(rung_dream_job, jobs[c:c + DREAM_CHUNK]))
    n_ex = sum(len(x) for x in out)
    if jobs and n_ex < MIN_EXAMPLES_PER_DREAM * len(jobs):
        raise RuntimeError(f"{what}: {n_ex} examples from {len(jobs)} dreams ({sum(bool(x) for x in out)} with examples): broken dream batch")
    return [e for x in out for e in x], sum(bool(x) for x in out)


# ---------------------------------------------------------------------------------------------------------------------
# the generator: the curriculum's (curriculum_check.draw), for d variables and no forced new piece
# ---------------------------------------------------------------------------------------------------------------------

def piece_keys(d):
    import library_ops as L
    import search_g1 as SG
    pats = {lv: {CC.canon(p) for p in CC.patterns(lv)} for lv in CC.LEVELS}
    out = {lv: [] for lv in CC.LEVELS}
    for key in SG.vocabulary_order(d):
        c = CC.canon(L.tree(key, 0))
        for lv in CC.LEVELS:
            if c in pats[lv]:
                out[lv].append(key)
    # rung e (owner, 17 Nov): the constant (inflow) and Hill activation x^2/(K+x^2), beyond the curriculum's levels
    out[CONSTANT] = [k for k in SG.vocabulary_order(d) if k == "Integer(1)"]
    out[HILL] = [k for k in SG.vocabulary_order(d) if k.startswith("Mul(Pow(Symbol") and k.count("Symbol('x") == 2
                 and "Integer(2)), Pow(Add(Symbol('K0', real=True), Pow(" in k and k.split("Pow(Symbol(")[1][:4] == k.split("Pow(Symbol(")[2][:4]]
    return out


K_SHIFT = {5: "x", 6: "x", 7: "x2", 8: "x2", 10: "x2"}        # pieces with K inside: the quantity K is added to
_OF = {}


def piece_of(key, d):
    if d not in _OF:
        _OF[d] = {k: lv for lv, ks in piece_keys(d).items() for k in ks}
    return _OF[d].get(key)


def shifted_var(key):
    """The (1-based) variable inside the K-shifted denominator of a key: Add(K0, x_j) or Add(K0, Pow(x_j, 2))."""
    import re
    m = re.search(r"Add\(Symbol\('K0', real=True\), (?:Pow\()?Symbol\('x(\d)'", key)
    return int(m.group(1)) if m else None


def k_ok(desc, bursts):
    """The K rule's check on the recordings (owner, 19 Nov): every K-term's K lies between the 25th and 90th percentiles of
    its shifted quantity (|x| or x^2) over the recorded bursts."""
    import numpy as np
    b = np.asarray(bursts, dtype=float)
    for i, j, K, kind in desc.get("kterms", []):
        x = b[..., j].ravel()
        q = np.abs(x) if kind == "x" else x ** 2
        lo, hi = np.percentile(q, 25), np.percentile(q, 90)
        if not (lo <= K <= hi):
            return False
    return True


def draw(seed, allowed, n_terms, d, krule=False, max_k=None):
    """krule (owner, 19 Nov; rung e3k on): a piece with K inside gets a positive K drawn log-uniformly between the 25th and
    90th percentiles of its shifted quantity (|x| or x^2) over the starting region; k_ok checks it on the recordings."""
    import numpy as np
    import sympy as sp
    import torch
    import domains_g1 as DG
    import fitter_g1 as FG
    import language_g1 as LG
    rng = DG.rng_of(seed)
    Y, X = DG.Y4, LG.variables(4)
    params, rhs, kterms = {}, [], []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    for i in range(d):
        n = int(rng.integers(n_terms[0], n_terms[1] + 1))
        picks, nk = [], 0
        while len(picks) < n:
            lv = int(rng.choice(sorted(allowed)))
            if max_k is not None and lv in K_SHIFT and nk >= max_k:   # q33a: at most max_k terms with K inside per equation
                continue
            key = allowed[lv][int(rng.integers(len(allowed[lv])))]
            if key not in picks:
                picks.append(key)
                nk += lv in K_SHIFT
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
                lv = piece_of(key, d) if krule else None
                if lv in K_SHIFT and inf.k == 1 and shifted_var(key):
                    j = actual[shifted_var(key) - 1]
                    xs = Wp[:, j].numpy()
                    q = np.abs(xs) if K_SHIFT[lv] == "x" else xs ** 2
                    lo, hi = np.percentile(q, 25), np.percentile(q, 90)
                    th = np.array([float(np.exp(rng.uniform(np.log(lo), np.log(hi))))])
                    kterms.append([i, int(j), float(th[0]), K_SHIFT[lv]])
                for a in range(inf.k):
                    name = f"c{i}_{t}_{a}"
                    params[name] = float(th[a])
                    sub[sp.Symbol(f"K{a}", real=True)] = sp.Symbol(name)
            cname = f"a{i}_{t}"
            params[cname] = float(rng.choice([-1.0, 1.0]) * math.exp(rng.uniform(math.log(0.1), math.log(2.0))))
            e = e + sp.Symbol(cname) * expr.xreplace(sub)
        rhs.append(e)
    return {"seed": seed, "domain": "curriculum", "form": "curriculum", "d": d, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False, **({"kterms": kterms} if krule else {})}


def new_keys(rung, keys):
    """The rung's new-piece keys (role-relative), or None."""
    return sorted({k for lv in RUNGS[rung].get("new", ()) for k in keys[lv]}) or None


def has_new(structure, new, d):
    """At least one true term of the new pieces (owner, 17 Nov: every rung-e system contains one)."""
    import dream_g2 as DR
    return new is None or any(DR.role_key(k, i, d) in set(new) for i, eq in enumerate(structure) for k in eq)


def gen_job(args):
    prefix, k, allowed, n_terms, d = args[:5]
    new = args[5] if len(args) > 5 else None
    krule = args[6] if len(args) > 6 else False
    max_k = args[7] if len(args) > 7 else None
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    reasons = []
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw(seed, allowed, n_terms, d, krule, max_k)
        why = DR.prescreen(desc)
        if why is not None:
            reasons.append(why)
            continue
        try:
            syn = DG.build(desc)
        except Exception as ex:
            reasons.append(type(ex).__name__)
            continue
        if not has_new([eq["keys"] for eq in syn.structure], new, d):
            reasons.append("no new-piece term")
            continue
        b8 = syn.bursts(8)
        if krule and not k_ok(desc, b8):
            reasons.append("K outside the recorded range")
            continue
        return {"k": k, "accepted": True, "tries": r + 1, "structure": [eq["keys"] for eq in syn.structure],
                "bursts": b8.tolist()}
    return {"k": k, "accepted": False, "reasons": reasons[-5:]}


def data_key(bursts):
    import numpy as np
    return hashlib.sha1(np.round(np.asarray(bursts, dtype=float), 10).tobytes()).hexdigest()


def earlier_keys(rung):
    keys = set()
    dirs = [G2H / d for d in EARLIER] + [OUT / "gen"]
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.name.startswith(f"{rung}-"):
                continue
            if p.suffix == ".json":
                obj = json.loads(p.read_text())
                recs = obj if isinstance(obj, list) else [r for v in obj.values() for sp in (v.values() if isinstance(v, dict) else [v]) for r in (sp if isinstance(sp, list) else [])]
            elif p.suffix == ".pkl":
                obj = pickle.loads(p.read_bytes())
                recs = obj if isinstance(obj, list) else []
            else:
                continue
            keys |= {data_key(r["bursts"]) for r in recs if isinstance(r, dict) and r.get("bursts") is not None}
    return keys


def step_gen(workers, rung):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    R = RUNGS[rung]
    keys = piece_keys(R["d"])
    allowed = {lv: keys[lv] for lv in R["levels"]}
    new = new_keys(rung, keys)
    jobs = [((s, split, k), (f"ladder-{rung}-s{s}-{split}", k, allowed, R["terms"], R["d"], new, False, R.get("max_k")))
            for s in SEEDS for split, n in (("train", R.get("n_train", N_TRAIN)), ("test", N_TEST)) for k in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(gen_job, [a for _, a in jobs]))
    old = earlier_keys(rung)
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    out = {s: {"train": [], "test": []} for s in SEEDS}
    for ((s, split, k), _), r in zip(jobs, res):
        if r["accepted"]:
            out[s][split].append({"id": f"{rung}-s{s}-{split}-{k}", "rung": rung, "structure": r["structure"], "bursts": r["bursts"]})
    new = [data_key(r["bursts"]) for s in SEEDS for sp in out[s].values() for r in sp]
    clash, dup = sum(k in old for k in new), len(new) - len(set(new))
    print(f"gen {rung}: {sum(r['accepted'] for r in res)} of {len(jobs)} accepted in {time.time() - t0:.0f}s; per seed "
          f"{ {s: {sp: len(v) for sp, v in out[s].items()} for s in SEEDS} }; against {len(old)} earlier systems: {clash} identical, "
          f"{dup} repeated")
    if clash or dup:
        raise SystemExit("the rung's systems are not all new: not writing them")
    for s in SEEDS:
        (OUT / "gen" / f"{rung}-s{s}.json").write_text(json.dumps(out[s]))


# ---------------------------------------------------------------------------------------------------------------------
# priors and jobs
# ---------------------------------------------------------------------------------------------------------------------

class ConfidencePrior:
    """(1 - u) * p_network + u * p_blend, u = 1 - the network's top probability over the legal additions; p_blend the
    plain top-100 blend; removals stay the network's (quick check 1's rule). tables: {d: library table}."""

    def __init__(self, base, tables):
        self.base, self.tables = base, tables

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, node, ctx):
        import numpy as np
        import muzero_checks as MZ
        pr = self.base(node, ctx)
        legal = pr.legal_add
        d = ctx["W"].shape[1]
        if not legal.any() or d not in self.tables:
            return pr
        lib = self.tables[d][ctx["vocab"].gid, ctx["i"]]
        pn = np.where(legal, pr.p_swap_add, 0.0)
        pn = pn / pn.sum()
        pb = MZ.blend_dist(lib, pn, legal)
        u = 1.0 - float(pn.max())
        mix = (1.0 - u) * pn + u * pb
        share = pr.p_add.sum()
        if share > 0:
            pr.p_add = share * mix
        pr.p_swap_add = mix
        return pr


class GrammarByD:
    def __init__(self, tables):
        self.tables = tables

    def __call__(self, node, ctx):
        return CP.GrammarPrior(self.tables[ctx["W"].shape[1]])(node, ctx)


def make_prior(spec):
    """("grammar", {d: table}) | ("net", path) | ("confidence", path, {d: table})."""
    import numpy as np
    import movenet_g2 as MN
    if spec[0] == "grammar":
        return GrammarByD({int(d): np.load(p) for d, p in spec[1].items()})
    emb = {d: np.load(f"{spec[1]}_d{d}.npy") for d in DIMS if Path(f"{spec[1]}_d{d}.npy").exists()}
    base = MN.NetworkPrior(MN.load_net(spec[1] + ".pt").eval(), emb, shortlist=2048)
    if spec[0] == "net":
        return base
    return ConfidencePrior(base, {int(d): np.load(p) for d, p in spec[2].items()})


def search_job(args):
    rec, spec, seed, budget, cap = args
    import signal
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_g1 as SG
    out = {"id": rec["id"], "status": "done", "budget": budget}
    t0 = time.time()
    signal.signal(signal.SIGALRM, C._alarm)
    signal.alarm(int(2 * C.TIME_CAP + 600))
    try:
        res = SG.search_system(np.array(rec["bursts"]), make_prior(spec), budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed)
        out["equations"] = [{"eq": e["eq"], "first_exact": e["effort"]["first_exact"], "structures": e["effort"]["structures"],
                             "time_capped": e["effort"]["time_capped"], "solution": e["solution"]["keys"] if e.get("solution") else None}
                            for e in res["equations"]]
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
    prefix, k, probs, d = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import search_g1 as SG
    syn, _ = DR.accepted_fantasy(prefix, k, SG.vocabulary_order(d), np.asarray(probs), d)
    if syn is None:
        return []
    try:
        return DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
    except Exception:
        return []


def train_net_job(args):
    """Further training from the previous network (or a new one); encodings saved for 2, 3 and 4 variables."""
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
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in DIMS}
    ex = [e for p in paths for e in pickle.loads(Path(p).read_bytes())]
    tl = MN.train(net, ex, tables, steps, batch=C.BATCH, lr=C.LR, device=device, seed=seed, groups=None)
    for d in DIMS:
        np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    return {"examples": len(ex), "last_loss": tl[-1]["loss"] if tl else None}


def rung_dream_job(args):
    """One accepted dream from the rung's own generator, as training examples (the a2 dream check's version b); with new
    pieces given, only dreams containing one of them (teacher dreams of rung e)."""
    prefix, k, allowed, n_terms, d = args[:5]
    new = args[5] if len(args) > 5 else None
    krule = args[6] if len(args) > 6 else False
    max_k = args[7] if len(args) > 7 else None
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw(seed, allowed, n_terms, d, krule, max_k)
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
            if not has_new([eq["keys"] for eq in syn.structure], new, d):
                continue
            b8 = syn.bursts(8)
            if krule and not k_ok(desc, b8):
                continue
            return DR.make_examples(b8, [eq["keys"] for eq in syn.structure], seed=10_000 + k)
        except Exception:
            continue
    return []


def table_job(args):
    """The library grammar (context-free form; compression plus counting) as log P over the d-variable language."""
    equations, path, d = args
    import numpy as np
    import grammar_ops as GO
    import library_ops as L
    import search_g1 as SG
    order = SG.vocabulary_order(d)
    tokens = GO.alphabet(order, d=d)
    terms = Counter(L.tree(k, own) for own, keys in equations for k in keys)
    lib = L.learn(terms, tokens, "uni")[0] if terms else []
    g = L.LibGrammar.fit(equations, [], tokens, lib, form="uni") if equations else L.LibGrammar([], tokens)
    np.save(path, np.array([[g.logp_term(k, own) for own in range(d)] for k in order]))
    return len(lib)


def stable(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------------------------------------------------
# GPUs: those free at the start; one training at a time each; nvidia-smi before each training
# ---------------------------------------------------------------------------------------------------------------------

class GPUs:
    def __init__(self):
        rows = self.rows()
        self.free = [g for g, (_, u, m, tot) in sorted(rows.items()) if u < 10 and m < 1000]
        excl = {int(x) for x in os.environ.get("LADDER_EXCLUDE_GPUS", "").split(",") if x}   # another run's GPUs (22 Nov)
        self.free = [g for g in self.free if g not in excl]
        cap = int(os.environ.get("LADDER_MAX_TRAIN", "0"))      # at 4 variables a training's example file is ~2 GB (21 Nov)
        if cap:
            self.free = self.free[:cap]
        self.slots = {g: threading.Semaphore(1) for g in self.free}
        log(f"ladder GPUs: {self.free} (free at the start); one training at a time each")

    @staticmethod
    def rows():
        import subprocess
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        return {int(r[0]): [int(x) for x in r] for r in (line.split(",") for line in q.strip().splitlines())}

    def run(self, fn):
        if not self.slots:
            raise SystemExit("no free GPU: the network does not train on the CPU (owner's rule)")
        while True:
            rows = self.rows()
            for g, sem in self.slots.items():
                _, u, m, tot = rows[g]
                if tot - m >= 8000 and u <= 90 and sem.acquire(blocking=False):
                    try:
                        return fn(g)
                    finally:
                        sem.release()
            time.sleep(15)


# ---------------------------------------------------------------------------------------------------------------------
# a lineage on one rung
# ---------------------------------------------------------------------------------------------------------------------

class Lineage:
    def __init__(self, arm, seed, rung, pool, gpus):
        self.arm, self.seed, self.rung, self.pool, self.gpus = arm, seed, rung, pool, gpus
        self.d = RUNGS[rung]["d"]
        self.name = f"{arm}-s{seed}"
        self.dir = OUT / rung / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sols, self.recs, self.tried = {}, {}, []
        self.net, self.tables, self.n_learn = None, {}, 0
        self.canon, self.rec_files = {}, []
        if arm != "fresh":
            self.start()
        self.examples_path = self.dir / "examples.pkl"
        self.examples = pickle.loads(self.examples_path.read_bytes()) if self.examples_path.exists() else {}
        self.rec = {"arm": arm, "seed": seed, "rung": rung, "rounds": [], "mastered": False, "start": self.start_info()}

    def start_info(self):
        return {"solved systems": len(self.sols), "tried": len(self.tried), "network": self.net, "tables": self.tables}

    def start(self):
        """The lineage's state at the end of the previous rung, or of the curriculum for the first rung."""
        prev = RUNGS[self.rung]["after"]
        if prev:
            st = json.loads((G2H / "ladder" / prev / f"{self.name}.state.json").read_text())   # the ladder's own folder
            self.sols = {k: {int(e): v for e, v in d.items()} for k, d in st["sols"].items()}
            self.tried, self.net, self.tables = st["tried"], st["net"], {int(k): v for k, v in st["tables"].items()}
            for f in st["record_files"]:
                self.add_records(f)
            return
        base, name, sysfile = CURRIC[(self.arm, self.seed)]
        rec = json.loads((G2H / base / "lineages" / f"{name}.json").read_text())
        rounds = [r for L_ in rec["levels"] for r in L_["rounds"]]
        for r in rounds:
            for y in r["wake"]:
                if y["id"] not in self.tried:
                    self.tried.append(y["id"])
                if y["status"] == "done":
                    for e in y["equations"]:
                        if e["first_exact"] is not None and e["solution"]:
                            self.sols.setdefault(y["id"], {})[int(e["eq"])] = e["solution"]
        n = len(rounds)
        self.net = str(G2H / base / "lineages" / name / f"learn{n}")
        if self.arm == "blend":
            self.tables = {2: str(G2H / base / "lineages" / name / f"library{n}.npy")}
        self.add_records(str(G2H / sysfile))

    def add_records(self, path):
        """Training records from a curriculum systems file ({level: {train, test}}) or a rung's ({train, test})."""
        obj = json.loads(Path(path).read_text())
        for part in ([obj] if "train" in obj else obj.values()):
            for r in part["train"]:
                self.recs[r["id"]] = r
        self.rec_files.append(path)

    def spec(self):
        if self.net is None:
            return ("grammar", self.size_tables())
        if self.arm == "blend":
            return ("confidence", self.net, self.tables)
        return ("net", self.net)

    def size_tables(self):
        out = {}
        for d in sorted({self.d, 2}):
            p = OUT / "size" / f"size_d{d}.npy"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                self.pool.submit(table_job, ([], str(p), d)).result()
            out[d] = str(p)
        return out

    def search(self, path, recs, budget, cap):
        if path.exists():
            return json.loads(path.read_text())
        spec = self.spec()
        res = list(self.pool.map(search_job, [(r, spec, self.seed, budget, cap) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        return res

    def add(self, res):
        for x in res:
            if x["status"] != "done":
                continue
            for e in x["equations"]:
                if e["first_exact"] is not None and e["solution"]:
                    self.sols.setdefault(x["id"], {})[int(e["eq"])] = e["solution"]

    def unsolved(self, ids):
        return [s for s in ids if any(i not in self.sols.get(s, {}) for i in range(len(self.recs[s]["structure"])))]

    # -- learning --
    def learn(self):
        import dream_g2 as DR
        import dream_g1 as D1
        import search_g1 as SG
        self.n_learn += 1
        tag = f"learn{self.n_learn}"
        out = str(self.dir / tag)
        # examples are rebuilt whenever a system's set of solved equations changes (e.g. an equation Reanalyse rescued)
        new = [s for s in self.sols if self.examples.get(s, {}).get("eqs") != sorted(self.sols[s])]
        futs = {s: self.pool.submit(examples_job, (self.recs[s]["bursts"], [self.sols[s].get(i) for i in range(len(self.recs[s]["structure"]))],
                                                   stable(s, "examples", sorted(self.sols[s])))) for s in new}
        for s, f in futs.items():
            self.examples[s] = {"eqs": sorted(self.sols[s]), "ex": f.result()}
        self.examples_path.write_bytes(pickle.dumps(self.examples))
        if not Path(out + ".pt").exists():
            usage = {}
            for s, eqs in self.sols.items():
                dd = len(self.recs[s]["structure"])
                if dd != self.d:
                    continue
                for eq, keys in eqs.items():
                    for k in keys:
                        rk = DR.role_key(k, int(eq), dd)
                        if rk is not None:
                            usage[rk] = usage.get(rk, 0) + 1
            V = json.loads((SG.G1_DIR / f"vocab_d{self.d}.json").read_text())["terms"]
            probs = D1.term_distribution(SG.vocabulary_order(self.d), V, usage).tolist()
            R = RUNGS[self.rung]
            if R.get("own_dreams"):
                # piece-balanced own dreams (owner, 16 Nov; passed the a2 own-dream check 6 of 6): a piece type the learner
                # has used, uniformly, then a used term of it, uniformly; the rung's term count. They replace the general dreams.
                of = {k: lv for lv, ks in piece_keys(self.d).items() for k in ks}
                order = set(SG.vocabulary_order(self.d))
                groups = {}
                for k in sorted(usage):
                    if k in order:
                        groups.setdefault(of.get(k, OTHER), []).append(k)
                n_own = (R["own_dreams"] if not SMOKE else 4) if sum(len(v) for v in groups.values()) >= R["terms"][1] else 0
                fant, ok = dream_map([(f"ladder-{self.rung}-{self.name}-{tag}-own", k, groups, R["terms"], self.d) for k in range(n_own)],
                                     f"{self.rung} {self.name} {tag} own dreams")
                log(f"{self.rung} {self.name} {tag}: {n_own} piece-balanced own dreams from {sum(len(v) for v in groups.values())} used terms "
                    f"in {len(groups)} piece types ({ok} with examples; {len(fant)} examples)")
            else:
                fant = [e for f in [self.pool.submit(fantasy_job, (f"ladder-{self.rung}-{self.name}-{tag}", k, probs, self.d)) for k in range(N_FANT)]
                        for e in f.result()]
            replay = [e for s in self.sols for e in self.examples[s]["ex"]]
            steps, data = STEPS, replay + fant
            if R.get("rung_dreams"):
                keys = piece_keys(self.d)
                allowed = {lv: keys[lv] for lv in R["levels"]}
                n = R["rung_dreams"] if not SMOKE else 4
                if self.rung in SHARE_TEACHER_FROM:
                    rd = shared_teacher(self.rung, self.n_learn, allowed, R["terms"], self.d, n, new_keys(self.rung, keys), R.get("max_k"))
                else:
                    rd, ok = dream_map([(f"ladder-{self.rung}-{self.name}-{tag}-rung", k, allowed, R["terms"], self.d) for k in range(n)],
                                       f"{self.rung} {self.name} {tag} teacher dreams")
                    log(f"{self.rung} {self.name} {tag}: {n} teacher dreams ({ok} with examples; {len(rd)} examples)")
                others = fant + rd
                m = max(1, round(SHARE_REAL * len(others) / ((1 - SHARE_REAL) * max(len(replay), 1))))
                data = replay * m + others
                steps = round(STEPS * len(data) / max(len(replay) + len(fant), 1))
            if R.get("steps"):
                steps = R["steps"] if not SMOKE else 10
            p = (Path(TMP_TRAIN) / f"{self.rung}-{self.name}-{tag}_examples.pkl") if TMP_TRAIN else self.dir / f"{tag}_examples.pkl"

            def train(g):
                """The examples file is written only once a GPU slot is held (21 Nov: at 4 variables each is ~2 GB); with
                LADDER_TMP, on that disk, and only with 2 GB to spare after it (22 Nov)."""
                import shutil
                from concurrent.futures import ProcessPoolExecutor
                from multiprocessing import get_context
                blob = pickle.dumps(data)
                while shutil.disk_usage(p.parent).free < len(blob) + 2 * 2 ** 30:
                    log(f"{self.rung} {self.name} {tag}: waiting for disk space on {p.parent}")
                    time.sleep(120)
                p.write_bytes(blob)
                del blob
                try:
                    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
                        return one.submit(train_net_job, ([str(p)], self.net, out, g, stable(self.name, self.rung, tag), steps)).result(), g
                finally:
                    p.unlink(missing_ok=True)
            info, g = self.gpus.run(train)
            log(f"{self.rung} {self.name} {tag}: trained {steps} steps on {len(data)} examples ({len(replay)} replay, {len(fant)} {'own (piece-balanced)' if R.get('own_dreams') else 'general'} dream"
                f"{', with rung dreams' if R.get('rung_dreams') else ''}) on GPU {g}")
        self.net = out
        if self.arm == "blend":
            self.rebuild_library(tag)

    def rebuild_library(self, tag):
        import canon_check as CN
        todo = {}
        for s, eqs in self.sols.items():
            for eq, keys in eqs.items():
                if (s, eq, tuple(sorted(keys))) not in self.canon:
                    todo.setdefault(s, {})[str(eq)] = keys
        futs = {s: self.pool.submit(CN.canon_job, (self.recs[s], d)) for s, d in todo.items()}
        for s, f in futs.items():
            for eq, e in f.result()["equations"].items():
                self.canon[(s, int(eq), tuple(sorted(e["raw"])))] = e["canon_plus"]
        eqs = [(eq, self.canon[(s, eq, tuple(sorted(keys)))]) for s, d in self.sols.items() for eq, keys in d.items()]
        for d in sorted({self.d, 2}):
            p = self.dir / f"library_{tag}_d{d}.npy"
            if not p.exists():
                self.pool.submit(table_job, (eqs, str(p), d)).result()
            self.tables[d] = str(p)

    # -- the rung --
    def run(self, systems):
        train, test = systems["train"], systems["test"]
        for r in train + test:
            self.recs[r["id"]] = r
        for r in train:
            if r["id"] not in self.tried:
                self.tried.append(r["id"])
        todo = [r["id"] for r in train]
        R = RUNGS[self.rung]
        budgets = R.get("budgets", BUDGETS) if not SMOKE else BUDGETS
        test_budget = R.get("test_budget", TEST_BUDGET) if not SMOKE else TEST_BUDGET
        reanalyse = R.get("reanalyse", REANALYSE) if not SMOKE else REANALYSE
        for k, (budget, cap) in enumerate(budgets):
            rnd = k + 1
            t_round = time.time()
            w = self.search(self.dir / f"r{rnd}_wake.json", [self.recs[s] for s in todo], budget, cap)
            self.add(w)
            todo = self.unsolved(todo)
            self.learn()
            t = self.search(self.dir / f"r{rnd}_test.json", test, *test_budget)
            ok = sum(y["solved"] for y in t)
            mastered = ok >= MASTERY
            another = not mastered and rnd < len(budgets) and bool(todo)
            pool_ = [s for s in self.unsolved(self.tried) if not (another and s in todo)]
            ra = self.search(self.dir / f"r{rnd}_reanalyse.json", [self.recs[s] for s in pool_], *reanalyse) if pool_ else []
            before = {(s, i) for s in pool_ for i in self.sols.get(s, {})}
            self.add(ra)
            new_eq = sum(1 for s in pool_ for i in self.sols.get(s, {}) if (s, i) not in before)
            new_sys = sum(1 for s in pool_ if s not in self.unsolved([s]))
            yld = {"re_searched": len(pool_), "new_equations": new_eq, "new_systems": new_sys,
                   "structures": sum(e["structures"] for x in ra if x["status"] == "done" for e in x["equations"])}
            self.rec["rounds"].append({"round": rnd, "budget": budget, "wake": w, "test": t, "test_solved": ok, "reanalyse": yld,
                                       "seconds": time.time() - t_round})
            log(f"{self.rung} {self.name} round {rnd}: training solved {sum(y['solved'] for y in w)} of {len(w)} tried; held-out "
                f"{ok} of {len(t)}; Reanalyse {len(pool_)} re-searched, {new_eq} equations and {new_sys} systems newly solved")
            if mastered:
                self.rec["mastered"] = True
                break
            if not todo and not mastered and rnd < len(budgets):
                continue
        self.rec["rounds_to_mastery"] = len(self.rec["rounds"]) if self.rec["mastered"] else None
        (OUT / self.rung / f"{self.name}.json").write_text(json.dumps(self.rec, default=float))
        st = {"sols": self.sols, "tried": self.tried, "net": self.net, "tables": self.tables,
              "record_files": self.rec_files + [str(OUT / "gen" / f"{self.rung}-s{self.seed}.json")]}
        (OUT / self.rung / f"{self.name}.state.json").write_text(json.dumps(st))
        log(f"{self.rung} {self.name}: {'mastered in ' + str(self.rec['rounds_to_mastery']) + ' rounds' if self.rec['mastered'] else 'not mastered'}")


def step_run(workers, rung):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    gpus = GPUs()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        jobs = []
        only = [int(x) for x in os.environ.get("LADDER_SEEDS", "").split(",") if x]   # 22 Nov: a seed at a time (memory)
        for s in SEEDS:
            if only and s not in only:
                continue
            systems = json.loads((OUT / "gen" / f"{rung}-s{s}.json").read_text())
            for arm in ARMS:
                if not (OUT / rung / f"{arm}-s{s}.json").exists():
                    jobs.append((Lineage(arm, s, rung, pool, gpus), systems))
        log(f"ladder rung {rung} ({RUNGS[rung]['change']}): {len(jobs)} lineages; budgets {RUNGS[rung].get('budgets', BUDGETS)}; test {RUNGS[rung].get('test_budget', TEST_BUDGET)}; Reanalyse {RUNGS[rung].get('reanalyse', REANALYSE)}; time cap {C.TIME_CAP:.0f}s per system; workers {workers}")
        threads = [threading.Thread(target=x.run, args=(sy,), name=x.name) for x, sy in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    log(f"ladder rung {rung}: done")


def step_report(rung):
    import numpy as np
    lines = [f"== Ladder rung {rung} ({RUNGS[rung]['change']}): held-out systems solved per round (of {N_TEST}); mastery at {MASTERY}"]
    verdict = {}
    for arm in ARMS:
        m, rows = 0, []
        for s in SEEDS:
            p = OUT / rung / f"{arm}-s{s}.json"
            if not p.exists():
                rows.append(f"s{s}: -")
                continue
            r = json.loads(p.read_text())
            m += r["mastered"]
            fe = [e["first_exact"] for x in r["rounds"] for y in x["wake"] if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None]
            ra = [x["reanalyse"] for x in r["rounds"]]
            mins = " / ".join(str(round(x.get("seconds", 0) / 60)) for x in r["rounds"])
            first = sum(y["solved"] for y in r["rounds"][0]["wake"])
            rows.append(f"s{s}: {' / '.join(str(x['test_solved']) for x in r['rounds'])} [minutes per round {mins}; first-wake solves {first} of {len(r['rounds'][0]['wake'])}] ({'mastered in ' + str(r['rounds_to_mastery']) if r['mastered'] else 'not mastered'}; "
                        f"median structures before a training solution {np.median(fe) if fe else float('nan'):.0f}; Reanalyse "
                        f"{sum(x['new_equations'] for x in ra)} equations from {sum(x['re_searched'] for x in ra)} re-searched)")
        verdict[arm] = m >= 2 if not SMOKE else m >= 1
        lines.append(f"   {ARM_NAMES[arm]:>16}: " + "; ".join(rows) + f" -> {'masters' if verdict[arm] else 'does not master'} the rung")
        if "budgets" in RUNGS[rung]:
            ext, bok = [], 0
            for s in SEEDS:
                p = OUT / rung / f"{arm}-s{s}.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                med, w300, capped = [], [], 0
                for x in r["rounds"]:
                    fe = [e["first_exact"] for y in x["test"] if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None]
                    med.append(float(np.median(fe)) if fe else float("nan"))
                    w300.append(sum(y["status"] == "done" and all(e["first_exact"] is not None and e["first_exact"] <= 300 for e in y["equations"]) for y in x["test"]))
                    capped += sum(e["time_capped"] for y in x["wake"] + x["test"] if y["status"] == "done" for e in y["equations"])
                import paired_tries as PT                         # owner, 21 Nov: (b) on paired tries from the rung after q33g
                pt = PT.paired(r)
                b = pt is None or not (pt[2] > pt[1])
                bok += b
                med_txt = "one round" if pt is None else f"paired {pt[0]} equations {pt[1]:.0f} -> {pt[2]:.0f}"
                ext.append(f"s{s}: held-out median tries per solution {' / '.join(f'{v:.0f}' for v in med)} ({med_txt}); held-out solved within 300 "
                           f"{' / '.join(map(str, w300))}; time-capped equations {capped}{'' if b else ' (paired tries rose)'}")
            lines.append(f"   {'':>16}  " + "; ".join(ext) + f" -> (b) {'holds' if bok >= 2 else 'does not hold'} ({bok} of 3)")
    if RUNGS[rung].get("new"):
        # the new pieces' own solve rate on held-out equations (counts only), final round, against the other equations
        import dream_g2 as DR
        d = RUNGS[rung]["d"]
        keys = piece_keys(d)
        new = set(new_keys(rung, keys))
        hill = set(keys[HILL]) if HILL in RUNGS[rung]["new"] else None
        lines.append("   held-out equations solved in the final round: with a new-piece term / without" + ("; Hill equations solved with a Hill term" if hill else ""))
        for arm in ARMS:
            parts = []
            for s in SEEDS:
                p = OUT / rung / f"{arm}-s{s}.json"
                if not p.exists():
                    continue
                st = {x["id"]: x["structure"] for x in json.loads((OUT / "gen" / f"{rung}-s{s}.json").read_text())["test"]}
                t = json.loads(p.read_text())["rounds"][-1]["test"]
                a = b = na = nb = hh = nh = 0
                for y in t:
                    if y["status"] != "done":
                        continue
                    for e in y["equations"]:
                        i = int(e["eq"])
                        true = {DR.role_key(k, i, d) for k in st[y["id"]][i]}
                        ok = e["first_exact"] is not None
                        if true & new:
                            na += 1; a += ok
                        else:
                            nb += 1; b += ok
                        if hill and true & hill and ok:
                            nh += 1; hh += any(DR.role_key(k, i, d) in hill for k in (e["solution"] or []))
                parts.append(f"s{s} {a}/{na} vs {b}/{nb}" + (f"; Hill form {hh}/{nh}" if hill else ""))
            lines.append(f"   {ARM_NAMES[arm]:>16}: " + "; ".join(parts))
    drop = not (verdict.get("blend") or verdict.get("net"))
    lines.append(f"== Rung {rung}: {'DROP-OFF: neither learner arm masters it (stop and report)' if drop else 'mastered by ' + ', '.join(ARM_NAMES[a] for a in ('blend', 'net') if verdict.get(a))}")
    text = "\n".join(lines)
    (OUT / rung / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers, rung, step = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / rung).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT.resolve()).free / 2 ** 30
    if free < 2:
        raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {OUT.resolve()} (owner's crash guard)")
    {"gen": lambda: step_gen(workers, rung), "run": lambda: step_run(workers, rung), "report": lambda: step_report(rung)}[step]()


if __name__ == "__main__":
    main()
