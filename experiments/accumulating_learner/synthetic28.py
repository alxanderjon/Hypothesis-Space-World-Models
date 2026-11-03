"""Trial 28 step 2: synthetic systems built from the search vocabulary (generator v1, docs/research-log.md, 3 November).

Biochemistry-like positive systems: every equation is a sum of terms whose types are tokens of structure_search.Vocab28,
so each equation's true structure is known as a token set. The settle time follows the Trial 23 transient rule, and
simulation, references and bursts follow the Trial 27 rules for transients (harness27.Reference27), with the
right-hand side compiled exactly as for real systems.

  draw(seed, mode)    a system description (terms with physical constants and tokens); mode "train" excludes the held-
                      out family combinations, mode "test2" puts at least one of them in some equation
  build(desc)         compiled system, reference and acceptance checks (cheap-simulation cutoff and the logged checks);
                      returns a Synthetic object or raises Rejected with the reason
  Synthetic           .ref (reference, as harness27), .tokens (true token set per equation), .bursts(M, seed)
"""

import hashlib
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import sympy as sp

import compile_artifact as CA
import discovery_learner as DL
import harness27 as H27
import practice_net as PN

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
CA.CACHE = Path(SP) / "t28_synth_cache" / str(os.getpid())          # per-process compile cache (cleared after loading)

MAX_REFINE, MAX_NSUB, MAX_BUILD_SECONDS = 4, 64, 60.0
D_DRAW = [0.168, 0.328, 0.504]     # calibrated on development seeds so that ACCEPTED systems are about 40/35/25% d = 2/3/4
HELD_OUT = {("abs", "hill"), ("hill", "tanh"), ("pair_hill", "recip"), ("prod_hill", "recip"), ("recip", "recip"),
            ("recip", "sin")}
OTHER_PIECES = ("exp", "tanh", "sin", "abs", "pow")


class Rejected(Exception):
    pass


def rng_of(seed):
    return np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def family(tok):
    """Piece family of a token for combination purposes ('mono' for products of variables)."""
    if tok[0] == "mono":
        return "mono"
    if tok[0] == "piece":
        return tok[1]
    return f"{tok[0]}_{tok[1]}"


def families_pairs(toks):
    fams = sorted(family(t) for t in toks if t[0] != "mono")
    return {tuple(sorted(p)) for p in itertools.combinations(fams, 2)}


def has_held_out(toks):
    return bool(families_pairs(toks) & HELD_OUT)


# ---------------------------------------------------------------------------------------------------------------------
# drawing a system
# ---------------------------------------------------------------------------------------------------------------------

def _unit(d, j):
    return tuple(int(k == j) for k in range(d))


def _mult(rng, d, i):
    """Multiplier of a piece: none (0.5), the equation's own variable (0.25), another variable (0.25)."""
    u = rng.random()
    if u < 0.5:
        return tuple([0] * d)
    if u < 0.75 or d == 1:
        return _unit(d, i)
    return _unit(d, int(rng.choice([k for k in range(d) if k != i])))


def _term(rng, d, i, ic, kind):
    """One term of equation i: (token, constants dict). Constants are physical; signs follow production (+) and
    consumption of the equation's own variable (-)."""
    j = int(rng.integers(d))
    if kind == "mono":
        deg = int(rng.choice([0, 1, 2], p=[0.25, 0.45, 0.30]))
        e = [0] * d
        for _ in range(deg):
            e[int(rng.integers(d))] += 1
        e = tuple(e)
        sign = -1.0 if e[i] > 0 else 1.0
        return ("mono", e), {"c": sign * float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))}
    if kind == "hill":
        m = _mult(rng, d, i)
        n = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.15]))
        K = float((np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j]) ** n)
        sign = -1.0 if m[i] > 0 else 1.0
        return ("piece", "hill", j, m), {"c": sign * float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))), "K": K, "n": float(n)}
    if kind == "recip":
        m = _mult(rng, d, i)
        K = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j])
        sign = -1.0 if m[i] > 0 else 1.0
        return ("piece", "recip", j, m), {"c": sign * float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))) * ic[j], "K": K}
    if kind in OTHER_PIECES:
        m = _mult(rng, d, i)
        sign = -1.0 if m[i] > 0 else 1.0
        c = sign * float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))
        if kind == "exp":
            return ("piece", "exp", j, m), {"c": c, "a": float(rng.uniform(0.3, 2.0) / ic[j])}
        if kind == "tanh":
            return ("piece", "tanh", j, m), {"c": c, "a": float(rng.uniform(1.0, 4.0) / ic[j]), "x0": float(ic[j] * rng.uniform(0.7, 1.3))}
        if kind == "sin":
            return ("piece", "sin", j, m), {"c": c, "w": float(rng.uniform(0.5, 2.0) / ic[j]), "ph": float(rng.uniform(0, 2 * np.pi))}
        if kind == "abs":
            return ("piece", "abs", j, m), {"c": c, "x0": float(ic[j] * rng.uniform(0.8, 1.2))}
        return ("piece", "pow", j, m), {"c": abs(c), "p": float(rng.choice([0.5, 1.5, -0.5]))}
    if kind in ("pair_recip", "pair_hill"):
        if d < 2:
            return None
        j, k = sorted(int(v) for v in rng.choice(d, 2, replace=False))
        m = _mult(rng, d, i)
        w = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j] / ic[k])
        sign = -1.0 if m[i] > 0 else 1.0
        c = sign * float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
        if kind == "pair_recip":
            return ("pair", "recip", (j, k), m), {"c": c * ic[j], "w": w, "K": float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j])}
        n = int(rng.choice([1, 2]))
        return ("pair", "hill", (j, k), m), {"c": c, "w": w, "n": float(n), "K": float((np.exp(rng.uniform(np.log(0.5), np.log(4.0))) * ic[j]) ** n)}
    if kind in ("prod_recip", "prod_hill"):
        if d < 2:
            return None
        j, k = sorted(int(v) for v in rng.choice(d, 2, replace=False))
        sign = -1.0 if i in (j, k) else 1.0
        c = sign * float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
        if kind == "prod_recip":
            return ("prod", "recip", (j, k)), {"c": c * ic[j] * ic[k], "K1": float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j]),
                                               "K2": float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[k])}
        n1, n2 = (int(v) for v in rng.choice([1, 2], 2))
        return ("prod", "hill", (j, k)), {"c": c, "n1": float(n1), "n2": float(n2),
                                          "K1": float((np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[j]) ** n1),
                                          "K2": float((np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * ic[k]) ** n2)}
    raise ValueError(kind)


KINDS = ["mono", "hill", "recip", "other", "pair", "prod"]
KIND_P = [0.35, 0.35, 0.10, 0.10, 0.05, 0.05]


def _equation(rng, d, i, ic, extra=None, forbid_held_out=True):
    """Token -> constants for equation i: a degradation term (0.8), then 1-3 further terms; `extra` forces terms."""
    terms = {}
    if rng.random() < 0.8:
        terms[("mono", _unit(d, i))] = {"c": -float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))}
    for kind in extra or []:
        for _ in range(50):
            got = _term(rng, d, i, ic, kind)
            if got is not None and got[0] not in terms:
                terms[got[0]] = got[1]
                break
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(50):
            kind = str(rng.choice(KINDS, p=KIND_P))
            if kind == "other":
                kind = str(rng.choice(OTHER_PIECES))
            elif kind == "pair":
                kind = str(rng.choice(["pair_recip", "pair_hill"]))
            elif kind == "prod":
                kind = str(rng.choice(["prod_recip", "prod_hill"]))
            got = _term(rng, d, i, ic, kind)
            if got is None or got[0] in terms:
                continue
            if forbid_held_out and has_held_out(list(terms) + [got[0]]):
                continue
            terms[got[0]] = got[1]
            break
    return terms


def draw(seed, mode="train"):
    """A system description: {seed, d, ic, eqs: [ {token: constants} ]}. mode: "train" (no held-out combination in any
    equation) or "test2" (at least one equation built around a held-out combination)."""
    rng = rng_of(seed)
    d = int(rng.choice([2, 3, 4], p=D_DRAW))
    ic = np.exp(rng.uniform(np.log(0.3), np.log(3.0), d))
    eqs = []
    forced = int(rng.integers(d)) if mode == "test2" else None
    pair = sorted(HELD_OUT)[int(rng.integers(len(HELD_OUT)))] if mode == "test2" else None
    for i in range(d):
        if i == forced:                                        # both families of a held-out pair in this equation
            eqs.append(_equation(rng, d, i, ic, extra=list(pair), forbid_held_out=False))
        else:
            eqs.append(_equation(rng, d, i, ic, forbid_held_out=True))
    return {"seed": seed, "d": d, "ic": ic.tolist(), "eqs": eqs, "mode": mode}


# ---------------------------------------------------------------------------------------------------------------------
# symbolic right-hand side and true learner terms
# ---------------------------------------------------------------------------------------------------------------------

def _sym_term(tok, cst, X):
    d = len(X)
    if tok[0] == "mono":
        return cst["c"] * sp.Mul(*[X[k] ** e for k, e in enumerate(tok[1])])
    if tok[0] in ("piece", "pair"):
        m = sp.Mul(*[X[k] ** e for k, e in enumerate(tok[3])])
    fam = tok[1]
    if tok[0] == "piece":
        x = X[tok[2]]
        if fam == "hill":
            f = x ** cst["n"] / (cst["K"] + x ** cst["n"])
        elif fam == "recip":
            f = 1 / (x + cst["K"])
        elif fam == "exp":
            f = sp.exp(-cst["a"] * x)
        elif fam == "tanh":
            f = sp.tanh(cst["a"] * (x - cst["x0"]))
        elif fam == "sin":
            f = sp.sin(cst["w"] * x + cst["ph"])
        elif fam == "abs":
            f = sp.Abs(x - cst["x0"])
        else:
            f = x ** cst["p"]
        return cst["c"] * m * f
    if tok[0] == "pair":
        j, k = tok[2]
        u = X[j] + cst["w"] * X[k]
        f = 1 / (cst["K"] + u) if fam == "recip" else u ** cst["n"] / (cst["K"] + u ** cst["n"])
        return cst["c"] * m * f
    j, k = tok[2]
    if fam == "recip":
        return cst["c"] / ((X[j] + cst["K1"]) * (X[k] + cst["K2"]))
    return cst["c"] * X[j] ** cst["n1"] / (cst["K1"] + X[j] ** cst["n1"]) * X[k] ** cst["n2"] / (cst["K2"] + X[k] ** cst["n2"])


def exprs_of(desc):
    d = desc["d"]
    X = sp.symbols(" ".join(f"x{i}" for i in range(d)))
    X = X if isinstance(X, tuple) else (X,)
    return [sp.Add(*[_sym_term(t, c, X) for t, c in eq.items()]) for eq in desc["eqs"]]


def learner_term(tok, cst, mu, sig):
    """The true term as a learner Term in the learner's (uncentred) coordinates z = x / sig, with its constants."""
    sig = [float(s) for s in sig]
    if tok[0] == "mono":
        return DL.Term(tok[1], (), 1)
    fam = tok[1]
    if tok[0] == "piece":
        j = tok[2]
        s = sig[j]
        th = {"hill": ("hill", (cst.get("K", 0.0), cst.get("n", 1.0))), "recip": ("crecip", (1.0, cst.get("K", 0.0) / s)),
              "exp": ("cexp", (-cst.get("a", 0.0) * s, 0.0)), "tanh": ("ctanh", (cst.get("a", 0.0) * s, -cst.get("a", 0.0) * cst.get("x0", 0.0))),
              "sin": ("csin", (cst.get("w", 0.0) * s, cst.get("ph", 0.0))), "abs": ("cabs", (1.0, -cst.get("x0", 0.0) / s)),
              "pow": ("pow", (cst.get("p", 1.0),))}[fam]
        return DL.Term(tok[3], (DL.Factor(th[0], j, tuple(float(v) for v in th[1]), ()),), 1)
    if tok[0] == "pair":
        j, k = tok[2]
        w2 = cst["w"] * sig[k] / sig[j]
        if fam == "recip":
            fac = DL.Factor("crecip", (j, k), (w2, 1.0, cst["K"] / sig[j]), ())
        else:
            fac = DL.Factor("chill", (j, k), (w2, sig[j] / cst["K"] ** (1.0 / cst["n"]), 0.0, cst["n"]), ("zsnap",))
        return DL.Term(tok[3], (fac,), 1)
    j, k = tok[2]
    if fam == "recip":
        facs = (DL.Factor("crecip", j, (1.0, cst["K1"] / sig[j]), ()), DL.Factor("crecip", k, (1.0, cst["K2"] / sig[k]), ()))
    else:
        facs = (DL.Factor("hill", j, (cst["K1"], cst["n1"]), ()), DL.Factor("hill", k, (cst["K2"], cst["n2"]), ()))
    return DL.Term(tuple([0] * len(sig)), facs, 1)


# ---------------------------------------------------------------------------------------------------------------------
# building and checking
# ---------------------------------------------------------------------------------------------------------------------

def settle_time(f, x0):
    """Trial 23 transient rule (biomodels_convert.transient_data): integrate from the initial state up to t = 1e4; the
    settle time is the first time the distance to the final state falls below 1% of its starting value. Rejects
    systems that do not move or do not settle within the horizon."""
    import biomodels_convert as BC
    t = np.concatenate([[0.0], np.logspace(-4, 4, 4000)])
    with np.errstate(all="ignore"):
        traj = BC.integrate(lambda x: f(x[None, :])[0], x0, t, seconds=5)
    if traj is None:
        raise Rejected("simulation: settle-time integration failed")
    dist = np.linalg.norm(traj - traj[-1], axis=1)
    if dist[0] < 1e-3 * (np.linalg.norm(x0) + 1e-6):
        raise Rejected("transient: no motion")
    below = np.nonzero(dist < 0.01 * dist[0])[0]
    if len(below) == 0 or t[below[0]] <= 0:
        raise Rejected("transient: does not settle within the horizon")
    return float(t[below[0]])


class SynthSystem(H27.TrueSystem27):
    """A synthetic ground truth with the same interface as harness27.TrueSystem27 (compiled right-hand side)."""

    def __init__(self, desc):
        self.name = desc["seed"]
        exprs = exprs_of(desc)
        CA.CACHE.mkdir(parents=True, exist_ok=True)
        self._f, self._roll = CA.build_true(exprs)
        for p in CA.CACHE.glob("true_*"):                     # loaded: remove the compiled files (small footprint)
            try:
                p.unlink()
            except OSError:
                pass
        self.kind = "transient"
        self.ic = np.array(desc["ic"], dtype=float)
        self.positive = True
        self.d = len(self.ic)
        self.T = settle_time(self.f, self.ic)                 # the Trial 23 transient rule, as for real systems
        self.dt = self.T / H27.STEPS


class SynthReference(H27.Reference27):
    """harness27.Reference27 (transient rules) for a synthetic system, with the cheap-simulation cutoff applied while
    building: more than MAX_REFINE interval halvings, or more than MAX_NSUB reference substeps, rejects at once."""

    def __init__(self, system, seed=0):
        self.sys = s = system
        self.rng = rng = np.random.default_rng([2700, seed, sum(map(ord, s.name))])
        self.rng_held = np.random.default_rng([2701, seed, sum(map(ord, s.name))])
        self._transient(s, rng)

    def _refine_dt(self, s, states, scale):
        self.refine = 0
        nsub = H27.choose_nsub(s, states, scale, max_nsub=4 * H27.MAX_REF_NSUB)
        while nsub > H27.MAX_REF_NSUB:
            if self.refine >= MAX_REFINE:
                raise Rejected(f"cost: more than {MAX_REFINE} interval halvings")
            s.dt /= 2
            self.refine += 1
            nsub = H27.choose_nsub(s, states, scale, max_nsub=4 * H27.MAX_REF_NSUB)
        return nsub

    def _ref_nsub(self, s, sample, states):
        cap = MAX_NSUB // H27.REF_MARGIN
        n = max(H27.choose_nsub(s, sample, self.scale, max_nsub=2 * cap),
                H27.choose_nsub(s, states[:: max(1, len(states) // 2000)], self.scale, max_nsub=2 * cap))
        if n > cap:
            raise Rejected(f"cost: more than {MAX_NSUB} reference substeps")
        return H27.REF_MARGIN * n


class Synthetic:
    def __init__(self, desc, ref, build_seconds):
        self.desc, self.ref, self.build_seconds = desc, ref, build_seconds
        self.d = desc["d"]
        self.tokens = [tuple(sorted(eq, key=str)) for eq in desc["eqs"]]

    def bursts(self, M, seed=1):
        return np.asarray(H27.bursts_from(self.ref, seed, M))


def build(desc, Ms=(8,)):
    """Compile, build the reference under the Trial 27 transient rules, and apply the logged acceptance checks."""
    t0 = time.time()
    try:
        system = SynthSystem(desc)
        ref = SynthReference(system)
    except RuntimeError as ex:
        raise Rejected(f"simulation: {ex}")
    secs = time.time() - t0
    if ref.refine > MAX_REFINE or ref.nsub > MAX_NSUB or secs > MAX_BUILD_SECONDS:
        raise Rejected(f"cost: refine {ref.refine}, nsub {ref.nsub}, {secs:.0f}s")
    allx = np.concatenate([ref.learn_trajs.reshape(-1, desc["d"]), ref.held_trajs.reshape(-1, desc["d"])])
    if not np.all(np.isfinite(allx)) or allx.min() <= 0:
        raise Rejected("not positive and finite")
    if allx.max() > 100 * max(desc["ic"]):
        raise Rejected("unbounded")
    span = ref.learn_trajs.max(1) - ref.learn_trajs.min(1)                 # per recording and variable
    if np.any(np.median(span, 0) <= 1e-3 * ref.scale):
        raise Rejected("a variable does not move")
    syn = Synthetic(desc, ref, secs)
    for M in Ms:
        view = PN.learner_view(syn.bursts(M), seed=1)
        if view["centred"]:
            raise Rejected(f"centred at M={M}")
        if M == 8:
            Xc = view["Zc"] * view["sig"] + view["mu"]
            for eq in desc["eqs"]:
                for tok, cst in eq.items():
                    term = learner_term(tok, cst, view["mu"], view["sig"])
                    if term.factors and DL.degenerate(term, view["Zc"], Xc, strict=False):
                        raise Rejected(f"degenerate true piece {tok}")
    return syn


def draw_accepted(prefix, k, mode="train", tries=50):
    """The k-th accepted system of a seed series: seeds prefix-k, prefix-k-r1, prefix-k-r2, ... (replacement seeds
    in order when a draw is rejected). Returns (Synthetic, rejection reasons)."""
    reasons = []
    for r in range(tries):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw(seed, mode)
        try:
            return build(desc), reasons
        except Rejected as ex:
            reasons.append(str(ex))
    return None, reasons
