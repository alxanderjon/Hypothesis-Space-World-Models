"""Stage 1 of the general learner (research log, 6 November): three domain generators, fixed before any task is drawn.

  mechanics  (x, v): x' = v, v' = -k x minus 0-2 of {a x^3, a sin x, c v, c v^3, c v|v|, c x^2 v}
  circuits   one of five forms, equally likely: series RLC in charge form (q, i); a parallel circuit (v, i) with a
             linear, Van der Pol type or diode element; FitzHugh-Nagumo (v, w)
  ecology    (x, y): logistic prey with linear or Holling type II predation, or two-species competition

Constants are drawn in the physical ranges below (log-uniform unless noted). Recordings follow the existing transient
rules (synthetic28.SynthReference: 64 learning recordings from starting states around the initial state, cheap-reference
cutoffs) with a window from the system's own timescale (synthetic28.own_timescale, the Trial 29 rule) for every system,
since many mechanical and electrical systems do not settle. Acceptance: finite; positive (ecology only); bounded (within
100 times the largest initial magnitude); every variable moves; not centred by the learner's scaling at M = 8; every
true non-polynomial term visibly nonlinear on the learner's M = 8 data (relative residual against cubic polynomials at
least ACCEPT_COLLAPSE) and not singular there; every true term in the L1 vocabulary.

The true structure of each equation is the set of canonical L1 keys of its terms in the learner's coordinates
(x_j = s_j * w_j, s_j the learner's scale).
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp

import compile_artifact as CA
import harness27 as H27
import language_g1 as LG
import practice_net as PN
import synthetic28 as SY

DOMAINS = ("mechanics", "circuits", "ecology")
ACCEPT_COLLAPSE = 1e-6        # a true non-polynomial term must differ from every cubic polynomial by this much (relative)
VOCAB_PATH = Path(__file__).resolve().parents[2] / "outputs/accumulating_learner/g1/vocab_d2.json"
W = LG.variables(2)                                            # learner coordinates (by position), two variables
S = (sp.Symbol("s1", positive=True), sp.Symbol("s2", positive=True))
W3 = LG.variables(3)                                           # ... and up to three (Stage 2a)
S3 = tuple(sp.Symbol(f"s{j + 1}", positive=True) for j in range(3))


class Rejected(Exception):
    pass


def rng_of(seed):
    return np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def pm(rng, lo, hi):
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi))


# ---------------------------------------------------------------------------------------------------------------------
# the generators: right-hand sides as sympy expressions in symbols y0, y1 with named constants
# ---------------------------------------------------------------------------------------------------------------------

Y = sp.symbols("y0 y1", real=True)
Y3 = sp.symbols("y0 y1 y2", real=True)
MECH_EXTRAS = ("cubic", "sine", "damping", "cubic_damping", "drag", "vdp_damping")


def draw_mechanics(rng):
    x, v = Y
    p = {"k": logu(rng, 0.5, 5.0)}
    n = int(rng.integers(0, 3))
    extras = sorted(rng.choice(len(MECH_EXTRAS), n, replace=False).tolist())
    names = [MECH_EXTRAS[e] for e in extras]
    acc = -sp.Symbol("k") * x
    for e in names:
        if e == "cubic":
            p["a3"] = logu(rng, 0.1, 2.0); acc -= sp.Symbol("a3") * x ** 3
        elif e == "sine":
            p["a_s"] = logu(rng, 0.5, 5.0); acc -= sp.Symbol("a_s") * sp.sin(x)
        elif e == "damping":
            p["c1"] = logu(rng, 0.05, 1.0); acc -= sp.Symbol("c1") * v
        elif e == "cubic_damping":
            p["c3"] = logu(rng, 0.05, 1.0); acc -= sp.Symbol("c3") * v ** 3
        elif e == "drag":
            p["cd"] = logu(rng, 0.05, 1.0); acc -= sp.Symbol("cd") * v * sp.sqrt(v ** 2)
        else:
            p["cv"] = logu(rng, 0.05, 1.0); acc -= sp.Symbol("cv") * x ** 2 * v
    ic = [pm(rng, 0.5, 2.5), pm(rng, 0.5, 2.5) * math.sqrt(p["k"])]
    return "oscillator+" + "+".join(names) if names else "oscillator", [v, acc], p, ic, False


CIRCUIT_FORMS = ("rlc_series", "parallel_linear", "parallel_vdp", "parallel_diode", "fitzhugh_nagumo")


def draw_circuits(rng):
    form = CIRCUIT_FORMS[int(rng.integers(0, len(CIRCUIT_FORMS)))]
    a, b = Y
    if form == "rlc_series":                                     # (q, i)
        p = {"w2": logu(rng, 0.5, 5.0), "r": logu(rng, 0.05, 1.0)}
        rhs = [b, -sp.Symbol("r") * b - sp.Symbol("w2") * a]
        ic = [pm(rng, 0.5, 2.5), pm(rng, 0.5, 2.5) * math.sqrt(p["w2"])]
    elif form == "fitzhugh_nagumo":                              # (v, w)
        p = {"I": float(rng.uniform(0.0, 1.0)), "eps": logu(rng, 0.05, 0.5), "a": float(rng.uniform(0.5, 1.0)),
             "b": float(rng.uniform(0.5, 1.0))}
        rhs = [a - a ** 3 / 3 - b + sp.Symbol("I"), sp.Symbol("eps") * (a + sp.Symbol("a") - sp.Symbol("b") * b)]
        ic = [pm(rng, 0.3, 2.0), pm(rng, 0.3, 1.5)]
    else:                                                        # (v, i)
        p = {"C": logu(rng, 0.2, 2.0), "L": logu(rng, 0.2, 2.0), "R": logu(rng, 0.05, 1.0)}
        if form == "parallel_linear":
            p["g1"] = logu(rng, 0.05, 1.0); g = sp.Symbol("g1") * a
        elif form == "parallel_vdp":
            p["g1"] = logu(rng, 0.1, 1.0); p["g3"] = logu(rng, 0.05, 0.5)
            g = -sp.Symbol("g1") * a + sp.Symbol("g3") * a ** 3
        else:
            p["Is"] = logu(rng, 0.01, 0.5); p["Vt"] = logu(rng, 0.5, 3.0)
            g = sp.Symbol("Is") * (sp.exp(a / sp.Symbol("Vt")) - 1)
        rhs = [(b - g) / sp.Symbol("C"), (-a - sp.Symbol("R") * b) / sp.Symbol("L")]
        ic = [pm(rng, 0.5, 2.5), pm(rng, 0.5, 2.5)]
    return form, rhs, p, ic, False


ECOLOGY_FORMS = ("predator_prey_linear", "predator_prey_holling", "competition")


def draw_ecology(rng):
    form = ECOLOGY_FORMS[int(rng.integers(0, len(ECOLOGY_FORMS)))]
    x, y = Y
    if form == "competition":
        p = {"r1": logu(rng, 0.5, 2.0), "r2": logu(rng, 0.5, 2.0), "a11": logu(rng, 0.1, 1.0), "a22": logu(rng, 0.1, 1.0),
             "a12": logu(rng, 0.05, 1.0), "a21": logu(rng, 0.05, 1.0)}
        s = {k: sp.Symbol(k) for k in p}
        rhs = [s["r1"] * x - s["a11"] * x ** 2 - s["a12"] * x * y, s["r2"] * y - s["a22"] * y ** 2 - s["a21"] * x * y]
        ic = [float(rng.uniform(0.2, 2.0)) * p["r1"] / p["a11"], float(rng.uniform(0.2, 2.0)) * p["r2"] / p["a22"]]
    else:
        p = {"r": logu(rng, 0.5, 2.0), "K": logu(rng, 2.0, 10.0), "a": logu(rng, 0.2, 2.0), "e": logu(rng, 0.2, 1.0),
             "m": logu(rng, 0.1, 0.5)}
        s = {k: sp.Symbol(k) for k in p}
        f = s["a"] * x
        if form == "predator_prey_holling":
            p["h"] = logu(rng, 0.2, 2.0); f = s["a"] * x / (1 + sp.Symbol("h") * x)
        rhs = [s["r"] * x - s["r"] / s["K"] * x ** 2 - f * y, s["e"] * f * y - s["m"] * y]
        ic = [float(rng.uniform(0.2, 1.0)) * p["K"], float(rng.uniform(0.2, 2.0))]
    return form, rhs, p, ic, True


# ---------------------------------------------------------------------------------------------------------------------
# Stage 2a (research log, 7 November): chemistry (two or three species, positive) and chaotic flows (three variables:
# jerk and random sparse quadratic flows only, so that the named chaotic systems of dysts stay unfamiliar)
# ---------------------------------------------------------------------------------------------------------------------

CHEMISTRY_FORMS = ("mass_action", "michaelis_menten", "brusselator", "selkov", "goodwin", "oregonator")


def _mass_action(rng):
    """3 species, 3-5 distinct reactions from the six templates; every species takes part. Mass-action rates."""
    x = Y3
    templates = []
    for i in range(3):
        for j in range(3):
            if i != j:
                templates.append(("conv", i, j))                 # X_i -> X_j
                templates.append(("auto", i, j))                 # X_i + X_j -> 2 X_j
        templates.append(("in", i, None))                        # 0 -> X_i
        templates.append(("out", i, None))                       # X_i -> 0
        for j in range(3):
            if j != i:
                templates.append(("dimer", i, j))                # 2 X_i -> X_j
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for k in range(3):
            if k not in (i, j):
                templates.append(("bind", (i, j), k))            # X_i + X_j -> X_k
    for _ in range(200):
        n = int(rng.integers(3, 6))
        pick = [templates[int(a)] for a in rng.choice(len(templates), n, replace=False)]
        used = set()
        for kind, a, b in pick:
            used |= set(a) if isinstance(a, tuple) else {a}
            if b is not None:
                used.add(b)
        if used == {0, 1, 2}:
            break
    rhs, p = [sp.Integer(0)] * 3, {}
    for r, (kind, a, b) in enumerate(pick):
        k = sp.Symbol(f"k{r}")
        p[f"k{r}"] = logu(rng, 0.1, 2.0)
        if kind == "conv":
            rate = k * x[a]; rhs[a] -= rate; rhs[b] += rate
        elif kind == "auto":
            rate = k * x[a] * x[b]; rhs[a] -= rate; rhs[b] += rate
        elif kind == "in":
            rhs[a] += k
        elif kind == "out":
            rhs[a] -= k * x[a]
        elif kind == "dimer":
            rate = k * x[a] ** 2; rhs[a] -= 2 * rate; rhs[b] += rate
        else:
            i, j = a
            rate = k * x[i] * x[j]; rhs[i] -= rate; rhs[j] -= rate; rhs[b] += rate
    ic = [float(rng.uniform(0.5, 2.0)) for _ in range(3)]
    return rhs, p, ic


def draw_chemistry(rng):
    form = CHEMISTRY_FORMS[int(rng.integers(0, len(CHEMISTRY_FORMS)))]
    x = Y3
    if form == "mass_action":
        rhs, p, ic = _mass_action(rng)
        return form, rhs, p, ic, True
    if form == "michaelis_menten":                              # inflow, Michaelis-Menten steps, outflow
        d = int(rng.integers(2, 4))
        p = {"k0": logu(rng, 0.2, 2.0), "kout": logu(rng, 0.1, 1.0)}
        rhs = [sp.Symbol("k0")] + [sp.Integer(0)] * (d - 1)
        for st in range(d - 1):
            p[f"V{st}"], p[f"K{st}"] = logu(rng, 0.5, 3.0), logu(rng, 0.3, 3.0)
            rate = sp.Symbol(f"V{st}") * x[st] / (sp.Symbol(f"K{st}") + x[st])
            rhs[st] -= rate; rhs[st + 1] += rate
        rhs[d - 1] -= sp.Symbol("kout") * x[d - 1]
        return f"{form}_{d}", rhs, p, [float(rng.uniform(0.5, 2.0)) for _ in range(d)], True
    if form == "brusselator":
        p = {"a": logu(rng, 0.5, 2.0), "b": logu(rng, 1.0, 4.0)}
        a, b = sp.Symbol("a"), sp.Symbol("b")
        rhs = [a - (b + 1) * x[0] + x[0] ** 2 * x[1], b * x[0] - x[0] ** 2 * x[1]]
        return form, rhs, p, [float(rng.uniform(0.3, 2.0)) for _ in range(2)], True
    if form == "selkov":
        p = {"a": logu(rng, 0.02, 0.2), "b": logu(rng, 0.3, 1.0)}
        a, b = sp.Symbol("a"), sp.Symbol("b")
        rhs = [-x[0] + a * x[1] + x[0] ** 2 * x[1], b - a * x[1] - x[0] ** 2 * x[1]]
        return form, rhs, p, [float(rng.uniform(0.3, 2.0)) for _ in range(2)], True
    if form == "goodwin":                                       # repression loop with a Hill-type exponent n = 1-3
        n = int(rng.integers(1, 4))
        p = {k: logu(rng, lo, hi) for k, lo, hi in (("a", 0.5, 3.0), ("K", 0.2, 2.0), ("b", 0.1, 1.0), ("c", 0.2, 2.0),
                                                     ("dd", 0.1, 1.0), ("e", 0.2, 2.0), ("f", 0.1, 1.0))}
        P = {k: sp.Symbol(k) for k in p}
        rhs = [P["a"] / (P["K"] + x[2] ** n) - P["b"] * x[0], P["c"] * x[0] - P["dd"] * x[1], P["e"] * x[1] - P["f"] * x[2]]
        return f"{form}_n{n}", rhs, p, [float(rng.uniform(0.3, 2.0)) for _ in range(3)], True
    p = {"s": logu(rng, 0.5, 3.0), "q": logu(rng, 0.01, 0.1), "f": float(rng.uniform(0.5, 2.0)), "w": logu(rng, 0.1, 1.0)}
    P = {k: sp.Symbol(k) for k in p}                            # Oregonator-like, polynomial
    rhs = [P["s"] * (x[1] - x[0] * x[1] + x[0] - P["q"] * x[0] ** 2), (-x[1] - x[0] * x[1] + P["f"] * x[2]) / P["s"],
           P["w"] * (x[0] - x[2])]
    return form, rhs, p, [float(rng.uniform(0.3, 2.0)) for _ in range(3)], True


CHAOTIC_FORMS = ("jerk", "sparse_quadratic")
JERK_NONLINEAR = ("x2", "xy", "y2", "x3", "x2y")


def draw_chaotic(rng):
    form = CHAOTIC_FORMS[int(rng.integers(0, len(CHAOTIC_FORMS)))]
    x, y, z = Y3
    if form == "jerk":                                          # x' = y, y' = z, z' = linear + one nonlinearity (+ c)
        nl = JERK_NONLINEAR[int(rng.integers(0, len(JERK_NONLINEAR)))]
        p = {"a": logu(rng, 0.2, 2.0), "b": logu(rng, 0.2, 2.0), "c": pm(rng, 0.2, 2.0), "s": pm(rng, 0.5, 2.0)}
        N = {"x2": x ** 2, "xy": x * y, "y2": y ** 2, "x3": x ** 3, "x2y": x ** 2 * y}[nl]
        zdot = -sp.Symbol("a") * z - sp.Symbol("b") * y - sp.Symbol("c") * x + sp.Symbol("s") * N
        if rng.random() < 0.5:
            p["k"] = float(rng.uniform(-1.0, 1.0)); zdot += sp.Symbol("k")
        return f"jerk_{nl}", [y, z, zdot], p, [pm(rng, 0.3, 1.5) for _ in range(3)], False
    mons = [x, y, z, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z]  # random sparse quadratic: 2-3 terms per equation
    for _ in range(200):
        picks = [sorted(rng.choice(len(mons), int(rng.integers(2, 4)), replace=False).tolist()) for _ in range(3)]
        vars_ = set()
        for eq in picks:
            for m in eq:
                vars_ |= {str(v) for v in mons[m].free_symbols}
        if len(vars_) == 3 and any(m >= 3 for eq in picks for m in eq):
            break
    p, rhs = {}, []
    for i, eq in enumerate(picks):
        e = 0
        for t, m in enumerate(eq):
            p[f"a{i}_{t}"] = pm(rng, 0.2, 2.0)
            e = e + sp.Symbol(f"a{i}_{t}") * mons[m]
        rhs.append(e)
    return form, rhs, p, [pm(rng, 0.3, 1.5) for _ in range(3)], False


GENERATORS = {"mechanics": draw_mechanics, "circuits": draw_circuits, "ecology": draw_ecology, "chemistry": draw_chemistry, "chaotic": draw_chaotic}


def draw(seed, domain):
    form, rhs, p, ic, positive = GENERATORS[domain](rng_of(seed))
    return {"seed": seed, "domain": domain, "form": form, "d": len(rhs), "rhs": [sp.srepr(e) for e in rhs], "params": p,
            "ic": ic, "positive": positive}


# ---------------------------------------------------------------------------------------------------------------------
# true structure in L1
# ---------------------------------------------------------------------------------------------------------------------

def _sym_rhs(desc):
    return [eval(e, sp.__dict__) for e in desc["rhs"]]


def numeric_rhs(desc):
    """Right-hand sides with the drawn constants, in plain symbols x0, x1, .. (compile_artifact.build_true)."""
    d = desc["d"]
    xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
    sub = {sp.Symbol(k): v for k, v in desc["params"].items()}
    return [e.subs(sub).subs({Y3[j]: xs[j] for j in range(d)}) for e in _sym_rhs(desc)]


def _terms(e):
    return sp.Add.make_args(sp.expand(e, power_exp=False, log=False))


def true_structure(desc):
    """Per equation: the sorted canonical L1 keys of its terms in the learner's coordinates, and which are
    non-polynomial (as expressions in the learner's coordinates with symbolic constants)."""
    d = desc["d"]
    Wd, Sd = LG.variables(d), (S if d == 2 else S3[:d])
    out = []
    for e in _sym_rhs(desc):
        e = e.subs({Y3[j]: Sd[j] * Wd[j] for j in range(d)})
        keys, nonpoly = set(), []
        for t in _terms(e):
            if not (t.free_symbols & set(Wd)):
                keys.add(sp.srepr(sp.Integer(1)))
                continue
            r = LG.canonical(t, Wd)
            if r is None:
                raise Rejected(f"a true term has no canonical form")
            keys.add(r[0])
            if not t.is_polynomial(*Wd):
                nonpoly.append(r[0])
        out.append({"keys": sorted(keys), "nonpoly": sorted(set(nonpoly))})
    return out


def swapped(key):
    """The canonical key of a term with the two variables exchanged."""
    e = LG.expr_of(key).xreplace({W[0]: W[1], W[1]: W[0]})
    if not (e.free_symbols & set(W)):
        return key
    r = LG.canonical(e, W)
    return r[0] if r else None


def relabelled(key, perm):
    """The canonical key of a term with variable x_{j+1} renamed x_{perm[j]+1} (up to three variables)."""
    e = LG.expr_of(key)
    used = e.free_symbols & set(W3)
    if not used:
        return key
    e = e.xreplace({W3[j]: W3[perm[j]] for j in range(len(perm))})
    r = LG.canonical(e, W3)
    return r[0] if r else None


def largest_lyapunov(syn, steps=4000, burn=500, eps=1e-8):
    """Benettin estimate of the largest Lyapunov exponent (per unit time) from the reference's compiled flow, reported
    only (chaos is not an acceptance rule)."""
    s, nsub = syn.ref.sys, syn.ref.nsub
    x = np.asarray(syn.desc["ic"], dtype=float)[None, :]
    with np.errstate(all="ignore"):
        x = s.rollout(x, nsub, burn)[:, -1]
        v = np.random.default_rng(0).standard_normal(x.shape)
        y = x + eps * v / np.linalg.norm(v)
        acc = 0.0
        for _ in range(steps):
            x, y = s.one_step(x, nsub), s.one_step(y, nsub)
            dist = float(np.linalg.norm(y - x))
            if not np.isfinite(dist) or dist == 0:
                return float("nan")
            acc += math.log(dist / eps)
            y = x + (y - x) * (eps / dist)
    return acc / (steps * s.dt)


def numeric_terms(desc, eq):
    """The physical terms (with drawn constants) of equation eq that are non-polynomial, as numpy functions of x."""
    xs = sp.symbols(" ".join(f"x{j}" for j in range(desc["d"])))
    e = numeric_rhs(desc)[eq]
    out = []
    for t in _terms(e):
        if t.free_symbols and not t.is_polynomial(*xs):
            f = sp.lambdify(xs, t, "numpy")
            f.own = sorted(j for j, x in enumerate(xs) if x in t.free_symbols)   # the term's own variables
            out.append(f)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# building and acceptance
# ---------------------------------------------------------------------------------------------------------------------

class G1System(H27.TrueSystem27):
    """A Stage 1 ground truth with harness27's interface (compiled right-hand side); window from its own timescale."""

    def __init__(self, desc):
        self.name = desc["seed"]
        CA.CACHE.mkdir(parents=True, exist_ok=True)
        self._f, self._roll = CA.build_true(numeric_rhs(desc))
        for p in CA.CACHE.glob("true_*"):                     # loaded: remove the compiled files (small footprint)
            try:
                p.unlink()
            except OSError:
                pass
        self.kind = "transient"
        self.ic = np.array(desc["ic"], dtype=float)
        self.positive = bool(desc["positive"])
        self.d = len(self.ic)
        self.T = SY.own_timescale(self.f, self.ic)
        self.dt = self.T / H27.STEPS


def _collapse(col, Xc, own=None):
    """Relative residual of a column against all monomials of degree <= 3 in the term's own variables (all variables
    when `own` is None). Stage 2a (research log, 7 November): own variables, because with three variables the 20 cubic
    monomials in all of them, on 30 rate points, mimic almost any smooth term (a spurious rejection)."""
    import itertools
    own = list(range(Xc.shape[1])) if own is None else list(own)
    mons = [np.prod([Xc[:, j] ** e[k] for k, j in enumerate(own)], axis=0) for e in
            [e for e in itertools.product(range(4), repeat=len(own)) if sum(e) <= 3]]
    A = np.stack(mons, 1)
    A = A / np.maximum(np.abs(A).max(0), 1e-300)
    coef, *_ = np.linalg.lstsq(A, col, rcond=None)
    return float(np.linalg.norm(col - A @ coef) / (np.linalg.norm(col) + 1e-300))


_VOCAB, _MERGED = {}, {}


def vocabulary(d=2):
    if d not in _VOCAB:
        path = VOCAB_PATH if d == 2 else VOCAB_PATH.parent / f"vocab_d{d}.json"
        data = json.loads(path.read_text())
        _VOCAB[d], _MERGED[d] = data["terms"], data.get("merged", {})
    return _VOCAB[d]


def representative(key, d=2):
    """A key's vocabulary representative: itself, or the family it was merged into (g1_families.py)."""
    vocabulary(d)
    seen = set()
    while key in _MERGED[d] and key not in seen:
        seen.add(key)
        key = _MERGED[d][key]
    return key


class G1:
    """An accepted system: description, reference, true structure, bursts."""

    def __init__(self, desc, ref, structure, build_seconds):
        self.desc, self.ref, self.structure, self.build_seconds = desc, ref, structure, build_seconds

    def bursts(self, M, seed=1):
        return np.asarray(H27.bursts_from(self.ref, seed, M))


def build(desc):
    t0 = time.time()
    try:
        structure = true_structure(desc)
    except Rejected:
        raise
    V = vocabulary(desc["d"])
    for eq in structure:
        eq["keys"] = sorted({representative(k, desc["d"]) for k in eq["keys"]})
        eq["nonpoly"] = sorted({representative(k, desc["d"]) for k in eq["nonpoly"]})
        if any(k not in V for k in eq["keys"]):
            raise Rejected("a true term outside the L1 vocabulary")
    try:
        with np.errstate(all="ignore"):
            system = G1System(desc)
            ref = SY.SynthReference(system)
    except (RuntimeError, SY.Rejected) as ex:
        raise Rejected(f"simulation or cost: {ex}")
    secs = time.time() - t0
    if ref.refine > SY.MAX_REFINE or ref.nsub > SY.MAX_NSUB or secs > SY.MAX_BUILD_SECONDS:
        raise Rejected(f"cost: refine {ref.refine}, nsub {ref.nsub}, {secs:.0f}s")
    allx = np.concatenate([ref.learn_trajs.reshape(-1, desc["d"]), ref.held_trajs.reshape(-1, desc["d"])])
    if not np.all(np.isfinite(allx)):
        raise Rejected("not finite")
    if desc["positive"] and allx.min() <= 0:
        raise Rejected("not positive")
    if np.abs(allx).max() > 100 * max(abs(v) for v in desc["ic"]):
        raise Rejected("unbounded")
    span = ref.learn_trajs.max(1) - ref.learn_trajs.min(1)
    if np.any(np.median(span, 0) <= 1e-3 * ref.scale):
        raise Rejected("a variable does not move")
    syn = G1(desc, ref, structure, secs)
    view = PN.learner_view(syn.bursts(8), seed=1)
    if view["centred"]:
        raise Rejected("centred at M=8")
    Xc = (view["Zc"] * view["sig"] + view["mu"]).numpy()
    for i in range(desc["d"]):
        for f in numeric_terms(desc, i):
            with np.errstate(all="ignore"):
                col = np.asarray(f(*[Xc[:, j] for j in range(desc["d"])]), dtype=float) * np.ones(len(Xc))
            if not np.all(np.isfinite(col)):
                raise Rejected("a true term is singular on the data")
            if _collapse(col, Xc, f.own) < ACCEPT_COLLAPSE:
                raise Rejected("a true term is not visibly nonlinear on the data")
    return syn


def draw_accepted(prefix, k, domain, tries=500):
    """The k-th accepted system of a seed series: seeds prefix-k, prefix-k-r1, ... (replacement seeds in order when a
    draw is rejected). Returns (G1, rejection reasons)."""
    reasons = []
    for r in range(tries):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw(seed, domain)
        try:
            return build(desc), reasons
        except Rejected as ex:
            reasons.append(str(ex))
    return None, reasons
