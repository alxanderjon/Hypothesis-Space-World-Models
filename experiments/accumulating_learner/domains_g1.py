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
W = LG.variables(2)                                            # learner coordinates (by position)
S = (sp.Symbol("s1", positive=True), sp.Symbol("s2", positive=True))


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


GENERATORS = {"mechanics": draw_mechanics, "circuits": draw_circuits, "ecology": draw_ecology}


def draw(seed, domain):
    form, rhs, p, ic, positive = GENERATORS[domain](rng_of(seed))
    return {"seed": seed, "domain": domain, "form": form, "d": 2, "rhs": [sp.srepr(e) for e in rhs], "params": p,
            "ic": ic, "positive": positive}


# ---------------------------------------------------------------------------------------------------------------------
# true structure in L1
# ---------------------------------------------------------------------------------------------------------------------

def _sym_rhs(desc):
    return [eval(e, sp.__dict__) for e in desc["rhs"]]


def numeric_rhs(desc):
    """Right-hand sides with the drawn constants, in plain symbols x0, x1 (compile_artifact.build_true)."""
    xs = sp.symbols("x0 x1")
    sub = {sp.Symbol(k): v for k, v in desc["params"].items()}
    return [e.subs(sub).subs({Y[0]: xs[0], Y[1]: xs[1]}) for e in _sym_rhs(desc)]


def _terms(e):
    return sp.Add.make_args(sp.expand(e, power_exp=False, log=False))


def true_structure(desc):
    """Per equation: the sorted canonical L1 keys of its terms in the learner's coordinates, and which are
    non-polynomial (as expressions in the learner's coordinates with symbolic constants)."""
    out = []
    for e in _sym_rhs(desc):
        e = e.subs({Y[0]: S[0] * W[0], Y[1]: S[1] * W[1]})
        keys, nonpoly = set(), []
        for t in _terms(e):
            if not (t.free_symbols & set(W)):
                keys.add(sp.srepr(sp.Integer(1)))
                continue
            r = LG.canonical(t, W)
            if r is None:
                raise Rejected(f"a true term has no canonical form")
            keys.add(r[0])
            if not t.is_polynomial(*W):
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


def numeric_terms(desc, eq):
    """The physical terms (with drawn constants) of equation eq that are non-polynomial, as numpy functions of x."""
    xs = sp.symbols("x0 x1")
    e = numeric_rhs(desc)[eq]
    out = []
    for t in _terms(e):
        if t.free_symbols and not t.is_polynomial(*xs):
            out.append(sp.lambdify(xs, t, "numpy"))
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


def _collapse(col, Xc):
    """Relative residual of a column against all monomials of degree <= 3 in the variables."""
    d = Xc.shape[1]
    mons = [np.prod([Xc[:, j] ** e[j] for j in range(d)], axis=0) for e in
            [(a, b) for a in range(4) for b in range(4) if a + b <= 3]]
    A = np.stack(mons, 1)
    A = A / np.maximum(np.abs(A).max(0), 1e-300)
    coef, *_ = np.linalg.lstsq(A, col, rcond=None)
    return float(np.linalg.norm(col - A @ coef) / (np.linalg.norm(col) + 1e-300))


_VOCAB = None


def vocabulary():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = json.loads(VOCAB_PATH.read_text())["terms"]
    return _VOCAB


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
    V = vocabulary()
    for eq in structure:
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
    allx = np.concatenate([ref.learn_trajs.reshape(-1, 2), ref.held_trajs.reshape(-1, 2)])
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
    for i in range(2):
        for f in numeric_terms(desc, i):
            with np.errstate(all="ignore"):
                col = np.asarray(f(Xc[:, 0], Xc[:, 1]), dtype=float) * np.ones(len(Xc))
            if not np.all(np.isfinite(col)):
                raise Rejected("a true term is singular on the data")
            if _collapse(col, Xc) < ACCEPT_COLLAPSE:
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
