"""Stage 1 of the general learner (research log, 6 November): the generic fitter for L1 structures.

v2-strong (fitter30.FitterV2S) with its stages unchanged, applied to L1 terms, with starting values for each internal
constant from generic rules on the data, by the constant's role in its term:
  shift        a constant added to a subexpression f (K + f): the zero crossing of K + f across f's observed range,
               extended by half a span on each side; where a zero crossing inside the data would be singular (in a
               denominator, a log or a fractional power), at log-spaced distances of 0.01-100 spans outside the range
               (v2's pole rule);
  multiplier   a constant multiplying a subexpression f: a log scale from 0.01 to 100 of 1/scale, both signs, where scale
               is f's largest magnitude on the data (inside a function or power), or that magnitude relative to the other
               parts of the sum the product sits in;
  exponent     a fitted exponent: [-3, 3].
Stages: ordinary fit = grid sweep (coordinate sweeps over each term's grid placements, then Levenberg-Marquardt with
variable projection from the best and next-best placements), joint beam, algebraic start for a rational term (detected
automatically: constants only in a denominator that is affine in them); thorough fit = the sweep with more basins, the
algebraic start, the joint global search over the natural ranges, the fresh global search's best 4 pre-polished 200
iterations (v2-strong), each polished; then identifiability-gated snapping of fitted exponents to whole numbers, judged by
the held-back step-integral error. evaluate_structure() judges a structure as the search does (plausibly exact: the
step-integral error on training and held-back bursts within the equation's tolerance, and no degenerate term).
"""

import itertools
import math
import time
import zlib
from functools import lru_cache

import numpy as np
import sympy as sp
import torch

import fitter30 as F30
import language_g1 as LG
import practice_net as PN
import structure_search as SS

COLLAPSE_EXACT = 1e-10       # a fitted term this close to a cubic polynomial over the data is degenerate (exactly so)
D_DEFAULT = 3                  # terms compile for up to three variables; a system with fewer leaves the rest unused


# ---------------------------------------------------------------------------------------------------------------------
# term information
# ---------------------------------------------------------------------------------------------------------------------

def compile_expr(e, d, nconst):
    """f(W (n, d), theta (k,) or (B, k)) -> (n,) or (n, B) for expression e in x1..xd and K0..K{nconst-1}."""
    X = LG.variables(d)
    src = LG.torch_source(e, X)
    args = ", ".join([f"W{j}" for j in range(d)] + [f"K{c}" for c in range(nconst)])
    fn = eval(f"lambda {args}: {src}", {"torch": torch})

    def f(W, theta=None):
        cols = [W[:, j:j + 1] for j in range(d)]
        batched = theta is not None and torch.as_tensor(theta).dim() == 2
        if nconst == 0:
            out = fn(*cols)
        else:
            th = torch.as_tensor(theta, dtype=W.dtype)
            ks = [th[:, c].reshape(1, -1) for c in range(nconst)] if batched else [th[c] for c in range(nconst)]
            out = fn(*cols, *ks)
        out = torch.as_tensor(out, dtype=W.dtype)
        if batched:
            B = torch.as_tensor(theta).shape[0]
            if out.dim() < 2:
                out = out.reshape(-1, 1)
            return out.expand(W.shape[0], B)
        if out.dim() == 2:
            out = out[:, 0] if out.shape[1] == 1 else out.reshape(-1)
        return out.expand(W.shape[0]) if out.numel() == 1 else out
    return f


def _paths(e, target):
    """Paths (lists of (node, child index)) from e down to every occurrence of the symbol target."""
    out = []

    def walk(u, path):
        if u == target:
            out.append(path)
            return
        for j, a in enumerate(u.args):
            walk(a, path + [(u, j)])
    walk(e, [])
    return out


def _singular_context(path):
    """True if the node at the end of path sits (anywhere above) in a denominator, a log or a fractional power."""
    for node, j in path:
        if isinstance(node, sp.log):
            return True
        if node.is_Pow and j == 0:
            x = node.exp
            if not x.is_number or not x.is_Integer or x < 0:
                return True
    return False


class TermInfo:
    """Compiled form, constant roles, singular subexpressions and rational form of one L1 term."""

    def __init__(self, key, d=D_DEFAULT):
        self.key, self.d = key, d
        X = LG.variables(d)
        Xs = set(X)
        e = LG.expr_of(key)
        self.expr = e
        consts = sorted(e.free_symbols - Xs, key=lambda s: s.name)
        self.k = len(consts)
        assert [c.name for c in consts] == [f"K{c}" for c in range(self.k)], key
        self.fn = compile_expr(e, d, self.k)
        self.roles = []
        for c in consts:
            ps = _paths(e, c)
            role = {"kind": "generic"}
            if len(ps) == 1 and ps[0]:
                path = ps[0]
                parent, j = path[-1]
                if parent.is_Add:
                    rest = sp.Add(*[a for i, a in enumerate(parent.args) if i != j])
                    role = {"kind": "shift", "sub": compile_expr(rest, d, self.k), "singular": _singular_context(path),
                            "dep": bool(rest.free_symbols & set(consts))}
                elif parent.is_Mul:
                    rest = sp.Mul(*[a for i, a in enumerate(parent.args) if i != j])
                    role = {"kind": "mult", "sub": compile_expr(rest, d, self.k), "dep": bool(rest.free_symbols & set(consts))}
                    if len(path) >= 2 and path[-2][0].is_Add:
                        gp, jj = path[-2]
                        others = [a for i, a in enumerate(gp.args) if i != jj and (a.free_symbols & Xs)]
                        if others:
                            g = sp.Add(*others)
                            role.update(weight=compile_expr(g, d, self.k), dep=role["dep"] or bool(g.free_symbols & set(consts)))
                elif parent.is_Pow and j == 1:
                    role = {"kind": "exponent", "dep": False}
            self.roles.append(role)
        # subexpressions that must not reach zero on the data: denominators, log arguments, bases of non-integer powers
        sing = []
        for u in sp.preorder_traversal(e):
            if isinstance(u, sp.log):
                sing.append(u.args[0])
            elif u.is_Pow and (u.base.free_symbols & Xs):
                x = u.exp
                if not x.is_number or not x.is_Integer or x < 0:
                    sing.append(u.base)
        self.singular = [compile_expr(s, d, self.k) for s in sing]
        # rational form: constants only in a denominator affine in them
        self.rational = None
        if self.k:
            num, den = sp.fraction(sp.together(e))
            if not (num.free_symbols & set(consts)) and den.free_symbols & set(consts):
                try:
                    P = sp.Poly(sp.expand(den), *consts)
                    if P.total_degree() == 1:
                        d0 = P.coeff_monomial(1)
                        dj = [P.coeff_monomial(c) for c in consts]
                        if all(not (q.free_symbols & set(consts)) for q in [d0] + dj):
                            self.rational = (compile_expr(num, d, 0), compile_expr(d0, d, 0), [compile_expr(q, d, 0) for q in dj])
                except (sp.PolynomialError, sp.GeneratorsNeeded):
                    pass
        self.exponents = [a for a, r in enumerate(self.roles) if r["kind"] == "exponent"]


@lru_cache(maxsize=None)
def info(key, d=D_DEFAULT):
    return TermInfo(key, d)


def natural_thetas(inf, U, W):
    """Constant values (B, k) for unit points U (B, k) by each constant's role (module docstring), constants whose
    subexpression involves no other constant first."""
    B = U.shape[0]
    P = np.ones((B, inf.k))
    order = sorted(range(inf.k), key=lambda a: inf.roles[a].get("dep", False))
    logs = lambda u: 10.0 ** (-2.0 + 4.0 * u)
    for a in order:
        r = inf.roles[a]
        u = U[:, a]
        if r["kind"] == "exponent":
            P[:, a] = -3.0 + 6.0 * u
            continue
        if r["kind"] == "generic":
            P[:, a] = np.where(u >= 0.5, 1.0, -1.0) * logs(np.abs(2 * u - 1))
            continue
        with torch.no_grad(), np.errstate(all="ignore"):
            sub = r["sub"](W, torch.tensor(P)).numpy()             # (n, B), with the constants sampled so far
            wt = r["weight"](W, torch.tensor(P)).numpy() if "weight" in r else None
        for b in range(B):
            f = sub[:, b]
            f = f[np.isfinite(f)]
            if not len(f):
                P[b, a] = np.nan
                continue
            if r["kind"] == "shift":
                lo, hi = float(f.min()), float(f.max())
                s = max(hi - lo, 1e-9 * max(abs(lo), abs(hi), 1e-300))
                if r["singular"]:
                    dist = s * logs((2 * u[b]) % 1.0)
                    p = lo - dist if u[b] < 0.5 else hi + dist
                else:
                    p = lo - 0.5 * s + 2.0 * s * u[b]
                P[b, a] = -p
            else:
                typ = float(np.abs(f).max()) or 1.0
                scale = 1.0 / typ
                if wt is not None:
                    g = wt[:, b]
                    g = g[np.isfinite(g)]
                    if len(g) and float(np.abs(g).max()) > 0:
                        scale = float(np.abs(g).max()) / typ
                P[b, a] = (1.0 if u[b] >= 0.5 else -1.0) * scale * logs(abs(2 * u[b] - 1))
    return P


def unit_grid(k):
    if k == 1:
        return ((np.arange(32) + 0.5) / 32)[:, None]
    g = (np.arange(10) + 0.5) / 10
    return np.array(list(itertools.product(g, g)))


# ---------------------------------------------------------------------------------------------------------------------
# terms with constants
# ---------------------------------------------------------------------------------------------------------------------

class LTerm:
    """A term of a structure: key, constants, constants held fixed (snapped exponents), grid placement."""

    def __init__(self, key, theta=(), fixed=(), grid_index=None):
        self.key, self.theta, self.fixed, self.grid_index = key, tuple(float(v) for v in theta), frozenset(fixed), grid_index

    @property
    def k(self):
        return len(self.theta)

    def with_theta(self, theta, fixed=None):
        return LTerm(self.key, theta, self.fixed if fixed is None else fixed, self.grid_index)

    def __repr__(self):
        return f"LTerm({self.key!r}, {self.theta})"


def value(term, W, theta=None):
    inf = info(term.key)
    th = term.theta if theta is None else theta
    with torch.no_grad():
        return inf.fn(W, torch.tensor(th, dtype=W.dtype) if inf.k else None)


class Bank:
    """Every key's columns on the rate points at its grid placements: bank[key] = (columns (n, g), thetas (g, k))."""

    def __init__(self, keys, W):
        self.bank, self.W = {}, W
        for key in keys:
            self.add(key)

    def add(self, key):
        if key in self.bank:
            return self.bank[key]
        inf = info(key)
        W = self.W
        with torch.no_grad(), np.errstate(all="ignore"):
            if inf.k == 0:
                v = inf.fn(W)
                if torch.isfinite(v).all() and float(v.abs().max()) < 1e12 and float(v.norm()) > 0:
                    self.bank[key] = (v.reshape(-1, 1), np.zeros((1, 0)))
                return self.bank.get(key)
            th = natural_thetas(inf, unit_grid(inf.k), W)
            th = th[np.all(np.isfinite(th), 1)]
            if not len(th):
                return None
            C = inf.fn(W, torch.tensor(th))
        ok = torch.isfinite(C).all(0) & (C.abs().max(0).values < 1e12) & (C.norm(dim=0) > 0)
        if bool(ok.any()):
            self.bank[key] = (C[:, ok], th[ok.numpy()])
        return self.bank.get(key)


# ---------------------------------------------------------------------------------------------------------------------
# variable projection
# ---------------------------------------------------------------------------------------------------------------------

def vp_lm(terms, column, y, iters=10, tol_gain=1e-6, lam_max=1e4, stop_below=0.0):
    """Levenberg-Marquardt on the free constants of `terms`, linear coefficients projected out (Kaufman's Jacobian, finite
    differences), as structure_search._vp_lm. column(key, theta) gives a term's column. Returns (terms, residual, coefs)."""
    slots = [(m, a) for m, t in enumerate(terms) for a in range(t.k) if a not in t.fixed]
    th = [list(t.theta) for t in terms]
    cols = [column(t.key, th[m]) for m, t in enumerate(terms)]
    out = SS._project(torch.stack(cols, 1), y) if terms else None
    if out is None or not bool(torch.isfinite(out[0]).all()):
        return terms, float("inf"), None
    r, c, U = out
    rn = float(r.norm())
    live = sorted({m for m, _ in slots})
    lam = 1e-3
    for _ in range(iters if slots else 0):
        J = torch.empty(len(y), len(slots), dtype=torch.float64)
        for s_, (m, a) in enumerate(slots):
            v = th[m][a]
            h = 1e-7 * (1.0 + abs(v))
            thm = list(th[m]); thm[a] = v + h
            dc = (column(terms[m].key, thm) - cols[m]) / h
            J[:, s_] = -(dc - U @ (U.T @ dc)) * c[m]
        if not bool(torch.isfinite(J).all()):
            break
        H, g = J.T @ J, J.T @ r
        D = torch.diag(torch.diag(H) + 1e-12)
        gain, tries = None, 0
        while lam <= lam_max and tries < 4:
            tries += 1
            try:
                step = torch.linalg.solve(H + lam * D, -g)
            except RuntimeError:
                lam *= 10
                continue
            th2 = [list(x) for x in th]
            for s_, (m, a) in enumerate(slots):
                th2[m][a] += float(step[s_])
            cols2 = [column(terms[m].key, th2[m]) if m in live else cols[m] for m in range(len(terms))]
            o2 = SS._project(torch.stack(cols2, 1), y)
            if o2 is not None and bool(torch.isfinite(o2[0]).all()) and float(o2[0].norm()) < rn:
                gain = 1.0 - float(o2[0].norm()) / rn
                th, cols, (r, c, U), rn = th2, cols2, o2, float(o2[0].norm())
                lam = max(lam * 0.3, 1e-9)
                break
            lam *= 10
        if gain is None or gain < tol_gain or rn <= stop_below:
            break
    new = [t.with_theta(th[m]) for m, t in enumerate(terms)]
    return new, rn, c


def rate_column(W):
    def col(key, theta):
        return value(LTerm(key, theta), W)
    return col


def fast_refine(terms, W, y, iters=10, stop_below=0.0):
    return vp_lm(terms, rate_column(W), y, iters=iters, stop_below=stop_below)


class StepIntegrals:
    """structure_search.StepIntegrals for L1 terms (states at the quadrature nodes in the learner's coordinates)."""

    def __init__(self, Zb, mu, sig, nodes=8):
        base = SS.StepIntegrals(Zb, mu, sig, nodes)
        self.W = base.X / sig                                   # x / sig: the learner's coordinates when uncentred
        self.Wt, self.dZ, self.nodes = base.W, base.dZ, nodes

    def col(self, key, theta):
        v = value(LTerm(key, theta), self.W)
        return (v * self.Wt).reshape(-1, self.nodes).sum(1)

    def design(self, terms):
        return torch.stack([self.col(t.key, t.theta) for t in terms], 1)


def refine_integral(terms, si, i, iters=12):
    y = si.dZ[:, i]
    yn = float(y.norm()) + 1e-300
    new, rn, c = vp_lm(terms, si.col, y, iters=iters, lam_max=1e6, stop_below=1e-15 * yn)
    return new, rn / yn, c


def heldback_error(terms, coefs, si, i):
    y = si.dZ[:, i]
    with torch.no_grad():
        pred = si.design(terms) @ coefs if terms else torch.zeros_like(y)
    e = float((y - pred).norm()) / (float(y.norm()) + 1e-300)
    return e if math.isfinite(e) else float("inf")


# ---------------------------------------------------------------------------------------------------------------------
# degenerate terms
# ---------------------------------------------------------------------------------------------------------------------

def _cubic_residual(col, W):
    d = W.shape[1]
    exps = [e for e in itertools.product(range(4), repeat=d) if sum(e) <= 3]
    A = torch.stack([torch.prod(torch.stack([W[:, j] ** e[j] for j in range(d)]), 0) for e in exps], 1)
    A = A / A.abs().max(0).values.clamp(min=1e-300)
    sol = torch.linalg.lstsq(A, col.reshape(-1, 1), driver="gelsd").solution[:, 0]
    return float((col - A @ sol).norm()) / (float(col.norm()) + 1e-300)


def degenerate(term, W):
    """True if a term with constants is singular on the data (non-finite, or a denominator, log argument or base of a
    non-integer power reaching zero there), or has collapsed exactly into a cubic polynomial over the data."""
    if term.k == 0:
        return False
    if not all(math.isfinite(v) for v in term.theta):
        return True
    inf = info(term.key)
    th = torch.tensor(term.theta, dtype=W.dtype)
    with torch.no_grad(), np.errstate(all="ignore"):
        col = inf.fn(W, th)
        if not bool(torch.isfinite(col).all()):
            return True
        for s in inf.singular:
            v = s(W, th)
            if not bool(torch.isfinite(v).all()):
                return True
            vmin, vmax = float(v.min()), float(v.max())
            if vmin <= 0 <= vmax or min(abs(vmin), abs(vmax)) < 1e-9 * max(abs(vmin), abs(vmax), 1e-300):
                return True
        return _cubic_residual(col, W) < COLLAPSE_EXACT


# ---------------------------------------------------------------------------------------------------------------------
# global search and the algebraic start
# ---------------------------------------------------------------------------------------------------------------------

def _free(terms):
    return [(m, a) for m, t in enumerate(terms) for a in range(t.k) if a not in t.fixed]


def _thetas_from_unit(terms, slots, U, W):
    """Constant arrays per term (B, k) for unit points U (B, len(slots)); fixed constants keep their values."""
    B = U.shape[0]
    out = []
    for m, t in enumerate(terms):
        if t.k == 0:
            out.append(None)
            continue
        Um = np.full((B, t.k), 0.5)
        for s_, (mm, a) in enumerate(slots):
            if mm == m:
                Um[:, a] = U[:, s_]
        P = natural_thetas(info(t.key), Um, W)
        for a in t.fixed:
            P[:, a] = t.theta[a]
        out.append(P)
    return out


def _batch_resid(terms, thetas, W, y):
    S = next(P.shape[0] for P in thetas if P is not None)
    cols = []
    for t, P in zip(terms, thetas):
        with torch.no_grad(), np.errstate(all="ignore"):
            if P is None:
                v = value(t, W).reshape(1, -1).expand(S, -1)
            else:
                v = info(t.key).fn(W, torch.tensor(P)).T
        cols.append(v)
    Phi = torch.stack(cols, 2)
    good = torch.isfinite(Phi).all(dim=(1, 2)) & (Phi.abs().amax(dim=(1, 2)) < 1e12)
    Phi = torch.where(good[:, None, None], Phi, torch.zeros_like(Phi))
    try:
        sol = torch.linalg.lstsq(Phi, y.reshape(1, -1, 1).expand(S, -1, 1), driver="gelsd").solution
        rn = ((Phi @ sol).squeeze(2) - y.unsqueeze(0)).norm(dim=1)
    except RuntimeError:
        rn = torch.full((S,), float("inf"), dtype=W.dtype)
    return torch.where(good & torch.isfinite(rn), rn, torch.full_like(rn, float("inf"))).numpy()


def global_search(terms, W, y, rng):
    """Joint search over all free constants in their natural ranges (Sobol points, then refinement around the best),
    as fitter30.global_search. Returns up to REFINE_TOP distinct term lists, best first."""
    from scipy.stats import qmc
    slots = _free(terms)
    if not slots:
        return []
    h = len(slots)
    sob = qmc.Sobol(h, scramble=True, seed=int(rng.integers(1 << 31)))
    U = sob.random(min(2048, max(F30.SOBOL_N // 4, 256 * h)))
    TH = _thetas_from_unit(terms, slots, U, W)
    rn = _batch_resid(terms, TH, W, y)
    for rnd in range(F30.REFINE_ROUNDS):
        order = np.argsort(rn)
        top = []
        for k in order:
            if not np.isfinite(rn[k]):
                break
            if all(np.max(np.abs(U[k] - U[j])) > 0.02 for j in top):
                top.append(k)
            if len(top) >= F30.REFINE_TOP:
                break
        if not top:
            break
        rad = 0.15 * 0.4 ** rnd
        newU = np.concatenate([np.clip(U[k] + rad * (rng.random((F30.REFINE_N, h)) * 2 - 1), 0, 1) for k in top])
        newTH = _thetas_from_unit(terms, slots, newU, W)
        U = np.concatenate([U, newU])
        TH = [None if a is None else np.concatenate([a, b]) for a, b in zip(TH, newTH)]
        rn = np.concatenate([rn, _batch_resid(terms, newTH, W, y)])
    order = np.argsort(rn)
    picks, outs = [], []
    for k in order:
        if not np.isfinite(rn[k]):
            break
        if all(np.max(np.abs(U[k] - U[j])) > 0.02 for j in picks):
            picks.append(k)
            outs.append([t if P is None else t.with_theta(P[k]) for t, P in zip(terms, TH)])
        if len(picks) >= F30.REFINE_TOP:
            break
    return outs


def algebraic_start(terms, Wd, yd, rounds=F30.SK_ROUNDS):
    """One rational term (constants only in a denominator affine in them) fitted jointly with the others by clearing
    the denominator: one linear least-squares problem in the lifted unknowns, Sanathanan-Koerner reweighting, on dense
    rate data. Other terms enter at their current constants. Returns a list of candidate term lists (at most one)."""
    rat = [m for m, t in enumerate(terms) if info(t.key).rational is not None and t.k and not t.fixed]
    if len(rat) != 1:
        return []
    m0 = rat[0]
    num, d0f, djf = info(terms[m0].key).rational
    with torch.no_grad(), np.errstate(all="ignore"):
        N, d0, dj = num(Wd), d0f(Wd), [q(Wd) for q in djf]
        phis = [value(t, Wd) for m, t in enumerate(terms) if m != m0]
    J = len(dj)
    wts = torch.ones_like(yd)
    K = None
    for _ in range(rounds):
        cols = [-(yd * q) for q in dj] + [p * d0 for p in phis] + [p * q for p in phis for q in dj] + [N]
        A = torch.stack(cols, 1) / wts.unsqueeze(1)
        b = yd * d0 / wts
        if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(b).all()):
            return []
        try:
            z = torch.linalg.lstsq(A, b.reshape(-1, 1), driver="gelsd").solution[:, 0]
        except RuntimeError:
            return []
        K = z[:J]
        den = d0 + sum(K[j] * dj[j] for j in range(J))
        if not bool(torch.isfinite(den).all()) or float(den.abs().min()) == 0:
            break
        wts = den
    if K is None or not bool(torch.isfinite(K).all()):
        return []
    out = list(terms)
    out[m0] = terms[m0].with_theta([float(v) for v in K])
    return [out]


def exponent_uncertainty(terms, W, y, floor):
    """[(term, position, value, sigma)] for every free fitted exponent (fitter30.exponent_uncertainty)."""
    slots = _free(terms)
    ex = [(s_, m, a) for s_, (m, a) in enumerate(slots) if a in info(terms[m].key).exponents]
    if not ex:
        return []
    th0 = [list(t.theta) for t in terms]

    def resid(th):
        with torch.no_grad():
            Phi = torch.stack([value(t, W, th[m]) for m, t in enumerate(terms)], 1)
        out = SS._project(Phi, y)
        return None if out is None else out[0]

    r0 = resid(th0)
    if r0 is None:
        return []
    Jm = torch.empty(len(y), len(slots), dtype=W.dtype)
    for s_, (m, a) in enumerate(slots):
        v = th0[m][a]
        h = 1e-6 * (1.0 + abs(v))
        th = [list(x) for x in th0]
        th[m][a] = v + h
        r1 = resid(th)
        if r1 is None:
            return [(m_, a_, terms[m_].theta[a_], float("inf")) for _, m_, a_ in ex]
        Jm[:, s_] = (r1 - r0) / h
    sig_y = floor / math.sqrt(len(y))
    H = Jm.T @ Jm
    try:
        cov = torch.linalg.pinv(H, rtol=1e-14) * sig_y ** 2
    except RuntimeError:
        return []
    full = bool(torch.isfinite(H).all()) and int(torch.linalg.matrix_rank(H)) == len(slots)
    out = []
    for s_, m, a in ex:
        var = float(cov[s_, s_])
        sigma = math.sqrt(var) if var > 0 and math.isfinite(var) and full else float("inf")
        out.append((m, a, terms[m].theta[a], sigma))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------------------------------------------------

class FitterL1:
    """v2-strong on L1 terms (module docstring). bank: Bank on the rate points; dense: fitter30.DenseRates."""

    def __init__(self, bank, dense, seed=0, lm_iters=10, prepolish=200, keep=4, sweeps=2, basins=2):
        self.bank, self.dense = bank, dense
        self.Wd = dense.X / dense.X.new_tensor(1.0) if dense is not None else None
        self.rng = np.random.default_rng(seed)
        self.lm_iters, self.prepolish, self.keep, self.sweeps, self.basins = lm_iters, prepolish, keep, sweeps, basins
        self.stats = {"ordinary": 0, "ordinary_seconds": 0.0, "algebraic_tried": 0, "algebraic_won": 0, "beam_won": 0,
                      "thorough": 0, "thorough_seconds": 0.0, "global_won": 0, "extra_global_won": 0,
                      "exponents_checked": 0, "snaps_allowed": 0, "snaps_kept": 0}
        self.last = None

    def set_dense_coordinates(self, sig, mu):
        self.Wd = (self.dense.Z * sig + mu) / sig

    @staticmethod
    def nonlinear(keys):
        return sum(1 for k in keys if info(k).k > 0)

    def precise_gate(self, keys, res, rel_rate, floor):
        if rel_rate <= SS.PRECISE_GATE and res <= SS.PRECISE_FLOOR * floor:
            return True
        return self.nonlinear(keys) >= 2 and rel_rate <= 1e-2

    # -- the grid sweep (structure_search.GridSweepFitter) --
    def _profile(self, key, keys, idx, y):
        B = self.bank.bank
        others = [B[u][0][:, idx[u]] for u in keys if u != key]
        C = B[key][0]
        if others:
            Q = torch.linalg.qr(torch.stack(others, 1))[0]
            r0 = y - Q @ (Q.T @ y)
            Cp = C - Q @ (Q.T @ C)
        else:
            r0, Cp = y, C
        cn = (Cp * Cp).sum(0)
        return torch.where(cn > 1e-24, (Cp.T @ r0) ** 2 / cn, torch.zeros_like(cn))

    def _term(self, key, g):
        th = self.bank.bank[key][1][g] if info(key).k else ()
        return LTerm(key, th, grid_index=g)

    def sweep_fit(self, keys, W, y, warm=None, floor=0.0, basins=None):
        B = self.bank.bank
        warm = warm or {}
        eff = {"starts": 0, "polished": 0}
        if not keys:
            return [], float(y.norm()), torch.zeros(0, dtype=torch.float64), eff
        for k in keys:
            self.bank.add(k)
        if any(k not in B for k in keys):
            return [LTerm(k, [1.0] * info(k).k) for k in keys], float("inf"), None, eff
        idx = {}
        for k in keys:
            C = B[k][0]
            w = warm.get(k)
            idx[k] = w if isinstance(w, int) and w < C.shape[1] else None
            if idx[k] is None:
                cn = (C * C).sum(0) + 1e-300
                idx[k] = int(torch.argmax((C.T @ y) ** 2 / cn)) if C.shape[1] > 1 else 0
        pieces = [k for k in keys if B[k][0].shape[1] > 1]
        for _ in range(self.sweeps if pieces else 0):
            for k in pieces:
                idx[k] = int(torch.argmax(self._profile(k, keys, idx, y)))
                eff["starts"] += B[k][0].shape[1]
        Phi = torch.stack([B[k][0][:, idx[k]] for k in keys], 1)
        out = SS._project(Phi, y)
        res = float(out[0].norm()) if out is not None else float("inf")
        c = out[1] if out is not None else None
        terms = [self._term(k, idx[k]) for k in keys]
        if pieces and math.isfinite(res):
            starts = [dict(idx)]
            inherited = {k: warm[k] if isinstance(warm.get(k), int) and warm[k] < B[k][0].shape[1] else idx[k] for k in keys}
            if inherited != idx:
                starts.append(inherited)
            fresh = pieces if basins else [k for k in pieces if not isinstance(warm.get(k), int)]
            for k in fresh:
                order = torch.argsort(self._profile(k, keys, idx, y), descending=True).tolist()
                starts += [{**idx, k: g} for g in order if g != idx[k]][:max((basins or self.basins) - 1, 0)]
            for st in starts:
                cand = [self._term(k, st[k]) for k in keys]
                try:
                    t2, r2, c2 = fast_refine(cand, W, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                eff["polished"] += 1
                if math.isfinite(r2) and r2 < res:
                    for tm, k in zip(t2, keys):
                        tm.grid_index = st[k]
                    terms, res, c = t2, r2, c2
        return terms, res, c, eff

    def _beam(self, keys, terms, W, y, width=3, keep=1):
        B = self.bank.bank
        pieces = [j for j, k in enumerate(keys) if B[k][0].shape[1] > 1]
        if len(pieces) < 2:
            return []
        idx = {k: (t.grid_index if isinstance(t.grid_index, int) and t.grid_index < B[k][0].shape[1] else 0)
               for k, t in zip(keys, terms)}
        tops = [torch.argsort(self._profile(keys[j], keys, idx, y), descending=True)[:width].tolist() for j in pieces]
        combos = list(itertools.product(*tops))
        if len(combos) > 729:
            return []
        Phi = torch.stack([torch.stack([B[k][0][:, dict(zip([keys[j] for j in pieces], cb)).get(k, idx[k])] for k in keys], 1)
                           for cb in combos], 0)
        try:
            sol = torch.linalg.lstsq(Phi, y.reshape(1, -1, 1).expand(len(combos), -1, 1), driver="gelsd").solution
            rn = ((Phi @ sol).squeeze(2) - y.unsqueeze(0)).norm(dim=1)
        except RuntimeError:
            return []
        rn = torch.where(torch.isfinite(rn), rn, torch.full_like(rn, float("inf")))
        out = []
        for ci in torch.argsort(rn)[:keep].tolist():
            sel = dict(idx)
            sel.update(dict(zip([keys[j] for j in pieces], combos[ci])))
            out.append([self._term(k, sel[k]) for k in keys])
        return out

    def fit(self, keys, W, y, warm=None, floor=0.0, eq=None):
        """Ordinary fit: sweep, joint beam, algebraic start; the best kept. Returns (terms, residual, coefs, eff)."""
        t0 = time.perf_counter()
        terms, res, c, eff = self.sweep_fit(keys, W, y, warm, floor)
        if math.isfinite(res) and res > 2.0 * floor:
            for cand in self._beam(keys, terms, W, y):
                try:
                    t2, r2, c2 = fast_refine(cand, W, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                if math.isfinite(r2) and r2 < res:
                    terms, res, c = t2, r2, c2
                    self.stats["beam_won"] += 1
        if any(info(k).rational is not None for k in keys) and math.isfinite(res) and res > 2.0 * floor and eq is not None:
            self.stats["algebraic_tried"] += 1
            for cand in algebraic_start(terms, self.Wd, self.dense.dZ[:, eq]):
                try:
                    t2, r2, c2 = fast_refine(cand, W, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                if math.isfinite(r2) and r2 < res:
                    terms, res, c = t2, r2, c2
                    self.stats["algebraic_won"] += 1
        dt = time.perf_counter() - t0
        eff["seconds"] = dt
        self.stats["ordinary"] += 1
        self.stats["ordinary_seconds"] += dt
        return terms, res, c, eff

    def thorough(self, keys, W, y, floor, current, eq=None, si_tr=None, si_va=None):
        """Thorough fit: sweep with more basins, algebraic start, joint global search, the fresh global search's best
        `keep` pre-polished, each polished; then identifiability-gated exponent snapping judged by held-back error."""
        t0 = time.perf_counter()
        best = current
        cands = []
        t1, r1, c1, _ = self.sweep_fit(keys, W, y, {}, floor, basins=SS.PRECISE_BASINS)
        cands.append((t1, "sweep"))
        if eq is not None:
            for cand in algebraic_start(current[0], self.Wd, self.dense.dZ[:, eq]):
                cands.append((cand, "algebraic"))
        for cand in global_search(current[0], W, y, self.rng):
            cands.append((cand, "global"))
        if eq is not None:
            rng = np.random.default_rng([31, int(eq), zlib.crc32(repr(tuple(keys)).encode())])
            for cand in global_search(current[0], W, y, rng)[:self.keep]:
                if self.prepolish:
                    try:
                        cand = fast_refine(cand, W, y, iters=self.prepolish, stop_below=1e-14 * float(y.norm()))[0]
                    except (RuntimeError, ValueError):
                        pass
                cands.append((cand, "extra_global"))
        winner = "current"
        for terms, tag in cands:
            try:
                t2, r2, c2 = fast_refine(terms, W, y, iters=2 * self.lm_iters, stop_below=1.5 * floor)
            except (RuntimeError, ValueError):
                continue
            if math.isfinite(r2) and r2 < best[1]:
                best, winner = (t2, r2, c2), tag
        if winner == "global":
            self.stats["global_won"] += 1
        if winner == "extra_global":
            self.stats["extra_global_won"] += 1
        terms, res, coefs = best
        unc = exponent_uncertainty(terms, W, y, floor)
        self.stats["exponents_checked"] += len(unc)
        alt = None
        for m, a, val, sigma in unc:
            if sigma > F30.SNAP_SIGMA and round(val) != 0:
                self.stats["snaps_allowed"] += 1
                trial = list(terms)
                th = list(trial[m].theta); th[a] = float(round(val))
                trial[m] = trial[m].with_theta(th, fixed=trial[m].fixed | {a})
                try:
                    t3, r3, c3 = fast_refine(trial, W, y, iters=2 * self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                if math.isfinite(r3):
                    alt = (t3, r3, c3)
                    terms = t3
        if alt is not None and si_tr is not None and si_va is not None and eq is not None:
            scores = []
            for cand in (best, alt):
                try:
                    tt, rt, ct = refine_integral(cand[0], si_tr, eq)
                    rv = heldback_error(tt, ct, si_va, eq) if ct is not None else float("inf")
                except (RuntimeError, ValueError):
                    rv = float("inf")
                scores.append(rv)
            if scores[1] < scores[0]:
                best = alt
                self.stats["snaps_kept"] += 1
        dt = time.perf_counter() - t0
        self.stats["thorough"] += 1
        self.stats["thorough_seconds"] += dt
        self.last = {"winner": winner, "snap_kept": alt is not None and best is alt}
        return best


# ---------------------------------------------------------------------------------------------------------------------
# a system's context and the judgement of one structure
# ---------------------------------------------------------------------------------------------------------------------

def system_context(bursts, seed=1, keys=()):
    """The learner's view of the bursts, its coordinates and step integrals, a bank for `keys`, and the fitter."""
    view = PN.learner_view(bursts, seed=seed)
    if view["centred"]:
        raise ValueError("centred coordinates")
    mu, sig = view["mu"], view["sig"]
    W = (view["Zc"] * sig + mu) / sig
    dense = F30.DenseRates(view["Ztr"], mu, sig)
    ctx = {"view": view, "W": W, "G": view["G"], "noise": view["noise"],
           "si_tr": StepIntegrals(view["Ztr"], mu, sig), "si_va": StepIntegrals(view["Zva"], mu, sig),
           "bank": Bank(keys, W), "dense": dense}
    return ctx


def make_fitter(ctx, seed=1):
    f = FitterL1(ctx["bank"], ctx["dense"], seed=seed)
    f.set_dense_coordinates(ctx["view"]["sig"], ctx["view"]["mu"])
    return f


def equation_targets(ctx, i):
    y = ctx["G"][:, i]
    yn = float(y.norm()) + 1e-300
    floor = max(float(ctx["noise"][i]) * math.sqrt(len(y)), 1e-15 * yn)
    tau = max(SS.TAU_EXACT, SS.TAU_FLOOR * floor / yn)
    return y, yn, floor, tau


def evaluate_structure(keys, ctx, i, fitter, warm=None):
    """Judge one structure for equation i as EquationSearch.evaluate does (no cap): ordinary fit, the precise stage's
    gate, a thorough fit unless already at the floor, step-integral refit, held-back check. Returns a node dict."""
    W = ctx["W"]
    y, yn, floor, tau = equation_targets(ctx, i)
    keys = tuple(keys)
    t0 = time.perf_counter()
    terms, res, coefs, e = fitter.fit(keys, W, y, warm, floor=floor, eq=i)
    t_fit = time.perf_counter() - t0
    rel_rate = res / yn if math.isfinite(res) else float("inf")
    rel_t = rel_v = float("inf")
    thorough_now, t_th = False, 0.0
    gate = bool(keys) and fitter.precise_gate(keys, res, rel_rate, floor)
    if gate:
        try:
            if res > 2.0 * floor:
                th = time.perf_counter()
                terms, res, coefs = fitter.thorough(keys, W, y, floor, (terms, res, coefs), eq=i, si_tr=ctx["si_tr"],
                                                    si_va=ctx["si_va"])
                t_th = time.perf_counter() - th
                thorough_now = True
            terms2, rel_t, c2 = refine_integral(terms, ctx["si_tr"], i)
            if c2 is not None and math.isfinite(rel_t):
                terms, coefs = terms2, c2
                rel_v = heldback_error(terms, coefs, ctx["si_va"], i)
        except (RuntimeError, ValueError):
            pass
    try:
        degen = any(degenerate(t, W) for t in terms)
    except (RuntimeError, ValueError, ZeroDivisionError):
        degen = True
    return {"keys": keys, "terms": terms, "coefs": coefs, "res": res, "rel_rate": rel_rate, "rel_t": rel_t, "rel_v": rel_v,
            "tau": tau, "floor_rel": floor / yn, "gate": gate, "thorough": thorough_now, "degenerate": degen,
            "exact": bool(rel_v <= tau and rel_t <= tau and not degen), "fit_seconds": t_fit, "thorough_seconds": t_th,
            "seconds": time.perf_counter() - t0}
