"""Trial 26 learner: expression-network discovery with machine-precision polishing on the exact one-step map.

The learner receives only noise-free observation bursts (M, L, d) at a unit sampling interval. It never sees the true
vector field and never calls the simulator. See docs/research-log.md (Trial 26).

Pipeline per system:
  1. derivative screening (9-point differences; their error is estimated from the data) proposes a generous support;
  2. Levenberg-Marquardt polishes all constants on the one-step flow map, integrated by the model's own fixed-step
     8th-order Runge-Kutta;
  3. stepwise refinement on the flow map: terms whose removal costs nothing are pruned, and missing terms are added
     by their exact flow-map sensitivity;
  4. acceptance only if held-back bursts are reproduced to <= 1e-13 (scaled).
Knowledge from history enters through `Knowledge`: a template prior and promoted composites (memory), and/or
predicted influences / term counts / levels (guide or statistical screen) that restrict a first search pass.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.integrate._ivp import dop853_coefficients as D

torch.set_default_dtype(torch.float64)
FD9 = torch.tensor([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
FD7 = torch.tensor([0.0, -1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60, 0.0])
FD5 = torch.tensor([0.0, 0.0, 1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12, 0.0, 0.0])
A = torch.tensor(D.A[:D.N_STAGES, :D.N_STAGES]); B = torch.tensor(D.B)
STAGES = [[(j, float(A[i, j])) for j in range(i) if A[i, j] != 0] for i in range(D.N_STAGES)]
BW = [(i, float(B[i])) for i in range(D.N_STAGES) if B[i] != 0]
NONPOLY = ("sin", "exp", "tanh", "abs", "recip")
GRIDS = {"sin": [(w, p) for w in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) for p in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)],
         "exp": [(a,) for a in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)],
         "tanh": [(a, b) for a in (0.5, 1.0, 2.0, 4.0) for b in (-1.0, 0.0, 1.0)],
         "abs": [(b,) for b in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)],
         "pow": [(p,) for p in (0.5, 1.5, 2.5, -1.0, -2.0)]}
VAL_TOL = 1e-13
CENTER_RATIO = 3.0   # centre a variable only if |mean| > 3 std (keeps polynomial structure sparse)


# ---------------- terms ----------------

@dataclass
class Factor:
    family: str
    var: int
    theta: tuple


@dataclass
class Term:
    mono: tuple                      # exponents over scaled coordinates z
    factors: tuple = ()              # up to two Factors
    level: int = 1

    def key(self):
        """Template key for the memory prior: shape of the term, independent of which variables it uses."""
        deg = sum(self.mono)
        fam = tuple(sorted(f.family for f in self.factors))
        if not fam:
            return ("mono", deg)
        same = any(self.mono[f.var] > 0 for f in self.factors) if deg else None
        return ("prod", deg, fam, same)

    def shape(self, comp):
        """Role shape relative to the component it appears in (ignores variable labels): used for similarity."""
        self_e = self.mono[comp]
        others = tuple(sorted(e for k, e in enumerate(self.mono) if k != comp and e))
        facs = tuple(sorted((f.family, f.var == comp) for f in self.factors))
        return (self_e, others, facs)

    def variables(self):
        return {k for k, e in enumerate(self.mono) if e} | {f.var for f in self.factors}

    def sig(self):
        return (self.mono, tuple((f.family, f.var) for f in self.factors))

    def nparams(self):
        return sum(len(f.theta) for f in self.factors)


def factor_value(fac, Z, X, theta):
    z, x = Z[..., fac.var], X[..., fac.var]
    t = theta
    if fac.family == "sin":
        return torch.sin(t[0] * z + t[1])
    if fac.family == "exp":
        return torch.exp(t[0] * z)
    if fac.family == "tanh":
        return torch.tanh(t[0] * z + t[1])
    if fac.family == "abs":
        return torch.abs(z - t[0])
    if fac.family == "recip":
        return 1.0 / (t[0] + z)
    if fac.family == "pow":
        return x ** t[0]
    if fac.family == "hill":
        xp = x ** t[1]
        return xp / (t[0] + xp)
    if fac.family == "hillr":
        return t[0] / (t[0] + x ** t[1])
    raise ValueError(fac.family)


def term_value(term, Z, X, thetas=None):
    v = torch.ones(Z.shape[:-1], dtype=Z.dtype)
    for k, e in enumerate(term.mono):
        if e:
            v = v * Z[..., k] ** e
    for i, fac in enumerate(term.factors):
        th = thetas[i] if thetas is not None else torch.tensor(fac.theta)
        v = v * factor_value(fac, Z, X, th)
    return v


# ---------------- candidate generation ----------------

def monomials(d, maxdeg=3):
    out = []
    def rec(prefix, left, k):
        if k == d:
            out.append(tuple(prefix)); return
        for e in range(left + 1):
            rec(prefix + [e], left - e, k + 1)
    rec([], maxdeg, 0)
    return [m for m in out if sum(m) <= maxdeg]


def factor_grid(fam, j, Z, X):
    zj = Z[:, j]
    if fam == "abs":      # breakpoints at data quantiles (coordinates are not necessarily centred)
        q = torch.quantile(zj, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        return [Factor("abs", j, (float(c),)) for c in q]
    if fam == "tanh":     # transition centres at data quantiles
        q = torch.quantile(zj, torch.tensor([0.25, 0.5, 0.75]))
        return [Factor("tanh", j, (a, -a * float(c))) for a in (0.5, 1.0, 2.0, 4.0) for c in q]
    if fam in GRIDS and fam != "pow":
        return [Factor(fam, j, th) for th in GRIDS[fam]]
    if fam == "recip":
        lo, hi, span = float(zj.min()), float(zj.max()), float(zj.max() - zj.min()) + 1e-9
        return [Factor("recip", j, (-lo + c * span,)) for c in (0.25, 0.5, 1.0, 2.0)] + \
               [Factor("recip", j, (-hi - c * span,)) for c in (0.25, 0.5, 1.0, 2.0)]
    xj = X[:, j]
    if float(xj.min()) <= 0:
        return []
    if fam == "pow":
        return [Factor("pow", j, th) for th in GRIDS["pow"]]
    q = torch.quantile(xj, torch.tensor([0.1, 0.5, 0.9]))
    return [Factor(fam, j, (float(qq) ** n, float(n))) for n in (1.0, 2.0, 4.0) for qq in q]


def frequency_scan(Z, G, n_top=4):
    """Data-adaptive sine frequencies per variable: peaks of the energy of the derivative data (all components,
    with the global linear part removed) captured by span{sin(w z_j), cos(w z_j)}, scanned over w in [0.2, 80]."""
    Zn, Gn = Z.numpy(), G.numpy()
    n, d = Zn.shape
    base = np.concatenate([np.ones((n, 1)), Zn], 1)
    Qb = np.linalg.qr(base)[0]
    Y = Gn - Qb @ (Qb.T @ Gn)
    Y = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-300)
    ws = np.geomspace(0.2, 80.0, 500)
    out = {}
    for j in range(d):
        Sx, Cx = np.sin(np.outer(Zn[:, j], ws)), np.cos(np.outer(Zn[:, j], ws))
        Sx = Sx - Qb @ (Qb.T @ Sx); Cx = Cx - Qb @ (Qb.T @ Cx)
        ss, cc, sc = (Sx * Sx).sum(0), (Cx * Cx).sum(0), (Sx * Cx).sum(0)
        det = np.clip(ss * cc - sc ** 2, 1e-300, None)
        gain = np.zeros(len(ws))
        for i in range(Y.shape[1]):
            sy, cy = Sx.T @ Y[:, i], Cx.T @ Y[:, i]
            gain += (cc * sy ** 2 - 2 * sc * sy * cy + ss * cy ** 2) / det
        gain[(ss < 1e-12) | (cc < 1e-12)] = 0
        peaks = [k for k in range(1, len(ws) - 1) if gain[k] >= gain[k - 1] and gain[k] >= gain[k + 1]]
        peaks.sort(key=lambda k: -gain[k])
        out[j] = [float(ws[k]) for k in peaks[:n_top]]
    return out


def candidates(d, Z, X, levels, promoted=(), G=None):
    """Candidate terms (with grid-initialized nonlinear parameters) for the requested levels."""
    lin = [tuple(int(i == k) for i in range(d)) for k in range(d)]
    zero = tuple([0] * d)
    base_f = [f for fam in NONPOLY + ("pow",) for j in range(d) for f in factor_grid(fam, j, Z, X)]
    if G is not None:   # data-adaptive sine frequencies (generic, from this system's own data)
        fs = frequency_scan(Z, G)
        have = {(f.var, round(f.theta[0], 6)) for f in base_f if f.family == "sin"}
        base_f += [Factor("sin", j, (w, ph)) for j, wl in fs.items() for w in wl if (j, round(w, 6)) not in have
                   for ph in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)]
    sat_f = [f for fam in ("hill", "hillr") for j in range(d) for f in factor_grid(fam, j, Z, X)]
    out = []
    if 0 in levels or 1 in levels:
        out += [Term(m, (), 1) for m in monomials(d)]
    if 1 in levels:
        out += [Term(zero, (f,), 1) for f in base_f] + [Term(l, (f,), 1) for f in base_f for l in lin]
    if 2 in levels:
        out += [Term(zero, (f, g), 2) for i, f in enumerate(base_f) for g in base_f[i + 1:] if (f.family, f.var) != (g.family, g.var)]
        out += [Term(zero, (f,), 2) for f in sat_f] + [Term(l, (f,), 2) for f in sat_f for l in lin]
    if 3 in levels:
        out += [Term(zero, (f, g), 3) for i, f in enumerate(sat_f) for g in sat_f[i + 1:] if f.var != g.var]
    if promoted and levels != (0,):  # memory: composite templates promoted to level 1
        keys = set(promoted)
        extra = [t for lv in (2, 3) if lv not in levels for t in candidates(d, Z, X, (lv,), G=G) if t.key() in keys]
        for t in extra:
            t.level = 1
        out += extra
    return out


# ---------------- derivative screening ----------------

def design(terms, Z, X, thetas=None):
    cols = [term_value(t, Z, X, None if thetas is None else thetas[i]) for i, t in enumerate(terms)]
    return torch.stack(cols, -1) if cols else torch.zeros(Z.shape[:-1] + (0,))


def _factor_block(fam, idx_var, thetas, Z, X):
    """Vectorized values of one factor family for many (variable, theta) pairs: (n, k)."""
    z, x = Z[..., idx_var], X[..., idx_var]
    th = thetas
    if fam == "sin":
        return torch.sin(z * th[:, 0] + th[:, 1])
    if fam == "exp":
        return torch.exp(z * th[:, 0])
    if fam == "tanh":
        return torch.tanh(z * th[:, 0] + th[:, 1])
    if fam == "abs":
        return torch.abs(z - th[:, 0])
    if fam == "recip":
        return 1.0 / (th[:, 0] + z)
    if fam == "pow":
        return x ** th[:, 0]
    if fam == "hill":
        xp = x ** th[:, 1]
        return xp / (th[:, 0] + xp)
    if fam == "hillr":
        return th[:, 0] / (th[:, 0] + x ** th[:, 1])
    raise ValueError(fam)


def design_fast(terms, Z, X):
    """Same columns as design(terms, Z, X) for fixed terms, computed in a few vectorized operations."""
    n = Z.shape[:-1]
    if not terms:
        return torch.zeros(n + (0,))
    monos, facs = {}, {}
    mi, f1, f2 = [], [], []
    for t in terms:
        mi.append(monos.setdefault(t.mono, len(monos)))
        fl = [facs.setdefault((f.family, f.var, tuple(f.theta)), len(facs)) for f in t.factors]
        f1.append(fl[0] if fl else -1)
        f2.append(fl[1] if len(fl) > 1 else -1)
    E = torch.tensor(list(monos.keys()), dtype=Z.dtype)
    Mc = torch.prod(Z.unsqueeze(-2) ** E, dim=-1)                          # (..., n_mono)
    Fc = torch.ones(n + (len(facs) + 1,), dtype=Z.dtype)                    # last column = 1 (no factor)
    by_fam = {}
    for (fam, var, th), k in facs.items():
        by_fam.setdefault(fam, []).append((k, var, th))
    for fam, lst in by_fam.items():
        cols = torch.tensor([k for k, _, _ in lst])
        vars_ = torch.tensor([v for _, v, _ in lst])
        width = max(len(th) for _, _, th in lst)
        th = torch.tensor([list(t) + [0.0] * (width - len(t)) for _, _, t in lst], dtype=Z.dtype)
        Fc[..., cols] = _factor_block(fam, vars_, th, Z, X)
    return Mc[..., mi] * Fc[..., f1] * Fc[..., f2]


def lstsq_resid(Phi, y):
    if Phi.shape[1] == 0:
        return y.clone(), torch.zeros(0)
    if not bool(torch.isfinite(Phi).all()):          # unusable design: infinite residual (filtered by callers)
        return torch.full_like(y, float("inf")), torch.zeros(Phi.shape[1], dtype=y.dtype)
    try:
        c = torch.linalg.lstsq(Phi, y.unsqueeze(1), driver="gelsd").solution.squeeze(1)
    except RuntimeError:                              # LAPACK failure: same treatment
        return torch.full_like(y, float("inf")), torch.zeros(Phi.shape[1], dtype=y.dtype)
    return y - Phi @ c, c


def _split_thetas(terms, p):
    k, thetas = 0, []
    for t in terms:
        tt = []
        for f in t.factors:
            n = len(f.theta); tt.append(p[k:k + n]); k += n
        thetas.append(tt)
    return thetas


def _with_thetas(terms, thetas):
    return [Term(t.mono, tuple(Factor(f.family, f.var, tuple(float(v) for v in th)) for f, th in zip(t.factors, tt)), t.level)
            for t, tt in zip(terms, thetas)]


def degenerate(term, Z, X):
    """True if a nonlinear factor has collapsed, over the observed range, into something simpler (a constant, a
    linear trend or a monomial) or is singular there. Generic limits, the same for every system."""
    for f in term.factors:
        z, x = Z[:, f.var], X[:, f.var]
        zmin, zmax = float(z.min()), float(z.max())
        s = max(zmax - zmin, 1e-12)
        th = f.theta
        if not all(math.isfinite(v) for v in th):
            return True
        if f.family == "sin" and abs(th[0]) * s < 0.5:
            return True
        if f.family == "exp" and (abs(th[0]) * s < 0.2 or abs(th[0]) * max(abs(zmin), abs(zmax)) > 50):
            return True
        if f.family == "tanh":
            lo, hi = sorted((th[0] * zmin + th[1], th[0] * zmax + th[1]))
            if hi - lo < 0.3 or lo > 4 or hi < -4:
                return True
        if f.family == "abs" and not (zmin < th[0] < zmax):
            return True
        if f.family == "recip":
            lo, hi = th[0] + zmin, th[0] + zmax
            if lo <= 0 <= hi or min(abs(lo), abs(hi)) > 2 * s:
                return True
        if f.family in ("pow", "hill", "hillr"):
            if float(x.min()) <= 0:
                return True
            pw = th[0] if f.family == "pow" else th[1]
            if f.family == "pow" and min(abs(pw - k) for k in (0, 1, 2, 3)) < 0.02:
                return True
            if f.family != "pow":
                if pw <= 0.05:
                    return True
                xp = x ** pw
                if th[0] < 0.05 * float(xp.min()) or th[0] > 20 * float(xp.max()):
                    return True
    return False


def simplify_degenerate(model, Zc, Xc):
    """Replace every degenerate term by the monomials it imitates on the data (constant and linear in its variables).
    Returns (model, changed)."""
    changed = False
    d = len(model.supports)
    for i in range(d):
        terms, coefs = model.supports[i], model.coefs[i]
        keep = [a for a, t in enumerate(terms) if not degenerate(t, Zc, Xc)]
        if len(keep) == len(terms):
            continue
        changed = True
        add = set()
        for a, t in enumerate(terms):
            if a not in keep:
                add.add(tuple([0] * d))
                for v in t.variables():
                    add.add(tuple(int(k == v) for k in range(d)))
                if sum(t.mono):
                    add.add(tuple(t.mono))
        new_terms = [terms[a] for a in keep]
        have = {u.sig() for u in new_terms}
        new_coefs = [float(coefs[a]) for a in keep]
        for m in sorted(add):
            tm = Term(m, (), 1)
            if tm.sig() not in have:
                new_terms.append(tm); new_coefs.append(0.0); have.add(tm.sig())
        model.supports[i] = new_terms
        model.coefs[i] = torch.tensor(new_coefs)
    return model, changed


def refine_support(terms, Z, X, y, iters=10):
    """Gauss-Newton on the nonlinear parameters of a support, linear coefficients projected out (VarPro)."""
    flat = [float(v) for t in terms for f in t.factors for v in f.theta]
    if not flat:
        r, c = lstsq_resid(design(terms, Z, X), y)
        return terms, float(r.norm()), c
    resid = lambda p: lstsq_resid(design(terms, Z, X, _split_thetas(terms, p)), y)[0]
    p = torch.tensor(flat)
    r = resid(p)
    lam = 1e-3
    for _ in range(iters):
        J = torch.func.jacfwd(resid)(p)
        if not torch.all(torch.isfinite(J)):
            break
        H = J.T @ J
        try:
            step = torch.linalg.solve(H + lam * torch.diag(torch.diag(H) + 1e-12), -J.T @ r)
        except RuntimeError:
            break
        rn = resid(p + step)
        if torch.isfinite(rn).all() and float(rn.norm()) < float(r.norm()):
            p, r, lam = p + step, rn, lam * 0.3
        else:
            lam *= 10
            if lam > 1e8:
                break
    new = _with_thetas(terms, _split_thetas(terms, p))
    r, c = lstsq_resid(design(new, Z, X), y)
    if not torch.isfinite(r).all():
        return terms, float("inf"), c
    return new, float(r.norm()), c


COMPLEXITY = [0.1]   # generic Occam penalty (log-residual units) per constant; set per system from the data size


def set_complexity(n_points):
    """Information-criterion scale: 2 log(n) / n per fitted constant (4x the BIC weight), at least 0.1.
    Generic, not history-derived."""
    COMPLEXITY[0] = max(2.0 * math.log(max(n_points, 2)) / max(n_points, 2), 0.1)


def term_cost(t, prior):
    return prior(t.key()) + COMPLEXITY[0] * (1 + t.nparams())


def screen_component(y, Z, X, cands, prior, tol, beam=6, max_terms=12, budget=None, rng=None, seeds=()):
    """Beam search over supports on derivative data. Score = log(relative residual) + complexity + memory penalty.
    Stops when a support reaches `tol` (the estimated derivative-error level). Returns ranked supports."""
    ny = float(y.norm()) + 1e-300
    vecs = design_fast(cands, Z, X)
    ok = torch.isfinite(vecs).all(0) & (vecs.norm(dim=0) > 1e-12)
    idx_ok = torch.nonzero(ok).squeeze(1).tolist()
    if budget is not None and len(idx_ok) > budget:
        order = sorted(idx_ok, key=lambda i: prior(cands[i].key()) + (rng.random() * 1e-6 if rng is not None else 0))
        idx_ok = order[:budget]
    if not idx_ok:
        return [([], 1.0)]
    V = vecs[:, idx_ok]
    V = V / V.norm(dim=0, keepdim=True)
    pen = torch.tensor([term_cost(cands[i], prior) for i in idx_ok])
    beams = [([], y.clone(), 1.0)]
    for sd in seeds:                       # warm start: supports found by a simpler pass, and their prefixes
        for k in range(1, len(sd) + 1):
            r0, _ = lstsq_resid(design(sd[:k], Z, X), y)
            beams.append((list(sd[:k]), r0, float(r0.norm()) / ny))
    finished = []
    for _ in range(max_terms):
        expansions = []
        for support, r, _ in beams:
            if support:
                Q = torch.linalg.qr(design(support, Z, X))[0]
                Vp = V - Q @ (Q.T @ V)
            else:
                Vp = V
            nrm = (Vp * Vp).sum(0)
            gain = (Vp.T @ r) ** 2 / torch.clamp(nrm, min=1e-30)
            gain = torch.where(nrm > 1e-20, gain, torch.zeros_like(gain))
            new_r2 = torch.clamp(float(r @ r) - gain, min=0.0)
            score = 0.5 * torch.log(new_r2 / ny ** 2 + 1e-40) + pen
            have = {t.sig() for t in support}
            n_exp = 0
            for k in torch.argsort(score)[: 3 * beam].tolist():
                term = cands[idx_ok[k]]
                if term.sig() in have:
                    continue
                terms, res, _ = refine_support(support + [term], Z, X, y)
                if math.isfinite(res) and not any(degenerate(t, Z, X) for t in terms):
                    expansions.append((terms, res / ny))
                    n_exp += 1
                if n_exp >= beam:
                    break
        if not expansions:
            break
        expansions.sort(key=lambda e: math.log(e[1] + 1e-40) + sum(term_cost(t, prior) for t in e[0]))
        seen, beams = set(), []
        for terms, rel in expansions:
            s = tuple(sorted(str(t.sig()) for t in terms))
            if s in seen:
                continue
            seen.add(s)
            r, _ = lstsq_resid(design(terms, Z, X), y)
            beams.append((terms, r, rel))
            if rel <= tol:
                finished.append((terms, rel))
            if len(beams) >= beam:
                break
        if finished:
            break
    have_f = [f[0] for f in finished]
    finished += [(b[0], b[2]) for b in beams if all(b[0] is not f for f in have_f)]
    finished.sort(key=lambda e: (e[1] > tol, len(e[0]), e[1]))
    return finished[:beam]


# ---------------- the model and its own integrator ----------------

def rk8_torch(f, z, nsub):
    h = 1.0 / nsub  # time in units of the sampling interval
    for _ in range(nsub):
        K = []
        for i in range(len(STAGES)):
            zi = z if i == 0 else z + h * sum(a * K[j] for j, a in STAGES[i])
            K.append(f(zi))
        z = z + h * sum(b * K[i] for i, b in BW)
    return z


class Model:
    """Expression network g(z) = dz/dtau in scaled coordinates (tau = t / sampling interval)."""

    def __init__(self, supports, coefs, mu, sig):
        self.supports = [list(s) for s in supports]
        self.coefs = [torch.as_tensor(c, dtype=torch.float64).clone() for c in coefs]
        self.mu, self.sig = mu, sig

    def copy(self):
        return Model([list(s) for s in self.supports], [c.clone() for c in self.coefs], self.mu, self.sig)

    def params(self):
        p = []
        for terms, c in zip(self.supports, self.coefs):
            p += list(c.tolist())
            for t in terms:
                for f in t.factors:
                    p += list(f.theta)
        return torch.tensor(p)

    def unpack(self, p):
        k, out = 0, []
        for terms, c in zip(self.supports, self.coefs):
            cc = p[k:k + len(c)]; k += len(c)
            thetas = []
            for t in terms:
                tt = []
                for f in t.factors:
                    n = len(f.theta); tt.append(p[k:k + n]); k += n
                thetas.append(tt)
            out.append((cc, thetas))
        return out

    def coef_index(self):
        """(component, term index) -> position of that term's linear coefficient in params()."""
        k, out = 0, {}
        for i, (terms, c) in enumerate(zip(self.supports, self.coefs)):
            for a in range(len(c)):
                out[(i, a)] = k + a
            k += len(c) + sum(len(f.theta) for t in terms for f in t.factors)
        return out

    def g(self, z, p=None):
        p = self.params() if p is None else p
        X = z * self.sig + self.mu
        cols = []
        for terms, (cc, thetas) in zip(self.supports, self.unpack(p)):
            if terms:
                cols.append(design(terms, z, X, thetas) @ cc)
            else:
                cols.append(torch.zeros(z.shape[:-1], dtype=z.dtype))
        return torch.stack(cols, -1)

    def set_params(self, p):
        new_s, new_c = [], []
        for terms, (cc, thetas) in zip(self.supports, self.unpack(p)):
            new_c.append(cc.detach().clone())
            new_s.append(_with_thetas(terms, thetas))
        self.supports, self.coefs = new_s, new_c

    def n_terms(self):
        return [len(s) for s in self.supports]

    def describe(self):
        return [[{"mono": list(t.mono), "factors": [[f.family, f.var, list(f.theta)] for f in t.factors], "level": t.level,
                  "coef": float(c)} for t, c in zip(terms, cc)] for terms, cc in zip(self.supports, self.coefs)]


def model_from_description(desc, mu, sig):
    sups, cfs = [], []
    for comp in desc:
        sups.append([Term(tuple(t["mono"]), tuple(Factor(f[0], f[1], tuple(f[2])) for f in t["factors"]), t["level"]) for t in comp])
        cfs.append(torch.tensor([t["coef"] for t in comp]))
    return Model(sups, cfs, torch.as_tensor(mu), torch.as_tensor(sig))


def self_nsub(model, z, tol=1e-14, max_nsub=64):
    """The model's own substep count (no simulator): the harness rule in scaled coordinates (scale 1).
    Returns None if the model's one-step map is not finite (an unusable candidate)."""
    n, last = 1, float("inf")
    with torch.no_grad():
        prev = rk8_torch(model.g, z, n)
        if not torch.isfinite(prev).all():
            return None
        while n < max_nsub:
            nxt = rk8_torch(model.g, z, 2 * n)
            diff = float((nxt - prev).abs().max())
            if not math.isfinite(diff):
                return None
            if diff <= tol or (diff <= 1e-13 and diff > last / 8):
                return n
            n, prev, last = 2 * n, nxt, diff
    return max_nsub


# ---------------- polishing and stepwise refinement on the flow map ----------------

def polish(model, z0, z1, iters=30, nsub=None):
    """Levenberg-Marquardt on one-step residuals rk8(g, z0) - z1 over all training pairs (float64)."""
    if nsub is None:
        nsub = self_nsub(model, z0[: min(len(z0), 200)], max_nsub=16)   # provisional; final count re-chosen below
        if nsub is None:
            return model, 1, float("inf"), None
    p = model.params()
    f_res = lambda q: (rk8_torch(lambda zz: model.g(zz, q), z0, nsub) - z1).reshape(-1)
    r = f_res(p)
    if not torch.isfinite(r).all():
        return model, nsub, float("inf"), None
    lam, J = 1e-6, None
    for _ in range(iters):
        J = torch.func.jacfwd(f_res)(p)
        if not torch.all(torch.isfinite(J)):
            break
        Hm = J.T @ J
        g = J.T @ r
        try:
            step = torch.linalg.solve(Hm + lam * torch.diag(torch.diag(Hm) + 1e-30), -g)
        except RuntimeError:
            lam *= 10
            continue
        rn = f_res(p + step)
        if torch.isfinite(rn).all() and float(rn.norm()) <= float(r.norm()):
            done = float(step.abs().max()) <= 1e-14 * (float(p.abs().max()) + 1e-30) or float(rn.abs().max()) < 1e-15
            p, r, lam = p + step, rn, max(lam * 0.1, 1e-12)
            if done:
                break
        else:
            lam *= 10
            if lam > 1e10:
                break
    model.set_params(p)
    nsub2 = self_nsub(model, z0[: min(len(z0), 200)])
    if nsub2 is None:
        return model, nsub, float("inf"), None
    if nsub2 != nsub:
        nsub = nsub2
        r = f_res(p)
    return model, nsub, float(r.abs().max()), J


def _removal_costs(model, J):
    """Linearized RMS cost of removing each term (others re-optimized): sqrt(c^2 / Cov_cc / n_res)."""
    p = model.params()
    try:
        Cov = torch.linalg.pinv(J.T @ J, rtol=1e-15)
    except RuntimeError:
        return None
    n_res = J.shape[0]
    out = []
    for (i, a), k in model.coef_index().items():
        cost = float(p[k] ** 2 / Cov[k, k]) if Cov[k, k] > 0 else 0.0
        out.append((math.sqrt(max(cost, 0.0) / n_res), i, a))
    return sorted(out)


def _without(model, removals):
    trial = model.copy()
    for i, a in sorted(removals, reverse=True):
        del trial.supports[i][a]
        trial.coefs[i] = torch.cat([trial.coefs[i][:a], trial.coefs[i][a + 1:]])
    return trial


def polish_clean(model, z0, z1, Zc, Xc, iters=30):
    """Polish; if any term has drifted into a degenerate form, replace it by the monomials it imitates and polish
    again (at most twice)."""
    model, nsub, err, J = polish(model, z0, z1, iters=iters)
    for _ in range(2):
        model, changed = simplify_degenerate(model, Zc, Xc)
        if not changed:
            break
        model, nsub, err, J = polish(model, z0, z1, iters=iters)
    return model, nsub, err, J


def prune(model, z0, z1, nsub, err, J, floor=1e-14, max_rounds=12):
    """Remove terms whose (linearized, others re-optimized) removal cost keeps the residual at the floor.
    First all such terms at once; if that breaks the fit, one at a time (cheapest first)."""
    for _ in range(max_rounds):
        if J is None or not math.isfinite(err):
            break
        costs = _removal_costs(model, J)
        if not costs:
            break
        thr = max(floor, 0.1 * err)
        cheap = [(i, a) for c, i, a in costs if c <= thr]
        if not cheap:
            break
        accepted = False
        tries = [cheap] if len(cheap) > 1 else []
        tries.append([cheap[0]])
        for rem in tries:
            trial, nsub_t, err_t, J_t = polish(_without(model, rem), z0, z1, iters=15)
            if math.isfinite(err_t) and err_t <= max(10 * err, 1e-13):
                model, nsub, err, J = trial, nsub_t, err_t, J_t
                accepted = True
                break
        if not accepted:
            break
    return model, nsub, err, J


def stage_adjoints(model, z0, z1, nsub):
    """One reverse pass: every RK8 stage state Z_l and the adjoint dL/dK_l of its stage derivative, for
    L = 0.5 * ||rk8(g, z0) - z1||^2. Then dL/dc for ANY added term phi (coefficient c = 0) is sum_l adj_l . phi(Z_l)."""
    p = model.params().detach()
    leaves, states = [], []

    def f(zz):
        e = torch.zeros(zz.shape, dtype=zz.dtype, requires_grad=True)
        leaves.append(e); states.append(zz.detach())
        return model.g(zz, p) + e

    loss = 0.5 * ((rk8_torch(f, z0, nsub) - z1) ** 2).sum()
    adj = torch.autograd.grad(loss, leaves)
    return torch.cat(states, 0), torch.cat([a.detach() for a in adj], 0)


def flowmap_add(model, z0, z1, nsub, cands_per_comp, Zc, Xc, prior, n_pre=48, n_add=1, verbose=False):
    """Add the candidate term with the largest exact flow-map gain on the current one-step residual.
    Prefilter: the exact flow-map loss gradient for ALL candidates from one reverse pass (stage adjoints), normalized
    by each candidate's scale; the top `n_pre` then get exact forward-mode sensitivities and projected gains."""
    base_p = model.params().detach()
    d = z0.shape[-1]
    Zs, adj = stage_adjoints(model, z0, z1, nsub)
    Xs = Zs * model.sig + model.mu
    scored = []
    for i, cands in enumerate(cands_per_comp):
        if not cands:
            continue
        have = {t.sig() for t in model.supports[i]}
        pool_i = [t for t in cands if t.sig() not in have]
        if not pool_i:
            continue
        V = design_fast(pool_i, Zs, Xs)
        ok = torch.isfinite(V).all(0)
        V = torch.where(ok, V, torch.zeros_like(V))
        scale = V.pow(2).mean(0).sqrt()
        val = (V.T @ adj[:, i]).abs() / torch.clamp(scale, min=1e-30)
        val = torch.where(ok & (scale > 1e-12) & torch.isfinite(val), val, torch.zeros_like(val))
        scored += [(float(val[k]), i, pool_i[k]) for k in range(len(pool_i)) if val[k] > 0]
    scored.sort(key=lambda e: -e[0])
    pool = [(i, t) for _, i, t in scored[:n_pre]]
    if not pool:
        return None
    pools = [[t for i2, t in pool if i2 == i] for i in range(d)]

    def phi(zz, cvec):
        X = zz * model.sig + model.mu
        out = model.g(zz, base_p)
        cols, k = [], 0
        for i in range(d):
            if pools[i]:
                cols.append(design_fast(pools[i], zz, X) @ cvec[k:k + len(pools[i])]); k += len(pools[i])
            else:
                cols.append(torch.zeros(zz.shape[:-1], dtype=zz.dtype))
        return out + torch.stack(cols, -1)

    order = [(i, t) for i in range(d) for t in pools[i]]
    r = (rk8_torch(lambda zz: model.g(zz, base_p), z0, nsub) - z1).reshape(-1).detach()
    S = torch.func.jacfwd(lambda cvec: rk8_torch(lambda zz: phi(zz, cvec), z0, nsub).reshape(-1))(torch.zeros(len(order)))
    ok = torch.isfinite(S).all(0)
    if not ok.any():
        return None
    S = torch.where(ok, S, torch.zeros_like(S))
    Jm = torch.func.jacfwd(lambda q: rk8_torch(lambda zz: model.g(zz, q), z0, nsub).reshape(-1))(base_p)
    Q = torch.linalg.qr(Jm)[0] if Jm.shape[1] else torch.zeros(len(r), 0)
    Sp = S - Q @ (Q.T @ S)
    rp = r - Q @ (Q.T @ r)
    nrm = (Sp * Sp).sum(0)
    gain = torch.where(ok & (nrm > 1e-28), (Sp.T @ rp) ** 2 / torch.clamp(nrm, min=1e-300), torch.zeros(len(order)))
    cost = torch.tensor([term_cost(t, prior) for _, t in order])
    score = 0.5 * torch.log(torch.clamp(float(rp @ rp) - gain, min=1e-300)) + cost
    picks = torch.argsort(score)[:n_add].tolist()
    new = model.copy()
    for k in picks:
        i, t = order[k]
        new.supports[i] = new.supports[i] + [t]
        new.coefs[i] = torch.cat([new.coefs[i], torch.zeros(1)])
        if verbose:
            print(f"     add comp {i}: {t.sig()} gain {float(gain[k]):.2e} of {float(rp @ rp):.2e}", flush=True)
    return new


# ---------------- knowledge (memory / guide / statistics) ----------------

@dataclass
class Knowledge:
    """What the learner brings to a new system. Empty = learning from scratch."""
    template_counts: dict = field(default_factory=dict)   # memory: template key -> count
    promoted: tuple = ()                                   # memory: composite template keys promoted to level 1
    influence: object = None                               # (d, d) probabilities: component i depends on variable j
    n_terms: object = None                                 # (d,) predicted number of terms per component
    level: object = None                                   # (d,) predicted max level per component
    label: str = "scratch"

    def prior_fn(self, alpha=1.0):
        total = sum(self.template_counts.values())
        if total == 0:
            return lambda key: 0.0
        nkeys = len(self.template_counts)
        def pen(key):
            p = (self.template_counts.get(key, 0) + alpha) / (total + alpha * (nkeys + 1))
            return -0.1 * math.log(p)
        return pen


def restrict(cands, allowed_vars, max_level):
    return [t for t in cands if t.variables() <= allowed_vars and t.level <= max_level]


# ---------------- the learner ----------------

FD13 = torch.tensor([1 / 5544, -1 / 385, 1 / 56, -5 / 63, 15 / 56, -6 / 7, 0.0, 6 / 7, -15 / 56, 5 / 63, -1 / 56, 1 / 385, -1 / 5544])


def deriv_noise(Ztr):
    """9-point derivatives at burst interiors, and a per-component estimate of their error from the data:
    RMS difference between the 9-point and the 13-point (12th-order) estimate at each burst's centre."""
    L = Ztr.shape[1]
    cs = range(4, L - 4)
    G9 = torch.stack([sum(FD9[k] * Ztr[:, c - 4 + k] for k in range(9)) for c in cs], 1).reshape(-1, Ztr.shape[-1])
    if L >= 13:
        c = L // 2
        g9 = sum(FD9[k] * Ztr[:, c - 4 + k] for k in range(9))
        g13 = sum(FD13[k] * Ztr[:, c - 6 + k] for k in range(13))
        est = ((g9 - g13) ** 2).mean(0).sqrt()
    else:
        G7 = torch.stack([sum(FD7[k] * Ztr[:, c - 4 + k] for k in range(9)) for c in cs], 1).reshape(-1, Ztr.shape[-1])
        est = ((G7 - G9) ** 2).mean(0).sqrt()
    return G9, est


def learn(bursts, knowledge=None, val_frac=0.25, seed=0, time_limit=600, verbose=False, max_terms=12,
          stepwise_rounds=8, levels_plan=((0,), (1,), (1, 2), (1, 2, 3)), composite_budget=4000):
    """bursts: (M, L, d) noise-free states at unit sampling interval. Returns a result dict."""
    t_start = time.time()
    kn = knowledge or Knowledge()
    rng = np.random.default_rng(seed)
    Bt = torch.as_tensor(np.asarray(bursts, dtype=np.float64))
    M, L, d = Bt.shape
    nval = max(1, int(round(val_frac * M)))
    perm = rng.permutation(M)
    val_idx, tr_idx = perm[:nval], (perm[nval:] if M > 1 else perm)
    allx = Bt.reshape(-1, d)
    mu, sig = allx.mean(0), allx.std(0) + 1e-300
    mu = torch.where(mu.abs() > CENTER_RATIO * sig, mu, torch.zeros_like(mu))   # sparse monomials unless badly offset
    Ztr = (Bt[tr_idx] - mu) / sig
    Zva = (Bt[val_idx] - mu) / sig
    G, noise = deriv_noise(Ztr)
    Zc = torch.stack([Ztr[:, c] for c in range(4, L - 4)], 1).reshape(-1, d)
    Xc = Zc * sig + mu
    z0, z1 = Ztr[:, :-1].reshape(-1, d), Ztr[:, 1:].reshape(-1, d)
    zv0, zv1 = Zva[:, :-1].reshape(-1, d), Zva[:, 1:].reshape(-1, d)
    prior = kn.prior_fn()
    set_complexity(len(Zc))
    report = {"phases": [], "polishes": 0, "knowledge": kn.label}
    cand_cache = {}
    best = None

    def out_of_time():
        return time.time() - t_start > time_limit

    def cands_for(levels):
        if levels not in cand_cache:
            cand_cache[levels] = candidates(d, Zc, Xc, levels, promoted=tuple(kn.promoted), G=G)
        return cand_cache[levels]

    # search passes: guided (restricted) first if guidance is available, then the full search
    passes = []
    if kn.influence is not None:
        infl = np.asarray(kn.influence)
        allowed = [set(np.nonzero(infl[i] >= 0.5)[0].tolist()) or {int(np.argmax(infl[i]))} for i in range(d)]
        lv = [int(kn.level[i]) if kn.level is not None else 3 for i in range(d)]
        nt = [int(kn.n_terms[i]) if kn.n_terms is not None else max_terms for i in range(d)]
        for levels in levels_plan:
            passes.append(("guided", levels, allowed, lv, nt))
    for levels in levels_plan:
        passes.append(("full", levels, None, None, None))

    tried = set()
    seeds = [[] for _ in range(d)]
    for kind, levels, allowed, lv, nt in passes:
        if out_of_time():
            break
        if kind == "guided" and max(levels) > max(lv):
            continue
        cands_all = cands_for(levels)
        per_comp = [cands_all if allowed is None else restrict(cands_all, allowed[i], lv[i]) for i in range(d)]
        sig_pass = (kind, levels, tuple(len(c) for c in per_comp))
        if sig_pass[1:] in tried:
            continue
        tried.add(sig_pass[1:])
        # 1. derivative screening per component -> ranked supports
        ranked = []
        for i in range(d):
            y = G[:, i]
            tol = max(3.0 * float(noise[i]) / (float(y.norm()) / math.sqrt(len(y)) + 1e-300), 1e-12)
            mt = max_terms if nt is None else min(max_terms, nt[i] + 3)
            budget = composite_budget if max(levels) > 1 else None
            sd = [t for t in seeds[i] if t.variables() <= (allowed[i] if allowed is not None else set(range(d)))]
            res = screen_component(y, Zc, Xc, per_comp[i], prior, tol, max_terms=mt, budget=budget, rng=rng,
                                   seeds=[sd] if sd else ())
            ranked.append(res)
            if verbose:
                print(f"  [{kind} L{max(levels)}] comp {i}: cands {len(per_comp[i])} tol {tol:.1e} best rel {res[0][1]:.2e} "
                      f"terms {len(res[0][0])} ({time.time() - t_start:.1f}s)", flush=True)
            if out_of_time():
                break
        if len(ranked) < d:
            break
        if levels == (0,):
            seeds = [ranked[i][0][0] for i in range(d)]
            far = any(ranked[i][0][1] > 30 * max(3.0 * float(noise[i]) / (float(G[:, i].norm()) / math.sqrt(len(G)) + 1e-300), 1e-12)
                      for i in range(d))
            if far:                         # polynomial terms alone clearly cannot explain the data: next pass
                report["phases"].append(sig_pass[:2])
                continue
        # 2.-3. polish combinations of ranked supports, with stepwise refinement on the flow map
        combos = [[0] * d] + [[int(j == i) * a for j in range(d)] for a in range(1, 4) for i in range(d)]
        for combo in combos:
            if out_of_time():
                break
            if any(a >= len(ranked[i]) for i, a in enumerate(combo)):
                continue
            sups, cfs = [], []
            for i, a in enumerate(combo):
                terms = ranked[i][a][0]
                _, c = lstsq_resid(design(terms, Zc, Xc), G[:, i])
                sups.append(list(terms)); cfs.append(c)
            model = Model(sups, cfs, mu, sig)
            try:
                t_c = time.time()
                model, nsub, err, J = polish_clean(model, z0, z1, Zc, Xc)
                report["polishes"] += 1
                if verbose:
                    print(f"   polished {model.n_terms()} nsub {nsub} train {err:.2e} ({time.time() - t_c:.1f}s)", flush=True)
                if not math.isfinite(err):
                    continue
                model, nsub, err, J = prune(model, z0, z1, nsub, err, J)
                if verbose:
                    print(f"   start {model.n_terms()} train {err:.2e} (pruned; {time.time() - t_c:.1f}s)", flush=True)
                    for ci, terms in enumerate(model.supports):
                        print(f"      c{ci}: " + " ".join(f"{t.mono}{[(f.family, f.var, tuple(round(v, 3) for v in f.theta)) for f in t.factors] or ''}"
                                                       for t in terms), flush=True)
                for _ in range(stepwise_rounds):
                    if err <= VAL_TOL or out_of_time():
                        break
                    new = flowmap_add(model, z0, z1, nsub, per_comp, Zc, Xc, prior, verbose=verbose)
                    if new is None:
                        break
                    new, nsub_n, err_n, J_n = polish_clean(new, z0, z1, Zc, Xc)
                    report["polishes"] += 1
                    if verbose:
                        print(f"     -> {new.n_terms()} train {err_n:.2e}", flush=True)
                    if not (math.isfinite(err_n) and err_n < 0.5 * err):
                        break
                    model, nsub, err, J = prune(new, z0, z1, nsub_n, err_n, J_n)
                with torch.no_grad():
                    val_err = float((rk8_torch(model.g, zv0, nsub) - zv1).abs().max())
            except (RuntimeError, ValueError) as e:  # numerical failure of a candidate
                if verbose:
                    print("   candidate failed:", str(e)[:120], flush=True)
                continue
            if not math.isfinite(val_err):
                continue
            if verbose:
                print(f"   polish {kind} L{max(levels)} {combo}: terms {model.n_terms()} nsub {nsub} train {err:.2e} "
                      f"val {val_err:.2e} ({time.time() - t_start:.1f}s)", flush=True)
            if best is None or val_err < best["val_err_scaled"]:
                best = {"model": model, "nsub": nsub, "val_err_scaled": val_err, "train_err_scaled": err,
                        "pass": kind, "levels": max(levels)}
            if val_err <= VAL_TOL:
                break
        report["phases"].append(sig_pass[:2])
        if best is not None and best["val_err_scaled"] <= VAL_TOL:
            break
    if best is not None and best["val_err_scaled"] <= VAL_TOL:
        # model chosen by internal validation; its constants are then refit on ALL observed bursts (more coverage)
        za0 = torch.cat([z0, zv0]); za1 = torch.cat([z1, zv1])
        try:
            m_all, nsub_all, err_all, _ = polish(best["model"].copy(), za0, za1, iters=10)
            if math.isfinite(err_all) and err_all <= VAL_TOL:
                best.update({"model": m_all, "nsub": nsub_all, "all_err_scaled": err_all})
        except (RuntimeError, ValueError):
            pass
    if best is not None:
        # the artifact's own substep count: self-convergence on ALL observed states, then one more doubling as the
        # safety margin of an 8th-order method (truncation error / 256); no simulator involved
        za = torch.cat([Ztr.reshape(-1, d), Zva.reshape(-1, d)])
        n_conv = self_nsub(best["model"], za, max_nsub=128)
        if n_conv is not None:
            best["nsub"] = 2 * n_conv
    elapsed = time.time() - t_start
    if best is None:
        return {"status": "no_model", "seconds": elapsed, **report}
    best["status"] = "accepted" if best["val_err_scaled"] <= VAL_TOL else "rejected"
    best["seconds"] = elapsed
    best["n_terms"] = best["model"].n_terms()
    best.update(report)
    return best


def artifact(result, dt):
    """Numpy-facing artifact: learned vector field in original units and its own substep count."""
    model = result["model"]
    sig, mu = model.sig.numpy(), model.mu.numpy()
    p = model.params()
    def f(x):
        z = torch.as_tensor((np.asarray(x) - mu) / sig)
        with torch.no_grad():
            return (model.g(z, p).numpy() * sig) / dt
    return f, result["nsub"]
