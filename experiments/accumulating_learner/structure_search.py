"""Trial 28: search over equation structures. Moves add, remove or swap one term in one equation; a candidate structure
is fitted by a swappable fitter; priorities over moves come from a prior function, data-only for the baseline and a
network for the with-history learner. The search always explores: every move keeps an exploration bonus and a fixed
share of moves is drawn at random. Effort (structures fitted, fitting starts, time) is recorded from the start.

Pieces of the vocabulary (fixed 2 November, docs/research-log.md):
  Vocab28           term types: products of variables (degree <= 3); single-variable pieces (reciprocal, Hill-type
                    with a free exponent, exp, tanh, sin, |.|, free-exponent power), alone or times one variable;
                    two-variable weighted-sum reciprocal and Hill pieces, alone or times one variable; products of two
                    single-variable switches. Each piece type comes with a grid of starting values for its constants.
  MultiStartFitter  constants of a candidate: many starting values (grid, warm starts from the parent structure, random
                    restarts), Levenberg-Marquardt with the linear coefficients projected out, best kept. Swappable:
                    any object with the same fit() can replace it (e.g. a learned initializer).
  data_prior        baseline priorities from this system's data only: evidence for adding a term, cost of removing one.
  search_equation   tree search over one equation's term set (PUCT with an exploration share, transpositions).
  learn28           per-equation searches, assembly, exact fitting on the step-to-step map, held-back check, ties.
"""

import itertools
import math
import time

import numpy as np
import torch

import discovery_learner as DL
import practice_net as PN
import ties as TI

PIECES = ("recip", "hill", "exp", "tanh", "sin", "abs", "pow")
PAIR_PIECES = ("recip", "hill")
PROD_PIECES = ("recip", "hill")
MAX_TERMS = 8
PRECISE_GATE = 1e-3    # precise stage only for candidates within 0.1% on the rate data
PRECISE_FLOOR = 1e6    # ... and within this factor of the rate data's noise floor
TAU_EXACT = 1e-9       # plausibly exact: step-integral error <= max(TAU_EXACT, TAU_FLOOR x the equation's relative
TAU_FLOOR = 30.0       # rate-data noise floor) on training and held-back bursts (calibrated on practice systems:
                       # 60 of 61 correct structures fitted to their rate floor pass; the one miss is data-limited)
ASSEMBLY_CAP = 300.0   # seconds for assembling and exactly fitting whole models
PRECISE_BASINS = 4     # polishing starts per piece in the precise stage (2 in the ordinary fit)
SNAP_POS = {"hill": 1, "pow": 0}   # exponent position of the Hill-type and power pieces (snapped to integers)
LOG_POS = {"hill": 0}              # positive scale constant moved on a log scale while polishing


# ---------------------------------------------------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------------------------------------------------

class Vocab28:
    """Term types of the search for d variables, with start grids for their constants from the system's own data
    (scaled states Zc, physical states Xc). A token is hashable; term(token, thetas) builds the learner Term.
    Single-variable Hill-type and power pieces act on the physical variable (zero point at its zero, as logged for the
    vocabulary), so they stay available when the learner's scaling centres a variable."""

    SINGLE = {"recip": "crecip", "hill": "hill", "exp": "cexp", "tanh": "ctanh", "sin": "csin", "abs": "cabs", "pow": "pow"}
    PAIR = {"recip": "crecip", "hill": "chill"}

    def __init__(self, d, Zc, Xc=None):
        self.d = d
        Xc = Zc if Xc is None else Xc
        lo, hi = Zc.min(0).values.numpy(), Zc.max(0).values.numpy()
        span = np.maximum(hi - lo, 1e-9)
        zmax = np.maximum(np.abs(Zc).max(0).values.numpy(), 1e-9)
        xmax = np.maximum(np.abs(Xc).max(0).values.numpy(), 1e-300)
        q = {j: np.quantile(Zc[:, j].numpy(), [0.1, 0.3, 0.5, 0.7, 0.9]) for j in range(d)}
        self.positive = [bool((Zc[:, j] > 0).all()) for j in range(d)]          # weighted-sum Hill pieces (scaled)
        self.positive_x = [bool((Xc[:, j] > 0).all()) for j in range(d)]        # Hill-type and power pieces (physical)
        mults = [tuple([0] * d)] + [tuple(int(k == j) for k in range(d)) for j in range(d)]
        toks = [("mono", tuple(m)) for m in DL.monomials(d, 3)]
        for fam in PIECES:
            for j in range(d):
                if fam in ("hill", "pow") and not self.positive_x[j]:
                    continue
                toks += [("piece", fam, j, m) for m in mults]
        for fam in PAIR_PIECES:
            for j, k in itertools.combinations(range(d), 2):
                if fam == "hill" and not (self.positive[j] and self.positive[k]):
                    continue
                toks += [("pair", fam, (j, k), m) for m in mults]
        for fam in PROD_PIECES:
            for j, k in itertools.combinations(range(d), 2):
                if fam == "hill" and not (self.positive_x[j] and self.positive_x[k]):
                    continue
                toks += [("prod", fam, (j, k))]
        self.tokens = toks
        self.index = {t: i for i, t in enumerate(toks)}
        # start grids per piece family and variable (thetas of one factor)
        g = {}
        for j in range(d):
            poles = np.concatenate([lo[j] - np.geomspace(0.02, 50, 8) * span[j], hi[j] + np.geomspace(0.02, 50, 8) * span[j]])
            g[("recip", j)] = [(1.0, float(-p)) for p in poles]
            g[("hill", j)] = [(float((h * xmax[j]) ** n), n) for h in np.geomspace(0.03, 30, 12) for n in (1.0, 2.0, 3.0, 4.0)]
            g[("exp", j)] = [(float(s / zmax[j]), 0.0) for s in (-3.0, -1.0, -0.3, 0.3, 1.0, 3.0)]
            g[("tanh", j)] = [(float(b / span[j]), float(-b / span[j] * c)) for b in (1.0, 3.0, 10.0) for c in q[j][1:4]]
            g[("sin", j)] = [(float(2 * math.pi * w / span[j]), ph) for w in (0.5, 1.0, 2.0, 4.0) for ph in (0.0, math.pi / 2)]
            g[("abs", j)] = [(1.0, float(-c)) for c in q[j]]
            g[("pow", j)] = [(p,) for p in (0.5, 1.5, 2.5, -0.5, -1.0, -2.0)]
        for j, k in itertools.combinations(range(d), 2):
            ws = [float(r * span[j] / span[k]) for r in (0.3, 1.0, 3.0)]
            for w in ws:
                ulo, uhi = lo[j] + w * lo[k], hi[j] + w * hi[k]
                us = max(uhi - ulo, 1e-9)
                poles = np.concatenate([ulo - np.geomspace(0.05, 20, 4) * us, uhi + np.geomspace(0.05, 20, 4) * us])
                g.setdefault(("pair_recip", j, k), []).extend([(w, 1.0, float(-p)) for p in poles])
                g.setdefault(("pair_hill", j, k), []).extend(
                    [(w, float(1.0 / (h * max(abs(uhi), 1e-9))), 0.0, n) for h in (0.1, 1.0, 10.0) for n in (1.0, 2.0)])
        self.grid = g

    def __len__(self):
        return len(self.tokens)

    def factor_families(self, tok):
        """[(learner family, var, grid key)] for the factors of a token."""
        if tok[0] == "mono":
            return []
        if tok[0] == "piece":
            _, fam, j, _ = tok
            return [(self.SINGLE[fam], j, (fam, j))]
        if tok[0] == "pair":
            _, fam, (j, k), _ = tok
            return [(self.PAIR[fam], (j, k), ("pair_" + fam, j, k))]
        _, fam, (j, k) = tok
        return [(self.SINGLE[fam], j, (fam, j)), (self.SINGLE[fam], k, (fam, k))]

    def mono_of(self, tok):
        if tok[0] in ("mono",):
            return tok[1]
        if tok[0] in ("piece", "pair"):
            return tok[3]
        return tuple([0] * self.d)

    def term(self, tok, thetas):
        """Learner Term for a token with constants thetas (one tuple per factor)."""
        facs = []
        for (fam, var, key), th in zip(self.factor_families(tok), thetas):
            meta = ("zsnap",) if key[0] == "pair_hill" else ()   # weighted-sum Hill: zero point at the sum's zero
            facs.append(DL.Factor(fam, var, tuple(float(v) for v in th), meta))
        return DL.Term(self.mono_of(tok), tuple(facs), 1)

    def starts(self, tok, limit=None):
        """Grid starting values for a token: a list of per-factor theta tuples; with `limit`, each factor's grid is
        subsampled evenly so that the combinations number at most `limit`."""
        keys = [k for _, _, k in self.factor_families(tok)]
        if not keys:
            return [()]
        grids = [self.grid[k] for k in keys]
        if limit is not None:
            per = max(1, int(limit ** (1.0 / len(grids)) + 1e-9))
            grids = [g if len(g) <= per else [g[int(round(a))] for a in np.linspace(0, len(g) - 1, per)] for g in grids]
        if len(grids) == 1:
            return [(t,) for t in grids[0]]
        return [tuple(c) for c in itertools.product(*grids)]


# ---------------------------------------------------------------------------------------------------------------------
# fitting (swappable)
# ---------------------------------------------------------------------------------------------------------------------

class MultiStartFitter:
    """Brute-force fitting of one equation's candidate term set on the rate data: starting values from the vocabulary
    grids (each new piece over its whole grid, pieces shared with the parent structure at their fitted values, plus
    random combinations), Levenberg-Marquardt with linear coefficients projected out (DL.refine_support) from each,
    best kept. Returns (fitted terms, residual norm, coefficients, effort)."""

    def __init__(self, max_starts=32, lm_iters=15, seed=0):
        self.max_starts, self.lm_iters = max_starts, lm_iters
        self.rng = np.random.default_rng(seed)

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0, basins=None):
        t0 = time.perf_counter()
        warm = {t: w for t, w in (warm or {}).items() if not isinstance(w, int)}
        toks = list(toks)
        if not toks:
            return [], float(y.norm()), torch.zeros(0, dtype=torch.float64), {"starts": 0, "seconds": 0.0}
        free = [t for t in toks if vocab.factor_families(t)]
        if not free:
            terms = [vocab.term(t, ()) for t in toks]
            r, c = DL.lstsq_resid(DL.design(terms, Zc, Xc), y)
            return terms, float(r.norm()), c, {"starts": 1, "seconds": time.perf_counter() - t0}
        combos = []
        new = [t for t in free if t not in warm]
        base = {t: warm[t] for t in free if t in warm}
        if len(new) == 1:                                      # the new piece over its grid, others warm
            combos += [{**base, new[0]: s} for s in vocab.starts(new[0])]
        elif not new:
            combos.append(dict(base))
        grids = {t: vocab.starts(t) for t in free}
        while len(combos) < self.max_starts:                   # random combinations (all pieces restarted)
            combos.append({t: grids[t][int(self.rng.integers(len(grids[t])))] for t in free})
        combos = combos[:self.max_starts]
        best = None
        for cmb in combos:
            terms = [vocab.term(t, cmb.get(t, ())) for t in toks]
            try:
                terms, rn, c = DL.refine_support(terms, Zc, Xc, y, iters=self.lm_iters)
            except (RuntimeError, ValueError):
                continue
            if math.isfinite(rn) and (best is None or rn < best[1]):
                best = (terms, rn, c)
        if best is None:
            return [vocab.term(t, vocab.starts(t)[0]) for t in toks], float("inf"), None, \
                {"starts": len(combos), "seconds": time.perf_counter() - t0}
        return best[0], best[1], best[2], {"starts": len(combos), "seconds": time.perf_counter() - t0}


class ScreenPolishFitter(MultiStartFitter):
    """Global search on the nonlinear constants, then polishing: every starting value is screened with one linear
    least-squares fit (coefficients only), and Levenberg-Marquardt runs only from the `n_polish` best. Same interface
    and same starting values as MultiStartFitter, at a fraction of the cost."""

    def __init__(self, max_starts=48, lm_iters=10, n_polish=3, seed=0):
        super().__init__(max_starts=max_starts, lm_iters=lm_iters, seed=seed)
        self.n_polish = n_polish

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0, basins=None):
        t0 = time.perf_counter()
        warm = {t: w for t, w in (warm or {}).items() if not isinstance(w, int)}
        toks = list(toks)
        if not toks:
            return [], float(y.norm()), torch.zeros(0, dtype=torch.float64), {"starts": 0, "polished": 0, "seconds": 0.0}
        free = [t for t in toks if vocab.factor_families(t)]
        if not free:
            terms = [vocab.term(t, ()) for t in toks]
            r, c = DL.lstsq_resid(DL.design(terms, Zc, Xc), y)
            return terms, float(r.norm()), c, {"starts": 1, "polished": 0, "seconds": time.perf_counter() - t0}
        new = [t for t in free if t not in warm]
        base = {t: warm[t] for t in free if t in warm}
        combos = [{**base, new[0]: s} for s in vocab.starts(new[0])] if len(new) == 1 else ([dict(base)] if not new else [])
        grids = {t: vocab.starts(t) for t in free}
        while len(combos) < self.max_starts:
            combos.append({t: grids[t][int(self.rng.integers(len(grids[t])))] for t in free})
        combos = combos[:self.max_starts]
        screened = []
        for cmb in combos:
            terms = [vocab.term(t, cmb.get(t, ())) for t in toks]
            with torch.no_grad():
                Phi = DL.design(terms, Zc, Xc)
            if not torch.isfinite(Phi).all():
                continue
            r, _ = DL.lstsq_resid(Phi, y)
            rn = float(r.norm())
            if math.isfinite(rn):
                screened.append((rn, terms))
        screened.sort(key=lambda x: x[0])
        best = None
        for rn0, terms in screened[:self.n_polish]:
            try:
                terms, rn, c = DL.refine_support(terms, Zc, Xc, y, iters=self.lm_iters)
            except (RuntimeError, ValueError):
                continue
            if math.isfinite(rn) and (best is None or rn < best[1]):
                best = (terms, rn, c)
        eff = {"starts": len(combos), "polished": min(len(screened), self.n_polish), "seconds": time.perf_counter() - t0}
        if best is None:
            return [vocab.term(t, vocab.starts(t)[0]) for t in toks], float("inf"), None, eff
        return best[0], best[1], best[2], eff


def _project(Phi, y):
    """Least squares through the SVD, with gelsd's rank cutoff: (residual, coefficients, left basis), or None for an
    unusable design."""
    if Phi.shape[1] == 0:
        return y.clone(), torch.zeros(0, dtype=y.dtype), torch.zeros(len(y), 0, dtype=y.dtype)
    if not bool(torch.isfinite(Phi).all()):
        return None
    try:
        U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
    except RuntimeError:
        return None
    keep = S > S[0] * max(Phi.shape) * 2.2e-16
    U, S, Vh = U[:, keep], S[keep], Vh[keep]
    uty = U.T @ y
    return y - U @ uty, Vh.T @ (uty / S), U


def _vp_lm(terms, column, y, iters=10, tol_gain=1e-6, lam_max=1e4, stop_below=0.0):
    """Levenberg-Marquardt on the free constants of `terms` with the linear coefficients projected out (variable
    projection with Kaufman's Jacobian: a constant moves only its own term's column, so each Jacobian column needs one
    term evaluation, by a finite difference, and one projection). column(term, thetas) gives a term's column for
    per-factor constants. Stops when converged (gain below tol_gain), stuck, or below stop_below.
    Returns (terms, residual norm, coefficients); residual inf for an unusable design."""
    frozen, slots, k = set(DL.frozen_positions(terms)), [], 0
    for m, t in enumerate(terms):                              # (term, factor, position) of each free constant
        for fi, f in enumerate(t.factors):
            for a_ in range(len(f.theta)):
                snapped = "snap" in (f.meta or ()) and SNAP_POS.get(f.family) == a_   # exponent held at an integer
                if k not in frozen and not snapped:
                    slots.append((m, fi, a_))
                k += 1
    logs = [terms[m].factors[fi].family in LOG_POS and LOG_POS[terms[m].factors[fi].family] == a_ for m, fi, a_ in slots]
    th = [[list(f.theta) for f in t.factors] for t in terms]

    def col(m, thm):
        with torch.no_grad():
            return column(terms[m], [torch.tensor(v, dtype=torch.float64) for v in thm])

    cols = [col(m, th[m]) for m in range(len(terms))]
    out = _project(torch.stack(cols, 1), y) if terms else None
    if out is None or not bool(torch.isfinite(out[0]).all()):
        return terms, float("inf"), None
    r, c, U = out
    rn = float(r.norm())
    live = sorted({m for m, _, _ in slots})
    lam = 1e-3
    for _ in range(iters if slots else 0):
        J = torch.empty(len(y), len(slots), dtype=torch.float64)
        for s_, (m, fi, a_) in enumerate(slots):
            v = th[m][fi][a_]
            h = 1e-7 if logs[s_] else 1e-7 * (1.0 + abs(v))   # positive scale constants move on a log scale
            thm = [list(x) for x in th[m]]
            thm[fi][a_] = v * math.exp(h) if logs[s_] else v + h
            dc = (col(m, thm) - cols[m]) / h
            J[:, s_] = -(dc - U @ (U.T @ dc)) * c[m]
        if not bool(torch.isfinite(J).all()):
            break
        H, g = J.T @ J, J.T @ r
        D = torch.diag(torch.diag(H) + 1e-12)
        gain, tries = None, 0
        while lam <= lam_max and tries < 4:                    # a few damping increases; then converged or stuck
            tries += 1
            try:
                step = torch.linalg.solve(H + lam * D, -g)
            except RuntimeError:
                lam *= 10
                continue
            th2 = [[list(x) for x in thm] for thm in th]
            for s_, (m, fi, a_) in enumerate(slots):
                st_ = float(step[s_])
                th2[m][fi][a_] = th2[m][fi][a_] * math.exp(max(min(st_, 30.0), -30.0)) if logs[s_] else th2[m][fi][a_] + st_
            cols2 = [col(m, th2[m]) if m in live else cols[m] for m in range(len(terms))]
            out = _project(torch.stack(cols2, 1), y)
            if out is not None and bool(torch.isfinite(out[0]).all()) and float(out[0].norm()) < rn:
                gain = 1.0 - float(out[0].norm()) / rn
                th, cols, (r, c, U), rn = th2, cols2, out, float(out[0].norm())
                lam = max(lam * 0.3, 1e-9)
                break
            lam *= 10
        if gain is None or gain < tol_gain or rn <= stop_below:
            break
    new = DL._with_thetas(terms, th)
    for a_, b_ in zip(new, terms):                             # keep the grid placement for warm starts
        if hasattr(b_, "grid_index"):
            a_.__dict__["grid_index"] = b_.grid_index
    return new, rn, c


def fast_refine(terms, Z, X, y, iters=10, stop_below=0.0):
    """Levenberg-Marquardt polishing of a candidate's constants on the rate data (variable projection, _vp_lm), stopping
    at the noise floor if given. Returns (terms, residual norm, coefficients)."""
    return _vp_lm(terms, lambda t, th: DL.term_value(t, Z, X, th), y, iters=iters, stop_below=stop_below)


class StepIntegrals:
    """Precise step-by-step form of a set of bursts (noise-free): each burst's states between samples come from the
    degree-(L-1) interpolating polynomial through its L samples (barycentric Lagrange weights, computed once), and each
    step's change z(k+1) - z(k) equals the integral of the right-hand side over the step (8-node Gauss-Legendre).
    Interior steps only (k = 1 .. L-3), away from the interpolation's edge effects."""

    def __init__(self, Zb, mu, sig, nodes=8):
        Zb = torch.as_tensor(Zb, dtype=torch.float64)
        nb, L, d = Zb.shape
        tau = np.arange(L, dtype=float)
        gx, gw = np.polynomial.legendre.leggauss(nodes)
        steps = list(range(1, L - 2))
        tq = np.concatenate([k + 0.5 * (gx + 1) for k in steps])
        bw = np.array([1.0 / np.prod([tau[m] - tau[n] for n in range(L) if n != m]) for m in range(L)])
        Lm = np.zeros((len(tq), L))
        for a, t in enumerate(tq):
            v = bw / (t - tau)
            Lm[a] = v / v.sum()
        Lmat = torch.tensor(Lm)
        self.P = torch.einsum("qm,bmd->bqd", Lmat, Zb).reshape(-1, d)       # states at quadrature nodes
        self.X = self.P * sig + mu
        self.W = torch.tensor(np.tile(0.5 * gw, nb * len(steps)))
        self.dZ = torch.stack([Zb[:, k + 1] - Zb[:, k] for k in steps], 1).reshape(-1, d)
        self.nodes = nodes

    def design(self, terms, thetas=None):
        Phi = DL.design(terms, self.P, self.X, thetas)
        return (Phi * self.W.unsqueeze(1)).reshape(-1, self.nodes, Phi.shape[1]).sum(1)


def snap_exponents(terms, Z, X, y, floor, iters=10, tol=0.25):
    """Exponent snapping, as in the Trial 27 learner: each free Hill-type or power exponent within `tol` of a nonzero
    integer is held at that integer and the other constants are re-polished on the rate data; the snap is kept when the
    residual grows by less than half (or stays within 1.5x the noise floor). Returns (terms, residual, coefficients),
    or None when nothing was snapped."""
    out = _vp_lm(terms, lambda t, th: DL.term_value(t, Z, X, th), y, iters=0)
    cur, res, c = out
    changed = False
    for m, t in enumerate(terms):
        for fi, f in enumerate(t.factors):
            pos = SNAP_POS.get(f.family)
            if pos is None or "snap" in (f.meta or ()):
                continue
            n = cur[m].factors[fi].theta[pos]
            nr = float(round(n))
            if nr == 0.0 or abs(n - nr) >= tol:
                continue
            trial = list(cur)
            fs = list(trial[m].factors)
            th = list(fs[fi].theta); th[pos] = nr
            fs[fi] = DL.Factor(fs[fi].family, fs[fi].var, tuple(th), tuple(fs[fi].meta or ()) + ("snap",))
            trial[m] = DL.Term(trial[m].mono, tuple(fs), trial[m].level)
            if hasattr(cur[m], "grid_index"):
                trial[m].__dict__["grid_index"] = cur[m].grid_index
            t2, r2, c2 = _vp_lm(trial, lambda t_, th_: DL.term_value(t_, Z, X, th_), y, iters=iters, stop_below=1.5 * floor)
            if math.isfinite(r2) and r2 <= max(1.5 * res, 1.5 * floor):
                cur, res, c, changed = t2, r2, c2, True
    return (cur, res, c) if changed else None


def thorough_fit(fitter, vocab, toks, Zc, Xc, y, warm, floor, current=None):
    """The precise stage's fit on the rate data: the fitter with more polishing starts per piece (and shared placements
    for pieces of one family on one variable), then exponent snapping; the better of this and `current` (terms,
    residual, coefficients) is returned."""
    terms, res, coefs, _ = fitter.fit(vocab, toks, Zc, Xc, y, warm, floor=floor, basins=PRECISE_BASINS)
    if current is not None and not (math.isfinite(res) and res < current[1]):
        terms, res, coefs = current
    sn = snap_exponents(terms, Zc, Xc, y, floor)
    return sn if sn is not None else (terms, res, coefs)


def refine_integral(terms, si, i, iters=12):
    """Levenberg-Marquardt (variable projection, _vp_lm) on the step integrals of equation i, to the precision floor.
    Returns (terms, relative residual, coefficients)."""
    y = si.dZ[:, i]
    yn = float(y.norm()) + 1e-300
    new, rn, c = _vp_lm(terms, lambda t, th: si.design([t], [th])[:, 0], y, iters=iters, lam_max=1e6, stop_below=1e-15 * yn)
    return new, rn / yn, c


def heldback_error(terms, coefs, si, i):
    """Relative step-integral prediction error of equation i on the held-back bursts, coefficients fixed."""
    y = si.dZ[:, i]
    with torch.no_grad():
        pred = si.design(terms) @ coefs if terms else torch.zeros_like(y)
    e = float((y - pred).norm()) / (float(y.norm()) + 1e-300)
    return e if math.isfinite(e) else float("inf")


class ColumnBank:
    """Every token's columns at its grid starting values, computed once per system (they do not depend on the
    equation): bank[token] = (columns (n, g), list of per-factor theta tuples)."""

    def __init__(self, vocab, Zc, Xc, max_grid=256):
        self.bank = {}
        for tok in vocab.tokens:
            ths = vocab.starts(tok, limit=max_grid) if vocab.factor_families(tok) else [()]
            cols, keep = [], []
            for th in ths:
                try:
                    with torch.no_grad():
                        v = DL.term_value(vocab.term(tok, th), Zc, Xc)
                except (RuntimeError, ValueError, ZeroDivisionError):
                    continue
                if torch.isfinite(v).all() and float(v.abs().max()) < 1e12 and float(v.norm()) > 0:
                    cols.append(v); keep.append(th)
            if cols:
                self.bank[tok] = (torch.stack(cols, 1), keep)


class GridSweepFitter:
    """Fast brute-force fitting: a global search over the grid placements of each piece (coordinate sweeps, every
    placement of one piece tried at once by projection, the others fixed), then Levenberg-Marquardt polishing from the
    best placement and from each piece's next-best placements (`basins` starts per piece in all: a correct structure's
    true valley can be narrow and miss the grid, while a wrong valley holds the best grid point), only for structures
    within `polish_within` of the noise floor. Swappable like the other fitters; it needs the system's ColumnBank (set
    by learn28) and the equation's noise floor (passed by the search)."""

    def __init__(self, bank, sweeps=2, polish_within=float("inf"), lm_iters=10, basins=2):
        self.bank, self.sweeps, self.polish_within, self.lm_iters = bank, sweeps, polish_within, lm_iters
        self.basins = basins

    def _profile(self, t, toks, idx, y):
        """Residual reduction of every grid placement of piece t, the other pieces fixed at idx."""
        others = [self.bank.bank[u][0][:, idx[u]] for u in toks if u != t]
        C = self.bank.bank[t][0]
        if others:
            Q = torch.linalg.qr(torch.stack(others, 1))[0]
            r0 = y - Q @ (Q.T @ y)
            Cp = C - Q @ (Q.T @ C)
        else:
            r0, Cp = y, C
        cn = (Cp * Cp).sum(0)
        return torch.where(cn > 1e-24, (Cp.T @ r0) ** 2 / cn, torch.zeros_like(cn))

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0, basins=None):
        t0 = time.perf_counter()
        toks = [t for t in toks]
        eff = {"starts": 0, "polished": 0, "seconds": 0.0}
        if not toks:
            eff["seconds"] = time.perf_counter() - t0
            return [], float(y.norm()), torch.zeros(0, dtype=torch.float64), eff
        if any(t not in self.bank.bank for t in toks):
            eff["seconds"] = time.perf_counter() - t0
            return [vocab.term(t, vocab.starts(t)[0]) for t in toks], float("inf"), None, eff
        warm = warm or {}
        idx = {}
        for t in toks:
            C, ths = self.bank.bank[t]
            idx[t] = warm.get(t, None) if isinstance(warm.get(t, None), int) else None
            if idx[t] is None or idx[t] >= C.shape[1]:
                cn = (C * C).sum(0) + 1e-300                    # start each new piece at its best single fit
                idx[t] = int(torch.argmax((C.T @ y) ** 2 / cn)) if C.shape[1] > 1 else 0
        piece_toks = [t for t in toks if self.bank.bank[t][0].shape[1] > 1]
        for _ in range(self.sweeps if piece_toks else 0):
            for t in piece_toks:
                idx[t] = int(torch.argmax(self._profile(t, toks, idx, y)))
                eff["starts"] += self.bank.bank[t][0].shape[1]
        Phi = torch.stack([self.bank.bank[t][0][:, idx[t]] for t in toks], 1)
        r, c = DL.lstsq_resid(Phi, y)
        res = float(r.norm())
        terms = [vocab.term(t, self.bank.bank[t][1][idx[t]]) for t in toks]
        best_idx = dict(idx)
        if piece_toks and math.isfinite(res) and res <= self.polish_within * max(floor, 1e-300):
            starts = [dict(idx)]
            inherited = {t: warm[t] if isinstance(warm.get(t), int) and warm[t] < self.bank.bank[t][0].shape[1] else idx[t]
                         for t in toks}
            if inherited != idx:                               # the parent's placements, if the sweep moved them
                starts.append(inherited)
            if basins:                                         # thorough fit: pieces of one family on one variable
                for t1, t2 in itertools.combinations(piece_toks, 2):   # also tried at one shared placement
                    if t1[0] != "piece" or t2[0] != "piece" or t1[1:3] != t2[1:3]:
                        continue
                    best_k, best_r = None, float("inf")
                    rest = [self.bank.bank[u][0][:, idx[u]] for u in toks if u not in (t1, t2)]
                    for k in range(min(self.bank.bank[t1][0].shape[1], self.bank.bank[t2][0].shape[1])):
                        Phi = torch.stack(rest + [self.bank.bank[t1][0][:, k], self.bank.bank[t2][0][:, k]], 1)
                        rk = float(DL.lstsq_resid(Phi, y)[0].norm())
                        if rk < best_r:
                            best_k, best_r = k, rk
                    if best_k is not None:
                        starts.append({**idx, t1: best_k, t2: best_k})
            fresh = piece_toks if basins else [t for t in piece_toks if not isinstance(warm.get(t), int)]
            for t in fresh:                                    # next-best placements of the pieces new to this structure
                                                               # (all pieces when `basins` is given), others fixed
                order = torch.argsort(self._profile(t, toks, idx, y), descending=True).tolist()
                starts += [{**idx, t: k} for k in order if k != idx[t]][:max((basins or self.basins) - 1, 0)]
            for st in starts:
                cand = [vocab.term(t, self.bank.bank[t][1][st[t]]) for t in toks]
                try:
                    t2, res2, c2 = fast_refine(cand, Zc, Xc, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                eff["polished"] += 1
                if math.isfinite(res2) and res2 < res:
                    terms, res, c, best_idx = t2, res2, c2, st
        eff["seconds"] = time.perf_counter() - t0
        for tm, t in zip(terms, toks):                         # remember the grid placement for warm starts
            tm.__dict__["grid_index"] = best_idx[t]
        return terms, res, c, eff


def thetas_of(term):
    return tuple(tuple(f.theta) for f in term.factors)


# ---------------------------------------------------------------------------------------------------------------------
# priorities
# ---------------------------------------------------------------------------------------------------------------------

class CandidateColumns:
    """Every token's columns at its grid placements (for the data-only evidence of adding it)."""

    def __init__(self, vocab, Zc, Xc):
        cols, owner = [], []
        for k, tok in enumerate(vocab.tokens):
            for th in vocab.starts(tok, limit=48):
                try:
                    v = DL.term_value(vocab.term(tok, th), Zc, Xc)
                except (RuntimeError, ValueError, ZeroDivisionError):
                    continue
                if torch.isfinite(v).all() and float(v.abs().max()) < 1e12 and float(v.norm()) > 0:
                    cols.append(v / v.norm()); owner.append(k)
        self.A = torch.stack(cols, 1)
        self.owner = torch.tensor(owner)
        self.V = len(vocab)

    def add_gain(self, Q, r):
        """Best partial R^2 of each token's columns given the span Q of the current terms and residual r."""
        C = self.A - Q @ (Q.T @ self.A) if Q.shape[1] else self.A
        cn = (C * C).sum(0)
        g = torch.where(cn > 1e-20, (C.T @ r) ** 2 / (cn * float(r @ r) + 1e-300), torch.zeros_like(cn)).clamp(0, 1)
        return torch.zeros(self.V, dtype=g.dtype).scatter_reduce(0, self.owner, g, reduce="amax")


def _evidence(node, ctx):
    """Data-only evidence at a node: each token's best partial R^2 for being added (given the current terms and
    residual), and for each current term the fraction of the remaining residual its removal would add."""
    y, S = ctx["y"], node["toks"]
    Phi = DL.design(node["terms"], ctx["Zc"], ctx["Xc"]) if S else torch.zeros(len(y), 0, dtype=y.dtype)
    Q = torch.linalg.qr(Phi)[0] if S else Phi
    r = y - Q @ (Q.T @ y) if S else y
    gain = ctx["cols"].add_gain(Q, r).numpy()
    rem = {}
    if S:
        XtX = Phi.T @ Phi + 1e-12 * torch.eye(Phi.shape[1], dtype=Phi.dtype)
        cov = torch.linalg.inv(XtX)
        c = DL.lstsq_resid(Phi, y)[1]                          # rate-data coefficients (the node's may be step-integral ones)
        delta = (c ** 2 / torch.diag(cov)).numpy()           # increase in squared residual if the term is dropped
        rr = float(r @ r) + 1e-300
        for a_, t in enumerate(S):
            rem[t] = float(delta[a_]) / rr
    return gain, rem


def data_prior(node, ctx):
    """Baseline priorities over a node's moves from this system's data only (no history). Adding a token: in
    proportion to 1 / (fraction of the residual it leaves unexplained), so a token explaining 99.99% outranks one
    explaining 99% by 100x; removing a term: in proportion to 1 / (1 + residual increase / residual); swapping: both.
    Move types are mixed 45% add, 15% remove, 40% swap."""
    vocab, S = ctx["vocab"], node["toks"]
    gain, rem = _evidence(node, ctx)
    add = -np.log(np.clip(1.0 - gain, 1e-12, 1.0))
    moves, logit = [], []
    inS = set(S)
    if len(S) < MAX_TERMS:
        for k, t in enumerate(vocab.tokens):
            if t not in inS:
                moves.append(("add", t, None)); logit.append(("add", float(add[k])))
    for t in S:
        lr = -math.log1p(rem[t])
        moves.append(("remove", t, None)); logit.append(("remove", lr))
        for k, b in enumerate(vocab.tokens):
            if b not in inS:
                moves.append(("swap", t, b)); logit.append(("swap", lr + float(add[k])))
    return moves, _mix(logit)


def data_prior_v1(node, ctx):
    """The first data-only priorities (kept for reference): adding in proportion to the partial R^2 itself, which barely
    separates tokens that all explain most of a smooth residual."""
    vocab, S = ctx["vocab"], node["toks"]
    gain, rem = _evidence(node, ctx)
    moves, logit = [], []
    inS = set(S)
    if len(S) < MAX_TERMS:
        for k, t in enumerate(vocab.tokens):
            if t not in inS:
                moves.append(("add", t, None)); logit.append(("add", math.log(gain[k] + 1e-9)))
    for t in S:
        lr = math.log(1.0 / (1.0 + rem[t]) + 1e-9)
        moves.append(("remove", t, None)); logit.append(("remove", lr))
        for k, b in enumerate(vocab.tokens):
            if b not in inS:
                moves.append(("swap", t, b)); logit.append(("swap", lr + math.log(gain[k] + 1e-9)))
    return moves, _mix(logit)


def uniform_prior(node, ctx):
    moves = _moves(node, ctx)
    return moves, np.full(len(moves), 1.0 / max(len(moves), 1))


def _moves(node, ctx):
    vocab, S = ctx["vocab"], node["toks"]
    inS = set(S)
    out = [("add", t, None) for t in vocab.tokens if t not in inS] if len(S) < MAX_TERMS else []
    for t in S:
        out.append(("remove", t, None))
        out += [("swap", t, b) for b in vocab.tokens if b not in inS]
    return out


def _mix(logit, weights={"add": 0.45, "remove": 0.15, "swap": 0.40}):
    kinds = np.array([k for k, _ in logit])
    lv = np.array([v for _, v in logit], dtype=float)
    p = np.zeros(len(lv))
    present = [k for k in weights if (kinds == k).any()]
    wsum = sum(weights[k] for k in present)
    for k in present:
        sel = kinds == k
        e = np.exp(lv[sel] - lv[sel].max())
        p[sel] = weights[k] / wsum * e / e.sum()
    return p


# ---------------------------------------------------------------------------------------------------------------------
# tree search over one equation
# ---------------------------------------------------------------------------------------------------------------------

def _apply(S, mv):
    kind, a, b = mv
    if kind == "add":
        return tuple(sorted(S + (a,), key=str))
    if kind == "remove":
        return tuple(t for t in S if t != a)
    return tuple(sorted(tuple(t for t in S if t != a) + (b,), key=str))


def search_equation(i, ctx, budget, prior_fn, fitter, rng, c_puct=1.5, eps=0.15, time_cap=float("inf"), lam=0.02,
                    extra_after_exact=1.0, min_extra=500):
    """Tree search over equation i's term set within `budget` fitted structures. Returns (table of evaluated
    structures, effort). Every evaluated structure has: toks, terms (fitted), coefs, res, value. Once a plausibly exact
    structure is found at structure k, the search goes on for max(min_extra, extra_after_exact * k) more structures
    (looking for a smaller one) and stops. Strict supersets of a plausibly exact structure skip the precise stage.
    Pruning: every plausibly exact structure's one-term-smaller structures are evaluated next (recursively), so the
    smallest exact structure near it is always tried. A structure with a degenerate piece (DL.degenerate: collapsed to
    something simpler over the observed range, or singular there) cannot be plausibly exact."""
    vocab, Zc, Xc = ctx["vocab"], ctx["Zc"], ctx["Xc"]
    y = ctx["G"][:, i]
    ctx = dict(ctx, y=y)
    yn = float(y.norm()) + 1e-300
    floor = max(float(ctx["noise"][i]) * math.sqrt(len(y)), 1e-15 * yn)
    span = max(math.log10(yn / floor), 1e-3)
    tau = max(TAU_EXACT, TAU_FLOOR * floor / yn)
    t0 = time.perf_counter()
    table, eff = {}, {"structures": 0, "starts": 0, "fit_seconds": 0.0, "prior_seconds": 0.0, "first_exact": None, "tau": tau,
                      "first_exact_seconds": None, "floor_rel": floor / yn, "precise": 0, "precise_seconds": 0.0}
    si_tr, si_va = ctx.get("si_tr"), ctx.get("si_va")
    exact_sets = []                                            # plausibly exact structures found so far
    prune = []                                                 # (structure, parent node) waiting to be evaluated

    def evaluate(S, parent):
        warm = {}
        if parent is not None:
            for t, term in zip(parent["toks"], parent["terms"]):
                if t in S and term.factors:
                    warm[t] = getattr(term, "grid_index", None)
                    if warm[t] is None:
                        warm[t] = thetas_of(term)
        terms, res, coefs, e = fitter.fit(vocab, S, Zc, Xc, y, warm, floor=floor)
        eff["structures"] += 1; eff["starts"] += e["starts"]; eff["fit_seconds"] += e["seconds"]
        rel_rate = res / yn if math.isfinite(res) else float("inf")
        rel_t = rel_v = float("inf")
        superset = any(E < set(S) for E in exact_sets)
        if S and si_tr is not None and rel_rate <= PRECISE_GATE and res <= PRECISE_FLOOR * floor and not superset:
            tp = time.perf_counter()                               # precise stage: thorough refit (unless the fit is
            try:                                                   # already at the floor), step integrals, held-back
                if res > 2.0 * floor:
                    terms, res, coefs = thorough_fit(fitter, vocab, S, Zc, Xc, y, warm, floor, (terms, res, coefs))
                terms2, rel_t, c2 = refine_integral(terms, si_tr, i)
                if c2 is not None and math.isfinite(rel_t):
                    terms, coefs = terms2, c2
                    rel_v = heldback_error(terms, coefs, si_va, i)
            except (RuntimeError, ValueError):
                pass
            eff["precise"] += 1; eff["precise_seconds"] += time.perf_counter() - tp
        try:
            degen = any(DL.degenerate(t, Zc, Xc, strict=False) for t in terms)
        except (RuntimeError, ValueError, ZeroDivisionError):
            degen = True
        rel = rel_v if math.isfinite(rel_v) and not degen else max(rel_rate, PRECISE_GATE)
        v = 0.0 if not math.isfinite(rel) else min(1.0, max(0.0, math.log10(1.0 / max(rel, 1e-14)) / 14.0))
        node = {"toks": S, "terms": terms, "coefs": coefs, "res": res, "rel_t": rel_t, "rel_v": rel_v,
                "value": v - lam * len(S), "n": 0, "k": eff["structures"], "degenerate": degen,
                "best": v - lam * len(S), "children": {}, "moves": None, "prior": None,
                "exact": bool(rel_v <= tau and rel_t <= tau and not degen)}
        if node["exact"]:
            exact_sets.append(set(S))
            for t in S:                                        # pruning: the one-term-smaller structures next
                sub = tuple(u for u in S if u != t)
                if sub and sub not in table:
                    prune.append((sub, node))
            if eff["first_exact"] is None:
                eff["first_exact"] = eff["structures"]; eff["first_exact_seconds"] = time.perf_counter() - t0
        table[S] = node
        return node

    root = evaluate((), None)
    while eff["structures"] < budget and time.perf_counter() - t0 < time_cap:
        if prune:                                              # pruning before the tree search continues
            sub, par = prune.pop(0)
            if sub not in table:
                evaluate(sub, par)
                eff["pruned"] = eff.get("pruned", 0) + 1
            continue
        if eff["first_exact"] is not None and eff["structures"] >= eff["first_exact"] + max(min_extra, extra_after_exact * eff["first_exact"]):
            eff["stopped_after_exact"] = True
            break
        node, path = root, [root]
        for _depth in range(2 * MAX_TERMS + 4):
            if node["moves"] is None:
                tp = time.perf_counter()
                node["moves"], node["prior"] = prior_fn(node, ctx)
                node["tried"] = np.zeros(len(node["moves"]), dtype=bool)   # moves already followed from this node
                node["tried_idx"], node["child_of"] = [], {}
                eff["prior_seconds"] += time.perf_counter() - tp
            moves, P = node["moves"], node["prior"]
            if not moves:
                break
            untried = ~node["tried"]
            if untried.any() and rng.random() < eps:          # exploration share: a random untried move
                k = int(rng.choice(np.flatnonzero(untried)))
            else:
                q = np.full(len(moves), node["value"] - 0.05)
                nv = np.zeros(len(moves))
                for kk in node["tried_idx"]:
                    ch = table[node["child_of"][kk]]
                    q[kk] = ch["best"]; nv[kk] = ch["n"]
                score = q + c_puct * P * math.sqrt(node["n"] + 1) / (1 + nv)
                k = int(np.argmax(score))
            mv = moves[k]
            S2 = node["child_of"].get(k) or _apply(node["toks"], mv)
            if not node["tried"][k]:
                node["tried"][k] = True; node["tried_idx"].append(k); node["child_of"][k] = S2
            if S2 in table:
                nxt = table[S2]
                if any(nxt is p_ for p_ in path):               # a cycle: stop descending here
                    break
                node = nxt; path.append(node)
                continue
            child = evaluate(S2, node)
            path.append(child)
            break
        leaf_v = path[-1]["value"]
        for nd in path:                                        # risk-seeking backup: the best value found below
            nd["n"] += 1
            nd["best"] = max(nd["best"], leaf_v)
    eff["seconds"] = time.perf_counter() - t0
    return table, eff


def candidates_for(table, top=3):
    """An equation's candidate structures for assembly: plausibly exact ones (fewest terms first), then the best."""
    nodes = list(table.values())
    ex = sorted([n for n in nodes if n["exact"]], key=lambda n: (len(n["toks"]), n["rel_v"]))
    rest = sorted([n for n in nodes if not n["exact"]], key=lambda n: -n["value"])
    return (ex + rest)[:top]


# ---------------------------------------------------------------------------------------------------------------------
# a whole learning run
# ---------------------------------------------------------------------------------------------------------------------

def learn28(bursts, budget, prior="data", fitter=None, seed=1, time_cap=float("inf"), max_assemblies=6):
    """Search each equation within `budget` fitted structures, assemble, fit exactly on the step-to-step map, check on
    held-back bursts, apply ties. Returns a result dict with the model, held-back error, acceptance and effort."""
    t0 = time.perf_counter()
    view = PN.learner_view(bursts, seed=seed)
    Zc, G = view["Zc"], view["G"]
    Xc = Zc * view["sig"] + view["mu"]
    d = Zc.shape[1]
    vocab = Vocab28(d, Zc, Xc)
    ctx = {"vocab": vocab, "Zc": Zc, "Xc": Xc, "G": G, "noise": view["noise"],
           "si_tr": StepIntegrals(view["Ztr"], view["mu"], view["sig"]), "si_va": StepIntegrals(view["Zva"], view["mu"], view["sig"])}
    tp = time.perf_counter()
    ctx["cols"] = CandidateColumns(vocab, Zc, Xc)
    setup = time.perf_counter() - tp
    effort = {"vocabulary": len(vocab), "equations": []}
    tp = time.perf_counter()
    bank = ColumnBank(vocab, Zc, Xc)
    setup += time.perf_counter() - tp
    effort["setup_seconds"] = setup
    fitter = fitter or GridSweepFitter(bank)
    prior_fn = {"data": data_prior, "data_v1": data_prior_v1, "uniform": uniform_prior}[prior] if isinstance(prior, str) else prior
    tables = []
    for i in range(d):                                         # one random stream per equation: a smaller budget is
        rng = np.random.default_rng([seed, i])                 # exactly a prefix of a larger one
        table, eff = search_equation(i, ctx, budget, prior_fn, fitter, rng, time_cap=time_cap / d)
        tables.append(table); effort["equations"].append(eff)
    cands = [candidates_for(t) for t in tables]
    all_plausible = all(c and c[0]["exact"] for c in cands)
    combos = sorted(itertools.product(*[range(len(c)) for c in cands]), key=sum)[:max_assemblies if all_plausible else 1]
    effort["all_equations_plausible"] = all_plausible
    best, assemblies = None, 0
    ta = time.perf_counter()
    for combo in combos:
        if time.perf_counter() - ta > ASSEMBLY_CAP:
            effort["assembly_capped"] = True
            break
        sups = [cands[i][k]["terms"] for i, k in enumerate(combo)]
        cfs = [cands[i][k]["coefs"] if cands[i][k]["coefs"] is not None else torch.zeros(len(sups[i]), dtype=torch.float64)
               for i, k in enumerate(combo)]
        model = DL.Model(sups, cfs, view["mu"], view["sig"])
        assemblies += 1
        try:
            model, nsub, err, _ = DL.polish(model, view["z0"], view["z1"], iters=40)
            if not math.isfinite(err):
                continue
            model, val, info = TI.finalize(model, view["z0"], view["z1"], view["zv0"], view["zv1"])
        except (RuntimeError, ValueError):
            continue
        if best is None or val < best[1]:
            best = (model, val, info, combo)
        if val <= DL.VAL_TOL:
            break
    effort["assemblies"] = assemblies
    effort["assembly_seconds"] = time.perf_counter() - ta
    effort["structures"] = sum(e["structures"] for e in effort["equations"])
    effort["fit_seconds"] = sum(e["fit_seconds"] for e in effort["equations"])
    effort["seconds"] = time.perf_counter() - t0
    out = {"effort": effort, "view": view}
    if best is None:
        out.update(status="no_model", val=float("inf"))
        return out
    model, val, info, combo = best
    out.update(status="accepted" if val <= DL.VAL_TOL else "rejected", val=val, model=model, ties=info,
               combo=list(combo), terms_per_eq=[len(cands[i][k]["toks"]) for i, k in enumerate(combo)],
               chosen_k=[cands[i][k]["k"] for i, k in enumerate(combo)])   # when each chosen structure was first fitted
    return out
