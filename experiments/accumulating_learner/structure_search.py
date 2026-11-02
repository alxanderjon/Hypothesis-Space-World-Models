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


# ---------------------------------------------------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------------------------------------------------

class Vocab28:
    """Term types of the search for d variables, with start grids for their constants from the system's own data
    (scaled states Zc). A token is hashable; term(token, thetas) builds the learner Term."""

    def __init__(self, d, Zc):
        self.d = d
        lo, hi = Zc.min(0).values.numpy(), Zc.max(0).values.numpy()
        span = np.maximum(hi - lo, 1e-9)
        zmax = np.maximum(np.abs(Zc).max(0).values.numpy(), 1e-9)
        q = {j: np.quantile(Zc[:, j].numpy(), [0.1, 0.3, 0.5, 0.7, 0.9]) for j in range(d)}
        self.positive = [bool((Zc[:, j] > 0).all()) for j in range(d)]
        mults = [tuple([0] * d)] + [tuple(int(k == j) for k in range(d)) for j in range(d)]
        toks = [("mono", tuple(m)) for m in DL.monomials(d, 3)]
        for fam in PIECES:
            for j in range(d):
                if fam in ("hill", "pow") and not self.positive[j]:
                    continue
                toks += [("piece", fam, j, m) for m in mults]
        for fam in PAIR_PIECES:
            for j, k in itertools.combinations(range(d), 2):
                if fam == "hill" and not (self.positive[j] and self.positive[k]):
                    continue
                toks += [("pair", fam, (j, k), m) for m in mults]
        for fam in PROD_PIECES:
            for j, k in itertools.combinations(range(d), 2):
                if fam == "hill" and not (self.positive[j] and self.positive[k]):
                    continue
                toks += [("prod", fam, (j, k))]
        self.tokens = toks
        self.index = {t: i for i, t in enumerate(toks)}
        # start grids per piece family and variable (thetas of one factor)
        g = {}
        for j in range(d):
            poles = np.concatenate([lo[j] - np.geomspace(0.02, 50, 8) * span[j], hi[j] + np.geomspace(0.02, 50, 8) * span[j]])
            g[("recip", j)] = [(1.0, float(-p)) for p in poles]
            g[("hill", j)] = [(float(1.0 / (h * zmax[j])), 0.0, n) for h in np.geomspace(0.03, 30, 6) for n in (1.0, 2.0, 4.0)]
            g[("exp", j)] = [(float(s / zmax[j]), 0.0) for s in (-3.0, -1.0, -0.3, 0.3, 1.0, 3.0)]
            g[("tanh", j)] = [(float(b / span[j]), float(-b / span[j] * c)) for b in (1.0, 3.0, 10.0) for c in q[j][1:4]]
            g[("sin", j)] = [(float(2 * math.pi * w / span[j]), ph) for w in (0.5, 1.0, 2.0, 4.0) for ph in (0.0, math.pi / 2)]
            g[("abs", j)] = [(1.0, float(-c)) for c in q[j]]
            g[("pow", j)] = [(1.0, 0.0, p) for p in (0.5, 1.5, 2.5, -0.5, -1.0, -2.0)]
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
        self.meta = {"hill": ("zsnap",), "pow": ("zsnap",)}

    def __len__(self):
        return len(self.tokens)

    def factor_families(self, tok):
        """[(learner family, var, grid key)] for the factors of a token."""
        if tok[0] == "mono":
            return []
        fam_map = {"recip": "crecip", "hill": "chill", "exp": "cexp", "tanh": "ctanh", "sin": "csin", "abs": "cabs",
                   "pow": "cpow"}
        if tok[0] == "piece":
            _, fam, j, _ = tok
            return [(fam_map[fam], j, (fam, j))]
        if tok[0] == "pair":
            _, fam, (j, k), _ = tok
            return [(fam_map[fam], (j, k), ("pair_" + fam, j, k))]
        _, fam, (j, k) = tok
        return [(fam_map[fam], j, (fam, j)), (fam_map[fam], k, (fam, k))]

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
            base = key[0].replace("pair_", "")
            facs.append(DL.Factor(fam, var, tuple(float(v) for v in th), self.meta.get(base, ())))
        return DL.Term(self.mono_of(tok), tuple(facs), 1)

    def starts(self, tok):
        """All grid starting values for a token: a list of per-factor theta tuples."""
        keys = [k for _, _, k in self.factor_families(tok)]
        if not keys:
            return [()]
        grids = [self.grid[k] for k in keys]
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

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0):
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

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0):
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


def fast_refine(terms, Z, X, y, iters=10):
    """Levenberg-Marquardt on the nonlinear constants with the linear coefficients projected out (as DL.refine_support)
    but with a finite-difference Jacobian: a candidate has only a few nonlinear constants, so this is far cheaper than
    automatic differentiation. Returns (terms, residual norm, coefficients)."""
    flat = [float(v) for t in terms for f in t.factors for v in f.theta]
    if not flat:
        r, c = DL.lstsq_resid(DL.design(terms, Z, X), y)
        return terms, float(r.norm()), c
    p, full, _ = DL._free_map(torch.tensor(flat, dtype=torch.float64), DL.frozen_positions(terms))
    if p.numel() == 0:
        r, c = DL.lstsq_resid(DL.design(terms, Z, X), y)
        return terms, float(r.norm()), c

    def resid(q):
        with torch.no_grad():
            return DL.lstsq_resid(DL.design(terms, Z, X, DL._split_thetas(terms, full(q))), y)[0]

    r = resid(p)
    if not torch.isfinite(r).all():
        return terms, float("inf"), None
    lam = 1e-3
    for _ in range(iters):
        J = torch.empty(len(r), len(p), dtype=torch.float64)
        for k in range(len(p)):
            h = 1e-7 * (1.0 + abs(float(p[k])))
            q = p.clone(); q[k] += h
            J[:, k] = (resid(q) - r) / h
        if not torch.isfinite(J).all():
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
    new = DL._with_thetas(terms, DL._split_thetas(terms, full(p)))
    r, c = DL.lstsq_resid(DL.design(new, Z, X), y)
    if not torch.isfinite(r).all():
        return terms, float("inf"), c
    return new, float(r.norm()), c


class ColumnBank:
    """Every token's columns at its grid starting values, computed once per system (they do not depend on the
    equation): bank[token] = (columns (n, g), list of per-factor theta tuples)."""

    def __init__(self, vocab, Zc, Xc, max_grid=24):
        self.bank = {}
        for tok in vocab.tokens:
            ths = vocab.starts(tok)[:max_grid] if vocab.factor_families(tok) else [()]
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
    placement of one piece tried at once by projection, the others fixed), then Levenberg-Marquardt polishing only for
    structures that come within `polish_within` of the noise floor. Swappable like the other fitters; it needs the
    system's ColumnBank (set by learn28) and the equation's noise floor (passed by the search)."""

    def __init__(self, bank, sweeps=2, polish_within=1e3, lm_iters=10):
        self.bank, self.sweeps, self.polish_within, self.lm_iters = bank, sweeps, polish_within, lm_iters

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0):
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
                others = [self.bank.bank[u][0][:, idx[u]] for u in toks if u != t]
                C = self.bank.bank[t][0]
                if others:
                    Q = torch.linalg.qr(torch.stack(others, 1))[0]
                    r0 = y - Q @ (Q.T @ y)
                    Cp = C - Q @ (Q.T @ C)
                else:
                    r0, Cp = y, C
                cn = (Cp * Cp).sum(0)
                red = torch.where(cn > 1e-24, (Cp.T @ r0) ** 2 / cn, torch.zeros_like(cn))
                idx[t] = int(torch.argmax(red))
                eff["starts"] += C.shape[1]
        Phi = torch.stack([self.bank.bank[t][0][:, idx[t]] for t in toks], 1)
        r, c = DL.lstsq_resid(Phi, y)
        res = float(r.norm())
        terms = [vocab.term(t, self.bank.bank[t][1][idx[t]]) for t in toks]
        if piece_toks and math.isfinite(res) and res <= self.polish_within * max(floor, 1e-300):
            try:
                t2, res2, c2 = fast_refine(terms, Zc, Xc, y, iters=self.lm_iters)
                eff["polished"] = 1
                if math.isfinite(res2) and res2 < res:
                    terms, res, c = t2, res2, c2
            except (RuntimeError, ValueError):
                pass
        eff["seconds"] = time.perf_counter() - t0
        terms_out = terms
        for tm, t in zip(terms_out, toks):                     # remember the grid placement for warm starts
            tm.__dict__["grid_index"] = idx[t]
        return terms_out, res, c, eff


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
            for th in vocab.starts(tok)[:24]:
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


def data_prior(node, ctx):
    """Baseline priorities over a node's moves from this system's data only (no history): adding a token in
    proportion to its evidence given the current fit, removing a term in proportion to how little it contributes,
    swapping by both. Move types are mixed 45% add, 15% remove, 40% swap."""
    vocab, y = ctx["vocab"], ctx["y"]
    S = node["toks"]
    Phi = DL.design(node["terms"], ctx["Zc"], ctx["Xc"]) if S else torch.zeros(len(y), 0, dtype=y.dtype)
    Q = torch.linalg.qr(Phi)[0] if S else Phi
    r = y - Q @ (Q.T @ y) if S else y
    gain = ctx["cols"].add_gain(Q, r).numpy()
    rem = {}
    if S:
        XtX = Phi.T @ Phi + 1e-12 * torch.eye(Phi.shape[1], dtype=Phi.dtype)
        cov = torch.linalg.inv(XtX)
        c = node["coefs"] if node["coefs"] is not None else torch.linalg.lstsq(Phi, y).solution
        delta = (c ** 2 / torch.diag(cov)).numpy()           # increase in squared residual if the term is dropped
        rr = float(r @ r) + 1e-300
        for a, t in enumerate(S):
            rem[t] = 1.0 / (1.0 + float(delta[a]) / rr)
    moves, logit = [], []
    inS = set(S)
    if len(S) < MAX_TERMS:
        for k, t in enumerate(vocab.tokens):
            if t not in inS:
                moves.append(("add", t, None)); logit.append(("add", math.log(gain[k] + 1e-9)))
    for t in S:
        moves.append(("remove", t, None)); logit.append(("remove", math.log(rem[t] + 1e-9)))
        for k, b in enumerate(vocab.tokens):
            if b not in inS:
                moves.append(("swap", t, b)); logit.append(("swap", math.log(rem[t] + 1e-9) + math.log(gain[k] + 1e-9)))
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


def search_equation(i, ctx, budget, prior_fn, fitter, rng, c_puct=1.5, eps=0.15, time_cap=float("inf"), lam=0.02):
    """Tree search over equation i's term set within `budget` fitted structures. Returns (table of evaluated
    structures, effort). Every evaluated structure has: toks, terms (fitted), coefs, res, value."""
    vocab, Zc, Xc = ctx["vocab"], ctx["Zc"], ctx["Xc"]
    y = ctx["G"][:, i]
    ctx = dict(ctx, y=y)
    yn = float(y.norm()) + 1e-300
    floor = max(float(ctx["noise"][i]) * math.sqrt(len(y)), 1e-15 * yn)
    span = max(math.log10(yn / floor), 1e-3)
    t0 = time.perf_counter()
    table, eff = {}, {"structures": 0, "starts": 0, "fit_seconds": 0.0, "prior_seconds": 0.0, "first_exact": None,
                      "first_exact_seconds": None, "floor_rel": floor / yn}

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
        v = 0.0 if not math.isfinite(res) else min(1.0, max(0.0, math.log10(yn / max(res, floor)) / span))
        node = {"toks": S, "terms": terms, "coefs": coefs, "res": res, "value": v - lam * len(S), "n": 0, "k": eff["structures"],
                "best": v - lam * len(S), "children": {}, "moves": None, "prior": None,
                "exact": bool(math.isfinite(res) and res <= 3 * floor)}
        if node["exact"] and eff["first_exact"] is None:
            eff["first_exact"] = eff["structures"]; eff["first_exact_seconds"] = time.perf_counter() - t0
        table[S] = node
        return node

    root = evaluate((), None)
    while eff["structures"] < budget and time.perf_counter() - t0 < time_cap:
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
    ex = sorted([n for n in nodes if n["exact"]], key=lambda n: (len(n["toks"]), n["res"]))
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
    vocab = Vocab28(d, Zc)
    ctx = {"vocab": vocab, "Zc": Zc, "Xc": Xc, "G": G, "noise": view["noise"]}
    tp = time.perf_counter()
    ctx["cols"] = CandidateColumns(vocab, Zc, Xc)
    setup = time.perf_counter() - tp
    effort = {"vocabulary": len(vocab), "equations": []}
    tp = time.perf_counter()
    bank = ColumnBank(vocab, Zc, Xc)
    setup += time.perf_counter() - tp
    effort["setup_seconds"] = setup
    fitter = fitter or GridSweepFitter(bank)
    prior_fn = {"data": data_prior, "uniform": uniform_prior}[prior] if isinstance(prior, str) else prior
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(d):
        table, eff = search_equation(i, ctx, budget, prior_fn, fitter, rng, time_cap=time_cap / d)
        tables.append(table); effort["equations"].append(eff)
    cands = [candidates_for(t) for t in tables]
    all_plausible = all(c and c[0]["exact"] for c in cands)
    combos = sorted(itertools.product(*[range(len(c)) for c in cands]), key=sum)[:max_assemblies if all_plausible else 1]
    effort["all_equations_plausible"] = all_plausible
    best, assemblies = None, 0
    ta = time.perf_counter()
    for combo in combos:
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
