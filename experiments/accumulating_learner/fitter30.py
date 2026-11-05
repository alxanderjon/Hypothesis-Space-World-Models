"""Trial 30: fitter v2 for equations with several nonlinear pieces (research log, 4 November).

Same interface as structure_search.GridSweepFitter (fit(...) for the ordinary fit; thorough(...) for the precise stage),
so it is swappable and used by both learners. Stages:
  (a) algebraic start: rational pieces (single-variable and weighted-sum reciprocals, Hill-type pieces with whole-number
      exponents 1-4, products of reciprocals or of Hill-type pieces) are fitted jointly by clearing denominators: one
      linear least-squares problem in the lifted unknowns (normalized by fixing one coefficient), with Sanathanan-Koerner
      reweighting, on dense rate data (derivatives of each training burst's interpolating polynomial at interior points);
      other terms enter at their current placements;
  (b) joint global search (precise stage): all of the structure's hard constants together, linear coefficients solved
      exactly at every candidate (Sobol points, then refinement around the best);
  (c) Levenberg-Marquardt with variable projection from the best distinct candidates;
  (d) identifiability check: an exponent may be snapped to the nearest whole number only if the data cannot pin it down
      (uncertainty from the Jacobian and the precision floor above SNAP_SIGMA); both versions are then judged by the
      held-back step-integral error and the lower is kept. Counts are kept in .stats.
"""

import itertools
import math
import time

import numpy as np
import torch

import discovery_learner as DL
import structure_search as SS

SNAP_SIGMA = 0.05            # exponent uncertainty above which snapping to a whole number is allowed
HILL_N = (1.0, 2.0, 3.0, 4.0)
SK_ROUNDS = 3
SOBOL_N, REFINE_ROUNDS, REFINE_TOP, REFINE_N = 1024, 3, 8, 128


# ---------------------------------------------------------------------------------------------------------------------
# dense rate data from the training bursts
# ---------------------------------------------------------------------------------------------------------------------

class DenseRates:
    """States and time derivatives (per step) of each training burst's degree-(L-1) interpolating polynomial at interior
    points t = k + 1/4 and k + 3/4 (k = 1 .. L-3), computed once per system (barycentric weights)."""

    def __init__(self, Zb, mu, sig):
        Zb = torch.as_tensor(Zb, dtype=torch.float64)
        nb, L, d = Zb.shape
        tau = np.arange(L, dtype=float)
        bw = np.array([1.0 / np.prod([tau[m] - tau[n] for n in range(L) if n != m]) for m in range(L)])
        ts = np.array([k + f for k in range(1, L - 2) for f in (0.25, 0.75)])
        Lm, Dm = np.zeros((len(ts), L)), np.zeros((len(ts), L))
        for a, t in enumerate(ts):
            q = bw / (t - tau)
            s, s1 = q.sum(), -(bw / (t - tau) ** 2).sum()
            Lm[a] = q / s
            Dm[a] = (-(bw / (t - tau) ** 2) * s - q * s1) / s ** 2
        Lm, Dm = torch.tensor(Lm), torch.tensor(Dm)
        self.Z = torch.einsum("qm,bmd->bqd", Lm, Zb).reshape(-1, d)
        self.X = self.Z * sig + mu
        self.dZ = torch.einsum("qm,bmd->bqd", Dm, Zb).reshape(-1, d)


# ---------------------------------------------------------------------------------------------------------------------
# (a) algebraic start for rational pieces
# ---------------------------------------------------------------------------------------------------------------------

def _rational_factors(tok, n_choice):
    """Denominator factors of a rational token as [(basis kinds, var(s))], its numerator spec, and how to rebuild thetas.
    Returns None for non-rational tokens. n_choice: whole-number exponents for its Hill-type factors (tuple)."""
    if tok[0] == "piece" and tok[1] == "recip":
        return {"factors": [("lin", tok[2])], "num": ("one",), "kind": "recip"}
    if tok[0] == "piece" and tok[1] == "hill":
        return {"factors": [("pow", tok[2], n_choice[0])], "num": ("xpow", tok[2], n_choice[0]), "kind": "hill"}
    if tok[0] == "pair" and tok[1] == "recip":
        return {"factors": [("pair", tok[2])], "num": ("one",), "kind": "pair_recip"}
    if tok[0] == "prod" and tok[1] == "recip":
        j, k = tok[2]
        return {"factors": [("lin", j), ("lin", k)], "num": ("one",), "kind": "prod_recip"}
    if tok[0] == "prod" and tok[1] == "hill":
        j, k = tok[2]
        return {"factors": [("pow", j, n_choice[0]), ("pow", k, n_choice[1])], "num": ("xpow2", j, n_choice[0], k, n_choice[1]),
                "kind": "prod_hill"}
    return None


def _n_hill(tok):
    if tok[0] == "piece" and tok[1] == "hill":
        return 1
    if tok[0] == "prod" and tok[1] == "hill":
        return 2
    return 0


def _basis(fac, Z, X):
    """Known basis columns of a denominator factor; the last one's coefficient is fixed to 1."""
    if fac[0] == "lin":
        j = fac[1]
        return [torch.ones_like(Z[:, j]), Z[:, j]]                  # c + z_j
    if fac[0] == "pow":
        j, n = fac[1], fac[2]
        return [torch.ones_like(X[:, j]), X[:, j] ** n]             # K + x_j^n
    j, k = fac[1]
    return [torch.ones_like(Z[:, j]), Z[:, k], Z[:, j]]             # c + w z_k + z_j


def _numerator(spec, Z, X):
    if spec[0] == "one":
        return torch.ones(Z.shape[0], dtype=Z.dtype)
    if spec[0] == "xpow":
        return X[:, spec[1]] ** spec[2]
    return X[:, spec[1]] ** spec[2] * X[:, spec[3]] ** spec[4]


def _mono(e, Z):
    v = torch.ones(Z.shape[0], dtype=Z.dtype)
    for k, p in enumerate(e):
        if p:
            v = v * Z[:, k] ** p
    return v


def algebraic_start(vocab, toks, terms, Z, X, ydot, max_combos=64):
    """Joint starting values for all rational pieces of the structure `toks` by clearing denominators (lifted linear
    least squares + Sanathanan-Koerner reweighting) on rate data (Z, X, ydot). Non-rational pieces enter as fixed
    columns at their placements in `terms`. Returns a list of candidate term lists (one per exponent combination that
    gave a valid start), best first, or [] if the structure has no rational piece or the problem is underdetermined."""
    toks = list(toks)
    rat = [t for t in toks if _rational_factors(t, (1.0, 1.0)) is not None]
    if not rat:
        return []
    fixed_cols = []                                              # monomials and non-rational pieces (fixed placements)
    for t, tm in zip(toks, terms):
        if t in rat:
            continue
        if t[0] == "mono":
            fixed_cols.append(_mono(t[1], Z))
        else:
            with torch.no_grad():
                fixed_cols.append(DL.term_value(tm, Z, X))
    n_h = [_n_hill(t) for t in rat]
    choices = [list(itertools.product(HILL_N, repeat=h)) if h else [()] for h in n_h]
    combos = list(itertools.product(*choices))[:max_combos]
    out = []
    for combo in combos:
        specs = [_rational_factors(t, ch if ch else (1.0, 1.0)) for t, ch in zip(rat, combo)]
        facs = [f for s in specs for f in s["factors"]]
        owner = [p for p, s in enumerate(specs) for _ in s["factors"]]
        bases = [_basis(f, Z, X) for f in facs]
        dims = [len(b) for b in bases]
        idx_all = list(itertools.product(*[range(m) for m in dims]))
        last = tuple(m - 1 for m in dims)
        U = torch.stack([torch.prod(torch.stack([bases[f][i_] for f, i_ in enumerate(ix)]), 0) for ix in idx_all], 1)
        # per rational piece: tensor basis over the other pieces' factors
        U_minus = []
        for p in range(len(specs)):
            keep = [f for f in range(len(facs)) if owner[f] != p]
            ix_k = list(itertools.product(*[range(dims[f]) for f in keep])) if keep else [()]
            cols = [torch.prod(torch.stack([bases[f][i_] for f, i_ in zip(keep, ix)] or [torch.ones_like(Z[:, 0])]), 0) for ix in ix_k]
            U_minus.append((keep, ix_k, torch.stack(cols, 1) * _numerator(specs[p]["num"], Z, X).unsqueeze(1)
                            * _mono(vocab.mono_of(rat[p]), Z).unsqueeze(1)))
        n_unk = (len(idx_all) - 1) + len(fixed_cols) * len(idx_all) + sum(u[2].shape[1] for u in U_minus)
        if n_unk > 0.8 * len(ydot):
            continue                                             # underdetermined lifted problem: skip this start
        fix_col = idx_all.index(last)
        free = [a for a in range(len(idx_all)) if a != fix_col]
        blocks = [ydot.unsqueeze(1) * U[:, free]]
        blocks += [-(b.unsqueeze(1) * U) for b in fixed_cols]
        blocks += [-u[2] for u in U_minus]
        Mtx = torch.cat(blocks, 1)
        rhs = -ydot * U[:, fix_col]
        w = torch.ones_like(ydot)
        A = None
        ok = True
        for _sk in range(SK_ROUNDS + 1):
            try:
                sol = torch.linalg.lstsq(Mtx * w.unsqueeze(1), (rhs * w).unsqueeze(1), driver="gelsd").solution.squeeze(1)
            except RuntimeError:
                ok = False
                break
            A = torch.zeros(len(idx_all), dtype=Z.dtype)
            A[free] = sol[:len(free)]
            A[fix_col] = 1.0
            Dh = U @ A
            if not torch.isfinite(Dh).all() or float(Dh.abs().min()) <= 0:
                break
            w = 1.0 / Dh.abs()                                   # Sanathanan-Koerner reweighting
        if not ok or A is None or not torch.isfinite(A).all():
            continue
        # read each factor's coefficients off the entries where all other factors take their last basis function
        coef = []
        for f in range(len(facs)):
            vals = []
            for q in range(dims[f]):
                ix = tuple(q if g == f else dims[g] - 1 for g in range(len(facs)))
                vals.append(float(A[idx_all.index(ix)]))
            coef.append(vals)
        thetas, valid = [], True
        for p, s in enumerate(specs):
            fl = [coef[f] for f in range(len(facs)) if owner[f] == p]
            if s["kind"] == "recip":
                th = ((1.0, fl[0][0]),)
            elif s["kind"] == "hill":
                th = ((fl[0][0], s["factors"][0][2]),)
                valid &= fl[0][0] > 0
            elif s["kind"] == "pair_recip":
                th = ((fl[0][1], 1.0, fl[0][0]),)
            elif s["kind"] == "prod_recip":
                th = ((1.0, fl[0][0]), (1.0, fl[1][0]))
            else:
                th = ((fl[0][0], s["factors"][0][2]), (fl[1][0], s["factors"][1][2]))
                valid &= fl[0][0] > 0 and fl[1][0] > 0
            thetas.append(th)
        if not valid:
            continue
        new_terms = []
        k_r = 0
        for t, tm in zip(toks, terms):
            if t in rat:
                new_terms.append(vocab.term(t, thetas[rat.index(t)]))
                k_r += 1
            else:
                new_terms.append(tm)
        try:
            bad = any(DL.degenerate(tm, Z, X, strict=False) for tm in new_terms if tm.factors)
        except (RuntimeError, ValueError, ZeroDivisionError):
            bad = True
        if bad:
            continue
        out.append(new_terms)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# (b) joint global search over all hard constants
# ---------------------------------------------------------------------------------------------------------------------

def _free_slots(terms):
    """(term, factor, position, lo, hi, log-scale) for every free constant, with search bounds from the vocabulary's own
    start grids (widened) or from the data."""
    frozen, slots, k = set(DL.frozen_positions(terms)), [], 0
    for m, t in enumerate(terms):
        for fi, f in enumerate(t.factors):
            for a_ in range(len(f.theta)):
                if k not in frozen and not ("snap" in (f.meta or ()) and SS.SNAP_POS.get(f.family) == a_):
                    slots.append((m, fi, a_))
                k += 1
    return slots


def _samplers(term, fi, Z, X):
    """Natural search ranges, from the data, for the free constants of factor fi of `term`: a list of
    (position, map u in [0, 1] -> value). Frequencies and rates span gentle to steep on a log scale, switch centres and
    kinks the data range, poles log-spaced distances on either side of the data, Hill constants a log scale."""
    f = term.factors[fi]
    fam = f.family
    lin = lambda u, a, b: a + (b - a) * u
    logs = lambda u, a, b: 10.0 ** (a + (b - a) * u)
    def rng_of(v):
        lo, hi = float(v.min()), float(v.max())
        return lo, hi, max(hi - lo, 1e-9)
    if isinstance(f.var, tuple):
        j, k = f.var
        _, _, sj = rng_of(Z[:, j]); _, _, sk = rng_of(Z[:, k])
        wmap = lambda u: (1.0 if u >= 0.5 else -1.0) * logs(abs(2 * u - 1), -1.5, 1.5) * sj / sk
        if fam == "crecip":                                  # theta (w, b = 1 frozen, c)
            def cmap(u, th):
                uu = Z[:, j] + th[0] * Z[:, k]
                lo, hi, sp = rng_of(uu)
                dist = sp * logs((2 * u) % 1.0, -2, 2)
                return -(lo - dist) if u < 0.5 else -(hi + dist)
            return [(0, wmap), (2, cmap)]
        if fam == "chill":                                   # theta (w, b, c = 0 frozen by zsnap, n)
            def bmap(u, th):
                uu = (Z[:, j] + th[0] * Z[:, k]).abs()
                return logs(u, -2, 2) / max(float(uu.max()), 1e-9)
            return [(0, wmap), (1, bmap), (3, lambda u, th=None: lin(u, 0.5, 5.0))]
        return []
    j = f.var
    zlo, zhi, zs = rng_of(Z[:, j])
    xmed = float(X[:, j].abs().median()) + 1e-300
    if fam == "crecip":                                      # theta (b = 1 frozen, c); pole outside the data
        def cmap(u, th=None):
            dist = zs * logs((2 * u) % 1.0, -2, 2)
            return -(zlo - dist) if u < 0.5 else -(zhi + dist)
        return [(1, cmap)]
    if fam == "hill":                                        # theta (K, n)
        return [(0, None), (1, lambda u, th=None: lin(u, 0.5, 5.0))]
    if fam == "cexp":                                        # theta (b, c = 0 frozen)
        return [(0, lambda u, th=None: (1.0 if u >= 0.5 else -1.0) * logs(abs(2 * u - 1), -1.5, 1.5) / zs)]
    if fam == "ctanh":                                       # theta (b, c): steepness and centre
        return [(0, lambda u, th=None: (1.0 if u >= 0.5 else -1.0) * logs(abs(2 * u - 1), -1.0, 2.0) / zs),
                (1, lambda u, th: -th[0] * lin(u, zlo - 0.5 * zs, zhi + 0.5 * zs))]
    if fam == "csin":                                        # theta (w, phase)
        return [(0, lambda u, th=None: 2 * math.pi * logs(u, -1.5, 1.0) / zs), (1, lambda u, th=None: 2 * math.pi * u)]
    if fam == "cabs":                                        # theta (b = 1 frozen, c): kink inside the data
        return [(1, lambda u, th=None: -lin(u, zlo, zhi))]
    if fam == "pow":                                         # theta (p,)
        return [(0, lambda u, th=None: lin(u, -3.0, 3.0))]
    return []


def _sample_terms(terms, U, Z, X):
    """Map unit points U (S, h) to constant tensors for every free constant (natural ranges), and the slot list."""
    slots, maps = [], []
    for m, t in enumerate(terms):
        for fi, f in enumerate(t.factors):
            if "snap" in (f.meta or ()):
                continue
            for pos, mp in _samplers(t, fi, Z, X):
                slots.append((m, fi, pos)); maps.append(mp)
    return slots, maps


def _apply_samples(terms, slots, maps, U, Z, X):
    """Constant values (S, len(slots)) for unit points U, resolving maps that depend on earlier constants of the same
    factor (tanh centre on steepness, pair pole on weight) and Hill K on its exponent."""
    S = U.shape[0]
    P = np.zeros((S, len(slots)))
    for s_ in range(S):
        cur = {}
        for q, ((m, fi, pos), mp) in enumerate(zip(slots, maps)):
            th = list(terms[m].factors[fi].theta)
            for (mm, ff, pp), v in cur.items():
                if mm == m and ff == fi:
                    th[pp] = v
            if mp is None:                                   # Hill K: half-saturation log-uniform around the data
                n = th[1]
                for (mm, ff, pp), v in cur.items():
                    if mm == m and ff == fi and pp == 1:
                        n = v
                j = terms[m].factors[fi].var
                xmed = float(X[:, j].abs().median()) + 1e-300
                val = (10.0 ** (-1.5 + 3.0 * U[s_, q]) * xmed) ** n
            else:
                try:
                    val = mp(U[s_, q], th)
                except TypeError:
                    val = mp(U[s_, q])
            cur[(m, fi, pos)] = float(val)
            P[s_, q] = float(val)
    return P


def _batch_resid(terms, toks, slots, P, Z, X, y):
    """Residual norms of the structure for a batch of constant vectors P (S, n_slots) by variable projection."""
    S = P.shape[0]
    cols = []
    for m, t in enumerate(terms):
        th = []
        for fi, f in enumerate(t.factors):
            base = torch.tensor(f.theta, dtype=Z.dtype).reshape(-1, 1, 1).expand(-1, S, 1).clone()
            for s_, (mm, ff, aa) in enumerate(slots):
                if mm == m and ff == fi:
                    base[aa, :, 0] = P[:, s_]
            th.append(base)
        with torch.no_grad():
            v = DL.term_value(t, Z, X, th if t.factors else None)
        cols.append(v.expand(S, -1) if v.dim() == 2 else v.unsqueeze(0).expand(S, -1))
    Phi = torch.stack(cols, 2)                                   # (S, n, T)
    good = torch.isfinite(Phi).all(dim=(1, 2))
    Phi = torch.where(good[:, None, None], Phi, torch.zeros_like(Phi))
    try:
        sol = torch.linalg.lstsq(Phi, y.reshape(1, -1, 1).expand(S, -1, 1), driver="gelsd").solution
        r = (Phi @ sol).squeeze(2) - y.unsqueeze(0)
        rn = r.norm(dim=1)
    except RuntimeError:
        rn = torch.full((S,), float("inf"), dtype=Z.dtype)
    return torch.where(good & torch.isfinite(rn), rn, torch.full_like(rn, float("inf")))


def global_search(vocab, toks, terms, Z, X, y, rng):
    """Joint search over all free constants of `terms` in natural ranges from the data (Sobol points, then refinement
    around the best). Returns up to REFINE_TOP distinct term lists, best first."""
    from scipy.stats import qmc
    slots, maps = _sample_terms(terms, None, Z, X)
    if not slots:
        return []
    # Hill exponents first within a factor so that K can follow them: order maps so dependencies resolve
    order = sorted(range(len(slots)), key=lambda q: (slots[q][0], slots[q][1], 0 if maps[q] is not None else 1, slots[q][2]))
    slots = [slots[q] for q in order]; maps = [maps[q] for q in order]
    h = len(slots)
    sob = qmc.Sobol(h, scramble=True, seed=int(rng.integers(1 << 31)))
    U = sob.random(min(2048, max(SOBOL_N // 4, 256 * h)))
    P = _apply_samples(terms, slots, maps, U, Z, X)
    rn = _batch_resid(terms, toks, slots, torch.tensor(P), Z, X, y).numpy()
    for rnd in range(REFINE_ROUNDS):
        order_ = np.argsort(rn)
        top = []
        for k in order_:
            if not np.isfinite(rn[k]):
                break
            if all(np.max(np.abs(U[k] - U[j_])) > 0.02 for j_ in top):
                top.append(k)
            if len(top) >= REFINE_TOP:
                break
        if not top:
            break
        rad = 0.15 * 0.4 ** rnd
        newU = np.concatenate([np.clip(U[k] + rad * (rng.random((REFINE_N, h)) * 2 - 1), 0, 1) for k in top])
        newP = _apply_samples(terms, slots, maps, newU, Z, X)
        U = np.concatenate([U, newU]); P = np.concatenate([P, newP])
        rn = np.concatenate([rn, _batch_resid(terms, toks, slots, torch.tensor(newP), Z, X, y).numpy()])
    order_ = np.argsort(rn)
    picks, outs = [], []
    for k in order_:
        if not np.isfinite(rn[k]):
            break
        if all(np.max(np.abs(U[k] - U[j_])) > 0.02 for j_ in picks):
            picks.append(k)
            th = [[list(f.theta) for f in t.factors] for t in terms]
            for s_, (m, fi, a_) in enumerate(slots):
                th[m][fi][a_] = float(P[k, s_])
            outs.append(DL._with_thetas(terms, th))
        if len(picks) >= REFINE_TOP:
            break
    return outs


class NaturalBank:
    """v2's start grids: for every token, v1's grid placements plus a grid over the natural data-based ranges of its
    free constants (32 points for one constant, 10 x 10 for two, 6 per axis for three, 256 Sobol points beyond), as
    columns on the rate-data points. Same layout as structure_search.ColumnBank (bank[token] = (columns, thetas))."""

    def __init__(self, vocab, Zc, Xc, v1_bank=None, seed=0):
        from scipy.stats import qmc
        self.bank = {}
        for tok in vocab.tokens:
            if not vocab.factor_families(tok):
                C = DL.term_value(vocab.term(tok, ()), Zc, Xc).unsqueeze(1)
                self.bank[tok] = (C, [()])
                continue
            template = vocab.term(tok, vocab.starts(tok)[0])
            slots, maps = _sample_terms([template], None, Zc, Xc)
            order = sorted(range(len(slots)), key=lambda q: (slots[q][1], 0 if maps[q] is not None else 1, slots[q][2]))
            slots = [slots[q] for q in order]; maps = [maps[q] for q in order]
            h = len(slots)
            if h == 0:
                U = np.zeros((1, 0))
            elif h == 1:
                U = ((np.arange(32) + 0.5) / 32)[:, None]
            elif h == 2:
                g = (np.arange(10) + 0.5) / 10
                U = np.array(list(itertools.product(g, g)))
            elif h == 3:
                g = (np.arange(6) + 0.5) / 6
                U = np.array(list(itertools.product(g, g, g)))
            else:
                U = qmc.Sobol(h, scramble=True, seed=seed).random(256)
            P = _apply_samples([template], slots, maps, U, Zc, Xc) if h else np.zeros((1, 0))
            thetas = []
            for row in P:
                th = [list(f.theta) for f in template.factors]
                for s_, (m, fi, a_) in enumerate(slots):
                    th[fi][a_] = float(row[s_])
                thetas.append(tuple(tuple(t) for t in th))
            if v1_bank is not None and tok in v1_bank.bank:
                thetas = list(v1_bank.bank[tok][1]) + thetas
            cols, keep = [], []
            for th in thetas:
                try:
                    with torch.no_grad():
                        v = DL.term_value(vocab.term(tok, th), Zc, Xc)
                except (RuntimeError, ValueError, ZeroDivisionError):
                    continue
                if torch.isfinite(v).all() and float(v.abs().max()) < 1e12 and float(v.norm()) > 0:
                    cols.append(v); keep.append(th)
            if cols:
                self.bank[tok] = (torch.stack(cols, 1), keep)


# ---------------------------------------------------------------------------------------------------------------------
# (d) identifiability-gated snapping
# ---------------------------------------------------------------------------------------------------------------------

def exponent_uncertainty(terms, Z, X, y, floor):
    """Per free exponent (Hill-type n, power p): its standard uncertainty from the variable-projection Jacobian and the
    data's precision floor (noise per point = floor / sqrt(n)). Returns [(term, factor, position, value, sigma)]."""
    slots = _free_slots(terms)
    if not slots:
        return []
    th0 = [[list(f.theta) for f in t.factors] for t in terms]

    def resid(th):
        with torch.no_grad():
            Phi = DL.design(DL._with_thetas(terms, th), Z, X)
        return DL.lstsq_resid(Phi, y)[0]

    r0 = resid(th0)
    J = torch.empty(len(y), len(slots), dtype=Z.dtype)
    for s_, (m, fi, a_) in enumerate(slots):
        v = th0[m][fi][a_]
        h = 1e-6 * (1.0 + abs(v))
        th = [[list(x) for x in tt] for tt in th0]
        th[m][fi][a_] = v + h
        J[:, s_] = (resid(th) - r0) / h
    sig_y = floor / math.sqrt(len(y))
    H = J.T @ J
    try:
        cov = torch.linalg.pinv(H, rtol=1e-14) * sig_y ** 2
    except RuntimeError:
        return []
    out = []
    for s_, (m, fi, a_) in enumerate(slots):
        f = terms[m].factors[fi]
        if SS.SNAP_POS.get(f.family) == a_:
            var = float(cov[s_, s_])
            sigma = math.sqrt(var) if var > 0 and math.isfinite(var) else float("inf")
            if not torch.isfinite(H).all() or float(torch.linalg.matrix_rank(H)) < len(slots):
                sigma = float("inf")
            out.append((m, fi, a_, f.theta[a_], sigma))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------------------------------------------------

class FitterV2:
    """Fitter v2: ordinary fit = the v1 sweep fit plus the algebraic start (best of both); thorough fit = v1's thorough
    candidates, the algebraic start and the joint global search, polished, then identifiability-gated snapping judged by
    the held-back step-integral error."""

    wants_eq = True                                            # the search passes the equation index (dense rates)
    trace = None                                               # fit log (fitlog.py): a list collects every internal
    last = None                                                # start of a thorough fit; `last` its winner

    def precise_gate(self, toks, res, rel_rate, floor):
        """v2's precise-stage gate: the search's usual gate, or, for structures with two or more nonlinear pieces (whose
        cheap fit is unreliable), any fit explaining 99% of the rates."""
        if rel_rate <= SS.PRECISE_GATE and res <= SS.PRECISE_FLOOR * floor:
            return True
        return sum(1 for t in toks if t[0] != "mono") >= 2 and rel_rate <= 1e-2

    def __init__(self, bank, dense=None, seed=0, lm_iters=10, natural=None):
        self.v1 = SS.GridSweepFitter(bank)                       # v1's thorough candidates (original grids)
        self.sweep = SS.GridSweepFitter(natural) if natural is not None else self.v1   # v2's sweep (natural grids)
        self.dense = dense
        self.rng = np.random.default_rng(seed)
        self.lm_iters = lm_iters
        self.stats = {"ordinary": 0, "ordinary_seconds": 0.0, "algebraic_tried": 0, "algebraic_won": 0, "thorough": 0,
                      "thorough_seconds": 0.0, "global_won": 0, "exponents_checked": 0, "snaps_allowed": 0, "snaps_kept": 0,
                      "not_identifiable": 0}

    def _beam(self, vocab, toks, terms, Zc, Xc, y, width=3, keep=2):
        """Joint placements: each piece's `width` best grid placements (others at the sweep's), all combinations
        evaluated jointly in one batched solve; the `keep` best as term lists."""
        bank = self.sweep.bank.bank
        pieces = [k for k, t in enumerate(toks) if t in bank and bank[t][0].shape[1] > 1]
        if len(pieces) < 2:
            return []
        idx = {t: getattr(tm, "grid_index", None) for t, tm in zip(toks, terms)}
        for t in toks:
            if not isinstance(idx[t], int) or idx[t] >= bank[t][0].shape[1]:
                idx[t] = 0
        tops = []
        for k in pieces:
            t = toks[k]
            red = self.sweep._profile(t, list(toks), idx, y)
            tops.append(torch.argsort(red, descending=True)[:width].tolist())
        combos = list(itertools.product(*tops))
        if len(combos) > 729:
            return []
        cols = []
        for combo in combos:
            sel = dict(idx)
            for k, g in zip(pieces, combo):
                sel[toks[k]] = g
            cols.append(torch.stack([bank[t][0][:, sel[t]] for t in toks], 1))
        Phi = torch.stack(cols, 0)
        try:
            sol = torch.linalg.lstsq(Phi, y.reshape(1, -1, 1).expand(len(combos), -1, 1), driver="gelsd").solution
            rn = ((Phi @ sol).squeeze(2) - y.unsqueeze(0)).norm(dim=1)
        except RuntimeError:
            return []
        rn = torch.where(torch.isfinite(rn), rn, torch.full_like(rn, float("inf")))
        out = []
        for c in torch.argsort(rn)[:keep].tolist():
            sel = dict(idx)
            for k, g in zip(pieces, combos[c]):
                sel[toks[k]] = g
            tl = [vocab.term(t, bank[t][1][sel[t]]) for t in toks]
            for tm, t in zip(tl, toks):
                tm.__dict__["grid_index"] = sel[t]
            out.append(tl)
        return out

    def _rates(self, eq, Zc, Xc, y):
        if self.dense is not None and eq is not None:
            return self.dense.Z, self.dense.X, self.dense.dZ[:, eq]
        return Zc, Xc, y

    def fit(self, vocab, toks, Zc, Xc, y, warm=None, floor=0.0, basins=None, eq=None):
        t0 = time.perf_counter()
        terms, res, c, eff = self.sweep.fit(vocab, toks, Zc, Xc, y, warm, floor=floor, basins=basins)
        if math.isfinite(res) and res > 2.0 * floor:
            for cand in self._beam(vocab, toks, terms, Zc, Xc, y, keep=1):
                try:
                    t2, r2, c2 = SS.fast_refine(cand, Zc, Xc, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                eff["polished"] = eff.get("polished", 0) + 1
                if math.isfinite(r2) and r2 < res:
                    terms, res, c = t2, r2, c2
                    self.stats["beam_won"] = self.stats.get("beam_won", 0) + 1
        if any(_rational_factors(t, (1.0, 1.0)) is not None for t in toks) and math.isfinite(res) and res > 2.0 * floor:
            Zd, Xd, yd = self._rates(eq, Zc, Xc, y)
            self.stats["algebraic_tried"] += 1
            for cand in algebraic_start(vocab, toks, terms, Zd, Xd, yd)[:1]:
                try:
                    t2, r2, c2 = SS.fast_refine(cand, Zc, Xc, y, iters=self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                eff["polished"] = eff.get("polished", 0) + 1
                if math.isfinite(r2) and r2 < res:
                    terms, res, c = t2, r2, c2
                    self.stats["algebraic_won"] += 1
                    for tm, tk in zip(terms, toks):              # keep grid indices for warm starts
                        if not hasattr(tm, "grid_index"):
                            tm.__dict__["grid_index"] = None
        dt = time.perf_counter() - t0
        eff["seconds"] = dt
        self.stats["ordinary"] += 1
        self.stats["ordinary_seconds"] += dt
        return terms, res, c, eff

    def thorough(self, vocab, toks, Zc, Xc, y, warm, floor, current, eq=None, si_tr=None, si_va=None, i=None):
        t0 = time.perf_counter()
        toks = list(toks)
        best = current
        cands = []
        t1, r1, c1, _ = self.v1.fit(vocab, toks, Zc, Xc, y, {}, floor=floor, basins=SS.PRECISE_BASINS)   # own grid: no warm index
        cands.append((t1, r1, c1, "v1"))
        Zd, Xd, yd = self._rates(eq, Zc, Xc, y)
        for cand in algebraic_start(vocab, toks, current[0], Zd, Xd, yd)[:3]:
            cands.append((cand, None, None, "algebraic"))
        for cand in global_search(vocab, toks, current[0], Zc, Xc, y, self.rng):
            cands.append((cand, None, None, "global"))
        winner = "current"
        for terms, res, coefs, tag in cands:
            try:
                t2, r2, c2 = SS.fast_refine(terms, Zc, Xc, y, iters=2 * self.lm_iters, stop_below=1.5 * floor)
            except (RuntimeError, ValueError):
                if self.trace is not None:
                    self.trace.append((tag, terms, None, float("inf")))
                continue
            if self.trace is not None:
                self.trace.append((tag, terms, t2, r2))
            if math.isfinite(r2) and r2 < best[1]:
                best, winner = (t2, r2, c2), tag
        if winner == "global":
            self.stats["global_won"] += 1
        # (d) identifiability-gated snapping, judged by the held-back step-integral error
        terms, res, coefs = best
        unc = exponent_uncertainty(terms, Zc, Xc, y, floor)
        self.stats["exponents_checked"] += len(unc)
        alt = None
        for m, fi, a_, val, sigma in unc:
            if sigma > SNAP_SIGMA and round(val) != 0:
                self.stats["snaps_allowed"] += 1
                if not math.isfinite(sigma):
                    self.stats["not_identifiable"] += 1
                trial = list(terms)
                fs = list(trial[m].factors)
                th = list(fs[fi].theta); th[a_] = float(round(val))
                fs[fi] = DL.Factor(fs[fi].family, fs[fi].var, tuple(th), tuple(fs[fi].meta or ()) + ("snap",))
                trial[m] = DL.Term(trial[m].mono, tuple(fs), trial[m].level)
                try:
                    t3, r3, c3 = SS.fast_refine(trial, Zc, Xc, y, iters=2 * self.lm_iters, stop_below=1.5 * floor)
                except (RuntimeError, ValueError):
                    continue
                if self.trace is not None:
                    self.trace.append(("snap", trial, t3, r3))
                if math.isfinite(r3):
                    alt = (t3, r3, c3)
                    terms = t3                                   # further exponents are checked on the snapped fit
        if alt is not None and si_tr is not None and si_va is not None and i is not None:
            scores = []
            for cand in (best, alt):
                try:
                    tt, rt, ct = SS.refine_integral(cand[0], si_tr, i)
                    rv = SS.heldback_error(tt, ct, si_va, i) if ct is not None else float("inf")
                except (RuntimeError, ValueError):
                    rv = float("inf")
                scores.append(rv)
            if scores[1] < scores[0]:                            # the exactness test decides
                best = alt
                self.stats["snaps_kept"] += 1
        dt = time.perf_counter() - t0
        self.stats["thorough"] += 1
        self.stats["thorough_seconds"] += dt
        self.last = {"winner": winner, "snap_kept": alt is not None and best is alt}
        return best
