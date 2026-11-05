"""Trial 31 (rules: research log, 5 November): a learned starting-value network for thorough fits.

Coordinates. Every free constant of a structure is expressed in fitter v2's natural-range unit cube: the coordinates
its joint global search samples (fitter30._samplers): switch centres across the data, rates and frequencies on log
scales relative to the data range, poles at log distances outside it, Hill constants around the data's median.

Labels. For a true structure, the generator's true constants (synthetic28.learner_term) mapped into the cube; for
any recorded fit that ended plausibly exact, its final constants. Valid answers also include symmetry images: constant
transformations that leave a factor's column unchanged up to scale (found numerically: sign flips, half-period phase
shifts) and permutations of interchangeable factors or terms. Constants the data cannot pin down (uncertainty in cube
coordinates above MASK_SIGMA, from the Jacobian at the label and the data's noise level) are masked.

Training. Winner-takes-all: an example's loss is the distance from the best of the K hypotheses to the nearest valid
answer; the other hypotheses get no loss, so alternative exact answers are never penalized.

Fitter v3 = fitter v2 plus the network's starts, added after v2's own candidates through v2's hook (never replacing
them; its own random stream), so its best fit never has a higher residual than v2's on the same inputs."""
import itertools
import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

import discovery_learner as DL
import fitter30 as F30
import structure_search as SS

K_HYP, D_MAX, N_TERMS_MAX = 16, 6, 10
FAMILIES = ("hill", "crecip", "cexp", "ctanh", "csin", "cabs", "pow", "chill")
CIRCULAR = {("csin", 1)}                                        # periodic constants (period 2 pi; u = phase / 2 pi)
MASK_SIGMA = 0.05
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------------------------
# unit-cube coordinates (the global search's)
# ---------------------------------------------------------------------------------------------------------------------

def slots_of(terms, Z, X):
    """Free constants of `terms` in the global search's order (Hill exponents before their K), with their maps."""
    slots, maps = F30._sample_terms(terms, None, Z, X)
    order = sorted(range(len(slots)), key=lambda q: (slots[q][0], slots[q][1], 0 if maps[q] is not None else 1, slots[q][2]))
    return [slots[q] for q in order], [maps[q] for q in order]


def _rng(v):
    lo, hi = float(v.min()), float(v.max())
    return lo, hi, max(hi - lo, 1e-9)


def _signed_log_inv(v, a, b, scale):
    """Inverse of u -> sign(u - 1/2) 10^(a + (b - a)|2u - 1|) / scale (u >= 1/2 positive)."""
    s = v * scale
    if s == 0 or not math.isfinite(s):
        return 0.5, False
    t = (math.log10(abs(s)) - a) / (b - a)
    return (0.5 + t / 2) if s > 0 else (0.5 - t / 2), 0.0 <= t <= 1.0


def _pole_inv(c, lo, hi, sp):
    """Inverse of the pole map: c = -(lo - dist) (u < 1/2) or -(hi + dist) (u >= 1/2), dist = sp 10^(-2 + 4 ((2u) mod 1))."""
    p = -c
    if p < lo:
        w = (math.log10((lo - p) / sp) + 2) / 4
        return w / 2, 0.0 <= w < 1.0
    if p > hi:
        w = (math.log10((p - hi) / sp) + 2) / 4
        return 0.5 + w / 2, 0.0 <= w < 1.0
    return 0.5, False                                           # a pole inside the data: not in the search space


def _forward(terms, slot, mp, u, th, X):
    """One slot's value at u given its factor's current constants (as fitter30._apply_samples)."""
    m, fi, pos = slot
    if mp is None:                                              # Hill K follows the exponent
        j = terms[m].factors[fi].var
        xmed = float(X[:, j].abs().median()) + 1e-300
        return (10.0 ** (-1.5 + 3.0 * u) * xmed) ** th[1]
    try:
        return float(mp(u, th))
    except TypeError:
        return float(mp(u))


def invert(terms, slots, Z, X, maps=None):
    """Cube coordinates of the terms' own constants: (u (h,) clipped to the cube, status (h,): 0 inside, 1 outside the
    natural range (u is the nearest point of the cube), 2 not representable (e.g. a pole inside the data)). With
    `maps`, a constant that depends on an earlier, clipped one of its factor is inverted given that clipped value."""
    us, ins = [], []
    cur = {}
    for q, (m, fi, pos) in enumerate(slots):
        f = terms[m].factors[fi]
        th = cur.setdefault((m, fi), [float(v) for v in f.theta])
        v, fam = float(f.theta[pos]), f.family
        if isinstance(f.var, tuple):
            j, k = f.var
            _, _, sj = _rng(Z[:, j]); _, _, sk = _rng(Z[:, k])
            if pos == 0:
                u, ok = _signed_log_inv(v, -1.5, 1.5, sk / sj)
            elif fam == "crecip" and pos == 2:
                lo, hi, sp = _rng(Z[:, j] + th[0] * Z[:, k])
                u, ok = _pole_inv(v, lo, hi, sp)
            elif fam == "chill" and pos == 1:
                mx = max(float((Z[:, j] + th[0] * Z[:, k]).abs().max()), 1e-9)
                u = (math.log10(v * mx) + 2) / 4 if v > 0 else 0.5
                ok = v > 0 and 0.0 <= u <= 1.0
            elif fam == "chill" and pos == 3:
                u = (v - 0.5) / 4.5; ok = 0.0 <= u <= 1.0
            else:
                u, ok = 0.5, False
        else:
            j = f.var
            zlo, zhi, zs = _rng(Z[:, j])
            if fam == "crecip":
                u, ok = _pole_inv(v, zlo, zhi, zs)
            elif fam == "hill" and pos == 0:
                n = th[1]
                xmed = float(X[:, j].abs().median()) + 1e-300
                ok = v > 0 and n > 0
                u = (math.log10(v ** (1.0 / n) / xmed) + 1.5) / 3 if ok else 0.5
                ok = ok and 0.0 <= u <= 1.0
            elif fam == "hill" and pos == 1:
                u = (v - 0.5) / 4.5; ok = 0.0 <= u <= 1.0
            elif fam == "cexp":
                u, ok = _signed_log_inv(v, -1.5, 1.5, zs)
            elif fam == "ctanh" and pos == 0:
                u, ok = _signed_log_inv(v, -1.0, 2.0, zs)
            elif fam == "ctanh" and pos == 1:
                x0 = -v / th[0] if th[0] != 0 else float("nan")
                u = (x0 - (zlo - 0.5 * zs)) / (2 * zs); ok = math.isfinite(u) and 0.0 <= u <= 1.0
            elif fam == "csin" and pos == 0:
                u = (math.log10(v * zs / TWO_PI) + 1.5) / 2.5 if v > 0 else 0.5
                ok = v > 0 and 0.0 <= u <= 1.0
            elif fam == "csin" and pos == 1:
                u, ok = (v % TWO_PI) / TWO_PI, True
            elif fam == "cabs":
                u = (-v - zlo) / zs; ok = 0.0 <= u <= 1.0
            elif fam == "pow":
                u = (v + 3.0) / 6.0; ok = 0.0 <= u <= 1.0
            else:
                u, ok = 0.5, False
        st = 0 if ok else (1 if (math.isfinite(u) and _representable(f, pos, v, th, Z)) else 2)
        u = min(max(u, 0.0), 1.0) if math.isfinite(u) else 0.5
        if st and maps is not None:                             # later constants of this factor see the clipped value
            try:
                th[pos] = _forward(terms, (m, fi, pos), maps[q], u, th, X)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        us.append(u)
        ins.append(st)
    return np.array(us), np.array(ins, dtype=np.int8)


def _representable(f, pos, v, th, Z):
    """Whether an out-of-range constant lies on the cube's side (so the nearest cube point is a sensible start)."""
    fam = f.family
    if fam == "crecip" and (pos == 1 or pos == 2):
        if isinstance(f.var, tuple):
            j, k = f.var
            lo, hi, _ = _rng(Z[:, j] + th[0] * Z[:, k])
        else:
            lo, hi, _ = _rng(Z[:, f.var])
        return not (lo <= -v <= hi)                             # a pole inside the data cannot be a start
    if (fam == "hill" and pos == 0) or (fam == "chill" and pos == 1) or (fam == "csin" and pos == 0):
        return v > 0
    return True


def constants_of(terms, slots, maps, U, Z, X):
    """Constants (S, h) for cube points U (S, h): the global search's own forward map."""
    return F30._apply_samples(terms, slots, maps, np.atleast_2d(U), Z, X)


def with_constants(terms, slots, P_row):
    th = [[list(f.theta) for f in t.factors] for t in terms]
    for s_, (m, fi, a_) in enumerate(slots):
        th[m][fi][a_] = float(P_row[s_])
    return DL._with_thetas(terms, th)


def circular_flags(terms, slots):
    return np.array([(terms[m].factors[fi].family, pos) in CIRCULAR for m, fi, pos in slots])


# ---------------------------------------------------------------------------------------------------------------------
# valid alternative answers (symmetry images) and identifiability
# ---------------------------------------------------------------------------------------------------------------------

def _parallel(a, b, tol=1e-10):
    na, nb = float(a.norm()), float(b.norm())
    if not (math.isfinite(na) and math.isfinite(nb)) or na == 0 or nb == 0:
        return False
    return abs(float(a @ b)) / (na * nb) > 1.0 - tol


def factor_symmetries(f, Z, X):
    """Constant transformations of one factor that leave its column unchanged up to a scalar (sign flips of any subset
    of its constants, with or without a half-period shift of one of them), found numerically on the data."""
    th0 = [float(v) for v in f.theta]
    base = DL.factor_value(f, Z, X, torch.tensor(th0, dtype=Z.dtype))
    out = []
    n = len(th0)
    for mask in range(1 << n):
        for shift in [None] + list(range(n)):
            if mask == 0 and shift is None:
                continue
            th = [(-v if (mask >> a_) & 1 else v) for a_, v in enumerate(th0)]
            if shift is not None:
                th[shift] = th[shift] + math.pi
            try:
                col = DL.factor_value(f, Z, X, torch.tensor(th, dtype=Z.dtype))
            except (RuntimeError, ValueError, ZeroDivisionError):
                continue
            if torch.isfinite(col).all() and _parallel(col, base):
                out.append(tuple(th))
    return out


def _shape(t):
    return (tuple(t.mono), tuple((f.family, f.var, tuple(f.meta or ())) for f in t.factors))


def images(terms, Z, X, cap=48):
    """Term lists equivalent to `terms` on the data: per-factor symmetries x permutations of interchangeable factors
    (same family, variable and meta within a term) and of interchangeable terms (same monomial and factor forms)."""
    per = []
    for t in terms:
        per.append([[tuple(float(v) for v in f.theta)] + factor_symmetries(f, Z, X) for f in t.factors])
    out, seen = [], set()

    def add(th_lists):
        key = tuple(tuple(tuple(round(v, 9) for v in th) for th in tl) for tl in th_lists)
        if key not in seen:
            seen.add(key); out.append(th_lists)

    choices = [list(itertools.product(*p)) if p else [()] for p in per]
    for combo in itertools.islice(itertools.product(*choices), cap):
        th_lists = [list(c) for c in combo]
        # permutations of interchangeable factors within a term
        variants = [th_lists]
        for m, t in enumerate(terms):
            groups = {}
            for fi, f in enumerate(t.factors):
                groups.setdefault((f.family, f.var, tuple(f.meta or ())), []).append(fi)
            for g in groups.values():
                if len(g) > 1:
                    new = []
                    for v_ in variants:
                        for perm in itertools.permutations(g):
                            w = [list(x) for x in v_]
                            w[m] = list(w[m])
                            for a_, b_ in zip(g, perm):
                                w[m][a_] = v_[m][b_]
                            new.append(w)
                    variants = new[:cap]
        # permutations of interchangeable terms
        tg = {}
        for m, t in enumerate(terms):
            if t.factors:
                tg.setdefault(_shape(t), []).append(m)
        for g in tg.values():
            if len(g) > 1:
                new = []
                for v_ in variants:
                    for perm in itertools.permutations(g):
                        w = list(v_)
                        for a_, b_ in zip(g, perm):
                            w[a_] = v_[b_]
                        new.append(w)
                variants = new[:cap]
        for v_ in variants:
            add(v_)
            if len(out) >= cap:
                break
        if len(out) >= cap:
            break
    return [DL._with_thetas(terms, [[list(th) for th in tl] for tl in v_]) for v_ in out]


def vp_residual_vectors(terms, slots, P, Z, X, y):
    """Residual vectors (S, n) of the structure for constant rows P (S, h), coefficients solved exactly."""
    S = P.shape[0]
    cols = []
    for m, t in enumerate(terms):
        th = []
        for fi, f in enumerate(t.factors):
            base = torch.tensor(f.theta, dtype=Z.dtype).reshape(-1, 1, 1).expand(-1, S, 1).clone()
            for s_, (mm, ff, aa) in enumerate(slots):
                if mm == m and ff == fi:
                    base[aa, :, 0] = torch.as_tensor(P[:, s_], dtype=Z.dtype)
            th.append(base)
        with torch.no_grad():
            v = DL.term_value(t, Z, X, th if t.factors else None)
        cols.append(v.expand(S, -1) if v.dim() == 2 else v.unsqueeze(0).expand(S, -1))
    Phi = torch.stack(cols, 2)
    sol = torch.linalg.lstsq(Phi, y.reshape(1, -1, 1).expand(S, -1, 1), driver="gelsd").solution
    return (y.unsqueeze(0) - (Phi @ sol).squeeze(2))


def slot_sigma(terms, slots, maps, u0, Z, X, y, noise, eps=1e-4):
    """Uncertainty of each cube coordinate at u0: noise * sqrt(diag((J^T J)^-1)), J the residual's Jacobian in cube
    coordinates (central differences)."""
    h = len(slots)
    U = np.repeat(u0[None, :], 2 * h, 0)
    steps = np.zeros(h)
    for s_ in range(h):
        up, dn = min(u0[s_] + eps, 1.0), max(u0[s_] - eps, 0.0)
        U[2 * s_, s_], U[2 * s_ + 1, s_] = up, dn
        steps[s_] = up - dn
    P = constants_of(terms, slots, maps, U, Z, X)
    try:
        R = vp_residual_vectors(terms, slots, P, Z, X, y).numpy()
    except RuntimeError:
        return np.full(h, np.inf)
    J = np.stack([(R[2 * s_] - R[2 * s_ + 1]) / max(steps[s_], 1e-12) for s_ in range(h)], 1)
    if not np.all(np.isfinite(J)):
        return np.full(h, np.inf)
    G = J.T @ J
    try:
        cov = np.linalg.pinv(G, rcond=1e-14, hermitian=True)
    except np.linalg.LinAlgError:
        return np.full(h, np.inf)
    d = np.diag(cov)
    sig = noise * np.sqrt(np.clip(d, 0, None))
    sig[~np.isfinite(sig)] = np.inf
    sig[np.diag(G) <= 1e-30 * max(float(np.max(np.diag(G))), 1e-300)] = np.inf   # no information at all
    return sig


def label(terms, Z, X, y, noise, extra=()):
    """Training label for one structure on one equation's data: slots, the valid answers in cube coordinates
    (A, h), which answers lie inside the search space, and the slot mask (identifiable constants)."""
    slots, maps = slots_of(terms, Z, X)
    if not slots:
        return None
    answers, inside = [], []
    for alt in [terms] + list(extra):
        for img in images(alt, Z, X):
            u, st = invert(img, slots, Z, X, maps)
            answers.append(u); inside.append(st)
    u0 = answers[0]
    sig = slot_sigma(terms, slots, maps, u0, Z, X, y, noise)
    return {"slots": slots, "answers": np.array(answers), "status": np.array(inside), "sigma": sig,
            "mask": sig <= MASK_SIGMA, "circular": circular_flags(terms, slots)}


# ---------------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------------

F_POINT = 2 * D_MAX + 1
F_SLOT = len(FAMILIES) + 4 + 2 * D_MAX + 2 + N_TERMS_MAX + D_MAX + D_MAX + 3
F_MONO = N_TERMS_MAX + D_MAX + D_MAX


def features(terms, slots, Zc, Xc, Zd, yd, eq):
    """Tokens of one example: points (dense rate samples, states normalized to each variable's observed range in the
    cheap-fit data, rates by their RMS), one per free constant, one per monomial term."""
    d = Zc.shape[1]
    lo = torch.stack([Zc[:, j].min() for j in range(d)]); sp = torch.stack([Zc[:, j].max() - Zc[:, j].min() for j in range(d)])
    sp = torch.clamp(sp, min=1e-9)
    zn = ((Zd - lo) / sp).float()
    yr = yd / (yd.pow(2).mean().sqrt() + 1e-300)
    pts = torch.zeros(Zd.shape[0], F_POINT)
    pts[:, :d] = zn
    pts[:, D_MAX:D_MAX + d] = 1.0
    pts[:, 2 * D_MAX] = yr.float().clamp(-50, 50)
    srows = []
    for (m, fi, pos) in slots:
        t = terms[m]; f = t.factors[fi]
        r = torch.zeros(F_SLOT)
        o = 0
        if f.family in FAMILIES:
            r[FAMILIES.index(f.family)] = 1.0
        o += len(FAMILIES)
        r[o + min(pos, 3)] = 1.0; o += 4
        if isinstance(f.var, tuple):
            r[o + f.var[0]] = 1.0; r[o + D_MAX + f.var[1]] = 1.0
        else:
            r[o + f.var] = 1.0
        o += 2 * D_MAX
        r[o + min(fi, 1)] = 1.0; o += 2
        r[o + min(m, N_TERMS_MAX - 1)] = 1.0; o += N_TERMS_MAX
        r[o:o + d] = torch.tensor([e / 3.0 for e in t.mono], dtype=torch.float32); o += D_MAX
        r[o + min(eq, D_MAX - 1)] = 1.0; o += D_MAX
        j = f.var[0] if isinstance(f.var, tuple) else f.var
        r[o] = float(torch.log10(Xc[:, j].abs().median() + 1e-300)) / 3.0
        r[o + 1] = float((f.family, pos) in CIRCULAR)
        r[o + 2] = float(isinstance(f.var, tuple))
        srows.append(r)
    mrows = []
    for m, t in enumerate(terms):
        if not t.factors:
            r = torch.zeros(F_MONO)
            r[min(m, N_TERMS_MAX - 1)] = 1.0
            r[N_TERMS_MAX:N_TERMS_MAX + d] = torch.tensor([e / 3.0 for e in t.mono], dtype=torch.float32)
            r[N_TERMS_MAX + D_MAX + min(eq, D_MAX - 1)] = 1.0
            mrows.append(r)
    return pts, torch.stack(srows), (torch.stack(mrows) if mrows else torch.zeros(0, F_MONO))


# ---------------------------------------------------------------------------------------------------------------------
# the network and its loss
# ---------------------------------------------------------------------------------------------------------------------

class StartNet(nn.Module):
    def __init__(self, width=128, layers=3, heads=4, k=K_HYP, dropout=0.0):
        super().__init__()
        self.k = k
        self.ep, self.es, self.em = nn.Linear(F_POINT, width), nn.Linear(F_SLOT, width), nn.Linear(F_MONO, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, k)

    def forward(self, pts, pm, sl, sm, mo, mm):
        x = torch.cat([self.ep(pts), self.es(sl), self.em(mo)], 1)
        mask = torch.cat([pm, sm, mm], 1)
        h = self.enc(x, src_key_padding_mask=~mask)
        P = pts.shape[1]
        return torch.sigmoid(self.head(h[:, P:P + sl.shape[1]]))      # (B, S, K)


def collate(batch):
    """batch: list of (pts, slots, monos, answers (A, S), answer_ok (A,), mask (S,), circular (S,))."""
    B = len(batch)
    P = max(b[0].shape[0] for b in batch); S = max(b[1].shape[0] for b in batch)
    Mo = max(max(b[2].shape[0] for b in batch), 1); A = max(b[3].shape[0] for b in batch)
    pts = torch.zeros(B, P, F_POINT); pm = torch.zeros(B, P, dtype=torch.bool)
    sl = torch.zeros(B, S, F_SLOT); sm = torch.zeros(B, S, dtype=torch.bool)
    mo = torch.zeros(B, Mo, F_MONO); mm = torch.zeros(B, Mo, dtype=torch.bool)
    ans = torch.zeros(B, A, S); aok = torch.zeros(B, A, dtype=torch.bool)
    msk = torch.zeros(B, S, dtype=torch.bool); circ = torch.zeros(B, S, dtype=torch.bool)
    for i, (p, s, m, a, ok, mk, ci) in enumerate(batch):
        pts[i, :p.shape[0]] = p; pm[i, :p.shape[0]] = True
        sl[i, :s.shape[0]] = s; sm[i, :s.shape[0]] = True
        if m.shape[0]:
            mo[i, :m.shape[0]] = m; mm[i, :m.shape[0]] = True
        ans[i, :a.shape[0], :a.shape[1]] = a; aok[i, :a.shape[0]] = ok
        msk[i, :mk.shape[0]] = mk; circ[i, :ci.shape[0]] = ci
    return pts, pm, sl, sm, mo, mm, ans, aok, msk, circ


def wta_loss(pred, ans, aok, msk, circ, beta=0.02):
    """Winner-takes-all over hypotheses and valid answers: per example, min over (answer, hypothesis) of the mean
    Huber distance over identifiable constants (circular for phases). Returns (loss, per-example distance)."""
    diff = (pred[:, None, :, :] - ans[:, :, :, None]).abs()            # (B, A, S, K)
    diff = torch.where(circ[:, None, :, None], torch.minimum(diff, 1.0 - diff), diff)
    hub = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    w = msk[:, None, :, None].float()
    d = (hub * w).sum(2) / w.sum(2).clamp(min=1.0)                     # (B, A, K)
    d = torch.where(aok[:, :, None], d, torch.full_like(d, float("inf")))
    best = d.flatten(1).amin(1)
    return best.mean(), best


# ---------------------------------------------------------------------------------------------------------------------
# fitter v3: v2 plus the network's starts
# ---------------------------------------------------------------------------------------------------------------------

class FitterV3(F30.FitterV2S):
    """The network's starting points added after the fitter's own candidates (never replacing them). Without
    `with_global` it is Trial 31's v3 (fitter v2 + learned picks); with it, the combined fitter: v2-strong's own global
    picks first (same random stream as v2-strong, so all of v2-strong's candidates are included), then the learned
    picks (their own stream)."""

    def __init__(self, bank, dense=None, seed=0, lm_iters=10, natural=None, net=None, keep=4, refine_rounds=2,
                 refine_n=32, radius=0.05, cloud=0, cloud_radius=0.1, prepolish=0, with_global=False):
        super().__init__(bank, dense, seed=seed, lm_iters=lm_iters, natural=natural, prepolish=prepolish, keep=keep)
        self.net, self.refine_rounds, self.refine_n, self.radius = net, refine_rounds, refine_n, radius
        self.cloud, self.cloud_radius = cloud, cloud_radius        # points sampled around every hypothesis first
        self.with_global = with_global
        self.stats.update(learned_tried=0, learned_won=0, learned_seconds=0.0)

    def _extra_candidates(self, vocab, toks, current, Zc, Xc, y, eq):
        outs = self._global_picks(vocab, toks, current, Zc, Xc, y, eq) if self.with_global else []
        return outs + [("learned", c) for c in self._learned_picks(vocab, toks, current, Zc, Xc, y, eq)]

    def _learned_picks(self, vocab, toks, current, Zc, Xc, y, eq):
        if self.net is None or self.dense is None or eq is None:
            return []
        t0 = time.perf_counter()
        terms = current[0]
        slots, maps = slots_of(terms, Zc, Xc)
        if not slots:
            return []
        rng = np.random.default_rng([33 if self.with_global else 31, int(eq), zlib.crc32(repr(tuple(toks)).encode())])
        if isinstance(self.net, str) and self.net == "global":     # control: v2's own global search (fresh stream),
            outs = []                                                # its best 4, with the same pre-polish
            for cand in F30.global_search(vocab, toks, terms, Zc, Xc, y, rng)[:self.keep]:
                if self.prepolish:
                    try:
                        cand = SS.fast_refine(cand, Zc, Xc, y, iters=self.prepolish, stop_below=1e-14 * float(y.norm()))[0]
                    except (RuntimeError, ValueError):
                        pass
                outs.append(cand)
            self.stats["learned_tried"] += 1
            self.stats["learned_seconds"] += time.perf_counter() - t0
            return outs
        if isinstance(self.net, str) and self.net == "random":     # control: random points instead of the network's
            U = rng.random((K_HYP, len(slots)))
        else:
            pts, sl, mo = features(terms, slots, Zc, Xc, self.dense.Z, self.dense.dZ[:, eq], eq)
            with torch.no_grad():
                U = self.net(pts[None], torch.ones(1, pts.shape[0], dtype=torch.bool), sl[None],
                             torch.ones(1, sl.shape[0], dtype=torch.bool), mo[None] if mo.shape[0] else torch.zeros(1, 1, F_MONO),
                             torch.ones(1, max(mo.shape[0], 1), dtype=torch.bool) if mo.shape[0] else torch.zeros(1, 1, dtype=torch.bool))
            U = U[0].T.double().numpy()                               # (K, h)
        if self.cloud:
            U = np.concatenate([U] + [np.clip(U + self.cloud_radius * (rng.random(U.shape) * 2 - 1), 0, 1)
                                      for _ in range(self.cloud)])
        P = constants_of(terms, slots, maps, U, Zc, Xc)
        rn = F30._batch_resid(terms, toks, slots, torch.tensor(P), Zc, Xc, y).numpy()
        for rnd in range(self.refine_rounds):
            order = np.argsort(rn)
            top = []
            for k in order:
                if not np.isfinite(rn[k]):
                    break
                if all(np.max(np.abs(U[k] - U[j_])) > 0.02 for j_ in top):
                    top.append(k)
                if len(top) >= self.keep:
                    break
            if not top:
                break
            rad = self.radius * 0.4 ** rnd
            newU = np.concatenate([np.clip(U[k] + rad * (rng.random((self.refine_n, U.shape[1])) * 2 - 1), 0, 1) for k in top])
            newP = constants_of(terms, slots, maps, newU, Zc, Xc)
            U = np.concatenate([U, newU]); P = np.concatenate([P, newP])
            rn = np.concatenate([rn, F30._batch_resid(terms, toks, slots, torch.tensor(newP), Zc, Xc, y).numpy()])
        outs, picks = [], []
        for k in np.argsort(rn):
            if not np.isfinite(rn[k]):
                break
            if all(np.max(np.abs(U[k] - U[j_])) > 0.02 for j_ in picks):
                picks.append(k)
                cand = with_constants(terms, slots, P[k])
                if self.prepolish:
                    try:
                        cand = SS.fast_refine(cand, Zc, Xc, y, iters=self.prepolish, stop_below=1e-14 * float(y.norm()))[0]
                    except (RuntimeError, ValueError):
                        pass
                outs.append(cand)
            if len(picks) >= self.keep:
                break
        self.stats["learned_tried"] += 1
        self.stats["learned_seconds"] += time.perf_counter() - t0
        return outs

    def thorough(self, *a, **kw):
        best = super().thorough(*a, **kw)
        if (self.last or {}).get("winner") == "learned":
            self.stats["learned_won"] += 1
        return best


# ---------------------------------------------------------------------------------------------------------------------
# training examples from a collected system (t31_collect.py's summary, known-structure and search logs)
# ---------------------------------------------------------------------------------------------------------------------

def make_example(terms, extra, Zc, Xc, y, noise, dense, eq):
    """(points, slot tokens, monomial tokens, answers (A, S), answer usable (A,), mask (S,), circular (S,)) or None."""
    lab = label(terms, Zc, Xc, y, noise, extra=extra)
    if lab is None or not lab["mask"].any():
        return None
    ok = np.array([bool((st[lab["mask"]] <= 1).all()) for st in lab["status"]])   # clipped answers allowed
    if not ok.any():
        return None
    pts, sl, mo = features(terms, lab["slots"], Zc, Xc, dense.Z, dense.dZ[:, eq], eq)
    return (pts, sl, mo, torch.tensor(lab["answers"], dtype=torch.float32), torch.tensor(ok), torch.tensor(lab["mask"]),
            torch.tensor(lab["circular"]))


def system_examples(rec, data_dir):
    """Examples of one collected system: each equation's true structure (answers: the generator's constants, recorded
    exact fits of it, and their images), and every other structure a recorded thorough or shadow fit made exact."""
    import ast
    import os
    import fitlog as FL
    import practice_net as PN
    import synthetic28 as SY
    paths = [os.path.join(data_dir, "logs", f"{rec['prefix']}-{rec['k']}_{p}.fitlog.npz") for p in ("known", "search")]
    runs = [FL.load([p], purpose="train")[0] for p in paths if os.path.exists(p)]
    if not runs:
        return []
    h = runs[0]["header"]
    view = PN.learner_view(runs[0]["bursts"], seed=h["seed"])
    Zc, G = view["Zc"], view["G"]; Xc = Zc * view["sig"] + view["mu"]; d = Zc.shape[1]
    vocab = SS.Vocab28(d, Zc, Xc)
    dense = F30.DenseRates(view["Ztr"], view["mu"], view["sig"])
    fits = {}
    for run in runs:
        for r in FL.rows(run):
            if r["kind"] in (0, 2) and r["exact"] and not r["failed"] and r["final"][0]:
                try:
                    tl = [vocab.term(t, th) for t, th in zip(r["tokens"], r["final"][0])]
                except (KeyError, ValueError, TypeError):
                    continue
                fits.setdefault((r["eq"], r["tokens"]), []).append(tl)
    out, true_S = [], {}
    for i, eq in enumerate(rec["labels"]["eqs"]):
        toks_c = sorted([(ast.literal_eval(t), cst) for t, cst in eq], key=lambda tc: str(tc[0]))
        S = tuple(t for t, _ in toks_c)
        true_S[i] = S
        terms = [SY.learner_term(t, cst, view["mu"], view["sig"]) for t, cst in toks_c]
        ex = make_example(terms, fits.get((i, S), []), Zc, Xc, G[:, i], float(view["noise"][i]), dense, i)
        if ex is not None:
            out.append(ex + ({"prefix": rec["prefix"], "k": rec["k"], "eq": i, "source": "true", "pieces": sum(1 for t in S if t[0] != "mono"),
                              "slots": int(ex[1].shape[0])},))
    for (i, S), tls in fits.items():
        if true_S.get(i) == S:
            continue
        ex = make_example(tls[0], tls[1:], Zc, Xc, G[:, i], float(view["noise"][i]), dense, i)
        if ex is not None:
            out.append(ex + ({"prefix": rec["prefix"], "k": rec["k"], "eq": i, "source": "recorded",
                              "pieces": sum(1 for t in S if t[0] != "mono"), "slots": int(ex[1].shape[0])},))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the screen (secondary; option C): which structures most deserve a thorough fit
# ---------------------------------------------------------------------------------------------------------------------

SCREEN_FAM = ("hill", "recip", "exp", "tanh", "sin", "abs", "pow")


def screen_features(S, gate_code, res, floor, scale, step):
    """What is known before a thorough fit: size, pieces, families, the gate's verdict, the cheap fit's distance from
    the floor and from the rates' scale, and how far the search has gone (no feature of the thorough-fit policy)."""
    fam = np.zeros(len(SCREEN_FAM) + 2)
    for t in S:
        if t[0] == "mono":
            fam[-2] += 1
        else:
            if t[1] in SCREEN_FAM:
                fam[SCREEN_FAM.index(t[1])] += 1
            fam[-1] += t[0] in ("pair", "prod")
    g = np.zeros(3); g[int(gate_code)] = 1
    pieces = sum(1 for t in S if t[0] != "mono")
    r = math.log10(max(res, 1e-300) / floor) if math.isfinite(res) else 20.0
    rs = math.log10(max(res, 1e-300) / scale) if math.isfinite(res) else 5.0
    return np.concatenate([[len(S), pieces], fam, g, [r, rs, math.log10(1 + step)]]).astype(np.float32)


class Screen:
    """Callable for EquationSearch(screen=...): a score (logit) per structure, higher = more likely exact."""

    def __init__(self, path):
        z = torch.load(path, weights_only=False)
        self.mu, self.sd = z["mu"], z["sd"]
        self.net = nn.Sequential(nn.Linear(len(self.mu), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.net.load_state_dict(z["state"]); self.net.eval()

    def __call__(self, S, gate_code, res, floor, scale, step):
        x = (screen_features(S, gate_code, res, floor, scale, step) - self.mu) / self.sd
        with torch.no_grad():
            return float(self.net(torch.tensor(x, dtype=next(self.net.parameters()).dtype)[None])[0, 0])


def generated_examples(bursts, eqs, meta, seed=1):
    """True-structure examples of a generated system from its own bursts and drawn constants (no search logs):
    the learner's view of the bursts exactly as at test time, labels from the generator's constants."""
    import ast
    import practice_net as PN
    import synthetic28 as SY
    view = PN.learner_view(bursts, seed=seed)
    Zc, G = view["Zc"], view["G"]; Xc = Zc * view["sig"] + view["mu"]
    dense = F30.DenseRates(view["Ztr"], view["mu"], view["sig"])
    out = []
    for i, eq in enumerate(eqs):
        toks_c = sorted([(ast.literal_eval(t), cst) for t, cst in eq], key=lambda tc: str(tc[0]))
        terms = [SY.learner_term(t, cst, view["mu"], view["sig"]) for t, cst in toks_c]
        ex = make_example(terms, [], Zc, Xc, G[:, i], float(view["noise"][i]), dense, i)
        if ex is not None:
            out.append(ex + (dict(meta, eq=i, source="true", pieces=sum(1 for t, _ in toks_c if t[0] != "mono"),
                                  slots=int(ex[1].shape[0])),))
    return out
