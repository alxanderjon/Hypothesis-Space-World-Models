"""Option (b), approved 1 November: experience as practice. After learning systems, the learner generates many variations
of its learned models (a term added or dropped, a variable swapped, a switch moved, other constants), simulates them on
the fly, and trains a recognition network to read the structure from a few short bursts. On a new system the network's
top suggestions are fitted exactly and ADDED to the candidates of the learner's own search (which always runs in full);
the learner's usual rule chooses (owner condition 1: suggestions can never rule out an answer the learner would find).

Pieces:
  BStruct           batched evaluation of one model structure with many constant draws (the learner's term language)
  variants          variations of a learned model; random toolkit systems for the generic (no-history) control
  simulate          bursts of 13 steps from starting states around the source system's observed region
  Vocab / features  per-equation candidate tokens (monomials, reciprocals, Hill switches x multipliers) and data features
  PracticeNet       transformer over the tokens of all equations: presence, and placement on the token's grid
  propose / fit     top suggestions -> learner models -> exact fit, held-back check, ties (the shared final step)
Everything is simulated in memory; nothing is written except trained weights and small JSON records.
"""

import copy
import itertools
import math

import numpy as np
import torch
import torch.nn as nn

import discovery_learner as DL

L = 13
RECIP_GRID = np.geomspace(0.02, 50.0, 12)          # pole distance beyond the observed range, in observed spans
HILL_GRID = np.geomspace(0.01, 100.0, 16)          # half-saturation point, in units of the variable's largest |z|
HILL_N = (1.0, 2.0, 3.0, 4.0)
N_GRID = len(RECIP_GRID) * 2                         # 24 placement cells (hill tokens use the first 16)


# ---------------------------------------------------------------------------------------------------------------------
# batched evaluation of a structure
# ---------------------------------------------------------------------------------------------------------------------

def _fac_b(fam, var, th, Z, X, meta):
    """factor value for a batch of constant draws: th (B, nt), Z, X (B, n, d) -> (B, n). Mirrors DL.factor_value."""
    t = [th[:, k:k + 1] for k in range(th.shape[1])]
    if fam in DL.LIB_FAMILIES:
        if isinstance(var, tuple):
            u, o = Z[..., var[0]] + t[0] * Z[..., var[1]], 1
        else:
            u, o = Z[..., var], 0
        if fam == "cheb":
            lo, hi = meta[0], meta[1]
            return DL.cheb_eval(2 * (u - lo) / (hi - lo) - 1, [c for c in t[o:]])
        if fam == "libcurve":
            coefs = torch.as_tensor(meta[2:], dtype=Z.dtype)
            return DL.cheb_eval(t[o] * u + t[o + 1], coefs)
        v = t[o] * u + t[o + 1]
        if fam == "crecip":
            return 1.0 / v
        if fam == "cexp":
            return torch.exp(v)
        if fam == "ctanh":
            return torch.tanh(v)
        if fam == "csin":
            return torch.sin(v)
        if fam == "cabs":
            return torch.abs(v)
        if fam == "cpow":
            return v ** t[o + 2]
        if fam == "chill":
            vp = v ** t[o + 2]
            return vp / (1.0 + vp)
        if fam == "chillr":
            return 1.0 / (1.0 + v ** t[o + 2])
    z, x = Z[..., var], X[..., var]
    if fam == "sin":
        return torch.sin(t[0] * z + t[1])
    if fam == "exp":
        return torch.exp(t[0] * z)
    if fam == "tanh":
        return torch.tanh(t[0] * z + t[1])
    if fam == "abs":
        return torch.abs(z - t[0])
    if fam == "recip":
        return 1.0 / (t[0] + z)
    if fam == "pow":
        return x ** t[0]
    if fam == "hill":
        xp = x ** t[1]
        return xp / (t[0] + xp)
    if fam == "hillr":
        return t[0] / (t[0] + x ** t[1])
    raise ValueError(fam)


class BStruct:
    """One structure (learner description) evaluated for B constant draws at once."""

    def __init__(self, desc, mu, sig):
        self.desc = desc
        self.d = len(desc)
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.sig = torch.as_tensor(sig, dtype=torch.float64)
        self.terms, P, self.kind = [], [], []
        for i, comp in enumerate(desc):
            for t in comp:
                co = len(P)
                P.append(float(t["coef"]))
                self.kind.append(("coef", None, None))
                facs = []
                for f in t["factors"]:
                    var = tuple(f[1]) if isinstance(f[1], list) else f[1]
                    off = len(P)
                    for q, v in enumerate(f[2]):
                        P.append(float(v))
                        self.kind.append(("theta", f[0], q - (1 if isinstance(var, tuple) else 0)))
                    facs.append((f[0], var, len(f[2]), off, tuple(f[3]) if len(f) > 3 else ()))
                self.terms.append((i, tuple(t["mono"]), facs, co))
        self.P0 = torch.tensor(P, dtype=torch.float64)

    def g(self, z, P):
        X = z * self.sig + self.mu
        cols = [torch.zeros(z.shape[:-1], dtype=z.dtype) for _ in range(self.d)]
        for i, mono, facs, co in self.terms:
            v = P[:, co:co + 1].expand(z.shape[0], z.shape[1])
            for k, e in enumerate(mono):
                if e:
                    v = v * z[..., k] ** e
            for fam, var, nt, off, meta in facs:
                v = v * _fac_b(fam, var, P[:, off:off + nt], z, X, meta)
            cols[i] = cols[i] + v
        return torch.stack(cols, -1)

    def with_params(self, p):
        """Description with one draw of constants."""
        desc, k = copy.deepcopy(self.desc), 0
        for comp in desc:
            for t in comp:
                t["coef"] = float(p[k]); k += 1
                for f in t["factors"]:
                    f[2] = [float(v) for v in p[k:k + len(f[2])]]
                    k += len(f[2])
        return desc


# ---------------------------------------------------------------------------------------------------------------------
# variations
# ---------------------------------------------------------------------------------------------------------------------

def _monomials(d, maxdeg):
    return [m for m in DL.monomials(d, maxdeg)]


def _random_piece(d, rng, zabs, lo, hi):
    """A toolkit-family piece with a placement relative to the observed region (scaled units)."""
    j = int(rng.integers(d))
    if rng.random() < 0.5:                                   # reciprocal: pole beyond the observed range
        span = max(hi[j] - lo[j], 1e-6)
        s = float(np.exp(rng.uniform(np.log(0.05), np.log(10.0))))
        u0 = lo[j] - s * span if rng.random() < 0.5 else hi[j] + s * span
        return ["crecip", j, [1.0, -u0], []]
    n = float(rng.choice(HILL_N))
    zh = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0)))) * max(zabs[j], 1e-6)
    return ["chill", j, [1.0 / zh, 0.0, n], ["snap", "zsnap"]]


def edit(desc, rng, zabs, lo, hi):
    """One structural variation: drop a term, add a toolkit term, swap a variable, change an exponent, or none."""
    desc = copy.deepcopy(desc)
    d = len(desc)
    r = rng.random()
    comps = [i for i in range(d) if len(desc[i]) >= 2]
    if r < 0.25 and comps:
        i = int(rng.choice(comps))
        desc[i].pop(int(rng.integers(len(desc[i]))))
    elif r < 0.55:
        i = int(rng.integers(d))
        mags = [abs(t["coef"]) for t in desc[i]] or [1.0]
        coef = float(np.median(mags) * np.exp(rng.normal(0, 1.0)) * rng.choice([-1, 1]))
        if rng.random() < 0.5:
            mono = list(_monomials(d, 2)[int(rng.integers(len(_monomials(d, 2))))])
            desc[i].append({"mono": mono, "factors": [], "level": 1, "coef": coef})
        else:
            mono = [0] * d
            if rng.random() < 0.4:
                mono[int(rng.integers(d))] = 1
            desc[i].append({"mono": mono, "factors": [_random_piece(d, rng, zabs, lo, hi)], "level": 1, "coef": coef})
    elif r < 0.7:
        i = int(rng.integers(d))
        if desc[i]:
            t = desc[i][int(rng.integers(len(desc[i])))]
            j_new = int(rng.integers(d))
            if t["factors"] and not isinstance(t["factors"][0][1], list) and rng.random() < 0.5:
                t["factors"][0][1] = j_new
            elif sum(t["mono"]):
                js = [k for k, e in enumerate(t["mono"]) if e]
                j = int(rng.choice(js))
                e = t["mono"][j]
                t["mono"][j] = 0
                t["mono"][j_new] += e
    elif r < 0.8:
        hs = [(i, a, k) for i, comp in enumerate(desc) for a, t in enumerate(comp) for k, f in enumerate(t["factors"])
              if f[0] in ("chill", "chillr")]
        if hs:
            i, a, k = hs[int(rng.integers(len(hs)))]
            desc[i][a]["factors"][k][2][-1] = float(rng.choice(HILL_N))
    return desc


def relabel(desc, perm):
    """Variables renamed: new index perm[old]."""
    d = len(desc)
    out = [None] * d
    for i, comp in enumerate(desc):
        new = []
        for t in comp:
            mono = [0] * d
            for k, e in enumerate(t["mono"]):
                mono[perm[k]] = e
            facs = []
            for f in t["factors"]:
                var = [perm[v] for v in f[1]] if isinstance(f[1], list) else perm[f[1]]
                facs.append([f[0], var, list(f[2]), list(f[3]) if len(f) > 3 else []])
            new.append({"mono": mono, "factors": facs, "level": t.get("level", 1), "coef": t["coef"]})
        out[perm[i]] = new
    return out


def draw_constants(bs, B, rng, sc=0.5, st=0.3):
    """B constant draws around the structure's own constants: coefficients x exp(N(0, sc)) (sign kept), free nonlinear
    constants x exp(N(0, st)); exponents, held-fixed constants and zero points at 0 unchanged."""
    P = bs.P0.unsqueeze(0).repeat(B, 1)
    for k, (kind, fam, q) in enumerate(bs.kind):
        if kind == "coef":
            P[:, k] *= torch.as_tensor(np.exp(rng.normal(0, sc, B)))
        else:
            fixed = (fam in ("crecip", "cpow", "cabs") and q == 0) or (fam == "cexp" and q == 1) or \
                    (fam in ("cpow", "chill", "chillr") and q == 2) or (fam in ("hill", "hillr") and q == 1) or \
                    (fam == "pow") or float(bs.P0[k]) == 0.0
            if not fixed:
                P[:, k] *= torch.as_tensor(np.exp(rng.normal(0, st, B)))
    return P


def random_toolkit_desc(d, rng, zabs, lo, hi):
    """Generic control: a random system from the toolkit vocabulary (no history)."""
    desc = []
    for i in range(d):
        comp = [{"mono": [int(k == i) for k in range(d)], "factors": [], "level": 1, "coef": -float(np.exp(rng.normal(-1, 1)))}]
        for _ in range(int(rng.integers(1, 4))):
            coef = float(np.exp(rng.normal(-1, 1)) * rng.choice([-1, 1]))
            if rng.random() < 0.5:
                mono = list(_monomials(d, 2)[int(rng.integers(len(_monomials(d, 2))))])
                comp.append({"mono": mono, "factors": [], "level": 1, "coef": coef})
            else:
                mono = [0] * d
                if rng.random() < 0.4:
                    mono[int(rng.integers(d))] = 1
                comp.append({"mono": mono, "factors": [_random_piece(d, rng, zabs, lo, hi)], "level": 1, "coef": coef})
        desc.append(comp)
    return desc


def scramble_desc(desc, rng):
    """Scrambled lessons: the same terms (shapes, constants) reassigned to random equations and variables, so the term
    statistics are kept and the learned structure is broken."""
    d = len(desc)
    terms = [copy.deepcopy(t) for comp in desc for t in comp]
    out = [[] for _ in range(d)]
    for t in terms:
        perm = list(rng.permutation(d))
        t = relabel([[t]] + [[] for _ in range(d - 1)], perm)[perm[0]][0]
        out[int(rng.integers(d))].append(t)
    for i in range(d):
        if not out[i]:
            out[i].append({"mono": [int(k == i) for k in range(d)], "factors": [], "level": 1, "coef": -0.1})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# simulation of bursts
# ---------------------------------------------------------------------------------------------------------------------

def simulate(bs, P, starts, nsub, rng, n_bursts=16):
    """Bursts (B, n_bursts, L, d) from starting states drawn around `starts` (scaled states of the source system).
    Draws that leave a sane region, stop moving, or are not finite are dropped (returned mask)."""
    B = P.shape[0]
    idx = rng.integers(len(starts), size=(B, n_bursts))
    z = torch.as_tensor(starts[idx]) * torch.as_tensor(np.exp(rng.normal(0, 0.2, (B, n_bursts, starts.shape[1]))))
    traj = [z]
    with torch.no_grad():
        for _ in range(L - 1):
            z = DL.rk8_torch(lambda zz: bs.g(zz, P), z, nsub)
            traj.append(z)
    T = torch.stack(traj, 2)                                  # (B, n_bursts, L, d)
    fin = torch.isfinite(T).all(dim=(1, 2, 3)) & (T.abs().amax(dim=(1, 2, 3)) < 1e3)
    move = (T[:, :, -1] - T[:, :, 0]).abs().amax(dim=(1, 2)) > 1e-6
    return T, fin & move


# ---------------------------------------------------------------------------------------------------------------------
# vocabulary, features and labels
# ---------------------------------------------------------------------------------------------------------------------

class Vocab:
    """Per-equation tokens: monomials of degree <= 3; reciprocal and Hill (n = 1..4) switches of one variable, times 1
    or one variable."""

    def __init__(self, d):
        self.d = d
        self.monos = [tuple(m) for m in DL.monomials(d, 3)]
        mults = [tuple([0] * d)] + [tuple(int(k == j) for k in range(d)) for j in range(d)]
        self.pieces = [(fam, j, m) for fam in ("recip",) + tuple(f"hill{int(n)}" for n in HILL_N) for j in range(d) for m in mults]
        self.tokens = [("mono", m) for m in self.monos] + [("piece",) + p for p in self.pieces]
        self.index = {t: k for k, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)


def _grid_columns(fam, j, Z, lo, hi, zabs):
    """Columns of one switch family on variable j at every grid placement: (n, cells), placements."""
    zj = Z[:, j:j + 1]
    if fam == "recip":
        span = max(hi[j] - lo[j], 1e-9)
        u0 = np.concatenate([lo[j] - RECIP_GRID * span, hi[j] + RECIP_GRID * span])
        return 1.0 / (zj - torch.as_tensor(u0)), u0
    n = float(fam[4:])
    zh = HILL_GRID * max(zabs[j], 1e-9)
    v = (zj.clamp(min=0) / torch.as_tensor(zh)) ** n
    return v / (1.0 + v), zh


def _gain(y, base, cols):
    """Partial R^2 of each column given the base columns: (cols ...) -> same shape."""
    Q = torch.linalg.qr(base)[0]
    r = y - Q @ (Q.T @ y)
    C = cols - Q @ (Q.T @ cols)
    num = (C * r.unsqueeze(1)).sum(0) ** 2
    den = (C * C).sum(0) * (r * r).sum() + 1e-300
    return (num / den).clamp(0, 1)


def _static_features(vocab):
    """Type and role descriptors of every token (independent of data): (d, V, 16)."""
    d = vocab.d
    S = torch.zeros(d, len(vocab), 16)
    for i in range(d):
        for k, m in enumerate(vocab.monos):
            S[i, k, 0] = 1.0
            S[i, k, 6] = m[i]
            others = sorted([e for q, e in enumerate(m) if q != i], reverse=True) + [0, 0, 0]
            S[i, k, 7:10] = torch.tensor(others[:3], dtype=torch.float32)
            S[i, k, 10] = sum(m)
        for k, (fam, j, mult) in enumerate(vocab.pieces):
            f = S[i, len(vocab.monos) + k]
            f[1 + (0 if fam == "recip" else int(float(fam[4:])))] = 1.0
            f[11] = float(j == i)
            f[12] = float(sum(mult) == 0)
            f[13] = float(sum(mult) > 0 and mult[i] == 1)
            f[14] = float(sum(mult) > 0 and mult[i] == 0)
    return S


_STATIC = {}


def features(vocab, Zc, G, noise):
    """Token features for one system (scaled states Zc, rate estimates G at burst interiors): per token its type/role
    descriptors and its data evidence, the partial R^2 over the placement grid given {1} and given {1, best monomial}."""
    d, n = vocab.d, Zc.shape[0]
    lo, hi = Zc.min(0).values.numpy(), Zc.max(0).values.numpy()
    zabs = Zc.abs().max(0).values.numpy()
    one = torch.ones(n, 1, dtype=Zc.dtype)
    monos = torch.stack([torch.prod(Zc ** torch.as_tensor(m, dtype=Zc.dtype), 1) for m in vocab.monos], 1)
    blocks, slots = [], []                     # piece columns of all tokens, and (token index, cells) for each block
    grids = {}
    for k, (fam, j, mult) in enumerate(vocab.pieces):
        if (fam, j) not in grids:
            grids[(fam, j)] = _grid_columns(fam, j, Zc, lo, hi, zabs)[0]
        C = grids[(fam, j)] * torch.prod(Zc ** torch.as_tensor(mult, dtype=Zc.dtype), 1, keepdim=True)
        blocks.append(C); slots.append((len(vocab.monos) + k, C.shape[1]))
    PC = torch.cat(blocks, 1)
    okc = torch.isfinite(PC).all(0)
    PC = torch.where(okc.unsqueeze(0), PC, torch.zeros_like(PC))
    if vocab.d not in _STATIC:
        _STATIC[vocab.d] = _static_features(vocab)
    F = torch.zeros(d, len(vocab), N_FEAT, dtype=torch.float32)
    F[..., :16] = _STATIC[vocab.d]
    for i in range(d):
        y = G[:, i]
        g0 = _gain(y, one, monos)
        kb = int(torch.argmax(g0))
        b1 = torch.cat([one, monos[:, kb:kb + 1]], 1)
        g1 = _gain(y, b1, monos)
        F[i, :len(vocab.monos), 16] = g0.float()
        F[i, :len(vocab.monos), 16 + N_GRID] = g1.float()
        p0 = (_gain(y, one, PC) * okc).float()
        p1 = (_gain(y, b1, PC) * okc).float()
        c = 0
        for tk, w in slots:
            F[i, tk, 16:16 + w] = p0[c:c + w]
            F[i, tk, 16 + N_GRID:16 + N_GRID + w] = p1[c:c + w]
            c += w
    F[..., 15] = float(math.log(max(float(noise.mean()), 1e-16)) / 30.0)
    return F


def _piece_token(f, sig_scale, lo, hi, zabs):
    """Map a learner factor (in the source scaling, multiplied by per-variable scale s = sig_target/sig_source) to a
    (family token, variable, grid cell) or None if it is not in the vocabulary."""
    fam, var, th = f[0], f[1], f[2]
    if isinstance(var, list):
        return None
    j = var
    s = sig_scale[j]
    span = max(hi[j] - lo[j], 1e-9)
    if fam in ("crecip", "cpow") and (fam == "crecip" or abs(th[2] + 1.0) < 1e-9):
        b, c = th[0] * s, th[1]
        if b == 0:
            return None
        u0 = -c / b
        name = "recip"
    elif fam == "recip":
        u0, name = -th[0] / s, "recip"
    elif fam in ("chill", "chillr"):
        n = th[2]
        if min(abs(n - h) for h in HILL_N) > 1e-6:
            return None
        b, c = th[0] * s, th[1]
        if b == 0:
            return None
        zh = (1.0 - c) / b
        name = f"hill{int(round(n))}"
    elif fam in ("hill", "hillr"):
        n = th[1]
        if min(abs(n - h) for h in HILL_N) > 1e-6:
            return None
        return ("hill" + str(int(round(n))), j, None, th[0] ** (1.0 / n))     # half-saturation in X units
    else:
        return None
    if name == "recip":
        if lo[j] <= u0 <= hi[j]:
            return None
        dist = (lo[j] - u0) / span if u0 < lo[j] else (u0 - hi[j]) / span
        k = int(np.argmin(np.abs(np.log(RECIP_GRID) - np.log(max(dist, 1e-12)))))
        return ("recip", j, k if u0 < lo[j] else k + len(RECIP_GRID), None)
    if zh <= 0:
        return None
    k = int(np.argmin(np.abs(np.log(HILL_GRID * max(zabs[j], 1e-9)) - np.log(zh))))
    return (name, j, k, None)


def labels(vocab, desc, sig_src, sig_tgt, Zc):
    """Presence (d, V) and placement cell (d, V; -1 where none) of a description in the target scaling."""
    d = vocab.d
    s = (np.asarray(sig_tgt) / np.asarray(sig_src))
    lo, hi = Zc.min(0).values.numpy(), Zc.max(0).values.numpy()
    zabs = Zc.abs().max(0).values.numpy()
    pres = torch.zeros(d, len(vocab))
    cell = torch.full((d, len(vocab)), -1, dtype=torch.long)
    for i, comp in enumerate(desc):
        for t in comp:
            mono = tuple(t["mono"])
            if not t["factors"]:
                k = vocab.index.get(("mono", mono))
                if k is not None:
                    pres[i, k] = 1
                continue
            if len(t["factors"]) != 1 or sum(mono) > 1:
                continue
            pt = _piece_token(t["factors"][0], s, lo, hi, zabs)
            if pt is None:
                continue
            name, j, kcell, xh = pt
            if xh is not None:                                # Hill in X units: half-saturation -> target z units
                zh = xh / sig_tgt[j]
                kcell = int(np.argmin(np.abs(np.log(HILL_GRID * max(zabs[j], 1e-9)) - np.log(max(zh, 1e-300)))))
            k = vocab.index.get(("piece", name, j, mono))
            if k is not None:
                pres[i, k] = 1
                cell[i, k] = kcell
    return pres, cell


def learner_view(bursts, seed=1, val_frac=0.25):
    """What the learner computes from bursts (M, L, d): split, scaling, training states and rate estimates."""
    Bt = torch.as_tensor(np.asarray(bursts, dtype=np.float64))
    M, Lb, d = Bt.shape
    rng = np.random.default_rng(seed)
    nval = max(1, int(round(val_frac * M)))
    perm = rng.permutation(M)
    val_idx, tr_idx = perm[:nval], (perm[nval:] if M > 1 else perm)
    allx = Bt.reshape(-1, d)
    mu, sig = allx.mean(0), allx.std(0) + 1e-300
    centred = bool((mu.abs() > DL.CENTER_RATIO * sig).any())
    mu = torch.where(mu.abs() > DL.CENTER_RATIO * sig, mu, torch.zeros_like(mu))
    Ztr, Zva = (Bt[tr_idx] - mu) / sig, (Bt[val_idx] - mu) / sig
    G, noise = DL.deriv_noise(Ztr)
    Zc = torch.stack([Ztr[:, c] for c in range(4, Lb - 4)], 1).reshape(-1, d)
    return {"mu": mu, "sig": sig, "centred": centred, "Ztr": Ztr, "Zva": Zva, "G": G, "noise": noise, "Zc": Zc,
            "z0": Ztr[:, :-1].reshape(-1, d), "z1": Ztr[:, 1:].reshape(-1, d),
            "zv0": Zva[:, :-1].reshape(-1, d), "zv1": Zva[:, 1:].reshape(-1, d)}


def dream_samples(desc, mu, sig, starts, nsub, rng, n_struct=60, n_draw=12, Ms=(4, 8, 16), scrambled=False,
                  generic=False, max_tokens_d=4):
    """Training samples (features, presence, cells, d, M) from variations of one learned model (or, for the controls,
    from scrambled versions of it or from random toolkit systems placed in the same observed region)."""
    d = len(desc)
    vocab = Vocab(d)
    zabs = np.abs(starts).max(0)
    lo, hi = starts.min(0), starts.max(0)
    out = []
    for _ in range(n_struct):
        if generic:
            base = random_toolkit_desc(d, rng, zabs, lo, hi)
        elif scrambled:
            base = scramble_desc(desc, rng)
        else:
            base = desc
        var = edit(base, rng, zabs, lo, hi)
        perm = list(rng.permutation(d))
        var = relabel(var, perm)
        st = starts[:, [perm.index(k) for k in range(d)]]          # new coordinate perm[k] = old coordinate k
        sig_p = np.asarray(sig)[[perm.index(k) for k in range(d)]]
        try:
            bs = BStruct(var, torch.zeros(d), torch.as_tensor(sig_p))
            P = draw_constants(bs, n_draw, rng)
            T, ok = simulate(bs, P, st, nsub, rng, n_bursts=max(Ms))
        except (RuntimeError, ValueError, IndexError, ZeroDivisionError):
            continue
        for b in torch.nonzero(ok).flatten().tolist():
            dd = bs.with_params(P[b])
            X = T[b] * torch.as_tensor(sig_p)                  # physical-like units of the source scaling
            for M in Ms:
                view = learner_view(X[:M].numpy())
                if view["centred"] or view["Zc"].shape[0] < 4:
                    continue
                F = features(vocab, view["Zc"], view["G"], view["noise"])
                pres, cell = labels(vocab, dd, sig_p, view["sig"].numpy(), view["Zc"])
                out.append({"F": F, "pres": pres, "cell": cell, "d": d, "M": M})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------------------------------

N_FEAT = 16 + 2 * N_GRID
N_VAR_EMB = 8


class PracticeNet(nn.Module):
    """Transformer over the tokens of all equations of a system. Variable identities enter through embeddings drawn
    at random per sample (consistent within a system), so the network is equivariant to relabelling variables while
    still able to relate tokens about the same variables (e.g. one switch in two equations)."""

    def __init__(self, width=128, layers=3, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N_FEAT + 4 * N_VAR_EMB, width), nn.GELU(), nn.Linear(width, width))
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.pres = nn.Linear(width, 1)
        self.cell = nn.Linear(width, N_GRID)

    def forward(self, F, V, mask):
        h = self.embed(torch.cat([F, V], -1))
        h = self.tf(h, src_key_padding_mask=~mask)
        return self.pres(h).squeeze(-1), self.cell(h)


def var_features(vocab, rng_t):
    """Random variable embeddings for one system: (d, V, 4 * N_VAR_EMB) [equation var, piece var, multiplier var,
    monomial variables weighted by exponent]."""
    d = vocab.d
    E = torch.randn(d, N_VAR_EMB, generator=rng_t)
    out = torch.zeros(d, len(vocab), 4 * N_VAR_EMB)
    for i in range(d):
        for k, tok in enumerate(vocab.tokens):
            out[i, k, :N_VAR_EMB] = E[i]
            if tok[0] == "mono":
                out[i, k, 3 * N_VAR_EMB:] = sum(e * E[q] for q, e in enumerate(tok[1]) if e) if sum(tok[1]) else 0
            else:
                _, fam, j, mult = tok
                out[i, k, N_VAR_EMB:2 * N_VAR_EMB] = E[j]
                if sum(mult):
                    out[i, k, 2 * N_VAR_EMB:3 * N_VAR_EMB] = E[mult.index(1)]
    return out


def collate(samples, rng_t):
    """Pad samples of possibly different dimension into one batch of token sequences."""
    vocabs = {}
    seqs = []
    for s in samples:
        v = vocabs.setdefault(s["d"], Vocab(s["d"]))
        Vf = var_features(v, rng_t)
        seqs.append((s["F"].reshape(-1, N_FEAT), Vf.reshape(-1, 4 * N_VAR_EMB), s["pres"].reshape(-1), s["cell"].reshape(-1)))
    n = max(x[0].shape[0] for x in seqs)
    B = len(seqs)
    F = torch.zeros(B, n, N_FEAT); V = torch.zeros(B, n, 4 * N_VAR_EMB)
    P = torch.zeros(B, n); C = torch.full((B, n), -1, dtype=torch.long); Mk = torch.zeros(B, n, dtype=torch.bool)
    for b, (f, vf, p, c) in enumerate(seqs):
        k = f.shape[0]
        F[b, :k], V[b, :k], P[b, :k], C[b, :k], Mk[b, :k] = f, vf, p, c, True
    return F, V, P, C, Mk


def train(samples, device="cuda:0", epochs=30, batch=32, lr=3e-4, seed=0, log=None):
    """Train a PracticeNet on dream samples. Returns the network (on CPU) and the last epoch's mean loss."""
    torch.manual_seed(seed)
    rng_t = torch.Generator().manual_seed(seed)
    net = PracticeNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = sum(float(s["pres"].sum()) for s in samples)
    tot = sum(float(s["pres"].numel()) for s in samples)
    pw = torch.tensor(min(50.0, (tot - pos) / max(pos, 1.0)), device=device)
    order = np.random.default_rng(seed)
    last = float("nan")
    for ep in range(epochs):
        idx = order.permutation(len(samples))
        # batches of one dimension at a time keep padding small
        by_d = {}
        for k in idx:
            by_d.setdefault(samples[k]["d"], []).append(k)
        batches = [v[i:i + batch] for v in by_d.values() for i in range(0, len(v), batch)]
        order.shuffle(batches)
        losses = []
        for bk in batches:
            F, V, P, C, Mk = (x.to(device) for x in collate([samples[k] for k in bk], rng_t))
            lp, lc = net(F, V, Mk)
            l1 = nn.functional.binary_cross_entropy_with_logits(lp[Mk], P[Mk], pos_weight=pw)
            sel = Mk & (C >= 0)
            l2 = nn.functional.cross_entropy(lc[sel], C[sel]) if sel.any() else torch.zeros((), device=device)
            loss = l1 + 0.3 * l2
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        last = float(np.mean(losses))
        if log:
            log(f"epoch {ep + 1}/{epochs}: loss {last:.4f}")
    return net.cpu().eval(), last


# ---------------------------------------------------------------------------------------------------------------------
# suggestions -> learner models
# ---------------------------------------------------------------------------------------------------------------------

def predict(net, vocab, Zc, G, noise, seed=0):
    F = features(vocab, Zc, G, noise)
    Vf = var_features(vocab, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        lp, lc = net(F.reshape(1, -1, N_FEAT), Vf.reshape(1, -1, 4 * N_VAR_EMB),
                     torch.ones(1, F.shape[0] * F.shape[1], dtype=torch.bool))
    d = vocab.d
    return torch.sigmoid(lp).reshape(d, len(vocab)), torch.softmax(lc, -1).reshape(d, len(vocab), N_GRID)


def _subsets(p, k_top=6, max_terms=8):
    """Most probable supports of one equation under independent presences p (V,): the MAP set, then sets obtained by
    flipping the most uncertain tokens. Returns [(log probability, token indices)]."""
    p = p.clamp(1e-6, 1 - 1e-6)
    order = torch.argsort(-p)
    cand = order[:12].tolist()
    base = {k for k in cand if p[k] > 0.5}
    lp_in, lp_out = torch.log(p), torch.log(1 - p)
    total_out = float(lp_out.sum())
    def score(S):
        return total_out + sum(float(lp_in[k] - lp_out[k]) for k in S)
    sets = {frozenset(base)}
    unc = sorted(cand, key=lambda k: abs(float(p[k]) - 0.5))[:6]
    for r in (1, 2):
        for flip in itertools.combinations(unc, r):
            S = set(base) ^ set(flip)
            if 0 < len(S) <= max_terms:
                sets.add(frozenset(S))
    if not base:
        sets.add(frozenset(cand[:1]))
    ranked = sorted(((score(S), sorted(S)) for S in sets if S), key=lambda x: -x[0])
    return ranked[:k_top]


def proposals(net, vocab, view, n_models=8, seed=0):
    """Top whole-model suggestions: per-equation supports combined by total log probability, placements at the most
    probable grid cell. Returns a list of (log probability, [[token, cell] per equation])."""
    pres, cells = predict(net, vocab, view["Zc"], view["G"], view["noise"], seed=seed)
    per = [_subsets(pres[i]) for i in range(vocab.d)]
    combos = [(0.0, [])]
    for i in range(vocab.d):
        combos = sorted(((s + si, sup + [Si]) for s, sup in combos for si, Si in per[i]), key=lambda x: -x[0])[:n_models]
    out = []
    for s, sup in combos:
        eqs = []
        for i, S in enumerate(sup):
            eqs.append([(vocab.tokens[k], int(torch.argmax(cells[i, k]))) for k in S])
        out.append((s, eqs))
    return out


def tokens_to_terms(eq, vocab, Zc):
    """A suggested support (tokens with grid cells) as learner terms with initial constants."""
    lo, hi = Zc.min(0).values.numpy(), Zc.max(0).values.numpy()
    zabs = Zc.abs().max(0).values.numpy()
    terms = []
    for tok, cell in eq:
        if tok[0] == "mono":
            terms.append(DL.Term(tok[1], (), 1))
            continue
        _, fam, j, mult = tok
        if fam == "recip":
            span = max(hi[j] - lo[j], 1e-9)
            u0 = lo[j] - RECIP_GRID[cell] * span if cell < len(RECIP_GRID) else hi[j] + RECIP_GRID[cell - len(RECIP_GRID)] * span
            f = DL.Factor("crecip", j, (1.0, float(-u0)), ("practice",))
        else:
            n = float(fam[4:])
            zh = HILL_GRID[min(cell, len(HILL_GRID) - 1)] * max(zabs[j], 1e-9)
            f = DL.Factor("chill", j, (float(1.0 / zh), 0.0, n), ("practice", "snap", "zsnap"))
        terms.append(DL.Term(tuple(mult), (f,), 1))
    return terms


def fit_suggestion(eqs, vocab, view, iters=40):
    """Exact fit of one suggestion: per-equation fit on the rate data (placements refined, coefficients projected out),
    then Levenberg-Marquardt on the step-to-step map and the held-back check. Returns (model, held-back error)."""
    Zc, G = view["Zc"], view["G"]
    Xc = Zc * view["sig"] + view["mu"]
    sups, cfs = [], []
    for i, eq in enumerate(eqs):
        terms = tokens_to_terms(eq, vocab, Zc)
        if not terms:
            sups.append([]); cfs.append(torch.zeros(0, dtype=torch.float64)); continue
        terms, _, c = DL.refine_support(terms, Zc, Xc, G[:, i], iters=15)
        sups.append(terms); cfs.append(c.detach().clone())
    model = DL.Model(sups, cfs, view["mu"], view["sig"])
    model, nsub, err, _ = DL.polish(model, view["z0"], view["z1"], iters=iters)
    if not math.isfinite(err):
        return model, float("inf")
    val, _, _ = DL.converged_val(model, view["z0"], view["zv0"], view["zv1"])
    return model, val


def choose(cands, n_points):
    """The learner's own rule over candidates [(model, held-back error, tag)]: log(max(error, tolerance)) plus the
    complexity cost per fitted constant (DL.set_complexity scale). Returns the chosen candidate."""
    DL.set_complexity(n_points)
    best, bs = None, float("inf")
    for m, v, tag in cands:
        if m is None or not math.isfinite(v):
            continue
        s = math.log(max(v, DL.VAL_TOL)) + DL.COMPLEXITY[0] * DL.n_fitted(m)
        if s < bs:
            best, bs = (m, v, tag), s
    return best
