"""Trial 28 step 2: what the network sees, and the training history built from synthetic systems.

Features of a search state S in equation i (one row per vocabulary token, the same for training and for the search):
  - the token's type (piece family) and how it relates to equation i (own variable, multiplier, degree);
  - random per-sample codes of the token's variables, so the network can tell which tokens share a variable without
    being able to remember variable indices;
  - whether the token is in S;
  - this system's data evidence: the token's best partial R^2 with the rates alone and given S's fitted terms, and, for
    tokens in S, the share of the residual their removal would add;
  - the state's fit quality, size, the system's size, the number of bursts and the noise floor.
Labels: tokens to add (in the true structure, not in S) and tokens to remove (in S, not in the true structure).
"""

import math
import time

import numpy as np
import torch

import discovery_learner as DL
import practice_net as PN
import structure_search as SS
import synthetic28 as SY

FAMILIES = ["mono", "recip", "hill", "exp", "tanh", "sin", "abs", "pow", "pair_recip", "pair_hill", "prod_recip", "prod_hill"]
FAM_INDEX = {f: k for k, f in enumerate(FAMILIES)}
CODE_DIM = 8
N_STATIC, N_EVID, N_STATE = 9, 6, 5
N_FEAT = N_STATIC + CODE_DIM + N_EVID + N_STATE


def static_features(tok, i, d):
    """(family index, relation features) of a token for equation i."""
    fam = FAM_INDEX[SY.family(tok)]
    f = np.zeros(N_STATIC, dtype=np.float32)
    if tok[0] == "mono":
        e = tok[1]
        f[0], f[1], f[2], f[3] = sum(e) / 3, e[i] / 3, sum(1 for v in e if v) / 3, max(e) / 3
        f[8] = float(e[i] > 0)
        return fam, f
    if tok[0] in ("piece", "pair"):
        m = tok[3]
        vars_ = [tok[2]] if tok[0] == "piece" else list(tok[2])
        f[4] = float(i in vars_)
        f[5], f[6], f[7] = float(sum(m) == 0), float(m[i] > 0), float(sum(m) > 0 and m[i] == 0)
        f[8] = float(i in vars_ or m[i] > 0)
        f[0] = sum(m) / 3
        return fam, f
    vars_ = list(tok[2])
    f[4] = float(i in vars_)
    f[8] = f[4]
    return fam, f


def variables_of(tok, d):
    """Weights of the variables a token involves (pieces, multipliers, monomial exponents)."""
    w = np.zeros(d, dtype=np.float32)
    if tok[0] == "mono":
        w += np.asarray(tok[1], dtype=np.float32)
    elif tok[0] == "piece":
        w[tok[2]] += 1
        w += np.asarray(tok[3], dtype=np.float32)
    elif tok[0] == "pair":
        for j in tok[2]:
            w[j] += 1
        w += np.asarray(tok[3], dtype=np.float32)
    else:
        for j in tok[2]:
            w[j] += 1
    return w


class Context:
    """Everything the featurizer needs for one system's data (the same objects the search builds)."""

    def __init__(self, bursts, M, seed=1):
        self.view = view = PN.learner_view(bursts, seed=seed)
        self.Zc, self.G = view["Zc"], view["G"]
        self.Xc = self.Zc * view["sig"] + view["mu"]
        self.d = self.Zc.shape[1]
        self.M = M
        self.vocab = SS.Vocab28(self.d, self.Zc, self.Xc)
        self.cols = SS.CandidateColumns(self.vocab, self.Zc, self.Xc)
        self.bank = SS.ColumnBank(self.vocab, self.Zc, self.Xc)
        self.fitter = SS.GridSweepFitter(self.bank)
        self._g0 = {}

    @classmethod
    def from_parts(cls, view, vocab, cols, M):
        """A featurizer context on objects a learning run already built (no second vocabulary or column set)."""
        ctx = cls.__new__(cls)
        ctx.view, ctx.Zc, ctx.G = view, view["Zc"], view["G"]
        ctx.Xc = ctx.Zc * view["sig"] + view["mu"]
        ctx.d, ctx.M, ctx.vocab, ctx.cols = ctx.Zc.shape[1], M, vocab, cols
        ctx._g0 = {}
        return ctx

    def floor(self, i):
        y = self.G[:, i]
        yn = float(y.norm()) + 1e-300
        return max(float(self.view["noise"][i]) * math.sqrt(len(y)), 1e-15 * yn), yn

    def g0(self, i):
        if i not in self._g0:
            y = self.G[:, i]
            self._g0[i] = self.cols.add_gain(torch.zeros(len(y), 0, dtype=y.dtype), y).numpy()
        return self._g0[i]


def features(ctx, i, toks, terms, res, codes):
    """(family indices (V,), features (V, N_FEAT)) for state toks (fitted terms, rate residual res) of equation i."""
    vocab, d = ctx.vocab, ctx.d
    y = ctx.G[:, i]
    floor, yn = ctx.floor(i)
    node = {"toks": tuple(toks), "terms": terms, "coefs": None}
    gain, rem = SS._evidence(node, {"vocab": vocab, "y": y, "Zc": ctx.Zc, "Xc": ctx.Xc, "cols": ctx.cols})
    g0 = ctx.g0(i)
    inS = set(toks)
    V = len(vocab.tokens)
    fam = np.zeros(V, dtype=np.int64)
    F = np.zeros((V, N_FEAT), dtype=np.float32)
    rel = res / yn if math.isfinite(res) else 1.0
    state = np.array([math.log10(max(rel, 1e-16)) / 14, len(toks) / 8, d / 4, ctx.M / 16, math.log10(max(floor / yn, 1e-16)) / 14],
                     dtype=np.float32)
    for k, t in enumerate(vocab.tokens):
        fam[k], st = static_features(t, i, d)
        F[k, :N_STATIC] = st
        F[k, N_STATIC:N_STATIC + CODE_DIM] = variables_of(t, d) @ codes
        o = N_STATIC + CODE_DIM
        g, g_0 = float(gain[k]), float(g0[k])
        F[k, o:o + N_EVID] = [float(t in inS), g, -math.log10(max(1 - g, 1e-12)) / 12, g_0, -math.log10(max(1 - g_0, 1e-12)) / 12,
                              math.log10(1 + rem.get(t, 0.0)) / 12 if t in inS else 0.0]
        F[k, o + N_EVID:] = state
    return fam, F


def random_codes(rng, d):
    c = rng.standard_normal((d, CODE_DIM)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def states_for(rng, T, vocab_tokens, n_sub=2, n_wrong=1, n_over=1):
    """Training states for an equation with true structure T: proper subsets of T, subsets with 1-2 wrong tokens, and
    T with one wrong token."""
    T = list(T)
    others = [t for t in vocab_tokens if t not in T]
    out = []
    for _ in range(n_sub):
        k = int(rng.integers(len(T)))
        out.append(tuple(sorted([T[a] for a in rng.choice(len(T), k, replace=False)], key=str)))
    for _ in range(n_wrong):
        k = int(rng.integers(len(T)))
        base = [T[a] for a in rng.choice(len(T), k, replace=False)]
        w = [others[a] for a in rng.choice(len(others), int(rng.integers(1, 3)), replace=False)]
        out.append(tuple(sorted(base + w, key=str)))
    for _ in range(n_over):
        out.append(tuple(sorted(T + [others[int(rng.integers(len(others)))]], key=str)))
    return [s for s in out if len(s) <= SS.MAX_TERMS]


def examples_for(syn, M, rng, n_sub=2, n_wrong=1, n_over=1):
    """Training examples of one synthetic system at M bursts: list of dicts (eq, toks, fam, F, add, rem)."""
    ctx = Context(syn.bursts(M), M)
    codes = random_codes(rng, ctx.d)
    out = []
    for i in range(ctx.d):
        T = set(syn.tokens[i])
        if any(t not in ctx.vocab.index for t in T):
            continue                                           # (a true token outside this view's vocabulary)
        y = ctx.G[:, i]
        floor, _ = ctx.floor(i)
        for S in states_for(rng, sorted(T, key=str), ctx.vocab.tokens, n_sub=n_sub, n_wrong=n_wrong, n_over=n_over):
            terms, res, coefs, _ = ctx.fitter.fit(ctx.vocab, S, ctx.Zc, ctx.Xc, y, {}, floor=floor)
            fam, F = features(ctx, i, S, terms, res, codes)
            add = np.array([t in T and t not in S for t in ctx.vocab.tokens])
            rem = np.array([t in S and t not in T for t in ctx.vocab.tokens])
            out.append({"eq": i, "toks": S, "fam": fam, "F": F, "add": add, "rem": rem})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the training history on disk (shards of examples; deleted after training)
# ---------------------------------------------------------------------------------------------------------------------

M_DRAW = ((4, 0.2), (8, 0.6), (16, 0.2))


def history_shard(prefix, ks, path, mode="train"):
    """Examples of the accepted systems prefix-k (k in ks), each at one number of bursts drawn from M_DRAW, saved as
    one compressed shard: concatenated rows (F float16, fam, add, rem), offsets, and per-example metadata."""
    rows_F, rows_fam, rows_add, rows_rem, offs, meta = [], [], [], [], [0], []
    stats = {"systems": 0, "skipped_centred": 0, "rejected_draws": 0, "seconds": 0.0}
    for k in ks:
        t0 = time.time()
        syn, reasons = SY.draw_accepted(prefix, k, mode)
        stats["rejected_draws"] += len(reasons)
        if syn is None:
            continue
        rng = SY.rng_of(f"{prefix}-{k}-states")
        M = int(rng.choice([m for m, _ in M_DRAW], p=[p for _, p in M_DRAW]))
        if PN.learner_view(syn.bursts(M), seed=1)["centred"]:
            stats["skipped_centred"] += 1
            M = 8                                              # accepted systems are never centred at M = 8
        for ex in examples_for(syn, M, rng):
            rows_F.append(ex["F"].astype(np.float16)); rows_fam.append(ex["fam"].astype(np.int8))
            rows_add.append(ex["add"]); rows_rem.append(ex["rem"])
            offs.append(offs[-1] + len(ex["fam"]))
            meta.append((k, ex["eq"], syn.d, M, len(ex["toks"])))
        stats["systems"] += 1
        stats["seconds"] += time.time() - t0
    np.savez_compressed(path, F=np.concatenate(rows_F) if rows_F else np.zeros((0, N_FEAT), np.float16),
                        fam=np.concatenate(rows_fam) if rows_fam else np.zeros(0, np.int8),
                        add=np.concatenate(rows_add) if rows_add else np.zeros(0, bool),
                        rem=np.concatenate(rows_rem) if rows_rem else np.zeros(0, bool),
                        offs=np.asarray(offs, dtype=np.int64), meta=np.asarray(meta, dtype=np.int32))
    return stats


def load_shards(paths):
    """All examples of the given shards as a list of dicts (fam, F float32, add, rem, meta)."""
    out = []
    for p in paths:
        z = np.load(p)
        offs = z["offs"]
        for e in range(len(offs) - 1):
            a, b = offs[e], offs[e + 1]
            out.append({"fam": z["fam"][a:b].astype(np.int64), "F": z["F"][a:b].astype(np.float32), "add": z["add"][a:b],
                        "rem": z["rem"][a:b], "meta": tuple(int(v) for v in z["meta"][e])})
    return out
