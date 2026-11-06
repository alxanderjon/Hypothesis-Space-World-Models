"""Stage 1 of the general learner (research log, 6 November): dreaming.

Training examples (search states with the answer known) come from two sources:
  replays    the learner's own verified solutions to its tasks so far (per equation, the simplest plausibly exact
             structure its searches found), on the task's own data;
  fantasies  new systems sampled from the current language: each equation takes 1-4 terms drawn, half the time, from
             the terms of the verified solutions so far in proportion to their use (each term read relative to its
             equation: its own variable and the other one), and otherwise from the whole vocabulary with every term
             size (number of nodes) equally likely; coefficients of random sign with magnitudes log-uniform in [0.1, 2];
             internal constants from their natural ranges (fitter_g1) on states around the initial state; initial states
             +-[0.5, 2.5]. Simulated and accepted with the domains' machinery and rules (domains_g1.build), so their
             answers are known.
States per equation (Trial 28's history format, history28.states_for): the empty state; 2 proper subsets of the
structure; 1 subset with 1-2 wrong terms; the structure with 1 wrong term. Wrong terms: half from the 64 terms with the
most evidence on their own, half from the whole vocabulary. Each example keeps every term of the system's vocabulary with
its evidence features (as the network sees them in the search).
"""

import collections
import json
import math

import numpy as np
import sympy as sp
import torch

import domains_g1 as DG
import fitter_g1 as FG
import language_g1 as LG
import movenet_g1 as MN
import search_g1 as SG

def role_key(key, own, d=2):
    """The key read relative to equation `own` (own variable -> x1): unchanged for own = 0, variables swapped for 1."""
    return key if own == 0 else DG.swapped(key)


def states_for(rng, T, hard, V, n_sub=2, n_wrong=1, n_over=1):
    T = list(T)
    Ts = set(T)
    hard = [h for h in hard if h not in Ts]

    def wrong(k):
        out = []
        while len(out) < k:
            w = int(hard[int(rng.integers(len(hard)))]) if hard and rng.random() < 0.5 else int(rng.integers(V))
            if w not in Ts and w not in out:
                out.append(w)
        return out
    out = [()]
    for _ in range(n_sub):
        k = int(rng.integers(len(T)))
        out.append(tuple(sorted(T[a] for a in rng.choice(len(T), k, replace=False))))
    for _ in range(n_wrong):
        k = int(rng.integers(len(T)))
        base = [T[a] for a in rng.choice(len(T), k, replace=False)]
        out.append(tuple(sorted(base + wrong(int(rng.integers(1, 3))))))
    for _ in range(n_over):
        out.append(tuple(sorted(T + wrong(1))))
    seen, uniq = set(), []
    for s in out:
        if len(s) <= SG.MAX_TERMS and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def make_examples(bursts, structures, seed, M=8):
    """Examples for one system: structures[i] is the list of keys of equation i's structure (None to skip)."""
    rng = np.random.default_rng(seed)
    ctx = SG.system_context(np.asarray(bursts))
    vocab, W = ctx["vocab"], ctx["W"]
    fit = FG.make_fitter(ctx)
    d = W.shape[1]
    out = []
    for i, keys in enumerate(structures):
        if not keys or any(k not in vocab.index for k in keys):
            continue
        T = sorted(vocab.index[k] for k in keys)
        y, yn, floor, tau = FG.equation_targets(ctx, i)
        c2 = dict(ctx, y=y, i=i)
        g0 = SG.evidence({"keys": (), "terms": [], "idx": ()}, c2)[0].astype(np.float32)
        hard = [int(h) for h in np.argsort(-g0)[:64]]
        for S in states_for(rng, T, hard, vocab.V):
            ks = tuple(vocab.keys[x] for x in S)
            try:
                terms, res, coefs, _ = fit.fit(ks, W, y, {}, floor=floor, eq=i)
                gain, rem = SG.evidence({"keys": ks, "terms": terms}, c2)
            except (RuntimeError, ValueError):
                continue
            inS = np.zeros(vocab.V, dtype=bool)
            inS[list(S)] = True
            rem_full = np.zeros(vocab.V, dtype=np.float32)
            rem_full[list(S)] = rem
            F = MN.evidence_features(gain.astype(np.float32), g0, inS, rem_full)
            add = np.zeros(vocab.V, dtype=bool)
            add[[t for t in T if t not in S]] = True
            remv = np.zeros(vocab.V, dtype=bool)
            remv[[x for x in S if x not in T]] = True
            rel = res / yn if math.isfinite(res) else 1.0
            out.append({"eq": i, "gids": vocab.gid.astype(np.int32), "F": F.astype(np.float16), "add": add, "rem": remv,
                        "inS": inS, "state": MN.state_features(rel, len(S), d, M, floor / yn)})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# fantasies
# ---------------------------------------------------------------------------------------------------------------------

def size_prior(order, V_info):
    """Every term size (nodes of its smallest generating tree) equally likely, then every term of that size."""
    sizes = np.array([V_info[k]["nodes"] for k in order])
    p = np.zeros(len(order))
    for s in np.unique(sizes):
        sel = sizes == s
        p[sel] = 1.0 / sel.sum()
    return p / p.sum()


def term_distribution(order, V_info, usage):
    """Probabilities over role-relative keys: half usage (counts of role-relative keys in verified solutions), half the
    size prior; the size prior alone when nothing has been used yet."""
    q = size_prior(order, V_info)
    if not usage:
        return q
    idx = {k: g for g, k in enumerate(order)}
    u = np.zeros(len(order))
    for k, c in usage.items():
        if k in idx:
            u[idx[k]] += c
    if u.sum() == 0:
        return q
    return 0.5 * u / u.sum() + 0.5 * q


def draw_fantasy(seed, order, probs):
    """A fantasy system description (domains_g1 format; constants as named parameters)."""
    rng = DG.rng_of(seed)
    Y = DG.Y
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(2)]
    for i in range(2):
        n = int(rng.integers(1, 5))
        picks = []
        while len(picks) < n:
            g = int(rng.choice(len(order), p=probs))
            if g not in picks:
                picks.append(g)
        e = 0
        for t, g in enumerate(picks):
            key = order[g]
            expr = LG.expr_of(key)
            X = LG.variables(2)
            expr = expr.xreplace({X[0]: Y[i], X[1]: Y[1 - i]})          # role-relative: x1 = own variable
            inf = FG.info(key)
            sub = {}
            if inf.k:
                W = torch.tensor(np.asarray(ic) * rng.uniform(0.5, 1.5, (64, 2)))
                Wr = W[:, [i, 1 - i]]
                th = FG.natural_thetas(inf, rng.random((1, inf.k)), Wr)[0]
                if not np.all(np.isfinite(th)):
                    th = np.ones(inf.k)
                for a in range(inf.k):
                    name = f"c{i}_{t}_{a}"
                    params[name] = float(th[a])
                    sub[sp.Symbol(f"K{a}", real=True)] = sp.Symbol(name)
            cname = f"a{i}_{t}"
            params[cname] = float(rng.choice([-1.0, 1.0]) * math.exp(rng.uniform(math.log(0.1), math.log(2.0))))
            e = e + sp.Symbol(cname) * expr.xreplace(sub)
        rhs.append(e)
    return {"seed": seed, "domain": "fantasy", "form": "fantasy", "d": 2, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False}


def accepted_fantasy(prefix, k, order, probs, tries=200):
    """The k-th accepted fantasy of a seed series (replacement seeds in order). Returns (G1 system or None, reasons)."""
    reasons = []
    for r in range(tries):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw_fantasy(seed, order, probs)
        try:
            return DG.build(desc), reasons
        except DG.Rejected as ex:
            reasons.append(str(ex).split(":")[0])
        except Exception as ex:                                       # sympy or numeric failure: a rejection
            reasons.append(type(ex).__name__)
    return None, reasons
