"""Stage 2a of the general learner (research log, 7 November): dreaming for up to three variables.

dream_g1 with the Stage 2a changes:
  - every training example's system is relabelled at random (its variables permuted, with the structure relabelled to
    match), so the network does not depend on the order of the variables;
  - fantasies of two or three variables: each equation takes 1-4 terms drawn, half the time, from the terms of the
    lineage's verified solutions in proportion to their use (each term read relative to its equation: its own variable,
    then the others in index order), and otherwise from the whole search vocabulary with every term size equally
    likely; the made-up-systems control draws only from the latter (no usage). Coefficients, internal constants,
    initial states and acceptance as in dream_g1.
"""

import math

import numpy as np
import sympy as sp
import torch

import domains_g1 as DG
import dream_g1 as D1
import fitter_g1 as FG
import language_g1 as LG
import movenet_g2 as MN
import search_g1 as SG


def role_perm(own, d):
    """old variable j -> its position relative to equation `own`: own -> 0, the others in index order -> 1, 2."""
    others = [j for j in range(d) if j != own]
    perm = [0] * d
    for r, j in enumerate(others):
        perm[j] = r + 1
    return perm


def role_key(key, own, d):
    r = DG.relabelled(key, role_perm(own, d))
    return DG.representative(r, d) if r is not None else None


def make_examples(bursts, structures, seed, M=8, relabel=True):
    """Examples for one system (structures[i]: keys of equation i's structure, or None to skip), after a random
    relabelling of its variables."""
    rng = np.random.default_rng(seed)
    b = np.asarray(bursts, dtype=np.float64)
    d = b.shape[-1]
    structures = list(structures)
    if relabel and d > 1:
        pi = rng.permutation(d)                                  # new variable j is old variable pi[j]
        inv = [int(v) for v in np.argsort(pi)]                   # old variable i becomes new variable inv[i]
        b = b[..., pi]
        new = []
        for j in range(d):
            ks = structures[int(pi[j])]
            if not ks:
                new.append(None)
                continue
            r = [DG.relabelled(k, inv) for k in ks]
            new.append(None if any(x is None for x in r) else sorted({DG.representative(x, d) for x in r}))
        structures = new
    ctx = SG.system_context(b)
    vocab, W = ctx["vocab"], ctx["W"]
    fit = FG.make_fitter(ctx)
    out = []
    for i, keys in enumerate(structures):
        if not keys or any(k not in vocab.index for k in keys):
            continue
        T = sorted(vocab.index[k] for k in keys)
        y, yn, floor, tau = FG.equation_targets(ctx, i)
        c2 = dict(ctx, y=y, i=i)
        g0 = SG.evidence({"keys": (), "terms": [], "idx": ()}, c2)[0].astype(np.float32)
        hard = [int(h) for h in np.argsort(-g0)[:64]]
        for S in D1.states_for(rng, T, hard, vocab.V):
            ks = tuple(vocab.keys[x] for x in S)
            try:
                terms, res, coefs, _ = fit.fit(ks, W, y, {}, floor=floor, eq=i)
                gain, rem = SG.evidence({"keys": ks, "terms": terms}, c2)
            except (RuntimeError, ValueError, TypeError, ZeroDivisionError, IndexError):
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
            out.append({"eq": i, "d": d, "gids": vocab.gid.astype(np.int16), "F": F.astype(np.float16), "add": add,
                        "rem": remv, "inS": inS, "state": MN.state_features(rel, len(S), d, M, floor / yn)})
    return out


def draw_fantasy(seed, order, probs, d):
    """A fantasy system of d variables (domains_g1 format; constants as named parameters)."""
    rng = DG.rng_of(seed)
    Y = DG.Y3
    X = LG.variables(3)
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    for i in range(d):
        n = int(rng.integers(1, 5))
        picks = []
        while len(picks) < n:
            g = int(rng.choice(len(order), p=probs))
            if g not in picks:
                picks.append(g)
        others = [j for j in range(d) if j != i]
        actual = [i] + others                                    # role r -> actual variable
        e = 0
        for t, g in enumerate(picks):
            key = order[g]
            expr = LG.expr_of(key).xreplace({X[r]: Y[actual[r]] for r in range(d)})
            inf = FG.info(key)
            sub = {}
            if inf.k:
                Wp = torch.tensor(np.asarray(ic) * rng.uniform(0.5, 1.5, (64, d)))
                th = FG.natural_thetas(inf, rng.random((1, inf.k)), Wp[:, actual])[0]
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
    return {"seed": seed, "domain": "fantasy", "form": "fantasy", "d": d, "rhs": [sp.srepr(e) for e in rhs],
            "params": params, "ic": ic, "positive": False}


def accepted_fantasy(prefix, k, order, probs, d, tries=200):
    reasons = []
    for r in range(tries):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw_fantasy(seed, order, probs, d)
        try:
            return DG.build(desc), reasons
        except DG.Rejected as ex:
            reasons.append(str(ex).split(":")[0])
        except Exception as ex:
            reasons.append(type(ex).__name__)
    return None, reasons
