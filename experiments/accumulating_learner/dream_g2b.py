"""Stage B of the redesign (research log, "Stage B: dreams from each lineage's grammar"): fantasies drawn from the
lineage's learned grammar (grammar_g2), so dreams reflect accumulated experience, as DreamCoder's dreams are drawn from
its current library and weights.

dream_g2's fantasies with one change: each fantasy system is drawn whole either from the grammar (its number of terms
per equation and its terms, read relative to each equation) or, with probability SHARE (exploration), from the open
language as before Stage B for made-up systems (the size prior, 1-4 terms per equation). The branch is fixed per
fantasy before any retry, so the realized share does not drift with acceptance. Coefficients, internal constants,
initial states, the pre-screen and acceptance are dream_g2's.
"""

import math

import numpy as np
import sympy as sp
import torch

import domains_g1 as DG
import dream_g2 as DR
import fitter_g1 as FG
import language_g1 as LG

SHARE = 0.5
UNIFORM_N = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])        # 1-4 terms per equation (dream_g2)


def draw_fantasy(seed, order, probs, nterms, d):
    """dream_g2.draw_fantasy with the number of terms per equation drawn from nterms (probabilities of 1..6 terms)."""
    rng = DG.rng_of(seed)
    Y = DG.Y4
    X = LG.variables(4)
    params, rhs = {}, []
    ic = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)) for _ in range(d)]
    for i in range(d):
        n = 1 + int(rng.choice(len(nterms), p=nterms))
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


def accepted_fantasy(prefix, k, order, grammar_p, grammar_n, size_p, d, share=SHARE, tries=200):
    """dream_g2.accepted_fantasy for Stage B. grammar_p: the grammar's probabilities over role-relative keys of the
    d-variable vocabulary (own variable x1); grammar_n: its probabilities of 1..6 terms; size_p: the size prior.
    Returns (system or None, reasons, branch)."""
    branch = "open language" if DG.rng_of(f"{prefix}-{k}|branch").random() < share else "grammar"
    probs, nterms = (size_p, UNIFORM_N) if branch == "open language" else (grammar_p, grammar_n)
    reasons = []
    for r in range(tries):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = draw_fantasy(seed, order, probs, nterms, d)
        why = DR.prescreen(desc)
        if why is not None:
            reasons.append(why)
            continue
        try:
            return DG.build(desc), reasons, branch
        except DG.Rejected as ex:
            reasons.append(str(ex).split(":")[0])
        except Exception as ex:
            reasons.append(type(ex).__name__)
    return None, reasons, branch
