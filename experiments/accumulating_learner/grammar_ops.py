"""K0 of the compositional plan (owner, 11 November; research log "Compositional term building: assessment and staged
plan"): a grammar over operations, DreamCoder's contextual grammar for L1's terms.

A term's canonical tree is read relative to its equation: the equation's own variable ("own"), any other variable
("other"; which one does not matter), a constant ("K"), numbers ("n2", "n-1", ...), and the operations Add and Mul (as
nested pairs, Op(a0, Op(a1, ...))), Pow (base, exponent), exp, log, sin, Abs. Its probability is the product over its
nodes of P(child | parent, argument position) (the root's parent is "ROOT"). A component holds these counts (and the
number of terms per equation) over a set of equations, Dirichlet-smoothed per context with a uniform base over the
tokens; alpha is chosen from a grid by held-out likelihood as in grammar_g2 (leave one domain out for earlier
experience with two or more domains, else leave one equation out; fewer than MIN_LOO equations: ALPHA_DEFAULT). The
grammar mixes the components that have counts half and half at the term level, as grammar_g2 does.
"""

import math
from collections import Counter, defaultdict

import numpy as np

import language_g1 as LG

ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ALPHA_DEFAULT = 1.0
MIN_LOO = 20
MAX_N = 6
_TRIPLES = {}


def triples(key, own):
    """[(parent, argument position, child)] of the term's tree for the equation of variable x_{own+1}."""
    if (key, own) in _TRIPLES:
        return _TRIPLES[(key, own)]
    X = LG.variables(4)
    out = []

    def tok(e):
        if e.is_Symbol:
            if e in X:
                return "own" if X.index(e) == own else "other"
            return "K"
        if e.is_Number or e.is_NumberSymbol:
            return f"n{e}"
        return type(e).__name__

    def walk(e, parent, arg):
        t = tok(e)
        out.append((parent, arg, t))
        if e.is_Symbol or e.is_Number or e.is_NumberSymbol:
            return
        if e.is_Add or e.is_Mul:
            args = list(e.args)
            for j, a in enumerate(args):
                if j < len(args) - 1:
                    walk(a, t, 0)
                    if j < len(args) - 2:
                        out.append((t, 1, t))                    # the nested pair in argument 1
                else:
                    walk(a, t, 1)
            return
        for j, a in enumerate(e.args):
            walk(a, t, j)
    walk(LG.expr_of(key), "ROOT", 0)
    _TRIPLES[(key, own)] = out
    return out


class OpComponent:
    def __init__(self, equations, tokens, alpha=None, groups=None):
        """equations: [(own, keys)]; tokens: the token alphabet (the uniform base)."""
        self.tokens = list(tokens)
        self.eqs = [(int(own), list(keys)) for own, keys in equations]
        self.c, self.tot = defaultdict(Counter), Counter()
        self.nterms = np.zeros(MAX_N + 1)
        for own, keys in self.eqs:
            for k in keys:
                for p, a, ch in triples(k, own):
                    self.c[(p, a)][ch] += 1
                    self.tot[(p, a)] += 1
            self.nterms[min(len(keys), MAX_N)] += 1
        self.alpha = self._choose(groups) if alpha is None else float(alpha)

    def logp_term(self, key, own, alpha=None, minus=None):
        a = self.alpha if alpha is None else alpha
        u = 1.0 / len(self.tokens)
        lp = 0.0
        for p, arg, ch in triples(key, own):
            cnt, tot = self.c[(p, arg)][ch], self.tot[(p, arg)]
            if minus is not None:
                cnt -= minus[0].get((p, arg, ch), 0)
                tot -= minus[1].get((p, arg), 0)
            lp += math.log((cnt + a * u) / (tot + a))
        return lp

    def _choose(self, groups):
        if len(self.eqs) < MIN_LOO:
            return ALPHA_DEFAULT
        labels = list(groups) if groups is not None and len(set(groups)) >= 2 else list(range(len(self.eqs)))
        parts = defaultdict(list)
        for (own, keys), g in zip(self.eqs, labels):
            parts[g].append((own, keys))
        held = {}
        for g, eqs in parts.items():
            cc, tt = Counter(), Counter()
            for own, keys in eqs:
                for k in keys:
                    for p, a, ch in triples(k, own):
                        cc[(p, a, ch)] += 1
                        tt[(p, a)] += 1
            held[g] = (cc, tt)
        best, best_ll = ALPHA_DEFAULT, -math.inf
        for a in ALPHAS:
            ll = sum(self.logp_term(k, own, a, held[g]) for g, eqs in parts.items() for own, keys in eqs for k in keys)
            if ll > best_ll:
                best, best_ll = a, ll
        return best

    def nterms_logp(self):
        return np.log((self.nterms[1:] + 1.0) / (self.nterms[1:].sum() + MAX_N))


class OpGrammar:
    def __init__(self, components, tokens):
        self.components = [c for c in components if c is not None and c.eqs]
        self.tokens = list(tokens)

    @classmethod
    def fit(cls, earlier, newest, tokens, earlier_groups=None):
        """earlier, newest: [(own, keys)] (verified solutions; a made-up lineage's made-up structures as earlier)."""
        comp = lambda eqs, g: OpComponent(eqs, tokens, groups=g) if eqs else None
        return cls([comp(earlier, earlier_groups), comp(newest, None)], tokens)

    def logp_term(self, key, own):
        if not self.components:                                  # no experience: the uniform base at every node
            return len(triples(key, own)) * math.log(1.0 / len(self.tokens))
        L = [c.logp_term(key, own) for c in self.components]
        return float(np.logaddexp.reduce(L) - math.log(len(L)))

    def bits(self, own, keys):
        """Description length in bits of an equation's term set: its number of terms, then its terms (unordered)."""
        n = len(keys)
        if self.components:
            P = np.mean([np.exp(c.nterms_logp()) for c in self.components], axis=0)
            lpn = math.log(P[min(n, MAX_N) - 1])
        else:
            lpn = -math.log(MAX_N)
        return -(lpn + sum(self.logp_term(k, own) for k in keys) - math.lgamma(n + 1)) / math.log(2)


def alphabet(keys, d=4):
    """Every token in the trees of the given terms (the uniform base)."""
    toks = set()
    for k in keys:
        for own in range(d):
            for _, _, ch in triples(k, own):
                toks.add(ch)
    return sorted(toks)
