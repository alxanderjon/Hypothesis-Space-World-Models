"""Stage A of the redesign (owner, 9 November; research log "DreamCoder comparison and a staged redesign"): a learned
grammar of equation structures, the generative half of the prior that the move network (the recognition half) lacked.

What it is:
  - every term of an equation is read by its shape: the term with the equation's own variable written x1 and the other
    variables x2, x3, x4 in whichever order gives the smallest canonical key (which other variable is which does not
    matter); a shape is the same key in two, three or four variables, so what was learned on two-variable systems
    carries over to four-variable ones (a few dozen terms with a constant inside a logarithm keep canonical forms
    that depend on the variable order, so the shape counts m_s below are kept per own variable);
  - a component is the counts of shapes (and of the number of terms per equation) over a set of verified solutions;
    its probability of a concrete term k for an equation of a d-variable system is
        P(k) = (n_s / m_s + alpha / V_d) / (N_d + alpha),
    with n_s the count of k's shape s, m_s the number of terms of the d-variable search vocabulary with that shape,
    N_d the count of shapes that exist in d variables and V_d the vocabulary size: shapes seen often are likely,
    unseen terms keep a share alpha / (N_d + alpha) spread evenly (a Dirichlet-smoothed count, the Bayesian posterior
    predictive under a uniform base);
  - alpha is chosen per component from a fixed grid by how well the component predicts held-out parts of its own
    data: for the earlier component, each earlier domain from the other earlier domains (leave one domain out: the
    grammar will be used on a domain it has not seen, so its share for unseen terms is set by how often a new domain
    brought unseen terms before); for the newest component (or earlier experience with no domains, such as made-up
    systems), each equation from the others (leave one equation out); with fewer than MIN_LOO equations it is
    ALPHA_DEFAULT. The domains are those of the lineage's own training tasks, in the order it met them; nothing is
    known about the domain of a system being searched;
  - the grammar is the equal mixture of the components that have counts in d variables: "earlier" (everything before
    the newest domain) and "newest" (the newest domain's verified solutions), the same half-and-half rule as the
    balanced review rounds; with no counts it is uniform, so an empty grammar changes nothing.
Uses: log_ratio(d) for the search and training (log P(k) + log V_d: zero for the uniform prior), bits() for library
learning's description length (Stage C) and sample_terms() for dreams (Stage B).
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHAPES = HERE.parents[1] / "outputs/accumulating_learner/g1/shapes_d234.json"
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
ALPHA_DEFAULT = 10.0
MIN_LOO = 20
MAX_N = 6                                                       # search_g1.MAX_TERMS
_TABLES = {}


def shape_key(key, own, d):
    """The term's shape for the equation of variable x_{own+1} in d variables (module docstring)."""
    import domains_g1 as DG
    others = [j for j in range(d) if j != own]
    best = None
    for p in itertools.permutations(range(1, d)):
        perm = list(range(4))
        perm[own] = 0
        for r, j in enumerate(others):
            perm[j] = p[r]
        k = DG.relabelled(key, perm[:d])
        if k is not None and (best is None or k < best):
            best = k
    return best if best is not None else key


def _build():
    import search_g1 as SG
    out = {"shapes": [], "tables": {}}
    sid = {}
    for d in (2, 3, 4):
        order = SG.vocabulary_order(d)
        T = np.zeros((len(order), d), dtype=np.int32)
        for g, key in enumerate(order):
            for own in range(d):
                s = shape_key(key, own, d)
                if s not in sid:
                    sid[s] = len(out["shapes"])
                    out["shapes"].append(s)
                T[g, own] = sid[s]
        out["tables"][str(d)] = T.tolist()
    SHAPES.write_text(json.dumps(out))


def tables():
    """(shape keys, {d: (V_d, d) shape ids of the search vocabulary's terms per own variable}, {d: (d, S) counts m_s
    per own variable})."""
    if not _TABLES:
        if not SHAPES.exists():
            _build()
        raw = json.loads(SHAPES.read_text())
        _TABLES["shapes"] = raw["shapes"]
        _TABLES["T"] = {int(d): np.asarray(t, dtype=np.int64) for d, t in raw["tables"].items()}
        S = len(raw["shapes"])
        _TABLES["m"] = {d: np.stack([np.bincount(T[:, own], minlength=S) for own in range(d)]).astype(np.float64)
                        for d, T in _TABLES["T"].items()}
        _TABLES["gid"] = {}
    return _TABLES["shapes"], _TABLES["T"], _TABLES["m"]


def gid_of(d):
    if d not in _TABLES.get("gid", {}):
        import search_g1 as SG
        tables()
        _TABLES["gid"][d] = {k: g for g, k in enumerate(SG.vocabulary_order(d))}
    return _TABLES["gid"][d]


def equation_shapes(d, own, keys):
    """Shape ids of an equation's terms (keys of the d-variable search vocabulary; other keys get their shape computed)."""
    shapes, T, _ = tables()
    gid = gid_of(d)
    out = []
    for k in keys:
        if k in gid:
            out.append(int(T[d][gid[k], own]))
        else:
            s = shape_key(k, own, d)
            out.append(shapes.index(s) if s in shapes else -1)
    return out


class Component:
    """Counts of shapes and of terms per equation over a set of equations [(d, own, [shape ids])]."""

    def __init__(self, equations, alpha=None, groups=None):
        _, _, m = tables()
        S = m[2].shape[1]
        self.n = np.zeros(S)
        self.nterms = np.zeros(MAX_N + 1)
        self.eqs = [(int(d), int(own), [int(s) for s in ss if s >= 0]) for d, own, ss in equations]
        for d, own, ss in self.eqs:
            np.add.at(self.n, ss, 1.0)
            self.nterms[min(len(ss), MAX_N)] += 1
        self.held_out = None
        self.alpha = self._choose_alpha(groups) if alpha is None else float(alpha)

    def has(self, d):
        _, _, m = tables()
        return float(self.n[m[d][0] > 0].sum()) > 0

    def logp_shapes(self, d, own, alpha=None):
        """log P(k) for one term of each shape for the equation of variable own in d variables (per shape id; -inf
        where m_s = 0)."""
        _, _, m = tables()
        a = self.alpha if alpha is None else alpha
        ms = m[d][own]
        Vd = float(ms.sum())
        avail = ms > 0
        Nd = float(self.n[avail].sum())
        with np.errstate(divide="ignore"):
            return np.log(np.where(avail, (self.n / np.where(avail, ms, 1.0) + a / Vd) / (Nd + a), 0.0))

    def _choose_alpha(self, groups=None):
        """The grid value that best predicts held-out parts from the rest: each group (domain) from the other groups
        when there are two or more, else each equation from the other equations (module docstring)."""
        if len(self.eqs) < MIN_LOO:
            return ALPHA_DEFAULT
        _, _, m = tables()
        S = len(self.n)
        labels = list(groups) if groups is not None and len(set(groups)) >= 2 else list(range(len(self.eqs)))
        self.held_out = "domain" if groups is not None and len(set(groups)) >= 2 else "equation"
        parts = {}
        for (d, own, ss), g in zip(self.eqs, labels):
            parts.setdefault(g, []).append((d, own, ss))
        best, best_ll = ALPHA_DEFAULT, -math.inf
        for a in ALPHAS:
            ll = 0.0
            for g, eqs in parts.items():
                c = np.zeros(S)
                for d, own, ss in eqs:
                    np.add.at(c, ss, 1.0)
                rest = self.n - c
                for d, own, ss in eqs:
                    if not ss:
                        continue
                    ms = m[d][own]
                    Vd = float(ms.sum())
                    Nd = float(rest[ms > 0].sum())
                    p = (rest[ss] / np.maximum(ms[ss], 1.0) + a / Vd) / (Nd + a)
                    ll += float(np.log(np.maximum(p, 1e-300)).sum())
            if ll > best_ll:
                best, best_ll = a, ll
        return best

    def to_json(self):
        return {"eqs": self.eqs, "alpha": self.alpha, "held_out": self.held_out}


class Grammar:
    """The equal mixture of the components with counts in d variables (module docstring)."""

    def __init__(self, components):
        self.components = [c for c in components if c is not None and c.eqs]
        self._lr = {}

    @classmethod
    def fit(cls, earlier, newest, earlier_groups=None):
        """earlier, newest: lists of equations [(d, own, keys)] (verified solutions; for a made-up-systems lineage its
        made-up systems' structures count as its earlier experience). earlier_groups: the domain of each earlier
        equation's training task (None for made-up systems)."""
        comp = lambda eqs, g: Component([(d, own, equation_shapes(d, own, keys)) for d, own, keys in eqs], groups=g) \
            if eqs else None
        return cls([comp(earlier, earlier_groups), comp(newest, None)])

    def logp(self, d, own):
        """log P(k) per shape id for one term of the equation of variable own in d variables (the mixture)."""
        cs = [c for c in self.components if c.has(d)]
        if not cs:
            _, _, m = tables()
            with np.errstate(divide="ignore"):
                return np.where(m[d][own] > 0, -math.log(float(m[d][own].sum())), -np.inf)
        L = np.stack([c.logp_shapes(d, own) for c in cs])
        return np.logaddexp.reduce(L, axis=0) - math.log(len(cs))

    def log_ratio(self, d):
        """(V_d, d) float32: log P(term | own variable) + log V_d over the d-variable search vocabulary (0 = uniform)."""
        if d not in self._lr:
            _, T, m = tables()
            V = T[d].shape[0]
            self._lr[d] = np.stack([self.logp(d, own)[T[d][:, own]] + math.log(V) for own in range(d)], 1).astype(np.float32)
        return self._lr[d]

    def nterms_logp(self, d):
        """log P(number of terms = n), n = 1..MAX_N (Laplace-smoothed counts, the same mixture)."""
        cs = [c for c in self.components if c.has(d)]
        if not cs:
            return np.full(MAX_N, -math.log(MAX_N))
        P = np.mean([(c.nterms[1:] + 1.0) / (c.nterms[1:].sum() + MAX_N) for c in cs], axis=0)
        return np.log(P)

    def bits(self, d, own, keys):
        """Description length of an equation's term set in bits: its number of terms, then its terms as an unordered
        set (stage C's compression score)."""
        lp = self.logp(d, own)
        ss = equation_shapes(d, own, keys)
        n = len(keys)
        out = -self.nterms_logp(d)[min(n, MAX_N) - 1] - sum(float(lp[s]) for s in ss) - math.lgamma(n + 1)
        return out / math.log(2)

    def sample_terms(self, rng, d, own, n=None):
        """Indices into the d-variable search vocabulary for one equation (stage B's dreams): the number of terms from
        the grammar unless given, then distinct terms in proportion to the grammar."""
        if n is None:
            n = 1 + int(rng.choice(MAX_N, p=np.exp(self.nterms_logp(d))))
        p = np.exp(self.log_ratio(d)[:, own].astype(np.float64))
        p /= p.sum()
        return [int(x) for x in rng.choice(len(p), size=min(n, len(p)), replace=False, p=p)]

    def to_json(self):
        return {"components": [c.to_json() for c in self.components]}

    @classmethod
    def from_json(cls, obj):
        return cls([Component(c["eqs"], alpha=c["alpha"]) for c in obj["components"]])


def load(path):
    p = Path(str(path) + "_grammar.json")
    if not p.exists():
        return None, False
    obj = json.loads(p.read_text())
    return Grammar.from_json(obj["grammar"]), bool(obj.get("trained_with_grammar", False))


def save(path, grammar, trained):
    Path(str(path) + "_grammar.json").write_text(json.dumps({"grammar": grammar.to_json(), "trained_with_grammar": trained}))
