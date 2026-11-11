"""K0b of the compositional plan (owner's approval, 11 November; research log "Owner: K0b ..."): DreamCoder-style library
learning by compression over L1's term trees, and grammar_ops' operation grammar extended with the learned pieces.

A term's tree is grammar_ops' reading (relative to its equation: "own", "other", "K", numbers "n2", ...; Add and Mul as
nested pairs Op(a0, Op(a1, ...))). A piece is a subtree of some verified solution with up to MAX_HOLES of its proper,
non-nested subtrees replaced by holes; the grammar uses it as one operation whose holes are its arguments.

Two forms of grammar, each learning its own library: "ctx", grammar_ops' contextual grammar (a choice's distribution
depends on its parent and argument position; a piece's holes are contexts of their own), and "uni", the context-free
form of DreamCoder's prior (a choice's distribution depends only on the slot's type: an exponent of Pow, or an
expression; a hole takes the type of its place in the piece), under which a piece learned in one context is as likely
in any other.

Library learning is greedy compression (DreamCoder's objective, a description length): starting from no pieces, the
candidate that most reduces
    DL(library) + DL(solutions | library)
is added while it reduces it (at most MAX_PIECES). DL(library) is each piece's bits under the form's grammar without
pieces; DL(solutions | library) is the code length of the solutions' parses with the library under the form's grammar
with its parameters integrated out (a Dirichlet-multinomial per context, concentration ALPHA_LEARN, uniform base), so a
piece pays for any new contexts it opens. A term is parsed by taking at every node, from the root, the largest matching
piece (pieces of equal size in the order learned). Candidates: abstractions of subtrees found in at least MIN_USES
distinct solution terms; at each step the PRESELECT with the largest estimated saving (uses at the current parse's
nodes x the choices they save) are evaluated exactly.

The grammar with a library (LibComponent, LibGrammar) is grammar_ops' (counts over the parses, Dirichlet-smoothed toward
uniform, alpha by held-out likelihood, components mixed half and half) and scores a term by its best parse: at each node,
the more probable of its own operation and every matching piece (Viterbi; its probability is at most the term's total,
so the bits are conservative). The held-out likelihood that chooses alpha scores the held-out terms by their best parse
too; when it leaves out one domain at a time, the library is learned again without that domain, so the held-out domain
is new to the library as well as to the counts. With no library both forms' parse is the tree itself; the "ctx" form is
then grammar_ops' grammar exactly.
"""

import itertools
import math
from collections import Counter, defaultdict

import numpy as np

import language_g1 as LG

MAX_HOLES, MAX_PIECES, MIN_USES, PRESELECT = 2, 20, 2, 30
ALPHA_LEARN = 1.0
ALPHAS, ALPHA_DEFAULT, MIN_LOO, MAX_N = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0), 1.0, 20, 6    # as grammar_ops
HOLE = ("?",)
_TREES, _SIZE, _ABS = {}, {}, {}


def tree(key, own):
    """The term's tree as nested tuples (token, child, ...) for the equation of variable x_{own+1}: grammar_ops.triples'
    reading."""
    if (key, own) in _TREES:
        return _TREES[(key, own)]
    X = LG.variables(4)

    def tok(e):
        if e.is_Symbol:
            if e in X:
                return "own" if X.index(e) == own else "other"
            return "K"
        if e.is_Number or e.is_NumberSymbol:
            return f"n{e}"
        return type(e).__name__

    def build(e):
        t = tok(e)
        if e.is_Symbol or e.is_Number or e.is_NumberSymbol:
            return (t,)
        args = [build(a) for a in e.args]
        if (e.is_Add or e.is_Mul) and len(args) > 2:
            node = (t, args[-2], args[-1])
            for a in reversed(args[:-2]):
                node = (t, a, node)
            return node
        return (t, *args)
    out = build(LG.expr_of(key))
    _TREES[(key, own)] = out
    return out


def size(t):
    if t not in _SIZE:
        _SIZE[t] = 1 if t == HOLE else 1 + sum(size(c) for c in t[1:])
    return _SIZE[t]


def holes(t):
    return 1 if t == HOLE else sum(holes(c) for c in t[1:])


def gain(piece):
    """Choices a use of the piece saves: its nodes but one."""
    return 0 if piece is None else size(piece) - holes(piece) - 1


def subtrees(t):
    yield t
    for c in t[1:]:
        yield from subtrees(c)


def positions(t, path=()):
    """Paths to every proper subtree."""
    for j, c in enumerate(t[1:]):
        yield path + (j,)
        yield from positions(c, path + (j,))


def replace(t, path, new):
    if not path:
        return new
    kids = list(t[1:])
    kids[path[0]] = replace(kids[path[0]], path[1:], new)
    return (t[0], *kids)


def abstractions(t):
    """The pieces that match t at its root: t and every version of it with up to MAX_HOLES non-nested proper subtrees
    replaced by holes, keeping at least two nodes of its own."""
    if t in _ABS:
        return _ABS[t]
    out = {t}
    ps = list(positions(t))
    for n in range(1, MAX_HOLES + 1):
        for combo in itertools.combinations(ps, n):
            if any(a != b and b[:len(a)] == a for a in combo for b in combo):
                continue
            u = t
            for p in combo:
                u = replace(u, p, HOLE)
            out.add(u)
    _ABS[t] = tuple(a for a in out if size(a) - holes(a) >= 2)
    return _ABS[t]


def match(piece, t):
    """The subtrees bound to the piece's holes (in order), or None."""
    if piece == HOLE:
        return [t]
    if piece[0] != t[0] or len(piece) != len(t):
        return None
    out = []
    for a, b in zip(piece[1:], t[1:]):
        m = match(a, b)
        if m is None:
            return None
        out += m
    return out


def hole_slots(piece):
    """(parent token, argument position) of each of the piece's holes, in the order match binds them."""
    out = []

    def go(u):
        for j, c in enumerate(u[1:]):
            if c == HOLE:
                out.append((u[0], j))
            else:
                go(c)
    go(piece)
    return out


def context(form, library):
    """The context of a choice made below (parent, argument position): itself ("ctx"), or the slot's type ("uni")."""
    if form == "ctx":
        return lambda p, a: (p, a)
    slots = {name: hole_slots(piece) for name, piece in library}

    def f(p, a):
        if p in slots:
            p, a = slots[p][a]
        return "exponent" if (p == "Pow" and a == 1) else "expression"
    return f


def ordered(library):
    """The parsing order: largest first, pieces of equal size in the order learned."""
    return sorted(library, key=lambda x: -size(x[1]))


def parse(t, library):
    """A tree's parse with an ordered library: ([(parent, argument position, symbol)] like grammar_ops.triples, with
    pieces as symbols and their holes as argument positions; [(node, piece used there or None)])."""
    trip, visits = [], []

    def go(u, parent, arg):
        for name, piece in library:
            m = match(piece, u)
            if m is not None:
                trip.append((parent, arg, name))
                visits.append((u, piece))
                for h, sub in enumerate(m):
                    go(sub, name, h)
                return
        trip.append((parent, arg, u[0]))
        visits.append((u, None))
        for j, c in enumerate(u[1:]):
            go(c, u[0], j)
    go(t, "ROOT", 0)
    return trip, visits


def data_bits(c, V, alpha=ALPHA_LEARN):
    """Code length (bits) of the parses' choices with each context's distribution integrated out (Dirichlet-multinomial,
    concentration alpha, uniform base over V symbols): the product of the grammar's sequential predictive probabilities."""
    b, lg, out = alpha / V, math.lgamma, 0.0
    for cnt in c.values():
        n = sum(cnt.values())
        if n:
            out += lg(alpha + n) - lg(alpha) - sum(lg(x + b) - lg(b) for x in cnt.values() if x)
    return out / math.log(2)


def piece_bits(piece, c0, V0, key0):
    """A piece's description length (bits) under the form's grammar without pieces (plug-in, alpha 1); holes are free."""
    out = 0.0

    def go(u, parent, arg):
        nonlocal out
        if u == HOLE:
            return
        cnt = c0.get(key0(parent, arg), {})
        out -= math.log((cnt.get(u[0], 0) + 1.0 / V0) / (sum(cnt.values()) + 1.0))
        for j, ch in enumerate(u[1:]):
            go(ch, u[0], j)
    go(piece, "ROOT", 0)
    return out / math.log(2)


def _counts(parses, weights, key):
    c = defaultdict(Counter)
    for (trip, _), w in zip(parses, weights):
        for p, a, ch in trip:
            c[key(p, a)][ch] += w
    return c


def learn(terms, tokens, form="ctx"):
    """Greedy compression over the solutions' trees (a Counter of trees, with multiplicity) under the form's grammar:
    ([(name, piece)] in the order learned, [{piece, bits saved when added, uses in the final parses}])."""
    trees = list(terms)
    w = [terms[t] for t in trees]
    uses = defaultdict(set)
    for i, t in enumerate(trees):
        for s in set(subtrees(t)):
            if size(s) >= 3:
                for a in abstractions(s):
                    uses[a].add(i)
    cands = {a for a, s in uses.items() if len(s) >= MIN_USES}
    V0, key0 = len(tokens), context(form, [])
    library, lib_bits, log = [], 0.0, []
    parses = [parse(t, []) for t in trees]
    c = _counts(parses, w, key0)
    c0 = {k: dict(v) for k, v in c.items()}
    current = data_bits(c, V0)
    for _ in range(MAX_PIECES):
        score = Counter()
        chosen = {p for _, p in library}
        for (_, visits), wi in zip(parses, w):
            for u, q in visits:
                if size(u) < 3:
                    continue
                for a in abstractions(u):
                    if a in cands and a not in chosen and size(a) > (size(q) if q is not None else 0):
                        score[a] += wi * (gain(a) - gain(q))
        best = None
        for a, s in score.most_common(PRESELECT):
            if s <= 0:
                break
            lib2 = ordered(library + [(f"P{len(library)}", a)])
            key2 = context(form, lib2)
            new = {i: parse(trees[i], lib2) for i in uses[a]}
            c2 = defaultdict(Counter, {k: Counter(v) for k, v in c.items()})
            for i, pi in new.items():
                for p, a_, ch in parses[i][0]:
                    c2[key2(p, a_)][ch] -= w[i]
                for p, a_, ch in pi[0]:
                    c2[key2(p, a_)][ch] += w[i]
            pb = piece_bits(a, c0, V0, key0)
            dl = data_bits(c2, V0 + len(library) + 1) + lib_bits + pb
            if best is None or dl < best[0]:
                best = (dl, a, new, c2, pb)
        if best is None or best[0] >= current + lib_bits - 1e-9:
            break
        dl, a, new, c2, pb = best
        log.append({"piece": a, "saved": current + lib_bits - dl})
        library.append((f"P{len(library)}", a))
        for i, pi in new.items():
            parses[i] = pi
        c, lib_bits = c2, lib_bits + pb
        current = dl - lib_bits
    used = Counter()
    for (trip, _), wi in zip(parses, w):
        for _, _, ch in trip:
            used[ch] += wi
    for (name, _), x in zip(library, log):
        x["uses"] = used[name]
    return library, log


class LibComponent:
    """grammar_ops.OpComponent over the parses with a library, in either form; terms scored by their best parse."""

    def __init__(self, equations, library, tokens, alpha=None, groups=None, form="ctx"):
        self.library = ordered(library)
        self.tokens, self.form = list(tokens), form
        self.V = len(tokens) + len(library)
        self.key = context(form, self.library)
        self.root = self.key("ROOT", 0)
        self.eqs = [(int(own), list(keys)) for own, keys in equations]
        self._trip, self._best, self._matches = {}, {}, {}
        self.c, self.tot = defaultdict(Counter), Counter()
        self.nterms = np.zeros(MAX_N + 1)
        for own, keys in self.eqs:
            for k in keys:
                for ck, ch in self.triples(k, own):
                    self.c[ck][ch] += 1
                    self.tot[ck] += 1
            self.nterms[min(len(keys), MAX_N)] += 1
        self.alpha = self._choose(groups) if alpha is None else float(alpha)

    def triples(self, key, own):
        """[(context, symbol)] of the term's parse with the library."""
        if (key, own) not in self._trip:
            self._trip[(key, own)] = [(self.key(p, a), ch) for p, a, ch in parse(tree(key, own), self.library)[0]]
        return self._trip[(key, own)]

    def matches(self, t):
        if t not in self._matches:
            self._matches[t] = [(name, m) for name, piece in self.library for m in [match(piece, t)] if m is not None]
        return self._matches[t]

    def lp(self, ck, ch, alpha, minus=None):
        cnt, tot = self.c.get(ck, {}).get(ch, 0), self.tot.get(ck, 0)
        if minus is not None:
            cnt -= minus[0].get((ck, ch), 0)
            tot -= minus[1].get(ck, 0)
        return math.log((cnt + alpha / self.V) / (tot + alpha))

    def _parse_best(self, t, ck, alpha, minus, cache):
        """log P of the tree's most probable parse in the context ck."""
        k = (t, ck)
        if k not in cache:
            v = self.lp(ck, t[0], alpha, minus) + sum(self._parse_best(c, self.key(t[0], j), alpha, minus, cache)
                                                      for j, c in enumerate(t[1:]))
            for name, m in self.matches(t):
                v = max(v, self.lp(ck, name, alpha, minus) + sum(self._parse_best(s, self.key(name, h), alpha, minus, cache)
                                                                for h, s in enumerate(m)))
            cache[k] = v
        return cache[k]

    def logp_term(self, key, own):
        return self._parse_best(tree(key, own), self.root, self.alpha, None, self._best)

    def _choose(self, groups):
        """grammar_ops.OpComponent._choose with the held-out terms scored by their best parse."""
        if len(self.eqs) < MIN_LOO:
            return ALPHA_DEFAULT
        by_domain = groups is not None and len(set(groups)) >= 2
        labels = list(groups) if by_domain else list(range(len(self.eqs)))
        parts = defaultdict(list)
        for (own, keys), g in zip(self.eqs, labels):
            parts[g].append((own, keys))
        if by_domain and self.library:                           # each held-out domain new to the library too
            rest = {g: [e for h, es in parts.items() if h != g for e in es] for g in parts}
            fits = {g: LibComponent(r, learn(Counter(tree(k, own) for own, keys in r for k in keys), self.tokens,
                                              self.form)[0], self.tokens, alpha=ALPHA_DEFAULT, form=self.form)
                    for g, r in rest.items()}
            best, best_ll = ALPHA_DEFAULT, -math.inf
            for a in ALPHAS:
                ll = 0.0
                for g, eqs in parts.items():
                    f, cache = fits[g], {}
                    ll += sum(f._parse_best(tree(k, own), f.root, a, None, cache) for own, keys in eqs for k in keys)
                if ll > best_ll:
                    best, best_ll = a, ll
            return best
        held = {}
        for g, eqs in parts.items():
            cc, tt = Counter(), Counter()
            for own, keys in eqs:
                for k in keys:
                    for ck, ch in self.triples(k, own):
                        cc[(ck, ch)] += 1
                        tt[ck] += 1
            held[g] = (cc, tt)
        best, best_ll = ALPHA_DEFAULT, -math.inf
        for a in ALPHAS:
            ll = 0.0
            for g, eqs in parts.items():
                cache = {}
                ll += sum(self._parse_best(tree(k, own), self.root, a, held[g], cache) for own, keys in eqs for k in keys)
            if ll > best_ll:
                best, best_ll = a, ll
        return best

    def nterms_logp(self):
        return np.log((self.nterms[1:] + 1.0) / (self.nterms[1:].sum() + MAX_N))


class LibGrammar:
    """grammar_ops.OpGrammar with a library: the components mixed half and half at the term level."""

    def __init__(self, components, tokens):
        self.components = [c for c in components if c is not None and c.eqs]
        self.tokens = list(tokens)

    @classmethod
    def fit(cls, earlier, newest, tokens, library=(), earlier_groups=None, form="ctx"):
        """earlier, newest: [(own, keys)]; library: [(name, piece)] (shared by the components)."""
        comp = lambda eqs, g: LibComponent(eqs, list(library), tokens, groups=g, form=form) if eqs else None
        return cls([comp(earlier, earlier_groups), comp(newest, None)], tokens)

    def logp_term(self, key, own):
        if not self.components:                                  # no experience: the uniform base at every node
            return size(tree(key, own)) * math.log(1.0 / len(self.tokens))
        L = [c.logp_term(key, own) for c in self.components]
        return float(np.logaddexp.reduce(L) - math.log(len(L)))

    def bits(self, own, keys):
        """grammar_ops.OpGrammar.bits: an equation's number of terms, then its terms (unordered)."""
        n = len(keys)
        if self.components:
            P = np.mean([np.exp(c.nterms_logp()) for c in self.components], axis=0)
            lpn = math.log(P[min(n, MAX_N) - 1])
        else:
            lpn = -math.log(MAX_N)
        return -(lpn + sum(self.logp_term(k, own) for k in keys) - math.lgamma(n + 1)) / math.log(2)


def show(piece):
    """A piece in plain notation (u = the equation's own variable, v = another variable, K = a constant, □ = a hole)."""
    t = piece[0]
    if len(piece) == 1:
        return {"own": "u", "other": "v", "K": "K", "?": "□"}.get(t, t[1:] if t.startswith("n") else t)
    args = [show(c) for c in piece[1:]]
    if t == "Add":
        return f"({args[0]} + {args[1]})"
    if t == "Mul":
        return f"{args[0]}·{args[1]}"
    if t == "Pow":
        return f"({args[0]})^{args[1]}"
    return f"{t}({', '.join(args)})"
