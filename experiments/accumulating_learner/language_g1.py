"""Stage 1 of the general learner (research log, 6 November): the expression language L1, in place of the toolkit.

A term is an expression over the state variables x1 .. xd (the learner's scaled coordinates, named by position only)
and fitted constants, built with +, *, /, power (exponent 2, 3, 1/2 or a fitted constant), exp, log and sin: at most
7 nodes (operations, variables, constants and fixed exponents all count), at most 2 fitted constants, and no exp or log
inside another exp or log. A term counts as in L1 if it, or an equivalent form up to a constant factor, meets these
limits. An equation is a sum of at most 6 terms whose coefficients are solved exactly (variable projection); the
constant 1 is one of the terms.

Canonical form (canonical()):
  - every constant-only subexpression becomes one fresh constant (siblings in a sum or product merge);
  - exponentials of sums are split (exp(a + b) = exp(a) exp(b));
  - a term whose expansion is a sum of parts that share no constant is a combination of simpler terms: not kept;
  - rational parts go over a common denominator (sympy cancel); every sum in multiplicative position is made monic
    (the coefficient of its leading part is pulled out); overall constant factors are dropped;
  - the constants are renamed K0, K1 in the order that gives the smallest printed form.
Two canonical terms without constants are merged when their values agree up to a scale on random points (merge_numeric).

The enumeration (enumerate_trees) produces every tree within the limits; the vocabulary is the set of canonical forms.
"""

import itertools
import math
from functools import lru_cache

import numpy as np
import sympy as sp
import torch

MAX_NODES, MAX_CONSTS = 7, 2
FIXED_EXPONENTS = {"2": sp.Integer(2), "3": sp.Integer(3), "h": sp.Rational(1, 2)}


def variables(d):
    return tuple(sp.Symbol(f"x{j + 1}", real=True) for j in range(d))


CONSTS = tuple(sp.Symbol(f"K{k}", real=True) for k in range(4))


# ---------------------------------------------------------------------------------------------------------------------
# enumeration of trees
# ---------------------------------------------------------------------------------------------------------------------
# Trees are nested tuples: ("x", j), ("c",), (op, a) for op in exp, log, sin, (op, a, b) for op in add, mul, div,
# ("pow", a, e) with a fixed exponent e in "2", "3", "h" (1/2), and ("powc", a) with a fitted exponent.

def _info(t, cache):
    key = t
    if key in cache:
        return cache[key]
    op = t[0]
    if op == "x":
        out = (1, 0, True, False)
    elif op == "c":
        out = (1, 1, False, False)
    elif op in ("exp", "log", "sin"):
        n, k, v, el = _info(t[1], cache)
        out = (n + 1, k, v, el or op in ("exp", "log"))
    elif op == "pow":
        n, k, v, el = _info(t[1], cache)
        out = (n + 2, k, v, el)
    elif op == "powc":
        n, k, v, el = _info(t[1], cache)
        out = (n + 2, k + 1, v, el)
    else:
        a, b = _info(t[1], cache), _info(t[2], cache)
        out = (a[0] + b[0] + 1, a[1] + b[1], a[2] or b[2], a[3] or b[3])
    cache[key] = out
    return out


def enumerate_trees(d, max_nodes=MAX_NODES, max_consts=MAX_CONSTS):
    """Every tree with at most max_nodes nodes and max_consts fitted constants, no constant-only subtree other than a
    single constant, no exp or log inside an exp or log, commutative children in a fixed order, no division by a lone
    constant (a scale). Returns {size: [tree, ...]}."""
    cache = {}
    by = {1: [("x", j) for j in range(d)] + [("c",)]}
    for t in by[1]:
        _info(t, cache)
    for n in range(2, max_nodes + 1):
        out = []
        for a in by[n - 1]:                                   # unary functions
            _, k, v, el = cache[a]
            if not v:
                continue
            out.append(("sin", a))
            if not el:
                out += [("exp", a), ("log", a)]
        if n - 2 >= 1:                                        # powers: the exponent is one node
            for a in by[n - 2]:
                _, k, v, el = cache[a]
                if not v:
                    continue
                out += [("pow", a, e) for e in ("2", "3", "h")]
                if k < max_consts:
                    out.append(("powc", a))
        for n1 in range(1, n - 1):                            # binary operations
            n2 = n - 1 - n1
            for a in by[n1]:
                ia = cache[a]
                for b in by[n2]:
                    ib = cache[b]
                    if not (ia[2] or ib[2]) or ia[1] + ib[1] > max_consts:
                        continue
                    if repr(a) <= repr(b):
                        out += [("add", a, b), ("mul", a, b)]
                    if b != ("c",):
                        out.append(("div", a, b))
        for t in out:
            _info(t, cache)
        by[n] = out
    return by


def to_sympy(t, X, fresh):
    op = t[0]
    if op == "x":
        return X[t[1]]
    if op == "c":
        return fresh()
    if op == "exp":
        return sp.exp(to_sympy(t[1], X, fresh))
    if op == "log":
        return sp.log(to_sympy(t[1], X, fresh))
    if op == "sin":
        return sp.sin(to_sympy(t[1], X, fresh))
    if op == "pow":
        return sp.Pow(to_sympy(t[1], X, fresh), FIXED_EXPONENTS[t[2]])
    if op == "powc":
        base = to_sympy(t[1], X, fresh)
        return sp.Pow(base, fresh())
    a, b = to_sympy(t[1], X, fresh), to_sympy(t[2], X, fresh)
    return {"add": sp.Add, "mul": sp.Mul}[op](a, b) if op != "div" else a / b


class _Fresh:
    def __init__(self, prefix="_c"):
        self.n, self.prefix = 0, prefix

    def __call__(self):
        self.n += 1
        return sp.Symbol(f"{self.prefix}{self.n}", real=True)


# ---------------------------------------------------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------------------------------------------------

def _has_var(e, Xs):
    return bool(e.free_symbols & Xs)


def _replace_consts(e, Xs, fresh):
    """Every maximal constant-only subexpression (one containing a constant symbol) becomes a fresh constant; within a
    sum or a product, all constant-only parts merge into one fresh constant."""
    if e.is_Number or e.is_NumberSymbol:
        return e
    if not _has_var(e, Xs):
        return fresh() if e.free_symbols else e
    if e.is_Atom:
        return e
    if e.is_Add or e.is_Mul:
        const = [a for a in e.args if not _has_var(a, Xs)]
        rest = [_replace_consts(a, Xs, fresh) for a in e.args if _has_var(a, Xs)]
        if any(a.free_symbols for a in const):
            rest.append(fresh())
        else:
            rest += const
        return e.func(*rest)
    return e.func(*[_replace_consts(a, Xs, fresh) for a in e.args])


def _split_exp(e):
    """exp(a + b) -> exp(a) exp(b) everywhere (sympy keeps exp of a sum as is)."""
    return e.replace(lambda u: isinstance(u, sp.exp) and u.args[0].is_Add,
                     lambda u: sp.Mul(*[sp.exp(a) for a in u.args[0].args]))


def _abs_split(e, Xs):
    """|c * f| -> |c| |f| (the constant factor of an absolute value is a constant)."""
    def f(u):
        inner = u.args[0]
        if inner.is_Mul:
            const = [a for a in inner.args if not _has_var(a, Xs)]
            if const:
                return sp.Abs(sp.Mul(*const)) * sp.Abs(sp.Mul(*[a for a in inner.args if _has_var(a, Xs)]))
        return u
    return e.replace(lambda u: isinstance(u, sp.Abs), f)


def _denest(e, Xs):
    """Rewrites valid wherever the term is defined: (u**a)**b -> u**(a b) for a constant exponent b (|u|**(a b) when a is
    an even integer), and log(u**b) -> b log(u) (log|u| for an even integer b)."""
    def is_const(u):
        return not _has_var(u, Xs) and bool(u.free_symbols)

    def pw(u):
        inner, b = u.base, u.exp
        if not (inner.is_Pow and (is_const(b) or is_const(inner.exp))):
            return u
        a = inner.exp
        base = sp.Abs(inner.base) if (a.is_Integer and int(a) % 2 == 0) else inner.base
        return sp.Pow(base, a * b)

    def lg(u):
        arg = u.args[0]
        if arg.is_Pow and not _has_var(arg.exp, Xs):
            b = arg.exp
            base = sp.Abs(arg.base) if (b.is_Integer and int(b) % 2 == 0) else arg.base
            return b * sp.log(base)
        return u
    e = e.replace(lambda u: u.is_Pow and u.base.is_Pow, pw)
    return e.replace(lambda u: isinstance(u, sp.log), lg)


def _redundant(e, Xs):
    """True if the expansion of e is a sum whose parts share no constant inside their variable parts: then e is a linear
    combination of simpler terms (the coefficients are free in an equation)."""
    ex = sp.expand(e, power_exp=False, log=False, deep=False)
    if not ex.is_Add:
        return False
    seen = set()
    for a in ex.args:
        _, varpart = a.as_independent(*Xs, as_Add=False)
        cs = varpart.free_symbols - Xs
        if cs & seen:
            return False
        seen |= cs
    return True


def _monic(add, Xs, fresh):
    """A sum divided by the constant coefficient of its leading part (the variable part with the smallest sort key):
    returns the monic sum (the pulled-out factor is dropped by the caller)."""
    parts = [a for a in add.args if _has_var(a, Xs)]
    if not parts:
        return add
    lead = min(parts, key=lambda a: sp.default_sort_key(a.as_independent(*Xs, as_Add=False)[1]))
    coef, _ = lead.as_independent(*Xs, as_Add=False)
    if coef == 1:
        return add
    return _replace_consts(sp.expand(add / coef, power_exp=False, log=False, deep=False), Xs, fresh)


def _strip(e, Xs, fresh):
    """Drop overall constant factors and make sums in multiplicative position monic (numerator, denominator and the
    bases of powers at the top level)."""
    factors = sp.Mul.make_args(e)
    keep = []
    for f in factors:
        if not _has_var(f, Xs):
            continue
        if f.is_Add:
            f = _monic(f, Xs, fresh)
        elif f.is_Pow and f.base.is_Add and not _has_var(f.exp, Xs):
            f = sp.Pow(_monic(f.base, Xs, fresh), f.exp)
        elif f.is_Pow and f.base.is_Mul and not _has_var(f.exp, Xs):
            f = sp.Mul(*[sp.Pow(a, f.exp) for a in f.base.args if _has_var(a, Xs)])
        keep.append(f)
    out = sp.Mul(*keep)
    if out.is_Mul and any(not _has_var(a, Xs) for a in out.args):   # factors produced by the monic step
        out = sp.Mul(*[a for a in out.args if _has_var(a, Xs)])
    return out


def _rename(e, Xs):
    cs = sorted(e.free_symbols - Xs, key=lambda s: s.name)
    if len(cs) > len(CONSTS):
        return None
    best = None
    for perm in itertools.permutations(CONSTS[:len(cs)]):
        f = e.xreplace(dict(zip(cs, perm)))
        s = sp.srepr(f)
        if best is None or (len(s), s) < (len(best[1]), best[1]):
            best = (f, s)
    return best[0]


BAD = (sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I)


def canonical(e, X):
    """(key, expression, number of constants) of a term, or None when it is not kept (constant, a combination of
    simpler terms, more than MAX_CONSTS constants after simplification, or invalid)."""
    Xs = set(X)
    fresh = _Fresh()
    try:
        e = _replace_consts(e, Xs, fresh)
        e = sp.powsimp(e, combine="exp")                     # x * x**K -> x**(K + 1) (then a fresh constant)
        e = _abs_split(_split_exp(e), Xs)
        e = _replace_consts(e, Xs, fresh)
        e = _replace_consts(sp.powsimp(_denest(e, Xs), combine="exp"), Xs, fresh)
        if not _has_var(e, Xs) or any(e.has(b) for b in BAD):
            return None
        if _redundant(e, Xs):
            return None
        e = sp.cancel(e) if not e.has(sp.Abs) else sp.together(e)
        e = _replace_consts(e, Xs, fresh)
        if e.is_Add and _redundant(e, Xs):
            return None
        num, den = sp.fraction(e)
        num, den = _strip(num, Xs, fresh), _strip(den, Xs, fresh)
        e = _replace_consts(num / den, Xs, fresh)
        e = _strip(e, Xs, fresh)
        if not _has_var(e, Xs) or e.is_Add or any(e.has(b) for b in BAD):
            return None
        e = _rename(e, Xs)
        if e is None:
            return None
        k = len(e.free_symbols - Xs)
        if k > MAX_CONSTS:
            return None
        return sp.srepr(e), e, k
    except (TypeError, ValueError, ZeroDivisionError, RecursionError, sp.PolynomialError, NotImplementedError,
            AttributeError):
        return None


def canonical_tree(t, d):
    X = variables(d)
    return canonical(to_sympy(t, X, _Fresh("_t")), X)


# ---------------------------------------------------------------------------------------------------------------------
# evaluation (torch)
# ---------------------------------------------------------------------------------------------------------------------

def torch_source(e, X):
    """Python source of e with W0, W1, .. (columns, shape (n, 1)) and K0, K1 (constants, scalars or (1, B))."""
    names = {x: f"W{j}" for j, x in enumerate(X)}
    names.update({c: c.name for c in CONSTS})

    def src(u):
        if u in names:
            return names[u]
        if u.is_number and not u.is_Integer:                    # a numeric subexpression, e.g. log(2)
            try:
                return f"{float(u)!r}"
            except TypeError:
                return "float('nan')"
        if u.is_Integer:
            return f"{float(int(u))!r}"
        if u.is_Rational or u.is_Float or u.is_NumberSymbol:
            return f"{float(u)!r}"
        if u.is_Add:
            return "(" + " + ".join(src(a) for a in u.args) + ")"
        if u.is_Mul:
            return "(" + " * ".join(src(a) for a in u.args) + ")"
        if u.is_Pow:
            b, x = u.args
            if x == -1:
                return f"(1.0 / {src(b)})"
            if x == sp.Rational(1, 2):
                return f"torch.sqrt({src(b)})"
            if x == sp.Rational(-1, 2):
                return f"(1.0 / torch.sqrt({src(b)}))"
            if x.is_Integer:
                return f"({src(b)}) ** {int(x)}"
            return f"torch.pow({src(b)}, {src(x)})"
        if isinstance(u, sp.exp):
            return f"torch.exp({src(u.args[0])})"
        if isinstance(u, sp.log):
            return f"torch.log({src(u.args[0])})"
        if isinstance(u, sp.sin):
            return f"torch.sin({src(u.args[0])})"
        if isinstance(u, sp.cos):
            return f"torch.cos({src(u.args[0])})"
        if isinstance(u, sp.Abs):
            return f"torch.abs({src(u.args[0])})"
        if isinstance(u, sp.sign):
            return f"torch.sign({src(u.args[0])})"
        raise NotImplementedError(f"no torch form for {u.func}")
    return src(e)


@lru_cache(maxsize=None)
def compiled(key, d):
    """A function f(W (n, d), K (k,) or (B, k)) -> (n,) or (n, B) for the canonical term with this key."""
    X = variables(d)
    e = sp.sympify(eval(key, sp.__dict__)) if isinstance(key, str) else key
    k = len(e.free_symbols - set(X))
    src = torch_source(e, X)
    args = ", ".join([f"W{j}" for j in range(d)] + [f"K{c}" for c in range(k)])
    fn = eval(f"lambda {args}: {src}", {"torch": torch})

    def f(W, K=None):
        cols = [W[:, j:j + 1] for j in range(d)]
        if k == 0:
            out = fn(*cols)
        else:
            K = torch.as_tensor(K, dtype=W.dtype)
            if K.dim() == 1:
                out = fn(*cols, *[K[c] for c in range(k)])
            else:
                out = fn(*cols, *[K[:, c].reshape(1, -1) for c in range(k)])
        out = torch.as_tensor(out, dtype=W.dtype)
        if K is not None and torch.as_tensor(K).dim() == 2:
            return out.expand(W.shape[0], K.shape[0]) if out.dim() == 2 else out.reshape(-1, 1).expand(W.shape[0], K.shape[0])
        out = out.reshape(-1) if out.dim() == 2 else out
        return out.expand(W.shape[0]) if out.numel() == 1 else out
    return f


def expr_of(key):
    return eval(key, sp.__dict__) if isinstance(key, str) else key


# ---------------------------------------------------------------------------------------------------------------------
# numerical merging of constant-free terms
# ---------------------------------------------------------------------------------------------------------------------

def fingerprint(key, d, W, K=None, digits=9):
    """Values on random points, up to a scale and sign (None if undefined everywhere or constant)."""
    with torch.no_grad(), np.errstate(all="ignore"):
        try:
            v = compiled(key, d)(W, K).numpy()
        except (RuntimeError, ValueError, ZeroDivisionError, OverflowError, NotImplementedError):
            return None
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return None
    vv = v[ok]
    if np.ptp(vv) <= 1e-12 * (np.abs(vv).max() + 1e-300):
        return None
    j = int(np.argmax(np.abs(vv)))
    vv = vv / vv[j]
    return tuple(ok.tolist()), tuple(np.round(vv, digits).tolist())
