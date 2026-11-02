"""Trial 28, owner addition 1: is a system's true right-hand side expressible in the search vocabulary?

Used only to report real-system results on the toolkit-expressible subset; the learner never sees the truth.
The vocabulary (structure_search.Vocab28) is the Trial 26 toolkit plus the weighted-sum argument approved in Trial 27,
fixed on 2 November before any search was run:
  - products of variables of total degree <= 3;
  - single-variable pieces: reciprocal 1/(a + b x), Hill-type 1/(a + b x^n) with a free exponent n > 0, exp, tanh,
    sin and |.| of an affine function of x, and x^p with a free exponent;
  - two-variable weighted-sum pieces: 1/(a + b x + c y);
  - any piece alone or times one variable;
  - products of two single-variable switches (Michaelis-Menten x Michaelis-Menten, Hill x Hill).
A component is expressible if every additive term of its right-hand side, after polynomial division of each
switch denominator, is one of these types. Piecewise ramps and clips of one variable (max(x - c, 0), min(x, c)) are
exact combinations of |.| and products of variables, so they count as expressible; logarithms do not.
"""

import sympy as sp

MONO_DEGREE, MULT_DEGREE = 3, 1
OK_CLASSES = ("mono", "switch", "pair", "function", "product")


def _deg(m, X):
    return sum(int(sp.degree(m, x)) for x in X if m.has(x))


def _monomial_ok(m, X):
    """m: a product of a number and nonnegative integer powers of the variables."""
    c, rest = m.as_coeff_Mul()
    for f in sp.Mul.make_args(rest):
        if f.is_Number:
            continue
        b, e = f.as_base_exp()
        if b not in X or not (e.is_Integer and e >= 0):
            return False
    return True


def _den_kind(f, X):
    """('single', x, n) for (a + b x^n) with a, b != 0 and n > 0; ('pair', x, y) for a + b x + c y; else None."""
    base, k = f.as_base_exp()
    if k != 1:
        return None
    syms = [x for x in X if base.has(x)]
    terms = sp.Add.make_args(sp.expand(base))
    const = [t for t in terms if not t.free_symbols & set(X)]
    var = [t for t in terms if t.free_symbols & set(X)]
    if len(syms) == 1 and len(const) == 1 and len(var) == 1:
        x = syms[0]
        _, rest = var[0].as_coeff_Mul()
        b, e = rest.as_base_exp()
        if b == x and e.is_Number and float(e) > 0:
            return ("single", x, float(e))
        return None
    if len(syms) == 2 and len(var) == 2 and len(const) <= 1:
        for t in var:
            _, rest = t.as_coeff_Mul()
            if rest not in syms:
                return None
        return ("pair", syms[0], syms[1])
    return None


def _piecewise_ok(f, X):
    """A piecewise ramp or clip of one variable with affine pieces: an exact combination of |.| and monomials."""
    if not isinstance(f, sp.Piecewise):
        return False
    vs = [x for x in X if f.has(x)]
    if len(vs) != 1:
        return False
    for expr, cond in f.args:
        if expr.free_symbols & set(X) and not (sp.Poly(expr, vs[0]).degree() <= 1):
            return False
    return True


def _function_term(t, X):
    """exp/tanh/sin/abs of an affine function of one variable, x^p with a free exponent, or a piecewise ramp/clip,
    times at most one variable."""
    c, rest = t.as_coeff_Mul()
    funcs, mult = [], sp.Integer(1)
    for f in sp.Mul.make_args(rest):
        if f.is_Number or not (f.free_symbols & set(X)):
            continue
        b, e = f.as_base_exp()
        if isinstance(f, (sp.exp, sp.tanh, sp.sin, sp.Abs)) or b.func == sp.exp:
            funcs.append(f)
        elif b in X and e.is_Number and not e.is_Integer:
            funcs.append(f)
        elif _piecewise_ok(f, X):
            funcs.append(f)
        else:
            mult = mult * f
    if len(funcs) != 1 or not _monomial_ok(mult, X) or _deg(mult, X) > MULT_DEGREE:
        return False
    return len([x for x in X if funcs[0].has(x)]) == 1


def _single_ok(m, x, n, X):
    """A numerator monomial m over a single-variable switch (a + b x^n)."""
    ex = int(sp.degree(m, x)) if m.has(x) else 0
    other = sp.simplify(m / x ** ex)
    if float(n) != int(n):                                  # free (non-integer) exponent: repressor or activator form
        return ex == 0 and _deg(other, X) <= MULT_DEGREE
    n = int(n)
    q = ex
    while q >= n:                                            # quotient monomials from polynomial division
        q -= n
        if _deg(other * x ** q, X) > MONO_DEGREE:
            return False
    r = ex % n
    if r == 0:
        return _deg(other, X) <= MULT_DEGREE
    if r == 1:
        return _deg(other, X) == 0
    return False


def classify_term(t, X):
    """One of OK_CLASSES, or the reason the term is not expressible."""
    t = sp.together(t)
    num, den = sp.fraction(t)
    if not den.free_symbols & set(X):
        if _monomial_ok(t, X):
            return "mono" if _deg(t, X) <= MONO_DEGREE else "monomial degree > 3"
        return "function" if _function_term(t, X) else "form outside the toolkit"
    dens = [f for f in sp.Mul.make_args(den) if f.free_symbols & set(X)]
    kinds = [_den_kind(f, X) for f in dens]
    if any(k is None for k in kinds):
        return "denominator outside the toolkit"
    numx = sp.expand(num)
    mons = sp.Add.make_args(numx)
    if any(not _monomial_ok(m, X) for m in mons):
        return "numerator not polynomial"
    if len(kinds) == 1 and kinds[0][0] == "pair":
        return "pair" if all(_deg(m, X) <= MULT_DEGREE for m in mons) else "two-variable switch times a product of degree > 1"
    if len(kinds) == 1:
        _, x, n = kinds[0]
        return "switch" if all(_single_ok(m, x, n, X) for m in mons) else "switch times a product outside the toolkit"
    if len(kinds) == 2 and all(k[0] == "single" for k in kinds) and kinds[0][1] != kinds[1][1]:
        (_, x1, n1), (_, x2, n2) = kinds
        for m in mons:
            e1 = float(sp.degree(m, x1)) if m.has(x1) else 0.0
            e2 = float(sp.degree(m, x2)) if m.has(x2) else 0.0
            rest = m / (x1 ** int(e1)) / (x2 ** int(e2))
            if e1 not in (0.0, n1) or e2 not in (0.0, n2) or _deg(rest, X) > 0:
                return "product of switches with an extra factor"
        return "product"
    return "more than two switch denominators"


def _integer_powers(e):
    """The converter writes integer powers as floats (x**4.0); make them integers so polynomial tools apply."""
    return e.replace(lambda a: isinstance(a, sp.Pow) and a.exp.is_Float and float(a.exp) == int(float(a.exp)),
                     lambda a: sp.Pow(a.base, sp.Integer(int(float(a.exp)))))


def system_expressible(exprs):
    """(expressible, per-component dict of class -> count)"""
    exprs = [_integer_powers(e) for e in exprs]
    d = len(exprs)
    X = sp.symbols(" ".join(f"x{i}" for i in range(d)))
    X = X if isinstance(X, tuple) else (X,)
    ok, out = True, []
    for e in exprs:
        classes = {}
        for t in sp.Add.make_args(sp.expand(e)):
            try:
                c = classify_term(t, X)
            except (sp.PolynomialError, TypeError, ValueError):
                c = "unparsed"
            classes[c] = classes.get(c, 0) + 1
            if c not in OK_CLASSES:
                ok = False
        out.append(classes)
    return ok, out
