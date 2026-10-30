"""Classify a system's nonlinearity 'kinds' from its equation structure (no data): polynomial, trig, exponential,
rational (state-dependent denominators, incl. Michaelis-Menten / Hill), piecewise (abs/sign/heaviside/max/min/clip/
piecewise), tanh (sigmoidal network units), power (non-integer powers of states), log, other."""

import ast

import sympy as sp

import systems as S

NP_KIND = {"sin": "trig", "cos": "trig", "tan": "trig", "arctan": "trig", "arctan2": "trig", "arcsin": "trig", "arccos": "trig",
           "exp": "exponential", "sinh": "exponential", "cosh": "exponential", "tanh": "tanh", "log": "log", "sqrt": "power",
           "abs": "piecewise", "sign": "piecewise", "heaviside": "piecewise", "maximum": "piecewise", "minimum": "piecewise",
           "clip": "piecewise", "floor": "piecewise", "mod": "piecewise", "where": "piecewise", "power": "power", "cbrt": "power"}


def dysts_kinds(name):
    fn = S._rhs_node(name)
    states = {a.arg for a in fn.args.args[:3]}
    kinds = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in NP_KIND:
            kinds.add(NP_KIND[node.attr])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if {n.id for n in ast.walk(node.right) if isinstance(n, ast.Name)} & states:
                kinds.add("rational")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp_ = node.right
            const = isinstance(exp_, ast.Constant) and float(exp_.value).is_integer() and exp_.value >= 0
            if not const and ({n.id for n in ast.walk(node.left) if isinstance(n, ast.Name)} & states):
                kinds.add("power")
    return kinds or {"polynomial"}


def sympy_kinds(exprs):
    X = set().union(*[e.free_symbols for e in exprs])
    kinds = set()
    for e in exprs:
        for a in sp.preorder_traversal(e):
            if isinstance(a, (sp.sin, sp.cos, sp.tan, sp.atan)):
                kinds.add("trig")
            elif isinstance(a, sp.exp):
                kinds.add("exponential")
            elif isinstance(a, sp.tanh):
                kinds.add("tanh")
            elif isinstance(a, (sp.Abs, sp.Piecewise, sp.Heaviside, sp.Max, sp.Min)):
                kinds.add("piecewise")
            elif isinstance(a, sp.log):
                kinds.add("log")
            elif isinstance(a, sp.Pow) and (a.base.free_symbols & X):
                if a.exp.is_number and a.exp.is_negative:
                    kinds.add("rational")
                elif a.exp.is_number and not a.exp.is_integer:
                    kinds.add("power")
        _, den = sp.fraction(sp.together(e))
        if den.free_symbols & X:
            kinds.add("rational")
    return kinds or {"polynomial"}
