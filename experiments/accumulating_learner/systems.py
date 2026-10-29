"""Load dysts vector fields without installing or executing dysts, and fix the Trial 1 system split.

`flows.py` is parsed with `ast`; it is never executed as a module. Eligibility and the split use only
the class graph and the parameter JSON. Each `_rhs` function is compiled on its own, and only after an
AST whitelist confirms it contains nothing but arithmetic, comparisons, conditional expressions,
tuple/list literals, names, numeric constants and calls to whitelisted `np` math functions.
Confirmation systems are listed by name, but their vector fields are never compiled or evaluated.
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DYSTS = ROOT / "third_party" / "dysts_0_96"
SPLIT_SALT = "accumulating-learner-trial1-split-v1"

NP_FUNCS = {
    "sin", "cos", "tan", "arctan", "arctan2", "sinh", "cosh", "tanh", "exp", "log", "sqrt",
    "abs", "sign", "heaviside", "maximum", "minimum", "power", "pi", "e", "cbrt", "square", "clip",
    "arcsin", "arccos", "floor", "mod", "where",
}
SAFE_NODES = (
    ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.Name,
    ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List, ast.IfExp,
    ast.Compare, ast.BoolOp, ast.Attribute, ast.Call, ast.Expr,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or, ast.Not,
)

TREE = ast.parse((DYSTS / "flows.py").read_text())
CLASSES = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
METADATA = json.loads((DYSTS / "chaotic_attractors.json").read_text())


def _bases(name):
    return [ast.unparse(b) for b in CLASSES[name].bases]


def _ancestors(name):
    out, todo = [], [name]
    while todo:
        c = todo.pop(0)
        out.append(c)
        todo += [b for b in _bases(c) if b in CLASSES]
    return out


def _own_methods(name):
    return {m.name: m for m in CLASSES[name].body if isinstance(m, ast.FunctionDef)}


def _rhs_node(name):
    """The `_rhs` FunctionDef the class resolves to, or None if `rhs` is overridden first."""
    for c in _ancestors(name):
        own = _own_methods(c)
        if "rhs" in own:
            return None
        if "_rhs" in own:
            return own["_rhs"]
    return None


def _check_safe(fn):
    for node in ast.walk(fn):
        if not isinstance(node, SAFE_NODES):
            raise ValueError(f"disallowed node {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in NP_FUNCS):
                raise ValueError(f"disallowed attribute {ast.unparse(node)}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute):
            raise ValueError(f"disallowed call {ast.unparse(node)}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
    for d in fn.decorator_list:
        if ast.unparse(d) != "staticjit":
            raise ValueError("unexpected decorator")


def eligibility(name):
    """Return None if eligible, else the reason. Metadata, class graph and AST checks only."""
    meta = METADATA[name]
    if name not in CLASSES:
        return "no class"
    if meta.get("embedding_dimension") != 3:
        return "dimension != 3"
    if meta.get("delay") or "DynSysDelay" in sum((_bases(c) for c in _ancestors(name)), []):
        return "delay system"
    if meta.get("nonautonomous"):
        return "nonautonomous"
    if meta.get("unbounded_indices"):
        return "unbounded coordinate"
    fn = _rhs_node(name)
    if fn is None:
        return "custom rhs method (no static _rhs)"
    try:
        _check_safe(fn)
    except ValueError as e:
        return f"rhs failed whitelist: {e}"
    return None


def split_of(name):
    h = int(hashlib.sha256(f"{SPLIT_SALT}:{name}".encode()).hexdigest()[:12], 16) / 16**12
    return "meta_train" if h < 0.6 else ("development" if h < 0.8 else "confirmation")


def eligible_split():
    out = {"meta_train": [], "development": [], "confirmation": []}
    for name in sorted(METADATA):
        if eligibility(name) is None:
            out[split_of(name)].append(name)
    return out


class System:
    """Vector field f(x; params) for one dysts system. Refuses confirmation systems."""

    def __init__(self, name, params=None, allow_confirmation=False):
        if split_of(name) == "confirmation" and not allow_confirmation:
            raise RuntimeError(f"{name} is a reserved confirmation system")
        if eligibility(name) is not None:
            raise RuntimeError(f"{name} is not eligible: {eligibility(name)}")
        self.name = name
        self.meta = METADATA[name]
        self.params = dict(self.meta["parameters"] if params is None else params)
        fn = _rhs_node(name)
        fn = ast.FunctionDef(name="_rhs", args=fn.args, body=fn.body, decorator_list=[],
                             returns=None, type_params=[])
        mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        ns = {"np": np, "__builtins__": {}}
        exec(compile(mod, f"dysts:{name}._rhs", "exec"), ns)
        self._rhs = ns["_rhs"]

    def param_list(self):
        return [np.asarray(self.params[k], dtype=float) for k in sorted(self.params)]

    def __call__(self, x):
        """x: (..., 3) array -> (..., 3) derivative."""
        x = np.asarray(x, dtype=float)
        out = self._rhs(x[..., 0], x[..., 1], x[..., 2], 0.0, *self.param_list())
        return np.stack([np.broadcast_to(np.asarray(o, dtype=float), x.shape[:-1]) for o in out], -1)


if __name__ == "__main__":
    reasons = {}
    for n in sorted(METADATA):
        r = eligibility(n)
        if r:
            reasons.setdefault(r, []).append(n)
    for r, ns in reasons.items():
        print(f"excluded ({r}): {len(ns)} {ns if 'whitelist' in r else ''}")
    sp = eligible_split()
    for k, v in sp.items():
        print(k, len(v), v if k != "confirmation" else "(names withheld from printing)")
