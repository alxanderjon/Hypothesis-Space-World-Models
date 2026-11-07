"""Stage 1 of the general learner: which vocabulary terms keep their canonical form when the variables are rescaled
(x_j -> s_j x_j, s_j > 0). True terms of systems in the learner's coordinates always do (the learner's scales are
absorbed); fantasies draw only from these. Writes the list next to the vocabulary. Usage: g1_scale_closed.py <workers> [d]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
G1 = HERE.parents[1] / "outputs/accumulating_learner/g1"


def work(args):
    keys, d = args
    sys.path.insert(0, str(HERE))
    import sympy as sp
    import language_g1 as LG
    X = LG.variables(d)
    s = [sp.Symbol(f"s{j + 1}", positive=True) for j in range(d)]
    out = []
    for k in keys:
        if k == sp.srepr(sp.Integer(1)):
            out.append(True); continue
        e = LG.expr_of(k).xreplace({X[j]: s[j] * X[j] for j in range(d)})
        r = LG.canonical(e, X)
        out.append(r is not None and r[0] == k)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    V = json.loads((G1 / f"vocab_d{d}.json").read_text())["terms"]
    order = sorted(V, key=lambda k: (V[k]["nodes"], V[k]["consts"], len(k), k))
    chunks = [(order[i:i + 300], d) for i in range(0, len(order), 300)]
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        flags = [f for part in ex.map(work, chunks) for f in part]
    closed = [k for k, f in zip(order, flags) if f]
    (G1 / f"scale_closed_d{d}.json").write_text(json.dumps(closed))
    print(f"scale-closed terms: {len(closed)} of {len(order)}")
