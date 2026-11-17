"""Stitch check (owner, 17 November): can Stitch (stitch_core; LILO, Grand et al. 2024) replace or speed up our library
compression (library_ops.learn, the "uni" form used by the ladder's tables)? Minutes-scale, CPU, no change to any run.
Input: the verified solutions of each curriculum lineage the ladder started from (and of the end-of-rung-c lineages), as
library_ops term trees with multiplicity. Both methods get the same trees and the same limits (MAX_PIECES pieces, at most
MAX_HOLES holes). Reported: seconds; the pieces; each library scored under OUR objective (DL(library) + DL(solutions |
library), bits) and under Stitch's (its rewritten corpus cost, via stitch_core.rewrite).
Usage: PYTHONPATH=<stitch lib> stitch_check.py"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stitch_check")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402  (read only)
import library_ops as L                                          # noqa: E402

G2H = LD.G2H


def curriculum_equations(arm, seed):
    base, name, _ = LD.CURRIC[(arm, seed)]
    rec = json.loads((G2H / base / "lineages" / f"{name}.json").read_text())
    eqs = []
    for L_ in rec["levels"]:
        for r in L_["rounds"]:
            for y in r["wake"]:
                if y["status"] == "done":
                    for e in y["equations"]:
                        if e["first_exact"] is not None and e["solution"]:
                            eqs.append((int(e["eq"]), list(e["solution"])))
    return eqs


def rung_equations(rung, arm, seed):
    st = json.loads((G2H / "ladder" / rung / f"{arm}-s{seed}.state.json").read_text())
    return [(int(eq), list(keys)) for s, d in st["sols"].items() for eq, keys in d.items()]


# -- trees <-> s-expressions --
def tok(t):
    return re.sub(r"[^A-Za-z0-9_]", lambda m: {"-": "m", "/": "_", ".": "p"}.get(m.group(), "x"), t)


def sexpr(t):
    if t == L.HOLE:
        raise ValueError("hole")
    if len(t) == 1:
        return tok(t[0])
    return "(" + " ".join([tok(t[0])] + [sexpr(c) for c in t[1:]]) + ")"


def parse_sexpr(s, untok):
    """Stitch body -> our piece (holes #i -> HOLE); None if a hole repeats (our pieces cannot express that)."""
    toks = re.findall(r"\(|\)|[^\s()]+", s)
    seen = []

    def go(i):
        if toks[i] == "(":
            head = toks[i + 1]
            j, kids = i + 2, []
            while toks[j] != ")":
                c, j = go(j)
                kids.append(c)
            return (untok.get(head, head), *kids), j + 1
        x = toks[i]
        if x.startswith("#"):
            seen.append(x)
            return L.HOLE, i + 1
        return (untok.get(x, x),), i + 1
    t, _ = go(0)
    return None if len(seen) != len(set(seen)) else t


def our_dl(terms, tokens, library):
    """DL(library) + DL(solutions | library) in bits, as library_ops.learn computes it (uni form)."""
    trees = list(terms)
    w = [terms[t] for t in trees]
    V0, key0 = len(tokens), L.context("uni", [])
    c0 = {k: dict(v) for k, v in L._counts([L.parse(t, []) for t in trees], w, key0).items()}
    lib = L.ordered(library)
    c = L._counts([L.parse(t, lib) for t in trees], w, L.context("uni", lib))
    return L.data_bits(c, V0 + len(lib)) + sum(L.piece_bits(p, c0, V0, key0) for _, p in library)


def main():
    import grammar_ops as GO
    import search_g1 as SG
    import stitch_core as S
    out = []
    cases = [(f"curriculum {a}-s{s}", lambda a=a, s=s: curriculum_equations(a, s), 2) for a in ("blend", "net") for s in (1, 2, 3)]
    cases += [(f"end of rung c {a}-s{s}", lambda a=a, s=s: rung_equations("c", a, s), 3) for a in ("blend",) for s in (1, 2, 3)]
    for name, get, d in cases:
        eqs = get()
        order = SG.vocabulary_order(d)
        tokens = GO.alphabet(order, d=d)
        terms = Counter(L.tree(k, own) for own, keys in eqs for k in keys)
        t0 = time.time()
        ours, _ = L.learn(terms, tokens, "uni")
        t_ours = time.time() - t0
        untok = {}
        for t in terms:
            for u in L.subtrees(t):
                untok[tok(u[0])] = u[0]
        progs = [sexpr(t) for t, m in terms.items() for _ in range(m)]
        t0 = time.time()
        res = S.compress(progs, iterations=L.MAX_PIECES, max_arity=L.MAX_HOLES)
        t_st = time.time() - t0
        st_pieces, unrep = [], 0
        for a in res.abstractions:
            body = a.body
            for j in range(L.MAX_HOLES + 1):
                body = body.replace(f"#{j}", f"#{j}")
            p = parse_sexpr(body, untok)
            if p is None:
                unrep += 1
                continue
            st_pieces.append((a.name, p))
        dl0 = our_dl(terms, tokens, [])
        dl_ours, dl_st = our_dl(terms, tokens, ours), our_dl(terms, tokens, st_pieces)
        # Stitch's own objective for our library: rewrite the corpus with our pieces as abstractions
        ab = []
        for nm, p in ours:
            k = [0]

            def hs(u):
                if u == L.HOLE:
                    k[0] += 1
                    return f"#{k[0] - 1}"
                return tok(u[0]) if len(u) == 1 else "(" + " ".join([tok(u[0])] + [hs(c) for c in u[1:]]) + ")"
            ab.append(S.Abstraction(name=f"ours_{nm}", body=hs(p), arity=L.holes(p)))
        cost0 = sum(len(re.findall(r"[^\s()]+", x)) for x in progs)
        try:
            rw = S.rewrite(progs, ab) if ab else None
            cost_ours_st = sum(len(re.findall(r"[^\s()]+", x)) for x in rw.rewritten) + sum(len(re.findall(r"[^\s()]+", x.body)) for x in ab) if rw else cost0
        except Exception as ex:
            cost_ours_st = f"rewrite failed: {str(ex)[:60]}"
        cost_st = sum(len(re.findall(r"[^\s()]+", x)) for x in res.rewritten) + sum(len(re.findall(r"[^\s()]+", a.body)) for a in res.abstractions)
        same = len({p for _, p in ours} & {p for _, p in st_pieces})
        row = {"case": name, "equations": len(eqs), "distinct terms": len(terms), "term occurrences": sum(terms.values()),
               "seconds ours": round(t_ours, 2), "seconds stitch": round(t_st, 2),
               "pieces ours": len(ours), "pieces stitch": len(res.abstractions), "stitch pieces with a repeated hole": unrep,
               "pieces in common": same,
               "our DL bits: none / ours / stitch": [round(dl0, 1), round(dl_ours, 1), round(dl_st, 1)],
               "stitch size (symbols incl. library): none / ours / stitch": [cost0, cost_ours_st, cost_st],
               "ours top": [L.show(p) for _, p in ours[:6]], "stitch top": [a.body for a in res.abstractions[:6]]}
        out.append(row)
        print(json.dumps(row), flush=True)
    (LD.OUT / "stitch_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
