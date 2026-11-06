# Stage 1 (general learner): enumerate L1 for d variables, canonicalize every tree, merge equal constant-free terms
# numerically, and save the vocabulary. Usage: g1_vocab.py <d> <workers> <out.json>
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")


def work(args):
    trees, d = args
    import language_g1 as LG
    out = []
    for t in trees:
        r = LG.canonical_tree(t, d)
        out.append(None if r is None else (r[0], r[2]))
    return out


if __name__ == "__main__":
    d, workers, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    import numpy as np, torch
    import language_g1 as LG
    t0 = time.time()
    by = LG.enumerate_trees(d)
    trees = [(n, t) for n in sorted(by) for t in by[n]]
    chunks = [[t for _, t in trees[i:i + 400]] for i in range(0, len(trees), 400)]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = [r for part in ex.map(work, [(c, d) for c in chunks]) for r in part]
    keys = {}
    for (n, t), r in zip(trees, res):
        if r is None:
            continue
        k, nc = r
        if k not in keys or n < keys[k]["nodes"]:
            keys[k] = {"nodes": n, "consts": nc, "tree": repr(t)}
    t1 = time.time()
    # numerical merge of constant-free terms (values up to a scale on random mixed-sign points)
    rng = np.random.default_rng(7)
    W = torch.tensor(rng.uniform(-2.5, 2.5, (64, d)))
    groups = {}
    for k, v in keys.items():
        if v["consts"] == 0:
            fp = LG.fingerprint(k, d, W)
            if fp is None:
                v["dropped"] = "undefined or constant on the test points"
                continue
            groups.setdefault(fp, []).append(k)
    merged = 0
    for fp, ks in groups.items():
        ks.sort(key=lambda k: (keys[k]["nodes"], len(k), k))
        for k in ks[1:]:
            keys[k]["merged_into"] = ks[0]; merged += 1
    # families: drop those undefined or constant for every one of 8 random constant draws
    dropped_f = 0
    for k, v in keys.items():
        if v["consts"] > 0:
            ok = False
            for _ in range(8):
                K = torch.tensor(rng.choice([-1, 1], v["consts"]) * 10 ** rng.uniform(-1, 1, v["consts"]))
                if LG.fingerprint(k, d, W, K) is not None:
                    ok = True; break
            if not ok:
                v["dropped"] = "undefined or constant for random constants"; dropped_f += 1
    vocab = {k: v for k, v in keys.items() if "merged_into" not in v and "dropped" not in v}
    vocab["Integer(1)"] = {"nodes": 1, "consts": 0, "tree": "('c',)"}       # the constant term
    by_c = {c: sum(1 for v in vocab.values() if v["consts"] == c) for c in range(3)}
    by_n = {n: sum(1 for v in vocab.values() if v["nodes"] == n) for n in range(1, 8)}
    summary = {"d": d, "trees": len(trees), "canonical_forms": len(keys), "merged_numerically": merged,
               "dropped_constant_free": sum(1 for v in keys.values() if v.get("dropped") and v["consts"] == 0),
               "dropped_families": dropped_f, "vocabulary": len(vocab), "by_constants": by_c, "by_nodes": by_n,
               "seconds": round(time.time() - t0, 1), "canonical_seconds": round(t1 - t0, 1)}
    json.dump({"summary": summary, "terms": vocab}, open(path, "w"))
    print(json.dumps(summary))
