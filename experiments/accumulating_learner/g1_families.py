# Stage 1 (general learner): merge L1 families that are the same up to a reparametrization of their constants and a scale
# (research log, Stage 1 rules, section 4): for random values of one term's constants, the other's constants can be
# fitted to reproduce its values exactly (relative 1e-9), up to a scale, on random points; tested both ways.
# Candidate pairs come from nearest neighbours among normalized columns on a constant grid.
# Usage (from experiments/accumulating_learner): g1_families.py <vocab.json (from g1_vocab.py)> <workers> <out.json>
import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")
D = 2


def grid(k):
    g1 = np.concatenate([-np.logspace(-1.5, 1.5, 25), np.logspace(-1.5, 1.5, 25)])
    if k == 1:
        return g1[:, None]
    g = np.concatenate([-np.logspace(-1.5, 1.5, 7), np.logspace(-1.5, 1.5, 7)])
    return np.array([[a, b] for a in g for b in g])


def cols(key, k, W, K):
    import torch, language_g1 as LG
    with np.errstate(all="ignore"):
        try:
            return LG.compiled(key, D)(torch.tensor(W), torch.tensor(K)).numpy()   # (n, B)
        except Exception:
            return None


def contains(B, kB, y, W, rng):
    """Does family B (kB constants) reproduce the column y (finite on all points W) up to a scale? Min relative residual."""
    from scipy.optimize import least_squares
    G = grid(kB)
    C = cols(B, kB, W, G)
    if C is None:
        return math.inf
    yn = np.linalg.norm(y)
    best = []
    for j in range(C.shape[1]):
        c = C[:, j]
        if not np.all(np.isfinite(c)) or np.linalg.norm(c) == 0:
            continue
        r = y - c * (c @ y) / (c @ c)
        best.append((np.linalg.norm(r) / yn, j))
    best.sort()
    out = best[0][0] if best else math.inf
    import torch, language_g1 as LG
    f = LG.compiled(B, D)
    Wt = torch.tensor(W)

    def res(b):
        with np.errstate(all="ignore"):
            c = f(Wt, torch.tensor(b)).numpy()
        if not np.all(np.isfinite(c)) or np.linalg.norm(c) == 0:
            return np.full(len(y), 1e3)
        return (y - c * (c @ y) / (c @ c)) / yn
    for r0, j in best[:3]:
        try:
            sol = least_squares(res, G[j], method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400)
            out = min(out, float(np.linalg.norm(sol.fun)))
        except Exception:
            pass
        if out < 1e-9:
            break
    return out


def pair_job(args):
    A, kA, B, kB, seed = args
    rng = np.random.default_rng(seed)
    W = rng.uniform(-2.5, 2.5, (40, D))
    W = W[np.abs(W).min(1) > 0.1]
    ok = 0
    for direction in ((A, kA, B, kB), (B, kB, A, kA)):
        P, kP, Q, kQ = direction
        passed = 0
        for _ in range(2):
            for _try in range(20):
                a = rng.choice([-1, 1], kP) * 10 ** rng.uniform(-1, 1, kP)
                y = cols(P, kP, W, a[None, :])
                if y is not None and np.all(np.isfinite(y[:, 0])) and np.ptp(y[:, 0]) > 1e-9 * np.abs(y).max():
                    break
            else:
                # P undefined on mixed points for random constants: test on positive points instead
                Wp = np.abs(W) + 0.2
                y = cols(P, kP, Wp, a[None, :])
                if y is None or not np.all(np.isfinite(y[:, 0])):
                    return (A, B, False)
                if contains(Q, kQ, y[:, 0], Wp, rng) < 1e-9:
                    passed += 1
                continue
            if contains(Q, kQ, y[:, 0], W, rng) < 1e-9:
                passed += 1
        ok += passed == 2
    return (A, B, ok == 2)


if __name__ == "__main__":
    src, workers, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    import torch
    t0 = time.time()
    data = json.load(open(src))
    V = data["terms"]
    fam = [k for k, v in V.items() if v["consts"] > 0]
    rng = np.random.default_rng(11)
    Wp = rng.uniform(0.3, 2.5, (32, D))
    pool, owner = [], []
    for i, k in enumerate(fam):
        C = cols(k, V[k]["consts"], Wp, grid(V[k]["consts"]))
        if C is None:
            continue
        for j in range(C.shape[1]):
            c = C[:, j]
            if np.all(np.isfinite(c)) and np.ptp(c) > 1e-9 * (np.abs(c).max() + 1e-300):
                c = c / np.linalg.norm(c); c = c * np.sign(c[np.argmax(np.abs(c))])
                pool.append(c); owner.append(i)
    P = torch.tensor(np.array(pool)); owner = torch.tensor(np.array(owner))
    queries, qowner = [], []
    for i, k in enumerate(fam):
        kk = V[k]["consts"]
        for _ in range(2):
            a = rng.choice([-1, 1], kk) * 10 ** rng.uniform(-1, 1, kk)
            q = cols(k, kk, Wp, a[None, :])
            if q is None or not np.all(np.isfinite(q)) or np.ptp(q) == 0:
                continue
            q = q[:, 0] / np.linalg.norm(q[:, 0]); q = q * np.sign(q[np.argmax(np.abs(q))])
            queries.append(q); qowner.append(i)
    Q = torch.tensor(np.array(queries))
    pairs = set()
    nf = len(fam)
    for s0 in range(0, len(Q), 64):                          # per query: nearest column of every other family
        dist = torch.cdist(Q[s0:s0 + 64], P)                   # (q, pool)
        best = torch.full((dist.shape[0], nf), 9.0, dtype=dist.dtype)
        best.scatter_reduce_(1, owner.expand(dist.shape[0], -1), dist, reduce="amin")
        for r in range(dist.shape[0]):
            i = qowner[s0 + r]
            best[r, i] = 9.0
            vals, idx = torch.topk(best[r], 5, largest=False)
            for dd, o in zip(vals.tolist(), idx.tolist()):
                if dd < 0.1:
                    pairs.add(tuple(sorted((i, int(o)))))
    t1 = time.time()
    jobs = [(fam[a], V[fam[a]]["consts"], fam[b], V[fam[b]]["consts"], 1000 + n) for n, (a, b) in enumerate(sorted(pairs))]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(pair_job, jobs, chunksize=8))
    parent = {k: k for k in fam}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    rank = lambda k: (V[k]["nodes"], V[k]["consts"], len(k), k)
    for A, B, same in res:
        if same:
            ra, rb = find(A), find(B)
            if ra != rb:
                keep, drop = (ra, rb) if rank(ra) <= rank(rb) else (rb, ra)
                parent[drop] = keep
    merged = 0
    for k in fam:
        r = find(k)
        if r != k:
            V[k]["merged_into"] = r; merged += 1
    vocab = {k: v for k, v in V.items() if "merged_into" not in v}
    s = dict(data["summary"])
    s.update(family_candidate_pairs=len(pairs), families_merged=merged, vocabulary=len(vocab),
             by_constants={c: sum(1 for v in vocab.values() if v["consts"] == c) for c in range(3)},
             by_nodes={n: sum(1 for v in vocab.values() if v["nodes"] == n) for n in range(1, 8)},
             family_seconds=round(time.time() - t0, 1), candidate_seconds=round(t1 - t0, 1))
    json.dump({"summary": s, "terms": vocab, "merged": {k: v["merged_into"] for k, v in V.items() if "merged_into" in v}},
              open(path, "w"))
    print(json.dumps(s))
