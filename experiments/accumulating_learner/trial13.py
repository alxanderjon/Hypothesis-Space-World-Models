"""Trial 13 (evaluation only): pool-based active observation guided by experience. See docs/research-log.md.

  python trial13.py run --seed 1 --device cuda:0
  python trial13.py analyze
"""

import argparse
import json

import numpy as np
import torch

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import N_QUERY, SEEDS, load, nmse
from trial4 import Library, krr_with_score
from trial5 import library_fb
from trial9 import OUT as TRIAL9
from trial12 import feats, krr_select, ridge_select

OUT = S.ROOT / "outputs/accumulating_learner/trial13"
KS, POOL = (8, 16, 32), 400
STRATS = ("random", "coverage", "experience")


def make_episode(x, gen, dev, h):
    """One instance x (T,3): pool (noisy states, hidden derivative estimates) and clean queries."""
    T = x.shape[0]
    ci = (1 + torch.randperm(T // 2 - 2, generator=gen)[:POOL]).to(dev)
    qi = (T // 2 + 1 + torch.randperm(T // 2 - 2, generator=gen)[:N_QUERY]).to(dev)
    noise = torch.randn(3, POOL, 3, generator=gen).to(dev) * T8.NOISE_X * x.std(0)
    xp, xm, xn = x[ci - 1] + noise[0], x[ci] + noise[1], x[ci + 1] + noise[2]
    return xm.double(), ((xn - xp) / (2 * h)).double(), x[qi].double(), ((x[qi + 1] - x[qi - 1]) / (2 * h)).double()


def standardize(xm, d, idx, xq, dq):
    xc, dc = xm[idx], d[idx]
    mx, sx, sd = xc.mean(0), xc.std(0) + 1e-8, dc.std(0) + 1e-8
    return (xc - mx) / sx, dc / sd, (xq - mx) / sx, dq / sd


def coverage(pool_u, start, K):
    chosen = [start]
    dmin = torch.cdist(pool_u, pool_u[start:start + 1]).squeeze(1)
    while len(chosen) < K:
        j = int(dmin.argmax())
        chosen.append(j)
        dmin = torch.minimum(dmin, torch.cdist(pool_u, pool_u[j:j + 1]).squeeze(1))
    return chosen


@torch.no_grad()
def experience(lib, xm, d, rand_idx, K, pool_u):
    first = list(rand_idx[: K // 2])
    xc, dc = xm[first], d[first]
    mx, sx, sd = xc.mean(0), xc.std(0) + 1e-8, dc.std(0) + 1e-8
    uc, yc = (xc - mx) / sx, dc / sd
    sels = [ridge_select(feats(b, uc), yc) for b in lib.bases]
    kr = krr_select(uc, yc)
    loos = [v for _, v in sels] + [kr[2]]
    o = int(np.argmin(loos))
    if o == len(lib.bases):  # kernel ridge selected: complete the design by coverage
        chosen = list(first)
        dmin = torch.cdist(pool_u, pool_u[chosen]).min(1).values
        while len(chosen) < K:
            j = int(dmin.argmax()); chosen.append(j)
            dmin = torch.minimum(dmin, torch.cdist(pool_u, pool_u[j:j + 1]).squeeze(1))
        return chosen, "krr"
    lam = sels[o][0]
    Phi_pool = feats(lib.bases[o], (xm - mx) / sx)
    Phi = Phi_pool[first]
    Fd = Phi.shape[1]
    Minv = torch.linalg.inv(Phi.T @ Phi + lam * K * torch.eye(Fd, device=Phi.device, dtype=Phi.dtype))
    chosen = list(first)
    taken = torch.zeros(len(xm), dtype=torch.bool, device=xm.device); taken[first] = True
    while len(chosen) < K:
        lev = ((Phi_pool @ Minv) * Phi_pool).sum(1).masked_fill(taken, -1.0)
        j = int(lev.argmax())
        chosen.append(j); taken[j] = True
        v = Minv @ Phi_pool[j]
        Minv = Minv - torch.outer(v, v) / (1 + Phi_pool[j] @ v)
    return chosen, f"basis{o}"


def run(args):
    dev, s = args.device, args.seed
    names = S.eligible_split()["development"]
    xs, dxs = load(names, dev)  # dxs is never used
    lib = Library().to(dev)
    lib.load_state_dict(torch.load(TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    rows = []
    for si, name in enumerate(names):
        h = T8.spacing(name)
        for K in KS:
            gen = torch.Generator().manual_seed(13000 + 97 * si + K)
            batches = {st: [] for st in STRATS}
            picks = []
            for i in range(xs[si].shape[0]):
                for _ in range(5):
                    xm, d, xq, dq = make_episode(xs[si][i], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    rand_idx = torch.randperm(POOL, generator=gen)[:K].tolist()
                    idx = {"random": rand_idx, "coverage": coverage(pool_u, rand_idx[0], K)}
                    idx["experience"], pick = experience(lib, xm, d, rand_idx, K, pool_u)
                    picks.append(pick)
                    for st in STRATS:
                        batches[st].append(standardize(xm, d, idx[st], xq, dq))
            with torch.no_grad():
                for st in STRATS:
                    uc, yc, uq, yq = (torch.stack(v).float() for v in zip(*batches[st]))
                    krr = krr_with_score(uc, yc, uq, yq)
                    e, _ = library_fb(lib, uc, yc, uq, yq, krr)
                    rows.append({"system": name, "K": K, "method": f"learner_{st}", "episodes": e.float().cpu().tolist()})
                    if st in ("random", "coverage"):
                        rows.append({"system": name, "K": K, "method": f"krr_{st}", "episodes": krr[0].float().cpu().tolist()})
            rows.append({"system": name, "K": K, "method": "experience_picks", "picks": picks})
            print(s, name, K, {r["method"]: round(float(np.median(r["episodes"])), 5) for r in rows[-6:-1]}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps(rows))


def analyze(args):
    out = {}
    for s in SEEDS:
        rows = [r for r in json.loads((OUT / f"rows_s{s}.json").read_text()) if "episodes" in r]
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        names = sorted({r["system"] for r in rows})
        med = lambda m, K, ns=names: float(np.median([sc[(n, K, m)] for n in ns]))
        methods = sorted({r["method"] for r in rows})
        i1 = med("learner_experience", 16) / med("learner_random", 16)
        i2 = med("learner_experience", 16) / med("learner_coverage", 16)
        out[s] = {"I1_ratio_K16": i1, "I2_ratio_K16": i2, "criteria": {"I1": bool(i1 <= 0.67), "I2": bool(i2 <= 0.83)},
                  "aggregates": {m: {K: med(m, K) for K in KS} for m in methods},
                  "nonpoly": {m: {K: med(m, K, sorted(NONPOLY)) for K in KS} for m in methods},
                  "wins_experience_vs_random_K16": sum(sc[(n, 16, "learner_experience")] < sc[(n, 16, "learner_random")] for n in names)}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
