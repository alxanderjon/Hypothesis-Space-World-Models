"""Trial 17 (evaluation only): observe where the learner's stored hypotheses disagree, combined with coverage.
See docs/research-log.md.

  OMP_NUM_THREADS=4 python trial17.py run --seed 1 --device cuda:0
  python trial17.py analyze
"""

import argparse
import json

import numpy as np
import torch

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import SEEDS, load
from trial4 import Library, krr_with_score
from trial5 import library_fb
from trial12 import feats, ridge_fit, ridge_select
from trial13 import POOL, coverage, make_episode, standardize
from trial16 import OUT as TRIAL16
from trial16 import load_any, real_info

OUT = S.ROOT / "outputs/accumulating_learner/trial17"
KS = (8, 16, 32)
RESELECT = {2, 4, 8, 16}
STRATS = ("random", "coverage", "disagree_cov", "leverage_cov")


@torch.no_grad()
def greedy(lib, xm, d, pool_u, start, K, mode):
    chosen = list(start)
    lams = None
    while len(chosen) < K:
        xc, dc = xm[chosen], d[chosen]
        mx, sx, sd = xc.mean(0), xc.std(0) + 1e-8, dc.std(0) + 1e-8
        uc, yc = (xc - mx) / sx, dc / sd
        if lams is None or len(chosen) in RESELECT:
            sel = [ridge_select(feats(b, uc), yc) for b in lib.bases]
            lams = [lam for lam, _ in sel]
            best = int(np.argmin([v for _, v in sel]))
        upool = (xm - mx) / sx
        dmin = torch.cdist(pool_u, pool_u[chosen]).min(1).values
        if mode == "disagree_cov":
            preds = []
            for b, lam in zip(lib.bases, lams):
                P = feats(b, upool)
                preds.append(P @ ridge_fit(P[chosen], yc, lam))
            score = torch.stack(preds).std(0).pow(2).mean(1).sqrt()
        else:
            P = feats(lib.bases[best], upool)
            Pc = P[chosen]
            k, F = Pc.shape
            Ainv = torch.linalg.inv(Pc.T @ Pc + lams[best] * k * torch.eye(F, device=P.device, dtype=P.dtype))
            score = ((P @ Ainv) * P).sum(1).clamp_min(0).sqrt()
        score = score * dmin
        score[chosen] = -float("inf")
        chosen.append(int(score.argmax()))
    return chosen


def systems_for_eval(dev):
    devnames = S.eligible_split()["development"]
    xs, _ = load(devnames, dev)
    out = [(n, xs[i], T8.spacing(n), "familiar" if n not in NONPOLY else "unfamiliar") for i, n in enumerate(devnames)]
    split, conv = real_info()
    for n in sorted(k for k, v in split.items() if v == "heldout"):
        x, _, h = load_any(n, dev, conv)
        out.append((n, x, h, "unfamiliar"))
    return out


def run(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    lib = Library(4).to(dev)
    lib.load_state_dict(torch.load(TRIAL16 / "learners" / f"catalogbio_final_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    rows = []
    for i, (name, x, h, group) in enumerate(systems_for_eval(dev)):
        for K in KS:
            gen = torch.Generator().manual_seed(17000 + 97 * i + K)
            batches = {st: [] for st in STRATS}
            for j in range(min(4, x.shape[0])):  # SprottQ has 3 instances (Trial 1 record)
                for _ in range(5):
                    xm, d, xq, dq = make_episode(x[j], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    rand_idx = torch.randperm(POOL, generator=gen)[:K].tolist()
                    idx = {"random": rand_idx, "coverage": coverage(pool_u, rand_idx[0], K),
                           "disagree_cov": greedy(lib, xm, d, pool_u, rand_idx[:2], K, "disagree_cov"),
                           "leverage_cov": greedy(lib, xm, d, pool_u, rand_idx[:2], K, "leverage_cov")}
                    for st in STRATS:
                        batches[st].append(standardize(xm, d, idx[st], xq, dq))
            with torch.no_grad():
                for st in STRATS:
                    uc, yc, uq, yq = (torch.stack(v).float() for v in zip(*batches[st]))
                    krr = krr_with_score(uc, yc, uq, yq)
                    e, _ = library_fb(lib, uc, yc, uq, yq, krr)
                    rows.append({"system": name, "group": group, "K": K, "method": f"learner_{st}", "episodes": e.float().cpu().tolist()})
                    if st == "coverage":
                        rows.append({"system": name, "group": group, "K": K, "method": "krr_coverage", "episodes": krr[0].float().cpu().tolist()})
            print(s, name, group, K, {r["method"]: round(float(np.median(r["episodes"])), 5) for r in rows[-5:]}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps(rows))


def analyze(args):
    out = {}
    for s in SEEDS:
        rows = json.loads((OUT / f"rows_s{s}.json").read_text())
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        groups = {r["system"]: r["group"] for r in rows}
        unf = sorted(n for n, g in groups.items() if g == "unfamiliar")
        fam = sorted(n for n, g in groups.items() if g == "familiar")
        med = lambda ns, K, m: float(np.median([sc[(n, K, m)] for n in ns]))
        u1 = med(unf, 16, "learner_disagree_cov") / med(unf, 16, "learner_coverage")
        u2 = med(fam, 16, "learner_disagree_cov") / med(fam, 16, "learner_coverage")
        methods = sorted({r["method"] for r in rows})
        out[s] = {"U1_ratio_unfamiliar_K16": u1, "U2_ratio_familiar_K16": u2,
                  "criteria": {"U1": bool(u1 <= 0.83), "U2": bool(u2 <= 1.0)},
                  "unfamiliar": {m: {K: med(unf, K, m) for K in KS} for m in methods},
                  "familiar": {m: {K: med(fam, K, m) for K in KS} for m in methods},
                  "wins_disagree_vs_coverage_unfamiliar_K16": sum(sc[(n, 16, "learner_disagree_cov")] < sc[(n, 16, "learner_coverage")] for n in unf)}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
