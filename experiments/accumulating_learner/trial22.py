"""Trial 22 (evaluation only): gate on relative fit (best basis vs kernel ridge LOO) at half budget, fresh episodes,
with coverage and disagreement baselines started from the same two random points. See docs/research-log.md.

  OMP_NUM_THREADS=4 python trial22.py run --seed 1 --device cuda:0
  python trial22.py analyze
"""

import argparse
import json

import numpy as np
import torch

import systems as S
import trial8 as T8
from trial1 import SEEDS
from trial4 import Library, krr_with_score
from trial5 import library_fb
from trial12 import feats, krr_select, ridge_select
from trial13 import POOL, make_episode, standardize
from trial16 import OUT as TRIAL16
from trial17 import KS, greedy, systems_for_eval
from trial19 import extend_coverage
from trial21 import CONF

OUT = S.ROOT / "outputs/accumulating_learner/trial22"
STRATS = ("coverage2", "disagree2", "gated_rel")


@torch.no_grad()
def rel_fit(lib, xm, d, chosen):
    xc, dc = xm[chosen], d[chosen]
    uc, yc = (xc - xc.mean(0)) / (xc.std(0) + 1e-8), dc / (dc.std(0) + 1e-8)
    best = min(ridge_select(feats(b, uc), yc)[1] for b in lib.bases)
    return best / max(krr_select(uc, yc)[2], 1e-12)


def calibrate(lib, dev):
    summ = json.loads((CONF / "summary_confirmation.json").read_text())
    names = sorted(n for n, p in summ["systems_polynomial"].items() if p)
    vals = {k0: [] for k0 in (4, 8, 16)}
    for i, name in enumerate(names):
        x = torch.tensor(np.load(CONF / "data" / f"{name}.npz")["x"], device=dev)
        for k0 in vals:
            gen = torch.Generator().manual_seed(22500 + 97 * i + k0)
            for e in range(10):
                xm, d, _, _ = make_episode(x[e % x.shape[0]], gen, dev, T8.spacing(name))
                pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                ri = torch.randperm(POOL, generator=gen)[:2].tolist()
                vals[k0].append(rel_fit(lib, xm, d, extend_coverage(pool_u, ri, k0)))
    return {k0: float(np.percentile(v, 90)) for k0, v in vals.items()}


def run(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    lib = Library(4).to(dev)
    lib.load_state_dict(torch.load(TRIAL16 / "learners" / f"catalogbio_final_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    thr = calibrate(lib, dev)
    print("seed", s, "relative-fit thresholds:", thr, flush=True)
    rows, stat = [], []
    for i, (name, x, h, group) in enumerate(systems_for_eval(dev)):
        for K in KS:
            gen = torch.Generator().manual_seed(22000 + 97 * i + K)
            k0 = K // 2
            batches, opened = {st: [] for st in STRATS}, []
            for j in range(min(4, x.shape[0])):
                for _ in range(5):
                    xm, d, xq, dq = make_episode(x[j], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    ri = torch.randperm(POOL, generator=gen)[:2].tolist()
                    pre = extend_coverage(pool_u, ri, k0)
                    r = rel_fit(lib, xm, d, pre)
                    gate = r > thr[k0]
                    stat.append({"system": name, "group": group, "K": K, "r": r})
                    opened.append(gate)
                    cov = extend_coverage(pool_u, pre, K)
                    dis = greedy(lib, xm, d, pool_u, ri, K, "disagree_cov")
                    idx = {"coverage2": cov, "disagree2": dis, "gated_rel": greedy(lib, xm, d, pool_u, pre, K, "disagree_cov") if gate else cov}
                    for st in STRATS:
                        batches[st].append(standardize(xm, d, idx[st], xq, dq))
            with torch.no_grad():
                for st in STRATS:
                    uc, yc, uq, yq = (torch.stack(t).float() for t in zip(*batches[st]))
                    krr = krr_with_score(uc, yc, uq, yq)
                    e, _ = library_fb(lib, uc, yc, uq, yq, krr)
                    rows.append({"system": name, "group": group, "K": K, "method": f"learner_{st}", "episodes": e.float().cpu().tolist()})
                    if st == "coverage2":
                        rows.append({"system": name, "group": group, "K": K, "method": "krr_coverage2", "episodes": krr[0].float().cpu().tolist()})
            rows.append({"system": name, "group": group, "K": K, "method": "gate", "open_rate": float(np.mean(opened))})
            print(s, name, group, K, {r_["method"]: round(float(np.median(r_["episodes"])), 5) for r_ in rows[-5:-1]}, "open", round(float(np.mean(opened)), 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps({"thresholds": thr, "rows": rows, "stat": stat}))


def auc(a, b):
    a, b = np.array(a), np.array(b)
    return float(((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()))


def analyze(args):
    out = {}
    for s in SEEDS:
        D = json.loads((OUT / f"rows_s{s}.json").read_text())
        rows = [r for r in D["rows"] if "episodes" in r]
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        groups = {r["system"]: r["group"] for r in rows}
        rate = {(r["system"], r["K"]): r["open_rate"] for r in D["rows"] if r["method"] == "gate"}
        unf = sorted(n for n, g in groups.items() if g == "unfamiliar"); fam = sorted(n for n, g in groups.items() if g == "familiar")
        med = lambda ns, K, m: float(np.median([sc[(n, K, m)] for n in ns]))
        gu = med(unf, 16, "learner_gated_rel") / med(unf, 16, "learner_coverage2")
        gf = med(fam, 16, "learner_gated_rel") / med(fam, 16, "learner_coverage2")
        ms = ("learner_coverage2", "learner_disagree2", "learner_gated_rel", "krr_coverage2")
        out[s] = {"thresholds": D["thresholds"], "GU_ratio": gu, "GF_ratio": gf, "criteria": {"GU": bool(gu <= 0.83), "GF": bool(gf <= 1.0)},
                  "medians": {g: {m: {K: med(ns, K, m) for K in KS} for m in ms} for g, ns in (("unfamiliar", unf), ("familiar", fam))},
                  "gate_open_rate": {g: {K: float(np.mean([rate[(n, K)] for n in ns])) for K in KS} for g, ns in (("unfamiliar", unf), ("familiar", fam))},
                  "auc_r_fresh": {K: auc([x["r"] for x in D["stat"] if x["K"] == K and x["group"] == "unfamiliar"],
                                         [x["r"] for x in D["stat"] if x["K"] == K and x["group"] == "familiar"]) for K in KS},
                  "gated_beats_krr_cov_unfamiliar_K16": sum(sc[(n, 16, "learner_gated_rel")] < sc[(n, 16, "krr_coverage2")] for n in unf)}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
