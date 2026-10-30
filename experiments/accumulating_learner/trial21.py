"""Trial 21 (evaluation only): unfamiliarity gate calibrated on new-but-familiar systems (polynomial confirmation
systems, already opened, never trained on). Same learner, pool contract and episodes as Trials 17 and 19.

  OMP_NUM_THREADS=4 python trial21.py run --seed 1 --device cuda:0
  python trial21.py analyze
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
from trial12 import feats, ridge_select
from trial13 import POOL, make_episode, standardize
from trial16 import OUT as TRIAL16
from trial17 import KS, greedy, systems_for_eval
from trial17 import OUT as TRIAL17
from trial19 import extend_coverage

OUT = S.ROOT / "outputs/accumulating_learner/trial21"
CONF = S.ROOT / "outputs/accumulating_learner/confirmation"
K0S = (4, 8, 16)
VARIANTS = {"gated_half": lambda K: K // 2, "gated_quarter": lambda K: max(4, K // 4)}


@torch.no_grad()
def best_loo(lib, xm, d, chosen):
    xc, dc = xm[chosen], d[chosen]
    uc, yc = (xc - xc.mean(0)) / (xc.std(0) + 1e-8), dc / (dc.std(0) + 1e-8)
    return min(ridge_select(feats(b, uc), yc)[1] for b in lib.bases)


def calibrate(lib, dev):
    summ = json.loads((CONF / "summary_confirmation.json").read_text())
    names = sorted(n for n, p in summ["systems_polynomial"].items() if p)
    vals = {k0: [] for k0 in K0S}
    for i, name in enumerate(names):
        x = torch.tensor(np.load(CONF / "data" / f"{name}.npz")["x"], device=dev)
        h = T8.spacing(name)
        for k0 in K0S:
            gen = torch.Generator().manual_seed(21000 + 97 * i + k0)
            for e in range(10):
                xm, d, _, _ = make_episode(x[e % x.shape[0]], gen, dev, h)
                pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                ri = torch.randperm(POOL, generator=gen)[:2].tolist()
                vals[k0].append(best_loo(lib, xm, d, extend_coverage(pool_u, ri, k0)))
    return {k0: float(np.percentile(v, 90)) for k0, v in vals.items()}, names


@torch.no_grad()
def gated(lib, xm, d, pool_u, rand_idx, K, k0, thr):
    chosen = extend_coverage(pool_u, rand_idx[:2], k0)
    unfamiliar = best_loo(lib, xm, d, chosen) > thr[k0]
    if unfamiliar and len(chosen) < K:
        return greedy(lib, xm, d, pool_u, chosen, K, "disagree_cov"), True
    return extend_coverage(pool_u, chosen, K), bool(unfamiliar)


def run(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    lib = Library(4).to(dev)
    lib.load_state_dict(torch.load(TRIAL16 / "learners" / f"catalogbio_final_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    thr, cal_names = calibrate(lib, dev)
    print("seed", s, "thresholds (90th pct best-basis LOO on new familiar systems):", thr, flush=True)
    rows = []
    for i, (name, x, h, group) in enumerate(systems_for_eval(dev)):
        for K in KS:
            gen = torch.Generator().manual_seed(17000 + 97 * i + K)  # identical to Trials 17 and 19
            batches, opened = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
            for j in range(min(4, x.shape[0])):
                for _ in range(5):
                    xm, d, xq, dq = make_episode(x[j], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    rand_idx = torch.randperm(POOL, generator=gen)[:K].tolist()
                    for v, k0f in VARIANTS.items():
                        idx, gate = gated(lib, xm, d, pool_u, rand_idx, K, k0f(K), thr)
                        batches[v].append(standardize(xm, d, idx, xq, dq)); opened[v].append(gate)
            with torch.no_grad():
                for v in VARIANTS:
                    uc, yc, uq, yq = (torch.stack(t).float() for t in zip(*batches[v]))
                    e, _ = library_fb(lib, uc, yc, uq, yq, krr_with_score(uc, yc, uq, yq))
                    rows.append({"system": name, "group": group, "K": K, "method": f"learner_{v}", "episodes": e.float().cpu().tolist(),
                                 "gate_open_rate": float(np.mean(opened[v]))})
            print(s, name, group, K, {r["method"]: (round(float(np.median(r["episodes"])), 5), round(r["gate_open_rate"], 2)) for r in rows[-2:]}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps({"thresholds": thr, "calibration_systems": cal_names, "rows": rows}))


def analyze(args):
    out = {}
    for s in SEEDS:
        mine = json.loads((OUT / f"rows_s{s}.json").read_text())
        rows = mine["rows"] + json.loads((TRIAL17 / f"rows_s{s}.json").read_text())
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        groups = {r["system"]: r["group"] for r in rows}
        rate = {(r["system"], r["K"], r["method"]): r["gate_open_rate"] for r in mine["rows"]}
        unf = sorted(n for n, g in groups.items() if g == "unfamiliar")
        fam = sorted(n for n, g in groups.items() if g == "familiar")
        med = lambda ns, K, m: float(np.median([sc[(n, K, m)] for n in ns]))
        gu = med(unf, 16, "learner_gated_half") / med(unf, 16, "learner_coverage")
        gf = med(fam, 16, "learner_gated_half") / med(fam, 16, "learner_coverage")
        ms = ("learner_coverage", "learner_disagree_cov", "learner_gated_half", "learner_gated_quarter", "krr_coverage")
        out[s] = {"thresholds": mine["thresholds"], "GU_ratio_unfamiliar_K16": gu, "GF_ratio_familiar_K16": gf,
                  "criteria": {"GU": bool(gu <= 0.83), "GF": bool(gf <= 1.0)},
                  "medians": {g: {m: {K: med(ns, K, m) for K in KS} for m in ms} for g, ns in (("unfamiliar", unf), ("familiar", fam))},
                  "gate_open_rate": {v: {g: {K: float(np.mean([rate[(n, K, f"learner_{v}")] for n in ns])) for K in KS}
                                         for g, ns in (("unfamiliar", unf), ("familiar", fam))} for v in VARIANTS},
                  "gated_half_beats_krr_cov_unfamiliar_K16": sum(sc[(n, 16, "learner_gated_half")] < sc[(n, 16, "krr_coverage")] for n in unf)}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
