"""Trial 19 (evaluation only): disagreement-guided observation only when the system looks unfamiliar.
Same learner, pool contract, systems and episodes as Trial 17. See docs/research-log.md.

  OMP_NUM_THREADS=4 python trial19.py run --seed 1 --device cuda:0
  python trial19.py analyze
"""

import argparse
import json

import numpy as np
import torch

import systems as S
from trial1 import SEEDS
from trial4 import Library, krr_with_score
from trial5 import library_fb
from trial12 import feats, krr_select, ridge_select
from trial13 import POOL, make_episode, standardize
from trial16 import OUT as TRIAL16
from trial17 import KS, greedy, systems_for_eval
from trial17 import OUT as TRIAL17

OUT = S.ROOT / "outputs/accumulating_learner/trial19"


def extend_coverage(pool_u, chosen, K):
    chosen = list(chosen)
    dmin = torch.cdist(pool_u, pool_u[chosen]).min(1).values
    while len(chosen) < K:
        j = int(dmin.argmax())
        chosen.append(j)
        dmin = torch.minimum(dmin, torch.cdist(pool_u, pool_u[j:j + 1]).squeeze(1))
    return chosen


@torch.no_grad()
def gated(lib, xm, d, pool_u, rand_idx, K):
    k0 = max(4, K // 4)
    chosen = extend_coverage(pool_u, rand_idx[:2], k0)
    xc, dc = xm[chosen], d[chosen]
    uc, yc = (xc - xc.mean(0)) / (xc.std(0) + 1e-8), dc / (dc.std(0) + 1e-8)
    loos = [ridge_select(feats(b, uc), yc)[1] for b in lib.bases] + [krr_select(uc, yc)[2]]
    unfamiliar = int(np.argmin(loos)) == len(lib.bases)
    if unfamiliar:
        return greedy(lib, xm, d, pool_u, chosen, K, "disagree_cov"), True
    return extend_coverage(pool_u, chosen, K), False


def run(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    lib = Library(4).to(dev)
    lib.load_state_dict(torch.load(TRIAL16 / "learners" / f"catalogbio_final_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    rows = []
    for i, (name, x, h, group) in enumerate(systems_for_eval(dev)):
        for K in KS:
            gen = torch.Generator().manual_seed(17000 + 97 * i + K)  # identical to Trial 17
            batch, opened = [], []
            for j in range(min(4, x.shape[0])):
                for _ in range(5):
                    xm, d, xq, dq = make_episode(x[j], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    rand_idx = torch.randperm(POOL, generator=gen)[:K].tolist()
                    idx, gate = gated(lib, xm, d, pool_u, rand_idx, K)
                    opened.append(gate)
                    batch.append(standardize(xm, d, idx, xq, dq))
            with torch.no_grad():
                uc, yc, uq, yq = (torch.stack(v).float() for v in zip(*batch))
                e, _ = library_fb(lib, uc, yc, uq, yq, krr_with_score(uc, yc, uq, yq))
            rows.append({"system": name, "group": group, "K": K, "method": "learner_gated", "episodes": e.float().cpu().tolist(),
                         "gate_open_rate": float(np.mean(opened))})
            print(s, name, group, K, round(float(np.median(e.float().cpu().numpy())), 5), "gate", round(float(np.mean(opened)), 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps(rows))


def analyze(args):
    out = {}
    for s in SEEDS:
        rows = json.loads((OUT / f"rows_s{s}.json").read_text()) + json.loads((TRIAL17 / f"rows_s{s}.json").read_text())
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        groups = {r["system"]: r["group"] for r in rows}
        gate = {(r["system"], r["K"]): r["gate_open_rate"] for r in rows if r["method"] == "learner_gated"}
        unf = sorted(n for n, g in groups.items() if g == "unfamiliar")
        fam = sorted(n for n, g in groups.items() if g == "familiar")
        med = lambda ns, K, m: float(np.median([sc[(n, K, m)] for n in ns]))
        gu = med(unf, 16, "learner_gated") / med(unf, 16, "learner_coverage")
        gf = med(fam, 16, "learner_gated") / med(fam, 16, "learner_coverage")
        out[s] = {"GU_ratio_unfamiliar_K16": gu, "GF_ratio_familiar_K16": gf,
                  "criteria": {"GU": bool(gu <= 0.83), "GF": bool(gf <= 1.0)},
                  "medians": {grp: {m: {K: med(ns, K, m) for K in KS} for m in ("learner_coverage", "learner_disagree_cov", "learner_gated", "krr_coverage")}
                              for grp, ns in (("unfamiliar", unf), ("familiar", fam))},
                  "gate_open_rate": {grp: {K: float(np.mean([gate[(n, K)] for n in ns])) for K in KS} for grp, ns in (("unfamiliar", unf), ("familiar", fam))}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
