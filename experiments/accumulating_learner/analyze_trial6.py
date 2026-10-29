"""Pre-registered Trial 6 criteria G1-G2 against Trial 5's seq-replay-400 (G3 = C1 and C2 from analyze_trial1.py
--prefix seqsparse400 --out <trial6>). Also reports the gap closed toward poly2-sparse at K = 8 and coefficient sparsity."""

import json

import numpy as np
import torch

import systems as S
from sparse_head import sparse_loo_with_score
from trial1 import SEEDS, episode, load, poly_features
from trial5 import OUT as TRIAL5
from trial6 import ARM, OUT


def poly2_sparse_k8(names, dev="cuda:0"):
    xs, dxs = load(names, dev)
    out = {}
    with torch.no_grad():
        for si, name in enumerate(names):
            gen = torch.Generator().manual_seed(5000 + 97 * si + 8)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], 8, gen, dev) for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            e, _ = sparse_loo_with_score(poly_features(uc, 2), yc, poly_features(uq, 2), yq)
            out[name] = float(np.median(e.float().cpu().numpy()))
    return out


def main():
    ev = json.loads((OUT / "eval_development.json").read_text())
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in ev["rows"]}
    t5 = json.loads((TRIAL5 / "eval_development.json").read_text())["rows"]
    score.update({(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t5
                  if r["method"].startswith("seqreplay400_N51")})
    systems = S.eligible_split()["development"]
    med = lambda m, K: float(np.median([score[(s, K, m)] for s in systems]))
    ps = poly2_sparse_k8(systems)
    ps_agg = float(np.median(list(ps.values())))
    out = {"poly2_sparse_K8": ps_agg}
    for seed in SEEDS:
        new, ref = f"{ARM}_N51_s{seed}", f"seqreplay400_N51_s{seed}"
        r8, r32 = med(new, 8) / med(ref, 8), med(new, 32) / med(ref, 32)
        wins8 = sum(score[(s, 8, new)] < score[(s, 8, ref)] for s in systems)
        gap = (np.log(med(ref, 8)) - np.log(med(new, 8))) / (np.log(med(ref, 8)) - np.log(ps_agg))
        sp = [x["frac_coeffs_below_1pct"] for x in ev["sparsity"] if x["seed"] == seed and x["K"] == 32]
        crit = {"G1": bool(r8 <= 0.75 and wins8 >= 9), "G2": bool(r32 <= 1.5)}
        out[seed] = {"K8": {"new": med(new, 8), "ref": med(ref, 8), "ratio": r8, "wins": wins8},
                     "K16": {"new": med(new, 16), "ref": med(ref, 16)},
                     "K32": {"new": med(new, 32), "ref": med(ref, 32), "ratio": r32},
                     "log_gap_closed_toward_poly2_sparse_K8": float(gap),
                     "median_frac_coeffs_below_1pct_K32": float(np.median(sp)), "criteria": crit}
        print(f"seed {seed}:", json.dumps(out[seed], default=float))
    print("poly2-sparse K=8 aggregate:", round(ps_agg, 4))
    (OUT / "summary_mechanism.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
