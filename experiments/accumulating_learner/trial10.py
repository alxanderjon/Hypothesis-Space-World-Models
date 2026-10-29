"""Trial 10: prequential test-time accumulation for the observable-only library. See docs/research-log.md.

Each development system in a random order is scored at encounter (before learning), then learned from (observable
data only, with replay). Arms: continual, replay-only (equal compute, never learns from stream systems), frozen.

  python trial10.py run --seed 1 --order 0 --device cuda:0
  python trial10.py analyze
"""

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
from torch import nn

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import BATCH, LR, SEEDS, load, nmse, ridge
from trial4 import RELAX, Library, krr_with_score
from trial5 import LR_MIN, library_fb
from trial9 import OUT as TRIAL9
from trial9 import ObsBank, episode_obs

OUT = S.ROOT / "outputs/accumulating_learner/trial10"
ORDERS, STEPS, MEM_INST = 6, 400, 2


@torch.no_grad()
def score(lib, x, dx, si, h, dev, Ks=(16, 32)):
    """Trial 9's evaluation episodes for development system index si; np.median over episodes."""
    out = {}
    for K in Ks:
        gen = torch.Generator().manual_seed(5000 + 97 * si + K)
        eps = [episode_obs(x[i:i + 1], dx[i:i + 1], K, gen, dev, h) for i in range(x.shape[0]) for _ in range(5)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        e, _ = library_fb(lib.eval(), uc, yc, uq, yq, krr_with_score(uc, yc, uq, yq))
        out[K] = float(np.median(e.float().cpu().numpy()))
    return out


@torch.no_grad()
def retention(lib, seed, dev):
    order = list(np.random.default_rng(seed).permutation(S.eligible_split()["meta_train"]))[:8]
    xs, dxs = load(order, dev)
    vals = []
    for pos, name in enumerate(order):
        gen = torch.Generator().manual_seed(7000 + 97 * pos + seed)
        eps = [episode_obs(xs[pos][0:1], dxs[pos][0:1], 32, gen, dev, T8.spacing(name)) for _ in range(20)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        e, _ = library_fb(lib.eval(), uc, yc, uq, yq, krr_with_score(uc, yc, uq, yq))
        vals.append(float(np.median(e.float().cpu().numpy())))
    return vals


def learn(lib, new, mem, gen, mode):
    opt = learn.opt
    lib.train()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / STEPS))
        K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
        if mode == "continual":
            a, b = new.sample(BATCH // 2, K, gen), mem.sample(BATCH // 2, K, gen)
            uc, yc, uq, yq = (torch.cat([p, q]) for p, q in zip(a, b))
        else:
            uc, yc, uq, yq = mem.sample(BATCH, K, gen)
        errs = []
        for basis in lib.bases:
            W = ridge(basis(uc), yc, basis.log_lam.exp())
            errs.append(nmse(basis(uq) @ W, yq).clamp_max(10.0))
        L = torch.log(torch.stack(errs, 1) + 1e-4)
        loss = (L.min(1).values + RELAX * L.mean(1)).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
        opt.step()


def run(args):
    dev, s, r = args.device, args.seed, args.order
    devnames = S.eligible_split()["development"]
    order = list(np.random.default_rng(1000 * s + r).permutation(devnames))
    xs, dxs = load(devnames, dev)
    meta = S.eligible_split()["meta_train"]
    mx, mdx = load(meta, dev)
    base_x, base_dx, base_h = [x[:MEM_INST] for x in mx], [d[:MEM_INST] for d in mdx], [T8.spacing(n) for n in meta]
    start = Library().to(dev)
    start.load_state_dict(torch.load(TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", map_location=dev, weights_only=True))
    frozen = {n: score(start, xs[devnames.index(n)], dxs[devnames.index(n)], devnames.index(n), T8.spacing(n), dev)
              for n in devnames}
    result = {"seed": s, "order_index": r, "order": order, "frozen": frozen,
              "retention_before": retention(start, s, dev), "arms": {}}
    for mode in ("continual", "replayonly"):
        lib = copy.deepcopy(start)
        learn.opt = torch.optim.Adam(lib.parameters(), lr=LR)
        gen = torch.Generator().manual_seed(300 + 1000 * s + 10 * r)
        mem_x, mem_dx, mem_h = list(base_x), list(base_dx), list(base_h)
        stages, t0 = [], time.time()
        for pos, name in enumerate(order, 1):
            si = devnames.index(name)
            h = T8.spacing(name)
            enc = score(lib, xs[si], dxs[si], si, h, dev)
            new = ObsBank([xs[si]], [dxs[si]], [h], dev)
            mem = ObsBank(mem_x, mem_dx, mem_h, dev)
            learn(lib, new, mem, gen, mode)
            if mode == "continual":
                mem_x.append(xs[si][:MEM_INST]); mem_dx.append(dxs[si][:MEM_INST]); mem_h.append(h)
            stages.append({"position": pos, "system": name, "encounter": enc})
            print(s, r, mode, pos, name, enc, f"{time.time() - t0:.0f}s", flush=True)
        result["arms"][mode] = {"stages": stages, "retention_after": retention(lib, s, dev)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stream_s{s}_o{r}.json").write_text(json.dumps(result))


def analyze(args):
    out = {}
    for s in SEEDS:
        pairs = []  # (position, system, L32, L16, Lfrozen32)
        ret = []
        for r in range(ORDERS):
            res = json.loads((OUT / f"stream_s{s}_o{r}.json").read_text())
            c = {st["system"]: st for st in res["arms"]["continual"]["stages"]}
            p = {st["system"]: st for st in res["arms"]["replayonly"]["stages"]}
            for name, st in c.items():
                ce, pe, fe = st["encounter"], p[name]["encounter"], res["frozen"][name]
                pairs.append((st["position"], name, math.log(ce["32"] / pe["32"]), math.log(ce["16"] / pe["16"]),
                              math.log(ce["32"] / fe["32"])))
            ret.append(np.median(np.array(res["arms"]["continual"]["retention_after"]) / np.array(res["retention_before"])))
        pos = np.array([q[0] for q in pairs]); L32 = np.array([q[2] for q in pairs]); L16 = np.array([q[3] for q in pairs])
        late = pos >= 7
        nonpoly_late = late & np.array([q[1] in NONPOLY for q in pairs])
        from scipy.stats import spearmanr
        rho32, rho16 = spearmanr(pos, L32).correlation, spearmanr(pos, L16).correlation
        crit = {"P1": bool(rho32 <= -0.3 and np.median(L32[late]) < 0),
                "P2": bool(np.median(L32[nonpoly_late]) < 0),
                "P3": bool(np.mean(ret) <= 1.5)}
        per_system_late = {n: float(np.median([q[2] for q in pairs if q[1] == n and q[0] >= 7] or [np.nan]))
                           for n in sorted({q[1] for q in pairs})}
        out[s] = {"spearman_K32": rho32, "spearman_K16": rho16, "median_L32_late": float(np.median(L32[late])),
                  "median_L16_late": float(np.median(L16[late])), "median_L32_nonpoly_late": float(np.median(L32[nonpoly_late])),
                  "median_Lfrozen32_late": float(np.median(np.array([q[4] for q in pairs])[late])),
                  "retention_ratio_mean_over_orders": float(np.mean(ret)), "per_system_median_L32_late": per_system_late,
                  "criteria": crit}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
