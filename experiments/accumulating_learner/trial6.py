"""Trial 6: Trial 5's sequential library (seq-replay-400) trained and evaluated through a sparse head.

  python trial6.py train --seed 1 --device cuda:0
  python trial6.py eval --device cuda:0
See docs/research-log.md for the pre-registered protocol.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

import systems as S
from sparse_head import sparse_fit, sparse_loo_with_score
from trial1 import BATCH, K_GRID, LR, N_GRID, SEEDS, episode, load, nmse
from trial4 import M, RELAX, Library, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES, make_bank

OUT = S.ROOT / "outputs/accumulating_learner/trial6"
TRIAL1 = S.ROOT / "outputs/accumulating_learner/trial1"
ARM, STEPS_PER_STAGE = "seqsparse400", 400


@torch.no_grad()
def library_fb_sparse(lib, uc, yc, uq, yq, krr, return_w=False):
    """Sparse-head library + kernel-ridge fallback, chosen per episode by context LOO."""
    outs = [sparse_loo_with_score(b(uc), yc, b(uq), yq, return_w=True) for b in lib.bases]
    errs, loos, ws = zip(*outs)
    E = torch.cat([torch.stack(errs, 1), krr[0].unsqueeze(1)], 1)
    sel = torch.cat([torch.stack(loos, 1), krr[1].unsqueeze(1)], 1).argmin(1)
    e = E.gather(1, sel.unsqueeze(1)).squeeze(1)
    if not return_w:
        return e, sel
    best_basis = torch.stack(loos, 1).argmin(1)
    W = torch.stack(ws, 1)[torch.arange(len(sel)), best_basis]  # (B,F,3) of the LOO-best basis
    return e, sel, W


def train(args):
    dev = args.device
    torch.manual_seed(1000 * args.seed)  # same initial library as Trials 4-5
    lib = Library().to(dev)
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    tag = lambda t: f"{ARM}_N{t:02d}_s{args.seed}"
    torch.save(lib.state_dict(), OUT / "learners" / f"{tag(0)}.pt")
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * args.seed)
    mem_x, mem_dx, log = [], [], {"arm": ARM, "seed": args.seed, "order": order, "stages": []}
    t0, nonfinite = time.time(), 0
    for t, name in enumerate(order, 1):
        (x_t,), (dx_t,) = load([name], dev)
        eg = torch.Generator().manual_seed(9100 + 97 * t + args.seed)
        eps = [episode(x_t[0:1], dx_t[0:1], 32, eg, dev) for _ in range(20)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        e_lib, sel = library_fb_sparse(lib.eval(), uc, yc, uq, yq, krr)
        lib.train()
        new = make_bank([x_t], [dx_t], dev)
        mem = make_bank(mem_x, mem_dx, dev) if mem_x else None
        run, wins, stage_nonfinite = 0.0, torch.zeros(M), 0
        for step in range(STEPS_PER_STAGE):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / STEPS_PER_STAGE))
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            if mem is None:
                uc, yc, uq, yq = new.sample(BATCH, K, gen)
            else:
                a, b = new.sample(BATCH // 2, K, gen), mem.sample(BATCH // 2, K, gen)
                uc, yc, uq, yq = (torch.cat([p, q]) for p, q in zip(a, b))
            errs = []
            for basis in lib.bases:
                W, _ = sparse_fit(basis(uc), yc, basis.log_lam.exp())
                errs.append(nmse(basis(uq) @ W, yq).clamp_max(10.0))
            L = torch.log(torch.stack(errs, 1) + 1e-4)
            best = L.min(1)
            loss = (best.values + RELAX * L.mean(1)).mean()
            if not torch.isfinite(loss):
                stage_nonfinite += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step()
            run += best.values.exp().mean().item()
            wins += torch.bincount(best.indices.cpu(), minlength=M)
        nonfinite += stage_nonfinite
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES])
        stage = {"t": t, "system": name, "encounter_libraryfb": float(e_lib.median()), "encounter_krr": float(krr[0].median()),
                 "encounter_selection": torch.bincount(sel.cpu(), minlength=M + 1).tolist(),
                 "train_min_nmse": run / max(1, STEPS_PER_STAGE - stage_nonfinite), "nonfinite_steps": stage_nonfinite,
                 "wins": wins.tolist(), "seconds": time.time() - t0}
        log["stages"].append(stage)
        print(ARM, args.seed, t, name, f"enc lib {stage['encounter_libraryfb']:.4f} krr {stage['encounter_krr']:.4f}",
              f"train {stage['train_min_nmse']:.4f} nonfinite {stage_nonfinite}", f"{stage['seconds']:.0f}s", flush=True)
        if t in N_GRID:
            torch.save(lib.state_dict(), OUT / "learners" / f"{tag(t)}.pt")
    log["nonfinite_steps_total"] = nonfinite
    (OUT / "learners" / f"{ARM}_s{args.seed}.json").write_text(json.dumps(log))


def load_libs(dev):
    libs = {}
    for t in N_GRID:
        for s in SEEDS:
            lib = Library().to(dev)
            lib.load_state_dict(torch.load(OUT / "learners" / f"{ARM}_N{t:02d}_s{s}.pt", map_location=dev, weights_only=True))
            libs[(t, s)] = lib.eval()
    return libs


def evaluate(args):
    dev = args.device
    dev_names = S.eligible_split()["development"]
    xs, dxs = load(dev_names, dev)
    libs = load_libs(dev)
    rows = [r for r in json.loads((TRIAL1 / "eval_development.json").read_text())["rows"]
            if not r["method"].startswith("learned")]
    sparsity = []
    for si, name in enumerate(dev_names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev)
                   for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            summary = {}
            for (t, s), lib in libs.items():
                e, sel, W = library_fb_sparse(lib, uc, yc, uq, yq, krr, return_w=True)
                rows.append({"system": name, "K": K, "method": f"{ARM}_N{t:02d}_s{s}", "episodes": e.float().cpu().tolist()})
                if t == 51:
                    a = W.abs()
                    frac_small = (a < 0.01 * a.amax(1, keepdim=True)).float().mean((1, 2))
                    sparsity.append({"system": name, "K": K, "seed": s, "frac_coeffs_below_1pct": float(frac_small.median()),
                                     "selection": torch.bincount(sel.cpu(), minlength=M + 1).tolist()})
                    summary[f"s{s}"] = round(float(np.median(e.float().cpu().numpy())), 4)
            print(name, K, summary, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows, "sparsity": sparsity}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)
