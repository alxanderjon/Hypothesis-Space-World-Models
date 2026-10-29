"""Trial 4: a library of M bases with leave-one-out selection and a generic kernel-ridge fallback.

Training: M = 4 copies of Trial 1's feature network, relaxed winner-take-all on per-task log NMSE.
Encounter: each basis (ridge, lambda by LOO) and kernel ridge (length scale, lambda by LOO) are scored by their
leave-one-out MSE on the K context observations; the lowest is used. See docs/research-log.md.

  python trial4.py train --n 8 --seed 1 --device cuda:0
  python trial4.py eval --device cuda:0
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

import systems as S
from trial1 import (BATCH, K_GRID, LAMBDAS, LR, N_GRID, SEEDS, STEPS, Phi, episode, load, nmse, ridge,
                    ridge_loo_with_score)
from trial3 import Bank, Shared

OUT = S.ROOT / "outputs/accumulating_learner/trial4"
TRIAL1 = S.ROOT / "outputs/accumulating_learner/trial1"
TRIAL3 = S.ROOT / "outputs/accumulating_learner/trial3"
M = 4
RELAX = 0.05


class Library(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.bases = nn.ModuleList([Phi() for _ in range(m)])


def train(args):
    torch.manual_seed(1000 * args.seed + args.n)
    dev = args.device
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    names = order[: args.n]
    lib = Library().to(dev)
    tag = f"library_N{args.n:02d}_s{args.seed}"
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    log = {"n": args.n, "seed": args.seed, "systems": names, "params": sum(p.numel() for p in lib.parameters()),
           "steps": 0, "loss": [], "wins": []}
    if args.n > 0:
        bank = Bank(names, dev)
        opt = torch.optim.Adam(lib.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        gen = torch.Generator().manual_seed(77 + 1000 * args.seed + args.n)
        t0, run, wins = time.time(), 0.0, torch.zeros(M)
        for step in range(STEPS):
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            uc, yc, uq, yq = bank.sample(BATCH, K, gen)
            errs = []
            for b in lib.bases:
                W = ridge(b(uc), yc, b.log_lam.exp())
                errs.append(nmse(b(uq) @ W, yq).clamp_max(10.0))
            L = torch.log(torch.stack(errs, 1) + 1e-4)
            best = L.min(1)
            loss = (best.values + RELAX * L.mean(1)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step(); sched.step()
            run += best.values.exp().mean().item()
            wins += torch.bincount(best.indices.cpu(), minlength=M)
            if (step + 1) % 1000 == 0:
                log["loss"].append([step + 1, run / 1000]); log["wins"].append(wins.tolist())
                print(tag, step + 1, f"{run / 1000:.4f}", wins.tolist(), f"{time.time() - t0:.0f}s", flush=True)
                run, wins = 0.0, torch.zeros(M)
        log["steps"], log["train_seconds"] = STEPS, time.time() - t0
    torch.save(lib.state_dict(), OUT / "learners" / f"{tag}.pt")
    (OUT / "learners" / f"{tag}.json").write_text(json.dumps(log))


def krr_with_score(uc, yc, uq, yq):
    """Trial 1's kernel ridge (same grid, same selection), also returning the selected LOO MSE."""
    uc, yc, uq, yq = (t.double() for t in (uc, yc, uq, yq))
    K = uc.shape[1]
    d_cc, d_qc = torch.cdist(uc, uc) ** 2, torch.cdist(uq, uc) ** 2
    best_loo = best_err = None
    for ell in [0.25, 0.5, 1.0, 2.0, 4.0]:
        Kc, Kq = torch.exp(-d_cc / (2 * ell**2)), torch.exp(-d_qc / (2 * ell**2))
        for lam in LAMBDAS.tolist():
            A = Kc + lam * K * torch.eye(K, device=uc.device, dtype=uc.dtype)
            alpha = torch.linalg.solve(A, yc)
            H = (Kc @ torch.linalg.inv(A)).diagonal(dim1=1, dim2=2)
            loo = (((yc - Kc @ alpha) / (1 - H).clamp_min(1e-9).unsqueeze(-1)) ** 2).mean((1, 2))
            err = nmse(Kq @ alpha, yq)
            if best_loo is None:
                best_loo, best_err = loo, err
            else:
                m = loo < best_loo
                best_loo, best_err = torch.where(m, loo, best_loo), torch.where(m, err, best_err)
    return best_err, best_loo


def load_models(dev):
    libs, wides = {}, {}
    for n in N_GRID:
        for s in SEEDS:
            lib = Library().to(dev)
            lib.load_state_dict(torch.load(OUT / "learners" / f"library_N{n:02d}_s{s}.pt", map_location=dev, weights_only=True))
            libs[(n, s)] = lib.eval()
            w = Shared().to(dev)
            w.load_state_dict(torch.load(TRIAL3 / "learners" / f"wide_N{n:02d}_s{s}.pt", map_location=dev, weights_only=True))
            wides[(n, s)] = w.eval()
    return libs, wides


@torch.no_grad()
def arms(lib, wide, uc, yc, uq, yq, krr):
    """Returns {arm: per-episode NMSE} and the per-episode selection of library+fb (0..M-1 basis, M = kernel ridge)."""
    errs, loos = zip(*[ridge_loo_with_score(b(uc), yc, b(uq), yq) for b in lib.bases])
    E, Lo = torch.stack(errs, 1), torch.stack(loos, 1)
    E_fb, Lo_fb = torch.cat([E, krr[0].unsqueeze(1)], 1), torch.cat([Lo, krr[1].unsqueeze(1)], 1)
    sel_fb = Lo_fb.argmin(1)
    ew, lw = ridge_loo_with_score(wide(uc, None), yc, wide(uq, None), yq)
    out = {
        "library": E.gather(1, Lo.argmin(1, keepdim=True)).squeeze(1),
        "libraryfb": E_fb.gather(1, sel_fb.unsqueeze(1)).squeeze(1),
        "oracle": E.min(1).values,
        "widefb": torch.where(lw <= krr[1], ew, krr[0]),
    }
    return out, sel_fb


def evaluate(args):
    dev = args.device
    dev_names = S.eligible_split()["development"]
    xs, dxs = load(dev_names, dev)
    libs, wides = load_models(dev)
    rows = [r for r in json.loads((TRIAL1 / "eval_development.json").read_text())["rows"]
            if not r["method"].startswith("learned")]
    selections = []
    for si, name in enumerate(dev_names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev)
                   for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            summary = {}
            for (n, s), lib in libs.items():
                res, sel = arms(lib, wides[(n, s)], uc, yc, uq, yq, krr)
                for arm, e in res.items():
                    rows.append({"system": name, "K": K, "method": f"{arm}_N{n:02d}_s{s}", "episodes": e.float().cpu().tolist()})
                selections.append({"system": name, "K": K, "N": n, "seed": s,
                                   "counts": torch.bincount(sel.cpu(), minlength=M + 1).tolist()})
                if s == 1 and n in (0, 51):
                    summary.update({f"{a}_N{n}": round(float(v.median()), 4) for a, v in res.items()})
            print(name, K, summary, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows, "selections": selections}))


def evaluate_retention(args):
    dev = args.device
    meta = S.eligible_split()["meta_train"]
    libs, wides = load_models(dev)
    rows = []
    for s in SEEDS:
        order = list(np.random.default_rng(s).permutation(meta))[:8]
        xs, dxs = load(order, dev)
        for si, name in enumerate(order):
            gen = torch.Generator().manual_seed(7000 + 97 * si + s)
            eps = [episode(xs[si][0:1], dxs[si][0:1], 32, gen, dev) for _ in range(20)]
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            for n in N_GRID:
                res, _ = arms(libs[(n, s)], wides[(n, s)], uc, yc, uq, yq, krr)
                for arm, e in res.items():
                    rows.append({"arm": arm, "seed": s, "position": si, "system": name, "N": n, "seen": si < n,
                                 "median_nmse": float(e.median())})
    (OUT / "eval_retention.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "retention"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "retention": evaluate_retention}[a.cmd](a)
