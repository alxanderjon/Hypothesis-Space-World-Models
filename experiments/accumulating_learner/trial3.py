"""Trial 3: context-conditioned features versus a capacity-matched shared basis. See docs/research-log.md.

Arms:
  context: a set encoder reads the K standardized (state, derivative) pairs, mean-pools them to c (64-d), and
           FiLM-modulates both hidden layers of the feature network phi(u; c). The ridge head (k = 96) is then
           fitted on the context exactly as in Trial 1. FiLM is zero-initialized, so at N = 0 it is a plain network.
  wide:    Trial 1's shared phi at width 256 (no context input), matching the context arm's parameter count.

Evaluation episodes, controls and analysis are identical to Trial 1 (same seeded `episode` function), so the
control rows are copied from Trial 1's evaluation file.

  python trial3.py train --arch context --n 8 --seed 1 --device cuda:0 --loss lognmse
  python trial3.py eval --device cuda:0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import systems as S
from trial1 import (BATCH, F, K_GRID, LR, N_GRID, N_QUERY, NOISE, SEEDS, STEPS, episode, load, nmse, ridge,
                    ridge_loo)

OUT = S.ROOT / "outputs/accumulating_learner/trial3"
TRIAL1 = S.ROOT / "outputs/accumulating_learner/trial1"
ARCHS = ["context", "wide"]


class Shared(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, F - 1))
        self.log_lam = nn.Parameter(torch.tensor(-3.0))

    def context(self, uc, yc):
        return None

    def forward(self, u, c):
        h = self.net(u)
        return torch.cat([h, torch.ones_like(h[..., :1])], -1)


class Context(nn.Module):
    def __init__(self, width=128, cdim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, cdim))
        self.l1, self.l2, self.l3 = nn.Linear(3, width), nn.Linear(width, width), nn.Linear(width, F - 1)
        self.film = nn.Linear(cdim, 4 * width)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.log_lam = nn.Parameter(torch.tensor(-3.0))

    def context(self, uc, yc):
        return self.enc(torch.cat([uc, yc], -1)).mean(1)

    def forward(self, u, c):
        g1, b1, g2, b2 = self.film(c).unsqueeze(1).chunk(4, -1)
        h = nn.functional.silu(self.l1(u) * (1 + g1) + b1)
        h = nn.functional.silu(self.l2(h) * (1 + g2) + b2)
        h = self.l3(h)
        return torch.cat([h, torch.ones_like(h[..., :1])], -1)


def make(arch):
    return Context() if arch == "context" else Shared()


class Bank:
    """All of one learner's meta-train data padded into one tensor, for vectorized episode sampling."""

    def __init__(self, names, device):
        xs, dxs = load(names, device)
        self.n_inst = torch.tensor([x.shape[0] for x in xs], device=device)
        imax, T = int(self.n_inst.max()), xs[0].shape[1]
        self.x = torch.zeros(len(xs), imax, T, 3, device=device)
        self.dx = torch.zeros_like(self.x)
        for j, (x, dx) in enumerate(zip(xs, dxs)):
            self.x[j, : x.shape[0]], self.dx[j, : x.shape[0]] = x, dx
        self.T, self.device = T, device

    def sample(self, B, K, gen):
        d, h = self.device, self.T // 2
        sys_ = torch.randint(len(self.n_inst), (B,), generator=gen).to(d)
        inst = (torch.rand(B, generator=gen).to(d) * self.n_inst[sys_]).long()
        ci = torch.rand(B, h, generator=gen).argsort(1)[:, :K].to(d)
        qi = h + torch.rand(B, h, generator=gen).argsort(1)[:, :N_QUERY].to(d)
        noise = torch.randn(B, K, 3, generator=gen).to(d)
        traj_x, traj_dx = self.x[sys_, inst], self.dx[sys_, inst]
        g = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, 3))
        xc, dc, xq, dq = g(traj_x, ci), g(traj_dx, ci), g(traj_x, qi), g(traj_dx, qi)
        dc = dc + NOISE * dc.std(1, keepdim=True) * noise
        mx, sx, sd = xc.mean(1, keepdim=True), xc.std(1, keepdim=True) + 1e-8, dc.std(1, keepdim=True) + 1e-8
        return (xc - mx) / sx, dc / sd, (xq - mx) / sx, dq / sd


def train(args):
    torch.manual_seed(1000 * args.seed + args.n)
    dev = args.device
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    names = order[: args.n]
    phi = make(args.arch).to(dev)
    tag = f"{args.arch}_N{args.n:02d}_s{args.seed}"
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    log = {"arch": args.arch, "n": args.n, "seed": args.seed, "systems": names, "loss_kind": args.loss,
           "params": sum(p.numel() for p in phi.parameters()), "steps": 0, "loss": []}
    if args.n > 0:
        bank = Bank(names, dev)
        opt = torch.optim.Adam(phi.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        gen = torch.Generator().manual_seed(77 + 1000 * args.seed + args.n)
        t0, run = time.time(), 0.0
        for step in range(STEPS):
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            uc, yc, uq, yq = bank.sample(BATCH, K, gen)
            c = phi.context(uc, yc)
            W = ridge(phi(uc, c), yc, phi.log_lam.exp())
            err = nmse(phi(uq, c) @ W, yq).clamp_max(10.0)
            loss = torch.log(err + 1e-4).mean() if args.loss == "lognmse" else err.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 1.0)
            opt.step(); sched.step()
            run += err.mean().item()
            if (step + 1) % 1000 == 0:
                log["loss"].append([step + 1, run / 1000]); run = 0.0
                print(tag, step + 1, f"{log['loss'][-1][1]:.4f}", f"{time.time() - t0:.0f}s", flush=True)
        log["steps"], log["train_seconds"] = STEPS, time.time() - t0
    log["log_lam"] = phi.log_lam.item()
    torch.save(phi.state_dict(), OUT / "learners" / f"{tag}.pt")
    (OUT / "learners" / f"{tag}.json").write_text(json.dumps(log))


def load_learners(dev):
    out = {}
    for arch in ARCHS:
        for n in N_GRID:
            for s in SEEDS:
                p = OUT / "learners" / f"{arch}_N{n:02d}_s{s}.pt"
                if p.exists():
                    phi = make(arch).to(dev)
                    phi.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); phi.eval()
                    out[(arch, n, s)] = phi
    return out


def evaluate(args):
    """Learned arms on the Trial 1 development episodes; controls copied from Trial 1's evaluation."""
    dev = args.device
    dev_names = S.eligible_split()["development"]
    xs, dxs = load(dev_names, dev)
    learners = load_learners(dev)
    rows = [r for r in json.loads((TRIAL1 / "eval_development.json").read_text())["rows"]
            if not r["method"].startswith("learned")]
    for si, name in enumerate(dev_names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev)
                   for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            # Wrong-context diagnostic: the encoder reads the next development system's context (same K).
            sj = (si + 1) % len(dev_names)
            gw = torch.Generator().manual_seed(5000 + 97 * sj + K)
            wrong = [episode(xs[sj][i % xs[sj].shape[0]:i % xs[sj].shape[0] + 1], dxs[sj][i % xs[sj].shape[0]:i % xs[sj].shape[0] + 1], K, gw, dev)
                     for i in range(len(eps))]
            ucw, ycw = (torch.stack(t) for t in list(zip(*wrong))[:2])
            summary = {}
            with torch.no_grad():
                for (arch, n, s), phi in learners.items():
                    c = phi.context(uc, yc)
                    e = ridge_loo(phi(uc, c), yc, phi(uq, c), yq)
                    rows.append({"system": name, "K": K, "method": f"{arch}_N{n:02d}_s{s}", "episodes": e.float().cpu().tolist()})
                    if arch == "context":
                        cw = phi.context(ucw, ycw)
                        ew = ridge_loo(phi(uc, cw), yc, phi(uq, cw), yq)
                        rows.append({"system": name, "K": K, "method": f"contextwrong_N{n:02d}_s{s}", "episodes": ew.float().cpu().tolist()})
                    if s == 1 and n in (0, 51):
                        summary[f"{arch}_N{n}"] = round(float(e.median()), 4)
            print(name, K, summary, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows}))


def evaluate_retention(args):
    dev = args.device
    meta = S.eligible_split()["meta_train"]
    learners = load_learners(dev)
    rows = []
    for s in SEEDS:
        order = list(np.random.default_rng(s).permutation(meta))[:8]
        xs, dxs = load(order, dev)
        for si, name in enumerate(order):
            gen = torch.Generator().manual_seed(7000 + 97 * si + s)
            eps = [episode(xs[si][0:1], dxs[si][0:1], 32, gen, dev) for _ in range(20)]
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            for arch in ARCHS:
                for n in N_GRID:
                    phi = learners[(arch, n, s)]
                    with torch.no_grad():
                        c = phi.context(uc, yc)
                        e = ridge_loo(phi(uc, c), yc, phi(uq, c), yq)
                    rows.append({"arch": arch, "seed": s, "position": si, "system": name, "N": n, "seen": si < n,
                                 "median_nmse": float(e.median())})
    (OUT / "eval_retention.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "retention"])
    ap.add_argument("--arch", choices=ARCHS, default="context")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--loss", choices=["nmse", "lognmse"], default="lognmse")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "retention": evaluate_retention}[a.cmd](a)
