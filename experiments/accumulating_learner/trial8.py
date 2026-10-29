"""Trial 8: noisy-trajectory observation contract (central-difference derivative estimates). See docs/research-log.md.

Context: K snippets (x(t-h), x(t), x(t+h)) with 0.5% state noise; the learner sees the noisy middle states and
(x(t+h) - x(t-h)) / 2h. Queries: clean segment-B states scored against the true vector field.

  python trial8.py train --seed 1 --device cuda:0
  python trial8.py eval --device cuda:0
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

import systems as S
from make_data import BURN, N_POINTS, SPAN
from trial1 import BATCH, K_GRID, LR, N_GRID, N_QUERY, SEEDS, load, nmse, poly_features, rff_eval, ridge, ridge_loo
from trial4 import M, RELAX, Library, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES, library_fb
from trial5 import OUT as TRIAL5

OUT = S.ROOT / "outputs/accumulating_learner/trial8"
NOISE_X = 0.005
ARM, STEPS_PER_STAGE = "seqtraj400", 400


def spacing(name):
    return SPAN * float(S.METADATA[name]["period"]) / (N_POINTS - 1)


def standardize(xc, dc, xq, dq):
    mx, sx, sd = xc.mean(-2, keepdim=True), xc.std(-2, keepdim=True) + 1e-8, dc.std(-2, keepdim=True) + 1e-8
    return (xc - mx) / sx, dc / sd, (xq - mx) / sx, dq / sd


def episode_traj(x, dx, K, gen, device, h):
    """x, dx: (1, T, 3) one instance. Returns standardized (u_c, y_c, u_q, y_q)."""
    T = x.shape[1]
    ci = (1 + torch.randperm(T // 2 - 2, generator=gen)[:K]).to(device)
    qi = (T // 2 + torch.randperm(T // 2, generator=gen)[:N_QUERY]).to(device)
    noise = torch.randn(3, K, 3, generator=gen).to(device) * NOISE_X * x[0].std(0)
    xp, xm, xn = x[0, ci - 1] + noise[0], x[0, ci] + noise[1], x[0, ci + 1] + noise[2]
    return standardize(xm, (xn - xp) / (2 * h), x[0, qi], dx[0, qi])


class TrajBank:
    def __init__(self, xs, dxs, hs, device):
        self.n_inst = torch.tensor([x.shape[0] for x in xs], device=device)
        imax, self.T = int(self.n_inst.max()), xs[0].shape[1]
        self.x = torch.zeros(len(xs), imax, self.T, 3, device=device)
        self.dx = torch.zeros_like(self.x)
        for j, (x, dx) in enumerate(zip(xs, dxs)):
            self.x[j, : x.shape[0]], self.dx[j, : x.shape[0]] = x, dx
        self.std = self.x.std(2)  # (S, I, 3) per-instance state std
        self.h = torch.tensor(hs, device=device, dtype=torch.float32)
        self.device = device

    def sample(self, B, K, gen):
        d, half = self.device, self.T // 2
        sys_ = torch.randint(len(self.n_inst), (B,), generator=gen).to(d)
        inst = (torch.rand(B, generator=gen).to(d) * self.n_inst[sys_]).long()
        ci = (1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :K]).to(d)
        qi = (half + torch.rand(B, half, generator=gen).argsort(1)[:, :N_QUERY]).to(d)
        noise = torch.randn(3, B, K, 3, generator=gen).to(d) * NOISE_X * self.std[sys_, inst].unsqueeze(1)
        tx, tdx = self.x[sys_, inst], self.dx[sys_, inst]
        g = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, 3))
        xp, xm, xn = g(tx, ci - 1) + noise[0], g(tx, ci) + noise[1], g(tx, ci + 1) + noise[2]
        dest = (xn - xp) / (2 * self.h[sys_].view(B, 1, 1))
        return standardize(xm, dest, g(tx, qi), g(tdx, qi))


def train(args):
    dev = args.device
    torch.manual_seed(1000 * args.seed)  # same initial library as Trials 4-6
    lib = Library().to(dev)
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    tag = lambda t: f"{ARM}_N{t:02d}_s{args.seed}"
    torch.save(lib.state_dict(), OUT / "learners" / f"{tag(0)}.pt")
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * args.seed)
    mem_x, mem_dx, mem_h, log = [], [], [], {"arm": ARM, "seed": args.seed, "order": order, "stages": []}
    t0 = time.time()
    for t, name in enumerate(order, 1):
        (x_t,), (dx_t,) = load([name], dev)
        h_t = spacing(name)
        eg = torch.Generator().manual_seed(9100 + 97 * t + args.seed)
        eps = [episode_traj(x_t[0:1], dx_t[0:1], 32, eg, dev, h_t) for _ in range(20)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        e_lib, sel = library_fb(lib.eval(), uc, yc, uq, yq, krr)
        lib.train()
        new = TrajBank([x_t], [dx_t], [h_t], dev)
        mem = TrajBank(mem_x, mem_dx, mem_h, dev) if mem_x else None
        run, wins = 0.0, torch.zeros(M)
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
                W = ridge(basis(uc), yc, basis.log_lam.exp())
                errs.append(nmse(basis(uq) @ W, yq).clamp_max(10.0))
            L = torch.log(torch.stack(errs, 1) + 1e-4)
            best = L.min(1)
            loss = (best.values + RELAX * L.mean(1)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step()
            run += best.values.exp().mean().item()
            wins += torch.bincount(best.indices.cpu(), minlength=M)
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES]); mem_h.append(h_t)
        stage = {"t": t, "system": name, "encounter_libraryfb": float(e_lib.median()), "encounter_krr": float(krr[0].median()),
                 "encounter_selection": torch.bincount(sel.cpu(), minlength=M + 1).tolist(),
                 "train_min_nmse": run / STEPS_PER_STAGE, "wins": wins.tolist(), "seconds": time.time() - t0}
        log["stages"].append(stage)
        print(ARM, args.seed, t, name, f"enc lib {stage['encounter_libraryfb']:.4f} krr {stage['encounter_krr']:.4f}",
              f"train {stage['train_min_nmse']:.4f}", f"{stage['seconds']:.0f}s", flush=True)
        if t in N_GRID:
            torch.save(lib.state_dict(), OUT / "learners" / f"{tag(t)}.pt")
    (OUT / "learners" / f"{ARM}_s{args.seed}.json").write_text(json.dumps(log))


def evaluate(args):
    dev = args.device
    names = S.eligible_split()["development"]
    xs, dxs = load(names, dev)
    libs = {}
    for t in N_GRID:
        for s in SEEDS:
            for arm, path in (("frozen5", TRIAL5 / "learners" / f"seqreplay400_N{t:02d}_s{s}.pt"),
                              (ARM, OUT / "learners" / f"{ARM}_N{t:02d}_s{s}.pt")):
                lib = Library().to(dev)
                lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
                libs[f"{arm}_N{t:02d}_s{s}"] = lib.eval()
    rows = []
    for si, name in enumerate(names):
        h = spacing(name)
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode_traj(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev, h)
                   for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            with torch.no_grad():
                krr = krr_with_score(uc, yc, uq, yq)
                res = {"krr": krr[0], "rff": rff_eval(uc, yc, uq, yq, torch.Generator().manual_seed(9000 + 97 * si + K)),
                       "poly2": ridge_loo(poly_features(uc, 2), yc, poly_features(uq, 2), yq)}
                for tag, lib in libs.items():
                    res[tag] = library_fb(lib, uc, yc, uq, yq, krr)[0]
            for m, v in res.items():
                rows.append({"system": name, "K": K, "method": m, "episodes": v.float().cpu().tolist()})
            print(name, K, {m: round(float(np.median(v.float().cpu().numpy())), 4) for m, v in res.items()
                            if m in ("krr", "poly2", "frozen5_N51_s1", f"{ARM}_N51_s1")}, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)
