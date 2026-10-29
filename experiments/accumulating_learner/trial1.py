"""Trial 1: does an experience curve exist? See docs/research-log.md for the pre-registered protocol.

Learner: shared MLP phi (3 -> 128 -> 128 -> 31, SiLU) plus a constant feature gives F = 32 features of the
context-standardized state. A new system gets a 32 x 3 ridge head fitted from its K observations (k = 96).
phi is meta-trained by differentiating query NMSE through the ridge solve across N meta-train systems.

  python trial1.py train --n 8 --seed 1 --device cuda:0
  python trial1.py eval --device cuda:0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import systems as S

OUT = S.ROOT / "outputs/accumulating_learner/trial1"
DATA = OUT / "data"
N_GRID = [0, 1, 2, 4, 8, 16, 32, 51]
K_GRID = [8, 16, 32, 64, 128]
SEEDS = [1, 2]
F = 32
NOISE = 0.01
N_QUERY = 128
LAMBDAS = torch.logspace(-6, 2, 17)
STEPS, BATCH, LR = 20000, 64, 1e-3


def load(names, device):
    xs, dxs = [], []
    for n in names:
        z = np.load(DATA / f"{n}.npz")
        xs.append(torch.tensor(z["x"], device=device)); dxs.append(torch.tensor(z["dx"], device=device))
    return xs, dxs


def episode(x, dx, K, gen, device):
    """x, dx: (I, T, 3) for one system. Returns standardized (u_c, y_c, u_q, y_q) with batch dim 1."""
    T = x.shape[1]
    i = torch.randint(x.shape[0], (1,), generator=gen).item()
    ci = torch.randperm(T // 2, generator=gen)[:K]
    qi = T // 2 + torch.randperm(T // 2, generator=gen)[:N_QUERY]
    noise = torch.randn(K, 3, generator=gen).to(device)
    xc, dc, xq, dq = x[i, ci.to(device)], dx[i, ci.to(device)], x[i, qi.to(device)], dx[i, qi.to(device)]
    dc = dc + NOISE * dc.std(0) * noise
    mx, sx, sd = xc.mean(0), xc.std(0) + 1e-8, dc.std(0) + 1e-8
    return (xc - mx) / sx, dc / sd, (xq - mx) / sx, dq / sd


def batch_episodes(xs, dxs, idx, K, gen, device):
    eps = [episode(xs[j], dxs[j], K, gen, device) for j in idx]
    return [torch.stack(t) for t in zip(*eps)]


class Phi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                                       torch.nn.SiLU(), torch.nn.Linear(128, F - 1))
        self.log_lam = torch.nn.Parameter(torch.tensor(-3.0))

    def forward(self, u):
        h = self.net(u)
        return torch.cat([h, torch.ones_like(h[..., :1])], -1)


def ridge(Pc, yc, lam):
    """Pc (B,K,F), yc (B,K,3), lam (B,) or scalar -> W (B,F,3). Penalty scaled by K."""
    B, K, Fd = Pc.shape
    A = Pc.transpose(1, 2) @ Pc + (lam * K).reshape(-1, 1, 1) * torch.eye(Fd, device=Pc.device, dtype=Pc.dtype)
    return torch.linalg.solve(A, Pc.transpose(1, 2) @ yc)


def nmse(pred, yq):
    """Per-episode NMSE averaged over the three output dims. (B,Q,3) -> (B,)"""
    return (((pred - yq) ** 2).mean(1) / (yq.var(1) + 1e-12)).mean(-1)


def ridge_loo(Pc, yc, Pq, yq):
    """Select lambda per episode by closed-form leave-one-out on the context only; return query NMSE (B,)."""
    Pc, yc, Pq, yq = (t.double() for t in (Pc, yc, Pq, yq))
    best_loo, best_err = None, None
    for lam in LAMBDAS.tolist():
        W = ridge(Pc, yc, torch.tensor(lam, device=Pc.device, dtype=Pc.dtype))
        K, Fd = Pc.shape[1:]
        A = Pc.transpose(1, 2) @ Pc + lam * K * torch.eye(Fd, device=Pc.device, dtype=Pc.dtype)
        H = (Pc @ torch.linalg.solve(A, Pc.transpose(1, 2))).diagonal(dim1=1, dim2=2)
        loo = (((yc - Pc @ W) / (1 - H).clamp_min(1e-9).unsqueeze(-1)) ** 2).mean((1, 2))
        err = nmse(Pq @ W, yq)
        if best_loo is None:
            best_loo, best_err = loo, err
        else:
            better = loo < best_loo
            best_loo, best_err = torch.where(better, loo, best_loo), torch.where(better, err, best_err)
    return best_err


def train(args):
    torch.manual_seed(1000 * args.seed + args.n)
    dev = args.device
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    names = order[: args.n]
    phi = Phi().to(dev)
    tag = f"N{args.n:02d}_s{args.seed}"
    ckpt = OUT / "learners" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log = {"n": args.n, "seed": args.seed, "systems": names, "steps": 0, "loss": []}
    if args.n > 0:
        xs, dxs = load(names, dev)
        opt = torch.optim.Adam(phi.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        gen = torch.Generator().manual_seed(77 + 1000 * args.seed + args.n)
        t0, run = time.time(), 0.0
        for step in range(STEPS):
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            idx = torch.randint(len(names), (BATCH,), generator=gen).tolist()
            uc, yc, uq, yq = batch_episodes(xs, dxs, idx, K, gen, dev)
            W = ridge(phi(uc), yc, phi.log_lam.exp())
            loss = nmse(phi(uq) @ W, yq).clamp_max(10.0).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
            if (step + 1) % 1000 == 0:
                log["loss"].append([step + 1, run / 1000]); run = 0.0
                print(tag, step + 1, f"{log['loss'][-1][1]:.4f}", f"{time.time() - t0:.0f}s", flush=True)
        log["steps"], log["train_seconds"] = STEPS, time.time() - t0
    log["log_lam"] = phi.log_lam.item()
    torch.save(phi.state_dict(), ckpt)
    (OUT / "learners" / f"{tag}.json").write_text(json.dumps(log))


# ---------------- evaluation ----------------

def poly_features(u, deg):
    cols = [torch.ones_like(u[..., :1])]
    for d in range(1, deg + 1):
        for a in range(d + 1):
            for b in range(d + 1 - a):
                c = d - a - b
                cols.append((u[..., 0:1] ** a) * (u[..., 1:2] ** b) * (u[..., 2:3] ** c))
    return torch.cat(cols, -1)


def rff_eval(uc, yc, uq, yq, gen):
    best = None
    for ell in [0.25, 0.5, 1.0, 2.0, 4.0]:
        Wf = torch.randn(3, F - 1, generator=gen).to(uc) / ell
        bf = (torch.rand(F - 1, generator=gen) * 2 * math.pi).to(uc)
        feat = lambda u: torch.cat([torch.cos(u @ Wf + bf) * math.sqrt(2), torch.ones_like(u[..., :1])], -1)
        err, loo = ridge_loo_with_score(feat(uc), yc, feat(uq), yq)
        best = (err, loo) if best is None else (torch.where(loo < best[1], err, best[0]), torch.minimum(loo, best[1]))
    return best[0]


def ridge_loo_with_score(Pc, yc, Pq, yq):
    Pc, yc, Pq, yq = (t.double() for t in (Pc, yc, Pq, yq))
    K, Fd = Pc.shape[1:]
    best_loo = best_err = None
    for lam in LAMBDAS.tolist():
        A = Pc.transpose(1, 2) @ Pc + lam * K * torch.eye(Fd, device=Pc.device, dtype=Pc.dtype)
        W = torch.linalg.solve(A, Pc.transpose(1, 2) @ yc)
        H = (Pc @ torch.linalg.solve(A, Pc.transpose(1, 2))).diagonal(dim1=1, dim2=2)
        loo = (((yc - Pc @ W) / (1 - H).clamp_min(1e-9).unsqueeze(-1)) ** 2).mean((1, 2))
        err = nmse(Pq @ W, yq)
        if best_loo is None:
            best_loo, best_err = loo, err
        else:
            m = loo < best_loo
            best_loo, best_err = torch.where(m, loo, best_loo), torch.where(m, err, best_err)
    return best_err, best_loo


def krr_eval(uc, yc, uq, yq):
    """Gaussian-kernel ridge with centres at the context states; length scale and lambda by LOO."""
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
    return best_err


def scratch_mlp_eval(uc, yc, uq, yq, seed, steps=500):
    """Per-episode training of the phi architecture plus linear head from scratch (k = P + 96). Batched."""
    g = torch.Generator().manual_seed(seed)
    B, dev = uc.shape[0], uc.device
    def init(i, o):
        return [torch.nn.Parameter((torch.rand(B, i, o, generator=g) * 2 - 1).to(dev) / math.sqrt(i)),
                torch.nn.Parameter(torch.zeros(B, 1, o, device=dev))]
    ps = init(3, 128) + init(128, 128) + init(128, F - 1) + init(F - 1, 3)
    def f(u):
        h = torch.nn.functional.silu(u @ ps[0] + ps[1])
        h = torch.nn.functional.silu(h @ ps[2] + ps[3])
        h = h @ ps[4] + ps[5]
        return h @ ps[6] + ps[7]
    opt = torch.optim.AdamW(ps, lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        loss = ((f(uc) - yc) ** 2).mean((1, 2)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return nmse(f(uq), yq)


def evaluate(args):
    dev = args.device
    split = S.eligible_split()
    dev_names = split["development"]
    xs, dxs = load(dev_names, dev)
    learners = {}
    for n in N_GRID:
        for s in SEEDS:
            p = OUT / "learners" / f"N{n:02d}_s{s}.pt"
            if p.exists():
                phi = Phi().to(dev); phi.load_state_dict(torch.load(p, map_location=dev)); phi.eval()
                learners[f"learned_N{n:02d}_s{s}"] = phi
    rows = []
    timing = {}
    for si, name in enumerate(dev_names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = []
            for inst in range(xs[si].shape[0]):
                for _ in range(5):
                    eps.append(episode(xs[si][inst:inst + 1], dxs[si][inst:inst + 1], K, gen, dev))
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            res = {}
            with torch.no_grad():
                for tag, phi in learners.items():
                    t0 = time.time()
                    res[tag] = ridge_loo(phi(uc), yc, phi(uq), yq)
                    timing.setdefault("learned_loo", []).append((time.time() - t0) / len(eps))
                    W = ridge(phi(uc).double(), yc.double(), torch.tensor(math.exp(phi.log_lam.item()), device=dev, dtype=torch.float64))
                    res[tag.replace("learned", "learnedlam")] = nmse(phi(uq).double() @ W, yq.double())
                res["rff"] = rff_eval(uc, yc, uq, yq, torch.Generator().manual_seed(9000 + 97 * si + K))
                t0 = time.time()
                res["krr"] = krr_eval(uc, yc, uq, yq)
                timing.setdefault("krr", []).append((time.time() - t0) / len(eps))
                res["poly2"] = ridge_loo(poly_features(uc, 2), yc, poly_features(uq, 2), yq)
                res["poly3"] = ridge_loo(poly_features(uc, 3), yc, poly_features(uq, 3), yq)
            t0 = time.time()
            res["scratch_mlp"] = scratch_mlp_eval(uc, yc, uq, yq, seed=31 * si + K)
            timing.setdefault("scratch_mlp", []).append((time.time() - t0) / len(eps))
            for m, v in res.items():
                rows.append({"system": name, "K": K, "method": m, "episodes": v.float().cpu().tolist()})
            print(name, K, {m: round(float(v.median()), 4) for m, v in res.items() if "_s1" in m or "_" not in m or m == "scratch_mlp"}, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows, "timing_s_per_episode": {k: float(np.mean(v)) for k, v in timing.items()}}))


def evaluate_retention(args):
    """In-distribution check on each seed's first 8 meta-train systems (published-parameter instance)."""
    dev = args.device
    meta = S.eligible_split()["meta_train"]
    rows = []
    for s in SEEDS:
        order = list(np.random.default_rng(s).permutation(meta))[:8]
        xs, dxs = load(order, dev)
        for si, name in enumerate(order):
            gen = torch.Generator().manual_seed(7000 + 97 * si + s)
            eps = [episode(xs[si][0:1], dxs[si][0:1], 32, gen, dev) for _ in range(20)]
            uc, yc, uq, yq = (torch.stack(t) for t in zip(*eps))
            for n in N_GRID:
                phi = Phi().to(dev); phi.load_state_dict(torch.load(OUT / "learners" / f"N{n:02d}_s{s}.pt", map_location=dev))
                with torch.no_grad():
                    e = ridge_loo(phi(uc), yc, phi(uq), yq)
                rows.append({"seed": s, "position": si, "system": name, "N": n, "seen": si < n, "median_nmse": float(e.median())})
    (OUT / "eval_retention.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "retention"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "retention": evaluate_retention}[a.cmd](a)
