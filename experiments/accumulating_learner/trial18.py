"""Trial 18: a pairwise feature library that works for any state dimension, with a broad real history.
See docs/research-log.md for the pre-registered protocol.

  OMP_NUM_THREADS=4 nice -n 10 python trial18.py train --arm all --seed 1 --device cuda:0   # arms: catalog, all
  python trial18.py eval --device cuda:0
  python trial18.py analyze
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import BATCH, LAMBDAS, LR, SEEDS, load, nmse
from trial4 import RELAX, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES
from trial16 import OUT as TRIAL16

OUT = S.ROOT / "outputs/accumulating_learner/trial18"
REAL = OUT / "realfam_any"
F1, F2, F3, N_QUERY, M = 16, 8, 4, 128, 4
KS = (16, 32, 64)


# ---------------- data ----------------

def real_info():
    split = json.loads((REAL / "split.json").read_text())
    conv = {r["id"]: r for r in json.loads((REAL / "conversion.json").read_text())["records"]}
    return split, conv


def load_sys(name, dev, conv, real_dir=REAL):
    if name.startswith("BIOMD"):
        return torch.tensor(np.load(real_dir / f"{name}.npz")["x"], device=dev), conv[name]["h"]
    (x,), _ = load([name], dev)
    return x, T8.spacing(name)


def central(xp, xn, h):
    return (xn - xp) / (2 * h)


def standardize(xc, dc, xq, dq):
    mx, sx, sd = xc.mean(-2, keepdim=True), xc.std(-2, keepdim=True) + 1e-8, dc.std(-2, keepdim=True) + 1e-8
    return (xc - mx) / sx, dc / sd, (xq - mx) / sx, dq / sd


def episode_d(x, K, gen, dev, h):
    """Evaluation episode from one instance x (T, d); consumes gen exactly like trial9.episode_obs when d = 3."""
    T, d = x.shape
    ci = (1 + torch.randperm(T // 2 - 2, generator=gen)[:K]).to(dev)
    qi = (T // 2 + 1 + torch.randperm(T // 2 - 2, generator=gen)[:N_QUERY]).to(dev)
    noise = torch.randn(3, K, d, generator=gen).to(dev) * T8.NOISE_X * x.std(0)
    xp, xm, xn = x[ci - 1] + noise[0], x[ci] + noise[1], x[ci + 1] + noise[2]
    return standardize(xm, central(xp, xn, h), x[qi], central(x[qi - 1], x[qi + 1], h))


class BankD:
    """Training sampler for one system (instances, T, d): noisy context and noisy query (all observable)."""

    def __init__(self, x, h):
        self.x, self.h, self.std = x, h, x.std(1)

    def sample(self, B, K, gen):
        I, T, d = self.x.shape
        dev, half = self.x.device, T // 2
        inst = torch.randint(I, (B,), generator=gen).to(dev)
        ci = (1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :K]).to(dev)
        qi = (half + 1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :N_QUERY]).to(dev)
        std = T8.NOISE_X * self.std[inst].unsqueeze(1)
        nc = torch.randn(3, B, K, d, generator=gen).to(dev) * std
        nq = torch.randn(3, B, N_QUERY, d, generator=gen).to(dev) * std
        tx = self.x[inst]
        g = lambda i: torch.gather(tx, 1, i.unsqueeze(-1).expand(-1, -1, d))
        xp, xm, xn = g(ci - 1) + nc[0], g(ci) + nc[1], g(ci + 1) + nc[2]
        qp, qm, qn = g(qi - 1) + nq[0], g(qi) + nq[1], g(qi + 1) + nq[2]
        return standardize(xm, central(xp, xn, self.h), qm, central(qp, qn, self.h))


# ---------------- learner ----------------

class PairBasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, F1))
        self.p = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, F2))
        self.o = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, F3))
        self.log_lam = nn.Parameter(torch.tensor(-3.0))

    def forward(self, u):
        """u (B, N, d) -> per-variable design rows (B, N, d, k), k = F1 + (d-1)(F2+F3) + 1."""
        B, N, d = u.shape
        su, ou = self.s(u.unsqueeze(-1)), self.o(u.unsqueeze(-1))
        parts = [su]
        if d > 1:
            idx = torch.tensor([[b for b in range(d) if b != i] for i in range(d)], device=u.device)
            ui = u.unsqueeze(-1).expand(B, N, d, d - 1)
            pu = self.p(torch.stack([ui, u[..., idx]], -1))
            parts += [pu.flatten(-2), ou[..., idx, :].flatten(-2)]
        return torch.cat(parts + [torch.ones(B, N, d, 1, device=u.device, dtype=u.dtype)], -1)


class PairLibrary(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.bases = nn.ModuleList([PairBasis() for _ in range(m)])


def ridge_pw(Pc, yc, lam):
    """Pc (B,K,d,k), yc (B,K,d) -> W (B,d,k); one head per variable, penalty lam*K."""
    P = Pc.permute(0, 2, 1, 3)
    K, k = P.shape[2], P.shape[3]
    A = P.transpose(-1, -2) @ P + lam * K * torch.eye(k, device=P.device, dtype=P.dtype)
    return torch.linalg.solve(A, P.transpose(-1, -2) @ yc.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)


def predict_pw(Pq, W):
    return torch.einsum("bqdk,bdk->bqd", Pq, W)


def ridge_loo_pw(Pc, yc, Pq, yq):
    """Shared lambda chosen per episode by exact closed-form LOO (mean over variables); returns (NMSE, LOO MSE)."""
    Pc, yc, Pq, yq = (t.double() for t in (Pc, yc, Pq, yq))
    P = Pc.permute(0, 2, 1, 3)
    Y = yc.permute(0, 2, 1)
    K, k = P.shape[2], P.shape[3]
    best_loo = best_err = None
    for lam in LAMBDAS.tolist():
        A = P.transpose(-1, -2) @ P + lam * K * torch.eye(k, device=P.device, dtype=P.dtype)
        X = torch.linalg.solve(A, P.transpose(-1, -2))                    # (B,d,k,K)
        W = (X @ Y.unsqueeze(-1)).squeeze(-1)                            # (B,d,k)
        H = (P * X.transpose(-1, -2)).sum(-1)                            # (B,d,K)
        r = (Y - (P @ W.unsqueeze(-1)).squeeze(-1)) / (1 - H).clamp_min(1e-9)
        loo = (r**2).mean((1, 2))
        loo = torch.where(torch.isfinite(loo), loo, torch.full_like(loo, float("inf")))
        err = nmse(predict_pw(Pq, W), yq)
        if best_loo is None:
            best_loo, best_err = loo, err
        else:
            m = loo < best_loo
            best_loo, best_err = torch.where(m, loo, best_loo), torch.where(m, err, best_err)
    return best_err, best_loo


@torch.no_grad()
def library_fb_pw(lib, uc, yc, uq, yq, krr):
    errs, loos = zip(*[ridge_loo_pw(b(uc), yc, b(uq), yq) for b in lib.bases])
    E = torch.cat([torch.stack(errs, 1), krr[0].unsqueeze(1)], 1)
    sel = torch.cat([torch.stack(loos, 1), krr[1].unsqueeze(1)], 1).argmin(1)
    return E.gather(1, sel.unsqueeze(1)).squeeze(1), sel


def task_loss(lib, batch):
    uc, yc, uq, yq = batch
    errs = []
    for b in lib.bases:
        W = ridge_pw(b(uc), yc, b.log_lam.exp())
        errs.append(nmse(predict_pw(b(uq), W), yq).clamp_max(10.0))
    L = torch.log(torch.stack(errs, 1) + 1e-4)
    best = L.min(1)
    return (best.values + RELAX * L.mean(1)).mean(), best.values.exp().mean().item()


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    split, conv = real_info()
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    real_train = sorted(k for k, v in split.items() if v == "train")
    order = list(np.random.default_rng(1800 + s).permutation(catalog + real_train)) if args.arm == "all" else catalog
    torch.manual_seed(1000 * s)
    lib = PairLibrary().to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem, log, t0 = [], {"arm": args.arm, "seed": s, "order": order, "stages": []}, time.time()
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        x, h = load_sys(name, dev, conv)
        new = BankD(x, h)
        lib.train()
        run = 0.0
        for step in range(400):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / 400))
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            if mem:
                la, ea = task_loss(lib, new.sample(BATCH // 2, K, gen))
                mb = mem[int(torch.randint(len(mem), (1,), generator=gen))]
                lb, eb = task_loss(lib, mb.sample(BATCH // 2, K, gen))
                loss, e = 0.5 * (la + lb), 0.5 * (ea + eb)
            else:
                loss, e = task_loss(lib, new.sample(BATCH, K, gen))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step()
            run += e
        mem.append(BankD(x[:MEMORY_INSTANCES], h))
        log["stages"].append({"t": t, "system": name, "d": int(x.shape[-1]), "train_min_nmse": run / 400})
        if t % 10 == 0 or t == len(order):
            print(args.arm, s, t, name, f"d={x.shape[-1]} train {run / 400:.4f}", f"{time.time() - t0:.0f}s", flush=True)
    torch.save(lib.state_dict(), OUT / "learners" / f"pw{args.arm}_final_s{s}.pt")
    (OUT / "learners" / f"pw{args.arm}_s{s}.json").write_text(json.dumps(log))


# ---------------- evaluation ----------------

@torch.no_grad()
def score(eps, libs, tag):
    uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
    krr = krr_with_score(uc, yc, uq, yq)
    res = {"krr": krr[0], **{name: library_fb_pw(lib, uc, yc, uq, yq, krr)[0] for name, lib in libs.items()}}
    print(tag, {m: round(float(np.median(v.float().cpu().numpy())), 5) for m, v in res.items() if m.endswith("s1") or m == "krr"}, flush=True)
    return res


def evaluate(args):
    dev = args.device
    torch.set_num_threads(4)
    split, conv = real_info()
    libs = {}
    for s in SEEDS:
        for arm in ("catalog", "all"):
            lib = PairLibrary().to(dev)
            lib.load_state_dict(torch.load(OUT / "learners" / f"pw{arm}_final_s{s}.pt", map_location=dev, weights_only=True))
            libs[f"pw{arm}_s{s}"] = lib.eval()
    rows = []
    held = sorted(k for k, v in split.items() if v == "heldout")
    devnames = S.eligible_split()["development"]
    targets = [(n, "heldout") for n in held] + [(n, "development") for n in devnames]
    for i, (name, part) in enumerate(targets):
        x, h = load_sys(name, dev, conv)
        for K in KS:
            gen = torch.Generator().manual_seed(18000 + 97 * i + K)
            eps = [episode_d(x[j], K, gen, dev, h) for j in range(min(4, x.shape[0])) for _ in range(5)]
            for m, v in score(eps, libs, f"{part} {name} d={x.shape[-1]} K={K}").items():
                rows.append({"part": part, "system": name, "d": int(x.shape[-1]), "K": K, "method": m, "episodes": v.float().cpu().tolist()})
    # Trial 16's held-out systems on Trial 16's exact episodes
    s16 = json.loads((TRIAL16 / "realfam" / "split.json").read_text())
    c16 = {r["id"]: r for r in json.loads((TRIAL16 / "realfam" / "conversion.json").read_text())["records"]}
    for i, name in enumerate(sorted(k for k, v in s16.items() if v == "heldout")):
        x, h = load_sys(name, dev, c16, TRIAL16 / "realfam")
        for K in KS:
            gen = torch.Generator().manual_seed(16000 + 97 * i + K)
            eps = [episode_d(x[j], K, gen, dev, h) for j in range(4) for _ in range(5)]
            for m, v in score(eps, libs, f"trial16eps {name} K={K}").items():
                rows.append({"part": "trial16_heldout", "system": name, "d": 3, "K": K, "method": m, "episodes": v.float().cpu().tolist()})
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["part"], r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    t16 = json.loads((TRIAL16 / "eval.json").read_text())["rows"]
    sc16 = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t16 if r["part"] == "heldout"}
    split, conv = real_info()
    held = sorted(k for k, v in split.items() if v == "heldout")
    held16 = sorted({r["system"] for r in rows if r["part"] == "trial16_heldout"})
    poly = [n for n in S.eligible_split()["development"] if n not in NONPOLY]
    med = lambda part, ns, K, m: float(np.median([sc[(part, n, K, m)] for n in ns]))
    out = {}
    for s in SEEDS:
        a, c = f"pwall_s{s}", f"pwcatalog_s{s}"
        x1 = {"ratio_vs_pwcatalog": med("heldout", held, 32, a) / med("heldout", held, 32, c),
              "wins_vs_krr": sum(sc[("heldout", n, 32, a)] < sc[("heldout", n, 32, "krr")] for n in held)}
        x2 = med("trial16_heldout", held16, 32, a) / float(np.median([sc16[(n, 32, f"catalogbio_s{s}")] for n in held16]))
        x3 = med("development", poly, 32, a) / med("development", poly, 32, c)
        crit = {"X1": bool(x1["ratio_vs_pwcatalog"] <= 0.67 and x1["wins_vs_krr"] >= 12), "X2": bool(x2 <= 0.8), "X3": bool(x3 <= 1.5)}
        out[s] = {"X1": x1, "X2_ratio_vs_trial16_catalogbio": x2, "X3_ratio_poly_dev": x3, "criteria": crit,
                  "heldout_medians": {m: {K: med("heldout", held, K, m if m == "krr" else f"{m}_s{s}") for K in KS} for m in ("pwcatalog", "pwall", "krr")},
                  "heldout_per_system_K32": {f"{n} d={len(conv[n]['states'])} {conv[n]['name'][:28]}": {m: sc[("heldout", n, 32, m if m == "krr" else f"{m}_s{s}")] for m in ("pwcatalog", "pwall", "krr")} for n in held},
                  "dev_poly_K32": {m: med("development", poly, 32, m if m == "krr" else f"{m}_s{s}") for m in ("pwcatalog", "pwall", "krr")},
                  "dev_nonpoly_K32": {m: med("development", sorted(NONPOLY), 32, m if m == "krr" else f"{m}_s{s}") for m in ("pwcatalog", "pwall", "krr")}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--arm", choices=["catalog", "all"], default="all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
