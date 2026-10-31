"""Trial 24: amortized inference (prior-fitted network). A Transformer reads K noisy observations of a new system and
predicts its central-difference velocity at query states. Training systems are simulated on the fly on the GPU from
generators of several *kinds* of dynamics (no datasets on disk), plus stored catalog systems of the allowed kinds.
Evaluation is on real systems of kinds never seen in training. See docs/research-log.md.

  python pfn.py probe --device cuda:0
  OMP_NUM_THREADS=4 nice -n 10 python pfn.py train --variety 4 --npool 65536 --seed 1 --device cuda:0
  python pfn.py eval --device cuda:0
  python pfn.py analyze
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
import trial8 as T8
from trial1 import load, nmse
from trial4 import krr_with_score
from trial1 import rff_eval
from trial9 import ObsBank, episode_obs

OUT = S.ROOT / "outputs/accumulating_learner/trial24pfn"
KINDS_FILE = S.ROOT / "outputs/accumulating_learner/trial24pfn/kinds.json"  # fixed held-out-by-kind split (Trial 24)
KINDS = json.loads(KINDS_FILE.read_text()) if KINDS_FILE.exists() else {}
VARIETIES = {1: ["polynomial"], 2: ["polynomial", "trig"], 3: ["polynomial", "trig", "piecewise"],
             4: ["polynomial", "trig", "piecewise", "tanh", "mixed"]}
REAL_ALLOWED = {1: {"polynomial"}, 2: {"polynomial", "trig"}, 3: {"polynomial", "trig", "piecewise"},
                4: {"polynomial", "trig", "piecewise", "tanh"}}
HELD_OUT_KINDS = {"rational", "exponential"}
SEG, SUB = 600, 4            # samples per segment (context A / query B); RK4 substeps per sample
N_QUERY, K_MAX = 128, 64


# ---------------- generators (batched over systems; 3-D) ----------------

def _mask(shape, p, g, dev):
    return (torch.rand(shape, generator=g) < p).float().to(dev)


def generate(kind, B, g, dev):
    """Returns f(x) for a batch of B random 3-D vector fields of the given kind (x: (B, 3))."""
    r = lambda *s: torch.rand(*s, generator=g).to(dev)
    n = lambda *s: torch.randn(*s, generator=g).to(dev)
    gamma = 0.1 + 0.9 * r(B, 3)
    if kind == "polynomial":
        A = n(B, 3, 3) * _mask((B, 3, 3), 0.5, g, dev)
        Q = n(B, 3, 6) * _mask((B, 3, 6), 0.35, g, dev) * (0.5 + 1.5 * r(B, 1, 1))
        c = 0.005 + 0.045 * r(B, 1)

        def f(x):
            m = torch.stack([x[:, 0] ** 2, x[:, 1] ** 2, x[:, 2] ** 2, x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2]], 1)
            return (A @ x.unsqueeze(-1)).squeeze(-1) - gamma * x + (Q @ m.unsqueeze(-1)).squeeze(-1) - c * (x ** 2).sum(1, keepdim=True) * x
        return f
    m = 6
    U = n(B, m, 3) / math.sqrt(3) * (0.5 + 1.5 * r(B, 1, 1))
    b = (2 * r(B, m) - 1) * math.pi
    W = n(B, 3, m) * (1 + 3 * r(B, 1, 1)) * _mask((B, 1, m), 0.75, g, dev)
    if kind == "trig":
        sig = [torch.sin] * m
    elif kind == "piecewise":
        sig = [lambda z: z.clamp(-1, 1), lambda z: (1 - z.abs()).clamp(min=0)] * (m // 2)
    elif kind == "tanh":
        sig = [torch.tanh] * m
    elif kind == "mixed":
        pool = [torch.sin, torch.cos, torch.tanh, lambda z: z.clamp(-1, 1)]
        sig = [pool[int(i)] for i in torch.randint(4, (m,), generator=g)]
    else:
        raise ValueError(kind)

    def f(x):
        z = (U @ x.unsqueeze(-1)).squeeze(-1) + b
        act = torch.stack([sig[k](z[:, k]) for k in range(m)], 1)
        return -gamma * x + (W @ act.unsqueeze(-1)).squeeze(-1)
    return f


def rk4(f, x, dt, steps):
    for _ in range(steps):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def simulate_batch(kind, B, g, dev):
    """Generate B candidates, keep those with sustained bounded motion; returns (x (n, 2*SEG, 3), h (n,))."""
    f = generate(kind, B, g, dev)
    x = torch.randn(B, 3, generator=g).to(dev)
    dt0 = 0.02
    traj = []
    for _ in range(100):                         # pilot: 100 time units, 50 snapshots
        x = rk4(f, x, dt0, 50)
        traj.append(x)
    traj = torch.stack(traj, 1)                  # (B, 100, 3)
    ok = torch.isfinite(traj).all(dim=(1, 2)) & (traj.abs().amax(dim=(1, 2)) < 1e4)
    tail = traj[:, 50:]
    v = torch.stack([f(tail[:, i]) for i in range(0, 50, 5)], 1).norm(dim=-1).mean(1)
    rad = (tail - tail.mean(1, keepdim=True)).norm(dim=-1).mean(1)
    ok &= (rad > 1e-3 * (tail.abs().mean(dim=(1, 2)) + 1e-3)) & (v > 1e-6)
    P = 2 * math.pi * rad / v.clamp(min=1e-9)
    ok &= torch.isfinite(P) & (P > 0.05) & (P < 50)
    if ok.sum() == 0:
        return None, None
    idx = ok.nonzero().squeeze(1)
    h = (P[idx] / 40)
    xs = tail[idx, -1]
    fsub = lambda y, fi=f, ii=idx: f_sel(fi, ii, B)(y)
    out = []
    dt = (h / SUB).unsqueeze(1)
    for _ in range(2 * SEG):
        xs = rk4(fsub, xs, dt, SUB)
        out.append(xs)
    X = torch.stack(out, 1)                      # (n, 2*SEG, 3)
    good = torch.isfinite(X).all(dim=(1, 2)) & (X.abs().amax(dim=(1, 2)) < 1e4)
    std = X[:, SEG:].std(1)
    good &= (std > 1e-3 * (X.abs().mean(1) + 1e-3)).all(1)
    vA = (X[:, 1:SEG] - X[:, :SEG - 1]).norm(dim=-1).mean(1)
    vB = (X[:, SEG + 1:] - X[:, SEG:-1]).norm(dim=-1).mean(1)
    good &= vB >= 0.1 * vA
    return X[good], h[good]


def f_sel(f, idx, B):
    """Evaluate a batched field on a subset idx of its systems (pads to the full batch)."""
    def g(y):
        full = torch.zeros(B, 3, device=y.device, dtype=y.dtype)
        full[idx] = y
        return f(full)[idx]
    return g


def build_pool(variety, npool, seed, dev, log):
    g = torch.Generator().manual_seed(77000 + 1000 * seed + 10 * variety + int(math.log2(npool)))
    kinds = VARIETIES[variety]
    per = [npool // len(kinds) + (1 if i < npool % len(kinds) else 0) for i in range(len(kinds))]
    xs, hs, counts = [], [], {}
    for kind, target in zip(kinds, per):
        got, tried = 0, 0
        while got < target and tried < 80 * target + 16384:
            X, h = simulate_batch(kind, 16384, g, dev)
            tried += 16384
            if X is not None:
                take = min(target - got, len(X))
                xs.append(X[:take]); hs.append(h[:take]); got += take
        counts[kind] = {"accepted": got, "tried": tried}
        print(f"pool variety {variety} kind {kind}: {got}/{target} accepted after {tried} candidates", flush=True)
    log["pool_counts"] = counts
    return torch.cat(xs), torch.cat(hs)


def real_training_systems(variety):
    allowed = REAL_ALLOWED[variety]
    return sorted(n for n, v in KINDS.items() if v["source"] == "dysts:meta_train" and set(v["kinds"]) <= allowed)


def heldout_test_systems():
    return sorted(n for n, v in KINDS.items() if HELD_OUT_KINDS & set(v["kinds"]))


# ---------------- sampling ----------------

def sample_pool(X, H, B, K, g, dev):
    """Observable training episodes from generated trajectories: context snippets in segment A, noisy query snippets in B."""
    n = X.shape[0]
    s = torch.randint(n, (B,), generator=g).to(dev)
    ci = (1 + torch.rand(B, SEG - 2, generator=g).argsort(1)[:, :K]).to(dev)
    qi = (SEG + 1 + torch.rand(B, SEG - 2, generator=g).argsort(1)[:, :N_QUERY]).to(dev)
    tx = X[s]
    std = T8.NOISE_X * tx.std(1, keepdim=True)
    nc = torch.randn(3, B, K, 3, generator=g).to(dev) * std
    nq = torch.randn(3, B, N_QUERY, 3, generator=g).to(dev) * std
    gt = lambda i: torch.gather(tx, 1, i.unsqueeze(-1).expand(-1, -1, 3))
    hh = H[s].view(B, 1, 1)
    xp, xm, xn = gt(ci - 1) + nc[0], gt(ci) + nc[1], gt(ci + 1) + nc[2]
    qp, qm, qn = gt(qi - 1) + nq[0], gt(qi) + nq[1], gt(qi + 1) + nq[2]
    return T8.standardize(xm, (xn - xp) / (2 * hh), qm, (qn - qp) / (2 * hh))


# ---------------- model ----------------

class PFN(nn.Module):
    def __init__(self, d=192, heads=6, layers=6, ff=512):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.qry = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, uc, yc, uq):
        K, Q = uc.shape[1], uq.shape[1]
        tok = torch.cat([self.ctx(torch.cat([uc, yc], -1)), self.qry(uq)], 1)
        mask = torch.full((K + Q, K + Q), float("-inf"), device=uc.device)
        mask[:, :K] = 0.0
        mask[K:, K:] = torch.where(torch.eye(Q, device=uc.device, dtype=torch.bool), 0.0, float("-inf"))
        return self.out(self.enc(tok, mask=mask)[:, K:])


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    tag = f"V{args.variety}_N{args.npool}_s{s}"
    log = {"variety": args.variety, "npool": args.npool, "seed": s, "kinds": VARIETIES[args.variety]}
    t0 = time.time()
    X, H = build_pool(args.variety, args.npool, s, dev, log)
    log["pool_seconds"] = time.time() - t0
    reals = real_training_systems(args.variety)
    log["real_systems"] = reals
    xs, dxs = load(reals, dev)
    real_bank = ObsBank(xs, dxs, [T8.spacing(n) for n in reals], dev)
    from trial9 import ObsBank as _OB  # noqa: F401  (ObsBank already yields observable noisy queries)
    real_share = len(reals) / (len(reals) + len(X))
    torch.manual_seed(1000 * s + args.variety)
    arch = {"d": args.d, "heads": args.heads, "layers": args.layers, "ff": args.ff}
    log["arch"], log["lr"], log["warm"] = arch, args.lr, args.warm
    model = PFN(**arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps, warm = args.steps, args.warm
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    g = torch.Generator().manual_seed(88000 + 1000 * s + args.variety)
    run, hist = 0.0, []
    for step in range(steps):
        K = int(round(math.exp(torch.empty(1).uniform_(math.log(4), math.log(K_MAX), generator=g).item())))
        nreal = int((torch.rand(64, generator=g) < real_share).sum())
        parts = []
        if 64 - nreal:
            parts.append(sample_pool(X, H, 64 - nreal, K, g, dev))
        if nreal:
            parts.append(real_bank.sample(nreal, K, g))
        uc, yc, uq, yq = (torch.cat(p) for p in zip(*parts))
        err = nmse(model(uc, yc, uq), yq).clamp_max(10.0)
        loss = torch.log(err + 1e-4).mean()
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += float(err.mean())
        if (step + 1) % 1000 == 0:
            hist.append([step + 1, run / 1000]); print(tag, step + 1, f"{run / 1000:.4f}", f"{time.time() - t0:.0f}s", flush=True); run = 0.0
    log.update(train_curve=hist, seconds=time.time() - t0, params=sum(p.numel() for p in model.parameters()))
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "learners" / f"pfn_{tag}.pt")
    (OUT / "learners" / f"pfn_{tag}.json").write_text(json.dumps(log))


# ---------------- evaluation ----------------

def load_real(name, dev):
    src = KINDS[name]["source"]
    if src.startswith("biomodels3d"):
        conv = {r["id"]: r for r in json.loads((S.ROOT / "outputs/accumulating_learner/trial16/realfam/conversion.json").read_text())["records"]}
        z = np.load(S.ROOT / "outputs/accumulating_learner/trial16/realfam" / f"{name}.npz")
        return torch.tensor(z["x"], device=dev), torch.tensor(z["dx"], device=dev), conv[name]["h"]
    if "confirmation" in src:
        z = np.load(S.ROOT / "outputs/accumulating_learner/confirmation/data" / f"{name}.npz")
        return torch.tensor(z["x"], device=dev), torch.tensor(z["dx"], device=dev), T8.spacing(name)
    (x,), (dx,) = load([name], dev)
    return x, dx, T8.spacing(name)


def evaluate(args):
    dev = args.device
    torch.set_num_threads(4)
    models = {}
    for p in sorted((OUT / "learners").glob("pfn_*.pt")):
        arch = json.loads(p.with_suffix(".json").read_text()).get("arch", {})
        m = PFN(**arch).to(dev); m.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); models[p.stem] = m.eval()
    tests = [(n, "unfamiliar") for n in heldout_test_systems()]
    tests += [(n, "familiar_dev_poly") for n in S.eligible_split()["development"] if KINDS[n]["kinds"] == ["polynomial"]]
    rows = []
    for i, (name, group) in enumerate(tests):
        x, dx, h = load_real(name, dev)
        for K in (8, 16, 32, 64):
            gen = torch.Generator().manual_seed(args.episode_base + 97 * i + K)
            eps = [episode_obs(x[j:j + 1], dx[j:j + 1], K, gen, dev, h) for j in range(min(4, x.shape[0])) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            with torch.no_grad():
                res = {"krr": krr_with_score(uc, yc, uq, yq)[0], "rff": rff_eval(uc, yc, uq, yq, torch.Generator().manual_seed(9000 + 97 * i + K))}
                for tag, m in models.items():
                    res[tag] = nmse(m(uc, yc, uq).double(), yq.double())
            for meth, v in res.items():
                rows.append({"system": name, "group": group, "K": K, "method": meth, "episodes": v.float().cpu().tolist()})
            print(group, name, K, {k: round(float(np.median(v.float().cpu().numpy())), 4) for k, v in res.items() if k in ("krr", "rff") or "V4_N65536" in k}, flush=True)
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    unf = sorted({r["system"] for r in rows if r["group"] == "unfamiliar"})
    fam = sorted({r["system"] for r in rows if r["group"] == "familiar_dev_poly"})
    med = lambda ns, K, m: float(np.median([sc[(n, K, m)] for n in ns]))
    out = {"n_unfamiliar": len(unf), "primary": args.primary}
    seeds = [int(x) for x in args.seeds.split(",")]
    numbers = [int(x) for x in args.numbers.split(",")]
    levels = [int(x) for x in args.varieties.split(",")]
    for s in seeds:
        P = lambda v, n: f"pfn_V{v}_N{n}_s{s}"
        ctrl = {K: min(("krr", "rff"), key=lambda c: med(unf, K, c)) for K in (8, 16, 32, 64)}
        adv = lambda m, K: math.log(med(unf, K, ctrl[K]) / med(unf, K, m))
        main = P(4, numbers[-1])
        s1 = {K: med(unf, K, main) / med(unf, K, ctrl[K]) for K in (16, 32)}
        wins = sum(sc[(n, 32, main)] < sc[(n, 32, "krr")] for n in unf)
        var_adv = [adv(P(v, args.vary_n), 32) for v in levels]
        num_adv = [adv(P(4, n), 32) for n in numbers]
        from scipy.stats import spearmanr
        rho = spearmanr(levels, var_adv).correlation if len(levels) > 2 else float("nan")
        crit = {"S1": bool(all(v <= 0.8 for v in s1.values()) and wins >= math.ceil(0.6 * len(unf))),
                "S2_variety": bool((len(levels) < 3 or rho >= 0.8) and var_adv[-1] - var_adv[0] >= math.log(1.25)),
                "S2_number": bool(num_adv[-1] - num_adv[0] >= math.log(1.25) and all(b >= a for a, b in zip(num_adv, num_adv[1:])))}
        out[s] = {"best_control": ctrl, "S1_ratio_vs_best_control": s1, "S1_wins_vs_krr_K32": wins,
                  "variety_advantage_K32": dict(zip([f"V{v}" for v in levels], var_adv)), "variety_spearman": rho,
                  "number_advantage_K32": dict(zip([f"N{n}" for n in numbers], num_adv)), "criteria": crit,
                  "pass": bool(all(crit.values())) if args.primary == "prereg" else bool(crit["S1"] and crit["S2_number"]),
                  "unfamiliar_medians": {m: {K: med(unf, K, m) for K in (8, 16, 32, 64)} for m in ("krr", "rff", main, P(levels[-1], args.vary_n), P(levels[0], args.vary_n))},
                  "familiar_dev_poly_K32": {m: med(fam, 32, m) for m in ("krr", "rff", main)}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


def probe(args):
    """Acceptance rates and timing of each generator (no training)."""
    dev = args.device
    g = torch.Generator().manual_seed(1)
    for kind in VARIETIES[4]:
        t0 = time.time()
        X, h = simulate_batch(kind, 16384, g, dev)
        n = 0 if X is None else len(X)
        print(f"{kind:10s} accepted {n}/16384 ({100 * n / 16384:.1f}%) in {time.time() - t0:.1f}s; h median {float(h.median()) if n else float('nan'):.3f}", flush=True)
    print("real training systems by variety:", {v: len(real_training_systems(v)) for v in VARIETIES})
    print("held-out-kind test systems:", len(heldout_test_systems()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "train", "eval", "analyze"])
    ap.add_argument("--variety", type=int, default=4)
    ap.add_argument("--npool", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warm", type=int, default=1000)
    ap.add_argument("--vary-n", type=int, default=8192)
    ap.add_argument("--episode-base", type=int, default=24000)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--numbers", default="512,8192,65536")
    ap.add_argument("--varieties", default="1,2,3,4")
    ap.add_argument("--primary", choices=["prereg", "owner"], default="prereg")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    {"probe": probe, "train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
