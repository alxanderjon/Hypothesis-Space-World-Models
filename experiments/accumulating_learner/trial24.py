"""Trial 24: resampled (learnable) transients with time stamps, a kernel-ridge solvability gate, then the large-history
test of Trial 23 again. See docs/research-log.md.

  python trial24.py check --device cuda:0                 # step 2: kernel-ridge solvability per scheme (training transients)
  OMP_NUM_THREADS=4 nice -n 10 python trial24.py train --scheme arclength --seed 1 --device cuda:0
  python trial24.py eval --scheme arclength --device cuda:0
  python trial24.py analyze
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
from trial1 import BATCH, LR, SEEDS, load
from trial4 import krr_with_score
from trial5 import LR_MIN
from trial18 import N_QUERY, OUT as TRIAL18, REAL as REAL18, BankD, PairLibrary, episode_d, library_fb_pw, standardize, task_loss
from trial23 import REAL as REAL23
from trial23 import info as info23

OUT = S.ROOT / "outputs/accumulating_learner/trial24"
KS = (16, 32, 64)


def scheme_ids(scheme):
    return {p.stem for p in (OUT / scheme).glob("BIOMD*.npz")}


def load_transient(name, dev, scheme):
    z = np.load(OUT / scheme / f"{name}.npz")
    return torch.tensor(z["x"], device=dev), torch.tensor(z["t"], device=dev, dtype=torch.float32)


def draw(x, t, n_ctx, n_q, gen, dev, noisy_query):
    """Context and query snippets at disjoint (instance, index) pairs from instances 0-3; per-sample central differences."""
    _, T, d = x.shape
    flat = torch.randperm(4 * (T - 2), generator=gen)[: n_ctx + n_q]
    inst, idx = (flat // (T - 2)).to(dev), (1 + flat % (T - 2)).to(dev)
    std = T8.NOISE_X * x[:4].reshape(-1, d).std(0)
    noise = torch.randn(3, n_ctx + n_q, d, generator=gen).to(dev) * std
    if not noisy_query:
        noise[:, n_ctx:] = 0
    xp, xm, xn = x[inst, idx - 1] + noise[0], x[inst, idx] + noise[1], x[inst, idx + 1] + noise[2]
    dt = (t[inst, idx + 1] - t[inst, idx - 1]).unsqueeze(-1)
    dest = (xn - xp) / dt
    return standardize(xm[:n_ctx], dest[:n_ctx], xm[n_ctx:], dest[n_ctx:])


def episode_tt(x, t, K, gen, dev):
    return draw(x, t, K, N_QUERY, gen, dev, noisy_query=False)


class StampedBank:
    def __init__(self, x, t):
        self.x, self.t = x, t

    def sample(self, B, K, gen):
        """Vectorized version of draw() for B episodes (noisy context and noisy queries, all observable)."""
        I, T, d = self.x.shape
        dev = self.x.device
        flat = torch.rand(B, I * (T - 2), generator=gen).argsort(1)[:, : K + N_QUERY].to(dev)
        inst, idx = flat // (T - 2), 1 + flat % (T - 2)
        std = T8.NOISE_X * self.x.reshape(-1, d).std(0)
        noise = torch.randn(3, B, K + N_QUERY, d, generator=gen).to(dev) * std
        xp, xm, xn = self.x[inst, idx - 1] + noise[0], self.x[inst, idx] + noise[1], self.x[inst, idx + 1] + noise[2]
        dest = (xn - xp) / (self.t[inst, idx + 1] - self.t[inst, idx - 1]).unsqueeze(-1)
        return standardize(xm[:, :K], dest[:, :K], xm[:, K:], dest[:, K:])


def check(args):
    """Step 2: kernel-ridge median NMSE at K = 32 on the training transients, for each scheme."""
    dev = args.device
    split, _ = info23()
    out = {}
    for scheme in ("logtime", "arclength"):
        ids = sorted(k for k in scheme_ids(scheme) if split.get(k) == "train")
        vals = []
        for i, n in enumerate(ids):
            x, t = load_transient(n, dev, scheme)
            g = torch.Generator().manual_seed(24500 + i)
            eps = [episode_tt(x, t, 32, g, dev) for _ in range(10)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            vals.append(float(np.median(krr_with_score(uc, yc, uq, yq)[0].cpu().numpy())))
        out[scheme] = {"n": len(ids), "median": float(np.median(vals)), "share_below_1": float(np.mean(np.array(vals) < 1)),
                       "share_below_0.1": float(np.mean(np.array(vals) < 0.1))}
        print(scheme, out[scheme], flush=True)
    best = min(out, key=lambda k: out[k]["median"])
    out["chosen"] = best
    out["gate_passed"] = bool(out[best]["median"] <= 0.5)
    print("chosen scheme:", best, "| gate (median <= 0.5) passed:", out["gate_passed"])
    (OUT / "solvability.json").write_text(json.dumps(out, indent=1))


def system_banks(name, dev, conv, scheme):
    if name.startswith("BIOMD") and conv[name].get("kind") == "transient":
        x, t = load_transient(name, dev, scheme)
        return StampedBank(x[:4], t[:4]), StampedBank(x[:2], t[:2]), int(x.shape[-1])
    if name.startswith("BIOMD"):
        x = torch.tensor(np.load(REAL23 / f"{name}.npz")["x"], device=dev)
        h = conv[name]["h"]
    else:
        (x,), _ = load([name], dev)
        h = T8.spacing(name)
    return BankD(x, h), BankD(x[:2], h), int(x.shape[-1])


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    split, conv = info23()
    have = scheme_ids(args.scheme)
    real_train = sorted(k for k, v in split.items() if v == "train" and (conv[k].get("kind") != "transient" or k in have))
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    order = list(np.random.default_rng(2400 + s).permutation(catalog + real_train))
    torch.manual_seed(1000 * s)
    lib = PairLibrary().to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem, nonfinite, t0 = [], 0, time.time()
    log = {"seed": s, "scheme": args.scheme, "order": order, "stages": []}
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        new, keep, d = system_banks(name, dev, conv, args.scheme)
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
            if not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step()
            run += e
        mem.append(keep)
        log["stages"].append({"t": t, "system": name, "d": d, "train_min_nmse": run / 400, "nonfinite_so_far": nonfinite})
        if t % 10 == 0 or t == len(order):
            print(s, t, "/", len(order), name, f"d={d} train {run / 400:.4f} nonfinite {nonfinite}", f"{time.time() - t0:.0f}s", flush=True)
    torch.save(lib.state_dict(), OUT / "learners" / f"pwbig_final_s{s}.pt")
    (OUT / "learners" / f"pwbig_s{s}.json").write_text(json.dumps(log))


@torch.no_grad()
def score(eps, libs):
    uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
    krr = krr_with_score(uc, yc, uq, yq)
    return {"krr": krr[0], **{n: library_fb_pw(lib, uc, yc, uq, yq, krr)[0] for n, lib in libs.items()}}


def evaluate(args):
    dev = args.device
    torch.set_num_threads(4)
    split, conv = info23()
    have = scheme_ids(args.scheme)
    libs = {}
    for s in SEEDS:
        for tag, path in (("pwbig", OUT / "learners" / f"pwbig_final_s{s}.pt"),
                          ("pwall", TRIAL18 / "learners" / f"pwall_final_s{s}.pt"),
                          ("pwcatalog", TRIAL18 / "learners" / f"pwcatalog_final_s{s}.pt")):
            lib = PairLibrary().to(dev)
            lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
            libs[f"{tag}_s{s}"] = lib.eval()
    rows = []
    s18 = json.loads((REAL18 / "split.json").read_text())
    new_held = sorted(k for k, v in split.items() if v == "heldout" and k not in s18 and k in have)
    for i, name in enumerate(new_held):
        x, t = load_transient(name, dev, args.scheme)
        for K in KS:
            gen = torch.Generator().manual_seed(24000 + 97 * i + K)
            res = score([episode_tt(x, t, K, gen, dev) for _ in range(20)], libs)
            rows += [{"part": "new_heldout_transient", "system": name, "d": int(x.shape[-1]), "K": K, "method": m, "episodes": v.float().cpu().tolist()} for m, v in res.items()]
            print("transient", name, K, {m: round(float(np.median(v.float().cpu().numpy())), 5) for m, v in res.items() if m.endswith("s1") or m == "krr"}, flush=True)
    c18 = {r["id"]: r for r in json.loads((REAL18 / "conversion.json").read_text())["records"]}
    held18 = sorted(k for k, v in s18.items() if v == "heldout")
    targets = [(n, "t18_heldout") for n in held18] + [(n, "development") for n in S.eligible_split()["development"]]
    for i, (name, part) in enumerate(targets):  # Trial 18's exact K = 32 episodes
        if name.startswith("BIOMD"):
            x, h = torch.tensor(np.load(REAL18 / f"{name}.npz")["x"], device=dev), c18[name]["h"]
        else:
            (x,), _ = load([name], dev); h = T8.spacing(name)
        gen = torch.Generator().manual_seed(18000 + 97 * i + 32)
        res = score([episode_d(x[j], 32, gen, dev, h) for j in range(min(4, x.shape[0])) for _ in range(5)], libs)
        rows += [{"part": part, "system": name, "d": int(x.shape[-1]), "K": 32, "method": m, "episodes": v.float().cpu().tolist()} for m, v in res.items()]
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["part"], r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    names = lambda part: sorted({r["system"] for r in rows if r["part"] == part})
    newn, h18 = names("new_heldout_transient"), names("t18_heldout")
    poly = [n for n in names("development") if n not in NONPOLY]
    out = {}
    for s in SEEDS:
        b, c, a = f"pwbig_s{s}", f"pwcatalog_s{s}", f"pwall_s{s}"
        mn = lambda m, K: float(np.median([sc[("new_heldout_transient", n, K, m)] for n in newn]))
        b1 = {"ratio_vs_pwcatalog": mn(b, 32) / mn(c, 32), "wins_vs_krr": sum(sc[("new_heldout_transient", n, 32, b)] < sc[("new_heldout_transient", n, 32, "krr")] for n in newn), "n": len(newn)}
        m18 = lambda m: float(np.median([sc[("t18_heldout", n, 32, m)] for n in h18]))
        b2 = {"ratio_vs_pwall": m18(b) / m18(a), "wins_vs_krr": sum(sc[("t18_heldout", n, 32, b)] < sc[("t18_heldout", n, 32, "krr")] for n in h18)}
        b3 = float(np.median([sc[("development", n, 32, b)] for n in poly])) / float(np.median([sc[("development", n, 32, c)] for n in poly]))
        crit = {"B1": bool(b1["ratio_vs_pwcatalog"] <= 0.67 and b1["wins_vs_krr"] >= 0.6 * b1["n"]),
                "B2": bool(b2["ratio_vs_pwall"] <= 0.8 and b2["wins_vs_krr"] >= 10), "B3": bool(b3 <= 1.5)}
        out[s] = {"B1": b1, "B2": b2, "B3_ratio_poly_dev": b3, "criteria": crit,
                  "new_heldout_medians": {m: {K: mn(m if m == "krr" else f"{m}_s{s}", K) for K in KS} for m in ("pwcatalog", "pwall", "pwbig", "krr")},
                  "t18_medians": {m: m18(m if m == "krr" else f"{m}_s{s}") for m in ("pwcatalog", "pwall", "pwbig", "krr")}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "train", "eval", "analyze"])
    ap.add_argument("--scheme", default="arclength")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"check": check, "train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
