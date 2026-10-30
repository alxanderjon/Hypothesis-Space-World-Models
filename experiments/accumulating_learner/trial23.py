"""Trial 23: the any-dimension pairwise library with a large real history (BioModels oscillators and transients).
See docs/research-log.md for the pre-registered protocol.

  OMP_NUM_THREADS=4 nice -n 10 python trial23.py train --seed 1 --device cuda:0
  python trial23.py eval --device cuda:0
  python trial23.py analyze
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
from trial18 import (N_QUERY, OUT as TRIAL18, REAL as REAL18, BankD, PairLibrary, central, episode_d, library_fb_pw,
                     standardize, task_loss)

OUT = S.ROOT / "outputs/accumulating_learner/trial23"
REAL = OUT / "realfam_big"
KS = (16, 32, 64)
STEPS = 400  # per stage; may be set lower in the pre-registration once the system count is known


def info():
    split = json.loads((REAL / "split.json").read_text())
    excluded = set(json.loads((REAL / "validity_excluded.json").read_text()))  # near-constant variable or non-finite samples
    split = {k: v for k, v in split.items() if k not in excluded}
    conv = {r["id"]: r for r in json.loads((REAL / "conversion.json").read_text())["records"]}
    return split, conv


def load_sys(name, dev, conv):
    if name.startswith("BIOMD"):
        z = np.load(REAL / f"{name}.npz")
        return torch.tensor(z["x"], device=dev), conv[name]["h"], conv[name].get("kind", "oscillator")
    (x,), _ = load([name], dev)
    return x, T8.spacing(name), "oscillator"


def episode_transient(x, K, gen, dev, h):
    """Context: K snippets from instances 0-3 (any time); queries: clean central differences on instances 4-7."""
    I, T, d = x.shape
    ci_inst = torch.randint(4, (K,), generator=gen)
    ci = 1 + torch.randint(T - 2, (K,), generator=gen)
    qi_inst = 4 + torch.randint(I - 4, (N_QUERY,), generator=gen)
    qi = 1 + torch.randint(T - 2, (N_QUERY,), generator=gen)
    noise = torch.randn(3, K, d, generator=gen).to(dev) * T8.NOISE_X * x[:4].reshape(-1, d).std(0)
    ci_inst, ci, qi_inst, qi = (t.to(dev) for t in (ci_inst, ci, qi_inst, qi))
    xp, xm, xn = x[ci_inst, ci - 1] + noise[0], x[ci_inst, ci] + noise[1], x[ci_inst, ci + 1] + noise[2]
    return standardize(xm, central(xp, xn, h), x[qi_inst, qi], central(x[qi_inst, qi - 1], x[qi_inst, qi + 1], h))


class TransientBank:
    """Training sampler for a transient system: context instances and query instances are disjoint (all observable)."""

    def __init__(self, x_ctx, x_qry, h):
        self.xc, self.xq, self.h = x_ctx, x_qry, h
        self.std = x_ctx.reshape(-1, x_ctx.shape[-1]).std(0)

    def sample(self, B, K, gen):
        Ic, T, d = self.xc.shape
        Iq = self.xq.shape[0]
        dev = self.xc.device
        ci_i, ci = torch.randint(Ic, (B, K), generator=gen).to(dev), (1 + torch.randint(T - 2, (B, K), generator=gen)).to(dev)
        qi_i, qi = torch.randint(Iq, (B, N_QUERY), generator=gen).to(dev), (1 + torch.randint(T - 2, (B, N_QUERY), generator=gen)).to(dev)
        s = T8.NOISE_X * self.std
        nc = torch.randn(3, B, K, d, generator=gen).to(dev) * s
        nq = torch.randn(3, B, N_QUERY, d, generator=gen).to(dev) * s
        xp, xm, xn = self.xc[ci_i, ci - 1] + nc[0], self.xc[ci_i, ci] + nc[1], self.xc[ci_i, ci + 1] + nc[2]
        qp, qm, qn = self.xq[qi_i, qi - 1] + nq[0], self.xq[qi_i, qi] + nq[1], self.xq[qi_i, qi + 1] + nq[2]
        return standardize(xm, central(xp, xn, self.h), qm, central(qp, qn, self.h))


def banks(x, h, kind):
    """(bank for the full system, bank for its replay memory)."""
    if kind == "transient":
        return TransientBank(x[:4], x[4:], h), TransientBank(x[:2], x[4:6], h)
    return BankD(x, h), BankD(x[:2], h)


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    split, conv = info()
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    real_train = sorted(k for k, v in split.items() if v == "train")
    order = list(np.random.default_rng(2300 + s).permutation(catalog + real_train))
    torch.manual_seed(1000 * s)
    lib = PairLibrary().to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem, log, t0 = [], {"seed": s, "order": order, "steps_per_stage": STEPS, "stages": []}, time.time()
    nonfinite = 0
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        x, h, kind = load_sys(name, dev, conv)
        new, keep = banks(x, h, kind)
        lib.train()
        run = 0.0
        for step in range(STEPS):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / STEPS))
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            if mem:
                la, ea = task_loss(lib, new.sample(BATCH // 2, K, gen))
                mb = mem[int(torch.randint(len(mem), (1,), generator=gen))]
                lb, eb = task_loss(lib, mb.sample(BATCH // 2, K, gen))
                loss, e = 0.5 * (la + lb), 0.5 * (ea + eb)
            else:
                loss, e = task_loss(lib, new.sample(BATCH, K, gen))
            if not torch.isfinite(loss):  # a bad batch must never corrupt the weights (Trial 23 deviation)
                nonfinite += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lib.parameters(), 1.0)
            opt.step()
            run += e
        mem.append(keep)
        log["stages"].append({"t": t, "system": name, "kind": kind, "d": int(x.shape[-1]), "train_min_nmse": run / STEPS, "nonfinite_so_far": nonfinite})
        if t % 10 == 0 or t == len(order):
            print(s, t, "/", len(order), name, kind, f"d={x.shape[-1]} train {run / STEPS:.4f} nonfinite {nonfinite}", f"{time.time() - t0:.0f}s", flush=True)
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
    split, conv = info()
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
    new_held = sorted(k for k, v in split.items() if v == "heldout" and k not in s18)
    for i, name in enumerate(new_held):  # new held-out systems (transient or oscillatory), own contract
        x, h, kind = load_sys(name, dev, conv)
        for K in KS:
            gen = torch.Generator().manual_seed(23000 + 97 * i + K)
            if kind == "transient":
                eps = [episode_transient(x, K, gen, dev, h) for _ in range(20)]
            else:
                eps = [episode_d(x[j], K, gen, dev, h) for j in range(4) for _ in range(5)]
            res = score(eps, libs)
            rows += [{"part": f"new_heldout_{kind}", "system": name, "d": int(x.shape[-1]), "K": K, "method": m, "episodes": v.float().cpu().tolist()} for m, v in res.items()]
            print(kind, name, K, {m: round(float(np.median(v.float().cpu().numpy())), 5) for m, v in res.items() if m.endswith("s1") or m == "krr"}, flush=True)
    c18 = {r["id"]: r for r in json.loads((REAL18 / "conversion.json").read_text())["records"]}
    held18 = sorted(k for k, v in s18.items() if v == "heldout")
    devnames = S.eligible_split()["development"]
    targets = [(n, "t18_heldout") for n in held18] + [(n, "development") for n in devnames]
    for i, (name, part) in enumerate(targets):  # Trial 18's exact episodes
        if name.startswith("BIOMD"):
            x, h = torch.tensor(np.load(REAL18 / f"{name}.npz")["x"], device=dev), c18[name]["h"]
        else:
            (x,), _ = load([name], dev); h = T8.spacing(name)
        K = 32
        gen = torch.Generator().manual_seed(18000 + 97 * i + K)
        eps = [episode_d(x[j], K, gen, dev, h) for j in range(min(4, x.shape[0])) for _ in range(5)]
        res = score(eps, libs)
        rows += [{"part": part, "system": name, "d": int(x.shape[-1]), "K": K, "method": m, "episodes": v.float().cpu().tolist()} for m, v in res.items()]
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["part"], r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    parts = sorted({r["part"] for r in rows})
    names = lambda part: sorted({r["system"] for r in rows if r["part"] == part})
    newp = [p for p in parts if p.startswith("new_heldout")]
    new_names = [(p, n) for p in newp for n in names(p)]
    poly = [n for n in names("development") if n not in NONPOLY]
    out = {}
    for s in SEEDS:
        b, c, a = f"pwbig_s{s}", f"pwcatalog_s{s}", f"pwall_s{s}"
        med_new = lambda m, K: float(np.median([sc[(p, n, K, m)] for p, n in new_names]))
        b1 = {"ratio_vs_pwcatalog": med_new(b, 32) / med_new(c, 32),
              "wins_vs_krr": sum(sc[(p, n, 32, b)] < sc[(p, n, 32, "krr")] for p, n in new_names), "n": len(new_names)}
        h18 = names("t18_heldout")
        m18 = lambda m: float(np.median([sc[("t18_heldout", n, 32, m)] for n in h18]))
        b2 = {"ratio_vs_pwall": m18(b) / m18(a), "wins_vs_krr": sum(sc[("t18_heldout", n, 32, b)] < sc[("t18_heldout", n, 32, "krr")] for n in h18)}
        mp = lambda m: float(np.median([sc[("development", n, 32, m)] for n in poly]))
        b3 = mp(b) / mp(c)
        crit = {"B1": bool(b1["ratio_vs_pwcatalog"] <= 0.67 and b1["wins_vs_krr"] >= 0.6 * b1["n"]),
                "B2": bool(b2["ratio_vs_pwall"] <= 0.8 and b2["wins_vs_krr"] >= 10), "B3": bool(b3 <= 1.5)}
        out[s] = {"B1": b1, "B2": b2, "B3_ratio_poly_dev": b3, "criteria": crit,
                  "new_heldout_medians": {m: {K: med_new(m if m == "krr" else f"{m}_s{s}", K) for K in KS} for m in ("pwcatalog", "pwall", "pwbig", "krr")},
                  "by_kind_K32": {p: {m: float(np.median([sc[(p, n, 32, m if m == 'krr' else f'{m}_s{s}')] for n in names(p)])) for m in ("pwcatalog", "pwall", "pwbig", "krr")} for p in newp}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    STEPS = a.steps
    {"train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
