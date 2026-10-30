"""Trial 20: replay prioritized by remaining error ("what to study"), otherwise identical to Trial 16's catalog+bio arm.
See docs/research-log.md.

  OMP_NUM_THREADS=4 nice -n 10 python trial20.py train --seed 1 --device cuda:0
  python trial20.py eval --device cuda:0
  python trial20.py analyze
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
from trial1 import BATCH, LR, N_QUERY, SEEDS, load, nmse, ridge
from trial4 import RELAX, Library, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES, library_fb
from trial9 import ObsBank, episode_obs
from trial10 import retention
from trial16 import OUT as TRIAL16
from trial16 import load_any, real_info

OUT = S.ROOT / "outputs/accumulating_learner/trial20"
EMA, MIX = 0.9, 0.5


class PrioBank(ObsBank):
    """ObsBank whose system draw follows given probabilities; also returns the drawn system indices."""

    def sample_prio(self, B, K, gen, probs):
        d, half = self.device, self.T // 2
        sys_ = torch.multinomial(probs, B, replacement=True, generator=gen).to(d)
        inst = (torch.rand(B, generator=gen).to(d) * self.n_inst[sys_]).long()
        ci = (1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :K]).to(d)
        qi = (half + 1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :N_QUERY]).to(d)
        std = T8.NOISE_X * self.std[sys_, inst].unsqueeze(1)
        nc = torch.randn(3, B, K, 3, generator=gen).to(d) * std
        nq = torch.randn(3, B, N_QUERY, 3, generator=gen).to(d) * std
        tx = self.x[sys_, inst]
        g = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, 3))
        hh = self.h[sys_].view(B, 1, 1)
        c = lambda a, b: (b - a) / (2 * hh)
        xp, xm, xn = g(tx, ci - 1) + nc[0], g(tx, ci) + nc[1], g(tx, ci + 1) + nc[2]
        qp, qm, qn = g(tx, qi - 1) + nq[0], g(tx, qi) + nq[1], g(tx, qi + 1) + nq[2]
        return T8.standardize(xm, c(xp, xn), qm, c(qp, qn)), sys_


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    split, conv = real_info()
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    bio_train = sorted(k for k, v in split.items() if v == "train")
    order = list(np.random.default_rng(1600 + s).permutation(catalog + bio_train))  # identical stream to Trial 16
    torch.manual_seed(1000 * s)
    lib = Library(4).to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem_x, mem_dx, mem_h, ema = [], [], [], []
    log, t0 = {"seed": s, "order": order, "stages": []}, time.time()
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        x_t, dx_t, h_t = load_any(name, dev, conv)
        new = ObsBank([x_t], [dx_t], [h_t], dev)
        mem = PrioBank(mem_x, mem_dx, mem_h, dev) if mem_x else None
        if mem is not None:
            e_t = torch.tensor(ema, dtype=torch.float32)
        new_err, lib_ = 0.0, lib.train()
        for step in range(400):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / 400))
            K = int(round(math.exp(torch.empty(1).uniform_(math.log(8), math.log(128), generator=gen).item())))
            if mem is None:
                uc, yc, uq, yq = new.sample(BATCH, K, gen)
                sys_ = None
            else:
                probs = MIX * e_t / e_t.sum() + (1 - MIX) / len(e_t)
                a = new.sample(BATCH // 2, K, gen)
                b, sys_ = mem.sample_prio(BATCH // 2, K, gen, probs)
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
            werr = best.values.exp().detach().cpu()
            if sys_ is None:
                new_err += float(werr.mean())
            else:
                new_err += float(werr[: BATCH // 2].mean())
                for j, v in zip(sys_.cpu().tolist(), werr[BATCH // 2:].tolist()):
                    e_t[j] = EMA * e_t[j] + (1 - EMA) * v
        if mem is not None:
            ema = e_t.tolist()
        ema.append(new_err / 400)
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES]); mem_h.append(h_t)
        log["stages"].append({"t": t, "system": name, "new_err": new_err / 400})
        if t % 10 == 0 or t == len(order):
            top = np.argsort(ema)[-3:][::-1]
            print(s, t, name, f"new {new_err / 400:.4f}", "highest-priority memory:", [order[j] for j in top], f"{time.time() - t0:.0f}s", flush=True)
    log["final_ema"] = dict(zip(order, ema))
    torch.save(lib.state_dict(), OUT / "learners" / f"prio_final_s{s}.pt")
    (OUT / "learners" / f"prio_s{s}.json").write_text(json.dumps(log))


@torch.no_grad()
def evaluate(args):
    dev = args.device
    torch.set_num_threads(4)
    split, conv = real_info()
    libs = {}
    for s in SEEDS:
        for tag, path in (("prio", OUT / "learners" / f"prio_final_s{s}.pt"), ("uniform", TRIAL16 / "learners" / f"catalogbio_final_s{s}.pt")):
            lib = Library(4).to(dev)
            lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
            libs[f"{tag}_s{s}"] = lib.eval()
    rows = []
    held = sorted(k for k, v in split.items() if v == "heldout")
    for i, name in enumerate(held):
        x, dx, h = load_any(name, dev, conv)
        gen = torch.Generator().manual_seed(16000 + 97 * i + 32)  # Trial 16's K = 32 episodes
        eps = [episode_obs(x[j:j + 1], dx[j:j + 1], 32, gen, dev, h) for j in range(4) for _ in range(5)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        for tag, lib in libs.items():
            rows.append({"part": "heldout", "system": name, "method": tag, "median": float(np.median(library_fb(lib, uc, yc, uq, yq, krr)[0].cpu().numpy()))})
    devnames = S.eligible_split()["development"]
    xs, dxs = load(devnames, dev)
    for si, name in enumerate(devnames):
        if name in NONPOLY:
            continue
        gen = torch.Generator().manual_seed(5000 + 97 * si + 32)
        eps = [episode_obs(xs[si][j:j + 1], dxs[si][j:j + 1], 32, gen, dev, T8.spacing(name)) for j in range(xs[si].shape[0]) for _ in range(5)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        for tag, lib in libs.items():
            rows.append({"part": "dev_poly", "system": name, "method": tag, "median": float(np.median(library_fb(lib, uc, yc, uq, yq, krr)[0].cpu().numpy()))})
    for tag, lib in libs.items():
        s = int(tag[-1])
        for pos, v in enumerate(retention(lib, s, dev)):
            rows.append({"part": "retention", "system": f"pos{pos}", "method": tag, "median": v})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["part"], r["system"], r["method"]): r["median"] for r in rows}
    out = {}
    for s in SEEDS:
        p, u = f"prio_s{s}", f"uniform_s{s}"
        ns = lambda part: sorted({r["system"] for r in rows if r["part"] == part})
        med = lambda part, m: float(np.median([sc[(part, n, m)] for n in ns(part)]))
        p1 = med("heldout", p) / med("heldout", u)
        p2 = med("dev_poly", p) / med("dev_poly", u)
        p3 = float(np.median([sc[("retention", n, p)] / sc[("retention", n, u)] for n in ns("retention")]))
        out[s] = {"P1_heldout_ratio": p1, "P2_devpoly_ratio": p2, "P3_retention_ratio": p3,
                  "criteria": {"P1": bool(p1 <= 0.8), "P2": bool(p2 <= 1.2), "P3": bool(p3 <= 1.0)},
                  "heldout_per_system": {n: {"prio": sc[("heldout", n, p)], "uniform": sc[("heldout", n, u)]} for n in ns("heldout")}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
