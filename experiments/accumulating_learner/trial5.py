"""Trial 5: the Trial 4 library trained sequentially over a stream of systems, with an optional replay memory.

  python trial5.py train --arm seqreplay400 --seed 1 --device cuda:0
  python trial5.py eval --device cuda:0
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
from trial1 import BATCH, K_GRID, LR, N_GRID, SEEDS, episode, load, nmse, ridge, ridge_loo_with_score
from trial3 import Bank
from trial4 import M, RELAX, Library, krr_with_score

OUT = S.ROOT / "outputs/accumulating_learner/trial5"
TRIAL1 = S.ROOT / "outputs/accumulating_learner/trial1"
ARMS = {"seqreplay400": (True, 400), "seqnoreplay400": (False, 400), "seqreplay2000": (True, 2000)}
MEMORY_INSTANCES = 2
LR_MIN = 1e-4


def make_bank(xs, dxs, device):
    b = Bank.__new__(Bank)
    b.n_inst = torch.tensor([x.shape[0] for x in xs], device=device)
    imax, T = int(b.n_inst.max()), xs[0].shape[1]
    b.x = torch.zeros(len(xs), imax, T, 3, device=device)
    b.dx = torch.zeros_like(b.x)
    for j, (x, dx) in enumerate(zip(xs, dxs)):
        b.x[j, : x.shape[0]], b.dx[j, : x.shape[0]] = x, dx
    b.T, b.device = T, device
    return b


@torch.no_grad()
def library_fb(lib, uc, yc, uq, yq, krr):
    """Library + kernel-ridge fallback, chosen per episode by context LOO. Returns (NMSE, selection)."""
    errs, loos = zip(*[ridge_loo_with_score(b(uc), yc, b(uq), yq) for b in lib.bases])
    E = torch.cat([torch.stack(errs, 1), krr[0].unsqueeze(1)], 1)
    sel = torch.cat([torch.stack(loos, 1), krr[1].unsqueeze(1)], 1).argmin(1)
    return E.gather(1, sel.unsqueeze(1)).squeeze(1), sel


def train(args):
    replay, steps = ARMS[args.arm]
    dev = args.device
    torch.manual_seed(1000 * args.seed)  # identical initial library to Trial 4's N = 0
    lib = Library().to(dev)
    order = list(np.random.default_rng(args.seed).permutation(S.eligible_split()["meta_train"]))
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    tag = lambda t: f"{args.arm}_N{t:02d}_s{args.seed}"
    torch.save(lib.state_dict(), OUT / "learners" / f"{tag(0)}.pt")
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * args.seed)
    mem_x, mem_dx, log = [], [], {"arm": args.arm, "seed": args.seed, "order": order, "stages": []}
    t0 = time.time()
    for t, name in enumerate(order, 1):
        (x_t,), (dx_t,) = load([name], dev)
        # Encounter evaluation before any training on this system.
        eg = torch.Generator().manual_seed(9100 + 97 * t + args.seed)
        eps = [episode(x_t[0:1], dx_t[0:1], 32, eg, dev) for _ in range(20)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        e_lib, sel = library_fb(lib.eval(), uc, yc, uq, yq, krr)
        lib.train()
        new = make_bank([x_t], [dx_t], dev)
        mem = make_bank(mem_x, mem_dx, dev) if (replay and mem_x) else None
        run, wins = 0.0, torch.zeros(M)
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / steps))
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
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES])
        stage = {"t": t, "system": name, "encounter_libraryfb": float(e_lib.median()), "encounter_krr": float(krr[0].median()),
                 "encounter_selection": torch.bincount(sel.cpu(), minlength=M + 1).tolist(),
                 "train_min_nmse": run / steps, "wins": wins.tolist(), "seconds": time.time() - t0}
        log["stages"].append(stage)
        print(args.arm, args.seed, t, name, f"enc lib {stage['encounter_libraryfb']:.4f} krr {stage['encounter_krr']:.4f}",
              f"train {stage['train_min_nmse']:.4f}", f"{stage['seconds']:.0f}s", flush=True)
        if t in N_GRID:
            torch.save(lib.state_dict(), OUT / "learners" / f"{tag(t)}.pt")
    log["retained_examples"] = {"instances_per_system": MEMORY_INSTANCES, "points_per_instance": int(x_t.shape[1])}
    (OUT / "learners" / f"{args.arm}_s{args.seed}.json").write_text(json.dumps(log))


def load_libs(dev):
    libs = {}
    for arm in ARMS:
        for t in N_GRID:
            for s in SEEDS:
                p = OUT / "learners" / f"{arm}_N{t:02d}_s{s}.pt"
                if p.exists():
                    lib = Library().to(dev)
                    lib.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
                    libs[(arm, t, s)] = lib.eval()
    return libs


def evaluate(args):
    dev = args.device
    dev_names = S.eligible_split()["development"]
    xs, dxs = load(dev_names, dev)
    libs = load_libs(dev)
    rows = [r for r in json.loads((TRIAL1 / "eval_development.json").read_text())["rows"]
            if not r["method"].startswith("learned")]
    selections = []
    for si, name in enumerate(dev_names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev)
                   for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            summary = {}
            for (arm, t, s), lib in libs.items():
                e, sel = library_fb(lib, uc, yc, uq, yq, krr)
                rows.append({"system": name, "K": K, "method": f"{arm}_N{t:02d}_s{s}", "episodes": e.float().cpu().tolist()})
                selections.append({"system": name, "K": K, "arm": arm, "N": t, "seed": s,
                                   "counts": torch.bincount(sel.cpu(), minlength=M + 1).tolist()})
                if s == 1 and t == 51:
                    summary[arm] = round(float(e.median()), 4)
            print(name, K, summary, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows, "selections": selections}))


def evaluate_retention(args):
    dev = args.device
    meta = S.eligible_split()["meta_train"]
    libs = load_libs(dev)
    rows = []
    for s in SEEDS:
        order = list(np.random.default_rng(s).permutation(meta))[:8]
        xs, dxs = load(order, dev)
        for si, name in enumerate(order):
            gen = torch.Generator().manual_seed(7000 + 97 * si + s)
            eps = [episode(xs[si][0:1], dxs[si][0:1], 32, gen, dev) for _ in range(20)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            for arm in ARMS:
                for t in N_GRID:
                    if (arm, t, s) in libs:
                        e, _ = library_fb(libs[(arm, t, s)], uc, yc, uq, yq, krr)
                        rows.append({"arm": arm, "seed": s, "position": si, "system": name, "N": t, "seen": si < t,
                                     "median_nmse": float(e.median())})
    (OUT / "eval_retention.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "retention"])
    ap.add_argument("--arm", choices=list(ARMS), default="seqreplay400")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "retention": evaluate_retention}[a.cmd](a)
