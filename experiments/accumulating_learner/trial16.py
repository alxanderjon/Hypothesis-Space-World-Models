"""Trial 16: does related real history (BioModels families) transfer to held-out real mechanisms? See docs/research-log.md.

  OMP_NUM_THREADS=4 nice -n 10 python trial16.py train --seed 1 --device cuda:0
  python trial16.py eval --device cuda:0
  python trial16.py analyze
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
from trial1 import BATCH, LR, SEEDS, load, nmse, ridge
from trial4 import RELAX, Library, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES, library_fb
from trial9 import OUT as TRIAL9
from trial9 import ObsBank, episode_obs
from trial15 import OUT as TRIAL15

OUT = S.ROOT / "outputs/accumulating_learner/trial16"
REAL = OUT / "realfam"
KS = (16, 32, 64)


def real_info():
    split = json.loads((REAL / "split.json").read_text())
    conv = {r["id"]: r for r in json.loads((REAL / "conversion.json").read_text())["records"]}
    return split, conv


def load_any(name, dev, conv):
    if name.startswith("BIOMD"):
        z = np.load(REAL / f"{name}.npz")
        return torch.tensor(z["x"], device=dev), torch.tensor(z["dx"], device=dev), conv[name]["h"]
    (x,), (dx,) = load([name], dev)
    return x, dx, T8.spacing(name)


def train(args):
    dev, s = args.device, args.seed
    torch.set_num_threads(4)
    split, conv = real_info()
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    bio_train = sorted(k for k, v in split.items() if v == "train")
    order = list(np.random.default_rng(1600 + s).permutation(catalog + bio_train))
    torch.manual_seed(1000 * s)
    lib = Library(4).to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem_x, mem_dx, mem_h, log = [], [], [], {"seed": s, "order": order, "stages": []}
    t0 = time.time()
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        x_t, dx_t, h_t = load_any(name, dev, conv)
        new = ObsBank([x_t], [dx_t], [h_t], dev)
        mem = ObsBank(mem_x, mem_dx, mem_h, dev) if mem_x else None
        run = 0.0
        lib.train()
        for step in range(400):
            for g in opt.param_groups:
                g["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / 400))
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
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES]); mem_h.append(h_t)
        log["stages"].append({"t": t, "system": name, "train_min_nmse": run / 400})
        if t % 10 == 0 or t == len(order) or name.startswith("BIOMD"):
            print(s, t, name, f"train {run / 400:.4f}", f"{time.time() - t0:.0f}s", flush=True)
    torch.save(lib.state_dict(), OUT / "learners" / f"catalogbio_final_s{s}.pt")
    (OUT / "learners" / f"catalogbio_s{s}.json").write_text(json.dumps(log))


def load_libs(dev):
    libs = {}
    for s in SEEDS:
        for tag, path, m in (("catalogonly", TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", 4),
                             ("catalogbio", OUT / "learners" / f"catalogbio_final_s{s}.pt", 4),
                             ("catalogsyn", TRIAL15 / "learners" / f"broad_final_s{s}.pt", 4)):
            lib = Library(m).to(dev)
            lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
            libs[f"{tag}_s{s}"] = lib.eval()
    return libs


def evaluate(args):
    dev = args.device
    torch.set_num_threads(4)
    split, conv = real_info()
    libs = load_libs(dev)
    rows = []
    held = sorted(k for k, v in split.items() if v == "heldout")
    for i, name in enumerate(held):
        x, dx, h = load_any(name, dev, conv)
        for K in KS:
            gen = torch.Generator().manual_seed(16000 + 97 * i + K)
            eps = [episode_obs(x[j:j + 1], dx[j:j + 1], K, gen, dev, h) for j in range(4) for _ in range(5)]
            rows += score(eps, libs, name, K, "heldout")
    devnames = S.eligible_split()["development"]
    xs, dxs = load(devnames, dev)
    for si, name in enumerate(devnames):
        h = T8.spacing(name)
        gen = torch.Generator().manual_seed(5000 + 97 * si + 32)  # Trial 9's K = 32 episodes
        eps = [episode_obs(xs[si][j:j + 1], dxs[si][j:j + 1], 32, gen, dev, h) for j in range(xs[si].shape[0]) for _ in range(5)]
        rows += score(eps, libs, name, 32, "development")
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}))


@torch.no_grad()
def score(eps, libs, name, K, part):
    uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
    krr = krr_with_score(uc, yc, uq, yq)
    res = {"krr": krr[0], **{tag: library_fb(lib, uc, yc, uq, yq, krr)[0] for tag, lib in libs.items()}}
    print(part, name, K, {m: round(float(np.median(v.float().cpu().numpy())), 5) for m, v in res.items() if m.endswith("s1") or m == "krr"}, flush=True)
    return [{"part": part, "system": name, "K": K, "method": m, "episodes": v.float().cpu().tolist()} for m, v in res.items()]


def analyze(args):
    rows = json.loads((OUT / "eval.json").read_text())["rows"]
    sc = {(r["part"], r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    split, conv = real_info()
    held = sorted(k for k, v in split.items() if v == "heldout")
    poly = [n for n in S.eligible_split()["development"] if n not in NONPOLY]
    med = lambda part, ns, K, m: float(np.median([sc[(part, n, K, m)] for n in ns]))
    out = {}
    for s in SEEDS:
        cb, co, cs = f"catalogbio_s{s}", f"catalogonly_s{s}", f"catalogsyn_s{s}"
        r1 = {"ratio_vs_catalogonly": med("heldout", held, 32, cb) / med("heldout", held, 32, co),
              "wins_vs_krr": sum(sc[("heldout", n, 32, cb)] < sc[("heldout", n, 32, "krr")] for n in held)}
        r2 = med("heldout", held, 32, cb) / med("heldout", held, 32, cs)
        r3 = med("development", poly, 32, cb) / med("development", poly, 32, co)
        crit = {"R1": bool(r1["ratio_vs_catalogonly"] <= 0.67 and r1["wins_vs_krr"] >= 4), "R2": bool(r2 <= 0.8), "R3": bool(r3 <= 1.5)}
        out[s] = {"R1": r1, "R2_ratio_vs_synthetic": r2, "R3_ratio_poly_dev": r3, "criteria": crit,
                  "heldout_medians": {m: {K: med("heldout", held, K, m if m == "krr" else f"{m}_s{s}") for K in KS}
                                      for m in ("catalogonly", "catalogbio", "catalogsyn", "krr")},
                  "heldout_per_system_K32": {f"{n} {conv[n]['name'][:30]}": {m: sc[("heldout", n, 32, m if m == "krr" else f"{m}_s{s}")]
                                                                             for m in ("catalogonly", "catalogbio", "catalogsyn", "krr")} for n in held},
                  "dev_nonpoly_K32": {m: med("development", sorted(NONPOLY), 32, m if m == "krr" else f"{m}_s{s}") for m in ("catalogonly", "catalogbio", "krr")}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
