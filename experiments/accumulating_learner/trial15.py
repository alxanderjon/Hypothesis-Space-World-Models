"""Trial 15: does a procedurally broadened history help unfamiliar mechanisms? See docs/research-log.md.

  python trial15.py train --arm broad --seed 1 --device cuda:0      # arms: cataloglong, broad, broad8
  python trial15.py eval --device cuda:0
  python trial15.py analyze
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
from synthetic import OUT as SYN
from trial1 import BATCH, LR, SEEDS, load, nmse, ridge
from trial4 import RELAX, Library, krr_with_score
from trial5 import LR_MIN, MEMORY_INSTANCES, library_fb
from trial9 import OUT as TRIAL9
from trial9 import ObsBank, episode_obs

OUT = S.ROOT / "outputs/accumulating_learner/trial15"
ARMS = {"cataloglong": (False, 1970, 4), "broad": (True, 400, 4), "broad8": (True, 400, 8)}
KS = (16, 32, 64)


def syn_meta():
    return json.loads((SYN / "synthetic.json").read_text())["systems"]


def load_any(name, dev, meta):
    if name.startswith("syn"):
        z = np.load(SYN / f"{name}.npz")
        return torch.tensor(z["x"], device=dev), torch.tensor(z["dx"], device=dev), meta[name]["h"]
    (x,), (dx,) = load([name], dev)
    return x, dx, T8.spacing(name)


def train(args):
    use_syn, steps, m = ARMS[args.arm]
    dev, s = args.device, args.seed
    meta = syn_meta() if use_syn else {}
    catalog = list(np.random.default_rng(s).permutation(S.eligible_split()["meta_train"]))
    order = list(np.random.default_rng(1500 + s).permutation(catalog + sorted(meta))) if use_syn else catalog
    torch.manual_seed(1000 * s)
    lib = Library(m).to(dev)
    opt = torch.optim.Adam(lib.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(500 + 1000 * s)
    mem_x, mem_dx, mem_h, log = [], [], [], {"arm": args.arm, "seed": s, "order": order, "stages": []}
    t0 = time.time()
    (OUT / "learners").mkdir(parents=True, exist_ok=True)
    for t, name in enumerate(order, 1):
        x_t, dx_t, h_t = load_any(name, dev, meta)
        eg = torch.Generator().manual_seed(9100 + 97 * t + s)
        eps = [episode_obs(x_t[0:1], dx_t[0:1], 32, eg, dev, h_t) for _ in range(20)]
        uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
        krr = krr_with_score(uc, yc, uq, yq)
        e_lib, _ = library_fb(lib.eval(), uc, yc, uq, yq, krr)
        lib.train()
        new = ObsBank([x_t], [dx_t], [h_t], dev)
        mem = ObsBank(mem_x, mem_dx, mem_h, dev) if mem_x else None
        run = 0.0
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
        mem_x.append(x_t[:MEMORY_INSTANCES]); mem_dx.append(dx_t[:MEMORY_INSTANCES]); mem_h.append(h_t)
        log["stages"].append({"t": t, "system": name, "encounter_libraryfb": float(e_lib.median()),
                              "encounter_krr": float(krr[0].median()), "train_min_nmse": run / steps})
        if t % 10 == 0 or t == len(order):
            print(args.arm, s, t, name, f"enc {float(e_lib.median()):.4f} krr {float(krr[0].median()):.4f}",
                  f"train {run / steps:.4f}", f"{time.time() - t0:.0f}s", flush=True)
    torch.save(lib.state_dict(), OUT / "learners" / f"{args.arm}_final_s{s}.pt")
    (OUT / "learners" / f"{args.arm}_s{s}.json").write_text(json.dumps(log))


def evaluate(args):
    dev = args.device
    names = S.eligible_split()["development"]
    xs, dxs = load(names, dev)
    libs = {}
    for s in SEEDS:
        lib = Library(4).to(dev)
        lib.load_state_dict(torch.load(TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", map_location=dev, weights_only=True))
        libs[f"catalogonly_s{s}"] = lib.eval()
        for arm, (_, _, m) in ARMS.items():
            lib = Library(m).to(dev)
            lib.load_state_dict(torch.load(OUT / "learners" / f"{arm}_final_s{s}.pt", map_location=dev, weights_only=True))
            libs[f"{arm}_s{s}"] = lib.eval()
    rows = []
    for si, name in enumerate(names):
        h = T8.spacing(name)
        for K in KS:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)  # Trial 9's development episodes
            eps = [episode_obs(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev, h) for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            with torch.no_grad():
                krr = krr_with_score(uc, yc, uq, yq)
                res = {"krr": krr[0]}
                for tag, lib in libs.items():
                    res[tag] = library_fb(lib, uc, yc, uq, yq, krr)[0]
            for mth, v in res.items():
                rows.append({"system": name, "K": K, "method": mth, "episodes": v.float().cpu().tolist()})
            print(name, K, {mth: round(float(np.median(v.float().cpu().numpy())), 4) for mth, v in res.items() if mth.endswith("s1") or mth == "krr"}, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows}))


def analyze(args):
    rows = json.loads((OUT / "eval_development.json").read_text())["rows"]
    sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
    names = S.eligible_split()["development"]
    npn, pn = sorted(NONPOLY), [n for n in names if n not in NONPOLY]
    med = lambda m, K, ns: float(np.median([sc[(n, K, m)] for n in ns]))
    primary = min(("broad", "broad8"), key=lambda a: med(f"{a}_s1", 32, npn))  # rule fixed in the pre-registration
    out = {"primary_arm": primary}
    for s in SEEDS:
        p = f"{primary}_s{s}"
        sa = {"vs_catalogonly": med(p, 32, npn) / med(f"catalogonly_s{s}", 32, npn),
              "vs_cataloglong": med(p, 32, npn) / med(f"cataloglong_s{s}", 32, npn),
              "wins_vs_krr": sum(sc[(n, 32, p)] < sc[(n, 32, "krr")] for n in npn)}
        sb = med(p, 32, pn) / med(f"catalogonly_s{s}", 32, pn)
        crit = {"S-A": bool(sa["vs_catalogonly"] <= 0.75 and sa["vs_cataloglong"] <= 0.75 and sa["wins_vs_krr"] >= 3),
                "S-B": bool(sb <= 1.5)}
        out[s] = {"S-A": sa, "S-B_ratio": sb, "criteria": crit,
                  "nonpoly_median": {m: {K: med(f"{m}_s{s}" if m != "krr" else m, K, npn) for K in KS}
                                     for m in ("catalogonly", "cataloglong", "broad", "broad8", "krr")},
                  "poly_median": {m: {K: med(f"{m}_s{s}" if m != "krr" else m, K, pn) for K in KS}
                                  for m in ("catalogonly", "cataloglong", "broad", "broad8", "krr")},
                  "nonpoly_per_system_K32": {n: {m: sc[(n, 32, f"{m}_s{s}" if m != "krr" else m)]
                                                 for m in ("catalogonly", "cataloglong", "broad", "broad8", "krr")} for n in npn}}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    print("primary arm (by seed-1 non-polynomial median):", primary)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--arm", choices=list(ARMS), default="broad")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "analyze": analyze}[a.cmd](a)
