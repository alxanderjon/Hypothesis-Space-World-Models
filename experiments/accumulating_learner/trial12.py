"""Trial 12 (evaluation only): encounter-time fine-tuning of the selected basis, with held-out validation.
See docs/research-log.md for the pre-registered protocol.

  python trial12.py run --seed 1 --device cuda:0
  python trial12.py analyze
"""

import argparse
import copy
import json

import numpy as np
import torch

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import LAMBDAS, SEEDS, load, nmse
from trial4 import Library
from trial9 import OUT as TRIAL9
from trial9 import episode_obs

OUT = S.ROOT / "outputs/accumulating_learner/trial12"
KS, STEPS, LR_FT, FIT_FRAC = (32, 64), 100, 3e-4, 0.75
ELLS = [0.25, 0.5, 1.0, 2.0, 4.0]


def ridge_loo_terms(P, y, lam):
    """Closed-form ridge LOO residuals (differentiable). P (K,F), y (K,3); penalty lam*K."""
    K, F = P.shape
    A = P.T @ P + lam * K * torch.eye(F, device=P.device, dtype=P.dtype)
    W = torch.linalg.solve(A, P.T @ y)
    H = (P * torch.linalg.solve(A, P.T).T).sum(1)
    return (y - P @ W) / (1 - H).clamp_min(1e-9).unsqueeze(-1), W


def ridge_select(P, y):
    """Exact-LOO lambda selection for one episode; returns (lam, loo_mse)."""
    best = (None, float("inf"))
    for lam in LAMBDAS.tolist():
        r, _ = ridge_loo_terms(P, y, lam)
        v = float((r**2).mean())
        if np.isfinite(v) and v < best[1]:
            best = (lam, v)
    return best


def ridge_fit(P, y, lam):
    K, F = P.shape
    return torch.linalg.solve(P.T @ P + lam * K * torch.eye(F, device=P.device, dtype=P.dtype), P.T @ y)


def krr_select(u, y):
    best = (None, None, float("inf"))
    d = torch.cdist(u, u) ** 2
    K = u.shape[0]
    for ell in ELLS:
        Kc = torch.exp(-d / (2 * ell**2))
        for lam in LAMBDAS.tolist():
            A = Kc + lam * K * torch.eye(K, device=u.device, dtype=u.dtype)
            Ainv = torch.linalg.inv(A)
            r = (y - Kc @ (Ainv @ y)) / (1 - (Kc @ Ainv).diagonal()).clamp_min(1e-9).unsqueeze(-1)
            v = float((r**2).mean())
            if np.isfinite(v) and v < best[2]:
                best = (ell, lam, v)
    return best


def krr_predict(uc, yc, uq, ell, lam):
    K = uc.shape[0]
    Kc = torch.exp(-torch.cdist(uc, uc) ** 2 / (2 * ell**2))
    alpha = torch.linalg.solve(Kc + lam * K * torch.eye(K, device=uc.device, dtype=uc.dtype), yc)
    return torch.exp(-torch.cdist(uq, uc) ** 2 / (2 * ell**2)) @ alpha


def feats(basis, u):
    return basis(u.float()).double()


def finetune(basis, u_fit, y_fit, lam):
    b = copy.deepcopy(basis).train()
    opt = torch.optim.Adam(b.parameters(), lr=LR_FT)
    for _ in range(STEPS):
        r, _ = ridge_loo_terms(feats(b, u_fit), y_fit, lam)
        loss = (r**2).mean()
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return b.eval()


def val_mse_basis(b, u_fit, y_fit, u_val, y_val, lam):
    with torch.no_grad():
        W = ridge_fit(feats(b, u_fit), y_fit, lam)
        return float(((feats(b, u_val) @ W - y_val) ** 2).mean())


def run(args):
    dev, s = args.device, args.seed
    names = S.eligible_split()["development"]
    xs, dxs = load(names, dev)
    lib = Library().to(dev)
    lib.load_state_dict(torch.load(TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    rows = []
    for si, name in enumerate(names):
        h = T8.spacing(name)
        for K in KS:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode_obs(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev, h) for i in range(xs[si].shape[0]) for _ in range(5)]
            base, ft, chose = [], [], []
            for e_i, (uc, yc, uq, yq) in enumerate(eps):
                uc, yc, uq, yq = (t.double() for t in (uc, yc, uq, yq))
                with torch.no_grad():
                    sels = [ridge_select(feats(b, uc), yc) for b in lib.bases]
                    kr = krr_select(uc, yc)
                loos = [v for _, v in sels] + [kr[2]]
                o = int(np.argmin(loos))
                with torch.no_grad():
                    if o < len(lib.bases):
                        W = ridge_fit(feats(lib.bases[o], uc), yc, sels[o][0])
                        base_err = float(nmse((feats(lib.bases[o], uq) @ W)[None], yq[None]))
                    else:
                        base_err = float(nmse(krr_predict(uc, yc, uq, kr[0], kr[1])[None], yq[None]))
                base.append(base_err)
                # held-out split of the context, seeded per episode
                pg = torch.Generator().manual_seed(11000 + 97 * si + K + 1000 * e_i)
                perm = torch.randperm(K, generator=pg).to(dev)
                nf = int(FIT_FRAC * K)
                fi, vi = perm[:nf], perm[nf:]
                bi = o if o < len(lib.bases) else int(np.argmin(loos[:-1]))
                lam = sels[bi][0]
                b_ft = finetune(lib.bases[bi], uc[fi], yc[fi], lam)
                v_ft = val_mse_basis(b_ft, uc[fi], yc[fi], uc[vi], yc[vi], lam)
                if o < len(lib.bases):
                    v_alt = val_mse_basis(lib.bases[bi], uc[fi], yc[fi], uc[vi], yc[vi], lam)
                else:
                    with torch.no_grad():
                        v_alt = float(((krr_predict(uc[fi], yc[fi], uc[vi], kr[0], kr[1]) - yc[vi]) ** 2).mean())
                use_ft = v_ft < v_alt
                if use_ft:
                    with torch.no_grad():
                        W = ridge_fit(feats(b_ft, uc), yc, lam)
                        err = float(nmse((feats(b_ft, uq) @ W)[None], yq[None]))
                else:
                    err = base_err
                ft.append(err); chose.append(bool(use_ft))
            rows.append({"system": name, "K": K, "seed": s, "libfb": base, "finetune": ft, "chose_finetuned": chose})
            print(s, name, K, f"libfb {np.median(base):.5f} finetune {np.median(ft):.5f} chose {np.mean(chose):.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps(rows))


def analyze(args):
    t9 = json.loads((TRIAL9 / "eval_development.json").read_text())["rows"]
    krr = {(r["system"], r["K"]): float(np.median(r["episodes"])) for r in t9 if r["method"] == "krr"}
    out = {}
    for s in SEEDS:
        rows = json.loads((OUT / f"rows_s{s}.json").read_text())
        sc = {(r["system"], r["K"], m): float(np.median(r[m])) for r in rows for m in ("libfb", "finetune")}
        names = sorted({r["system"] for r in rows})
        med = lambda m, K, ns=names: float(np.median([sc[(n, K, m)] for n in ns]))
        np_names = sorted(NONPOLY)
        a1 = {"ratio_nonpoly_K64": med("finetune", 64, np_names) / med("libfb", 64, np_names),
              "wins_vs_krr_nonpoly_K64": sum(sc[(n, 64, "finetune")] < krr[(n, 64)] for n in np_names)}
        a2 = {K: med("finetune", K) / med("libfb", K) for K in KS}
        crit = {"A1": bool(a1["ratio_nonpoly_K64"] <= 0.8 and a1["wins_vs_krr_nonpoly_K64"] >= 3),
                "A2": bool(all(v <= 1.1 for v in a2.values()))}
        out[s] = {"A1": a1, "A2_ratios": a2, "ratio_nonpoly_K32": med("finetune", 32, np_names) / med("libfb", 32, np_names),
                  "aggregates": {m: {K: med(m, K) for K in KS} for m in ("libfb", "finetune")},
                  "nonpoly_per_system_K64": {n: {"finetune": sc[(n, 64, "finetune")], "libfb": sc[(n, 64, "libfb")], "krr": krr[(n, 64)]} for n in np_names},
                  "chose_finetuned_rate": float(np.mean([c for r in rows for c in r["chose_finetuned"]])),
                  "criteria": crit}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
