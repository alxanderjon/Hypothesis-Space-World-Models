"""Trial 14: an investigation policy learned from history (meta-train systems), evaluated on Trial 13's episodes.
See docs/research-log.md.

  python trial14.py run --seed 1 --device cuda:0
  python trial14.py analyze
"""

import argparse
import json

import numpy as np
import torch
from torch import nn

import systems as S
import trial8 as T8
from analyze_trial1 import NONPOLY
from trial1 import SEEDS, load
from trial4 import Library, krr_with_score
from trial5 import library_fb
from trial9 import OUT as TRIAL9
from trial12 import feats, ridge_fit, ridge_select
from trial13 import KS, POOL, coverage, make_episode, standardize
from trial13 import OUT as TRIAL13

OUT = S.ROOT / "outputs/accumulating_learner/trial14"
N_DESC, EPISODES_PER_SYSTEM, CANDS, TRAIN_STEPS = 6, 40, 64, 3000
RESELECT = {2, 4, 8, 16}


class State:
    """Current design state for one episode: chosen set, standardization, selected basis, ridge fit."""

    def __init__(self, lib, xm, d, pool_u, dens):
        self.lib, self.xm, self.d, self.pool_u, self.dens = lib, xm, d, pool_u, dens
        self.basis = self.lam = self.loo = None

    @torch.no_grad()
    def update(self, chosen, reselect):
        xc, dc = self.xm[chosen], self.d[chosen]
        self.mx, self.sx, self.sd = xc.mean(0), xc.std(0) + 1e-8, dc.std(0) + 1e-8
        uc, yc = (xc - self.mx) / self.sx, dc / self.sd
        if reselect or self.basis is None:
            sels = [ridge_select(feats(b, uc), yc) for b in self.lib.bases]
            o = int(np.argmin([v for _, v in sels]))
            self.basis, self.lam, self.loo = self.lib.bases[o], sels[o][0], sels[o][1]
        else:
            self.loo = ridge_select(feats(self.basis, uc), yc)[1] if len(chosen) > 2 else self.loo
        self.Phi_pool = feats(self.basis, (self.xm - self.mx) / self.sx)
        Pc = self.Phi_pool[chosen]
        k, F = Pc.shape
        self.Ainv = torch.linalg.inv(Pc.T @ Pc + self.lam * k * torch.eye(F, device=Pc.device, dtype=Pc.dtype))
        self.b = Pc.T @ yc
        self.W = self.Ainv @ self.b
        self.chosen = list(chosen)

    @torch.no_grad()
    def descriptors(self):
        lev = ((self.Phi_pool @ self.Ainv) * self.Phi_pool).sum(1)
        dmin = torch.cdist(self.pool_u, self.pool_u[self.chosen]).min(1).values
        pred = (self.Phi_pool @ self.W).norm(dim=1)
        k = torch.full_like(lev, float(len(self.chosen)))
        loo = torch.full_like(lev, float(self.loo))
        return torch.stack([torch.log(lev + 1e-12), torch.log(dmin + 1e-6), torch.log(self.dens),
                            torch.log(pred + 1e-6), torch.log(k), torch.log(loo + 1e-12)], 1)


def density(pool_u):
    dd = torch.cdist(pool_u, pool_u)
    return dd.topk(11, largest=False).values[:, 1:].mean(1)


@torch.no_grad()
def make_training_data(lib, s, dev):
    meta = S.eligible_split()["meta_train"]
    xs, _ = load(meta, dev)
    gen = torch.Generator().manual_seed(14000 + s)
    X, Y = [], []
    for j, name in enumerate(meta):
        h = T8.spacing(name)
        for _ in range(EPISODES_PER_SYSTEM):
            inst = int(torch.randint(xs[j].shape[0], (1,), generator=gen))
            xm, d, xq, dq = make_episode(xs[j][inst], gen, dev, h)
            pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
            Kt = [8, 16, 32][int(torch.randint(3, (1,), generator=gen))]
            k = int(torch.randint(2, Kt, (1,), generator=gen))
            perm = torch.randperm(POOL, generator=gen).tolist()
            chosen, cands = perm[:k], perm[k:k + CANDS]
            st = State(lib, xm, d, pool_u, density(pool_u))
            st.update(chosen, reselect=True)
            D = st.descriptors()[cands]
            Pq = feats(st.basis, (xq - st.mx) / st.sx)
            yq = dq / st.sd
            mse_before = ((Pq @ st.W - yq) ** 2).mean()
            phi = st.Phi_pool[cands]                                    # (C,F)
            yc = (d[cands] / st.sd)                                     # (C,3)
            Av = phi @ st.Ainv                                          # (C,F)
            denom = 1 + (Av * phi).sum(1, keepdim=True)
            b_new = st.b.unsqueeze(0) + phi.unsqueeze(2) * yc.unsqueeze(1)   # (C,F,3)
            W_new = torch.einsum("fg,cgo->cfo", st.Ainv, b_new) - torch.einsum("cf,cg,cgo->cfo", Av, Av, b_new) / denom.unsqueeze(2)
            mse_after = ((torch.einsum("qf,cfo->cqo", Pq, W_new) - yq.unsqueeze(0)) ** 2).mean((1, 2))
            X.append(D); Y.append(torch.log(mse_before / mse_after))
    return torch.cat(X).float(), torch.cat(Y).float()


def train_policy(X, Y, cols, dev, seed):
    torch.manual_seed(seed)
    Xc = X[:, cols]
    mu, sdv = Xc.mean(0), Xc.std(0) + 1e-8
    ym, ys = Y.mean(), Y.std() + 1e-8
    net = nn.Sequential(nn.Linear(len(cols), 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(TRAIN_STEPS):
        i = torch.randint(len(X), (1024,), generator=g).to(dev)
        loss = ((net((Xc[i] - mu) / sdv).squeeze(1) - (Y[i] - ym) / ys) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda D: net(((D[:, cols].float() - mu) / sdv)).squeeze(1)


@torch.no_grad()
def policy_select(lib, policy, xm, d, pool_u, dens, rand_idx, K):
    chosen = list(rand_idx[:2])
    st = State(lib, xm, d, pool_u, dens)
    st.update(chosen, reselect=True)
    while len(chosen) < K:
        sc = policy(st.descriptors())
        sc[chosen] = -float("inf")
        chosen.append(int(sc.argmax()))
        st.update(chosen, reselect=len(chosen) in RESELECT)
    return chosen


def run(args):
    dev, s = args.device, args.seed
    lib = Library().to(dev)
    lib.load_state_dict(torch.load(TRIAL9 / "learners" / f"seqobs400_N51_s{s}.pt", map_location=dev, weights_only=True))
    lib.eval()
    X, Y = make_training_data(lib, s, dev)
    print("training examples", len(X), "label mean/std", float(Y.mean()), float(Y.std()), flush=True)
    policies = {"learned": train_policy(X, Y, list(range(N_DESC)), dev, 100 + s),
                "learned_distonly": train_policy(X, Y, [1], dev, 200 + s)}
    names = S.eligible_split()["development"]
    xs, _ = load(names, dev)
    rows = []
    for si, name in enumerate(names):
        h = T8.spacing(name)
        for K in KS:
            gen = torch.Generator().manual_seed(13000 + 97 * si + K)  # identical to Trial 13
            batches = {k: [] for k in ("coverage", *policies)}
            for i in range(xs[si].shape[0]):
                for _ in range(5):
                    xm, d, xq, dq = make_episode(xs[si][i], gen, dev, h)
                    pool_u = (xm - xm.mean(0)) / (xm.std(0) + 1e-8)
                    rand_idx = torch.randperm(POOL, generator=gen)[:K].tolist()
                    dens = density(pool_u)
                    idx = {"coverage": coverage(pool_u, rand_idx[0], K)}
                    for pname, pol in policies.items():
                        idx[pname] = policy_select(lib, pol, xm, d, pool_u, dens, rand_idx, K)
                    for st in batches:
                        batches[st].append(standardize(xm, d, idx[st], xq, dq))
            with torch.no_grad():
                for st in batches:
                    uc, yc, uq, yq = (torch.stack(v).float() for v in zip(*batches[st]))
                    e, _ = library_fb(lib, uc, yc, uq, yq, krr_with_score(uc, yc, uq, yq))
                    rows.append({"system": name, "K": K, "method": f"learner_{st}", "episodes": e.float().cpu().tolist()})
            print(s, name, K, {r["method"]: round(float(np.median(r["episodes"])), 5) for r in rows[-3:]}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_s{s}.json").write_text(json.dumps(rows))


def analyze(args):
    out = {}
    for s in SEEDS:
        rows = json.loads((OUT / f"rows_s{s}.json").read_text())
        t13 = [r for r in json.loads((TRIAL13 / f"rows_s{s}.json").read_text()) if "episodes" in r]
        sc = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in rows}
        sc13 = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t13}
        names = sorted({r["system"] for r in rows})
        med = lambda tbl, m, K, ns=names: float(np.median([tbl[(n, K, m)] for n in ns]))
        cov_check = {K: (med(sc, "learner_coverage", K), med(sc13, "learner_coverage", K)) for K in KS}
        r = {K: med(sc, "learner_learned", K) / med(sc, "learner_coverage", K) for K in KS}
        crit = {"L1": bool(r[16] <= 0.83), "L2": bool(r[8] <= 1.0 and r[32] <= 1.0)}
        out[s] = {"ratio_learned_vs_coverage": r, "criteria": crit, "coverage_reproduces_trial13": cov_check,
                  "aggregates": {m: {K: med(sc, m, K) for K in KS} for m in ("learner_coverage", "learner_learned", "learner_learned_distonly")},
                  "experience_trial13": {K: med(sc13, "learner_experience", K) for K in KS},
                  "nonpoly": {m: {K: med(sc, m, K, sorted(NONPOLY)) for K in KS} for m in ("learner_coverage", "learner_learned")},
                  "wins_learned_vs_coverage_K16": sum(sc[(n, 16, "learner_learned")] < sc[(n, 16, "learner_coverage")] for n in names)}
        print(f"seed {s}:", json.dumps(out[s], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
