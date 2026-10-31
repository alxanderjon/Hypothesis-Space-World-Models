"""Trial 26 "influence guide": a small network trained on the learner's OWN solutions of past systems.

From a system's observation bursts it predicts, for every component i of the vector field:
  - which variables j the equation for dz_i/dt involves (influence),
  - how many terms the equation has, and
  - the highest grammar level among its terms.
Labels come only from models the learner itself recovered and accepted (internal validation), never from the true
equations. Inputs are only the bursts the learner observed for those systems (random subsets of them).
The network is permutation-equivariant in the variables: pair (i, j) features keep i and j explicit and pool the rest.
Also here: the no-history statistical influence screen used as a control.
"""

import math

import numpy as np
import torch
import torch.nn as nn

FD9 = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
MAX_TERMS = 12


def burst_features(bursts):
    """(M, L, d) raw states -> (M, d, F) per-burst, per-variable features in scaled units."""
    b = np.asarray(bursts, dtype=np.float64)
    M, L, d = b.shape
    flat = b.reshape(-1, d)
    z = (b - flat.mean(0)) / (flat.std(0) + 1e-300)
    der = np.stack([sum(FD9[k] * z[:, c - 4 + k] for k in range(9)) for c in range(4, L - 4)], 1)   # (M, L-8, d)
    dsc = np.abs(der).reshape(-1, d).max(0) + 1e-300
    feats = np.concatenate([z.transpose(0, 2, 1), (der / dsc).transpose(0, 2, 1)], axis=2)        # (M, d, L + L-8)
    return torch.tensor(feats, dtype=torch.float32)


class Guide(nn.Module):
    def __init__(self, F, h=64):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(3 * F + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.pair_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.comp = nn.Sequential(nn.Linear(2 * F, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.count_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, MAX_TERMS))
        self.level_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 3))

    def forward(self, feats):
        """feats: (M, d, F). Returns influence logits (d, d), count logits (d, MAX_TERMS), level logits (d, 3)."""
        M, d, F = feats.shape
        tot = feats.sum(1)                                                       # (M, F)
        pair_in = []
        for i in range(d):
            for j in range(d):
                rest = (tot - feats[:, i] - (feats[:, j] if j != i else 0)) / max(d - (1 if i == j else 2), 1)
                flag = torch.full((M, 1), float(i == j))
                pair_in.append(torch.cat([feats[:, i], feats[:, j], rest, flag], 1))
        P = self.pair(torch.stack(pair_in, 0)).mean(1)                          # (d*d, h) pooled over bursts
        infl = self.pair_head(P).reshape(d, d)
        comp_in = torch.stack([torch.cat([feats[:, i], (tot - feats[:, i]) / max(d - 1, 1)], 1) for i in range(d)], 0)
        C = self.comp(comp_in).mean(1)                                          # (d, h)
        Pm = P.reshape(d, d, -1).mean(1)                                        # (d, h) pair summary per component
        H = torch.cat([C, Pm], 1)
        return infl, self.count_head(H), self.level_head(H)


def labels_from_description(desc, d):
    """Influence (d, d), term count (d,), max level (d,) of a recovered model description."""
    infl = np.zeros((d, d), dtype=np.float32)
    count = np.zeros(d, dtype=np.int64)
    level = np.zeros(d, dtype=np.int64)
    for i, comp in enumerate(desc):
        for t in comp:
            for k, e in enumerate(t["mono"]):
                if e:
                    infl[i, k] = 1
            for f in t["factors"]:
                infl[i, f[1]] = 1
        count[i] = min(len(comp), MAX_TERMS)
        level[i] = max([t["level"] for t in comp] or [1])
    return infl, count, level


def train_guide(examples, steps=3000, seed=0, scramble=False, lr=2e-3, verbose=False):
    """examples: list of (bursts (64, L, d) observed for a past system, recovered model description)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [(np.asarray(b), labels_from_description(desc, np.asarray(b).shape[-1])) for b, desc in examples]
    if scramble and len(data) > 1:                     # same inputs, same label sets, paired with the wrong systems
        perm = np.arange(len(data))
        while np.any(perm == np.arange(len(data))):
            perm = rng.permutation(len(data))
        data = [(data[k][0], data[perm[k]][1]) for k in range(len(data))]
    F = burst_features(data[0][0][:2]).shape[-1]
    net = Guide(F)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    for step in range(steps):
        loss = 0.0
        for _ in range(8):                             # mini-batch of 8 systems
            b, (infl, count, level) = data[rng.integers(len(data))]
            M = b.shape[0]
            m = int(round(2 ** rng.uniform(1, math.log2(M))))
            sub = b[rng.choice(M, m, replace=False)]
            d = b.shape[-1]
            pm = rng.permutation(d)                    # relabel variables
            sub = sub[..., pm]
            infl_p, count_p, level_p = infl[np.ix_(pm, pm)], count[pm], level[pm]
            li, lc, ll = net(burst_features(sub))
            loss = loss + bce(li, torch.tensor(infl_p)) + 0.5 * ce(lc, torch.tensor(count_p - 1)) + 0.5 * ce(ll, torch.tensor(level_p - 1))
        opt.zero_grad()
        (loss / 8).backward()
        opt.step()
        sched.step()
        if verbose and step % 500 == 0:
            print(f"  guide step {step} loss {float(loss) / 8:.3f}", flush=True)
    net.eval()
    return net


def predict(net, bursts, n_perm=6):
    """Averaged over variable relabelings (test-time equivariance). Returns influence probs, counts, levels."""
    b = np.asarray(bursts)
    d = b.shape[-1]
    rng = np.random.default_rng(0)
    perms = [np.arange(d)] + [rng.permutation(d) for _ in range(n_perm - 1)]
    P = np.zeros((d, d)); C = np.zeros((d, MAX_TERMS)); Lv = np.zeros((d, 3))
    with torch.no_grad():
        for pm in perms:
            li, lc, ll = net(burst_features(b[..., pm]))
            inv = np.argsort(pm)
            P += torch.sigmoid(li).numpy()[np.ix_(inv, inv)]
            C += torch.softmax(lc, 1).numpy()[inv]
            Lv += torch.softmax(ll, 1).numpy()[inv]
    P /= len(perms); C /= len(perms); Lv /= len(perms)
    count = np.array([int(np.argmax(C[i])) + 1 for i in range(d)])
    level = np.array([int(np.argmax(Lv[i])) + 1 for i in range(d)])
    return P, count, level


def save_guide(net, path):
    torch.save(net.state_dict(), path)


def load_guide(path, F):
    net = Guide(F)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


# ---------------- the no-history statistical influence screen (control) ----------------

def statistical_influence(bursts, val_frac=0.25, seed=0, ratio=1.2):
    """Kernel ridge regression of each derivative component on the states, leave-one-out error with all variables
    versus without variable j. j counts as an influence if dropping it raises the LOO error by >= `ratio`.
    Uses only the new system's training bursts (same split as the learner)."""
    b = np.asarray(bursts, dtype=np.float64)
    M, L, d = b.shape
    rng = np.random.default_rng(seed)
    nval = max(1, int(round(val_frac * M)))
    perm = rng.permutation(M)
    tr = b[perm[nval:]] if M > 1 else b
    flat = b.reshape(-1, d)
    z = (tr - flat.mean(0)) / (flat.std(0) + 1e-300)
    X = np.stack([z[:, c] for c in range(4, L - 4)], 1).reshape(-1, d)
    Y = np.stack([sum(FD9[k] * z[:, c - 4 + k] for k in range(9)) for c in range(4, L - 4)], 1).reshape(-1, d)

    def loo(Xs):
        D2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        med = np.median(D2[D2 > 0]) if np.any(D2 > 0) else 1.0
        K = np.exp(-D2 / med)
        Hm = K @ np.linalg.inv(K + 1e-8 * np.eye(len(K)))
        R = (Y - Hm @ Y) / np.clip(1 - np.diag(Hm), 1e-12, None)[:, None]
        return (R ** 2).mean(0)

    full = loo(X)
    P = np.zeros((d, d))
    for j in range(d):
        keep = [k for k in range(d) if k != j]
        drop = loo(X[:, keep]) if keep else np.full(d, np.inf)
        P[:, j] = (drop >= ratio * full).astype(float)
    return P
