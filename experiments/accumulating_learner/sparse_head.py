"""Sparse per-system head: adaptive (iteratively reweighted) ridge, an approximation to an L0/L1 penalty.

W0 = ridge(P, Y, lam); then R times: w_fc = 1 / (W_fc^2 + eps_c), eps_c = EPS_REL * mean_f W0_fc^2, and each
output column c is re-solved with penalty lam * K * diag(w_:c). Leave-one-out uses the final weighted hat matrix
(weights treated as fixed), so selection still uses only the K context observations.
"""

import torch

from trial1 import LAMBDAS, nmse

ITERS, EPS_REL = 5, 1e-2


def sparse_fit(P, Y, lam, iters=ITERS):
    """P (B,K,F), Y (B,K,3), lam scalar tensor or float -> W (B,F,3), A (B,3,F,F) final per-output normal matrices."""
    B, K, F = P.shape
    G, b = P.transpose(1, 2) @ P, P.transpose(1, 2) @ Y
    eye = torch.eye(F, device=P.device, dtype=P.dtype)
    W = torch.linalg.solve(G + lam * K * eye, b)
    eps = EPS_REL * (W**2).mean(1, keepdim=True) + 1e-12
    A = (G + lam * K * eye).unsqueeze(1).expand(B, 3, F, F)
    for _ in range(iters):
        w = 1.0 / (W**2 + eps)
        A = G.unsqueeze(1) + lam * K * torch.diag_embed(w.transpose(1, 2))
        W = torch.linalg.solve(A, b.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
    return W, A


def sparse_loo_with_score(Pc, yc, Pq, yq, return_w=False):
    """Lambda chosen per episode by approximate LOO on the context; returns (query NMSE, LOO MSE[, W])."""
    Pc, yc, Pq, yq = (t.double() for t in (Pc, yc, Pq, yq))
    best_loo = best_err = best_w = None
    for lam in LAMBDAS.tolist():
        W, A = sparse_fit(Pc, yc, lam)
        H = (Pc.unsqueeze(1) @ torch.linalg.solve(A, Pc.transpose(1, 2).unsqueeze(1))).diagonal(dim1=2, dim2=3)  # (B,3,K)
        res = (yc - Pc @ W) / (1 - H.transpose(1, 2)).clamp_min(1e-9)
        loo = (res**2).mean((1, 2))
        loo = torch.where(torch.isfinite(loo), loo, torch.full_like(loo, float("inf")))
        err = nmse(Pq @ W, yq)
        if best_loo is None:
            best_loo, best_err, best_w = loo, err, W
        else:
            m = loo < best_loo
            best_loo, best_err = torch.where(m, loo, best_loo), torch.where(m, err, best_err)
            best_w = torch.where(m.view(-1, 1, 1), W, best_w)
    return (best_err, best_loo, best_w) if return_w else (best_err, best_loo)
