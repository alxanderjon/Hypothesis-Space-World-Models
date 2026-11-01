# PRACTICE half only: export, for every (system, M), exactly the data our learner sees (same bursts, same train/held-back
# split, same scaled coordinates and rate estimates), for the PySR benchmark.
import json, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np

warnings.filterwarnings("ignore")
SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
OUT = f"{SP}/bench/data"
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")


def export(name):
    import torch
    import harness27 as H27, discovery_learner as DL
    assert H27.split_half(name) == "dev"
    todo = [M for M in (8, 16, 32, 64) if not os.path.exists(f"{OUT}/{name}_M{M}.npz")]
    if not todo:
        return f"{name}: done"
    try:
        ref = H27.Reference27(name, seed=1)
    except Exception as e:
        return f"{name}: reference failed ({str(e)[:60]})"
    for M in todo:
        export_one(ref, name, M)
    return f"{name}: exported {todo}"


def export_one(ref, name, M):
    import torch
    import harness27 as H27, discovery_learner as DL
    out = f"{OUT}/{name}_M{M}.npz"
    Bt = torch.as_tensor(np.asarray(H27.bursts_from(ref, 1, M)))
    Mb, L, d = Bt.shape
    rng = np.random.default_rng(1)                       # learn(..., seed=1, val_frac=0.25): identical split
    nval = max(1, int(round(0.25 * Mb)))
    perm = rng.permutation(Mb)
    val_idx, tr_idx = perm[:nval], (perm[nval:] if Mb > 1 else perm)
    allx = Bt.reshape(-1, d)
    mu, sig = allx.mean(0), allx.std(0) + 1e-300
    mu = torch.where(mu.abs() > DL.CENTER_RATIO * sig, mu, torch.zeros_like(mu))
    Ztr, Zva = (Bt[tr_idx] - mu) / sig, (Bt[val_idx] - mu) / sig
    G, noise = DL.deriv_noise(Ztr)
    Zc = torch.stack([Ztr[:, c] for c in range(4, L - 4)], 1).reshape(-1, d)
    np.savez(out, Zc=Zc.numpy(), G=G.numpy(), noise=noise.numpy(), mu=mu.numpy(), sig=sig.numpy(), dt=ref.sys.dt,
             z0=Ztr[:, :-1].reshape(-1, d).numpy(), z1=Ztr[:, 1:].reshape(-1, d).numpy(),
             zv0=Zva[:, :-1].reshape(-1, d).numpy(), zv1=Zva[:, 1:].reshape(-1, d).numpy(),
             bursts=Bt.numpy(), tr_idx=tr_idx, val_idx=val_idx)
    return out


if __name__ == "__main__":
    import harness27 as H27
    os.makedirs(OUT, exist_ok=True)
    names = sorted(n for n in H27.pool_records() if H27.split_half(n) == "dev")
    jobs = names
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for r in ex.map(export, jobs):
            print(r, flush=True)
