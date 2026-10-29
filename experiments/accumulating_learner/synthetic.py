"""Procedurally generated non-polynomial 3-D dynamical systems for Trial 15 (generic recipe; see docs/research-log.md).

f(x) = -Gamma x + sum_k a_k sigma_k(u_k . x + b_k), sigma_k in {sin, tanh, z/(1+z^2), exp(-z^2), cos}.
Accepted systems are stored like the catalog bank: syn###.npz with x, dx of shape (8, 4000, 3), plus synthetic.json with
each system's parameters, characteristic period P and sample spacing h = 100 P / 3999.

  python synthetic.py --n 200 --workers 16
"""

import argparse
import json
import signal
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

import systems as S

OUT = S.ROOT / "outputs/accumulating_learner/trial15/synthetic"
SIGMAS = {
    "sin": np.sin,
    "tanh": np.tanh,
    "rational": lambda z: z / (1 + z**2),
    "bump": lambda z: np.exp(-z**2),
    "cos": np.cos,
}
N_POINTS, BURN, SPAN, INSTANCES = 4000, 50, 100, 8


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def sample_params(rng):
    m = int(rng.integers(3, 9))
    s = rng.uniform(1, 4)
    return {
        "gamma": rng.uniform(0.1, 1.0, 3).tolist(),
        "u": (rng.normal(size=(m, 3)) / np.sqrt(3)).tolist(),
        "b": rng.uniform(-np.pi, np.pi, m).tolist(),
        "a": (rng.normal(size=(m, 3)) * s).tolist(),
        "sigma": [str(k) for k in rng.choice(list(SIGMAS), m)],
    }


def make_rhs(p):
    g, u, b, a = (np.asarray(p[k]) for k in ("gamma", "u", "b", "a"))
    fs = [SIGMAS[k] for k in p["sigma"]]

    def f(x):
        x = np.asarray(x, dtype=float)
        z = x @ u.T + b                                    # (..., m)
        act = np.stack([fs[k](z[..., k]) for k in range(len(fs))], -1)
        return -g * x + act @ a
    return f


def integrate(f, x0, t_eval, seconds=60):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        sol = solve_ivp(lambda t, x: f(x), (0, t_eval[-1]), x0, method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    except (Timeout, FloatingPointError, ValueError):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or sol.y.shape[1] != len(t_eval) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


def accept(x, dx):
    a, b = slice(0, len(x) // 2), slice(len(x) // 2, None)
    scale = np.abs(x).mean(0) + 1e-3
    if np.any(x[b].std(0) < 1e-3 * scale):
        return False
    rms = lambda d: np.sqrt((d**2).mean())
    return rms(dx[b]) >= 0.1 * rms(dx[a])


def try_system(seed):
    np.seterr(all="ignore")
    rng = np.random.default_rng([15, seed])
    p = sample_params(rng)
    f = make_rhs(p)
    pilot = integrate(f, rng.normal(size=3), np.linspace(0, 200, 4001))
    if pilot is None:
        return None
    xp = pilot[2000:]
    speed = np.linalg.norm(f(xp), axis=1).mean()
    radius = np.linalg.norm(xp - xp.mean(0), axis=1).mean()
    if speed < 1e-6 or radius < 1e-6:
        return None
    P = 2 * np.pi * radius / speed
    if not (0.05 < P < 200):
        return None
    t_eval = np.linspace(BURN * P, (BURN + SPAN) * P, N_POINTS)
    xs, dxs = [], []
    for _ in range(INSTANCES):
        x0 = xp[int(rng.integers(len(xp)))] + 0.1 * rng.normal(size=3) * (xp.std(0) + 1e-3)
        x = integrate(f, x0, t_eval)
        if x is None:
            return None
        dx = f(x)
        if not accept(x, dx):
            return None
        xs.append(x.astype(np.float32)); dxs.append(dx.astype(np.float32))
    p.update({"seed": seed, "period": float(P), "h": float(SPAN * P / (N_POINTS - 1))})
    return p, np.stack(xs), np.stack(dxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--probe", type=int, default=0, help="only report acceptance over this many seeds; write nothing")
    a = ap.parse_args()
    if a.probe:
        with Pool(a.workers) as pool:
            res = pool.map(try_system, range(a.probe))
        ok = [r[0] for r in res if r is not None]
        print(f"accepted {len(ok)}/{a.probe}; periods", np.round(sorted(r["period"] for r in ok), 2).tolist()[:40])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta, seed = {}, 0
    with Pool(a.workers) as pool:
        while len(meta) < a.n:
            for r in pool.map(try_system, range(seed, seed + 4 * a.workers)):
                if r is not None and len(meta) < a.n:
                    p, x, dx = r
                    name = f"syn{len(meta):03d}"
                    np.savez(OUT / f"{name}.npz", x=x, dx=dx)
                    meta[name] = p
            seed += 4 * a.workers
            print("accepted", len(meta), "after seeds", seed, flush=True)
    (OUT / "synthetic.json").write_text(json.dumps({"systems": meta, "seeds_tried": seed}, indent=1))


if __name__ == "__main__":
    main()
