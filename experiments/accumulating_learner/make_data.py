"""Trial 1 data bank: on-attractor states and exact vector-field values for meta-train and development systems.

Per system, instance 0 uses published parameters; other instances multiply each parameter by
exp(U(-0.1, 0.1)). Each instance: integrate from the published initial condition for 20 characteristic
periods (discarded), then 100 periods sampled at 4000 equally spaced times. Segment A (first half) supplies
context states, segment B (second half) supplies query states. An instance is rejected and redrawn (max 4
draws) if integration fails, times out, diverges, or collapses (a coordinate's std in segment B below 1e-6 of
its scale, or derivative RMS in B below 1e-6 of A's). Confirmation systems are never loaded.
"""

import argparse
import json
import signal
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import systems as S

N_POINTS, BURN, SPAN, PERTURB = 4000, 20, 100, 0.1


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def integrate(sys_, period, seconds=60):
    ic = np.asarray(sys_.meta["initial_conditions"], dtype=float)
    t_eval = np.linspace(BURN * period, (BURN + SPAN) * period, N_POINTS)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        sol = solve_ivp(lambda t, x: sys_(x), (0, t_eval[-1]), ic, method="LSODA",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
    except (Timeout, FloatingPointError, ValueError) as e:
        return None, f"integration error {type(e).__name__}"
    finally:
        signal.alarm(0)
    if not sol.success or sol.y.shape[1] != N_POINTS:
        return None, f"solver: {sol.message}"
    x = sol.y.T
    if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e6:
        return None, "diverged"
    dx = sys_(x)
    a, b = slice(0, N_POINTS // 2), slice(N_POINTS // 2, None)
    scale = np.abs(x).mean(0) + 1e-3
    if np.any(x[b].std(0) < 1e-6 * scale):
        return None, "collapsed coordinate"
    rms = lambda d: np.sqrt((d**2).mean())
    if rms(dx[b]) < 1e-6 * rms(dx[a]):
        return None, "converged to fixed point"
    return (x.astype(np.float32), dx.astype(np.float32)), "ok"


def make_system(args):
    name, n_inst, seed, allow_confirmation = args
    np.seterr(all="ignore")
    meta = S.METADATA[name]
    period = float(meta["period"])
    rng = np.random.default_rng([seed, int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little")])
    xs, dxs, params, log = [], [], [], []
    for i in range(n_inst):
        for draw in range(4 if i else 1):
            p = dict(meta["parameters"])
            if i:
                p = {k: (np.asarray(v) * np.exp(rng.uniform(-PERTURB, PERTURB))).tolist() for k, v in p.items()}
            t0 = time.time()
            out, msg = integrate(S.System(name, p, allow_confirmation), period)
            log.append({"instance": i, "draw": draw, "status": msg, "seconds": round(time.time() - t0, 2)})
            if out is not None:
                xs.append(out[0]); dxs.append(out[1]); params.append(p)
                break
        if i == 0 and not xs:
            break
    return name, xs, dxs, params, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S.ROOT / "outputs/accumulating_learner/trial1/data"))
    ap.add_argument("--meta-instances", type=int, default=32)
    ap.add_argument("--dev-instances", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--confirmation", action="store_true",
                    help="generate ONLY the reserved confirmation systems (owner-approved one-shot run)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = S.eligible_split()
    if a.confirmation:
        jobs = [(n, a.dev_instances, a.seed, True) for n in split["confirmation"]]
    else:
        jobs = [(n, a.meta_instances, a.seed, False) for n in split["meta_train"]]
        jobs += [(n, a.dev_instances, a.seed, False) for n in split["development"]]
    report = {}
    with Pool(a.workers) as pool:
        for name, xs, dxs, params, log in pool.imap_unordered(make_system, jobs):
            report[name] = {"split": S.split_of(name), "kept_instances": len(xs), "log": log}
            if xs:
                np.savez(out / f"{name}.npz", x=np.stack(xs), dx=np.stack(dxs))
                (out / f"{name}.params.json").write_text(json.dumps(params))
            print(name, S.split_of(name), len(xs), "instances", flush=True)
    (out / "generation_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
