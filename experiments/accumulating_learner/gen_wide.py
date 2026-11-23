"""Generator for the restored definition of solved (owner, 23 November): (1) systems must stay bounded: a system is
rejected if its true trajectories blow up within 10 of its time constants from any of its 20 test starting states (10
inside the recorded range, 10 at 1.5-2 times its half-width outside it, within the valid range); (3) recordings from a
wider spread of starting states within the valid range: each variable's start is the initial state times U(SPREAD), so
its sign is kept (before: U(0.5, 1.5)); the valid range keeps each variable's sign where the recordings never change it.
build_wide mirrors domains_g1.build exactly apart from the reference's starting states and the bounded rule; the test
starts and the horizon are stored with the system. Usage (pilot): gen_wide.py <workers> <n per shape>"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPREAD_BY_D = {2: (0.2, 3.0), 3: (0.2, 3.0), 4: (0.3, 2.0)}     # owner, 23 Nov: narrower at 4 variables
MAX_DRAWS = 1000                                                 # owner, 23 Nov: raised so rungs still get their systems
N_IN, N_OUT, HORIZON, MAX_CALLS = 10, 10, 10.0, 200_000


def make_starts(bursts, seed):
    """The 20 test starting states: 10 inside the recorded box, 10 at 1.5-2 times its half-width outside it; a variable
    whose recordings never change sign keeps its sign (its outside starts go out on the far side from zero)."""
    B = np.asarray(bursts, dtype=float).reshape(-1, np.asarray(bursts).shape[-1])
    d = B.shape[1]
    lo, hi = B.min(0), B.max(0)
    c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    keep = (lo > 0) | (hi < 0)
    rng = np.random.default_rng(seed)
    starts = [lo + (hi - lo) * rng.random(d) for _ in range(N_IN)]
    for _ in range(N_OUT):
        sign = np.where(keep, np.sign(c), rng.choice([-1.0, 1.0], d))
        starts.append(c + sign * h * rng.uniform(1.5, 2.0, d))
    return np.array(starts)


def bounded_runs(f, starts, T, scale):
    """Per start: True if the trajectory stays finite and within 100 x max(|start|, scale) for the whole horizon T."""
    from scipy.integrate import solve_ivp
    out = []
    for x0 in starts:
        bound = 100 * max(np.abs(x0).max(), scale)
        n = [0]

        def g(t, x):
            n[0] += 1
            if n[0] > MAX_CALLS:
                raise OverflowError()
            return f(x)

        def leave(t, x):
            return bound - np.abs(x).max()
        leave.terminal, leave.direction = True, -1
        try:
            sol = solve_ivp(g, (0, T), x0, method="DOP853", rtol=1e-12, atol=1e-14 * scale, events=leave)
            out.append(bool(sol.success and sol.status == 0 and np.all(np.isfinite(sol.y))))
        except Exception:
            out.append(False)
    return out


def wide_reference(system):
    import harness27 as H27
    import synthetic28 as SY

    class WideReference(SY.SynthReference):
        def __init__(self, system, seed=0):
            self.sys = s = system
            self.rng = rng = np.random.default_rng([2700, seed, sum(map(ord, s.name))])
            self.rng_held = np.random.default_rng([2701, seed, sum(map(ord, s.name))])
            self._transient_wide(s, rng)

        def _transient_wide(self, s, rng):
            """harness27.Reference27._transient with the learning and held-out starts drawn from U(SPREAD)."""
            d = s.d
            spread = SPREAD_BY_D[d]
            f_learn = s.ic * rng.uniform(*spread, (H27.N_LEARN_REC, d))
            f_held = s.ic * rng.uniform(*spread, (16, d))
            lo = rng.uniform(0.1, 0.15, (8, d)); hi = rng.uniform(3.5, 4.5, (8, d))
            f_oor = s.ic * np.concatenate([lo, hi])
            pre = 0
            prov = s.rollout(f_learn[:4], 64, H27.REC_STEPS)
            while not np.all(np.isfinite(prov)) and pre < 6:
                s.dt /= 2
                pre += 1
                prov = s.rollout(f_learn[:4], 64, H27.REC_STEPS * 2 ** pre)
            if not np.all(np.isfinite(prov)):
                raise RuntimeError("provisional transient diverged")
            flat = prov.reshape(-1, d)
            scale0 = flat.std(0) + 1e-12 * (np.abs(flat).mean(0) + 1e-12)
            nsub = self._refine_dt(s, flat[::7], scale0)
            self.refine += pre
            steps = H27.REC_STEPS * 2 ** self.refine
            learn = s.rollout(f_learn, nsub, steps)
            flat = learn.reshape(-1, d)
            self.scale = flat.std(0) + 1e-12 * (np.abs(flat).mean(0) + 1e-12)
            self.nsub = self._ref_nsub(s, flat[rng.choice(len(flat), 100, replace=False)], flat)
            self.learn_trajs = s.rollout(f_learn, self.nsub, steps)
            self.held_trajs = s.rollout(f_held, self.nsub, steps)
            self.oor_trajs = s.rollout(f_oor, self.nsub, steps)
            for a in (self.learn_trajs, self.held_trajs):
                if not np.all(np.isfinite(a)):
                    raise RuntimeError("reference transients diverged")
            self.cap = min(steps, 2000)
    return WideReference(system)


def build_wide(desc, seed_starts=0):
    """domains_g1.build with the wide reference, then the bounded rule; syn.test = {"starts", "T"}."""
    import sympy as sp
    import domains_g1 as DG
    import synthetic28 as SY
    import practice_net as PN
    t0 = time.time()
    structure = DG.true_structure(desc)
    V = DG.vocabulary(desc["d"])
    for eq in structure:
        eq["keys"] = sorted({DG.representative(k, desc["d"]) for k in eq["keys"]})
        eq["nonpoly"] = sorted({DG.representative(k, desc["d"]) for k in eq["nonpoly"]})
        if any(k not in V for k in eq["keys"]):
            raise DG.Rejected("a true term outside the L1 vocabulary")
    try:
        with np.errstate(all="ignore"):
            system = DG.G1System(desc)
            ref = wide_reference(system)
    except (RuntimeError, SY.Rejected) as ex:
        raise DG.Rejected(f"simulation or cost: {ex}")
    secs = time.time() - t0
    if ref.refine > SY.MAX_REFINE or ref.nsub > SY.MAX_NSUB or secs > SY.MAX_BUILD_SECONDS:
        raise DG.Rejected(f"cost: refine {ref.refine}, nsub {ref.nsub}, {secs:.0f}s")
    allx = np.concatenate([ref.learn_trajs.reshape(-1, desc["d"]), ref.held_trajs.reshape(-1, desc["d"])])
    if not np.all(np.isfinite(allx)):
        raise DG.Rejected("not finite")
    if desc["positive"] and allx.min() <= 0:
        raise DG.Rejected("not positive")
    if np.abs(allx).max() > 100 * max(abs(v) for v in desc["ic"]):
        raise DG.Rejected("unbounded")
    span = ref.learn_trajs.max(1) - ref.learn_trajs.min(1)
    if np.any(np.median(span, 0) <= 1e-3 * ref.scale):
        raise DG.Rejected("a variable does not move")
    syn = DG.G1(desc, ref, structure, secs)
    view = PN.learner_view(syn.bursts(8), seed=1)
    if view["centred"]:
        raise DG.Rejected("centred at M=8")
    Xc = (view["Zc"] * view["sig"] + view["mu"]).numpy()
    for i in range(desc["d"]):
        for f in DG.numeric_terms(desc, i):
            with np.errstate(all="ignore"):
                col = np.asarray(f(*[Xc[:, j] for j in range(desc["d"])]), dtype=float) * np.ones(len(Xc))
            if not np.all(np.isfinite(col)):
                raise DG.Rejected("a true term is singular on the data")
            if DG._collapse(col, Xc, f.own) < DG.ACCEPT_COLLAPSE:
                raise DG.Rejected("a true term is not visibly nonlinear on the data")
    # (1) bounded from every test start over 10 time constants
    d = desc["d"]
    xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
    f_ = sp.lambdify(xs, DG.numeric_rhs(desc), "numpy")

    def ftrue(x):
        with np.errstate(all="ignore"):
            return np.array([float(v) for v in f_(*x)])
    b8 = syn.bursts(8)
    starts = make_starts(b8, seed_starts)
    T = HORIZON * SY.own_timescale(lambda x: np.atleast_2d(ftrue(np.ravel(x))), np.asarray(desc["ic"], dtype=float), factor=1.0)
    scale = max(float(np.abs(b8).max()), 1e-12)
    ok = bounded_runs(ftrue, starts, T, scale)
    if not all(ok):
        raise DG.Rejected(f"blows up within the horizon from {sum(not v for v in ok)} of 20 test starts")
    syn.test = {"starts": starts.tolist(), "T": T}
    return syn


def pilot_job(args):
    shape, k = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import ladder as LD
    d, terms = shape
    keys = LD.piece_keys(d)
    allowed = {lv: keys[lv] for lv in LD._ALL}
    t0, why = time.time(), {}
    for r in range(MAX_DRAWS):
        seed = f"wide-pilot-d{d}-{k}" if r == 0 else f"wide-pilot-d{d}-{k}-r{r}"
        desc = LD.draw(seed, allowed, terms, d)
        if DR.prescreen(desc) is not None:
            why["pre-screen"] = why.get("pre-screen", 0) + 1
            continue
        try:
            syn = build_wide(desc, seed_starts=LD.stable(seed, "test starts"))
        except Exception as ex:
            w = str(ex)[:40]
            why[w] = why.get(w, 0) + 1
            continue
        return {"shape": shape, "k": k, "accepted": True, "tries": r + 1, "seconds": time.time() - t0, "why": why}
    return {"shape": shape, "k": k, "accepted": False, "tries": MAX_DRAWS, "seconds": time.time() - t0, "why": why}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers, n = int(sys.argv[1]), int(sys.argv[2])
    shapes = [tuple(json.loads(x)) for x in os.environ.get("WIDE_SHAPES", "").split(";") if x] or [(3, (3, 3)), (4, (3, 3))]
    shapes = [(sh[0], tuple(sh[1])) for sh in shapes]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(pilot_job, [(sh, k) for sh in shapes for k in range(n)]))
    for sh in shapes:
        R = [r for r in res if r["shape"] == sh]
        acc = [r for r in R if r["accepted"]]
        why = {}
        for r in R:
            for k_, v in r["why"].items():
                why[k_] = why.get(k_, 0) + v
        print(f"{sh[0]} variables, {sh[1]} terms: {len(acc)} of {len(R)} accepted; median draws "
              f"{np.median([r['tries'] for r in acc]) if acc else None}; median seconds {np.median([r['seconds'] for r in acc]) if acc else float('nan'):.0f}; "
              f"top rejections {dict(sorted(why.items(), key=lambda x: -x[1])[:6])}", flush=True)


if __name__ == "__main__":
    main()
