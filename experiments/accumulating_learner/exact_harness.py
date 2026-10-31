"""Exactness harness (goal correction, 31 October): reference one-step maps, learning / held-out trajectories,
chaos-aware tolerance envelopes, exactness tests and cost comparison. Everything is generated in memory.
See docs/research-log.md, "How exactness is defined and measured, per system".
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp import dop853_coefficients as D

import systems as S

ROOT = S.ROOT
REAL3D = ROOT / "outputs/accumulating_learner/trial16/realfam"
SBML = Path("/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad/dl/biomodels")
STEPS_PER_PERIOD = 100
A, B, C = D.A[:D.N_STAGES, :D.N_STAGES], D.B, D.C[:D.N_STAGES]


def rk8(f, x, dt, nsub):
    """Fixed-step 8th-order Runge-Kutta (DOP853 tableau) over one interval dt with nsub substeps. x: (n, d)."""
    h = dt / nsub
    for _ in range(nsub):
        K = []
        for i in range(D.N_STAGES):
            xi = x if i == 0 else x + h * sum(A[i, j] * K[j] for j in range(i) if A[i, j] != 0)
            K.append(f(xi))
        x = x + h * sum(B[i] * K[i] for i in range(D.N_STAGES) if B[i] != 0)
    return x


def rollout(f, x0, dt, nsub, steps):
    out = [x0]
    x = x0
    for _ in range(steps):
        x = rk8(f, x, dt, nsub)
        out.append(x)
    return np.stack(out, 1)  # (n, steps+1, d)


class TrueSystem:
    """Ground truth for evaluation and for generating observations (the learner never receives this object)."""

    def __init__(self, name):
        self.name = name
        if name.startswith("BIOMD"):
            import biomodels_convert as C_
            man = {m["id"]: m for m in json.loads((SBML / "manifest.json").read_text())["models"]}
            info, exprs = C_.convert(SBML / man[name]["file"], 3, 3)
            self._f = C_.numeric(exprs)
            conv = {r["id"]: r for r in json.loads((REAL3D / "conversion.json").read_text())["records"]}
            self.P, self.lyap = conv[name]["period"], None
            self.ic = np.array(info["x0"], dtype=float)
        else:
            self._sys = S.System(name, allow_confirmation=True)
            meta = S.METADATA[name]
            self.P, self.lyap = float(meta["period"]), meta.get("maximum_lyapunov_estimated")
            self.ic = np.array(meta["initial_conditions"], dtype=float)
            self._f = self._sys
        self.d = len(self.ic)
        self.dt = self.P / STEPS_PER_PERIOD

    def f(self, x):
        return np.asarray(self._f(x), dtype=float)


def choose_nsub(f, states, dt, scale, tol=1e-14, max_nsub=1024):
    """Smallest power-of-two substep count whose doubling changes the one-step map by <= tol * scale (max norm), or,
    once that change is below 1e-13 * scale, whose doubling no longer shrinks it 8x (floating-point rounding floor)."""
    n, prev, last = 1, rk8(f, states, dt, 1), np.inf
    while n < max_nsub:
        nxt = rk8(f, states, dt, 2 * n)
        diff = float(np.max(np.abs(nxt - prev) / scale))
        if diff <= tol or (diff <= 1e-13 and diff > last / 8):
            return n
        n, prev, last = 2 * n, nxt, diff
    return max_nsub


class Reference:
    """Per-system reference data: scales, substeps, learning and held-out trajectories, envelope and horizon."""

    def __init__(self, name, seed=0, traj_steps=3000, burn_periods=50):
        self.sys = sys_ = TrueSystem(name)
        rng = np.random.default_rng([2600, seed, sum(map(ord, name))])
        f, dt = sys_.f, sys_.dt
        # short provisional burn-in, then the converged substep count, then the real burn-in (all batched)
        x = sys_.ic[None, :]
        x = rollout(f, x, dt, 64, 2 * STEPS_PER_PERIOD)[0]
        if not np.all(np.isfinite(x)):
            raise RuntimeError("provisional burn-in diverged")
        scale0 = x.std(0) + 1e-12 * (np.abs(x).mean(0) + 1e-12)
        self.nsub = choose_nsub(f, x[::4], dt, scale0)
        y = x[-1:]
        for _ in range(burn_periods):
            y = rollout(f, y, dt, self.nsub, STEPS_PER_PERIOD)[:, -1]
        long = rollout(f, y, dt, self.nsub, 2000)[0]
        if not np.all(np.isfinite(long)):
            raise RuntimeError("reference burn-in diverged")
        self.scale = long.std(0) + 1e-12 * (np.abs(long).mean(0) + 1e-12)
        idx = rng.choice(len(long), 100, replace=False)
        self.nsub = choose_nsub(f, long[idx], dt, self.scale)
        # 8 independent trajectories (4 for learning, 4 held out), integrated as one batch
        starts = long[rng.choice(len(long), 8, replace=False)] * (1 + 1e-3 * rng.standard_normal((8, sys_.d)))
        y = starts
        for _ in range(10):  # 10 periods of burn-in each
            y = rollout(f, y, dt, self.nsub, STEPS_PER_PERIOD)[:, -1]
        trajs = rollout(f, y, dt, self.nsub, traj_steps)
        if not np.all(np.isfinite(trajs)):
            raise RuntimeError("reference trajectories diverged")
        self.learn_trajs, self.held_trajs = trajs[:4], trajs[4:]
        self.rng = rng

    def one_step(self, x):
        return rk8(self.sys.f, x, self.sys.dt, self.nsub)

    def held_states(self, n=1000):
        flat = self.held_trajs[:, :-1].reshape(-1, self.sys.d)
        return flat[self.rng.choice(len(flat), n, replace=False)]

    def envelope(self, n_starts=100, runs=20, cap=20 * STEPS_PER_PERIOD, eps=1e-12):
        """E*(k): median scaled error of the true map with eps*scale perturbations injected every step; horizon H."""
        flat = self.held_trajs[:, :-cap - 1].reshape(-1, self.sys.d)
        x0 = flat[self.rng.choice(len(flat), n_starts, replace=False)]
        ref = rollout(self.sys.f, x0, self.sys.dt, self.nsub, cap)
        x = np.repeat(x0, runs, axis=0)                     # all Monte-Carlo runs as one batch
        refr = np.repeat(ref, runs, axis=0)
        e = [np.zeros(len(x))]
        for k in range(cap):
            x = self.one_step(x) + eps * self.scale * self.rng.standard_normal(x.shape)
            e.append(np.max(np.abs(x - refr[:, k + 1]) / self.scale, axis=1))
        E = np.median(np.stack(e, 1), axis=0)
        over = np.nonzero(E >= 1e-6)[0]
        H = int(over[0]) if len(over) else cap
        self.env_starts, self.env_ref, self.E, self.H = x0, ref, E, max(H, 1)
        return E, self.H


def evaluate_artifact(ref, f_hat, nsub_hat):
    """Exactness test of an artifact (learned vector field + its own RK8 substeps) against the reference."""
    xs = ref.held_states(1000)
    one = np.max(np.abs(rk8(f_hat, xs, ref.sys.dt, nsub_hat) - ref.one_step(xs)) / ref.scale)
    if not hasattr(ref, "E"):
        ref.envelope()
    H = ref.H
    xh = rollout(f_hat, ref.env_starts, ref.sys.dt, nsub_hat, H)
    e = np.max(np.abs(xh - ref.env_ref[:, :H + 1]) / ref.scale, axis=2)  # (starts, H+1)
    med, p90 = np.median(e, 0), np.percentile(e, 90, axis=0)
    Ek = np.maximum(ref.E[:H + 1], 1e-16)
    multi_ok = bool(np.all(med[1:] <= 10 * Ek[1:]) and np.all(p90[1:] <= 100 * Ek[1:]))
    return {"one_step_max_scaled_error": float(one), "one_step_ok": bool(one <= 1e-12),
            "horizon_steps": H, "multi_step_ok": multi_ok,
            "multi_step_worst_ratio_median": float(np.max(med[1:] / Ek[1:])),
            "exact": bool(one <= 1e-12 and multi_ok)}


def cost(ref, f_hat, nsub_hat, steps=1000, batch=100):
    """Wall time: artifact (vectorized RK8 on the learned field) vs original simulation (scipy DOP853 on the true f)."""
    x0 = ref.held_trajs[:, 0][np.arange(batch) % 4] * (1 + 1e-6 * ref.rng.standard_normal((batch, ref.sys.d)))
    out = {}
    for label, n in (("single", 1), ("batch", batch)):
        t0 = time.perf_counter(); rollout(f_hat, x0[:n], ref.sys.dt, nsub_hat, steps); ta = time.perf_counter() - t0
        T = steps * ref.sys.dt
        t_eval = np.linspace(0, T, steps + 1)
        fun = lambda t, y: ref.sys.f(y[None, :])[0]
        t0 = time.perf_counter()
        for i in range(n):
            solve_ivp(fun, (0, T), x0[i], method="DOP853", t_eval=t_eval, rtol=1e-13, atol=1e-15 * ref.scale)
        tr = time.perf_counter() - t0
        out[label] = {"artifact_s": ta, "reference_s": tr, "speedup": tr / ta}
    out["cheap"] = bool(out["single"]["speedup"] > 1 and out["batch"]["speedup"] > 1)
    return out


def evaluate_rollout(ref, roll, nsub_hat):
    """Exactness test of a compiled artifact: roll(x0 (n, d), steps, nsub) -> (n, steps+1, d)."""
    xs = ref.held_states(1000)
    one = np.max(np.abs(roll(xs, 1, nsub_hat)[:, 1] - ref.one_step(xs)) / ref.scale)
    if not hasattr(ref, "E"):
        ref.envelope()
    H = ref.H
    xh = roll(ref.env_starts, H, nsub_hat)
    e = np.max(np.abs(xh - ref.env_ref[:, :H + 1]) / ref.scale, axis=2)
    med, p90 = np.median(e, 0), np.percentile(e, 90, axis=0)
    Ek = np.maximum(ref.E[:H + 1], 1e-16)
    multi_ok = bool(np.all(med[1:] <= 10 * Ek[1:]) and np.all(p90[1:] <= 100 * Ek[1:]))
    return {"one_step_max_scaled_error": float(one), "one_step_ok": bool(one <= 1e-12),
            "horizon_steps": H, "multi_step_ok": multi_ok,
            "multi_step_worst_ratio_median": float(np.max(med[1:] / Ek[1:])),
            "exact": bool(one <= 1e-12 and multi_ok)}


def cost_compiled(ref, roll, nsub_hat, steps=1000, batch=100):
    """Wall time: compiled artifact vs the original simulation (scipy DOP853, rtol 1e-13, on the true f)."""
    x0 = ref.held_trajs[:, 0][np.arange(batch) % 4] * (1 + 1e-6 * ref.rng.standard_normal((batch, ref.sys.d)))
    T = steps * ref.sys.dt
    t_eval = np.linspace(0, T, steps + 1)
    fun = lambda t, y: ref.sys.f(y[None, :])[0]
    out = {}
    for label, n in (("single", 1), ("batch", batch)):
        t0 = time.perf_counter(); roll(x0[:n], steps, nsub_hat); ta = time.perf_counter() - t0
        n_ref = min(n, 10)                     # the reference is timed on up to 10 trajectories and scaled
        t0 = time.perf_counter()
        nfev = 0
        for i in range(n_ref):
            sol = solve_ivp(fun, (0, T), x0[i], method="DOP853", t_eval=t_eval, rtol=1e-13, atol=1e-15 * ref.scale)
            nfev += sol.nfev
        tr = (time.perf_counter() - t0) * n / n_ref
        out[label] = {"artifact_s": ta, "reference_s": tr, "speedup": tr / ta,
                      "reference_rhs_evals_per_step": nfev / n_ref / steps, "artifact_rhs_evals_per_step": 12 * nsub_hat}
    out["cheap"] = bool(out["single"]["speedup"] > 1 and out["batch"]["speedup"] > 1)
    return out
