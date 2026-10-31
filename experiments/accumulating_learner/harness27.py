"""Trial 27 exactness harness: reference data for the fresh BioModels pool (2-D/4-D; oscillators and settling
transients) and for Trial 26's opened systems, in-range and out-of-range exactness tests. Truth is used only here,
for generating observations and judging artifacts; the learner never receives it. See docs/research-log.md (Trial 27).

Exactness is defined exactly as in Trial 26 (one-step 1e-12 of scale; multi-step within the ideal-model envelope).
Transient systems: the step is settle_time/100 and every recording covers 1.5 settle times (150 steps); learning and
held-out recordings start from different random scalings (0.5-1.5) of the published initial state.
Out-of-range states (owner addition, Trial 27): starting states outside the region the learner observed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import compile_artifact as CA
import exact_harness as H

ROOT = H.ROOT
POOL_DIRS = {2: ROOT / "outputs/accumulating_learner/trial27/pool_d2", 4: ROOT / "outputs/accumulating_learner/trial27/pool_d4"}
SBML = H.SBML
STEPS = 100            # steps per period (oscillator) or per settle time (transient)
REC_STEPS = 150        # transient recording length in steps (before interval refinement)
N_LEARN_REC = 64       # transient learning recordings (varied starting states)
MAX_REF_NSUB = 2       # sampling rule: the true system needs at most this many substeps per interval
REF_MARGIN = 4         # reference substeps = 4 x the converged count (RK8 error / 65536); Trial 27 build fix


def split_half(mid):
    return "eval" if int(hashlib.sha256(("trial27-split-v1:" + mid).encode()).hexdigest()[0], 16) % 2 else "dev"


def ill_defined(rec):
    """Pool validity filter (both halves alike): converted equations containing infinite or undefined constants."""
    return any(tok in e for e in rec.get("exprs", []) for tok in ("ComplexInfinity", "zoo", "NaN", "Infinity"))


def pool_records():
    out = {}
    for d, p in POOL_DIRS.items():
        f = p / "conversion.json"
        if f.exists():
            for r in json.loads(f.read_text())["records"]:
                if r["status"] == "accepted" and not ill_defined(r):
                    out[r["id"]] = dict(r, d=d)
    return out


class TrueSystem27:
    """Ground truth for evaluation and observation generation. Pool systems use a compiled right-hand side."""

    def __init__(self, name):
        self.name = name
        recs = pool_records()
        if name in recs:
            import biomodels_convert as BC
            rec = recs[name]
            man = {m["id"]: m for m in json.loads((SBML / "manifest.json").read_text())["models"]}
            info, exprs = BC.convert(SBML / man[name]["file"], rec["d"], rec["d"])
            self._f, self._roll = CA.build_true(exprs)
            self.kind = rec["kind"]
            self.T = float(rec["period"] if self.kind == "oscillator" else rec["settle_time"])
            self.ic = np.array(info["x0"], dtype=float)
            self.positive = True
        elif name.startswith("BIOMD"):  # Trial 26 opened 3-D BioModels (development only): compiled as well
            import biomodels_convert as BC
            ts = H.TrueSystem(name)
            man = {m["id"]: m for m in json.loads((SBML / "manifest.json").read_text())["models"]}
            _, exprs = BC.convert(SBML / man[name]["file"], 3, 3)
            self._f, self._roll = CA.build_true(exprs)
            self.kind, self.T, self.ic = "oscillator", ts.P, ts.ic
            self.positive = True
        else:                           # Trial 26 opened catalog systems (development only)
            ts = H.TrueSystem(name)
            self._f, self._roll = ts.f, None
            self.kind, self.T, self.ic = "oscillator", ts.P, ts.ic
            self.positive = False
        self.d = len(self.ic)
        self.dt = self.T / STEPS

    def f(self, x):
        return np.asarray(self._f(x), dtype=float)

    def rollout(self, x0, nsub, steps):
        if self._roll is not None:
            return self._roll(x0, steps, nsub, self.dt)
        return H.rollout(self.f, np.asarray(x0, dtype=float), self.dt, nsub, steps)

    def one_step(self, x, nsub):
        return self.rollout(x, nsub, 1)[:, 1]


def choose_nsub(sys_, states, scale, tol=1e-14, max_nsub=8192):
    n, prev, last = 1, sys_.one_step(states, 1), np.inf
    while n < max_nsub:
        nxt = sys_.one_step(states, 2 * n)
        diff = float(np.max(np.abs(nxt - prev) / scale))
        if not np.isfinite(diff):
            n, prev, last = 2 * n, nxt, np.inf
            continue
        if diff <= tol or (diff <= 1e-13 and diff > last / 8):
            return n
        n, prev, last = 2 * n, nxt, diff
    return max_nsub


class Reference27:
    """Per-system reference: learning / held-out / out-of-range recordings, scale, substeps, envelope."""

    def __init__(self, name, seed=0):
        self.sys = s = TrueSystem27(name)
        self.rng = rng = np.random.default_rng([2700, seed, sum(map(ord, name))])
        self.rng_held = np.random.default_rng([2701, seed, sum(map(ord, name))])   # held-out test states (own stream)
        if s.kind == "oscillator":
            self._oscillator(s, rng)
        else:
            self._transient(s, rng)

    def _ref_nsub(self, s, sample, states):
        """Reference substeps (Trial 27 build fix): the doubling rule on the 100 sampled states and on a deterministic
        stride of up to 2000 recorded states, times REF_MARGIN (a 100-state sample missed demanding states: on a
        development system the 2-substep reference was off by 6e-10 at held-out states, failing an exact model)."""
        n = max(choose_nsub(s, sample, self.scale), choose_nsub(s, states[:: max(1, len(states) // 2000)], self.scale))
        return REF_MARGIN * n

    # -- oscillators: as in Trial 26 (burn-in on the attractor) --
    def _refine_dt(self, s, states, scale):
        """Sampling rule (Trial 27, both halves alike): halve the recording interval until the true system needs
        <= MAX_REF_NSUB substeps per interval (dynamics resolved). Returns the substep count."""
        self.refine = 0
        nsub = choose_nsub(s, states, scale)
        while nsub > MAX_REF_NSUB and self.refine < 6:
            s.dt /= 2
            self.refine += 1
            nsub = choose_nsub(s, states, scale)
        return nsub

    def _oscillator(self, s, rng):
        x = s.rollout(s.ic[None, :], 64, 2 * STEPS)[0]
        if not np.all(np.isfinite(x)):
            raise RuntimeError("provisional burn-in diverged")
        scale0 = x.std(0) + 1e-12 * (np.abs(x).mean(0) + 1e-12)
        nsub = self._refine_dt(s, x[::4], scale0)
        y = s.rollout(x[-1:], nsub, 50 * STEPS * 2 ** self.refine)[:, -1]
        long = s.rollout(y, nsub, 2000 * 2 ** self.refine)[0]
        if not np.all(np.isfinite(long)):
            raise RuntimeError("reference burn-in diverged")
        self.scale = long.std(0) + 1e-12 * (np.abs(long).mean(0) + 1e-12)
        self.nsub = self._ref_nsub(s, long[rng.choice(len(long), 100, replace=False)], long)
        starts = long[rng.choice(len(long), 8, replace=False)] * (1 + 1e-3 * rng.standard_normal((8, s.d)))
        y = s.rollout(starts, self.nsub, 10 * STEPS * 2 ** self.refine)[:, -1]
        trajs = s.rollout(y, self.nsub, 3000 * 2 ** self.refine)
        if not np.all(np.isfinite(trajs)):
            raise RuntimeError("reference trajectories diverged")
        self.learn_trajs, self.held_trajs = trajs[:4], trajs[4:]
        self.cap = 20 * STEPS
        # out of range: attractor states pushed outward (multiplicatively for positive systems, radially otherwise)
        xa = trajs[4:, ::97].reshape(-1, s.d)[:16]
        if s.positive:
            xo = xa * np.exp(rng.uniform(np.log(1.5), np.log(2.5), xa.shape) * rng.choice([-1, 1], xa.shape))
        else:
            c = long.mean(0)
            xo = c + (xa - c) * rng.uniform(1.5, 2.5, (len(xa), 1))
        self.oor_trajs = s.rollout(xo, self.nsub, REC_STEPS)

    # -- settling transients: recordings from varied starting states --
    def _transient(self, s, rng):
        d = s.d
        f_learn = s.ic * rng.uniform(0.5, 1.5, (N_LEARN_REC, d))
        f_held = s.ic * rng.uniform(0.5, 1.5, (16, d))
        lo = rng.uniform(0.2, 0.4, (8, d)); hi = rng.uniform(2.0, 3.0, (8, d))
        f_oor = s.ic * np.concatenate([lo, hi])
        pre = 0                                        # stiff start: refine the interval until a provisional run is finite
        prov = s.rollout(f_learn[:4], 64, REC_STEPS)
        while not np.all(np.isfinite(prov)) and pre < 6:
            s.dt /= 2
            pre += 1
            prov = s.rollout(f_learn[:4], 64, REC_STEPS * 2 ** pre)
        if not np.all(np.isfinite(prov)):
            raise RuntimeError("provisional transient diverged")
        flat = prov.reshape(-1, d)
        scale0 = flat.std(0) + 1e-12 * (np.abs(flat).mean(0) + 1e-12)
        nsub = self._refine_dt(s, flat[::7], scale0)
        self.refine += pre
        steps = REC_STEPS * 2 ** self.refine
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

    # -- shared --
    def one_step(self, x):
        return self.sys.one_step(x, self.nsub)

    def held_states(self, n=1000):
        flat = self.held_trajs[:, :-1].reshape(-1, self.sys.d)
        return flat[self.rng_held.choice(len(flat), min(n, len(flat)), replace=False)]

    def learned_box(self):
        flat = self.learn_trajs.reshape(-1, self.sys.d)
        return flat.min(0), flat.max(0)

    def oor_states(self, n=400):
        """Out-of-range starting states: states of the out-of-range recordings lying outside the learned box."""
        lo, hi = self.learned_box()
        flat = self.oor_trajs.reshape(-1, self.sys.d)
        flat = flat[np.all(np.isfinite(flat), 1)]
        out = flat[np.any((flat < lo) | (flat > hi), 1)]
        if len(out) > n:
            out = out[self.rng.choice(len(out), n, replace=False)]
        return out

    def envelope(self, starts, cap, runs=20, eps=1e-12):
        ref = self.sys.rollout(starts, self.nsub, cap)
        x = np.repeat(starts, runs, axis=0)
        refr = np.repeat(ref, runs, axis=0)
        e = [np.zeros(len(x))]
        for k in range(cap):
            x = self.one_step(x) + eps * self.scale * self.rng.standard_normal(x.shape)
            e.append(np.max(np.abs(x - refr[:, k + 1]) / self.scale, axis=1))
        E = np.median(np.stack(e, 1), axis=0)
        over = np.nonzero(E >= 1e-6)[0]
        Hz = max(int(over[0]) if len(over) else cap, 1)
        return ref, E, Hz

    def prepare(self, n_starts=100):
        """Envelope for the in-range multi-step test (held-out recordings)."""
        d = self.sys.d
        if self.sys.kind == "oscillator":
            flat = self.held_trajs[:, :-self.cap - 1].reshape(-1, d)
            starts = flat[self.rng.choice(len(flat), n_starts, replace=False)]
        else:
            starts = self.held_trajs[:, 0]
        self.env_starts = starts
        self.env_ref, self.E, self.H = self.envelope(starts, self.cap)


def _multi_ok(xh, ref, E, Hz, scale):
    e = np.max(np.abs(xh - ref[:, :Hz + 1]) / scale, axis=2)
    med, p90 = np.median(e, 0), np.percentile(e, 90, axis=0)
    Ek = np.maximum(E[:Hz + 1], 1e-16)
    return bool(np.all(med[1:] <= 10 * Ek[1:]) and np.all(p90[1:] <= 100 * Ek[1:])), float(np.max(med[1:] / Ek[1:]))


def evaluate(ref, f_hat, nsub_hat, rollout_hat=None):
    """In-range exactness (Trial 26 definition) of an artifact: vector field f_hat (or a compiled rollout)."""
    roll = rollout_hat or (lambda x0, steps: H.rollout(f_hat, x0, ref.sys.dt, nsub_hat, steps))
    xs = ref.held_states(1000)
    with np.errstate(all="ignore"):
        one = float(np.max(np.abs(roll(xs, 1)[:, 1] - ref.one_step(xs)) / ref.scale))
    if not (one <= 1e-12):             # exact needs both tests: skip the long rollouts of a model that already failed
        return {"one_step_max_scaled_error": one if np.isfinite(one) else float("inf"), "one_step_ok": False,
                "horizon_steps": getattr(ref, "H", None), "multi_step_ok": False, "multi_step_worst_ratio_median": float("nan"),
                "multi_step_skipped": True, "exact": False}
    if not hasattr(ref, "E"):          # envelope for the multi-step test, only for models that passed one-step
        ref.prepare()
    xh = roll(ref.env_starts, ref.H)
    ok, worst = _multi_ok(xh, ref.env_ref, ref.E, ref.H, ref.scale)
    return {"one_step_max_scaled_error": one, "one_step_ok": bool(one <= 1e-12), "horizon_steps": ref.H,
            "multi_step_ok": ok, "multi_step_worst_ratio_median": worst, "exact": bool(one <= 1e-12 and ok)}


def _converged_nsub(step_fn, xs, scale, tol=1e-14, max_nsub=8192):
    """Doubling rule for any one-step function step_fn(x, nsub) at the given states."""
    n, prev, last = 1, step_fn(xs, 1), np.inf
    while n < max_nsub:
        nxt = step_fn(xs, 2 * n)
        with np.errstate(all="ignore"):
            diff = float(np.max(np.abs(nxt - prev) / scale))
        if not np.isfinite(diff):
            n, prev, last = 2 * n, nxt, np.inf
            continue
        if diff <= tol or (diff <= 1e-13 and diff > last / 8):
            return n
        n, prev, last = 2 * n, nxt, diff
    return max_nsub


def evaluate_out_of_range(ref, roll_n, n=400):
    """Owner addition: is the learned VECTOR FIELD exact at starting states outside the observed region?
    roll_n(x0, steps, nsub) runs the artifact with a given substep count. Both the truth and the artifact use substep
    counts converged at these states (the artifact's by its own doubling rule; no simulator), so the test measures the
    learned equations, not step-size choices made for the observed region."""
    xs = ref.oor_states(n)
    if len(xs) == 0:
        return {"n_states": 0}
    ns_true = _converged_nsub(lambda x, k: ref.sys.one_step(x, k), xs, ref.scale)
    ns_hat = 2 * _converged_nsub(lambda x, k: roll_n(x, 1, k)[:, 1], xs, ref.scale)
    with np.errstate(all="ignore"):
        pred = roll_n(xs, 1, ns_hat)[:, 1]
        truth = ref.sys.one_step(xs, ns_true)
        truth2 = ref.sys.one_step(xs, min(2 * ns_true, 16384))
    # states where the reference itself is not trustworthy (non-finite or not converged) are excluded and counted
    trust = np.all(np.isfinite(truth), 1) & (np.max(np.abs(truth2 - truth) / ref.scale, axis=1) <= 1e-13)
    n_excluded = int((~trust).sum())
    xs, pred, truth = xs[trust], pred[trust], truth[trust]
    if len(xs) == 0:
        return {"n_states": 0, "n_excluded_untrusted_reference": n_excluded}
    with np.errstate(all="ignore"):
        err = np.max(np.abs(pred - truth) / ref.scale, axis=1)
    err = np.where(np.isfinite(err), err, np.inf)
    starts = xs[:min(len(xs), 40)]
    cap = min(ref.cap, 150)
    if not (err.max() <= 1e-12):       # the multi-step part cannot change "exact" once one-step failed
        return {"n_states": int(len(xs)), "n_excluded_untrusted_reference": n_excluded, "one_step_max_scaled_error": float(err.max()),
                "fraction_states_one_step_exact": float(np.mean(err <= 1e-12)), "nsub_true": int(ns_true), "nsub_artifact": int(ns_hat),
                "one_step_ok": False, "multi_step_ok": False, "multi_step_skipped": True, "exact": False}
    try:
        r = ref.sys.rollout(starts, ns_true, cap)
        x = np.repeat(starts, 10, axis=0); rr = np.repeat(r, 10, axis=0)
        e = [np.zeros(len(x))]
        for k in range(cap):
            x = ref.sys.one_step(x, ns_true) + 1e-12 * ref.scale * ref.rng.standard_normal(x.shape)
            e.append(np.max(np.abs(x - rr[:, k + 1]) / ref.scale, axis=1))
        E = np.median(np.stack(e, 1), axis=0)
        over = np.nonzero(E >= 1e-6)[0]
        Hz = max(int(over[0]) if len(over) else cap, 1)
        with np.errstate(all="ignore"):
            ok, worst = _multi_ok(roll_n(starts, Hz, ns_hat), r, E, Hz, ref.scale)
    except Exception:
        ok, worst, Hz = False, float("inf"), 0
    return {"n_states": int(len(xs)), "n_excluded_untrusted_reference": n_excluded, "one_step_max_scaled_error": float(err.max()),
            "fraction_states_one_step_exact": float(np.mean(err <= 1e-12)), "nsub_true": int(ns_true), "nsub_artifact": int(ns_hat),
            "one_step_ok": bool(err.max() <= 1e-12), "multi_step_ok": ok, "horizon_steps": Hz,
            "multi_step_worst_ratio_median": worst, "exact": bool(err.max() <= 1e-12 and ok)}


def bursts_from(ref, seed, M, L=13, pool=64):
    """First M of a seeded order of burst starts over ALL learning recordings (nested across M)."""
    n, T = ref.learn_trajs.shape[0], ref.learn_trajs.shape[1]
    rng = np.random.default_rng([2701, seed, sum(map(ord, ref.sys.name))])
    picks = rng.permutation(n * (T - L))[:pool]
    starts = [(int(k // (T - L)), int(k % (T - L))) for k in picks]
    return np.stack([ref.learn_trajs[a, s:s + L] for a, s in starts[:M]])
