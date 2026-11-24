"""Rollout check (owner, 22 November; research log "Owner: restore 'solved exactly'"): the owner's criterion for a solved
held-out system, applied to every system a rung's tests counted as solved. The true system is recovered by redrawing it
from its seed until the built recordings match (or read from its stored description). Each found equation is refitted
from its keys on the system's data (fitter_g1.evaluate_structure; must still pass the recording checks), and the found
model and the true system are compared from 20 new starting states (10 inside the recorded range, 10 at 1.5-2 times its
half-width outside it, every variable):
  one step: the relative error of the found right-hand side against the true one at each new state, <= ONE_STEP (1e-12);
  rollout: both integrated over 10 of the system's time constants (DOP853, rtol 1e-13), the trajectory's relative error
  (max over time of |x_found - x_true| / max over time of |x_true|) <= ROLLOUT (1e-9), judged on the starting states where
  the true system stays bounded (finite, within 100 times the larger of its start and the recorded range).
A system passes if every equation still passes the recording checks, every one-step error is within ONE_STEP, at least
one starting state is bounded and every bounded rollout is within ROLLOUT. Also: the conversion between the learner's
coordinates (x / sigma per variable, time in sample steps) and physical units is checked on the recorded states.
CPU only. Usage: rollout_check.py <workers> <rung> [<rung> ...]"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

ONE_STEP, ROLLOUT, N_IN, N_OUT, HORIZON = 1e-12, 1e-9, 10, 10, 10.0
PRECISE = os.environ.get("ROLLOUT_PRECISE", "1") == "1"         # owner, 23 Nov: the high-precision fit after acceptance
MARGIN = float(os.environ.get("ROLLOUT_MARGIN", "10"))           # owner, 24 Nov (option B): found <= MARGIN x the reference
UNJUDGEABLE = 1e-6                                               # a reference worse than this: the system cannot be judged
FULL = os.environ.get("ROLLOUT_FULL", "1") == "1"               # owner, 23 Nov: the horizon as written (bounded systems only);
                                                                 # ROLLOUT_FULL=0 for the old rungs' retroactive checks
MAX_CALLS = 200_000                                              # a rollout needing more right-hand-side calls is cut off


class TooLong(Exception):
    pass


def capped(f):
    n = [0]

    def g(t, x):
        n[0] += 1
        if n[0] > MAX_CALLS:
            raise TooLong()
        return f(x)
    return g


def recover(rec, rung, s, k):
    import domains_g1 as DG
    import dream_g2 as DR
    if rec.get("desc") and rec.get("test_starts"):              # strict generator (24 Nov): its own wide reference
        import gen_wide as GW
        syn = GW.rebuild_wide(rec["desc"])
        if LD.data_key(syn.bursts(8)) != LD.data_key(rec["bursts"]):
            raise RuntimeError(f"{rec['id']}: rebuilt recordings differ from the stored ones")
        return rec["desc"], syn
    if rec.get("desc"):
        return rec["desc"], DG.build(rec["desc"])
    R = LD.RUNGS[rung]
    keys = LD.piece_keys(R["d"])
    allowed = {lv: keys[lv] for lv in R["levels"]}
    want = LD.data_key(rec["bursts"])
    for r in range(200):
        seed = f"ladder-{rung}-s{s}-test-{k}" if r == 0 else f"ladder-{rung}-s{s}-test-{k}-r{r}"
        desc = LD.draw(seed, allowed, R["terms"], R["d"], False, R.get("max_k"))
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
        except Exception:
            continue
        if LD.data_key(syn.bursts(8)) == want:
            return desc, syn
    return None, None


def check_job(args):
    rec, rung, s, k, sols = args[:5]                               # sols: {(tag, round): [keys per equation]}
    opts = args[5] if len(args) > 5 else {}                      # {"perturb": [seeds], "eps": e}: calibration refits
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import sympy as sp
    import torch
    torch.set_num_threads(1)
    from scipy.integrate import solve_ivp
    import domains_g1 as DG
    import fitter_g1 as FG
    import search_g1 as SG
    import synthetic28 as SY
    t0 = time.time()
    try:
        desc, syn = recover(rec, rung, s, k)
        why = None
    except Exception as ex:                                      # counted as not solved, with the reason, not a crash
        desc, why = None, f"{type(ex).__name__}: {ex}"[:200]
    if desc is None:
        return [{"id": rec["id"], "tag": t, "round": rd, "recovered": False, "passes": False, "error": why} for (t, rd) in sols]
    d = desc["d"]
    xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
    ftrue_ = sp.lambdify(xs, DG.numeric_rhs(desc), "numpy")

    def ftrue(x):
        with np.errstate(all="ignore"):
            return np.array([float(v) for v in ftrue_(*x)])
    dt = float(syn.ref.sys.dt)
    B = np.asarray(rec["bursts"], dtype=float).reshape(-1, d)
    lo, hi = B.min(0), B.max(0)
    c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    rng = np.random.default_rng(LD.stable(rec["id"], "rollout"))
    starts = [lo + (hi - lo) * rng.random(d) for _ in range(N_IN)] if not rec.get("test_starts") else []
    keep = (lo > 0) | (hi < 0)                                   # variables that never change sign on the recordings
    for _ in range(N_OUT if not rec.get("test_starts") else 0):  # owner, 23 Nov: within the system's valid range
        sign = np.where(keep, np.sign(c), rng.choice([-1.0, 1.0], d))
        starts.append(c + sign * h * rng.uniform(1.5, 2.0, d))
    if rec.get("test_starts"):                                   # systems from the strict generator carry their test starts
        starts = [np.asarray(x, dtype=float) for x in rec["test_starts"]]
    scale = max(float(np.abs(B).max()), 1e-12)
    T = HORIZON * SY.own_timescale(lambda x: np.atleast_2d(ftrue(np.ravel(x))), np.asarray(desc["ic"], dtype=float), factor=1.0)
    tc = T / HORIZON
    true_runs = []                                               # (times, states, reached the full horizon) or None
    for x0 in starts:
        bound = 100 * max(np.abs(x0).max(), scale)

        def leave(t, x):
            return bound - np.abs(x).max()
        leave.terminal, leave.direction = True, -1
        segs, x, t_end = [], np.asarray(x0, dtype=float), 0.0
        for kseg in range(int(HORIZON)):                        # one time constant at a time; keep what stayed bounded
            try:
                sol = solve_ivp(capped(ftrue), (t_end, t_end + tc), x, method="DOP853", rtol=1e-13, atol=1e-14 * scale,
                                events=leave, dense_output=True)
            except Exception:
                break
            if not sol.success or sol.status == 1 or not np.all(np.isfinite(sol.y)):
                break                                            # a blow-up, or the bound left: stop before this segment
            segs.append(sol.sol)
            t_end, x = float(sol.t[-1]), sol.y[:, -1]
        if t_end < tc * 0.999:
            true_runs.append(None)
            continue
        tt = np.linspace(0, t_end, 201)
        yt = np.empty((len(x0), len(tt)))
        for j, t_ in enumerate(tt):
            seg = segs[min(int(t_ / tc), len(segs) - 1)]
            yt[:, j] = seg(t_)
        true_runs.append((tt, yt, t_end >= 0.999 * T))
    if not sols and not opts.get("perturb"):                     # an unsolved system: its boundedness only
        return [{"id": rec["id"], "tag": None, "round": None, "recovered": True,
                 "system_bounded": all(r_ is not None and r_[2] for r_ in true_runs)}]
    ctx = SG.system_context(np.array(rec["bursts"]), seed=s)
    fitter = FG.make_fitter(ctx, seed=s)
    sig = np.asarray(ctx["view"]["sig"], dtype=float)

    def judge(keys_per_eq, perturb=None):
        """Refit a structure on the recordings (recording checks, then the precision fit) and measure it against the
        exact true system from the 20 starts. perturb: (seed, eps) multiplies the precision fit's starting parameters
        by (1 + eps z), for the margin's calibration (a different route to the same structure)."""
        row = {}
        models, rec_ok = [], True
        for i, keys in enumerate(keys_per_eq):
            nd = FG.evaluate_structure(tuple(keys), ctx, i, fitter)
            rec_ok &= bool(nd["exact"])
            models.append((nd["terms"], np.asarray(nd["coefs"], dtype=float) if nd["coefs"] is not None else None))
        row["recordings_exact"] = rec_ok
        pmodel = pp = None
        if PRECISE and rec_ok:
            import precise_fit as PF
            tpe = [[(t.key, tuple(t.theta)) for t in terms] for terms, _ in models]
            cpe = [co for _, co in models]
            if perturb is not None:
                pr = np.random.default_rng(perturb[0])
                tpe = [[(kk, tuple(v * (1 + perturb[1] * pr.standard_normal()) for v in th)) for kk, th in eq] for eq in tpe]
                cpe = [co * (1 + perturb[1] * pr.standard_normal(len(co))) for co in cpe]
            _, _, r_before, r_after, pmodel, pp = PF.precise(tpe, cpe, rec["bursts"], sig, dt)
            row["shooting_residual_before"], row["shooting_residual_after"] = r_before, r_after

        def ffound(x):
            if pmodel is not None:
                return pmodel.rhs(np.asarray(x, dtype=float)[None, :], pp)[0]
            W = torch.tensor((np.asarray(x, dtype=float) / sig)[None, :])
            v = []
            for i, (terms, co) in enumerate(models):
                if co is None:
                    return np.full(d, np.nan)
                with torch.no_grad():
                    phi = np.array([float(FG.value(t, W)[0]) for t in terms]) if terms else np.zeros(0)
                v.append(sig[i] * float(phi @ co) / dt)
            return np.array(v)
        sample = B[rng_units.choice(len(B), size=min(20, len(B)), replace=False)]
        units = [np.linalg.norm(ffound(x) - ftrue(x)) / (np.linalg.norm(ftrue(x)) + 1e-300) for x in sample]
        row["units_check"] = float(np.median(units))
        one = []
        for x0 in starts:
            a, b = ffound(x0), ftrue(x0)
            one.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300)) if np.all(np.isfinite(b)) else float("nan"))
        roll, full = [], []
        for x0, run in zip(starts, true_runs):
            if run is None:
                roll.append(None)
                full.append(None)
                continue
            tt, yt, reached = run
            try:
                sol = solve_ivp(capped(ffound), (0, tt[-1]), x0, method="DOP853", rtol=1e-13, atol=1e-14 * scale, t_eval=tt)
                yf = sol.y if sol.success and sol.y.shape == yt.shape else None
            except Exception:
                yf = None
            roll.append(float(np.abs(yf - yt).max() / (np.abs(yt).max() + 1e-300)) if yf is not None and np.all(np.isfinite(yf)) else float("inf"))
            full.append(bool(reached))
        if opts.get("dense"):                                    # diagnostic: one step at many states of the same region
            dr = np.random.default_rng(LD.stable(rec["id"], "dense"))
            n = int(opts["dense"])
            pts = [lo + (hi - lo) * dr.random(d) for _ in range(n // 2)]
            for _ in range(n - n // 2):
                sign = np.where(keep, np.sign(c), dr.choice([-1.0, 1.0], d))
                pts.append(c + sign * h * dr.uniform(1.0, 2.0, d))
            dv = []
            for x0 in pts:
                a_, b_ = ffound(x0), ftrue(x0)
                if np.all(np.isfinite(b_)) and np.linalg.norm(b_) > 0:
                    dv.append(float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_)))
            dv = np.array([v if math.isfinite(v) else 1e300 for v in dv])
            row["one_step_dense_max"] = float(dv.max()) if len(dv) else None
            row["one_step_dense_q99"] = float(np.percentile(dv, 99)) if len(dv) else None
        one_f = [v for v in one if math.isfinite(v)]
        bounded = [v for v in roll if v is not None]
        strict = [v for v, f_ in zip(roll, full) if v is not None and f_]
        row.update(n_full_horizon=len(strict), rollout_max_full=max(strict) if strict else None,
                   horizon_reached=[None if r_ is None else float(r_[0][-1] / tc) for r_ in true_runs])
        row.update(one_step=one, rollout=roll, one_step_max=max(one_f) if one_f else float("inf"),
                   one_step_inside=max(one_f[:N_IN]) if one_f[:N_IN] else float("inf"),
                   rollout_max=max(bounded) if bounded else None, n_bounded=len(bounded), horizon=T)
        return row

    sys_bounded = all(r_ is not None and r_[2] for r_ in true_runs)
    rng_units = np.random.default_rng(LD.stable(rec["id"], "units"))
    truth = [e["keys"] if isinstance(e, dict) else e for e in rec["structure"]]
    ref = judge(truth)                                           # owner, 24 Nov (option B): the true structure refitted
    ref_one, ref_roll = ref["one_step_max"], ref["rollout_max"]
    unjudgeable = (not ref["recordings_exact"] or ref["n_bounded"] == 0 or not math.isfinite(ref_one) or ref_one > UNJUDGEABLE
                   or ref_roll is None or not math.isfinite(ref_roll) or ref_roll > UNJUDGEABLE)
    bar_one = max(MARGIN * ref_one, ONE_STEP) if math.isfinite(ref_one) else ONE_STEP
    bar_roll = max(MARGIN * ref_roll, ROLLOUT) if ref_roll is not None and math.isfinite(ref_roll) else ROLLOUT
    common = {"system_bounded": sys_bounded, "unjudgeable": bool(unjudgeable), "ref_recordings_exact": ref["recordings_exact"],
              "ref_one_step_max": ref_one, "ref_rollout_max": ref_roll, "bar_one_step": bar_one, "bar_rollout": bar_roll,
              "margin": MARGIN}
    out = []
    todo = list(sols.items()) + [((("perturb", j), None), None) for j in range(len(opts.get("perturb", [])))]
    for (tag, rd), keys_per_eq in todo:
        if keys_per_eq is None:                                  # a calibration refit of the true structure
            j = tag[1]
            pj = opts["perturb"][j]
            row = judge(truth, perturb=tuple(pj) if isinstance(pj, (list, tuple)) else (pj, opts.get("eps", 1e-6)))
            row["perturb"] = list(pj) if isinstance(pj, (list, tuple)) else [pj, opts.get("eps", 1e-6)]
            tag = f"perturb{j}"
        else:
            row = judge(keys_per_eq)
        row.update({"id": rec["id"], "tag": tag, "round": rd, "recovered": True}, **common)
        ok_bounded = row["n_bounded"] > 0 and (sys_bounded or not FULL)
        row["passes_absolute"] = bool(row["recordings_exact"] and row["one_step_max"] <= ONE_STEP and ok_bounded
                                      and row["rollout_max"] is not None and row["rollout_max"] <= ROLLOUT)
        row["passes"] = bool(not unjudgeable and row["recordings_exact"] and ok_bounded and row["one_step_max"] <= bar_one
                             and row["rollout_max"] is not None and row["rollout_max"] <= bar_roll)
        out.append(row)
    ref_row = dict(ref, id=rec["id"], tag="reference", round=None, recovered=True, **common)
    out.append(ref_row)
    for r in out:
        r["seconds"] = time.time() - t0
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    workers, rungs = int(sys.argv[1]), sys.argv[2:]
    for rung in rungs:
        jobs = {}
        for s in LD.SEEDS:
            test = {r["id"]: r for r in json.loads((LD.OUT / "gen" / f"{rung}-s{s}.json").read_text())["test"]}
            for rid, r in test.items():
                jobs.setdefault(rid, (r, s, {}))
            for arm in LD.ARMS:
                p = LD.OUT / rung / f"{arm}-s{s}.json"
                if not p.exists():
                    continue
                for rnd in json.loads(p.read_text())["rounds"]:
                    for y in rnd["test"]:
                        if y["solved"]:
                            sol = [None] * len(test[y["id"]]["structure"])
                            for e in y["equations"]:
                                sol[int(e["eq"])] = e["solution"]
                            jobs.setdefault(y["id"], (test[y["id"]], s, {}))[2][(f"{arm}-s{s}", rnd["round"])] = sol
        args = [(rec, rung, s, int(rec["id"].rsplit("-", 1)[1]), sols) for rec, s, sols in jobs.values()]
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            rows = [x for part in ex.map(check_job, args) for x in part]
        (LD.OUT / rung / "rollout_check.json").write_text(json.dumps(rows, indent=1, default=float))
        bounded = {r["id"]: r["system_bounded"] for r in rows if "system_bounded" in r}
        rows = [r for r in rows if r.get("tag") is not None and r.get("tag") != "reference"]
        R = [r for r in rows if r["recovered"]]
        print(f"   held-out systems bounded over the full horizon from all 20 test starts (would pass the new generator): "
              f"{sum(bounded.values())} of {len(bounded)}")
        print(f"== {rung}: {len(rows)} solved (system, lineage, round) results; {len(R)} with the true system recovered; "
              f"{time.time() - t0:.0f}s")
        print(f"   units check on recorded states: median relative error {np.median([r['units_check'] for r in R]):.1e}")
        print(f"   recordings still exact when refitted: {sum(r['recordings_exact'] for r in R)}; one step <= {ONE_STEP:g}: "
              f"{sum(r['one_step_max'] <= ONE_STEP for r in R)} (inside the range only: {sum(r['one_step_inside'] <= ONE_STEP for r in R)}); "
              f"rollout <= {ROLLOUT:g} on bounded starts: {sum(r['rollout_max'] is not None and r['rollout_max'] <= ROLLOUT for r in R)}; "
              f"no bounded start {sum(r['n_bounded'] == 0 for r in R)}; PASS {sum(r['passes'] for r in R)}")
        print(f"   starts judged (bounded for at least 1 time constant): median {np.median([r['n_bounded'] for r in R]):.0f} of 20; "
              f"starts bounded over the full 10 time constants: median {np.median([r['n_full_horizon'] for r in R]):.0f} of 20; "
              f"systems with at least one full-horizon start {sum(r['n_full_horizon'] > 0 for r in R)}")
        print(f"   one-step error quantiles (50/90/max): {np.quantile([min(r['one_step_max'], 1e3) for r in R], [0.5, 0.9, 1.0])}")
        print(f"   rollout error quantiles (50/90/max): {np.quantile([min(r['rollout_max'], 1e3) for r in R if r['rollout_max'] is not None], [0.5, 0.9, 1.0])}")
        for arm in LD.ARMS:
            for s in LD.SEEDS:
                tag = f"{arm}-s{s}"
                rec_ = json.loads((LD.OUT / rung / f"{tag}.json").read_text()) if (LD.OUT / rung / f"{tag}.json").exists() else None
                if rec_ is None:
                    continue
                per = []
                for rnd in rec_["rounds"]:
                    n_new = sum(r["passes"] for r in rows if r["tag"] == tag and r["round"] == rnd["round"])
                    per.append(f"{rnd['test_solved']} -> {n_new}")
                best = max(sum(r["passes"] for r in rows if r["tag"] == tag and r["round"] == rnd["round"]) for rnd in rec_["rounds"])
                nb = sum(v for k, v in bounded.items() if k.startswith(f"{rung}-s{s}-")) if FULL else \
                    sum(1 for k in bounded if k.startswith(f"{rung}-s{s}-"))      # judged up to escape: every held-out system counts
                print(f"   {tag}: held-out solved per round, old -> new definition: {' / '.join(per)} (of {nb} bounded held-out systems) -> "
                      f"{'still mastered' if nb and best >= 0.8 * nb else 'NOT mastered under the new definition'}")


if __name__ == "__main__":
    main()
