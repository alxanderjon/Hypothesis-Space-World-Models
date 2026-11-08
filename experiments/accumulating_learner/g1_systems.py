"""Stage 1 of the general learner: whole systems exact (reported alongside the bars). For every test system whose equations
were all solved in a test evaluation: assemble the model from each equation's simplest plausibly exact structure (the
step-integral fit), polish it on the step-to-step map of the training bursts (Levenberg-Marquardt on all coefficients
and constants, the learner's own converged substep count), check it on the held-back bursts (internal acceptance:
error <= discovery_learner.VAL_TOL in the learner's coordinates), then the external test (harness27.evaluate: one-step
error <= 1e-12 of the scale on 1,000 held-out reference states, and the multi-step envelope), with the artifact's substep
count (NSUB_MARGIN x the converged count). Prints counts only. Usage: g1_systems.py <workers> <test result .json> ..."""

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYS = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"


class L1Model:
    """dz/dt (per sampling interval) = sum_k c_k t_k(z; theta_k) per equation, in the learner's (uncentred) coordinates."""

    def __init__(self, sols):
        import torch
        self.eqs = [[(k, list(th)) for k, th in zip(s["keys"], s["thetas"])] for s in sols]
        p = []
        for s, eq in zip(sols, self.eqs):
            for (k, th), c in zip(eq, s["coefs"]):
                p += [float(c)] + [float(v) for v in th]
        self.p = torch.tensor(p, dtype=torch.float64)

    def g(self, z, p=None):
        import torch
        import fitter_g1 as FG
        p = self.p if p is None else p
        cols, j = [], 0
        for eq in self.eqs:
            f = torch.zeros(z.shape[0], dtype=z.dtype)
            for k, th in eq:
                c = p[j]
                th_t = p[j + 1:j + 1 + len(th)]
                j += 1 + len(th)
                f = f + c * FG.info(k).fn(z, th_t if len(th) else None)
            cols.append(f)
        return torch.stack(cols, 1)


def nsub_of(model, z, max_nsub=128):
    import torch
    import discovery_learner as DL
    n, last = 1, float("inf")
    with torch.no_grad():
        prev = DL.rk8_torch(model.g, z, n)
        if not bool(torch.isfinite(prev).all()):
            return None
        while n < max_nsub:
            nxt = DL.rk8_torch(model.g, z, 2 * n)
            diff = float((nxt - prev).abs().max())
            if not math.isfinite(diff):
                return None
            if diff <= DL.NSUB_TOL or (diff <= 1e-13 and diff > last / 8):
                return n
            n, prev, last = 2 * n, nxt, diff
    return max_nsub


def polish(model, z0, z1, iters=40):
    """Levenberg-Marquardt on the step-to-step map (all coefficients and constants)."""
    import torch
    import discovery_learner as DL
    n = nsub_of(model, DL.spread(z0))
    if n is None:
        return model, float("inf"), None
    res = lambda p: (DL.rk8_torch(lambda zz: model.g(zz, p), z0, n) - z1).reshape(-1)
    p = model.p.clone()
    with torch.no_grad():
        r = res(p)
    if not bool(torch.isfinite(r).all()):
        return model, float("inf"), n
    lam = 1e-6
    for _ in range(iters):
        J = torch.autograd.functional.jacobian(res, p)
        if not bool(torch.isfinite(J).all()):
            break
        H, g = J.T @ J, J.T @ r
        D = torch.diag(torch.diag(H) + 1e-300)
        improved = False
        for _try in range(8):
            try:
                step = torch.linalg.solve(H + lam * D, -g)
            except RuntimeError:
                lam *= 10
                continue
            with torch.no_grad():
                rn = res(p + step)
            if bool(torch.isfinite(rn).all()) and float(rn.norm()) < float(r.norm()):
                gain = 1 - float(rn.norm()) / max(float(r.norm()), 1e-300)
                p, r, lam, improved = p + step, rn, max(lam * 0.3, 1e-15), True
                break
            lam *= 10
        if not improved or gain < 1e-4 or float(r.abs().max()) < 1e-16:
            break
    model.p = p.detach()
    return model, float(r.abs().max()), n


def job(args):
    rec, sols = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import discovery_learner as DL
    import domains_g1 as DG
    import harness27 as H27
    import practice_net as PN
    out = {"id": rec["id"]}
    t0 = time.time()
    try:
        view = PN.learner_view(np.array(rec["bursts"]), seed=1)
        model = L1Model(sols)
        model, fit_err, n = polish(model, view["z0"], view["z1"])
        zz = torch.cat([DL.spread(view["z0"]), view["zv0"]])
        n_conv = nsub_of(model, zz)
        if n_conv is None:
            out.update(status="unusable", accepted=False, exact=False)
            return out
        n_art = DL.NSUB_MARGIN * n_conv
        with torch.no_grad():
            val = float((DL.rk8_torch(model.g, view["zv0"], n_art) - view["zv1"]).abs().max())
        out.update(fit_error=fit_err, val=val, accepted=bool(val <= DL.VAL_TOL), nsub=n_art)
        syn = DG.build(rec["desc"])                       # the same system, rebuilt (deterministic)
        b = np.ascontiguousarray(syn.bursts(8), dtype=np.float64)
        out["identical"] = bool(np.array_equal(b, np.array(rec["bursts"])))
        sig = view["sig"].numpy()

        def roll(x0, steps):
            z = torch.tensor(np.asarray(x0) / sig)
            xs = [np.asarray(x0, dtype=float)]
            with torch.no_grad():
                for _ in range(steps):
                    z = DL.rk8_torch(model.g, z, n_art)
                    xs.append(z.numpy() * sig)
            return np.stack(xs, 1)
        ev = H27.evaluate(syn.ref, None, None, rollout_hat=roll)
        out.update(status="done", one_step=ev["one_step_max_scaled_error"], multi_step_ok=ev["multi_step_ok"], exact=ev["exact"])
    except Exception as ex:
        out.update(status="error", error=type(ex).__name__, accepted=False, exact=False)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    workers = int(sys.argv[1])
    sys.path.insert(0, str(HERE))
    for path in sys.argv[2:]:
        path = Path(path)
        res = json.loads(path.read_text())
        letter = path.stem.rsplit("_", 1)[1]
        dom = {"A": "mechanics", "B": "circuits", "C": "ecology"}.get(letter, letter)   # Stage 2a files end in the domain
        for d2 in ("reactions_L1", "reactions_L2", "reactions_L3", "flows_L1", "flows_L2", "flows_L3"):
            if path.stem.endswith("_" + d2):                     # Stage 2a'' domain names hold an underscore
                dom = d2
        recs = {r["id"]: r for r in json.loads((SYS / f"test_{dom}.json").read_text())}
        jobs = [(recs[r["id"]], [e["solution"] for e in r["equations"]]) for r in res
                if r["status"] == "done" and all(e["solution"] for e in r["equations"])]
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            out = list(ex.map(job, jobs))
        (path.parent / f"{path.stem}_systems.json").write_text(json.dumps(out, default=float))
        print(f"{path.stem}: systems with every equation solved {len(jobs)} of {len(res)}; accepted internally "
              f"{sum(o.get('accepted', False) for o in out)}; exact (one-step and multi-step) {sum(o.get('exact', False) for o in out)}; "
              f"errors {sum(o['status'] == 'error' for o in out)}; rebuilt systems identical {sum(o.get('identical', False) for o in out)}",
              flush=True)
