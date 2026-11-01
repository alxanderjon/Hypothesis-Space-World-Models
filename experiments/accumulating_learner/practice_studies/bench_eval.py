# PRACTICE half only: high-precision finish and external exactness test for PySR's equations.
# For each (system, M): candidate equations from each component's Pareto front (lowest losses) -> constants refitted by
# Levenberg-Marquardt on the step-to-step map (8th-order RK, float64), exactly like our learner's polish -> held-back
# check (same split, 3e-13) -> refit on all bursts -> compiled artifact -> the harness's in-range exactness test.
import glob, json, math, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

warnings.filterwarnings("ignore")
SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
DATA, PYSR, OUT = f"{SP}/bench/data", f"{SP}/bench/pysr", f"{SP}/bench/eval"
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")
VAL_TOL = 3e-13


def parametrize(expr, d):
    """Numeric constants of a SymPy expression -> parameters (values returned); exact integers stay fixed."""
    import sympy as sp
    consts, k = {}, 0
    for a in sorted(expr.atoms(sp.Float), key=lambda v: str(v)):
        consts[a] = sp.Symbol(f"p{len(consts)}")
    e = expr.xreplace(consts)
    return e, [float(c) for c in consts], [consts[c] for c in consts]


def build_g(exprs_p, zs, psyms):
    import sympy as sp
    import torch
    fns = [sp.lambdify(list(zs) + list(ps), e, modules=[{"Abs": torch.abs, "exp": torch.exp, "tanh": torch.tanh,
                                                          "sin": torch.sin, "sqrt": torch.sqrt}, "math"])
           for e, ps in zip(exprs_p, psyms)]
    sizes = [len(ps) for ps in psyms]

    def g(z, p):
        out, k = [], 0
        for f, n in zip(fns, sizes):
            args = [z[..., j] for j in range(z.shape[-1])] + [p[k + q] for q in range(n)]
            v = f(*args)
            out.append(v * torch.ones_like(z[..., 0]) if not torch.is_tensor(v) or v.dim() == 0 else v)
            k += n
        return torch.stack(out, -1)
    return g


def finish(tag):
    import sympy as sp
    import torch
    torch.set_num_threads(1)
    import discovery_learner as DL, harness27 as H27, compile_artifact as CA
    out = f"{OUT}/{tag}.json"
    if os.path.exists(out):
        return json.load(open(out))
    t0 = time.time()
    rec = json.load(open(f"{PYSR}/{tag}.json"))
    D = np.load(f"{DATA}/{tag}.npz")
    d = rec["d"]
    name, M = tag.split("_M")[0], int(tag.split("_M")[1])
    z0, z1, zv0, zv1 = (torch.as_tensor(D[k]) for k in ("z0", "z1", "zv0", "zv1"))
    zs = sp.symbols(" ".join(f"z{j}" for j in range(d)))
    zs = zs if isinstance(zs, tuple) else (zs,)
    res = {"tag": tag, "name": name, "M": M, "pysr_seconds": rec["seconds"], "status": "no_model", "exact": False}
    cands = []
    for i, fr in enumerate(rec["fronts"]):
        if isinstance(fr, dict):
            cands.append([]); continue
        # candidates across the front (the simplest model exact on held-back data is the goal): the simplest member
        # within 10x and within 1000x of the best fit, the most accurate member, and the best accuracy-per-size elbow
        fr = sorted(fr, key=lambda r: r["complexity"])
        lmin = min(r["loss"] for r in fr)
        pick = []
        for fac in (10.0, 1e3):
            pick.append(next(r for r in fr if r["loss"] <= fac * lmin))
        pick.append(min(fr, key=lambda r: r["loss"]))
        sc = [(math.log(max(fr[k - 1]["loss"], 1e-300)) - math.log(max(fr[k]["loss"], 1e-300))) /
              max(fr[k]["complexity"] - fr[k - 1]["complexity"], 1) for k in range(1, len(fr))]
        if sc:
            pick.append(fr[1 + int(np.argmax(sc))])
        seen, uniq = set(), []
        for r in pick:
            if r["sympy"] not in seen:
                seen.add(r["sympy"]); uniq.append(r)
        # screening like our learner's: the rate data are accurate to their estimated noise, so a candidate missing them
        # by more than 100x that level cannot be exact and is not refitted
        Gi = D["G"][:, i]
        rms = float(np.sqrt(np.mean(Gi ** 2))) + 1e-300
        tol = max(3.0 * float(D["noise"][i]) / rms, 1e-12)
        # constants refitted precisely on the rate data first (PySR's own constant fit stops at a few digits), then the
        # noise-level screen: a candidate missing the rate data by more than 100x their noise cannot be exact
        kept, rels = [], []
        Zc_t, Gi_t = torch.as_tensor(D["Zc"]), torch.as_tensor(Gi)
        for r in uniq:
            e = sp.sympify(r["sympy"], locals={f"z{j}": zs[j] for j in range(d)})
            e_p, vals, ps = parametrize(e, d)
            rel = math.sqrt(max(r["loss"], 0.0)) / rms
            if vals:
                try:
                    gi = build_g([e_p], zs, [ps])
                    f_r = lambda q: gi(Zc_t, q)[:, 0] - Gi_t
                    q = torch.tensor(vals, dtype=torch.float64); rr = f_r(q); lam = 1e-6
                    t_c = time.time()
                    for _ in range(30):
                        if time.time() - t_c > 20 or len(vals) > 40:     # cap: pathological formulas
                            break
                        J = torch.func.jacfwd(f_r)(q)
                        if not torch.isfinite(J).all():
                            break
                        H = J.T @ J
                        step = torch.linalg.solve(H + lam * torch.diag(torch.diag(H) + 1e-30), -(J.T @ rr))
                        rn = f_r(q + step)
                        if torch.isfinite(rn).all() and float(rn.norm()) <= float(rr.norm()):
                            q, rr, lam = q + step, rn, max(lam * 0.1, 1e-12)
                        else:
                            lam *= 10
                            if lam > 1e10:
                                break
                    rel = min(rel, float(rr.norm()) / (float(Gi_t.norm()) + 1e-300))
                    e = e_p.xreplace({ps[k]: float(q[k]) for k in range(len(ps))})
                except Exception:
                    pass
            rels.append(rel)
            if rel <= 100 * tol:
                kept.append(e)
        res.setdefault("screen", []).append({"component": i, "best_rel": min(rels) if rels else None, "tol": tol, "kept": len(kept)})
        cands.append(kept)
    if any(not c for c in cands):
        res["status"] = "rejected"; res["reason"] = "rate-data misfit far above the noise level in some component"
        res["finish_seconds"] = time.time() - t0
        json.dump(res, open(out, "w"), default=float); return res
    combos = [[a] * d for a in range(4)] + [[int(j == i) * a for j in range(d)] for a in (1, 2, 3) for i in range(d)]
    combos = [c for k, c in enumerate(combos) if c not in combos[:k]][:8]
    best = None
    for combo in combos:
        if any(a >= len(cands[i]) for i, a in enumerate(combo)):
            continue
        exprs = [cands[i][a] for i, a in enumerate(combo)]
        parts = [parametrize(e, d) for e in exprs]
        g = build_g([p[0] for p in parts], zs, [p[2] for p in parts])
        p = torch.tensor([v for q in parts for v in q[1]], dtype=torch.float64)

        class M_:                                                     # minimal model interface for DL's integrator
            pass
        try:
            nsub = 1
            with torch.no_grad():
                prev = DL.rk8_torch(lambda z: g(z, p), DL.spread(z0), 1)
                for nsub_try in (2, 4, 8, 16, 32, 64):
                    nxt = DL.rk8_torch(lambda z: g(z, p), DL.spread(z0), nsub_try)
                    if float((nxt - prev).abs().max()) <= 1e-13:
                        break
                    prev, nsub = nxt, nsub_try
            nsub = 4 * nsub
            f_res = lambda q: (DL.rk8_torch(lambda z: g(z, q), z0, nsub) - z1).reshape(-1)
            r = f_res(p)
            if not torch.isfinite(r).all():
                continue
            lam = 1e-6
            if len(p) > 40:                                           # too many constants to refit precisely
                continue
            t_c = time.time()
            for _ in range(30):                                       # Levenberg-Marquardt on the step-to-step map
                if time.time() - t_c > 300:                           # cap per candidate combination
                    break
                J = torch.func.jacfwd(f_res)(p)
                if not torch.isfinite(J).all():
                    break
                H = J.T @ J
                step = torch.linalg.solve(H + lam * torch.diag(torch.diag(H) + 1e-30), -(J.T @ r))
                rn = f_res(p + step)
                if torch.isfinite(rn).all() and float(rn.norm()) <= float(r.norm()):
                    p, r, lam = p + step, rn, max(lam * 0.1, 1e-12)
                    if float(step.abs().max()) <= 1e-14 * (float(p.abs().max()) + 1e-30):
                        break
                else:
                    lam *= 10
                    if lam > 1e10:
                        break
            with torch.no_grad():
                val = float((DL.rk8_torch(lambda z: g(z, p), zv0, nsub) - zv1).abs().max())
        except Exception:
            continue
        if math.isfinite(val) and (best is None or val < best[0]):
            best = (val, combo, exprs, parts, p.clone(), nsub, g)
        if val <= VAL_TOL:
            break
    if best is None:
        json.dump(res, open(out, "w")); return res
    val, combo, exprs, parts, p, nsub, g = best
    res.update(status="accepted" if val <= VAL_TOL else "rejected", val=val, combo=combo, nsub=nsub,
               equations=[str(e) for e in exprs])
    if val <= VAL_TOL:                                                 # refit on all bursts, then the external test
        za0, za1 = torch.cat([z0, zv0]), torch.cat([z1, zv1])
        f_all = lambda q: (DL.rk8_torch(lambda z: g(z, q), za0, nsub) - za1).reshape(-1)
        try:
            r = f_all(p); lam = 1e-6
            for _ in range(30):
                J = torch.func.jacfwd(f_all)(p)
                H = J.T @ J
                step = torch.linalg.solve(H + lam * torch.diag(torch.diag(H) + 1e-30), -(J.T @ r))
                rn = f_all(p + step)
                if torch.isfinite(rn).all() and float(rn.norm()) <= float(r.norm()):
                    p, r, lam = p + step, rn, max(lam * 0.1, 1e-12)
                else:
                    lam *= 10
                    if lam > 1e10:
                        break
        except Exception:
            pass
        mu, sig, dt = D["mu"], D["sig"], float(D["dt"])
        xs = sp.symbols(" ".join(f"x{j}" for j in range(d)))
        xs = xs if isinstance(xs, tuple) else (xs,)
        k, xexprs = 0, []
        for i, (e_p, vals, ps) in enumerate(parts):
            sub = {ps[q]: float(p[k + q]) for q in range(len(ps))}
            k += len(ps)
            zsub = {zs[j]: (xs[j] - float(mu[j])) / float(sig[j]) for j in range(d)}
            xexprs.append(sp.Float(float(sig[i]) / dt) * e_p.xreplace(sub).xreplace(zsub))
        try:
            ref = H27.Reference27(name, seed=1)
            f_true, roll = CA.build_true(xexprs)
            ev = H27.evaluate(ref, None, nsub, rollout_hat=lambda x0, steps: roll(x0, steps, nsub, ref.sys.dt))
            res["exact"], res["one"] = bool(ev["exact"]), ev["one_step_max_scaled_error"]
        except Exception as e:
            res["eval_error"] = repr(e)[:200]
    res["finish_seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), default=float)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tags = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{PYSR}/*.json"))
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for f in as_completed([ex.submit(finish, t) for t in tags]):
            r = f.result()
            print(f"{r['tag']}: {r['status']} val {r.get('val', float('nan')):.2e} exact {r['exact']} "
                  f"(pysr {r.get('pysr_seconds', 0):.0f}s + finish {r.get('finish_seconds', 0):.0f}s)", flush=True)
