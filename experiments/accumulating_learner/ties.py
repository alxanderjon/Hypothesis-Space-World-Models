"""Ties between equations (option (b), 1 November): a shared capability of every arm, the learner from scratch included.

A fitted model often contains the same piece in several terms (one variable's loss is another's gain, so one rate law
appears in two equations), or the same term in two equations with a simple physical coefficient ratio (+-1, 2, 3, 4,
1/2, 1/3, 1/4). Tying them makes one constant serve several equations' data: fewer unknowns, better-determined constants
(the ceiling check: 114 at 8 bursts is exact only with ties). Ties are detected on a fitted model with a loose tolerance,
imposed exactly, refitted on the step-to-step map, and kept only if the learner's own held-back check accepts the tied
model. No truth is used.
"""

import math

import torch

import discovery_learner as DL

SIMPLE = (1.0, 2.0, 3.0, 4.0, 0.5, 1 / 3, 0.25)


def detect(model, Z, tol=1e-2):
    """Ties of a fitted model on scaled states Z. Returns (rewritten model, owner map, info).
    Pieces (same family and variable) whose values agree up to a constant factor within `tol` (relative spread of the
    ratio on the data) are rewritten to share the first piece's constants, the factor moving into the term's coefficient.
    Equal terms in different equations whose physical coefficient ratio is within `tol` of a simple number share one
    coefficient. owner maps a dependent position of params() to (representative position, multiplier)."""
    m = model.copy()
    X = Z * m.sig + m.mu
    facs = [(i, a, k) for i, terms in enumerate(m.supports) for a, t in enumerate(terms) for k, f in enumerate(t.factors)
            if f.theta and f.family not in ("cheb", "libcurve") and not isinstance(f.var, tuple)]
    with torch.no_grad():
        vals = {x: DL.factor_value(m.supports[x[0]][x[1]].factors[x[2]], Z, X,
                                   torch.tensor(m.supports[x[0]][x[1]].factors[x[2]].theta)) for x in facs}
    groups, used = [], set()
    for x in facs:
        if x in used:
            continue
        fx, g = m.supports[x[0]][x[1]].factors[x[2]], [x]
        for y in facs:
            if y in used or y == x:
                continue
            fy = m.supports[y[0]][y[1]].factors[y[2]]
            if fy.family != fx.family or fy.var != fx.var or len(fy.theta) != len(fx.theta):
                continue
            ratio = vals[y] / vals[x]
            if torch.isfinite(ratio).all() and float(ratio.std() / (ratio.abs().mean() + 1e-300)) < tol:
                g.append(y)
        if len(g) > 1:
            used |= set(g)
            groups.append(g)
    for g in groups:                                     # rewrite members to the first member's constants
        i0, a0, k0 = g[0]
        f0 = m.supports[i0][a0].factors[k0]
        for (i, a, k) in g[1:]:
            t = m.supports[i][a]
            ratio = float((vals[(i, a, k)] / vals[g[0]]).mean())
            fl = list(t.factors)
            fl[k] = DL.Factor(f0.family, f0.var, f0.theta, t.factors[k].meta)
            m.supports[i][a] = DL.Term(t.mono, tuple(fl), t.level)
            m.coefs[i] = m.coefs[i].clone()
            m.coefs[i][a] = m.coefs[i][a] * ratio
    pos, k = {}, 0
    for i, (terms, c) in enumerate(zip(m.supports, m.coefs)):
        for a in range(len(c)):
            pos[("c", i, a)] = k + a
        k += len(c)
        for a, t in enumerate(terms):
            for fi, f in enumerate(t.factors):
                for q in range(len(f.theta)):
                    pos[("t", i, a, fi, q)] = k
                    k += 1
    owner = {}
    for g in groups:
        i0, a0, k0 = g[0]
        for (i, a, kk) in g[1:]:
            for q in range(len(m.supports[i][a].factors[kk].theta)):
                owner[pos[("t", i, a, kk, q)]] = (pos[("t", i0, a0, k0, q)], 1.0)
    sigs = {}
    for i, terms in enumerate(m.supports):
        for a, t in enumerate(terms):
            key = (t.mono, tuple((f.family, f.var, tuple(round(v, 12) for v in f.theta)) for f in t.factors))
            sigs.setdefault(key, []).append((i, a))
    sg = m.sig.numpy()
    n_coef = 0
    for members in sigs.values():
        i0, a0 = members[0]
        c0 = float(m.coefs[i0][a0]) * sg[i0]
        if c0 == 0:
            continue
        for (i, a) in members[1:]:
            if i == i0:
                continue
            r = float(m.coefs[i][a]) * sg[i] / c0
            hit = next((sgn * s for s in SIMPLE for sgn in (1.0, -1.0) if abs(r - sgn * s) <= tol * s), None)
            if hit is not None:
                owner[pos[("c", i, a)]] = (pos[("c", i0, a0)], hit * sg[i0] / sg[i])
                n_coef += 1
    return m, owner, {"piece_ties": sum(len(g) - 1 for g in groups), "coef_ties": n_coef}


def tie_matrix(model, owner):
    """full = base + T q over params(): frozen constants in base, tied ones expressed through their representative."""
    p = model.params()
    frozen = set(model.frozen())
    reps = [j for j in range(len(p)) if j not in frozen and j not in owner]
    col = {j: c for c, j in enumerate(reps)}
    T = torch.zeros(len(p), len(reps), dtype=torch.float64)
    for j in range(len(p)):
        if j in frozen:
            continue
        if j in owner:
            rj, mult = owner[j]
            while rj in owner:
                rj, m2 = owner[rj]
                mult *= m2
            if rj in col:
                T[j, col[rj]] = mult
        else:
            T[j, col[j]] = 1.0
    base = torch.where(T.abs().sum(1) > 0, torch.zeros_like(p), p)
    return base, T


def fit_tied(model, z0, z1, base, T, iters=40):
    """Levenberg-Marquardt on the step-to-step map in tied coordinates (as DL.polish). In DL's precise final stage the
    steps are scaled least squares, iterations run longer, and the fit is redone if the final substep count changes."""
    nsub = DL.self_nsub(model, DL.spread(z0), max_nsub=16 if not DL.PRECISE else 512)
    if nsub is None or T.shape[1] == 0:
        return model, nsub or 1, float("inf")
    full = lambda q: base + T @ q
    q = torch.linalg.lstsq(T, model.params() - base).solution
    n2 = nsub
    for _round in range(3 if DL.PRECISE else 1):
        f_res = lambda q_, n_=nsub: (DL.rk8_torch(lambda zz: model.g(zz, full(q_)), z0, n_) - z1).reshape(-1)
        r, lam = f_res(q), 1e-6
        if not torch.isfinite(r).all():
            return model, nsub, float("inf")
        for _ in range(iters if not DL.PRECISE else 5 * iters):
            J = torch.func.jacfwd(f_res)(q)
            if not torch.all(torch.isfinite(J)):
                break
            try:
                step = DL.lm_step(J, r, lam)
            except RuntimeError:
                lam *= 10
                continue
            rn = f_res(q + step)
            if torch.isfinite(rn).all() and float(rn.norm()) <= float(r.norm()):
                done = float(step.abs().max()) <= 1e-14 * (float(q.abs().max()) + 1e-30) or float(rn.abs().max()) < 1e-15
                q, r, lam = q + step, rn, max(lam * 0.1, 1e-12)
                if done:
                    break
            else:
                lam *= 10
                if lam > 1e10:
                    break
        model.set_params(full(q))
        n2 = DL.self_nsub(model, DL.spread(z0))
        if n2 is None:
            return model, nsub, float("inf")
        if n2 == nsub:
            break
        nsub = n2
    with torch.no_grad():
        err = float((DL.rk8_torch(model.g, z0, n2) - z1).abs().max())
    return model, n2, err


def finalize(model, z0, z1, zv0, zv1, tol=1e-2):
    """The shared final step of every arm: the model as is, or tied if the tied refit passes the held-back check.
    Returns (model, held-back error, info). A tied model is preferred when both pass (fewer unknowns); otherwise the
    lower held-back error wins. The chosen model is refitted on all bursts (keeping its ties), kept unless worse."""
    val0, _, _ = DL.converged_val(model, z0, zv0, zv1)
    best, best_val, info, tie = model, val0, {"tied": False}, None
    try:
        mt, owner, ti = detect(model, torch.cat([z0, zv0]), tol=tol)
        if owner:
            base, T = tie_matrix(mt, owner)
            mt, _, _ = fit_tied(mt, z0, z1, base, T)
            vt, _, _ = DL.converged_val(mt, z0, zv0, zv1)
            ok_t, ok_0 = vt <= DL.VAL_TOL, val0 <= DL.VAL_TOL
            if ok_t or (not ok_0 and vt < val0):          # tied preferred when it passes (fewer unknowns)
                best, best_val, tie = mt, vt, (base, T)
                info = {"tied": True, **ti, "val_untied": val0}
            else:
                info = {"tied": False, "tie_candidates": ti, "val_tied": vt}
    except (RuntimeError, ValueError) as e:
        info = {"tied": False, "tie_error": repr(e)[:120]}
    if best_val <= DL.VAL_TOL:                           # final refit on all bursts, kept unless worse (as learn())
        za0, za1 = torch.cat([z0, zv0]), torch.cat([z1, zv1])
        try:
            n_prev = DL.self_nsub(best, DL.spread(za0))
            with torch.no_grad():
                err_prev = float((DL.rk8_torch(best.g, za0, n_prev) - za1).abs().max()) if n_prev else float("inf")
            if tie is None:
                m_all, _, err_all, _ = DL.polish(best.copy(), za0, za1, iters=30)
            else:
                m_all, _, err_all = fit_tied(best.copy(), za0, za1, tie[0], tie[1], iters=30)
            if math.isfinite(err_all) and err_all <= max(DL.VAL_TOL, err_prev):
                best = m_all
        except (RuntimeError, ValueError):
            pass
    return best, best_val, info
