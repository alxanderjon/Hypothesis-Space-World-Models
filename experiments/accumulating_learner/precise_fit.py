"""High-precision fit of an accepted structure's constants (owner, 23 November: "add a final high-precision fit of the
constants after a structure is accepted, aiming for one-step error of 1e-12 or better"). Shooting: the candidate model
(every equation's coefficients and constants together) is integrated from each recorded state over one sample interval
(DOP853, rtol 1e-14) and compared with the next recorded state; Levenberg-Marquardt on all parameters from the accepted
fit. The recordings come from an 8th-order Runge-Kutta reference converged to 1e-14 per step (harness27.choose_nsub), so
the fit is limited by them, not by quadrature or interpolation. Model in physical units:
dx_i/dt = sigma_i / dt * sum_k c_ik phi_k(x / sigma; theta_k)  (the learner's coordinates and sample-step time)."""

import numpy as np


def compile_terms(keys, d):
    """Vectorized numpy functions phi(X (n, d) scaled, theta) for each key."""
    import sympy as sp
    import language_g1 as LG
    X = LG.variables(4)[:d]
    K = [sp.Symbol(f"K{a}", real=True) for a in range(3)]
    out = []
    for k in keys:
        e = LG.expr_of(k)
        ks = [s for s in K if s in e.free_symbols]
        f = sp.lambdify(list(X) + ks, e, "numpy")
        out.append((f, len(ks)))
    return out


class Model:
    def __init__(self, terms_per_eq, sig, dt):
        """terms_per_eq: per equation, [(key, theta tuple)]; parameters: per equation its coefficients then its thetas."""
        self.d = len(terms_per_eq)
        self.sig, self.dt = np.asarray(sig, float), float(dt)
        self.fns = [compile_terms([k for k, _ in eq], self.d) for eq in terms_per_eq]
        self.shape = [(len(eq), [len(th) for _, th in eq]) for eq in terms_per_eq]

    def pack(self, coefs_per_eq, terms_per_eq):
        p = []
        for c, eq in zip(coefs_per_eq, terms_per_eq):
            p += list(np.asarray(c, float))
            for _, th in eq:
                p += list(th)
        return np.array(p, float)

    def unpack(self, p):
        out, j = [], 0
        for n, ks in self.shape:
            c = p[j:j + n]; j += n
            th = []
            for k in ks:
                th.append(p[j:j + k]); j += k
            out.append((c, th))
        return out

    def rhs(self, x, p):
        """x (n, d) physical -> (n, d)."""
        W = np.atleast_2d(x) / self.sig
        cols = [W[:, j] for j in range(self.d)]
        out = np.zeros_like(W)
        with np.errstate(all="ignore"):
            for i, ((c, th), fns) in enumerate(zip(self.unpack(p), self.fns)):
                v = np.zeros(len(W))
                for ck, t, (f, nk) in zip(c, th, fns):
                    v = v + ck * np.broadcast_to(f(*cols, *t), (len(W),))
                out[:, i] = self.sig[i] * v / self.dt
        return out


MAX_CALLS = 20_000                                               # 24 Nov: a stiff candidate cannot stall the fit


class TooLong(Exception):
    pass


def shoot(model, p, x0, dt):
    """One sample interval from every state of x0 (n, d), all at once (NaN if it needs more than MAX_CALLS calls)."""
    from scipy.integrate import solve_ivp
    n, d = x0.shape
    scale = max(np.abs(x0).max(), 1e-12)
    calls = [0]

    def f(t, y):
        calls[0] += 1
        if calls[0] > MAX_CALLS:
            raise TooLong()
        return model.rhs(y.reshape(n, d), p).ravel()
    try:
        sol = solve_ivp(f, (0, dt), x0.ravel(), method="DOP853", rtol=1e-14, atol=1e-16 * scale)
    except TooLong:
        return np.full_like(x0, np.nan)
    if not sol.success:
        return np.full_like(x0, np.nan)
    return sol.y[:, -1].reshape(n, d)


def precise(terms_per_eq, coefs_per_eq, bursts, sig, dt, max_nfev=60):
    """Returns (terms_per_eq with refined thetas, coefs_per_eq refined, relative one-step residual before, after)."""
    from scipy.optimize import least_squares
    B = np.asarray(bursts, float)                                # (M, L, d)
    x0, x1 = B[:, :-1].reshape(-1, B.shape[-1]), B[:, 1:].reshape(-1, B.shape[-1])
    dx = x1 - x0
    norm = np.abs(dx).max(0) + 1e-300                            # per variable, the scale of a step's change
    model = Model(terms_per_eq, sig, dt)
    p0 = model.pack(coefs_per_eq, terms_per_eq)

    def res(p):
        r = (shoot(model, p, x0, dt) - x1) / norm
        return np.nan_to_num(r.ravel(), nan=1e6)
    r0 = res(p0)
    try:
        if not np.all(np.abs(r0) < 1e5):                         # the accepted fit cannot even be shot: keep it
            raise ValueError("unshootable")
        fit = least_squares(res, p0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev * len(p0), x_scale="jac")
        p = fit.x if np.linalg.norm(fit.fun) <= np.linalg.norm(r0) else p0
    except Exception:
        p = p0
    r1 = res(p)
    out_terms, out_coefs = [], []
    for (c, th), eq in zip(model.unpack(p), terms_per_eq):
        out_terms.append([(k, tuple(t)) for (k, _), t in zip(eq, th)])
        out_coefs.append(np.asarray(c))
    rel = lambda r: float(np.sqrt(np.mean(r ** 2)))
    return out_terms, out_coefs, rel(r0), rel(r1), model, p
