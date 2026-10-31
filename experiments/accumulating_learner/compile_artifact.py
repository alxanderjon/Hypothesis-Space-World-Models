"""Compile a learned model (the artifact) to native code: its vector field plus its own fixed-step 8th-order
Runge-Kutta integrator, generated as C and built with gcc. The compiled artifact needs nothing but the learned
equations; it never calls the simulator. Used for the cost comparison against the original simulation."""

import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as D

CACHE = Path(tempfile.gettempdir()) / "trial26_artifacts"


def _num(v):
    return f"({float(v):.17g})"


_PRE = []        # static declarations collected while generating one source (curve coefficient tables)


def _compact_c(fam, v, th, o):
    if fam == "crecip":
        return f"(1.0/({v}))"
    if fam == "cexp":
        return f"exp({v})"
    if fam == "ctanh":
        return f"tanh({v})"
    if fam == "csin":
        return f"sin({v})"
    if fam == "cabs":
        return f"fabs({v})"
    if fam == "cpow":
        return f"pow({v},{_num(th[o + 2])})"
    if fam == "chill":
        return f"(pow({v},{_num(th[o + 2])})/(1.0+pow({v},{_num(th[o + 2])})))"
    if fam == "chillr":
        return f"(1.0/(1.0+pow({v},{_num(th[o + 2])})))"
    raise ValueError(fam)


def _factor_c(f, z, x):
    fam, var, th = f[0], f[1], f[2]
    meta = f[3] if len(f) > 3 else []
    if fam in ("cheb", "libcurve", "crecip", "cexp", "ctanh", "csin", "cabs", "cpow", "chill", "chillr"):
        if isinstance(var, (list, tuple)):
            u, o = f"({z}{var[0]}+{_num(th[0])}*{z}{var[1]})", 1
        else:
            u, o = f"{z}{var}", 0
        if fam in ("cheb", "libcurve"):
            if fam == "cheb":
                lo, hi = meta[0], meta[1]
                t = f"(2.0*({u}-{_num(lo)})/({_num(hi)}-{_num(lo)})-1.0)"
                coefs = th[o:]
            else:
                t = f"({_num(th[o])}*{u}+{_num(th[o + 1])})"
                coefs = meta[2:]
            name = f"C{len(_PRE)}"
            _PRE.append(f"static const double {name}[{len(coefs)}] = {{" + ",".join(f"{float(c):.17g}" for c in coefs) + "};")
            return f"cheb({t},{name},{len(coefs)})"
        return _compact_c(fam, f"({_num(th[o])}*{u}+{_num(th[o + 1])})", th, o)
    zv, xv = f"{z}{var}", f"{x}{var}"
    if fam == "sin":
        return f"sin({_num(th[0])}*{zv}+{_num(th[1])})"
    if fam == "exp":
        return f"exp({_num(th[0])}*{zv})"
    if fam == "tanh":
        return f"tanh({_num(th[0])}*{zv}+{_num(th[1])})"
    if fam == "abs":
        return f"fabs({zv}-{_num(th[0])})"
    if fam == "recip":
        return f"(1.0/({_num(th[0])}+{zv}))"
    if fam == "pow":
        return f"pow({xv},{_num(th[0])})"
    if fam == "hill":
        return f"(pow({xv},{_num(th[1])})/({_num(th[0])}+pow({xv},{_num(th[1])})))"
    if fam == "hillr":
        return f"({_num(th[0])}/({_num(th[0])}+pow({xv},{_num(th[1])})))"
    raise ValueError(fam)


def c_source(desc, mu, sig, dt):
    d = len(desc)
    _PRE.clear()
    lines = ["static void g(const double* x, double* o) {"]
    for k in range(d):
        lines.append(f"  double z{k} = (x[{k}] - {_num(mu[k])}) / {_num(sig[k])}; double X{k} = x[{k}];")
    for i, comp in enumerate(desc):
        terms = []
        for t in comp:
            parts = [_num(t["coef"])]
            for k, e in enumerate(t["mono"]):
                parts += [f"z{k}"] * int(e)
            parts += [_factor_c(f, "z", "X") for f in t["factors"]]
            terms.append("*".join(parts))
        body = " + ".join(terms) if terms else "0.0"
        lines.append(f"  o[{i}] = ({body}) * {_num(sig[i])} / {_num(dt)};")
    lines.append("}")
    head = ["#include <math.h>", "#include <string.h>",
            "static double cheb(double t, const double* c, int n) { double T0 = 1.0, T1 = t, s = c[0];"
            " if (n > 1) s += c[1] * T1; for (int k = 2; k < n; k++) { double T2 = 2.0 * t * T1 - T0; s += c[k] * T2; T0 = T1; T1 = T2; }"
            " return s; }"] + list(_PRE)
    lines = head + lines
    S = D.N_STAGES
    Am = D.A[:S, :S]
    lines.append(f"static const double A[{S}][{S}] = {{" + ",".join("{" + ",".join(f"{v:.17g}" for v in row) + "}" for row in Am) + "};")
    lines.append(f"static const double Bw[{S}] = {{" + ",".join(f"{v:.17g}" for v in D.B) + "};")
    lines.append(f"""
void rollout(const double* x0, long n, long steps, long nsub, double* out) {{
  const int d = {d}; const double h = {_num(dt)} / (double)nsub;
  double x[{d}], xi[{d}], K[{S}][{d}];
  for (long t = 0; t < n; t++) {{
    memcpy(x, x0 + t * d, sizeof(double) * d);
    double* o = out + t * (steps + 1) * d;
    memcpy(o, x, sizeof(double) * d);
    for (long s = 0; s < steps; s++) {{
      for (long q = 0; q < nsub; q++) {{
        for (int i = 0; i < {S}; i++) {{
          for (int k = 0; k < d; k++) {{
            double acc = x[k];
            for (int j = 0; j < i; j++) if (A[i][j] != 0.0) acc += h * A[i][j] * K[j][k];
            xi[k] = acc;
          }}
          g(xi, K[i]);
        }}
        for (int k = 0; k < d; k++) {{
          double acc = 0.0;
          for (int i = 0; i < {S}; i++) if (Bw[i] != 0.0) acc += Bw[i] * K[i][k];
          x[k] += h * acc;
        }}
      }}
      memcpy(o + (s + 1) * d, x, sizeof(double) * d);
    }}
  }}
}}""")
    return "\n".join(lines)


def build(desc, mu, sig, dt):
    """Returns a callable rollout(x0 (n, d), steps, nsub) -> (n, steps+1, d)."""
    src = c_source(desc, mu, sig, dt)
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    so = CACHE / f"art_{key}.so"
    if not so.exists():
        c = CACHE / f"art_{key}.c"
        c.write_text(src)
        # -ffp-contract=off keeps IEEE arithmetic (no fused multiply-add reassociation) so results match float64 numpy
        subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(so), str(c), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    lib.rollout.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
    d = len(desc)

    def rollout(x0, steps, nsub):
        x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(-1, d))
        out = np.empty((x0.shape[0], steps + 1, d))
        lib.rollout(x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x0.shape[0], steps, nsub,
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out
    return rollout


# ---------------- compiled TRUE right-hand sides (reference simulation only; never given to the learner) ----------------

def c_source_sympy(exprs):
    """C source for a vectorised true right-hand side from SymPy expressions in symbols x0..x{d-1}, plus the same
    fixed-step 8th-order integrator used everywhere (so the compiled reference equals the Python reference up to rounding)."""
    import sympy as sp
    d = len(exprs)
    X = sp.symbols(" ".join(f"x{i}" for i in range(d)))
    X = X if isinstance(X, tuple) else (X,)
    sub = {X[i]: sp.Symbol(f"X[{i}]") for i in range(d)}
    lines = ["#include <math.h>", "#include <string.h>", "static void g(const double* X, double* o) {"]
    for i, e in enumerate(exprs):
        code = sp.ccode(sp.sympify(e).xreplace(sub), standard="c99")
        lines.append(f"  o[{i}] = {code};")
    lines.append("}")
    S = D.N_STAGES
    lines.append(f"static const double A[{S}][{S}] = {{" + ",".join("{" + ",".join(f"{v:.17g}" for v in row) + "}" for row in D.A[:S, :S]) + "};")
    lines.append(f"static const double Bw[{S}] = {{" + ",".join(f"{v:.17g}" for v in D.B) + "};")
    lines.append(f"""
void rhs(const double* x, long n, double* out) {{ for (long t = 0; t < n; t++) g(x + t * {d}, out + t * {d}); }}
void rollout(const double* x0, long n, long steps, long nsub, double h_total, double* out) {{
  const int d = {d}; const double h = h_total / (double)nsub;
  double x[{d}], xi[{d}], K[{S}][{d}];
  for (long t = 0; t < n; t++) {{
    memcpy(x, x0 + t * d, sizeof(double) * d);
    double* o = out + t * (steps + 1) * d;
    memcpy(o, x, sizeof(double) * d);
    for (long s = 0; s < steps; s++) {{
      for (long q = 0; q < nsub; q++) {{
        for (int i = 0; i < {S}; i++) {{
          for (int k = 0; k < d; k++) {{
            double acc = x[k];
            for (int j = 0; j < i; j++) if (A[i][j] != 0.0) acc += h * A[i][j] * K[j][k];
            xi[k] = acc;
          }}
          g(xi, K[i]);
        }}
        for (int k = 0; k < d; k++) {{
          double acc = 0.0;
          for (int i = 0; i < {S}; i++) if (Bw[i] != 0.0) acc += Bw[i] * K[i][k];
          x[k] += h * acc;
        }}
      }}
      memcpy(o + (s + 1) * d, x, sizeof(double) * d);
    }}
  }}
}}""")
    return "\n".join(lines)


def build_true(exprs):
    """Returns (f, rollout): f(x (..., d)) -> (..., d) and rollout(x0 (n, d), steps, nsub, dt) -> (n, steps+1, d)."""
    src = c_source_sympy(exprs)
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    so = CACHE / f"true_{key}.so"
    if not so.exists():
        c = CACHE / f"true_{key}.c"
        c.write_text(src)
        subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(so), str(c), "-lm"], check=True,
                       capture_output=True)
    lib = ctypes.CDLL(str(so))
    P = ctypes.POINTER(ctypes.c_double)
    lib.rhs.argtypes = [P, ctypes.c_long, P]
    lib.rollout.argtypes = [P, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_double, P]
    d = len(exprs)

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        flat = np.ascontiguousarray(x.reshape(-1, d))
        out = np.empty_like(flat)
        lib.rhs(flat.ctypes.data_as(P), flat.shape[0], out.ctypes.data_as(P))
        return out.reshape(x.shape)

    def rollout(x0, steps, nsub, dt):
        x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(-1, d))
        out = np.empty((x0.shape[0], steps + 1, d))
        lib.rollout(x0.ctypes.data_as(P), x0.shape[0], steps, nsub, dt, out.ctypes.data_as(P))
        return out
    return f, rollout
