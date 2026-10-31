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


def _factor_c(f, z, x):
    fam, var, th = f[0], f[1], f[2]
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
    lines = ["#include <math.h>", "#include <string.h>", "static void g(const double* x, double* o) {"]
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
