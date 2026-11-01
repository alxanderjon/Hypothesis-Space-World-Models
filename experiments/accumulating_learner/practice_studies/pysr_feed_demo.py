# PRACTICE half only: can discovered pieces and equation patterns be fed into PySR as building blocks?
# Target 115 (M = 32). Piece from 114: a Hill switch with exponent 4 -> custom operator hill4(x) = x^4/(1+x^4) (placement
# via PySR's own constants inside its argument). Pattern from 114's equations -> a starting expression (guess).
import sys, time, json
import numpy as np
from pysr import PySRRegressor
SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
D = np.load(f"{SP}/bench/data/BIOMD0000000115_M32.npz")
X, Y = D["Zc"], D["G"]
out = {}
for label, extra in (("plain", {}),
                     ("with piece + pattern", {
                         "unary_operators": ["exp", "tanh", "sin", "abs", "hill4(x) = x^4 / (1 + x^4)"],
                         "extra_sympy_mappings": {"hill4": lambda x: x ** 4 / (1 + x ** 4)},
                         # 114's pattern, relabelled to 115's variables: z_self, z_other, z_self * hill4(c * z_other)
                         "guesses": [["0.01*z1 - 0.1*z0*hill4(0.6*z1) - 0.0003*z0"], ["0.02 - 0.1*z1 + 0.003*z0 + 1.4*z0*hill4(0.6*z1)"]]})):
    t0 = time.time()
    kw = dict(binary_operators=["+", "-", "*", "/", "^"], unary_operators=["exp", "tanh", "sin", "abs"],
              constraints={"^": (-1, 1)}, maxsize=40, niterations=100000, timeout_in_seconds=150, precision=64,
              parallelism="serial", deterministic=True, random_state=1, model_selection="accuracy", verbosity=0, progress=False,
              temp_equation_file=True, delete_tempfiles=True)
    kw.update(extra)
    m = PySRRegressor(**kw)
    m.fit(X, Y, variable_names=["z0", "z1"])
    rms = np.sqrt((Y ** 2).mean(0))
    res = []
    for i, eqs in enumerate(m.equations_):
        best = eqs.sort_values("loss").iloc[0]
        res.append({"component": i, "rel_misfit": float(np.sqrt(best.loss) / rms[i]), "equation": str(best.sympy_format)[:160]})
    out[label] = {"seconds": time.time() - t0, "components": res}
    print(label, json.dumps(out[label], indent=1), flush=True)
