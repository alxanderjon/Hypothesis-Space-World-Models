# PRACTICE half only: sanity check of the PySR configuration on a simple polynomial component (753, M = 16, comp 0).
import time, numpy as np
from pysr import PySRRegressor
SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
D = np.load(f"{SP}/bench/data/BIOMD0000000753_M16.npz")
X, y = D["Zc"], D["G"][:, 0]
rms = np.sqrt((y ** 2).mean())
for label, kw in (("lean + - * /", dict(binary_operators=["+", "-", "*", "/"])),
                  ("benchmark set", dict(binary_operators=["+", "-", "*", "/", "^"], unary_operators=["exp", "tanh", "sin", "abs"],
                                         constraints={"^": (-1, 1)}))):
    t0 = time.time()
    m = PySRRegressor(maxsize=40, niterations=100000, timeout_in_seconds=60, precision=64, parallelism="serial",
                      deterministic=True, random_state=1, model_selection="accuracy", verbosity=0, progress=False,
                      temp_equation_file=True, delete_tempfiles=True, **kw)
    m.fit(X, y, variable_names=["z0", "z1"])
    eq = m.equations_.sort_values("loss")
    for r in m.equations_.sort_values("complexity").itertuples():
        if r.complexity <= 13:
            print(f"  [{label}] c={r.complexity} rel={np.sqrt(r.loss) / rms:.2e} {str(r.sympy_format)[:100]}", flush=True)
    print(f"[{label}] best rel {np.sqrt(eq.iloc[0].loss) / rms:.2e} in {time.time() - t0:.0f}s", flush=True)
