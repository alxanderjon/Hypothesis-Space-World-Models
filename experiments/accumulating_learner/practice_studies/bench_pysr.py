# PRACTICE half only: PySR from scratch on exactly the data our learner sees (scaled states at burst interiors -> rate
# estimates, training bursts only). One long-lived process per worker (Julia compiles once), jobs taken in turn.
# Output: each component's Pareto front (complexity, loss, sympy string) and the search time.
import json, os, sys, time, glob

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
DATA, OUT = f"{SP}/bench/data", f"{SP}/bench/pysr"


def main(worker, nworkers, only=None, budget_scale=1.0):
    import numpy as np
    from pysr import PySRRegressor
    os.makedirs(OUT, exist_ok=True); os.makedirs(OUT + "_test", exist_ok=True)
    idle = 0
    while idle < 40:                                          # files keep arriving from the export; claim by lock file
        files = sorted(glob.glob(f"{DATA}/*.npz"), key=lambda p: (int(p.split("_M")[1].split(".")[0]), p))
        path = None
        for cand in files:
            if only and only not in cand:
                continue
            tag = os.path.basename(cand)[:-4]
            out = f"{OUT}/{tag}.json" if budget_scale == 1.0 else f"{OUT}_test/{tag}.json"
            if os.path.exists(out):
                continue
            try:
                os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            path = cand
            break
        if path is None:
            if only or os.path.exists(f"{DATA}/EXPORT_DONE"):
                break
            idle += 1
            time.sleep(30)
            continue
        idle = 0
        try:
            D = np.load(path)
            _ = D["Zc"], D["G"]
        except Exception:                                     # still being written: release and retry later
            os.remove(out + ".lock")
            time.sleep(10)
            continue
        X, Y = D["Zc"], D["G"]
        d = X.shape[1]
        M = int(tag.split("_M")[1])
        budget = (300 if M <= 8 else 900) * budget_scale     # same wall-clock budget as our learner
        fronts, t0 = [], time.time()
        for i in range(d):                                    # one search per component, budget shared equally
            model = PySRRegressor(
                binary_operators=["+", "-", "*", "/"],        # lean set: polynomials and ratios. Adding sin/exp/tanh/abs/^
                                                              # trapped PySR in sine approximations (sanity check, 753)
                maxsize=40, niterations=100000, timeout_in_seconds=budget / d,
                precision=64, parallelism="serial", deterministic=True, random_state=1,
                model_selection="accuracy", verbosity=0, progress=False,
                temp_equation_file=True, delete_tempfiles=True)
            try:
                model.fit(X, Y[:, i], variable_names=[f"z{j}" for j in range(d)])
                eqs = model.equations_
                fronts.append([{"complexity": int(r.complexity), "loss": float(r.loss), "sympy": str(r.sympy_format)}
                               for r in eqs.itertuples()])
            except Exception as e:
                fronts.append({"error": repr(e)[:300]})
        rec = {"tag": tag, "d": d, "M": M, "seconds": time.time() - t0, "fronts": fronts}
        json.dump(rec, open(out, "w"))
        os.remove(out + ".lock")
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None,
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
