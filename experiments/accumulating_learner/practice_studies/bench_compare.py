# Practice half: our learner vs PySR (lean operators + high-precision finish), exact recoveries and time per run.
import glob, json, os
import numpy as np

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
EXCLUDED = {"BIOMD0000000741", "BIOMD0000000802"}          # reference cannot be built


def ours():
    out = {}
    for d in ("scratch2", "bench_ours"):
        for p in glob.glob(f"{SP}/{d}/*_scratch.json"):
            r = json.load(open(p))
            out[(r["name"], r["M"])] = {"exact": bool(r.get("exact")), "accepted": r.get("status") == "accepted",
                                        "seconds": r.get("seconds")}
    return out


def pysr(folder="eval"):
    out = {}
    for p in glob.glob(f"{SP}/bench/{folder}/*.json"):
        r = json.load(open(p))
        out[(r["name"], r["M"])] = {"exact": bool(r.get("exact")), "accepted": r.get("status") == "accepted",
                                    "seconds": (r.get("pysr_seconds") or 0) + (r.get("finish_seconds") or 0)}
    return out


def main():
    A, B = ours(), pysr()
    names = sorted({k[0] for k in list(A) + list(B)} - EXCLUDED)
    print(f"{len(names)} practice systems")
    print(f"{'M':>3} | {'ours exact':>10} {'ours med s':>10} | {'PySR exact':>10} {'PySR med s':>10} | both  ours-only  PySR-only  (cells with both results)")
    for M in (8, 16, 32, 64):
        cells = [n for n in names if (n, M) in A and (n, M) in B]
        ea = [n for n in cells if A[(n, M)]["exact"]]
        eb = [n for n in cells if B[(n, M)]["exact"]]
        ta = [A[(n, M)]["seconds"] for n in cells if A[(n, M)]["seconds"]]
        tb = [B[(n, M)]["seconds"] for n in cells if B[(n, M)]["seconds"]]
        print(f"{M:>3} | {len(ea):>4}/{len(cells):<5} {np.median(ta) if ta else float('nan'):>10.0f} | {len(eb):>4}/{len(cells):<5} "
              f"{np.median(tb) if tb else float('nan'):>10.0f} | {len(set(ea) & set(eb)):>4}  {len(set(ea) - set(eb)):>9}  "
              f"{len(set(eb) - set(ea)):>9}")
        if set(eb) - set(ea):
            print("      PySR-only exact:", sorted(n[-4:] for n in set(eb) - set(ea)))
    print("ours exact at M=64:", sorted(n[-4:] for n in names if A.get((n, 64), {}).get("exact")))
    print("PySR exact at M=64:", sorted(n[-4:] for n in names if B.get((n, 64), {}).get("exact")))


if __name__ == "__main__":
    main()
