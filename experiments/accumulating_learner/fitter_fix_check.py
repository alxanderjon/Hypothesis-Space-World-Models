"""Fitter fix check (owner, 21 November): the proposed fix, applied in this process only (fitter_g1.py on disk is not
changed while q33g runs): for structures with two or more terms with K inside, (i) the precise gate opens at an ordinary
relative error of 0.1 instead of 0.01, and (ii) the thorough fit's extra global starts are doubled (keep 4 -> 8). On q33's
36 multi-K held-out equations (counts only), the true structure through the normal path (evaluate_structure), with and
without the fix, for fitter seeds 1, 2, 3; plus the seconds per structure. CPU only. Usage: fitter_fix_check.py <workers>"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

K_LEVELS = (5, 6, 7, 8, 10)


def patch(FG):
    """The fix, as a patch of this process's fitter class."""
    if getattr(FG.FitterL1, "_fixed", False):
        return
    gate0, thorough0 = FG.FitterL1.precise_gate, FG.FitterL1.thorough

    def precise_gate(self, keys, res, rel_rate, floor):
        if self.nonlinear(keys) >= 2 and rel_rate <= 1e-1:
            return True
        return gate0(self, keys, res, rel_rate, floor)

    def thorough(self, keys, W, y, floor, current, eq=None, si_tr=None, si_va=None):
        keep = self.keep
        if self.nonlinear(keys) >= 2:
            self.keep = 2 * keep
        try:
            return thorough0(self, keys, W, y, floor, current, eq=eq, si_tr=si_tr, si_va=si_va)
        finally:
            self.keep = keep
    FG.FitterL1.precise_gate, FG.FitterL1.thorough, FG.FitterL1._fixed = precise_gate, thorough, True


def job(args):
    rec, fixed, fseed = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import fitter_g1 as FG
    import search_g1 as SG
    if fixed:
        patch(FG)
    of = {k: lv for lv, ks in LD.piece_keys(3).items() for k in ks}
    out = []
    ctx = SG.system_context(np.array(rec["bursts"]), seed=fseed)
    fitter = FG.make_fitter(ctx, seed=fseed)
    for i, ks in enumerate(rec["structure"]):
        if sum(of.get(DR.role_key(x, i, 3)) in K_LEVELS for x in ks) < 2:
            continue
        t0 = time.perf_counter()
        nd = FG.evaluate_structure(tuple(ks), ctx, i, fitter)
        out.append({"id": rec["id"], "eq": i, "fixed": fixed, "seed": fseed, "exact": nd["exact"], "gate": nd["gate"],
                    "thorough": nd["thorough"], "seconds": time.perf_counter() - t0})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    recs = [r for s in (1, 2, 3) for r in json.loads((LD.OUT / "gen" / f"q33-s{s}.json").read_text())["test"]]
    jobs = [(r, fixed, fs) for r in recs for fixed in (False, True) for fs in (1, 2, 3)]
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(job, jobs) for x in part]
    (LD.OUT / "q33" / "fitter_fix_check.json").write_text(json.dumps(rows, indent=1))
    print("== Fitter fix check: q33's multi-K held-out equations, true structure, the normal path (evaluate_structure)")
    for fixed in (False, True):
        for fs in (1, 2, 3):
            R = [r for r in rows if r["fixed"] == fixed and r["seed"] == fs]
            print(f"   {'fixed' if fixed else 'base '} seed {fs}: exact {sum(r['exact'] for r in R)} of {len(R)}; gate open {sum(r['gate'] for r in R)}; "
                  f"thorough ran {sum(r['thorough'] for r in R)}; median seconds {np.median([r['seconds'] for r in R]):.1f}, mean {np.mean([r['seconds'] for r in R]):.1f}")
    for fixed in (False, True):
        R = [r for r in rows if r["fixed"] == fixed]
        by = {}
        for r in R:
            by.setdefault((r["id"], r["eq"]), []).append(r["exact"])
        print(f"== {'fixed' if fixed else 'base '}: exact {sum(r['exact'] for r in R)} of {len(R)} fits; equations exact in all 3 seeds "
              f"{sum(all(v) for v in by.values())} of {len(by)}, in none {sum(not any(v) for v in by.values())}")
    base = {(r["id"], r["eq"], r["seed"]): r["exact"] for r in rows if not r["fixed"]}
    fix = {(r["id"], r["eq"], r["seed"]): r["exact"] for r in rows if r["fixed"]}
    print(f"   fix gains {sum(fix[k] and not base[k] for k in base)} fits, loses {sum(base[k] and not fix[k] for k in base)}")


if __name__ == "__main__":
    main()
