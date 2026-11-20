"""Fitter check (owner, 20 November): why does the fitter miss the true constants of q33's multi-K equations (2-3 terms
with K inside) even given the true structure? For every such held-out equation (counts only), with the true structure:
  1. the normal path (fitter_g1.evaluate_structure): ordinary fit, the precise gate, the thorough fit if the gate opens;
  2. the thorough fit forced (gate ignored), then the same step-integral and held-back checks;
  3. a polish started from the TRUE constants (recovered by redrawing the system from its seed, converted to the
     learner's coordinates), then the same checks;
  4. where the true K values lie against each term's grid of constants.
CPU only, no search, no training. Usage: fitter_check.py <workers>"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

RUNG, K_LEVELS = "q33", (5, 6, 7, 8, 10)


def recover(rec, s, k, allowed):
    """The system's description, by redrawing from its seed until the built bursts match the record's."""
    import numpy as np
    import domains_g1 as DG
    import dream_g2 as DR
    want = LD.data_key(rec["bursts"])
    prefix = f"ladder-{RUNG}-s{s}-test"
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = LD.draw(seed, allowed, (3, 3), 3)
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
        except Exception:
            continue
        if LD.data_key(syn.bursts(8)) == want:
            return desc
    return None


def true_thetas(desc, i, keys, sig):
    """Per true key of equation i, its constant K in the learner's coordinates (None if the key has no K)."""
    import sympy as sp
    import language_g1 as LG
    e = sp.sympify(eval(desc["rhs"][i], sp.__dict__))
    K0 = sp.Symbol("K0", real=True)
    out = {}
    for key in keys:
        kk = LG.expr_of(key)
        if K0 not in kk.free_symbols:
            out[key] = None
            continue
        for arg in sp.Add.make_args(sp.expand(e, deep=False) if False else e):
            cs = [s_ for s_ in arg.free_symbols if s_.name.startswith(f"c{i}_")]
            if len(cs) != 1:
                continue
            a = [s_ for s_ in arg.free_symbols if s_.name.startswith(f"a{i}_")]
            body = (arg / a[0]) if a else arg
            body = body.xreplace({cs[0]: K0})
            # compare by name of the variables (desc uses domains_g1's symbols, keys language_g1's)
            ren = {s_: sp.Symbol(f"x{int(s_.name[1:]) + 1}", real=True) for s_ in body.free_symbols if s_.name[0] == "y" and s_.name[1:].isdigit()}
            if sp.simplify(body.xreplace(ren) - kk.xreplace({s_: sp.Symbol(s_.name, real=True) for s_ in kk.free_symbols})) == 0:
                K = float(desc["params"][cs[0].name])
                j = LD.shifted_var(key)
                sq = "Pow(Symbol('x" in key.split("Add(Symbol('K0', real=True), ")[1][:12]
                out[key] = K / (sig[j - 1] ** (2 if sq else 1))
                break
        else:
            out[key] = "unmatched"
    return out


def job(args):
    rec, s, k, allowed = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import fitter_g1 as FG
    import search_g1 as SG
    of = {kk: lv for lv, ks in LD.piece_keys(3).items() for kk in ks}
    rows = []
    multi = [i for i, ks in enumerate(rec["structure"]) if sum(of.get(DR.role_key(x, i, 3)) in K_LEVELS for x in ks) >= 2]
    if not multi:
        return rows
    desc = recover(rec, s, k, allowed)
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx)
    W, sig = ctx["W"], np.asarray(ctx["view"]["sig"], dtype=float)
    for i in multi:
        keys = tuple(rec["structure"][i])
        y, yn, floor, tau = FG.equation_targets(ctx, i)
        nk = sum(of.get(DR.role_key(x, i, 3)) in K_LEVELS for x in keys)
        row = {"id": rec["id"], "eq": i, "K_terms": nk, "recovered": desc is not None}
        # 1. normal path
        nd = FG.evaluate_structure(keys, ctx, i, fitter)
        row.update(normal_exact=nd["exact"], ordinary_rel=nd["rel_rate"], gate=nd["gate"], thorough_ran=nd["thorough"])

        def judge(terms):
            try:
                t2, r2, c2 = FG.fast_refine(terms, W, y, iters=40, stop_below=1.5 * floor)
                tt, rt, ct = FG.refine_integral(t2, ctx["si_tr"], i)
                rv = FG.heldback_error(tt, ct, ctx["si_va"], i) if ct is not None else float("inf")
                return {"rel_rate": r2 / yn, "rel_t": rt, "rel_v": rv, "exact": bool(rt <= tau and rv <= tau), "terms": t2}
            except (RuntimeError, ValueError):
                return {"rel_rate": float("inf"), "exact": False, "terms": terms}
        # 2. thorough forced
        terms, res, coefs, _ = fitter.fit(keys, W, y, None, floor=floor, eq=i)
        try:
            th = fitter.thorough(keys, W, y, floor, (terms, res, coefs), eq=i, si_tr=ctx["si_tr"], si_va=ctx["si_va"])
            j2 = judge(th[0])
        except (RuntimeError, ValueError):
            j2 = {"rel_rate": float("inf"), "exact": False, "terms": terms}
        row.update(forced_exact=j2["exact"], forced_rel=j2["rel_rate"])
        # 3. from the true constants
        if desc is not None:
            tt = true_thetas(desc, i, keys, sig)
            if all(v != "unmatched" for v in tt.values()):
                start = [FG.LTerm(kk, [tt[kk]] if tt[kk] is not None else ()) for kk in keys]
                j3 = judge(start)
                row.update(true_start_exact=j3["exact"], true_start_rel=j3["rel_rate"])
                # 4. the grid: each true K against its key's grid of constants
                g = []
                for kk in keys:
                    if tt[kk] is None or kk not in ctx["bank"].bank:
                        continue
                    grid = ctx["bank"].bank[kk][1][:, 0]
                    same = grid[np.sign(grid) == np.sign(tt[kk])]
                    if len(same):
                        g.append({"inside": bool(np.abs(same).min() <= abs(tt[kk]) <= np.abs(same).max()),
                                  "log10_to_nearest": float(np.min(np.abs(np.log10(np.abs(same)) - math.log10(abs(tt[kk])))))})
                    else:
                        g.append({"inside": False, "log10_to_nearest": None})
                row["grid"] = g
                # the ordinary fit's constants against the true ones (did it land in another basin?)
                fit_k = {t.key: t.theta[0] for t in nd["terms"] if t.theta}
                row["ordinary_K_log10_off"] = [abs(math.log10(abs(fit_k[kk]) / abs(tt[kk]))) if kk in fit_k and tt[kk] and fit_k[kk]
                                               else None for kk in keys if tt[kk] is not None]
            else:
                row["true_start_exact"] = None
        rows.append(row)
    return rows


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    keys = LD.piece_keys(3)
    allowed = {lv: keys[lv] for lv in LD.RUNGS[RUNG]["levels"]}
    jobs = []
    for s in (1, 2, 3):
        for r in json.loads((LD.OUT / "gen" / f"{RUNG}-s{s}.json").read_text())["test"]:
            jobs.append((r, s, int(r["id"].rsplit("-", 1)[1]), allowed))
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(job, jobs) for x in part]
    (LD.OUT / RUNG / "fitter_check.json").write_text(json.dumps(rows, indent=1, default=float))
    n = len(rows)
    print(f"multi-K held-out equations: {n} ({sum(r['recovered'] for r in rows)} with the system recovered from its seed)")
    ne = [r for r in rows if not r["normal_exact"]]
    print(f"1. normal path exact: {n - len(ne)} of {n}; of the {len(ne)} not exact: gate closed {sum(not r['gate'] for r in ne)}, "
          f"thorough ran {sum(r['thorough_ran'] for r in ne)}; ordinary fit rel. error median {np.median([r['ordinary_rel'] for r in ne]) if ne else float('nan'):.1e}")
    print(f"2. thorough forced: exact {sum(r['forced_exact'] for r in rows)} of {n}; of the normal path's failures: {sum(r['forced_exact'] for r in ne)} of {len(ne)}")
    ts = [r for r in rows if r.get("true_start_exact") is not None]
    print(f"3. from the true constants: exact {sum(r['true_start_exact'] for r in ts)} of {len(ts)}; of the normal path's failures: "
          f"{sum(r['true_start_exact'] for r in ts if not r['normal_exact'])} of {sum(1 for r in ts if not r['normal_exact'])}")
    G = [g for r in ts for g in r.get("grid", [])]
    if G:
        print(f"4. grid: true K inside the grid's range {np.mean([g['inside'] for g in G]):.0%}; median distance to the nearest grid value "
              f"{np.median([g['log10_to_nearest'] for g in G if g['log10_to_nearest'] is not None]):.3f} decades")
    off = [x for r in ne for x in (r.get("ordinary_K_log10_off") or []) if x is not None]
    if off:
        print(f"   normal-path failures: ordinary fit's K off the true K by a median of {np.median(off):.2f} decades (max {max(off):.2f})")
    for nk in (2, 3):
        R = [r for r in rows if r["K_terms"] == nk]
        if R:
            print(f"   {nk} K terms: {len(R)} equations; normal exact {sum(r['normal_exact'] for r in R)}, forced thorough exact "
                  f"{sum(r['forced_exact'] for r in R)}, true start exact {sum(bool(r.get('true_start_exact')) for r in R)}")


if __name__ == "__main__":
    main()
