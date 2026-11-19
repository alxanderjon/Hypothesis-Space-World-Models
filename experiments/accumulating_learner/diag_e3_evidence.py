"""e3 evidence diagnosis (owner, 19 November): on the e3-style development systems of the emphasized-dream check (60
systems; counts only), for every true x/(K+x^2) term, with no search and no training:
1. evidence: its rank by the data evidence alone (pooled partial R^2 against the residual, as the network sees it) at
   the empty start and one short, against the other true terms; and how many look-alikes on the same variable (x, x^2,
   x^3, 1/(K+x), x/(K+x), 1/(K+x^2), x·y with it) have stronger evidence;
2. the K grid: the true K in the learner's coordinates (from an exact fit of the true structure) against the grid of K
   values the evidence uses (inside its range? distance to the nearest grid value), and the evidence lost to the grid
   (best grid column's gain against the exact-K column's);
3. the data range: q = max w^2 / |K| over the recordings (w the learner's coordinate of the term's variable); for q << 1
   the term behaves like w/K - w^3/K^2. And how much of the true column's shape look-alikes copy: R^2 of the column on
   its variable's best single look-alike column, and on span{w, w^3}.
Usage: diag_e3_evidence.py <workers>"""

import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

DEV = LD.G2H / "emph_check" / "gen"
NAMES = {1: "x", 2: "x*y", 3: "x^2", 4: "x^3", 5: "1/(K+x)", 6: "x/(K+x)", 7: "1/(K+x^2)", 8: "x/(K+x^2)", 9: "constant", 10: "Hill"}


def variables(key):
    return sorted({int(m) for m in re.findall(r"Symbol\('x(\d)'", key)})


def job(rec):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import fitter_g1 as FG
    import muzero_checks as MZ
    import search_g1 as SG
    d = 3
    keys = LD.piece_keys(d)
    of = {k: lv for lv, ks in keys.items() for k in ks}
    ctx, fitter = MZ.system(rec)
    v, W = ctx["vocab"], ctx["W"]
    A = v.A
    out = []

    def r2(col, B):
        """R^2 of col on the span of B's columns."""
        col = col / (col.norm() + 1e-300)
        Q = torch.linalg.qr(B)[0]
        return float(((Q.T @ col) ** 2).sum())

    def best_single(col, key):
        """Best R^2 of col on one grid column of `key`."""
        if key not in ctx["bank"].bank:
            return None
        C = ctx["bank"].bank[key][0]
        C = C / C.norm(dim=0, keepdim=True)
        c = col / (col.norm() + 1e-300)
        return float(((C.T @ c) ** 2).max())

    for i, true_keys in enumerate(rec["structure"]):
        if any(k not in v.index for k in true_keys):
            continue
        roles = {k: of.get(DR.role_key(k, i, d)) for k in true_keys}
        new = [k for k in true_keys if roles[k] == 8]
        if not new:
            continue
        S = tuple(sorted(v.index[k] for k in true_keys))
        eq = MZ.Eq(ctx, fitter, i)
        node = eq.node(S)
        if not node["exact"]:
            continue
        c2 = eq.se.ctx
        for k in new:
            t = v.index[k]
            j = variables(k)[0]                                   # the term's variable (1-based in the key)
            term = [u for u in node["terms"] if u.key == k][0]
            K = float(term.theta[0])
            w = W[:, j - 1]
            q = float((w ** 2).max()) / abs(K)
            col = FG.value(term, W)
            # 1. evidence at the empty start and one short
            res = {}
            for point, Sp in (("empty start", ()), ("one short", tuple(x for x in S if x != t))):
                nd = eq.node(Sp)
                gain, _ = SG.evidence(nd, c2)
                gain = np.nan_to_num(np.asarray(gain), nan=0.0)
                legal = np.ones(v.V, dtype=bool)
                legal[list(Sp)] = False
                rank = int((np.where(legal, gain, -1) > gain[t]).sum()) + 1
                look = [u for u in range(v.V) if legal[u] and u != t and j in variables(v.keys[u])
                        and of.get(DR.role_key(v.keys[u], i, d)) in (1, 2, 3, 4, 5, 6, 7)]
                stronger = [u for u in look if gain[u] > gain[t]]
                others = [x for x in S if x != t and legal[x]]
                res[point] = {"rank": rank, "lookalikes_stronger": len(stronger),
                              "stronger_types": sorted({NAMES[of.get(DR.role_key(v.keys[u], i, d))] for u in stronger}),
                              "other_true_ranks": [int((np.where(legal, gain, -1) > gain[x]).sum()) + 1 for x in others]}
                res[point]["gain_grid"] = float(gain[t])
                res[point]["gain_best_lookalike"] = float(max([gain[u] for u in look], default=0.0))
                # exact-K column's evidence against the best grid column's (the grid's loss)
                if True:
                    import fitter_g1 as FG2
                    y = c2["y"]
                    if Sp:
                        Phi = torch.stack([FG2.value(u, W) for u in nd["terms"]], 1)
                        Q = torch.linalg.qr(torch.nan_to_num(Phi))[0]
                        r_ = y - Q @ (Q.T @ y)
                        ce = col - Q @ (Q.T @ col)
                    else:
                        r_, ce = y, col
                    res[point]["gain_exactK"] = float((ce @ r_) ** 2 / ((ce @ ce) * (r_ @ r_) + 1e-300))
            # 2. the K grid
            th = ctx["bank"].bank[k][1][:, 0]
            pos = th[th > 0]
            same = th[np.sign(th) == np.sign(K)]
            inside = bool(len(same)) and float(np.abs(same).min()) <= abs(K) <= float(np.abs(same).max())
            near = float(np.min(np.abs(np.log10(np.abs(same)) - math.log10(abs(K))))) if len(same) else None
            # 3. data range and look-alike copies of the column's shape
            look_best = {}
            for u in range(v.V):
                ku = v.keys[u]
                lv = of.get(DR.role_key(ku, i, d))
                if lv in (1, 3, 4, 5, 6, 7) and variables(ku) == [j]:
                    b = best_single(col, ku)
                    if b is not None:
                        look_best[NAMES[lv]] = max(look_best.get(NAMES[lv], 0.0), b)
            lin_cub = r2(col, torch.stack([w, w ** 3], 1))
            out.append({"id": rec["id"], "eq": i, "K": K, "K_sign": int(np.sign(K)), "q": q, "grid_inside": inside,
                        "grid_log10_distance": near, "grid_range": [float(np.abs(same).min()), float(np.abs(same).max())] if len(same) else None,
                        "evidence": res, "lookalike_R2": look_best, "R2_on_w_w3": lin_cub})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    recs = [r for s in (1, 2, 3) for r in json.loads((DEV / f"dev-s{s}.json").read_text())]
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(job, recs) for x in part]
    (LD.OUT / "e3" / "evidence_diagnosis.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    E, O = [r["evidence"]["empty start"] for r in rows], [r["evidence"]["one short"] for r in rows]
    other_e = [x for e in E for x in e["other_true_ranks"]]
    other_o = [x for e in O for x in e["other_true_ranks"]]
    print(f"x/(K+x^2) true terms: {n} (in {len({(r['id'], r['eq']) for r in rows})} equations of {len(recs)} systems)")
    for nm, P, oth in (("empty start", E, other_e), ("one short", O, other_o)):
        print(f"1. evidence, {nm}: median rank {np.median([e['rank'] for e in P]):.0f} (top 10 {np.mean([e['rank'] <= 10 for e in P]):.0%}); "
              f"other true terms median rank {np.median(oth):.0f} (top 10 {np.mean([x <= 10 for x in oth]):.0%}); "
              f"a look-alike on the same variable stronger in {np.mean([e['lookalikes_stronger'] > 0 for e in P]):.0%} "
              f"(median {np.median([e['lookalikes_stronger'] for e in P]):.0f} stronger)")
        from collections import Counter
        c = Counter(t for e in P for t in e["stronger_types"])
        print(f"   stronger look-alike types (share of terms): " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
    for nm, P in (("empty start", E), ("one short", O)):
        print(f"   {nm}: median gain of the true term at its best grid K {np.median([e['gain_grid'] for e in P]):.4f}, at the exact K "
              f"{np.median([e['gain_exactK'] for e in P]):.4f}, best same-variable look-alike {np.median([e['gain_best_lookalike'] for e in P]):.4f}; "
              f"look-alike beats the grid in {np.mean([e['gain_best_lookalike'] > e['gain_grid'] for e in P]):.0%}, beats the exact K in "
              f"{np.mean([e['gain_best_lookalike'] > e['gain_exactK'] for e in P]):.0%}")
    print(f"2. K grid: true K inside the grid's range (same sign) {np.mean([r['grid_inside'] for r in rows]):.0%}; "
          f"median distance to the nearest grid K {np.median([r['grid_log10_distance'] for r in rows if r['grid_log10_distance'] is not None]):.3f} decades "
          f"(spacing 0.25); negative K {np.mean([r['K_sign'] < 0 for r in rows]):.0%}")
    qs = np.array([r["q"] for r in rows])
    print(f"3. data range q = max w^2/|K|: median {np.median(qs):.2f}; q < 0.1 in {np.mean(qs < 0.1):.0%}, 0.1-1 in "
          f"{np.mean((qs >= 0.1) & (qs < 1)):.0%}, >= 1 in {np.mean(qs >= 1):.0%}")
    lb = [max(r["lookalike_R2"].values()) for r in rows if r["lookalike_R2"]]
    print(f"   column copied by one look-alike: median R^2 {np.median(lb):.4f}; > 0.999 in {np.mean(np.array(lb) > 0.999):.0%}; "
          f"by span{{w, w^3}}: median R^2 {np.median([r['R2_on_w_w3'] for r in rows]):.4f}, > 0.999 in "
          f"{np.mean(np.array([r['R2_on_w_w3'] for r in rows]) > 0.999):.0%}")
    for lo, hi in ((0, 0.1), (0.1, 1), (1, 1e9)):
        sel = [r for r in rows if lo <= r["q"] < hi]
        if sel:
            print(f"   q in [{lo}, {hi}): {len(sel)} terms; evidence rank at the empty start median "
                  f"{np.median([r['evidence']['empty start']['rank'] for r in sel]):.0f}; look-alike stronger "
                  f"{np.mean([r['evidence']['empty start']['lookalikes_stronger'] > 0 for r in sel]):.0%}; best look-alike R^2 median "
                  f"{np.median([max(r['lookalike_R2'].values()) for r in sel if r['lookalike_R2']]):.4f}")


if __name__ == "__main__":
    main()
