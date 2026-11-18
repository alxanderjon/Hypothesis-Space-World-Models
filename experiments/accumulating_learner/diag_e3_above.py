"""e3 diagnosis, part 2: for every held-out x/(K+x^2) true term (counts only), the prior's rank at the empty start for each
lineage's final prior, and the piece types the prior ranks above it. Read only. Usage: diag_e3_above.py <workers>"""
import json, os, sys
from collections import Counter
from pathlib import Path
os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

NAMES = {1: "x", 2: "x*y", 3: "x^2", 4: "x^3", 5: "1/(K+x)", 6: "x/(K+x)", 7: "1/(K+x^2)", 8: "x/(K+x^2)", 9: "constant", 10: "Hill"}


def job(args):
    rec, spec, of = args
    import warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np, torch
    torch.set_num_threads(1)
    import dream_g2 as DR, muzero_checks as MZ
    ctx, fitter = MZ.system(rec)
    v = ctx["vocab"]
    prior = LD.make_prior(spec)
    out = []
    for i, keys in enumerate(rec["structure"]):
        eq = MZ.Eq(ctx, fitter, i)
        pr = prior(eq.se.root, eq.se.ctx)
        legal = MZ.legal_adds(v.V, ())
        sc = np.where(legal, np.log(np.maximum(pr.p_swap_add, 1e-300)), -np.inf)
        for k in keys:
            rk = DR.role_key(k, i, 3)
            if of.get(rk) != 8 or k not in v.index:
                continue
            t = v.index[k]
            above = [u for u in np.nonzero(sc > sc[t])[0]]
            out.append({"rank": MZ.rank(sc, legal, t),
                        "above": dict(Counter(NAMES.get(of.get(DR.role_key(v.keys[u], i, 3)), "other") for u in above))})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    keys = LD.piece_keys(3)
    of = {k: lv for lv, ks in keys.items() for k in ks}
    jobs, meta = [], []
    for arm in LD.ARMS:
        for s in LD.SEEDS:
            st = json.loads((LD.OUT / "e3" / f"{arm}-s{s}.state.json").read_text())
            tables = {int(k): v for k, v in st["tables"].items()}
            spec = ("confidence", st["net"], tables) if arm == "blend" else ("net", st["net"])
            for r in json.loads((LD.OUT / "gen" / f"e3-s{s}.json").read_text())["test"]:
                jobs.append((r, spec, of)); meta.append((arm, s))
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        res = list(ex.map(job, jobs))
    for arm in LD.ARMS:
        R = [x for m, rr in zip(meta, res) if m[0] == arm for x in rr]
        ab = Counter()
        for x in R:
            ab.update(x["above"])
        n = len(R)
        print(f"{arm}: {n} held-out x/(K+x^2) terms; rank median {np.median([x['rank'] for x in R]):.0f}, share in top 10 "
              f"{np.mean([x['rank'] <= 10 for x in R]):.0%}; mean terms ranked above it by type: "
              + ", ".join(f"{k} {v / n:.1f}" for k, v in ab.most_common(6)))


if __name__ == "__main__":
    main()
