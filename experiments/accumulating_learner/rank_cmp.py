"""Rank comparison (owner, 20 November): how the learner networks ranked each mastered piece with K inside on its own
level's held-out systems at mastery (curriculum levels 5, 6, 7: 2 variables; the network after the level's last round),
against how the e3 networks rank x/(K+x^2) on e3's held-out systems; and, as a control, how the e3 networks rank the
mastered K pieces on e3's held-out systems. Ranks of true terms (counts only) at the empty start and one short, no
search, no training. Usage: rank_cmp.py <workers>"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

G2H = LD.G2H
NAMES = {5: "1/(K+x)", 6: "x/(K+x)", 7: "1/(K+x^2)", 8: "x/(K+x^2)"}
CUR_SYS = {1: "cpilot/gen/systems-s1.json", 2: "cpilot/gen/systems-s2.json", 3: "cseed3/gen/systems-s3.json"}


def job(args):
    rec, spec, tag, levels, d = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import dream_g2 as DR
    import muzero_checks as MZ
    of = {k: lv for lv, ks in LD.piece_keys(d).items() for k in ks}
    ctx, fitter = MZ.system(rec)
    v = ctx["vocab"]
    prior = LD.make_prior(spec)
    rows = []
    for i, keys in enumerate(rec["structure"]):
        true = sorted({v.index[k] for k in keys if k in v.index})
        eq = MZ.Eq(ctx, fitter, i)
        for t in true:
            lv = of.get(DR.role_key(v.keys[t], i, d))
            if lv not in levels:
                continue
            others = [u for u in true if u != t]
            for point, S in (("empty start", ()), ("one short", tuple(others))):
                nd = eq.node(S)
                legal = MZ.legal_adds(v.V, S)
                pr = prior(nd, eq.se.ctx)
                sc = np.log(np.maximum(np.where(legal, pr.p_swap_add, 0), 1e-300))
                rows.append({"tag": tag, "piece": lv, "point": point, "rank": MZ.rank(sc, legal, t), "V": int(v.V)})
    return rows


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = []
    # curriculum: each K level's held-out systems, the network at that level's mastery (end of the level)
    for (arm, s), (base, name, _) in LD.CURRIC.items():
        rec = json.loads((G2H / base / "lineages" / f"{name}.json").read_text())
        cur = json.loads((G2H / CUR_SYS[s]).read_text())
        n = 0
        for L in rec["levels"]:
            n += len(L["rounds"])
            lv = L["level"]
            if lv not in (5, 6, 7):
                continue
            net = str(G2H / base / "lineages" / name / f"learn{n}")
            spec = ("confidence", net, {2: str(G2H / base / "lineages" / name / f"library{n}.npy")}) if arm == "blend" else ("net", net)
            tag = f"curriculum L{lv} {arm}-s{s} ({'mastered' if L['mastered'] else 'not mastered'})"
            jobs += [(r, spec, tag, (lv,), 2) for r in cur[str(lv)]["test"]]
    # e3: the e3 networks on e3's held-out systems, x/(K+x^2) and the mastered K pieces
    for arm in ("blend", "net"):
        for s in (1, 2, 3):
            st = json.loads((G2H / "ladder" / "e3" / f"{arm}-s{s}.state.json").read_text())
            tables = {int(k): v for k, v in st["tables"].items()}
            spec = ("confidence", st["net"], tables) if arm == "blend" else ("net", st["net"])
            test = json.loads((G2H / "ladder" / "gen" / f"e3-s{s}.json").read_text())["test"]
            jobs += [(r, spec, f"e3 {arm}-s{s}", (5, 6, 7, 8), 3) for r in test]
    with ProcessPoolExecutor(int(sys.argv[1]), mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(job, jobs) for x in part]
    (LD.OUT / "e3" / "rank_cmp.json").write_text(json.dumps(rows))
    import numpy as np
    print("== ranks of true terms (median; share in the top 10) at the empty start | one short")

    def show(label, R):
        e = [x["rank"] for x in R if x["point"] == "empty start"]
        o = [x["rank"] for x in R if x["point"] == "one short"]
        if e:
            print(f"   {label}: {len(e)} terms; empty start median {np.median(e):.0f}, top 10 {np.mean(np.array(e) <= 10):.0%} | one short "
                  f"median {np.median(o):.0f}, top 10 {np.mean(np.array(o) <= 10):.0%}; vocabulary {R[0]['V']}")
    for lv in (5, 6, 7):
        show(f"curriculum, {NAMES[lv]} at its level's mastery (2 variables), all lineages",
             [x for x in rows if x["tag"].startswith(f"curriculum L{lv} ")])
    for lv in (5, 6, 7, 8):
        show(f"e3 networks, {NAMES[lv]} on e3's held-out (3 variables)", [x for x in rows if x["tag"].startswith("e3 ") and x["piece"] == lv])
    print("== per lineage")
    for tag in sorted({x["tag"] for x in rows}):
        for lv in (5, 6, 7, 8):
            R = [x for x in rows if x["tag"] == tag and x["piece"] == lv]
            if R:
                show(f"{tag} {NAMES[lv]}", R)


if __name__ == "__main__":
    main()
