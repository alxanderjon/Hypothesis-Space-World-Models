"""Equivalent forms as one piece (owner, 13 November; research log "Owner: finish the pilot; count equivalent forms as
one piece; ..."): before the library counts a verified solution, the solution is rewritten into its canonical
equivalent form with the system's own data, repeating until neither applies: drop a term if the rest still fits exactly
(the search's exactness test); merge a pair of terms into one term t if the rest plus t fits exactly (candidates: the
5 terms with the most evidence, at least 0.99 of the residual, at the node without the pair, simplest first). The check:
the curriculum pilot's library lineages' solutions at levels 1, 2, 3, 5, raw and canonical, rebuilt into libraries and
ranked on the follow-up check's held-out systems against a library from the true structures.
Usage: canon_check.py <workers> <canon|rank|report> [lineage ...]"""

import json
import os
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("G2H_RUN", "cpilot")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_pilot as CP                                    # noqa: E402

C, D = CP.C, CP.D
PILOT = C.ROOT / "outputs/accumulating_learner/g2h/cpilot"
CHECK1 = C.ROOT / "outputs/accumulating_learner/g2h/curriculum"
CHECK2 = C.ROOT / "outputs/accumulating_learner/g2h/curriculum2"
OUT = C.ROOT / "outputs/accumulating_learner/g2h/canon"
MASTERED = (1, 2, 3, 5)
COMBOS = ("A1", "A3", "B1", "B3")


def solutions(lineage):
    """{system id: {eq: keys}} of a pilot lineage, rebuilt from its wakes (later solutions replace earlier ones)."""
    rec = json.loads((PILOT / "lineages" / f"{lineage}.json").read_text())
    rounds = [r for L in rec["levels"] for r in L["rounds"]] + [r for B in rec["blocks"] for r in B["rounds"]]
    sols, level = {}, {}
    for r in rounds:
        for y in r["wake"]:
            level[y["id"]] = y["level"]
            if y["status"] == "done":
                found = {str(e["eq"]): e["solution"] for e in y["equations"] if e["first_exact"] is not None and e["solution"]}
                if found:
                    sols.setdefault(y["id"], {}).update(found)
    return sols, level


def canon_job(args):
    rec, sols = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    out = {}
    for eq, keys in sols.items():
        i = int(eq)
        if any(k not in vocab.index for k in keys):
            out[eq] = {"raw": keys, "canon": keys, "canon_plus": keys, "steps": [], "note": "outside the vocabulary"}
            continue
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))

        def node(T):
            return se.root if not T else se.evaluate(tuple(sorted(T)), None)
        S = set(vocab.index[k] for k in keys)
        if not node(S)["exact"]:
            out[eq] = {"raw": keys, "canon": keys, "canon_plus": keys, "steps": [], "note": "not exact when refitted"}
            continue
        steps, changed = [], True
        while changed:
            changed = False
            for a in sorted(S):
                if len(S) > 1 and node(S - {a})["exact"]:
                    S, changed = S - {a}, True
                    steps.append(["drop", vocab.keys[a]])
                    break
            if changed:
                continue
            for a, b in combinations(sorted(S), 2):
                T = S - {a, b}
                gain = SG.evidence(node(T), se.ctx)[0]
                cand = sorted(t for t in np.argsort(-gain)[:5] if int(t) not in S and gain[t] >= 0.99)
                for t in cand:
                    if node(T | {int(t)})["exact"]:
                        steps.append(["merge", vocab.keys[a], vocab.keys[b], vocab.keys[int(t)]])
                        S, changed = T | {int(t)}, True
                        break
                if changed:
                    break
        pre = [vocab.keys[t] for t in sorted(S)]
        changed = True                                           # the extension: simplify, then drop and merge again
        while changed:
            changed = False
            for a in sorted(S, reverse=True):
                T = S - {a}
                gain = SG.evidence(node(T), se.ctx)[0]
                cand = sorted(t for t in np.argsort(-gain)[:5] if int(t) not in S and int(t) < a and gain[t] >= 0.99)
                for t in cand:
                    if node(T | {int(t)})["exact"]:
                        steps.append(["simplify", vocab.keys[a], vocab.keys[int(t)]])
                        S, changed = T | {int(t)}, True
                        break
                if changed:
                    break
            if changed:
                continue
            for a in sorted(S):
                if len(S) > 1 and node(S - {a})["exact"]:
                    S, changed = S - {a}, True
                    steps.append(["drop", vocab.keys[a]])
                    break
        out[eq] = {"raw": keys, "canon": pre, "canon_plus": [vocab.keys[t] for t in sorted(S)], "steps": steps}
    return {"id": rec["id"], "equations": out}


def step_canon(workers, lineages):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    OUT.mkdir(parents=True, exist_ok=True)
    for lin in lineages:
        s = int(lin.rsplit("-s", 1)[1])
        systems = {r["id"]: r for lv, d in json.loads((PILOT / "gen" / f"systems-s{s}.json").read_text()).items()
                   for r in d["train"]}
        sols, _ = solutions(lin)
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(canon_job, [(systems[sid], d) for sid, d in sols.items()]))
        (OUT / f"canon_{lin}.json").write_text(json.dumps(res))
        eqs = [e for r in res for e in r["equations"].values()]
        changed = [e for e in eqs if sorted(e["canon"]) != sorted(e["raw"])]
        plus = [e for e in eqs if sorted(e["canon_plus"]) != sorted(e["raw"])]
        print(f"{lin}: {len(eqs)} solved equations in {time.time() - t0:.0f}s; rewritten by the pre-set fix {len(changed)}, by the "
              f"extended fix {len(plus)} (steps: drop {sum(s_[0] == 'drop' for e in plus for s_ in e['steps'])}, merge "
              f"{sum(s_[0] == 'merge' for e in plus for s_ in e['steps'])}, simplify "
              f"{sum(s_[0] == 'simplify' for e in plus for s_ in e['steps'])}); notes {sum(1 for e in eqs if e.get('note'))}")


def library_from(equations, path):
    """The pilot's library (curriculum_pilot.grammar_table) from [(own, keys)], saved as .npy and as .npz (key "t")."""
    import numpy as np
    import grammar_ops as GO
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    n = CP.grammar_table(equations, path, GO.alphabet(order, d=D), order)
    np.savez(path.replace(".npy", ".npz"), t=np.load(path))
    return n


def step_rank(workers, lineages):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import curriculum_check2 as C2
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tabs, info = {}, {}
    for lin in lineages:
        s = int(lin.rsplit("-s", 1)[1])
        _, level = solutions(lin)
        res = json.loads((OUT / f"canon_{lin}.json").read_text())
        for form in ("raw", "canon", "canon_plus"):
            eqs = [(int(eq), e[form]) for r in res if level[r["id"]] in MASTERED for eq, e in r["equations"].items()]
            p = str(OUT / "tables" / f"{form}-{lin}.npy")
            info[f"{form}-{lin}"] = {"equations": len(eqs), "pieces": library_from(eqs, p)}
            tabs[f"{form}-{lin}"] = (p.replace(".npy", ".npz"), "t")
        true = [(i, keys) for lv in MASTERED for r in pickle.loads((CHECK1 / "gen" / f"train-s{s}-L{lv}.pkl").read_bytes())
                for i, keys in enumerate(r["structure"]) if keys]
        p = str(OUT / "tables" / f"true-s{s}.npy")
        info[f"true-s{s}"] = {"equations": len(true), "pieces": library_from(true, p)}
        tabs[f"true-s{s}"] = (p.replace(".npy", ".npz"), "t")
    rows = {}
    for name in COMBOS:
        _, _, _, pats = C2.combos()[name]
        dev = pickle.loads((CHECK2 / "gen" / f"dev-{name}.pkl").read_bytes())
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            rows[name] = [x for part in ex.map(C2.rank_job, [(r, {}, tabs, pats) for r in dev]) for x in part]
    (OUT / "ranks.json").write_text(json.dumps({"rows": rows, "info": info}))
    print("libraries: " + "; ".join(f"{k} {v['equations']} equations, {v['pieces']} pieces" for k, v in info.items()))


def step_report(lineages):
    import numpy as np
    d = json.loads((OUT / "ranks.json").read_text())
    rows = d["rows"]
    lines = ["== Equivalent forms as one piece: median rank of the new term (of about 1,100), empty start / one short / two short"]
    better = {lin: 0 for lin in lineages}
    within = {lin: True for lin in lineages}
    better_p = {lin: 0 for lin in lineages}
    within_p = {lin: True for lin in lineages}
    for name in COMBOS:
        lines.append(f"   {name}:")
        R = rows[name]
        for lin in lineages:
            s = lin.rsplit("-s", 1)[1]
            med = {}
            for col in (f"raw-{lin}", f"canon-{lin}", f"canon_plus-{lin}", f"true-s{s}"):
                med[col] = [float(np.median([x[col] for x in R if x["point"] == pt])) if any(x["point"] == pt for x in R) else float("nan")
                            for pt in ("empty start", "one short", "two short")]
            raw1, can1, plus1, tru1 = (med[f"raw-{lin}"][1], med[f"canon-{lin}"][1], med[f"canon_plus-{lin}"][1],
                                       med[f"true-s{s}"][1])
            better[lin] += can1 < raw1
            within[lin] &= can1 <= 2 * tru1
            better_p[lin] += plus1 < raw1
            within_p[lin] &= plus1 <= 2 * tru1
            lines.append(f"      {lin}: raw {' / '.join(f'{v:.0f}' for v in med[f'raw-{lin}'])} | canonical "
                         f"{' / '.join(f'{v:.0f}' for v in med[f'canon-{lin}'])} | canonical+ "
                         f"{' / '.join(f'{v:.0f}' for v in med[f'canon_plus-{lin}'])} | true structures {' / '.join(f'{v:.0f}' for v in med[f'true-s{s}'])}")
    ok = all(better[lin] >= 3 and within[lin] for lin in lineages)
    ok_p = all(better_p[lin] >= 3 and within_p[lin] for lin in lineages)
    lines.append("== The fix works (canonical better than raw in >= 3 of 4 at one term short, within 2x of the true-structure "
                 f"library, both seeds): " + "; ".join(f"{lin}: better in {better[lin]} of 4, within 2x {within[lin]}" for lin in lineages)
                 + f" -> {'YES' if ok else 'NO'}")
    lines.append("== The extended fix (with simplify), same rule: " + "; ".join(
        f"{lin}: better in {better_p[lin]} of 4, within 2x {within_p[lin]}" for lin in lineages) + f" -> {'YES' if ok_p else 'NO'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    lineages = sys.argv[3:] or ["lib-s1", "lib-s2"]
    {"canon": lambda: step_canon(workers, lineages), "rank": lambda: step_rank(workers, lineages),
     "report": lambda: step_report(lineages)}[step]()


if __name__ == "__main__":
    main()
