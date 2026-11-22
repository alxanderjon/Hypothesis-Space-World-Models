"""Look-alike check (owner, 22 November): every held-out system counted as solved in a rung's tests, each equation's
found structure against the true structure. Identical term sets count as true; otherwise both are put into their
canonical equivalent form with the system's data (canon_check.canon_job: drop or merge terms while the rest still fits
exactly) and compared; a solved equation whose canonical form still differs is a look-alike (exact on the recordings,
not the true equation). Counts only. Usage: struct_check.py <workers> <rung> [<rung> ...]"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402


def canon_pair(args):
    rec, i, found, true = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import canon_check as CN
    out = CN.canon_job((rec, {str(i): list(found)}))["equations"][str(i)]
    tru = CN.canon_job((rec, {str(i): list(true)}))["equations"][str(i)]
    return {"found_canon": sorted(out["canon_plus"]), "true_canon": sorted(tru["canon_plus"]), "note_found": out.get("note"),
            "note_true": tru.get("note")}


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers, rungs = int(sys.argv[1]), sys.argv[2:]
    report = []
    for rung in rungs:
        rows, jobs = [], []
        for arm in LD.ARMS:
            for s in LD.SEEDS:
                p = LD.OUT / rung / f"{arm}-s{s}.json"
                if not p.exists():
                    continue
                test = {r["id"]: r for r in json.loads((LD.OUT / "gen" / f"{rung}-s{s}.json").read_text())["test"]}
                rec = json.loads(p.read_text())
                for rnd in rec["rounds"]:
                    for y in rnd["test"]:
                        if not y["solved"]:
                            continue
                        for e in y["equations"]:
                            i = int(e["eq"])
                            true, found = sorted(test[y["id"]]["structure"][i]), sorted(e["solution"] or [])
                            row = {"rung": rung, "arm": arm, "seed": s, "round": rnd["round"], "id": y["id"], "eq": i,
                                   "identical": found == true, "found": found, "true": true}
                            rows.append(row)
                            if found != true:
                                jobs.append((len(rows) - 1, (test[y["id"]], i, found, true)))
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(canon_pair, [a for _, a in jobs]))
        for (k, _), r in zip(jobs, res):
            rows[k].update(r, equivalent=r["found_canon"] == r["true_canon"])
        for r in rows:
            r["lookalike"] = not r["identical"] and not r.get("equivalent", False)
        (LD.OUT / rung / "struct_check.json").write_text(json.dumps(rows, indent=1))
        print(f"== {rung}: solved held-out equations (all rounds, all lineages): {len(rows)}; identical to the truth "
              f"{sum(r['identical'] for r in rows)}; equivalent form {sum(r.get('equivalent', False) for r in rows)}; "
              f"look-alikes {sum(r['lookalike'] for r in rows)}")
        for arm in LD.ARMS:
            for s in LD.SEEDS:
                R = [r for r in rows if r["arm"] == arm and r["seed"] == s]
                if not R:
                    continue
                last = max(r["round"] for r in R)
                L = [r for r in R if r["round"] == last]
                sys_ = {}
                for r in L:
                    sys_.setdefault(r["id"], []).append(r["lookalike"])
                print(f"   {arm}-s{s} final round: {len(sys_)} solved systems, {sum(any(v) for v in sys_.values())} with a look-alike equation")
        report.append(rows)


if __name__ == "__main__":
    main()
