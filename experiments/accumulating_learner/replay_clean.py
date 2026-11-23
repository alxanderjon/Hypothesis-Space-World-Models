"""Replay cleaning before the strict q43 rerun (owner, 23 November): every stored solution of each learner lineage's state
at the end of q41 is checked against its true system under the restored definition of solved (rollout_check.check_job:
the precision fit, the valid range, rollouts judged while the true system stays bounded, at least one time constant,
since these systems were drawn under the old rules). The true system is redrawn from the system's seed (curriculum
systems by the curriculum's generator, ladder systems by the ladder's). Equations a lineage had not solved are filled
with their true structure, so the rollout tests the stored ones only. A system whose stored solution fails is removed
from the lineage's solutions and from its tried list (so Reanalyse does not re-find the same look-alike); the cleaned
states are written to ladder/q41c/. Usage: replay_clean.py <workers> <check|write|report>"""

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
os.environ["ROLLOUT_FULL"] = "0"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

SRC, DST = LD.G2H / "ladder" / "q41", LD.G2H / "ladder" / "q41c"
LINEAGES = [f"{a}-s{s}" for a in ("blend", "net") for s in LD.SEEDS]


def recover(rec):
    """(desc, syn) for a curriculum or ladder record, by redrawing from its seed until the recordings match."""
    import domains_g1 as DG
    import dream_g2 as DR
    want = LD.data_key(rec["bursts"])
    m = re.fullmatch(r"s(\d)-L(\d+)-(train|test)-(\d+)", rec["id"])
    if m:
        import curriculum_check as CC
        import curriculum_pilot as CP
        s, lv, split, k = int(m[1]), int(m[2]), m[3], int(m[4])
        keys = CC.piece_keys()
        allowed = {l_: keys[l_] for l_ in CP.LEVELS if l_ <= lv}
        prefix = f"pilot-s{s}-L{lv}-{split}"
        drawf = lambda seed: CC.draw(seed, allowed, lv, CC.TRAIN_TERMS, None)
    else:
        rung, s, split, k = re.fullmatch(r"(.+)-s(\d)-(train|test)-(\d+)", rec["id"]).groups()
        R = LD.RUNGS[rung]
        keys = LD.piece_keys(R["d"])
        allowed = {lv: keys[lv] for lv in R["levels"]}
        prefix = f"ladder-{rung}-s{s}-{split}"
        drawf = lambda seed: LD.draw(seed, allowed, R["terms"], R["d"], False, R.get("max_k"))
    for r in range(200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = drawf(seed)
        if DR.prescreen(desc) is not None:
            continue
        try:
            syn = DG.build(desc)
        except Exception:
            continue
        if LD.data_key(syn.bursts(8)) == want:
            return desc, syn
    return None, None


def job(args):
    rec, seed, sols = args                                         # sols: {lineage: {eq: keys}}
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import rollout_check as RC
    t0 = time.time()
    desc, _ = recover(rec)
    if desc is None:
        return [{"id": rec["id"], "lineage": L, "recovered": False, "passes": False} for L in sols]
    rec = dict(rec, desc=desc)
    full = {}
    for L, eqs in sols.items():
        full[(L, 0)] = [eqs.get(i, rec["structure"][i]) for i in range(len(rec["structure"]))]
    rows = RC.check_job((rec, None, seed, None, full))
    out = []
    for r in rows:
        if r.get("tag") is None:
            continue
        out.append({"id": rec["id"], "lineage": r["tag"], "recovered": True, "passes": bool(r["passes"]),
                    "recordings_exact": r["recordings_exact"], "one_step_max": r["one_step_max"], "rollout_max": r["rollout_max"],
                    "n_bounded": r["n_bounded"], "seconds": time.time() - t0})
    return out


def records_for(state):
    recs = {}
    for f in state["record_files"]:
        obj = json.loads(Path(f).read_text())
        for part in ([obj] if "train" in obj else obj.values()):
            for r in part["train"]:
                recs[r["id"]] = r
    return recs


def step_check(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    jobs = {}
    for L in LINEAGES:
        st = json.loads((SRC / f"{L}.state.json").read_text())
        recs = records_for(st)
        s = int(L[-1])
        for sid, eqs in st["sols"].items():
            jobs.setdefault(sid, (recs[sid], s, {}))[2][L] = {int(e): v for e, v in eqs.items()}
    args = list(jobs.values())
    print(f"replay check: {len(args)} systems, {sum(len(a[2]) for a in args)} lineage solutions", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for k, part in enumerate(ex.map(job, args, chunksize=1)):
            rows += part
            if k % 100 == 0:
                print(f"   {k} of {len(args)} systems ({time.time() - t0:.0f}s)", flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "replay_check.json").write_text(json.dumps(rows, default=float))
    print(f"replay check done in {time.time() - t0:.0f}s")


def step_write():
    rows = json.loads((DST / "replay_check.json").read_text())
    fail = {(r["lineage"], r["id"]) for r in rows if not r["passes"]}
    for L in LINEAGES:
        st = json.loads((SRC / f"{L}.state.json").read_text())
        drop = {sid for sid in st["sols"] if (L, sid) in fail}
        st["sols"] = {sid: v for sid, v in st["sols"].items() if sid not in drop}
        st["tried"] = [sid for sid in st["tried"] if sid not in drop]
        st["cleaned"] = {"removed": sorted(drop), "rule": "restored definition (judged up to escape), 23 Nov"}
        (DST / f"{L}.state.json").write_text(json.dumps(st))


def step_report():
    from collections import Counter
    rows = json.loads((DST / "replay_check.json").read_text())
    print("== Replay cleaning: stored solutions at the end of q41 against their true systems (restored definition)")
    for L in LINEAGES:
        R = [r for r in rows if r["lineage"] == L]
        src = Counter(re.sub(r"-s\d-(train|test)-\d+$", "", r["id"]) if not r["id"].startswith("s") else "curriculum" for r in R if not r["passes"])
        why = Counter("not recovered" if not r["recovered"] else ("recordings" if not r["recordings_exact"] else
                      ("one step" if r["one_step_max"] > 1e-12 else "rollout")) for r in R if not r["passes"])
        print(f"   {L}: {len(R)} solved systems; removed {sum(not r['passes'] for r in R)} "
              f"({100 * sum(not r['passes'] for r in R) / max(1, len(R)):.0f}%); by rung {dict(src)}; why {dict(why)}")


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    {"check": lambda: step_check(workers), "write": step_write, "report": step_report}[step]()


if __name__ == "__main__":
    main()
