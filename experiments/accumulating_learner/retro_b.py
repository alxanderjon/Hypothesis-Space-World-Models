"""Retroactive re-check of every earlier rung under option B (owner, 25 Nov): the curriculum levels, every ladder rung
through q43, and q43s's held-out (judgeability only: its learners never reached a test). For each held-out system the
true system is redrawn from its seed; the reference (the true structure refitted from the system's recordings) and
every distinct found solution of every lineage and round go through rollout_check.check_job (recording checks; one step
at the 20 starts and at 2,000 states; rollouts; margin; ceilings). Systems drawn before 23 Nov are judged while the
true system stays bounded (ROLLOUT_FULL=0). A system that does not finish within CAP seconds counts as failed for
every solution (strict) and is reported. Mastery per lineage: some round solves at least 0.8 x the seed's judgeable
held-out systems; a rung holds for an arm in at least 2 of 3 seeds.
Usage: retro_b.py <workers> <check|report>; results g2h/retro_b/rows.jsonl (resumable)"""

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("G2H_RUN", "ladder")
import ladder as LD                                              # noqa: E402

G2H = LD.G2H
OUTD = G2H / "retro_b"
CAP = int(os.environ.get("RETRO_CAP", "1800"))
LADDER = ["a", "a1w", "a2", "b", "c", "e12", "e3", "p8", "p10", "q23", "q31", "q33", "q33g", "q41", "q43"]
HISTORY = {"a": "dropped off", "e3": "dropped off", "q33": "dropped off at budget 300 (rerun as q33g)"}
CURRIC_LEVELS = [1, 2, 3, 5, 6, 7]


class Cap(BaseException):
    pass


def _alarm(*_):
    raise Cap()


def curriculum_jobs():
    jobs = {}
    for (arm, s), (base, name, sysfile) in LD.CURRIC.items():
        gen = json.loads((G2H / sysfile).read_text())
        lin = json.loads((G2H / base / "lineages" / f"{name}.json").read_text())
        for L in lin["levels"]:
            lv = L["level"]
            test = {r["id"]: r for r in gen[str(lv)]["test"]}
            for rid, r in test.items():
                jobs.setdefault(("curriculum", lv, rid), {"rec": r, "s": s, "group": f"curriculum L{lv}", "full": False, "sols": {}})
            for rnd in L["rounds"]:
                for y in rnd["test"]:
                    if y.get("solved"):
                        eqs = y["equations"] if isinstance(y["equations"], list) else [y["equations"]]
                        sol = [None] * len(test[y["id"]]["structure"])
                        for e in eqs:
                            sol[int(e["eq"])] = e["solution"]
                        jobs[("curriculum", lv, y["id"])]["sols"][f"{arm}-s{s}|{rnd['round']}"] = sol
    return jobs


def ladder_jobs():
    jobs = {}
    for rung in LADDER + ["q43s"]:
        for s in LD.SEEDS:
            test = {r["id"]: r for r in json.loads((LD.OUT / "gen" / f"{rung}-s{s}.json").read_text())["test"]}
            for rid, r in test.items():
                jobs.setdefault((rung, s, rid), {"rec": r, "s": s, "group": rung, "full": rung == "q43s", "sols": {}})
            for arm in LD.ARMS:
                p = LD.OUT / rung / f"{arm}-s{s}.json"
                if not p.exists():
                    continue
                for rnd in json.loads(p.read_text())["rounds"]:
                    for y in rnd["test"]:
                        if y["solved"]:
                            sol = [None] * len(test[y["id"]]["structure"])
                            for e in y["equations"]:
                                sol[int(e["eq"])] = e["solution"]
                            jobs[(rung, s, y["id"])]["sols"][f"{arm}-s{s}|{rnd['round']}"] = sol
    return jobs


def job(j):
    os.environ["ROLLOUT_FULL"] = "1" if j["full"] else "0"
    import warnings
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    t0 = time.time()
    rec, s, group = j["rec"], j["s"], j["group"]
    base = {"id": rec["id"], "group": group, "seed": s}
    try:
        import rollout_check as RC
        RC.FULL = j["full"]
        rung = None if group.startswith("curriculum") else group
        k = int(rec["id"].rsplit("-", 1)[1])
        if group.startswith("curriculum"):
            import replay_clean as RCL
            desc, _ = RCL.recover(rec)
            if desc is None:
                return dict(base, status="not recovered")
            rec = dict(rec, desc=desc)
        sols = {tuple(t.split("|")): v for t, v in j["sols"].items()}
        sols = {(t, int(rd)): v for (t, rd), v in sols.items()}
        rows = RC.check_job((rec, rung, s, k, sols, {"reference": True}))
        ref = [r for r in rows if r.get("tag") == "reference"]
        if not ref:
            return dict(base, status="not recovered")
        ref = ref[0]
        return dict(base, status="done", unjudgeable=ref["unjudgeable"], why=ref.get("unjudgeable_why"),
                    ref={k_: ref.get(k_) for k_ in ("recordings_exact", "one_step_max", "one_step_dense_max", "rollout_max")},
                    results=[{"tag": r["tag"], "round": r["round"], "passes": r["passes"], "passes_absolute": r["passes_absolute"]}
                             for r in rows if r.get("tag") not in (None, "reference")], seconds=time.time() - t0)
    except Cap:
        return dict(base, status="timed out", results=[{"tag": t.split("|")[0], "round": int(t.split("|")[1]), "passes": False,
                                                        "passes_absolute": False} for t in j["sols"]], seconds=time.time() - t0)
    finally:
        signal.alarm(0)


def step_check(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    OUTD.mkdir(parents=True, exist_ok=True)
    part = OUTD / "rows.jsonl"
    done = set()
    if part.exists():
        done = {(r["group"], r["seed"], r["id"]) for r in map(json.loads, part.read_text().splitlines())}
    jobs = list(curriculum_jobs().values()) + list(ladder_jobs().values())
    todo = [j for j in jobs if (j["group"], j["s"], j["rec"]["id"]) not in done]
    print(f"retro B: {len(jobs)} held-out systems ({len(done)} already done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, open(part, "a") as fh:
        futs = [ex.submit(job, j) for j in todo]
        for n, f in enumerate(as_completed(futs)):
            fh.write(json.dumps(f.result(), default=float) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"   {n} of {len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def step_report():
    from collections import Counter, defaultdict
    rows = [json.loads(l) for l in (OUTD / "rows.jsonl").read_text().splitlines()]
    groups = [f"curriculum L{lv}" for lv in CURRIC_LEVELS] + LADDER + ["q43s"]
    print("== Retroactive check under option B (margin 30; one step at the 20 starts and at 2,000 states; ceilings 1e-8 / 1e-5)")
    first_fail = None
    for g in groups:
        R = [r for r in rows if r["group"] == g]
        if not R:
            continue
        status = Counter(r["status"] for r in R)
        judg = {s: sum(1 for r in R if r["seed"] == s and r["status"] == "done" and not r["unjudgeable"]) for s in LD.SEEDS}
        unj = Counter(w for r in R if r["status"] == "done" and r["unjudgeable"] for w in (r["why"] or ["?"])[:1])
        per = defaultdict(lambda: defaultdict(int))                # lineage -> round -> solved
        per_abs = defaultdict(lambda: defaultdict(int))
        lineages = set()
        for r in R:
            for x in r.get("results", []):
                lineages.add(x["tag"])
                if r["status"] == "done" and not r["unjudgeable"] and x["passes"]:
                    per[x["tag"]][x["round"]] += 1
                if x["passes_absolute"]:
                    per_abs[x["tag"]][x["round"]] += 1
        line = (f"{g}{' (' + HISTORY[g] + ')' if g in HISTORY else ''}: {len(R)} held-out systems; unjudgeable "
                f"{sum(1 for r in R if r['status'] == 'done' and r['unjudgeable'])} {dict(unj)}; not finished "
                f"{status.get('timed out', 0)}, not recovered {status.get('not recovered', 0)}; judgeable per seed {judg}")
        print(line)
        if g == "q43s":
            continue
        holds = {}
        for arm in LD.ARMS + (["blend", "net"] if g.startswith("curriculum") else []):
            if arm in holds:
                continue
            m, txt = 0, []
            for s in LD.SEEDS:
                tag = f"{arm}-s{s}"
                if tag not in lineages and not any(t.startswith(tag) for t in lineages):
                    continue
                best = max(per[tag].values(), default=0)
                ok = judg[s] > 0 and best >= 0.8 * judg[s]
                m += ok
                txt.append(f"s{s} {best}/{judg[s]}{'*' if ok else ''}")
            if txt:
                holds[arm] = m >= 2
                print(f"   {arm:6s}: best round {' '.join(txt)} -> {'mastery holds' if m >= 2 else 'mastery does NOT hold'}")
        learners = [a for a in ("blend", "net") if a in holds]
        if first_fail is None and learners and not any(holds[a] for a in learners) and g not in HISTORY:
            first_fail = g
    print(f"== first rung (of those once mastered) where no learner arm holds mastery: {first_fail}")


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    step_check(workers) if step == "check" else step_report()


if __name__ == "__main__":
    main()
