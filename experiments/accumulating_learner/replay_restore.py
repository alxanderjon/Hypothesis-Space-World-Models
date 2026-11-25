"""Restore removed replay solutions that pass option B (owner, 25 Nov). The replay cleaning (replay_clean.py, 24 Nov)
removed every stored solution at the end of q41 that failed the absolute definition. Each removed solution is judged
again against its true system (redrawn from its seed) under option B: the same recording checks; one step at the 20
starts and at 2,000 states and rollouts within max(30 x the reference, 1e-12 / 1e-9), the reference being the true
structure refitted from the same recordings; unjudgeable systems (reference failing the recordings or above 1e-6, or a
raised bar above 1e-8 one step / 1e-5 rollout) are not restored. Judged while the true system stays bounded (these
systems were drawn before 23 Nov). Equations a lineage had not solved are filled with their true structure, as in the
cleaning. Passing solutions are added back to ladder/q41c/<lineage>.state.json (solutions and tried list); a "restored"
record is kept. Usage: replay_restore.py <workers> <check|write|report>"""

import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
os.environ["ROLLOUT_FULL"] = "0"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_clean as RCL                                       # noqa: E402

PART = RCL.DST / "replay_restore.jsonl"
CAP = 1800


class Cap(BaseException):
    pass


def _alarm(*_):
    raise Cap()


def job(args):
    rec, seed, sols = args
    import warnings
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    t0 = time.time()
    try:
        import rollout_check as RC
        RC.FULL = False
        desc, _ = RCL.recover(rec)
        if desc is None:
            return [{"id": rec["id"], "lineage": L, "status": "not recovered", "passes": False} for L in sols]
        rec = dict(rec, desc=desc)
        full = {(L, 0): [eqs.get(i, rec["structure"][i]) for i in range(len(rec["structure"]))] for L, eqs in sols.items()}
        rows = RC.check_job((rec, None, seed, None, full))
        return [{"id": rec["id"], "lineage": r["tag"], "status": "done", "passes": bool(r["passes"]), "unjudgeable": r["unjudgeable"],
                 "why": r.get("unjudgeable_why"), "seconds": time.time() - t0} for r in rows if r.get("tag") not in (None, "reference")]
    except Cap:
        return [{"id": rec["id"], "lineage": L, "status": "timed out", "passes": False} for L in sols]
    finally:
        signal.alarm(0)


def removed():
    """{system id: (record, seed, {lineage: {eq: keys}})} for every removed solution."""
    jobs = {}
    for L in RCL.LINEAGES:
        src = json.loads((RCL.SRC / f"{L}.state.json").read_text())
        dst = json.loads((RCL.DST / f"{L}.state.json").read_text())
        recs = RCL.records_for(src)
        for sid in dst["cleaned"]["removed"]:
            jobs.setdefault(sid, (recs[sid], int(L[-1]), {}))[2][L] = {int(e): v for e, v in src["sols"][sid].items()}
    return jobs


def step_check(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    jobs = removed()
    done = {json.loads(l)["id"] for l in PART.read_text().splitlines()} if PART.exists() else set()
    todo = [a for sid, a in jobs.items() if sid not in done]
    print(f"replay restore: {len(jobs)} systems, {sum(len(a[2]) for a in jobs.values())} removed solutions ({len(done)} systems done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, open(PART, "a") as fh:
        for n, f in enumerate(as_completed([ex.submit(job, a) for a in todo])):
            for r in f.result():
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"   {n} of {len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def step_write():
    rows = [json.loads(l) for l in PART.read_text().splitlines()]
    ok = {(r["lineage"], r["id"]) for r in rows if r["passes"]}
    for L in RCL.LINEAGES:
        src = json.loads((RCL.SRC / f"{L}.state.json").read_text())
        p = RCL.DST / f"{L}.state.json"
        st = json.loads(p.read_text())
        back = sorted(sid for sid in st["cleaned"]["removed"] if (L, sid) in ok)
        for sid in back:
            st["sols"][sid] = src["sols"][sid]
            if sid not in st["tried"]:
                st["tried"].append(sid)
        st["restored"] = {"restored": back, "rule": "option B (margin 30, dense 2,000 states, ceilings 1e-8 / 1e-5), 25 Nov"}
        p.write_text(json.dumps(st))


def step_report():
    from collections import Counter
    rows = [json.loads(l) for l in PART.read_text().splitlines()]
    print("== Replay restoration under option B: removed solutions judged again")
    for L in RCL.LINEAGES:
        R = [r for r in rows if r["lineage"] == L]
        st = Counter("restored" if r["passes"] else (r["status"] if r["status"] != "done" else ("unjudgeable" if r["unjudgeable"] else "still fails"))
                     for r in R)
        print(f"   {L}: {len(R)} removed; {dict(st)}")


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    {"check": lambda: step_check(workers), "write": step_write, "report": step_report}[step]()


if __name__ == "__main__":
    main()
