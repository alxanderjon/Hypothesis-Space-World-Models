"""The re-climb's starting state (owner, 25 Nov): each learner lineage as it was at the end of curriculum level 3 (levels
1-3 hold under option B), written as ladder/cL3/<arm>-s<seed>.state.json for the ladder's rung cL5s.
- network: the curriculum checkpoint after level 3 (learn3: levels 1-3 took one round each in every lineage); the
  blend arm also its library table (library3);
- replay: the lineage's level 1-3 solutions from the end of q41 (cleaned and restored), each judged again under option
  B against its true system (redrawn from its seed; judged while the true system stays bounded), keeping only those
  that pass; the tried list is the lineage's level 1-3 systems less the removed ones.
Usage: make_cl3.py <workers> <check|write|report>"""

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
os.environ["ROLLOUT_FULL"] = "0"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402
import replay_clean as RCL                                       # noqa: E402

DST = LD.OUT / "cL3"
PART = DST / "judge.jsonl"
CAP = 1800
PAT = re.compile(r"s(\d)-L([123])-train-\d+$")


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
    try:
        import rollout_check as RC
        RC.FULL = False
        desc, _ = RCL.recover(rec)
        if desc is None:
            return [{"id": rec["id"], "lineage": L, "status": "not recovered", "passes": False} for L in sols]
        rec = dict(rec, desc=desc)
        full = {(L, 0): [eqs.get(i, rec["structure"][i]) for i in range(len(rec["structure"]))] for L, eqs in sols.items()}
        rows = RC.check_job((rec, None, seed, None, full))
        return [{"id": rec["id"], "lineage": r["tag"], "status": "done", "passes": bool(r["passes"]), "unjudgeable": r["unjudgeable"]}
                for r in rows if r.get("tag") not in (None, "reference")]
    except Cap:
        return [{"id": rec["id"], "lineage": L, "status": "timed out", "passes": False} for L in sols]
    finally:
        signal.alarm(0)


def lineages():
    for (arm, s), (base, name, sysfile) in LD.CURRIC.items():
        yield f"{arm}-s{s}", arm, s, base, name, sysfile


def step_check(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    DST.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for L, arm, s, base, name, sysfile in lineages():
        st = json.loads((RCL.DST / f"{L}.state.json").read_text())
        recs = RCL.records_for(st)
        for sid, eqs in st["sols"].items():
            if PAT.match(sid):
                jobs.setdefault(sid, (recs[sid], s, {}))[2][L] = {int(e): v for e, v in eqs.items()}
    done = {json.loads(l)["id"] for l in PART.read_text().splitlines()} if PART.exists() else set()
    todo = [a for sid, a in jobs.items() if sid not in done]
    print(f"cL3: {len(jobs)} level 1-3 systems, {sum(len(a[2]) for a in jobs.values())} solutions ({len(done)} done)", flush=True)
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
    for L, arm, s, base, name, sysfile in lineages():
        st = json.loads((RCL.DST / f"{L}.state.json").read_text())
        sols = {sid: v for sid, v in st["sols"].items() if PAT.match(sid) and (L, sid) in ok}
        dropped = sorted(sid for sid in st["sols"] if PAT.match(sid) and (L, sid) not in ok)
        removed = set(st["cleaned"]["removed"]) - set(st.get("restored", {}).get("restored", []))
        tried = [sid for sid in st["tried"] if PAT.match(sid) and sid not in removed and sid not in dropped]
        net = str(LD.G2H / base / "lineages" / name / "learn3")
        assert Path(net + ".pt").exists(), net
        tables = {}
        if arm == "blend":
            tables = {2: str(LD.G2H / base / "lineages" / name / "library3.npy")}
            assert Path(tables[2]).exists(), tables[2]
        out = {"sols": sols, "tried": tried, "net": net, "tables": tables, "record_files": [str(LD.G2H / sysfile)],
               "made": {"rule": "end of curriculum level 3; replay judged under option B (25 Nov)", "dropped": dropped}}
        (DST / f"{L}.state.json").write_text(json.dumps(out))


def step_report():
    from collections import Counter
    rows = [json.loads(l) for l in PART.read_text().splitlines()]
    for L, *_ in lineages():
        R = [r for r in rows if r["lineage"] == L]
        st = Counter("passes" if r["passes"] else (r["status"] if r["status"] != "done" else ("unjudgeable" if r["unjudgeable"] else "fails")) for r in R)
        print(f"   {L}: {len(R)} level 1-3 solutions; {dict(st)}")


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    {"check": lambda: step_check(workers), "write": step_write, "report": step_report}[step]()


if __name__ == "__main__":
    main()
