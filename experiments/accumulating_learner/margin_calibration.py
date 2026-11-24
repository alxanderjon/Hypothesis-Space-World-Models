"""Calibrating option B's margin (owner, 24 Nov: "propose the margin and justify it; report it before using it").
The reference is the true structure refitted from the system's recordings. Refitting the same true structure from
starting constants kicked by 1e-6 or 1e-3 (relative) reaches it by another route; how much worse such a refit can be
than the reference, on one step and on rollouts from the 20 starts, is how much a correct answer can differ from the
reference by chance. The margin must cover that spread, and no more. Training systems only: q43s's (wide recordings,
stored test starts, full horizon) and, for the retroactive checks, 20 per earlier rung (narrow recordings, redrawn from
their seeds, judged up to escape). Usage: margin_calibration.py <workers>; writes g2h/margin_calibration.jsonl"""

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

KICKS = [(1, 1e-6), (2, 1e-6), (3, 1e-3), (4, 1e-3)]
PER_RUNG = 20
OLD = ["curriculum", "a2", "c", "p8", "q23", "q33g", "q41"]


class Cap(BaseException):
    pass


def _alarm(*_):
    raise Cap()


def job(args):
    rec, s, full = args
    os.environ["ROLLOUT_FULL"] = "1" if full else "0"
    import warnings
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(1500)
    t0 = time.time()
    try:
        import importlib
        import rollout_check as RC
        RC.FULL = full
        if not rec.get("desc"):
            import replay_clean as RCL
            desc, _ = RCL.recover(rec)
            if desc is None:
                return {"id": rec["id"], "skip": "not recovered"}
            rec = dict(rec, desc=desc)
        rows = RC.check_job((rec, None, s, None, {}, {"perturb": KICKS}))
        return {"id": rec["id"], "full": full, "rows": [{k: r.get(k) for k in ("tag", "perturb", "recordings_exact", "one_step_max",
                "rollout_max", "n_bounded", "unjudgeable")} for r in rows], "seconds": time.time() - t0}
    except Cap:
        return {"id": rec["id"], "skip": "timed out"}
    finally:
        signal.alarm(0)


def cases():
    out = [(r, s, True) for s in LD.SEEDS for r in json.loads((LD.OUT / "gen" / f"q43s-s{s}.json").read_text())["train"]]
    import replay_clean as RCL
    pool = {}
    for L in [f"blend-s{s}" for s in LD.SEEDS]:
        st = json.loads((RCL.SRC / f"{L}.state.json").read_text())
        for sid, r in RCL.records_for(st).items():
            rung = "curriculum" if re.fullmatch(r"s\d-L\d+-(train|test)-\d+", sid) else re.sub(r"-s\d-(train|test)-\d+$", "", sid)
            if rung in OLD and "-train-" in sid:
                pool.setdefault(rung, {})[sid] = r
    for rung in OLD:
        ids = sorted(pool.get(rung, {}))
        step = max(1, len(ids) // PER_RUNG)
        for sid in ids[::step][:PER_RUNG]:
            out.append((pool[rung][sid], int(re.search(r"s(\d)-", sid)[1]), False))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    workers = int(sys.argv[1])
    todo = cases()
    path = LD.OUT.parent / "margin_calibration.jsonl"
    print(f"{len(todo)} training systems", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, open(path, "w") as fh:
        for n, f in enumerate(as_completed([ex.submit(job, a) for a in todo])):
            fh.write(json.dumps(f.result(), default=float) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"   {n} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
