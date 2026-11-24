"""Why true structures fail the real definition of solved (owner, 24 Nov: fix the fitter so true equations pass).
Training systems only (held-out systems are used for counts, never for diagnosis). For each system, its own true
structure goes through rollout_check.check_job; failing systems are dissected per equation: the rate fit's residual
against the gate, whether the thorough fit and the step-integral refit ran, the refit and held-back errors against the
bar, degenerate terms; and for systems past the recordings, the shooting fit's residuals and the one-step and rollout
errors. Usage: diag_truth_fit.py <workers> <set> where set is q43s-train or replay; writes g2h/diag_truth_<set>.jsonl"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("G2H_RUN", "ladder")
import ladder as LD                                              # noqa: E402


def truth_of(rec):
    return [e["keys"] if isinstance(e, dict) else e for e in rec["structure"]]


CAP = int(os.environ.get("DIAG_CAP", "900"))                       # seconds per system; a straggler is recorded as timed out


class _Cap(BaseException):                                          # not swallowed by the fitters' except Exception
    pass


def _alarm(*_):
    raise _Cap()


def job(args):
    import signal
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    t0 = time.time()
    try:
        return _job(args)
    except _Cap:
        return {"id": args[0]["id"], "passes": False, "recovered": True, "timed_out": True, "seconds": time.time() - t0}
    finally:
        signal.alarm(0)


def _job(args):
    rec, rung, s, k = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import rollout_check as RC
    t0 = time.time()
    row = [x for x in RC.check_job((rec, rung, s, k, {("truth", 0): truth_of(rec)})) if x.get("tag")][0]
    out = {"id": rec["id"], "passes": bool(row.get("passes")), "recovered": row.get("recovered"),
           "recordings": row.get("recordings_exact"), "one_step": row.get("one_step_max"), "rollout": row.get("rollout_max"),
           "shoot_before": row.get("shooting_residual_before"), "shoot_after": row.get("shooting_residual_after"),
           "n_bounded": row.get("n_bounded"), "error": row.get("error")}
    if not out["passes"] and out["recovered"]:
        ctx = FG.system_context(rec["bursts"], seed=s)
        fitter = FG.make_fitter(ctx, seed=s)
        eqs = []
        for i, keys in enumerate(truth_of(rec)):
            nd = FG.evaluate_structure(tuple(keys), ctx, i, fitter)
            eqs.append({"n_terms": len(keys), "nonlinear": sum(1 for kk in keys if FG.info(kk).k > 0),
                        "rel_rate": nd["rel_rate"], "floor_rel": nd["floor_rel"], "gate": nd["gate"], "thorough": nd["thorough"],
                        "rel_t": nd["rel_t"], "rel_v": nd["rel_v"], "tau": nd["tau"], "degenerate": nd["degenerate"], "exact": nd["exact"]})
        out["eqs"] = eqs
    out["seconds"] = time.time() - t0
    return out


def cases(which):
    if which == "q43s-train":
        for s in LD.SEEDS:
            for r in json.loads((LD.OUT / "gen" / f"q43s-s{s}.json").read_text())["train"]:
                yield r, "q43s", s, int(r["id"].rsplit("-", 1)[1])
    elif which == "q43s-dev":                                    # development set: the baseline's failures and 20 of its passes
        base = [json.loads(l) for l in (LD.OUT.parent / "diag_truth_q43s-train.baseline.jsonl").read_text().splitlines()]
        keep = {r["id"] for r in base if not r["passes"]} | {r["id"] for r in base if r["passes"]}.__class__(
            sorted(r["id"] for r in base if r["passes"])[:20])
        for s in LD.SEEDS:
            for r in json.loads((LD.OUT / "gen" / f"q43s-s{s}.json").read_text())["train"]:
                if r["id"] in keep:
                    yield r, "q43s", s, int(r["id"].rsplit("-", 1)[1])
    elif which == "replay":                                      # the replay check's failures that had exactly the true terms
        import replay_clean as RCL
        rows = json.loads((RCL.DST / "replay_check.json").read_text())
        fail = {(r["lineage"], r["id"]) for r in rows if not r["passes"]}
        seen = set()
        for L in RCL.LINEAGES:
            st = json.loads((RCL.SRC / f"{L}.state.json").read_text())
            recs = RCL.records_for(st)
            for sid, eqs in st["sols"].items():
                if (L, sid) not in fail or sid in seen:
                    continue
                truth = truth_of(recs[sid])
                if all(sorted(v) == sorted(truth[int(e)]) for e, v in eqs.items()):
                    seen.add(sid)
                    yield recs[sid], None, int(L[-1]), None


def replay_job(args):
    rec, _, s, _ = args
    import replay_clean as RCL
    desc, _ = RCL.recover(rec)
    if desc is None:
        return {"id": rec["id"], "passes": False, "recovered": False}
    return job((dict(rec, desc=desc), None, s, None))


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    workers, which = int(sys.argv[1]), sys.argv[2]
    if which == "replay":
        os.environ["ROLLOUT_FULL"] = "0"
    todo = list(cases(which))
    path = LD.OUT.parent / f"diag_truth_{which}.jsonl"
    print(f"{which}: {len(todo)} systems", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, open(path, "w") as fh:
        fn = replay_job if which == "replay" else job
        futs = [ex.submit(fn, a) for a in todo]
        for n, f in enumerate(as_completed(futs)):
            fh.write(json.dumps(f.result(), default=float) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"   {n} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
