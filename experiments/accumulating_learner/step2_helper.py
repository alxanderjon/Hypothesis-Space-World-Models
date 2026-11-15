"""A helper for step 2 of the harder fields (owner, 15 November: "if more space opens up allowing you to run faster take
it"): with spare cores, run step 2's remaining tests whose inputs are already saved (the lineage's network and, for the
confidence blend and the made-up control, its library table), writing the same result files the run would write
(atomically), so the run skips any test it finds done. A test is skipped when the run's own thread has reached it (its
Reanalyse for that checkpoint is recorded). Same searches as the run: the same prior, test systems, budget and seed.
Usage: step2_helper.py <workers>"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harder_step2 as S2                                        # noqa: E402  (read only: definitions and paths)

RUN = S2.RUN


def eligible():
    out = []
    for field in S2.FIELDS:
        for seed in S2.SEEDS:
            for arm in S2.ARMS:
                name = f"{arm}-{field}-s{seed}"
                for k in S2.POINTS:
                    tpath = RUN / "test" / f"{name}_t{k}_{field}.json"
                    if tpath.exists() or (RUN / "reanalyse" / f"{name}_r{k}_yield.json").exists():
                        continue
                    tag = f"{field}_d{k}"
                    net = RUN / "nets" / f"{name}_{tag}"
                    if not Path(f"{net}_d4.npy").exists():
                        continue
                    if arm in ("blend", "ctrl"):
                        tab = RUN / "tables" / f"{name}_{tag}.npy"
                        if not tab.exists():
                            continue
                        spec = ("confidence", str(net), str(tab))
                    else:
                        spec = ("net", str(net))
                    out.append((name, field, k, spec, tpath))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    workers = int(sys.argv[1])
    done = 0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        while True:
            todo = eligible()
            if not todo:
                break
            name, field, k, spec, tpath = todo[0]
            t0 = time.time()
            recs = S2.test_records(field)
            res = list(pool.map(S2.search_job, [(r, spec, 1, S2.B, S2.CAP) for r in recs]))
            if not tpath.exists():
                tmp = tpath.with_name(tpath.name + ".helper.tmp")
                tmp.write_text(json.dumps(res, default=float))
                os.replace(tmp, tpath)
                done += 1
                eqs = [e for r in res if r["status"] == "done" for e in r["equations"]]
                S2.log(f"helper: test {name} after {k} tasks: {len(recs)} systems in {time.time() - t0:.0f}s; equations solved "
                       f"{sum(e['first_exact'] is not None for e in eqs)} of {sum(len(r['structure']) for r in recs)}; time-capped "
                       f"{sum(e['time_capped'] for e in eqs)}; failed {sum(r['status'] != 'done' for r in res)}")
            else:
                S2.log(f"helper: test {name} after {k} tasks finished by the run first; the helper's copy is discarded")
    S2.log(f"helper: no eligible test left; {done} tests written")


if __name__ == "__main__":
    main()
