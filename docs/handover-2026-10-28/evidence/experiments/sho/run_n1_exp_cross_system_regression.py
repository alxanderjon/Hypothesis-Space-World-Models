"""Resumable process orchestrator for the recovered-exp closed-loop corpus."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


OUT = Path("outputs/sho/stage1B/n1_scalar_exp_cross_system_closed_loop")
CASES = OUT / "cases"
LOGS = OUT / "logs"
COUNTS = {"sho4": 256, "ball4": 50, "ball5": 8}


def result_path(system: str, index: int) -> Path:
    return CASES / system / f"{index:03d}.json"


def valid_result(system: str, index: int) -> bool:
    path = result_path(system, index)
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
        return (row["system"] == system and row["case_index"] == index
                and row["checkpoint_hashes_unchanged"]
                and (not row["production"]["recovered"]
                     or row["recovered_exp"]["recovered"]))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--systems", nargs="*", default=list(COUNTS))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(system, index) for system in args.systems
            for index in range(args.start_index,
                               min(COUNTS[system], args.end_index or COUNTS[system]))
            if not valid_result(system, index)]
    total = sum(max(0, min(COUNTS[s], args.end_index or COUNTS[s]) - args.start_index)
                for s in args.systems)
    reused = total - len(jobs)
    print(json.dumps(dict(total=total, reused=reused, pending=len(jobs),
                          workers=args.workers)), flush=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    active: dict[subprocess.Popen, tuple[str, int, object, float]] = {}
    completed = reused
    failed = None
    iterator = iter(jobs)
    exhausted = False
    while active or not exhausted:
        while not exhausted and len(active) < args.workers and failed is None:
            try:
                system, index = next(iterator)
            except StopIteration:
                exhausted = True
                break
            log_path = LOGS / f"{system}_{index:03d}.log"
            handle = log_path.open("w")
            command = [sys.executable, "-m",
                       "experiments.sho.validate_n1_exp_cross_system_closed_loop",
                       "--system", system, "--index", str(index)]
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=environment)
            active[process] = (system, index, handle, time.monotonic())
        if not active:
            break
        time.sleep(1.0)
        for process in list(active):
            code = process.poll()
            if code is None:
                continue
            system, index, handle, started = active.pop(process)
            handle.close()
            if code != 0 or not valid_result(system, index):
                failed = (system, index, code)
                print(json.dumps(dict(failed=failed)), flush=True)
                for child in active:
                    child.terminate()
                for child, (_, _, child_handle, _) in list(active.items()):
                    child.wait()
                    child_handle.close()
                active.clear()
                break
            completed += 1
            row = json.loads(result_path(system, index).read_text())
            print(json.dumps(dict(completed=completed, total=total, system=system,
                                  index=index,
                                  n1=row["recovered_exp"]["n1_evaluations"],
                                  elapsed=round(time.monotonic() - started, 2))), flush=True)
        if failed is not None:
            raise SystemExit(1)
    print(json.dumps(dict(status="COMPLETE", completed=completed, total=total)), flush=True)


if __name__ == "__main__":
    main()
