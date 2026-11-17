"""Dream-building speed (owner, 17 November, before rung d): worker-seconds per finished dream for teacher dreams and for
own dreams (a learner lineage and a fresh one), split into pre-screen, build and example time; and a validation of the
pre-screen on pilot seeds: every draw it rejects is also given the full build, which must accept none of them.
Rung c's shape (3 variables, 3 terms). Read only on ladder.py. Usage: dream_speed.py <workers>"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "dream_speed")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

N_DREAMS, N_VALIDATE = 40, 300


def timed_dream(args):
    """rung_dream_job's loop, timed per stage; with validate=True, a single draw that also gets the full build if the
    pre-screen rejects it."""
    prefix, k, allowed, terms, d, validate = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    t = {"pre": 0.0, "build": 0.0, "examples": 0.0, "draws": 0, "pre_rejects": 0, "build_rejects": 0}
    for r in range(1 if validate else 200):
        seed = f"{prefix}-{k}" if r == 0 else f"{prefix}-{k}-r{r}"
        desc = LD.draw(seed, allowed, terms, d)
        t["draws"] += 1
        t0 = time.perf_counter()
        why = DR.prescreen(desc)
        t["pre"] += time.perf_counter() - t0
        if why is not None and not validate:
            t["pre_rejects"] += 1
            continue
        t0 = time.perf_counter()
        try:
            syn = DG.build(desc)
        except Exception as ex:
            t["build"] += time.perf_counter() - t0
            t["build_rejects"] += 1
            if validate:
                return {"pre": why, "build": f"rejected: {str(ex)[:60]}", "pre_s": t["pre"], "build_s": t["build"]}
            continue
        t["build"] += time.perf_counter() - t0
        if validate:
            return {"pre": why, "build": "ACCEPTED", "pre_s": t["pre"], "build_s": t["build"]}
        t0 = time.perf_counter()
        try:
            ex = DR.make_examples(syn.bursts(8), [eq["keys"] for eq in syn.structure], seed=10_000 + k)
        except Exception:
            ex = []
        t["examples"] += time.perf_counter() - t0
        t["n_examples"] = len(ex)
        return t
    t["n_examples"] = 0
    return t


def own_groups(sols, recs, d):
    import dream_g2 as DR
    import search_g1 as SG
    of = {k: lv for lv, ks in LD.piece_keys(d).items() for k in ks}
    order = set(SG.vocabulary_order(d))
    usage = {}
    for s, eqs in sols.items():
        if len(recs[s]["structure"]) != d:
            continue
        for eq, keys in eqs.items():
            for k in keys:
                rk = DR.role_key(k, int(eq), d)
                if rk is not None and rk in order:
                    usage[rk] = 1
    groups = {}
    for k in sorted(usage):
        groups.setdefault(of.get(k, LD.OTHER), []).append(k)
    return groups


def main():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import numpy as np
    workers = int(sys.argv[1])
    d, terms = 3, (3, 3)
    keys = LD.piece_keys(d)
    teacher = {lv: keys[lv] for lv in (1, 2, 3, 5, 6, 7)}
    # a learner lineage's own pieces: blend-s1 after rung b plus its rung c first wake; the fresh lineage's: its first wake
    G = LD.G2H / "ladder"
    st = json.loads((G / "b" / "blend-s1.state.json").read_text())
    sols = {k: {int(e): v for e, v in x.items()} for k, x in st["sols"].items()}
    recs = {}
    for f in st["record_files"] + [str(G / "gen" / "c-s1.json")]:
        obj = json.loads(Path(f).read_text())
        for part in ([obj] if "train" in obj else obj.values()):
            for r in part["train"]:
                recs[r["id"]] = r
    def add_wake(path, into):
        for y in json.loads(Path(path).read_text()):
            if y["status"] == "done":
                for e in y["equations"]:
                    if e["first_exact"] is not None and e["solution"]:
                        into.setdefault(y["id"], {})[int(e["eq"])] = e["solution"]
    add_wake(G / "c" / "blend-s1" / "r1_wake.json", sols)
    fresh = {}
    add_wake(G / "c" / "fresh-s1" / "r1_wake.json", fresh)
    sets = {"teacher": teacher, "own (learner, blend-s1)": own_groups(sols, recs, d), "own (fresh-s1)": own_groups(fresh, recs, d)}
    out = {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name, allowed in sets.items():
            t0 = time.time()
            res = list(ex.map(timed_dream, [(f"dream-speed-{name}", k, allowed, terms, d, False) for k in range(N_DREAMS)]))
            tot = {k: float(np.sum([r[k] for r in res])) for k in ("pre", "build", "examples", "draws", "pre_rejects", "build_rejects", "n_examples")}
            out[name] = dict(tot, dreams=N_DREAMS, with_examples=sum(r["n_examples"] > 0 for r in res),
                             worker_seconds_per_dream=(tot["pre"] + tot["build"] + tot["examples"]) / N_DREAMS,
                             pieces=sum(len(v) for v in allowed.values()), wall=time.time() - t0)
            print(name, json.dumps(out[name]), flush=True)
        t0 = time.time()
        val = list(ex.map(timed_dream, [("dream-speed-validate", k, teacher, terms, d, True) for k in range(N_VALIDATE)]))
    rej = [v for v in val if v["pre"] is not None]
    passed = [v for v in val if v["pre"] is None]
    out["validation"] = {"draws": N_VALIDATE, "pre_rejected": len(rej), "pre_rejected_but_build_accepts": sum(v["build"] == "ACCEPTED" for v in rej),
                         "passed_pre": len(passed), "passed_pre_build_accepts": sum(v["build"] == "ACCEPTED" for v in passed),
                         "mean_pre_s": float(np.mean([v["pre_s"] for v in val])),
                         "mean_build_s_of_pre_rejected": float(np.mean([v["build_s"] for v in rej])) if rej else None,
                         "mean_build_s_of_pre_passed": float(np.mean([v["build_s"] for v in passed])) if passed else None,
                         "wall": time.time() - t0}
    print("validation", json.dumps(out["validation"]), flush=True)
    (LD.OUT / "dream_speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
