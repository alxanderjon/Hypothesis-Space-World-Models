"""The blend test's third seed (owner, 14 November; research log "Owner: a third seed of the blend test, and the
MuZero-family quick checks alongside it"): new systems generated exactly as the curriculum pilot's, from seed 3's names,
checked to be unlike every earlier system set; then the top-100 blend, the network alone and the fresh start at each
level, side by side in one process, with the pilot's budgets, mastery rule and final test.
Usage: curriculum_seed3.py <workers> <gen|run|report> [smoke]"""

import hashlib
import json
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

ARGS = list(sys.argv)
SMOKE = "smoke" in ARGS[2:]
os.environ.setdefault("G2H_RUN", "cseed3_smoke" if SMOKE else "cseed3")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum_pilot as CP                                    # noqa: E402
import curriculum_blend as CB                                    # noqa: E402

C, D, CC = CP.C, CP.D, CP.CC
OUT = C.RUN
G2H = C.ROOT / "outputs/accumulating_learner/g2h"
SEED = 3
EARLIER = [G2H / "cpilot/gen", G2H / "curriculum/gen", G2H / "curriculum2/gen"]   # every earlier curriculum system set


def data_key(bursts):
    import numpy as np
    return hashlib.sha1(np.round(np.asarray(bursts, dtype=float), 10).tobytes()).hexdigest()


def earlier_keys():
    """Data fingerprints of every system in the earlier curriculum sets (pilot seeds 1-2, both curriculum checks)."""
    keys, n = set(), 0
    for d in EARLIER:
        for p in sorted(d.iterdir()):
            if p.suffix == ".json":
                recs = [r for lv in json.loads(p.read_text()).values() for sp in lv.values() for r in sp]
            elif p.suffix == ".pkl":
                obj = pickle.loads(p.read_bytes())
                recs = obj if isinstance(obj, list) else []
            else:
                continue
            for r in recs:
                if isinstance(r, dict) and r.get("bursts") is not None:
                    keys.add(data_key(r["bursts"]))
                    n += 1
    return keys, n


def step_gen(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    keys = CC.piece_keys()
    jobs = []
    for lv in CP.LEVELS:
        allowed = {l_: keys[l_] for l_ in CP.LEVELS if l_ <= lv}
        for split, n in (("train", CP.N_TRAIN), ("test", CP.N_TEST)):
            jobs += [((lv, split, k), (f"{'smoke' if SMOKE else 'pilot'}-s{SEED}-L{lv}-{split}", k, allowed, lv,
                                       CC.TRAIN_TERMS, None, False))
                     for k in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(CC.gen_job, [a for _, a in jobs]))
    systems = {lv: {"train": [], "test": []} for lv in CP.LEVELS}
    for ((lv, split, k), _), r in zip(jobs, res):
        if r["accepted"]:
            systems[lv][split].append({"id": f"s{SEED}-L{lv}-{split}-{k}", "level": lv, "structure": r["structure"],
                                       "bursts": r["bursts"]})
    old, n_old = earlier_keys()
    new = [data_key(r["bursts"]) for lv in systems.values() for sp in lv.values() for r in sp]
    clash = sum(k in old for k in new)
    dup = len(new) - len(set(new))
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    counts = {lv: {sp: len(v) for sp, v in d.items()} for lv, d in systems.items()}
    print(f"gen: {sum(r['accepted'] for r in res)} of {len(jobs)} systems accepted in {time.time() - t0:.0f}s; per level "
          f"{counts}; checked against {n_old} earlier systems: {clash} with identical data, {dup} repeated within seed {SEED}")
    if clash or dup:
        raise SystemExit("seed 3's systems are not all new: not running")
    (OUT / "gen" / f"systems-s{SEED}.json").write_text(json.dumps(systems))


def step_run(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import grammar_ops as GO
    import search_g1 as SG
    order = SG.vocabulary_order(D)
    tokens = GO.alphabet(order, d=D)
    gpus = queue.Queue()
    free = CC.free_gpus()
    if not free:
        raise SystemExit("no free GPU now; the network does not train on the CPU (owner's rule)")
    for g in free[:2]:
        gpus.put(g)
    systems = {int(k): v for k, v in json.loads((OUT / "gen" / f"systems-s{SEED}.json").read_text()).items()}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        lineages = []
        # the blend first: its constructor clears the parent class's folder under the network lineage's name
        if not (OUT / "lineages" / f"net-blend-s{SEED}.json").exists():
            lineages.append(CB.BlendLineage(SEED, pool, gpus, systems, tokens, order))
        for kind in ("net", "lib-fresh"):
            if not (OUT / "lineages" / f"{kind}-s{SEED}.json").exists():
                lineages.append(CP.Lineage(kind, SEED, pool, gpus, systems, [], tokens, order))
        CP.log(f"seed {SEED}: {len(lineages)} lineages ({', '.join(x.name for x in lineages)}); levels {CP.LEVELS}; budgets "
               f"{CP.BUDGETS}; GPUs {free[:2]}; workers {workers}")
        threads = [threading.Thread(target=x.run, name=x.name) for x in lineages]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    CP.log(f"seed {SEED}: all lineages done")


def per_level(rec):
    return {lv: sum(y["solved"] for y in rec["final"] if y["level"] == lv) for lv in CP.LEVELS}


def step_report():
    arms = (("blend", "net-blend"), ("network alone", "net"), ("fresh start", "lib-fresh"))
    recs = {name: json.loads((OUT / "lineages" / f"{kind}-s{SEED}.json").read_text()) for name, kind in arms}
    per = {name: per_level(r) for name, r in recs.items()}
    hi = {k: sum(v.get(lv, 0) for lv in (6, 7)) for k, v in per.items()}
    lo = {k: sum(v.get(lv, 0) for lv in (1, 2, 3, 5)) for k, v in per.items()}
    lines = [f"== Seed {SEED}, final test: held-out systems solved per level (of {CP.N_TEST})"]
    for name, _ in arms:
        rounds = " ".join(f"L{L_['level']} {len(L_['rounds']) if L_['mastered'] else 'x'}" for L_ in recs[name]["levels"])
        lines.append(f"   {name:>13}: " + "; ".join(f"L{lv} {per[name][lv]}" for lv in CP.LEVELS)
                     + f" | levels 6-7 {hi[name]}, levels 1-5 {lo[name]}, total {sum(per[name].values())} | rounds to mastery: {rounds}")
    ok1, ok2 = hi["blend"] > hi["network alone"], lo["blend"] >= lo["network alone"] - 1
    lines.append(f"== Pass rule (seed {SEED}): levels 6-7 more with the blend {ok1}; levels 1-5 no loss beyond 1 {ok2} -> "
                 f"{'CONFIRMED' if ok1 and ok2 else 'NOT confirmed'}")
    if not SMOKE:
        lines.append("== All three seeds, final-test totals (of 60): seed 1 / seed 2 / seed 3")
        for name, kind in arms:
            tot = []
            for s in (1, 2):
                p = (G2H / "cblend/lineages" / f"net-blend-s{s}.json") if kind == "net-blend" else (G2H / "cpilot/lineages" / f"{kind}-s{s}.json")
                tot.append(sum(per_level(json.loads(p.read_text())).values()))
            tot.append(sum(per[name].values()))
            lines.append(f"   {name:>13}: " + " / ".join(str(t) for t in tot))
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers = int(ARGS[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ARGS[2:] if s != "smoke"]:
        free = shutil.disk_usage(OUT.resolve()).free / 2 ** 30
        if free < 2:
            raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {OUT.resolve()} (owner's crash guard)")
        {"gen": lambda: step_gen(workers), "run": lambda: step_run(workers), "report": step_report}[step]()


if __name__ == "__main__":
    main()
