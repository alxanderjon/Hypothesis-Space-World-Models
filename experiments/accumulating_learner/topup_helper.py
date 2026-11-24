"""Spare strict systems for a rung's top-up, on otherwise idle cores (24 Nov). The top-up (ladder.py <w> <rung> topup)
saves nothing until its pass ends, and if a cell is still short a second pass costs hours. This helper draws systems for
the same cells under the same rules (ladder.gen_job, strict) from seed indices the top-up never uses (from BASE on),
saving each result as it finishes. After the top-up ends, `merge` fills cells that are still short from the helper's
accepted systems, checked against every earlier set and against each other, in seed-index order.
Usage: topup_helper.py <workers> <rung> <draw|merge>"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

BASE = 5000                                                      # top-up passes use k0 + 1000 * pass, pass < 4


def cells(rung):
    R = LD.RUNGS[rung]
    want = {"train": R.get("n_train", LD.N_TRAIN), "test": LD.N_TEST}
    out = {s: json.loads((LD.OUT / "gen" / f"{rung}-s{s}.json").read_text()) for s in LD.SEEDS}
    return R, want, out


def step_draw(workers, rung):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    R, want, out = cells(rung)
    keys = LD.piece_keys(R["d"])
    allowed = {lv: keys[lv] for lv in R["levels"]}
    new = LD.new_keys(rung, keys)
    part = LD.OUT / "gen" / f"{rung}.helper.jsonl"
    done = set()
    if part.exists():
        done = {(r["s"], r["sp"], r["k"]) for r in map(json.loads, part.read_text().splitlines())}
    jobs = []
    for s in LD.SEEDS:
        for sp in want:
            n = want[sp] - len(out[s][sp])
            jobs += [(s, sp, k) for k in range(BASE, BASE + max(0, n)) if (s, sp, k) not in done]
    print(f"helper {rung}: {len(jobs)} draws ({len(done)} already done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex, open(part, "a") as fh:
        futs = {ex.submit(LD.gen_job, (f"ladder-{rung}-s{s}-{sp}", k, allowed, R["terms"], R["d"], new, False,
                                       R.get("max_k"), R.get("strict", False))): (s, sp, k) for s, sp, k in jobs}
        for i, f in enumerate(as_completed(futs)):
            s, sp, k = futs[f]
            r = f.result()
            rec = {"s": s, "sp": sp, "k": k, "accepted": bool(r["accepted"])}
            if r["accepted"]:
                rec.update(structure=r["structure"], bursts=r["bursts"], desc=r.get("desc"), test=r.get("test"))
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(f"   {i + 1} of {len(jobs)} draws, {s} {sp} {k}: {'accepted' if r['accepted'] else 'none'} ({time.time() - t0:.0f}s)", flush=True)


def step_merge(rung):
    R, want, out = cells(rung)
    part = LD.OUT / "gen" / f"{rung}.helper.jsonl"
    extra = sorted((r for r in map(json.loads, part.read_text().splitlines()) if r["accepted"]), key=lambda r: (r["s"], r["sp"], r["k"]))
    old = LD.earlier_keys(rung)
    seen = {LD.data_key(r["bursts"]) for s in LD.SEEDS for sp in out[s].values() for r in sp}
    added = 0
    for r in extra:
        s, sp = r["s"], r["sp"]
        if len(out[s][sp]) >= want[sp]:
            continue
        key = LD.data_key(r["bursts"])
        if key in old or key in seen:
            continue
        seen.add(key)
        t = r.get("test") or {}
        out[s][sp].append({"id": f"{rung}-s{s}-{sp}-{r['k']}", "rung": rung, "structure": r["structure"], "bursts": r["bursts"],
                           "desc": r.get("desc"), "test_starts": t.get("starts"), "horizon": t.get("T")})
        added += 1
    for s in LD.SEEDS:
        (LD.OUT / "gen" / f"{rung}-s{s}.json").write_text(json.dumps(out[s]))
    print(f"merge {rung}: {added} helper systems added; now { {s: {sp: len(v) for sp, v in out[s].items()} for s in LD.SEEDS} }")


def main():
    workers, rung, step = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    step_draw(workers, rung) if step == "draw" else step_merge(rung)


if __name__ == "__main__":
    main()
