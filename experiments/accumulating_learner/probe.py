"""The fixed probe set (owner, 22 November): accumulation measured directly on the same held-out systems from rung to rung.
Probe set (chosen once, never trained on): the first held-out system of seeds 1 and 2 from each of a1, a2, b, c, e12,
p8, p10, q23, q31, q33g, and the first held-out system of curriculum levels 3 and 7 (seed 1's curriculum set). Each
learner lineage's end-of-rung network (curriculum end, a1w, a2, b, c, e12, p8, p10, q23, q31, q33g, then each new rung)
searches every probe from the probe's own rung on: the real search, 3,000 structures and 600 thorough fits per equation,
stopping at an equation's first exact solution (tries = structures to it). The fresh networks of q33g search all probes as
a reference. CPU only; a job is submitted only while >= PROBE_SPARE cores are idle (q41/q43 first).
Usage: probe.py <workers> <select|run [checkpoint ...]|report>"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "probe")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402

G2H, OUT = LD.G2H, LD.OUT
L_DIR = G2H / "ladder"
SOURCES = ["curriculum", "a1", "a2", "b", "c", "e12", "p8", "p10", "q23", "q31", "q33g"]
CHECKPOINTS = ["curriculum", "a1w", "a2", "b", "c", "e12", "p8", "p10", "q23", "q31", "q33g"]   # then q41, q43, ...
SOURCE_AT = {"curriculum": "curriculum", "a1": "a1w", "a2": "a2", "b": "b", "c": "c", "e12": "e12", "p8": "p8", "p10": "p10",
             "q23": "q23", "q31": "q31", "q33g": "q33g"}           # the checkpoint at which a probe's rung was learned
BUDGET, CAP = 3000, 600
SPARE = int(os.environ.get("PROBE_SPARE", "4"))


def step_select():
    probes = []
    cur = json.loads((G2H / "cpilot/gen/systems-s1.json").read_text())
    for lv in ("3", "7"):
        r = cur[lv]["test"][0]
        probes.append(dict(r, id=f"probe-curriculum-L{lv}", source="curriculum"))
    for src in SOURCES[1:]:
        for s in (1, 2):
            r = json.loads((L_DIR / "gen" / f"{src}-s{s}.json").read_text())["test"][0]
            probes.append(dict(r, id=f"probe-{src}-s{s}", source=src, orig_id=r["id"]))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "probes.json"
    if p.exists():
        raise SystemExit("the probe set exists already (chosen once)")
    p.write_text(json.dumps(probes))
    print(f"probe set: {len(probes)} systems: " + ", ".join(f"{x['id']} ({len(x['structure'])} variables)" for x in probes))


def spec_at(arm, s, cp):
    """The lineage's network (and the blend's library tables) at the end of checkpoint cp."""
    if cp == "curriculum":
        base, name, _ = LD.CURRIC[(arm, s)]
        rec = json.loads((G2H / base / "lineages" / f"{name}.json").read_text())
        n = sum(len(L["rounds"]) for L in rec["levels"])
        net = str(G2H / base / "lineages" / name / f"learn{n}")
        return ("confidence", net, {2: str(G2H / base / "lineages" / name / f"library{n}.npy")}) if arm == "blend" else ("net", net)
    st = json.loads((L_DIR / cp / f"{arm}-s{s}.state.json").read_text())
    tables = {int(k): v for k, v in st["tables"].items()}
    return ("confidence", st["net"], tables) if arm == "blend" else ("net", st["net"])


def search_job(args):
    """The real search per equation, stopping at the first exact solution."""
    rec, spec, seed = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import search_g1 as SG
    t0 = time.time()
    ctx = SG.system_context(np.array(rec["bursts"]), seed=seed)
    fitter = FG.make_fitter(ctx, seed=seed)
    prior = LD.make_prior(spec)
    d = len(rec["structure"])
    eqs = []
    for i in range(d):
        se = SG.EquationSearchL1(i, ctx, BUDGET, prior, fitter, np.random.default_rng([seed, i]), time_cap=LD.C.TIME_CAP / d,
                                 thorough_cap=CAP).run(extra_after_exact=0.0, min_extra=0)
        eqs.append({"first_exact": se.eff["first_exact"], "structures": se.eff["structures"], "time_capped": se.eff.get("time_capped", False)})
    return {"id": rec["id"], "solved": all(e["first_exact"] is not None for e in eqs), "equations": eqs, "seconds": time.time() - t0}


def jobs_for(cps):
    probes = json.loads((OUT / "probes.json").read_text())
    order = {cp: k for k, cp in enumerate(CHECKPOINTS + ["q41", "q43"])}
    jobs = []
    for cp in cps:
        for arm in ("blend", "net"):
            for s in LD.SEEDS:
                tag = f"{arm}-s{s}-{cp}"
                if (OUT / "results" / f"{tag}.json").exists():
                    continue
                todo = [p for p in probes if order[SOURCE_AT[p["source"]]] <= order[cp]]
                jobs.append((tag, spec_at(arm, s, cp), s, todo))
    if "q33g" in cps:                                            # the reference: q33g's fresh networks on every probe
        for s in LD.SEEDS:
            tag = f"fresh-s{s}-q33g"
            if not (OUT / "results" / f"{tag}.json").exists():
                st = json.loads((L_DIR / "q33g" / f"fresh-s{s}.state.json").read_text())
                jobs.append((tag, ("net", st["net"]), s, probes))
    return jobs


def step_run(workers, cps):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from multiprocessing import get_context
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(cps)
    flat = [(tag, (p, spec, s)) for tag, spec, s, todo in jobs for p in todo]
    print(f"probe run: {len(jobs)} checkpoints x lineages, {len(flat)} searches", flush=True)
    res = {tag: [] for tag, _, _, _ in jobs}
    need = {tag: len(todo) for tag, _, _, todo in jobs}
    t0, waited = time.time(), 0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        pend, nxt = {}, 0
        while nxt < len(flat) or pend:
            while nxt < len(flat) and len(pend) < workers:
                if os.cpu_count() - os.getloadavg()[0] < SPARE:
                    break
                tag, a = flat[nxt]
                pend[ex.submit(search_job, a)] = tag
                nxt += 1
            done, _ = wait(list(pend), timeout=60, return_when=FIRST_COMPLETED)
            if not done and not pend:
                time.sleep(60)
                waited += 60
            for f in done:
                tag = pend.pop(f)
                res[tag].append(f.result())
                if len(res[tag]) == need[tag]:
                    p = OUT / "results" / f"{tag}.json"
                    tmp = p.with_name(p.name + ".tmp")
                    tmp.write_text(json.dumps(res[tag]))
                    os.replace(tmp, p)
                    print(f"   {tag}: {sum(x['solved'] for x in res[tag])} of {need[tag]} probes solved ({time.time() - t0:.0f}s)", flush=True)
    print(f"probe run done in {time.time() - t0:.0f}s ({waited}s waiting for spare cores)")


def step_report():
    import numpy as np
    probes = json.loads((OUT / "probes.json").read_text())
    order = {cp: k for k, cp in enumerate(CHECKPOINTS + ["q41", "q43"])}
    cps = [cp for cp in CHECKPOINTS + ["q41", "q43"] if any((OUT / "results" / f"{a}-s{s}-{cp}.json").exists() for a in ("blend", "net") for s in LD.SEEDS)]
    lines = ["== Probe set: per learner lineage and end-of-rung checkpoint, probes solved (of those from rungs already reached); "
             "median tries per probe equation, on the probe equations solved at both this and the previous checkpoint (previous -> now); "
             "probes lost (solved at the previous checkpoint, not now)"]
    for arm in ("blend", "net"):
        for s in LD.SEEDS:
            prev, row = None, []
            for cp in cps:
                p = OUT / "results" / f"{arm}-s{s}-{cp}.json"
                if not p.exists():
                    row.append(f"{cp} -")
                    continue
                R = {x["id"]: x for x in json.loads(p.read_text())}
                txt = f"{cp} {sum(x['solved'] for x in R.values())}/{len(R)}"
                if prev:
                    pairs = [(a_["first_exact"], b_["first_exact"]) for pid in R if pid in prev
                             for a_, b_ in zip(prev[pid]["equations"], R[pid]["equations"])
                             if a_["first_exact"] is not None and b_["first_exact"] is not None]
                    lost = sum(prev[pid]["solved"] and not R[pid]["solved"] for pid in R if pid in prev)
                    if pairs:
                        a_, b_ = np.array(pairs, float).T
                        txt += f" (tries {np.median(a_):.0f} -> {np.median(b_):.0f} on {len(pairs)} equations"
                    else:
                        txt += " ("
                    txt += f"; lost {lost})"
                row.append(txt)
                prev = R
            lines.append(f"   {LD.ARM_NAMES[arm]} s{s}: " + "; ".join(row))
    # accumulation in one number per lineage: each probe equation's tries at its own rung's checkpoint vs at the latest
    lines.append("== Per lineage: probe equations' tries at the checkpoint where their rung was learned -> at the latest checkpoint "
                 "(median; share needing fewer tries); probes solved then -> now")
    latest = cps[-1]
    for arm in ("blend", "net"):
        for s in LD.SEEDS:
            pl = OUT / "results" / f"{arm}-s{s}-{latest}.json"
            if not pl.exists():
                continue
            L_ = {x["id"]: x for x in json.loads(pl.read_text())}
            a_, b_, then, now = [], [], 0, 0
            for pr in probes:
                p0 = OUT / "results" / f"{arm}-s{s}-{SOURCE_AT[pr['source']]}.json"
                if not p0.exists() or pr["id"] not in L_:
                    continue
                R0 = {x["id"]: x for x in json.loads(p0.read_text())}
                if pr["id"] not in R0:
                    continue
                then += R0[pr["id"]]["solved"]
                now += L_[pr["id"]]["solved"]
                for e0, e1 in zip(R0[pr["id"]]["equations"], L_[pr["id"]]["equations"]):
                    if e0["first_exact"] is not None and e1["first_exact"] is not None:
                        a_.append(e0["first_exact"])
                        b_.append(e1["first_exact"])
            if a_:
                a_, b_ = np.array(a_, float), np.array(b_, float)
                lines.append(f"   {LD.ARM_NAMES[arm]} s{s}: {np.median(a_):.0f} -> {np.median(b_):.0f} tries on {len(a_)} equations "
                             f"({np.mean(b_ < a_):.0%} fewer, {np.mean(b_ > a_):.0%} more); probes solved {then} -> {now}")
    for s in LD.SEEDS:
        p = OUT / "results" / f"fresh-s{s}-q33g.json"
        if p.exists():
            R = json.loads(p.read_text())
            fe = [e["first_exact"] for x in R for e in x["equations"] if e["first_exact"] is not None]
            lines.append(f"   reference, q33g's fresh network s{s}: {sum(x['solved'] for x in R)} of {len(R)} probes solved; "
                         f"median tries {np.median(fe) if fe else float('nan'):.0f}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    workers, step = int(sys.argv[1]), sys.argv[2]
    {"select": step_select, "run": lambda: step_run(workers, sys.argv[3:] or CHECKPOINTS), "report": step_report}[step]()


if __name__ == "__main__":
    main()
