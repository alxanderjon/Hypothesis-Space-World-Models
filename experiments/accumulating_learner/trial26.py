"""Trial 26 driver: exact recovery with knowledge from history, tested on systems with never-seen building blocks.

Stages (see docs/research-log.md, Trial 26 and its refinements):
  history  learn each of the 45 history systems from scratch at M = 64; keep the learner's own accepted solutions
  guides   train influence guides on the first N history solutions (N = 8, 24, 45), plus the scrambled-lesson guide
  test     run every arm on every test system and observation budget; judge exactness with the harness
  a3       least-similar vs random half-histories for G-novel systems the full learner recovered
Everything is simulated in memory; only small JSON records and guide weights are written.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parents[1]
OUT = ROOT / "outputs/accumulating_learner/trial26"
KINDS = json.loads((ROOT / "outputs/accumulating_learner/trial24pfn/kinds.json").read_text())
HISTORY_KINDS = {"polynomial", "trig", "piecewise", "tanh"}
HISTORY = sorted(n for n, v in KINDS.items() if v["source"] == "dysts:meta_train" and set(v["kinds"]) <= HISTORY_KINDS)
G_NOVEL = sorted([n for n, v in KINDS.items() if set(v["kinds"]) & {"rational", "exponential"}
                  and v["source"] != "biomodels3d:train-excluded"] + ["WindmiReduced"])
G_FAMILIAR = sorted(n for n, v in KINDS.items() if v["source"] == "dysts:development" and set(v["kinds"]) <= HISTORY_KINDS)
MS = (2, 4, 8, 16, 32, 64)
NS = (8, 24, 45)
L = 13


def setup_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def history_order(seed):
    return [HISTORY[k] for k in np.random.default_rng([2626, seed]).permutation(len(HISTORY))]


def bursts_from(ref, seed, M):
    """First M of a seeded order of burst starts on the 4 learning trajectories (nested across M)."""
    T = ref.learn_trajs.shape[1]
    rng = np.random.default_rng([2627, seed, sum(map(ord, ref.sys.name))])
    starts = [(int(k // (T - L)), int(k % (T - L))) for k in rng.permutation(4 * (T - L))[:64]]
    return np.stack([ref.learn_trajs[a, s:s + L] for a, s in starts[:M]])


def time_limit(M):
    return {2: 240, 4: 240, 8: 300, 16: 360, 32: 480, 64: 600}[M]


# ---------------- history ----------------

def history_job(args):
    name, seed, tl = args
    out = OUT / f"history_s{seed}" / f"{name}.json"
    if out.exists():
        return name, "cached"
    setup_worker()
    import exact_harness as H
    import exact_learner as EL
    t0 = time.time()
    rec = {"name": name, "seed": seed}
    try:
        ref = H.Reference(name, seed=seed)
        b = bursts_from(ref, seed, 64)
        res = EL.learn(b, seed=seed, time_limit=tl)
        rec.update({k: res.get(k) for k in ("status", "val_err_scaled", "train_err_scaled", "n_terms", "seconds", "polishes", "levels", "pass")})
        rec["bursts"] = b.tolist()
        if "model" in res:
            m = res["model"]
            rec.update({"model": m.describe(), "mu": m.mu.tolist(), "sig": m.sig.tolist(), "nsub": res["nsub"]})
            fh, nh = EL.artifact(res, ref.sys.dt)            # truth-based check, REPORTING ONLY (never used for learning)
            rec["exactness_report_only"] = H.evaluate_artifact(ref, fh, nh)
    except Exception as e:
        rec.update({"status": "error", "error": repr(e), "trace": traceback.format_exc()[-2000:]})
    rec["wall"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    return name, rec.get("status")


def load_history(seed):
    recs = {}
    for name in HISTORY:
        p = OUT / f"history_s{seed}" / f"{name}.json"
        if p.exists():
            recs[name] = json.loads(p.read_text())
    return recs


def solved(rec):
    return rec is not None and rec.get("status") == "accepted" and "model" in rec


# ---------------- knowledge ----------------

def memory_from(recs, names):
    import exact_learner as EL
    counts, promoted = {}, set()
    for n in names:
        r = recs.get(n)
        if not solved(r):
            continue
        m = EL.model_from_description(r["model"], r["mu"], r["sig"])
        for terms in m.supports:
            for t in terms:
                k = t.key()
                counts[k] = counts.get(k, 0) + 1
                if t.level >= 2:
                    promoted.add(k)
    return counts, tuple(sorted(promoted, key=str))


def guide_path(seed, tag):
    return OUT / f"guides_s{seed}" / f"{tag}.pt"


def train_and_save_guide(args):
    seed, tag, names, scramble = args
    path = guide_path(seed, tag)
    if path.exists():
        return tag, "cached"
    setup_worker()
    import guide as GD
    recs = load_history(seed)
    ex = [(np.asarray(recs[n]["bursts"]), recs[n]["model"]) for n in names if solved(recs.get(n))]
    if len(ex) < 2:
        return tag, "too few solved"
    net = GD.train_guide(ex, seed=seed, scramble=scramble)
    path.parent.mkdir(parents=True, exist_ok=True)
    GD.save_guide(net, path)
    (path.with_suffix(".json")).write_text(json.dumps({"names": names, "used": len(ex), "scramble": scramble}))
    return tag, f"trained on {len(ex)}"


def knowledge_for(arm, N, seed, bursts, recs, subset=None, tag=None):
    import exact_learner as EL
    import guide as GD
    if arm == "scratch":
        return EL.Knowledge(label="scratch")
    if arm == "stats":
        P = GD.statistical_influence(bursts, seed=seed)
        return EL.Knowledge(influence=P, label="stats")
    names = subset if subset is not None else history_order(seed)[:N]
    counts, promoted = memory_from(recs, names) if arm in ("memory", "full", "scrambled") else ({}, ())
    if arm == "memory":
        return EL.Knowledge(template_counts=counts, promoted=promoted, label=f"memory{N}")
    gtag = tag or {"full": f"N{N}", "guide_only": f"N{N}", "scrambled": f"N{N}_scrambled"}[arm]
    F = GD.burst_features(np.asarray(bursts)[:2]).shape[-1]
    net = GD.load_guide(guide_path(seed, gtag), F)
    P, count, level = GD.predict(net, bursts)
    return EL.Knowledge(template_counts=counts, promoted=promoted, influence=P, n_terms=count, level=level,
                        label=f"{arm}{N}")


# ---------------- test ----------------

ARMS = [("scratch", 0), ("stats", 0)] + [("memory", n) for n in NS] + [("full", n) for n in NS] + \
       [("scrambled", 45), ("guide_only", 45), ("copy_closest", 45)]


def copy_closest(bursts, recs, names, seed, tl):
    """Baseline: structure of the past system most similar in data (all variable relabelings), constants refit."""
    import itertools
    import torch
    import exact_learner as EL
    import guide as GD
    b = np.asarray(bursts)
    d = b.shape[-1]
    fb = GD.burst_features(b).mean(0).numpy()          # (d, F)
    best = None
    for n in names:
        r = recs.get(n)
        if not solved(r):
            continue
        fh = GD.burst_features(np.asarray(r["bursts"])).mean(0).numpy()
        for pm in itertools.permutations(range(d)):
            dist = float(np.sum((fb - fh[list(pm)]) ** 2))
            if best is None or dist < best[0]:
                best = (dist, n, pm)
    if best is None:
        return {"status": "no_model", "seconds": 0.0}
    _, n, pm = best
    src = recs[n]["model"]
    inv = {old: new for new, old in enumerate(pm)}       # history variable `old` plays the role of test variable `new`
    desc = [None] * d
    for new_i in range(d):
        comp = src[pm[new_i]]
        desc[new_i] = [{"mono": [t["mono"][pm[k]] for k in range(d)],
                        "factors": [[f[0], inv[f[1]], f[2]] for f in t["factors"]], "level": t["level"], "coef": 0.0}
                       for t in comp]
    t0 = time.time()
    rng = np.random.default_rng(seed)
    M = b.shape[0]
    nval = max(1, int(round(0.25 * M)))
    perm = rng.permutation(M)
    Bt = torch.as_tensor(b)
    mu, sig = Bt.reshape(-1, d).mean(0), Bt.reshape(-1, d).std(0) + 1e-300
    Ztr, Zva = (Bt[perm[nval:] if M > 1 else perm] - mu) / sig, (Bt[perm[:nval]] - mu) / sig
    G, _ = EL.deriv_noise(Ztr)
    Zc = torch.stack([Ztr[:, c] for c in range(4, L - 4)], 1).reshape(-1, d)
    model = EL.model_from_description(desc, mu, sig)
    for i in range(d):
        _, c = EL.lstsq_resid(EL.design(model.supports[i], Zc, Zc * sig + mu), G[:, i])
        model.coefs[i] = c
    z0, z1 = Ztr[:, :-1].reshape(-1, d), Ztr[:, 1:].reshape(-1, d)
    try:
        model, nsub, err, _ = EL.polish(model, z0, z1)
        with torch.no_grad():
            val = float((EL.rk8_torch(model.g, Zva[:, :-1].reshape(-1, d), nsub) - Zva[:, 1:].reshape(-1, d)).abs().max())
    except (RuntimeError, ValueError):
        return {"status": "no_model", "seconds": time.time() - t0, "copied": n}
    return {"status": "accepted" if val <= EL.VAL_TOL else "rejected", "model": model, "nsub": nsub,
            "val_err_scaled": val, "train_err_scaled": err, "seconds": time.time() - t0, "copied": n,
            "n_terms": model.n_terms()}


def test_job(args):
    name, seed, M, arms, extra = args
    tagkey = extra.get("tagkey", "")
    outdir = OUT / f"test_s{seed}" / name
    todo = [(a, n) for a, n in arms if not (outdir / f"M{M}_{a}{n}{tagkey}.json").exists()]
    if not todo:
        return name, M, "cached"
    setup_worker()
    import exact_harness as H
    import exact_learner as EL
    recs = load_history(seed)
    ref = H.Reference(name, seed=seed)
    ref.envelope()
    b = bursts_from(ref, seed, M)
    outdir.mkdir(parents=True, exist_ok=True)
    for arm, N in todo:
        t0 = time.time()
        rec = {"name": name, "seed": seed, "M": M, "arm": arm, "N": N, "tagkey": tagkey}
        try:
            if arm == "copy_closest":
                res = copy_closest(b, recs, history_order(seed)[:N], seed, time_limit(M))
            else:
                kn = knowledge_for(arm, N, seed, b, recs, subset=extra.get("subset"), tag=extra.get("tag"))
                rec["guidance"] = {"influence": None if kn.influence is None else np.asarray(kn.influence).round(3).tolist(),
                                   "n_terms": None if kn.n_terms is None else np.asarray(kn.n_terms).tolist(),
                                   "level": None if kn.level is None else np.asarray(kn.level).tolist()}
                res = EL.learn(b, knowledge=kn, seed=seed, time_limit=time_limit(M))
            rec.update({k: res.get(k) for k in ("status", "val_err_scaled", "train_err_scaled", "n_terms", "seconds",
                                                 "polishes", "levels", "pass", "copied")})
            if "model" in res:
                m = res["model"]
                rec.update({"model": m.describe(), "mu": m.mu.tolist(), "sig": m.sig.tolist(), "nsub": res["nsub"]})
                fh, nh = EL.artifact(res, ref.sys.dt)
                rec["exactness"] = H.evaluate_artifact(ref, fh, nh)
                rec["exact"] = bool(rec["exactness"]["exact"])
            else:
                rec["exact"] = False
        except Exception as e:
            rec.update({"status": "error", "exact": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]})
        rec["wall"] = time.time() - t0
        (outdir / f"M{M}_{arm}{N}{tagkey}.json").write_text(json.dumps(rec))
    return name, M, "done"


def run_pool(fn, jobs, workers):
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, jobs)):
            print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(jobs)} {r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["history", "guides", "test", "lists"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--systems", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--hist-time", type=float, default=1200)
    a = ap.parse_args()
    if a.stage == "lists":
        print("history", len(HISTORY), HISTORY)
        print("G-novel", len(G_NOVEL), G_NOVEL)
        print("G-familiar", len(G_FAMILIAR), G_FAMILIAR)
        return
    if a.stage == "history":
        names = a.systems.split(",") if a.systems else HISTORY
        run_pool(history_job, [(n, a.seed, a.hist_time) for n in names], a.workers)
    elif a.stage == "guides":
        order = history_order(a.seed)
        jobs = [(a.seed, f"N{n}", order[:n], False) for n in NS] + [(a.seed, "N45_scrambled", order, True)]
        run_pool(train_and_save_guide, jobs, a.workers)
    elif a.stage == "test":
        names = a.systems.split(",") if a.systems else G_NOVEL + G_FAMILIAR
        ms = [int(m) for m in a.ms.split(",")] if a.ms else list(MS)
        jobs = [(n, a.seed, M, ARMS, {}) for M in sorted(ms, reverse=True) for n in names]
        run_pool(test_job, jobs, a.workers)


if __name__ == "__main__":
    main()
