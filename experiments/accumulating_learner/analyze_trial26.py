"""Trial 26 analysis: criteria C0, A1, A2, A3 per seed, exactly as pre-registered (docs/research-log.md)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from trial26 import G_FAMILIAR, G_NOVEL, MS, OUT

FAIL_OBS = 2 * 64 * 13          # a system never recovered counts as twice the largest observation count


def load(seed):
    recs = {}
    base = OUT / f"test_s{seed}"
    for p in base.glob("*/*.json"):
        r = json.loads(p.read_text())
        recs[(r["name"], r["arm"], r["N"], r.get("tagkey", ""), r["M"])] = r
    return recs


def exact(recs, name, arm, N, M, tag=""):
    r = recs.get((name, arm, N, tag, M))
    return None if r is None else bool(r.get("exact"))


def success_rate(recs, names, arm, N, M, tag=""):
    v = [exact(recs, n, arm, N, M, tag) for n in names]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def min_obs(recs, name, arm, N, tag="", ms=MS, fail=FAIL_OBS):
    for M in sorted(ms):
        if exact(recs, name, arm, N, M, tag):
            return 13 * M
    return fail


def median_min_obs(recs, names, arm, N, tag="", ms=MS, fail=FAIL_OBS):
    return float(np.median([min_obs(recs, n, arm, N, tag, ms, fail) for n in names]))


def gains(recs, names, a, b):
    """Gain of arm a over arm b: (success points at M = 8, log ratio of median minimal observations)."""
    s_a, _ = success_rate(recs, names, *a, 8)
    s_b, _ = success_rate(recs, names, *b, 8)
    m_a, m_b = median_min_obs(recs, names, *a), median_min_obs(recs, names, *b)
    return 100 * (s_a - s_b), float(np.log(m_b / m_a))


def criteria(seed):
    recs = load(seed)
    out = {"seed": seed}
    nov = G_NOVEL
    full, scratch, memory, scr, stats = ("full", 45), ("scratch", 0), ("memory", 45), ("scrambled", 45), ("stats", 0)
    # C0: capability prerequisite
    c0, n0 = success_rate(recs, nov, "scratch", 0, 64)
    out["C0_scratch_rate_M64"] = c0
    out["C0_pass"] = bool(c0 >= 0.25)
    # A1
    g_fs = gains(recs, nov, full, scratch)
    g_fm = gains(recs, nov, full, memory)
    g_sm = gains(recs, nov, scr, memory)
    g_ft = gains(recs, nov, full, stats)
    ok1 = [g_fs[0] >= 15, g_fs[1] >= np.log(2)]
    beyond_memory = any(ok1[k] and g_fm[k] >= 0.5 * g_fs[k] for k in range(2))
    scrambled_loses = any(ok1[k] and g_sm[k] <= 0.5 * g_fm[k] for k in range(2))
    beats_stats = g_ft[0] > 0 or g_ft[1] > 0
    out["A1"] = {"full_vs_scratch": g_fs, "full_vs_memory": g_fm, "scrambled_vs_memory": g_sm, "full_vs_stats": g_ft,
                 "gain_ok": any(ok1), "beyond_memory": beyond_memory, "scrambled_loses": scrambled_loses,
                 "beats_stats": beats_stats}
    out["A1_pass"] = bool(any(ok1) and beyond_memory and scrambled_loses and beats_stats)
    # A2: benefit at M = 8 grows with N
    s0, _ = success_rate(recs, nov, "scratch", 0, 8)
    ben = [0.0] + [100 * (success_rate(recs, nov, "full", n, 8)[0] - s0) for n in (8, 24, 45)]
    rho = spearmanr([0, 8, 24, 45], ben).correlation if len(set(ben)) > 1 else 0.0
    out["A2"] = {"benefit_points_by_N": ben, "spearman": rho}
    out["A2_pass"] = bool(rho >= 0.8 and ben[-1] > 0)
    # A3: least-similar vs random half-histories (M <= 16), over G-novel systems the full learner recovered
    plan_p = OUT / f"a3_plan_s{seed}.json"
    if plan_p.exists():
        plan = json.loads(plan_p.read_text())["least_similar"]
        names = sorted(plan)
        ms = (2, 4, 8, 16)
        cap = 2 * 16 * 13
        s0, _ = success_rate(recs, names, "scratch", 0, 8)
        m0 = median_min_obs(recs, names, "scratch", 0, ms=ms, fail=cap)
        ls_s, _ = success_rate(recs, names, "full", 22, 8, "_ls")
        r_s = np.nanmean([success_rate(recs, names, "full", 22, 8, f"_r{k}")[0] for k in range(3)])
        ls_m = median_min_obs(recs, names, "full", 22, "_ls", ms=ms, fail=cap)
        r_m = float(np.mean([median_min_obs(recs, names, "full", 22, f"_r{k}", ms=ms, fail=cap) for k in range(3)]))
        b_ls, b_r = 100 * (ls_s - s0), 100 * (r_s - s0)
        l_ls, l_r = float(np.log(m0 / ls_m)), float(np.log(m0 / r_m))
        ok = (b_r > 0 and b_ls >= 0.5 * b_r) or (l_r > 0 and l_ls >= 0.5 * l_r)
        out["A3"] = {"systems": names, "benefit_points_M8": {"least_similar": b_ls, "random": b_r},
                     "log_obs_ratio": {"least_similar": l_ls, "random": l_r}, "pass": bool(ok)}
        out["A3_pass"] = bool(ok)
    # rates for the report
    rates = {}
    for arm, N in [("scratch", 0), ("stats", 0), ("memory", 8), ("memory", 24), ("memory", 45), ("full", 8), ("full", 24),
                   ("full", 45), ("scrambled", 45), ("guide_only", 45), ("copy_closest", 45)]:
        rates[f"{arm}{N}"] = {"novel": [success_rate(recs, nov, arm, N, M)[0] for M in MS],
                              "familiar": [success_rate(recs, G_FAMILIAR, arm, N, M)[0] for M in MS],
                              "novel_median_min_obs": median_min_obs(recs, nov, arm, N)}
    out["rates_by_M"] = {"M": list(MS), **rates}
    out["pass"] = bool(out["C0_pass"] and out["A1_pass"] and out["A2_pass"] and out.get("A3_pass", False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    res = [criteria(int(s)) for s in a.seeds.split(",")]
    print(json.dumps(res, indent=1, default=float))
    (OUT / "criteria.json").write_text(json.dumps(res, indent=1, default=float))
