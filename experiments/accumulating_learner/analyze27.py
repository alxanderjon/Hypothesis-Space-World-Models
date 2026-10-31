"""Trial 27 analysis: criteria D, A1-A4 (both data draws, averaged over the 3 orders), piece classification and
out-of-range exactness (owner additions). See docs/research-log.md (Trial 27)."""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/accumulating_learner/trial27"
MS = (2, 4, 8, 16, 32, 64)
FAIL = 2 * 64 * 13


def load(tag, seed):
    base = OUT / tag / f"seed{seed}"
    scratch = {}
    for p in glob.glob(str(base / "scratch" / "*.json")):
        r = json.load(open(p))
        scratch[(r["name"], r["M"])] = r
    streams = defaultdict(dict)                 # order -> (pos, name, M, mode) -> record
    for p in glob.glob(str(base / "order*" / "pos*.json")):
        r = json.load(open(p))
        streams[r["order"]][(r["pos"], r["name"], r["M"], r["mode"])] = r
    removal = {}
    for p in glob.glob(str(base / "removal" / "*.json")):
        r = json.load(open(p))
        removal[(r["order"], r["pos"], r["M"])] = r
    return scratch, streams, removal


def exact(r):
    return bool(r and r.get("exact"))


def min_obs(get, name, ms=MS):
    for M in sorted(ms):
        if exact(get(name, M)):
            return 13 * M
    return FAIL


def reference_failed(r):
    """Fixed before the evaluation: a system whose reference cannot be built (the true system diverges from the protocol's
    starting states even with the interval refined) is excluded from every criterion, in every arm, and counted."""
    return bool(r) and r.get("status") == "error" and "diverged" in (r.get("error") or "")


def criteria(tag, seed):
    scratch, streams, removal = load(tag, seed)
    excluded = sorted({k[0] for k, r in scratch.items() if reference_failed(r)})
    names = sorted({k[0] for k in scratch} - set(excluded))
    out = {"seed": seed, "n_systems": len(names), "excluded_reference_failed": excluded}
    # ---- D: discovery capability ----
    toolkit_ok, unlearnable, disc_ok = [], [], []
    for n in names:
        r = scratch.get((n, 64))
        new = [p for p in (r or {}).get("pieces") or [] if p.get("origin") == "new"]
        if exact(r) and not new:
            toolkit_ok.append(n)
        else:
            unlearnable.append(n)
            if exact(r):
                disc_ok.append(n)
    out["D"] = {"toolkit_learnable": len(toolkit_ok), "toolkit_unlearnable": len(unlearnable), "discovery_exact": len(disc_ok),
                "rate": len(disc_ok) / max(len(unlearnable), 1)}
    out["D_pass"] = bool(out["D"]["rate"] >= 0.30)
    # ---- A1-A3 over the streams ----
    per_order = []
    for o, recs in sorted(streams.items()):
        order = sorted({(k[0], k[1]) for k in recs if k[1] not in excluded})
        half = len(order) // 2
        rows = []
        for rank, (pos, name) in enumerate(order):
            g = lambda mode: (lambda nm, M: recs.get((pos, nm, M, mode)))
            lib_get, dec_get, cop_get = g("library"), g("decoy"), g("copy_closest")
            scr_get = lambda nm, M: scratch.get((nm, M))
            rows.append({"pos": pos, "name": name, "second": rank >= half,
                         "lib8": exact(lib_get(name, 8)), "scr8": exact(scr_get(name, 8)),
                         "dec8": exact(dec_get(name, 8)) if dec_get(name, 8) else None,
                         "cop8": exact(cop_get(name, 8)) if cop_get(name, 8) else None,
                         "lib_obs": min_obs(lib_get, name), "scr_obs": min_obs(scr_get, name),
                         "dec_obs": min_obs(dec_get, name) if dec_get(name, 64) else None,
                         "cop_obs": min_obs(cop_get, name) if cop_get(name, 64) else None})
        sec = [r for r in rows if r["second"]]
        adv8 = 100 * (np.mean([r["lib8"] for r in sec]) - np.mean([r["scr8"] for r in sec])) if sec else float("nan")
        ratio = (np.median([r["lib_obs"] for r in sec]) / np.median([r["scr_obs"] for r in sec])) if sec else float("nan")
        dec = [r for r in sec if r["dec8"] is not None]
        cop = [r for r in sec if r["cop8"] is not None]
        dec_adv8 = 100 * (np.mean([r["dec8"] for r in dec]) - np.mean([r["scr8"] for r in dec])) if dec else float("nan")
        cop_adv8 = 100 * (np.mean([r["cop8"] for r in cop]) - np.mean([r["scr8"] for r in cop])) if cop else float("nan")
        # A2: advantage by position bins (success averaged over all M, library minus scratch)
        bins = np.array_split(np.arange(len(rows)), 5)
        adv_bins = []
        for b in bins:
            if len(b) == 0:
                continue
            lib = np.mean([[exact(recs.get((rows[i]["pos"], rows[i]["name"], M, "library"))) for M in MS] for i in b])
            scr = np.mean([[exact(scratch.get((rows[i]["name"], M))) for M in MS] for i in b])
            adv_bins.append(100 * (lib - scr))
        rho = spearmanr(range(len(adv_bins)), adv_bins).correlation if len(set(adv_bins)) > 1 else 0.0
        first = [r for r in rows if not r["second"]]
        adv_first = 100 * (np.mean([r["lib8"] for r in first]) - np.mean([r["scr8"] for r in first])) if first else float("nan")
        per_order.append({"order": o, "adv8_second_half_points": adv8, "median_obs_ratio_second_half": ratio,
                          "adv8_first_half_points": adv_first, "decoy_adv8": dec_adv8, "copy_adv8": cop_adv8,
                          "advantage_by_bin": adv_bins, "spearman": rho})
    if per_order:
        mean = lambda k: float(np.nanmean([p[k] for p in per_order]))
        a1 = {k: mean(k) for k in ("adv8_second_half_points", "median_obs_ratio_second_half", "adv8_first_half_points",
                                    "decoy_adv8", "copy_adv8", "spearman")}
        out["orders"] = per_order
        out["A1"] = a1
        out["A1_pass"] = bool(a1["adv8_second_half_points"] >= 15 or a1["median_obs_ratio_second_half"] <= 0.5)
        out["A2_pass"] = bool(a1["adv8_second_half_points"] > a1["adv8_first_half_points"] and a1["spearman"] >= 0.5)
        base = a1["adv8_second_half_points"]
        out["A3_pass"] = bool(base > 0 and a1["decoy_adv8"] <= 0.5 * base and a1["copy_adv8"] <= 0.5 * base)
    # ---- A4: removal check ----
    wins = [r for r in removal.values()]
    lost = [r for r in wins if not exact(r)]
    out["A4"] = {"cases": len(wins), "advantage_lost_without_pieces": len(lost),
                 "fraction": len(lost) / len(wins) if wins else float("nan")}
    out["A4_pass"] = bool(wins and len(lost) / len(wins) >= 0.8)
    # ---- owner additions: classification and out-of-range ----
    cls, oor = Counter(), []
    every = list(scratch.values()) + [r for recs in streams.values() for r in recs.values()]
    for r in every:
        for p in r.get("pieces") or []:
            if p.get("origin") == "new":
                cls[p.get("classification")] += 1
        if exact(r) and r.get("out_of_range") and r["out_of_range"].get("n_states"):
            o = r["out_of_range"]
            kinds = {p.get("classification") or "reused" for p in r.get("pieces") or []}
            oor.append({"name": r["name"], "M": r["M"], "mode": r.get("mode"), "kinds": sorted(kinds), "exact": o.get("exact"),
                        "fraction_states_exact": o.get("fraction_states_one_step_exact")})
    out["piece_classification"] = dict(cls)
    out["out_of_range"] = {"systems_with_pieces_exact_in_range": len(oor),
                           "exact_out_of_range": sum(1 for x in oor if x["exact"]),
                           "mean_fraction_of_states_exact": float(np.mean([x["fraction_states_exact"] for x in oor])) if oor else float("nan"),
                           "rows": oor[:60]}
    out["pass"] = bool(out["D_pass"] and out.get("A1_pass") and out.get("A2_pass") and out.get("A3_pass") and out["A4_pass"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    res = [criteria(a.tag, int(s)) for s in a.seeds.split(",")]
    print(json.dumps(res, indent=1, default=float))
    (OUT / f"criteria_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
