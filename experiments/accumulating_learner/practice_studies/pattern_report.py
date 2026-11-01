# Report for probe_pattern.py (practice half): success by arm and budget, paired comparisons, removal check, and the
# provenance of every pattern-based success (source system; how different the two systems are beyond the pattern).
import glob, itertools, json, os, sys
import numpy as np

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
OUT = f"{SP}/pattern"
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")


def canon(p):
    """Structural signature of a pattern, invariant to how its 'other' roles are numbered."""
    k = p["k"]
    best = None
    for perm in itertools.permutations(range(1, k)):
        rm = {0: 0, **{r: perm[r - 1] for r in range(1, k)}}
        terms = []
        for t in p["terms"]:
            mono = tuple(sorted((rm[r], e) for r, e in t["mono"]))
            facs = tuple(sorted((f["family"], str([rm[v] for v in f["var"]] if isinstance(f["var"], list) else rm[f["var"]]),
                                 None if f["n"] is None else round(f["n"] * 2) / 2) for f in t["factors"]))
            terms.append((mono, facs))
        sig = tuple(sorted(terms))
        best = sig if best is None or sig < best else best
    return best


FEAT = {}


def features(name):
    """Mean burst features of a system's M = 64 recordings (the copy-the-closest similarity features)."""
    if name not in FEAT:
        import harness27 as H27, guide as GD
        ref = H27.Reference27(name, seed=1)
        FEAT[name] = GD.burst_features(np.asarray(H27.bursts_from(ref, 1, 64))).mean(0).numpy()
    return FEAT[name]


def feature_distance(a, b):
    fa, fb = features(a), features(b)
    if fa.shape != fb.shape:
        return float("nan")
    # as in copy-the-closest: the closest match over every relabelling of the variables
    return float(min(np.sqrt(np.sum((fa - fb[list(pm)]) ** 2)) for pm in itertools.permutations(range(fa.shape[0]))))


def main():
    import torch
    import discovery_learner as DL
    plan = json.load(open(f"{OUT}/plan.json"))
    lib = {e["id"]: e for e in plan["library"]}
    recs = {}
    for p in glob.glob(f"{OUT}/*_M*_*.json"):
        if p.endswith("plan.json"):
            continue
        r = json.load(open(p))
        recs[(r["name"], r["M"], r["arm"])] = r
    arms = ("scratch", "pieces", "pattern", "pattern_decoy", "copy")
    print("exact recoveries, second half of the order (practice systems):")
    for M in (8, 16, 32):
        row = []
        for a in arms:
            rs = [r for (n, m, aa), r in recs.items() if m == M and aa == a]
            row.append(f"{a} {sum(bool(r.get('exact')) for r in rs)}/{len(rs)}")
        print(f"  M={M:2d}: " + "  ".join(row))
    for a in ("pieces", "pattern", "pattern_decoy", "copy"):
        w = l = 0
        for (n, M, aa), r in recs.items():
            if aa != a:
                continue
            s = recs.get((n, M, "scratch"))
            if s is None:
                continue
            w += bool(r.get("exact")) and not s.get("exact")
            l += bool(s.get("exact")) and not r.get("exact")
        print(f"  {a:13s} vs scratch (paired): {a} exact & scratch not = {w}; scratch exact & {a} not = {l}")
    print("removal check (pattern wins rerun without the used patterns):")
    for (n, M, a), r in sorted(recs.items()):
        if a == "removal":
            print(f"  {n} M={M} removed {r.get('removed')}: exact {r.get('exact')} (advantage {'lost' if not r.get('exact') else 'kept'})")
    print("provenance of every pattern-based success (pattern arm, exact, final model from the pattern stage):")
    mat = plan["material"]
    for (n, M, a), r in sorted(recs.items()):
        if a != "pattern" or not r.get("exact") or not r.get("patterns_used"):
            continue
        tgt_model = DL.model_from_description(r["model"], torch.tensor(r["mu"]), torch.tensor(r["sig"]))
        tgt_pats = DL.motifs_from_model(tgt_model, torch.zeros(4, len(r["mu"])) + 1.0)
        tgt_sigs = [canon(p) for p in tgt_pats]
        for comp, eid in r["patterns_used"]:
            src = lib[eid]["source"]
            sm = mat.get(src, {})
            src_sigs = {canon(p) for p in sm.get("patterns", [])}
            shared = sum(1 for sg in tgt_sigs if sg in src_sigs)
            others = [k for k in plan["order"] if k in mat and k != n and len(mat[k].get("mu", [])) == len(r["mu"])]
            dists = sorted((feature_distance(n, k), k) for k in others) if others else []
            rank = [k for _, k in dists].index(src) + 1 if src in [k for _, k in dists] else None
            print(f"  {n} (d={len(r['mu'])}) M={M}: component {comp} used a pattern from {src} (d={len(sm.get('mu', []))}); "
                  f"target equations sharing a pattern with {src}: {shared}/{len(tgt_sigs)} "
                  f"(beyond the reused one: {max(shared - 1, 0)}/{max(len(tgt_sigs) - 1, 0)}); data-feature distance "
                  f"{feature_distance(n, src) if len(sm.get('mu', [])) == len(r['mu']) else float('nan'):.3g} "
                  f"(rank {rank} of {len(dists)} same-dimension sources by closeness)")


if __name__ == "__main__":
    main()


def criteria_like():
    """Practice-half analogue of the pre-registered accumulation criteria (A1 and A3 parts; A4 from the removal check)."""
    recs = {}
    for p in glob.glob(f"{OUT}/*_M*_*.json"):
        if p.endswith("plan.json"):
            continue
        r = json.load(open(p))
        recs[(r["name"], r["M"], r["arm"])] = r
    names = sorted({k[0] for k in recs})
    FAIL = 2 * 64 * 13

    def min_obs(n, arm):
        for M in (8, 16, 32):
            if recs.get((n, M, arm), {}).get("exact"):
                return 13 * M
        s64 = f"{SP}/scratch3/{n}_M64_scratch.json"
        return 13 * 64 if os.path.exists(s64) and json.load(open(s64)).get("exact") else FAIL

    base = None
    for arm in ("scratch", "pieces", "pattern", "pattern_decoy", "copy"):
        e8 = [bool(recs.get((n, 8, arm), {}).get("exact")) for n in names if (n, 8, arm) in recs]
        mo = [min_obs(n, arm) for n in names if (n, 8, arm) in recs]
        rate = 100 * np.mean(e8) if e8 else float("nan")
        med = float(np.median(mo)) if mo else float("nan")
        if arm == "scratch":
            base = (rate, med)
        print(f"  {arm:13s}: exact at M=8 {rate:5.1f}%  median minimal observations {med:.0f}"
              + ("" if arm == "scratch" else f"  (advantage {rate - base[0]:+.1f} points; ratio {med / base[1]:.2f})"))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "criteria":
    criteria_like()
