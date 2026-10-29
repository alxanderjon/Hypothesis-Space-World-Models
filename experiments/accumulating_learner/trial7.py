"""Trial 7 (evaluation-only): exact leave-one-out selection for the sparse head. See docs/research-log.md."""

import json

import numpy as np
import torch

import systems as S
from sparse_head import sparse_exact_loo_with_score
from trial1 import K_GRID, SEEDS, episode, load, ridge_loo_with_score
from trial4 import M, Library, krr_with_score
from trial5 import OUT as TRIAL5
from trial6 import OUT as TRIAL6

OUT = S.ROOT / "outputs/accumulating_learner/trial7"
TRIAL1 = S.ROOT / "outputs/accumulating_learner/trial1"


@torch.no_grad()
def select(lib, uc, yc, uq, yq, krr, heads):
    """Options: each basis with each head in `heads`, then kernel ridge last. Lowest exact LOO wins."""
    E, L = [], []
    for head in heads:
        fn = sparse_exact_loo_with_score if head == "sparse" else ridge_loo_with_score
        for b in lib.bases:
            e, l = fn(b(uc), yc, b(uq), yq)
            E.append(e); L.append(l)
    E, L = torch.stack(E + [krr[0]], 1), torch.stack(L + [krr[1]], 1)
    sel = L.argmin(1)
    return E.gather(1, sel.unsqueeze(1)).squeeze(1), sel


def lib_from(path, dev):
    lib = Library().to(dev)
    lib.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return lib.eval()


def main(dev="cuda:0"):
    OUT.mkdir(parents=True, exist_ok=True)
    names = S.eligible_split()["development"]
    xs, dxs = load(names, dev)
    arms = {}
    for s in SEEDS:
        l6 = lib_from(TRIAL6 / "learners" / f"seqsparse400_N51_s{s}.pt", dev)
        l5 = lib_from(TRIAL5 / "learners" / f"seqreplay400_N51_s{s}.pt", dev)
        arms[f"hybrid6_N51_s{s}"] = (l6, ("sparse", "ridge"))
        arms[f"sparse6exact_N51_s{s}"] = (l6, ("sparse",))
        arms[f"hybrid5_N51_s{s}"] = (l5, ("sparse", "ridge"))
    rows = [r for r in json.loads((TRIAL1 / "eval_development.json").read_text())["rows"] if not r["method"].startswith("learned")]
    selections = []
    for si, name in enumerate(names):
        for K in K_GRID:
            gen = torch.Generator().manual_seed(5000 + 97 * si + K)
            eps = [episode(xs[si][i:i + 1], dxs[si][i:i + 1], K, gen, dev) for i in range(xs[si].shape[0]) for _ in range(5)]
            uc, yc, uq, yq = (torch.stack(v) for v in zip(*eps))
            krr = krr_with_score(uc, yc, uq, yq)
            summary = {}
            for tag, (lib, heads) in arms.items():
                e, sel = select(lib, uc, yc, uq, yq, krr, heads)
                rows.append({"system": name, "K": K, "method": tag, "episodes": e.float().cpu().tolist()})
                selections.append({"system": name, "K": K, "method": tag, "heads": heads,
                                   "counts": torch.bincount(sel.cpu(), minlength=M * len(heads) + 1).tolist()})
                summary[tag] = round(float(np.median(e.float().cpu().numpy())), 4)
            print(name, K, summary, flush=True)
    (OUT / "eval_development.json").write_text(json.dumps({"rows": rows, "selections": selections}))
    analyze(names)


def analyze(names):
    ev = json.loads((OUT / "eval_development.json").read_text())
    score = {(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in ev["rows"]}
    t5 = json.loads((TRIAL5 / "eval_development.json").read_text())["rows"]
    score.update({(r["system"], r["K"], r["method"]): float(np.median(r["episodes"])) for r in t5
                  if r["method"].startswith("seqreplay400_N")})
    med = lambda m, K: float(np.median([score[(s, K, m)] for s in names]))
    thr = lambda m, s: next((K for K in K_GRID if score[(s, K, m)] <= 0.1), np.inf)
    out = {}
    for arm in ("hybrid6", "sparse6exact", "hybrid5"):
        for s in SEEDS:
            new, ref = f"{arm}_N51_s{s}", f"seqreplay400_N51_s{s}"
            r8, r32 = med(new, 8) / med(ref, 8), med(new, 32) / med(ref, 32)
            wins8 = sum(score[(x, 8, new)] < score[(x, 8, ref)] for x in names)
            h1 = {x: score[(x, 8, new)] / score[(x, 8, ref)] for x in ("Thomas", "WindmiReduced")}
            ctrls = ["rff", "krr", f"seqreplay400_N00_s{s}"]
            best8 = min(ctrls, key=lambda m: med(m, 8)); best32 = min(ctrls, key=lambda m: med(m, 32))
            s1 = {"wins": sum(score[(x, 8, new)] < score[(x, 8, best8)] for x in names), "ratio": med(new, 8) / med(best8, 8)}
            c3 = {"learner": float(np.median([thr(new, x) for x in names])), "control": float(np.median([thr(best32, x) for x in names]))}
            sel = np.sum([x["counts"] for x in ev["selections"] if x["method"] == new and x["K"] in (8, 32)], 0).tolist()
            crit = {"G1": bool(r8 <= 0.75 and wins8 >= 9), "G2": bool(r32 <= 1.5), "H1": all(v <= 1.2 for v in h1.values())}
            out[new] = {"K8": med(new, 8), "K16": med(new, 16), "K32": med(new, 32), "ratio8": r8, "wins8": wins8, "ratio32": r32,
                        "H1_ratios": h1, "S1": s1, "C3": c3, "selection_counts_K8_K32": sel, "criteria": crit}
            print(new, json.dumps(out[new], default=float))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
