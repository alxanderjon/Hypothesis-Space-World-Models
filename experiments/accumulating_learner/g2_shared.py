"""Stage 2a of the general learner: classify each new-domain test equation as shared-form or new-form for each position
and order (research log, 7 November): an equation is shared-form if its canonical term set, under some relabelling of
the variables (a two-variable form may match inside a three-variable system), equals the true term set of an equation of
a training task of an earlier domain in that order. Orders: 1: mechanics, circuits, ecology, chemistry, chaotic flows;
2: ecology, circuits, mechanics, chaotic flows, chemistry. Uses the known structures only; prints counts only.
Usage: g2_shared.py"""
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"
ORDERS = {1: ("mechanics", "circuits", "ecology", "chemistry", "chaotic"), 2: ("ecology", "circuits", "mechanics", "chaotic", "chemistry")}

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import domains_g1 as DG
    perms = list(itertools.permutations(range(3)))
    cache = {}

    def rep(keys):
        """A representative of a term set under every relabelling of up to three variables."""
        ks = tuple(sorted(keys))
        if ks not in cache:
            forms = []
            for p in perms:
                r = [DG.relabelled(k, p) for k in ks]
                if all(x is not None for x in r):
                    forms.append(tuple(sorted(r)))
            cache[ks] = min(forms) if forms else ks
        return cache[ks]

    train = {d: [eq["keys"] for r in json.loads((OUT / f"train_{d}.json").read_text()) if r["status"] == "accepted"
                 for eq in r["structure"]] for d in ORDERS[1]}
    reps = {d: {rep(k) for k in train[d]} for d in train}
    out = {}
    for o, seq in ORDERS.items():
        for pos, dom in enumerate(seq):
            if dom not in ("chemistry", "chaotic"):
                continue
            seen = set().union(*[reps[e] for e in seq[:pos]])
            tests = [r for r in json.loads((OUT / f"test_{dom}.json").read_text()) if r["status"] == "accepted"]
            flags = {r["id"]: [rep(eq["keys"]) in seen for eq in r["structure"]] for r in tests}
            out[f"order{o}_{dom}"] = flags
            n_eq = sum(len(v) for v in flags.values())
            n_sh = sum(sum(v) for v in flags.values())
            print(f"order {o}, {dom} (position {pos + 1}): shared-form equations {n_sh} of {n_eq}; new-form {n_eq - n_sh}; "
                  f"shared-form systems {sum(all(v) for v in flags.values())} of {len(flags)}")
    (OUT / "shared_forms_g2.json").write_text(json.dumps(out))
