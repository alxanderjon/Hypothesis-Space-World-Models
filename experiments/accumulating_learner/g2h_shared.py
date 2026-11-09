"""Stage 2a'' of the general learner: classify each harder-domain equation as shared-form or new-form (Stage 2a's rule,
for up to four variables): an equation is shared-form if its canonical term set, under some relabelling of the
variables (a two-variable form may match inside a four-variable system), equals the true term set of an equation of a
training task of an earlier domain. For the development systems (the saturation check) the earlier domains are
mechanics, circuits and ecology ("mce_dev_<domain>"); for the test systems, each order and position ("order<o>_<domain>";
order 1: mechanics, circuits, ecology, reactions, flows; order 2: ecology, circuits, mechanics, flows, reactions).
Uses the known structures only; prints counts only. Usage: g2h_shared.py <level> <set ...> (dev, test)"""
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"

if __name__ == "__main__":
    level, sets = int(sys.argv[1]), sys.argv[2:]
    sys.path.insert(0, str(HERE))
    import domains_g1 as DG
    R, F = f"reactions_L{level}", f"flows_L{level}"
    ORDERS = {1: ("mechanics", "circuits", "ecology", R, F), 2: ("ecology", "circuits", "mechanics", F, R)}
    perms = list(itertools.permutations(range(4)))
    cache = {}

    def rep(keys):
        """A representative of a term set under every relabelling of up to four variables."""
        ks = tuple(sorted(keys))
        if ks not in cache:
            forms = []
            for p in perms:
                r = [DG.relabelled(k, p) for k in ks]
                if all(x is not None for x in r):
                    forms.append(tuple(sorted(r)))
            cache[ks] = min(forms) if forms else ks
        return cache[ks]

    extra = [w for w in sets if w not in ("dev", "test")]
    path = OUT / (f"shared_forms_g2h_L{level}.json" if not extra else f"shared_forms_g2h_L{level}_{extra[0]}.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    train = {d: [eq["keys"] for r in json.loads((OUT / f"train_{d}.json").read_text()) if r["status"] == "accepted"
                 for eq in r["structure"]] for d in ORDERS[1]}
    reps = {d: {rep(k) for k in train[d]} for d in train}
    for which in sets:
        jobs = [("mce", None, d, ORDERS[1][:3]) for d in (R, F)] if which == "dev" else \
               [(f"order{o}", o, dom, seq[:pos]) for o, seq in ORDERS.items() for pos, dom in enumerate(seq) if pos >= 3]
        for name, o, dom, earlier in jobs:
            seen = set().union(*[reps[e] for e in earlier])
            recs = [r for r in json.loads((OUT / f"{which}_{dom}.json").read_text()) if r["status"] == "accepted"]
            flags = {r["id"]: [rep(eq["keys"]) in seen for eq in r["structure"]] for r in recs}
            key = f"mce_dev_{dom}" if which == "dev" else f"order{o}_{dom}"
            out[key] = flags
            n_eq = sum(len(v) for v in flags.values())
            n_sh = sum(sum(v) for v in flags.values())
            print(f"{which} {dom} ({'after ' + ', '.join(earlier)}): shared-form equations {n_sh} of {n_eq}; new-form "
                  f"{n_eq - n_sh}; shared-form systems {sum(all(v) for v in flags.values())} of {len(flags)}")
    path.write_text(json.dumps(out))
