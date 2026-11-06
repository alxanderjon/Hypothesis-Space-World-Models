"""Stage 1 of the general learner (owner's addition 2): classify each test equation of circuits and ecology as shared-form
(its canonical term set, constants ignored and the two variables possibly swapped, equals the true term set of an
equation of some training task in an earlier domain) or novel-form; a test system is shared-form when all its equations
are. Uses the known structures only; writes the classification and prints counts only. Usage: g1_shared.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/accumulating_learner/g1/systems"
ORDER = ("mechanics", "circuits", "ecology")

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import domains_g1 as DG
    seen, out = set(), {}
    for n, dom in enumerate(ORDER):
        if n:
            tests = json.loads((OUT / f"test_{dom}.json").read_text())
            eqs_shared, sys_shared, total_eq, total_sys = 0, 0, 0, 0
            for r in tests:
                if r["status"] != "accepted":
                    continue
                flags = []
                for eq in r["structure"]:
                    ks = frozenset(eq["keys"])
                    sw = [DG.swapped(k) for k in eq["keys"]]
                    ksw = frozenset(sw) if all(s is not None for s in sw) else None
                    flags.append(bool(ks in seen or (ksw is not None and ksw in seen)))
                out[r["id"]] = flags
                eqs_shared += sum(flags); total_eq += len(flags)
                sys_shared += all(flags); total_sys += 1
            print(f"{dom}: shared-form equations {eqs_shared} of {total_eq}; shared-form systems {sys_shared} of {total_sys}")
        for r in json.loads((OUT / f"train_{dom}.json").read_text()):
            if r["status"] == "accepted":
                for eq in r["structure"]:
                    seen.add(frozenset(eq["keys"]))
    (OUT / "shared_forms.json").write_text(json.dumps(out))
