"""Stage 2a'' of the general learner: the direct shortlist check (owner, 8 November). The 10-task tests of the continued
network, the rehearsal-matched control and (as context) the fresh network, rerun with every term of the search
vocabulary attended (no shortlist), on the same networks, test systems and search seeds; then bar (B) and the
continued-vs-fresh comparison recomputed next to the run's. Counts only. Usage: g2h_fullcheck.py <workers> <level>"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    workers, level = int(sys.argv[1]), int(sys.argv[2])
    os.environ["G2H_SHORTLIST"] = "1000000"                     # every term attended (inherited by the workers)
    sys.path.insert(0, str(HERE))
    import cycles_g2h as C
    import g2h_report as R
    D = C.Driver(workers, level)
    O = C.orders(level)
    C.log("shortlist check: the 10-task tests rerun with every term attended")
    nets = {}
    for o in (1, 2):
        for dom in O[o][3:]:
            nets[(o, dom, "continued")] = (str(C.RUN / "nets" / f"o{o}_{dom}_d10"), f"o{o}_{dom}_t10full")
            nets[(o, dom, "control")] = (str(C.RUN / "nets" / f"c{o}_{dom}_{dom}_d10"), f"c{o}_{dom}_t10full")
    if os.environ.get("G2H_CHECK_FRESH", "1") == "1":            # context only; skipped when the check runs long
        for dom in C.domains(level):
            nets[(0, dom, "fresh")] = (str(C.RUN / "nets" / f"fresh_{dom}_{dom}_d10"), f"fresh_{dom}_t10full")
    for (o, dom, arm), (net, label) in nets.items():
        D.test(net, dom, label, *C.NEWB)
    shared = json.loads((C.SYS / f"shared_forms_g2h_L{level}.json").read_text())
    for o in (1, 2):
        for dom in O[o][3:]:
            tests = [r for r in json.loads((C.SYS / f"test_{dom}.json").read_text()) if r["status"] == "accepted"]
            flags = shared[f"order{o}_{dom}"]
            new = [(r["id"], i) for r in tests for i in range(len(r["structure"])) if not flags[r["id"]][i]]
            lines = []
            for tag, cl, kl, fl in (("shortlist 768 (the run)", f"o{o}_{dom}_t10", f"c{o}_{dom}_t10", f"fresh_{dom}_t10"),
                                    ("every term attended", f"o{o}_{dom}_t10full", f"c{o}_{dom}_t10full", f"fresh_{dom}_t10full")):
                c, k, f = R.load(cl, dom), R.load(kl, dom), R.load(fl, dom)
                met, pts, a_, b_ = R.bar_b(c, k, new)
                fa, fb = R.paired(c, f, new) if f is not None else (0, 0)
                f = f if f is not None else []
                lines.append(f"{tag}: continued {R.counts(c, new)}, control {R.counts(k, new)}, fresh {R.counts(f, new)} of "
                             f"{len(new)} new-form; (B) {pts:+.1f} points, only continued {a_}, only control {b_}, p = "
                             f"{R.sign_p(a_, b_):.2g} -> {'MET' if met else 'NOT MET'}; vs fresh only continued {fa}, only fresh {fb}, "
                             f"p = {R.sign_p(fa, fb):.2g}")
            for arm, a, b in (("continued", f"o{o}_{dom}_t10", f"o{o}_{dom}_t10full"),
                              ("control", f"c{o}_{dom}_t10", f"c{o}_{dom}_t10full")):
                x, y = R.paired(R.load(b, dom), R.load(a, dom), new)
                lines.append(f"{arm}, every term attended vs shortlist: solved only with every term {x}, only with the "
                             f"shortlist {y}, p = {R.sign_p(x, y):.2g}")
            C.log(f"shortlist check, order {o}, {R.name(dom)}: " + " | ".join(lines))
    C.log("shortlist check done")
