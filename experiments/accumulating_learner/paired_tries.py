"""Paired tries (owner, 21 November): search tries per solution compared only on held-out equations solved in both the
first and the last round, per lineage, so harder equations solved later do not raise the median. Appended to a rung's
report without changing its pass rule (q33g); from the next rung on, ladder.step_report uses it for pass rule (b).
Usage: paired_tries.py <rung>"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("G2H_RUN", "ladder")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder as LD                                              # noqa: E402


def paired(r):
    """(n, median first-round tries, median last-round tries, share of pairs with fewer tries) over held-out equations solved
    in both the first and the last round; None if the rung took one round."""
    import numpy as np
    if len(r["rounds"]) < 2:
        return None
    first, last = r["rounds"][0]["test"], r["rounds"][-1]["test"]
    f = {(y["id"], e["eq"]): e["first_exact"] for y in first if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None}
    l_ = {(y["id"], e["eq"]): e["first_exact"] for y in last if y["status"] == "done" for e in y["equations"] if e["first_exact"] is not None}
    both = sorted(set(f) & set(l_))
    if not both:
        return (0, float("nan"), float("nan"), float("nan"))
    a, b = np.array([f[k] for k in both], float), np.array([l_[k] for k in both], float)
    return (len(both), float(np.median(a)), float(np.median(b)), float(np.mean(b < a)))


def main():
    rung = sys.argv[1]
    lines = [f"== Paired tries ({rung}): held-out equations solved in both the first and last rounds; median tries first -> last; "
             f"share of those equations needing fewer tries"]
    for arm in LD.ARMS:
        parts, ok = [], 0
        for s in LD.SEEDS:
            p = LD.OUT / rung / f"{arm}-s{s}.json"
            if not p.exists():
                continue
            x = paired(json.loads(p.read_text()))
            if x is None:
                parts.append(f"s{s}: one round")
                ok += 1
                continue
            n, a, b, sh = x
            ok += b <= a
            parts.append(f"s{s}: {n} equations, {a:.0f} -> {b:.0f} ({sh:.0%} fewer)")
        lines.append(f"   {LD.ARM_NAMES[arm]:>16}: " + "; ".join(parts) + f" -> paired tries do not rise in {ok} of 3")
    text = "\n".join(lines)
    with open(LD.OUT / rung / "report.txt", "a") as fh:
        fh.write("\n" + text)
    print(text)


if __name__ == "__main__":
    main()
