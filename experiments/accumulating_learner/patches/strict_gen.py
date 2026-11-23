"""Apply after q43, after patches/solved_def.py (owner, 23 November): training and test systems of rungs marked
"strict" come from gen_wide.build_wide (bounded from all 20 test starts over 10 time constants; recordings from the
initial state times U(0.2, 3.0) at 2-3 variables, U(0.3, 2.0) at 4; up to gen_wide.MAX_DRAWS draws), and carry their
description, test starts and horizon; dreams keep the cheaper generator (domains_g1.build). The precision fit is part
of the rollout check, which judges the held-out test. Usage: python patches/strict_gen.py"""

from pathlib import Path

p = Path(__file__).resolve().parent.parent / "ladder.py"
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''    max_k = args[7] if len(args) > 7 else None
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    reasons = []
    for r in range(200):''', '''    max_k = args[7] if len(args) > 7 else None
    strict = args[8] if len(args) > 8 else False                 # owner, 23 Nov: the strict generator for training and test
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    import domains_g1 as DG
    import dream_g2 as DR
    import gen_wide as GW
    reasons = []
    for r in range(GW.MAX_DRAWS if strict else 200):''')
rep('''        try:
            syn = DG.build(desc)
        except Exception as ex:
            reasons.append(type(ex).__name__)''', '''        try:
            syn = GW.build_wide(desc, seed_starts=stable(seed, "test starts")) if strict else DG.build(desc)
        except Exception as ex:
            reasons.append(type(ex).__name__)''')
rep('''                "bursts": b8.tolist(), "desc": desc}''', '''                "bursts": b8.tolist(), "desc": desc, "test": getattr(syn, "test", None)}''')
rep('''                                  "desc": r.get("desc")})''', '''                                  "desc": r.get("desc"), "test_starts": (r.get("test") or {}).get("starts"),
                                  "horizon": (r.get("test") or {}).get("T")})''')
rep('''    jobs = [((s, split, k), (f"ladder-{rung}-s{s}-{split}", k, allowed, R["terms"], R["d"], new, False, R.get("max_k")))''',
    '''    jobs = [((s, split, k), (f"ladder-{rung}-s{s}-{split}", k, allowed, R["terms"], R["d"], new, False, R.get("max_k"), R.get("strict", False)))''')
p.write_text(s)
print("patched ladder.py: strict generator for training and test systems of strict rungs")
