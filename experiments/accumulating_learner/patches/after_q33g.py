"""Apply after q33g's processes have exited (owner, 21 November): q41 and q43 under the generous-budget rule, q41 after
q33g; pass rule (b) in ladder.step_report on paired tries (held-out equations solved in both the first and last round).
Usage: python patches/after_q33g.py"""

from pathlib import Path

p = Path(__file__).resolve().parent.parent / "ladder.py"
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''    "q41": dict(_R, d=4, terms=(1, 2), levels=_ALL, change="all ten pieces, 4 variables, 1-2 terms (after q33)", after="q33"),
    "q43": dict(_R, d=4, terms=(3, 3), levels=_ALL, change="all ten pieces, 4 variables, 3 terms (after q41)", after="q41")})''',
    '''    "q41": dict(_R, d=4, terms=(1, 2), levels=_ALL, change="all ten pieces, 4 variables, 1-2 terms (after q33)", after="q33"),
    "q43": dict(_R, d=4, terms=(3, 3), levels=_ALL, change="all ten pieces, 4 variables, 3 terms (after q41)", after="q41")})
_GB = {"budgets": ((3000, 600), (3000, 600), (3000, 600)), "test_budget": (3000, 600), "reanalyse": (1000, 200)}
RUNGS["q41"].update(_GB, after="q33g", change="all ten pieces, 4 variables, 1-2 terms, generous budgets (after q33g)")   # owner, 21 Nov
RUNGS["q43"].update(_GB, change="all ten pieces, 4 variables, 3 terms, generous budgets (after q41)")''')
rep('''                b = len(med) == 1 or med[-1] <= med[0]
                bok += b''', '''                import paired_tries as PT                         # owner, 21 Nov: (b) on paired tries from the rung after q33g
                pt = PT.paired(r)
                b = pt is None or not (pt[2] > pt[1])
                bok += b
                med_txt = "one round" if pt is None else f"paired {pt[0]} equations {pt[1]:.0f} -> {pt[2]:.0f}"''')
rep('''                ext.append(f"s{s}: held-out median tries per solution {' / '.join(f'{v:.0f}' for v in med)}; held-out solved within 300 {' / '.join(map(str, w300))}; "
                           f"time-capped equations {capped}{'' if b else ' (tries rose)'}")''', '''                ext.append(f"s{s}: held-out median tries per solution {' / '.join(f'{v:.0f}' for v in med)} ({med_txt}); held-out solved within 300 "
                           f"{' / '.join(map(str, w300))}; time-capped equations {capped}{'' if b else ' (paired tries rose)'}")''')
p.write_text(s)
print("patched ladder.py")
