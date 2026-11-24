"""Fitter fix 2 (owner, 24 Nov: fix the fitter so true equations pass the real definition of solved). The constant of a
shift inside a denominator, log or fractional power (K in 1/(K + x^2)) was only ever tried with the pole outside the
range of the data. With wide recordings, recordings can lie on both sides of a true pole, each staying on its side; that
pole sits in a gap between the recordings' ranges and was never tried. With the recordings known (run: rows per
recording), the gaps between the recordings' ranges of the subexpression are found; then half of the unit interval
keeps the old outside placement and half places the pole inside a gap (away from both edges). No gaps: as before.
Usage: python patches/pole_gaps.py"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
p = HERE / "fitter_g1.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''def natural_thetas(inf, U, W):
    """Constant values (B, k) for unit points U (B, k) by each constant's role (module docstring), constants whose
    subexpression involves no other constant first."""''', '''def _gaps(f, run):
    """Open gaps between the recordings' ranges of values f (n,), rows grouped by recording (run rows each)."""
    if not run or f.size % run:
        return []
    iv = sorted((float(b.min()), float(b.max())) for b in f.reshape(-1, run) if np.all(np.isfinite(b)))
    out, hi = [], None
    for lo_, hi_ in iv:
        if hi is not None and lo_ > hi:
            out.append((hi, lo_))
        hi = hi_ if hi is None else max(hi, hi_)
    return out


def natural_thetas(inf, U, W, run=None):
    """Constant values (B, k) for unit points U (B, k) by each constant's role (module docstring), constants whose
    subexpression involves no other constant first. run: rows per recording; a singular shift's pole is then also
    placed in the gaps between the recordings' ranges (24 Nov, patches/pole_gaps.py)."""''')
rep('''            if r["kind"] == "shift":
                lo, hi = float(f.min()), float(f.max())
                s = max(hi - lo, 1e-9 * max(abs(lo), abs(hi), 1e-300))
                if r["singular"]:
                    dist = s * logs((2 * u[b]) % 1.0)
                    p = lo - dist if u[b] < 0.5 else hi + dist''', '''            if r["kind"] == "shift":
                lo, hi = float(f.min()), float(f.max())
                s = max(hi - lo, 1e-9 * max(abs(lo), abs(hi), 1e-300))
                gaps = _gaps(sub[:, b], run) if r["singular"] and run else []
                if r["singular"] and gaps and u[b] >= 0.5:
                    v = (u[b] - 0.5) * 2 * len(gaps)
                    g0, g1 = gaps[min(int(v), len(gaps) - 1)]
                    p = g0 + (g1 - g0) * (0.02 + 0.96 * (v % 1.0))
                elif r["singular"]:
                    uu = 2 * u[b] if gaps else u[b]
                    dist = s * logs((2 * uu) % 1.0)
                    p = lo - dist if uu < 0.5 else hi + dist''')
rep('''    def __init__(self, keys, W):
        self.bank, self.W = {}, W''', '''    def __init__(self, keys, W, run=None):
        self.bank, self.W, self.run = {}, W, run''')
rep('''            th = natural_thetas(inf, unit_grid(inf.k), W)''', '''            th = natural_thetas(inf, unit_grid(inf.k), W, self.run)''')
rep('''def _thetas_from_unit(terms, slots, U, W):''', '''def _thetas_from_unit(terms, slots, U, W, run=None):''')
rep('''        P = natural_thetas(info(t.key), Um, W)''', '''        P = natural_thetas(info(t.key), Um, W, run)''')
rep('''def global_search(terms, W, y, rng):''', '''def global_search(terms, W, y, rng, run=None):''')
rep('''    TH = _thetas_from_unit(terms, slots, U, W)''', '''    TH = _thetas_from_unit(terms, slots, U, W, run)''')
rep('''        newTH = _thetas_from_unit(terms, slots, newU, W)''', '''        newTH = _thetas_from_unit(terms, slots, newU, W, run)''')
rep('''        for cand in global_search(current[0], W, y, self.rng):''', '''        for cand in global_search(current[0], W, y, self.rng, self.bank.run):''')
rep('''            for cand in global_search(current[0], W, y, rng)[:self.keep]:''', '''            for cand in global_search(current[0], W, y, rng, self.bank.run)[:self.keep]:''')
rep('''           "bank": Bank(keys, W), "dense": dense}''', '''           "bank": Bank(keys, W, view["Ztr"].shape[1] - 8), "dense": dense}''')
p.write_text(s)

q = HERE / "search_g1.py"
t = q.read_text()
old_a = '''    def __init__(self, W, order=None):
        order = order or vocabulary_order(W.shape[1])
        self.global_index = {k: g for g, k in enumerate(order)}
        self.bank = FG.Bank(order, W)'''
new_a = '''    def __init__(self, W, order=None, run=None):
        order = order or vocabulary_order(W.shape[1])
        self.global_index = {k: g for g, k in enumerate(order)}
        self.bank = FG.Bank(order, W, run)'''
old_b = '''    ctx["vocab"] = SystemVocab(ctx["W"])'''
new_b = '''    ctx["vocab"] = SystemVocab(ctx["W"], run=ctx.get("run"))'''
for o, n in ((old_a, new_a), (old_b, new_b)):
    assert t.count(o) == 1, o[:80]
    t = t.replace(o, n)
q.write_text(t)
print("patched fitter_g1.py and search_g1.py: poles also tried in the gaps between recordings")
