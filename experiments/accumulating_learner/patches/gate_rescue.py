"""Fitter fix 3 (owner, 24 Nov: true equations must pass). A structure with two or more terms with constants whose
ordinary fit lands its constants in a wrong basin (rate-fit error 4-8e-2 on q43s's true structures) missed the precise
stage's gate (1e-2 for such structures) and never got the thorough fit (joint global search over the constants), which
rescues 4 of 6 such true structures in 1-3 s each. Now such a structure within RESCUE_MAX gets the thorough fit, then the
gate is re-taken. In the search, rescues count against the search's thorough cap (no exception for a best-so-far).
Usage: python patches/gate_rescue.py"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def patch(name, pairs):
    p = HERE / name
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:80])
        s = s.replace(old, new)
    p.write_text(s)


patch("fitter_g1.py", [
    ('''    def precise_gate(self, keys, res, rel_rate, floor):''',
     '''    def rescuable(self, keys, res, rel_rate, floor):
        """Missed the gate, but two or more terms with constants and within RESCUE_MAX: worth a thorough fit (24 Nov)."""
        return (bool(keys) and math.isfinite(rel_rate) and self.nonlinear(keys) >= 2 and rel_rate <= RESCUE_MAX
                and not self.precise_gate(keys, res, rel_rate, floor))

    def precise_gate(self, keys, res, rel_rate, floor):'''),
    ('''D_DEFAULT = 4 ''', '''RESCUE_MAX = 0.1              # 24 Nov: a gate miss this close, with two or more constants, gets the thorough fit
D_DEFAULT = 4 '''),
    ('''    gate = bool(keys) and fitter.precise_gate(keys, res, rel_rate, floor)
    if gate:''', '''    gate = bool(keys) and fitter.precise_gate(keys, res, rel_rate, floor)
    rescued = False
    if not gate and fitter.rescuable(keys, res, rel_rate, floor):    # 24 Nov: a wrong basin for the constants
        try:
            th = time.perf_counter()
            terms, res, coefs = fitter.thorough(keys, W, y, floor, (terms, res, coefs), eq=i, si_tr=ctx["si_tr"], si_va=ctx["si_va"])
            t_th += time.perf_counter() - th
            rel_rate = res / yn if math.isfinite(res) else float("inf")
            gate = fitter.precise_gate(keys, res, rel_rate, floor)
            rescued = thorough_now = True
        except (RuntimeError, ValueError):
            pass
    if gate:'''),
    ('''            if res > 2.0 * floor:
                th = time.perf_counter()''', '''            if res > 2.0 * floor and not rescued:
                th = time.perf_counter()'''),
])

patch("search_g1.py", [
    ('''        gate = bool(S) and self.fitter.precise_gate(keys, res, rel_rate, floor)
        thorough_now = False
        if gate and not superset:''', '''        gate = bool(S) and self.fitter.precise_gate(keys, res, rel_rate, floor)
        thorough_now = False
        if (not gate and not superset and self.fitter.rescuable(keys, res, rel_rate, floor)
                and (self.thorough_cap is None or eff["thorough"] < self.thorough_cap)):     # 24 Nov: a wrong basin
            try:
                eff["thorough"] += 1
                eff["rescues"] = eff.get("rescues", 0) + 1
                th = time.perf_counter()
                terms, res, coefs = self.fitter.thorough(keys, W, y, floor, (terms, res, coefs), eq=i, si_tr=self.si_tr, si_va=self.si_va)
                eff["thorough_seconds"] += time.perf_counter() - th
                rel_rate = res / yn if math.isfinite(res) else float("inf")
                gate = self.fitter.precise_gate(keys, res, rel_rate, floor)
                thorough_now = True
            except FIT_ERRORS:
                eff["fit_errors"] = eff.get("fit_errors", 0) + 1
        if gate and not superset:'''),
    ('''                if res > 2.0 * floor:
                    if self.thorough_cap is None''', '''                if res > 2.0 * floor and not thorough_now:
                    if self.thorough_cap is None'''),
])
print("patched fitter_g1.py and search_g1.py: gate rescue by the thorough fit")
