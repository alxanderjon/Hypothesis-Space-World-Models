"""Apply after q43 (owner, 22 November: "From now on, rung pass rules use the new definition of solved"), to ladder.py
once ladder_next.py has replaced it: (1) gen_job stores each system's description, so the true system is always
available; (2) after each held-out test, every system the search solved is put through rollout_check.check_job (one step
<= 1e-12 from 20 new starting states, rollouts <= 1e-9 where the true system stays bounded) and the round's test_solved,
and so mastery, count only systems that pass both the recording checks and this check; the recording-only count is kept
as test_solved_recordings. Usage: python patches/solved_def.py"""

from pathlib import Path

p = Path(__file__).resolve().parent.parent / "ladder.py"
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''        return {"k": k, "accepted": True, "tries": r + 1, "structure": [eq["keys"] for eq in syn.structure],
                "bursts": b8.tolist()}''', '''        return {"k": k, "accepted": True, "tries": r + 1, "structure": [eq["keys"] for eq in syn.structure],
                "bursts": b8.tolist(), "desc": desc}''')
rep('''            out[s][split].append({"id": f"{rung}-s{s}-{split}-{k}", "rung": rung, "structure": r["structure"], "bursts": r["bursts"]})''',
    '''            out[s][split].append({"id": f"{rung}-s{s}-{split}-{k}", "rung": rung, "structure": r["structure"], "bursts": r["bursts"],
                                  "desc": r.get("desc")})''')
rep('''            t = self.search(self.dir / f"r{rnd}_test.json", test, *test_budget)
            ok = sum(y["solved"] for y in t)''', '''            t = self.search(self.dir / f"r{rnd}_test.json", test, *test_budget)
            ok_rec = sum(y["solved"] for y in t)
            ok = self.strict(rnd, t, test) if not SMOKE else ok_rec          # owner, 22 Nov: the restored definition of solved''')
rep('''            self.rec["rounds"].append({"round": rnd, "budget": budget, "wake": w, "test": t, "test_solved": ok, "reanalyse": yld,''',
    '''            self.rec["rounds"].append({"round": rnd, "budget": budget, "wake": w, "test": t, "test_solved": ok, "test_solved_recordings": ok_rec,
                                       "reanalyse": yld,''')
rep('''    # -- the rung --
    def run(self, systems):''', '''    def strict(self, rnd, t, test):
        """Rollout check (rollout_check.check_job) on every held-out system the search solved; cached per round."""
        import rollout_check as RC
        p = self.dir / f"r{rnd}_rollout.json"
        if p.exists():
            rows = json.loads(p.read_text())
        else:
            by = {r["id"]: r for r in test}
            jobs = []
            for y in t:
                if y["solved"]:
                    sol = [None] * len(by[y["id"]]["structure"])
                    for e in y["equations"]:
                        sol[int(e["eq"])] = e["solution"]
                    jobs.append((by[y["id"]], self.rung, self.seed, int(y["id"].rsplit("-", 1)[1]), {(self.name, rnd): sol}))
            rows = [x for part in self.pool.map(RC.check_job, jobs) for x in part]
            p.write_text(json.dumps(rows, default=float))
        passed = {r["id"] for r in rows if r.get("passes")}
        for y in t:
            y["solved_strict"] = y["id"] in passed
        log(f"{self.rung} {self.name} round {rnd}: held-out solved {sum(y['solved'] for y in t)} on the recordings, "
            f"{len(passed)} under the rollout check")
        return len(passed)

    # -- the rung --
    def run(self, systems):''')
p.write_text(s)
print("patched ladder.py: descriptions stored; held-out solved = recordings + rollout check")
