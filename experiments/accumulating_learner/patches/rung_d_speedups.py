"""Apply after rung c finishes (its dream pools import ladder.py and dream_g2.py afresh every 100 dreams, so neither may
change while it runs). Owner, 17 November: from rung d on, (1) one fresh set of teacher dreams per round, shared by every
lineage in that round; (2) the pre-screen applied identically to every arm: it was already in use, but its starting states
were seeded with Python's per-process string hash, so the same draw could pass in one worker and fail in another; now a
stable hash. The empty-dream guard stays. Usage: python patches/rung_d_speedups.py"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def patch(path, pairs):
    p = HERE / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60])
        s = s.replace(old, new)
    p.write_text(s)
    print(f"patched {path}")


patch("dream_g2.py", [(
    '''    rng = np.random.default_rng(abs(hash(desc["seed"])) % (1 << 32))''',
    '''    import hashlib                                               # stable across processes (17 Nov; was hash(), salted per process)
    rng = np.random.default_rng(int(hashlib.sha1(str(desc["seed"]).encode()).hexdigest()[:8], 16))''')])

patch("ladder.py", [
    ('''MIN_EXAMPLES_PER_DREAM = 2.0                                     # below this a dream batch is treated as broken: stop loudly
''',
     '''MIN_EXAMPLES_PER_DREAM = 2.0                                     # below this a dream batch is treated as broken: stop loudly
SHARE_TEACHER_FROM = ("d", "e", "f", "g")                        # owner, 17 Nov: one teacher-dream set per round, shared by every lineage
_TEACHER, _TEACHER_LOCKS = {}, {}
'''),
    ('''                rd, ok = dream_map([(f"ladder-{self.rung}-{self.name}-{tag}-rung", k, allowed, R["terms"], self.d) for k in range(n)],
                                   f"{self.rung} {self.name} {tag} teacher dreams")
                log(f"{self.rung} {self.name} {tag}: {n} teacher dreams ({ok} with examples; {len(rd)} examples)")
''',
     '''                if self.rung in SHARE_TEACHER_FROM:
                    rd = shared_teacher(self.rung, self.n_learn, allowed, R["terms"], self.d, n)
                else:
                    rd, ok = dream_map([(f"ladder-{self.rung}-{self.name}-{tag}-rung", k, allowed, R["terms"], self.d) for k in range(n)],
                                       f"{self.rung} {self.name} {tag} teacher dreams")
                    log(f"{self.rung} {self.name} {tag}: {n} teacher dreams ({ok} with examples; {len(rd)} examples)")
'''),
    ('''def dream_map(jobs, what):''',
     '''def shared_teacher(rung, rnd, allowed, terms, d, n):
    """The round's teacher dreams: built once by the first lineage to need them, then given to every lineage in that
    round (and saved, so a restarted run reuses them). A new set every round."""
    with _LOCK:
        lock = _TEACHER_LOCKS.setdefault((rung, rnd), threading.Lock())
    with lock:
        if (rung, rnd) not in _TEACHER:
            p = OUT / rung / f"teacher_round{rnd}.pkl"
            if p.exists():
                _TEACHER[(rung, rnd)] = pickle.loads(p.read_bytes())
                log(f"{rung} round {rnd}: shared teacher dreams loaded ({len(_TEACHER[(rung, rnd)])} examples)")
            else:
                t0 = time.time()
                ex, ok = dream_map([(f"ladder-{rung}-round{rnd}-teacher", k, allowed, terms, d) for k in range(n)],
                                   f"{rung} round {rnd} shared teacher dreams")
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_bytes(pickle.dumps(ex))
                tmp.replace(p)
                _TEACHER[(rung, rnd)] = ex
                log(f"{rung} round {rnd}: {n} shared teacher dreams built ({ok} with examples; {len(ex)} examples) in {time.time() - t0:.0f}s")
    return _TEACHER[(rung, rnd)]


def dream_map(jobs, what):'''),
])
