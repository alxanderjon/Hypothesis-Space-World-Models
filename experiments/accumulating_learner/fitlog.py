"""A compact record of every thorough fit (owner request, 4 November): training data for a later learned starting-value
network and a learned screen for which candidates deserve a thorough fit. Off unless a FitLog is passed to learn28.

One compressed file per run holds:
  - a header: the run's metadata (with its set tag), the vocabulary, each equation's floor, scale and exactness
    threshold, and per-equation totals; plus the bursts the learner saw (everything else is rebuilt from them);
  - one row per record:
      kind 0: a thorough fit made by the search;
      kind 1: a structure the thorough-fit cap refused (inputs only, no outcome);
      kind 2: a shadow thorough fit (optional, off by default): a small random sample of the structures the gate or the
              cap turned away, fitted with the fitter's random state and statistics restored afterwards and its time
              kept off the search's clock, so the search's decisions are unchanged; never for headline comparisons;
    each with the structure, the warm start, the cheap fit (the thorough fit's starting values), the fitter's random
    state before the fit, the constants after the thorough fit and after the step-integral refit, the residuals, the
    held-back error, the degenerate and plausibly-exact flags, the snapping counts and the times;
  - for fitter v2, a trace of every internal start (v1 sweep, algebraic, global, snap): its start and polished
    constants and its polished residual.
Term lists are stored as shapes (per term: number of factors, then each factor's number of constants) and the flat
constants, in float64; vocab.term(token, constants) rebuilds each term exactly (only a snapped factor's "snap"
provenance tag is not kept). res0/res1 are the residuals the fitter reported, which the search acted on; recomputing
|y - Phi c| from the stored terms and coefficients can differ slightly (up to ~4e-5 of |y| in ~3% of v1 fits, a
property of the fitter's own residual). The loader refuses test and reserve sets for training."""
import ast
import json
import math
import os
import time

import numpy as np

KINDS = {"thorough": 0, "refused": 1, "shadow": 2}
TAGS = {"current": 0, "v1": 1, "algebraic": 2, "global": 3, "snap": 4, "learned": 5}
TRAIN_SETS = ("train", "dev")                                   # usable for training without an explicit allowance
NEVER_TRAIN = ("test", "reserve")                               # refused for training, always
VERSION = 1


def set_of(prefix, explicit=None):
    """The set tag of a run from its system prefix: test and reserve prefixes can never be tagged otherwise."""
    p = prefix.lower()
    inferred = "test" if "test" in p else "reserve" if any(w in p for w in ("reserve", "fresh", "held")) else \
        "dev" if "dev" in p.split("-") or p.endswith("-dev") else None
    if inferred in NEVER_TRAIN and explicit not in (None, inferred):
        raise ValueError(f"a {inferred} prefix cannot be tagged '{explicit}'")
    return explicit or inferred or "unknown"


def enc_terms(terms):
    """Term list -> (shapes, constants)."""
    shapes, vals = [], []
    for tm in terms or []:
        shapes.append(len(tm.factors))
        for f in tm.factors:
            th = tuple(float(x) for x in f.theta)
            shapes.append(len(th)); vals.extend(th)
    return shapes, vals


def dec_thetas(shapes, vals):
    """(shapes, constants) -> per term, the tuple of per-factor constant tuples (vocab.term's second argument)."""
    out, a, b = [], 0, 0
    while a < len(shapes):
        nf = int(shapes[a]); a += 1
        th = []
        for _ in range(nf):
            nt = int(shapes[a]); a += 1
            th.append(tuple(float(v) for v in vals[b:b + nt])); b += nt
        out.append(tuple(th))
    return out


def _coefs(c):
    if c is None:
        return []
    try:
        return [float(v) for v in c.detach().cpu().numpy().ravel()]
    except AttributeError:
        return [float(v) for v in np.asarray(c).ravel()]


def rng_state(fitter):
    """The fitter's random state (PCG64) as 6 integers, or None if it has none."""
    g = getattr(fitter, "rng", None)
    if g is None:
        return None
    s = g.bit_generator.state
    st, inc = int(s["state"]["state"]), int(s["state"]["inc"])
    m = (1 << 64) - 1
    return [st >> 64, st & m, inc >> 64, inc & m, int(s["has_uint32"]), int(s["uinteger"])]


def set_rng_state(fitter, v):
    g = fitter.rng
    s = g.bit_generator.state
    s["state"]["state"] = (int(v[0]) << 64) | int(v[1])
    s["state"]["inc"] = (int(v[2]) << 64) | int(v[3])
    s["has_uint32"], s["uinteger"] = int(v[4]), int(v[5])
    g.bit_generator.state = s


class _Ragged:
    def __init__(self):
        self.vals, self.off = [], [0]

    def add(self, xs):
        self.vals.extend(xs); self.off.append(len(self.vals))


class FitLog:
    """Recorder for one run. learn28 calls begin(); the search calls before()/after() around each thorough fit and
    row() for every record; learn28 calls finish(); the runner calls save()."""

    def __init__(self, meta, shadow_rate=0.0, shadow_seed=0, trace=True):
        if "set" not in meta:
            raise ValueError("a fit log needs a set tag")
        self.meta, self.shadow_rate, self.shadow_seed = dict(meta), float(shadow_rate), int(shadow_seed)
        self.keep_trace = bool(trace)                           # v2's internal starts (about half of a log's size)
        self.header, self.bursts = {}, None
        self.cols = {k: [] for k in ("kind", "eq", "step", "seq", "gate", "breaker", "failed", "res0", "res1", "rel_t",
                                     "rel_v", "degen", "exact", "t_thorough", "t_precise", "snaps_checked",
                                     "snaps_allowed", "snaps_kept", "winner", "has_rng", "rng")}
        self.rag = {k: _Ragged() for k in ("tok", "warm", "s0", "th0", "c0", "s1", "th1", "c1", "s2", "th2", "c2")}
        self.tr = {k: [] for k in ("row", "tag", "res")}
        self.tr_rag = {k: _Ragged() for k in ("ss", "sth", "ps", "pth")}
        self.index = None

    # -- run level ---------------------------------------------------------------------------------------------------
    def begin(self, bursts, seed, vocab, floors, yns, taus):
        self.bursts = np.asarray(bursts, dtype=np.float64)
        self.index = vocab.index
        self.header = {"version": VERSION, "meta": self.meta, "seed": int(seed), "d": int(vocab.d),
                       "vocab": [repr(t) for t in vocab.tokens], "floors": [float(v) for v in floors],
                       "scales": [float(v) for v in yns], "taus": [float(v) for v in taus],
                       "shadow_rate": self.shadow_rate, "shadow_seed": self.shadow_seed, "trace": self.keep_trace}

    def finish(self, equations):
        self.header["equations"] = [{k: e.get(k) for k in ("structures", "precise", "thorough", "thorough_capped",
                                                           "shadow", "first_exact")} for e in equations]

    # -- around a thorough fit ---------------------------------------------------------------------------------------
    def before(self, fitter):
        rec = {"rng": rng_state(fitter), "stats": dict(getattr(fitter, "stats", {}) or {}), "t0": time.perf_counter()}
        if hasattr(fitter, "trace") and self.keep_trace:
            fitter.trace = []
        return rec

    def after(self, fitter, rec, out):
        rec["t"] = time.perf_counter() - rec["t0"]
        rec["out"] = out
        rec["trace"] = getattr(fitter, "trace", None)
        if hasattr(fitter, "trace"):
            fitter.trace = None
        st = getattr(fitter, "stats", {}) or {}
        rec["snaps"] = [st.get(k, 0) - rec["stats"].get(k, 0) for k in ("exponents_checked", "snaps_allowed", "snaps_kept")]
        rec["winner"] = (getattr(fitter, "last", None) or {}).get("winner")
        if (getattr(fitter, "last", None) or {}).get("snap_kept"):
            rec["winner"] = "snap"

    @staticmethod
    def freeze(fitter):
        """For a shadow fit: the fitter's random state and statistics, to restore afterwards."""
        g = getattr(fitter, "rng", None)
        return (g.bit_generator.state if g is not None else None,
                {k: v for k, v in (getattr(fitter, "stats", {}) or {}).items()}, getattr(fitter, "last", None))

    @staticmethod
    def thaw(fitter, snap):
        state, stats, last = snap
        if state is not None:
            fitter.rng.bit_generator.state = state
        if hasattr(fitter, "stats"):
            fitter.stats.clear(); fitter.stats.update(stats)
        if hasattr(fitter, "last"):
            fitter.last = last

    # -- one record --------------------------------------------------------------------------------------------------
    def row(self, kind, eq, step, seq, gate, S, warm, cheap, rec=None, final=None, rel_t=float("inf"),
            rel_v=float("inf"), degen=False, exact=False, t_precise=0.0, breaker=False):
        c, r = self.cols, self.rag
        n = len(c["kind"])
        c["kind"].append(KINDS[kind]); c["eq"].append(eq); c["step"].append(step); c["seq"].append(seq)
        c["gate"].append(gate); c["breaker"].append(bool(breaker))
        r["tok"].add([self.index[t] for t in S])
        r["warm"].add([warm[t] if isinstance(warm.get(t), int) else -1 for t in S])
        sh, th = enc_terms(cheap[0]); r["s0"].add(sh); r["th0"].add(th); r["c0"].add(_coefs(cheap[2]))
        c["res0"].append(float(cheap[1]))
        out = rec.get("out") if rec else None
        c["failed"].append(bool(rec is not None and out is None))
        if out is not None:
            sh, th = enc_terms(out[0]); r["s1"].add(sh); r["th1"].add(th); r["c1"].add(_coefs(out[2]))
            c["res1"].append(float(out[1]))
        else:
            r["s1"].add([]); r["th1"].add([]); r["c1"].add([]); c["res1"].append(float("nan"))
        if final is not None and rec is not None:
            sh, th = enc_terms(final[0]); r["s2"].add(sh); r["th2"].add(th); r["c2"].add(_coefs(final[1]))
        else:
            r["s2"].add([]); r["th2"].add([]); r["c2"].add([])
        c["rel_t"].append(float(rel_t)); c["rel_v"].append(float(rel_v))
        c["degen"].append(bool(degen)); c["exact"].append(bool(exact))
        c["t_thorough"].append(float(rec["t"]) if rec and "t" in rec else 0.0); c["t_precise"].append(float(t_precise))
        sn = rec.get("snaps", [0, 0, 0]) if rec else [0, 0, 0]
        c["snaps_checked"].append(sn[0]); c["snaps_allowed"].append(sn[1]); c["snaps_kept"].append(sn[2])
        c["winner"].append(TAGS.get(rec.get("winner"), -1) if rec else -1)
        st = rec.get("rng") if rec else None
        c["has_rng"].append(st is not None); c["rng"].append(st if st is not None else [0] * 6)
        for tag, start, pol, res in (rec.get("trace") or []) if rec else []:
            self.tr["row"].append(n); self.tr["tag"].append(TAGS.get(tag, -1)); self.tr["res"].append(float(res))
            sh, th = enc_terms(start); self.tr_rag["ss"].add(sh); self.tr_rag["sth"].add(th)
            sh, th = enc_terms(pol); self.tr_rag["ps"].add(sh); self.tr_rag["pth"].add(th)

    # -- file --------------------------------------------------------------------------------------------------------
    def arrays(self):
        c = self.cols
        a = {"header": np.array(json.dumps(self.header)), "bursts": self.bursts,
             "kind": np.array(c["kind"], np.int8), "eq": np.array(c["eq"], np.int8), "step": np.array(c["step"], np.int32),
             "seq": np.array(c["seq"], np.int32), "gate": np.array(c["gate"], np.int8),
             "breaker": np.array(c["breaker"], bool), "failed": np.array(c["failed"], bool),
             "degen": np.array(c["degen"], bool), "exact": np.array(c["exact"], bool),
             "has_rng": np.array(c["has_rng"], bool), "rng": np.array(c["rng"], np.uint64).reshape(-1, 6),
             "winner": np.array(c["winner"], np.int8)}
        for k in ("res0", "res1", "rel_t", "rel_v"):
            a[k] = np.array(c[k], np.float64)
        for k in ("t_thorough", "t_precise"):
            a[k] = np.array(c[k], np.float32)
        for k in ("snaps_checked", "snaps_allowed", "snaps_kept"):
            a[k] = np.array(c[k], np.int16)
        dt = {"tok": np.int16, "warm": np.int32, "s0": np.int8, "s1": np.int8, "s2": np.int8}
        for k, rg in self.rag.items():
            a[k] = np.array(rg.vals, dt.get(k, np.float64)); a[k + "_off"] = np.array(rg.off, np.int32)
        a["tr_row"] = np.array(self.tr["row"], np.int32); a["tr_tag"] = np.array(self.tr["tag"], np.int8)
        a["tr_res"] = np.array(self.tr["res"], np.float64)
        for k, rg in self.tr_rag.items():
            a["tr_" + k] = np.array(rg.vals, np.int8 if k in ("ss", "ps") else np.float64)
            a["tr_" + k + "_off"] = np.array(rg.off, np.int32)
        return a

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **self.arrays())
        os.replace(tmp, path)
        return os.path.getsize(path)


# ---------------------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------------------

def header_of(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["header"]))


def load(paths, purpose="train", allow=()):
    """Read fit logs. For training, files from test or reserve sets are always refused, and other sets need to be in
    TRAIN_SETS or `allow`; for evaluation every set is read. Returns a list of runs (header, bursts and columns)."""
    if purpose not in ("train", "evaluate"):
        raise ValueError("purpose is 'train' or 'evaluate'")
    bad = [a for a in allow if a in NEVER_TRAIN]
    if bad:
        raise ValueError(f"sets {bad} can never be used for training")
    runs, refused = [], 0
    for p in paths:
        h = header_of(p)
        s = h["meta"]["set"]
        if purpose == "train" and (s in NEVER_TRAIN or s not in TRAIN_SETS + tuple(allow)):
            refused += 1
            continue
        with np.load(p, allow_pickle=False) as z:
            run = {k: z[k] for k in z.files if k != "header"}
        run["header"] = h
        runs.append(run)
    if refused and purpose == "train":
        print(f"fitlog.load: refused {refused} file(s) not allowed for training (test, reserve or untagged sets)")
    return runs


def rows(run):
    """Every record of a run as a dict: tokens (decoded), warm start, and the cheap, thorough and final constants."""
    vocab = [ast.literal_eval(s) for s in run["header"]["vocab"]]
    sl = lambda k, j: run[k][run[k + "_off"][j]:run[k + "_off"][j + 1]]
    for j in range(len(run["kind"])):
        yield {"kind": int(run["kind"][j]), "eq": int(run["eq"][j]), "step": int(run["step"][j]),
               "seq": int(run["seq"][j]), "gate": int(run["gate"][j]), "breaker": bool(run["breaker"][j]),
               "failed": bool(run["failed"][j]),
               "tokens": tuple(vocab[int(t)] for t in sl("tok", j)), "warm": [int(w) for w in sl("warm", j)],
               "cheap": (dec_thetas(sl("s0", j), sl("th0", j)), sl("c0", j), float(run["res0"][j])),
               "thorough": (dec_thetas(sl("s1", j), sl("th1", j)), sl("c1", j), float(run["res1"][j])),
               "final": (dec_thetas(sl("s2", j), sl("th2", j)), sl("c2", j)),
               "rel_t": float(run["rel_t"][j]), "rel_v": float(run["rel_v"][j]), "degen": bool(run["degen"][j]),
               "exact": bool(run["exact"][j]), "t_thorough": float(run["t_thorough"][j]),
               "t_precise": float(run["t_precise"][j]), "winner": int(run["winner"][j]),
               "rng": [int(v) for v in run["rng"][j]] if run["has_rng"][j] else None}


def trace_of(run, j):
    """Fitter v2's internal starts for record j: (tag, start constants, polished constants, polished residual)."""
    sel = np.flatnonzero(run["tr_row"] == j)
    sl = lambda k, e: run["tr_" + k][run["tr_" + k + "_off"][e]:run["tr_" + k + "_off"][e + 1]]
    inv = {v: k for k, v in TAGS.items()}
    return [(inv.get(int(run["tr_tag"][e]), "?"), dec_thetas(sl("ss", e), sl("sth", e)), dec_thetas(sl("ps", e), sl("pth", e)),
             float(run["tr_res"][e])) for e in sel]
