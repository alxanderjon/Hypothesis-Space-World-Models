"""Stage 1 of the general learner (research log, 6 November): the tree search over L1 term sets.

structure_search.EquationSearch on L1 terms with the generic fitter (fitter_g1): moves add, remove or swap one term; PUCT
with an exploration share and transpositions; every new structure fitted (ordinary fit), the precise stage's gate, a
thorough fit (capped per equation: the first `thorough_cap`, then only record-breakers), step-integral refit and held-back
check; strict supersets of a plausibly exact structure skip the precise stage; every plausibly exact structure's
one-term-smaller structures are evaluated next. At most MAX_TERMS terms per equation (Stage 1 rules: 6).

With 10^4 terms the move list is not stored: priors are kept factored (add log-probabilities over the vocabulary, removal
log-probabilities over the state, swap = removal x addition), as every prior in the toolkit search was (logits add(t),
remove(s), remove(s) + add(t), mixed 45% / 15% / 40% by move type). The uniform prior gives every legal move the same
probability (structure_search.uniform_prior); ties among untried moves go to the first legal move in the vocabulary's
order (simplest terms first: fewest nodes, then fewest constants), as in the toolkit search.
"""

import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import fitter_g1 as FG
import structure_search as SS

MAX_TERMS = 6
FIT_ERRORS = (RuntimeError, ValueError, TypeError, ZeroDivisionError, IndexError, FloatingPointError, OverflowError)
WEIGHTS = {"add": 0.45, "remove": 0.15, "swap": 0.40}
VOCAB_PATH = Path(__file__).resolve().parents[2] / "outputs/accumulating_learner/g1/vocab_d2.json"
_ORDER = None


CLOSED_PATH = VOCAB_PATH.parent / "scale_closed_d2.json"


def vocabulary_order():
    """The search's vocabulary in the search's order (fewest nodes, then fewest constants, then the key): the terms
    that keep their canonical form when the variables are rescaled (g1_scale_closed.py). The learner's coordinates are
    the physical ones divided by scales from the data, so only these can be exact; a term like sin(x1) is the K0 = 1
    member of sin(K0 x1), which is kept (research log, 6 November)."""
    global _ORDER
    if _ORDER is None:
        V = json.loads(VOCAB_PATH.read_text())["terms"]
        closed = set(json.loads(CLOSED_PATH.read_text()))
        _ORDER = sorted([k for k in V if k in closed], key=lambda k: (V[k]["nodes"], V[k]["consts"], len(k), k))
    return _ORDER


class SystemVocab:
    """The vocabulary on one system: the terms defined (finite, nonzero) on its rate points at some grid placement, in
    the global order, with their grid columns (bank) and the normalized columns for the data evidence."""

    def __init__(self, W, order=None):
        order = order or vocabulary_order()
        self.global_index = {k: g for g, k in enumerate(order)}
        self.bank = FG.Bank(order, W)
        self.keys = [k for k in order if k in self.bank.bank]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.gid = np.array([self.global_index[k] for k in self.keys])
        cols, owner = [], []
        for i, k in enumerate(self.keys):
            C = self.bank.bank[k][0]
            cols.append(C / C.norm(dim=0, keepdim=True))
            owner.append(torch.full((C.shape[1],), i, dtype=torch.long))
        self.A = torch.cat(cols, 1)
        self.owner = torch.cat(owner)
        self.V = len(self.keys)

    def add_gain(self, Q, r):
        """Best partial R^2 of each term's grid columns given the span Q of the current terms and the residual r."""
        C = self.A - Q @ (Q.T @ self.A) if Q.shape[1] else self.A
        cn = (C * C).sum(0)
        g = torch.where(cn > 1e-20, (C.T @ r) ** 2 / (cn * float(r @ r) + 1e-300), torch.zeros_like(cn)).clamp(0, 1)
        return torch.zeros(self.V, dtype=g.dtype).scatter_reduce(0, self.owner, g, reduce="amax")


def evidence(node, ctx):
    """Each term's best partial R^2 for being added given the node's fitted terms, and for each current term the
    fraction of the remaining residual its removal would add (structure_search._evidence)."""
    y, S, W = ctx["y"], node["keys"], ctx["W"]
    if S:
        Phi = torch.stack([FG.value(t, W) for t in node["terms"]], 1)
        if not bool(torch.isfinite(Phi).all()):
            Phi = torch.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)
        Q = torch.linalg.qr(Phi)[0]
        r = y - Q @ (Q.T @ y)
    else:
        Phi = Q = torch.zeros(len(y), 0, dtype=y.dtype)
        r = y
    gain = ctx["vocab"].add_gain(Q, r).numpy()
    rem = np.zeros(len(S))
    if S:
        XtX = Phi.T @ Phi + 1e-12 * torch.eye(Phi.shape[1], dtype=Phi.dtype)
        try:
            cov = torch.linalg.inv(XtX)
            c = torch.linalg.lstsq(Phi, y.reshape(-1, 1), driver="gelsd").solution[:, 0]
            dg = torch.diag(cov)
            ok = (dg > 0) & torch.isfinite(dg) & torch.isfinite(c)
            delta = torch.where(ok, c ** 2 / torch.where(ok, dg, torch.ones_like(dg)), torch.zeros_like(dg)).numpy()
            rr = float(r @ r) + 1e-300
            rem = np.array([min(max(float(v) / rr, 0.0), 1e30) if math.isfinite(float(v) / rr) else 0.0 for v in delta])   # capped: a near-exact fit makes it astronomically large (float32 overflow)
        except RuntimeError:
            pass
    return gain, rem


# ---------------------------------------------------------------------------------------------------------------------
# factored priors
# ---------------------------------------------------------------------------------------------------------------------

class Prior:
    """Move probabilities at a node, factored. Moves are indexed: adds k (vocabulary index), then removals s (position
    in the state), then swaps s -> k (index V + nS + s * V + k)."""

    def __init__(self, V, inS_idx, add_logit=None, rem_logit=None, uniform=False):
        self.V, self.inS, self.nS = V, np.asarray(inS_idx, dtype=np.int64), len(inS_idx)
        self.M = V + self.nS + self.nS * V
        legal_add = np.ones(V, dtype=bool)
        legal_add[self.inS] = False
        self.legal_add = legal_add
        n_out = int(legal_add.sum())
        can_add = self.nS < MAX_TERMS and n_out > 0
        if uniform:
            n_moves = (n_out if can_add else 0) + self.nS + self.nS * n_out
            p = 1.0 / max(n_moves, 1)
            self.p_add = np.where(legal_add, p, 0.0) if can_add else np.zeros(V)
            self.p_rem = np.full(self.nS, p)
            self.p_swap_rem = np.full(self.nS, p)                # swap probability = p_swap_rem[s] * p_swap_add[k]
            self.p_swap_add = np.where(legal_add, 1.0, 0.0)
            return
        present = {}
        if can_add:
            present["add"] = True
        if self.nS:
            present["remove"] = True
            if n_out:
                present["swap"] = True
        wsum = sum(WEIGHTS[k] for k in present) or 1.0
        a = np.where(legal_add, add_logit, -np.inf)
        pa = np.exp(a - a[legal_add].max()) if n_out else np.zeros(V)
        pa = pa / pa.sum() if n_out else pa
        pr = np.zeros(self.nS)
        if self.nS:
            r = np.asarray(rem_logit, dtype=float)
            pr = np.exp(r - r.max()); pr /= pr.sum()
        self.p_add = WEIGHTS["add"] / wsum * pa if "add" in present else np.zeros(V)
        self.p_rem = WEIGHTS["remove"] / wsum * pr if "remove" in present else np.zeros(self.nS)
        self.p_swap_rem = WEIGHTS["swap"] / wsum * pr if "swap" in present else np.zeros(self.nS)
        self.p_swap_add = pa

    def vector(self):
        parts = [self.p_add, self.p_rem]
        if self.nS:
            parts.append(np.outer(self.p_swap_rem, self.p_swap_add).ravel())
        return np.concatenate(parts)

    def legal(self):
        parts = [self.legal_add & (self.nS < MAX_TERMS), np.ones(self.nS, dtype=bool)]
        if self.nS:
            parts.append(np.tile(self.legal_add, self.nS))
        return np.concatenate(parts)


def apply_move(S_idx, m, V):
    """New state (sorted tuple of vocabulary indices) after move m."""
    nS = len(S_idx)
    if m < V:
        return tuple(sorted(S_idx + (m,)))
    if m < V + nS:
        s = m - V
        return tuple(x for j, x in enumerate(S_idx) if j != s)
    s, k = divmod(m - V - nS, V)
    return tuple(sorted(tuple(x for j, x in enumerate(S_idx) if j != s) + (k,)))


def uniform_prior(node, ctx):
    return Prior(ctx["vocab"].V, node["idx"], uniform=True)


def data_prior(node, ctx):
    """Priorities from this system's data only (structure_search.data_prior): add in proportion to 1 / (fraction of the
    residual a term leaves unexplained); remove in proportion to 1 / (1 + residual increase / residual)."""
    gain, rem = evidence(node, ctx)
    add = -np.log(np.clip(1.0 - gain, 1e-12, 1.0))
    return Prior(ctx["vocab"].V, node["idx"], add, -np.log1p(rem))


# ---------------------------------------------------------------------------------------------------------------------
# the tree search over one equation
# ---------------------------------------------------------------------------------------------------------------------

class EquationSearchL1:
    """structure_search.EquationSearch for L1 (module docstring). States are sorted tuples of vocabulary indices."""

    def __init__(self, i, ctx, budget, prior_fn, fitter, rng, c_puct=1.5, eps=0.15, time_cap=float("inf"), lam=0.02,
                 thorough_cap=None):
        self.i, self.budget, self.prior_fn, self.fitter, self.rng = i, budget, prior_fn, fitter, rng
        self.c_puct, self.eps, self.time_cap, self.lam, self.thorough_cap = c_puct, eps, time_cap, lam, thorough_cap
        self.vocab, self.W = ctx["vocab"], ctx["W"]
        self.y, self.yn, self.floor, self.tau = FG.equation_targets(ctx, i)
        self.ctx = dict(ctx, y=self.y, i=i)
        self.si_tr, self.si_va = ctx["si_tr"], ctx["si_va"]
        self.table, self.exact_sets, self.prune = {}, [], []
        self.eff = {"structures": 0, "fit_seconds": 0.0, "prior_seconds": 0.0, "first_exact": None, "first_exact_seconds": None,
                    "tau": self.tau, "floor_rel": self.floor / self.yn, "precise": 0, "precise_seconds": 0.0, "pruned": 0,
                    "seconds": 0.0, "stalls": 0, "thorough": 0, "thorough_seconds": 0.0, "thorough_capped": 0,
                    "thorough_best_rel": float("inf"), "thorough_exact": 0, "forced": 0, "time_capped": False}
        self._t = time.perf_counter()
        self.root = self.evaluate((), None)
        self.eff["seconds"] += time.perf_counter() - self._t

    @property
    def exhausted(self):
        return self.eff["structures"] >= self.budget or self.eff["seconds"] >= self.time_cap or self.eff["stalls"] > 2000

    def keys_of(self, S):
        return tuple(self.vocab.keys[k] for k in S)

    def evaluate(self, S, parent):
        y, floor, yn, i, eff, W = self.y, self.floor, self.yn, self.i, self.eff, self.W
        keys = self.keys_of(S)
        warm = {}
        if parent is not None:
            for k, term in zip(parent["keys"], parent["terms"]):
                if k in keys and term.k and isinstance(term.grid_index, int):
                    warm[k] = term.grid_index
        tf = time.perf_counter()
        try:
            terms, res, coefs, _ = self.fitter.fit(keys, W, y, warm, floor=floor, eq=i)
        except FIT_ERRORS:
            terms, res, coefs = [FG.LTerm(k, [1.0] * FG.info(k).k) for k in keys], float("inf"), None
            eff["fit_errors"] = eff.get("fit_errors", 0) + 1
        eff["structures"] += 1
        eff["fit_seconds"] += time.perf_counter() - tf
        rel_rate = res / yn if math.isfinite(res) else float("inf")
        rel_t = rel_v = float("inf")
        superset = any(E < set(S) for E in self.exact_sets)
        gate = bool(S) and self.fitter.precise_gate(keys, res, rel_rate, floor)
        thorough_now = False
        if gate and not superset:
            tp = time.perf_counter()
            try:
                if res > 2.0 * floor:
                    if self.thorough_cap is None or eff["thorough"] < self.thorough_cap or rel_rate < eff["thorough_best_rel"]:
                        eff["thorough"] += 1
                        eff["thorough_best_rel"] = min(eff["thorough_best_rel"], rel_rate)
                        th = time.perf_counter()
                        terms, res, coefs = self.fitter.thorough(keys, W, y, floor, (terms, res, coefs), eq=i,
                                                                 si_tr=self.si_tr, si_va=self.si_va)
                        eff["thorough_seconds"] += time.perf_counter() - th
                        thorough_now = True
                    else:
                        eff["thorough_capped"] += 1
                terms2, rel_t, c2 = FG.refine_integral(terms, self.si_tr, i)
                if c2 is not None and math.isfinite(rel_t):
                    terms, coefs = terms2, c2
                    rel_v = FG.heldback_error(terms, coefs, self.si_va, i)
            except FIT_ERRORS:
                eff["fit_errors"] = eff.get("fit_errors", 0) + 1
            eff["precise"] += 1
            eff["precise_seconds"] += time.perf_counter() - tp
        try:
            degen = any(FG.degenerate(t, W) for t in terms)
        except FIT_ERRORS:
            degen = True
        rel = rel_v if math.isfinite(rel_v) and not degen else max(rel_rate, SS.PRECISE_GATE)
        v = 0.0 if not math.isfinite(rel) else min(1.0, max(0.0, math.log10(1.0 / max(rel, 1e-14)) / 14.0))
        node = {"idx": S, "keys": keys, "terms": terms, "coefs": coefs, "res": res, "rel_t": rel_t, "rel_v": rel_v,
                "value": v - self.lam * len(S), "n": 0, "k": eff["structures"], "degenerate": degen,
                "best": v - self.lam * len(S), "prior": None, "exact": bool(rel_v <= self.tau and rel_t <= self.tau and not degen)}
        self.table[S] = node
        if node["exact"] and thorough_now:
            eff["thorough_exact"] += 1
        if node["exact"]:
            self.exact_sets.append(set(S))
            queued = {q for q, _ in self.prune}
            for t in S:
                q = tuple(u for u in S if u != t)
                if q and q not in self.table and q not in queued:
                    self.prune.append((q, node)); queued.add(q)
            if eff["first_exact"] is None:
                eff["first_exact"] = eff["structures"]
                eff["first_exact_seconds"] = eff["seconds"] + time.perf_counter() - self._t
        return node

    def _expand(self, node):
        if node["prior"] is None:
            tp = time.perf_counter()
            node["prior"] = self.prior_fn(node, self.ctx)
            node["P"] = node["prior"].vector()
            node["legal"] = node["prior"].legal()
            node["tried"], node["child_of"] = [], {}
            self.eff["prior_seconds"] += time.perf_counter() - tp

    def step(self):
        table, eff, rng, V = self.table, self.eff, self.rng, self.vocab.V
        if self.prune:
            sub, par = self.prune.pop(0)
            if sub not in table:
                self.evaluate(sub, par)
                eff["pruned"] += 1
            return
        n0 = eff["structures"]
        node, path = self.root, [self.root]
        on_path = {self.root["idx"]}
        for _depth in range(2 * MAX_TERMS + 4):
            self._expand(node)
            P, legal = node["P"], node["legal"]
            if not legal.any():
                break
            tried = node["tried"]
            banned = [m for m in tried if node["child_of"][m] in on_path]
            S2 = None
            for _pick in range(20):
                untried = legal.copy()
                if tried:
                    untried[tried] = False
                if untried.any() and rng.random() < self.eps:
                    cand_m = np.flatnonzero(untried)
                    m = int(cand_m[int(rng.integers(len(cand_m)))])
                else:
                    q = np.full(len(P), node["value"] - 0.05)
                    nv = np.zeros(len(P))
                    for mm in tried:
                        ch = table[node["child_of"][mm]]
                        q[mm] = ch["best"]; nv[mm] = ch["n"]
                    score = q + self.c_puct * P * math.sqrt(node["n"] + 1) / (1 + nv)
                    score[~legal] = -np.inf
                    if banned:
                        score[banned] = -np.inf
                    m = int(np.argmax(score))
                    if not np.isfinite(score[m]):
                        break
                cand = node["child_of"].get(m) or apply_move(node["idx"], m, V)
                if m not in node["child_of"]:
                    tried.append(m); node["child_of"][m] = cand
                if cand in on_path:
                    banned.append(m)
                    continue
                S2 = cand
                break
            if S2 is None:
                break
            if S2 in table:
                node = table[S2]; path.append(node); on_path.add(S2)
                continue
            child = self.evaluate(S2, node)
            path.append(child)
            break
        if eff["structures"] == n0:
            child = self._force_new(path)
            if child is not None:
                path.append(child)
        leaf_v = path[-1]["value"]
        for nd in path:
            nd["n"] += 1
            nd["best"] = max(nd["best"], leaf_v)
        eff["stalls"] = 0 if eff["structures"] > n0 else eff["stalls"] + 1

    def _force_new(self, path, tries=50):
        rng, table, V = self.rng, self.table, self.vocab.V
        keys = list(table)
        extra = [table[keys[int(a)]] for a in rng.choice(len(keys), size=min(20, len(keys)), replace=False)]
        for node in list(reversed(path)) + extra:
            self._expand(node)
            untried = node["legal"].copy()
            if node["tried"]:
                untried[node["tried"]] = False
            cand_m = np.flatnonzero(untried)
            rng.shuffle(cand_m)
            for m in cand_m[:tries]:
                m = int(m)
                S2 = apply_move(node["idx"], m, V)
                node["tried"].append(m); node["child_of"][m] = S2
                if S2 not in table:
                    self.eff["forced"] += 1
                    return self.evaluate(S2, node)
        return None

    def run(self, extra_after_exact=1.0, min_extra=500):
        """Search until the budget or the time cap, or until a plausibly exact structure has been found and
        max(min_extra, extra_after_exact * k) more structures have been tried (structure_search.EquationSearch.run)."""
        eff = self.eff
        self._t = time.perf_counter()
        while not self.exhausted:
            if not self.prune and eff["first_exact"] is not None and \
                    eff["structures"] >= eff["first_exact"] + max(min_extra, extra_after_exact * eff["first_exact"]):
                break
            self.step()
            now = time.perf_counter()
            eff["seconds"] += now - self._t
            self._t = now
        eff["time_capped"] = eff["seconds"] >= self.time_cap
        return self

    def solutions(self):
        """Plausibly exact structures, simplest first (fewest terms, then fewest terms with constants, then smallest
        held-back error)."""
        ex = [n for n in self.table.values() if n["exact"]]
        return sorted(ex, key=lambda n: (len(n["keys"]), sum(FG.info(k).k > 0 for k in n["keys"]), n["rel_v"]))


def system_context(bursts, seed=1):
    ctx = FG.system_context(bursts, seed=seed, keys=())
    ctx["vocab"] = SystemVocab(ctx["W"])
    ctx["bank"] = ctx["vocab"].bank
    return ctx


def search_system(bursts, prior_fn, budget=1000, thorough_cap=200, time_cap=float("inf"), seed=1, ctx=None):
    """Search every equation of one system. Returns per equation: effort, the simplest plausibly exact structure (keys,
    constants, coefficients, held-back error) and the number of plausibly exact structures found."""
    t0 = time.perf_counter()
    ctx = ctx or system_context(bursts, seed=seed)
    setup = time.perf_counter() - t0
    fitter = FG.make_fitter(ctx, seed=seed)
    out = {"setup_seconds": setup, "vocabulary": ctx["vocab"].V, "equations": []}
    d = ctx["W"].shape[1]
    for i in range(d):
        rng = np.random.default_rng([seed, i])
        se = EquationSearchL1(i, ctx, budget, prior_fn, fitter, rng, time_cap=time_cap / d, thorough_cap=thorough_cap).run()
        sols = se.solutions()
        rec = {"eq": i, "effort": dict(se.eff), "n_exact": len(sols)}
        if sols:
            b = sols[0]
            rec["solution"] = {"keys": list(b["keys"]), "thetas": [list(t.theta) for t in b["terms"]],
                               "coefs": [float(c) for c in b["coefs"]], "rel_v": b["rel_v"], "rel_t": b["rel_t"], "k": b["k"]}
        out["equations"].append(rec)
    out["fitter_stats"] = dict(fitter.stats)
    out["seconds"] = time.perf_counter() - t0
    return out
