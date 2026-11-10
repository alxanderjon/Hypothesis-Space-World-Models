"""Stage X of the redesign (owner, 10 November; research log "Stage X: exploration as a safety net"): the equation search
with two general exploration rules, the same for every network and system.

search_g1.EquationSearchL1 with:
  - never blocked (rho): at every visit of a node in the network-guided search, with probability `rho` the search tries
    the next untried term among those the prior rates below even (below 1 / the number of legal adds), in the
    language's order (the vocabulary's order: fewest nodes, then fewest constants, the order in which the uniform search
    breaks ties); otherwise it chooses as before. Like the search's existing random exploration share (eps), this is a
    share of choices, not of prior mass: a share of prior mass is not a share of choices in PUCT once the network's
    children fit well (research log, 10 November). Learned preferences order everything else, so an unfamiliar term can
    be delayed but not blocked.
  - exploration share (share): the first `share` of the equation's structure budget is searched with even weights (the
    uniform prior); then the network's prior takes over: every node's prior is recomputed when it is next visited, and
    every structure already fitted is reused.
Recorded per equation (owner, 10 November): structures searched with even weights; the distinct low-rated terms tried
by any rule (terms rated below even at the node where they were tried; counted with the rules off too, for the
baseline) and how deep into the simplest-first order of low-rated terms any node got (the longest unbroken run from
the simplest); the solution's terms that the network rated below even at the root; the time split (ordinary fits,
thorough fits, precise checks, prior).
"""

import json
import time

import numpy as np

import fitter_g1 as FG
import search_g1 as SG

_NODES = {}


def term_nodes(key, d):
    if d not in _NODES:
        _NODES[d] = {k: v["nodes"] for k, v in json.loads((SG.G1_DIR / f"vocab_d{d}.json").read_text())["terms"].items()}
    return _NODES[d].get(key)


class ExploringSearch(SG.EquationSearchL1):
    def __init__(self, i, ctx, budget, prior_fn, fitter, rng, share=0.0, rho=0.0, **kw):
        self.network_prior, self.share, self.rho = prior_fn, float(share), float(rho)
        self.switch_at = int(round(self.share * budget))
        self.phase = "even" if self.switch_at > 0 else "network"
        self.low_tried, self.low_depth, self.root_low = set(), 0, None
        self.xrng = np.random.default_rng(int(rng.integers(1 << 31))) if self.rho else None   # the never-blocked draws
        super().__init__(i, ctx, budget, SG.uniform_prior if self.phase == "even" else prior_fn, fitter, rng, **kw)
        self.eff["even_structures"] = self.eff["structures"] if self.phase == "even" else 0

    def step(self):
        if self.phase == "even" and self.eff["structures"] >= self.switch_at:
            self.phase, self.prior_fn = "network", self.network_prior
            for node in self.table.values():                     # the network's preferences take over
                node["prior"] = None
                node.pop("base_P", None)
        n0 = self.eff["structures"]
        super().step()
        if self.phase == "even":
            self.eff["even_structures"] += self.eff["structures"] - n0

    def _expand(self, node):
        super()._expand(node)
        if self.phase != "network":
            return
        if node.get("base_P") is None:
            pr = node["prior"]
            can_add = pr.nS < SG.MAX_TERMS
            n_legal = int(pr.legal_add.sum()) if can_add else 0
            add_total = float(pr.p_add.sum())
            node["base_P"] = node["P"]
            node["low"] = np.flatnonzero(pr.legal_add & (pr.p_add < add_total / n_legal)) if n_legal and add_total > 0 \
                else np.zeros(0, dtype=np.int64)                 # vocabulary order: simplest first
            node["low_ptr"], node["add_total"] = 0, add_total
            node["low_set"] = {int(t) for t in node["low"]}
            if node is self.root:
                self.root_low = set(node["low_set"])
        low = node["low"]
        if not len(low):
            return
        tried = [m for m in node["tried"] if m in node["low_set"]]   # low-rated adds tried here, by any rule
        for m in tried:
            self.low_tried.add(m)
        ptr = node["low_ptr"]
        done = set(tried)
        while ptr < len(low) and int(low[ptr]) in done:
            ptr += 1
        node["low_ptr"] = ptr
        self.low_depth = max(self.low_depth, ptr)
        if not self.rho:
            return
        if ptr < len(low) and self.xrng.random() < self.rho:     # this visit tries the next unfamiliar term
            P = node["base_P"].copy()
            P[int(low[ptr])] = 1e9
            node["P"] = P
        else:
            node["P"] = node["base_P"]


def search_system_x(bursts, prior_fn, budget=1000, thorough_cap=200, time_cap=float("inf"), seed=1, ctx=None, probes=None,
                    share=0.0, rho=0.0):
    """search_g1.search_system with ExploringSearch (module docstring). Per equation, besides search_g1's fields: the
    exploration counts and the time split."""
    t0 = time.perf_counter()
    ctx = ctx or SG.system_context(bursts, seed=seed)
    setup = time.perf_counter() - t0
    fitter = FG.make_fitter(ctx, seed=seed)
    out = {"setup_seconds": setup, "vocabulary": ctx["vocab"].V, "equations": []}
    d = ctx["W"].shape[1]
    vocab = ctx["vocab"]
    for i in range(d):
        rng = np.random.default_rng([seed, i])
        ci = ctx
        if probes is not None:
            ci = dict(ctx, probe=[vocab.index[k] for k in probes[i] if k in vocab.index])
        se = ExploringSearch(i, ci, budget, prior_fn, fitter, rng, share=share, rho=rho, time_cap=time_cap / d,
                             thorough_cap=thorough_cap).run()
        sols = se.solutions()
        eff = dict(se.eff)
        eff["low_tried"] = len(se.low_tried)
        eff["low_depth"] = se.low_depth
        nodes = [term_nodes(vocab.keys[t], d) for t in se.low_tried]
        eff["low_max_nodes"] = max([n for n in nodes if n is not None], default=0)
        rec = {"eq": i, "effort": eff, "n_exact": len(sols)}
        if sols:
            b = sols[0]
            rec["solution"] = {"keys": list(b["keys"]), "thetas": [list(t.theta) for t in b["terms"]],
                               "coefs": [float(c) for c in b["coefs"]], "rel_v": b["rel_v"], "rel_t": b["rel_t"], "k": b["k"]}
            if se.root_low is not None:                          # solution terms the network rated below even at the root
                rec["solution_low_terms"] = sum(1 for k in b["keys"] if vocab.index.get(k) in se.root_low)
        out["equations"].append(rec)
    out["fitter_stats"] = dict(fitter.stats)
    out["seconds"] = time.perf_counter() - t0
    return out
