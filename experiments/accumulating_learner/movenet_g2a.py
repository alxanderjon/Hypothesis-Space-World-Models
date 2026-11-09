"""Stage A of the redesign (owner, 9 November): the move network with the learned grammar (grammar_g2) as its base.

The network corrects the grammar per system instead of replacing it: every term's add score is the network's output
plus L(t) = log P_grammar(t) + log V (zero for the uniform prior, so an empty grammar changes nothing):
  - the scorer that picks the attended shortlist ranks by its score + L;
  - an attended term's add logit is the attention head's logit + L;
  - the rest slot carries the grammar's mass outside the attended set: its logit is the head's rest logit + log of the
    mean of exp(L) over the rest, and the rest's probability is spread over its terms in proportion to the grammar
    instead of evenly.
Training (the review rounds) uses the same sums, so the network learns the correction to the grammar (a product of
experts). A network trained without the grammar (the history networks, before their first Stage A review round) keeps
its own scores; for it the grammar only spreads the rest slot's probability (the rest's total is unchanged).
Removal moves are unchanged.

Copies of movenet_g2's loss, training loop and prior with the grammar added, kept in their own module so the claim run
in progress (which re-imports movenet_g2 for every review round) is untouched; with no grammar they compute exactly
what movenet_g2 computes.
"""

import math
import time

import numpy as np
import torch

import movenet_g2 as MN
import search_g1 as SG


def loss_fn(net, batch, tables, lg=None):
    """movenet_g2.loss_fn with the grammar's log ratios lg[d] ((V_d, d), on the device) added (module docstring)."""
    gids, F, add, rem, inS, st, own, dim = batch
    B, N = gids.shape
    d = int(dim[0])
    pad = gids < 0
    big = -1e9
    pairs = torch.stack([gids.clamp(min=0), own.unsqueeze(1).expand(-1, N)], -1).reshape(-1, 2)
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    E = net.terms(tables[d][uniq[:, 0], uniq[:, 1]])[inv].reshape(B, N, -1)
    L = lg[d][gids.clamp(min=0), own.unsqueeze(1).expand(-1, N)] if lg is not None else torch.zeros(B, N, device=gids.device)
    add, rem = add & ~inS & ~pad, rem & inS & ~pad
    has_a, has_r = add.any(-1), rem.any(-1)
    s0 = (net.score(E, F, st) + L).masked_fill(pad | inS, big)
    ls0 = torch.log_softmax(s0, -1)
    l_score = -(torch.logsumexp(ls0.masked_fill(~add, big), -1)[has_a]).mean() if has_a.any() else s0.sum() * 0
    with torch.no_grad():
        k = min(MN.SHORTLIST, N)
        sel = torch.zeros(B, N, dtype=torch.bool, device=gids.device)
        sel.scatter_(1, torch.topk(s0, k, dim=-1).indices, True)
        att = (sel & ~pad & ~inS) | inS
        order = torch.argsort((~att).float(), dim=1, stable=True)
        n_att = int(att.sum(1).max())
        idx = order[:, :n_att]
    ga = lambda x: torch.gather(x, 1, idx) if x.dim() == 2 else torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    apad = ~ga(att)
    rest = ~pad & ~att
    n_rest = rest.sum(1).float().clamp(min=1.0)
    a, r, ar = net.rerank(ga(E), ga(F), st, n_rest, apad)
    lse_rest = torch.logsumexp(L.masked_fill(~rest, big), -1)
    a = a + ga(L)
    ar = ar + torch.where(rest.any(-1), lse_rest - torch.log(n_rest), torch.zeros_like(lse_rest))   # unchanged when no rest
    ainS, aadd, arem = ga(inS) & ~apad, ga(add) & ~apad, ga(rem) & ~apad
    lsa = torch.log_softmax(torch.cat([ar.unsqueeze(1), a.masked_fill(ainS | apad, big)], 1), -1)
    out_true = add & ~att
    mass_in = torch.logsumexp(lsa[:, 1:].masked_fill(~aadd, big), -1)
    mass_rest = torch.where(out_true.any(-1), lsa[:, 0] + torch.logsumexp(L.masked_fill(~out_true, big), -1) - lse_rest,
                            torch.full_like(lse_rest, big))
    mass = torch.logsumexp(torch.stack([mass_in, mass_rest], 1), 1)
    l_add = -(mass[has_a]).mean() if has_a.any() else a.sum() * 0
    lsr = torch.log_softmax(r.masked_fill(~ainS, big), -1)
    l_rem = -(torch.logsumexp(lsr.masked_fill(~arem, big), -1)[has_r]).mean() if has_r.any() else r.sum() * 0
    with torch.no_grad():
        in_short = ((add & att).any(-1) | ~has_a)[has_a].float().mean() if has_a.any() else torch.tensor(1.0)
    return l_score + l_add + l_rem, {"score": float(l_score), "add": float(l_add), "remove": float(l_rem),
                                     "true_add_in_shortlist": float(in_short)}


def train(net, examples, tables_np, steps, batch=32, lr=3e-4, device="cpu", seed=0, log_every=200, groups=None, share=0.5,
          grammar_tables=None):
    """movenet_g2.train with the grammar (grammar_tables[d]: (V_d, d) log ratios, grammar_g2.Grammar.log_ratio)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device).train()
    tables = {d: torch.as_tensor(t).to(device) for d, t in tables_np.items()}
    lg = {d: torch.as_tensor(t, dtype=torch.float32).to(device) for d, t in grammar_tables.items()} if grammar_tables else None
    by_g = {}
    for k, e in enumerate(examples):
        g = 0 if groups is None else int(groups[k])
        by_g.setdefault(g, {}).setdefault(int(e["d"]), []).append(k)
    choose = {}
    for g, by_d in by_g.items():
        dims = sorted(by_d)
        w = np.array([len(by_d[d]) for d in dims], dtype=float)
        choose[g] = (dims, w / w.sum())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    log, t0, skipped = [], time.time(), 0
    for step in range(steps):
        g = 0
        if groups is not None:
            g = 1 if rng.random() < share else 0
            g = g if g in by_g else 1 - g
        dims, w = choose[g]
        d = dims[int(rng.choice(len(dims), p=w))]
        pool = by_g[g][d]
        pick = rng.choice(len(pool), size=min(batch, len(pool)), replace=False)
        loss, parts = loss_fn(net, MN.collate([examples[pool[k]] for k in pick], device), tables, lg)
        opt.zero_grad()
        if not bool(torch.isfinite(loss)):
            skipped += 1
            continue
        loss.backward()
        if not all(bool(torch.isfinite(q.grad).all()) for q in net.parameters() if q.grad is not None):
            skipped += 1
            opt.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            log.append({"step": step, "d": d, "loss": float(loss), **parts, "skipped": skipped, "seconds": round(time.time() - t0, 1)})
    net.eval()
    return log


class GrammarPrior(MN.NetworkPrior):
    """movenet_g2.NetworkPrior with the grammar (module docstring). trained: the network was trained with it."""

    def __init__(self, net, emb, shortlist=None, grammar=None, trained=False):
        super().__init__(net, emb, shortlist)
        self.grammar, self.trained = grammar, bool(trained)
        self._lr = {}

    def __call__(self, node, ctx):
        t0 = time.perf_counter()
        vocab, i = ctx["vocab"], ctx["i"]
        d = ctx["W"].shape[1]
        key = (id(vocab), i)
        if key not in self._g0:
            self._g0[key] = SG.evidence({"keys": (), "terms": [], "idx": ()}, ctx)[0]
        if key not in self._lr:
            self._lr[key] = self.grammar.log_ratio(d)[vocab.gid, i].astype(np.float64) if self.grammar is not None else None
        L = self._lr[key]
        F, st, inS = MN.node_features(node, ctx, self._g0[key])
        V = vocab.V
        E = self.emb[d][torch.as_tensor(vocab.gid), i]
        Ft, stt = torch.as_tensor(F), torch.as_tensor(st)
        idx = list(node["idx"])
        with torch.no_grad():
            s0 = self.net.score(E, Ft, stt).double().numpy()
            if L is not None and self.trained:
                s0 = s0 + L
            s0[inS] = -np.inf
            n_short = min(self.shortlist, V - len(idx))
            short = np.argpartition(-s0, n_short - 1)[:n_short] if n_short > 0 else np.zeros(0, dtype=np.int64)
            att = np.concatenate([np.asarray(idx, dtype=np.int64), short])
            n_rest = V - len(att)
            a, r, ar = self.net.rerank(E[att][None], Ft[att][None], stt[None], torch.tensor([float(n_rest)], dtype=torch.float32),
                                       torch.zeros(1, len(att), dtype=torch.bool))
        a, r, ar = a[0].double().numpy(), r[0].double().numpy(), float(ar[0])
        if L is None:
            add = np.full(V, ar - math.log(max(n_rest, 1)))
            add[short] = a[len(idx):]
        elif self.trained:
            add = ar - math.log(max(n_rest, 1)) + L
            add[short] = a[len(idx):] + L[short]
        else:
            restm = np.ones(V, dtype=bool)
            restm[att] = False
            lse = float(np.logaddexp.reduce(L[restm])) if restm.any() else 0.0
            add = ar + L - lse
            add[short] = a[len(idx):]
        prior = SG.Prior(V, node["idx"], add, r[:len(idx)])
        probe = ctx.get("probe")                               # counting only: never used by the search
        pe = self.probe_eq.setdefault(int(i), {"expansions": 0, "with_missing_true": 0, "cut": 0, "root_with_missing_true": 0,
                                               "root_cut": 0})
        if probe:
            missing = [t for t in probe if t not in set(idx)]
            if missing:
                attended = set(att.tolist())
                cut = any(t not in attended for t in missing)
                for q in (self.probe, pe):
                    q["with_missing_true"] += 1
                    q["cut"] += cut
                    if not idx:
                        q["root_with_missing_true"] += 1
                        q["root_cut"] += cut
        self.probe["expansions"] += 1
        pe["expansions"] += 1
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return prior
