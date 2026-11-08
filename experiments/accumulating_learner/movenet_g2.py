"""Stage 2a of the general learner (research log, 7 November): the guiding network for up to three variables
(four from Stage 2a'': the role "other3", appended to the tokens; load_net; balanced dreams in train).

movenet_g1's network (Trial 28's move network with a term encoder), with the Stage 2a changes:
  - variables are read relative to the equation: its own variable, then the others in index order ("other1",
    "other2"); training examples are relabelled at random (dream_g2), so the network does not depend on the order;
  - a per-term scorer (a small network reading each term's encoding and its evidence) ranks every term of the system's
    vocabulary; attention covers the terms in the state plus the scorer's top SHORTLIST; the remaining terms share one
    "rest" slot whose probability is spread evenly over them. The shortlist only limits what the network attends to:
    every term keeps a nonzero prior, and the search's random exploration share draws from every legal move (owner,
    7 November);
  - a probe that never influences the search records, at every expansion, whether a missing true term was outside the
    attended set (the true structure is used only for this count).
Training: Trial 28's multi-positive loss, for the scorer over every stored term, and for the attention part over the
state, the scorer's top SHORTLIST and the rest slot.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

import fitter_g1 as FG
import language_g1 as LG
import search_g1 as SG

SHORTLIST = 768
MAX_TOK = 32
N_EVID, N_STATE = 6, 5
NUMBERS = ["-1", "2", "1/2", "3", "-2", "-3", "-1/2", "1", "4", "1/4", "6", "3/2", "9"]
TOKENS = ["<pad>", "own", "other1", "other2", "K", "Add", "Mul", "Pow", "exp", "log", "sin", "Abs", "<num>"] + \
         [f"n{v}" for v in NUMBERS] + ["other3"]                # other3 appended for four variables (Stage 2a'')
TOK = {t: k for k, t in enumerate(TOKENS)}


def term_tokens(key, own, d):
    """Prefix tokens of a canonical term for the equation of variable x_{own+1}: own variable, then the others in
    index order (n-ary sums and products written as nested binary ones)."""
    X = LG.variables(4)
    others = [j for j in range(d) if j != own]
    out = []

    def walk(e):
        if e.is_Symbol:
            if e in X:
                j = X.index(e)
                out.append(TOK["own"] if j == own else TOK[f"other{others.index(j) + 1}"])
            else:
                out.append(TOK["K"])
            return
        if e.is_Number or e.is_NumberSymbol:
            out.append(TOK.get(f"n{e}", TOK["<num>"]))
            return
        name = type(e).__name__
        if e.is_Add or e.is_Mul:
            args = list(e.args)
            out.extend([TOK[name]] * (len(args) - 1))
            for a in args:
                walk(a)
            return
        out.append(TOK.get(name, TOK["<num>"]))
        for a in e.args:
            walk(a)
    walk(LG.expr_of(key))
    return out[:MAX_TOK]


def token_table(order, d):
    T = np.zeros((len(order), d, MAX_TOK), dtype=np.int64)
    for g, key in enumerate(order):
        for own in range(d):
            tk = term_tokens(key, own, d)
            T[g, own, :len(tk)] = tk
    return T


class TermEncoder(nn.Module):
    def __init__(self, width=64, out=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(len(TOKENS), width, padding_idx=0)
        self.pos = nn.Embedding(MAX_TOK, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out))

    def forward(self, tok):
        pad = tok == 0
        h = self.emb(tok) + self.pos(torch.arange(tok.shape[1], device=tok.device))[None]
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        return self.out((h * keep).sum(1) / keep.sum(1).clamp(min=1.0))


class MoveNetL2(nn.Module):
    def __init__(self, d_model=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.terms = TermEncoder(out=d_model)
        self.scorer = nn.Sequential(nn.Linear(d_model + N_EVID + N_STATE, d_model), nn.GELU(), nn.Linear(d_model, d_model),
                                    nn.GELU(), nn.Linear(d_model, 1))
        self.inp = nn.Sequential(nn.Linear(N_EVID + N_STATE, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.rest = nn.Parameter(torch.zeros(d_model))
        self.rest_inp = nn.Linear(N_STATE + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.float()

    def score(self, E, F, st):
        stx = st.unsqueeze(-2).expand(*F.shape[:-1], N_STATE)
        return self.scorer(torch.cat([E, F, stx], -1)).squeeze(-1)

    def rerank(self, E, F, st, n_rest, pad):
        """Attention over the attended terms (B, N) plus the rest slot -> (add (B, N), remove (B, N), rest add (B,))."""
        stx = st.unsqueeze(1).expand(-1, F.shape[1], -1)
        h = E + self.inp(torch.cat([F, stx], -1))
        r = self.rest + self.rest_inp(torch.cat([st, torch.log1p(n_rest).unsqueeze(-1) / 10.0], -1))
        h = torch.cat([r.unsqueeze(1), h], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        o = self.head(self.enc(h, src_key_padding_mask=pad))
        return o[:, 1:, 0], o[:, 1:, 1], o[:, 0, 0]


def state_features(rel, nS, d, M, floor_rel):
    return np.array([math.log10(max(rel, 1e-16)) / 14, nS / SG.MAX_TERMS, d / 4, M / 16, math.log10(max(floor_rel, 1e-16)) / 14],
                    dtype=np.float32)


def evidence_features(gain, g0, inS, rem_full):
    rem_full = np.clip(np.nan_to_num(np.asarray(rem_full, dtype=np.float64), nan=0.0, posinf=1e30), 0.0, 1e30)
    gain, g0 = np.nan_to_num(gain, nan=0.0), np.nan_to_num(g0, nan=0.0)
    return np.stack([inS.astype(np.float32), gain, -np.log10(np.clip(1 - gain, 1e-12, 1)) / 12, g0,
                     -np.log10(np.clip(1 - g0, 1e-12, 1)) / 12, np.log10(1 + rem_full) / 12], 1).astype(np.float32)


def node_features(node, ctx, g0):
    vocab, i, y = ctx["vocab"], ctx["i"], ctx["y"]
    gain, rem = SG.evidence(node, ctx)
    idx = list(node["idx"])
    inS = np.zeros(vocab.V, dtype=bool)
    inS[idx] = True
    rem_full = np.zeros(vocab.V, dtype=np.float32)
    rem_full[idx] = rem
    yn = float(y.norm()) + 1e-300
    _, _, floor, _ = FG.equation_targets(ctx, i)
    rel = node["res"] / yn if math.isfinite(node["res"]) else 1.0
    st = state_features(rel, len(idx), ctx["W"].shape[1], ctx.get("M", 8), floor / yn)
    F = evidence_features(gain.astype(np.float32), g0.astype(np.float32), inS, rem_full)
    return np.nan_to_num(F, nan=0.0, posinf=2.5, neginf=0.0), np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=-1.2), inS


class NetworkPrior:
    """search_g1 prior from the network (CPU inference). emb[d]: (V_d, d, d_model) encodings of the search vocabulary
    for d variables (encode_vocabulary). Probe counts (missing true terms outside the attended set) in .probe."""

    def __init__(self, net, emb):
        self.net = net.eval()
        self.emb = {d: torch.as_tensor(e, dtype=torch.float32) for d, e in emb.items()}
        self.seconds, self.calls = 0.0, 0
        self.probe = {"expansions": 0, "with_missing_true": 0, "cut": 0, "root_with_missing_true": 0, "root_cut": 0}
        self._g0 = {}

    def __call__(self, node, ctx):
        t0 = time.perf_counter()
        vocab, i = ctx["vocab"], ctx["i"]
        d = ctx["W"].shape[1]
        key = (id(vocab), i)
        if key not in self._g0:
            self._g0[key] = SG.evidence({"keys": (), "terms": [], "idx": ()}, ctx)[0]
        F, st, inS = node_features(node, ctx, self._g0[key])
        V = vocab.V
        E = self.emb[d][torch.as_tensor(vocab.gid), i]
        Ft, stt = torch.as_tensor(F), torch.as_tensor(st)
        idx = list(node["idx"])
        with torch.no_grad():
            s0 = self.net.score(E, Ft, stt).double().numpy()
            s0[inS] = -np.inf
            n_short = min(SHORTLIST, V - len(idx))
            short = np.argpartition(-s0, n_short - 1)[:n_short] if n_short > 0 else np.zeros(0, dtype=np.int64)
            att = np.concatenate([np.asarray(idx, dtype=np.int64), short])
            n_rest = V - len(att)
            a, r, ar = self.net.rerank(E[att][None], Ft[att][None], stt[None], torch.tensor([float(n_rest)], dtype=torch.float32),
                                       torch.zeros(1, len(att), dtype=torch.bool))
        a, r, ar = a[0].double().numpy(), r[0].double().numpy(), float(ar[0])
        add = np.full(V, ar - math.log(max(n_rest, 1)))
        add[short] = a[len(idx):]
        prior = SG.Prior(V, node["idx"], add, r[:len(idx)])
        probe = ctx.get("probe")                               # counting only: never used by the search
        if probe:
            missing = [t for t in probe if t not in set(idx)]
            if missing:
                attended = set(att.tolist())
                cut = any(t not in attended for t in missing)
                self.probe["with_missing_true"] += 1
                self.probe["cut"] += cut
                if not idx:
                    self.probe["root_with_missing_true"] += 1
                    self.probe["root_cut"] += cut
        self.probe["expansions"] += 1
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return prior


def load_net(path):
    """A saved network. Networks saved before the fourth variable role existed (Stage 2a) get its embedding row as
    the mean of the "other1" and "other2" rows (research log, Stage 2a'' proposal); nothing else changes."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    w = sd["terms.emb.weight"]
    if w.shape[0] == len(TOKENS) - 1:
        sd["terms.emb.weight"] = torch.cat([w, 0.5 * (w[TOK["other1"]] + w[TOK["other2"]])[None]], 0)
    net = MoveNetL2()
    net.load_state_dict(sd)
    return net


def encode_vocabulary(net, table, device="cpu", batch=4096):
    net = net.to(device).eval()
    V, d, L = table.shape
    flat = torch.as_tensor(table.reshape(V * d, L))
    out = []
    with torch.no_grad():
        for s in range(0, len(flat), batch):
            out.append(net.terms(flat[s:s + batch].to(device)).float().cpu())
    return torch.cat(out).reshape(V, d, -1).numpy()


# ---------------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------------

def collate(examples, device):
    N = max(len(e["gids"]) for e in examples)
    B = len(examples)
    gids = torch.full((B, N), -1, dtype=torch.long)
    F = torch.zeros(B, N, N_EVID, dtype=torch.float32)
    ta = torch.zeros(B, N, dtype=torch.bool)
    tr = torch.zeros(B, N, dtype=torch.bool)
    ins = torch.zeros(B, N, dtype=torch.bool)
    st = torch.zeros(B, N_STATE, dtype=torch.float32)
    own = torch.zeros(B, dtype=torch.long)
    dim = torch.zeros(B, dtype=torch.long)
    for b, e in enumerate(examples):
        n = len(e["gids"])
        gids[b, :n] = torch.as_tensor(e["gids"].astype(np.int64))
        F[b, :n] = torch.nan_to_num(torch.as_tensor(e["F"].astype(np.float32)), nan=0.0, posinf=2.5, neginf=0.0)
        ta[b, :n] = torch.as_tensor(e["add"])
        tr[b, :n] = torch.as_tensor(e["rem"])
        ins[b, :n] = torch.as_tensor(e["inS"])
        st[b] = torch.nan_to_num(torch.as_tensor(e["state"]), nan=0.0, posinf=0.0, neginf=-1.2)
        own[b], dim[b] = int(e["eq"]), int(e["d"])
    return [x.to(device) for x in (gids, F, ta, tr, ins, st, own, dim)]


def loss_fn(net, batch, tables):
    """tables[d]: (V_d, d, MAX_TOK) token ids (torch, on the device). Every example in a batch has the same d."""
    gids, F, add, rem, inS, st, own, dim = batch
    B, N = gids.shape
    d = int(dim[0])
    pad = gids < 0
    big = -1e9
    pairs = torch.stack([gids.clamp(min=0), own.unsqueeze(1).expand(-1, N)], -1).reshape(-1, 2)
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    E = net.terms(tables[d][uniq[:, 0], uniq[:, 1]])[inv].reshape(B, N, -1)
    add, rem = add & ~inS & ~pad, rem & inS & ~pad
    has_a, has_r = add.any(-1), rem.any(-1)
    s0 = net.score(E, F, st).masked_fill(pad | inS, big)
    ls0 = torch.log_softmax(s0, -1)
    l_score = -(torch.logsumexp(ls0.masked_fill(~add, big), -1)[has_a]).mean() if has_a.any() else s0.sum() * 0
    with torch.no_grad():
        k = min(SHORTLIST, N)
        sel = torch.zeros(B, N, dtype=torch.bool, device=gids.device)
        sel.scatter_(1, torch.topk(s0, k, dim=-1).indices, True)
        att = (sel & ~pad & ~inS) | inS
        order = torch.argsort((~att).float(), dim=1, stable=True)
        n_att = int(att.sum(1).max())
        idx = order[:, :n_att]
    ga = lambda x: torch.gather(x, 1, idx) if x.dim() == 2 else torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    apad = ~ga(att)
    n_rest = ((~pad).sum(1).float() - att.sum(1).float()).clamp(min=1.0)
    a, r, ar = net.rerank(ga(E), ga(F), st, n_rest, apad)
    ainS, aadd, arem = ga(inS) & ~apad, ga(add) & ~apad, ga(rem) & ~apad
    lsa = torch.log_softmax(torch.cat([ar.unsqueeze(1), a.masked_fill(ainS | apad, big)], 1), -1)
    n_out = (add & ~att).sum(1).float()
    mass_in = torch.logsumexp(lsa[:, 1:].masked_fill(~aadd, big), -1)
    mass_rest = torch.where(n_out > 0, lsa[:, 0] + torch.log(n_out.clamp(min=1.0)) - torch.log(n_rest), torch.full_like(n_out, big))
    mass = torch.logsumexp(torch.stack([mass_in, mass_rest], 1), 1)
    l_add = -(mass[has_a]).mean() if has_a.any() else a.sum() * 0
    lsr = torch.log_softmax(r.masked_fill(~ainS, big), -1)
    l_rem = -(torch.logsumexp(lsr.masked_fill(~arem, big), -1)[has_r]).mean() if has_r.any() else r.sum() * 0
    with torch.no_grad():
        in_short = ((add & att).any(-1) | ~has_a)[has_a].float().mean() if has_a.any() else torch.tensor(1.0)
    return l_score + l_add + l_rem, {"score": float(l_score), "add": float(l_add), "remove": float(l_rem),
                                     "true_add_in_shortlist": float(in_short)}


def train(net, examples, tables_np, steps, batch=32, lr=3e-4, device="cpu", seed=0, log_every=200, groups=None, share=0.5):
    """Train further for `steps` steps; every batch is drawn from examples of one dimension (in proportion to their
    numbers). Non-finite steps are skipped and counted. Balanced dreams (Stage 2a''): with `groups` (per example, 1 for
    the newest domain's examples, 0 for everything earlier) a batch comes from group 1 with probability `share` (from
    the other group when one is empty), then from one dimension within the group in proportion to its numbers."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device).train()
    tables = {d: torch.as_tensor(t).to(device) for d, t in tables_np.items()}
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
        loss, parts = loss_fn(net, collate([examples[pool[k]] for k in pick], device), tables)
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
