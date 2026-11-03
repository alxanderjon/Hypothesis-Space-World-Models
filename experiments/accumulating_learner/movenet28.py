"""Trial 28 step 2: the move-ranking network (the with-history learner's priorities).

MoveNet reads one row of features per vocabulary token (history28.features) and gives two scores per token: for adding
it (tokens not in the state) and for removing it (tokens in the state). Trained on synthetic search states with the
true structure known: the loss raises the total probability of the correct additions among all additions, and of the
correct removals among all removals.

network_prior(net) builds a prior function for structure_search: the search's moves (add, remove, swap) get logits
add(t), remove(s), remove(s) + add(t), mixed by move type as for every prior; the search keeps its exploration share and
bonus. Its computing time is recorded separately (effort "network_seconds").
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

import history28 as HI
import structure_search as SS


class MoveNet(nn.Module):
    def __init__(self, n_feat=HI.N_FEAT, d_model=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.fam = nn.Embedding(len(HI.FAMILIES), d_model)
        self.inp = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.float()                                           # float32 regardless of the learner's float64 default

    def forward(self, F, fam, pad):
        """F (B, V, n_feat), fam (B, V), pad (B, V) True on padding -> (B, V, 2) logits (add, remove)."""
        h = self.inp(F) + self.fam(fam)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.out(h)


def loss_fn(logits, inS, add, rem, pad):
    """Multi-positive cross-entropy: -log of the probability mass on the correct additions (over tokens not in the
    state) plus the same for removals (over tokens in the state), for samples that have any."""
    big = torch.finfo(logits.dtype).min / 4
    la = logits[..., 0].masked_fill(inS | pad, big)
    lr = logits[..., 1].masked_fill(~inS | pad, big)
    lsa = torch.log_softmax(la, -1)
    lsr = torch.log_softmax(lr, -1)
    has_a, has_r = add.any(-1), rem.any(-1)
    pa = torch.logsumexp(lsa.masked_fill(~add, big), -1)
    pr = torch.logsumexp(lsr.masked_fill(~rem, big), -1)
    la_ = -(pa[has_a]).mean() if has_a.any() else logits.sum() * 0
    lr_ = -(pr[has_r]).mean() if has_r.any() else logits.sum() * 0
    return la_ + lr_, la_.detach(), lr_.detach()


def collate(batch, device):
    """Pad a list of examples (fam (V,), F (V, n_feat), add, rem, inS) to tensors."""
    Vm = max(len(b["fam"]) for b in batch)
    B = len(batch)
    F = torch.zeros(B, Vm, HI.N_FEAT, dtype=torch.float32)
    fam = torch.zeros(B, Vm, dtype=torch.long)
    pad = torch.ones(B, Vm, dtype=torch.bool)
    add = torch.zeros(B, Vm, dtype=torch.bool)
    rem = torch.zeros(B, Vm, dtype=torch.bool)
    inS = torch.zeros(B, Vm, dtype=torch.bool)
    for k, b in enumerate(batch):
        V = len(b["fam"])
        F[k, :V] = torch.as_tensor(b["F"], dtype=torch.float32)
        fam[k, :V] = torch.as_tensor(b["fam"])
        pad[k, :V] = False
        add[k, :V] = torch.as_tensor(b["add"])
        rem[k, :V] = torch.as_tensor(b["rem"])
        inS[k, :V] = torch.as_tensor(b["F"][:, HI.N_STATIC + HI.CODE_DIM] > 0.5)
    return [x.to(device) for x in (F, fam, pad, add, rem, inS)]


# ---------------------------------------------------------------------------------------------------------------------
# priorities for the search
# ---------------------------------------------------------------------------------------------------------------------

def move_logits(vocab_tokens, toks, scores):
    """The search's moves at state toks and their logits from per-token scores (V, 2): add t -> add(t), remove s ->
    remove(s), swap s -> t -> remove(s) + add(t). Returns (moves, [(kind, logit)])."""
    index = {t: k for k, t in enumerate(vocab_tokens)}
    inS = set(toks)
    moves, logit = [], []
    if len(toks) < SS.MAX_TERMS:
        for k, t in enumerate(vocab_tokens):
            if t not in inS:
                moves.append(("add", t, None)); logit.append(("add", float(scores[k, 0])))
    for s in toks:
        r = float(scores[index[s], 1])
        moves.append(("remove", s, None)); logit.append(("remove", r))
        for k, b in enumerate(vocab_tokens):
            if b not in inS:
                moves.append(("swap", s, b)); logit.append(("swap", r + float(scores[k, 0])))
    return moves, logit


class NetworkPrior:
    """Prior function for structure_search with the trained network (CPU inference, one state at a time). The time it
    takes (features and network) accumulates in .seconds, reported separately from the search's own effort."""

    needs_history_context = True

    def __init__(self, net, seed=0):
        self.net = net.eval()
        self.rng = np.random.default_rng(seed)
        self.seconds = 0.0
        self.calls = 0
        self._codes = {}
        self._ctx = {}

    def __call__(self, node, ctx):
        t0 = time.perf_counter()
        vocab, i = ctx["vocab"], ctx["i"]
        key = id(ctx["hctx"])
        if key not in self._codes:
            self._codes[key] = HI.random_codes(self.rng, vocab.d)
        hctx = ctx["hctx"]
        res = node["res"] if math.isfinite(node["res"]) else float(ctx["y"].norm())
        fam, F = HI.features(hctx, i, node["toks"], node["terms"], res, self._codes[key])
        with torch.no_grad():
            sc = self.net(torch.as_tensor(F, dtype=torch.float32)[None], torch.as_tensor(fam)[None],
                          torch.zeros(1, len(fam), dtype=torch.bool))[0].double().numpy()
        moves, logit = move_logits(vocab.tokens, node["toks"], sc)
        P = SS._mix(logit)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return moves, P


# ---------------------------------------------------------------------------------------------------------------------
# the ranking measure of the go/no-go bar (a)
# ---------------------------------------------------------------------------------------------------------------------

def move_priorities(scores, inS, max_terms=SS.MAX_TERMS, weights={"add": 0.45, "remove": 0.15, "swap": 0.40}):
    """Priorities of all legal moves from per-token scores (V, 2) and the state mask inS (V,), exactly as the search
    mixes them (SS._mix). Returns (priorities, kinds, add_token_index) for adds first, then removes, then swaps."""
    a, r = scores[:, 0], scores[:, 1]
    outs, ins = np.flatnonzero(~inS), np.flatnonzero(inS)
    parts, kinds, add_idx = [], [], []
    present = {}
    if len(ins) < max_terms and len(outs):
        present["add"] = a[outs]
    if len(ins):
        present["remove"] = r[ins]
        if len(outs):
            present["swap"] = (r[ins][:, None] + a[outs][None, :]).ravel()
    wsum = sum(weights[k] for k in present)
    for k in ("add", "remove", "swap"):
        if k in present:
            v = present[k]
            e = np.exp(v - v.max())
            parts.append(weights[k] / wsum * e / e.sum())
            kinds += [k] * len(v)
            add_idx += list(outs) if k == "add" else [-1] * len(v)
    return np.concatenate(parts), kinds, np.asarray(add_idx)


def rank_hit(scores, inS, add, k=5):
    """(hit, chance): whether a correct addition is among the k highest-priority moves, and the chance rate of that
    under a uniform order, 1 - C(N - c, k) / C(N, k)."""
    P, kinds, add_idx = move_priorities(scores, inS)
    N = len(P)
    correct = (add_idx >= 0) & np.isin(add_idx, np.flatnonzero(add))
    c = int(correct.sum())
    top = np.argsort(-P, kind="stable")[:k]
    hit = bool(correct[top].any())
    chance = (1.0 - math.comb(N - c, k) / math.comb(N, k)) if N >= k else float(c > 0)
    return hit, chance
