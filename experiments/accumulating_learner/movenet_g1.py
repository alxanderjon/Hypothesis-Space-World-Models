"""Stage 1 of the general learner (research log, 6 November): the guiding network on L1 terms.

One network for all domains (no per-domain parts, no domain label). Trial 28's move network (movenet28.MoveNet:
attention across every candidate term of the system, add and remove scores per term), with each term described by what
it is: a small encoder reads the term's expression in prefix form (operations; the equation's own variable and the other
variable; fitted constants; numbers), instead of a toolkit slot number, so the same term has the same description in
every domain. Every term of the search vocabulary that is defined on the system attends (at most 1,143); the Stage 1 rules'
shortlist of 512 was a compute measure for 10^3-10^4 terms (research log, 6 November).
Inputs from the data, per term, as in Trial 28: membership in the state; the best partial R^2 for being added given the
state's fitted terms, and alone; for terms in the state, the share of the residual their removal would add; and the
state's fit quality, size, the system's size, the number of bursts and the noise floor.
Outputs: add and remove scores, which become the search's priors (search_g1.Prior: add t, remove s, swap s -> t with
remove(s) + add(t), mixed 45% / 15% / 40%). Training: Trial 28's multi-positive loss (movenet28.loss_fn).
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

import fitter_g1 as FG
import language_g1 as LG
import search_g1 as SG

MAX_TOK = 32
N_EVID, N_STATE = 6, 5
NUMBERS = ["-1", "2", "1/2", "3", "-2", "-3", "-1/2", "1", "4", "1/4", "6", "3/2", "9"]
TOKENS = ["<pad>", "own", "other", "K", "Add", "Mul", "Pow", "exp", "log", "sin", "Abs", "<num>"] + [f"n{v}" for v in NUMBERS]
TOK = {t: k for k, t in enumerate(TOKENS)}


# ---------------------------------------------------------------------------------------------------------------------
# term tokens
# ---------------------------------------------------------------------------------------------------------------------

def term_tokens(key, own, d=2):
    """Prefix tokens of a canonical term for an equation whose own variable is x_{own+1} (n-ary sums and products
    written as nested binary ones)."""
    X = LG.variables(d)
    out = []

    def walk(e):
        if e.is_Symbol:
            out.append((TOK["own"] if X.index(e) == own else TOK["other"]) if e in X else TOK["K"])
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


def token_table(order, d=2):
    """(V, d, MAX_TOK) token ids of every vocabulary term for each choice of own variable."""
    T = np.zeros((len(order), d, MAX_TOK), dtype=np.int64)
    for g, key in enumerate(order):
        for own in range(d):
            tk = term_tokens(key, own, d)
            T[g, own, :len(tk)] = tk
    return T


# ---------------------------------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------------------------------

class TermEncoder(nn.Module):
    def __init__(self, width=64, out=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(len(TOKENS), width, padding_idx=0)
        self.pos = nn.Embedding(MAX_TOK, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out))

    def forward(self, tok):
        """tok (N, MAX_TOK) -> (N, out)."""
        pad = tok == 0
        h = self.emb(tok) + self.pos(torch.arange(tok.shape[1], device=tok.device))[None]
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        return self.out((h * keep).sum(1) / keep.sum(1).clamp(min=1.0))


class MoveNetL1(nn.Module):
    """movenet28.MoveNet with the term encoder in place of the family embedding."""

    def __init__(self, d_model=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.terms = TermEncoder(out=d_model)
        self.inp = nn.Sequential(nn.Linear(N_EVID + N_STATE, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.float()

    def forward(self, E, F, st, pad):
        """E (B, N, d_model) term encodings, F (B, N, N_EVID), st (B, N_STATE), pad (B, N) -> (B, N, 2) (add, remove)."""
        stx = st.unsqueeze(1).expand(-1, F.shape[1], -1)
        h = E + self.inp(torch.cat([F, stx], -1))
        return self.out(self.enc(h, src_key_padding_mask=pad))


def state_features(rel, nS, d, M, floor_rel):
    return np.array([math.log10(max(rel, 1e-16)) / 14, nS / SG.MAX_TERMS, d / 4, M / 16, math.log10(max(floor_rel, 1e-16)) / 14],
                    dtype=np.float32)


def evidence_features(gain, g0, inS, rem_full):
    """(V, N_EVID): membership, gain given the state and its -log10(1 - .) / 12, gain alone and its transform, removal
    share (terms in the state)."""
    rem_full = np.clip(np.nan_to_num(np.asarray(rem_full, dtype=np.float64), nan=0.0, posinf=1e30), 0.0, 1e30)
    gain, g0 = np.nan_to_num(gain, nan=0.0), np.nan_to_num(g0, nan=0.0)
    return np.stack([inS.astype(np.float32), gain, -np.log10(np.clip(1 - gain, 1e-12, 1)) / 12, g0,
                     -np.log10(np.clip(1 - g0, 1e-12, 1)) / 12, np.log10(1 + rem_full) / 12], 1).astype(np.float32)


def node_features(node, ctx, g0):
    """(evidence features (V, N_EVID), state features, membership mask) at a search node."""
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
    return evidence_features(gain.astype(np.float32), g0.astype(np.float32), inS, rem_full), st, inS


# ---------------------------------------------------------------------------------------------------------------------
# the prior for the search
# ---------------------------------------------------------------------------------------------------------------------

class NetworkPrior:
    """search_g1 prior from the network (CPU inference, one state at a time). emb: (V_global, d, d_model) term
    encodings of the whole search vocabulary (encode_vocabulary). Time accumulates in .seconds."""

    def __init__(self, net, emb):
        self.net = net.eval()
        self.emb = torch.as_tensor(emb, dtype=torch.float32)
        self.seconds, self.calls = 0.0, 0
        self._g0 = {}

    def __call__(self, node, ctx):
        t0 = time.perf_counter()
        vocab, i = ctx["vocab"], ctx["i"]
        key = (id(vocab), i)
        if key not in self._g0:
            self._g0[key] = SG.evidence({"keys": (), "terms": [], "idx": ()}, ctx)[0]
        F, st, inS = node_features(node, ctx, self._g0[key])
        F, st = np.nan_to_num(F, nan=0.0, posinf=2.5, neginf=0.0), np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=-1.2)
        E = self.emb[torch.as_tensor(vocab.gid), i]
        with torch.no_grad():
            o = self.net(E[None], torch.as_tensor(F)[None], torch.as_tensor(st)[None],
                         torch.zeros(1, vocab.V, dtype=torch.bool))[0].double().numpy()
        prior = SG.Prior(vocab.V, node["idx"], o[:, 0], o[list(node["idx"]), 1])
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return prior


def encode_vocabulary(net, table, device="cpu", batch=4096):
    """(V, d, d_model) encodings of every vocabulary term for each own-variable choice."""
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
    """Pad stored examples (dream_g1.make_examples) to tensors: gids (B, N) (-1 padding), evidence, add and remove
    targets, membership, state, own variable."""
    N = max(len(e["gids"]) for e in examples)
    B = len(examples)
    gids = torch.full((B, N), -1, dtype=torch.long)
    F = torch.zeros(B, N, N_EVID, dtype=torch.float32)
    ta = torch.zeros(B, N, dtype=torch.bool)
    tr = torch.zeros(B, N, dtype=torch.bool)
    ins = torch.zeros(B, N, dtype=torch.bool)
    st = torch.zeros(B, N_STATE, dtype=torch.float32)
    own = torch.zeros(B, dtype=torch.long)
    for b, e in enumerate(examples):
        n = len(e["gids"])
        gids[b, :n] = torch.as_tensor(e["gids"].astype(np.int64))
        F[b, :n] = torch.nan_to_num(torch.as_tensor(e["F"].astype(np.float32)), nan=0.0, posinf=2.5, neginf=0.0)
        ta[b, :n] = torch.as_tensor(e["add"])
        tr[b, :n] = torch.as_tensor(e["rem"])
        ins[b, :n] = torch.as_tensor(e["inS"])
        st[b] = torch.as_tensor(e["state"])
        own[b] = int(e["eq"])
    return [x.to(device) for x in (gids, F, ta, tr, ins, st, own)]


def loss_fn(net, batch, table_t):
    """Trial 28's multi-positive cross-entropy: -log of the probability mass on the correct additions (over terms not in
    the state) plus the same for removals (over terms in the state), for examples that have any."""
    gids, F, add, rem, inS, st, own = batch
    B, N = gids.shape
    pad = gids < 0
    pairs = torch.stack([gids.clamp(min=0), own.unsqueeze(1).expand(-1, N)], -1).reshape(-1, 2)
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    E = net.terms(table_t[uniq[:, 0], uniq[:, 1]])[inv].reshape(B, N, -1)
    logits = net(E, F, st, pad)
    big = torch.finfo(logits.dtype).min / 4
    la = logits[..., 0].masked_fill(inS | pad, big)
    lr = logits[..., 1].masked_fill(~inS | pad, big)
    lsa, lsr = torch.log_softmax(la, -1), torch.log_softmax(lr, -1)
    add, rem = add & ~inS & ~pad, rem & inS & ~pad
    has_a, has_r = add.any(-1), rem.any(-1)
    pa = torch.logsumexp(lsa.masked_fill(~add, big), -1)
    pr = torch.logsumexp(lsr.masked_fill(~rem, big), -1)
    la_ = -(pa[has_a]).mean() if has_a.any() else logits.sum() * 0
    lr_ = -(pr[has_r]).mean() if has_r.any() else logits.sum() * 0
    with torch.no_grad():                                      # top-5 add hit rate (Trial 28's ranking measure, adds only)
        top = torch.topk(la, min(5, N), dim=-1).indices
        hit = torch.gather(add, 1, top).any(-1)[has_a].float().mean() if has_a.any() else torch.tensor(0.0)
    return la_ + lr_, {"add": float(la_), "remove": float(lr_), "top5_add": float(hit)}


def train(net, examples, table, steps, batch=32, lr=3e-4, device="cpu", seed=0, log_every=200):
    """Train `net` further for `steps` steps on uniformly drawn batches of `examples`. Returns the log."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device).train()
    table_t = torch.as_tensor(table).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    log, t0, skipped = [], time.time(), 0
    for step in range(steps):
        pick = rng.choice(len(examples), size=min(batch, len(examples)), replace=False)
        loss, parts = loss_fn(net, collate([examples[k] for k in pick], device), table_t)
        opt.zero_grad()
        if not bool(torch.isfinite(loss)):                     # a non-finite loss never reaches the weights
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
            log.append({"step": step, "loss": float(loss), **parts, "skipped": skipped, "seconds": round(time.time() - t0, 1)})
    net.eval()
    return log
