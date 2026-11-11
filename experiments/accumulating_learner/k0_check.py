"""K0 of the compositional plan (owner's approval, 11 November; research log "Owner: K0 approved"): where the development
systems' true new-form terms would rank under each lineage's operation grammar (grammar_ops), against their ranks now
(the lineage's network at the root, Stage A's shape grammar, the language's simplest-first order), and the bits check at
the operation level. No search, no training; development systems only; counts only.

Lineages: Stage X version n's (flows after order 2's history; reaction networks after order 1's): the continued network
(its history's verified solutions, then its new-domain wakes), the control (its made-up systems' structures, then its
wakes) and the fresh network (its wakes only), before the new domain and after 10 and 30 tasks. Unfamiliar true terms:
those whose kind of term (grammar_g2's shape) is not among the lineage's verified solutions (or made-up structures) so
far. Usage: k0_check.py <workers> [smoke] (smoke: two development systems per domain, written to k0_smoke, nothing
printed)"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageX")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402

X_RUN = C.ROOT / "outputs/accumulating_learner/g2h/stageX"
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
OUT = X_RUN / ("k0_smoke" if SMOKE else "k0")
SETUPS = {"flows": {"dom": "flows_L1", "o": 2, "hist": "o2_mechanics2", "cbase": "c2_mechanics2"},
          "reactions": {"dom": "reactions_L1", "o": 1, "hist": "o1_ecology2", "cbase": "c1_ecology2"}}
ARMS = ("continued", "control", "fresh")
POINTS = (0, 10, 30)


def lineage(arm, o):
    return {"continued": f"xn_o{o}", "control": f"xcn_{o}", "fresh": f"xfreshn_{o}"}[arm]


def network(arm, setup, k):
    if k == 0:
        return {"continued": setup["hist"], "control": setup["cbase"], "fresh": None}[arm]
    return f"{lineage(arm, setup['o'])}_{setup['dom']}_d{k}"


def experience(arm, setup, k):
    """(earlier [(d, own, keys)], their groups or None, newest [(d, own, keys)]) of the lineage before task k+1."""
    import stageA_pilot as A
    o = setup["o"]
    earlier, groups = [], None
    if arm == "continued":
        rep = json.loads((C.REUSE / f"replays_o{o}.json").read_text())
        hist = [(k_.rsplit("|", 1), v) for k_, v in rep.items() if C.dom_of(k_.rsplit("|", 1)[0]) in C.HISTORY]
        earlier = [(2, int(e), v) for (tid, e), v in hist]
        groups = [C.dom_of(tid) for (tid, e), v in hist]
    elif arm == "control":
        p = OUT / f"madeup_structures_c{o}.json"
        if not p.exists():
            p.write_text(json.dumps(A.madeup_structures(C.cbase_files(o))))
        earlier = [tuple(x) for x in json.loads(p.read_text())]
    newest = []
    for w in [p_ for p_ in (10, 30) if p_ <= k]:
        res = json.loads((X_RUN / "wake" / f"{lineage(arm, o)}_w{w}.json").read_text())
        newest += [(4, e["eq"], e["solution"]["keys"]) for r in res if r["status"] == "done" for e in r["equations"] if e["solution"]]
    return earlier, groups, newest


def job(args):
    rec, nets, table_path = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import movenet_g2 as MN
    import search_g1 as SG
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    gid = vocab.gid
    loaded = {n: (MN.load_net(str(X_RUN / "nets" / n) + ".pt").eval(), {4: np.load(str(X_RUN / "nets" / n) + "_d4.npy")})
              for n in set(nets.values()) if n}
    tab = np.load(table_path)
    out = []

    def ranks(score):                                            # 1 = first; ties by vocabulary order (simplest first)
        order = np.lexsort((np.arange(len(score)), -score))
        r = np.empty(len(score), dtype=np.int64)
        r[order] = np.arange(1, len(score) + 1)
        return r
    simplest = ranks(-np.arange(vocab.V, dtype=np.float64))
    for i, eq in enumerate(rec["structure"]):
        true = [vocab.index[k] for k in eq["keys"] if k in vocab.index]
        se = SG.EquationSearchL1(i, ctx, 0, SG.uniform_prior, fitter, np.random.default_rng([1, i]))
        net_rank = {}
        for n, (net, emb) in loaded.items():
            net_rank[n] = ranks(MN.NetworkPrior(net, emb, shortlist=2048)(se.root, se.ctx).p_add)
        for (arm, k), n in nets.items():
            op_r = ranks(tab[f"{arm}_{k}_op"][gid, i])
            sh_r = ranks(tab[f"{arm}_{k}_shape"][gid, i])
            for t in true:
                out.append({"id": rec["id"], "eq": i, "arm": arm, "k": k, "key": vocab.keys[t], "op": int(op_r[t]),
                            "shape": int(sh_r[t]), "simplest": int(simplest[t]), "net": int(net_rank[n][t]) if n else None,
                            "V": int(vocab.V)})
    return out


def main():
    import numpy as np
    import grammar_g2 as GR
    import grammar_ops as GO
    import search_g1 as SG
    import stageX_report as XR
    workers = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    order = SG.vocabulary_order(4)
    tokens = GO.alphabet(order + SG.vocabulary_order(2))
    shared = json.loads((C.SYS / "shared_forms_g2h_L1.json").read_text())
    lines = []
    for which, setup in SETUPS.items():
        dom, o = setup["dom"], setup["o"]
        dev = C.records("dev", dom)[:2] if SMOKE else C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        tables, nets, seen, bits = {}, {}, {}, {}
        new_eqs = [(i, eq["keys"]) for r in dev for i, eq in enumerate(r["structure"]) if not flags[r["id"]][i]]
        for arm in ARMS:
            for k in POINTS:
                earlier, groups, newest = experience(arm, setup, k)
                og = GO.OpGrammar.fit([(own, keys) for d, own, keys in earlier], [(own, keys) for d, own, keys in newest],
                                      tokens, earlier_groups=groups)
                sg = GR.Grammar.fit(earlier, newest, earlier_groups=groups)
                tables[f"{arm}_{k}_op"] = np.array([[og.logp_term(key, own) for own in range(4)] for key in order])
                tables[f"{arm}_{k}_shape"] = sg.log_ratio(4).astype(np.float64)
                nets[(arm, k)] = network(arm, setup, k)
                seen[(arm, k)] = {s for d, own, keys in earlier + newest for s in GR.equation_shapes(d, own, keys)}
                bits[(arm, k)] = (np.mean([og.bits(i, keys) for i, keys in new_eqs]),
                                  [c.alpha for c in og.components])
        table_path = OUT / f"tables_{which}.npz"
        np.savez(table_path, **tables)
        uni = GO.OpGrammar([], tokens)
        bits_uniform = np.mean([uni.bits(i, keys) for i, keys in new_eqs])
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = [x for part in ex.map(job, [(r, nets, str(table_path)) for r in dev]) for x in part]
        flagd = {(r["id"], i): not flags[r["id"]][i] for r in dev for i in range(len(r["structure"]))}
        res = [x for x in res if flagd[(x["id"], x["eq"])]]          # new-form equations only
        (OUT / f"ranks_{which}.json").write_text(json.dumps(res))
        lines.append(f"== {which}: {len(new_eqs)} new-form development equations; bits check at the operation level (mean bits per "
                     f"equation; smoothing): uniform {bits_uniform:.1f}; " + "; ".join(
                         f"{arm} {'before' if k == 0 else f'{k} tasks'} {bits[(arm, k)][0]:.1f} ({bits[(arm, k)][1]})"
                         for arm in ARMS for k in POINTS))
        for arm in ARMS:
            for k in POINTS:
                rows = [x for x in res if x["arm"] == arm and x["k"] == k]
                for x in rows:
                    x["unfamiliar"] = GR.equation_shapes(4, x["eq"], [x["key"]])[0] not in seen[(arm, k)]
                    x["kind"] = XR.kind(x["key"], x["eq"]) or "other"
                def med(sel, f):
                    v = [x[f] for x in sel if x[f] is not None]
                    return f"{np.median(v):.0f}" if v else "-"
                parts = []
                for title, sel in (("all", rows), ("unfamiliar", [x for x in rows if x["unfamiliar"]]),
                                   ("bump", [x for x in rows if x["kind"] == "bump"]),
                                   ("saturating", [x for x in rows if x["kind"] == "saturating"])):
                    if not sel:
                        continue
                    parts.append(f"{title} ({len(sel)}): operation grammar {med(sel, 'op')}, shape grammar {med(sel, 'shape')}, "
                                 f"network {med(sel, 'net')}, simplest-first {med(sel, 'simplest')}")
                lines.append(f"   {arm:>9}, {'before the domain' if k == 0 else f'after {k} tasks':>17}: median rank of true terms (of about "
                             f"{rows[0]['V'] if rows else 0}) -- " + " | ".join(parts))
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    if not SMOKE:
        print(text)


if __name__ == "__main__":
    main()
