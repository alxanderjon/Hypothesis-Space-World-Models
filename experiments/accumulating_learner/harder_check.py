"""Step 1 of the blend on the harder fields (owner, 14 November; research log "Owner: step 1 of the harder fields, with
the forced blend as an extra arm", where the design and pass rule were set before the run). No search, no training.
Claim run 2's networks and wakes (reaction networks and flows, four variables; order 1: mechanics, circuits, ecology,
reactions, flows; order 2: ecology, circuits, mechanics, flows, reactions), after 10 and 30 tasks of the new field:
  network alone   the continued network
  blend           its library from its own verified solutions, in the searched top-100 blend
  forced blend    the same library's top 100 ahead of every other term (arm 3's stand-in)
  made-up control the rehearsal-matched control network, its library from its own made-up systems (matched in number
                  to the blend's history equations) and its own verified solutions on the field, in the same blend
  fresh network
Ranks of the true terms of the development systems' new-form equations at the empty start, one short and two short.
Usage: harder_check.py <workers> <canon|tables|rank|report> [limit=N]"""

import itertools
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("G2H_RUN", "harder")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canon_check as CN                                         # noqa: E402  (the extended equivalence fix)

C = CN.C
ROOT = C.ROOT
OUT = C.RUN
RUN2 = ROOT / "outputs/accumulating_learner/g2h/run2"
SYS = ROOT / "outputs/accumulating_learner/g1/systems"
BASE_F = ROOT / "outputs/accumulating_learner/g2/run1/examples/fantasy"
EXT_F = RUN2 / "examples/fantasy"
D4 = 4
ORDERS = {1: ("mechanics", "circuits", "ecology", "reactions_L1", "flows_L1"),
          2: ("ecology", "circuits", "mechanics", "flows_L1", "reactions_L1")}
TASKS = (10, 30)
POINTS = ("empty start", "one short", "two short")
TOP, SHARE_TOP = 100, 0.95
OPTS = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
LIMIT = int(OPTS.get("limit", 0)) or None                        # smoke tests only: development systems per field


def configs():
    """(order, position (3 = 4th, 4 = 5th), field)."""
    return [(o, pos, ORDERS[o][pos]) for o in (1, 2) for pos in (3, 4)]


def records(which, dom):
    return [r for r in json.loads((SYS / f"{which}_{dom}.json").read_text()) if r["status"] == "accepted"]


def wake_solutions(name):
    """{(system id, eq): keys} of the verified solutions in one of claim run 2's wake files."""
    w = json.loads((RUN2 / "wake" / f"{name}.json").read_text())
    return {(y["id"], int(e["eq"])): e["solution"]["keys"] for y in w if y["status"] == "done" for e in y["equations"]
            if e["first_exact"] is not None and e.get("solution")}


def field_solutions(arm, o, field, T):
    """The continued network's ("o") or the control's ("c") verified solutions on the field's first T tasks."""
    out = {}
    for w in ("w10", "w30")[:1 if T < 30 else 2]:
        out.update(wake_solutions(f"{arm}{o}_{field}_{w}"))
    return out


def history_solutions(o, pos):
    """{(system id, eq): keys}: the continued network's replays of its history fields (and, in 5th position, of the 4th
    field), as kept by claim run 2."""
    rp = json.loads((RUN2 / f"replays_o{o}.json").read_text())
    hist = {f"g1-train-{d}" for d in ORDERS[o][:3]}
    fourth = {f"g2h-train-{ORDERS[o][3]}"} if pos == 4 else set()
    out = {}
    for k, keys in rp.items():
        sid, eq = k.rsplit("|", 1)
        if sid.rsplit("-", 1)[0] in hist | fourth:
            out[(sid, int(eq))] = keys
    return out


def madeup_equations(o, pos, n):
    """[(own, keys)] of the control's own made-up equations (base dreams, and in 5th position its 4th-field extension
    dreams): each root example (no current term) holds its equation's full true structure. A fixed-seed sample of n."""
    import numpy as np
    import search_g1 as SG
    files = [BASE_F / f"g2-control-c{o}-{d}{c}.pkl" for d in ORDERS[o][:3] for c in (1, 2)]
    if pos == 4:
        files += [EXT_F / f"g2h-control-c{o}x-{ORDERS[o][3]}{k}.pkl" for k in (10, 30, 100)]
    orders, out = {}, []
    for f in files:
        for e in pickle.loads(f.read_bytes()):
            if np.asarray(e["inS"]).any():
                continue
            d = int(e["d"])
            if d not in orders:
                orders[d] = SG.vocabulary_order(d)
            out.append((int(e["eq"]), [orders[d][int(g)] for g in np.asarray(e["gids"])[np.asarray(e["add"], dtype=bool)]]))
    random.Random(f"harder-madeup-c{o}-p{pos}").shuffle(out)
    return out[:n], len(out)


# ---------------------------------------------------------------------------------------------------------------------
# step canon: the extended equivalence fix on the new-field (and 4th-field) solutions
# ---------------------------------------------------------------------------------------------------------------------

def canon_key(sid, eq, keys):
    return f"{sid}|{eq}|{'+'.join(sorted(keys))}"


def to_fix():
    """{(system id, eq, raw keys tuple)} of every new-field and 4th-field solution any library uses."""
    items = set()
    for o, pos, field in configs():
        for arm in ("o", "c"):
            items |= {(sid, eq, tuple(sorted(k))) for (sid, eq), k in field_solutions(arm, o, field, 30).items()}
        if pos == 4:
            items |= {(sid, eq, tuple(sorted(k))) for (sid, eq), k in history_solutions(o, pos).items() if sid.startswith("g2h-")}
    return items


def step_canon(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    items = sorted(to_fix())
    recs = {r["id"]: r for f in ("reactions_L1", "flows_L1") for r in records("train", f)}
    by_sid = {}
    for sid, eq, keys in items:
        by_sid.setdefault(sid, {}).setdefault(eq, []).append(list(keys))
    jobs = []
    for sid, eqs in by_sid.items():
        for j in range(max(len(v) for v in eqs.values())):
            jobs.append((recs[sid], {str(eq): v[j] for eq, v in eqs.items() if j < len(v)}))
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(CN.canon_job, jobs))
    fixed = {canon_key(r["id"], int(eq), e["raw"]): e["canon_plus"] for r in res for eq, e in r["equations"].items()}
    (OUT / "canon.json").write_text(json.dumps(fixed))
    changed = sum(sorted(v) != sorted(k.split("|")[2].split("+")) for k, v in fixed.items())
    print(f"canon: {len(fixed)} solutions ({len(jobs)} jobs) in {time.time() - t0:.0f}s; rewritten {changed}")


# ---------------------------------------------------------------------------------------------------------------------
# step tables: the libraries (context-free form, compression plus counting) as log P tables over the 4-variable language
# ---------------------------------------------------------------------------------------------------------------------

def table_job(args):
    """curriculum_pilot.grammar_table for the four-variable language."""
    equations, path = args
    import numpy as np
    import grammar_ops as GO
    import library_ops as L
    import search_g1 as SG
    order = SG.vocabulary_order(D4)
    tokens = GO.alphabet(order, d=D4)
    terms = Counter(L.tree(k, own) for own, keys in equations for k in keys)
    lib = L.learn(terms, tokens, "uni")[0] if terms else []
    g = L.LibGrammar.fit(equations, [], tokens, lib, form="uni") if equations else L.LibGrammar([], tokens)
    np.save(path, np.array([[g.logp_term(k, own) for own in range(D4)] for k in order]))
    return {"path": path, "pieces": [L.show(p) for _, p in lib], "equations": len(equations)}


def step_tables(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    fixed = json.loads((OUT / "canon.json").read_text())

    def fix(sid, eq, keys):
        return fixed.get(canon_key(sid, eq, keys), keys)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    jobs, info = [], {}
    for o, pos, field in configs():
        hist = history_solutions(o, pos)
        hist_eqs = [(eq, fix(sid, eq, k) if sid.startswith("g2h-") else k) for (sid, eq), k in hist.items()]
        made, n_made = madeup_equations(o, pos, len(hist_eqs))
        for T in TASKS:
            mine = [(eq, fix(sid, eq, k)) for (sid, eq), k in field_solutions("o", o, field, T).items()]
            ctrl = [(eq, fix(sid, eq, k)) for (sid, eq), k in field_solutions("c", o, field, T).items()]
            for arm, eqs in (("blend", hist_eqs + mine), ("control", made + ctrl)):
                name = f"{arm}-o{o}-{field}-T{T}"
                jobs.append((eqs, str(OUT / "tables" / f"{name}.npy")))
                info[name] = {"history equations": len(hist_eqs) if arm == "blend" else len(made),
                              "made-up equations available": n_made if arm == "control" else None,
                              "field solutions": len(mine) if arm == "blend" else len(ctrl)}
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(table_job, jobs))
    for (eqs, path), r in zip(jobs, res):
        info[Path(path).stem]["pieces"] = r["pieces"]
    (OUT / "tables.json").write_text(json.dumps(info, indent=1))
    print(f"tables: {len(jobs)} libraries in {time.time() - t0:.0f}s; " + "; ".join(
        f"{k}: {v['history equations']}+{v['field solutions']} equations, {len(v['pieces'])} pieces" for k, v in info.items()))


# ---------------------------------------------------------------------------------------------------------------------
# step rank
# ---------------------------------------------------------------------------------------------------------------------

def rep_form(keys, cache):
    """A representative of a term set under every relabelling of up to four variables (g2h_shared.py's rule)."""
    import domains_g1 as DG
    ks = tuple(sorted(keys))
    if ks not in cache:
        forms = []
        for p in itertools.permutations(range(4)):
            r = [DG.relabelled(k, p) for k in ks]
            if all(x is not None for x in r):
                forms.append(tuple(sorted(r)))
        cache[ks] = min(forms) if forms else ks
    return cache[ks]


def newform_flags(o, pos, field, recs):
    """{system id: [new-form?] per equation} of the development systems for this order and position."""
    if pos == 3:
        sf = json.loads((SYS / "shared_forms_g2h_L1.json").read_text())[f"mce_dev_{field}"]
        return {r["id"]: [not s for s in sf[r["id"]]] for r in recs}
    cache = {}
    seen = {rep_form(eq["keys"], cache) for d in ORDERS[o][:4] for r in records("train", d) for eq in r["structure"]}
    return {r["id"]: [rep_form(eq["keys"], cache) not in seen for eq in r["structure"]] for r in recs}


def forced_score(lib, pn, legal):
    import numpy as np
    order = np.lexsort((np.arange(len(lib)), -np.where(legal, lib, -np.inf)))
    top = order[:min(TOP, int(legal.sum()))]
    s = lib.astype(float).copy()
    s[top] = 1e6 + 0.5 * lib[top] + 0.5 * np.log(np.maximum(pn[top], 1e-300))
    return s


def rank_job(args):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import fitter_g1 as FG
    import library_ops as L
    import movenet_g2 as MN
    import search_g1 as SG
    import muzero_checks as MZ
    rec, cfg, nets, tabs, newform, familiar = args
    ctx = SG.system_context(np.array(rec["bursts"]))
    fitter = FG.make_fitter(ctx, seed=1)
    vocab = ctx["vocab"]
    NP = {T: {a: MN.NetworkPrior(MN.load_net(p + ".pt").eval(), {D4: np.load(p + f"_d{D4}.npy")}, shortlist=2048)
              for a, p in nets[T].items()} for T in nets}
    TB = {T: {a: np.load(p)[vocab.gid] for a, p in tabs[T].items()} for T in tabs}
    rows = []
    for i, eqrec in enumerate(rec["structure"]):
        if not newform[i]:
            continue
        true = sorted({vocab.index[k] for k in eqrec["keys"] if k in vocab.index})
        if not true:
            continue
        eq = MZ.Eq(ctx, fitter, i)
        memo = {}
        for t in true:
            kind = "familiar" if L.tree(vocab.keys[t], i) in familiar else "new"
            others = [u for u in true if u != t]
            pts = [("empty start", ()), ("one short", tuple(others))] + ([("two short", tuple(others[1:]))] if len(true) >= 3 else [])
            for point, S in pts:
                if S not in memo:
                    nd = eq.node(S)
                    legal = MZ.legal_adds(vocab.V, S)
                    sc = {}
                    for T in TASKS:
                        pc, _ = MZ.net_dist(NP[T]["continued"], nd, eq.se.ctx, legal)
                        px, _ = MZ.net_dist(NP[T]["control"], nd, eq.se.ctx, legal)
                        pf, _ = MZ.net_dist(NP[T]["fresh"], nd, eq.se.ctx, legal)
                        lc, lx = TB[T]["blend"][:, i], TB[T]["control"][:, i]
                        sc[f"network alone T{T}"] = np.log(np.maximum(pc, 1e-300))
                        sc[f"blend T{T}"] = np.log(np.maximum(MZ.blend_dist(lc, pc, legal), 1e-300))
                        sc[f"forced blend T{T}"] = forced_score(lc, pc, legal)
                        sc[f"library T{T}"] = lc
                        sc[f"made-up control T{T}"] = np.log(np.maximum(MZ.blend_dist(lx, px, legal), 1e-300))
                        sc[f"control network T{T}"] = np.log(np.maximum(px, 1e-300))
                        sc[f"fresh network T{T}"] = np.log(np.maximum(pf, 1e-300))
                    memo[S] = (legal, sc)
                legal, sc = memo[S]
                row = {"order": cfg[0], "position": cfg[1] + 1, "field": cfg[2], "id": rec["id"], "eq": i,
                       "key": vocab.keys[t], "kind": kind, "point": point}
                row.update({k: MZ.rank(v, legal, t) for k, v in sc.items()})
                rows.append(row)
    return rows


def step_rank(workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import library_ops as L
    jobs = []
    for o, pos, field in configs():
        recs = records("dev", field)[:LIMIT] if LIMIT else records("dev", field)
        flags = newform_flags(o, pos, field, recs)
        familiar = {L.tree(k, eq) for (sid, eq), keys in history_solutions(o, pos).items() for k in keys}
        nets = {T: {"continued": str(RUN2 / "nets" / f"o{o}_{field}_d{T}"), "control": str(RUN2 / "nets" / f"c{o}_{field}_{field}_d{T}"),
                    "fresh": str(RUN2 / "nets" / f"fresh_{field}_{field}_d{T}")} for T in TASKS}
        tabs = {T: {a: str(OUT / "tables" / f"{a}-o{o}-{field}-T{T}.npy") for a in ("blend", "control")} for T in TASKS}
        jobs += [(r, (o, pos, field), nets, tabs, flags[r["id"]], familiar) for r in recs]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(rank_job, jobs) for x in part]
    (OUT / "ranks.json").write_text(json.dumps(rows))
    print(f"rank: {len(rows)} rows from {len(jobs)} development systems in {time.time() - t0:.0f}s")


ARMS = ("network alone", "blend", "forced blend", "library", "made-up control", "control network", "fresh network")


def step_report():
    import numpy as np
    rows = json.loads((OUT / "ranks.json").read_text())
    lines = ["== Harder fields, step 1: median rank of the true terms of new-form development equations (of about 4,700), "
             "empty start / one short / two short"]

    def med(R, col):
        return [float(np.median([x[col] for x in R if x["point"] == pt])) if any(x["point"] == pt for x in R) else float("nan")
                for pt in POINTS]
    ok = True
    for o, pos, field in configs():
        R = [x for x in rows if x["order"] == o and x["field"] == field]
        lines.append(f"   {field}, {pos + 1}th (order {o}): {len(R)} cases ({sum(x['kind'] == 'new' for x in R)} new, "
                     f"{sum(x['kind'] == 'familiar' for x in R)} familiar)")
        for T in TASKS:
            for kind in ("all", "new", "familiar"):
                S = R if kind == "all" else [x for x in R if x["kind"] == kind]
                if not S:
                    continue
                lines.append(f"      T{T} {kind:>8}: " + "; ".join(f"{a} {' / '.join(f'{v:.0f}' for v in med(S, f'{a} T{T}'))}" for a in ARMS))
        b, n, x = med(R, "blend T30"), med(R, "network alone T30"), med(R, "made-up control T30")
        this = all(b[j] < n[j] and b[j] < x[j] for j in (1, 2) if not math.isnan(b[j])) and b[0] <= 2 * n[0]
        ok &= this
        lines.append(f"      rule at 30 tasks: blend better than the network alone and the made-up control at one and two short, "
                     f"and within 2x the network alone at the empty start -> {this}")
    lines.append(f"== Step 1: {'PASSES' if ok else 'does not pass'}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text)
    print(text)


def main():
    import shutil
    workers, step = int(sys.argv[1]), sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT.resolve()).free / 2 ** 30
    if free < 2:
        raise SystemExit(f"not running {step}: free disk {free:.1f} GB < 2 GB on {OUT.resolve()} (owner's crash guard)")
    {"canon": lambda: step_canon(workers), "tables": lambda: step_tables(workers), "rank": lambda: step_rank(workers),
     "report": step_report}[step]()


if __name__ == "__main__":
    main()
