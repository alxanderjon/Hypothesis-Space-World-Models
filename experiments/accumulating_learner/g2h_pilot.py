"""Stage 2a'' of the general learner: the slowdown pilot (owner, 8 November; research log "Loss of plasticity: literature,
and the slowdown pilot's design"). Development systems only. Testbed: nonlinear flows (level 1) in 4th position after
order 2's history (ecology, circuits, mechanics). One fix at a time, judged by the gap to the fresh network after 30
tasks on new-form development equations:
  f1 focus-then-consolidate (continued and control): dreams on the new domain only until 30 tasks, then one 1,000-step
     consolidation dream on all domains' verified solutions shuffled together; retention on the history domains'
     development systems after consolidation;
  f2 shrink-and-perturb once before the first new-domain dream (0.4 theta + 0.1 theta_fresh), balanced dreams;
  f3 L2 Init (lambda 1e-2 toward the network's own initial weights) in the new-domain dreams, balanced;
  f4 ReDo (tau 0.1; incoming weights redrawn, outgoing zeroed) before every new-domain dream, balanced;
  f5 1,000-step new-domain dreams (continued balanced; the fresh network too);
  f6 a 2,048-term shortlist in training and search (run with G2H_SHORTLIST=2048; the fresh network too).
The first wake (tasks 1-10) of each fix uses the unchanged starting network, so where nothing differs it is the claim
run's wake (reused). Resumable; counts only. Usage: g2h_pilot.py <workers> <step ...> (base, f1 .. f6, report,
confirm-<fix>)"""

import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("G2H_RUN", "pilot")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402

RUN1 = C.ROOT / "outputs/accumulating_learner/g2h/run1"
LEVEL = 1
SETUPS = {"flows": {"dom": "flows_L1", "o": 2, "hist": "o2_mechanics2", "cbase": "c2_mechanics2"},
          "reactions": {"dom": "reactions_L1", "o": 1, "hist": "o1_ecology2", "cbase": "c1_ecology2"}}
CONSOLIDATE_STEPS = 1000


# ---------------------------------------------------------------------------------------------------------------------
# training with the fixes (a fresh process per dream, as in the claim run)
# ---------------------------------------------------------------------------------------------------------------------

def initial_weights():
    """The history networks' own initialization (Stage 2a: seed 1 with 26 tokens), with the fourth role's row as in
    movenet_g2.load_net."""
    import torch
    import movenet_g2 as MN
    full = MN.TOKENS
    MN.TOKENS = full[:-1]
    try:
        torch.manual_seed(1)
        net = MN.MoveNetL2()
    finally:
        MN.TOKENS = full
    sd = net.state_dict()
    w = sd["terms.emb.weight"]
    sd["terms.emb.weight"] = torch.cat([w, 0.5 * (w[MN.TOK["other1"]] + w[MN.TOK["other2"]])[None]], 0)
    return sd


def redo(net, examples, tables, tau, seed, n=512, device="cpu"):
    """ReDo (Sokar et al. 2023): units whose mean |activation| over n examples is at most tau times their layer's mean
    get their incoming weights redrawn (the layer's default initialization) and their outgoing weights set to zero.
    Layers: the feed-forward blocks of both transformers (ReLU) and the scorer's and input projection's GELU layers."""
    import numpy as np
    import torch
    import torch.nn as nn
    import movenet_g2 as MN
    layers = []
    for enc in (net.terms.enc, net.enc):
        for L in enc.layers:
            layers.append((L.linear1, L.linear1, L.linear2, torch.relu))
    for seq in (net.scorer, net.inp):
        mods = list(seq)
        for j, m in enumerate(mods):
            if isinstance(m, nn.GELU) and j + 1 < len(mods):
                layers.append((mods[j - 1], mods[j - 1], mods[j + 1], nn.functional.gelu))
    acts = [torch.zeros(lin.out_features) for _, lin, _, _ in layers]
    cnt = [0] * len(layers)
    hooks = []
    for k, (mod, _, _, f) in enumerate(layers):
        def hook(m, inp, out, k=k, f=f):
            a = f(out).abs().reshape(-1, out.shape[-1])
            acts[k] += a.sum(0).detach().cpu()
            cnt[k] += a.shape[0]
        hooks.append(mod.register_forward_hook(hook))
    net.train()
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    rng = np.random.default_rng(seed)
    by_d = {}
    for i, e in enumerate(examples):
        by_d.setdefault(int(e["d"]), []).append(i)
    tabs = {d: torch.as_tensor(t).to(device) for d, t in tables.items()}
    with torch.no_grad():
        for d, idx in by_d.items():
            m_ = max(1, round(n * len(idx) / len(examples)))
            pick = rng.choice(len(idx), size=min(len(idx), m_), replace=False)
            sel = [idx[int(j)] for j in pick]
            for s in range(0, len(sel), 32):
                MN.loss_fn(net, MN.collate([examples[i] for i in sel[s:s + 32]], device), tabs)
    for hk in hooks:
        hk.remove()
    reset = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for k, (_, lin_in, lin_out, _) in enumerate(layers):
            mean = acts[k] / max(cnt[k], 1)
            score = mean / (mean.mean() + 1e-12)
            dorm = torch.nonzero(score <= tau).flatten()
            if len(dorm):
                tmp = nn.Linear(lin_in.in_features, lin_in.out_features, dtype=lin_in.weight.dtype)
                lin_in.weight[dorm] = tmp.weight[dorm].to(lin_in.weight.device)
                if lin_in.bias is not None:
                    lin_in.bias[dorm] = tmp.bias[dorm].to(lin_in.bias.device)
                lin_out.weight[:, dorm] = 0.0
            reset.append([int(len(dorm)), int(lin_in.out_features)])
    return reset


def train_job(args):
    """cycles_g2h.train_job with the pilot's fixes (opts: sp (shrink, noise), l2init (lambda), redo (tau),
    shortlist (attended terms in training))."""
    sources, prev, out, gi, seed, steps, balanced, opts = args
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import movenet_g2 as MN
    import search_g1 as SG
    if opts.get("shortlist"):
        MN.SHORTLIST = int(opts["shortlist"])
    torch.manual_seed(1)
    net = MN.load_net(prev + ".pt") if prev is not None else MN.MoveNetL2()
    device = f"cuda:{gi}" if gi is not None else "cpu"
    if gi is None:
        torch.set_num_threads(8)
    tables = {d: MN.token_table(SG.vocabulary_order(d), d) for d in C.DIMS}
    examples, groups = C._load_examples(sources)
    extra = {}
    if opts.get("sp"):
        lam, gam = opts["sp"]
        torch.manual_seed(seed % (1 << 31))
        fresh = MN.MoveNetL2()
        with torch.no_grad():
            for (_, p1), (_, p2) in zip(net.named_parameters(), fresh.named_parameters()):
                p1.mul_(lam).add_(gam * p2)
        extra["shrink_perturb"] = [lam, gam]
    if opts.get("redo"):
        net = net.to(device)
        extra["redo_reset_per_layer"] = redo(net, examples, tables, float(opts["redo"]), seed % (1 << 31), device=device)
    penalty = None
    if opts.get("l2init"):
        lam = float(opts["l2init"])
        sd0 = initial_weights()
        theta0 = {n: sd0[n].to(device) for n, _ in net.named_parameters()}

        def penalty(m):
            return lam * sum(((p - theta0[n]) ** 2).sum() for n, p in m.named_parameters())
        extra["l2init"] = lam
    tl = MN.train(net, examples, tables, steps, batch=C.BATCH, lr=C.LR, device=device, seed=seed,
                  groups=groups if balanced else None, penalty=penalty)
    counts = {}
    for e, g in zip(examples, groups):
        c = counts.setdefault(str(int(g)), {})
        c[str(e["d"])] = c.get(str(e["d"]), 0) + 1
    for d in C.DIMS:
        np.save(f"{out}_d{d}.npy", MN.encode_vocabulary(net, tables[d], device=device))
    torch.save(net.cpu().state_dict(), out + ".pt")
    return tl, counts, extra


class Pilot(C.Driver):
    def replays(self, lineage):
        p = C.RUN / f"replays_{lineage}.json"
        if p.exists():
            return json.loads(p.read_text())
        rep = {}
        if lineage.endswith(("_o1", "_o2")):                      # experienced lineages start with their history's solutions
            rep = {k: v for k, v in json.loads((C.REUSE / f"replays_{lineage[-2:]}.json").read_text()).items()
                   if C.dom_of(k.rsplit("|", 1)[0]) in C.HISTORY}
        p.write_text(json.dumps(rep))
        return rep

    def replay_sources(self, rep, newest):
        out, missing = [], {}
        for k, keys in rep.items():
            tid, eq = k.rsplit("|", 1)
            missing.setdefault(tid, {})[int(eq)] = keys
        jobs = []
        for tid, eqs in sorted(missing.items()):
            d = self.recs[tid]["desc"]["d"]
            st = [eqs.get(i) for i in range(d)]
            name = f"{tid}__{C.h(st)}.pkl"
            p = next((q for q in (C.REUSE / "examples/replay" / name, RUN1 / "examples/replay" / name) if q.exists()),
                     C.RUN / "examples/replay" / name)
            if not p.exists():
                jobs.append((p, (self.recs[tid]["bursts"], st, int(C.h(tid), 16) % (1 << 31))))
            out.append((str(p), 1 if C.dom_of(tid) == newest else 0, None))
        if jobs:
            for (p, _), ex in zip(jobs, self.pool.map(C.examples_job, [a for _, a in jobs])):
                p.write_bytes(pickle.dumps(ex))
        return out

    def train(self, out, sources, prev, steps, label, info, balanced, opts=None):
        gi, gdesc = self.gpu()
        t0 = time.time()
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as one:
            tl, counts, extra = one.submit(train_job, (sources, prev, str(out), gi, int(C.h(label), 16) % (1 << 31), steps,
                                                       balanced, opts or {})).result()
        rec = dict(info, device=gdesc, seconds=time.time() - t0, examples_by_group_d=counts, steps=steps, balanced=balanced,
                   opts=opts or {}, extra=extra, log=tl)
        (C.RUN / "dream" / f"{label}_train.json").write_text(json.dumps(rec))
        n = sum(sum(c.values()) for c in counts.values())
        C.log(f"dream {label}: trained {steps} steps on {n} examples ({counts}){' balanced' if balanced else ''}; fixes "
              f"{opts or {}} {extra if extra else ''} on {gdesc} in {time.time() - t0:.0f}s; last loss {tl[-1]['loss']:.3f}; "
              f"skipped {tl[-1]['skipped']}")
        return rec

    # -- one pilot lineage --
    def wake(self, lineage, label, tasks, prior, reuse=None):
        path = C.RUN / "wake" / f"{label}.json"
        if not path.exists() and reuse is not None and Path(reuse).exists():
            shutil.copy(reuse, path)
            C.log(f"wake {label}: reused {Path(reuse).name} (identical network, tasks, budget and seed)")
        return self.search(path, tasks, prior, f"wake {label}", *C.NEWB)

    def new_dream(self, lineage, tag, prev, wake, dom, mode, steps, opts=None, earlier_files=None, mirror=None):
        """mode: 'focus' (the new domain only: its verified solutions and fantasies), 'balanced' (claim-run recipe),
        'fresh' (the fresh network's recipe)."""
        out = C.RUN / "nets" / f"{lineage}_{tag}"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        rep = self.add_replays(lineage, wake)
        src = self.replay_sources(rep, dom)
        if mode == "focus":
            src = [s for s in src if s[1] == 1]
        src.append((self.fantasies(f"g2h-fantasy-{lineage}-{tag}", {4: C.NEW["fantasies"]},
                                   self.usage({k: v for k, v in rep.items() if C.dom_of(k.rsplit("|", 1)[0]) == dom})), 1, None))
        if earlier_files is not None and mode == "balanced":    # a control in the balanced schedule
            src += self.rehearsal(f"{lineage}_{tag}", earlier_files, self.counts_of(mirror).get(0, {}))
        self.train(out, src, None if prev == "uniform" else prev, steps, f"{lineage}_{tag}", {"mode": mode},
                   mode == "balanced", opts)
        return str(out)

    def consolidate(self, lineage, prev, dom, earlier_files=None, target=None):
        """One dream on all domains' verified solutions shuffled together (control: its earlier made-up examples, as
        many per dimension as `target`)."""
        out = C.RUN / "nets" / f"{lineage}_consolidated"
        if Path(f"{out}_d4.npy").exists():
            return str(out)
        src = self.replay_sources(self.replays(lineage), dom)
        if earlier_files is not None:
            src += self.rehearsal(f"{lineage}_consolidated", earlier_files, target)
        self.train(out, src, prev, CONSOLIDATE_STEPS, f"{lineage}_consolidated", {"mode": "consolidate"}, False)
        return str(out)

    def curve(self, lineage, start, dom, mode, steps=None, opts10=None, opts30=None, reuse_w10=None, earlier_files=None,
              mirror_prefix=None):
        """Wakes on training tasks 1-10 and 11-30, a dream after each, development tests after each dream."""
        tasks = C.records("train", dom)
        steps = steps or C.NEW["steps"]
        w = self.wake(lineage, f"{lineage}_w10", tasks[:10], start, reuse=reuse_w10)
        mir = f"{mirror_prefix}_{dom}_d10" if mirror_prefix else None
        n10 = self.new_dream(lineage, f"{dom}_d10", start, w, dom, mode, steps, opts10, earlier_files, mir)
        self.test(n10, dom, f"{lineage}_t10dev", *C.NEWB, which="dev")
        w = self.wake(lineage, f"{lineage}_w30", tasks[10:30], n10)
        mir = f"{mirror_prefix}_{dom}_d30" if mirror_prefix else None
        n30 = self.new_dream(lineage, f"{dom}_d30", n10, w, dom, mode, steps, opts30, earlier_files, mir)
        self.test(n30, dom, f"{lineage}_t30dev", *C.NEWB, which="dev")
        return n30

    def retention_dev(self, net, label, o):
        for hd in C.orders(LEVEL)[o][:3]:
            self.test(net, hd, f"{label}_ret", C.HIST["budget"], C.HIST["cap"], which="dev")


def copy_net(src, dst):
    for suf in (".pt", "_d2.npy", "_d3.npy", "_d4.npy"):
        if not Path(f"{dst}{suf}").exists():
            shutil.copy(f"{src}{suf}", f"{dst}{suf}")
    return str(dst)


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    C.RUN.mkdir(parents=True, exist_ok=True)
    P = Pilot(workers, LEVEL)
    C.log(f"pilot started: steps {steps_}; shortlist {os.environ.get('G2H_SHORTLIST', '768 (default)')}")
    for step in steps_:
        setup = SETUPS["reactions" if step.startswith("confirm") else "flows"]
        dom, o = setup["dom"], setup["o"]
        hist = copy_net(RUN1 / "nets" / setup["hist"], C.RUN / "nets" / setup["hist"])
        cbase = copy_net(RUN1 / "nets" / setup["cbase"], C.RUN / "nets" / setup["cbase"])
        w10_cont = RUN1 / "wake" / f"o{o}_{dom}_w10.json"
        w10_ctrl = RUN1 / "wake" / f"c{o}_{dom}_w10.json"
        w10_fresh = RUN1 / "wake" / f"fresh_{dom}_w10.json"
        fix = step.split("-")[1] if step.startswith("confirm") else step
        if step in ("base",) or step.startswith("confirm"):
            for lab, net in ((f"base_cont_o{o}", RUN1 / "nets" / f"o{o}_{dom}_d10"), (f"base_ctrl_o{o}", RUN1 / "nets" / f"c{o}_{dom}_{dom}_d10"),
                             (f"base_fresh_{dom}", RUN1 / "nets" / f"fresh_{dom}_{dom}_d10")):
                P.test(str(net), dom, f"{lab}_t10dev", *C.NEWB, which="dev")
            for lab, net in ((f"base_cont_o{o}", RUN1 / "nets" / f"o{o}_{dom}_d30"), (f"base_ctrl_o{o}", RUN1 / "nets" / f"c{o}_{dom}_{dom}_d30")):
                P.test(str(net), dom, f"{lab}_t30dev", *C.NEWB, which="dev")
            src = RUN1 / "test" / f"fresh_{dom}_t30dev_{dom}.json"
            dst = C.RUN / "test" / f"base_fresh_{dom}_t30dev_{dom}.json"
            if not dst.exists():
                shutil.copy(src, dst)
            P.retention_dev(hist, f"base_hist_o{o}", o)
            P.retention_dev(cbase, f"base_cbase_o{o}", o)
            if step == "base":
                continue
        if fix == "f1":
            pre = "q" if step.startswith("confirm") else "p"
            cont, ctrl = f"{pre}1_o{o}", f"{pre}1c_{o}"
            n30 = P.curve(cont, hist, dom, "focus", reuse_w10=w10_cont)       # continued: focus, then consolidation
            cons = P.consolidate(cont, n30, dom)
            P.test(cons, dom, f"{cont}_t30cons_dev", *C.NEWB, which="dev")
            P.retention_dev(cons, f"{cont}_cons", o)
            n30 = P.curve(ctrl, cbase, dom, "focus", reuse_w10=w10_ctrl)      # control: the same schedule; its
            rec = json.loads((C.RUN / "dream" / f"{cont}_consolidated_train.json").read_text())  # consolidation reviews as
            target = {int(d): n for d, n in rec["examples_by_group_d"].get("0", {}).items()}     # many earlier made-up
            cons = P.consolidate(ctrl, n30, dom, earlier_files=C.cbase_files(o), target=target)  # examples per dimension
            P.test(cons, dom, f"{ctrl}_t30cons_dev", *C.NEWB, which="dev")
            P.retention_dev(cons, f"{ctrl}_cons", o)
        elif fix in ("f2", "f3", "f4"):
            opts = {"f2": ({"sp": [0.4, 0.1]}, None), "f3": ({"l2init": 1e-2}, {"l2init": 1e-2}),
                    "f4": ({"redo": 0.1}, {"redo": 0.1})}[fix]
            lin = f"p{fix[1]}_o{o}" if not step.startswith("confirm") else f"q{fix[1]}_o{o}"
            P.curve(lin, hist, dom, "balanced", opts10=opts[0], opts30=opts[1], reuse_w10=w10_cont)
        elif fix == "f5":
            for lin, start, mode, rw in ((f"p5_o{o}", hist, "balanced", w10_cont), (f"p5fresh_{o}", "uniform", "fresh", w10_fresh)):
                if step.startswith("confirm"):
                    lin = "q" + lin[1:]
                P.curve(lin, start, dom, mode, steps=1000, reuse_w10=rw)
        elif fix == "f6":
            assert os.environ.get("G2H_SHORTLIST") == "2048", "run f6 with G2H_SHORTLIST=2048"
            for lin, start, mode, rw in ((f"p6_o{o}", hist, "balanced", None), (f"p6fresh_{o}", "uniform", "fresh", w10_fresh)):
                if step.startswith("confirm"):
                    lin = "q" + lin[1:]
                P.curve(lin, start, dom, mode, opts10={"shortlist": 2048}, opts30={"shortlist": 2048}, reuse_w10=rw)
        elif step == "report":
            report()
    C.log("pilot steps done")


def report():
    import g2h_report as R
    shared = json.loads((C.SYS / f"shared_forms_g2h_L{LEVEL}.json").read_text())
    for name, setup in SETUPS.items():
        dom, o = setup["dom"], setup["o"]
        dev = C.records("dev", dom)
        flags = shared[f"mce_dev_{dom}"]
        new = [(r["id"], i) for r in dev for i in range(len(r["structure"])) if not flags[r["id"]][i]]
        def ld(lab, dom=dom):                                   # the pilot's own folder
            q = C.RUN / "test" / f"{lab}_{dom}.json"
            return json.loads(q.read_text()) if q.exists() else None
        rows = [("baseline (claim run's balanced schedule)", f"base_cont_o{o}", f"base_fresh_{dom}"),
                ("baseline control", f"base_ctrl_o{o}", f"base_fresh_{dom}")]
        pre = "p" if name == "flows" else "q"
        rows += [(f"f1 focus phase, continued", f"{pre}1_o{o}", f"base_fresh_{dom}"),
                 (f"f1 focus phase, control", f"{pre}1c_{o}", f"base_fresh_{dom}"),
                 ("f2 shrink-and-perturb", f"{pre}2_o{o}", f"base_fresh_{dom}"),
                 ("f3 L2 Init", f"{pre}3_o{o}", f"base_fresh_{dom}"),
                 ("f4 ReDo", f"{pre}4_o{o}", f"base_fresh_{dom}"),
                 ("f5 1,000-step dreams", f"{pre}5_o{o}", f"{pre}5fresh_{o}"),
                 ("f6 2,048-term shortlist", f"{pre}6_o{o}", f"{pre}6fresh_{o}")]
        print(f"== {name} (development, {len(new)} new-form equations); experienced vs fresh at equal exposure")
        for title, e, f in rows:
            parts = []
            for t in (10, 30):
                a, b = ld(f"{e}_t{t}dev"), ld(f"{f}_t{t}dev")
                if a is None or b is None:
                    continue
                x, y = R.paired(a, b, new)
                gap = 100.0 * (R.counts(b, new) - R.counts(a, new)) / len(new)
                parts.append(f"{t} tasks: experienced {R.counts(a, new)}, fresh {R.counts(b, new)}, gap {gap:+.1f} points "
                             f"(only experienced {x}, only fresh {y}, p = {R.sign_p(x, y):.2g})")
            if parts:
                print(f"   {title}: " + "; ".join(parts))
        for lin in (f"{pre}1_o{o}", f"{pre}1c_{o}"):
            a, b = ld(f"{lin}_t30cons_dev"), ld(f"{lin}_t30dev")
            if a is not None and b is not None:
                x, y = R.paired(a, b, new)
                print(f"   {lin} after consolidation: {R.counts(a, new)} new-form (focus end {R.counts(b, new)}; only after "
                      f"{x}, only before {y}, p = {R.sign_p(x, y):.2g})")
            for hd in C.orders(LEVEL)[o][:3]:
                b0 = ld(f"base_hist_o{o}_ret" if "c" not in lin.split("_")[0] else f"base_cbase_o{o}_ret", hd)
                a0 = ld(f"{lin}_cons_ret", hd)
                if a0 is not None and b0 is not None:
                    print(f"   {lin} retention on {hd} development systems: before the new domain "
                          f"{R.systems_solved(b0)} of {len(b0)} systems ({R.counts(b0, list(R.solved_map(b0)))} eq.), after "
                          f"consolidation {R.systems_solved(a0)} ({R.counts(a0, list(R.solved_map(a0)))} eq.)")


if __name__ == "__main__":
    main()
