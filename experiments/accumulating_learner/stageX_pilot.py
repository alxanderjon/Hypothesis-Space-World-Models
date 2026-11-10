"""Stage X of the redesign: exploration as a safety net, the development pilot (owner's approval, 10 November; research
log "Stage X: exploration as a safety net, the pilot's design").

The slowdown pilot's f6 recipe (the 2,048-term shortlist in training and search, balanced review rounds, fantasies from
single-term usage and the size prior, no grammar; wakes on training tasks 1-10 and 11-30, development tests after each
review round) with every search in the new domain (wakes and tests, every arm) run by search_x's rules:
  version n   never blocked (rho = 1/10);
  version ns  never blocked plus the adaptive exploration share: 1/4 in a lineage's first wake of the new domain, then
              half the share of the latest wake's equations left unsolved, kept between 1/10 and 1/4 (tests use the
              lineage's current share);
  version f6  neither (the no-exploration side of a two-sided rerun; identical to search_g1's search).
Arms x<v><tag>_o<o> (continued), xc<v><tag>_<o> (rehearsal-matched control), xfresh<v><tag>_<o> (fresh); tag "r2" is a
rerun with seed 2 (the rerun rule). The fresh network's first wake is claim run 1's uniform one (unchanged by either
rule), reused in the pilot. Steps: <testbed>:<version>[:r2[:arm,...]]; budget:<version> (the winner's 30-task
experienced and fresh flows networks tested at 1,000 structures per equation, thorough cap 200; run with
G2H_TIME_CAP=14400); ranks:<version>[,<version>] (root rankings on the development flows systems); report.
Development systems only; counts only. Usage: stageX_pilot.py <workers> <step ...>"""

import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("G2H_RUN", "stageX")
os.environ.setdefault("G2H_SHORTLIST", "2048")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_g2h as C                                           # noqa: E402
import g2h_pilot as P                                            # noqa: E402

PILOT = C.ROOT / "outputs/accumulating_learner/g2h/pilot"
SETUPS = P.SETUPS
POINTS = (10, 30) if not C.SMOKE else (2, 4)
RHO = 0.1
SHARE0, SHARE_MIN, SHARE_MAX = 0.25, 0.1, 0.25
VERSIONS = {"n": (RHO, False), "ns": (RHO, True), "f6": (0.0, False)}
BUDGET_CHECK = (1000, 200) if not C.SMOKE else (40, 8)


def search_job_x(args):
    """cycles_g2h.search_job with search_x's rules (share, rho)."""
    rec, prior_path, seed, budget, cap, share, rho = args
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import search_x as SX
    out = {"id": rec["id"], "domain": rec["domain"], "status": "done", "budget": budget, "share": share, "rho": rho}
    t0 = time.time()
    signal.signal(signal.SIGALRM, C._alarm)
    signal.alarm(int(2 * C.TIME_CAP + 600))
    try:
        prior = C._prior(prior_path)
        res = SX.search_system_x(np.array(rec["bursts"]), prior, budget=budget, thorough_cap=cap, time_cap=C.TIME_CAP, seed=seed,
                                 probes=[eq["keys"] for eq in rec["structure"]], share=share, rho=rho)
        keep = ("structures", "seconds", "thorough", "time_capped", "prior_seconds", "fit_seconds", "thorough_seconds",
                "precise_seconds", "even_structures", "low_tried", "low_depth", "low_max_nodes")
        out["equations"] = [dict({k: e["effort"].get(k) for k in keep}, eq=e["eq"], first_exact=e["effort"]["first_exact"],
                                 n_exact=e["n_exact"], solution=e.get("solution"), solution_low_terms=e.get("solution_low_terms"),
                                 fit_errors=e["effort"].get("fit_errors", 0)) for e in res["equations"]]
        out["network_seconds"] = getattr(prior, "seconds", 0.0)
        out["probe"] = getattr(prior, "probe", None)
        out["probe_eq"] = getattr(prior, "probe_eq", None)
    except C._Limit:
        out["status"] = "safety limit"
    except Exception as ex:
        out["status"] = "error"
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        signal.alarm(0)
    out["seconds"] = time.time() - t0
    return out


def unsolved_share(wake):
    eqs = [e for r in wake if r["status"] == "done" for e in r["equations"]]
    return sum(e["first_exact"] is None for e in eqs) / max(1, len(eqs))


class PilotX(P.Pilot):
    explore = (0.0, 0.0)                                         # (share, rho) of the searches being run
    search_seed = 1

    def search(self, path, recs, prior, label, budget, cap):
        """cycles_g2h.Driver.search with search_x's rules."""
        if path.exists():
            return json.loads(path.read_text())
        share, rho = self.explore
        C.log(f"{label}: searching {len(recs)} systems with prior {Path(prior).name} (budget {budget}, thorough cap {cap}; "
              f"even-weights share {share:.2f}, never-blocked share {rho:.2f}; seed {self.search_seed})")
        t0 = time.time()
        res = list(self.pool.map(search_job_x, [(r, prior, self.search_seed, budget, cap, share, rho) for r in recs]))
        path.write_text(json.dumps(res, default=float))
        truth = {r["id"]: r["structure"] for r in recs}
        eqs = [(r, e) for r in res if r["status"] == "done" for e in r["equations"]]
        solved = sum(e["first_exact"] is not None for _, e in eqs)
        systems = sum(1 for r in res if r["status"] == "done" and all(e["first_exact"] is not None for e in r["equations"]))
        low = [e["low_tried"] for _, e in eqs if e.get("low_tried") is not None]
        secs = sorted(r["seconds"] for r in res)
        C.log(f"{label}: done in {time.time() - t0:.0f}s; equations solved {solved} of {sum(len(truth[r['id']]) for r in res)}; "
              f"systems {systems} of {len(res)}; time-capped {sum(e['time_capped'] for _, e in eqs)}; failed "
              f"{sum(r['status'] != 'done' for r in res)}; low-rated terms tried per equation median "
              f"{sorted(low)[len(low) // 2] if low else 0}; seconds per system median {secs[len(secs) // 2]:.0f}, max {secs[-1]:.0f}")
        return res

    def share_for(self, lineage, version, wakes):
        """The adaptive exploration share before a search: 1/4 before the lineage's first wake of the domain, then half
        the share of its latest wake's equations left unsolved, kept between 1/10 and 1/4."""
        if not VERSIONS[version][1]:
            return 0.0
        if not wakes:
            return SHARE0
        return min(SHARE_MAX, max(SHARE_MIN, unsolved_share(wakes[-1]) / 2))

    def curve_x(self, lineage, version, start, dom, mode, reuse_w10=None, earlier_files=None, mirror_prefix=None):
        tasks = C.records("train", dom)
        rho = VERSIONS[version][0]
        net, prev, wakes = start, 0, []
        for k in POINTS:
            self.explore = (self.share_for(lineage, version, wakes), rho)
            w = self.wake(lineage, f"{lineage}_w{k}", tasks[prev:k], net, reuse=reuse_w10 if k == POINTS[0] else None)
            wakes.append(w)
            mir = f"{mirror_prefix}_{dom}_d{k}" if mirror_prefix else None
            net = self.new_dream(lineage, f"{dom}_d{k}", None if net == "uniform" else net, w, dom, mode, C.NEW["steps"],
                                 {"shortlist": int(os.environ["G2H_SHORTLIST"])}, earlier_files, mir)
            self.explore = (self.share_for(lineage, version, wakes), rho)
            self.test(net, dom, f"{lineage}_t{k}dev", *C.NEWB, which="dev")
            prev = k
        return net

    def test(self, net, dom, label, budget, cap, which="test"):
        if not C.SMOKE:
            return super().test(net, dom, label, budget, cap, which)
        return self.search(C.RUN / "test" / f"{label}_{dom}.json", C.records(which, dom)[:3], net, f"test {label} on {dom}",
                           budget, cap)                          # smoke tests: three development systems


def names(version, tag, o):
    return f"x{version}{tag}_o{o}", f"xc{version}{tag}_{o}", f"xfresh{version}{tag}_{o}"


def run(D, step):
    parts = step.split(":")
    which, version = parts[0], parts[1]
    tag = parts[2] if len(parts) > 2 else ""
    arms = parts[3].split(",") if len(parts) > 3 else ["continued", "control", "fresh"]
    setup = SETUPS[which]
    dom, o = setup["dom"], setup["o"]
    D.search_seed = 2 if tag else 1
    cont, ctrl, fresh = names(version, tag, o)
    if "continued" in arms or "control" in arms:                 # the control's rehearsal mirrors this continued run
        hist = P.copy_net(P.RUN1 / "nets" / setup["hist"], C.RUN / "nets" / setup["hist"])
        D.curve_x(cont, version, hist, dom, "balanced")
    if "control" in arms:
        cbase = P.copy_net(P.RUN1 / "nets" / setup["cbase"], C.RUN / "nets" / setup["cbase"])
        D.curve_x(ctrl, version, cbase, dom, "balanced", earlier_files=C.cbase_files(o), mirror_prefix=cont)
    if "fresh" in arms:
        reuse = None if (C.SMOKE or tag) else P.RUN1 / "wake" / f"fresh_{dom}_w10.json"
        D.curve_x(fresh, version, "uniform", dom, "fresh", reuse_w10=reuse)


def budget_check(D, version):
    """The winner's 30-task experienced and fresh flows networks at 1,000 structures per equation (owner, 10 November)."""
    setup = SETUPS["flows"]
    dom, o = setup["dom"], setup["o"]
    budget, cap = BUDGET_CHECK
    cont, _, fresh = names(version, "", o)
    for lin in (cont, fresh):
        wakes = [json.loads((C.RUN / "wake" / f"{lin}_w{k}.json").read_text()) for k in POINTS]
        D.explore = (D.share_for(lin, version, wakes), VERSIONS[version][0])
        D.test(str(C.RUN / "nets" / f"{lin}_{dom}_d{POINTS[-1]}"), dom, f"{lin}_t{POINTS[-1]}dev_b{budget}", budget, cap, which="dev")


def ranks(versions, workers):
    """Root rankings by every lineage's networks after 10 and 30 tasks on the 40 development flows systems (as in the
    slowdown diagnosis): written to ranks_flows.json."""
    import subprocess
    nets = [f"{lin}_flows_L1_d{k}" for v in versions for lin in names(v, "", 2) for k in POINTS]
    nets = [n for n in nets if (C.RUN / "nets" / f"{n}.pt").exists()]
    out = C.RUN / "ranks_flows.json"
    subprocess.run([sys.executable, str(HERE / "rank_terms_x.py"), str(workers), str(C.RUN / "nets"), str(out)] + nets, check=True)


def main():
    workers, steps_ = int(sys.argv[1]), sys.argv[2:]
    if steps_ == ["report"]:
        import stageX_report
        stageX_report.report()
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    D = PilotX(workers, 1)
    C.log(f"Stage X pilot started: steps {steps_}; shortlist {os.environ['G2H_SHORTLIST']}; time cap {C.TIME_CAP:.0f}s")
    for step in steps_:
        kind = step.split(":")[0]
        if kind in SETUPS:
            run(D, step)
        elif kind == "budget":
            budget_check(D, step.split(":")[1])
        elif kind == "ranks":
            ranks(step.split(":")[1].split(","), workers)
    C.log("Stage X pilot steps done")


if __name__ == "__main__":
    main()
