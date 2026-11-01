# Owner-approved option (b), PRACTICE (development) half only: equation-pattern library vs a strong from-scratch learner,
# with pieces-only library, decoy patterns, copy-the-closest, removal check, and provenance of every pattern success.
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import numpy as np

SP = "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad"
OUT = f"{SP}/pattern"
SRC = f"{SP}/scratch3"            # M = 64 runs of the current (strong) learner: the source of pieces and patterns
sys.path.insert(0, "/home/alex/Hypothesis-Space-World-Models/experiments/accumulating_learner")
sys.path.insert(0, SP)
MS = (8, 16, 32)


def src_record(name):
    p = f"{SRC}/{name}_M64_scratch.json"
    return json.load(open(p)) if os.path.exists(p) else None


def source_material(name):
    """(pieces, patterns, model description, bursts) of an earlier system's accepted M = 64 run (current learner)."""
    import torch
    import harness27 as H27, discovery_learner as DL
    r = src_record(name)
    if not r or r.get("status") != "accepted" or "model" not in r:
        return None
    ref = H27.Reference27(name, seed=1)
    B = np.asarray(H27.bursts_from(ref, 1, 64))
    mu, sig = torch.tensor(r["mu"]), torch.tensor(r["sig"])
    Z = (torch.as_tensor(B) - mu) / sig
    model = DL.model_from_description(r["model"], mu, sig)
    pats = DL.motifs_from_model(model, Z[:, 4:9].reshape(-1, Z.shape[-1]))
    return {"name": name, "pieces": [p for p in r.get("pieces") or [] if p.get("origin") == "new"], "patterns": pats,
            "model": r["model"], "mu": r["mu"], "sig": r["sig"], "bursts": B.tolist(), "status": "accepted"}


def run(spec):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(2)
    import harness27 as H27, discovery_learner as DL, compile_artifact as CA
    out = spec["out"]
    if os.path.exists(out):
        return json.load(open(out))
    assert H27.split_half(spec["name"]) == "dev"
    DL.MOTIFS[0] = spec["arm"] in ("pattern", "pattern_decoy", "removal")
    rec = {k: spec[k] for k in ("name", "M", "arm", "pos")}
    rec["removed"] = spec.get("removed")
    t0 = time.time()
    try:
        ref = H27.Reference27(spec["name"], seed=1)
        b = H27.bursts_from(ref, 1, spec["M"])
        if spec["arm"] == "copy":
            import trial27 as T27
            res = T27.copy_closest(b, spec["earlier"], 1)
        else:
            lib = DL.Library.from_json(spec["library"])
            res = DL.learn(b, seed=1, time_limit=300 if spec["M"] <= 8 else 900, library=lib, discover_pieces=True)
        rec.update(status=res.get("status"), val=res.get("val_err_scaled"), levels=str(res.get("levels")),
                   patterns_used=res.get("patterns_used"), copied=res.get("copied"),
                   reused=[p.get("library_id") for p in res.get("pieces", []) if p.get("origin") == "reused"], exact=False)
        if res.get("status") == "accepted":
            m = res["model"]
            rec["model"] = m.describe(); rec["mu"] = m.mu.tolist(); rec["sig"] = m.sig.tolist()
            roll = CA.build(rec["model"], rec["mu"], rec["sig"], ref.sys.dt)
            ns = res.get("nsub") or 4
            ev = H27.evaluate(ref, None, ns, rollout_hat=lambda x0, steps: roll.adaptive(x0, steps, nsub0=max(1, ns // 4)))
            rec["exact"], rec["one"] = bool(ev["exact"]), ev["one_step_max_scaled_error"]
    except Exception as e:
        rec.update(status="error", exact=False, error=repr(e)[:300])
    rec["wall"] = time.time() - t0
    json.dump(rec, open(out, "w"), default=float)
    return rec


def build_specs():
    import discovery_learner as DL
    order = json.load(open(f"{SP}/probe/plan.json"))["order"]
    N = len(order)
    lib_p, lib_full, earlier, material = DL.Library(), DL.Library(), [], {}
    specs = []
    for k, name in enumerate(order):
        if k >= N // 2:
            for M in MS:
                base = {"name": name, "M": M, "pos": k}
                specs.append({**base, "arm": "scratch", "library": [], "out": f"{OUT}/{name}_M{M}_scratch.json"})
                if lib_full.entries:
                    specs.append({**base, "arm": "pattern", "library": lib_full.to_json(), "out": f"{OUT}/{name}_M{M}_pattern.json"})
                    specs.append({**base, "arm": "pattern_decoy", "library": DL.decoy_library(lib_full, seed=900 + k).to_json(),
                                  "out": f"{OUT}/{name}_M{M}_patterndecoy.json"})
                if lib_p.entries:
                    specs.append({**base, "arm": "pieces", "library": lib_p.to_json(), "out": f"{OUT}/{name}_M{M}_pieces.json"})
                if earlier:
                    specs.append({**base, "arm": "copy", "earlier": earlier, "out": f"{OUT}/{name}_M{M}_copy.json"})
        mat = source_material(name)
        if mat:
            material[name] = {k2: mat[k2] for k2 in ("patterns", "model", "mu", "sig")}
            for pc in mat["pieces"]:
                lib_p.add(DL.as_library_piece(pc), source=name)
                lib_full.add(DL.as_library_piece(pc), source=name)
            for pat in mat["patterns"]:
                lib_full.entries.append(DL.LibEntry(id=len(lib_full.entries), kind="motif", family="motif", nargs=0,
                                                    multiplier=False, shape=(), source=name, motif=pat))
            earlier.append({k2: mat[k2] for k2 in ("name", "status", "model", "bursts")})
        print(f"pos {k} {name}: pieces {len(lib_p.entries)}, library with patterns {len(lib_full.entries)}", flush=True)
    json.dump({"order": order, "library": lib_full.to_json(), "material": material}, open(f"{OUT}/plan.json", "w"), default=float)
    return specs


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = build_specs()
    print(len(specs), "jobs", flush=True)
    specs.sort(key=lambda s: (s["arm"] == "copy", s["M"], s["pos"]))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(run, s) for s in specs]):
            r = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] pos {r['pos']} {r['name']} M{r['M']} {r['arm']}: {r.get('status')} exact={r.get('exact')} "
                  f"levels={r.get('levels')} patterns={r.get('patterns_used')} {r.get('wall', 0):.0f}s", flush=True)
    # removal check: every pattern-arm win (exact, scratch not exact, patterns used) rerun without those patterns
    plan = json.load(open(f"{OUT}/plan.json"))
    rem = []
    for s in specs:
        if s["arm"] != "pattern" or not os.path.exists(s["out"]):
            continue
        r = json.load(open(s["out"]))
        sc = json.load(open(s["out"].replace("_pattern.json", "_scratch.json"))) if os.path.exists(s["out"].replace("_pattern.json", "_scratch.json")) else {}
        if r.get("exact") and not sc.get("exact") and r.get("patterns_used"):
            used = sorted({e for _, e in r["patterns_used"]})
            lib = [e for e in s["library"] if e["id"] not in used]
            rem.append({**{k: s[k] for k in ("name", "M", "pos")}, "arm": "removal", "library": lib, "removed": used,
                        "out": s["out"].replace("_pattern.json", "_removal.json")})
    print(len(rem), "removal jobs", flush=True)
    if rem:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            for f in as_completed([ex.submit(run, s) for s in rem]):
                r = f.result()
                print(f"[{time.strftime('%H:%M:%S')}] REMOVAL pos {r['pos']} {r['name']} M{r['M']} removed {r.get('removed')}: "
                      f"{r.get('status')} exact={r.get('exact')}", flush=True)


if __name__ == "__main__":
    main()
