"""Trial 24 step 1: regenerate the accepted BioModels transients with learnable sampling (log-time and arc-length).
Same models, same initial-condition rule as Trial 23; time stamps are stored. See docs/research-log.md.

  python transient_resample.py --sbml <scratch>/dl/biomodels --workers 12
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import biomodels_convert as C
import systems as S

T23 = S.ROOT / "outputs/accumulating_learner/trial23/realfam_big"
OUT = S.ROOT / "outputs/accumulating_learner/trial24"
N = 4000


def initial_states(x0, rid):
    rng = np.random.default_rng(int(hashlib.sha256(("t" + rid).encode()).hexdigest()[:8], 16))
    return [x0] + [x0 * rng.uniform(0.5, 1.5, size=len(x0)) for _ in range(C.INSTANCES - 1)]


def arclength_times(f, start, H):
    t_dense = np.concatenate([[0.0], np.geomspace(1e-7 * H, H, 40000)])
    xd = C.integrate(f, start, t_dense)
    if xd is None:
        return None
    scale = xd.max(0) - xd.min(0) + 1e-12
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xd / scale, axis=0), axis=1))])
    if s[-1] <= 0:
        return None
    t = np.interp(np.linspace(0, s[-1], N), s, t_dense)
    t = np.maximum.accumulate(t)
    t[1:] = np.maximum(t[1:], t[:-1] + 1e-12 * H)  # strictly increasing
    return t


def process(args):
    rid, path, ts = args
    np.seterr(all="ignore")
    try:
        _, exprs = C.convert(Path(path), 2, 12)
        f = C.numeric(exprs)
        info, _ = C.convert(Path(path), 2, 12)
        x0 = np.array(info["x0"], dtype=float)
        H = 1.5 * ts
        out = {}
        for scheme in ("logtime", "arclength"):
            xs, tts = [], []
            for start in initial_states(x0, rid):
                if scheme == "logtime":
                    t = np.concatenate([[0.0], np.geomspace(1e-4 * ts, H, N - 1)])
                else:
                    t = arclength_times(f, start, H)
                    if t is None:
                        raise C.Reject("arc-length dense integration failed")
                x = C.integrate(f, start, t)
                if x is None:
                    raise C.Reject(f"{scheme} integration failed")
                xs.append(x.astype(np.float32)); tts.append(t)
            out[scheme] = (np.stack(xs), np.stack(tts))
        return rid, out, "ok"
    except Exception as e:
        return rid, None, f"{type(e).__name__}: {e}"[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbml", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    conv = {r["id"]: r for r in json.loads((T23 / "conversion.json").read_text())["records"]}
    split = json.loads((T23 / "split.json").read_text())
    excluded = set(json.loads((T23 / "validity_excluded.json").read_text()))
    manifest = {m["id"]: m for m in json.loads((Path(a.sbml) / "manifest.json").read_text())["models"]}
    ids = sorted(k for k in split if k not in excluded and conv[k].get("kind") == "transient")
    jobs = [(k, str(Path(a.sbml) / manifest[k]["file"]), conv[k]["settle_time"]) for k in ids]
    for scheme in ("logtime", "arclength"):
        (OUT / scheme).mkdir(parents=True, exist_ok=True)
    status = {}
    with Pool(a.workers) as pool:
        for rid, out, st in pool.imap_unordered(process, jobs):
            status[rid] = st
            if out:
                for scheme, (x, t) in out.items():
                    np.savez(OUT / scheme / f"{rid}.npz", x=x, t=t)
    (OUT / "resample_status.json").write_text(json.dumps(status, indent=1))
    print("resampled", sum(v == "ok" for v in status.values()), "of", len(status))


if __name__ == "__main__":
    main()
