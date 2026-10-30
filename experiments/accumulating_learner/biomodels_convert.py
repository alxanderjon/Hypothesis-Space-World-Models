"""Convert downloaded BioModels SBML (untrusted data) into 3-D ODE systems by the pre-registered sourcing rule.

SBML is parsed as XML data; MathML is converted to SymPy objects by walking the tree with a whitelist of operators.
No downloaded text is ever passed to eval/exec or to a string parser. Generated numeric functions contain only the
state symbols x0, x1, x2, numbers and NumPy elementary functions.

  python biomodels_convert.py --sbml <scratch>/dl/biomodels --out outputs/accumulating_learner/trial16/realfam
"""

import argparse
import hashlib
import json
import re
import signal
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

MAX_BYTES = 5_000_000
N_POINTS, BURN, SPAN, INSTANCES = 4000, 50, 100, 8
EXCLUDE_TITLE = re.compile(
    r"belousov|zhabotinsky|calcium|ca2\+|ca\(2\+\)|ca\^2|excitable|isothermal|colpitts|chua|hopfield|beer|ni[nñ]o|"
    r"anishchenko|cellular neural|vallis|"
    # development and confirmation catalog systems (by name)
    r"burke|halvorsen|kawczynski|rikitake|salton|shimizu|sprott|thomas|windmi|yu ?wang|"
    r"chen ?lee|dadras|genesio|glycolytic oscillation|hadley|hindmarsh|itik|banks|rabinovich|sakarya|tsucs|torus",
    re.I)


class Reject(Exception):
    pass


def tag(e):
    return e.tag.split("}")[-1]


# ---------------- MathML -> SymPy (whitelist) ----------------

UNARY = {"exp": sp.exp, "ln": sp.log, "abs": sp.Abs, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
         "sinh": sp.sinh, "cosh": sp.cosh, "arctan": sp.atan}


def mathml(e, env, funcs):
    t = tag(e)
    if t == "math":
        kids = list(e)
        if len(kids) != 1:
            raise Reject("math with != 1 child")
        return mathml(kids[0], env, funcs)
    if t == "cn":
        typ = e.get("type", "real")
        txt = "".join(e.itertext()).strip()
        if typ in ("real", "integer"):
            return sp.Float(float(txt)) if typ == "real" else sp.Integer(int(txt))
        if typ == "e-notation":
            parts = [p.strip() for p in e.itertext() if p.strip()]
            return sp.Float(float(parts[0]) * 10 ** float(parts[1]))
        if typ == "rational":
            parts = [p.strip() for p in e.itertext() if p.strip()]
            return sp.Rational(int(parts[0]), int(parts[1]))
        raise Reject(f"cn type {typ}")
    if t == "ci":
        name = "".join(e.itertext()).strip()
        if name not in env:
            raise Reject(f"unknown identifier {name}")
        return env[name]
    if t in ("exponentiale", "pi"):
        return sp.E if t == "exponentiale" else sp.pi
    if t in ("true", "false"):
        return sp.true if t == "true" else sp.false
    if t == "piecewise":
        pieces, other = [], None
        for k in e:
            kk = list(k)
            if tag(k) == "piece":
                pieces.append((mathml(kk[0], env, funcs), mathml(kk[1], env, funcs)))
            elif tag(k) == "otherwise":
                other = mathml(kk[0], env, funcs)
        if other is not None:
            pieces.append((other, sp.true))
        return sp.Piecewise(*pieces)
    if t == "csymbol":
        raise Reject("csymbol (time/delay/avogadro)")
    if t == "apply":
        kids = list(e)
        op, args = kids[0], kids[1:]
        o = tag(op)
        if o == "ci":  # function definition call
            fname = "".join(op.itertext()).strip()
            if fname not in funcs:
                raise Reject(f"unknown function {fname}")
            params, body = funcs[fname]
            vals = [mathml(a, env, funcs) for a in args]
            if len(vals) != len(params):
                raise Reject("arity")
            return mathml(body, {**env, **dict(zip(params, vals))}, funcs)
        q = [a for a in args if tag(a) not in ("degree", "logbase")]
        vals = [mathml(a, env, funcs) for a in q]
        if o == "plus":
            return sp.Add(*vals)
        if o == "times":
            return sp.Mul(*vals)
        if o == "minus":
            return -vals[0] if len(vals) == 1 else vals[0] - vals[1]
        if o == "divide":
            return vals[0] / vals[1]
        if o == "power":
            return vals[0] ** vals[1]
        if o == "root":
            deg = [a for a in args if tag(a) == "degree"]
            d = mathml(list(deg[0])[0], env, funcs) if deg else sp.Integer(2)
            return vals[0] ** (1 / d)
        if o == "log":
            base = [a for a in args if tag(a) == "logbase"]
            b = mathml(list(base[0])[0], env, funcs) if base else sp.Integer(10)
            return sp.log(vals[0]) / sp.log(b)
        if o in UNARY and len(vals) == 1:
            return UNARY[o](vals[0])
        REL = {"gt": sp.Gt, "lt": sp.Lt, "geq": sp.Ge, "leq": sp.Le, "eq": sp.Eq, "neq": sp.Ne}
        if o in REL and len(vals) == 2:
            return REL[o](vals[0], vals[1])
        if o in ("and", "or"):
            return (sp.And if o == "and" else sp.Or)(*vals)
        if o == "not" and len(vals) == 1:
            return sp.Not(vals[0])
        raise Reject(f"operator {o}")
    raise Reject(f"element {t}")


def parse_lambda(e):
    lam = [k for k in e.iter() if tag(k) == "lambda"]
    if len(lam) != 1:
        raise Reject("functionDefinition without a single lambda")
    lam = lam[0]
    params = ["".join(b.itertext()).strip() for b in lam if tag(b) == "bvar"]
    body = [b for b in lam if tag(b) != "bvar"]
    if len(body) != 1:
        raise Reject("lambda body")
    return params, body[0]


# ---------------- SBML model -> 3 ODEs ----------------

def find(root, name):
    return [e for e in root.iter() if tag(e) == name]


def convert(path, dmin=3, dmax=3):
    if path.stat().st_size > MAX_BYTES:
        raise Reject("file too large")
    root = ET.parse(path).getroot()
    model = [e for e in root if tag(e) == "model"]
    if not model:
        raise Reject("no model")
    model = model[0]
    for bad in ("listOfEvents", "listOfConstraints"):
        if any(tag(e) == bad and len(e) for e in model):
            raise Reject(bad)
    if find(model, "algebraicRule"):
        raise Reject("algebraic rule")
    if any(r.get("fast") == "true" for r in find(model, "reaction")):
        raise Reject("fast reaction")
    if find(model, "stoichiometryMath"):
        raise Reject("stoichiometryMath")
    funcs = {fd.get("id"): parse_lambda(fd) for fd in find(model, "functionDefinition")}
    comps = {c.get("id"): float(c.get("size", c.get("volume", "1")) or 1) for c in find(model, "compartment")}
    params = {p.get("id"): p for p in find(model, "parameter") if p.get("id")}
    species = {s.get("id"): s for s in find(model, "species")}
    rate_rules = {r.get("variable"): r for r in find(model, "rateRule")}
    assign_rules = {r.get("variable"): r for r in find(model, "assignmentRule")}
    init_assign = {ia.get("symbol"): ia for ia in find(model, "initialAssignment")}
    # state variables: species changed by reactions or rate rules (not constant/boundary/assigned), plus rate-ruled params/compartments
    changed = set()
    for r in find(model, "reaction"):
        for sr in find(r, "speciesReference"):
            changed.add(sr.get("species"))
    states = [sid for sid, s in species.items()
              if (sid in changed or sid in rate_rules) and s.get("boundaryCondition") != "true"
              and s.get("constant") != "true" and sid not in assign_rules]
    states += [v for v in rate_rules if v not in species]
    if not dmin <= len(states) <= dmax:
        raise Reject(f"{len(states)} state variables")
    X = sp.symbols(" ".join(f"x{i}" for i in range(len(states))))
    sym = dict(zip(states, X))
    env = {}
    for cid, v in comps.items():
        env[cid] = sp.Float(v)
    for pid, p in params.items():
        if pid in sym:
            continue
        if p.get("value") is None and pid not in assign_rules:
            raise Reject(f"parameter {pid} without value")
        if p.get("value") is not None:
            env[pid] = sp.Float(float(p.get("value")))
    for sid, s in species.items():
        if sid in sym:
            env[sid] = sym[sid]
        else:
            v = s.get("initialConcentration", s.get("initialAmount"))
            if v is None and sid not in assign_rules:
                raise Reject(f"species {sid} without initial value")
            if v is not None:
                env[sid] = sp.Float(float(v))
    for v in rate_rules:
        if v in sym:
            env[v] = sym[v]
    # initial assignments: must evaluate to constants (in parameters, compartments and initial values)
    init_vals = {}
    pending_ia = dict(init_assign)
    const_env = dict(env)
    for sid in species:
        if sid in sym:
            sd = species[sid]
            v = sd.get("initialConcentration", sd.get("initialAmount"))
            if v is not None:
                const_env[sid] = sp.Float(float(v))
    for _ in range(len(pending_ia) + 1):
        for v, ia in list(pending_ia.items()):
            try:
                val = mathml([k for k in ia if tag(k) == "math"][0], const_env, funcs)
                if val.free_symbols:
                    raise Reject("unknown identifier (symbolic initial assignment)")
                init_vals[v] = float(val)
                const_env[v] = sp.Float(init_vals[v])
                if v not in sym:
                    env[v] = sp.Float(init_vals[v])
                del pending_ia[v]
            except Reject as e:
                if "unknown identifier" not in str(e):
                    raise
        if not pending_ia:
            break
    if pending_ia:
        raise Reject("unresolved initial assignments")
    # assignment rules: resolve iteratively (algebraic in states/constants)
    pending = dict(assign_rules)
    for _ in range(len(pending) + 1):
        for v, r in list(pending.items()):
            try:
                env[v] = mathml(list(r)[-1] if tag(list(r)[-1]) == "math" else list(r)[0], env, funcs)
                del pending[v]
            except Reject as e:
                if "unknown identifier" not in str(e):
                    raise
        if not pending:
            break
    if pending:
        raise Reject("unresolved assignment rules")
    rhs = {s: sp.Integer(0) for s in states}
    for r in find(model, "reaction"):
        kl = [k for k in r if tag(k) == "kineticLaw"]
        if not kl:
            raise Reject("reaction without kinetic law")
        local = {p.get("id"): sp.Float(float(p.get("value"))) for p in find(kl[0], "parameter") + find(kl[0], "localParameter")
                 if p.get("value") is not None}
        m = [k for k in kl[0] if tag(k) == "math"]
        rate = mathml(m[0], {**env, **local}, funcs)
        for lst, sign in (("listOfReactants", -1), ("listOfProducts", 1)):
            for L in [k for k in r if tag(k) == lst]:
                for sr in L:
                    sid = sr.get("species")
                    if sid in rhs:
                        st = float(sr.get("stoichiometry", "1"))
                        comp = comps.get(species[sid].get("compartment"), 1.0)
                        div = 1.0 if species[sid].get("hasOnlySubstanceUnits") == "true" else comp
                        rhs[sid] += sign * st * rate / div
    for v, r in rate_rules.items():
        if v in rhs:
            rhs[v] = mathml(list(r)[0], env, funcs)
    exprs = [rhs[s] for s in states]
    free = set().union(*[e.free_symbols for e in exprs])
    if not free <= set(X):
        raise Reject(f"free symbols {free - set(X)}")
    if all(sp.expand(e).is_polynomial(*X) for e in exprs):
        raise Reject("polynomial right-hand side")
    x0 = []
    for s in states:
        if s in init_vals:
            x0.append(init_vals[s])
            continue
        if s in species:
            sd = species[s]
            if sd.get("initialConcentration") is not None:
                v = float(sd.get("initialConcentration"))
            elif sd.get("initialAmount") is not None:
                v = float(sd.get("initialAmount"))
                if sd.get("hasOnlySubstanceUnits") != "true":
                    v /= comps.get(sd.get("compartment"), 1.0)
            else:
                raise Reject(f"state {s} without initial value")
        else:
            src = params.get(s)
            if src is None or src.get("value") is None:
                raise Reject(f"state {s} without initial value")
            v = float(src.get("value"))
        x0.append(v)
    return {"states": states, "exprs": [sp.srepr(e) for e in exprs], "x0": x0}, exprs


def numeric(exprs):
    X = sp.symbols(" ".join(f"x{i}" for i in range(len(exprs))))
    f = sp.lambdify(X, exprs, modules="numpy")
    def rhs(x):
        x = np.asarray(x, dtype=float)
        out = f(*[x[..., i] for i in range(len(exprs))])
        return np.stack([np.broadcast_to(np.asarray(o, dtype=float), x.shape[:-1]) for o in out], -1)
    return rhs


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def integrate(f, x0, t_eval, seconds=60):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        sol = solve_ivp(lambda t, x: f(x), (0, t_eval[-1]), x0, method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    except (Timeout, FloatingPointError, ValueError, ZeroDivisionError):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or sol.y.shape[1] != len(t_eval) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


def accept(x, dx):
    a, b = slice(0, len(x) // 2), slice(len(x) // 2, None)
    scale = np.abs(x).mean(0) + 1e-3
    if np.any(x[b].std(0) < 1e-3 * scale):
        return False
    rms = lambda d: np.sqrt((d**2).mean())
    return rms(dx[b]) >= 0.1 * rms(dx[a])


def transient_data(f, x0, rng):
    """Trial 23 transient rule: moves and settles from the published state; 8 instances from scaled initial states."""
    t = np.concatenate([[0.0], np.logspace(-4, 4, 4000)])
    traj = integrate(f, x0, t)
    if traj is None:
        raise Reject("transient pilot integration failed")
    dist = np.linalg.norm(traj - traj[-1], axis=1)
    if dist[0] < 1e-3 * (np.linalg.norm(x0) + 1e-6):
        raise Reject("transient: no motion")
    below = np.nonzero(dist < 0.01 * dist[0])[0]
    if len(below) == 0 or t[below[0]] <= 0:
        raise Reject("transient: does not settle within horizon")
    H = 1.5 * t[below[0]]
    t_eval = np.linspace(0, H, N_POINTS)
    xs, dxs = [], []
    for i in range(INSTANCES):
        for _ in range(4 if i else 1):
            start = x0 if i == 0 else x0 * rng.uniform(0.5, 1.5, size=len(x0))
            x = integrate(f, start, t_eval)
            if x is not None:
                break
        if x is None:
            raise Reject(f"transient instance {i} integration failed")
        xs.append(x.astype(np.float32)); dxs.append(f(x).astype(np.float32))
    return np.stack(xs), np.stack(dxs), float(H / (N_POINTS - 1)), float(t[below[0]])


def process(args):
    path, entry = args[:2]
    dmin, dmax = (args[2], args[3]) if len(args) > 2 else (3, 3)
    transient = args[4] if len(args) > 4 else False
    np.seterr(all="ignore")
    rec = {"id": entry["id"], "name": entry.get("name", "")}
    try:
        if EXCLUDE_TITLE.search(entry.get("name", "")):
            raise Reject("excluded by title rule")
        info, exprs = convert(path, dmin, dmax)
        f = numeric(exprs)
        x0 = np.array(info["x0"], dtype=float)
        if transient:
            try:
                return oscillator(entry, rec, info, f, x0)
            except Reject:
                rng = np.random.default_rng(int(hashlib.sha256(("t" + entry["id"]).encode()).hexdigest()[:8], 16))
                x, dx, h, ts = transient_data(f, x0, rng)
                rec.update(info, h=h, settle_time=ts, kind="transient", status="accepted")
                return rec, x, dx
        return oscillator(entry, rec, info, f, x0)
    except Exception as e:  # any failure on untrusted input is a recorded rejection, never a crash
        rec["status"] = f"rejected: {e}"[:200]
        return rec, None, None


def oscillator(entry, rec, info, f, x0):
    pilot = integrate(f, x0, np.linspace(0, 2000, 8001))
    if pilot is None:
        raise Reject("pilot integration failed")
    xp = pilot[4000:]
    speed = np.linalg.norm(f(xp), axis=1).mean()
    radius = np.linalg.norm(xp - xp.mean(0), axis=1).mean()
    if speed < 1e-9 or radius < 1e-9:
        raise Reject("no motion in pilot")
    P = 2 * np.pi * radius / speed
    t_eval = np.linspace(BURN * P, (BURN + SPAN) * P, N_POINTS)
    rng = np.random.default_rng(int(hashlib.sha256(entry["id"].encode()).hexdigest()[:8], 16))
    xs, dxs = [], []
    for i in range(INSTANCES):
        start = x0 if i == 0 else np.abs(xp[int(rng.integers(len(xp)))] * (1 + 0.05 * rng.normal(size=len(x0))))
        x = integrate(f, start, t_eval)
        if x is None:
            raise Reject(f"instance {i} integration failed")
        dx = f(x)
        if not accept(x, dx):
            raise Reject(f"instance {i}: no sustained motion")
        xs.append(x.astype(np.float32)); dxs.append(dx.astype(np.float32))
    rec.update(info, period=float(P), h=float(SPAN * P / (N_POINTS - 1)), kind="oscillator", status="accepted")
    return rec, np.stack(xs), np.stack(dxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dmin", type=int, default=3)
    ap.add_argument("--dmax", type=int, default=3)
    ap.add_argument("--transient", action="store_true", help="Trial 23: also accept settling transients")
    a = ap.parse_args()
    src, out = Path(a.sbml), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())["models"]
    jobs = [(src / m["file"], m, a.dmin, a.dmax, a.transient) for m in manifest if "sha256" in m]
    records = []
    with Pool(a.workers) as pool:
        for rec, x, dx in pool.imap_unordered(process, jobs):
            records.append(rec)
            if x is not None:
                np.savez(out / f"{rec['id']}.npz", x=x, dx=dx)
    acc = [r for r in records if r["status"] == "accepted"]
    reasons = {}
    for r in records:
        if r["status"] != "accepted":
            k = re.sub(r"\d+ state variables", "N state variables", r["status"].split(":", 1)[1].strip()[:40])
            reasons[k] = reasons.get(k, 0) + 1
    (out / "conversion.json").write_text(json.dumps({"records": sorted(records, key=lambda r: r["id"]),
                                                      "accepted": len(acc), "rejection_reasons": reasons}, indent=1))
    print("accepted", len(acc), "of", len(records))
    print(json.dumps(dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:20]), indent=1))


if __name__ == "__main__":
    main()
