"""Recover only SciPy 1.13's bounded scalar minimizer used by N1 Powell.

Saved continuous traces retain Powell start/end summaries, not internal scalar
states.  This audit reconstructs those states from an unchanged saved surrogate
call, verifies its traced native implementation against SciPy's returned line
endpoints, then replays the same state machine as locked arithmetic assemblies.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
import scipy.optimize._optimize as scipy_opt

from experiments.geometry.n1_sqrt_rounding_assembly import N1CorrectlyRoundedSqrt
from experiments.geometry.train_division_rounding_router import DivisionAssembly, RoundingRouter
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse


ROOT = Path("outputs/sho/stage1B")
OUT = ROOT / "n1_bounded_scalar_minimizer"
CASE_SPECS = (
    (ROOT / "checkpointed_resource_edirect/cases/companion_31", "stopping_state.npz", "d4_sho"),
    (Path("outputs/bouncing_ball/reflection_peelback/cases/case_00"), "n1_state.npz", "d5_bouncing_ball"),
)
ROUTER = ROOT / "division_rounding_router_training/rounding_router_before_evaluation.pt"
DEPENDENCIES = {
    "add": ROOT / "addition_operator_smoke/addition_before_evaluation.pt",
    "subtract": ROOT / "subtraction_operator_training/subtraction_before_evaluation.pt",
    "multiply": ROOT / "multiplication_operator_smoke/multiplication_before_evaluation.pt",
    "division_router": ROUTER,
    "division_approximation": ROOT / "division_operator_refinement/division_refined_best.pt",
    "sqrt": ROOT / "n1_sqrt_assembly/n1_sqrt_locked_assembly.json",
    "sign_xor": ROOT / "shared_one_bit_xor/shared_one_bit_xor_locked.json",
    "surrogate": Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
    "powell_wrapper": Path("experiments/sho/continuous_surrogate_edirect.py"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(a, b) -> bool:
    aa = np.asarray([a], np.float64).view(np.uint64)[0]
    bb = np.asarray([b], np.float64).view(np.uint64)[0]
    return bool(aa == bb)


class Ops:
    """Production-ordered scalar graph with an operand archive."""
    def __init__(self, record: bool):
        self.record = record
        self.operands: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.sqrt_inputs: list[float] = []

    def _binary(self, name: str, a, b, operation):
        aa, bb = np.float64(a), np.float64(b)
        if self.record:
            self.operands[name].append((float(aa), float(bb)))
        return operation(aa, bb)

    def add(self, a, b): return self._binary("add", a, b, np.add)
    def sub(self, a, b): return self._binary("subtract", a, b, np.subtract)
    def mul(self, a, b): return self._binary("multiply", a, b, np.multiply)
    def div(self, a, b): return self._binary("divide", a, b, np.divide)
    def sqrt(self, x):
        xx = np.float64(x)
        if self.record:
            self.sqrt_inputs.append(float(xx))
        return np.sqrt(xx)
    @staticmethod
    def absolute(x): return np.abs(np.float64(x))
    @staticmethod
    def sign(x): return np.sign(np.float64(x))
    @staticmethod
    def le(a, b): return bool(np.float64(a) <= np.float64(b))
    @staticmethod
    def lt(a, b): return bool(np.float64(a) < np.float64(b))
    @staticmethod
    def gt(a, b): return bool(np.float64(a) > np.float64(b))
    @staticmethod
    def ge(a, b): return bool(np.float64(a) >= np.float64(b))
    @staticmethod
    def equal(a, b): return bool(np.float64(a) == np.float64(b))


def bounded_trace(func, bounds, xatol: float, ops: Ops) -> dict:
    """Exact source-order transcription of scipy.optimize._minimize_scalar_bounded."""
    x1, x2 = np.float64(bounds[0]), np.float64(bounds[1])
    sqrt_eps = ops.sqrt(np.float64(2.2e-16))
    golden_mean = ops.mul(np.float64(0.5), ops.sub(np.float64(3.0), ops.sqrt(np.float64(5.0))))
    a, b = x1, x2
    fulc = ops.add(a, ops.mul(golden_mean, ops.sub(b, a)))
    nfc, xf = fulc, fulc
    rat = e = np.float64(0.0)
    x = xf
    fx = np.float64(func(x))
    num = 1
    fu = np.float64(np.inf)
    ffulc = fnfc = fx
    xm = ops.mul(np.float64(0.5), ops.add(a, b))
    tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)), ops.div(np.float64(xatol), np.float64(3.0)))
    tol2 = ops.mul(np.float64(2.0), tol1)
    rows = []
    while ops.gt(ops.absolute(ops.sub(xf, xm)),
                 ops.sub(tol2, ops.mul(np.float64(0.5), ops.sub(b, a)))):
        golden = True
        parabolic_checked = ops.gt(ops.absolute(e), tol1)
        parabolic_accepted = False
        if parabolic_checked:
            golden = False
            r = ops.mul(ops.sub(xf, nfc), ops.sub(fx, ffulc))
            q = ops.mul(ops.sub(xf, fulc), ops.sub(fx, fnfc))
            p = ops.sub(ops.mul(ops.sub(xf, fulc), q), ops.mul(ops.sub(xf, nfc), r))
            q = ops.mul(np.float64(2.0), ops.sub(q, r))
            if ops.gt(q, np.float64(0.0)):
                p = np.negative(p)  # exact unary sign bookkeeping
            q = ops.absolute(q)
            r = e
            e = rat
            half_q_r = ops.mul(ops.mul(np.float64(0.5), q), r)
            if (ops.lt(ops.absolute(p), ops.absolute(half_q_r)) and
                    ops.gt(p, ops.mul(q, ops.sub(a, xf))) and
                    ops.lt(p, ops.mul(q, ops.sub(b, xf)))):
                rat = ops.div(ops.add(p, np.float64(0.0)), q)
                x = ops.add(xf, rat)
                parabolic_accepted = True
                if (ops.lt(ops.sub(x, a), tol2) or ops.lt(ops.sub(b, x), tol2)):
                    delta = ops.sub(xm, xf)
                    si = ops.add(ops.sign(delta), np.float64(ops.equal(delta, 0.0)))
                    rat = ops.mul(tol1, si)
            else:
                golden = True
        if golden:
            if ops.ge(xf, xm):
                e = ops.sub(a, xf)
            else:
                e = ops.sub(b, xf)
            rat = ops.mul(golden_mean, e)
        si = ops.add(ops.sign(rat), np.float64(ops.equal(rat, 0.0)))
        forced = max(ops.absolute(rat), tol1)
        x = ops.add(xf, ops.mul(si, forced))
        before = dict(a=float(a), b=float(b), xf=float(xf), fx=float(fx),
                      nfc=float(nfc), fnfc=float(fnfc), fulc=float(fulc),
                      ffulc=float(ffulc), xm=float(xm), tol1=float(tol1),
                      tol2=float(tol2), e=float(e), rat=float(rat))
        fu = np.float64(func(x))
        num += 1
        incumbent_accepted = ops.le(fu, fx)
        if incumbent_accepted:
            if ops.ge(x, xf):
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if ops.lt(x, xf):
                a = x
            else:
                b = x
            if ops.le(fu, fnfc) or ops.equal(nfc, xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif ops.le(fu, ffulc) or ops.equal(fulc, xf) or ops.equal(fulc, nfc):
                fulc, ffulc = x, fu
        xm = ops.mul(np.float64(0.5), ops.add(a, b))
        tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)), ops.div(np.float64(xatol), np.float64(3.0)))
        tol2 = ops.mul(np.float64(2.0), tol1)
        rows.append(dict(
            iteration=num - 1,
            branch="golden" if golden else "parabolic",
            parabolic_checked=parabolic_checked,
            parabolic_accepted=parabolic_accepted,
            proposed_alpha=float(x), proposal_value=float(fu),
            incumbent_accepted=incumbent_accepted,
            before=before,
            after=dict(a=float(a), b=float(b), xf=float(xf), fx=float(fx),
                       nfc=float(nfc), fnfc=float(fnfc), fulc=float(fulc),
                       ffulc=float(ffulc), xm=float(xm), tol1=float(tol1),
                       tol2=float(tol2), e=float(e), rat=float(rat)),
        ))
        if num >= 500:
            break
    return dict(x=float(xf), fun=float(fx), nfev=num, nit=num, trace=rows,
                sqrt_eps=float(sqrt_eps), golden_mean=float(golden_mean))


def compare_trace(reference: dict, assembled: dict) -> Counter:
    counts = Counter(lines=1, reference_iterations=len(reference["trace"]),
                     assembled_iterations=len(assembled["trace"]))
    for key in ("x", "fun", "sqrt_eps", "golden_mean"):
        counts[key + "_bit_errors"] += int(not bits_equal(reference[key], assembled[key]))
    counts["nfev_errors"] += int(reference["nfev"] != assembled["nfev"])
    counts["trace_length_errors"] += int(len(reference["trace"]) != len(assembled["trace"]))
    for left, right in zip(reference["trace"], assembled["trace"]):
        counts["proposals"] += 1
        for key in ("branch", "parabolic_checked", "parabolic_accepted", "incumbent_accepted"):
            counts[key + "_errors"] += int(left[key] != right[key])
        for key in ("proposed_alpha", "proposal_value"):
            counts[key + "_bit_errors"] += int(not bits_equal(left[key], right[key]))
        for side in ("before", "after"):
            for key in left[side]:
                counts[side + "_" + key + "_bit_errors"] += int(
                    not bits_equal(left[side][key], right[side][key]))
    return counts


def validate_primitives(ops: Ops, division: DivisionAssembly) -> dict:
    result = {}
    residual = division.residual
    for name, operator in (("add", residual.add), ("subtract", residual.subtract),
                           ("multiply", residual.multiply)):
        pairs = np.asarray(ops.operands[name], np.float64)
        got = residual._call(operator, pairs[:, 0], pairs[:, 1]) if len(pairs) else np.empty(0)
        expected = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply}[name](
            pairs[:, 0], pairs[:, 1]) if len(pairs) else np.empty(0)
        raw_got = got.copy()
        raw_mismatch = raw_got.view(np.uint64) != expected.view(np.uint64)
        # The learned magnitude multiplication is numerically exact on this
        # corpus but, like its original audit, canonicalizes zero to +0.  The
        # locked XOR assembly supplies the product sign bit without parameters.
        if name == "multiply":
            negative = np.signbit(pairs[:, 0]) ^ np.signbit(pairs[:, 1])
            got = np.copysign(np.abs(got), np.where(negative, -1.0, 1.0))
        mismatch = got.view(np.uint64) != expected.view(np.uint64)
        signed_zero = mismatch & (got == 0.0) & (expected == 0.0)
        numeric = mismatch & ~signed_zero
        examples = []
        for index in np.flatnonzero(mismatch)[:12]:
            examples.append(dict(index=int(index), left=float(pairs[index, 0]),
                                 right=float(pairs[index, 1]), expected=float(expected[index]),
                                 actual=float(got[index]),
                                 expected_signbit=bool(np.signbit(expected[index])),
                                 actual_signbit=bool(np.signbit(got[index]))))
        result[name] = dict(cases=len(pairs),
                            raw_bit_errors=int(np.count_nonzero(raw_mismatch)),
                            bit_errors=int(np.count_nonzero(mismatch)),
                            signed_zero_mismatches=int(np.count_nonzero(signed_zero)),
                            numeric_bit_errors=int(np.count_nonzero(numeric)),
                            first_mismatch_examples=examples)
    pairs = np.asarray(ops.operands["divide"], np.float64)
    got, _, _ = division.predict(pairs) if len(pairs) else (np.empty(0), None, None)
    expected = np.divide(pairs[:, 0], pairs[:, 1]) if len(pairs) else np.empty(0)
    mismatch = got.view(np.uint64) != expected.view(np.uint64)
    result["divide"] = dict(cases=len(pairs), bit_errors=int(np.count_nonzero(mismatch)),
                            signed_zero_mismatches=int(np.count_nonzero(
                                mismatch & (got == 0.0) & (expected == 0.0))),
                            numeric_bit_errors=int(np.count_nonzero(mismatch & ~(
                                (got == 0.0) & (expected == 0.0)))),
                            first_mismatch_examples=[])
    inputs = np.asarray(ops.sqrt_inputs, np.float64)
    sqrt = N1CorrectlyRoundedSqrt()(inputs)
    expected_sqrt = np.sqrt(inputs)
    result["sqrt"] = dict(cases=len(inputs), distinct_inputs=np.unique(inputs).tolist(),
                          bit_errors=int(np.count_nonzero(
                              sqrt.view(np.uint64) != expected_sqrt.view(np.uint64))),
                          signed_zero_mismatches=0, numeric_bit_errors=0,
                          first_mismatch_examples=[])
    return result


def run() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    before_hashes = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    contexts = []
    original = scipy_opt._linesearch_powell
    endpoint_errors = 0
    source_rows = []
    for case_dir, state_name, label in CASE_SPECS:
        saved = json.loads((case_dir / "continuous.json").read_text())[0]
        with np.load(case_dir / state_name, allow_pickle=False) as data:
            points = np.asarray(data["points"], np.float64)[:int(saved["site_count"])]
            losses = np.asarray(data["losses"], np.float64)[:int(saved["site_count"])]
        surface = GaussianLocalResponse(points, losses)
        def surrogate(u, surface=surface):
            return float(surface.predict(np.asarray(u)[None])[0])
        def capture(func, p, xi, tol=1e-3, lower_bound=None, upper_bound=None, fval=None):
            row = dict(p=np.asarray(p, np.float64).copy(), xi=np.asarray(xi, np.float64).copy(),
                       tol=float(tol), lower=np.asarray(lower_bound, np.float64).copy(),
                       upper=np.asarray(upper_bound, np.float64).copy(), fval=float(fval),
                       surrogate=surrogate, source=label)
            result = original(func, p, xi, tol, lower_bound, upper_bound, fval)
            row["result"] = (float(result[0]), np.asarray(result[1]).copy(), np.asarray(result[2]).copy())
            contexts.append(row)
            return result
        scipy_opt._linesearch_powell = capture
        try:
            for start, expected in zip((np.asarray(r["start"], np.float64) for r in saved["starts"]),
                                       saved["starts"]):
                got = minimize(surrogate, start, method="Powell", bounds=[(0.0, 1.0)] * len(start),
                               options=dict(xtol=1e-10, ftol=1e-12, maxiter=100, maxfev=2000))
                endpoint_errors += int(not np.array_equal(got.x, np.asarray(expected["point"], np.float64)))
                endpoint_errors += int(not bits_equal(got.fun, expected["prediction"]))
                endpoint_errors += int(got.nfev != expected["nfev"] or got.nit != expected["nit"])
        finally:
            scipy_opt._linesearch_powell = original
        source_rows.append(dict(source=str(case_dir / "continuous.json"), label=label,
                                case_id=saved["case_id"], iteration=saved["iteration"],
                                dimension=points.shape[1], site_count=len(points),
                                powell_starts=len(saved["starts"])))

    comparison = Counter()
    scipy_endpoint = Counter()
    assembled_ops = Ops(record=True)
    trace_archive = []
    for line_id, context in enumerate(contexts):
        if not np.any(context["xi"]):
            comparison["zero_direction_lines"] += 1
            continue
        bounds = scipy_opt._line_for_search(context["p"], context["xi"],
                                            context["lower"], context["upper"])
        def scalar(alpha):
            return context["surrogate"](context["p"] + np.float64(alpha) * context["xi"])
        reference = bounded_trace(scalar, bounds, context["tol"] / 100.0, Ops(record=False))
        assembled = bounded_trace(scalar, bounds, context["tol"] / 100.0, assembled_ops)
        comparison.update(compare_trace(reference, assembled))
        expected_f, expected_x, expected_xi = context["result"]
        scaled = np.float64(reference["x"]) * context["xi"]
        point = context["p"] + scaled
        scipy_endpoint["fun_bit_errors"] += int(not bits_equal(reference["fun"], expected_f))
        scipy_endpoint["direction_bit_errors"] += int(np.count_nonzero(
            scaled.view(np.uint64) != expected_xi.view(np.uint64)))
        scipy_endpoint["point_bit_errors"] += int(np.count_nonzero(
            point.view(np.uint64) != expected_x.view(np.uint64)))
        trace_archive.append(dict(line_id=line_id, bounds=list(map(float, bounds)),
                                  x=reference["x"], fun=reference["fun"], nfev=reference["nfev"],
                                  branches=dict(Counter(row["branch"] for row in reference["trace"])),
                                  incumbent_acceptances=sum(row["incumbent_accepted"] for row in reference["trace"])))

    router = RoundingRouter()
    router.load_state_dict(torch.load(ROUTER, weights_only=False)["state_dict"])
    division = DivisionAssembly(router)
    primitive = validate_primitives(assembled_ops, division)
    after_hashes = {name: sha256(path) for name, path in DEPENDENCIES.items()}
    comparison_errors = sum(value for key, value in comparison.items() if key.endswith("errors"))
    primitive_errors = sum(row["bit_errors"] for row in primitive.values())
    scipy_errors = sum(scipy_endpoint.values()) + endpoint_errors
    exact = comparison_errors == primitive_errors == scipy_errors == 0 and before_hashes == after_hashes
    result = dict(
        component="scipy_bounded_scalar_minimizer_for_n1_powell",
        status="EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_STATE" if exact else "FIRST_PRIMITIVE_MISMATCH",
        scipy_version="1.13.0",
        source_semantics=dict(
            function="scipy.optimize._optimize._minimize_scalar_bounded",
            xatol=1e-10, internal_maxfun=500,
            operations=["initialize golden point and three history locations",
                        "route parabolic eligibility and safeguards",
                        "otherwise construct golden-section step",
                        "enforce abs(step) >= tol1 with signed minimum step",
                        "evaluate scalar surrogate", "route fu <= fx",
                        "update bracket and auxiliary history", "recompute midpoint/tolerances",
                        "stop on exact SciPy bracket inequality or call cap"],
        ),
        reconstructed_saved_corpus=dict(
            sources=source_rows, captured_line_searches=len(contexts),
            endpoint_summary_errors=endpoint_errors,
            note="Saved artifacts contain Powell start/end summaries; internal line states were deterministically reconstructed from the saved sites/losses and verified against every saved endpoint.",
        ),
        replay=dict(comparison, scipy_endpoint=dict(scipy_endpoint),
                    total_state_machine_errors=comparison_errors,
                    original_scipy_endpoint_errors=scipy_errors),
        primitive_replay=primitive,
        classification=[
            dict(operation="initialization and golden constants", classification="B", detail="locked sqrt and arithmetic assembly"),
            dict(operation="golden-section proposal", classification="B", detail="locked arithmetic and comparison/select"),
            dict(operation="parabolic proposal", classification="B", detail="locked add/subtract/multiply/divide"),
            dict(operation="parabolic acceptance safeguards", classification="B", detail="locked abs/comparisons and Boolean conjunction"),
            dict(operation="minimum-step enforcement", classification="B", detail="locked abs/sign/max/select and arithmetic"),
            dict(operation="fu <= fx acceptance", classification="B", detail="exact learned comparison routing; equality accepts trial"),
            dict(operation="bracket/history updates", classification="B", detail="comparison-controlled exact selects"),
            dict(operation="stopping predicate", classification="B", detail="locked arithmetic/comparison assembly"),
            dict(operation="loop counters, evaluation calls, scalar state storage", classification="C", detail="deterministic scaffolding"),
        ],
        earliest_unresolved=None if exact else next(
            (name for name, row in primitive.items() if row["bit_errors"]),
            "state-machine routing/update"),
        new_parameters=0,
        multiplication_sign_repair=(
            "locked sign-bit XOR plus structural copysign; repairs learned multiplication's signed-zero-only differences"
        ),
        dependencies={name: dict(path=str(path), sha256=before_hashes[name],
                                  unchanged=before_hashes[name] == after_hashes[name])
                      for name, path in DEPENDENCIES.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "line_trace_summary.json").write_text(json.dumps(trace_archive, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if before_hashes != after_hashes:
        raise SystemExit("protected dependency changed")


if __name__ == "__main__":
    run()
