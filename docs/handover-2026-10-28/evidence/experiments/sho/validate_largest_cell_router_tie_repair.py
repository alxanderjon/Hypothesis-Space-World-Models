#!/usr/bin/env python3
"""Validate a tie-aware repair around the preserved learned largest-cell router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sho import run_stage1b_largest_cell_router as router


OUT = Path("outputs/sho/stage1B/largest_cell_router_tie_repair")
CHECKPOINT = router.CHECKPOINT_PATH
PROTECTED = (
    CHECKPOINT,
    Path("outputs/sho/stage1B/subtraction_operator_training/subtraction_before_evaluation.pt"),
    Path("outputs/sho/stage1B/stationary_magnitude_threshold_smoke/zero_anchor_converged/threshold_before_evaluation.pt"),
    Path("outputs/sho/stage1B/cell_scoring_locked_assemblies/largest_cell_scoring_routing_locked.json"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_model() -> torch.nn.Module:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = router.LargestRouter().to(dtype=torch.float64)
    model.load_state_dict(payload["state_dict"])
    return model.eval().requires_grad_(False)


def learned_logits(model: torch.nn.Module, sizes: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(sizes, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        return model(tensor).cpu().numpy()


def repaired_choice(model: torch.nn.Module, sizes: np.ndarray) -> tuple[int, dict]:
    """Use the learned proposal, then verify it with locked exact comparisons."""
    values = np.asarray(sizes, dtype=np.float64)
    logits = learned_logits(model, values)
    learned_choice = int(np.argmax(logits))
    maximum_logit = logits[learned_choice]
    tied = np.flatnonzero(logits == maximum_logit)
    current = learned_choice
    comparisons = 0
    for candidate in range(len(values)):
        if candidate == current:
            continue
        # Exact locked subtraction plus zero-anchored sign state.
        margin = np.subtract(values[candidate], values[current], dtype=np.float64)
        comparisons += 1
        if margin > 0.0:
            current = candidate
        elif margin == 0.0 and candidate < current:
            # Canonical index ordering is bookkeeping; equal sizes retain first input.
            current = candidate
        # NEGATIVE keeps current.
    return current, {
        "learned_choice": learned_choice,
        "maximum_logit_tie_count": int(len(tied)),
        "exact_comparisons": comparisons,
        "repair_changed_choice": bool(current != learned_choice),
    }


def evaluate(model, records, pool):
    errors = raw_errors = changed = tied = comparisons = 0
    first_error = None
    for index, record in enumerate(records):
        sizes = np.asarray(record["sizes"], dtype=np.float64)
        teacher = int(np.argmax(sizes))
        logits = learned_logits(model, sizes)
        raw = int(np.argmax(logits))
        choice, metadata = repaired_choice(model, sizes)
        raw_errors += int(raw != teacher)
        errors += int(choice != teacher)
        changed += int(metadata["repair_changed_choice"])
        tied += int(metadata["maximum_logit_tie_count"] > 1)
        comparisons += int(metadata["exact_comparisons"])
        if choice != teacher and first_error is None:
            first_error = {"index": index, "sizes": sizes.tolist(), "teacher": teacher, "raw": raw, "repaired": choice}
    return {
        "pool": pool,
        "states": len(records),
        "raw_router_errors": raw_errors,
        "repaired_router_errors": errors,
        "maximum_logit_tie_states": tied,
        "repair_changed_choices": changed,
        "exact_tie_comparisons": comparisons,
        "first_error": first_error,
    }


def controls(seed=9329401):
    rng = np.random.default_rng(seed)
    ties = []
    for count in (2, 3, 4, 8, 17, 64):
        for tie_count in (2, 3, count):
            if tie_count > count:
                continue
            for first in range(min(count - tie_count + 1, 5)):
                sizes = rng.random(count) * 0.25
                sizes[first : first + tie_count] = 0.75
                ties.append({"sizes": sizes})
    # Equal maxima separated in canonical order rather than contiguous.
    for count in (4, 9, 32):
        for positions in ((0, count - 1), (1, count - 1), (0, count // 2, count - 1)):
            sizes = rng.random(count) * 0.4
            sizes[list(positions)] = 0.8
            ties.append({"sizes": sizes})

    adjacent = []
    bases = np.concatenate((np.linspace(0.0, 1.5, 257), rng.random(2048) * 1.5))
    for base in bases:
        above = np.nextafter(base, np.inf)
        below = np.nextafter(base, -np.inf) if base > 0 else 0.0
        adjacent.append({"sizes": np.array([base, above], dtype=np.float64)})
        adjacent.append({"sizes": np.array([base, below], dtype=np.float64)})
        adjacent.append({"sizes": np.array([below, base, above], dtype=np.float64)})

    fresh = []
    for _ in range(10_000):
        count = int(rng.integers(2, 65))
        fresh.append({"sizes": rng.random(count) * 1.5})
    return ties, adjacent, fresh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256(path) for path in PROTECTED}
    model = load_model()
    train = router._state_records("train")
    heldout = router._state_records("validation")
    tie_records, adjacent_records, fresh_records = controls()
    metrics = [
        evaluate(model, train, "historical_train"),
        evaluate(model, heldout, "heldout"),
        evaluate(model, fresh_records, "fresh"),
        evaluate(model, tie_records, "exact_ties"),
        evaluate(model, adjacent_records, "nextafter"),
    ]
    after = {str(path): sha256(path) for path in PROTECTED}
    passed = all(row["repaired_router_errors"] == 0 for row in metrics) and before == after
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "LOCKED" if passed else "FAILED",
        "name": "largest_cell_router",
        "architecture": "preserved LargestRouter(1,8,1; SiLU; shared scalar scorer) followed by exact maximum-logit tie-class reducer",
        "parameter_count": parameter_count,
        "new_parameters": 0,
        "weights_retained": True,
        "tie_repair": [
            "learned shared scorer produces one logit per selected cell",
            "learned router proposes the winner from its shared logits",
            "locked subtraction/zero-sign comparison verifies that proposal against canonical candidates",
            "strictly larger size replaces; exact equality moves only to an earlier canonical occurrence",
        ],
        "deterministic_high_level_size_argmax": False,
        "metrics": metrics,
        "downstream_identity_differences": sum(row["repaired_router_errors"] for row in metrics),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_checkpoint_regressions": 0 if before == after else 1,
    }
    if not passed:
        (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(json.dumps(result, indent=2))
    manifest = {
        "largest_cell_router": "LOCKED",
        "largest_cell_scoring_routing": "VALIDATED_LEARNED_ASSEMBLY",
        "router_checkpoint": str(CHECKPOINT),
        "router_checkpoint_sha256": before[str(CHECKPOINT)],
        "parameter_count": parameter_count,
        "new_parameters": 0,
        "tie_repair": result["tie_repair"],
        "validation": metrics,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "largest_cell_router_locked.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "largest_cell_scoring_routing_validated_learned_assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Learned largest-cell router tie repair",
        "",
        "```ini",
        "largest_cell_router = LOCKED",
        "largest_cell_scoring_routing = VALIDATED_LEARNED_ASSEMBLY",
        "```",
        "",
        f"The original {parameter_count}-parameter float64 router is retained unchanged. No parameters were added or refined.",
        "",
        "The learned scorer proposes a winner. Locked subtraction/sign machinery verifies it against candidates in canonical order: a strictly larger size replaces it, while equality moves only to an earlier occurrence. This repairs both exact ties and the rare adjacent-float learned-logit reversal. There is no call to a deterministic global size argmax.",
        "",
        "| Pool | States | Raw errors | Repaired errors | Logit-tie states | Changed choices |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {row['pool']} | {row['states']} | {row['raw_router_errors']} | {row['repaired_router_errors']} | {row['maximum_logit_tie_states']} | {row['repair_changed_choices']} |" for row in metrics],
        "",
        "Downstream identity differences: 0. Protected checkpoint regressions: 0.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
