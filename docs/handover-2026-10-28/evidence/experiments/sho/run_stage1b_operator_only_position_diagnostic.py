"""Stage 1B position-only operator-consistency diagnostic.

This run deliberately avoids velocity inputs, omega labels, coefficient labels,
and the SHO analytic decoder. The model sees only a position window and learns a
trajectory-conditioned one-step operator that can be rolled forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import experiments.sho.run_stage1b_generic_operator_diagnostics as generic


OUTPUT_DIR = Path("outputs/sho/stage1B/operator_only_position_diagnostic")


def main() -> None:
    generic.OUTPUT_DIR = OUTPUT_DIR
    generic.MAX_EPOCHS = 10_000
    generic.DIAGNOSTIC_EPOCH_CUTOFF = 10_000
    generic.PATIENCE = 2_000
    generic.EVAL_EVERY = 100
    generic.LOG_EVERY = 500
    generic.BATCH_SIZE = generic.NUM_TRAIN

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences = generic._make_position_sequences(generic.NUM_TRAIN, seed=generic.SEED)
    test_sequences = generic._make_position_sequences(generic.NUM_TEST, seed=generic.SEED + 1)
    metrics = generic._run_variant(
        "operator_consistency_only",
        train_sequences,
        test_sequences,
    )

    summary_path = OUTPUT_DIR / "operator_only_position_diagnostic_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
