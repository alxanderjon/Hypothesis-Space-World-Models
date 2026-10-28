"""Refresh the Stage 1B K-landscape HTML report from saved metrics."""

from __future__ import annotations

import json

from experiments.sho.run_stage1b_k_landscape_and_basin_analysis import OUTPUT_DIR, _write_html


def main() -> None:
    payload = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    _write_html(
        payload["single_trajectory_landscapes"],
        payload["basin_widths"],
        payload["dense_perturbation_summary"],
        payload["basin_sweep_summary"],
    )


if __name__ == "__main__":
    main()
