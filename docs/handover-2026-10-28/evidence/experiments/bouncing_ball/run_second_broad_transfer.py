#!/usr/bin/env python3
"""Broad-domain invocation of the unchanged first K_ball transfer stack."""
from __future__ import annotations

import os

os.environ["BALL_TRANSFER_DOMAIN_LOWER"] = "0.5"
os.environ["BALL_TRANSFER_DOMAIN_UPPER"] = "1.5"
os.environ["BALL_TRANSFER_SEED"] = "20261001"
os.environ["BALL_TRANSFER_TRAJECTORIES"] = "8"
os.environ["BALL_TRANSFER_OUTPUT"] = "outputs/bouncing_ball/second_broad_n1_n2_n3_transfer"

from experiments.bouncing_ball.run_first_transfer import main


if __name__ == "__main__":
    main()
