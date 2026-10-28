#!/usr/bin/env python3
"""Frozen-policy P3 holdout; intentionally reuses the sealed P5 protocol."""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Execute a minimally patched in-memory copy so the scientific implementation
    # remains identical except for the prospectively declared degree 3.
    source = open(args.driver, "r", encoding="utf-8").read()
    source = source.replace("systems=m.make_systems(seed,5)", "systems=m.make_systems(seed,3)")
    source = source.replace("FROZEN_CONTEXT_POLICY_NOT_SUPPORTED_IID", "FROZEN_CONTEXT_POLICY_NOT_SUPPORTED_P3")
    source = source.replace("FROZEN_CONTEXT_POLICY_SUPPORTED_IID_DEVELOPMENT_ONLY", "FROZEN_CONTEXT_POLICY_SUPPORTED_P3_DEVELOPMENT_ONLY")
    old = sys.argv
    try:
        sys.argv = [args.driver, "--driver", args.driver.replace("learned_response_acquisition_policy_v1_iid_holdout.py", "learned_response_acquisition_policy_v1.py"), "--policy", args.policy, "--plan", args.plan, "--out", args.out]
        ns = {"__name__": "__main__", "__file__": args.driver}
        exec(compile(source, args.driver, "exec"), ns)
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
