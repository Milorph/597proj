#!/usr/bin/env python3
"""
Entry point for the two-stage IoT IDS.

Examples
--------
    python run.py                      # demo scale (fast, ~1-3 min)
    python run.py --scale full         # spec scale: 200k benign / 4k-6.2k attack
    python run.py --scale smoke         # tiny, for CI / quick checks
    python run.py --seed 7             # different random composition
    python run.py --no-flow-unsup      # skip optional Task 3.1

If the real CIC IoT-DIAD 2024 CSVs are placed under ./data (see README) they are
used automatically; otherwise a schema-faithful synthetic population is used.
"""
import argparse

from src import pipeline


def main():
    ap = argparse.ArgumentParser(description="Two-stage IoT IDS pipeline")
    ap.add_argument("--scale", choices=list(pipeline.SCALES.keys()), default="demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-flow-unsup", action="store_true",
                    help="skip optional Task 3.1 flow-level unsupervised stage")
    args = ap.parse_args()
    pipeline.run(scale=args.scale, seed=args.seed, do_flow_unsup=not args.no_flow_unsup)


if __name__ == "__main__":
    main()
