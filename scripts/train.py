#!/usr/bin/env python
"""Train the Layer 5 SAC driver and report real driving numbers.

Usage:
    python scripts/train.py --steps 50000 --out runs/sac_l5 --seed 42
    python scripts/train.py --resume runs/sac_l5 --steps 25000 --out runs/sac_l5

Trains on the benign Layer 5 preset (dry / mediums / fixed setup, randomized
start pose), saves model + VecNormalize stats + config to --out, then
evaluates the trained policy AND a random baseline on identical spread
rolling starts. Layer 6 extends this script with curriculum, weather/setup
domain randomization and long-run logging.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.agents.sac_driver import (BENIGN_EVAL, SACDriver, SACDriverConfig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50_000,
                   help="SAC policy steps (x action_repeat env steps)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--out", type=str, default="runs/sac_l5",
                   help="checkpoint + csv-log directory")
    p.add_argument("--resume", type=str, default=None,
                   help="directory of a previous save() to continue from")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="also snapshot every N policy steps")
    p.add_argument("--eval-episodes", type=int, default=5)
    args = p.parse_args()

    if args.resume:
        print(f"resuming from {args.resume}")
        driver = SACDriver.load(args.resume)
    else:
        driver = SACDriver(SACDriverConfig(seed=args.seed, n_envs=args.n_envs))

    driver.train(args.steps, log_dir=args.out,
                 checkpoint_every=args.checkpoint_every,
                 progress_every=max(args.steps // 20, 1))
    out = driver.save(args.out)
    print(f"saved -> {out}")

    eval_opts = dict(BENIGN_EVAL, v0=30.0)  # survivable at every spread start
    print("\ntrained policy (deterministic, spread rolling starts):")
    print(driver.evaluate(args.eval_episodes, options=eval_opts,
                          spread_starts=True).table())
    print("\nrandom baseline (same starts, same control rate):")
    print(driver.evaluate(args.eval_episodes, options=eval_opts,
                          spread_starts=True, policy="random").table())
    driver.close()


if __name__ == "__main__":
    main()
