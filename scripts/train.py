#!/usr/bin/env python
"""Train the racing driver: Layer 5 benign runs and the Layer 6 curriculum.

Recipes
    l5_benign      flat SAC on the benign preset (the Layer 5 behavior; default)
    l6_generalist  the Layer 6 curriculum (A_benign_laps -> B1_setup_near ->
                   B2_setup_full -> C_full_dr): tensorboard logging, rotating
                   checkpoints with the replay buffer, gate-driven stage
                   advancement, per-stage best-policy keeping.

Usage
    # Layer 5 smoke train (unchanged from Layer 5)
    python scripts/train.py --steps 50000 --out runs/sac_l5 --seed 42

    # Layer 6 generalist (hours of CPU at full budget; resumable any time)
    python scripts/train.py --recipe l6_generalist --out runs/l6_generalist
    # ... window/container died? continue exactly where it stopped:
    python scripts/train.py --recipe l6_generalist --out runs/l6_generalist --resume

    # whole-pipeline dry run (~40 min): scale every stage budget by 0.1
    python scripts/train.py --recipe l6_generalist --budget-scale 0.1 --out runs/l6_dry

    # reports on a trained driver (writes run_report.json + reports/layer6_report.md)
    python scripts/train.py --report runs/l6_generalist/final
    python scripts/train.py --compare runs/l6_generalist/final --specialist-steps 250000

Colab runbook (~12 h windows)
    !pip install -r requirements.txt
    !python scripts/train.py --recipe l6_generalist --out runs/l6_generalist
    # after a window reset, the SAME command plus --resume picks up the
    # rotating checkpoint (model + replay buffer + obs normalization + stage
    # counters). Keep runs/ on Drive to survive resets:
    #   --out /content/drive/MyDrive/f1/runs/l6_generalist
    # tensorboard: %load_ext tensorboard ; %tensorboard --logdir <out>/tb
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.agents.curriculum import (HELD_OUT_PANEL, VALIDATION_PANEL,
                                   CurriculumTrainer, build_report,
                                   compare_generalist_vs_specialists,
                                   evaluate_distribution, evaluate_panel,
                                   l6_default_curriculum, l6_driver_config,
                                   render_markdown)
from src.agents.sac_driver import BENIGN_EVAL, SACDriver, SACDriverConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", choices=("l5_benign", "l6_generalist"),
                   default="l5_benign")
    p.add_argument("--steps", type=int, default=50_000,
                   help="l5_benign: SAC policy steps")
    p.add_argument("--budget-scale", type=float, default=1.0,
                   help="l6: multiply every stage budget (0.1 = dry run)")
    p.add_argument("--step-limit", type=int, default=None,
                   help="l6: pause the run after N total policy steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--out", type=str, default="runs/sac_l5",
                   help="run directory (checkpoints, logs, final driver)")
    p.add_argument("--resume", nargs="?", const="", default=None,
                   metavar="DIR",
                   help="resume a previous run (bare flag: resume from --out)")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="l5_benign: also snapshot every N policy steps")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--no-tb", action="store_true",
                   help="l6: disable tensorboard logging (csv only)")
    p.add_argument("--report", type=str, default=None, metavar="DRIVER_DIR",
                   help="evaluate a saved driver: validation panel + lap-time "
                        "distribution -> run_report.json + reports/")
    p.add_argument("--compare", type=str, default=None, metavar="DRIVER_DIR",
                   help="--report plus the generalist-vs-specialists "
                        "comparison on the held-out panel")
    p.add_argument("--specialist-steps", type=int, default=250_000)
    p.add_argument("--specialist-out", type=str, default="runs/l6_specialists")
    p.add_argument("--distribution-episodes", type=int, default=40)
    p.add_argument("--reports-dir", type=str, default="reports")
    return p.parse_args()


def run_l5(args: argparse.Namespace) -> None:
    """The Layer 5 flat benign train (behavior unchanged)."""
    if args.resume == "":
        sys.exit("l5_benign needs an explicit directory: --resume DIR")
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


def run_l6(args: argparse.Namespace) -> None:
    """The Layer 6 curriculum: fresh start or resume, pausable."""
    out = Path(args.out)
    if args.resume is not None:
        resume_dir = args.resume or args.out
        print(f"resuming curriculum from {resume_dir}")
        trainer = CurriculumTrainer.resume(resume_dir)
    else:
        if (out / "checkpoint").exists():
            sys.exit(f"{out}/checkpoint exists — pass --resume to continue "
                     f"it, or point --out somewhere fresh")
        cfg = dataclasses.replace(l6_driver_config(seed=args.seed),
                                  n_envs=args.n_envs)
        trainer = CurriculumTrainer(l6_default_curriculum(), out,
                                    seed=args.seed,
                                    budget_scale=args.budget_scale,
                                    tensorboard=not args.no_tb,
                                    driver_config=cfg)
    state = trainer.run(step_limit=args.step_limit)
    if state.get("finished"):
        print("\nfinal validation panel (summaries):")
        print(json.dumps(state.get("final_panel", {}), indent=2))
        print(f"\nfinal driver: {out / 'final'} — next: "
              f"python scripts/train.py --report {out / 'final'}")
    else:
        print("\nrun paused — continue with: python scripts/train.py "
              f"--recipe l6_generalist --out {args.out} --resume")
    trainer.close()


def _load_state_near(driver_dir: Path) -> dict | None:
    for cand in (driver_dir / "curriculum_state.json",
                 driver_dir.parent / "curriculum_state.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def run_report(args: argparse.Namespace) -> None:
    """Panel + lap-time distribution (+ specialist comparison for --compare):
    the documented numbers of the Layer 6 acceptance."""
    driver_dir = Path(args.compare or args.report)
    print(f"loading driver from {driver_dir}")
    driver = SACDriver.load(driver_dir)

    print("evaluating the validation panel ...")
    panel = evaluate_panel(driver, VALIDATION_PANEL, episodes=3)
    for name, rep in panel.items():
        print(f"\n  {name}:")
        print(rep.table())

    print(f"\nlap-time distribution over {args.distribution_episodes} "
          f"random conditions ...")
    dist = evaluate_distribution(driver,
                                 n_episodes=args.distribution_episodes)
    print(dist.table())

    comparison = None
    if args.compare:
        print("\ngeneralist vs specialists on the held-out panel ...")
        comparison = compare_generalist_vs_specialists(
            driver, HELD_OUT_PANEL,
            specialist_steps=args.specialist_steps,
            episodes=args.eval_episodes, out_dir=args.specialist_out)
        print(comparison.table())

    report = build_report(state=_load_state_near(driver_dir), panel=panel,
                          distribution=dist, comparison=comparison)
    report_path = driver_dir / "run_report.json"
    if comparison is None and report_path.exists():
        # keep a previously documented comparison when only re-reporting
        old = json.loads(report_path.read_text())
        report["generalist_vs_specialists"] = old.get(
            "generalist_vs_specialists")
    report_path.write_text(json.dumps(report, indent=2))

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md = reports_dir / "layer6_report.md"
    md.write_text(render_markdown(report, distribution=dist,
                                  comparison=comparison))
    print(f"\nwrote {report_path} and {md}")
    driver.close()


def main() -> None:
    args = parse_args()
    if args.report or args.compare:
        run_report(args)
    elif args.recipe == "l6_generalist":
        run_l6(args)
    else:
        run_l5(args)


if __name__ == "__main__":
    main()
