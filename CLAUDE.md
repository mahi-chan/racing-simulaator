# CLAUDE.md — F1 Autonomous Racing RL (ground-up build)

Always-on project memory for Claude Code. Keep in the repo root. For the full spec of
whatever layer we're on, read `LAYER_SPECS.md`. The human's workflow is in `GUIDE.md`.

## What this is

An RL system that simulates real F1 tracks with a realistic car (tire, engine, aero)
under varying weather, trains a driver agent to extract each car setup's potential,
then searches for the **best car setup for a given track + weather** — i.e. fine-tuning
the car without physical testing. Two "brains": the DRIVER (an RL agent that drives any
given setup near its limit) and the SETUP optimizer (an outer search over car configs
that uses the trained driver as its evaluator). The driver is the RL centerpiece.

## How we work

Planning happens in Claude.ai chat; implementation happens here. Each layer has a full
spec in `LAYER_SPECS.md`. Build ONE layer at a time and validate it in isolation before
building the next. Don't start a layer before the one below it passes its validation.
If a spec is ambiguous or an acceptance threshold looks wrong, stop and ask — don't
weaken thresholds to force a pass. If a spec is unclear, ask before coding.

## Locked decisions

- Track geometry: reconstruct from real F1 telemetry via **FastF1** (Silverstone first).
- Driver algorithm: **SAC** (Stable-Baselines3), proven model-free baseline first.
- Setup search: Bayesian optimization / CMA-ES on top of the trained driver.
- Fresh, clean codebase built bottom-up. Old mixed-state repo is reference only.

## Build order & status

1. Vehicle model (car spec + dynamics) — ✅ DONE, validated.
2. Track environment (FastF1 Silverstone + synthetic fallback) — ▶ NEXT.
3. Conditions model (tire compounds, degradation, fuel burn, weather) — pending.
4. Gym environment wrapper (obs / action / reward / termination + domain randomization) — pending.
5. SAC driver agent — pending.
6. Training loop + curriculum + domain randomization (generalist driver) — pending.
7. Telemetry calibration & validation — pending (justifies "no physical testing").
8. Setup optimization (best setup per track+weather) — pending (the deliverable).
9. Showcase & reporting — optional.

## Layer 1 — vehicle model (built)

`src/physics/vehicle_model.py`, tested by `tests/validate_vehicle.py`. Dynamic bicycle
model: powertrain (torque curve + 8-speed + power limit), speed² aero drag/downforce,
load-sensitive Pacejka tires with a friction ellipse, longitudinal load transfer,
low-speed kinematic blend. numpy only. Validated: 0–100 in 2.7 s, top speed 344 km/h,
braking peak 4.8 G, cornering peak 4.1 G. Tire grip is deliberately conservative and is
a calibration target for Layer 7.

## Project structure

```
f1-racing-rl/
  CLAUDE.md  LAYER_SPECS.md  GUIDE.md  requirements.txt
  src/
    physics/  vehicle_model.py  conditions.py(L3)
    tracks/   track.py(L2)
    envs/     f1_env.py(L4)
    agents/   sac_driver.py(L5)
  scripts/    train.py(L5-6)  calibrate.py(L7)  optimize_setup.py(L8)
  tests/      validate_vehicle.py  test_track.py  test_conditions.py  test_env.py ...
  data/       fastf1_cache/   (gitignored)
```

## Environment & commands

- Python 3.10+. Use a venv. Deps added per layer, kept minimal:
  L1 `numpy` · L2 `+fastf1 scipy` · L3+ (same) · L4–6 `+gymnasium stable-baselines3 sb3-contrib torch` · L8 `+optuna`
- Run Layer 1 validation: `python tests/validate_vehicle.py`

## Conventions (important)

- One layer at a time; each ships a validation script and must pass before building up.
- Physics: numpy only. Env: gymnasium. Agent: stable-baselines3 / sb3-contrib.
- Colab-friendly: the env must sustain thousands of steps/sec on CPU.
- Reward stays interpretable — named, separable components, never one opaque scalar.
- Never claim a layer "works" from reward curves alone — verify lap times / G-forces /
  speed traces.
- All physics & condition parameters live in labeled config; they're placeholders until
  calibrated against real telemetry in Layer 7.

## Current task

Layer 2 — see `LAYER_SPECS.md` § Layer 2. Reconstruct Silverstone from FastF1 with a
synthetic offline fallback; build the queryable Track object and its tests. The real
FastF1 download needs internet + cache and runs on the user's machine/Colab.
