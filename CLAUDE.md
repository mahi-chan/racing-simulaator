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
2. Track environment (FastF1 Silverstone + synthetic fallback) — ✅ DONE, validated
   offline (T8 real-Silverstone check still needs one online run).
3. Conditions model (tire compounds, degradation, fuel burn, weather) — ✅ DONE, validated.
4. Gym environment wrapper (obs / action / reward / termination + domain randomization) — ✅ DONE, validated.
5. SAC driver agent — ▶ NEXT.
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

## Layer 2 — track environment (built)

`src/tracks/track.py`, tested by `tests/test_track.py` (pytest or standalone). Closed
centerline points → periodic smoothing spline → uniform 2 m arc-length resample.
Queryable at any s, wrap-safe: heading / curvature (signed, +left) / width; KD-tree
`nearest_point(x, y)` → (s, lateral offset, +left of travel); corner detection;
`raw_closure_gap` sanity metric. Sources: `Track.from_fastf1` (Silverstone — FastF1
X/Y are **decimeters**) and `Track.from_synthetic` (offline fallback, 5348.9 m, exact
ground-truth corner metadata). Validated offline: curvature matches designed arc radii,
all 8 designed corners found, ~24k nearest-point queries/s (floor 5k). T8 (real
Silverstone: 5891 m ±3%, ≥15 corners) auto-skips offline — run once online on the
user's machine/Colab.

## Layer 3 — conditions model (built)

`src/physics/conditions.py`, tested by `tests/test_conditions.py` (pytest or
standalone). Stint state that modulates the car: `TireCompound` registry
(soft/medium/hard/intermediate/wet — peak grip, wear rate, temp window) and
`Conditions` (compound, wear 0–1, tire temp, fuel, weather dry/damp/wet, rain
intensity, track temp). `step(dt, load, slip, throttle=1.0)` advances wear
(load/slip-driven, cliff at 0.75), temperature (friction heating vs Newtonian
cooling toward track temp), and fuel burn (throttle duty). `grip_multiplier()` =
compound × wear × temperature × weather (named breakdown via `grip_components()`);
`fuel_mass` feeds `CarSpec.fuel_mass` (env must re-run `__post_init__` — Layer 4's
job). One aggregate tire; weather fixed per stint. Pure-scalar stdlib hot loop,
~1M steps/s. Validated: wet 32% below dry, inter best in damp, dead tire −51%
grip, +100 kg fuel = +0.30 s over an 800 m sprint-and-brake, grip 0.52 → peak
lateral 1.5 G vs 3.0 G fresh.

## Layer 4 — gym environment wrapper (built)

`src/envs/f1_env.py`, tested by `tests/test_env.py` (pytest or standalone).
`F1Env(gymnasium.Env)` wraps vehicle + track + conditions. Obs: 40-dim float32 Box,
normalized — car dynamics, prev controls, 10-point look-ahead curvature (10–330 m),
tire/fuel/weather (one-hots), ERS battery, DRS flag, setup knobs. Action: 6-dim Box —
throttle, brake, steer, gear target, ERS deploy, DRS request. Reward = sum of named
components in `info["reward_components"]`: progress (wrap-aware Δs), speed, track,
smooth, tire, fuel, terminal. Each step rewrites the episode `CarSpec` from cached
bases: grip multiplier → mu, 4 MJ ERS battery → power (ICE 615 + MGU-K 120 kW = L1's
735 at full deploy), DRS (gated |κ| < 0.0025) → aero, fuel → mass via
`__post_init__()`; Layers 1–3 untouched. `reset()` domain-randomizes weather /
compound / rain / track temp / fuel (20–105 kg) / aero / brake bias / final drive /
start pose — every knob pinnable via `options` for evaluation. Termination
off_track / spin / fuel_out / stall (config-gated); truncation lap complete
(`info["lap_time"]`) or 15k steps. Validated: check_env clean; 10k random steps
finite; components sum exactly; seed-bit-reproducible; pursuit driver laps 140.7 s
(avg 137 km/h) vs random no-laps; damp +30% lap time on equal compound; +85 kg fuel
+0.7 s; 3,742 steps/s (floor 2000). Emergent, kept: slicks run cold after the long
straight (soft ≈ wet-wet pace with a cautious driver) — L7 calibration target.

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
- Run Layer 2 validation: `python tests/test_track.py` (or `pytest tests/`)
- Run Layer 3 validation: `python tests/test_conditions.py`
- Run Layer 4 validation: `python tests/test_env.py`

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

Layer 5 — see `LAYER_SPECS.md` § Layer 5: SAC driver agent. Plan in Claude.ai chat
before implementing here. Leftover from Layer 2: run `python tests/test_track.py`
once online to exercise T8 (real Silverstone reconstruction).
