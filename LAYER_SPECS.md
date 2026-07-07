# LAYER_SPECS.md — Layer Contracts (for Claude Code)

The authoritative spec for each layer. When implementing Layer N, read this section
in full and follow the interface and acceptance tests exactly. Do not weaken an
acceptance threshold to make a test pass; if a threshold is genuinely wrong, stop and
say so. Build one layer at a time; a layer is done only when its validation passes.

Global constraints: physics uses numpy only; the environment uses gymnasium; the
agent uses stable-baselines3 / sb3-contrib. Everything must run on CPU fast enough
for training on Google Colab (the env must sustain thousands of steps/sec). Keep all
physics/condition parameters in clearly-labeled config so Layer 7 can calibrate them.

---

## Layer 1 — Vehicle model  [DONE]

Built: `src/physics/vehicle_model.py`, validated by `tests/validate_vehicle.py`.
`CarSpec` + `F1Vehicle` dynamic bicycle model. Reference only; do not modify without
a planning decision.

---

## Layer 2 — Track environment (FastF1 reconstruction)

**Objective.** A queryable Silverstone track built from real F1 telemetry, with a
synthetic fallback so tests run offline.

**Interface** (`src/tracks/track.py`):
- `Track.from_fastf1(year=2023, gp="Silverstone", session="Q", driver="VER")` —
  caches to `data/fastf1_cache/`, extracts the fastest lap's XY trace, resamples to a
  smooth centerline.
- `Track.from_synthetic()` — a hand-built closed loop (~5 km, several corners) needing
  no network.
- Properties: `length` (m), `centerline` (N×2), `width_at(s)`, `curvature` (per point).
- Methods: `nearest_point(x, y) -> (s, lateral_offset)`, `heading_at(s)`,
  `is_on_track(x, y) -> bool`, `corners` (list of s-positions from curvature peaks).

**Acceptance** (`tests/test_track.py`):
- Synthetic track: closed loop (start within 1 m of end), positive length, monotonic `s`.
- `nearest_point` and `heading_at` are self-consistent (project a known point, recover it).
- `is_on_track` true on centerline, false beyond `width/2 + margin`.
- If run online, reconstructed Silverstone length is 5891 m ± 3% and detects ≥ 15 corners.
- Offline test suite passes using only the synthetic track.

**Constraints.** numpy + scipy (+ fastf1 for the online path). Never fail the test
suite because the network is unavailable — gate the FastF1 test behind availability.

---

## Layer 3 — Conditions model (tires, degradation, fuel, weather)

**Objective.** The dynamic state that modulates car performance over a stint — the
"real F1 features" beyond raw chassis physics.

**Interface** (`src/physics/conditions.py`):
- `TireCompound` (soft/medium/hard/intermediate/wet): peak grip, wear rate, optimal
  temperature window. Soft = highest peak grip, fastest wear; hard = opposite.
- `Conditions` holding: current compound + wear (0–1), tire temperature, fuel mass,
  weather (`dry`/`damp`/`wet`), track temperature, rain intensity.
- `step(dt, load, slip)` — advances tire wear/temperature and burns fuel.
- `grip_multiplier() -> float` — combined factor applied to tire mu in the vehicle
  (compound × wear × temperature × weather).
- Exposes `fuel_mass` so the env can update `CarSpec.fuel_mass` (heavier = slower).

**Acceptance** (`tests/test_conditions.py`):
- Peak grip ordering soft > medium > hard on a fresh tire; wear ordering reversed.
- Grip decreases monotonically as wear rises.
- Wet reduces grip 30–40% vs dry; intermediate sits between wet and dry in the damp.
- Fuel mass decreases over a stint; a lighter car yields a measurably faster lap
  (integrate with Layer 1 to confirm).
- Tire outside its temperature window loses grip.

**Constraints.** All coefficients labeled and config-driven for later calibration.

---

## Layer 4 — Gym environment wrapper

**Objective.** Wrap vehicle + track + conditions into one Gymnasium env the agent
trains against.

**Interface** (`src/envs/f1_env.py`, `F1Env(gymnasium.Env)`):
- **Observation** (Box): car dynamics (speed, slip, yaw rate, lateral offset, heading
  error), track look-ahead (curvature at several distances ahead), tire/fuel/weather
  state, and the current setup parameters (so a domain-randomized policy can adapt).
- **Action** (Box): throttle, brake, steering, gear, ERS deploy, DRS.
- **Reward** (named, separable components): forward progress along `s`; speed;
  staying on track; smoothness (penalize jerky inputs); tire/fuel management; large
  penalty for leaving the track or spinning. Expose the component breakdown in `info`.
- **reset()**: domain-randomize the car setup and weather within configured ranges.
- **Termination**: off-track, spun (excess yaw/slip), out of fuel. **Truncation**:
  max steps or lap completed.

**Acceptance** (`tests/test_env.py`):
- Passes `gymnasium.utils.env_checker.check_env`.
- 10k random steps produce no NaN/inf and no crash.
- Reward component breakdown present in `info` and sums to the total reward.
- A simple hand-coded pursuit controller (follow the centerline) completes a lap and
  scores higher than a random policy — proves the reward gradient points the right way.
- Sustains ≥ 2000 steps/sec single-process on CPU.

**Constraints.** Reward components stay interpretable; randomization ranges in config.

---

## Layer 5 — SAC driver agent

**Objective.** An SAC agent (Stable-Baselines3) that learns to drive a lap.

**Interface** (`src/agents/sac_driver.py`, `scripts/train.py`):
- Wrap SB3 `SAC`; observation normalization; support a vectorized env for parallelism.
- Config-driven hyperparameters (lr, buffer, batch, tau, gamma, net arch); fixed seed.
- Checkpoint save/load; an `evaluate()` reporting lap time, average speed, off-track count.

**Acceptance** (`tests/test_sac.py` + a short smoke train):
- A short training run (e.g. 50k steps on the synthetic track) measurably reduces lap
  time and off-track count vs a random policy.
- Training is stable (no divergence/NaN); checkpoints round-trip.

**Constraints.** Reproducible; all knobs in config, none hard-coded.

---

## Layer 6 — Training loop, curriculum, domain randomization

**Objective.** Produce a *generalist* driver: one policy that drives many setups and
weather conditions competently — the headline RL result.

**Interface** (`scripts/train.py` extended):
- Curriculum stages (e.g. wide/grippy/single-corner → full track → full randomization).
- Domain randomization over setup + weather every reset.
- Tensorboard logging; checkpoint/resume for Colab's ~12h windows.

**Acceptance**:
- A domain-randomized policy drives several held-out setups/weather within a
  documented % of setup-specialized policies.
- Lap-time distribution over random conditions is reported; training resumes cleanly.

**Constraints.** Sample-efficiency matters (Colab compute). Log everything.

---

## Layer 7 — Telemetry calibration & validation

**Objective.** Calibrate the simulator to real FastF1 telemetry until it is trustworthy
— the layer that justifies "fine-tuning without physical testing."

**Interface** (`src/utils/validation.py`, `scripts/calibrate.py`):
- Compare sim vs a real reference lap: lap-time error, speed-vs-distance RMSE, apex
  speeds, braking points.
- Fit a small set of physics params (tire grip, aero, drivetrain) to minimize error
  (grid search or an optimizer).

**Acceptance**:
- After calibration, sim lap time is within ~2–3% of the real reference lap and the
  speed trace RMSE is under a stated threshold.
- Calibrated parameters are written to config with provenance.

**Constraints.** Needs real telemetry (internet). Report error honestly; do not tune
to overfit a single lap — hold out a second lap to check.

---

## Layer 8 — Setup optimization (the deliverable)

**Objective.** Given a track + weather, find the best car setup, using the trained
generalist driver as the evaluator. This is the "fine-tune the car" outcome.

**Interface** (`scripts/optimize_setup.py`):
- Search space: aero level, gear ratios, brake bias, tire compound, fuel load,
  mechanical balance.
- Optimizer: Bayesian optimization (Optuna/scikit-optimize) or CMA-ES.
- Objective: lap time (or stint time) from the trained driver on the target
  track+weather. Cache evaluations; run per weather condition.
- Output: ranked setups, the best setup, and per-parameter sensitivity.

**Acceptance**:
- Finds a setup beating a sensible baseline setup on lap time.
- Produces a comparison report (best vs baseline, param importances) per weather.
- Flags any proposed setup the driver fails to generalize to (so results stay honest).

**Constraints.** Reuse the Layer 6 driver — no retraining per setup.

---

## Layer 9 — Showcase & reporting (optional)

Racing-line and speed-trace plots, setup-comparison tables, weather-sensitivity
charts, and a written summary framing the RL methods and results for a portfolio.
