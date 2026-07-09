"""
F1 Gym Environment — Layer 4 of the autonomous racing stack.

Wraps the Layer 1 vehicle (`F1Vehicle`), Layer 2 track (`Track`) and Layer 3
stint state (`Conditions`) into one `gymnasium.Env` the Layer 5 SAC driver
trains against.

How the layers are wired (each step):
  * The env owns a per-episode `CarSpec` and rewrites its modulated fields
    every step FROM CACHED BASE VALUES (never compounding):
      - mu_x / mu_y = base x conditions.grip_multiplier()
      - cda / cla   = base x DRS factors (open only where the track is straight)
      - max_power   = ICE + ERS deploy (battery-limited)
      - fuel_mass   = conditions.fuel_mass, then `spec.__post_init__()` re-caches
        total_mass — the wiring conditions.py's docstring assigns to Layer 4.
    `F1Vehicle` reads spec fields live each substep, so Layer 1 is untouched.
  * The load/slip fed to `conditions.step` mirror Layer 1's own formulas:
    total load = m*g + downforce (longitudinal transfer cancels in the sum);
    slip = max |front/rear slip angle|, recomputed here because `VehicleState`
    deliberately does not expose Layer 1 internals.

Conventions:
  * float32 Box spaces; observations are normalized to O(1) and clipped.
  * Reward is a sum of NAMED components, exposed every step in
    info["reward_components"] (project rule: never one opaque scalar).
  * Weather/compound/setup are drawn at reset (domain randomization); an
    `options` dict pins any subset for evaluation and tests. All randomness
    goes through `self.np_random`, so seeded episodes are bit-reproducible.
  * Every constant lives in the labeled configs below — placeholders until the
    Layer 7 telemetry calibration. ERS and DRS are deliberately simple
    env-level models (no real DRS zones or energy regulations yet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from src.physics.conditions import COMPOUNDS, WEATHERS, Conditions
from src.physics.vehicle_model import G, RHO_AIR, CarSpec, F1Vehicle
from src.tracks.track import Track

COMPOUND_ORDER = tuple(COMPOUNDS)  # one-hot order: soft/medium/hard/inter/wet
ACTION_NAMES = ("throttle", "brake", "steer", "gear", "ers", "drs")


# ----------------------------------------------------------------------------
# Config — every number labeled; Layer 7 calibrates, Layer 8 searches setups
# ----------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Weights of the named reward components (info["reward_components"])."""

    w_progress: float = 1.0        # per meter of progress along s (reward ~ meters)
    w_speed: float = 0.05          # per (m/s of vx) per second — small, spec lists it
    w_track: float = 5.0           # per m^2 of soft-edge overshoot, per second
    soft_edge_margin: float = 1.0  # m inside the physical edge where penalty starts
    w_smooth: float = 0.05         # per unit sum of squared control deltas
    w_tire: float = 20.0           # per unit of tire wear (a full tire life = -20)
    w_fuel: float = 1.0            # per kg of fuel burned
    p_terminal: float = 100.0      # one-off penalty: off_track / spin / stall


@dataclass
class DomainRandomizationConfig:
    """reset()-time ranges (spec: randomize "within configured ranges")."""

    weather_probs: dict[str, float] = field(default_factory=lambda: {
        "dry": 0.60, "damp": 0.25, "wet": 0.15})
    rain_intensity_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"dry": (0.0, 0.0), "damp": (0.0, 0.5),
                                 "wet": (0.3, 1.0)})
    track_temp_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"dry": (20.0, 45.0), "damp": (12.0, 30.0),
                                 "wet": (12.0, 30.0)})
    # compound draw per weather — mismatched pairs (slicks in rain) stay
    # possible on purpose: surviving them IS the domain randomization
    compound_probs: dict[str, tuple] = field(default_factory=lambda: {
        "dry": (("soft", 1 / 3), ("medium", 1 / 3), ("hard", 1 / 3)),
        "damp": (("intermediate", 0.60), ("soft", 0.2 / 3), ("medium", 0.2 / 3),
                 ("hard", 0.2 / 3), ("wet", 0.20)),
        "wet": (("wet", 0.70), ("intermediate", 0.30)),
    })
    fuel_range: tuple[float, float] = (20.0, 105.0)     # kg at episode start
    aero_level_range: tuple[float, float] = (0.0, 1.0)  # 0 = skinny .. 1 = max wing
    brake_bias_range: tuple[float, float] = (0.54, 0.62)
    final_drive_range: tuple[float, float] = (2.85, 3.15)
    start_speed_range: tuple[float, float] = (25.0, 60.0)          # m/s
    start_lateral_range: tuple[float, float] = (-1.5, 1.5)         # m
    start_heading_error_range: tuple[float, float] = (-0.05, 0.05)  # rad
    randomize_start_s: bool = True  # False -> every episode starts at s = 0


@dataclass
class EnvConfig:
    """Environment behavior. SI units unless noted."""

    # --- control loop ---
    dt: float = 0.02          # s per env step (50 Hz control)
    substeps: int = 2         # Layer 1 integration substeps per env step
    max_steps: int = 15_000   # truncation: 5 sim-minutes

    # --- observation ---
    lookahead_distances: tuple = (10.0, 20.0, 35.0, 55.0, 80.0, 110.0,
                                  150.0, 200.0, 260.0, 330.0)  # m ahead along s
    obs_clip: float = 5.0     # Box bound after normalization

    # --- termination ---
    off_track_margin: float = 1.0   # m beyond half-width before terminating
    spin_sideslip: float = 0.6      # rad body sideslip = spun (at speed)
    spin_min_speed: float = 5.0     # m/s — below this, sideslip is parking noise
    terminate_on_stall: bool = True  # beyond-spec addition: end idle episodes
    stall_speed: float = 1.0        # m/s
    stall_duration: float = 2.0     # s continuously below stall_speed

    # --- setup -> aero: scale CarSpec base cla/cda by aero_level in [0, 1] ---
    aero_cla_span: tuple[float, float] = (0.85, 1.15)  # downforce multiplier range
    aero_cda_span: tuple[float, float] = (0.92, 1.08)  # drag follows wing level

    # --- ERS (CarSpec.max_power 735 kW is documented as ICE + full deploy) ---
    ers_ice_power: float = 615_000.0      # W with zero deploy
    ers_deploy_power: float = 120_000.0   # W added at full deploy (MGU-K scale)
    ers_capacity: float = 4.0e6           # J battery
    ers_harvest_power: float = 120_000.0  # W recovered at full brake (speed-scaled)

    # --- DRS (placeholder gate: openable wherever the track is straight) ---
    drs_max_curvature: float = 0.0025  # 1/m — open only where R > 400 m
    drs_drag_factor: float = 0.88      # cda multiplier while open
    drs_downforce_factor: float = 0.90  # cla multiplier while open

    reward: RewardConfig = field(default_factory=RewardConfig)
    dr: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig)


# ----------------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------------
class F1Env(gym.Env):
    """One car, one track, one stint; episode = up to one flying lap.

    Action (Box, [-1, 1]^6): throttle, brake, steer, gear, ERS deploy, DRS.
    Termination: off_track / spin / fuel_out / stall (reason in info).
    Truncation: lap completed (info["lap_time"]) or max_steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, track: Track | None = None,
                 config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.track = track if track is not None else Track.from_synthetic()
        cfg = self.config

        self._look_d = np.asarray(cfg.lookahead_distances, dtype=np.float64)
        n_obs = (11                        # dynamics (8) + previous controls (3)
                 + len(self._look_d)       # curvature look-ahead
                 + 3                       # wear, temp-vs-window, grip multiplier
                 + len(COMPOUND_ORDER) + 1  # compound one-hot + fuel
                 + len(WEATHERS) + 2       # weather one-hot + rain + track temp
                 + 2                       # ERS battery fraction, DRS open
                 + 3)                      # setup: aero level, brake bias, final drive
        self.observation_space = gym.spaces.Box(
            -cfg.obs_clip, cfg.obs_clip, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,),
                                           dtype=np.float32)

        # normalization midpoints for the setup obs block, from the DR ranges
        dr = cfg.dr
        self._bb_mid = 0.5 * (dr.brake_bias_range[0] + dr.brake_bias_range[1])
        self._bb_half = max(
            0.5 * (dr.brake_bias_range[1] - dr.brake_bias_range[0]), 1e-6)
        self._fd_mid = 0.5 * (dr.final_drive_range[0] + dr.final_drive_range[1])
        self._fd_half = max(
            0.5 * (dr.final_drive_range[1] - dr.final_drive_range[0]), 1e-6)

        # episode state — fully (re)initialized in reset()
        self.vehicle = F1Vehicle()
        self.conditions = Conditions()
        self._base_mu_x = self.vehicle.spec.mu_x
        self._base_mu_y = self.vehicle.spec.mu_y
        self._base_cla = self.vehicle.spec.cla
        self._base_cda = self.vehicle.spec.cda
        self._setup = {"weather": "dry", "compound": "medium",
                       "rain_intensity": 0.0, "track_temp": 30.0, "fuel": 100.0,
                       "aero_level": 0.5, "brake_bias": self._bb_mid,
                       "final_drive": self._fd_mid}
        self._battery = cfg.ers_capacity
        self._drs_open = False
        self._prev_controls = (0.0, 0.0, 0.0)
        self._s = 0.0
        self._lat = 0.0
        self._heading_err = 0.0
        self._width = float(self.track.width_at(0.0))
        self._slip = 0.0
        self._steps = 0
        self._total_progress = 0.0
        self._stall_timer = 0.0
        self._lap_time: float | None = None

    # -- gymnasium API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, dr, rng = self.config, self.config.dr, self.np_random
        opt = dict(options) if options else {}

        def pick(key, sampler):
            v = opt.get(key)
            return sampler() if v is None else v

        # --- weather & stint state ---
        w_names = list(dr.weather_probs)
        weather = str(pick("weather", lambda: rng.choice(
            w_names, p=[dr.weather_probs[w] for w in w_names])))
        table = dr.compound_probs[weather]
        compound = str(pick("compound", lambda: rng.choice(
            [c for c, _ in table], p=[p for _, p in table])))
        rain = float(pick("rain_intensity",
                          lambda: rng.uniform(*dr.rain_intensity_range[weather])))
        track_temp = float(pick("track_temp",
                                lambda: rng.uniform(*dr.track_temp_range[weather])))
        fuel = float(pick("fuel", lambda: rng.uniform(*dr.fuel_range)))

        # --- car setup ---
        aero_level = float(pick("aero_level",
                                lambda: rng.uniform(*dr.aero_level_range)))
        brake_bias = float(pick("brake_bias",
                                lambda: rng.uniform(*dr.brake_bias_range)))
        final_drive = float(pick("final_drive",
                                 lambda: rng.uniform(*dr.final_drive_range)))
        base = CarSpec()  # reference car; setup scales it, conditions modulate it
        lo, hi = cfg.aero_cla_span
        cla = base.cla * (lo + (hi - lo) * aero_level)
        lo, hi = cfg.aero_cda_span
        cda = base.cda * (lo + (hi - lo) * aero_level)
        spec = CarSpec(fuel_mass=fuel, cla=cla, cda=cda,
                       brake_bias_front=brake_bias, final_drive=final_drive)
        self._base_mu_x, self._base_mu_y = spec.mu_x, spec.mu_y
        self._base_cla, self._base_cda = spec.cla, spec.cda
        self.vehicle = F1Vehicle(spec)
        self.conditions = Conditions(compound=compound, weather=weather,
                                     rain_intensity=rain, track_temp=track_temp,
                                     fuel_mass=fuel)

        # --- start pose ---
        L = self.track.length
        s0 = float(pick("s0", lambda: (rng.uniform(0.0, L)
                                       if dr.randomize_start_s else 0.0)))
        v0 = float(pick("v0", lambda: rng.uniform(*dr.start_speed_range)))
        lat0 = float(pick("lateral", lambda: rng.uniform(*dr.start_lateral_range)))
        herr0 = float(pick("heading_error",
                           lambda: rng.uniform(*dr.start_heading_error_range)))
        heading = float(self.track.heading_at(s0))
        x0, y0 = self._position_at(s0)
        x0 -= math.sin(heading) * lat0  # +lat0 = left of the direction of travel
        y0 += math.cos(heading) * lat0
        self.vehicle.reset(x=x0, y=y0, yaw=heading + herr0, speed=v0)
        self.vehicle.state.gear = self.vehicle.auto_gear(v0)

        # --- episode bookkeeping ---
        self._battery = cfg.ers_capacity
        self._drs_open = False
        self._prev_controls = (0.0, 0.0, 0.0)
        self._steps = 0
        self._total_progress = 0.0
        self._stall_timer = 0.0
        self._lap_time = None
        self._slip = 0.0
        self._s, self._lat = self.track.nearest_point(x0, y0)
        self._heading_err = self._wrap_pi(self.vehicle.state.yaw
                                          - float(self.track.heading_at(self._s)))
        self._width = float(self.track.width_at(self._s))
        self._setup = {"weather": weather, "compound": compound,
                       "rain_intensity": rain, "track_temp": track_temp,
                       "fuel": fuel, "aero_level": aero_level,
                       "brake_bias": brake_bias, "final_drive": final_drive}

        info = {"setup": dict(self._setup), "s": self._s,
                "grip_multiplier": self.conditions.grip_multiplier()}
        return self._make_obs(), info

    def step(self, action):
        cfg = self.config
        rw = cfg.reward
        cond = self.conditions
        veh = self.vehicle
        spec = veh.spec
        dt = cfg.dt

        # --- map the action ---
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        throttle = 0.5 * (float(a[0]) + 1.0)
        brake = 0.5 * (float(a[1]) + 1.0)
        steer = float(a[2])
        gear = int(min(max(round(1.0 + 3.5 * (float(a[3]) + 1.0)), 1), 8))
        deploy = 0.5 * (float(a[4]) + 1.0)
        drs_request = float(a[5]) > 0.0

        # DRS: request AND straight-enough track (placeholder for real zones)
        self._drs_open = bool(
            drs_request
            and abs(float(self.track.curvature_at(self._s))) < cfg.drs_max_curvature)
        # ERS: battery-limited boost (no free energy at the empty boundary)
        boost = min(deploy * cfg.ers_deploy_power, self._battery / dt)

        # --- modulate the spec from cached bases (Layer 1 reads these live) ---
        grip = cond.grip_multiplier()
        spec.mu_x = self._base_mu_x * grip
        spec.mu_y = self._base_mu_y * grip
        spec.cda = self._base_cda * (cfg.drs_drag_factor if self._drs_open else 1.0)
        spec.cla = self._base_cla * (cfg.drs_downforce_factor if self._drs_open
                                     else 1.0)
        spec.max_power = cfg.ers_ice_power + boost
        spec.fuel_mass = cond.fuel_mass
        spec.__post_init__()  # re-cache total_mass after the fuel burn

        # --- advance the car ---
        st = veh.step(throttle, brake, steer, gear=gear, dt=dt,
                      substeps=cfg.substeps)

        # --- advance the stint state (load/slip mirror Layer 1's formulas) ---
        wear_prev, fuel_prev = cond.wear, cond.fuel_mass
        load = spec.total_mass * G + 0.5 * RHO_AIR * spec.cla * st.vx * st.vx
        vx_safe = max(st.vx, 1.0)
        blend = min(max(st.vx / 3.0, 0.0), 1.0)
        delta = steer * spec.max_steer
        alpha_f = (math.atan2(st.vy + spec.lf * st.yaw_rate, vx_safe) - delta) * blend
        alpha_r = math.atan2(st.vy - spec.lr * st.yaw_rate, vx_safe) * blend
        self._slip = max(abs(alpha_f), abs(alpha_r))
        cond.step(dt, load, self._slip, throttle)

        speed = math.hypot(st.vx, st.vy)
        harvest = cfg.ers_harvest_power * brake * min(st.vx / 30.0, 1.0)
        self._battery = min(max(self._battery + (harvest - boost) * dt, 0.0),
                            cfg.ers_capacity)

        # --- track relation & wrap-aware progress ---
        L = self.track.length
        s_prev = self._s
        s, lat = self.track.nearest_point(st.x, st.y)
        ds = (s - s_prev + 0.5 * L) % L - 0.5 * L
        self._s, self._lat = s, lat
        self._total_progress += ds
        self._heading_err = self._wrap_pi(st.yaw - float(self.track.heading_at(s)))
        self._width = float(self.track.width_at(s))
        self._steps += 1

        # --- termination / truncation ---
        reason = None
        if abs(lat) > 0.5 * self._width + cfg.off_track_margin:
            reason = "off_track"
        elif (speed > cfg.spin_min_speed
              and abs(math.atan2(st.vy, max(st.vx, 0.5))) > cfg.spin_sideslip):
            reason = "spin"
        elif cond.fuel_mass <= 0.0:
            reason = "fuel_out"
        elif cfg.terminate_on_stall:
            self._stall_timer = (self._stall_timer + dt
                                 if speed < cfg.stall_speed else 0.0)
            if self._stall_timer >= cfg.stall_duration:
                reason = "stall"
        terminated = reason is not None

        if self._total_progress >= L and self._lap_time is None:
            self._lap_time = self._steps * dt
        truncated = (not terminated) and (self._lap_time is not None
                                          or self._steps >= cfg.max_steps)

        # --- reward: named separable components; their sum IS the reward ---
        p_thr, p_brk, p_str = self._prev_controls
        excess = max(0.0, abs(lat) - (0.5 * self._width - rw.soft_edge_margin))
        components = {
            "progress": rw.w_progress * ds,
            "speed": rw.w_speed * st.vx * dt,
            "track": -rw.w_track * excess * excess * dt,
            "smooth": -rw.w_smooth * ((throttle - p_thr) ** 2
                                      + (brake - p_brk) ** 2
                                      + (steer - p_str) ** 2),
            "tire": -rw.w_tire * (cond.wear - wear_prev),
            "fuel": -rw.w_fuel * (fuel_prev - cond.fuel_mass),
            "terminal": (-rw.p_terminal
                         if reason in ("off_track", "spin", "stall") else 0.0),
        }
        reward = float(sum(components.values()))
        self._prev_controls = (throttle, brake, steer)

        info = {
            "reward_components": components,
            "s": s, "lateral": lat, "speed": speed,
            "lap_fraction": min(max(self._total_progress / L, 0.0), 1.0),
            "wear": cond.wear, "tire_temp": cond.tire_temp,
            "fuel": cond.fuel_mass, "battery": self._battery,
            "grip_multiplier": cond.grip_multiplier(),
            "grip_components": cond.grip_components(),
            "drs_open": self._drs_open, "setup": self._setup,
        }
        if reason is not None:
            info["termination"] = reason
        if self._lap_time is not None:
            info["lap_time"] = self._lap_time
        return self._make_obs(), reward, terminated, truncated, info

    # -- internals -------------------------------------------------------------
    def _make_obs(self) -> np.ndarray:
        cfg = self.config
        st = self.vehicle.state
        cond = self.conditions
        lo, hi = cond.compound.temp_window
        look = np.asarray(self.track.curvature_at(self._s + self._look_d),
                          dtype=np.float64)
        p_thr, p_brk, p_str = self._prev_controls

        vals = [
            st.vx / 100.0,
            st.vy / 10.0,
            st.yaw_rate / 3.0,
            self._slip / 0.5,
            self._lat / (0.5 * self._width),   # +-1 = at the track edge
            self._heading_err / (0.5 * math.pi),
            (st.gear - 1) / 7.0,
            self.vehicle.engine_rpm(st.vx, st.gear) / self.vehicle.spec.rpm_limit,
            p_thr, p_brk, p_str,
        ]
        vals.extend((look * 100.0).tolist())    # kappa * 100: R=100 m -> 1.0
        vals.extend([
            cond.wear,
            (cond.tire_temp - 0.5 * (lo + hi)) / 50.0,
            cond.grip_multiplier(),
        ])
        vals.extend(1.0 if cond.compound.name == n else 0.0
                    for n in COMPOUND_ORDER)
        vals.append(cond.fuel_mass / 110.0)
        vals.extend(1.0 if cond.weather == w else 0.0 for w in WEATHERS)
        vals.extend([
            cond.rain_intensity,
            cond.track_temp / 50.0,
            self._battery / cfg.ers_capacity,
            1.0 if self._drs_open else 0.0,
            self._setup["aero_level"],
            (self._setup["brake_bias"] - self._bb_mid) / self._bb_half,
            (self._setup["final_drive"] - self._fd_mid) / self._fd_half,
        ])
        obs = np.asarray(vals, dtype=np.float32)
        np.clip(obs, -cfg.obs_clip, cfg.obs_clip, out=obs)
        return obs

    def _position_at(self, s: float) -> tuple[float, float]:
        """Centerline XY at arc length s (linear interp on the 2 m grid)."""
        track = self.track
        sm = s % track.length
        return (float(np.interp(sm, track.s, track.centerline[:, 0])),
                float(np.interp(sm, track.s, track.centerline[:, 1])))

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
