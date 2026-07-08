"""
F1 Vehicle Model — Layer 1 of the autonomous racing stack.

This is the "car": a physically grounded dynamic bicycle model with a proper
powertrain, aerodynamics, and load-sensitive tires. Everything above it in the
stack (track environment, Gym wrapper, SAC agent) depends on this being correct,
so it is written to be validated in isolation before anything is built on top.

Design choices:
  * Dynamic bicycle model (not point-mass): captures yaw, understeer/oversteer,
    and load transfer, which matter for racing lines and braking points — while
    staying light enough to run thousands of steps/second on a CPU (Colab-friendly).
  * Simplified Pacejka ("magic formula") tires with a friction ellipse, so the
    tire cannot spend 100% of its grip on cornering AND braking at once.
  * Longitudinal load transfer + aero downforce feed the tire normal loads, so
    grip rises with speed (the defining feature of an F1 car).
  * A kinematic/dynamic blend at low speed removes the divide-by-velocity
    singularity that naive bicycle models suffer from at a standing start.

Parameters are representative 2024-era F1 values from public sources, not any
team's confidential data. Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

G = 9.81            # gravity (m/s^2)
RHO_AIR = 1.225     # air density at sea level (kg/m^3)


# ----------------------------------------------------------------------------
# Car specification
# ----------------------------------------------------------------------------
@dataclass
class CarSpec:
    """Physical specification of the car. All SI units unless noted."""

    # --- Mass & geometry ---
    mass: float = 798.0            # min F1 weight incl. driver, no fuel (kg)
    fuel_mass: float = 0.0         # added on top of mass (0..~110 kg)
    wheelbase: float = 3.60        # front-to-rear axle distance (m)
    front_weight_dist: float = 0.46  # static fraction of weight on front axle
    cg_height: float = 0.30        # centre-of-gravity height (m), F1 is very low
    yaw_inertia: float = 1000.0    # moment of inertia about vertical axis (kg·m^2)

    # --- Aerodynamics ---  F = 0.5 * rho * (Cd*A or Cl*A) * v^2
    cda: float = 1.30              # drag area (m^2) — sets top speed
    cla: float = 3.40              # downforce area (m^2) — sets high-speed grip
    aero_balance_front: float = 0.45  # fraction of downforce on the front axle

    # --- Powertrain ---
    max_power: float = 735_000.0   # combined ICE+ERS peak power (W) ~986 hp
    drivetrain_efficiency: float = 0.95
    tire_radius: float = 0.33      # rolling radius, rear (m)
    final_drive: float = 3.0
    # 8-speed gearbox ratios (high -> low)
    gear_ratios: tuple = (2.90, 2.30, 1.90, 1.60, 1.36, 1.18, 1.04, 0.92)
    rpm_idle: float = 4000.0
    rpm_limit: float = 15000.0     # 2024 regs cap engine at 15,000 rpm
    peak_torque: float = 900.0     # crank torque at the fat part of the curve (N·m)

    # --- Braking ---
    max_brake_force: float = 32_000.0  # total, tire-limited in practice (N)
    brake_bias_front: float = 0.58

    # --- Tires (simplified Pacejka) ---
    mu_x: float = 1.65             # peak longitudinal friction coefficient
    mu_y: float = 1.60             # peak lateral friction coefficient
    pacejka_b: float = 10.0        # stiffness factor
    pacejka_c: float = 1.45        # shape factor
    rolling_resistance: float = 0.014
    # mild load sensitivity: grip coefficient falls slightly as load grows
    load_sensitivity: float = 0.20

    # --- Limits ---
    max_steer: float = np.radians(18.0)  # max front wheel angle (rad)

    def __post_init__(self):
        self.total_mass = self.mass + self.fuel_mass
        self.lf = self.wheelbase * (1.0 - self.front_weight_dist)  # CG->front axle
        self.lr = self.wheelbase * self.front_weight_dist          # CG->rear axle


# ----------------------------------------------------------------------------
# Vehicle state
# ----------------------------------------------------------------------------
@dataclass
class VehicleState:
    x: float = 0.0          # global position (m)
    y: float = 0.0
    yaw: float = 0.0        # heading (rad)
    vx: float = 0.0         # body-frame longitudinal velocity (m/s)
    vy: float = 0.0         # body-frame lateral velocity (m/s)
    yaw_rate: float = 0.0   # (rad/s)
    ax: float = 0.0         # last longitudinal accel (m/s^2), for load transfer
    ay: float = 0.0         # last lateral accel (m/s^2)
    gear: int = 1

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


# ----------------------------------------------------------------------------
# The vehicle
# ----------------------------------------------------------------------------
class F1Vehicle:
    """Steps the car forward given driver controls."""

    def __init__(self, spec: CarSpec | None = None):
        self.spec = spec or CarSpec()
        self.state = VehicleState()

    def reset(self, x=0.0, y=0.0, yaw=0.0, speed=0.0) -> VehicleState:
        self.state = VehicleState(x=x, y=y, yaw=yaw, vx=max(speed, 0.0), gear=1)
        return self.state

    # -- powertrain ---------------------------------------------------------
    def engine_rpm(self, vx: float, gear: int) -> float:
        s = self.spec
        wheel_rps = max(vx, 0.0) / (2 * np.pi * s.tire_radius)
        rpm = wheel_rps * 60.0 * s.gear_ratios[gear - 1] * s.final_drive
        return float(np.clip(rpm, s.rpm_idle, s.rpm_limit))

    def _torque_curve(self, rpm: float) -> float:
        """Crank torque (N·m) vs rpm — a plausible peaky turbo-hybrid curve."""
        s = self.spec
        x = rpm / s.rpm_limit
        # broad plateau across the upper rev range; a hybrid PU pulls hard
        # almost everywhere, so the power limit (P=F*v) is what usually binds.
        shape = np.exp(-((x - 0.72) ** 2) / (2 * 0.40 ** 2))
        return s.peak_torque * float(np.clip(shape, 0.60, 1.0))

    def auto_gear(self, vx: float) -> int:
        """Pick the gear giving the most wheel force at this speed (open-loop use)."""
        s = self.spec
        best_gear, best_force = 1, -1.0
        for g in range(1, len(s.gear_ratios) + 1):
            rpm = self.engine_rpm(vx, g)
            if rpm >= s.rpm_limit * 0.995 and g < len(s.gear_ratios):
                continue  # would be on the limiter, prefer taller gear
            force = (self._torque_curve(rpm) * s.gear_ratios[g - 1]
                     * s.final_drive * s.drivetrain_efficiency / s.tire_radius)
            if force > best_force:
                best_force, best_gear = force, g
        return best_gear

    def _drive_force(self, throttle: float, vx: float, gear: int) -> float:
        s = self.spec
        rpm = self.engine_rpm(vx, gear)
        wheel_force = (self._torque_curve(rpm) * s.gear_ratios[gear - 1]
                       * s.final_drive * s.drivetrain_efficiency / s.tire_radius)
        # never exceed the power limit: P = F * v
        power_limit = s.max_power / max(vx, 1.0)
        return throttle * min(wheel_force, power_limit)

    # -- tires --------------------------------------------------------------
    def _tire_lateral(self, alpha: float, fz: float) -> float:
        """Simplified Pacejka lateral force with load-sensitive grip."""
        s = self.spec
        mu = s.mu_y * (1.0 - s.load_sensitivity * (fz / (s.total_mass * G) - 1.0))
        mu = max(mu, 0.5)
        return -mu * fz * np.sin(s.pacejka_c * np.arctan(s.pacejka_b * alpha))

    # -- one integration substep -------------------------------------------
    def _substep(self, throttle, brake, steer, dt):
        s, st = self.spec, self.state
        m = s.total_mass
        vx = st.vx
        vy = st.vy
        r = st.yaw_rate
        delta = float(np.clip(steer, -1.0, 1.0)) * s.max_steer

        # --- normal loads: static + aero + longitudinal transfer ---
        downforce = 0.5 * RHO_AIR * s.cla * vx * vx
        df_f = downforce * s.aero_balance_front
        df_r = downforce * (1.0 - s.aero_balance_front)
        static_f = m * G * s.front_weight_dist
        static_r = m * G * (1.0 - s.front_weight_dist)
        transfer = m * st.ax * s.cg_height / s.wheelbase  # +accel -> load to rear
        fz_f = max(static_f + df_f - transfer, 100.0)
        fz_r = max(static_r + df_r + transfer, 100.0)

        # --- slip angles (blend to kinematic at low speed to avoid singularity) ---
        vx_safe = max(vx, 1.0)
        blend = float(np.clip(vx / 3.0, 0.0, 1.0))  # 0 at rest -> 1 above 3 m/s
        alpha_f = (np.arctan2(vy + s.lf * r, vx_safe) - delta) * blend
        alpha_r = (np.arctan2(vy - s.lr * r, vx_safe)) * blend

        fy_f = self._tire_lateral(alpha_f, fz_f)
        fy_r = self._tire_lateral(alpha_r, fz_r)

        # --- longitudinal forces ---
        drive = self._drive_force(throttle, vx, st.gear)          # rear axle
        brake_cmd = float(np.clip(brake, 0.0, 1.0)) * s.max_brake_force
        # vx is clamped >= 0 (no reversing in this baseline), so brake, drag
        # and rolling resistance always oppose forward motion.
        brake_f_axle = brake_cmd * s.brake_bias_front
        brake_r_axle = brake_cmd * (1.0 - s.brake_bias_front)

        drag = 0.5 * RHO_AIR * s.cda * vx * vx
        roll = s.rolling_resistance * (fz_f + fz_r)

        # --- friction ellipse: cap each axle's longitudinal force by remaining grip ---
        fx_r_cap = np.sqrt(max((s.mu_x * fz_r) ** 2 - fy_r ** 2, 0.0))
        fx_f_cap = np.sqrt(max((s.mu_x * fz_f) ** 2 - fy_f ** 2, 0.0))
        fx_r = np.clip(drive - brake_r_axle, -fx_r_cap, fx_r_cap)
        fx_f = np.clip(-brake_f_axle, -fx_f_cap, fx_f_cap)

        fx_total = fx_r + fx_f - drag - roll

        # --- equations of motion (body frame) ---
        # Specific forces (what an accelerometer reads = the "g-force" felt).
        # These are grip-limited and are the correct quantities for load
        # transfer and for reporting cornering/braking G.
        a_long = fx_total / m
        a_lat = (fy_f * np.cos(delta) + fy_r) / m
        # Rates of change of body-frame velocities include the rotation terms.
        dvx = a_long + vy * r
        dvy = a_lat - vx * r
        r_dot = (s.lf * fy_f * np.cos(delta) - s.lr * fy_r) / s.yaw_inertia

        # semi-implicit Euler
        vx = vx + dvx * dt
        vy = vy + dvy * dt
        r = r + r_dot * dt
        vx = max(vx, 0.0)  # no reversing in this baseline

        # --- global pose ---
        st.x += (vx * np.cos(st.yaw) - vy * np.sin(st.yaw)) * dt
        st.y += (vx * np.sin(st.yaw) + vy * np.cos(st.yaw)) * dt
        st.yaw += r * dt

        st.vx, st.vy, st.yaw_rate = vx, vy, r
        st.ax = a_long   # net long. specific force -> next step's load transfer
        st.ay = a_lat    # lateral specific force (true cornering G)

    def step(self, throttle, brake, steer, gear=None, dt=0.02, substeps=2):
        """Advance the car by dt seconds.

        Args:
            throttle, brake: 0..1
            steer: -1..1 (mapped to +/- max_steer)
            gear: 1..8, or None to auto-select
            dt: control timestep (s); split into `substeps` for stability
        """
        self.state.gear = self.auto_gear(self.state.vx) if gear is None \
            else int(np.clip(gear, 1, len(self.spec.gear_ratios)))
        h = dt / substeps
        for _ in range(substeps):
            self._substep(throttle, brake, steer, h)
        return self.state
