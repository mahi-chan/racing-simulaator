"""
Quasi-steady-state (QSS) lap-time profiler — Layer 7's calibration evaluator.

Answers "what lap does this car+track support at the limit?" without a driver.
The RL driver measures *driver* skill; calibration must measure the *car*, so
this module computes the classic three-pass QSS speed profile over a Track:

  1. corner-speed cap:   largest v with lateral demand m*v^2*|kappa| within the
                         load-sensitive lateral grip available at that v,
  2. forward pass:       traction-limited acceleration (engine vs rear grip),
  3. backward pass:      brake-limited deceleration (bias-split axle caps),

iterated to a periodic fixed point (a flying lap, matching a quali reference).

Every force formula deliberately mirrors `vehicle_model.py` so the profile is
the steady-state envelope of the SAME car Layers 4-6 drive (line references in
the code). What QSS adds/simplifies, stated honestly:
  * quasi-steady state: no yaw dynamics, no slip-angle transients — the car is
    assumed balanced on the friction limit (lateral demand split across axles
    in proportion to each axle's available grip),
  * longitudinal load transfer via a 2-step fixed point per grid point (the
    dynamic model integrates it; QSS converges it),
  * stint state frozen: one `grip_multiplier` scalar for the whole lap
    (Layer 3 wear/temperature do not evolve during a single flying lap),
  * DRS modeled with Layer 4's placeholder gate (open wherever |kappa| is
    below `DRS_MAX_CURVATURE`); ERS at full deploy is already CarSpec's
    `max_power` (735 kW documented as ICE + full MGU-K in `f1_env.py`).

numpy only (project convention: physics never imports the env stack). The
DRS constants are duplicated from `src/envs/f1_env.py:EnvConfig` defaults;
tests/test_calibration.py guards against drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from src.physics.vehicle_model import G, RHO_AIR, CarSpec

# Duplicated from src.envs.f1_env.EnvConfig defaults (drift-guarded by T5):
DRS_MAX_CURVATURE = 0.0025   # 1/m — open only where R > 400 m
DRS_DRAG_FACTOR = 0.88       # cda multiplier while open
DRS_DOWNFORCE_FACTOR = 0.90  # cla multiplier while open

VMAX_HARD = 110.0            # m/s bisection ceiling (drag-limited top ~101 with DRS)
V_FLOOR = 5.0                # m/s numerical floor (F1 corners never approach this)
MIN_AXLE_LOAD = 100.0        # N — same clamp as vehicle_model._substep


@dataclass
class LapProfile:
    """QSS result: the speed profile and its derived racing landmarks."""

    s: np.ndarray            # arc-length grid (m), length n, seam point once
    v: np.ndarray            # speed at each s (m/s)
    lap_time: float          # flying-lap time (s), trapezoidal ds/v
    v_corner_cap: np.ndarray  # pure-lateral speed cap used as initialization
    drs_open: np.ndarray     # bool mask of the Layer-4 DRS gate per point
    apex_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    apex_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    brake_s: np.ndarray = field(default_factory=lambda: np.empty(0))  # per apex

    @property
    def a_long(self) -> np.ndarray:
        """Longitudinal accel per segment, (v[i+1]^2 - v[i]^2) / 2ds, wrapped."""
        ds = self.s[1] - self.s[0]
        v2 = np.square(np.append(self.v, self.v[0]))
        return np.diff(v2) / (2.0 * ds)


# -----------------------------------------------------------------------------
# engine force lookup (mirrors vehicle_model auto_gear + _drive_force)
# -----------------------------------------------------------------------------
def _engine_force_table(spec: CarSpec, v_step: float = 0.25) -> tuple[list, float]:
    """Best-gear wheel force vs speed at full throttle.

    Mirrors `F1Vehicle._drive_force` (torque curve x ratio x final drive x
    efficiency / radius, capped by max_power / max(v, 1) — the power branch
    carries no efficiency factor, exactly as vehicle_model.py:156 does) and
    `auto_gear`'s best-force gear choice including its limiter skip rule.
    """
    v = np.arange(0.0, VMAX_HARD + v_step, v_step)
    wheel_rps = np.maximum(v, 0.0) / (2 * np.pi * spec.tire_radius)
    best = np.zeros_like(v)
    n_gears = len(spec.gear_ratios)
    for gi, ratio in enumerate(spec.gear_ratios):
        rpm = np.clip(wheel_rps * 60.0 * ratio * spec.final_drive,
                      spec.rpm_idle, spec.rpm_limit)
        shape = np.exp(-((rpm / spec.rpm_limit - 0.72) ** 2) / (2 * 0.40 ** 2))
        torque = spec.peak_torque * np.clip(shape, 0.60, 1.0)
        force = (torque * ratio * spec.final_drive
                 * spec.drivetrain_efficiency / spec.tire_radius)
        # auto_gear: a gear pinned on the limiter is skipped unless it is top
        on_limiter = (rpm >= spec.rpm_limit * 0.995) & (gi < n_gears - 1)
        force = np.where(on_limiter, 0.0, force)
        best = np.maximum(best, force)
    best = np.minimum(best, spec.max_power / np.maximum(v, 1.0))
    return best.tolist(), v_step


# -----------------------------------------------------------------------------
# the profiler
# -----------------------------------------------------------------------------
def qss_lap(track, spec: CarSpec | None = None, *, grip_multiplier: float = 1.0,
            drs: bool = True, max_rounds: int = 4, tol: float = 1e-6,
            landmarks: bool = True) -> LapProfile:
    """Compute the QSS speed profile and flying-lap time for `track` + `spec`.

    Args:
        track: a Layer 2 `Track` (its uniform s-grid and curvature are used).
        spec: CarSpec (default CarSpec()); `fuel_mass` should already reflect
            the lap being modeled (e.g. ~15 kg for a quali lap).
        grip_multiplier: frozen Layer 3 `Conditions.grip_multiplier()` scalar,
            applied to mu_x and mu_y exactly as f1_env.py:312 does.
        drs: apply the Layer 4 DRS gate on low-curvature stretches.
        max_rounds: forward+backward sweep rounds (each wraps the lap twice);
            iteration stops early once the profile changes < `tol` m/s.
        landmarks: also extract apexes and braking points.
    """
    spec = spec or CarSpec()
    m = spec.total_mass
    mu_x = spec.mu_x * grip_multiplier
    mu_y = spec.mu_y * grip_multiplier
    ls = spec.load_sensitivity
    weight = m * G
    static_f = weight * spec.front_weight_dist
    static_r = weight * (1.0 - spec.front_weight_dist)
    abf = spec.aero_balance_front
    bias_f = spec.brake_bias_front
    bias_r = 1.0 - bias_f
    trans_coef = m * spec.cg_height / spec.wheelbase  # * a_long -> axle shift
    rr = spec.rolling_resistance

    # -- grid (drop the duplicated closing point; index arithmetic wraps) ----
    s_grid = track.s[:-1]
    n = len(s_grid)
    ds = float(track.s[1] - track.s[0])
    kappa = np.abs(track.curvature[:-1])

    drs_mask = (kappa < DRS_MAX_CURVATURE) if drs else np.zeros(n, bool)
    q_cla = 0.5 * RHO_AIR * spec.cla * np.where(drs_mask, DRS_DOWNFORCE_FACTOR, 1.0)
    q_cda = 0.5 * RHO_AIR * spec.cda * np.where(drs_mask, DRS_DRAG_FACTOR, 1.0)

    def axle_mu(fz):
        # vehicle_model._tire_lateral:163 — load-sensitive lateral grip, floored
        return np.maximum(mu_y * (1.0 - ls * (fz / weight - 1.0)), 0.5)

    # -- pass 1: pure-lateral corner-speed cap, vectorized bisection ---------
    def lateral_reserve(v):
        """available lateral force minus demand at speed v (steady state)."""
        df = q_cla * v * v
        fz_f = np.maximum(static_f + df * abf, MIN_AXLE_LOAD)
        fz_r = np.maximum(static_r + df * (1.0 - abf), MIN_AXLE_LOAD)
        avail = axle_mu(fz_f) * fz_f + axle_mu(fz_r) * fz_r
        return avail - m * v * v * kappa

    lo = np.full(n, V_FLOOR)
    hi = np.full(n, VMAX_HARD)
    unlimited = lateral_reserve(hi) >= 0.0   # grip outgrows demand -> flat out
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        ok = lateral_reserve(mid) >= 0.0
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    v_cap = np.where(unlimited, VMAX_HARD, lo)

    # -- passes 2+3: python-float recurrences (Layer 3 hot-loop discipline) --
    feng, v_step = _engine_force_table(spec)
    feng_top = len(feng) - 1
    kap = kappa.tolist()
    qla = q_cla.tolist()
    qda = q_cda.tolist()
    v = v_cap.tolist()
    max_brake = spec.max_brake_force
    two_ds = 2.0 * ds
    vfloor2 = V_FLOOR * V_FLOOR

    def long_accel(vv: float, kk: float, ql: float, qd: float,
                   braking: bool) -> float:
        """Grip/power/brake-limited longitudinal accel at speed vv, |curv| kk."""
        a = 0.0
        for _ in range(2):  # load-transfer fixed point
            df = ql * vv * vv
            trans = trans_coef * a
            fz_f = static_f + df * abf - trans
            if fz_f < MIN_AXLE_LOAD:
                fz_f = MIN_AXLE_LOAD
            fz_r = static_r + df * (1.0 - abf) + trans
            if fz_r < MIN_AXLE_LOAD:
                fz_r = MIN_AXLE_LOAD
            mu_f = mu_y * (1.0 - ls * (fz_f / weight - 1.0))
            if mu_f < 0.5:
                mu_f = 0.5
            mu_r = mu_y * (1.0 - ls * (fz_r / weight - 1.0))
            if mu_r < 0.5:
                mu_r = 0.5
            fy_need = m * vv * vv * kk
            avail = mu_f * fz_f + mu_r * fz_r
            frac = fy_need / avail if avail > 0.0 else 1.0
            if frac > 1.0:
                frac = 1.0
            # friction ellipse per axle (vehicle_model:207-208, mu_x on load):
            # lateral demand split in proportion to each axle's available grip
            fy_f = frac * mu_f * fz_f
            fy_r = frac * mu_r * fz_r
            cap_f2 = (mu_x * fz_f) ** 2 - fy_f * fy_f
            cap_r2 = (mu_x * fz_r) ** 2 - fy_r * fy_r
            cap_f = math.sqrt(cap_f2) if cap_f2 > 0.0 else 0.0
            cap_r = math.sqrt(cap_r2) if cap_r2 > 0.0 else 0.0
            drag = qd * vv * vv
            roll = rr * (fz_f + fz_r)
            if braking:
                b = max_brake
                if cap_f / bias_f < b:
                    b = cap_f / bias_f
                if cap_r / bias_r < b:
                    b = cap_r / bias_r
                a = -(b + drag + roll) / m
            else:
                idx = vv / v_step
                fe = feng[int(idx) if idx < feng_top else feng_top]
                if fe > cap_r:
                    fe = cap_r
                a = (fe - drag - roll) / m
        return a

    prev = None
    for _ in range(max_rounds):
        # forward (traction): wrap the lap twice for periodicity
        for j in range(2 * n):
            i = j % n
            i1 = (i + 1) % n
            vi = v[i]
            a = long_accel(vi, kap[i], qla[i], qda[i], braking=False)
            v2 = vi * vi + two_ds * a
            vn = math.sqrt(v2) if v2 > vfloor2 else V_FLOOR
            if vn < v[i1]:
                v[i1] = vn
        # backward (braking)
        for j in range(2 * n, 0, -1):
            i = (j - 1) % n
            i1 = j % n
            vi1 = v[i1]
            a = long_accel(vi1, kap[i1], qla[i1], qda[i1], braking=True)
            v2 = vi1 * vi1 - two_ds * a  # a < 0: decel capability
            vn = math.sqrt(v2) if v2 > vfloor2 else V_FLOOR
            if vn < v[i]:
                v[i] = vn
        arr = np.asarray(v)
        if prev is not None and float(np.max(np.abs(arr - prev))) < tol:
            break
        prev = arr.copy()

    v_arr = np.asarray(v)
    # trapezoidal flying-lap time over the closed loop
    v_next = np.roll(v_arr, -1)
    lap_time = float(np.sum(two_ds / (v_arr + v_next)))

    profile = LapProfile(s=s_grid, v=v_arr, lap_time=lap_time,
                         v_corner_cap=v_cap, drs_open=drs_mask)
    if landmarks:
        profile.apex_s, profile.apex_v = find_apexes(s_grid, v_arr,
                                                     track.length)
        profile.brake_s = braking_points(s_grid, v_arr, profile.apex_s,
                                         track.length)
    return profile


# -----------------------------------------------------------------------------
# landmarks (shared with real-telemetry traces via src/utils/validation.py)
# -----------------------------------------------------------------------------
def find_apexes(s: np.ndarray, v: np.ndarray, length: float,
                min_separation: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Local speed minima over a closed lap (wrap-safe), min-separated.

    Mirrors Track._detect_corners' seam trick: roll so the global maximum sits
    at index 0, find local minima of the rolled trace, map back. numpy only
    (physics convention), so the peak picking is done by hand: strict-left /
    non-strict-right minima (a flat-bottom apex counts once), then greedy
    keep-the-slowest with a circular minimum separation.
    """
    n = len(s)
    if n < 8:
        return np.empty(0), np.empty(0)
    shift = int(np.argmax(v))
    r = np.roll(v, -shift)
    is_min = np.zeros(n, bool)
    is_min[1:-1] = (r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:])
    cand = np.flatnonzero(is_min)
    if len(cand) == 0:
        return np.empty(0), np.empty(0)
    spacing = length / n
    min_gap = min_separation / spacing
    kept: list[int] = []
    for c in cand[np.argsort(r[cand])]:          # deepest minima win ties
        if all(min(abs(c - k), n - abs(c - k)) >= min_gap for k in kept):
            kept.append(int(c))
    idx = np.sort([(k + shift) % n for k in kept])
    return s[idx], v[idx]


def braking_points(s: np.ndarray, v: np.ndarray, apex_s: np.ndarray,
                   length: float, decel_threshold: float = 2.0,
                   search_back: float = 250.0,
                   accel_abort: float = 1.5) -> np.ndarray:
    """Brake-onset s for each apex; NaN where the corner is taken flat-out.

    Walking upstream from the apex, first skip the trail-off/arc region where
    deceleration sits below `decel_threshold` m/s^2 (at most `search_back` m,
    aborting with NaN if a strongly accelerating stretch — a previous corner's
    exit, > `accel_abort` m/s^2 — is crossed first), then continue through the
    sustained-braking zone; its first sample is the braking point.
    """
    n = len(s)
    ds = length / n
    v2 = np.square(v)
    decel = (v2 - np.square(np.roll(v, -1))) / (2.0 * ds)  # >0 = slowing i->i+1
    out = np.empty(len(apex_s))
    for a, sa in enumerate(apex_s):
        i = int(round(sa / ds)) % n
        j = (i - 1) % n
        skipped = 0.0
        while decel[j] <= decel_threshold and skipped < search_back:
            if decel[j] < -accel_abort:      # crossed an exit under power
                skipped = search_back
                break
            j = (j - 1) % n
            skipped += ds
        if decel[j] <= decel_threshold or skipped >= search_back:
            out[a] = np.nan                  # no braking event: flat-out
            continue
        steps = 0
        while decel[j] > decel_threshold and steps < n:
            j = (j - 1) % n
            steps += 1
        out[a] = s[(j + 1) % n]
    return out
