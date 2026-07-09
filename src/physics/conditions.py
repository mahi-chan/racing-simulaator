"""
Conditions model — Layer 3 of the autonomous racing stack.

The dynamic state that modulates car performance over a stint: tire compound,
tire wear, tire temperature, fuel load, and weather. Layer 1 (the vehicle)
stays pure chassis physics; this module distills everything stint-related into
the two numbers the environment applies:

  * ``grip_multiplier()`` — scales the vehicle's tire friction (mu_x, mu_y):
        compound peak grip x wear x temperature x weather
  * ``fuel_mass``         — written into ``CarSpec.fuel_mass`` by the env
                            (heavier = slower). CarSpec caches ``total_mass``
                            in ``__post_init__``, so the env re-runs it after
                            writing — that wiring is Layer 4's job.

Conventions & units:
  * ONE aggregate tire state for the whole car, not four corners — a
    deliberate Layer 3 simplification.
  * ``load`` = total tire normal load (N), e.g. fz_f + fz_r from the vehicle.
    ``slip`` = representative slip magnitude (rad), e.g. max |slip angle|.
  * Temperatures in deg C, fuel in kg, wear dimensionless 0 (fresh) → 1 (dead).
  * Weather is held constant within a stint; changing it mid-episode is
    Layer 4/6 domain-randomization territory.
  * Every coefficient is a labeled, representative placeholder — Layer 7
    calibrates them against real telemetry. None are authoritative.

Pure-scalar stdlib Python on purpose: ``step()`` sits inside the env hot loop,
and tiny-array numpy calls cost more than they compute at this size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Tire compounds
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TireCompound:
    """Static properties of one compound. All values are Layer 7 calibration
    targets; relationships (orderings, window placement) are the contract."""

    name: str
    kind: str            # "slick" | "inter" | "wet" — row in the weather grip table
    peak_grip: float     # fresh, in-window grip multiplier (reference: soft = 1.0)
    wear_rate: float     # wear fraction per second at reference load & slip
    temp_window: tuple[float, float]  # (t_low, t_high) deg C — factor 1.0 inside


# Softer = more peak grip, faster wear, working window reached with less energy;
# rain tires run far cooler than slicks (which is also why they cook in the dry).
# wear_rate anchor: at racing load/slip (~1x, see ConditionsConfig) a soft loses
# ~0.45 of its life in a 10-minute push — a deliberately short placeholder stint.
COMPOUNDS: dict[str, TireCompound] = {
    "soft":         TireCompound("soft",         "slick", 1.000, 6.5e-4, (85.0, 105.0)),
    "medium":       TireCompound("medium",       "slick", 0.965, 4.5e-4, (90.0, 110.0)),
    "hard":         TireCompound("hard",         "slick", 0.930, 3.0e-4, (95.0, 115.0)),
    "intermediate": TireCompound("intermediate", "inter", 0.880, 5.5e-4, (60.0, 85.0)),
    "wet":          TireCompound("wet",          "wet",   0.800, 4.0e-4, (50.0, 75.0)),
}

WEATHERS = ("dry", "damp", "wet")


# ----------------------------------------------------------------------------
# Config — every stint-dynamics coefficient, labeled for Layer 7 calibration
# ----------------------------------------------------------------------------
@dataclass
class ConditionsConfig:
    # --- wear -> grip (piecewise linear, monotone decreasing by construction) ---
    wear_grip_slope: float = 0.25    # grip fraction lost per unit wear pre-cliff
    wear_cliff_start: float = 0.75   # wear where the performance cliff begins
    wear_cliff_slope: float = 1.30   # much steeper loss rate past the cliff
    wear_grip_floor: float = 0.45    # a dead tire still puts some rubber down

    # --- wear accumulation ---
    #   d(wear)/dt = wear_rate * (load / ref_load) * (c0 + c1 * |slip|)
    ref_load: float = 8000.0         # N — roughly a fueled car's static weight
    wear_slip_c0: float = 0.30       # rolling wear at zero slip
    wear_slip_c1: float = 12.0       # sliding wear per rad of slip
    #   (c0 + c1 * 0.06 rad ~= 1.0: representative racing pace = 1x wear_rate)

    # --- tire temperature ---
    #   dT/dt = q_heat * (load / ref_load) * |slip|  -  k_cool * (T - track_temp)
    q_heat: float = 55.0             # deg C/s heating at ref load and 1 rad slip
    k_cool: float = 0.06             # 1/s Newtonian cooling toward track temp
    initial_tire_temp: float = 70.0  # deg C — fresh out of the tire blankets
    tire_temp_min: float = 0.0       # state clamps (deg C)
    tire_temp_max: float = 200.0

    # --- temperature -> grip: 1.0 inside the compound window, quadratic
    #     falloff outside, floored ---
    temp_cold_width: float = 50.0    # deg C below the window over which grip fades
    temp_hot_width: float = 45.0     # deg C above the window over which grip fades
    temp_grip_floor: float = 0.60    # grip factor far outside the window

    # --- weather -> grip, by tire kind ---
    # Best tire per condition: dry = slick 1.00; damp = inter .88*.93 = .82;
    # wet = wet tire .80*.85 = .68 -> wet sits ~32% below dry (spec: 30-40%)
    # and damp between the two. Slicks in standing water are the disaster case.
    weather_grip: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "slick": {"dry": 1.00, "damp": 0.72, "wet": 0.42},
        "inter": {"dry": 0.88, "damp": 0.93, "wet": 0.80},
        "wet":   {"dry": 0.82, "damp": 0.88, "wet": 0.85},
    })
    # extra fractional grip loss per unit rain_intensity (applied in damp/wet
    # only): standing water punishes the wrong tire; full wets barely care
    rain_sensitivity: dict[str, float] = field(default_factory=lambda: {
        "slick": 0.25, "inter": 0.10, "wet": 0.0,
    })

    # --- fuel burn:  kg/s = coast + (full - coast) * throttle ---
    fuel_burn_coast: float = 0.004   # kg/s off-throttle (idle/overrun)
    fuel_burn_full: float = 0.035    # kg/s at full throttle
    #   (~55% average throttle duty -> ~0.02 kg/s ~= 105 kg over a 90-min race)
    initial_fuel: float = 100.0      # kg — race-start ballpark


# ----------------------------------------------------------------------------
# The stint state
# ----------------------------------------------------------------------------
class Conditions:
    """Mutable stint state: one tire set + fuel load + fixed weather.

    Construct at episode reset, call ``step`` every control step with the
    vehicle's current load/slip/throttle, and read ``grip_multiplier()`` and
    ``fuel_mass`` back into the vehicle.
    """

    def __init__(self, compound: str = "medium", weather: str = "dry",
                 rain_intensity: float = 0.0, track_temp: float = 30.0,
                 fuel_mass: float | None = None,
                 config: ConditionsConfig | None = None):
        if compound not in COMPOUNDS:
            raise ValueError(f"unknown compound {compound!r}, "
                             f"expected one of {sorted(COMPOUNDS)}")
        if weather not in WEATHERS:
            raise ValueError(f"unknown weather {weather!r}, "
                             f"expected one of {WEATHERS}")
        cfg = config or ConditionsConfig()
        if fuel_mass is None:
            fuel_mass = cfg.initial_fuel
        if fuel_mass < 0.0:
            raise ValueError(f"fuel_mass must be >= 0, got {fuel_mass}")
        self.config = cfg
        self.compound = COMPOUNDS[compound]
        self.weather = weather
        self.rain_intensity = min(max(float(rain_intensity), 0.0), 1.0)
        self.track_temp = float(track_temp)
        # stint state
        self.wear = 0.0                          # 0 fresh .. 1 dead
        self.tire_temp = cfg.initial_tire_temp   # deg C
        self.fuel_mass = float(fuel_mass)        # kg

    # -- dynamics -------------------------------------------------------------
    def step(self, dt: float, load: float, slip: float,
             throttle: float = 1.0) -> None:
        """Advance tire wear/temperature and burn fuel over dt seconds.

        Args:
            dt: timestep (s)
            load: total tire normal load (N), e.g. fz_f + fz_r
            slip: representative slip magnitude (rad), e.g. max |slip angle|
            throttle: 0..1 duty for fuel burn — optional beyond the spec's
                (dt, load, slip) signature; defaults to full power
        """
        cfg = self.config
        load_f = max(load, 0.0) / cfg.ref_load
        slip_m = abs(slip)

        # wear: load- and slip-proportional, never recovers, clamps at dead
        dwear = (self.compound.wear_rate * load_f
                 * (cfg.wear_slip_c0 + cfg.wear_slip_c1 * slip_m) * dt)
        self.wear = min(self.wear + dwear, 1.0)

        # temperature: friction work heats, Newtonian cooling toward the track
        heat = cfg.q_heat * load_f * slip_m
        cool = cfg.k_cool * (self.tire_temp - self.track_temp)
        t = self.tire_temp + (heat - cool) * dt
        self.tire_temp = min(max(t, cfg.tire_temp_min), cfg.tire_temp_max)

        # fuel: linear in throttle between coast and full-throttle burn rates
        thr = min(max(throttle, 0.0), 1.0)
        burn = cfg.fuel_burn_coast + (cfg.fuel_burn_full - cfg.fuel_burn_coast) * thr
        self.fuel_mass = max(self.fuel_mass - burn * dt, 0.0)

    # -- grip factors ---------------------------------------------------------
    def _wear_factor(self) -> float:
        """Piecewise-linear grip loss with a performance cliff, floored."""
        cfg = self.config
        w = self.wear
        f = 1.0 - cfg.wear_grip_slope * min(w, cfg.wear_cliff_start)
        if w > cfg.wear_cliff_start:
            f -= cfg.wear_cliff_slope * (w - cfg.wear_cliff_start)
        return max(f, cfg.wear_grip_floor)

    def _temp_factor(self) -> float:
        """1.0 inside the compound's window, quadratic falloff outside."""
        cfg = self.config
        lo, hi = self.compound.temp_window
        t = self.tire_temp
        if t < lo:
            d, width = lo - t, cfg.temp_cold_width
        elif t > hi:
            d, width = t - hi, cfg.temp_hot_width
        else:
            return 1.0
        return max(1.0 - (d / width) ** 2, cfg.temp_grip_floor)

    def _weather_factor(self) -> float:
        """Tire-kind vs weather table, with a rain-intensity penalty for the
        wrong tire when there is standing water."""
        cfg = self.config
        base = cfg.weather_grip[self.compound.kind][self.weather]
        if self.weather == "dry":
            return base
        penalty = cfg.rain_sensitivity[self.compound.kind] * self.rain_intensity
        return max(base * (1.0 - penalty), 0.05)

    # -- outputs --------------------------------------------------------------
    def grip_multiplier(self) -> float:
        """Combined factor the env applies to the vehicle's mu_x / mu_y."""
        return (self.compound.peak_grip * self._wear_factor()
                * self._temp_factor() * self._weather_factor())

    def grip_components(self) -> dict[str, float]:
        """Named factor breakdown (interpretability; Layer 4 surfaces it in info)."""
        return {
            "compound": self.compound.peak_grip,
            "wear": self._wear_factor(),
            "temperature": self._temp_factor(),
            "weather": self._weather_factor(),
        }

    def __repr__(self) -> str:
        return (f"Conditions({self.compound.name}, {self.weather}, "
                f"wear={self.wear:.2f}, temp={self.tire_temp:.0f}C, "
                f"fuel={self.fuel_mass:.1f}kg, grip={self.grip_multiplier():.3f})")
