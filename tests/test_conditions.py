"""Layer 3 acceptance tests — conditions model (LAYER_SPECS.md § Layer 3).

Runs two ways:
  * `pytest tests/test_conditions.py`   — standard test run
  * `python tests/test_conditions.py`   — standalone PASS/FAIL report

Fully offline: this layer has no network path at all. T5 and T6 integrate with
the Layer 1 vehicle to prove fuel mass and the grip multiplier actually change
car performance the way Layer 4 will wire them.
"""
import math
import sys
import time
from pathlib import Path
from unittest import SkipTest  # honored by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
from src.physics.conditions import COMPOUNDS, WEATHERS, Conditions
from src.physics.vehicle_model import CarSpec, F1Vehicle, G


def _mid_window(name: str) -> float:
    lo, hi = COMPOUNDS[name].temp_window
    return 0.5 * (lo + hi)


def _fresh(name: str, weather: str = "dry", **kw) -> Conditions:
    """Fresh tire forced to mid-window temperature, isolating other factors."""
    c = Conditions(compound=name, weather=weather, **kw)
    c.tire_temp = _mid_window(name)
    return c


# ---------------------------------------------------------------------------
# T1 — compound orderings: grip soft > medium > hard, wear reversed  [bullet 1]
# ---------------------------------------------------------------------------
def test_t1_compound_ordering():
    slicks = ("soft", "medium", "hard")
    grip = {n: _fresh(n).grip_multiplier() for n in slicks}
    assert grip["soft"] > grip["medium"] > grip["hard"], \
        f"peak grip ordering wrong: {grip}"
    wr = {n: COMPOUNDS[n].wear_rate for n in slicks}
    assert wr["soft"] > wr["medium"] > wr["hard"], f"wear rate ordering wrong: {wr}"
    # the ordering must also hold INTEGRATED over an identical hard stint,
    # not just as attributes
    wear = {}
    for n in slicks:
        c = _fresh(n)
        for _ in range(30_000):                    # 600 s at dt = 0.02
            c.step(0.02, load=9000.0, slip=0.06)   # sustained racing load/slip
        wear[n] = c.wear
    assert wear["soft"] > wear["medium"] > wear["hard"], \
        f"integrated wear ordering wrong: {wear}"
    assert 0.0 < wear["hard"] and wear["soft"] < 1.0, \
        f"10-min stint wear implausible: {wear}"
    print(f"    fresh grip soft {grip['soft']:.3f} > medium {grip['medium']:.3f} "
          f"> hard {grip['hard']:.3f}; 10-min wear "
          + ", ".join(f"{n} {wear[n]:.2f}" for n in slicks))


# ---------------------------------------------------------------------------
# T2 — grip decreases monotonically with wear                       [bullet 2]
# ---------------------------------------------------------------------------
def test_t2_wear_monotonic_grip_loss():
    c = _fresh("medium")
    grips = []
    for w in np.linspace(0.0, 1.0, 201):
        c.wear = float(w)
        grips.append(c.grip_multiplier())
    grips = np.array(grips)
    assert np.all(np.diff(grips) <= 1e-12), \
        "grip is not monotonically non-increasing in wear"
    total_drop = 1.0 - grips[-1] / grips[0]
    assert total_drop > 0.15, f"a worn-out tire only lost {total_drop:.0%} grip"
    # stepping never decreases wear, and wear clamps at exactly 1
    c.wear = 0.0
    prev = 0.0
    for _ in range(2000):
        c.step(0.5, load=20_000.0, slip=0.5)   # absurd abuse to reach the clamp
        assert c.wear >= prev, "wear went backwards"
        prev = c.wear
    assert c.wear == 1.0, f"wear did not clamp at 1 (got {c.wear})"
    print(f"    grip fresh {grips[0]:.3f} -> dead {grips[-1]:.3f} "
          f"(-{total_drop:.0%}), monotone throughout")


# ---------------------------------------------------------------------------
# T3 — weather: wet 30-40% below dry, inter between in the damp     [bullet 3]
# ---------------------------------------------------------------------------
def test_t3_weather():
    def grip(compound, weather, rain=0.0):
        return _fresh(compound, weather=weather,
                      rain_intensity=rain).grip_multiplier()

    g_dry = grip("soft", "dry")             # best dry choice
    g_damp = grip("intermediate", "damp")   # best damp choice
    g_wet = grip("wet", "wet")              # best wet choice
    ratio = g_wet / g_dry
    assert 0.60 <= ratio <= 0.70, \
        f"wet grip is {1 - ratio:.0%} below dry, spec wants 30-40%"
    assert g_wet < g_damp < g_dry, \
        f"damp does not sit between wet and dry: {g_wet:.3f} / {g_damp:.3f} / {g_dry:.3f}"
    # inters exist for a reason: best of the three tire classes in the damp
    assert g_damp > grip("soft", "damp") and g_damp > grip("wet", "damp"), \
        "intermediate is not the best tire in damp conditions"
    # slicks in standing water are the worst combination on the board
    best_slick_wet = max(grip(n, "wet") for n in ("soft", "medium", "hard"))
    worst_rain_tire = min(grip(n, w)
                          for n in ("intermediate", "wet") for w in WEATHERS)
    assert best_slick_wet < worst_rain_tire, \
        "a slick in the wet should be worse than any rain tire in any weather"
    # more rain never helps a slick
    seq = [grip("soft", "wet", rain=r) for r in np.linspace(0.0, 1.0, 11)]
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:])), \
        "slick grip increased with rain intensity"
    print(f"    dry {g_dry:.3f} / damp(inter) {g_damp:.3f} / wet {g_wet:.3f} "
          f"-> wet {1 - ratio:.0%} below dry")


# ---------------------------------------------------------------------------
# T4 — temperature window: unity inside, loss outside, real dynamics [bullet 5]
# ---------------------------------------------------------------------------
def test_t4_temperature_window():
    c = _fresh("soft")
    lo, hi = COMPOUNDS["soft"].temp_window
    for t in (lo, 0.5 * (lo + hi), hi):
        c.tire_temp = t
        assert c.grip_components()["temperature"] == 1.0, \
            f"temperature factor != 1 inside the window at {t} C"
    c.tire_temp = lo - 30.0
    cold = c.grip_components()["temperature"]
    c.tire_temp = hi + 40.0
    hot = c.grip_components()["temperature"]
    floor = c.config.temp_grip_floor
    assert cold <= 0.90 and hot <= 0.90, \
        f"outside the window barely penalized: cold {cold:.2f}, hot {hot:.2f}"
    assert cold >= floor and hot >= floor, "temperature factor fell through its floor"
    # monotone: grip only gets worse the further outside the window
    for sign, edge in ((-1.0, lo), (+1.0, hi)):
        seq = []
        for d in np.linspace(0.0, 80.0, 33):
            c.tire_temp = edge + sign * d
            seq.append(c.grip_components()["temperature"])
        assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:])), \
            f"temperature factor not monotone on the {'hot' if sign > 0 else 'cold'} side"
    # dynamics: hard running heats a cold tire...
    dyn = Conditions(compound="soft", weather="dry", track_temp=35.0)
    dyn.tire_temp = 35.0                            # stone cold
    for _ in range(1500):                           # 30 s at dt = 0.02
        dyn.step(0.02, load=12_000.0, slip=0.08)    # heavy cornering
    heated = dyn.tire_temp
    assert heated > 35.0 + 25.0, f"30 s of abuse only reached {heated:.0f} C"
    # ...and pure rolling cools it back toward track temperature
    for _ in range(6000):                           # 120 s at dt = 0.02
        dyn.step(0.02, load=8000.0, slip=0.0)
    assert abs(dyn.tire_temp - 35.0) < 5.0, \
        f"tire did not cool toward track temp: {dyn.tire_temp:.0f} C"
    print(f"    factor in-window 1.00, 30C cold {cold:.2f}, 40C hot {hot:.2f}; "
          f"dynamics 35 -> {heated:.0f} C under load, cools back")


# ---------------------------------------------------------------------------
# T5 — fuel burns down; heavier car is measurably slower (Layer 1)  [bullet 4]
# ---------------------------------------------------------------------------
def test_t5_fuel_burn_and_mass_effect():
    # burn is linear in time at fixed throttle, coast < full, clamps at zero
    full = Conditions(compound="medium", weather="dry")
    f0 = full.fuel_mass
    for _ in range(5000):                           # 100 s at dt = 0.02
        full.step(0.02, load=8000.0, slip=0.02, throttle=1.0)
    burn_full = f0 - full.fuel_mass
    coast = Conditions(compound="medium", weather="dry")
    for _ in range(5000):
        coast.step(0.02, load=8000.0, slip=0.02, throttle=0.0)
    burn_coast = f0 - coast.fuel_mass
    cfg = full.config
    assert abs(burn_full - 100.0 * cfg.fuel_burn_full) < 1e-6
    assert abs(burn_coast - 100.0 * cfg.fuel_burn_coast) < 1e-6
    assert 0.0 < burn_coast < burn_full
    nearly_dry = Conditions(compound="medium", weather="dry", fuel_mass=0.05)
    for _ in range(500):
        nearly_dry.step(0.02, load=8000.0, slip=0.02)
        assert nearly_dry.fuel_mass >= 0.0, "fuel went negative"
    assert nearly_dry.fuel_mass == 0.0, "empty tank did not clamp at zero"

    # Layer 1 integration. No driver exists until Layer 5, so the "faster
    # lap" is a deterministic scripted run: full-throttle 0 -> 800 m, then
    # threshold-brake to 20 m/s. Only CarSpec.fuel_mass differs.
    def timed_run(fuel_kg: float) -> float:
        car = F1Vehicle(CarSpec(fuel_mass=fuel_kg))
        car.reset()
        t, dt = 0.0, 0.02
        while car.state.x < 800.0:
            car.step(throttle=1.0, brake=0.0, steer=0.0, dt=dt)
            t += dt
            assert t < 60.0, "car never covered 800 m"
        while car.state.vx > 20.0:
            car.step(throttle=0.0, brake=1.0, steer=0.0, dt=dt)
            t += dt
            assert t < 90.0, "car never slowed to 20 m/s"
        return t

    t_light, t_mid, t_heavy = timed_run(0.0), timed_run(50.0), timed_run(100.0)
    assert t_light < t_mid < t_heavy, \
        f"run time not monotone in fuel: {t_light:.2f} / {t_mid:.2f} / {t_heavy:.2f} s"
    assert t_heavy - t_light > 0.25, \
        f"100 kg of fuel only cost {t_heavy - t_light:.3f} s"
    print(f"    burn {burn_full:.2f} kg per 100 s full / {burn_coast:.2f} coast; "
          f"scripted run 0/50/100 kg fuel: "
          f"{t_light:.2f}/{t_mid:.2f}/{t_heavy:.2f} s")


# ---------------------------------------------------------------------------
# T6 — grip_multiplier feeds the vehicle: degraded tires corner slower
# ---------------------------------------------------------------------------
def test_t6_grip_multiplier_into_vehicle():
    base = CarSpec()

    def peak_lateral_g(mult: float, kmh: float = 200.0) -> float:
        """Peak sustained lateral G, validate_vehicle.py's method, with the
        conditions multiplier applied to tire mu exactly as Layer 4 will."""
        spec = CarSpec(mu_x=base.mu_x * mult, mu_y=base.mu_y * mult)
        best = 0.0
        for steer in np.linspace(0.1, 1.0, 19):
            car = F1Vehicle(spec)
            car.reset(speed=kmh / 3.6)
            peak = 0.0
            for _ in range(200):    # 1 s at dt = 0.005, turn-in at the limit
                thr = 0.5 if car.state.vx > kmh / 3.6 * 0.97 else 1.0
                st = car.step(throttle=thr, brake=0.0, steer=steer, dt=0.005)
                peak = max(peak, abs(st.ay) / G)
            best = max(best, peak)
        return best

    fresh = _fresh("soft")
    gm_fresh = fresh.grip_multiplier()
    assert abs(gm_fresh - 1.0) < 1e-9, \
        f"reference state (fresh soft, in-window, dry) should be 1.0, got {gm_fresh}"

    degraded = Conditions(compound="hard", weather="damp")
    degraded.wear = 0.60
    degraded.tire_temp = COMPOUNDS["hard"].temp_window[0] - 15.0   # under-temp
    gm_deg = degraded.grip_multiplier()
    assert 0.35 < gm_deg < 0.75, \
        f"degraded multiplier {gm_deg:.3f} outside the intended test band"

    g_fresh = peak_lateral_g(gm_fresh)
    g_deg = peak_lateral_g(gm_deg)
    ratio = g_deg / g_fresh
    deficit = 1.0 - gm_deg
    assert ratio < 1.0 - 0.5 * deficit, \
        f"multiplier {gm_deg:.2f} only cut cornering to {ratio:.2f} of fresh"
    assert ratio > 0.25, "degraded car lost implausibly much cornering"
    print(f"    multiplier {gm_deg:.2f} -> peak lateral {g_deg:.1f} G "
          f"vs fresh {g_fresh:.1f} G (ratio {ratio:.2f})")


# ---------------------------------------------------------------------------
# T7 — robustness, determinism, throughput (env hot-loop budget)
# ---------------------------------------------------------------------------
def test_t7_robustness_performance_determinism():
    rng = np.random.default_rng(42)
    n = 100_000
    loads = rng.uniform(0.0, 25_000.0, n).tolist()
    slips = rng.uniform(0.0, 0.5, n).tolist()
    throttles = rng.uniform(0.0, 1.0, n).tolist()

    def run() -> Conditions:
        c = Conditions(compound="soft", weather="damp", rain_intensity=0.5,
                       track_temp=28.0, fuel_mass=105.0)
        prev_fuel = c.fuel_mass
        for i in range(n):
            c.step(0.02, loads[i], slips[i], throttles[i])
            if i % 5000 == 0:   # spot-check invariants along the way
                assert 0.0 <= c.wear <= 1.0
                assert (c.config.tire_temp_min <= c.tire_temp
                        <= c.config.tire_temp_max)
                assert 0.0 <= c.fuel_mass <= prev_fuel
                prev_fuel = c.fuel_mass
                g = c.grip_multiplier()
                assert math.isfinite(g) and 0.0 < g <= 1.2
        return c

    t0 = time.perf_counter()
    a = run()
    rate = n / (time.perf_counter() - t0)
    b = run()   # this layer has no RNG of its own -> bit-identical rerun
    assert (a.wear, a.tire_temp, a.fuel_mass) == (b.wear, b.tire_temp, b.fuel_mass), \
        "identical input sequences produced different states"
    for v in (a.wear, a.tire_temp, a.fuel_mass, a.grip_multiplier()):
        assert math.isfinite(v), "non-finite state after random abuse"
    assert 0.0 <= a.wear <= 1.0 and a.fuel_mass >= 0.0
    assert rate > 50_000.0, f"conditions.step too slow: {rate:,.0f} steps/s"
    print(f"    {n:,} random steps: {rate:,.0f} steps/s, deterministic, "
          f"state stayed bounded")


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_compound_ordering,
    test_t2_wear_monotonic_grip_loss,
    test_t3_weather,
    test_t4_temperature_window,
    test_t5_fuel_burn_and_mass_effect,
    test_t6_grip_multiplier_into_vehicle,
    test_t7_robustness_performance_determinism,
]

if __name__ == "__main__":
    print("=" * 58)
    print("F1 CONDITIONS MODEL — LAYER 3 VALIDATION")
    print("=" * 58)
    n_pass = n_fail = n_skip = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
        except SkipTest as e:
            print(f"  SKIP  {name}  ({e})")
            n_skip += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
            n_fail += 1
        else:
            print(f"  PASS  {name}")
            n_pass += 1
    print("=" * 58)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 58)
    sys.exit(1 if n_fail else 0)
