"""Layer 4 acceptance tests — F1Env gym wrapper (LAYER_SPECS.md § Layer 4).

Runs two ways:
  * `pytest tests/test_env.py`   — standard test run
  * `python tests/test_env.py`   — standalone PASS/FAIL report

Fully offline: everything runs on the synthetic track. T6/T7 use a hand-coded
centerline pursuit controller as the "driver" (no RL until Layer 5) and check
real lap numbers — times and speeds — not just reward. The controller is a
white-box test helper: it reads the env's internal track relation directly
instead of decoding the normalized observation.
"""
import math
import sys
import time
from pathlib import Path
from unittest import SkipTest  # honored by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np

from src.envs.f1_env import COMPOUND_ORDER, EnvConfig, F1Env
from src.physics.conditions import WEATHERS
from src.physics.vehicle_model import G, RHO_AIR
from src.tracks.track import Track

TRACK = Track.from_synthetic()  # built once and shared; construction ~1 s

REWARD_KEYS = {"progress", "speed", "track", "smooth", "tire", "fuel",
               "terminal"}

# fixed benign episode for scripted runs: dry, mid setup, start of the main straight
BENIGN = dict(weather="dry", compound="medium", rain_intensity=0.0,
              track_temp=30.0, fuel=30.0, aero_level=0.5, brake_bias=0.58,
              final_drive=3.0, s0=0.0, v0=40.0, lateral=0.0,
              heading_error=0.0)


def make_env(**cfg_kw) -> F1Env:
    return F1Env(track=TRACK, config=EnvConfig(**cfg_kw) if cfg_kw else None)


# ---------------------------------------------------------------------------
# hand-coded pursuit driver (spec: "simple hand-coded pursuit controller")
# ---------------------------------------------------------------------------
class PursuitDriver:
    """Pure-pursuit steering + physics-aware speed target.

    Corner speeds come from the car's actual lateral capability, which grows
    with speed through downforce:  v^2 * kappa = margin * mu_y * g * (1 + B v^2)
    with B = 0.5 * rho * cla / (m g)  — solved in closed form. mu is read from
    the live (grip-modulated) spec, so the driver naturally slows in the rain
    or on dead tires. Deliberately conservative margins: this driver exists to
    complete laps and beat a random policy, not to be fast.
    """

    V_CAP = 75.0        # m/s ceiling so braking always fits the horizon
    MARGIN_LAT = 0.70   # fraction of true lateral capability to use
    MARGIN_BRK = 0.60   # fraction of true braking capability to use
    BRAKE_EARLY = 10.0  # m of anticipation subtracted from braking distances
    HORIZON = np.arange(5.0, 425.0, 10.0)  # m ahead for braking anticipation

    def __init__(self, env: F1Env):
        self.env = env
        self._kappa_max = float(np.max(np.abs(env.track.curvature)))

    def _corner_speeds(self, kappa_abs: np.ndarray) -> np.ndarray:
        spec = self.env.vehicle.spec  # mu already carries the grip multiplier
        A = self.MARGIN_LAT * spec.mu_y * G
        B = 0.5 * RHO_AIR * spec.cla / (spec.total_mass * G)
        denom = kappa_abs - A * B          # <= 0 means flat-out at any speed
        v = np.full_like(kappa_abs, self.V_CAP)
        tight = denom > 1e-9
        v[tight] = np.sqrt(A / denom[tight])
        return np.minimum(v, self.V_CAP)

    def act(self) -> np.ndarray:
        env = self.env
        st = env.vehicle.state
        spec = env.vehicle.spec
        track = env.track
        s = env._s
        v = st.vx

        # steering: pure pursuit toward a point ~0.5 s ahead on the centerline
        look = min(max(0.5 * v, 10.0), 50.0)
        tx, ty = env._position_at(s + look)
        alpha = env._wrap_pi(math.atan2(ty - st.y, tx - st.x) - st.yaw)
        delta = math.atan2(2.0 * spec.wheelbase * math.sin(alpha), look)
        steer = max(min(delta / spec.max_steer, 1.0), -1.0)

        # speed target: worst upcoming corner, braking-distance aware, with
        # the braking budget evaluated at the (worst-case) corner speed
        kap = np.abs(np.asarray(track.curvature_at(s + self.HORIZON)))
        v_corner = self._corner_speeds(kap)
        B = 0.5 * RHO_AIR * spec.cla / (spec.total_mass * G)
        a_brk = self.MARGIN_BRK * spec.mu_x * G * (1.0 + B * v_corner ** 2)
        dist = np.maximum(self.HORIZON - self.BRAKE_EARLY, 0.0)
        v_allow = np.sqrt(v_corner ** 2 + 2.0 * a_brk * dist)
        k_here = abs(float(track.curvature_at(s)))
        v_here = float(self._corner_speeds(np.array([k_here]))[0])
        # never outrun the horizon: cap so the track's slowest corner is
        # reachable from top speed within it, whatever the current grip
        v_slow = float(self._corner_speeds(np.array([self._kappa_max]))[0])
        a_slow = self.MARGIN_BRK * spec.mu_x * G * (1.0 + B * v_slow ** 2)
        v_cap = math.sqrt(v_slow ** 2 + 2.0 * a_slow
                          * (self.HORIZON[-1] - self.BRAKE_EARLY - 20.0))
        v_target = min(v_here, float(v_allow.min()), v_cap)

        err = v_target - v
        throttle = max(min(0.35 * err, 1.0), 0.0)
        brake = max(min(-0.5 * err, 1.0), 0.0)
        if abs(math.atan2(st.vy, max(v, 0.5))) > 0.3:
            throttle = 0.0  # sliding: lift and let the tires recover
        gear = env.vehicle.auto_gear(max(v, 1.0))
        gear_axis = 2.0 * (gear - 1) / 7.0 - 1.0  # exact inverse of the env map
        ers = 1.0 if (throttle > 0.9 and k_here < 0.002) else -1.0
        return np.array([2.0 * throttle - 1.0, 2.0 * brake - 1.0, steer,
                         gear_axis, ers, 1.0], dtype=np.float32)


def drive_lap(options: dict, seed: int = 11):
    """Run the pursuit driver until lap truncation; hard-fail on termination.

    Returns (total_return, lap_time, avg_speed, max_speed).
    """
    env = make_env()
    env.reset(seed=seed, options=options)
    driver = PursuitDriver(env)
    total, speed_sum, speed_max = 0.0, 0.0, 0.0
    for i in range(env.config.max_steps):
        obs, r, term, trunc, info = env.step(driver.act())
        total += r
        speed_sum += info["speed"]
        speed_max = max(speed_max, info["speed"])
        if term:
            raise AssertionError(
                f"pursuit driver terminated ({info['termination']}) at "
                f"s={info['s']:.0f} m, lap fraction {info['lap_fraction']:.2f}")
        if trunc:
            assert "lap_time" in info, "truncated without completing the lap"
            return total, info["lap_time"], speed_sum / (i + 1), speed_max
    raise AssertionError("driver never finished the lap within max_steps")


# ---------------------------------------------------------------------------
# T1 — gymnasium API compliance                                     [bullet 1]
# ---------------------------------------------------------------------------
def test_t1_check_env():
    from gymnasium.utils.env_checker import check_env
    env = make_env()
    check_env(env, skip_render_check=True)
    print(f"    gymnasium check_env passed "
          f"(obs {env.observation_space.shape[0]} dims, action 6 dims)")


# ---------------------------------------------------------------------------
# T2 — 10k random steps: no NaN/inf, no crash, state stays bounded  [bullet 2]
# ---------------------------------------------------------------------------
def test_t2_random_steps():
    env = make_env()
    env.reset(seed=123)
    env.action_space.seed(123)
    lo, hi = env.observation_space.low, env.observation_space.high
    ends: dict[str, int] = {}
    episodes = 0
    for i in range(10_000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {i}"
        assert np.all(obs >= lo) and np.all(obs <= hi), f"obs out of space at {i}"
        assert math.isfinite(r), f"non-finite reward at step {i}"
        assert all(math.isfinite(v)
                   for v in info["reward_components"].values())
        assert 0.0 <= info["wear"] <= 1.0
        assert info["fuel"] >= 0.0
        assert 0.0 <= info["battery"] <= env.config.ers_capacity
        if term or trunc:
            key = info.get("termination", "truncated")
            ends[key] = ends.get(key, 0) + 1
            episodes += 1
            env.reset(seed=1000 + episodes)
    dist = ", ".join(f"{k} x{v}" for k, v in sorted(ends.items()))
    print(f"    10,000 random steps / {episodes} episodes, all finite "
          f"and in-space; endings: {dist}")


# ---------------------------------------------------------------------------
# T3 — reward components: named, complete, sum to the reward        [bullet 3]
# ---------------------------------------------------------------------------
def test_t3_reward_components():
    env = make_env()
    env.reset(seed=7)
    env.action_space.seed(7)
    for i in range(500):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        rc = info["reward_components"]
        assert set(rc) == REWARD_KEYS, f"unexpected component keys {set(rc)}"
        assert abs(sum(rc.values()) - r) < 1e-9, \
            f"components sum {sum(rc.values())} != reward {r}"
        if term or trunc:
            env.reset(seed=100 + i)

    # progress must be wrap-correct: crossing the start line is a small
    # POSITIVE step, not +/- track length
    env2 = make_env()
    env2.reset(seed=1, options=dict(BENIGN, s0=TRACK.length - 5.0))
    coast = np.array([-0.4, -1.0, 0.0, 0.2, -1.0, -1.0], dtype=np.float32)
    crossed = False
    for _ in range(50):
        obs, r, term, trunc, info = env2.step(coast)
        p = info["reward_components"]["progress"]
        assert -5.0 < p < 5.0, f"progress jumped by {p:.1f} at the line"
        if info["s"] < 10.0:
            crossed = True
            break
    assert crossed, "car never crossed the start line"
    print("    components named & sum equals reward on every step; "
          "start-line crossing is a small positive progress step")


# ---------------------------------------------------------------------------
# T4 — domain randomization coverage + seeded determinism        [reset bullet]
# ---------------------------------------------------------------------------
def test_t4_domain_randomization_and_determinism():
    env = make_env()
    dr = env.config.dr
    weathers, compounds, aeros = set(), set(), []
    for seed in range(60):
        obs, info = env.reset(seed=seed)
        su = info["setup"]
        weathers.add(su["weather"])
        compounds.add(su["compound"])
        aeros.append(su["aero_level"])
        assert su["weather"] in WEATHERS and su["compound"] in COMPOUND_ORDER
        assert dr.fuel_range[0] <= su["fuel"] <= dr.fuel_range[1]
        assert dr.aero_level_range[0] <= su["aero_level"] <= dr.aero_level_range[1]
        assert dr.brake_bias_range[0] <= su["brake_bias"] <= dr.brake_bias_range[1]
        assert dr.final_drive_range[0] <= su["final_drive"] <= dr.final_drive_range[1]
        assert 0.0 <= su["rain_intensity"] <= 1.0
    assert weathers == set(WEATHERS), f"60 resets only saw weathers {weathers}"
    assert len(compounds) >= 3, f"60 resets only saw compounds {compounds}"
    assert np.std(aeros) > 0.05, "aero level barely varies"

    # same seed -> bit-identical episode (reset AND a 20-step rollout)
    env_a, env_b = make_env(), make_env()
    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    assert info_a["setup"] == info_b["setup"]
    acts = np.random.default_rng(0).uniform(-1, 1, (20, 6)).astype(np.float32)
    for k in range(20):
        ra, rb = env_a.step(acts[k]), env_b.step(acts[k])
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1] \
            and ra[2] == rb[2] and ra[3] == rb[3]
        if ra[2] or ra[3]:
            break
    obs_c, info_c = env_a.reset(seed=43)
    assert info_c["setup"] != info_a["setup"], "different seeds, same setup"
    print(f"    60 resets: weathers {sorted(weathers)}, "
          f"{len(compounds)} compounds, all knobs in range; "
          f"seed 42 reproduces bit-identically")


# ---------------------------------------------------------------------------
# T5 — every termination mode fires; truncation is not termination
# ---------------------------------------------------------------------------
def test_t5_termination_modes():
    # off_track: gentle constant arc off the main straight
    env = make_env()
    env.reset(seed=3, options=dict(BENIGN, v0=30.0))
    arc = np.array([-0.2, -1.0, 0.12, 0.0, -1.0, -1.0], dtype=np.float32)
    reason, terminal = None, 0.0
    for _ in range(600):
        obs, r, term, trunc, info = env.step(arc)
        if term:
            reason = info["termination"]
            terminal = info["reward_components"]["terminal"]
            break
    assert reason == "off_track", f"expected off_track, got {reason}"
    assert terminal == -env.config.reward.p_terminal, \
        f"off-track terminal penalty {terminal}"

    # spin: force a huge body sideslip and let the check catch it
    env.reset(seed=4, options=BENIGN)
    env.vehicle.state.vy = 35.0
    obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term and info["termination"] == "spin", \
        f"expected spin, got {info.get('termination')}"
    assert info["reward_components"]["terminal"] == -env.config.reward.p_terminal

    # fuel_out: start nearly dry and hold full throttle
    env.reset(seed=5, options=dict(BENIGN, fuel=0.02))
    flat = np.array([1.0, -1.0, 0.0, 0.4, 1.0, -1.0], dtype=np.float32)
    reason = None
    for _ in range(200):
        obs, r, term, trunc, info = env.step(flat)
        if term:
            reason = info["termination"]
            break
    assert reason == "fuel_out", f"expected fuel_out, got {reason}"

    # stall: brake to a halt and sit (config-gated addition beyond the spec)
    env.reset(seed=8, options=dict(BENIGN, v0=15.0))
    stop = np.array([-1.0, 1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
    reason = None
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(stop)
        if term:
            reason = info["termination"]
            break
    assert reason == "stall", f"expected stall, got {reason}"

    # max_steps: truncated, NOT terminated
    env50 = make_env(max_steps=50)
    env50.reset(seed=6, options=BENIGN)
    coast = np.array([-0.1, -1.0, 0.0, 0.2, -1.0, -1.0], dtype=np.float32)
    steps = 0
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, info = env50.step(coast)
        steps += 1
        assert steps <= 50, "episode ran past max_steps"
    assert trunc and not term and steps == 50
    assert "termination" not in info
    print("    off_track / spin / fuel_out / stall all fire with the -100 "
          "penalty where due; max_steps truncates cleanly")


# ---------------------------------------------------------------------------
# T6 — pursuit completes a lap and beats a random policy            [bullet 4]
# ---------------------------------------------------------------------------
def test_t6_pursuit_beats_random():
    ret, lap_t, v_avg, v_max = drive_lap(BENIGN)
    # verify the physics numbers, not just the reward (project convention)
    assert 60.0 < lap_t < 300.0, f"implausible lap time {lap_t:.1f} s"
    assert v_max <= PursuitDriver.V_CAP * 1.15, f"v_max {v_max:.0f} m/s ran away"
    assert v_avg > 25.0, f"average speed {v_avg:.0f} m/s implausibly low"

    best_random = -math.inf
    env = make_env()
    for ep in range(5):
        env.reset(seed=200 + ep, options=BENIGN)
        env.action_space.seed(300 + ep)
        total = 0.0
        for _ in range(env.config.max_steps):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        best_random = max(best_random, total)
    assert ret > best_random, \
        f"pursuit return {ret:.0f} did not beat best random {best_random:.0f}"
    print(f"    pursuit lap {lap_t:.1f} s, avg {v_avg * 3.6:.0f} km/h, "
          f"max {v_max * 3.6:.0f} km/h; return {ret:.0f} vs best random "
          f"{best_random:.0f}")


# ---------------------------------------------------------------------------
# T7 — conditions shape real lap times through the full stack
# ---------------------------------------------------------------------------
def test_t7_conditions_affect_laps():
    # controlled weather comparison: same compound, same track temp, same
    # driver — ONLY the weather changes. (Comparing different compounds
    # instead confounds weather with the temperature window: slicks exit
    # this track's 1.5 km straight stone cold — see the print below.)
    _, t_dry, _, _ = drive_lap(BENIGN)
    _, t_damp, _, _ = drive_lap(dict(BENIGN, weather="damp",
                                     rain_intensity=0.5))
    assert t_damp > t_dry * 1.05, \
        f"damp lap {t_damp:.1f} s not >=5% slower than dry {t_dry:.1f} s"

    _, t_light, _, _ = drive_lap(dict(BENIGN, fuel=20.0))
    _, t_heavy, _, _ = drive_lap(dict(BENIGN, fuel=105.0))
    assert t_heavy > t_light + 0.3, \
        f"105 kg fuel ({t_heavy:.2f} s) vs 20 kg ({t_light:.2f} s): no cost"

    # observed, print-only: rain tires in the wet vs softs in the dry. The
    # gap is small because the cautious driver never works the softs into
    # their 85-105 C window — the Layer 3 temperature model, end to end.
    _, t_soft, _, _ = drive_lap(dict(BENIGN, compound="soft"))
    _, t_wet, _, _ = drive_lap(dict(BENIGN, weather="wet", compound="wet",
                                    rain_intensity=0.7, track_temp=18.0))
    print(f"    laps (same mediums): dry {t_dry:.1f} s / damp {t_damp:.1f} s "
          f"(+{(t_damp / t_dry - 1) * 100:.0f}%); fuel 20 kg {t_light:.1f} s "
          f"/ 105 kg {t_heavy:.1f} s (+{t_heavy - t_light:.1f} s); "
          f"dry-softs {t_soft:.1f} s vs wet-wets {t_wet:.1f} s (cold slicks)")


# ---------------------------------------------------------------------------
# T8 — throughput: >= 2000 steps/s single-process on CPU            [bullet 5]
# ---------------------------------------------------------------------------
def test_t8_throughput():
    env = make_env()
    env.reset(seed=99)
    env.action_space.seed(99)
    n = 12_000
    resets = 0
    t0 = time.perf_counter()
    for i in range(n):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset(seed=i)
            resets += 1
    rate = n / (time.perf_counter() - t0)
    assert rate >= 2000.0, f"env too slow: {rate:,.0f} steps/s (floor 2000)"
    print(f"    {n:,} random steps incl. {resets} resets: "
          f"{rate:,.0f} steps/s (spec floor 2000)")


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_check_env,
    test_t2_random_steps,
    test_t3_reward_components,
    test_t4_domain_randomization_and_determinism,
    test_t5_termination_modes,
    test_t6_pursuit_beats_random,
    test_t7_conditions_affect_laps,
    test_t8_throughput,
]

if __name__ == "__main__":
    print("=" * 58)
    print("F1 GYM ENVIRONMENT — LAYER 4 VALIDATION")
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
