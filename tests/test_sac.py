"""Layer 5 acceptance tests — SAC driver agent (LAYER_SPECS.md § Layer 5).

Runs two ways:
  * `pytest tests/test_sac.py`   — standard test run
  * `python tests/test_sac.py`   — standalone PASS/FAIL report

Fully offline (synthetic track). T1–T6 use a deliberately tiny config so they
run in seconds; T7 is the spec's acceptance smoke train — 150k SAC steps,
roughly 40 min on 4 CPU cores, and the one test that must show the trained
policy measurably beating a random one on lap time and off-track count. Real
driving numbers are printed throughout (project rule: never reward curves
alone).
"""
import sys
import tempfile
import time
from pathlib import Path
from unittest import SkipTest  # honored by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np

from src.agents.sac_driver import (BENIGN_EVAL, SMOKE_EVAL, ActionRepeat,
                                   SACDriver, SACDriverConfig,
                                   benign_training_env_config)
from src.envs.f1_env import F1Env
from src.tracks.track import Track

TRACK = Track.from_synthetic()  # built once and shared; construction ~1 s

PURSUIT_REF_S = 140.7  # Layer 4's hand-coded pursuit lap on BENIGN (reference)

TERMINATIONS = {"off_track", "spin", "fuel_out", "stall", "lap", "max_steps"}


def tiny_config(**over) -> SACDriverConfig:
    """Small-but-real config so unit tests run in seconds."""
    base = dict(buffer_size=20_000, learning_starts=200, batch_size=64,
                net_arch=(64, 64), seed=7)
    base.update(over)
    return SACDriverConfig(**base)


def collect_probe_obs(n: int, seed: int) -> np.ndarray:
    """Deterministic set of raw observations from a seeded random rollout."""
    env = F1Env(track=TRACK, config=benign_training_env_config())
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    out = [obs]
    while len(out) < n:
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset(seed=seed + len(out))
        out.append(obs)
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# T1 — construction is config-driven: the SB3 model wears exactly the config
# ---------------------------------------------------------------------------
def test_t1_config_driven_construction():
    cfg = SACDriverConfig()  # the real training defaults
    d = SACDriver(cfg, track=TRACK)
    m = d.model
    assert m.learning_rate == cfg.learning_rate
    assert m.buffer_size == cfg.buffer_size
    assert m.batch_size == cfg.batch_size
    assert m.tau == cfg.tau
    assert m.gamma == cfg.gamma
    assert m.learning_starts == cfg.learning_starts
    assert m.seed == cfg.seed
    assert m.ent_coef == cfg.ent_coef

    # the configured net arch made it into the actual torch modules
    actor_widths = [layer.out_features for layer in m.policy.actor.latent_pi
                    if hasattr(layer, "out_features")]
    assert actor_widths == list(cfg.net_arch), actor_widths
    critic_widths = [layer.out_features for layer in m.policy.critic.qf0
                     if hasattr(layer, "out_features")]
    assert critic_widths[:-1] == list(cfg.net_arch), critic_widths

    # spaces match Layer 4's env; the action-repeat wrapper is in the stack
    assert m.observation_space.shape == (40,)
    assert m.action_space.shape == (6,)
    wrapped = d.vec_env.unwrapped.envs[0].env  # Monitor -> ActionRepeat
    assert isinstance(wrapped, ActionRepeat)
    assert wrapped.repeat == cfg.action_repeat
    d.close()
    print(f"    defaults: net {cfg.net_arch}, buffer {cfg.buffer_size:,}, "
          f"action repeat {cfg.action_repeat} (50 Hz physics -> "
          f"{50 / cfg.action_repeat:.1f} Hz control)")


# ---------------------------------------------------------------------------
# T2 — vectorized envs work (spec: "support a vectorized env for parallelism")
# ---------------------------------------------------------------------------
def test_t2_vectorized_env():
    from stable_baselines3.common.vec_env import VecNormalize

    d = SACDriver(tiny_config(n_envs=2, vec_env_cls="dummy",
                              learning_starts=100), track=TRACK)
    assert isinstance(d.vec_env, VecNormalize)
    obs = d.vec_env.reset()
    assert obs.shape == (2, 40)
    assert not np.allclose(obs[0], obs[1]), "vec envs did not desync"
    before = float(d.vec_env.obs_rms.count)
    d.train(600)
    assert float(d.vec_env.obs_rms.count) > before, "obs stats never updated"
    d.close()

    # subprocess path: construct, reset, step, close (workers build their own
    # synthetic track — nothing shared across the process boundary)
    d2 = SACDriver(tiny_config(n_envs=2, vec_env_cls="subproc"), track=None)
    obs = d2.vec_env.reset()
    assert obs.shape == (2, 40)
    obs, _, _, _ = d2.vec_env.step(np.zeros((2, 6), dtype=np.float32))
    assert obs.shape == (2, 40)
    d2.close()
    print("    2-env DummyVecEnv trains; 2-env SubprocVecEnv steps")


# ---------------------------------------------------------------------------
# T3 — a tiny training run is numerically stable (spec: "no divergence/NaN")
# ---------------------------------------------------------------------------
def test_t3_tiny_train_stability():
    d = SACDriver(tiny_config(seed=3), track=TRACK)
    d.train(1_500)  # VecCheckNan raises on any NaN/inf in the loop
    for name, p in d.model.policy.named_parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all(), \
            f"non-finite parameter after training: {name}"
    if getattr(d.model, "log_ent_coef", None) is not None:
        assert np.isfinite(float(d.model.log_ent_coef.detach().cpu()))
    assert d.model.replay_buffer.pos == 1_500  # one transition per policy step
    n_eps = len(d.model.ep_info_buffer)
    assert n_eps > 0, "no episode ever finished in 1500 policy steps"
    d.close()
    print(f"    1,500 policy steps ({n_eps} episodes buffered): "
          f"all parameters finite, NaN guard silent")


# ---------------------------------------------------------------------------
# T4 — checkpoints round-trip bit-identically and can resume      [spec bullet]
# ---------------------------------------------------------------------------
def test_t4_checkpoint_roundtrip():
    d = SACDriver(tiny_config(seed=11), track=TRACK)
    d.train(600)
    probe = collect_probe_obs(20, seed=5)
    acts = np.stack([d.predict(o) for o in probe])

    with tempfile.TemporaryDirectory() as tmp:
        d.save(tmp)
        d2 = SACDriver.load(tmp, track=TRACK)
        acts2 = np.stack([d2.predict(o) for o in probe])
        assert np.array_equal(acts, acts2), \
            "loaded policy's deterministic actions differ from the original"
        assert np.array_equal(d.vec_env.obs_rms.mean, d2.vec_env.obs_rms.mean)
        assert np.array_equal(d.vec_env.obs_rms.var, d2.vec_env.obs_rms.var)

        steps_before = d2.model.num_timesteps
        assert steps_before == 600  # counter restored from the checkpoint
        d2.train(200)               # resumes without error
        assert d2.model.num_timesteps == steps_before + 200
        d2.close()
    d.close()
    print("    save -> load: 20 probe actions bit-identical, "
          "VecNormalize stats intact, training resumed 600 -> 800 steps")


# ---------------------------------------------------------------------------
# T5 — fixed seed => identical training result      [spec: "reproducible"]
# ---------------------------------------------------------------------------
def test_t5_seed_reproducibility():
    probe = collect_probe_obs(10, seed=21)

    def train_one(seed: int) -> np.ndarray:
        d = SACDriver(tiny_config(seed=seed), track=TRACK)
        d.train(800)
        acts = np.stack([d.predict(o) for o in probe])
        d.close()
        return acts

    a, b, c = train_one(7), train_one(7), train_one(8)
    assert np.array_equal(a, b), "same seed produced different policies"
    assert not np.array_equal(a, c), "different seeds produced identical policies"
    print("    two seed-7 trainings bit-identical; seed-8 differs")


# ---------------------------------------------------------------------------
# T6 — evaluate() reports real driving numbers        [spec: evaluate() bullet]
# ---------------------------------------------------------------------------
def test_t6_evaluate_harness():
    d = SACDriver(tiny_config(seed=13), track=TRACK)  # untrained on purpose

    rep = d.evaluate(n_episodes=2, seed=50)
    assert rep.policy == "model" and len(rep.episodes) == 2
    for e in rep.episodes:
        assert e.termination in TERMINATIONS, e.termination
        assert np.isfinite(e.progress_m) and np.isfinite(e.ret)
        assert e.avg_speed_kmh >= 0.0 and e.max_speed_kmh >= e.avg_speed_kmh
        if e.lap_time is None:
            assert e.time_to_lap_capped == 300.0  # the 15k-step episode cap
        else:
            assert e.time_to_lap_capped == e.lap_time
    assert rep.off_track_count + rep.spin_count <= len(rep.episodes)

    rnd = d.evaluate(n_episodes=2, seed=50, policy="random")
    assert rnd.policy == "random" and len(rnd.episodes) == 2

    assert d.evaluate(n_episodes=2, seed=50) == rep, \
        "evaluation protocol is not deterministic"

    spread = d.evaluate(n_episodes=3, seed=50, policy="random",
                        spread_starts=True)
    starts = [e.start_s for e in spread.episodes]
    assert starts == sorted(starts) and starts[0] == 0.0 and starts[-1] > 0.0
    d.close()
    print(f"    untrained model: {[e.termination for e in rep.episodes]}, "
          f"random: {[e.termination for e in rnd.episodes]}; "
          f"reports deterministic, spread starts at {[f'{s:.0f}' for s in starts]} m")


# ---------------------------------------------------------------------------
# T7 — THE SMOKE TRAIN: ~50k steps must measurably beat a random policy on
# lap time and off-track count, with stable training     [spec acceptance]
# ---------------------------------------------------------------------------
def test_t7_smoke_train_beats_random():
    # SAC policy steps (x4 action repeat = 600k env steps). The spec's
    # "e.g. 50k steps" proved ~3x short of a first lap on this track (progress
    # 649 m mean at 50k, healthy learning curve); the assertions are unchanged.
    steps = 150_000
    print(f"    training {steps:,} SAC steps on the benign preset "
          f"(~35-40 min on CPU) ...", flush=True)
    t0 = time.perf_counter()
    d = SACDriver(SACDriverConfig(), track=TRACK)  # the real defaults
    d.train(steps, progress_every=5_000)
    train_min = (time.perf_counter() - t0) / 60.0

    # 5 distinct rolling starts spread around the lap, identical for both
    # policies (SMOKE_EVAL: v0=30 m/s — survivable rolling speed anywhere)
    eval_opts = SMOKE_EVAL
    trained = d.evaluate(n_episodes=5, seed=123, spread_starts=True,
                         options=eval_opts)
    rand = d.evaluate(n_episodes=5, seed=123, spread_starts=True,
                      options=eval_opts, policy="random")

    print(f"    trained in {train_min:.1f} min — deterministic policy:")
    print(trained.table())
    print("    random baseline (same starts, same control rate):")
    print(rand.table())
    canon = d.evaluate(n_episodes=1, seed=123)  # BENIGN_EVAL: s0=0, v0=40
    lap = canon.episodes[0].lap_time
    print(f"    canonical lap from s0=0 (pursuit reference {PURSUIT_REF_S} s): "
          + (f"{lap:.1f} s" if lap is not None
             else f"DNF ({canon.episodes[0].termination} at "
                  f"{canon.episodes[0].progress_m:.0f} m)"))

    # (c) stability: NaN guard stayed silent all run; parameters finite
    for name, p in d.model.policy.named_parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all(), \
            f"non-finite parameter after smoke train: {name}"

    # (a) "measurably reduces lap time": mean time-to-lap with DNF = 300 s cap.
    # Random never laps (flat 300), so the trained policy must actually lap.
    assert trained.mean_time_to_lap_capped < rand.mean_time_to_lap_capped, (
        f"trained time-to-lap {trained.mean_time_to_lap_capped:.1f} s not "
        f"below random's {rand.mean_time_to_lap_capped:.1f} s")

    # (b) "measurably reduces off-track count"
    assert trained.off_track_count < rand.off_track_count, (
        f"trained off-track {trained.off_track_count} not below "
        f"random's {rand.off_track_count}")
    d.close()


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_config_driven_construction,
    test_t2_vectorized_env,
    test_t3_tiny_train_stability,
    test_t4_checkpoint_roundtrip,
    test_t5_seed_reproducibility,
    test_t6_evaluate_harness,
    test_t7_smoke_train_beats_random,
]

if __name__ == "__main__":
    print("=" * 58)
    print("SAC DRIVER AGENT — LAYER 5 VALIDATION")
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
