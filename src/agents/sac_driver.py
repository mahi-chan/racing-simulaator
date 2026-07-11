"""
SAC driver agent — Layer 5 of the autonomous racing stack.

Wraps Stable-Baselines3 SAC around the Layer 4 `F1Env` (LAYER_SPECS.md §
Layer 5): observation normalization, vectorized envs, config-driven
hyperparameters with a fixed seed, checkpoint save/load, and an `evaluate()`
that reports real driving numbers — lap time, average speed, off-track count —
never reward curves alone.

Design notes:
  * `ActionRepeat` holds each policy action for `action_repeat` env steps.
    The physics stays at the validated 50 Hz (`dt=0.02`, 2 substeps); the agent
    decides at 12.5 Hz by default — the band racing RL actually uses (GT Sophy
    runs 10 Hz). Rewards AND the named `reward_components` are summed across
    the held steps, so Layer 4's "components sum to the reward" invariant
    survives the wrapper; `lap_time` stays physics time. `action_repeat=1`
    disables it.
  * Layer 5 trains on ONE benign condition — dry, mediums, fixed mid setup —
    with only the start pose randomized so the replay buffer covers the whole
    track (`benign_training_env_config()`). Weather/setup domain randomization
    during training is Layer 6's job per LAYER_SPECS; the preset is built
    purely from Layer 4's config surface (degenerate ranges), no env changes.
  * Observation normalization = `VecNormalize` running z-score (observations
    only — rewards stay interpretable) on top of the env's static O(1)
    scaling. The running stats are saved with every checkpoint and are used
    frozen at evaluation time.
  * Two departures from vanilla SAC defaults, both forced by evidence from
    50k/150k-step smoke runs on this env: a best-policy keeper — auto-tuned
    entropy settles at alpha ~ 0.13 here and the mandated action noise
    degrades the LIVE policy after it peaks, so train() returns the best
    deterministic evaluator seen, not the last policy (a fixed low alpha was
    tried instead and starved exploration) — and spawn-at-speed training
    starts (`benign_training_dr`).
  * Layer 6 builds on this file without changing its behavior: save/load can
    persist the replay buffer (`include_buffer=True`) and
    `swap_env_config()` swaps the training env mid-run for curriculum stage
    transitions, carrying the VecNormalize statistics across the swap.
  * Every knob lives in `SACDriverConfig`; nothing is hard-coded. Values are
    standard SAC starting points, tuned only as far as the Layer 5 smoke-train
    acceptance demanded.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecCheckNan, VecNormalize)

from src.envs.f1_env import (DomainRandomizationConfig, EnvConfig, F1Env,
                             RewardConfig)
from src.tracks.track import Track

# Fixed benign episode for evaluation — identical to tests/test_env.py's
# BENIGN, so lap times stay comparable across layers (the Layer 4 hand-coded
# pursuit controller laps this in 140.7 s).
BENIGN_EVAL = dict(weather="dry", compound="medium", rain_intensity=0.0,
                   track_temp=30.0, fuel=30.0, aero_level=0.5, brake_bias=0.58,
                   final_drive=3.0, s0=0.0, v0=40.0, lateral=0.0,
                   heading_error=0.0)

# Rolling-start protocol for smoke evaluation and in-training model selection:
# the same benign stint, but v0=30 m/s so a start dropped anywhere on the lap
# is survivable in principle. Used with `spread_starts=True` (episode i of n
# starts at s = i/n of the lap — distinct trials for a deterministic policy).
SMOKE_EVAL = dict(BENIGN_EVAL, v0=30.0)


def benign_training_dr() -> DomainRandomizationConfig:
    """Layer 5 training preset: one benign condition, randomized start pose.

    Degenerate ranges pin weather/compound/fuel/setup to BENIGN_EVAL's values;
    the only randomness is WHERE the episode starts (anywhere on track, 20-45
    m/s, small lateral/heading noise) so experience covers every corner.
    """
    return DomainRandomizationConfig(
        weather_probs={"dry": 1.0},
        rain_intensity_range={"dry": (0.0, 0.0)},
        track_temp_range={"dry": (30.0, 30.0)},
        compound_probs={"dry": (("medium", 1.0),)},
        fuel_range=(30.0, 30.0),
        aero_level_range=(0.5, 0.5),
        brake_bias_range=(0.58, 0.58),
        final_drive_range=(3.0, 3.0),
        # Spawn-at-speed up to 65 m/s (234 km/h): high-speed corner approaches
        # are exactly the states a from-slow policy explores last (and died in
        # every probe run); seeding them directly teaches brake-or-crash from
        # step one. Fast spawns inside tight corners are unsavable — that is
        # acceptable replay noise, priced in.
        start_speed_range=(20.0, 65.0),
        start_lateral_range=(-1.0, 1.0),
        start_heading_error_range=(-0.03, 0.03),
        randomize_start_s=True,
    )


def benign_training_env_config() -> EnvConfig:
    return EnvConfig(dr=benign_training_dr())


class ActionRepeat(gym.Wrapper):
    """Hold each policy action for `repeat` env steps (control-rate shaping).

    Sums the reward and the named `reward_components` over the held steps and
    stops early on termination/truncation. Spaces are unchanged; `lap_time`
    (physics seconds) passes through untouched.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.repeat = int(repeat)

    def step(self, action):
        total = 0.0
        comps: dict | None = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total += float(r)
            rc = info["reward_components"]
            comps = dict(rc) if comps is None else {k: comps[k] + rc[k]
                                                    for k in comps}
            if terminated or truncated:
                break
        info = dict(info)
        info["reward_components"] = comps
        return obs, total, terminated, truncated, info


# ----------------------------------------------------------------------------
# Config — every Layer 5 knob in one labeled place (spec: none hard-coded)
# ----------------------------------------------------------------------------
@dataclass
class SACDriverConfig:
    """SAC hyperparameters + agent-side env interface knobs."""

    # --- SAC core (names match the SB3 arguments) ---
    learning_rate: float = 3e-4
    buffer_size: int = 300_000     # policy-step transitions (~105 MB at 40-dim obs)
    learning_starts: int = 10_000  # random-action warmup before updates
    batch_size: int = 256
    tau: float = 0.005             # target-network soft update
    gamma: float = 0.995           # 16 s horizon at 12.5 Hz control
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"  # SAC temperature auto-tuning. It settles
    #   at alpha ~ 0.13 here and the mandated noise degrades the LIVE policy
    #   after it peaks — but a fixed low alpha (0.02) was tried and starved
    #   exploration (slower learning, never found corner-1 braking). Keep the
    #   exploration; the best-policy keeper preserves the deterministic peak.
    net_arch: tuple = (256, 256)   # actor & critic MLP widths

    # --- control & env interface ---
    action_repeat: int = 4        # env steps per policy action (1 disables)
    n_envs: int = 1
    vec_env_cls: str = "auto"     # "auto" (subproc when n_envs>1) | "dummy" | "subproc"
    normalize_obs: bool = True    # VecNormalize running z-score (stats checkpointed)
    normalize_reward: bool = False  # keep logged returns interpretable
    vecnorm_clip_obs: float = 10.0
    check_nan: bool = True        # VecCheckNan raises on any NaN/inf in the loop

    # --- model selection (best-policy keeper) ---
    best_eval_every: int | None = 10_000  # deterministic SMOKE_EVAL cadence
    best_eval_episodes: int = 5           # (policy steps); None = keep last

    # --- reproducibility / hardware ---
    seed: int = 42
    device: str = "auto"


# ----------------------------------------------------------------------------
# Evaluation report — real driving numbers, printable
# ----------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    start_s: float                # arc-length of the episode's start (m)
    lap_time: float | None        # physics seconds; None = did not finish
    time_to_lap_capped: float     # lap_time, or the episode time cap on a DNF
    progress_m: float             # forward progress along s
    avg_speed_kmh: float
    max_speed_kmh: float
    ret: float                    # episode return (sum of env rewards)
    termination: str              # off_track/spin/fuel_out/stall/lap/max_steps


@dataclass
class EvalReport:
    policy: str                   # "model" | "random"
    episodes: list
    laps_completed: int
    mean_lap_time: float | None   # over completed laps; None if no lap
    mean_time_to_lap_capped: float  # DNF counts as the episode cap
    off_track_count: int
    spin_count: int
    mean_progress_m: float
    mean_return: float

    def table(self) -> str:
        rows = [f"    {'ep':>2}  {'start':>7}  {'end':<9}  {'lap time':>9}  "
                f"{'progress':>8}  {'avg':>5}  {'max':>5}  {'return':>9}"]
        for i, e in enumerate(self.episodes, 1):
            lap = f"{e.lap_time:7.1f} s" if e.lap_time is not None else "    DNF  "
            rows.append(f"    {i:>2}  {e.start_s:5.0f} m  {e.termination:<9}  "
                        f"{lap:>9}  {e.progress_m:6.0f} m  {e.avg_speed_kmh:3.0f}"
                        f"    {e.max_speed_kmh:3.0f}    {e.ret:9.1f}")
        mean_lap = (f"{self.mean_lap_time:.1f} s" if self.mean_lap_time is not None
                    else "-")
        rows.append(f"    laps {self.laps_completed}/{len(self.episodes)}"
                    f" | mean lap {mean_lap}"
                    f" | time-to-lap (DNF=cap) {self.mean_time_to_lap_capped:.1f} s"
                    f" | off-track {self.off_track_count}"
                    f" | spin {self.spin_count}"
                    f" | mean return {self.mean_return:.1f}")
        return "\n".join(rows)


class _BestPolicyKeeper(BaseCallback):
    """Evaluate deterministically every `every` steps; keep the best policy.

    Model selection: training returns the best policy seen, not the last one
    — SAC's exploration pressure can degrade the live policy after it peaks
    (observed here even with a fixed temperature the risk is cheap to remove).
    Rank: laps completed, then capped time-to-lap, then progress. The
    VecNormalize obs stats are snapshotted with the policy — a policy is only
    reproducible together with the normalization it was evaluated under.
    """

    def __init__(self, driver: "SACDriver", every: int, n_episodes: int):
        super().__init__()
        self.driver = driver
        self.every = max(int(every), 1)
        self.n_episodes = int(n_episodes)
        self._next = self.every
        self.best_key: tuple | None = None
        self.best_step: int | None = None
        self._best_policy: dict | None = None
        self._best_rms = None

    @staticmethod
    def key(rep: EvalReport) -> tuple:
        return (rep.laps_completed, -rep.mean_time_to_lap_capped,
                rep.mean_progress_m)

    def snapshot_if_better(self) -> None:
        rep = self.driver.evaluate(n_episodes=self.n_episodes,
                                   options=SMOKE_EVAL, seed=1234,
                                   spread_starts=True)
        k = self.key(rep)
        if self.best_key is None or k > self.best_key:
            self.best_key = k
            self.best_step = self.num_timesteps
            state = self.model.policy.state_dict()
            self._best_policy = {n: t.detach().clone()
                                 for n, t in state.items()}
            venv = self.driver.vec_env
            self._best_rms = (copy.deepcopy(venv.obs_rms)
                              if isinstance(venv, VecNormalize) else None)
            print(f"    [best] {self.num_timesteps:>8,d} steps: "
                  f"laps {rep.laps_completed}/{self.n_episodes}, "
                  f"time-to-lap {rep.mean_time_to_lap_capped:.1f} s, "
                  f"progress {rep.mean_progress_m:.0f} m", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.snapshot_if_better()
        return True

    def restore_best(self) -> None:
        """After training: final policy gets a last look, then best wins."""
        if self._best_policy is None:
            return
        self.snapshot_if_better()
        self.model.policy.load_state_dict(self._best_policy)
        if self._best_rms is not None and isinstance(self.driver.vec_env,
                                                     VecNormalize):
            self.driver.vec_env.obs_rms = self._best_rms
        print(f"    [best] kept the policy from step {self.best_step:,d}",
              flush=True)


class _Progress(BaseCallback):
    """One-line training heartbeat every `every` policy steps."""

    def __init__(self, every: int):
        super().__init__()
        self.every = max(int(every), 1)
        self._next = self.every

    def _on_training_start(self) -> None:
        # anchor to the current counter so resumed/chunked training doesn't
        # fire a print per step while catching up
        self._next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            buf = list(self.model.ep_info_buffer or [])
            if buf:
                rew = float(np.mean([e["r"] for e in buf]))
                length = float(np.mean([e["l"] for e in buf]))
                extra = f"ep_rew_mean {rew:9.1f}  ep_len_mean {length:6.0f}"
            else:
                extra = "no episodes finished yet"
            print(f"    [train] {self.num_timesteps:>8,d} steps  {extra}",
                  flush=True)
        return True


# ----------------------------------------------------------------------------
# config <-> json (checkpoint sidecar; JSON lists rebuild as tuples where the
# dataclasses declare tuples — the env consumes ranges positionally either way)
# ----------------------------------------------------------------------------
def _sac_config_from_dict(d: dict) -> SACDriverConfig:
    d = dict(d)
    if isinstance(d.get("net_arch"), list):
        d["net_arch"] = tuple(d["net_arch"])
    return SACDriverConfig(**d)


def _env_config_from_dict(blob: dict) -> EnvConfig:
    b = dict(blob)
    reward = RewardConfig(**b.pop("reward"))
    dr_raw = dict(b.pop("dr"))
    dr_raw["compound_probs"] = {
        w: tuple((str(c), float(p)) for c, p in table)
        for w, table in dr_raw["compound_probs"].items()}
    for key, v in dr_raw.items():
        if isinstance(v, list):
            dr_raw[key] = tuple(v)
        elif isinstance(v, dict) and key != "compound_probs":
            dr_raw[key] = {k: tuple(vv) if isinstance(vv, list) else vv
                           for k, vv in v.items()}
    dr = DomainRandomizationConfig(**dr_raw)
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in b.items()}
    return EnvConfig(reward=reward, dr=dr, **kw)


def _make_env_fn(env_config: EnvConfig, repeat: int, track: Track | None,
                 seed: int, rank: int):
    """Env factory for vec envs. With track=None each worker builds its own
    synthetic track (needed for subprocess workers)."""

    def _init():
        t = track if track is not None else Track.from_synthetic()
        env: gym.Env = F1Env(track=t, config=env_config)
        if repeat > 1:
            env = ActionRepeat(env, repeat)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ----------------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------------
class SACDriver:
    """SB3 SAC bound to `F1Env`: train / save / load / evaluate / predict."""

    def __init__(self, config: SACDriverConfig | None = None,
                 env_config: EnvConfig | None = None,
                 track: Track | None = None):
        self.config = config or SACDriverConfig()
        self.env_config = env_config or benign_training_env_config()
        self._track = track
        cfg = self.config

        vec = self._build_vec_env()
        self.vec_env = vec

        self.model = SAC(
            "MlpPolicy", vec,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            ent_coef=cfg.ent_coef,
            policy_kwargs={"net_arch": list(cfg.net_arch)},
            seed=cfg.seed,
            device=cfg.device,
            verbose=0,
        )

    def _build_vec_env(self):
        """Vec stack from the current `env_config`:
        (Subproc|Dummy)VecEnv -> VecCheckNan -> VecNormalize (fresh stats)."""
        cfg = self.config
        fns = [_make_env_fn(self.env_config, cfg.action_repeat, self._track,
                            cfg.seed, i) for i in range(cfg.n_envs)]
        use_subproc = (cfg.vec_env_cls == "subproc"
                       or (cfg.vec_env_cls == "auto" and cfg.n_envs > 1))
        vec = SubprocVecEnv(fns) if (use_subproc and cfg.n_envs > 1) \
            else DummyVecEnv(fns)
        if cfg.check_nan:
            vec = VecCheckNan(vec, raise_exception=True)
        if cfg.normalize_obs or cfg.normalize_reward:
            vec = VecNormalize(vec, training=True, norm_obs=cfg.normalize_obs,
                               norm_reward=cfg.normalize_reward,
                               clip_obs=cfg.vecnorm_clip_obs, gamma=cfg.gamma)
        return vec

    def swap_env_config(self, env_config: EnvConfig) -> None:
        """Swap the training environment mid-run (curriculum stage change).

        Rebuilds the vec stack from `env_config` and transplants the
        VecNormalize running statistics so observation scaling stays
        continuous across the swap; SB3 resets the new env on the next
        `learn()` call. The replay buffer is intentionally kept — transitions
        from the previous stage age out of the buffer window naturally.
        """
        old_rms = (self.vec_env.obs_rms
                   if isinstance(self.vec_env, VecNormalize) else None)
        self.vec_env.close()
        self.env_config = env_config
        vec = self._build_vec_env()
        if old_rms is not None and isinstance(vec, VecNormalize):
            vec.obs_rms = old_rms
        self.vec_env = vec
        self.model.set_env(vec)

    # -- training --------------------------------------------------------------
    def train(self, total_timesteps: int, log_dir: str | Path | None = None,
              checkpoint_every: int | None = None,
              progress_every: int | None = None) -> "SACDriver":
        """Run SAC for `total_timesteps` policy steps (resumes counters)."""
        callbacks = []
        if progress_every:
            callbacks.append(_Progress(progress_every))
        keeper = None
        if self.config.best_eval_every:
            keeper = _BestPolicyKeeper(self, self.config.best_eval_every,
                                       self.config.best_eval_episodes)
            callbacks.append(keeper)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.model.set_logger(configure_logger(str(log_dir), ["csv"]))
            if checkpoint_every:
                callbacks.append(CheckpointCallback(
                    save_freq=max(int(checkpoint_every)
                                  // max(self.config.n_envs, 1), 1),
                    save_path=str(log_dir), name_prefix="sac_ckpt"))
        self.model.learn(total_timesteps=int(total_timesteps),
                         callback=callbacks or None,
                         reset_num_timesteps=False)
        if keeper is not None:
            keeper.restore_best()
        return self

    # -- checkpointing -----------------------------------------------------------
    def save(self, path: str | Path, include_buffer: bool = False) -> Path:
        """Write model.zip + vecnormalize.pkl + config.json into `path`.

        `include_buffer` also writes replay_buffer.pkl (~110 MB at the
        default 300k capacity) so a resumed run continues from a warm buffer
        instead of refilling it — Layer 6's checkpoint/resume contract.
        """
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(str(d / "model.zip"))
        if include_buffer:
            self.model.save_replay_buffer(str(d / "replay_buffer.pkl"))
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.save(str(d / "vecnormalize.pkl"))
        blob = {"sac": dataclasses.asdict(self.config),
                "env": dataclasses.asdict(self.env_config)}
        (d / "config.json").write_text(json.dumps(blob, indent=2))
        return d

    @classmethod
    def load(cls, path: str | Path, track: Track | None = None) -> "SACDriver":
        """Rebuild a driver from `save()` output; training can continue.

        If `save(include_buffer=True)` wrote replay_buffer.pkl, it is
        restored too, so resumed training continues from a warm buffer;
        otherwise the buffer refills from scratch.
        """
        d = Path(path)
        blob = json.loads((d / "config.json").read_text())
        config = _sac_config_from_dict(blob["sac"])
        env_config = _env_config_from_dict(blob["env"])
        driver = cls(config=config, env_config=env_config, track=track)
        if isinstance(driver.vec_env, VecNormalize):
            inner = driver.vec_env.venv
            driver.vec_env = VecNormalize.load(str(d / "vecnormalize.pkl"),
                                               inner)
            driver.vec_env.training = True
        driver.model = SAC.load(str(d / "model.zip"), env=driver.vec_env,
                                device=config.device)
        buf = d / "replay_buffer.pkl"
        if buf.exists():
            driver.model.load_replay_buffer(str(buf))
        return driver

    # -- inference / evaluation --------------------------------------------------
    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.vec_env, VecNormalize):
            return self.vec_env.normalize_obs(obs)  # frozen stats, no update
        return obs

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(
            self._norm_obs(np.asarray(obs, dtype=np.float32)),
            deterministic=deterministic)
        return action

    def evaluate(self, n_episodes: int = 5, options: dict | None = None,
                 seed: int = 123, deterministic: bool = True,
                 policy: str = "model",
                 spread_starts: bool = False,
                 env_config: EnvConfig | None = None) -> EvalReport:
        """Run pinned evaluation episodes and report real driving numbers.

        `options` pins the env's reset (default BENIGN_EVAL). With
        `spread_starts` episode i starts at s = i/n of the lap instead —
        n distinct trials for a deterministic policy on a deterministic env
        (the same trajectory n times measures nothing). A DNF's
        `time_to_lap_capped` is the episode time cap, which makes "reduces
        lap time vs a random policy" well-defined when random never laps.
        `env_config` overrides the env the episodes run in (default: the
        training env config) — needed to pin a weather the training DR
        tables don't cover, since Layer 4's reset draws its defaults from
        those tables even when `options` pin every value.
        """
        if policy not in ("model", "random"):
            raise ValueError(f"policy must be 'model' or 'random': {policy!r}")
        base_options = BENIGN_EVAL if options is None else options
        track = self._track if self._track is not None else Track.from_synthetic()
        base = F1Env(track=track,
                     config=self.env_config if env_config is None
                     else env_config)
        env: gym.Env = (ActionRepeat(base, self.config.action_repeat)
                        if self.config.action_repeat > 1 else base)
        cap = base.config.max_steps * base.config.dt  # episode time limit (s)

        episodes = []
        for i in range(n_episodes):
            opts = dict(base_options)
            if spread_starts:
                opts["s0"] = i * track.length / n_episodes
            if policy == "random":
                env.action_space.seed(seed * 1000 + i)
            obs, info = env.reset(seed=seed + i, options=opts)
            start_s = float(info["s"])
            ret, speeds, lap, term = 0.0, [], None, None
            done = False
            while not done:
                action = (env.action_space.sample() if policy == "random"
                          else self.predict(obs, deterministic=deterministic))
                obs, r, terminated, truncated, info = env.step(action)
                ret += float(r)
                speeds.append(float(info["speed"]))
                if terminated or truncated:
                    done = True
                    lap = info.get("lap_time")
                    term = info.get("termination",
                                    "lap" if lap is not None else "max_steps")
            episodes.append(EpisodeStats(
                start_s=start_s,
                lap_time=None if lap is None else float(lap),
                time_to_lap_capped=float(cap if lap is None else lap),
                progress_m=float(info["lap_fraction"]) * track.length,
                avg_speed_kmh=float(np.mean(speeds)) * 3.6,
                max_speed_kmh=float(np.max(speeds)) * 3.6,
                ret=ret,
                termination=term,
            ))
        env.close()

        laps = [e.lap_time for e in episodes if e.lap_time is not None]
        return EvalReport(
            policy=policy,
            episodes=episodes,
            laps_completed=len(laps),
            mean_lap_time=float(np.mean(laps)) if laps else None,
            mean_time_to_lap_capped=float(np.mean(
                [e.time_to_lap_capped for e in episodes])),
            off_track_count=sum(e.termination == "off_track" for e in episodes),
            spin_count=sum(e.termination == "spin" for e in episodes),
            mean_progress_m=float(np.mean([e.progress_m for e in episodes])),
            mean_return=float(np.mean([e.ret for e in episodes])),
        )

    def close(self):
        self.vec_env.close()
