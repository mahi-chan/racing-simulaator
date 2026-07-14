"""
Training loop, curriculum, domain randomization — Layer 6 of the racing stack.

Produces the GENERALIST driver (LAYER_SPECS.md § Layer 6): one SAC policy that
drives many setups and weather conditions competently. Everything here is
built from validated lower-layer surfaces — `EnvConfig` /
`DomainRandomizationConfig` (Layer 4) supply every curriculum knob (including
`off_track_margin`, the "wide track" scaffold), and `SACDriver` (Layer 5)
supplies training, checkpointing and pinned-condition evaluation. No physics,
reward, or env-logic changes.

Design notes:
  * Curriculum = a tuple of `StageConfig`s. The v3 (attempt-3) sequence:
    A benign laps -> B1 narrow setup + dry-compound rotation -> B2 full
    setup -> C full Layer 4 randomization. The spec's "wide/grippy/
    single-corner" example maps to: grippy = dry mediums at 30 C (the
    max-grip realistic condition in this sim, Layer 5 evidence);
    single-corner = spawn-at-speed anywhere on track, training corners
    piecewise. The literal "wide track" scaffold (off_track_margin 3.0) was
    tried in attempt 1 and measured to neither corrupt nor accelerate
    learning — dropped since v2.
  * v3 exists because v2's one-shot jump from pinned-benign to the full
    setup hypercube never generalized (runs/l6_generalist_v2 +
    reports/layer6_report.md, on record): its best policy lapped benign at
    133.9 s but finished 0/40 under full randomization, tolerated only
    ~+/-15 kg fuel / +/-0.1 aero, died within ~30 m of ANY compound
    one-hot flip (soft and hard alike, dry), and stalled from spawns below
    its 20 m/s training floor. Meanwhile stage B's LIVE policy lost even
    the benign lap — 25-60 m/s spawns on random skinny-wing setups are
    frequently unsavable-by-any-action, and those doomed transitions
    poisoned training. Hence the four v3 changes: (1) setup ranges widen
    progressively instead of jumping, (2) spawns are 15-40 m/s in learning
    stages — the low end teaches slow-speed driving, the capped top end
    removes most unsavable spawns — widening to 15-60 in the terminal
    stage to cover the acceptance distribution's 25-60; (3) dry compounds
    rotate from B1 so the categorical obs dims are covered early; (4) the
    terminal stage uses Layer 4's DR tables verbatim, mismatched compounds
    included (~13% of the acceptance distribution — evaluating what was
    never trained proved fatal in v2).
  * Stages advance when their GATE passes (checked every `eval_every` steps);
    a `max_steps` failsafe advances anyway and records `gate_met=False` so an
    unattainable provisional gate can never stall a long unattended run. The
    hard acceptance lives in tests/test_train.py, not in the gates.
  * Model selection: per-stage best-policy keeping on the stage's own
    yardstick (benign spread starts for the benign stage, validation
    conditions once setups randomize), ranked by DISTINCT CONDITIONS LAPPED
    first, then total laps, capped time-to-lap, progress. v2 ranked total
    laps first and a benign specialist's 2 laps could never be displaced by
    a broader-but-slower policy — nothing ever beat stage C's entry eval in
    500k steps. Each stage hands its best self to the next; the final
    artifact is the best terminal-stage policy. Snapshots pair the policy
    with its VecNormalize statistics (Layer 5 discipline) and are mirrored
    to disk so a resumed run cannot lose a pre-crash best.
  * Evaluation hygiene: VALIDATION_PANEL steers gates + model selection;
    HELD_OUT_PANEL is never consulted during training — it exists only for
    the final generalist-vs-specialist comparison and the acceptance test.
  * Checkpoint/resume (Colab ~12 h windows): every chunk writes a rotating
    checkpoint (model + replay buffer + VecNormalize + curriculum state,
    previous checkpoint kept until the new one lands). Resume restores exact
    counters, stage, buffer and normalization; the contract is functional
    continuation, NOT bitwise-identical trajectories (SB3 does not persist
    RNG streams).
  * The generalist-vs-specialist "documented %" deliberately has no pass bar
    — the spec asks for the gap to be measured and documented. Specialists
    are single-stage curriculum runs pinned to one held-out condition.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import VecNormalize

from src.agents.sac_driver import (BENIGN_EVAL, SMOKE_EVAL, EvalReport,
                                   SACDriver, SACDriverConfig,
                                   benign_training_dr, build_episode_env)
from src.envs.f1_env import DomainRandomizationConfig, EnvConfig, F1Env
from src.tracks.track import Track


# ----------------------------------------------------------------------------
# Evaluation conditions and panels
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalCondition:
    """A named, fully pinned F1Env reset — one point in condition space."""

    name: str
    options: dict


def _cond(name: str, **overrides) -> EvalCondition:
    """Pinned condition on the SMOKE_EVAL protocol (v0=30 rolling start)."""
    return EvalCondition(name, dict(SMOKE_EVAL, **overrides))


# Gates + model selection look here. Never used for the final comparison.
# v3 panel: V4/V5 span the dry-compound axis (v2 evidence: the kept policy
# died within ~30 m of any compound one-hot flip because compound variety
# was neither trained nor selected for — the T7 distribution draws each dry
# slick 1/3 of the time, so compound competence must be able to win model
# selection). Their track temps sit toward each compound's window since the
# cold-slick temperature model is a known Layer 7 calibration target.
# VALIDATION_PANEL[:5] is the dry sub-panel (stage B1/B2 yardstick).
VALIDATION_PANEL: tuple[EvalCondition, ...] = (
    _cond("V1_benign"),
    _cond("V2_light_lowwing", fuel=25.0, aero_level=0.2),
    _cond("V3_heavy_highwing", fuel=95.0, aero_level=0.8),
    _cond("V4_dry_softs", compound="soft", track_temp=35.0),
    _cond("V5_dry_hards", compound="hard", fuel=70.0, track_temp=40.0),
    _cond("V6_damp_inters", weather="damp", compound="intermediate",
          rain_intensity=0.25, track_temp=22.0),
    _cond("V7_wet_wets", weather="wet", compound="wet", rain_intensity=0.7,
          track_temp=18.0, aero_level=0.8),
)

# Held out from ALL training-time decisions; consumed only by the
# generalist-vs-specialist comparison and the Layer 6 acceptance test.
# In-distribution but never trained or selected on.
HELD_OUT_PANEL: tuple[EvalCondition, ...] = (
    _cond("H1_dry_mid", fuel=60.0, aero_level=0.35, brake_bias=0.60,
          final_drive=2.90),
    _cond("H2_dry_heavy_maxwing", fuel=85.0, aero_level=0.9,
          brake_bias=0.55),
    _cond("H3_damp_inters_light", weather="damp", compound="intermediate",
          rain_intensity=0.35, track_temp=20.0, fuel=35.0, aero_level=0.6),
    _cond("H4_wet_wets_midfuel", weather="wet", compound="wet",
          rain_intensity=0.5, track_temp=16.0, fuel=40.0, aero_level=0.65),
)


def pinned_dr(condition: EvalCondition) -> DomainRandomizationConfig:
    """Degenerate DR that trains on exactly one condition (specialists).

    Same recipe as Layer 5's `benign_training_dr` (which equals
    `pinned_dr(V1_benign)` by construction): everything pinned to the
    condition, only the start pose randomized — spawn-at-speed anywhere on
    track so replay covers every corner.
    """
    o = condition.options
    w = str(o.get("weather", "dry"))

    def pin(key, default):
        v = float(o.get(key, default))
        return (v, v)

    return DomainRandomizationConfig(
        weather_probs={w: 1.0},
        rain_intensity_range={w: pin("rain_intensity", 0.0)},
        track_temp_range={w: pin("track_temp", 30.0)},
        compound_probs={w: ((str(o.get("compound", "medium")), 1.0),)},
        fuel_range=pin("fuel", 30.0),
        aero_level_range=pin("aero_level", 0.5),
        brake_bias_range=pin("brake_bias", 0.58),
        final_drive_range=pin("final_drive", 3.0),
        start_speed_range=(15.0, 40.0),   # v3 spawn policy (see module doc)
        start_lateral_range=(-1.0, 1.0),
        start_heading_error_range=(-0.03, 0.03),
        randomize_start_s=True,
    )


# ----------------------------------------------------------------------------
# Curriculum stages and gates
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GateCheck:
    """One requirement: an eval metric must reach a threshold.

    kind: "laps" (laps completed), "progress" (mean progress, m) — both read
    the stage-yardstick eval named by `target` ("spread" for the benign
    spread-start eval) — or "canonical_lap" (the fixed BENIGN_EVAL s0=0
    episode completes; target unused).
    """

    kind: str
    target: str = "spread"
    threshold: float = 1.0


@dataclass(frozen=True)
class StageGate:
    """Passes when every `all_of` check passes and (if any are listed) at
    least one `any_of` check passes."""

    all_of: tuple = ()
    any_of: tuple = ()

    def check(self, evals: dict, canonical_fn) -> tuple[bool, list]:
        cache: dict = {}

        def value(c: GateCheck) -> float:
            if c.kind == "laps":
                return float(evals[c.target].laps_completed)
            if c.kind == "progress":
                return float(evals[c.target].mean_progress_m)
            if c.kind == "canonical_lap":
                if "canon" not in cache:
                    cache["canon"] = canonical_fn()
                ep = cache["canon"].episodes[0]
                return 1.0 if ep.lap_time is not None else 0.0
            raise ValueError(f"unknown gate check kind: {c.kind!r}")

        entries, all_ok, any_ok = [], True, []
        for group, checks in (("all_of", self.all_of), ("any_of", self.any_of)):
            for c in checks:
                v = value(c)
                ok = v >= c.threshold
                entries.append({"group": group, "kind": c.kind,
                                "target": c.target, "threshold": c.threshold,
                                "value": round(v, 2), "passed": ok})
                if group == "all_of":
                    all_ok = all_ok and ok
                else:
                    any_ok.append(ok)
        passed = all_ok and (not self.any_of or any(any_ok))
        return passed, entries


@dataclass(frozen=True)
class StageConfig:
    """One curriculum stage: an env to train in, a yardstick to evaluate on,
    and a gate that advances to the next stage."""

    name: str
    env_config: EnvConfig
    max_steps: int                 # failsafe budget (policy steps)
    min_steps: int = 0             # no gate-advance before this
    gate: StageGate | None = None  # None = terminal/budget-bound stage
    # yardstick: what each chunk's eval runs — None = SMOKE_EVAL spread
    # starts; a tuple of EvalConditions = one spread eval per condition.
    yardstick: tuple[EvalCondition, ...] | None = None
    episodes: int = 5              # episodes per yardstick entry
    eval_every: int = 25_000       # chunk size: train -> eval -> gate -> ckpt


def l6_driver_config(seed: int = 42) -> SACDriverConfig:
    """Layer 5 defaults with the Layer 6 changes.

    Since v2: the in-train keeper is off (the curriculum trainer does
    per-stage best-keeping on stage-appropriate yardsticks) and actions are
    simplified to [steer, drive] (the attempt-1 restructure — see
    `SimplifiedActions`).

    v4 anti-forgetting changes (the mechanism v1-v3 left untreated: at
    buffer 300k every stage's experience was fully extinct from replay
    before the next stage ended, so the live policy TRADED skills instead
    of accumulating them — all three attempts ended with every end-of-stage
    gate at 0 laps on previously mastered conditions):
      * buffer_size 1.5M — replay spans the entire run; earlier stages'
        transitions keep gradient pressure on earlier skills (rehearsal).
        ~0.5 GB RAM / ~0.5 GB per checkpoint, headroom verified.
      * net_arch (512, 512) — capacity for lap-grade control across many
        setup/weather condition combinations at once (~1.5-2x slower
        gradient steps on CPU; accepted).
    """
    return SACDriverConfig(seed=seed, best_eval_every=None,
                           simplified_actions=True,
                           buffer_size=1_500_000,
                           net_arch=(512, 512))


def dry_rotation_probs() -> dict:
    """All three dry slicks, equal draw — covers the compound one-hot obs
    dims early (v2 evidence: the kept policy died within ~30 m of any
    compound flip, soft and hard alike, on the same dry track it lapped
    on mediums)."""
    return {"dry": (("soft", 1 / 3), ("medium", 1 / 3), ("hard", 1 / 3))}


def stage_dry_dr(fuel: tuple, aero: tuple, bias: tuple, fd: tuple,
                 temp: tuple = (20.0, 45.0)) -> DomainRandomizationConfig:
    """A dry stage: rotating slicks, the given setup ranges, v3 spawns
    (15-40 m/s: low end teaches slow-speed driving, capped top end keeps
    spawns savable — see module doc)."""
    return dataclasses.replace(
        DomainRandomizationConfig(),
        weather_probs={"dry": 1.0},
        rain_intensity_range={"dry": (0.0, 0.0)},
        track_temp_range={"dry": temp},
        compound_probs=dry_rotation_probs(),
        fuel_range=fuel,
        aero_level_range=aero,
        brake_bias_range=bias,
        final_drive_range=fd,
        start_speed_range=(15.0, 40.0),
        start_lateral_range=(-1.0, 1.0),
        start_heading_error_range=(-0.03, 0.03),
    )


def stage_full_dr() -> DomainRandomizationConfig:
    """The terminal stage: Layer 4's randomization tables VERBATIM —
    weather, rain, temperatures, compounds including mismatches (~13% of
    the acceptance distribution; v2 evidence: any never-trained category
    is an instant off). Spawn range 15-60 m/s: a superset of the
    acceptance distribution's 25-60 so evaluation speeds are never
    out-of-distribution, low end kept for slow-speed skill retention."""
    return dataclasses.replace(
        DomainRandomizationConfig(),
        start_speed_range=(15.0, 60.0),
        start_lateral_range=(-1.0, 1.0),
        start_heading_error_range=(-0.03, 0.03),
    )


def l6_default_curriculum() -> tuple[StageConfig, ...]:
    """The v3 (attempt-3) recipe. Every threshold/budget is a labeled,
    provisional knob (calibration-grade values are Layer 7's business).

    History, all on record: v1 (6-dim actions, wide-margin scaffold,
    models/l6_generalist_attempt1) ran 1.75M steps and never lapped. v2
    (SimplifiedActions, benign-first) mastered the benign lap at 133.9 s
    but its one-shot jump to the full setup hypercube produced a brittle
    point-specialist — 0/40 under full randomization
    (reports/layer6_report.md). v3 keeps v2's action space and benign-first
    entry and adds the four evidence-backed fixes documented in the module
    docstring: progressive setup widening (B1 narrow -> B2 full), 15-40 m/s
    spawns, dry-compound rotation from B1, and the terminal stage on Layer 4
    tables verbatim. T7's acceptance floors are unchanged throughout.
    """
    dry_panel = VALIDATION_PANEL[:5]
    return (
        # A: the Layer 5 obligation made the entry gate — laps on demand on
        # the benign preset, true edges from the start.
        StageConfig(
            name="A_benign_laps",
            env_config=EnvConfig(dr=dataclasses.replace(
                benign_training_dr(), start_speed_range=(15.0, 40.0))),
            gate=StageGate(all_of=(GateCheck("laps", "spread", 3),
                                   GateCheck("canonical_lap"))),
            max_steps=400_000,
        ),
        # B1: setup ranges open NARROW around benign; dry compounds rotate.
        # Gate = keep lapping benign while making real distance on the
        # setup-edge conditions (they sit outside B1's training ranges).
        StageConfig(
            name="B1_setup_near",
            env_config=EnvConfig(dr=stage_dry_dr(
                fuel=(25.0, 50.0), aero=(0.35, 0.65),
                bias=(0.56, 0.60), fd=(2.95, 3.10), temp=(25.0, 40.0))),
            gate=StageGate(all_of=(
                GateCheck("laps", "V1_benign", 2),
                GateCheck("progress", "V2_light_lowwing", 1000),
                GateCheck("progress", "V3_heavy_highwing", 1000))),
            yardstick=dry_panel, episodes=3,
            max_steps=300_000,
        ),
        # B2: full Layer 4 SETUP ranges (fuel 20-105, aero 0-1, bias, final
        # drive), still dry. Gate = a lap on each dry setup condition.
        StageConfig(
            name="B2_setup_full",
            env_config=EnvConfig(dr=stage_dry_dr(
                fuel=(20.0, 105.0), aero=(0.0, 1.0),
                bias=(0.54, 0.62), fd=(2.85, 3.15))),
            gate=StageGate(all_of=(
                GateCheck("laps", "V1_benign", 1),
                GateCheck("laps", "V2_light_lowwing", 1),
                GateCheck("laps", "V3_heavy_highwing", 1))),
            yardstick=dry_panel, episodes=3,
            max_steps=350_000,
        ),
        # C: full Layer 4 randomization, mismatched compounds included.
        # Terminal stage — budget-bound, the panel keeper selects the best
        # generalist seen (distinct conditions lapped first).
        StageConfig(
            name="C_full_dr",
            env_config=EnvConfig(dr=stage_full_dr()),
            gate=None,
            yardstick=VALIDATION_PANEL, episodes=3,
            max_steps=450_000,
        ),
    )


def validate_curriculum(stages: tuple[StageConfig, ...]) -> None:
    """Structural sanity — raises on a malformed recipe."""
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate stage names: {names}")
    for s in stages:
        if s.max_steps < 1 or s.min_steps < 0 or s.min_steps > s.max_steps:
            raise ValueError(f"{s.name}: bad budget "
                             f"(min {s.min_steps}, max {s.max_steps})")
        if s.eval_every < 1 or s.episodes < 1:
            raise ValueError(f"{s.name}: bad eval_every/episodes")
        valid_targets = ({"spread"} if s.yardstick is None
                         else {c.name for c in s.yardstick})
        if s.gate is not None:
            for c in s.gate.all_of + s.gate.any_of:
                if c.kind not in ("laps", "progress", "canonical_lap"):
                    raise ValueError(f"{s.name}: unknown check {c.kind!r}")
                if c.kind != "canonical_lap" and c.target not in valid_targets:
                    raise ValueError(f"{s.name}: gate target {c.target!r} "
                                     f"not in yardstick {sorted(valid_targets)}")


# ----------------------------------------------------------------------------
# Shared eval helpers
# ----------------------------------------------------------------------------
def evaluate_panel(driver: SACDriver, panel: tuple[EvalCondition, ...],
                   episodes: int = 3, seed: int = 1234) -> dict:
    """One spread-start EvalReport per named condition.

    Episodes run on the driver's env config with FULL Layer 4 DR tables
    swapped in, so any weather can be pinned even when the driver trained on
    a degenerate table (benign stages, specialists)."""
    env_config = dataclasses.replace(driver.env_config,
                                     dr=DomainRandomizationConfig())
    return {c.name: driver.evaluate(n_episodes=episodes, options=c.options,
                                    seed=seed, spread_starts=True,
                                    env_config=env_config)
            for c in panel}


def _panel_key(evals: dict) -> tuple:
    """Rank policies: DISTINCT conditions lapped, then total laps, then
    capped time-to-lap, then progress — aggregated over whatever evals the
    stage yardstick produced. Breadth outranks depth on purpose: under v2's
    total-laps-first key a benign specialist's 2 laps could never be
    displaced by a policy lapping two conditions once each (v2 stage C:
    zero keeper improvements in 500k steps)."""
    reps = list(evals.values())
    return (sum(1 for r in reps if r.laps_completed > 0),
            sum(r.laps_completed for r in reps),
            -float(np.mean([r.mean_time_to_lap_capped for r in reps])),
            float(np.mean([r.mean_progress_m for r in reps])))


def _summarize(rep: EvalReport) -> dict:
    return {"episodes": len(rep.episodes),
            "laps": rep.laps_completed,
            "mean_lap_time_s": (None if rep.mean_lap_time is None
                                else round(rep.mean_lap_time, 2)),
            "time_to_lap_capped_s": round(rep.mean_time_to_lap_capped, 2),
            "off_track": rep.off_track_count,
            "spin": rep.spin_count,
            "mean_progress_m": round(rep.mean_progress_m, 1),
            "mean_return": round(rep.mean_return, 1)}


def _versions() -> dict:
    import gymnasium
    import stable_baselines3
    import torch
    return {"python": sys.version.split()[0],
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__}


# ----------------------------------------------------------------------------
# The curriculum trainer
# ----------------------------------------------------------------------------
class CurriculumTrainer:
    """Chunked train -> evaluate -> gate -> checkpoint loop over stages.

    Fresh run:   CurriculumTrainer(stages, out_dir, seed=...).run()
    Resume:      CurriculumTrainer.resume(out_dir).run()
    Pause a run: run(step_limit=N) returns after the chunk that crosses N
                 total policy steps (the checkpoint makes it resumable).
    """

    def __init__(self, stages: tuple[StageConfig, ...], out_dir: str | Path,
                 seed: int = 42, budget_scale: float = 1.0,
                 tensorboard: bool = True,
                 driver_config: SACDriverConfig | None = None,
                 track: Track | None = None,
                 _driver: SACDriver | None = None, _state: dict | None = None):
        stages = tuple(stages)
        validate_curriculum(stages)
        self.stages = stages
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.budget_scale = float(budget_scale)
        self.tensorboard = bool(tensorboard)
        self._track = track

        if _driver is not None:                      # resume path
            self.driver = _driver
            st = _state or {}
            self.stage_idx = int(st["stage_idx"])
            self.steps_in_stage = int(st["steps_in_stage"])
            self.total_steps = int(st["total_steps"])
            self.history = list(st.get("history", []))
            self.stage_best_keys = dict(st.get("stage_best_keys", {}))
            self._env_stage = st.get("env_stage")
            self._wall_prev = float(st.get("wall_seconds", 0.0))
            self.finished = bool(st.get("finished", False))
        else:                                        # fresh run
            cfg = driver_config or l6_driver_config(seed=self.seed)
            self.driver = SACDriver(cfg, env_config=stages[0].env_config,
                                    track=track)
            self.stage_idx = 0
            self.steps_in_stage = 0
            self.total_steps = 0
            self.history = []
            self.stage_best_keys = {}
            self._env_stage = stages[0].name
            self._wall_prev = 0.0
            self.finished = False
        self._best_mem: dict | None = None           # in-memory best-in-stage
        self._t0 = time.perf_counter()

    # -- construction helpers --------------------------------------------------
    @classmethod
    def resume(cls, out_dir: str | Path,
               stages: tuple[StageConfig, ...] | None = None,
               track: Track | None = None) -> "CurriculumTrainer":
        """Rebuild a trainer from the latest rotating checkpoint."""
        out = Path(out_dir)
        ckpt = out / "checkpoint"
        if not (ckpt / "curriculum_state.json").exists():
            ckpt = out / "checkpoint_prev"
        state = json.loads((ckpt / "curriculum_state.json").read_text())
        stages = tuple(stages or l6_default_curriculum())
        names = [s.name for s in stages]
        if names != state["stage_names"]:
            raise ValueError(f"stage mismatch: checkpoint has "
                             f"{state['stage_names']}, code has {names}")
        driver = SACDriver.load(ckpt, track=track)
        return cls(stages, out_dir, seed=state["seed"],
                   budget_scale=state["budget_scale"],
                   tensorboard=state.get("tensorboard", True), track=track,
                   _driver=driver, _state=state)

    def _scaled(self, stage: StageConfig) -> StageConfig:
        if self.budget_scale == 1.0:
            return stage
        f = self.budget_scale
        return dataclasses.replace(
            stage,
            max_steps=max(int(stage.max_steps * f), 1),
            min_steps=int(stage.min_steps * f),
            eval_every=max(int(stage.eval_every * f), 250))

    # -- state / logging -------------------------------------------------------
    def _wall(self) -> float:
        return self._wall_prev + (time.perf_counter() - self._t0)

    def _state_dict(self) -> dict:
        return {"stage_names": [s.name for s in self.stages],
                "stage_idx": self.stage_idx,
                "steps_in_stage": self.steps_in_stage,
                "total_steps": self.total_steps,
                "seed": self.seed,
                "budget_scale": self.budget_scale,
                "tensorboard": self.tensorboard,
                "env_stage": self._env_stage,
                "stage_best_keys": self.stage_best_keys,
                "history": self.history,
                "finished": self.finished,
                "wall_seconds": round(self._wall(), 1),
                "versions": _versions()}

    def _record(self, event: str, stage_name: str, details=None) -> None:
        self.history.append({"event": event, "stage": stage_name,
                             "total_steps": self.total_steps,
                             "steps_in_stage": self.steps_in_stage,
                             "wall_s": round(self._wall(), 1),
                             "details": details})

    def _log_scalars(self, evals: dict, gate_passed: bool | None) -> None:
        lg = self.driver.model.logger
        lg.record("curriculum/stage_idx", self.stage_idx)
        lg.record("curriculum/steps_in_stage", self.steps_in_stage)
        if gate_passed is not None:
            lg.record("curriculum/gate_passed", int(gate_passed))
        for name, rep in evals.items():
            lg.record(f"panel/{name}/laps", rep.laps_completed)
            lg.record(f"panel/{name}/progress_m", rep.mean_progress_m)
            lg.record(f"panel/{name}/time_to_lap_s",
                      rep.mean_time_to_lap_capped)
        lg.dump(step=self.driver.model.num_timesteps)

    # -- checkpointing ---------------------------------------------------------
    def _checkpoint(self) -> None:
        """Rotating on-disk checkpoint; the previous one survives until the
        new one has fully landed (a mid-write crash loses nothing)."""
        import shutil
        tmp = self.out / "checkpoint_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        self.driver.save(tmp, include_buffer=True)
        (tmp / "curriculum_state.json").write_text(
            json.dumps(self._state_dict(), indent=2))
        cur, prev = self.out / "checkpoint", self.out / "checkpoint_prev"
        if prev.exists():
            shutil.rmtree(prev)
        if cur.exists():
            cur.rename(prev)
        tmp.rename(cur)
        # convenience mirror for humans/reports (tiny)
        (self.out / "curriculum_state.json").write_text(
            json.dumps(self._state_dict(), indent=2))

    # -- best-policy keeping (per stage, yardstick-scored) ----------------------
    def _keep_if_best(self, stage: StageConfig, key: tuple,
                      evals: dict) -> None:
        prev = self.stage_best_keys.get(stage.name)
        if prev is not None and tuple(prev) >= key:
            return
        self.stage_best_keys[stage.name] = list(key)
        policy = self.driver.model.policy.state_dict()
        self._best_mem = {
            "stage": stage.name,
            "policy": {n: t.detach().clone() for n, t in policy.items()},
            "rms": (copy.deepcopy(self.driver.vec_env.obs_rms)
                    if isinstance(self.driver.vec_env, VecNormalize)
                    else None)}
        self.driver.save(self.out / f"best_{stage.name}")  # survives crashes
        self._record("best", stage.name,
                     {"key": [round(float(k), 3) for k in key],
                      "evals": {n: _summarize(r) for n, r in evals.items()}})
        # refresh the checkpointed state too: a crash before the next chunk's
        # checkpoint must not forget this snapshot (the weights above are
        # already on disk; counters are unchanged since the last checkpoint)
        for state_file in (self.out / "checkpoint" / "curriculum_state.json",
                           self.out / "curriculum_state.json"):
            if state_file.parent.exists():
                state_file.write_text(json.dumps(self._state_dict(),
                                                 indent=2))
        print(f"    [best] {stage.name} @ {self.total_steps:,d} steps: "
              f"conds-lapped {int(key[0])}, laps {int(key[1])}, "
              f"time-to-lap {-key[2]:.1f} s, progress {key[3]:.0f} m",
              flush=True)

    def _restore_stage_best(self, stage: StageConfig) -> None:
        mem = self._best_mem
        if mem is not None and mem["stage"] == stage.name:
            self.driver.model.policy.load_state_dict(mem["policy"])
            if mem["rms"] is not None and isinstance(self.driver.vec_env,
                                                     VecNormalize):
                self.driver.vec_env.obs_rms = mem["rms"]
        elif (self.out / f"best_{stage.name}" / "model.zip").exists():
            self._load_weights(self.out / f"best_{stage.name}")
        else:
            return
        print(f"    [best] {stage.name}: handing its best policy to the "
              f"next stage", flush=True)

    def _load_weights(self, path: Path) -> None:
        """Policy weights + obs stats from a save() dir (no env rebuild)."""
        tmp = SAC.load(str(path / "model.zip"),
                       device=self.driver.config.device)
        self.driver.model.policy.load_state_dict(tmp.policy.state_dict())
        del tmp
        pkl = path / "vecnormalize.pkl"
        if pkl.exists() and isinstance(self.driver.vec_env, VecNormalize):
            with open(pkl, "rb") as f:
                self.driver.vec_env.obs_rms = pickle.load(f).obs_rms

    # -- the loop ----------------------------------------------------------------
    def _enter_stage_if_needed(self, stage: StageConfig) -> None:
        if self._env_stage == stage.name:
            return
        self.driver.swap_env_config(stage.env_config)
        self._env_stage = stage.name
        self._best_mem = None
        self._record("enter_stage", stage.name)
        print(f"[curriculum] entering stage {stage.name} "
              f"(total {self.total_steps:,d} steps)", flush=True)

    def _stage_eval(self, stage: StageConfig) -> dict:
        seed = self.seed + 1234                     # fixed: comparable chunks
        if stage.yardstick is None:
            return {"spread": self.driver.evaluate(
                n_episodes=stage.episodes, options=SMOKE_EVAL, seed=seed,
                spread_starts=True)}
        return evaluate_panel(self.driver, stage.yardstick,
                              episodes=stage.episodes, seed=seed)

    def _canonical(self) -> EvalReport:
        return self.driver.evaluate(n_episodes=1, options=BENIGN_EVAL,
                                    seed=self.seed + 1234)

    def run(self, step_limit: int | None = None) -> dict:
        """Drive the curriculum to completion (or to `step_limit`)."""
        fmts = ["csv", "tensorboard"] if self.tensorboard else ["csv"]
        self.driver.model.set_logger(configure_logger(str(self.out / "tb"),
                                                      fmts))
        if not self.history:
            self._record("enter_stage", self.stages[self.stage_idx].name)
            print(f"[curriculum] entering stage "
                  f"{self.stages[self.stage_idx].name}", flush=True)

        while self.stage_idx < len(self.stages):
            stage = self._scaled(self.stages[self.stage_idx])
            self._enter_stage_if_needed(stage)

            while True:
                evals = self._stage_eval(stage)
                key = _panel_key(evals)
                self._keep_if_best(stage, key, evals)
                if stage.gate is not None:
                    passed, entries = stage.gate.check(evals, self._canonical)
                else:
                    passed, entries = False, []
                self._log_scalars(evals, passed if stage.gate else None)

                if (stage.gate is not None and passed
                        and self.steps_in_stage >= stage.min_steps):
                    self._record("gate_met", stage.name, entries)
                    print(f"[curriculum] {stage.name}: gate met at "
                          f"{self.steps_in_stage:,d} steps in stage",
                          flush=True)
                    break
                if self.steps_in_stage >= stage.max_steps:
                    self._record("budget_exhausted", stage.name,
                                 {"gate_met": False if stage.gate else None,
                                  "checks": entries})
                    print(f"[curriculum] {stage.name}: budget exhausted at "
                          f"{self.steps_in_stage:,d} steps"
                          + (" (gate NOT met)" if stage.gate else ""),
                          flush=True)
                    break

                chunk = min(stage.eval_every,
                            stage.max_steps - self.steps_in_stage)
                self.driver.train(chunk,
                                  progress_every=max(chunk // 5, 1))
                self.steps_in_stage += chunk
                self.total_steps += chunk
                self._checkpoint()
                if step_limit is not None and self.total_steps >= step_limit:
                    self._record("paused", stage.name,
                                 {"step_limit": step_limit})
                    print(f"[curriculum] paused at {self.total_steps:,d} "
                          f"steps (limit {step_limit:,d})", flush=True)
                    return self._state_dict()

            self._restore_stage_best(stage)
            self.stage_idx += 1
            self.steps_in_stage = 0
            self._checkpoint()

        # final artifact: best policy of the last stage, no buffer
        self.finished = True
        final_dir = self.out / "final"
        self.driver.save(final_dir)
        final_panel = evaluate_panel(self.driver, VALIDATION_PANEL,
                                     episodes=3, seed=self.seed + 1234)
        state = self._state_dict()
        state["final_panel"] = {n: _summarize(r)
                                for n, r in final_panel.items()}
        (self.out / "curriculum_state.json").write_text(
            json.dumps(state, indent=2))
        print(f"[curriculum] finished: {self.total_steps:,d} policy steps, "
              f"final driver -> {final_dir}", flush=True)
        return state

    def close(self) -> None:
        self.driver.close()


# ----------------------------------------------------------------------------
# Lap-time distribution over random conditions (spec acceptance bullet)
# ----------------------------------------------------------------------------
@dataclass
class DistributionReport:
    episodes: list                 # one row dict per episode
    finish_rate: float
    lap_stats: dict | None         # over finishers; None if nobody lapped
    dnf_progress: dict | None      # over DNFs; None if everybody lapped
    by_weather: dict

    def table(self) -> str:
        rows = [f"    {'ep':>3}  {'weather':<5} {'compound':<12} {'fuel':>5} "
                f"{'aero':>5}  {'end':<9}  {'lap time':>9}  {'progress':>8}  "
                f"{'avg':>4}"]
        for i, e in enumerate(self.episodes, 1):
            lap = (f"{e['lap_time']:7.1f} s" if e["lap_time"] is not None
                   else "    DNF  ")
            rows.append(
                f"    {i:>3}  {e['weather']:<5} {e['compound']:<12} "
                f"{e['fuel']:5.1f} {e['aero_level']:5.2f}  "
                f"{e['termination']:<9}  {lap:>9}  {e['progress_m']:6.0f} m  "
                f"{e['avg_speed_kmh']:4.0f}")
        n = len(self.episodes)
        fin = int(round(self.finish_rate * n))
        rows.append(f"    finished {fin}/{n} ({self.finish_rate:.0%})")
        if self.lap_stats:
            s = self.lap_stats
            rows.append(f"    lap time over finishers: mean {s['mean']:.1f} s"
                        f" | median {s['median']:.1f} | p10 {s['p10']:.1f}"
                        f" | p90 {s['p90']:.1f} | min {s['min']:.1f}"
                        f" | max {s['max']:.1f}")
        if self.dnf_progress:
            rows.append(f"    DNF progress: mean "
                        f"{self.dnf_progress['mean']:.0f} m | median "
                        f"{self.dnf_progress['median']:.0f} m")
        for w, st in sorted(self.by_weather.items()):
            mean_lap = (f"{st['mean_lap_time']:.1f} s"
                        if st["mean_lap_time"] is not None else "-")
            rows.append(f"    {w:<5}: {st['finished']}/{st['episodes']} "
                        f"finished, mean lap {mean_lap}")
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def evaluate_distribution(driver: SACDriver, n_episodes: int = 40,
                          seed: int = 777, env_config: EnvConfig | None = None,
                          track: Track | None = None,
                          deterministic: bool = True) -> DistributionReport:
    """Drive `n_episodes` under full domain randomization (Layer 4 default DR
    unless overridden) and report the lap-time distribution — the spec's
    "lap-time distribution over random conditions" bullet."""
    env_config = env_config or EnvConfig(dr=DomainRandomizationConfig())
    track = track or driver._track or Track.from_synthetic()
    env = build_episode_env(track, env_config, driver.config)

    episodes = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)        # options=None -> DR draw
        setup = dict(info["setup"])
        ret, speeds, lap, term, done = 0.0, [], None, None, False
        while not done:
            action = driver.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ret += float(r)
            speeds.append(float(info["speed"]))
            if terminated or truncated:
                done = True
                lap = info.get("lap_time")
                term = info.get("termination",
                                "lap" if lap is not None else "max_steps")
        episodes.append({
            "weather": setup["weather"], "compound": setup["compound"],
            "rain_intensity": round(float(setup["rain_intensity"]), 3),
            "track_temp": round(float(setup["track_temp"]), 1),
            "fuel": round(float(setup["fuel"]), 1),
            "aero_level": round(float(setup["aero_level"]), 3),
            "brake_bias": round(float(setup["brake_bias"]), 3),
            "final_drive": round(float(setup["final_drive"]), 3),
            "lap_time": None if lap is None else round(float(lap), 2),
            "progress_m": round(float(info["lap_fraction"]) * track.length, 1),
            "avg_speed_kmh": round(float(np.mean(speeds)) * 3.6, 1),
            "termination": term, "return": round(ret, 1)})
    env.close()

    laps = [e["lap_time"] for e in episodes if e["lap_time"] is not None]
    dnfs = [e["progress_m"] for e in episodes if e["lap_time"] is None]
    lap_stats = None
    if laps:
        arr = np.asarray(laps)
        lap_stats = {"mean": float(arr.mean()),
                     "median": float(np.median(arr)),
                     "p10": float(np.percentile(arr, 10)),
                     "p90": float(np.percentile(arr, 90)),
                     "min": float(arr.min()), "max": float(arr.max())}
    dnf_progress = None
    if dnfs:
        arr = np.asarray(dnfs)
        dnf_progress = {"mean": float(arr.mean()),
                        "median": float(np.median(arr))}
    by_weather = {}
    for w in sorted({e["weather"] for e in episodes}):
        rows = [e for e in episodes if e["weather"] == w]
        wl = [e["lap_time"] for e in rows if e["lap_time"] is not None]
        by_weather[w] = {"episodes": len(rows), "finished": len(wl),
                         "mean_lap_time": (float(np.mean(wl)) if wl
                                           else None)}
    return DistributionReport(
        episodes=episodes,
        finish_rate=len(laps) / max(len(episodes), 1),
        lap_stats=lap_stats, dnf_progress=dnf_progress,
        by_weather=by_weather)


# ----------------------------------------------------------------------------
# Generalist vs specialists (spec acceptance bullet: the documented %)
# ----------------------------------------------------------------------------
@dataclass
class ConditionComparison:
    condition: str
    specialist_steps: int
    generalist: dict               # _summarize() of the generalist's eval
    specialist: dict
    gap_time_to_lap_pct: float     # +x% = generalist slower (capped metric)
    gap_lap_time_pct: float | None  # only when BOTH lap
    progress_ratio: float          # generalist progress / specialist progress


@dataclass
class ComparisonReport:
    items: list
    episodes_per_condition: int

    def table(self) -> str:
        rows = [f"    {'condition':<20} {'metric':<28} {'generalist':>10}  "
                f"{'specialist':>10}  {'gap':>8}"]
        for it in self.items:
            g, s = it.generalist, it.specialist
            rows.append(f"    {it.condition:<20} "
                        f"{'time-to-lap (DNF=cap), s':<28} "
                        f"{g['time_to_lap_capped_s']:>10.1f}  "
                        f"{s['time_to_lap_capped_s']:>10.1f}  "
                        f"{it.gap_time_to_lap_pct:>+7.1f}%")
            if it.gap_lap_time_pct is not None:
                rows.append(f"    {'':<20} {'mean lap time, s':<28} "
                            f"{g['mean_lap_time_s']:>10.1f}  "
                            f"{s['mean_lap_time_s']:>10.1f}  "
                            f"{it.gap_lap_time_pct:>+7.1f}%")
            rows.append(f"    {'':<20} {'laps / progress ratio':<28} "
                        f"{g['laps']:>10d}  {s['laps']:>10d}  "
                        f"{it.progress_ratio:>7.2f}x")
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {"episodes_per_condition": self.episodes_per_condition,
                "items": [dataclasses.asdict(i) for i in self.items]}


def compare_generalist_vs_specialists(
        generalist: SACDriver,
        conditions: tuple[EvalCondition, ...] = HELD_OUT_PANEL,
        specialist_steps: int = 250_000, episodes: int = 5,
        seed: int = 4242, out_dir: str | Path = "runs/l6_specialists",
        eval_every: int = 25_000, tensorboard: bool = False,
        reuse: bool = True, track: Track | None = None) -> ComparisonReport:
    """Train (or reuse) one pinned-condition specialist per held-out
    condition, evaluate both drivers on identical episodes, and document the
    gap. The spec sets no pass bar — the documented % IS the deliverable.

    Metric hierarchy (recorded per condition): mean lap time when both lap;
    the 300 s-capped time-to-lap always (well-defined even when nobody laps);
    progress ratio always.
    """
    out = Path(out_dir)
    items = []
    for i, cond in enumerate(conditions):
        sdir = out / f"specialist_{cond.name}"
        if reuse and (sdir / "final" / "model.zip").exists():
            print(f"[compare] reusing specialist {cond.name}", flush=True)
            spec = SACDriver.load(sdir / "final", track=track)
        else:
            print(f"[compare] training specialist {cond.name} "
                  f"({specialist_steps:,d} steps)", flush=True)
            stage = StageConfig(
                name=f"spec_{cond.name}",
                env_config=EnvConfig(dr=pinned_dr(cond)),
                max_steps=specialist_steps, gate=None,
                yardstick=(cond,), episodes=3, eval_every=eval_every)
            trainer = CurriculumTrainer(
                (stage,), sdir, seed=seed + 101 * (i + 1),
                tensorboard=tensorboard,
                driver_config=l6_driver_config(seed=seed + 101 * (i + 1)),
                track=track)
            trainer.run()
            spec = trainer.driver

        eval_seed = 9000 + 17 * i                   # identical episodes
        full = DomainRandomizationConfig()          # any weather pinnable
        g = generalist.evaluate(
            n_episodes=episodes, options=cond.options, seed=eval_seed,
            spread_starts=True,
            env_config=dataclasses.replace(generalist.env_config, dr=full))
        s = spec.evaluate(
            n_episodes=episodes, options=cond.options, seed=eval_seed,
            spread_starts=True,
            env_config=dataclasses.replace(spec.env_config, dr=full))
        spec.close()

        gap = (100.0 * (g.mean_time_to_lap_capped - s.mean_time_to_lap_capped)
               / max(s.mean_time_to_lap_capped, 1e-9))
        lap_gap = None
        if g.mean_lap_time is not None and s.mean_lap_time is not None:
            lap_gap = (100.0 * (g.mean_lap_time - s.mean_lap_time)
                       / s.mean_lap_time)
        items.append(ConditionComparison(
            condition=cond.name, specialist_steps=specialist_steps,
            generalist=_summarize(g), specialist=_summarize(s),
            gap_time_to_lap_pct=round(gap, 2),
            gap_lap_time_pct=None if lap_gap is None else round(lap_gap, 2),
            progress_ratio=round(g.mean_progress_m
                                 / max(s.mean_progress_m, 1e-9), 3)))
        print(f"[compare] {cond.name}: time-to-lap gap {gap:+.1f}% "
              f"(generalist vs specialist)", flush=True)
    return ComparisonReport(items=items, episodes_per_condition=episodes)


# ----------------------------------------------------------------------------
# Run report (json + human-readable markdown)
# ----------------------------------------------------------------------------
def build_report(state: dict | None = None, panel: dict | None = None,
                 distribution: DistributionReport | None = None,
                 comparison: ComparisonReport | None = None) -> dict:
    return {"versions": _versions(),
            "curriculum": state,
            "validation_panel": ({n: _summarize(r) for n, r in panel.items()}
                                 if panel else None),
            "lap_time_distribution": (distribution.to_dict()
                                      if distribution else None),
            "generalist_vs_specialists": (comparison.to_dict()
                                          if comparison else None)}


def render_markdown(report: dict, distribution: DistributionReport | None =
                    None, comparison: ComparisonReport | None = None) -> str:
    lines = ["# Layer 6 report — generalist driver", ""]
    v = report.get("versions", {})
    lines += [f"Versions: {', '.join(f'{k} {x}' for k, x in v.items())}", ""]
    cur = report.get("curriculum")
    if cur:
        lines += ["## Curriculum run", "",
                  f"- total policy steps: {cur.get('total_steps', 0):,d}",
                  f"- wall time: {cur.get('wall_seconds', 0) / 3600:.2f} h",
                  f"- budget scale: {cur.get('budget_scale', 1.0)}", ""]
        for h in cur.get("history", []):
            if h["event"] in ("enter_stage", "gate_met", "budget_exhausted",
                              "paused"):
                lines.append(f"- `{h['stage']}` {h['event']} at "
                             f"{h['total_steps']:,d} total steps "
                             f"({h['wall_s'] / 3600:.2f} h)")
        lines.append("")
    pan = report.get("validation_panel")
    if pan:
        lines += ["## Validation panel (final policy)", "",
                  "| condition | laps | mean lap (s) | time-to-lap (s) | "
                  "progress (m) |", "|---|---|---|---|---|"]
        for name, s in pan.items():
            lap = s["mean_lap_time_s"] if s["mean_lap_time_s"] else "-"
            lines.append(f"| {name} | {s['laps']}/{s['episodes']} | {lap} | "
                         f"{s['time_to_lap_capped_s']} | "
                         f"{s['mean_progress_m']} |")
        lines.append("")
    if distribution is not None:
        lines += ["## Lap-time distribution over random conditions", "",
                  "```", distribution.table(), "```", ""]
    if comparison is not None:
        lines += ["## Generalist vs setup-specialists (documented gap)", "",
                  "```", comparison.table(), "```", "",
                  "Positive gap = generalist slower than the specialist on "
                  "that condition (300 s-capped time-to-lap).", ""]
    return "\n".join(lines)
