"""Layer 6 acceptance tests — training loop, curriculum, DR (LAYER_SPECS § 6).

Runs two ways:
  * `pytest tests/test_train.py`   — standard test run
  * `python tests/test_train.py`   — standalone PASS/FAIL/SKIP report

Fully offline (synthetic track). T1–T6 validate the machinery at toy scale in
minutes. T7 and T8 are the result gates and need the trained generalist —
they SKIP (with the exact command to produce the artifact) until
`scripts/train.py --recipe l6_generalist` has run; this mirrors Layer 2's
online-gated T8. T7 carries the Layer 5 obligation: full-lap completion and
lap-time assertions against the random baseline, plus the spec's lap-time
distribution over random conditions. T8 asserts the generalist-vs-specialist
gap is measured and documented (the spec sets no numeric bar on the %; the
documentation is the deliverable).
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import SkipTest  # honored by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np

from src.agents.curriculum import (HELD_OUT_PANEL, VALIDATION_PANEL,
                                   CurriculumTrainer, GateCheck, StageConfig,
                                   StageGate, build_report,
                                   compare_generalist_vs_specialists,
                                   evaluate_distribution, evaluate_panel,
                                   l6_default_curriculum, pinned_dr,
                                   render_markdown, validate_curriculum)
from src.agents.sac_driver import (BENIGN_EVAL, SMOKE_EVAL, SACDriver,
                                   SACDriverConfig, SimplifiedActions,
                                   benign_training_dr,
                                   benign_training_env_config)
from src.envs.f1_env import DomainRandomizationConfig, EnvConfig, F1Env
from src.tracks.track import Track

TRACK = Track.from_synthetic()  # built once and shared; construction ~1 s

PURSUIT_REF_S = 140.7  # Layer 4's hand-coded pursuit lap on BENIGN (reference)

TERMINATIONS = {"off_track", "spin", "fuel_out", "stall", "lap", "max_steps"}


def tiny_cfg(**over) -> SACDriverConfig:
    """Small-but-real config so machinery tests run in seconds. The in-train
    keeper is off — the curriculum trainer owns model selection in Layer 6."""
    base = dict(buffer_size=20_000, learning_starts=200, batch_size=64,
                net_arch=(64, 64), seed=7, best_eval_every=None)
    base.update(over)
    return SACDriverConfig(**base)


def toy_stage(name: str, gate: StageGate | None, max_steps: int,
              eval_every: int = 600, min_steps: int = 0) -> StageConfig:
    return StageConfig(name=name,
                       env_config=EnvConfig(dr=benign_training_dr()),
                       gate=gate, max_steps=max_steps, min_steps=min_steps,
                       episodes=2, eval_every=eval_every)


def _within(rng, ref) -> bool:
    return ref[0] <= rng[0] and rng[1] <= ref[1]


# ---------------------------------------------------------------------------
# T1 — the default recipe is structurally sound and hygienic
# ---------------------------------------------------------------------------
def test_t1_curriculum_config_sanity():
    stages = l6_default_curriculum()
    validate_curriculum(stages)  # raises on malformed recipes
    assert len(stages) >= 3

    # every stage trains on the real track edges (the v1 wide-margin
    # scaffold was measured useless and dropped)
    for s in stages:
        assert s.env_config.off_track_margin == EnvConfig().off_track_margin

    # the spec's "full randomization" is reached on the setup + weather axes:
    # final stage == Layer 4 defaults for every condition range. Compounds
    # are weather-MATCHED (one per weather) — the documented post-Layer-7
    # descope while tire-temperature physics is a placeholder.
    ref = DomainRandomizationConfig()
    last = stages[-1].env_config.dr
    assert last.weather_probs == ref.weather_probs
    assert last.rain_intensity_range == ref.rain_intensity_range
    assert last.track_temp_range == ref.track_temp_range
    assert last.fuel_range == ref.fuel_range
    assert last.aero_level_range == ref.aero_level_range
    assert last.brake_bias_range == ref.brake_bias_range
    assert last.final_drive_range == ref.final_drive_range
    for w, table in last.compound_probs.items():
        assert len(table) == 1 and table[0][1] == 1.0, (w, table)

    # every stage's CONDITION ranges live inside Layer 4's default bounds
    # (start-pose ranges are exploration protocol, not conditions — exempt)
    ref_compounds = {w: {c for c, _ in tbl}
                     for w, tbl in ref.compound_probs.items()}
    for s in stages:
        dr = s.env_config.dr
        assert _within(dr.fuel_range, ref.fuel_range), s.name
        assert _within(dr.aero_level_range, ref.aero_level_range), s.name
        assert _within(dr.brake_bias_range, ref.brake_bias_range), s.name
        assert _within(dr.final_drive_range, ref.final_drive_range), s.name
        for w in dr.weather_probs:
            assert w in ref.weather_probs, (s.name, w)
            assert _within(dr.rain_intensity_range[w],
                           ref.rain_intensity_range[w]), (s.name, w)
            for c, _p in dr.compound_probs[w]:
                assert c in ref_compounds[w], (s.name, w, c)
        assert 0 < s.eval_every <= s.max_steps, s.name
        assert 0 <= s.min_steps <= s.max_steps, s.name

    # evaluation hygiene: the held-out panel shares nothing with validation
    v_names = {c.name for c in VALIDATION_PANEL}
    h_names = {c.name for c in HELD_OUT_PANEL}
    assert not (v_names & h_names)
    v_opts = {frozenset(c.options.items()) for c in VALIDATION_PANEL}
    h_opts = {frozenset(c.options.items()) for c in HELD_OUT_PANEL}
    assert not (v_opts & h_opts), "a held-out condition equals a validation one"
    assert len(HELD_OUT_PANEL) >= 3  # spec: "several held-out setups/weather"

    # every panel condition is inside Layer 4's DR bounds (an in-distribution
    # but never-trained-on point — the honest reading of "held-out")
    for c in VALIDATION_PANEL + HELD_OUT_PANEL:
        o = c.options
        w = o["weather"]
        assert w in ref.weather_probs, c.name
        lo, hi = ref.rain_intensity_range[w]
        assert lo <= o["rain_intensity"] <= hi, c.name
        lo, hi = ref.track_temp_range[w]
        assert lo <= o["track_temp"] <= hi, c.name
        assert ref.fuel_range[0] <= o["fuel"] <= ref.fuel_range[1], c.name
        assert (ref.aero_level_range[0] <= o["aero_level"]
                <= ref.aero_level_range[1]), c.name
        assert (ref.brake_bias_range[0] <= o["brake_bias"]
                <= ref.brake_bias_range[1]), c.name
        assert (ref.final_drive_range[0] <= o["final_drive"]
                <= ref.final_drive_range[1]), c.name
        assert o["compound"] in ref_compounds[w], c.name

    # specialists pin exactly their condition
    dr = pinned_dr(HELD_OUT_PANEL[0])
    o = HELD_OUT_PANEL[0].options
    assert dr.weather_probs == {o["weather"]: 1.0}
    assert dr.fuel_range == (o["fuel"], o["fuel"])
    assert dr.compound_probs[o["weather"]] == ((o["compound"], 1.0),)

    # a malformed recipe is rejected (gate targeting a missing eval)
    bad = (StageConfig(name="bad", env_config=EnvConfig(),
                       gate=StageGate(all_of=(GateCheck("laps", "nope", 1),)),
                       max_steps=100),)
    try:
        validate_curriculum(bad)
        raise AssertionError("validate_curriculum accepted a bad gate target")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# T2 — domain randomization actually happens on every reset (spec bullet)
# ---------------------------------------------------------------------------
def test_t2_domain_randomization_every_reset():
    stages = l6_default_curriculum()

    # final stage: setup + weather vary within the configured ranges
    env = F1Env(track=TRACK, config=stages[-1].env_config)
    ref = stages[-1].env_config.dr
    setups = []
    for i in range(200):
        _, info = env.reset(seed=10_000 + i)
        setups.append(info["setup"])
    env.close()

    assert {s["weather"] for s in setups} == set(ref.weather_probs), \
        "200 resets should draw every weather (probs 0.60/0.25/0.15)"
    # matched compounds: exactly one per weather, so three across 200 draws
    assert ({s["compound"] for s in setups}
            == {"medium", "intermediate", "wet"})
    fuels = [s["fuel"] for s in setups]
    aeros = [s["aero_level"] for s in setups]
    biases = [s["brake_bias"] for s in setups]
    fds = [s["final_drive"] for s in setups]
    assert max(fuels) - min(fuels) > 40.0      # range spans 85 kg
    assert max(aeros) - min(aeros) > 0.5       # range spans 1.0
    assert max(biases) - min(biases) > 0.03    # range spans 0.08
    assert max(fds) - min(fds) > 0.1           # range spans 0.30
    for s in setups:
        assert _within((s["fuel"], s["fuel"]), ref.fuel_range)
        assert _within((s["aero_level"], s["aero_level"]),
                       ref.aero_level_range)
        assert _within((s["brake_bias"], s["brake_bias"]),
                       ref.brake_bias_range)
        assert _within((s["final_drive"], s["final_drive"]),
                       ref.final_drive_range)
        lo, hi = ref.rain_intensity_range[s["weather"]]
        assert lo <= s["rain_intensity"] <= hi
        lo, hi = ref.track_temp_range[s["weather"]]
        assert lo <= s["track_temp"] <= hi

    # stage A: the CONDITION is pinned; only the start pose varies
    env = F1Env(track=TRACK, config=stages[0].env_config)
    infos = [env.reset(seed=20_000 + i)[1] for i in range(30)]
    env.close()
    conditions = {frozenset(i["setup"].items()) for i in infos}
    assert len(conditions) == 1, "benign stage must pin the condition"
    starts = {round(i["s"], 1) for i in infos}
    assert len(starts) > 20, "start position should randomize over the lap"


# ---------------------------------------------------------------------------
# T3 — stage machinery: gates advance, failsafes record, everything logs
# ---------------------------------------------------------------------------
def test_t3_stage_machinery_and_logging():
    out = Path(tempfile.mkdtemp(prefix="l6_t3_"))
    stages = (
        # met after one chunk (min_steps forces that one chunk of training)
        toy_stage("meetable",
                  StageGate(any_of=(GateCheck("progress", "spread", 0.0),)),
                  max_steps=2400, min_steps=600),
        # unmeetable for an untrained tiny policy -> budget failsafe
        toy_stage("unmeetable",
                  StageGate(all_of=(GateCheck("laps", "spread", 5),)),
                  max_steps=1200),
    )
    tr = CurriculumTrainer(stages, out, seed=7, driver_config=tiny_cfg(),
                           tensorboard=True)
    state = tr.run()
    tr.close()

    assert state["finished"] is True
    assert state["total_steps"] == 600 + 1200

    events = {(h["event"], h["stage"]) for h in state["history"]}
    assert ("enter_stage", "meetable") in events
    assert ("enter_stage", "unmeetable") in events
    assert ("best", "meetable") in events  # first eval always snapshots
    gate_hits = [h for h in state["history"] if h["event"] == "gate_met"]
    assert len(gate_hits) == 1 and gate_hits[0]["stage"] == "meetable"
    assert gate_hits[0]["steps_in_stage"] == 600
    exhausted = [h for h in state["history"]
                 if h["event"] == "budget_exhausted"]
    assert len(exhausted) == 1 and exhausted[0]["stage"] == "unmeetable"
    assert exhausted[0]["details"]["gate_met"] is False

    # log everything: tensorboard events + csv + rotating checkpoint + final
    assert list((out / "tb").glob("events.out.tfevents.*")), "no tb events"
    assert (out / "tb" / "progress.csv").exists()
    for f in ("model.zip", "replay_buffer.pkl", "vecnormalize.pkl",
              "curriculum_state.json"):
        assert (out / "checkpoint" / f).exists(), f"checkpoint missing {f}"
    assert (out / "final" / "model.zip").exists()
    assert (out / "best_meetable" / "model.zip").exists()

    # the state mirror on disk agrees and carries the final panel
    disk = json.loads((out / "curriculum_state.json").read_text())
    assert disk["total_steps"] == state["total_steps"]
    assert set(disk["final_panel"]) == {c.name for c in VALIDATION_PANEL}
    assert disk["versions"]["stable_baselines3"]


# ---------------------------------------------------------------------------
# T4 — checkpoint/resume mid-curriculum (spec: "training resumes cleanly")
# ---------------------------------------------------------------------------
def test_t4_checkpoint_resume_cleanly():
    # buffer round-trip parity through the Layer 5 extension first
    d = SACDriver(tiny_cfg(), track=TRACK)
    d.train(400)
    save_dir = Path(tempfile.mkdtemp(prefix="l6_t4buf_")) / "ckpt"
    d.save(save_dir, include_buffer=True)
    d2 = SACDriver.load(save_dir, track=TRACK)
    b1, b2 = d.model.replay_buffer, d2.model.replay_buffer
    assert b2.size() == b1.size() == 400
    assert np.array_equal(b1.observations[:400], b2.observations[:400])
    assert np.array_equal(b1.actions[:400], b2.actions[:400])
    assert np.array_equal(b1.rewards[:400], b2.rewards[:400])
    d.close()
    d2.close()

    # now a curriculum run paused mid-stage-2 and resumed
    out = Path(tempfile.mkdtemp(prefix="l6_t4_"))
    stages = (toy_stage("s1", None, max_steps=1200),
              toy_stage("s2", None, max_steps=1200))
    tr = CurriculumTrainer(stages, out, seed=7, driver_config=tiny_cfg(),
                           tensorboard=False)
    paused = tr.run(step_limit=1800)
    rms_at_pause = tr.driver.vec_env.obs_rms.mean.copy()
    tr.close()
    assert paused["finished"] is False
    assert paused["total_steps"] == 1800
    assert paused["stage_idx"] == 1 and paused["steps_in_stage"] == 600
    assert any(h["event"] == "paused" for h in paused["history"])

    tr2 = CurriculumTrainer.resume(out, stages=stages)
    assert tr2.total_steps == 1800
    assert tr2.stage_idx == 1 and tr2.steps_in_stage == 600
    assert tr2.driver.model.num_timesteps == 1800
    assert tr2.driver.model.replay_buffer.size() == 1800
    assert np.array_equal(tr2.driver.vec_env.obs_rms.mean, rms_at_pause), \
        "obs normalization stats must survive resume exactly"

    final = tr2.run()
    tr2.close()
    assert final["finished"] is True
    assert final["total_steps"] == 2400, \
        "resumed run must complete the exact configured budget"


# ---------------------------------------------------------------------------
# T5 — env swap mid-training: stats carry over, training continues
# ---------------------------------------------------------------------------
def test_t5_env_swap_continuity():
    d = SACDriver(tiny_cfg(), env_config=benign_training_env_config(),
                  track=TRACK)
    d.train(600)
    rms_mean = d.vec_env.obs_rms.mean.copy()
    count_before = float(d.vec_env.obs_rms.count)
    obs_space, act_space = d.model.observation_space, d.model.action_space

    d.swap_env_config(l6_default_curriculum()[0].env_config)  # 3 m margin
    assert np.array_equal(d.vec_env.obs_rms.mean, rms_mean), \
        "VecNormalize stats must transplant across the swap"
    assert d.model.observation_space == obs_space
    assert d.model.action_space == act_space

    d.train(400)
    assert d.model.num_timesteps == 1000
    assert float(d.vec_env.obs_rms.count) > count_before, \
        "stats must keep learning from the transplanted baseline, not reset"
    for name, p in d.model.policy.named_parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all(), name
    d.close()


# ---------------------------------------------------------------------------
# T6 — evaluators: panel, distribution, comparison (machinery at toy scale)
# ---------------------------------------------------------------------------
def test_t6_evaluators_toy_scale():
    d = SACDriver(tiny_cfg(), track=TRACK)  # untrained — machinery only

    panel = evaluate_panel(d, VALIDATION_PANEL[:2], episodes=2, seed=99)
    assert set(panel) == {c.name for c in VALIDATION_PANEL[:2]}
    for rep in panel.values():
        assert len(rep.episodes) == 2
        for e in rep.episodes:
            assert e.termination in TERMINATIONS
            assert np.isfinite(e.progress_m) and e.progress_m >= 0

    r1 = evaluate_distribution(d, n_episodes=10, seed=555, track=TRACK)
    r2 = evaluate_distribution(d, n_episodes=10, seed=555, track=TRACK)
    assert r1.to_dict() == r2.to_dict(), "distribution eval must be seeded"
    assert len({e["weather"] for e in r1.episodes}) >= 2
    assert 0.0 <= r1.finish_rate <= 1.0
    for e in r1.episodes:
        assert e["termination"] in TERMINATIONS
        assert np.isfinite(e["progress_m"])
    assert "finished" in r1.table()

    out = Path(tempfile.mkdtemp(prefix="l6_t6_"))
    comp = compare_generalist_vs_specialists(
        d, (VALIDATION_PANEL[0],), specialist_steps=900, episodes=2,
        seed=31, out_dir=out, eval_every=450)
    assert len(comp.items) == 1
    it = comp.items[0]
    assert it.condition == "V1_benign"
    assert it.specialist_steps == 900
    assert np.isfinite(it.gap_time_to_lap_pct)
    assert np.isfinite(it.progress_ratio) and it.progress_ratio > 0
    assert "V1_benign" in comp.table()

    report = build_report(state=None, panel=panel, distribution=r1,
                          comparison=comp)
    json.dumps(report)  # must be JSON-serializable end to end
    md = render_markdown(report, distribution=r1, comparison=comp)
    assert "Layer 6 report" in md
    assert "Generalist vs setup-specialists" in md
    d.close()


# ---------------------------------------------------------------------------
# T6b — the simplified action space maps exactly onto Layer 4's interface
# ---------------------------------------------------------------------------
def test_t6b_action_adapter():
    env = SimplifiedActions(F1Env(track=TRACK,
                                  config=benign_training_env_config()))
    assert env.action_space.shape == (2,)
    env.reset(seed=3, options=dict(BENIGN_EVAL))
    veh = env.unwrapped.vehicle

    # full drive: throttle only, ERS rides the drive axis, DRS requested,
    # gear encodes Layer 1's auto_gear for the live speed
    a6 = env.action(np.array([0.2, 1.0], dtype=np.float32))
    assert a6.shape == (6,)
    assert a6[0] == 1.0 and a6[1] == -1.0            # throttle 1, brake 0
    assert abs(float(a6[2]) - 0.2) < 1e-6            # steer passthrough
    g = veh.auto_gear(veh.state.vx)
    assert int(round(1.0 + 3.5 * (float(a6[3]) + 1.0))) == g
    assert a6[4] == 1.0 and a6[5] == 1.0             # ERS deploy, DRS on

    # full brake: mutually exclusive with throttle, no ERS deploy
    a6 = env.action(np.array([-0.5, -1.0], dtype=np.float32))
    assert a6[0] == -1.0 and a6[1] == 1.0 and a6[4] == -1.0
    env.close()

    # end-to-end through the driver: 2-dim policy trains and evaluates
    d = SACDriver(tiny_cfg(simplified_actions=True), track=TRACK)
    assert d.model.action_space.shape == (2,)
    d.train(300)
    rep = d.evaluate(n_episodes=1, options=SMOKE_EVAL, seed=5)
    assert rep.episodes[0].termination in TERMINATIONS
    for name, p in d.model.policy.named_parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all(), name
    d.close()


# ---------------------------------------------------------------------------
# T7 — the trained generalist drives (artifact-gated; Layer 5's deferred
#      full-lap + lap-time acceptance lands here)
# ---------------------------------------------------------------------------
def _find_generalist() -> Path | None:
    # models/l6_generalist is the ACCEPTED artifact path; attempt dirs and
    # in-flight runs are reachable via L6_DRIVER_DIR only, so a failed or
    # stale run can never masquerade as the accepted driver.
    for cand in (os.environ.get("L6_DRIVER_DIR"), "models/l6_generalist"):
        if cand and (Path(cand) / "model.zip").exists():
            return Path(cand)
    return None


def test_t7_generalist_drives():
    d_dir = _find_generalist()
    if d_dir is None:
        raise SkipTest("no trained generalist yet — run: python "
                       "scripts/train.py --recipe l6_generalist "
                       "--out runs/l6_generalist")
    driver = SACDriver.load(d_dir, track=TRACK)
    print(f"\n    generalist: {d_dir}")

    # (1) canonical benign lap COMPLETES — the recorded Layer 5 obligation.
    #     250 s is a deliberately generous sanity bound (realism is Layer 7's
    #     job); the pursuit reference is printed for comparison, not asserted.
    canon = driver.evaluate(1, options=BENIGN_EVAL, seed=123)
    e = canon.episodes[0]
    assert e.lap_time is not None, (
        f"canonical benign lap DNF: {e.termination} at {e.progress_m:.0f} m")
    assert e.lap_time < 250.0, f"canonical lap {e.lap_time:.1f} s >= 250 s"
    print(f"    canonical benign lap: {e.lap_time:.1f} s "
          f"(hand-coded pursuit reference {PURSUIT_REF_S} s)")

    # (2) spread rolling starts: laps on demand, measurably better than random
    trained = driver.evaluate(5, options=SMOKE_EVAL, seed=123,
                              spread_starts=True)
    rand = driver.evaluate(5, options=SMOKE_EVAL, seed=123,
                           spread_starts=True, policy="random")
    print("    trained (deterministic):")
    print(trained.table())
    print("    random baseline (same starts):")
    print(rand.table())
    assert trained.laps_completed >= 3, (
        f"only {trained.laps_completed}/5 spread-start laps")
    assert (trained.mean_time_to_lap_capped
            < rand.mean_time_to_lap_capped), "no lap-time edge over random"
    t_km = sum(ep.progress_m for ep in trained.episodes) / 1000.0
    r_km = sum(ep.progress_m for ep in rand.episodes) / 1000.0
    t_rate = trained.off_track_count / max(t_km, 1e-9)
    r_rate = rand.off_track_count / max(r_km, 1e-9)
    assert t_rate < r_rate, (
        f"off-track per km: trained {t_rate:.2f} not below random "
        f"{r_rate:.2f}")

    # (3) lap-time distribution over random conditions (spec bullet).
    #     30% finish floor is provisional — flagged, never silently lowered.
    dist = evaluate_distribution(driver, n_episodes=40, seed=777, track=TRACK)
    print("    lap-time distribution over 40 random conditions:")
    print(dist.table())
    assert dist.finish_rate >= 0.30, (
        f"finish rate {dist.finish_rate:.0%} under full DR < 30%")
    driver.close()


# ---------------------------------------------------------------------------
# T8 — the generalist-vs-specialist gap is measured and documented
# ---------------------------------------------------------------------------
def test_t8_documented_comparison():
    report_path = None
    d_dir = _find_generalist()
    candidates = [Path("models/l6_generalist/run_report.json")]
    if d_dir is not None:
        candidates.insert(0, d_dir / "run_report.json")
    for cand in candidates:
        if cand.exists():
            report_path = cand
            break
    if report_path is None:
        raise SkipTest("no run report yet — run: python scripts/train.py "
                       "--compare <driver_dir>")
    report = json.loads(report_path.read_text())
    comp = report.get("generalist_vs_specialists")
    if not comp:
        raise SkipTest(f"{report_path} has no comparison block yet — run: "
                       "python scripts/train.py --compare "
                       "runs/l6_generalist/final")

    items = comp["items"]
    assert len(items) >= 3, "spec: several held-out conditions"
    held_out_names = {c.name for c in HELD_OUT_PANEL}
    print(f"\n    documented generalist-vs-specialist gap ({report_path}):")
    for it in items:
        assert it["condition"] in held_out_names, it["condition"]
        assert it["specialist_steps"] > 0
        assert np.isfinite(it["gap_time_to_lap_pct"])
        assert np.isfinite(it["progress_ratio"])
        for side in ("generalist", "specialist"):
            assert "time_to_lap_capped_s" in it[side]
            assert "laps" in it[side]
        lap_gap = it["gap_lap_time_pct"]
        lap_str = (f", lap-time gap {lap_gap:+.1f}%" if lap_gap is not None
                   else " (lap-time gap undefined: a side never lapped)")
        print(f"      {it['condition']}: time-to-lap gap "
              f"{it['gap_time_to_lap_pct']:+.1f}%{lap_str}")
    assert report.get("lap_time_distribution"), \
        "the report must document the lap-time distribution too"


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_curriculum_config_sanity,
    test_t2_domain_randomization_every_reset,
    test_t3_stage_machinery_and_logging,
    test_t4_checkpoint_resume_cleanly,
    test_t5_env_swap_continuity,
    test_t6_evaluators_toy_scale,
    test_t6b_action_adapter,
    test_t7_generalist_drives,
    test_t8_documented_comparison,
]

if __name__ == "__main__":
    print("=" * 58)
    print("TRAINING LOOP + CURRICULUM — LAYER 6 VALIDATION")
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
