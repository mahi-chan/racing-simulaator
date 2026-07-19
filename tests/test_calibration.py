"""Layer 7 acceptance tests — telemetry calibration & validation (LAYER_SPECS § 7).

Runs two ways:
  * `pytest tests/test_calibration.py`   — standard test run
  * `python tests/test_calibration.py`   — standalone PASS/FAIL/SKIP report

T1-T5 validate the machinery fully OFFLINE (QSS profiler sanity, metric
exactness, planted-parameter self-recovery, bundle round-trip, artifact
schema + constant drift guards). T6-T8 are the real-data acceptance gates:
they need `data/telemetry_reference/silverstone_2024_Q.npz` (a one-time pull
via `scripts/fetch_telemetry.py` on a machine with F1-API access — the
sandbox proxy blocks those hosts) and SKIP with the exact runbook until the
bundle is committed; this mirrors Layer 2's online-gated T8.

Stated thresholds (src/utils/validation.py THRESHOLDS): fit lap |dt| <= 3.0 %
and speed RMSE <= 10 km/h; held-out lap |dt| <= 4.0 % and RMSE <= 12 km/h.
House rules: a miss is a stop-and-ask, never a silent edit.
"""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import SkipTest  # honored by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np

from src.physics.conditions import COMPOUNDS, Conditions
from src.physics.qss_lap import (DRS_DOWNFORCE_FACTOR, DRS_DRAG_FACTOR,
                                 DRS_MAX_CURVATURE, VMAX_HARD, braking_points,
                                 find_apexes, qss_lap)
from src.physics.vehicle_model import G, CarSpec
from src.tracks.track import Track
from src.utils.validation import (CAL_FIELDS, SCALE_BOUNDS, THRESHOLDS,
                                  RefLap, ReferenceBundle, align_offset,
                                  apply_scales, compare_traces, fit_scales,
                                  lap_time_error, load_calibrated_spec,
                                  load_reference_bundle, load_reference_track,
                                  render_comparison, speed_rmse,
                                  to_fraction_grid, write_calibration,
                                  write_reference_bundle)

REPO = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO / "data/telemetry_reference/silverstone_2024_Q.npz"
CALIBRATION_PATH = REPO / "data/calibrated/silverstone_2024.json"
FETCH_HINT = ("needs the real-telemetry bundle — run ONCE on a machine with "
              "F1-API access: `pip install fastf1 && python "
              "scripts/fetch_telemetry.py --year 2024 --gp Silverstone "
              "--session Q`, commit data/telemetry_reference/, then re-run")

_TRACK = None
_FIT_CACHE: dict = {}


def synthetic_track() -> Track:
    global _TRACK
    if _TRACK is None:
        _TRACK = Track.from_synthetic()
    return _TRACK


def circular_dist(a: float, b: float, length: float) -> float:
    d = abs(a - b) % length
    return min(d, length - d)


# ---------------------------------------------------------------------------
# T1 — QSS profiler sanity on the synthetic track
# ---------------------------------------------------------------------------
def test_t1_qss_sanity():
    track = synthetic_track()
    spec = CarSpec(fuel_mass=15.0)

    t0 = time.perf_counter()
    p = qss_lap(track, spec)
    wall = time.perf_counter() - t0
    assert wall < 1.0, f"QSS lap took {wall:.2f} s (must stay optimizer-fast)"

    # sanity envelope (decided before the first run; NOT a realism claim —
    # realism is exactly what the real-data calibration T6/T7 measure)
    assert 75.0 <= p.lap_time <= 220.0, f"lap_time {p.lap_time:.1f} s"
    assert np.all(np.isfinite(p.v))
    assert p.v.max() <= VMAX_HARD + 1e-6
    assert p.v.min() > 15.0, f"min speed {p.v.min():.1f} m/s — stall?"
    assert np.all(p.v <= p.v_corner_cap + 1e-9), "profile above lateral cap"

    # every track corner must own an apex within +-30 m
    for c in track.corners:
        d = min(circular_dist(c, a, track.length) for a in p.apex_s)
        assert d <= 30.0, f"corner at s={c:.0f} m has no apex within 30 m"

    a_long = p.a_long
    peak_brake_g = a_long.min() / G
    assert -6.0 <= peak_brake_g <= -3.5, f"peak braking {peak_brake_g:.2f} G"

    # physical directions: DRS helps, less grip hurts
    p_nodrs = qss_lap(track, spec, drs=False, landmarks=False)
    assert p_nodrs.lap_time > p.lap_time
    assert p_nodrs.v.max() < p.v.max()
    p_lowgrip = qss_lap(track, spec, grip_multiplier=0.8, landmarks=False)
    assert p_lowgrip.lap_time > p.lap_time + 1.0

    # deterministic
    p2 = qss_lap(track, spec)
    assert np.array_equal(p.v, p2.v) and p.lap_time == p2.lap_time

    print(f"    [t1] lap {p.lap_time:.2f} s ({track.length / p.lap_time * 3.6:.0f} km/h avg), "
          f"top {p.v.max() * 3.6:.0f} km/h, peak brake {peak_brake_g:.2f} G, "
          f"{len(p.apex_s)} apexes, {wall * 1e3:.0f} ms")


# ---------------------------------------------------------------------------
# T2 — metric exactness on fabricated traces
# ---------------------------------------------------------------------------
def test_t2_metric_units():
    # resampling: a linear ramp survives uneven sampling exactly AWAY from
    # the seam (the resampler is periodic — correct for a flying lap, where
    # speed is continuous around the loop — so a deliberately non-periodic
    # ramp gets blended only at the wrap)
    rng = np.random.default_rng(1)
    d = np.sort(rng.uniform(0, 1000.0, 300))
    d[0], d[-1] = 0.0, 1000.0
    ramp = 5.0 + 0.01 * d
    on_grid = to_fraction_grid(d, ramp, 500)
    expect = 5.0 + 0.01 * (np.arange(500) / 500) * 1000.0
    interior = slice(30, 470)
    err = np.abs(on_grid[interior] - expect[interior]).max()
    assert err < 1e-9, f"interior resample error {err}"
    assert np.all(np.isfinite(on_grid))

    # duplicate distance samples (GPS hiccup) are tolerated
    d2 = np.repeat(np.linspace(0, 100, 50), 2)
    v2 = np.repeat(np.linspace(10, 20, 50), 2)
    assert np.all(np.isfinite(to_fraction_grid(d2, v2, 64)))

    # alignment: a known circular shift is recovered exactly
    base = 50.0 + 10.0 * np.sin(np.linspace(0, 6 * np.pi, 400, endpoint=False))
    for k in (0, 17, 233):
        shifted = np.roll(base, -k)          # simulate a late start line
        assert align_offset(base, shifted) == k

    # rmse / lap-time error are exact
    assert abs(speed_rmse(base, base + 2.5) - 2.5) < 1e-12
    assert abs(lap_time_error(103.0, 100.0) - 0.03) < 1e-15

    # apexes: cosine minima at half-period, plateau counted once
    s = np.arange(0.0, 1000.0, 2.0)
    v = 60.0 + 20.0 * np.cos(2 * np.pi * 2 * s / 1000.0)  # minima at 250, 750
    a_s, a_v = find_apexes(s, v, 1000.0)
    assert len(a_s) == 2
    assert all(min(abs(a - e) for e in (250.0, 750.0)) <= 2.0 for a in a_s)
    v_plat = np.full_like(s, 80.0)
    v_plat[100:105] = 40.0                    # flat-bottom dip
    a_s, _ = find_apexes(s, v_plat, 1000.0)
    assert len(a_s) == 1

    # braking point: designed trapezoid decel zone found at its exact onset
    def vsq_profile(brake_at, apex_at, exit_at, a_brk=6.0, a_acc=3.0,
                    v_top=80.0, n=500, ds=2.0):
        v2p = np.full(n, v_top ** 2)
        srel = np.arange(n) * ds
        for i in range(n):
            x = srel[i]
            if brake_at <= x < apex_at:
                v2p[i] = v_top ** 2 - 2 * a_brk * (x - brake_at)
            elif apex_at <= x < exit_at:
                v_apex2 = v_top ** 2 - 2 * a_brk * (apex_at - brake_at)
                v2p[i] = min(v_apex2 + 2 * a_acc * (x - apex_at), v_top ** 2)
        return srel, np.sqrt(v2p)

    s, v = vsq_profile(400.0, 500.0, 700.0)
    a_s, _ = find_apexes(s, v, 1000.0)
    assert len(a_s) == 1 and abs(a_s[0] - 500.0) <= 2.0
    b = braking_points(s, v, a_s, 1000.0)
    assert abs(b[0] - 400.0) <= 4.0, f"onset {b[0]:.0f}, designed 400"

    # gentle dip below the decel threshold -> flat-out (NaN). The exp tail
    # is truncated to exactly zero: float dust in the "flat" region reads as
    # spurious strict minima otherwise (real traces are handled by the
    # min-separation pruning; fabricated data must just be clean).
    dip = 0.5 * np.exp(-((s - 500.0) ** 2) / (2 * 40.0 ** 2))
    dip[dip < 1e-6] = 0.0
    v_gentle = 80.0 - dip
    a_s, _ = find_apexes(s, v_gentle, 1000.0)
    b = braking_points(s, v_gentle, a_s, 1000.0)
    assert len(b) == 1, f"dip apexes {a_s.tolist()}"
    assert np.isnan(b[0]), f"expected flat-out NaN, got {b[0]}"

    # braking zone crossing the start/finish seam
    s = np.arange(0.0, 1000.0, 2.0)
    unwrapped = (s - 900.0) % 1000.0          # 0 at s=900, apex zone at 100
    v2w = np.full_like(s, 80.0 ** 2)
    zone = unwrapped < 100.0                  # brake from s=900 through 0
    v2w[zone] = 80.0 ** 2 - 2 * 6.0 * unwrapped[zone]
    arc = (unwrapped >= 100.0) & (unwrapped < 160.0)
    v2w[arc] = 80.0 ** 2 - 2 * 6.0 * 100.0
    vw = np.sqrt(v2w)
    a_s, _ = find_apexes(s, vw, 1000.0)
    assert len(a_s) == 1
    b = braking_points(s, vw, a_s, 1000.0)
    assert abs(b[0] - 900.0) <= 4.0, f"wrapped onset {b[0]:.0f}, designed 900"

    # end-to-end self comparison: rolled + resampled sim matches itself
    track = synthetic_track()
    p = qss_lap(track, CarSpec(fuel_mass=15.0))
    roll_m = 400.0
    ref_d = np.linspace(0.0, track.length, 1200, endpoint=False)
    ref_v = np.interp((ref_d + roll_m) % track.length, p.s, p.v)
    ref = RefLap("self", p.lap_time, ref_d, ref_v, np.zeros(1200),
                 np.zeros(1200), np.ones(1200))
    comp = compare_traces(ref, p.s, p.v, p.lap_time, track.length)
    assert abs(comp.lap_time_err) < 1e-12, f"self err {comp.lap_time_err}"
    # tolerance = measured pure-resampling noise (~1 km/h between the 2 m sim
    # grid and a 1200-point reference) with 2x margin; NOT an acceptance bar
    assert comp.rmse_kmh < 2.0, f"self RMSE {comp.rmse_kmh:.2f} km/h"
    # ref samples the sim AHEAD by roll_m, so sim must roll BACKWARD:
    # expected shift is n - roll_m/L*n (circular)
    n = len(comp.grid)
    expected_shift = (n - roll_m / track.length * n) % n
    circ = min(abs(comp.shift % n - expected_shift),
               n - abs(comp.shift % n - expected_shift))
    assert circ <= 2.0, \
        f"shift {comp.shift} vs expected {expected_shift:.1f} (circular)"
    paired = [r for r in comp.apex_rows if not np.isnan(r["dv_kmh"])]
    assert len(paired) >= 10, f"only {len(paired)} apexes paired"
    max_dv = max(abs(r["dv_kmh"]) for r in paired)
    assert max_dv <= 5.0, f"paired apex dv {max_dv:.2f} km/h (resample noise)"
    assert render_comparison(comp)            # renders without blowing up
    print(f"    [t2] alignment/rmse/apex/braking exact; self-comparison "
          f"rmse {comp.rmse_kmh:.2f} km/h, {len(paired)} apexes paired")


# ---------------------------------------------------------------------------
# T3 — the offline machinery proof: planted scales are recovered
# ---------------------------------------------------------------------------
def test_t3_self_recovery():
    # power_scale is deliberately NOT planted or fitted: an earlier version
    # of this test planted all five scales and exposed the cda/power ridge
    # (a speed trace only pins P/cda — cda came back 0.99 for a planted
    # 0.95). fit_scales therefore fixes power at 1.0 by default (engine
    # power is the regulation-known quantity); this test proves the shipped
    # 4-scale set is identifiable.
    track = synthetic_track()
    base = CarSpec(fuel_mass=15.0)
    truth = {"mu_scale": 0.90, "cla_scale": 1.15, "cda_scale": 0.95,
             "brake_scale": 0.85}

    p_true = qss_lap(track, apply_scales(base, truth), landmarks=False)
    ref = RefLap("planted truth", p_true.lap_time, p_true.s.copy(),
                 p_true.v.copy(), np.zeros_like(p_true.v),
                 np.zeros_like(p_true.v), np.ones_like(p_true.v))

    t0 = time.perf_counter()
    fit = fit_scales(track, base, 1.0, ref)
    wall = time.perf_counter() - t0
    assert wall < 180.0, f"fit took {wall:.0f} s (budget 180 s)"

    assert fit.fit_names == ("mu_scale", "cla_scale", "cda_scale",
                             "brake_scale")
    assert fit.scales["power_scale"] == 1.0, "power must stay fixed"
    for k, tv in truth.items():
        got = fit.scales[k]
        assert abs(got - tv) <= 0.03, \
            f"{k}: recovered {got:.4f}, planted {tv} (noiseless!)"
    assert fit.objective_after < fit.objective_before
    assert abs(fit.comparison.lap_time_err_pct) <= 0.3
    assert fit.comparison.rmse_kmh <= 1.0
    rec = {k: round(fit.scales[k], 4) for k in truth}
    print(f"    [t3] recovered {rec} in {fit.n_evals} evals / {wall:.0f} s; "
          f"residual {fit.comparison.rmse_kmh:.3f} km/h, "
          f"{fit.comparison.lap_time_err_pct:+.3f} %")


# ---------------------------------------------------------------------------
# T4 — reference-bundle round trip
# ---------------------------------------------------------------------------
def test_t4_bundle_roundtrip():
    track = synthetic_track()
    pts = track.centerline[::2].copy()
    rng = np.random.default_rng(4)
    laps = []
    for i in range(2):
        d = np.sort(rng.uniform(0, track.length, 700))
        laps.append(RefLap(
            label=f"FAKE{i} 1:2{i}.000 (fabricated)", lap_time=80.0 + i,
            dist=d, speed=55.0 + 15.0 * np.sin(2 * np.pi * d / track.length + i),
            throttle=rng.uniform(0, 1, 700), brake=rng.integers(0, 2, 700).astype(float),
            gear=rng.integers(1, 9, 700).astype(float)))
    meta = {"source": "fabricated:test_t4", "weather": {"track_temp": 33.0},
            "schema_version": 1}
    bundle = ReferenceBundle(track_points=pts, laps=laps, meta=meta)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fake.npz"
        write_reference_bundle(path, bundle)
        assert path.stat().st_size < 1_000_000, "bundle must stay small"
        rt = load_reference_bundle(path)
        assert rt.meta == meta
        assert np.allclose(rt.track_points, pts)
        for a, b in zip(rt.laps, laps):
            assert a.label == b.label and a.lap_time == b.lap_time
            for ch in ("dist", "speed", "throttle", "brake", "gear"):
                assert np.allclose(getattr(a, ch), getattr(b, ch))

        # a wrong schema version is refused loudly
        bad = dict(np.load(path, allow_pickle=False))
        bad["version"] = np.int64(99)
        np.savez_compressed(Path(td) / "bad.npz", **bad)
        try:
            load_reference_bundle(Path(td) / "bad.npz")
            raise AssertionError("schema v99 was accepted")
        except ValueError:
            pass

        # the real track rebuilds from bundled points (same geometry here)
        t2 = load_reference_track(rt)
        assert abs(t2.length - track.length) / track.length < 0.02
        assert len(t2.corners) >= 6
        assert t2.source == "reference:fabricated:test_t4"
    print(f"    [t4] round-trip exact; rebuilt track {t2.length:.0f} m, "
          f"{len(t2.corners)} corners")


# ---------------------------------------------------------------------------
# T5 — calibration artifact schema + cross-module constant drift guards
# ---------------------------------------------------------------------------
def test_t5_artifact_and_drift_guards():
    base = CarSpec(fuel_mass=15.0)
    scales = {"mu_scale": 0.9, "cla_scale": 1.2, "cda_scale": 1.05,
              "power_scale": 1.1, "brake_scale": 0.8}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cal.json"
        write_calibration(path, scales=scales,
                          provenance={"tool": "test_t5", "bundle_sha256": "x"},
                          metrics={"thresholds_met": {"fit_lap_time": True}})
        spec, doc = load_calibrated_spec(path, base=base)
        assert abs(spec.mu_x - base.mu_x * 0.9) < 1e-12
        assert abs(spec.mu_y - base.mu_y * 0.9) < 1e-12
        assert abs(spec.cla - base.cla * 1.2) < 1e-12
        assert abs(spec.cda - base.cda * 1.05) < 1e-12
        assert abs(spec.max_power - base.max_power * 1.1) < 1e-6
        assert abs(spec.max_brake_force - base.max_brake_force * 0.8) < 1e-9
        assert spec.total_mass == base.total_mass  # __post_init__ re-ran
        for key in ("schema_version", "created_utc", "scales", "scale_bounds",
                    "scale_fields", "metrics", "provenance"):
            assert key in doc, f"artifact missing {key!r}"

        # out-of-bounds scales are refused on load (honesty guard)
        write_calibration(Path(td) / "bad.json", scales={"mu_scale": 2.5},
                          provenance={}, metrics={})
        try:
            load_calibrated_spec(Path(td) / "bad.json", base=base)
            raise AssertionError("out-of-bounds scale was accepted")
        except ValueError:
            pass

    # unknown scale names are refused
    try:
        apply_scales(base, {"warp_scale": 1.0})
        raise AssertionError("unknown scale was accepted")
    except ValueError:
        pass

    # thresholds stay ordered (fit stricter than hold-out)
    assert THRESHOLDS["fit_max_err_pct"] <= THRESHOLDS["holdout_max_err_pct"]
    assert THRESHOLDS["fit_max_rmse_kmh"] <= THRESHOLDS["holdout_max_rmse_kmh"]
    assert set(CAL_FIELDS) == {"mu_scale", "cla_scale", "cda_scale",
                               "power_scale", "brake_scale"}
    assert SCALE_BOUNDS[0] < 1.0 < SCALE_BOUNDS[1]

    # qss_lap duplicates Layer 4's DRS constants (numpy-only physics rule):
    # guard against silent drift between the two modules
    from src.envs.f1_env import EnvConfig
    ec = EnvConfig()
    assert DRS_MAX_CURVATURE == ec.drs_max_curvature
    assert DRS_DRAG_FACTOR == ec.drs_drag_factor
    assert DRS_DOWNFORCE_FACTOR == ec.drs_downforce_factor
    print("    [t5] artifact round-trip + bounds guard + DRS drift guard ok")


# ---------------------------------------------------------------------------
# gated real-data helpers
# ---------------------------------------------------------------------------
def _require_bundle():
    if not BUNDLE_PATH.exists():
        raise SkipTest(FETCH_HINT)
    bundle = load_reference_bundle(BUNDLE_PATH)
    track = load_reference_track(bundle)
    return bundle, track


def _pinned_grip(bundle) -> tuple[float, CarSpec]:
    """Same quali pinning as scripts/calibrate.py (soft, in-window, dry)."""
    track_temp = float(bundle.meta.get("weather", {}).get("track_temp", 30.0))
    cond = Conditions(compound="soft", weather="dry", track_temp=track_temp,
                      fuel_mass=15.0)
    lo, hi = COMPOUNDS["soft"].temp_window
    cond.tire_temp = 0.5 * (lo + hi)
    return cond.grip_multiplier(), CarSpec(fuel_mass=15.0)


def _calibrated_comparisons():
    """Fit-lap and hold-out comparisons under the calibrated scales.

    Prefers the committed artifact (verifying it by recomputation); falls
    back to a fresh fit when no artifact exists yet. Cached across T6/T7.
    """
    if "comps" in _FIT_CACHE:
        return _FIT_CACHE["comps"]
    bundle, track = _require_bundle()
    grip, base = _pinned_grip(bundle)

    if CALIBRATION_PATH.exists():
        spec, doc = load_calibrated_spec(CALIBRATION_PATH, base=base)
        origin = f"artifact {CALIBRATION_PATH.name}"
    else:
        fit = fit_scales(track, base, grip, bundle.laps[0])
        spec = apply_scales(base, fit.scales)
        origin = f"fresh fit ({fit.n_evals} evals)"

    prof = qss_lap(track, spec, grip_multiplier=grip)
    comps = [compare_traces(lap, prof.s, prof.v, prof.lap_time, track.length)
             for lap in bundle.laps]
    _FIT_CACHE["comps"] = (comps, origin)
    return _FIT_CACHE["comps"]


# ---------------------------------------------------------------------------
# T6 — real-data calibration acceptance (fit lap)
# ---------------------------------------------------------------------------
def test_t6_real_calibration_acceptance():
    comps, origin = _calibrated_comparisons()
    fit = comps[0]
    print(f"    [t6] scales from {origin}")
    print(render_comparison(fit))
    assert abs(fit.lap_time_err_pct) <= THRESHOLDS["fit_max_err_pct"], \
        (f"fit lap time error {fit.lap_time_err_pct:+.2f} % exceeds "
         f"{THRESHOLDS['fit_max_err_pct']} % — stop and discuss, do not "
         f"weaken (CLAUDE.md)")
    assert fit.rmse_kmh <= THRESHOLDS["fit_max_rmse_kmh"], \
        (f"fit speed RMSE {fit.rmse_kmh:.2f} km/h exceeds "
         f"{THRESHOLDS['fit_max_rmse_kmh']} km/h — stop and discuss")


# ---------------------------------------------------------------------------
# T7 — hold-out honesty check (second lap, never fitted)
# ---------------------------------------------------------------------------
def test_t7_holdout_honesty():
    comps, origin = _calibrated_comparisons()
    if len(comps) < 2:
        raise SkipTest("bundle has no second lap — re-run fetch_telemetry")
    fit, hold = comps[0], comps[1]
    gap = abs(hold.lap_time_err_pct) - abs(fit.lap_time_err_pct)
    print(f"    [t7] hold-out {hold.ref_label}: "
          f"{hold.lap_time_err_pct:+.2f} % / {hold.rmse_kmh:.2f} km/h "
          f"(fit-vs-holdout gap {gap:+.2f} pp)")
    assert abs(hold.lap_time_err_pct) <= THRESHOLDS["holdout_max_err_pct"], \
        (f"hold-out lap time error {hold.lap_time_err_pct:+.2f} % exceeds "
         f"{THRESHOLDS['holdout_max_err_pct']} % — overfit or model-form "
         f"error; stop and discuss")
    assert hold.rmse_kmh <= THRESHOLDS["holdout_max_rmse_kmh"], \
        (f"hold-out RMSE {hold.rmse_kmh:.2f} km/h exceeds "
         f"{THRESHOLDS['holdout_max_rmse_kmh']} km/h — stop and discuss")


# ---------------------------------------------------------------------------
# T8 — real-geometry sanity (Layer 2's online T8, discharged offline)
# ---------------------------------------------------------------------------
def test_t8_real_geometry():
    _, track = _require_bundle()
    assert abs(track.length - 5891.0) / 5891.0 <= 0.03, \
        f"Silverstone length {track.length:.0f} m vs 5891 +-3 %"
    assert len(track.corners) >= 15, \
        f"only {len(track.corners)} corners found (>= 15 expected)"
    assert track.raw_closure_gap < 50.0, \
        f"raw closure gap {track.raw_closure_gap:.1f} m"
    print(f"    [t8] real Silverstone: {track.length:.0f} m, "
          f"{len(track.corners)} corners, closure {track.raw_closure_gap:.1f} m")


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_qss_sanity,
    test_t2_metric_units,
    test_t3_self_recovery,
    test_t4_bundle_roundtrip,
    test_t5_artifact_and_drift_guards,
    test_t6_real_calibration_acceptance,
    test_t7_holdout_honesty,
    test_t8_real_geometry,
]

if __name__ == "__main__":
    print("=" * 58)
    print("F1 TELEMETRY CALIBRATION — LAYER 7 VALIDATION")
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
