"""Layer 2 acceptance tests — track environment (LAYER_SPECS.md § Layer 2).

Runs two ways:
  * `pytest tests/test_track.py`        — standard test run
  * `python tests/test_track.py`        — standalone PASS/FAIL report (Layer 1 style)

T1-T7 use the synthetic track only and always run (offline). T8 reconstructs
Silverstone via FastF1 and is gated behind availability: it SKIPS — never
fails — when fastf1 is missing or the F1 API is unreachable with a cold cache.
"""
import sys
import time
from pathlib import Path
from unittest import SkipTest  # honored as a skip by pytest and by main() below

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
from src.tracks.track import Track, TrackConfig

# Build the synthetic track once; every offline test queries the same instance.
_SYN = None


def synthetic() -> Track:
    global _SYN
    if _SYN is None:
        _SYN = Track.from_synthetic()
    return _SYN


def _ang_diff(a, b) -> float:
    """Smallest signed angular difference a-b (rad)."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_ds(ds: float, length: float) -> float:
    """Arc-length difference accounting for the start/finish wrap."""
    return min(abs(ds), length - abs(ds))


def _left_normal(track: Track, s: float) -> np.ndarray:
    """Unit normal pointing LEFT of travel at s — the +lateral direction.

    Deliberately rebuilt from heading_at here (not from track internals), so
    the tests probe the documented sign convention from first principles.
    """
    h = track.heading_at(s)
    return np.array([-np.sin(h), np.cos(h)])


# ---------------------------------------------------------------------------
# T1 — closed loop, plausible length, monotonic s          [spec bullets 1]
# ---------------------------------------------------------------------------
def test_t1_closed_loop_and_length():
    t = synthetic()
    # spec's literal check — NOTE: the periodic fit forces the resampled
    # endpoints to coincide, so the falsifiable closure evidence is below
    gap = float(np.hypot(*(t.centerline[0] - t.centerline[-1])))
    assert gap < 1.0, f"loop does not close: start/end gap {gap:.3f} m"
    # falsifiable closure #1: the RAW input loop closed on its own, BEFORE the
    # periodic spline was allowed to bridge any gap
    max_raw_gap = 2.0 * TrackConfig().resample_spacing
    assert t.raw_closure_gap < max_raw_gap, \
        f"raw input loop has a {t.raw_closure_gap:.1f} m start/finish gap"
    # falsifiable closure #2: heading winds exactly once (+/-2 pi) around the
    # loop — an open or self-crossing curve cannot satisfy this
    seg = np.diff(t.centerline, axis=0)
    winding = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    assert abs(abs(winding[-1] - winding[0]) - 2.0 * np.pi) < np.radians(1.0), \
        f"heading winds {np.degrees(winding[-1] - winding[0]):.1f} deg, not 360"
    assert t.length > 0.0
    assert 4000.0 <= t.length <= 6000.0, f"synthetic length {t.length:.0f} m not ~5 km"
    ds = np.diff(t.s)
    assert np.all(ds > 0.0), "s is not strictly monotonic"
    assert t.s[0] == 0.0 and abs(t.s[-1] - t.length) < 1e-6
    print(f"    synthetic: length {t.length:.1f} m, {len(t.centerline)} points, "
          f"raw closure gap {t.raw_closure_gap:.2f} m")


# ---------------------------------------------------------------------------
# T2 — nearest_point / heading_at self-consistency          [spec bullet 2]
# ---------------------------------------------------------------------------
def test_t2_nearest_point_self_consistency():
    t = synthetic()
    n = len(t.centerline)
    for i in np.linspace(0, n - 2, 48, dtype=int):  # includes wrap region
        s_true = float(t.s[i])
        normal = _left_normal(t, s_true)
        for d in (-3.0, 0.0, 3.0):
            p = t.centerline[i] + d * normal
            s_hat, lat = t.nearest_point(p[0], p[1])
            ds = _wrap_ds(s_hat - s_true, t.length)
            assert ds < 3.0, f"s not recovered at s={s_true:.1f}, d={d}: off by {ds:.2f} m"
            assert abs(lat - d) < 0.1, \
                f"lateral offset wrong at s={s_true:.1f}: expected {d}, got {lat:.3f}"


# ---------------------------------------------------------------------------
# T3 — heading correctness and wrap continuity               [spec bullet 2]
# ---------------------------------------------------------------------------
def test_t3_heading():
    t = synthetic()
    c, n = t.centerline, len(t.centerline)
    # heading matches the central-difference direction of the centerline
    for i in np.linspace(1, n - 2, 60, dtype=int):
        chord = c[i + 1] - c[i - 1]
        h_chord = np.arctan2(chord[1], chord[0])
        err = abs(_ang_diff(t.heading_at(float(t.s[i])), h_chord))
        assert err < np.radians(2.0), \
            f"heading off by {np.degrees(err):.2f} deg at s={t.s[i]:.1f}"
    # seam continuity, crossing DIFFERENT interpolation cells (heading_at's
    # internal modulo makes heading_at(s + length) == heading_at(s) by
    # arithmetic, so that comparison proves nothing): heading just before the
    # finish line must match heading just after it, up to the rotation the
    # local curvature produces over the 2*eps span
    eps = 0.5
    kappa_seam = max(abs(float(t.curvature_at(t.length - eps))),
                     abs(float(t.curvature_at(eps))))
    tol = 2.0 * eps * kappa_seam + np.radians(0.1)
    seam_err = abs(_ang_diff(t.heading_at(t.length - eps), t.heading_at(eps)))
    assert seam_err < tol, \
        f"heading kink at the seam: {np.degrees(seam_err):.3f} deg"
    # wrap arithmetic: negative and beyond-length s map into [0, length)
    assert abs(_ang_diff(t.heading_at(-1.0), t.heading_at(t.length - 1.0))) < 1e-9
    assert abs(_ang_diff(t.heading_at(t.length + 7.0), t.heading_at(7.0))) < 1e-9


# ---------------------------------------------------------------------------
# T4 — is_on_track inside/outside the width                  [spec bullet 3]
# ---------------------------------------------------------------------------
def test_t4_is_on_track():
    t = synthetic()
    n = len(t.centerline)
    margin = 1.0  # m beyond the half-width must be off-track
    for i in np.linspace(0, n - 2, 40, dtype=int):
        s = float(t.s[i])
        normal = _left_normal(t, s)
        half_w = float(t.width_at(s)) / 2.0
        cx, cy = t.centerline[i]
        assert t.is_on_track(cx, cy), f"centerline point off-track at s={s:.1f}"
        for sign in (+1.0, -1.0):
            inside = t.centerline[i] + sign * 0.5 * half_w * normal
            assert t.is_on_track(*inside), f"point at 25% width off-track at s={s:.1f}"
            outside = t.centerline[i] + sign * (half_w + margin) * normal
            assert not t.is_on_track(*outside), \
                f"point {margin} m beyond half-width on-track at s={s:.1f}"


# ---------------------------------------------------------------------------
# T5 — curvature matches the designed geometry (ground truth)
# ---------------------------------------------------------------------------
def test_t5_curvature_ground_truth():
    t = synthetic()
    kappa = t.curvature
    assert np.all(np.isfinite(kappa)), "curvature has NaN/inf"
    assert len(kappa) == len(t.centerline)
    spacing = t.length / (len(t.centerline) - 1)
    design = t.design_corners
    assert design, "synthetic track carries no design metadata"
    # in each designed arc's midpoint: |kappa| ~ 1/R with the designed sign
    for c in design:
        s_mid = 0.5 * (c["s_start"] + c["s_end"])
        i = int(round(s_mid / spacing))
        k_here = float(kappa[i])
        expect = c["direction"] / c["radius"]
        assert abs(k_here - expect) < 0.10 * abs(expect), \
            (f"arc R={c['radius']:.0f} at s~{s_mid:.0f}: curvature {k_here:.5f}, "
             f"designed {expect:.5f}")
    # in the middle of each designed straight: |kappa| ~ 0
    for a, b in zip(design, design[1:] + [design[0]]):
        gap_start, gap_end = a["s_end"], b["s_start"]
        if gap_end < gap_start:  # wraps past s = 0
            gap_end += t.length
        s_mid = 0.5 * (gap_start + gap_end) % t.length
        i = int(round(s_mid / spacing))
        assert abs(float(kappa[i])) < 0.002, \
            f"straight at s~{s_mid:.0f} has curvature {kappa[i]:.5f}"


# ---------------------------------------------------------------------------
# T6 — corner detection recovers the designed corners        [spec: corners]
# ---------------------------------------------------------------------------
def test_t6_corner_detection():
    t = synthetic()
    detected = t.corners
    design = t.design_corners
    assert len(detected) == len(design), \
        f"detected {len(detected)} corners, designed {len(design)}: {detected}"
    tol = 15.0  # m slack on the designed arc span
    used = set()
    for s_c in detected:
        hit = None
        for j, c in enumerate(design):
            if j not in used and c["s_start"] - tol <= s_c <= c["s_end"] + tol:
                hit = j
                break
        assert hit is not None, f"detected corner at s={s_c:.0f} matches no designed arc"
        used.add(hit)
    print(f"    synthetic corners detected at s = "
          f"{', '.join(f'{s:.0f}' for s in detected)} (all {len(design)} designed)")


# ---------------------------------------------------------------------------
# T7 — query throughput smoke test (env needs 1000s of steps/s)
# ---------------------------------------------------------------------------
def test_t7_query_performance():
    t = synthetic()
    rng = np.random.default_rng(0)
    n_q = 5000
    idx = rng.integers(0, len(t.centerline) - 1, n_q)
    lat = rng.uniform(-8.0, 8.0, n_q)
    normals = np.array([_left_normal(t, float(t.s[i])) for i in idx])
    pts = t.centerline[idx] + lat[:, None] * normals
    t0 = time.perf_counter()
    for x, y in pts:
        t.nearest_point(float(x), float(y))
    rate = n_q / (time.perf_counter() - t0)
    assert rate > 5000.0, f"nearest_point too slow: {rate:.0f} queries/s"
    print(f"    nearest_point: {rate:,.0f} queries/s")


# ---------------------------------------------------------------------------
# T8 — Silverstone from FastF1 (gated: skips offline)        [spec bullet 4]
# ---------------------------------------------------------------------------
SILVERSTONE_LENGTH = 5891.0  # m, official


def _fastf1_reachable() -> bool:
    try:
        import requests
        requests.head("https://livetiming.formula1.com/static/", timeout=5)
        return True
    except Exception:
        return False


def test_t8_silverstone_reconstruction():
    try:
        import fastf1  # noqa: F401
    except ImportError:
        raise SkipTest("fastf1 not installed")
    cache = Path(__file__).resolve().parents[1] / "data" / "fastf1_cache"
    warm_cache = cache.exists() and any(cache.iterdir())
    if not warm_cache and not _fastf1_reachable():
        raise SkipTest("F1 API unreachable and no warm cache — run online (see GUIDE.md)")
    try:
        t = Track.from_fastf1(year=2023, gp="Silverstone", session="Q", driver="VER")
    except Exception as e:
        # spec: never fail the suite for lack of network/data — but ONLY
        # availability-shaped errors may skip. Contract bugs in our fastf1
        # usage (AttributeError, KeyError, ...) must FAIL when run online.
        import requests
        availability = (ImportError, OSError, ValueError, requests.RequestException)
        if isinstance(e, availability) or type(e).__module__.startswith("fastf1"):
            raise SkipTest(f"FastF1 data unavailable: {type(e).__name__}: {e}")
        raise

    lo, hi = 0.97 * SILVERSTONE_LENGTH, 1.03 * SILVERSTONE_LENGTH
    assert lo <= t.length <= hi, \
        f"Silverstone length {t.length:.0f} m outside {lo:.0f}-{hi:.0f} m"
    n_corners = len(t.corners)
    assert n_corners >= 15, f"only {n_corners} corners detected (Silverstone has 18)"
    # the RAW lap trace must nearly close on its own (a flying lap's GPS
    # start/finish gap is one sample, ~20 m at speed) — the fitted spline's
    # endpoint gap is forced to zero by construction and proves nothing
    assert t.raw_closure_gap < 50.0, \
        f"raw lap trace start/finish gap {t.raw_closure_gap:.1f} m"
    assert np.all(np.diff(t.s) > 0.0)
    print(f"    Silverstone: length {t.length:.1f} m "
          f"(official {SILVERSTONE_LENGTH:.0f}), {n_corners} corners")


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
TESTS = [
    test_t1_closed_loop_and_length,
    test_t2_nearest_point_self_consistency,
    test_t3_heading,
    test_t4_is_on_track,
    test_t5_curvature_ground_truth,
    test_t6_corner_detection,
    test_t7_query_performance,
    test_t8_silverstone_reconstruction,
]

if __name__ == "__main__":
    print("=" * 58)
    print("F1 TRACK ENVIRONMENT — LAYER 2 VALIDATION")
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
