"""
F1 Track Environment — Layer 2 of the autonomous racing stack.

A queryable closed-loop track. Two ways to build one:

  * `Track.from_fastf1(...)` — reconstruct a real circuit from the fastest lap's
    GPS trace in F1 telemetry (via FastF1). Needs network the first time; results
    are cached under `data/fastf1_cache/`. NOTE: FastF1 world coordinates are in
    DECIMETERS (0.1 m) — they are scaled to meters here.
  * `Track.from_synthetic()` — a hand-built ~5 km closed circuit (rounded polygon
    of straights + constant-radius arcs with known ground truth) that needs no
    network, so the full test suite runs offline.

Both constructors feed the same pipeline: fit a periodic smoothing spline to the
raw XY loop (guarantees closure, suppresses GPS noise), resample it at uniform
arc-length spacing, and precompute everything the environment will query per
step (arc length `s`, tangents/headings, signed curvature, width, a KD-tree for
nearest-point lookups). Queries are O(log N), fast enough for a Gym env running
thousands of steps/second.

Conventions (the Layer 4 env relies on these):
  * `s` is arc length along the centerline, in meters, in [0, length]; queries
    taking `s` wrap modulo `length`.
  * Lateral offset is signed: positive = LEFT of the direction of travel.
  * Curvature is signed: positive = turning left (CCW).

The track has no width channel in F1 telemetry, so width is a clearly-labeled
placeholder (constant by default) to be calibrated in Layer 7. numpy + scipy
only; fastf1 is imported lazily inside `from_fastf1` so offline use never
requires it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate, signal
from scipy.spatial import cKDTree

# Repo root (this file lives at src/tracks/track.py) — used to resolve the
# default FastF1 cache dir independently of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Configuration — every constant here is a labeled placeholder until Layer 7
# ----------------------------------------------------------------------------
@dataclass
class TrackConfig:
    """Track reconstruction & query parameters. SI units unless noted."""

    # --- centerline fit / resampling ---
    resample_spacing: float = 2.0     # m between resampled centerline points
    # splprep smoothing budget per input point (m^2). FastF1 GPS noise is ~1 m,
    # so allow ~1 m^2 per point. Synthetic points are exact: the tiny budget
    # only damps spline ringing at arc/straight junctions — anything larger
    # visibly flattens the tightest arcs' curvature.
    fastf1_smoothing: float = 1.0
    synthetic_smoothing: float = 1e-4

    # --- track width ---
    # F1 telemetry has no width channel; constant placeholder (real F1 tracks
    # are ~10-15 m wide). Calibration target for Layer 7.
    default_width: float = 12.0       # m
    # synthetic track only: gentle width variation around default_width so
    # width_at(s) interpolation is genuinely exercised by the offline tests
    synthetic_width_amplitude: float = 2.0  # m
    synthetic_width_periods: int = 3        # oscillations per lap

    # --- corner detection (peaks of |curvature|) ---
    corner_min_curvature: float = 0.005   # 1/m -> radius < 200 m counts as a corner
    corner_min_prominence: float = 0.003  # 1/m; rejects plateau ripple, keeps real peaks
    corner_min_separation: float = 50.0   # m along s between distinct corners

    # --- FastF1 ---
    fastf1_cache_dir: str = "data/fastf1_cache"  # relative paths resolve to repo root


# ----------------------------------------------------------------------------
# The track
# ----------------------------------------------------------------------------
class Track:
    """A closed racing circuit, queryable by position or arc length."""

    def __init__(self, points: np.ndarray, *, smoothing_per_point: float,
                 config: TrackConfig | None = None, width_profile=None,
                 source: str = "custom"):
        """Build a track from a closed loop of raw XY points (N x 2, meters).

        Args:
            points: raw centerline samples tracing the loop once (last point
                need not repeat the first; closure is enforced by the fit).
            smoothing_per_point: splprep residual budget per input point (m^2).
            width_profile: optional callable mapping the (N,) array of arc
                lengths s (whose last entry equals the track length) to track
                widths (m); default is the constant `config.default_width`.
            source: provenance string (e.g. "synthetic", "fastf1:...").
        """
        self.config = config or TrackConfig()
        self.source = source
        self.design_corners = None  # ground-truth corner spans (synthetic only)
        self._build(np.asarray(points, dtype=float), smoothing_per_point,
                    width_profile)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_fastf1(cls, year: int = 2023, gp: str = "Silverstone",
                    session: str = "Q", driver: str = "VER",
                    config: TrackConfig | None = None) -> "Track":
        """Reconstruct a track from the fastest lap's GPS trace via FastF1.

        Caches to `config.fastf1_cache_dir` (needs internet on the first run).
        """
        import fastf1  # lazy: offline/synthetic use must not require fastf1

        cfg = config or TrackConfig()
        cache = Path(cfg.fastf1_cache_dir)
        if not cache.is_absolute():
            cache = _REPO_ROOT / cache
        os.makedirs(cache, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache))

        ses = fastf1.get_session(year, gp, session)
        ses.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = ses.laps
        # fastf1 >= 3.1 uses pick_drivers; fall back for older versions
        picker = getattr(laps, "pick_drivers", None) or laps.pick_driver
        lap = picker(driver).pick_fastest()
        if lap is None:
            raise ValueError(
                f"no fastest lap for driver {driver!r} in {year} {gp} {session}")
        pos = lap.get_pos_data()

        # FastF1 world coordinates are in DECIMETERS -> scale to meters.
        xy = np.column_stack([pos["X"].to_numpy(dtype=float),
                              pos["Y"].to_numpy(dtype=float)]) * 0.1
        xy = xy[~np.isnan(xy).any(axis=1)]
        xy = xy[~np.all(xy == 0.0, axis=1)]  # (0,0) rows = missing GPS fixes
        if len(xy) < 8:
            raise ValueError(
                f"degenerate position trace for {driver!r}: "
                f"{len(xy)} usable GPS points")

        return cls(xy, smoothing_per_point=cfg.fastf1_smoothing, config=cfg,
                   source=f"fastf1:{year}-{gp}-{session}-{driver}")

    @classmethod
    def from_synthetic(cls, config: TrackConfig | None = None) -> "Track":
        """A hand-built ~5.3 km closed circuit needing no network.

        Rounded polygon: 8 corners (2 right-handers) with radii 30-150 m joined
        by straights, including a 1.54 km main straight. Designed corner spans
        and radii are attached as `design_corners` for ground-truth testing.
        """
        cfg = config or TrackConfig()
        points, corner_meta = _synthetic_layout(cfg.resample_spacing)

        def width_profile(s: np.ndarray) -> np.ndarray:
            # synthetic width placeholder (see TrackConfig); s[-1] == length
            return cfg.default_width + cfg.synthetic_width_amplitude * np.sin(
                2.0 * np.pi * cfg.synthetic_width_periods * s / s[-1])

        track = cls(points, smoothing_per_point=cfg.synthetic_smoothing,
                    config=cfg, width_profile=width_profile, source="synthetic")
        track.design_corners = corner_meta
        return track

    # -- construction pipeline -------------------------------------------------
    def _build(self, points: np.ndarray, smoothing_per_point: float,
               width_profile):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8:
            raise ValueError("points must be an (N>=8) x 2 array of XY meters")

        # 1) drop consecutive (near-)duplicates — splprep rejects zero chords
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.hypot(*np.diff(points, axis=0).T) > 0.01
        pts = points[keep]
        # Closure of the RAW input loop, measured BEFORE the periodic fit
        # enforces closure (the fitted spline's endpoints always coincide by
        # construction, so only this gap says whether the source trace really
        # closed). Exposed for the acceptance tests.
        self.raw_closure_gap = float(np.hypot(*(pts[0] - pts[-1])))
        # close the loop explicitly so splprep(per=1) doesn't warn/adjust
        if self.raw_closure_gap > 0.01:
            pts = np.vstack([pts, pts[0]])
        else:
            pts[-1] = pts[0]

        # 2) periodic cubic smoothing spline through the loop
        m = len(pts)
        tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], per=1, k=3,
                                     s=m * smoothing_per_point)

        # 3) arc-length reparametrization: dense sample -> cumulative length
        chord = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))  # ~ true length
        n_dense = max(4000, int(8 * chord / self.config.resample_spacing))
        u_dense = np.linspace(0.0, 1.0, n_dense)
        xd, yd = interpolate.splev(u_dense, tck)
        seg = np.hypot(np.diff(xd), np.diff(yd))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self._length = float(cum[-1])

        # uniform-s grid INCLUDING the closing point: s[0]=0 and s[-1]=length
        # map to the same location, which makes wrap handling trivial.
        n = max(int(round(self._length / self.config.resample_spacing)), 32) + 1
        self._s = np.linspace(0.0, self._length, n)
        u = np.interp(self._s, cum, u_dense)

        # 4) precompute per-point geometry from analytic spline derivatives
        x, y = interpolate.splev(u, tck)
        dx, dy = interpolate.splev(u, tck, der=1)
        ddx, ddy = interpolate.splev(u, tck, der=2)
        self._centerline = np.column_stack([x, y])
        norm = np.hypot(dx, dy)
        # unwrapped tangent heading per point: heading_at needs one interp
        # (not two), and a simple closed loop must wind by exactly +/-2 pi
        # from first to last entry — a testable closure invariant.
        self._heading_unwrapped = np.unwrap(np.arctan2(dy, dx))
        # signed curvature, parametrization-invariant; positive = left/CCW turn
        self._curvature = (dx * ddy - dy * ddx) / norm ** 3

        self._width = (width_profile(self._s) if width_profile is not None
                       else np.full(n, self.config.default_width))
        self._tree = cKDTree(self._centerline)
        # plain-float copies for the scalar-math hot path in nearest_point
        self._px = self._centerline[:, 0].tolist()
        self._py = self._centerline[:, 1].tolist()
        self._s_list = self._s.tolist()
        self._corners = self._detect_corners()

    def _detect_corners(self) -> list[float]:
        """s-positions of curvature peaks (the corners)."""
        cfg = self.config
        kappa = np.abs(self._curvature[:-1])  # drop duplicated closing point
        # roll so the straightest point sits at index 0 -> no peak is split
        # across the start/finish seam
        shift = int(np.argmin(kappa))
        rolled = np.roll(kappa, -shift)
        n_seg = len(self._s) - 1
        spacing = self._length / n_seg
        peaks, _ = signal.find_peaks(
            rolled,
            height=cfg.corner_min_curvature,
            prominence=cfg.corner_min_prominence,
            distance=max(int(cfg.corner_min_separation / spacing), 1))
        idx = (peaks + shift) % n_seg
        return sorted(float(self._s[i]) for i in idx)

    # -- properties (per spec) --------------------------------------------------
    @property
    def length(self) -> float:
        """Total centerline length (m)."""
        return self._length

    @property
    def centerline(self) -> np.ndarray:
        """(N x 2) resampled centerline, uniform in s; last point == first."""
        return self._centerline

    @property
    def s(self) -> np.ndarray:
        """(N,) arc length of each centerline point, 0 .. length inclusive."""
        return self._s

    @property
    def curvature(self) -> np.ndarray:
        """(N,) signed curvature per centerline point (1/m, positive = left)."""
        return self._curvature

    @property
    def corners(self) -> list[float]:
        """s-positions of detected corners (curvature peaks), ascending."""
        return list(self._corners)

    # -- queries (per spec) -------------------------------------------------------
    def width_at(self, s) -> float | np.ndarray:
        """Track width (m) at arc length s (wraps modulo length)."""
        return np.interp(np.asarray(s) % self._length, self._s, self._width)

    def curvature_at(self, s) -> float | np.ndarray:
        """Signed curvature (1/m) at arc length s (wraps modulo length) —
        wrap-safe look-ahead queries for the Layer 4 observation vector."""
        return np.interp(np.asarray(s) % self._length, self._s, self._curvature)

    def heading_at(self, s) -> float | np.ndarray:
        """Centerline heading (rad) at arc length s, wrapped to [-pi, pi)."""
        h = np.interp(np.asarray(s) % self._length, self._s,
                      self._heading_unwrapped)
        return (h + np.pi) % (2.0 * np.pi) - np.pi

    def nearest_point(self, x: float, y: float) -> tuple[float, float]:
        """Project (x, y) onto the centerline.

        Returns:
            (s, lateral_offset): arc length of the projection in [0, length),
            and the signed lateral distance (m, positive = left of travel).

        Scalar math on purpose: this runs every environment step, and numpy
        dispatch on 2-element arrays costs more than the arithmetic itself.
        Both KD-tree hits are considered so the projection cannot jump to the
        wrong branch where two track sections pass near each other.
        """
        _, idx = self._tree.query((x, y), k=2)
        px, py, s_arr = self._px, self._py, self._s_list
        n_seg = len(px) - 1  # segments 0..n-2; segment i joins points i, i+1
        cands = {(int(idx[0]) - 1) % n_seg, int(idx[0]) % n_seg,
                 (int(idx[1]) - 1) % n_seg, int(idx[1]) % n_seg}
        best_d2, best_s, best_lat = math.inf, 0.0, 0.0
        for a in cands:
            ax, ay = px[a], py[a]
            abx, aby = px[a + 1] - ax, py[a + 1] - ay
            seg2 = abx * abx + aby * aby
            if seg2 == 0.0:
                continue
            t = ((x - ax) * abx + (y - ay) * aby) / seg2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            fx, fy = ax + t * abx, ay + t * aby
            d2 = (x - fx) * (x - fx) + (y - fy) * (y - fy)
            if d2 < best_d2:
                seglen = math.sqrt(seg2)
                best_d2 = d2
                best_s = s_arr[a] + t * seglen
                # signed perpendicular distance: cross(ab, p-A) / |ab|
                best_lat = (abx * (y - ay) - aby * (x - ax)) / seglen
        return best_s % self._length, best_lat

    def is_on_track(self, x: float, y: float) -> bool:
        """True if (x, y) lies within half the local track width."""
        s, lat = self.nearest_point(x, y)
        return bool(abs(lat) <= float(self.width_at(s)) / 2.0)


# ----------------------------------------------------------------------------
# Synthetic circuit layout (offline ground truth)
# ----------------------------------------------------------------------------
# Rounded polygon: vertices of a 1.8 km x 0.7 km rectangle with a notch (two
# right-handers), each corner replaced by a tangent constant-radius arc. Turn
# order starts at vertex B so s = 0 sits at the start of the main straight.
_SYNTH_VERTICES = np.array([
    (1800.0, 0.0),    # B — end of the main straight, fast left-hand sweeper
    (1800.0, 700.0),  # C
    (1150.0, 700.0),  # D — slow corner into the notch
    (1150.0, 400.0),  # E — right-hander (notch)
    (650.0, 400.0),   # F — right-hander (notch exit)
    (650.0, 700.0),   # G
    (0.0, 700.0),     # H
    (0.0, 0.0),       # A — onto the 1.54 km main straight toward B
])
_SYNTH_RADII = np.array([150.0, 60.0, 30.0, 40.0, 70.0, 35.0, 90.0, 110.0])  # m


def _synthetic_layout(spacing: float) -> tuple[np.ndarray, list[dict]]:
    """Sample the synthetic circuit densely; return (points, corner metadata).

    Corner metadata (per arc, in construction order): s_start / s_end along the
    designed polyline, radius (m), and direction (+1 left / -1 right) — exact
    ground truth for curvature and corner-detection tests.
    """
    V, R = _SYNTH_VERTICES, _SYNTH_RADII
    n = len(V)
    arcs = []
    for i in range(n):
        p_prev, p, p_next = V[i - 1], V[i], V[(i + 1) % n]
        d_in = p - p_prev
        d_in = d_in / np.hypot(*d_in)
        d_out = p_next - p
        d_out = d_out / np.hypot(*d_out)
        # signed turn angle at this vertex (positive = left/CCW)
        theta = float(np.arctan2(d_in[0] * d_out[1] - d_in[1] * d_out[0],
                                 d_in @ d_out))
        t_len = R[i] * np.tan(abs(theta) / 2.0)  # tangent-point distance from vertex
        p_in = p - d_in * t_len
        p_out = p + d_out * t_len
        n_in = np.array([-d_in[1], d_in[0]]) * np.sign(theta)  # toward arc center
        center = p_in + n_in * R[i]
        phi0 = float(np.arctan2(p_in[1] - center[1], p_in[0] - center[0]))
        arcs.append((p_in, p_out, center, phi0, theta, float(R[i])))

    pts, meta, s = [], [], 0.0
    for i in range(n):
        # straight from the previous arc's exit to this arc's entry
        a_prev, a_here = arcs[i - 1], arcs[i]
        start, end = a_prev[1], a_here[0]
        seg = end - start
        seglen = float(np.hypot(*seg))
        k = max(int(np.ceil(seglen / spacing)), 2)
        ts = np.linspace(0.0, 1.0, k, endpoint=False)
        pts.append(start + ts[:, None] * seg)
        s += seglen
        # the arc itself (sampled excluding its endpoint; the next straight starts there)
        p_in, p_out, center, phi0, theta, radius = a_here
        arclen = radius * abs(theta)
        k = max(int(np.ceil(arclen / spacing)), 2)
        phis = phi0 + theta * np.linspace(0.0, 1.0, k, endpoint=False)
        pts.append(center + radius * np.column_stack([np.cos(phis), np.sin(phis)]))
        meta.append({"s_start": s, "s_end": s + arclen, "radius": radius,
                     "direction": 1 if theta > 0 else -1})
        s += arclen

    return np.vstack(pts), meta
