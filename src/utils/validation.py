"""
Layer 7 — telemetry reference bundles, sim-vs-real metrics, calibration fit.

The spec's `src/utils/validation.py`: everything needed to (a) carry real
FastF1 telemetry into the offline sandbox as a small versioned bundle,
(b) compare a QSS sim lap against a real reference lap (lap-time error,
speed-vs-distance RMSE, apex speeds, braking points), and (c) fit the five
calibration scales and read/write the calibrated-parameter artifact.

Design notes:
  * Traces are compared on a common FRACTIONAL-distance grid: GPS lap length
    and the smoothing spline's arc length disagree by O(1%), so comparing on
    absolute meters would drift progressively around the lap. Speeds stay in
    SI; only the abscissa is normalized. A circular shift (start-line offset)
    is estimated once by RMSE scan and reported.
  * Calibrated parameters ship as an OPT-IN artifact (JSON with provenance),
    never as mutated CarSpec defaults — Layers 1-6 validations and trained
    drivers stay bit-for-bit valid. `load_calibrated_spec` applies the scales
    to a base CarSpec on request.
  * The fit itself (coarse grid + Nelder-Mead polish over 5 scale factors)
    lives here so tests can exercise it offline; `scripts/calibrate.py` is a
    thin CLI around these functions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.physics.qss_lap import braking_points, find_apexes, qss_lap
from src.physics.vehicle_model import CarSpec
from src.tracks.track import Track, TrackConfig

SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1

# Stated Layer 7 acceptance thresholds (LAYER_SPECS: lap time "within ~2-3%",
# RMSE "under a stated threshold", hold-out honesty check). Provisional per
# house rules: a miss is a stop-and-ask, never a silent edit — in either
# direction.
THRESHOLDS = {
    "fit_max_err_pct": 3.0,        # |lap-time error| on the fitted lap
    "fit_max_rmse_kmh": 10.0,      # speed-trace RMSE on the fitted lap
    "holdout_max_err_pct": 4.0,    # |lap-time error| on the held-out lap
    "holdout_max_rmse_kmh": 12.0,  # speed-trace RMSE on the held-out lap
}

# scale name -> CarSpec fields it multiplies (the spec's "tire grip, aero,
# drivetrain" small parameter set)
CAL_FIELDS: dict[str, tuple[str, ...]] = {
    "mu_scale": ("mu_x", "mu_y"),
    "cla_scale": ("cla",),
    "cda_scale": ("cda",),
    "power_scale": ("max_power",),
    "brake_scale": ("max_brake_force",),
}
SCALE_BOUNDS = (0.7, 1.6)   # sanity envelope for every scale factor


# -----------------------------------------------------------------------------
# reference bundle (written online by scripts/fetch_telemetry.py, read offline)
# -----------------------------------------------------------------------------
@dataclass
class RefLap:
    """One real reference lap: distance-indexed SI channels."""

    label: str                 # e.g. "RUS 1:25.819 (2024 British GP Q)"
    lap_time: float            # s
    dist: np.ndarray           # m from the lap's start line, increasing
    speed: np.ndarray          # m/s
    throttle: np.ndarray       # 0..1
    brake: np.ndarray          # 0..1
    gear: np.ndarray           # 1..8


@dataclass
class ReferenceBundle:
    track_points: np.ndarray   # (N, 2) XY meters (fastest lap's GPS trace)
    laps: list[RefLap]
    meta: dict


def write_reference_bundle(path: str | Path, bundle: ReferenceBundle) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "version": np.int64(SCHEMA_VERSION),
        "meta": np.str_(json.dumps(bundle.meta)),
        "track_points": np.asarray(bundle.track_points, dtype=np.float64),
        "n_laps": np.int64(len(bundle.laps)),
    }
    for i, lap in enumerate(bundle.laps):
        payload[f"lap{i}_label"] = np.str_(lap.label)
        payload[f"lap{i}_time"] = np.float64(lap.lap_time)
        for ch in ("dist", "speed", "throttle", "brake", "gear"):
            payload[f"lap{i}_{ch}"] = np.asarray(getattr(lap, ch),
                                                 dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_reference_bundle(path: str | Path) -> ReferenceBundle:
    with np.load(Path(path), allow_pickle=False) as z:
        version = int(z["version"])
        if version != SCHEMA_VERSION:
            raise ValueError(f"bundle schema v{version}, expected "
                             f"v{SCHEMA_VERSION}: {path}")
        meta = json.loads(str(z["meta"]))
        laps = [RefLap(label=str(z[f"lap{i}_label"]),
                       lap_time=float(z[f"lap{i}_time"]),
                       **{ch: z[f"lap{i}_{ch}"]
                          for ch in ("dist", "speed", "throttle",
                                     "brake", "gear")})
                for i in range(int(z["n_laps"]))]
        return ReferenceBundle(track_points=z["track_points"], laps=laps,
                               meta=meta)


def load_reference_track(bundle: ReferenceBundle,
                         config: TrackConfig | None = None) -> Track:
    """Rebuild the real track offline from the bundled GPS centerline."""
    cfg = config or TrackConfig()
    src = bundle.meta.get("source", "reference")
    return Track(bundle.track_points, smoothing_per_point=cfg.fastf1_smoothing,
                 config=cfg, source=f"reference:{src}")


def bundle_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -----------------------------------------------------------------------------
# trace comparison
# -----------------------------------------------------------------------------
def to_fraction_grid(dist: np.ndarray, values: np.ndarray,
                     n_grid: int = 1500) -> np.ndarray:
    """Resample a distance-indexed channel onto a uniform lap-fraction grid."""
    dist = np.asarray(dist, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(dist) != len(values) or len(dist) < 8:
        raise ValueError("dist/values must be equal-length arrays (>= 8)")
    # enforce a strictly increasing abscissa (GPS hiccups produce repeats)
    d = np.maximum.accumulate(dist - dist[0])
    keep = np.concatenate([[True], np.diff(d) > 0])
    f = d[keep] / d[keep][-1]
    grid = np.arange(n_grid) / n_grid
    return np.interp(grid, f, values[keep], period=1.0)


def align_offset(ref_on_grid: np.ndarray, sim_on_grid: np.ndarray, *,
                 max_shift: int | None = None) -> int:
    """Circular shift (grid points) of the SIM trace minimizing speed RMSE.

    The two traces start at different points of the loop (spline seam vs the
    timing line). With `max_shift=None` every shift is scanned (reporting
    path). The fit passes a small window instead: for a real bundle the
    track spline and the reference lap start from the SAME GPS trace, so the
    true offset is near zero by construction — windowing both saves time and
    stops a badly-mismatched candidate car from "aligning" onto the wrong
    part of the lap (observed: a stale full-scan shift estimated from the
    uncalibrated base car poisoned the self-recovery fit).
    """
    if ref_on_grid.shape != sim_on_grid.shape:
        raise ValueError("traces must share one grid")
    n = len(sim_on_grid)
    shifts = (range(n) if max_shift is None
              else range(-max_shift, max_shift + 1))
    best_shift, best_cost = 0, np.inf
    for shift in shifts:
        cost = float(np.mean(np.square(np.roll(sim_on_grid, shift)
                                       - ref_on_grid)))
        if cost < best_cost:
            best_cost, best_shift = cost, shift
    return best_shift


def speed_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two speed traces on a shared grid (m/s)."""
    return float(np.sqrt(np.mean(np.square(np.asarray(a) - np.asarray(b)))))


def lap_time_error(sim_time: float, ref_time: float) -> float:
    """Signed relative lap-time error (sim - ref) / ref."""
    return (sim_time - ref_time) / ref_time


@dataclass
class TraceComparison:
    """Everything the spec asks to compare, for one sim-vs-reference lap."""

    ref_label: str
    ref_lap_time: float
    sim_lap_time: float
    lap_time_err: float          # signed fraction
    rmse: float                  # m/s on the aligned fraction grid
    shift: int                   # applied circular shift (grid points)
    grid: np.ndarray             # lap fraction in [0, 1)
    ref_v: np.ndarray            # m/s on grid
    sim_v: np.ndarray            # m/s on grid, aligned
    apex_rows: list = field(default_factory=list)
    # apex_rows: dicts with ref_s_frac, ref_v, sim_v, dv, ref_brake_frac,
    #            sim_brake_frac, brake_delta_m (NaN-tolerant)

    @property
    def rmse_kmh(self) -> float:
        return self.rmse * 3.6

    @property
    def lap_time_err_pct(self) -> float:
        return self.lap_time_err * 100.0


def compare_traces(ref: RefLap, sim_s: np.ndarray, sim_v: np.ndarray,
                   sim_lap_time: float, length: float, n_grid: int = 1500,
                   shift: int | None = None) -> TraceComparison:
    """Compare a sim speed profile against one reference lap.

    `sim_s`/`sim_v` are the QSS profile arrays (uniform s over `length`);
    pass `shift` to reuse a previously estimated start-line offset.
    """
    ref_v = to_fraction_grid(ref.dist, ref.speed, n_grid)
    sim_vg = to_fraction_grid(sim_s, sim_v, n_grid)
    if shift is None:
        shift = align_offset(ref_v, sim_vg)
    sim_al = np.roll(sim_vg, shift)
    grid = np.arange(n_grid) / n_grid

    comp = TraceComparison(
        ref_label=ref.label, ref_lap_time=ref.lap_time,
        sim_lap_time=sim_lap_time,
        lap_time_err=lap_time_error(sim_lap_time, ref.lap_time),
        rmse=speed_rmse(ref_v, sim_al), shift=shift, grid=grid,
        ref_v=ref_v, sim_v=sim_al)

    # per-corner apex + braking-point deltas on the common fraction grid
    ra_s, ra_v = find_apexes(grid, ref_v, 1.0, min_separation=50.0 / length)
    sa_s, sa_v = find_apexes(grid, sim_al, 1.0, min_separation=50.0 / length)
    # fraction-space decel = physical decel x length (ds shrinks by 1/length),
    # so the m/s^2 thresholds scale UP by length; distances scale down.
    rb = braking_points(grid, ref_v, ra_s, 1.0,
                        decel_threshold=2.0 * length,
                        search_back=250.0 / length,
                        accel_abort=1.5 * length)
    sb = braking_points(grid, sim_al, sa_s, 1.0,
                        decel_threshold=2.0 * length,
                        search_back=250.0 / length,
                        accel_abort=1.5 * length)
    for k, (rs, rv) in enumerate(zip(ra_s, ra_v)):
        if len(sa_s):
            d = np.abs(sa_s - rs)
            d = np.minimum(d, 1.0 - d)                 # circular
            jj = int(np.argmin(d))
            if d[jj] <= 60.0 / length:                 # paired within 60 m
                brake_delta = (sb[jj] - rb[k]) * length
                comp.apex_rows.append({
                    "ref_s_m": rs * length, "ref_v_kmh": rv * 3.6,
                    "sim_v_kmh": sa_v[jj] * 3.6,
                    "dv_kmh": (sa_v[jj] - rv) * 3.6,
                    "ref_brake_s_m": rb[k] * length,
                    "sim_brake_s_m": sb[jj] * length,
                    "brake_delta_m": brake_delta})
                continue
        comp.apex_rows.append({"ref_s_m": rs * length,
                               "ref_v_kmh": rv * 3.6, "sim_v_kmh": np.nan,
                               "dv_kmh": np.nan, "ref_brake_s_m": np.nan,
                               "sim_brake_s_m": np.nan,
                               "brake_delta_m": np.nan})
    return comp


def render_comparison(comp: TraceComparison) -> str:
    lines = [
        f"reference : {comp.ref_label}",
        f"lap time  : sim {comp.sim_lap_time:7.3f} s  vs  ref "
        f"{comp.ref_lap_time:7.3f} s   error {comp.lap_time_err_pct:+.2f} %",
        f"speed RMSE: {comp.rmse_kmh:.2f} km/h  (start-line shift "
        f"{comp.shift} grid pts)",
        "corner table (paired apexes):",
        "     s[m]   ref[km/h]  sim[km/h]   dv[km/h]   brake delta[m]",
    ]
    for r in comp.apex_rows:
        sim_v = ("     -   " if np.isnan(r["sim_v_kmh"])
                 else f"{r['sim_v_kmh']:9.1f}")
        dv = "    -  " if np.isnan(r["dv_kmh"]) else f"{r['dv_kmh']:+7.1f}"
        bd = ("    -  " if np.isnan(r["brake_delta_m"])
              else f"{r['brake_delta_m']:+7.1f}")
        lines.append(f"  {r['ref_s_m']:7.0f} {r['ref_v_kmh']:9.1f}  "
                     f"{sim_v}  {dv}    {bd}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# calibrated-parameter artifact
# -----------------------------------------------------------------------------
def apply_scales(spec: CarSpec, scales: dict[str, float]) -> CarSpec:
    """Return a new CarSpec with the calibration scales multiplied in."""
    unknown = set(scales) - set(CAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown calibration scales: {sorted(unknown)}")
    overrides = {}
    for name, value in scales.items():
        for fld in CAL_FIELDS[name]:
            overrides[fld] = getattr(spec, fld) * float(value)
    return dataclasses.replace(spec, **overrides)


def write_calibration(path: str | Path, *, scales: dict[str, float],
                      provenance: dict, metrics: dict,
                      bounds: tuple[float, float] = SCALE_BOUNDS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scales": {k: float(v) for k, v in scales.items()},
        "scale_bounds": list(bounds),
        "scale_fields": {k: list(v) for k, v in CAL_FIELDS.items()},
        "metrics": metrics,
        "provenance": provenance,
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def load_calibrated_spec(path: str | Path,
                         base: CarSpec | None = None) -> tuple[CarSpec, dict]:
    """Apply a calibration artifact to a base CarSpec (defaults untouched)."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"calibration schema {doc.get('schema_version')!r}, "
                         f"expected {CALIBRATION_SCHEMA_VERSION}: {path}")
    lo, hi = doc.get("scale_bounds", SCALE_BOUNDS)
    for k, v in doc["scales"].items():
        if not (lo - 1e-9 <= float(v) <= hi + 1e-9):
            raise ValueError(f"calibrated {k}={v} outside recorded bounds "
                             f"[{lo}, {hi}] — refusing to load")
    return apply_scales(base or CarSpec(), doc["scales"]), doc


# -----------------------------------------------------------------------------
# the fit: coarse grid + Nelder-Mead over the five scales
# -----------------------------------------------------------------------------
@dataclass
class FitResult:
    scales: dict[str, float]           # every CAL_FIELDS key (non-fitted = 1.0)
    fit_names: tuple[str, ...]         # which scales the optimizer moved
    objective_before: float
    objective_after: float
    n_evals: int
    comparison: TraceComparison        # fit lap, with the fitted scales


def _objective(comp_rmse_kmh: float, rel_dt: float,
               x: np.ndarray, bounds: tuple[float, float]) -> float:
    """1 % lap time trades 1:1 against 1 km/h RMSE; hard walls at bounds."""
    j = (100.0 * rel_dt) ** 2 + comp_rmse_kmh ** 2
    lo, hi = bounds
    over = np.maximum(0.0, np.maximum(lo - x, x - hi))
    return j + 1e3 * float(np.sum(np.square(over)))


DEFAULT_FIT_NAMES = ("mu_scale", "cla_scale", "cda_scale", "brake_scale")


def fit_scales(track, base_spec: CarSpec, grip_multiplier: float,
               ref: RefLap, *, drs: bool = True, n_grid: int = 1500,
               bounds: tuple[float, float] = SCALE_BOUNDS,
               grid_values: tuple[float, ...] = (0.85, 1.0, 1.25),
               nm_maxiter: int = 400,
               fit_names: tuple[str, ...] = DEFAULT_FIT_NAMES) -> FitResult:
    """Fit the calibration scales so QSS matches the reference lap.

    `power_scale` is NOT fitted by default: the noiseless self-recovery test
    exposed a cda/power ridge — a speed trace only pins the P/cda combination
    (top speed goes as (P/cda)^(1/3)), so fitting both lets errors alias
    between them. Engine power is the regulation-grade known quantity
    (fuel-flow-limited ICE + 120 kW MGU-K, CarSpec's documented 735 kW) while
    grip/aero are the declared placeholder calibration targets, so power
    stays fixed at 1.0 and the choice is recorded in provenance. Pass a
    custom `fit_names` to revisit (a stop-and-ask lever, per house rules).
    """
    from itertools import product

    from scipy.optimize import minimize

    unknown = set(fit_names) - set(CAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown fit scales: {sorted(unknown)}")
    names = list(fit_names)
    ref_v = to_fraction_grid(ref.dist, ref.speed, n_grid)

    # start-line offset: re-aligned per evaluation over a small window
    # (track spline and reference share a start by construction; a stale
    # shift estimated from one candidate car misaligns every other one)
    max_shift = max(int(150.0 / track.length * n_grid), 2)
    n_evals = 0

    def evaluate(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        spec = apply_scales(base_spec,
                            dict(zip(names, np.clip(x, *bounds))))
        prof = qss_lap(track, spec, grip_multiplier=grip_multiplier,
                       drs=drs, landmarks=False)
        sim = to_fraction_grid(prof.s, prof.v, n_grid)
        shift = align_offset(ref_v, sim, max_shift=max_shift)
        rmse_kmh = speed_rmse(ref_v, np.roll(sim, shift)) * 3.6
        rel_dt = lap_time_error(prof.lap_time, ref.lap_time)
        return _objective(rmse_kmh, rel_dt, x, bounds)

    j0 = evaluate(np.ones(len(names)))

    best_x, best_j = np.ones(len(names)), j0
    for combo in product(grid_values, repeat=len(names)):
        x = np.asarray(combo)
        j = evaluate(x)
        if j < best_j:
            best_j, best_x = j, x

    # Powell, then a short Nelder-Mead polish. Chosen from evidence: on the
    # noiseless self-recovery problem NM (even restarted) collapsed inside
    # the curved mu/cla valley (brake ended 1.38 for a planted 0.85, J~1.2),
    # while Powell's coordinate line searches walk the valley to J=0 and
    # recover every planted scale exactly. The polish is cheap insurance on
    # real, noisy references; the better result wins.
    res = minimize(evaluate, best_x, method="Powell",
                   options={"maxiter": nm_maxiter, "xtol": 1e-5,
                            "ftol": 1e-10})
    res2 = minimize(evaluate, res.x, method="Nelder-Mead",
                    options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-8})
    if res2.fun < res.fun:
        res = res2
    x_fit = np.clip(res.x, *bounds)
    scales = {k: 1.0 for k in CAL_FIELDS}       # non-fitted scales stay 1.0
    scales.update(zip(names, (float(v) for v in x_fit)))

    spec_fit = apply_scales(base_spec, scales)
    prof_fit = qss_lap(track, spec_fit, grip_multiplier=grip_multiplier,
                       drs=drs)
    fit_shift = align_offset(ref_v, to_fraction_grid(prof_fit.s, prof_fit.v,
                                                     n_grid),
                             max_shift=max_shift)
    comp = compare_traces(ref, prof_fit.s, prof_fit.v, prof_fit.lap_time,
                          track.length, n_grid=n_grid, shift=fit_shift)
    return FitResult(scales=scales, fit_names=tuple(fit_names),
                     objective_before=j0, objective_after=float(res.fun),
                     n_evals=n_evals, comparison=comp)
