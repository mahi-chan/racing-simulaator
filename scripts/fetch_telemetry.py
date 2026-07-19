#!/usr/bin/env python
"""Layer 7 — one-time ONLINE pull of the real telemetry reference bundle.

This script needs the F1 data API, which the Claude Code sandbox proxy
blocks (GUIDE.md), so run it ONCE on your own machine or Colab:

    pip install fastf1                      # already in requirements.txt
    python scripts/fetch_telemetry.py --year 2024 --gp Silverstone --session Q
    git add data/telemetry_reference
    git commit -m "Layer 7: telemetry reference bundle"
    git push

It writes a small (~100-300 KB) versioned bundle that everything offline
consumes (`scripts/calibrate.py`, `tests/test_calibration.py` T6-T8):

    data/telemetry_reference/<gp>_<year>_<session>.npz    the bundle
    data/telemetry_reference/<gp>_<year>_<session>.json   human-readable meta

Bundle contents: the session-fastest lap (calibration fit target) and the
second-fastest lap by a DIFFERENT driver (hold-out against overfitting) —
distance-indexed speed/throttle/brake/gear in SI units — plus the fastest
lap's GPS trace (meters) from which the real track is rebuilt offline, and
session weather. Colab tip: run from the repo root so the relative output
path lands inside the repo checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.utils.validation import (RefLap, ReferenceBundle,
                                  write_reference_bundle)

SANDBOX_NOTE = """\
Could not reach the F1 data API ({err}).

If you are inside the Claude Code sandbox: this is expected — the proxy
blocks the F1 hosts. Run this script on your own machine or Colab (see the
module docstring for the exact commands), commit the bundle, and everything
else runs offline."""


def _lap_label(lap, year: int, gp: str, session: str) -> str:
    lt = lap["LapTime"]
    total = lt.total_seconds()
    return (f"{lap['Driver']} {int(total // 60)}:{total % 60:06.3f} "
            f"({year} {gp} {session}, lap {int(lap['LapNumber'])})")


def _lap_channels(lap) -> dict[str, np.ndarray]:
    tel = lap.get_telemetry()          # merged car + pos data, adds Distance
    dist = tel["Distance"].to_numpy(dtype=float)
    return {
        "dist": dist - dist[0],
        "speed": tel["Speed"].to_numpy(dtype=float) / 3.6,     # km/h -> m/s
        "throttle": tel["Throttle"].to_numpy(dtype=float) / 100.0,
        "brake": tel["Brake"].to_numpy(dtype=float),
        "gear": tel["nGear"].to_numpy(dtype=float),
    }


def _track_points_m(lap) -> np.ndarray:
    """Fastest lap's GPS trace in meters, cleaned like Track.from_fastf1."""
    pos = lap.get_pos_data()
    xy = np.column_stack([pos["X"].to_numpy(dtype=float),
                          pos["Y"].to_numpy(dtype=float)]) * 0.1  # dm -> m
    xy = xy[~np.isnan(xy).any(axis=1)]
    xy = xy[~np.all(xy == 0.0, axis=1)]
    if len(xy) < 8:
        raise ValueError(f"degenerate GPS trace: {len(xy)} usable points")
    return xy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default="Silverstone")
    ap.add_argument("--session", default="Q")
    ap.add_argument("--out", default="data/telemetry_reference")
    ap.add_argument("--cache", default="data/fastf1_cache")
    args = ap.parse_args(argv)

    try:
        import fastf1
    except ImportError:
        print("fastf1 is not installed here — `pip install fastf1` first "
              "(it is in requirements.txt).")
        return 3

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)

    try:
        ses = fastf1.get_session(args.year, args.gp, args.session)
        ses.load(laps=True, telemetry=True, weather=True, messages=False)
    except Exception as err:  # network/proxy/API failures get the runbook
        print(SANDBOX_NOTE.format(err=err))
        return 3

    laps = ses.laps
    fastest = laps.pick_fastest()
    if fastest is None:
        print(f"no timed laps in {args.year} {args.gp} {args.session}")
        return 4
    others = laps[(laps["Driver"] != fastest["Driver"])
                  & laps["LapTime"].notna()]
    holdout = others.pick_fastest() if len(others) else None

    chosen = [fastest] + ([holdout] if holdout is not None else [])
    ref_laps = [RefLap(label=_lap_label(lap, args.year, args.gp, args.session),
                       lap_time=lap["LapTime"].total_seconds(),
                       **_lap_channels(lap))
                for lap in chosen]

    weather = {}
    try:
        wd = ses.weather_data
        weather = {"air_temp": float(wd["AirTemp"].mean()),
                   "track_temp": float(wd["TrackTemp"].mean())}
    except Exception as err:
        print(f"note: weather unavailable ({err}); calibrate.py will use "
              "its default track temp")

    meta = {
        "schema_version": 1,
        "source": f"fastf1:{args.year}-{args.gp}-{args.session}",
        "year": args.year, "gp": args.gp, "session": args.session,
        "laps": [{"label": rl.label, "lap_time_s": rl.lap_time,
                  "n_samples": len(rl.dist)} for rl in ref_laps],
        "weather": weather,
        "fastf1_version": fastf1.__version__,
        "pulled_utc": datetime.now(timezone.utc).isoformat(),
        "units": {"dist": "m", "speed": "m/s", "throttle": "0..1",
                  "brake": "0..1", "gear": "1..8",
                  "track_points": "m (FastF1 decimeters / 10)"},
    }

    bundle = ReferenceBundle(track_points=_track_points_m(fastest),
                             laps=ref_laps, meta=meta)
    stem = f"{args.gp.lower()}_{args.year}_{args.session}"
    out_npz = Path(args.out) / f"{stem}.npz"
    write_reference_bundle(out_npz, bundle)
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"bundle  : {out_npz}  ({out_npz.stat().st_size / 1024:.0f} KB)")
    print(f"meta    : {out_json}")
    for rl in ref_laps:
        print(f"lap     : {rl.label}  ({len(rl.dist)} samples)")
    if holdout is None:
        print("WARNING : only one usable lap — no hold-out; T7 will skip")
    print()
    print("Next: commit the bundle and re-run calibration in the sandbox:")
    print("    git add data/telemetry_reference")
    print('    git commit -m "Layer 7: telemetry reference bundle"')
    print("    git push")
    print("    python scripts/calibrate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
