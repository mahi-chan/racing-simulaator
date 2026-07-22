#!/usr/bin/env python3
"""
Fetch F1 telemetry data via FastF1 and save as a reference bundle.

Usage:
    python scripts/fetch_telemetry.py --year 2024 --gp Silverstone --session Q

Outputs data to data/telemetry_reference/ including:
    - fastest_lap_telemetry.csv  (car telemetry: speed, throttle, brake, RPM, gear, DRS)
    - fastest_lap_position.csv   (X, Y, Z position data)
    - lap_info.json              (metadata: driver, lap time, compound, sector times, etc.)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import fastf1


def fetch_telemetry(year: int, gp: str, session: str, driver: Optional[str] = None):
    """Fetch telemetry from FastF1 and save reference bundle."""

    # Set up cache
    cache_dir = Path("data/fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    # Output directory
    out_dir = Path("data/telemetry_reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {year} {gp} {session}...")
    sess = fastf1.get_session(year, gp, session)
    sess.load()

    # Get the fastest lap (optionally for a specific driver)
    if driver:
        laps = sess.laps.pick_drivers(driver)
    else:
        laps = sess.laps

    fastest = laps.pick_fastest()
    if fastest is None:
        print("ERROR: No valid fastest lap found.")
        sys.exit(1)

    driver_code = fastest["Driver"]
    lap_time = fastest["LapTime"]
    compound = fastest.get("Compound", "Unknown")
    lap_number = fastest.get("LapNumber", "?")

    print(f"Fastest lap: {driver_code} — {lap_time} (Lap {lap_number}, {compound})")

    # --- Car telemetry (speed, throttle, brake, RPM, gear, DRS) ---
    telemetry = fastest.get_car_data().add_distance()
    tel_path = out_dir / "fastest_lap_telemetry.csv"
    telemetry.to_csv(tel_path, index=False)
    print(f"  Saved telemetry → {tel_path}  ({len(telemetry)} samples)")

    # --- Position data (X, Y, Z) ---
    pos = fastest.get_pos_data()
    pos_path = out_dir / "fastest_lap_position.csv"
    pos.to_csv(pos_path, index=False)
    print(f"  Saved position  → {pos_path}  ({len(pos)} samples)")

    # --- Lap metadata ---
    lap_info = {
        "year": year,
        "grand_prix": gp,
        "session": session,
        "driver": driver_code,
        "lap_number": int(lap_number) if lap_number != "?" else None,
        "lap_time_s": lap_time.total_seconds() if hasattr(lap_time, "total_seconds") else str(lap_time),
        "compound": str(compound),
        "sector1": fastest.get("Sector1Time").total_seconds() if fastest.get("Sector1Time") is not None and hasattr(fastest.get("Sector1Time"), "total_seconds") else None,
        "sector2": fastest.get("Sector2Time").total_seconds() if fastest.get("Sector2Time") is not None and hasattr(fastest.get("Sector2Time"), "total_seconds") else None,
        "sector3": fastest.get("Sector3Time").total_seconds() if fastest.get("Sector3Time") is not None and hasattr(fastest.get("Sector3Time"), "total_seconds") else None,
    }
    info_path = out_dir / "lap_info.json"
    with open(info_path, "w") as f:
        json.dump(lap_info, f, indent=2)
    print(f"  Saved metadata  → {info_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Fetch F1 telemetry reference data")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g. 2024)")
    parser.add_argument("--gp", type=str, required=True, help="Grand Prix name (e.g. Silverstone)")
    parser.add_argument("--session", type=str, required=True, help="Session type: FP1, FP2, FP3, Q, R")
    parser.add_argument("--driver", type=str, default=None, help="Optional driver code (e.g. VER, HAM)")
    args = parser.parse_args()

    fetch_telemetry(args.year, args.gp, args.session, args.driver)


if __name__ == "__main__":
    main()
