#!/usr/bin/env python
"""Layer 7 — calibrate the simulator against real Silverstone telemetry.

Runs fully OFFLINE against a committed reference bundle (produced once, on a
machine with F1-API access, by `scripts/fetch_telemetry.py` — the sandbox
proxy blocks the F1 data hosts, see GUIDE.md). Fits five scale factors
(tire grip, downforce, drag, power, brake force) so the quasi-steady-state
lap profile of the Layer 1 car matches the reference lap, then checks the
result on a held-out second lap it never fitted.

Usage:
    python scripts/calibrate.py                       # fit + write artifacts
    python scripts/calibrate.py --report-only         # re-render from JSON
    python scripts/calibrate.py --bundle B --out J --report M

Outputs:
    data/calibrated/silverstone_2024.json   calibrated scales + provenance
    reports/layer7_report.md                before/after + per-corner tables

Exit codes: 0 all stated thresholds met; 1 calibration ran but a threshold
was missed (report says which — stop and discuss, do not weaken); 2 bundle
missing (fetch runbook printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.physics.conditions import COMPOUNDS, Conditions
from src.physics.qss_lap import qss_lap
from src.physics.vehicle_model import CarSpec
from src.utils.validation import (THRESHOLDS, TraceComparison, apply_scales,
                                  bundle_sha256, compare_traces, fit_scales,
                                  load_calibrated_spec, load_reference_bundle,
                                  load_reference_track, render_comparison,
                                  write_calibration)

DEFAULT_BUNDLE = "data/telemetry_reference/silverstone_2024_Q.npz"
DEFAULT_OUT = "data/calibrated/silverstone_2024.json"
DEFAULT_REPORT = "reports/layer7_report.md"

FETCH_RUNBOOK = """\
Reference bundle not found: {bundle}

The sandbox cannot reach the F1 data hosts (proxy-blocked), so the pull is a
one-time step on your machine or Colab:

    pip install fastf1
    python scripts/fetch_telemetry.py --year 2024 --gp Silverstone --session Q
    git add data/telemetry_reference && git commit -m "Layer 7: telemetry reference bundle" && git push

then re-run this script (everything from here on is offline)."""


def pinned_conditions(track_temp: float, fuel: float) -> tuple[Conditions, dict]:
    """Quali stint state: fresh softs in-window, dry, low fuel.

    Matches the reference laps (dry qualifying flying laps on new softs);
    fuel is an assumption (F1 quali fuel is not public) and is recorded in
    provenance so it can be revisited.
    """
    cond = Conditions(compound="soft", weather="dry", track_temp=track_temp,
                      fuel_mass=fuel)
    lo, hi = COMPOUNDS["soft"].temp_window
    cond.tire_temp = 0.5 * (lo + hi)      # tires in the middle of the window
    doc = {"compound": "soft", "weather": "dry", "wear": 0.0,
           "tire_temp_C": cond.tire_temp, "track_temp_C": track_temp,
           "fuel_kg": fuel, "grip_multiplier": cond.grip_multiplier(),
           "grip_components": cond.grip_components()}
    return cond, doc


def comparison_metrics(comp: TraceComparison) -> dict:
    return {"ref_label": comp.ref_label,
            "ref_lap_time_s": round(comp.ref_lap_time, 3),
            "sim_lap_time_s": round(comp.sim_lap_time, 3),
            "lap_time_err_pct": round(comp.lap_time_err_pct, 3),
            "speed_rmse_kmh": round(comp.rmse_kmh, 3)}


def driver_context() -> list[str]:
    """QSS car-potential vs the recorded Layer 5/6 driver laps (synthetic).

    Numbers quoted from the committed records (models/*/README.md,
    reports/layer6_report.md); the QSS values are recomputed live.
    """
    from src.tracks.track import Track

    lines = ["## Context: driver skill vs car potential (synthetic track)", ""]
    t = Track.from_synthetic()
    for name, compound, temp, fuel in (("benign (medium, 30 C)", "medium",
                                        30.0, 30.0),
                                       ("dry softs (35 C)", "soft", 35.0,
                                        30.0)):
        cond = Conditions(compound=compound, weather="dry", track_temp=temp,
                          fuel_mass=fuel)
        lo, hi = COMPOUNDS[compound].temp_window
        cond.tire_temp = 0.5 * (lo + hi)
        prof = qss_lap(t, CarSpec(fuel_mass=fuel),
                       grip_multiplier=cond.grip_multiplier(),
                       landmarks=False)
        lines.append(f"- QSS car potential, {name}: **{prof.lap_time:.1f} s** "
                     f"(uncalibrated placeholder physics)")
    lines += [
        "- recorded driver laps (same car, same track): pursuit baseline "
        "140.7 s (medium); L6 v3 generalist 138.1 s (benign); L6 v4 "
        "generalist 133.0 s (softs) — see `reports/layer6_report.md`.",
        "",
        "The gap between the RL/scripted laps and the QSS potential is "
        "driver skill and caution, which is why calibration fits physics "
        "through the QSS profile, not through a driver.", ""]
    return lines


def render_report(doc: dict, before_fit: TraceComparison,
                  after_fit: TraceComparison,
                  before_hold: TraceComparison | None,
                  after_hold: TraceComparison | None,
                  track_summary: str) -> str:
    th = THRESHOLDS
    m = doc["metrics"]
    cond_json = json.dumps(doc["provenance"]["pinned_conditions"],
                           sort_keys=True)
    lines = ["# Layer 7 — telemetry calibration report", ""]
    lines += [f"- created: {doc['created_utc']}",
              f"- bundle: `{doc['provenance']['bundle_path']}` "
              f"(sha256 `{doc['provenance']['bundle_sha256'][:16]}...`)",
              f"- track: {track_summary}",
              f"- pinned conditions: {cond_json}", ""]
    fit_names = doc["provenance"].get("fit_names", list(doc["scales"]))
    lines += ["## Fitted scales", "",
              "| scale | value | fields |", "|---|---|---|"]
    for k, v in doc["scales"].items():
        fields = ", ".join(doc["scale_fields"][k])
        mark = "" if k in fit_names else " (fixed)"
        lines.append(f"| {k}{mark} | {v:.4f} | {fields} |")
    lo, hi = doc["scale_bounds"]
    lines += ["", f"Bounds [{lo}, {hi}] per scale; objective "
              f"{m['objective_before']:.1f} -> {m['objective_after']:.3f} "
              f"({m['n_evals']} QSS evaluations).", ""]

    def block(title, before, after, err_key, rmse_key):
        rows = ["## " + title, "",
                "| | lap time err | speed RMSE |", "|---|---|---|",
                f"| before | {before.lap_time_err_pct:+.2f} % | "
                f"{before.rmse_kmh:.2f} km/h |",
                f"| after | **{after.lap_time_err_pct:+.2f} %** | "
                f"**{after.rmse_kmh:.2f} km/h** |",
                f"| threshold | +/-{th[err_key]:.1f} % | "
                f"{th[rmse_key]:.1f} km/h |", "",
                "```", render_comparison(after), "```", ""]
        return rows

    lines += block("Fit lap", before_fit, after_fit,
                   "fit_max_err_pct", "fit_max_rmse_kmh")
    if after_hold is not None:
        lines += block("Held-out lap (never fitted)", before_hold, after_hold,
                       "holdout_max_err_pct", "holdout_max_rmse_kmh")
        gap = abs(after_hold.lap_time_err_pct) - abs(after_fit.lap_time_err_pct)
        lines += [f"Fit-vs-holdout generalization gap: {gap:+.2f} % lap time, "
                  f"{after_hold.rmse_kmh - after_fit.rmse_kmh:+.2f} km/h "
                  "RMSE.", ""]
    lines += driver_context()
    lines += ["## Verdict", ""]
    for name, ok in doc["metrics"]["thresholds_met"].items():
        lines.append(f"- {name}: {'MET' if ok else '**MISSED**'}")
    lines += ["", "Thresholds are stated up front and never weakened; a miss "
              "is discussed, not edited away (see CLAUDE.md conventions).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--fit-lap", type=int, default=0,
                    help="bundle lap index to fit (default 0 = fastest)")
    ap.add_argument("--holdout-lap", type=int, default=1,
                    help="bundle lap index held out (-1 to disable)")
    ap.add_argument("--fuel", type=float, default=15.0,
                    help="assumed quali fuel load, kg (recorded in provenance)")
    ap.add_argument("--no-drs", action="store_true",
                    help="disable the DRS gate in the QSS profile")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the report from an existing --out JSON")
    args = ap.parse_args(argv)

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(FETCH_RUNBOOK.format(bundle=bundle_path))
        return 2

    bundle = load_reference_bundle(bundle_path)
    track = load_reference_track(bundle)
    track_summary = (f"{track.length:.0f} m, {len(track.corners)} corners, "
                     f"raw closure gap {track.raw_closure_gap:.1f} m "
                     f"(source {track.source})")
    print(f"track   : {track_summary}")
    for i, lap in enumerate(bundle.laps):
        print(f"lap {i}   : {lap.label}  ({lap.lap_time:.3f} s, "
              f"{len(lap.dist)} samples)")

    track_temp = float(bundle.meta.get("weather", {}).get("track_temp", 30.0))
    cond, cond_doc = pinned_conditions(track_temp, args.fuel)
    base_spec = CarSpec(fuel_mass=args.fuel)
    grip = cond.grip_multiplier()
    drs = not args.no_drs
    fit_ref = bundle.laps[args.fit_lap]
    hold_ref = (bundle.laps[args.holdout_lap]
                if 0 <= args.holdout_lap < len(bundle.laps)
                and args.holdout_lap != args.fit_lap else None)

    def compare_with(spec, ref):
        prof = qss_lap(track, spec, grip_multiplier=grip, drs=drs)
        return compare_traces(ref, prof.s, prof.v, prof.lap_time,
                              track.length)

    if args.report_only:
        _, doc = load_calibrated_spec(args.out, base=base_spec)
        scales = doc["scales"]
        print(f"loaded  : {args.out} (report-only)")
        fitted_spec = apply_scales(base_spec, scales)
        before_fit = compare_with(base_spec, fit_ref)
        after_fit = compare_with(fitted_spec, fit_ref)
        before_hold = compare_with(base_spec, hold_ref) if hold_ref else None
        after_hold = compare_with(fitted_spec, hold_ref) if hold_ref else None
    else:
        before_fit = compare_with(base_spec, fit_ref)
        print(f"before  : err {before_fit.lap_time_err_pct:+.2f} %  "
              f"RMSE {before_fit.rmse_kmh:.2f} km/h — fitting...")
        fit = fit_scales(track, base_spec, grip, fit_ref, drs=drs)
        after_fit = fit.comparison
        fitted_spec = apply_scales(base_spec, fit.scales)
        before_hold = compare_with(base_spec, hold_ref) if hold_ref else None
        after_hold = compare_with(fitted_spec, hold_ref) if hold_ref else None

        met = {
            "fit_lap_time": abs(after_fit.lap_time_err_pct)
            <= THRESHOLDS["fit_max_err_pct"],
            "fit_rmse": after_fit.rmse_kmh <= THRESHOLDS["fit_max_rmse_kmh"],
        }
        if after_hold is not None:
            met["holdout_lap_time"] = (abs(after_hold.lap_time_err_pct)
                                       <= THRESHOLDS["holdout_max_err_pct"])
            met["holdout_rmse"] = (after_hold.rmse_kmh
                                   <= THRESHOLDS["holdout_max_rmse_kmh"])

        metrics = {
            "fit": {"before": comparison_metrics(before_fit),
                    "after": comparison_metrics(after_fit)},
            "holdout": ({"before": comparison_metrics(before_hold),
                         "after": comparison_metrics(after_hold)}
                        if after_hold is not None else None),
            "objective_before": round(fit.objective_before, 3),
            "objective_after": round(fit.objective_after, 4),
            "n_evals": fit.n_evals,
            "thresholds": THRESHOLDS,
            "thresholds_met": met,
        }
        provenance = {
            "tool": "scripts/calibrate.py",
            "bundle_path": str(bundle_path),
            "bundle_sha256": bundle_sha256(bundle_path),
            "bundle_meta": bundle.meta,
            "fit_lap": fit_ref.label,
            "holdout_lap": hold_ref.label if hold_ref else None,
            "pinned_conditions": cond_doc,
            "drs_modeled": drs,
            "track": track_summary,
            "fit_names": list(fit.fit_names),
            "fixed_scales_note": ("power_scale fixed at 1.0: a speed trace "
                                  "only pins P/cda (cda/power ridge, see "
                                  "fit_scales docstring); engine power is "
                                  "the regulation-known quantity"),
        }
        write_calibration(args.out, scales=fit.scales, provenance=provenance,
                          metrics=metrics)
        print(f"written : {args.out}")
        _, doc = load_calibrated_spec(args.out, base=base_spec)

    report = render_report(doc, before_fit, after_fit, before_hold,
                           after_hold, track_summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"report  : {report_path}")
    print()
    print(render_comparison(after_fit))
    if after_hold is not None:
        print()
        print(render_comparison(after_hold))

    met = doc["metrics"]["thresholds_met"]
    print()
    for name, ok in met.items():
        print(f"  {'MET   ' if ok else 'MISSED'}  {name}")
    return 0 if all(met.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
