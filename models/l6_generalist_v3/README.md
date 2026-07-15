# Layer 6 generalist, attempt 3 — the DEPTH champion

The v3 curriculum's final driver (progressive setup widening, 15-40 m/s
spawns, dry-compound rotation, breadth-first keeper; 1.375M steps, 7.44 h,
seed 42; `runs/l6_generalist_v3`). Kept alongside attempt 4 because the two
sit on different Pareto points:

- v3 (this): best BENIGN driver — 3/3 spread laps at 138.1 s mean (quicker
  than the 140.7 s hand-coded pursuit baseline), plus greatly improved
  dry-soft survival (835 m). One-condition otherwise: 0/40 under the full
  acceptance distribution (DNF median 56 m).
- v4 (`models/l6_generalist_v4`): best BREADTH driver — laps benign AND
  dry softs (133.0 s), every panel condition >= 733 m, DNF median 951 m,
  still 0/40 finishes.

Not the accepted artifact (see the v4 README for the acceptance record and
the four-attempt history). Layers 7-8 wanting a reliable single-condition
lapper should start here; wanting broad-condition progress, start from v4.

Files: model.zip + vecnormalize.pkl (paired), config.json,
curriculum_state.json (stage/keeper/gate history), run_report.json.
