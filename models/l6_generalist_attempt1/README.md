# Layer 6 generalist — attempt 1 (evidence, NOT the accepted artifact)

1,750,000 policy steps of the default A→D curriculum (seed 42, commits
b828bb0 + b514f39, ~6.6 h CPU, three crash-resumes from the rotating
checkpoint — the resume contract held every time).

**Layer 6 acceptance status: T7 FAILED — kept on record, thresholds untouched.**
Canonical benign lap DNF: off_track at 2,950 m of 5,349 m (avg 169 km/h to
that point). Final validation panel (3 spread episodes each):

| condition        | laps | mean progress |
|------------------|------|---------------|
| V1_benign (med)  | 0/3  | 3,137 m       |
| V2_soft_light    | 0/3  | 30 m          |
| V3_hard_heavy    | 0/3  | 147 m         |
| V4_damp_inters   | 0/3  | 50 m          |
| V5_wet_wets      | 0/3  | 457 m         |

Two structural findings, documented for the restructure decision:
1. Benign progress shows strong diminishing returns (1,774 m @ 150k →
   2,203 m @ 650k → 3,137 m best): budget alone won't buy the first lap.
2. The policy is catastrophically compound-sensitive (30 m on cold softs) —
   consistent with Layer 4's emergent cold-slick physics quirk, which is a
   Layer 7 calibration target; compound generalization is being trained
   against physics known to be placeholder.

Weights kept for warm-starts and forensics. `training_curve.csv` is the SB3
progress log; `curriculum_state.json` holds the full stage/gate/best history.
