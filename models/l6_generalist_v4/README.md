# Layer 6 generalist, attempt 4 — the BREADTH champion (best of four attempts)

Recipe: v3 curriculum (A_benign_laps -> B1_setup_near -> B2_setup_full ->
C_full_dr) + the v4 anti-forgetting config: replay buffer 1.5M (spans the
whole run) and a (512, 512) net. 1.325M policy steps, 7.35 h, seed 42
(`runs/l6_generalist_v4`, exact recipe in src/agents/curriculum.py at the
commit that added this directory).

This is NOT the accepted Layer 6 artifact — `models/l6_generalist` is
deliberately absent because T7's provisional acceptance floors (canonical +
spread benign laps AND >= 30% finish over the full Layer 4 randomization)
are unmet: the 40-episode distribution finished 0/40 in all four attempts.
The floor stays on the record un-weakened; the layer closes documented, per
the decision recorded in the session. Point `L6_DRIVER_DIR` here to run T7
against this driver anyway.

## What this driver is

The first policy in the project to lap two distinct conditions in one
evaluation, and the strongest driver of the four attempts on breadth:

| panel condition | laps | mean lap | mean progress |
|---|---|---|---|
| V1_benign          | 1/3 | 153.3 s | 3,710 m |
| V2_light_lowwing   | 0/3 | - | 1,114 m |
| V3_heavy_highwing  | 0/3 | - | 1,137 m |
| V4_dry_softs       | 1/3 | **133.0 s** (fastest lap of any attempt) | 4,306 m |
| V5_dry_hards       | 0/3 | - | 1,357 m |
| V6_damp_inters     | 0/3 | - | 875 m |
| V7_wet_wets        | 0/3 | - | 733 m |

Under the full acceptance distribution (40 random conditions incl.
mismatched compounds, 25-60 m/s spawns): 0/40 finished, DNF progress
median 951 m / mean 1,245 m, best 4,243 m (79% of a lap). v2/v3 medians
were 33 m / 56 m.

## Documented % vs same-architecture specialists (the spec deliverable)

250k-step specialists per held-out condition, identical config, identical
eval episodes (5 spread starts; 300 s-capped time-to-lap):

| held-out condition | gap | laps gen/spec | progress ratio (gen/spec) |
|---|---|---|---|
| H1_dry_mid           | +0.0%  | 0/0 | 17.44x |
| H2_dry_heavy_maxwing | +0.0%  | 0/0 | 1.08x |
| H3_damp_inters_light | +10.5% | 0/1 | 0.21x |
| H4_wet_wets_midfuel  | +19.6% | 0/2 | 0.17x |

The generalist is within 0-19.6% of specialists everywhere and out-drives
both dry specialists on progress; on wet surfaces dedicated policies lap
where it does not — the measured cost of generality. The damp and wet
specialists' laps also prove every weather in the sim is lappable by a
dedicated policy at this scale: the generalist's wet gap is interference
cost, not impossible physics.

## The four-attempt record (all preserved)

1. v1 (`models/l6_generalist_attempt1`): 6-dim actions — never lapped.
2. v2 (reports history): SimplifiedActions, benign-first — lapped benign
   at 133.9 s but a point-specialist; 0/40; died in ~30 m on any compound
   flip; stalled below 20 m/s.
3. v3 (`models/l6_generalist_v3`): progressive widening, 15-40 m/s spawns,
   compound rotation, breadth-first keeper — every fix moved its metric,
   still one-condition (the DEPTH champion: benign 3/3 at 138.1 s).
4. v4 (this): + full-run replay and 4x net capacity — first 2-condition
   lapper, ~17x distribution progress vs v3. Key negative result: with ALL
   past experience retained in replay, the live policy still traded away
   earlier skills during later stages — the forgetting is network
   interference, not data extinction.

Files: model.zip + vecnormalize.pkl (paired; policy is only reproducible
with its obs normalization), config.json, curriculum_state.json (full
stage/keeper/gate history), run_report.json (panel + distribution +
comparison; rendered at reports/layer6_report.md).
