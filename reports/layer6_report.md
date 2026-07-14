# Layer 6 report — generalist driver

Versions: python 3.11.15, numpy 2.4.6, gymnasium 1.3.0, stable_baselines3 2.9.0, torch 2.13.0+cu130

## Curriculum run

- total policy steps: 1,375,000
- wall time: 7.44 h
- budget scale: 1.0

- `A_benign_laps` enter_stage at 0 total steps (0.00 h)
- `A_benign_laps` gate_met at 275,000 total steps (1.44 h)
- `B1_setup_near` enter_stage at 275,000 total steps (1.44 h)
- `B1_setup_near` budget_exhausted at 575,000 total steps (3.10 h)
- `B2_setup_full` enter_stage at 575,000 total steps (3.10 h)
- `B2_setup_full` budget_exhausted at 925,000 total steps (5.04 h)
- `C_full_dr` enter_stage at 925,000 total steps (5.04 h)
- `C_full_dr` budget_exhausted at 1,375,000 total steps (7.44 h)

## Validation panel (final policy)

| condition | laps | mean lap (s) | time-to-lap (s) | progress (m) |
|---|---|---|---|---|
| V1_benign | 3/3 | 138.1 | 138.1 | 5348.9 |
| V2_light_lowwing | 0/3 | - | 300.0 | 31.4 |
| V3_heavy_highwing | 0/3 | - | 300.0 | 128.3 |
| V4_dry_softs | 0/3 | - | 300.0 | 835.4 |
| V5_dry_hards | 0/3 | - | 300.0 | 362.6 |
| V6_damp_inters | 0/3 | - | 300.0 | 30.7 |
| V7_wet_wets | 0/3 | - | 300.0 | 33.2 |

## Lap-time distribution over random conditions

```
     ep  weather compound      fuel  aero  end         lap time  progress   avg
      1  damp  intermediate  36.7  0.34  off_track      DNF        52 m   143
      2  dry   soft          30.8  0.03  off_track      DNF        31 m   135
      3  dry   soft          96.5  0.47  off_track      DNF       209 m   210
      4  dry   medium        97.3  0.15  off_track      DNF       116 m   230
      5  dry   soft          60.9  0.99  off_track      DNF        47 m   178
      6  dry   soft          40.4  0.46  spin           DNF       622 m   224
      7  dry   medium        79.3  0.03  off_track      DNF       491 m   219
      8  dry   medium        52.3  0.30  off_track      DNF        19 m   133
      9  dry   hard          47.5  0.51  off_track      DNF      1336 m   164
     10  wet   wet           71.5  0.56  off_track      DNF       154 m   189
     11  dry   medium        71.7  0.71  spin           DNF        35 m   184
     12  dry   hard          58.5  0.82  off_track      DNF        95 m   152
     13  dry   soft          20.3  0.61  off_track      DNF      2632 m   186
     14  damp  wet           71.7  0.38  off_track      DNF        55 m   178
     15  wet   wet           36.2  0.21  off_track      DNF        40 m   168
     16  dry   soft          67.8  0.66  off_track      DNF        28 m   109
     17  damp  wet          102.6  0.10  off_track      DNF        52 m   176
     18  dry   hard          64.9  0.22  off_track      DNF       504 m   263
     19  damp  intermediate  82.6  0.88  off_track      DNF        27 m    99
     20  dry   soft          75.4  0.30  off_track      DNF       554 m   254
     21  dry   medium       104.8  0.63  off_track      DNF        43 m   178
     22  dry   medium        51.3  0.14  off_track      DNF        55 m   132
     23  dry   hard          94.5  0.29  off_track      DNF        49 m   154
     24  dry   soft          26.3  0.92  off_track      DNF        79 m   209
     25  dry   soft          99.9  0.53  off_track      DNF       440 m   213
     26  wet   wet           89.2  0.33  off_track      DNF        54 m   166
     27  dry   medium        76.9  0.22  off_track      DNF        65 m   186
     28  damp  medium        58.6  0.27  off_track      DNF        42 m   116
     29  dry   soft          41.4  0.29  off_track      DNF       278 m   262
     30  dry   soft          75.2  0.43  off_track      DNF      1593 m   299
     31  damp  wet           64.0  0.99  off_track      DNF        57 m   196
     32  dry   medium        41.4  0.26  off_track      DNF       121 m   231
     33  damp  wet           23.5  0.84  off_track      DNF        51 m   185
     34  dry   soft          71.8  0.32  off_track      DNF       391 m   192
     35  damp  soft          67.3  0.00  off_track      DNF        56 m   177
     36  dry   medium        55.2  0.14  off_track      DNF        30 m   118
     37  dry   medium        94.9  0.10  off_track      DNF        82 m   118
     38  dry   soft          32.9  0.94  spin           DNF       160 m   162
     39  damp  wet           93.4  0.57  off_track      DNF        47 m   212
     40  damp  intermediate  69.1  0.22  off_track      DNF        26 m   100
    finished 0/40 (0%)
    DNF progress: mean 270 m | median 56 m
    damp : 0/10 finished, mean lap -
    dry  : 0/27 finished, mean lap -
    wet  : 0/3 finished, mean lap -
```
