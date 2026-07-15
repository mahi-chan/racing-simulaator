# Layer 6 report — generalist driver

Versions: python 3.11.15, numpy 2.4.6, gymnasium 1.3.0, stable_baselines3 2.9.0, torch 2.13.0+cu130

## Curriculum run

- total policy steps: 1,325,000
- wall time: 7.35 h
- budget scale: 1.0

- `A_benign_laps` enter_stage at 0 total steps (0.00 h)
- `A_benign_laps` gate_met at 225,000 total steps (1.19 h)
- `B1_setup_near` enter_stage at 225,000 total steps (1.19 h)
- `B1_setup_near` budget_exhausted at 525,000 total steps (2.84 h)
- `B2_setup_full` enter_stage at 525,000 total steps (2.84 h)
- `B2_setup_full` budget_exhausted at 875,000 total steps (4.82 h)
- `C_full_dr` enter_stage at 875,000 total steps (4.82 h)
- `C_full_dr` budget_exhausted at 1,325,000 total steps (7.34 h)

## Validation panel (final policy)

| condition | laps | mean lap (s) | time-to-lap (s) | progress (m) |
|---|---|---|---|---|
| V1_benign | 1/3 | 153.34 | 251.11 | 3709.7 |
| V2_light_lowwing | 0/3 | - | 300.0 | 1114.4 |
| V3_heavy_highwing | 0/3 | - | 300.0 | 1136.7 |
| V4_dry_softs | 1/3 | 133.0 | 244.33 | 4306.2 |
| V5_dry_hards | 0/3 | - | 300.0 | 1357.1 |
| V6_damp_inters | 0/3 | - | 300.0 | 874.7 |
| V7_wet_wets | 0/3 | - | 300.0 | 733.3 |

## Lap-time distribution over random conditions

```
     ep  weather compound      fuel  aero  end         lap time  progress   avg
      1  damp  intermediate  36.7  0.34  off_track      DNF        56 m   124
      2  dry   soft          30.8  0.03  off_track      DNF      2351 m   159
      3  dry   soft          96.5  0.47  off_track      DNF      1460 m   151
      4  dry   medium        97.3  0.15  off_track      DNF      1746 m   188
      5  dry   soft          60.9  0.99  off_track      DNF      3114 m   189
      6  dry   soft          40.4  0.46  stall          DNF       641 m   130
      7  dry   medium        79.3  0.03  off_track      DNF      1181 m   179
      8  dry   medium        52.3  0.30  off_track      DNF        19 m   122
      9  dry   hard          47.5  0.51  off_track      DNF      1523 m   151
     10  wet   wet           71.5  0.56  spin           DNF       109 m   206
     11  dry   medium        71.7  0.71  off_track      DNF      2385 m   175
     12  dry   hard          58.5  0.82  off_track      DNF       930 m   166
     13  dry   soft          20.3  0.61  off_track      DNF      4088 m   179
     14  damp  wet           71.7  0.38  off_track      DNF       394 m   215
     15  wet   wet           36.2  0.21  off_track      DNF        57 m   159
     16  dry   soft          67.8  0.66  off_track      DNF        27 m   108
     17  damp  wet          102.6  0.10  off_track      DNF       972 m   217
     18  dry   hard          64.9  0.22  off_track      DNF      1929 m   182
     19  damp  intermediate  82.6  0.88  off_track      DNF       262 m   177
     20  dry   soft          75.4  0.30  off_track      DNF      2233 m   183
     21  dry   medium       104.8  0.63  stall          DNF       515 m   114
     22  dry   medium        51.3  0.14  spin           DNF      1347 m   148
     23  dry   hard          94.5  0.29  off_track      DNF       131 m   162
     24  dry   soft          26.3  0.92  off_track      DNF       363 m   174
     25  dry   soft          99.9  0.53  off_track      DNF      3396 m   171
     26  wet   wet           89.2  0.33  spin           DNF       125 m   191
     27  dry   medium        76.9  0.22  spin           DNF       378 m   231
     28  damp  medium        58.6  0.27  off_track      DNF      1450 m   222
     29  dry   soft          41.4  0.29  spin           DNF      2848 m   182
     30  dry   soft          75.2  0.43  off_track      DNF      2851 m   206
     31  damp  wet           64.0  0.99  off_track      DNF       549 m   234
     32  dry   medium        41.4  0.26  spin           DNF      1933 m   164
     33  damp  wet           23.5  0.84  off_track      DNF        67 m   198
     34  dry   soft          71.8  0.32  off_track      DNF      1783 m   174
     35  damp  soft          67.3  0.00  off_track      DNF       228 m   181
     36  dry   medium        55.2  0.14  off_track      DNF      4243 m   149
     37  dry   medium        94.9  0.10  off_track      DNF        23 m    86
     38  dry   soft          32.9  0.94  spin           DNF      1900 m   162
     39  damp  wet           93.4  0.57  off_track      DNF       162 m   225
     40  damp  intermediate  69.1  0.22  off_track      DNF        34 m    82
    finished 0/40 (0%)
    DNF progress: mean 1245 m | median 951 m
    damp : 0/10 finished, mean lap -
    dry  : 0/27 finished, mean lap -
    wet  : 0/3 finished, mean lap -
```

## Generalist vs setup-specialists (documented gap)

```
    condition            metric                       generalist  specialist       gap
    H1_dry_mid           time-to-lap (DNF=cap), s          300.0       300.0     +0.0%
                         laps / progress ratio                 0           0    17.44x
    H2_dry_heavy_maxwing time-to-lap (DNF=cap), s          300.0       300.0     +0.0%
                         laps / progress ratio                 0           0     1.08x
    H3_damp_inters_light time-to-lap (DNF=cap), s          300.0       271.4    +10.5%
                         laps / progress ratio                 0           1     0.21x
    H4_wet_wets_midfuel  time-to-lap (DNF=cap), s          300.0       250.8    +19.6%
                         laps / progress ratio                 0           2     0.17x
```

Positive gap = generalist slower than the specialist on that condition (300 s-capped time-to-lap).
