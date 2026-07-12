# Layer 6 report — generalist driver

Versions: python 3.11.15, numpy 2.4.6, gymnasium 1.3.0, stable_baselines3 2.9.0, torch 2.13.0+cu130

## Curriculum run

- total policy steps: 1,250,000
- wall time: 5.06 h
- budget scale: 1.0

- `A_benign_laps` enter_stage at 0 total steps (0.00 h)
- `A_benign_laps` gate_met at 350,000 total steps (1.37 h)
- `B_setup_dr` enter_stage at 350,000 total steps (1.37 h)
- `B_setup_dr` budget_exhausted at 750,000 total steps (2.97 h)
- `C_weather_matched` enter_stage at 750,000 total steps (2.97 h)
- `C_weather_matched` budget_exhausted at 1,250,000 total steps (5.05 h)

## Validation panel (final policy)

| condition | laps | mean lap (s) | time-to-lap (s) | progress (m) |
|---|---|---|---|---|
| V1_benign | 2/3 | 133.9 | 189.27 | 4050.4 |
| V2_light_lowwing | 0/3 | - | 300.0 | 61.4 |
| V3_heavy_highwing | 0/3 | - | 300.0 | 507.3 |
| V4_damp_inters | 0/3 | - | 300.0 | 28.3 |
| V5_wet_wets | 0/3 | - | 300.0 | 30.4 |

## Lap-time distribution over random conditions

```
     ep  weather compound      fuel  aero  end         lap time  progress   avg
      1  damp  intermediate  36.7  0.34  spin           DNF        31 m   131
      2  dry   soft          30.8  0.03  spin           DNF        27 m   122
      3  dry   soft          96.5  0.47  spin           DNF        26 m   177
      4  dry   medium        97.3  0.15  off_track      DNF        39 m   182
      5  dry   soft          60.9  0.99  spin           DNF        25 m   163
      6  dry   soft          40.4  0.46  spin           DNF        28 m   101
      7  dry   medium        79.3  0.03  off_track      DNF        35 m   124
      8  dry   medium        52.3  0.30  off_track      DNF        19 m   122
      9  dry   hard          47.5  0.51  spin           DNF        31 m   192
     10  wet   wet           71.5  0.56  spin           DNF        33 m   186
     11  dry   medium        71.7  0.71  off_track      DNF       882 m    82
     12  dry   hard          58.5  0.82  off_track      DNF        44 m   128
     13  dry   soft          20.3  0.61  off_track      DNF        31 m    62
     14  damp  wet           71.7  0.38  spin           DNF        33 m   167
     15  wet   wet           36.2  0.21  spin           DNF        34 m   156
     16  dry   soft          67.8  0.66  spin           DNF        26 m   106
     17  damp  wet          102.6  0.10  spin           DNF        33 m   167
     18  dry   hard          64.9  0.22  off_track      DNF        64 m   212
     19  damp  intermediate  82.6  0.88  off_track      DNF        24 m    87
     20  dry   soft          75.4  0.30  off_track      DNF        30 m   122
     21  dry   medium       104.8  0.63  stall          DNF       504 m    88
     22  dry   medium        51.3  0.14  off_track      DNF        64 m   122
     23  dry   hard          94.5  0.29  spin           DNF        30 m   138
     24  dry   soft          26.3  0.92  off_track      DNF        60 m   204
     25  dry   soft          99.9  0.53  off_track      DNF        39 m   128
     26  wet   wet           89.2  0.33  spin           DNF        40 m   157
     27  dry   medium        76.9  0.22  stall          DNF       104 m    57
     28  damp  medium        58.6  0.27  off_track      DNF        40 m   107
     29  dry   soft          41.4  0.29  spin           DNF        27 m   184
     30  dry   soft          75.2  0.43  spin           DNF        28 m   187
     31  damp  wet           64.0  0.99  spin           DNF        30 m   187
     32  dry   medium        41.4  0.26  off_track      DNF       159 m   192
     33  damp  wet           23.5  0.84  spin           DNF        32 m   171
     34  dry   soft          71.8  0.32  off_track      DNF        34 m   126
     35  damp  soft          67.3  0.00  spin           DNF        36 m   167
     36  dry   medium        55.2  0.14  off_track      DNF       353 m    86
     37  dry   medium        94.9  0.10  off_track      DNF       178 m   123
     38  dry   soft          32.9  0.94  off_track      DNF        29 m   121
     39  damp  wet           93.4  0.57  spin           DNF        32 m   198
     40  damp  intermediate  69.1  0.22  off_track      DNF        34 m    82
    finished 0/40 (0%)
    DNF progress: mean 84 m | median 33 m
    damp : 0/10 finished, mean lap -
    dry  : 0/27 finished, mean lap -
    wet  : 0/3 finished, mean lap -
```
