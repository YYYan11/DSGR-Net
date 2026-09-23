# A8 model table audit (2026-09-23)

All values below were read from the A8 `results.json` files under
the formal A8 experiment results directory. Standard deviations
use the sample definition (`ddof=1`).

| Model | Runs | Test RMSE (kg/ha) | Test MAE (kg/ha) | Test R² |
|---|---:|---:|---:|---:|
| Empirical baseline (WL-DNN) | 3 | 549.99 ± 41.27 | 427.66 ± 47.60 | 0.8050 ± 0.0384 |
| Vanilla DNN (Strong DNN) | 3 | 489.43 ± 57.01 | 375.13 ± 39.95 | 0.8444 ± 0.0406 |
| Irrigation-aware DNN | 3 | 458.43 ± 59.07 | 356.09 ± 49.73 | 0.8631 ± 0.0397 |
| Process-guided NN (V13 Process Gate) | 3 | 458.67 ± 54.51 | 350.74 ± 37.75 | 0.8632 ± 0.0369 |
| Gradient-informed NN | 3 | 491.60 ± 44.37 | 378.11 ± 34.47 | 0.8438 ± 0.0340 |
| CNN-based model | 3 | 489.90 ± 34.39 | 376.47 ± 27.74 | 0.8454 ± 0.0277 |
| Multi-scale CNN | 3 | 506.63 ± 47.17 | 388.56 ± 38.48 | 0.8339 ± 0.0376 |
| Hybrid NN-CNN | 3 | 517.54 ± 48.14 | 395.23 ± 34.71 | 0.8267 ± 0.0395 |
| DSGR-Net (V16, ours) | 3 | 450.82 ± 41.16 | 343.63 ± 29.98 | 0.8687 ± 0.0280 |

The draft value `403.49` is the seed-42 V16 result, not the three-seed mean. It
must not appear as `403.49 ± ...`. The original draft's `Process-guided NN`
row duplicated V16; the coherent controlled comparison uses V13 Process Gate
for that row and V16 for the final DSGR-Net row.

The missing seeds 43 and 44 for Gradient-informed NN, CNN-based model, and
Hybrid NN-CNN were run on A8 on 2026-09-23 using the exact arguments stored in
each seed-42 checkpoint. The six new raw result files are mirrored in this
directory. Their completion marker is
`logs/20260923_missing_seeds.done` on A8.
