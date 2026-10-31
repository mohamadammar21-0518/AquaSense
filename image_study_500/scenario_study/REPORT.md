# Expanded synthetic scenario study

**Simulation only.** No images, colors, metal concentrations, or accuracy values here are experimentally measured.

1,800 newly rendered primary images = 900 independent simulated concentration profiles × 2 captures, in 180 sessions. Another 120 intentionally corrupted images reuse challenge profiles and are QC stress cases, not independent samples.

The train set contains 1,200 primary images (120 sessions) from three scenarios. The grouped same-scenario test set contains 300 images (30 sessions). The 300-image camera/batch-shift challenge (30 sessions) is absent from all tuning and training.

## Four-model comparison

Mean normalized RMSE is the unweighted mean of per-metal RMSE divided by each metal’s simulated full-scale range. It is not a classification accuracy.

| Model | Same-scenario test NRMSE % | Shift challenge NRMSE % |
|---|---:|---:|
| XGBoost | 2.166 | 8.099 |
| RandomForest | 2.432 | 8.904 |
| RBF_SVR | 3.503 | 11.579 |
| PLS | 4.479 | 11.517 |

## Image acceptance

| Partition | Scenario | Images | Accepted | Acceptance rate |
|---|---|---:|---:|---:|
| challenge | camera_batch_shift | 300 | 295 | 98.3% |
| stress | heavy_blur | 30 | 0 | 0.0% |
| stress | missing_marker | 30 | 0 | 0.0% |
| stress | underexposure | 30 | 0 | 0.0% |
| stress | upside_down | 30 | 0 | 0.0% |
| test | low_range | 100 | 100 | 100.0% |
| test | mixed_extremes | 100 | 100 | 100.0% |
| test | standard | 100 | 100 | 100.0% |
| train | low_range | 400 | 400 | 100.0% |
| train | mixed_extremes | 400 | 400 | 100.0% |
| train | standard | 400 | 400 | 100.0% |

## Interpretation

Compare each metal’s MAE, RMSE, and R² in `results/metrics_by_scenario_metal.csv`; never treat its R² as a detection accuracy percentage. Metrics use only QC-accepted images, so read them alongside acceptance/rejection rates. The shifted-camera challenge measures transfer to one *simulated* change, not an ESP32-S3 or real-strip result. The intentionally corrupted images test whether bad captures are withheld; they do not enlarge the training dataset.

A high score would show that the model recovered the simulator’s invented concentration-color rules. It cannot establish chemical selectivity, detection limits, interference tolerance, or field performance. Collect physical-strip images paired with laboratory reference concentrations for those claims.
