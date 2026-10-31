# Interpretation of the synthetic-image experiment

**Scope:** 500 rendered images represent 250 simulated concentration profiles (two captures each) in 50 simulated sessions. The common held-out test set has 100 images from 50 profiles in 10 sessions. These are not physical water samples.

## Reading the metrics

- MAE and RMSE are in simulated µg/L for each metal. RMSE penalizes larger mistakes more strongly.
- R² describes fit to held-out simulated targets; it is not classification accuracy or a percentage of correctly detected metals.
- Normalized RMSE divides RMSE by that metal’s **simulated full-scale concentration**. Its mean permits an across-metal comparison; raw MAE/RMSE should not be averaged across different concentration ranges.
- Predictive intervals are separate from model-error metrics. An interval is not a concentration detection limit or a probability that a specific value is chemically true.

## Four-model held-out comparison

| Model | Mean R² | Mean normalized RMSE (% simulated full scale) |
|---|---:|---:|
| XGBoost | 0.985 | 3.006 |
| RandomForest | 0.981 | 3.397 |
| PLS | 0.961 | 4.821 |
| RBF_SVR | 0.915 | 7.078 |

XGBoost has the lowest mean normalized RMSE in this one held-out synthetic test. That is a descriptive ranking, not proof that it will perform best on another simulator or a manufactured strip.

## Per-metal winner on this test split

| Metal | Lowest test RMSE model | RMSE (simulated µg/L) | R² |
|---|---|---:|---:|
| Pb | XGBoost | 4.002 | 0.987 |
| Hg | XGBoost | 0.648 | 0.990 |
| Cd | XGBoost | 1.181 | 0.989 |
| As | XGBoost | 4.710 | 0.979 |
| Cr | XGBoost | 5.704 | 0.985 |
| Cu | XGBoost | 54.333 | 0.982 |
| Zn | XGBoost | 156.805 | 0.980 |

These per-metal test winners are **not** used to choose the deployed prototype models. Those were selected by grouped validation within the training sessions; a separate calibration set supplies synthetic prediction intervals.

## Interpretation flow for one uploaded image

1. Check file, card geometry, lighting and color references. Reject failures before showing concentrations.
2. Detect the fiducials, rectify the card, correct color, and extract pad features.
3. Check whether the feature vector resembles the synthetic training domain. Reject out-of-domain images.
4. Predict seven simulated concentrations. Withhold individual out-of-range predictions rather than clipping them.
5. Show each supported point estimate with its synthetic-session-calibrated interval and any range flag.
6. State that no safety, compliance, or real-world detection decision follows from these numbers.

## What can be claimed in a paper

The software pipeline recovered simulated concentration signals under the stated renderer and split. It has **not** established metal selectivity, limit of detection, recovery, reproducibility, or accuracy in real water. The simulator encodes concentration-dependent colors by assumption, and the training and test images share that simulator and card layout. Physical seven-metal strips, standardized ESP32-S3 captures, and independent reference-laboratory measurements are required for analytical validation.

Source tables: `results/test_metrics.csv`, `results/robustness/reliable_test_metrics.csv`, `results/reliability_validation.csv`, and `results/calibration_session_scores.csv`.
