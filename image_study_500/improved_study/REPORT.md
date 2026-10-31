# Improved synthetic models (v2)

All results concern simulated chemistry. Earlier test/challenge results informed the decision to investigate gaps; their reuse is exploratory. A fresh 200-image audit is generated after model selection, using new sessions and seed. It remains a test of the same assumed chemistry. Fresh audit concentrations are independent uniform fractions of each simulated range. "Fresh nominal" describes nominal imaging conditions, not identical concentration distributions.

V2 reserves 24 of the 120 original development sessions for interval calibration; fits use the other 96. V1 used all 120 for fitting. Thus these versions differ in both training method and calibration allocation.

Selection compares all-pad features, own-pad RGB, neighboring-pad RGB, and own-pad RGB with assumed nuisance augmentation. Three grouped training folds score clean and perturbed validation images equally. Scaling is fitted within each fold. Calibration and evaluation labels never select recipes or parameters.

Neighbor features use the simulator’s cyclic cross-response mapping (including the first/last pad pair). This mapping is an assumed mathematical relationship, not measured reagent interference.

| Metal | Selected family | Feature recipe |
|---|---|---|
| Pb | XGBoost | neighbors_clean |
| Hg | XGBoost | neighbors_clean |
| Cd | XGBoost | neighbors_clean |
| As | XGBoost | neighbors_clean |
| Cr | XGBoost | neighbors_clean |
| Cu | XGBoost | neighbors_clean |
| Zn | XGBoost | neighbors_clean |

| Scenario | Previous XGBoost NRMSE % | Selected v2 NRMSE % |
|---|---:|---:|
| camera_batch_shift | 8.099 | 7.711 |
| fresh_nominal | 3.112 | 2.997 |
| fresh_shift | 8.737 | 8.306 |
| low_range | 1.047 | 1.100 |
| mixed_extremes | 2.902 | 2.978 |
| standard | 2.548 | 2.730 |

| Family | V2 fresh nominal NRMSE % | V2 fresh shift NRMSE % |
|---|---:|---:|
| PLS | 5.918 | 9.006 |
| RandomForest | 3.237 | 8.674 |
| XGBoost | 2.997 | 8.306 |
| RBF_SVR | 3.594 | 9.297 |

| Scenario | Mean image interval coverage | Mean session coverage | Domain gate acceptance |
|---|---:|---:|---:|
| camera_batch_shift | 80.9% | 37.6% | 100.0% |
| fresh_nominal | 98.3% | 88.6% | 100.0% |
| fresh_shift | 77.9% | 38.6% | 99.0% |
| low_range | 100.0% | 100.0% | 100.0% |
| mixed_extremes | 98.9% | 92.9% | 100.0% |
| standard | 99.6% | 97.1% | 100.0% |

The 90% target is per metal over a new exchangeable session. Coverage under shifted imaging is substantially lower. The nearest-neighbor domain gate accepts most shifted captures, so passing that check does not demonstrate reliable uncertainty.

| Fresh audit | NRMSE change, v2 minus v1 (percentage points) | Approximate paired 95% interval |
|---|---:|---|
| fresh_nominal | -0.115 | [-0.264, 0.106] |
| fresh_shift | -0.431 | [-0.930, 0.156] |

The percentile intervals resample whole sessions jointly across all metals (2,000 resamples). There are only ten sessions per condition. An interval spanning zero does not establish a reliable improvement.

NRMSE uses simulated full-scale concentrations; it is not percent-correct accuracy. Tables include all QC-accepted images before domain/range withholding. Read fresh_audit_qc.csv and coverage_and_acceptance.csv alongside errors. Intervals are calibrated on clean synthetic sessions only; shifted-domain coverage can fail. Each fresh audit condition has only ten sessions, so differences are descriptive.

Remaining gaps: experimentally measured reagent response, real camera captures, reference-lab labels, and physical board configuration. Arbitrary pad-specific color offsets can be indistinguishable from concentration changes; more training cannot eliminate that ambiguity.
