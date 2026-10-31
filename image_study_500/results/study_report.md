# Synthetic image study results

All figures describe simulated images and assumed chemistry.

500 images; 250 profiles; 50 sessions. Train: 400 images, 200 profiles, 40 sessions. Test: 100 images, 50 profiles, 10 sessions.

| Model | Mean normalized RMSE (% full scale) | Mean R² |
|---|---:|---:|
| XGBoost | 3.006 | 0.985 |
| RandomForest | 3.397 | 0.981 |
| PLS | 4.821 | 0.961 |
| RBF_SVR | 7.078 | 0.915 |

These are post-evaluation comparisons, not validated deployment selections. The single-split experiment does not establish statistical superiority. Reagent-specific responses are assumed, and none of these results support real-water detection claims. See README.md for full limitations and differences from previous experiments.
