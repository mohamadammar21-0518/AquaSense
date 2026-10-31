# Expanded AquaSense synthetic scenario study

This is a software robustness experiment, **not physical heavy-metal detection**. All pad responses and concentrations are assumed by the renderer. Images are new, independently generated scenes, not augmented copies of the original 500-image dataset.

Run `AquaSense_Scenario_Study.ipynb` from the parent `image_study_500` folder. Its default view loads the completed results. Set `REBUILD = True` to regenerate and retrain, or run `python scenario_study.py`. The fixed seed is 20261025.

The 1,800 primary JPEGs represent 900 simulated profiles, two captures per profile, across 180 acquisition sessions. Each session has five profiles. The 1,200-image training set contains standard, low-range, and mixed-extreme scenarios (40 sessions each). The same-scenario test has another 10 sessions from each scenario: 300 images total. The 300-image camera/batch-shift challenge has 30 entirely separate sessions, and its scenario does not appear in training or validation. Grouped three-fold validation tunes all four model families on training sessions only. Images are 640 × 480 and retain the proposed seven-spot card layout.

The 120 extra stress JPEGs are transformations of one existing challenge image per challenge session: heavy blur, upside-down orientation, a covered marker, and severe underexposure. They are **not 120 new samples** and are never training data. Their intended use is to audit image rejection.

Read `results/metrics_by_scenario_metal.csv` together with `results/acceptance_by_scenario.csv`. Prediction metrics are computed only for QC-accepted images; a scenario with a high rejection rate must not be presented as having demonstrated reliable concentration prediction. `results/predictions.csv` contains row-level simulated truth and predictions. `REPORT.md` and the two PNG charts summarize the comparisons. Validation results and trained pipelines are in `results/validation.csv` and `models/`.

The held-out camera/batch scenario perturbs simulated white balance, gamma, JPEG quality, blur, gain, and strip-lot color. It is *not* an empirical model of a specific ESP32-S3 camera or a real reagent lot. A synthetic-to-real gap remains even if this challenge performs well. Future evaluation must collect manufactured strip captures paired with independent laboratory concentrations, with samples/lots/sessions kept together in splits.
