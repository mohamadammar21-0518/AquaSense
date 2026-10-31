# Real-data pilot: required before experimental accuracy claims

No real samples or reference measurements have been fabricated. The CSV is a blank collection template, not training data.

## Before collecting

Have the chemistry team specify each reagent, intended analyte/species, physical pad geometry, intended concentration range, sample matrix, preparation method, reaction time, storage conditions and stability expectations. Record the reference laboratory method and its uncertainty/quantification limits. The current simulated spot-to-metal mapping and concentration bounds are placeholders, not a chemical specification.

Use a fixed holder, diffuse controlled lighting and a visible reference card. Choose camera exposure/white-balance settings only after testing actual captures. Record distance, lighting settings, camera settings, temperature, pH, lot, date and reaction time. Use separate session groups when the setup changes. Physical dimensions must come from your team's final strip design; the simulator specifies pixels only.

## Collect paired evidence

For every physical sample, retain raw JPEGs and a matching laboratory report for each metal. Give all repeats of that sample the same sample_id. Images acquired together share a session_id. Record blanks, standards, duplicates, relevant matrix/interferent conditions and quality-control checks according to the laboratory's established method. Agree the calibration range and an external test campaign before model tuning.

In `measurements_template.csv`, enter paths relative to the manifest folder. Use partitions `train`, `calibration`, and `test`. Keep entire sample and session groups in one partition. For broader generalization, additionally reserve lots/devices/dates or sites as appropriate to the intended deployment. Never create an independent split for each crop or repeat capture.

The validator accepts finite nonnegative numeric laboratory values. It intentionally rejects blank/non-numeric values: do not replace `<LOD`, `<LOQ` or missing laboratory results with zero. Retain qualifiers in the laboratory records and agree a censored-data analysis before importing those records. All targets use µg/L; convert other units explicitly before import.

Run `python validate_real_data.py path/to/completed_manifest.csv` before any training. An empty template should fail validation. The validator does not certify chemistry or measurement accuracy and does not train models or merge real data into the synthetic dataset.

## Acceptance evidence still needed

Reference agreement, recovery, repeatability, interference/selectivity, relevant analytical detection/quantification limits, stability, capture failure rates, and external campaign performance. Predefine tolerances from the intended analytical use; no arbitrary 'accuracy percentage' is supplied by this software.
