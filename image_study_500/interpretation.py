"""Plain-language interpretation of synthetic AquaSense regression outputs."""

from __future__ import annotations

import pandas as pd

from image_pipeline import ROOT, METALS, MAXIMA


def interpret_prediction(result: dict) -> dict:
    """Describe an estimate without implying chemical or regulatory validity."""
    if result.get('status') != 'accepted':
        return {
            'decision': 'No result',
            'summary': 'The image did not pass the required checks; no concentrations were estimated.',
            'next_step': 'Retake the complete card under even light, with sharp visible markers and reference patches.',
            'limitations': 'The model is trained only on synthetic images and cannot validate a water sample.',
        }
    rows = result.get('predictions', [])
    withheld = [r['metal'] for r in rows if r.get('status') != 'estimate']
    range_flags = [r['metal'] for r in rows if 'interval_exceeds_simulated_range' in r.get('flags', [])]
    if withheld:
        decision = 'Partial result'
        summary = f"Estimates are withheld for {', '.join(withheld)} because their raw predictions fall outside the simulated training range."
        next_step = 'Review image quality and repeat the capture; do not replace withheld values with zero or an endpoint.'
    else:
        decision = 'Synthetic estimates available'
        summary = 'Seven simulated concentrations were estimated from the corrected strip image.'
        next_step = 'Use these values only to exercise the processing workflow. Obtain paired lab measurements before interpreting real water.'
    if range_flags:
        summary += f" Intervals for {', '.join(range_flags)} extend beyond the simulated concentration range."
    return {
        'decision': decision,
        'summary': summary,
        'next_step': next_step,
        'how_to_read': 'Each estimate is a continuous regression output in simulated µg/L. The nominal 90% interval is calibrated on clean held-out synthetic sessions. Coverage can be substantially lower under changed lighting or strip batches even when image checks pass; this is not chemical certainty, a detection limit, or a safety threshold.',
        'limitations': 'No real strip chemistry, ESP32 capture data, reference-laboratory values, or regulatory threshold has been validated.',
    }


def build_research_interpretation() -> str:
    """Rebuild a paper-ready, explicitly simulation-limited interpretation from saved CSVs."""
    metrics = pd.read_csv(ROOT / 'results/test_metrics.csv')
    summary = metrics.groupby('model', sort=False)[['R2', 'NRMSE_percent_full_scale']].mean().sort_values('NRMSE_percent_full_scale')
    lines = [
        '# Interpretation of the synthetic-image experiment', '',
        '**Scope:** 500 rendered images represent 250 simulated concentration profiles (two captures each) in 50 simulated sessions. '
        'The common held-out test set has 100 images from 50 profiles in 10 sessions. These are not physical water samples.', '',
        '## Reading the metrics', '',
        '- MAE and RMSE are in simulated µg/L for each metal. RMSE penalizes larger mistakes more strongly.',
        '- R² describes fit to held-out simulated targets; it is not classification accuracy or a percentage of correctly detected metals.',
        '- Normalized RMSE divides RMSE by that metal’s **simulated full-scale concentration**. Its mean permits an across-metal comparison; raw MAE/RMSE should not be averaged across different concentration ranges.',
        '- Predictive intervals are separate from model-error metrics. An interval is not a concentration detection limit or a probability that a specific value is chemically true.', '',
        '## Four-model held-out comparison', '',
        '| Model | Mean R² | Mean normalized RMSE (% simulated full scale) |',
        '|---|---:|---:|',
    ]
    for model, row in summary.iterrows():
        lines.append(f"| {model} | {row.R2:.3f} | {row.NRMSE_percent_full_scale:.3f} |")
    winners = metrics.loc[metrics.groupby('metal').RMSE.idxmin(), ['metal', 'model']].set_index('metal').loc[METALS]
    lines += ['', 'XGBoost has the lowest mean normalized RMSE in this one held-out synthetic test. '
              'That is a descriptive ranking, not proof that it will perform best on another simulator or a manufactured strip.', '',
              '## Per-metal winner on this test split', '',
              '| Metal | Lowest test RMSE model | RMSE (simulated µg/L) | R² |', '|---|---|---:|---:|']
    for metal, row in winners.iterrows():
        record = metrics[(metrics.metal == metal) & (metrics.model == row['model'])].iloc[0]
        lines.append(f"| {metal} | {row['model']} | {record.RMSE:.3f} | {record.R2:.3f} |")
    lines += ['', 'These per-metal test winners are **not** used to choose the deployed prototype models. '
              'Those were selected by grouped validation within the training sessions; a separate calibration set supplies synthetic prediction intervals.', '',
              '## Interpretation flow for one uploaded image', '',
              '1. Check file, card geometry, lighting and color references. Reject failures before showing concentrations.',
              '2. Detect the fiducials, rectify the card, correct color, and extract pad features.',
              '3. Check whether the feature vector resembles the synthetic training domain. Reject out-of-domain images.',
              '4. Predict seven simulated concentrations. Withhold individual out-of-range predictions rather than clipping them.',
              '5. Show each supported point estimate with its synthetic-session-calibrated interval and any range flag.',
              '6. State that no safety, compliance, or real-world detection decision follows from these numbers.', '',
              '## What can be claimed in a paper', '',
              'The software pipeline recovered simulated concentration signals under the stated renderer and split. '
              'It has **not** established metal selectivity, limit of detection, recovery, reproducibility, or accuracy in real water. '
              'The simulator encodes concentration-dependent colors by assumption, and the training and test images share that simulator and card layout. '
              'Physical seven-metal strips, standardized ESP32-S3 captures, and independent reference-laboratory measurements are required for analytical validation.', '',
              'Source tables: `results/test_metrics.csv`, `results/robustness/reliable_test_metrics.csv`, '
              '`results/reliability_validation.csv`, and `results/calibration_session_scores.csv`.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    destination = ROOT / 'results/result_interpretation.md'
    destination.write_text(build_research_interpretation(), encoding='utf-8')
    print(destination)
