"""Independent synthetic scenario study; no real strip or ESP32 measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from threadpoolctl import threadpool_limits

from image_pipeline import ROOT, METALS, TARGETS, MAXIMA, FEATURES, assay_colors, render_card, camera_capture, model_spaces, preprocess_image
from quality_control import ImageRejected, check_image

SEED = 20261025
OUT = ROOT / 'scenario_study'
IMAGE_DIR = OUT / 'images'
RESULTS = OUT / 'results'
MODELS = OUT / 'models'
SCENARIOS = [('standard', 50), ('low_range', 50), ('mixed_extremes', 50), ('camera_batch_shift', 30)]


def small_spaces():
    """Four prespecified compact tuning grids; no test or challenge feedback."""
    spaces = model_spaces()
    spaces['PLS'][1]['regressor__model__n_components'] = [4, 8]
    spaces['RandomForest'][1]['regressor__model__max_depth'] = [None]
    spaces['XGBoost'][1]['regressor__model__reg_lambda'] = [1.]
    spaces['RBF_SVR'][1]['regressor__model__gamma'] = ['scale']
    spaces['RBF_SVR'][1]['regressor__model__C'] = [10., 100.]
    return spaces


def _fraction(rng, scenario):
    if scenario == 'low_range':
        return rng.beta(1.1, 3.2, 7) * .22
    if scenario == 'mixed_extremes':
        values = rng.uniform(.03, .28, 7)
        high = rng.choice(7, size=int(rng.integers(2, 5)), replace=False)
        values[high] = rng.uniform(.76, 1., len(high))
        return values
    values = .88 * rng.beta(1.2, 1.5, 7) + .12 * rng.beta(2, 2, 7)
    values[rng.random(7) < .06] = rng.uniform(0, .015)
    return values


def _shift_capture(encoded, rng):
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)
    # Deliberate camera-domain shift, not a measured ESP32-S3 image model.
    gains = rng.uniform(.91, 1.09, 3)
    image = np.clip(image * gains, 0, 255)
    gamma = rng.uniform(.88, 1.12)
    image = 255 * (image / 255) ** gamma
    image = cv2.GaussianBlur(image.astype(np.uint8), (3, 3), rng.uniform(.2, .85))
    return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(63, 83))])[1]


def generate():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for scenario, count in SCENARIOS:
        for session in range(count):
            partition = 'challenge' if scenario == 'camera_batch_shift' else ('test' if session >= 40 else 'train')
            session_id = f'SCN-{scenario}-{session:03d}'
            session_gain = rng.normal(1, .025 if partition != 'challenge' else .055, 3)
            lot_shift = rng.normal(0, 1.5 if partition != 'challenge' else 6., (7, 3))
            for profile in range(5):
                sample_id = f'{session_id}-P{profile}'
                fractions = _fraction(rng, scenario)
                ph, temperature = rng.uniform(6, 8), rng.uniform(20, 30)
                colors = assay_colors(fractions, ph, temperature, lot_shift)
                card = render_card(colors, rng)
                for capture in range(2):
                    image_id = f'{sample_id}-C{capture}'
                    encoded, info = camera_capture(card, rng, session_gain)
                    if partition == 'challenge':
                        encoded = _shift_capture(encoded, rng)
                    relative = Path('scenario_study/images') / f'{image_id}.jpg'
                    (ROOT / relative).write_bytes(encoded.tobytes())
                    rows.append({'image_id': image_id, 'sample_id': sample_id, 'session_id': session_id,
                                 'scenario': scenario, 'partition': partition, 'capture': capture,
                                 'image_path': relative.as_posix(), 'is_synthetic': True,
                                 'ph_simulated': ph, 'temperature_C_simulated': temperature,
                                 **dict(zip(TARGETS, fractions * MAXIMA))})
        print(f'Generated {scenario}: {count * 10} images', flush=True)
    labels = pd.DataFrame(rows)
    assert len(labels) == 1800 and labels.sample_id.nunique() == 900 and labels.session_id.nunique() == 180
    assert labels.groupby('session_id').partition.nunique().eq(1).all()
    labels.to_csv(OUT / 'labels.csv', index=False, float_format='%.9f')
    return labels


def _stress_images(labels):
    """120 additional corrupted captures, not new independent samples."""
    rows = []
    for row in labels[labels.partition == 'challenge'].groupby('session_id').head(1).itertuples():
        original = cv2.imread(str(ROOT / row.image_path))
        _, _, _, markers, _ = preprocess_image(ROOT / row.image_path)
        px, py = np.rint(markers[0]).astype(int)
        missing_marker = original.copy()
        cv2.rectangle(missing_marker, (px - 32, py - 32), (px + 32, py + 32), (240, 240, 240), -1)
        conditions = {
            'heavy_blur': cv2.GaussianBlur(original, (31, 31), 8),
            'upside_down': cv2.rotate(original, cv2.ROTATE_180),
            'missing_marker': missing_marker,
            'underexposure': np.clip(original.astype(np.float32) * .22, 0, 255).astype(np.uint8),
        }
        for condition, image in conditions.items():
            image_id = f'{row.image_id}-{condition}'
            relative = Path('scenario_study/images') / f'{image_id}.jpg'
            cv2.imwrite(str(ROOT / relative), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({'image_id': image_id, 'source_image_id': row.image_id,
                         'session_id': row.session_id, 'condition': condition,
                         'image_path': relative.as_posix(), 'is_synthetic': True})
    frame = pd.DataFrame(rows)
    assert len(frame) == 120
    frame.to_csv(OUT / 'stress_labels.csv', index=False)
    return frame


def extract(labels, stress):
    limits = json.loads((ROOT / 'models/reliable/config.json').read_text(encoding='utf-8'))['quality_limits']
    # Set using original training data only; no new test/challenge feedback.
    rows, checks = [], []
    for n, row in enumerate(labels.itertuples(), 1):
        try:
            features, _, qc = check_image(ROOT / row.image_path, limits)
            rows.append({'image_id': row.image_id, **features})
            checks.append({'image_id': row.image_id, 'scenario': row.scenario, 'partition': row.partition,
                           'accepted': True, 'reason': '', **qc})
        except ImageRejected as error:
            checks.append({'image_id': row.image_id, 'scenario': row.scenario, 'partition': row.partition,
                           'accepted': False, 'reason': '; '.join(error.reasons), **error.qc})
        if n % 300 == 0:
            print(f'Quality checked {n}/1800', flush=True)
    for row in stress.itertuples():
        try:
            _, _, qc = check_image(ROOT / row.image_path, limits)
            accepted, reason = True, ''
        except ImageRejected as error:
            accepted, reason, qc = False, '; '.join(error.reasons), error.qc
        checks.append({'image_id': row.image_id, 'scenario': row.condition, 'partition': 'stress',
                       'accepted': accepted, 'reason': reason, **qc})
    features = pd.DataFrame(rows)
    qc = pd.DataFrame(checks)
    features.to_csv(OUT / 'features.csv', index=False, float_format='%.9f')
    qc.to_csv(OUT / 'quality_checks.csv', index=False)
    return features, qc


def evaluate(labels, features, qc):
    frame = labels.merge(features, on='image_id', how='left', validate='one_to_one')
    frame = frame.merge(qc[['image_id', 'accepted']], on='image_id', validate='one_to_one')
    train = frame[(frame.partition == 'train') & frame.accepted].copy()
    assert train.session_id.nunique() >= 100
    assert not set(train.session_id) & set(frame[frame.partition != 'train'].session_id)
    assert not set(train.sample_id) & set(frame[frame.partition != 'train'].sample_id)
    folds = list(GroupKFold(n_splits=3).split(train[FEATURES], groups=train.session_id))
    for a, b in folds:
        assert not set(train.iloc[a].session_id) & set(train.iloc[b].session_id)
    MODELS.mkdir(exist_ok=True)
    metrics, predictions, validation = [], [], []
    for metal, target, maximum in zip(METALS, TARGETS, MAXIMA):
        for family, (model, grid) in small_spaces().items():
            print(f'Training {metal}: {family}', flush=True)
            search = GridSearchCV(model, grid, cv=folds, scoring='neg_root_mean_squared_error',
                                  n_jobs=4, error_score='raise')
            with threadpool_limits(limits=1), joblib.parallel_backend('threading'):
                search.fit(train[FEATURES], train[target])
            joblib.dump(search.best_estimator_, MODELS / f'{metal}_{family}.joblib')
            validation.append({'metal': metal, 'model': family, 'validation_RMSE': -search.best_score_,
                               'best_parameters': json.dumps(search.best_params_, sort_keys=True)})
            for (partition, scenario), group in frame[(frame.partition != 'train') & frame.accepted].groupby(['partition', 'scenario']):
                pred = np.asarray(search.predict(group[FEATURES])).reshape(-1)
                truth = group[target].to_numpy()
                rmse = mean_squared_error(truth, pred) ** .5
                metrics.append({'partition': partition, 'scenario': scenario, 'metal': metal, 'model': family,
                                'n_images': len(group), 'n_profiles': group.sample_id.nunique(),
                                'n_sessions': group.session_id.nunique(), 'MAE': mean_absolute_error(truth, pred),
                                'RMSE': rmse, 'R2': r2_score(truth, pred),
                                'NRMSE_percent_full_scale': 100 * rmse / maximum, 'is_synthetic': True})
                predictions.extend({'image_id': image_id, 'partition': partition, 'scenario': scenario,
                                    'session_id': session_id, 'sample_id': sample_id, 'metal': metal, 'model': family,
                                    'actual_ug_L_simulated': actual, 'predicted_ug_L_simulated': estimated}
                                   for image_id, session_id, sample_id, actual, estimated in
                                   zip(group.image_id, group.session_id, group.sample_id, truth, pred))
    pd.DataFrame(validation).to_csv(RESULTS / 'validation.csv', index=False)
    pd.DataFrame(metrics).to_csv(RESULTS / 'metrics_by_scenario_metal.csv', index=False)
    pd.DataFrame(predictions).to_csv(RESULTS / 'predictions.csv', index=False)
    qc.groupby(['partition', 'scenario']).accepted.agg(['count', 'sum', 'mean']).to_csv(RESULTS / 'acceptance_by_scenario.csv')
    config = {'seed': SEED, 'source': 'SYNTHETIC ONLY', 'primary_images': 1800,
              'independent_profiles': 900, 'sessions': 180, 'stress_augmentations': 120,
              'model_selection': 'Grouped 3-fold validation on train sessions; no test/challenge tuning.',
              'challenge': 'camera_batch_shift withheld from training and validation',
              'quality_limits_source': 'original 500-image study training/calibration split',
              'model_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.glob('*.joblib')}}
    (OUT / 'manifest.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return pd.DataFrame(metrics)


def report(metrics):
    acceptance = pd.read_csv(RESULTS / 'acceptance_by_scenario.csv')
    mean = metrics.groupby(['partition', 'model']).NRMSE_percent_full_scale.mean().unstack(0)
    lines = ['# Expanded synthetic scenario study', '',
             '**Simulation only.** No images, colors, metal concentrations, or accuracy values here are experimentally measured.', '',
             '1,800 newly rendered primary images = 900 independent simulated concentration profiles × 2 captures, '
             'in 180 sessions. Another 120 intentionally corrupted images reuse challenge profiles and are QC stress cases, '
             'not independent samples.', '',
             'The train set contains 1,200 primary images (120 sessions) from three scenarios. '
             'The grouped same-scenario test set contains 300 images (30 sessions). '
             'The 300-image camera/batch-shift challenge (30 sessions) is absent from all tuning and training.', '',
             '## Four-model comparison', '',
             'Mean normalized RMSE is the unweighted mean of per-metal RMSE divided by each metal’s simulated full-scale range. '
             'It is not a classification accuracy.', '',
             '| Model | Same-scenario test NRMSE % | Shift challenge NRMSE % |', '|---|---:|---:|']
    for model, row in mean.sort_values('test').iterrows():
        lines.append(f"| {model} | {row['test']:.3f} | {row['challenge']:.3f} |")
    lines += ['', '## Image acceptance', '', '| Partition | Scenario | Images | Accepted | Acceptance rate |',
              '|---|---|---:|---:|---:|']
    for row in acceptance.itertuples():
        lines.append(f'| {row.partition} | {row.scenario} | {row.count} | {row.sum} | {row.mean:.1%} |')
    lines += ['', '## Interpretation', '',
              'Compare each metal’s MAE, RMSE, and R² in `results/metrics_by_scenario_metal.csv`; '
              'never treat its R² as a detection accuracy percentage. Metrics use only QC-accepted images, '
              'so read them alongside acceptance/rejection rates. The shifted-camera challenge measures transfer '
              'to one *simulated* change, not an ESP32-S3 or real-strip result. The intentionally corrupted images '
              'test whether bad captures are withheld; they do not enlarge the training dataset.', '',
              'A high score would show that the model recovered the simulator’s invented concentration-color rules. '
              'It cannot establish chemical selectivity, detection limits, interference tolerance, or field performance. '
              'Collect physical-strip images paired with laboratory reference concentrations for those claims.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def charts():
    metrics = pd.read_csv(RESULTS / 'metrics_by_scenario_metal.csv')
    values = metrics.groupby(['scenario', 'model']).NRMSE_percent_full_scale.mean().unstack('model')
    values = values.reindex(['standard', 'low_range', 'mixed_extremes', 'camera_batch_shift'])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    values.plot.bar(ax=ax, width=.8)
    ax.set(ylabel='Mean NRMSE (% of simulated full scale)', xlabel='Synthetic scenario',
           title='Four-model errors by synthetic scenario')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Model', ncol=4, loc='upper center', bbox_to_anchor=(.5, -.2))
    fig.tight_layout()
    fig.savefig(OUT / 'scenario_model_comparison.png', dpi=180)
    plt.close(fig)
    acceptance = pd.read_csv(RESULTS / 'acceptance_by_scenario.csv')
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(acceptance.scenario, acceptance['mean'] * 100, color='#118287')
    ax.set(ylabel='Images accepted (%)', ylim=(0, 105), title='Image-quality gate by synthetic scenario')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / 'image_acceptance.png', dpi=180)
    plt.close(fig)


def refresh_stress_qc():
    """Regenerate intentional corruptions and refresh their QC summary only."""
    labels = pd.read_csv(OUT / 'labels.csv')
    stress = _stress_images(labels)
    limits = json.loads((ROOT / 'models/reliable/config.json').read_text(encoding='utf-8'))['quality_limits']
    qc = pd.read_csv(OUT / 'quality_checks.csv')
    qc = qc[qc.partition != 'stress'].copy()
    rows = []
    for row in stress.itertuples():
        try:
            _, _, details = check_image(ROOT / row.image_path, limits)
            accepted, reason = True, ''
        except ImageRejected as error:
            accepted, reason, details = False, '; '.join(error.reasons), error.qc
        rows.append({'image_id': row.image_id, 'scenario': row.condition, 'partition': 'stress',
                     'accepted': accepted, 'reason': reason, **details})
    qc = pd.concat([qc, pd.DataFrame(rows)], ignore_index=True)
    qc.to_csv(OUT / 'quality_checks.csv', index=False)
    qc.groupby(['partition', 'scenario']).accepted.agg(['count', 'sum', 'mean']).to_csv(RESULTS / 'acceptance_by_scenario.csv')
    report(pd.read_csv(RESULTS / 'metrics_by_scenario_metal.csv'))
    charts()
    return qc[qc.partition == 'stress']


def main():
    labels = generate()
    stress = _stress_images(labels)
    features, qc = extract(labels, stress)
    metrics = evaluate(labels, features, qc)
    report(metrics)
    charts()
    print('Expanded synthetic study complete.', flush=True)


if __name__ == '__main__':
    main()
