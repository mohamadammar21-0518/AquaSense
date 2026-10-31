"""Version 2: training-only feature selection, nuisance robustness and calibration."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from image_pipeline import ROOT, METALS, MAXIMA, TARGETS, FEATURES, CHANNELS, model_spaces, lab_from_rgb, assay_colors, render_card, camera_capture
from quality_control import check_image, ImageRejected
from reliability import fit_domain

V2_DIR = ROOT / 'improved_study'
V2_MODELS = V2_DIR / 'models'
V2_SEED = 20261029
V2_RECIPES = ['all_clean', 'own_clean', 'neighbors_clean', 'own_augmented']


def v2_data():
    labels = pd.read_csv(ROOT / 'scenario_study/labels.csv')
    features = pd.read_csv(ROOT / 'scenario_study/features.csv')
    return labels.merge(features, on='image_id', validate='one_to_one')


def v2_split(frame):
    development = frame[frame.partition == 'train'].copy()
    calibration_groups = set()
    rng = np.random.default_rng(V2_SEED)
    for _, part in development.groupby('scenario', sort=True):
        groups = np.sort(part.session_id.unique())
        calibration_groups.update(rng.choice(groups, 8, replace=False))
    calibration = development[development.session_id.isin(calibration_groups)].copy()
    proper = development[~development.session_id.isin(calibration_groups)].copy()
    heldout = frame[frame.partition != 'train'].copy()
    assert [x.session_id.nunique() for x in (proper, calibration)] == [96, 24]
    for key in ['session_id', 'sample_id']:
        sets = [set(x[key]) for x in (proper, calibration, heldout)]
        assert all(not sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)])
    return proper.reset_index(drop=True), calibration, heldout


def v2_perturb(frame, seed):
    """Assumed RGB nuisance variation; HSV/LAB stay derived from RGB."""
    out = frame[FEATURES].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    for group in sorted(frame.session_id.unique()):
        indices = np.flatnonzero(frame.session_id.to_numpy() == group)
        offsets = rng.normal(0, 3., (7, 3))
        for i in indices:
            for metal, offset in zip(METALS, offsets):
                rgb = np.clip(out.loc[i, [f'{metal}_{c}' for c in ['R', 'G', 'B']]].to_numpy(dtype=float)
                              + offset + rng.normal(0, .5, 3), 0, 255)
                hsv = cv2.cvtColor((rgb / 255).astype(np.float32).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                hsv[1:] *= 100
                out.loc[i, [f'{metal}_{c}' for c in CHANNELS]] = np.r_[rgb, hsv, lab_from_rgb(rgb)]
    return out


def v2_columns(metal, recipe):
    if recipe == 'all_clean':
        return FEATURES
    pads = [metal]
    if recipe == 'neighbors_clean':
        i = METALS.index(metal)
        pads = [METALS[(i - 1) % 7], metal, METALS[(i + 1) % 7]]
    return [f'{pad}_{c}' for pad in pads for c in ['R', 'G', 'B']]


def v2_candidate(metal, family, recipe, setting):
    columns = v2_columns(metal, recipe)
    model = model_spaces()[family][0]
    if family == 'PLS':
        params = {'n_components': [2, min(8, len(columns))][setting]}
    elif family == 'RandomForest':
        params = {'min_samples_leaf': [1, 3][setting], 'max_depth': None}
    elif family == 'XGBoost':
        params = {'max_depth': [2, 3][setting], 'n_estimators': 350, 'learning_rate': .035, 'reg_lambda': 5.}
    else:
        params = {'C': [10., 100.][setting], 'gamma': 'scale', 'epsilon': .01}
    model.set_params(**{f'regressor__model__{k}': v for k, v in params.items()})
    return Pipeline([('columns', ColumnTransformer([('selected', 'passthrough', columns)])), ('regression', model)])


def train_improved():
    V2_MODELS.mkdir(parents=True, exist_ok=True)
    proper, calibration, _ = v2_split(v2_data())
    protocol = {'version': 'expanded-v2', 'seed': V2_SEED, 'recipes': V2_RECIPES,
                'proper_sessions': 96, 'calibration_sessions': 24,
                'selection': '3 grouped folds; equal-weight clean and assumed RGB-noise validation MSE; no test/challenge tuning',
                'perturbation': 'Per-session, per-pad RGB Gaussian sd=3; per-image sd=0.5; HSV/LAB recomputed; applied within training folds only',
                'old_holdout_status': 'Previously inspected; comparisons are exploratory. New audit generated only after selection.',
                'audit_seed': V2_SEED + 10000, 'audit_sessions': 20}
    (V2_DIR / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    assignment = pd.concat([proper.assign(role='proper_train'), calibration.assign(role='calibration')])
    assignment[['image_id', 'sample_id', 'session_id', 'scenario', 'role']].to_csv(V2_DIR / 'split.csv', index=False)
    folds = list(GroupKFold(3).split(proper, groups=proper.session_id))
    fold_data = []
    for fold, (a, b) in enumerate(folds):
        train, valid = proper.iloc[a], proper.iloc[b]
        assert not set(train.session_id) & set(valid.session_id)
        fold_data.append((train, valid, v2_perturb(train, V2_SEED + fold), v2_perturb(valid, V2_SEED + 100 + fold)))
    pd.concat([proper.iloc[b][['image_id', 'session_id']].assign(validation_fold=i) for i, (_, b) in enumerate(folds)]).to_csv(V2_DIR / 'validation_folds.csv', index=False)
    augmented_proper = v2_perturb(proper, V2_SEED + 500)
    validation_rows, selected, calibration_rows = [], {}, []
    for metal, target in zip(METALS, TARGETS):
        family_winners = []
        for family in ['PLS', 'RandomForest', 'XGBoost', 'RBF_SVR']:
            print(f'V2 validation: {metal} / {family}', flush=True)
            trials = []
            for recipe in V2_RECIPES:
                for setting in range(2):
                    squared_clean, squared_noise = [], []
                    for train, valid, noisy_train, noisy_valid in fold_data:
                        X, y = train[FEATURES], train[target]
                        if recipe.endswith('augmented'):
                            X = pd.concat([X, noisy_train], ignore_index=True)
                            y = np.tile(y, 2)
                        fitted = v2_candidate(metal, family, recipe, setting)
                        with threadpool_limits(limits=1):
                            fitted.fit(X, y)
                            clean_pred = np.asarray(fitted.predict(valid[FEATURES])).reshape(-1)
                            noisy_pred = np.asarray(fitted.predict(noisy_valid)).reshape(-1)
                        squared_clean.extend((clean_pred - valid[target].to_numpy()) ** 2)
                        squared_noise.extend((noisy_pred - valid[target].to_numpy()) ** 2)
                    clean = float(np.sqrt(np.mean(squared_clean)))
                    noise = float(np.sqrt(np.mean(squared_noise)))
                    score = float(np.sqrt((clean ** 2 + noise ** 2) / 2))
                    validation_rows.append({'metal': metal, 'model': family, 'recipe': recipe, 'setting': setting,
                                            'clean_RMSE': clean, 'perturbed_RMSE': noise, 'selection_RMSE': score})
                    trials.append((score, recipe, setting))
            score, recipe, setting = min(trials)
            fitted = v2_candidate(metal, family, recipe, setting)
            X, y = proper[FEATURES], proper[target]
            if recipe.endswith('augmented'):
                X, y = pd.concat([X, augmented_proper], ignore_index=True), np.tile(y, 2)
            with threadpool_limits(limits=1):
                fitted.fit(X, y)
            joblib.dump(fitted, V2_MODELS / f'{metal}_{family}.joblib')
            family_winners.append((score, family, recipe, setting, fitted))
        score, family, recipe, setting, fitted = min(family_winners, key=lambda item: item[0])
        residual = abs(np.asarray(fitted.predict(calibration[FEATURES])).reshape(-1) - calibration[target].to_numpy())
        session_scores = pd.DataFrame({'session_id': calibration.session_id.to_numpy(), 'error': residual}).groupby('session_id').error.max()
        rank = math.ceil((len(session_scores) + 1) * .9)
        radius = float(np.sort(session_scores.to_numpy())[rank - 1])
        calibration_rows.extend({'metal': metal, 'session_id': group, 'max_abs_error': error} for group, error in session_scores.items())
        selected[metal] = {'family': family, 'recipe': recipe, 'setting': setting, 'validation_RMSE': score,
                           'interval_radius_ug_L_simulated': radius, 'nominal_coverage': .9}
        joblib.dump(fitted, V2_MODELS / f'{metal}.joblib')
    pd.DataFrame(validation_rows).to_csv(V2_DIR / 'validation.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(V2_DIR / 'calibration_scores.csv', index=False)
    joblib.dump(fit_domain(proper), V2_MODELS / 'domain.joblib')
    config = {'version': 'expanded-v2', 'training_data': 'SYNTHETIC ONLY', 'metals': selected,
              'quality_limits': json.loads((ROOT / 'models/reliable/config.json').read_text())['quality_limits'],
              'proper_training_sessions': 96, 'calibration_sessions': 24,
              'interval_method': '90% per-metal session-max split-conformal on 24 clean synthetic sessions; no shifted-domain or real-water coverage guarantee.',
              'model_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in V2_MODELS.glob('*.joblib')}}
    (V2_MODELS / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    load_improved.cache_clear()
    return config


@lru_cache(maxsize=1)
def load_improved():
    config = json.loads((V2_MODELS / 'config.json').read_text(encoding='utf-8'))
    for name, digest in config['model_sha256'].items():
        if hashlib.sha256((V2_MODELS / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('V2 model checksum mismatch. Rebuild the improved bundle.')
    return config, {metal: joblib.load(V2_MODELS / f'{metal}.joblib') for metal in METALS}, joblib.load(V2_MODELS / 'domain.joblib')


def estimate_improved(image_path):
    config, models, domain = load_improved()
    features, corrected, qc = check_image(image_path, config['quality_limits'])
    X = pd.DataFrame([features], columns=FEATURES)
    distance = float(domain['neighbors'].kneighbors(domain['scaler'].transform(X[domain['columns']]))[0][0, 0])
    qc.update(domain_distance=distance, domain_threshold=domain['threshold'])
    if distance > domain['threshold']:
        raise ImageRejected(['Color features are outside the supported synthetic training domain.'], qc)
    rows = []
    for metal, maximum in zip(METALS, MAXIMA):
        value = float(np.asarray(models[metal].predict(X)).reshape(-1)[0])
        info = config['metals'][metal]
        radius = info['interval_radius_ug_L_simulated']
        supported = bool(np.isfinite(value) and 0 <= value <= maximum)
        flags = [] if supported else ['outside_supported_range']
        if supported and value + radius > maximum:
            flags.append('interval_exceeds_simulated_range')
        rows.append({'metal': metal, 'model': info['family'], 'feature_recipe': info['recipe'],
                     'estimate_ug_L_simulated': value if supported else None,
                     'raw_prediction': value if np.isfinite(value) else None,
                     'lower_90': max(0., value - radius) if supported else None,
                     'upper_90': value + radius if supported else None,
                     'status': 'estimate' if supported else 'outside_supported_range', 'flags': flags})
    return {'status': 'accepted', 'training_data': 'SYNTHETIC ONLY', 'model_version': config['version'],
            'warning': 'Simulated chemistry only; physical-strip validation is pending.',
            'interval_note': config['interval_method'], 'qc': qc, 'predictions': rows}, corrected


def generate_v2_audit():
    """New synthetic sessions generated only after the selection bundle exists."""
    config, _, _ = load_improved()
    rng = np.random.default_rng(V2_SEED + 10000)
    folder = V2_DIR / 'audit_images'
    folder.mkdir(exist_ok=True)
    rows, checks = [], []
    for session in range(20):
        shifted = session >= 10
        scenario = 'fresh_shift' if shifted else 'fresh_nominal'
        lot = rng.normal(0, 6. if shifted else 1.5, (7, 3))
        gain = rng.normal(1, .055 if shifted else .025, 3)
        for profile in range(5):
            fractions = rng.uniform(.01, .99, 7)
            card = render_card(assay_colors(fractions, rng.uniform(6, 8), rng.uniform(20, 30), lot), rng)
            for capture in range(2):
                image_id = f'AUDIT-S{session:02d}-P{profile}-C{capture}'
                encoded, _ = camera_capture(card, rng, gain)
                if shifted:
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)
                    image = np.clip(image * rng.uniform(.91, 1.09, 3), 0, 255)
                    image = (255 * (image / 255) ** rng.uniform(.88, 1.12)).astype(np.uint8)
                    encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]
                path = folder / f'{image_id}.jpg'
                path.write_bytes(encoded.tobytes())
                item = {'image_id': image_id, 'session_id': f'AUDIT-S{session:02d}', 'sample_id': f'AUDIT-S{session:02d}-P{profile}',
                        'scenario': scenario, 'partition': 'fresh_audit', 'image_path': path.relative_to(ROOT).as_posix(),
                        'is_synthetic': True, **dict(zip(TARGETS, fractions * MAXIMA))}
                try:
                    features, _, _ = check_image(path, config['quality_limits'])
                    item.update(features)
                    checks.append({'image_id': image_id, 'scenario': scenario, 'accepted': True, 'reason': ''})
                except ImageRejected as error:
                    checks.append({'image_id': image_id, 'scenario': scenario, 'accepted': False, 'reason': '; '.join(error.reasons)})
                rows.append(item)
    audit = pd.DataFrame(rows)
    audit.to_csv(V2_DIR / 'fresh_audit.csv', index=False)
    pd.DataFrame(checks).to_csv(V2_DIR / 'fresh_audit_qc.csv', index=False)
    return audit.dropna(subset=FEATURES)


def evaluate_improved(audit):
    config, models, domain = load_improved()
    _, _, heldout = v2_split(v2_data())
    evaluation = pd.concat([heldout, audit], ignore_index=True)
    rows, predictions, all_predictions = [], [], []
    for metal, target, maximum in zip(METALS, TARGETS, MAXIMA):
        candidates = {'selected_v2': models[metal]}
        for family in ['PLS', 'RandomForest', 'XGBoost', 'RBF_SVR']:
            candidates[f'v1_{family}'] = joblib.load(ROOT / 'scenario_study/models' / f'{metal}_{family}.joblib')
            candidates[f'v2_{family}'] = joblib.load(V2_MODELS / f'{metal}_{family}.joblib')
        for (partition, scenario), part in evaluation.groupby(['partition', 'scenario']):
            distance = domain['neighbors'].kneighbors(domain['scaler'].transform(part[domain['columns']]))[0][:, 0]
            for name, model in candidates.items():
                pred = np.asarray(model.predict(part[FEATURES])).reshape(-1)
                truth = part[target].to_numpy()
                rmse = mean_squared_error(truth, pred) ** .5
                rows.append({'partition': partition, 'scenario': scenario, 'metal': metal, 'model': name,
                             'n_images': len(part), 'MAE': mean_absolute_error(truth, pred), 'RMSE': rmse,
                             'R2': r2_score(truth, pred), 'NRMSE_percent_full_scale': 100 * rmse / maximum})
                all_predictions.extend({'image_id': image_id, 'session_id': session_id, 'scenario': scenario,
                                        'partition': partition, 'metal': metal, 'model': name, 'actual': a, 'predicted': p}
                                       for image_id, session_id, a, p in zip(part.image_id, part.session_id, truth, pred))
                if name == 'selected_v2':
                    radius = config['metals'][metal]['interval_radius_ug_L_simulated']
                    predictions.extend({'image_id': image_id, 'session_id': session_id, 'scenario': scenario,
                                        'partition': partition, 'metal': metal, 'actual': a, 'predicted': p,
                                        'interval_covered': bool(abs(a - p) <= radius), 'domain_supported': bool(d <= domain['threshold']),
                                        'range_supported': bool(0 <= p <= maximum)}
                                       for image_id, session_id, a, p, d in zip(part.image_id, part.session_id, truth, pred, distance))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(V2_DIR / 'comparison_metrics.csv', index=False)
    pd.DataFrame(all_predictions).to_csv(V2_DIR / 'all_predictions.csv', index=False)
    pred = pd.DataFrame(predictions)
    pred.to_csv(V2_DIR / 'selected_predictions.csv', index=False)
    coverage = pred.groupby(['scenario', 'metal']).agg(image_coverage=('interval_covered', 'mean'), domain_acceptance=('domain_supported', 'mean'), range_acceptance=('range_supported', 'mean'))
    sessions = pred.groupby(['scenario', 'metal', 'session_id']).interval_covered.all().groupby(['scenario', 'metal']).mean()
    coverage['session_coverage'] = sessions
    coverage.to_csv(V2_DIR / 'coverage_and_acceptance.csv')
    v2_bootstrap()
    v2_report(metrics, config)
    return metrics


def v2_bootstrap():
    """Paired session bootstrap; preserve dependence between captures and metals."""
    predictions = pd.read_csv(V2_DIR / 'all_predictions.csv')
    predictions = predictions[predictions.partition == 'fresh_audit'].copy()
    scale = predictions.metal.map(dict(zip(METALS, MAXIMA)))
    predictions['squared_normalized_error'] = ((predictions.predicted - predictions.actual) / scale) ** 2
    rng = np.random.default_rng(V2_SEED + 900)
    rows = []
    for scenario, part in predictions.groupby('scenario'):
        old = part[part.model == 'v1_XGBoost'].pivot_table(index='session_id', columns='metal', values='squared_normalized_error', aggfunc='mean').reindex(columns=METALS)
        new = part[part.model == 'selected_v2'].pivot_table(index='session_id', columns='metal', values='squared_normalized_error', aggfunc='mean').reindex(index=old.index, columns=METALS)
        indices = rng.integers(0, len(old), size=(2000, len(old)))
        changes = 100 * (np.sqrt(new.to_numpy()[indices].mean(axis=1)).mean(axis=1)
                         - np.sqrt(old.to_numpy()[indices].mean(axis=1)).mean(axis=1))
        low, high = np.percentile(changes, [2.5, 97.5])
        delta = 100 * (np.sqrt(new.mean()).mean() - np.sqrt(old.mean()).mean())
        rows.append({'scenario': scenario, 'sessions': len(old), 'paired_resamples': 2000,
                     'NRMSE_change_percentage_points': delta, 'lower_95': low, 'upper_95': high})
    pd.DataFrame(rows).to_csv(V2_DIR / 'fresh_audit_bootstrap.csv', index=False)


def v2_report(metrics, config):
    from matplotlib import pyplot as plt
    summary = metrics.groupby(['scenario', 'model']).NRMSE_percent_full_scale.mean().unstack()
    summary.to_csv(V2_DIR / 'summary.csv')
    lines = ['# Improved synthetic models (v2)', '',
             'All results concern simulated chemistry. Earlier test/challenge results informed the decision to investigate gaps; '
             'their reuse is exploratory. A fresh 200-image audit is generated after model selection, using new sessions and seed. '
             'It remains a test of the same assumed chemistry. Fresh audit concentrations are independent uniform fractions '
             'of each simulated range. "Fresh nominal" describes nominal imaging conditions, not identical concentration distributions.', '',
             'V2 reserves 24 of the 120 original development sessions for interval calibration; fits use the other 96. '
             'V1 used all 120 for fitting. Thus these versions differ in both training method and calibration allocation.', '',
             'Selection compares all-pad features, own-pad RGB, neighboring-pad RGB, and own-pad RGB with assumed nuisance augmentation. '
             'Three grouped training folds score clean and perturbed validation images equally. Scaling is fitted within each fold. '
             'Calibration and evaluation labels never select recipes or parameters.', '',
             'Neighbor features use the simulator’s cyclic cross-response mapping (including the first/last pad pair). '
             'This mapping is an assumed mathematical relationship, not measured reagent interference.', '',
             '| Metal | Selected family | Feature recipe |', '|---|---|---|']
    for metal, info in config['metals'].items():
        lines.append(f"| {metal} | {info['family']} | {info['recipe']} |")
    lines += ['', '| Scenario | Previous XGBoost NRMSE % | Selected v2 NRMSE % |', '|---|---:|---:|']
    for scenario, row in summary.iterrows():
        lines.append(f"| {scenario} | {row['v1_XGBoost']:.3f} | {row['selected_v2']:.3f} |")
    lines += ['', '| Family | V2 fresh nominal NRMSE % | V2 fresh shift NRMSE % |', '|---|---:|---:|']
    for family in ['PLS', 'RandomForest', 'XGBoost', 'RBF_SVR']:
        lines.append(f"| {family} | {summary.loc['fresh_nominal', 'v2_' + family]:.3f} | {summary.loc['fresh_shift', 'v2_' + family]:.3f} |")
    intervals = pd.read_csv(V2_DIR / 'coverage_and_acceptance.csv').groupby('scenario')[['image_coverage', 'session_coverage', 'domain_acceptance']].mean()
    lines += ['', '| Scenario | Mean image interval coverage | Mean session coverage | Domain gate acceptance |', '|---|---:|---:|---:|']
    for scenario, row in intervals.iterrows():
        lines.append(f"| {scenario} | {row.image_coverage:.1%} | {row.session_coverage:.1%} | {row.domain_acceptance:.1%} |")
    lines += ['', 'The 90% target is per metal over a new exchangeable session. Coverage under shifted imaging is substantially lower. '
              'The nearest-neighbor domain gate accepts most shifted captures, so passing that check does not demonstrate reliable uncertainty.', '',
              '| Fresh audit | NRMSE change, v2 minus v1 (percentage points) | Approximate paired 95% interval |', '|---|---:|---|']
    for row in pd.read_csv(V2_DIR / 'fresh_audit_bootstrap.csv').itertuples():
        lines.append(f'| {row.scenario} | {row.NRMSE_change_percentage_points:.3f} | [{row.lower_95:.3f}, {row.upper_95:.3f}] |')
    lines += ['', 'The percentile intervals resample whole sessions jointly across all metals (2,000 resamples). '
              'There are only ten sessions per condition. An interval spanning zero does not establish a reliable improvement.']
    lines += ['', 'NRMSE uses simulated full-scale concentrations; it is not percent-correct accuracy. '
              'Tables include all QC-accepted images before domain/range withholding. '
              'Read fresh_audit_qc.csv and coverage_and_acceptance.csv alongside errors. '
              'Intervals are calibrated on clean synthetic sessions only; shifted-domain coverage can fail. '
              'Each fresh audit condition has only ten sessions, so differences are descriptive.', '',
              'Remaining gaps: experimentally measured reagent response, real camera captures, reference-lab labels, '
              'and physical board configuration. Arbitrary pad-specific color offsets can be indistinguishable from concentration changes; '
              'more training cannot eliminate that ambiguity.', '']
    (V2_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11, 5))
    summary[['v1_XGBoost', 'selected_v2']].plot.bar(ax=ax)
    ax.set(ylabel='Mean RMSE / simulated full scale (%)', xlabel='', title='Synthetic model comparison: lower error is better')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(V2_DIR / 'comparison.png', dpi=160)
    plt.close(fig)


def run_improvement():
    train_improved()
    audit = generate_v2_audit()
    evaluate_improved(audit)
    print('Improved model study finished.', flush=True)


if __name__ == '__main__':
    run_improvement()
