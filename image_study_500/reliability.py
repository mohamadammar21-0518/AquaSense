"""Validation-selected models, independent calibration and synthetic-domain checks."""
import json
import hashlib
import math
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from image_pipeline import ROOT, SEED, METALS, MAXIMA, TARGETS, FEATURES, model_spaces
from quality_control import check_image, calibrate_quality, ImageRejected

RELIABLE_DIR = ROOT/'models/reliable'
RGB_FEATURES = [f'{metal}_{c}' for metal in METALS for c in ['R', 'G', 'B']]

def load_frame():
    labels = pd.read_csv(ROOT/'datasets/labels.csv')
    features = pd.read_csv(ROOT/'datasets/features.csv')
    return labels.merge(features[['image_id', *FEATURES]], on='image_id', validate='one_to_one')

def reliability_partition(frame):
    train = frame[frame.partition == 'train'].reset_index(drop=True)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED+73).split(train, groups=train.session_id))
    proper, calibration = train.iloc[a].copy(), train.iloc[b].copy()
    test = frame[frame.partition == 'test'].copy()
    assert [part.session_id.nunique() for part in [proper, calibration, test]] == [30, 10, 10]
    assert not set(proper.session_id) & set(calibration.session_id)
    assert not set(test.session_id) & (set(proper.session_id) | set(calibration.session_id))
    return proper, calibration, test

def small_spaces():
    """Prespecified compact search, independent of any test result."""
    spaces = model_spaces()
    spaces['PLS'][1]['regressor__model__n_components'] = [4, 8]
    spaces['RandomForest'][1]['regressor__model__max_depth'] = [None]
    spaces['XGBoost'][1]['regressor__model__reg_lambda'] = [1.]
    spaces['RBF_SVR'][1]['regressor__model__gamma'] = ['scale']
    spaces['RBF_SVR'][1]['regressor__model__C'] = [10., 100.]
    return spaces

def fit_domain(proper):
    scaler = StandardScaler().fit(proper[RGB_FEATURES])
    X = scaler.transform(proper[RGB_FEATURES])
    # Same-session captures are excluded from the reference distance calibration.
    distances = []
    for session in proper.session_id.unique():
        mask = (proper.session_id == session).to_numpy()
        nn = NearestNeighbors(n_neighbors=1).fit(X[~mask])
        distances.extend(nn.kneighbors(X[mask])[0].ravel())
    threshold = float(np.quantile(distances, .99)*1.5)
    return {'scaler': scaler, 'neighbors': NearestNeighbors(n_neighbors=1).fit(X),
            'threshold': threshold, 'columns': RGB_FEATURES}

def train_reliable():
    RELIABLE_DIR.mkdir(parents=True, exist_ok=True)
    proper, calibration, test = reliability_partition(load_frame())
    parts = []
    for name, part in [('proper_train', proper), ('calibration', calibration), ('test', test)]:
        part = part[['image_id', 'sample_id', 'session_id']].copy(); part['role'] = name; parts.append(part)
    pd.concat(parts).to_csv(ROOT/'results/reliability_split.csv', index=False)
    limits = calibrate_quality(proper)
    selected, searches, calibration_rows = {}, [], []
    folds = list(GroupKFold(n_splits=3).split(proper, groups=proper.session_id))
    for metal, target in zip(METALS, TARGETS):
        candidates = []
        for family, (model, grid) in small_spaces().items():
            print(f'Reliability training: {metal} / {family}', flush=True)
            search = GridSearchCV(model, grid, cv=folds, scoring='neg_root_mean_squared_error', n_jobs=4, error_score='raise')
            with threadpool_limits(limits=1), joblib.parallel_backend('threading'):
                search.fit(proper[FEATURES], proper[target])
            candidates.append((search.best_score_, family, search.best_estimator_, search.best_params_))
            searches.append({'metal': metal, 'model': family, 'validation_RMSE': -search.best_score_, 'best_parameters': json.dumps(search.best_params_)})
        score, family, fitted, params = max(candidates, key=lambda item: item[0])
        prediction = np.asarray(fitted.predict(calibration[FEATURES])).reshape(-1)
        residuals = np.abs(calibration[target].to_numpy()-prediction)
        # Cluster-level conformal score: maximum error across all 10 images of a session.
        session_scores = pd.DataFrame({'session': calibration.session_id, 'error': residuals}).groupby('session').error.max().to_numpy()
        alpha = .10
        rank = math.ceil((len(session_scores)+1)*(1-alpha))
        radius = float(np.sort(session_scores)[rank-1])
        joblib.dump(fitted, RELIABLE_DIR/f'{metal}.joblib')
        selected[metal] = {'family': family, 'validation_RMSE': -score, 'interval_radius_ug_L_simulated': radius,
                           'nominal_coverage': .90, 'calibration_sessions': len(session_scores)}
        calibration_rows.extend({'metal': metal, 'session_id': s, 'session_max_abs_error': v} for s,v in zip(sorted(calibration.session_id.unique()), session_scores))
    pd.DataFrame(searches).to_csv(ROOT/'results/reliability_validation.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(ROOT/'results/calibration_session_scores.csv', index=False)
    joblib.dump(fit_domain(proper), RELIABLE_DIR/'domain.joblib')
    config = {'training_data': 'SYNTHETIC ONLY', 'version': 'reliable-v1', 'proper_training_sessions': 30,
              'calibration_sessions': 10, 'test_sessions': 10, 'quality_limits': limits, 'metals': selected,
              'interval_method': '90% split-conformal with maximum absolute residual per calibration session, per metal; assumes exchangeable synthetic sessions of ten images. No real-world coverage claim.',
              'domain_check': 'Training-only standardized RGB nearest-neighbor distance. Heuristic, not proof of in-domain input.'}
    config['model_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RELIABLE_DIR.glob('*.joblib')}
    (RELIABLE_DIR/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Reliable models and independent calibration saved.', flush=True)
    return config

@lru_cache(maxsize=1)
def load_reliable():
    config = json.loads((RELIABLE_DIR/'config.json').read_text(encoding='utf-8'))
    # Reject an incomplete or changed bundle rather than mixing model versions.
    for name, digest in config['model_sha256'].items():
        if hashlib.sha256((RELIABLE_DIR/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Model bundle checksum mismatch. Rebuild the reliability bundle.')
    return config, {metal: joblib.load(RELIABLE_DIR/f'{metal}.joblib') for metal in METALS}, joblib.load(RELIABLE_DIR/'domain.joblib')

def estimate_checked(image_path):
    config, models, domain = load_reliable()
    features, corrected, qc = check_image(image_path, config['quality_limits'])
    X = pd.DataFrame([features], columns=FEATURES)
    scaled = domain['scaler'].transform(X[domain['columns']])
    distance = float(domain['neighbors'].kneighbors(scaled)[0][0,0])
    qc['domain_distance'] = distance; qc['domain_threshold'] = domain['threshold']
    if distance > domain['threshold']:
        raise ImageRejected(['Color features are outside the supported synthetic training domain. No estimate returned.'], qc)
    rows = []
    for metal, maximum in zip(METALS, MAXIMA):
        value = float(np.asarray(models[metal].predict(X)).reshape(-1)[0])
        info = config['metals'][metal]
        radius = info['interval_radius_ug_L_simulated']
        flags = []
        if not np.isfinite(value): flags.append('nonfinite_prediction')
        elif value < 0: flags.append('below_supported_range')
        elif value > maximum: flags.append('above_supported_range')
        lower, upper = (max(0., value-radius), value+radius) if np.isfinite(value) else (None, None)
        if upper is not None and upper > maximum: flags.append('interval_exceeds_simulated_range')
        supported = not any(f in flags for f in ['nonfinite_prediction','below_supported_range','above_supported_range'])
        rows.append({'metal': metal, 'model': info['family'], 'estimate_ug_L_simulated': value if supported else None,
                     'raw_prediction': value if np.isfinite(value) else None,
                     'lower_90': lower if supported else None, 'upper_90': upper if supported else None,
                     'status': 'estimate' if supported else 'outside_supported_range', 'flags': flags})
    result = {'status': 'accepted', 'training_data': 'SYNTHETIC ONLY', 'model_version': config['version'],
              'warning': 'Demonstration estimates from simulated chemistry. Not validated for real water or safety decisions.',
              'interval_note': config['interval_method'], 'qc': qc, 'predictions': rows}
    return result, corrected

if __name__ == '__main__':
    train_reliable()
