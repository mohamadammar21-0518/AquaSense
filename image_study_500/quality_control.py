"""Reject images outside the supported capture conditions before returning estimates."""
import json
from pathlib import Path
import cv2
import numpy as np
from image_pipeline import ROOT, SPOTS, REF_CENTERS, preprocess_image

DEFAULT_LIMITS = {'min_sharpness': 35., 'max_gray_error': 8., 'max_pad_spread': 25.,
                  'max_glare_fraction': .01, 'max_dark_fraction': .02, 'max_angle_deg': 20.}

class ImageRejected(ValueError):
    def __init__(self, reasons, qc=None):
        self.reasons, self.qc = list(reasons), qc or {}
        super().__init__('; '.join(self.reasons))

def check_image(path, limits=None):
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ImageRejected(['Image cannot be decoded. Upload a JPEG or PNG.'])
    if bgr.shape[:2] != (480, 640):
        raise ImageRejected(['Capture at 640 × 480 pixels; other sizes are not calibrated.'])
    try:
        features, rectified, corrected, points, qc = preprocess_image(Path(path))
    except (ValueError, cv2.error) as error:
        raise ImageRejected(['Card alignment or reference patches failed. Check orientation, lighting and all four markers.']) from error
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    qc['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    qc['angle_deg'] = float(abs(np.degrees(np.arctan2(*(points[1]-points[0])[::-1]))))
    qc['card_aspect'] = float(np.linalg.norm(points[1]-points[0])/np.linalg.norm(points[3]-points[0]))
    pixels = []
    spreads = []
    yy, xx = np.ogrid[:280, :560]
    for x, y in SPOTS:
        roi = (xx-x)**2+(yy-y)**2 <= 16**2
        values = rectified[roi]
        pixels.append(values)
        spreads.append(float(np.max(np.percentile(values, 95, axis=0)-np.percentile(values, 5, axis=0))))
    pixels = np.concatenate(pixels)
    qc['glare_fraction'] = float(np.mean(np.all(pixels >= 250, axis=1)))
    qc['dark_fraction'] = float(np.mean(np.all(pixels <= 15, axis=1)))
    qc['max_pad_spread'] = max(spreads)
    reasons = []
    if qc['fiducial_candidates'] != 4:
        reasons.append('Exactly four alignment markers are required; remove distracting dark shapes.')
    if not 1.7 <= qc['card_aspect'] <= 2.8 or qc['angle_deg'] > limits['max_angle_deg']:
        reasons.append('Card orientation or perspective is outside the supported capture range.')
    if qc['sharpness'] < limits['min_sharpness']:
        reasons.append('Image is too blurry. Refocus and keep the camera still.')
    if qc['gray_reference_abs_error'] > limits['max_gray_error']:
        reasons.append('Color reference check failed. Use even lighting and visible reference patches.')
    if qc['glare_fraction'] > limits['max_glare_fraction']:
        reasons.append('Glare or overexposure covers a reaction spot.')
    if qc['dark_fraction'] > limits['max_dark_fraction']:
        reasons.append('A reaction spot is obstructed or severely underexposed.')
    if qc['max_pad_spread'] > limits['max_pad_spread']:
        reasons.append('A reaction spot has excessive shadow, texture or obstruction.')
    if reasons:
        raise ImageRejected(reasons, qc)
    return features, corrected, qc

def calibrate_quality(labels):
    """Read proper-training images only; never select limits using test outcomes."""
    sharpness, spreads, gray_errors = [], [], []
    permissive = {**DEFAULT_LIMITS, 'min_sharpness': 0, 'max_pad_spread': 1000, 'max_gray_error': 1000}
    for row in labels.itertuples():
        _, _, qc = check_image(ROOT/row.image_path, permissive)
        sharpness.append(qc['sharpness']); spreads.append(qc['max_pad_spread']); gray_errors.append(qc['gray_reference_abs_error'])
    return {**DEFAULT_LIMITS, 'min_sharpness': float(max(10, np.quantile(sharpness, .01)*.45)),
            'max_pad_spread': float(max(18, np.quantile(spreads, .99)*2)),
            'max_gray_error': float(max(8, np.quantile(gray_errors, .99)*2))}
