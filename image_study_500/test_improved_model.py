"""Regression checks for data isolation and the actual v2 prediction endpoint."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from app import create_app
from image_pipeline import ROOT, FEATURES
from improved_model import v2_data, v2_split, v2_perturb, estimate_improved, load_improved
from quality_control import ImageRejected


class ImprovedTests(unittest.TestCase):
    def test_partitions_and_saved_folds_are_isolated(self):
        proper, calibration, heldout = v2_split(v2_data())
        self.assertEqual((len(proper), len(calibration)), (960, 240))
        folds = pd.read_csv(ROOT/'improved_study/validation_folds.csv')
        self.assertEqual(folds.groupby('session_id').validation_fold.nunique().max(), 1)
        self.assertEqual(set(folds.image_id), set(proper.image_id))
        for key in ['session_id', 'sample_id', 'image_id']:
            for a, b in [(proper, calibration), (proper, heldout), (calibration, heldout)]:
                self.assertFalse(set(a[key]) & set(b[key]))

    def test_augmentation_is_reproducible_and_preserves_input(self):
        frame = v2_data().head(4)
        before = frame.copy(deep=True)
        a, b = v2_perturb(frame, 35), v2_perturb(frame, 35)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(np.allclose(a, frame[FEATURES]))
        rgb = a.loc[0, ['Pb_R','Pb_G','Pb_B']].to_numpy(dtype=float)
        hsv = cv2.cvtColor((rgb/255).astype(np.float32).reshape(1,1,3), cv2.COLOR_RGB2HSV)[0,0]
        hsv[1:] *= 100
        np.testing.assert_allclose(a.loc[0, ['Pb_H','Pb_S','Pb_V']], hsv, atol=1e-5)

    def test_domain_rejection(self):
        features = dict.fromkeys(FEATURES, 100000.)
        with patch('improved_model.check_image', return_value=(features, None, {})):
            with self.assertRaises(ImageRejected):
                estimate_improved('not-read.jpg')

    def test_real_model_endpoint_and_rejected_image(self):
        runtime = ROOT/'runtime'
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime, prefix='v2-test-') as folder:
            client = create_app(storage=folder, token='test-token').test_client()
            self.assertEqual(client.get('/api/health').status_code, 401)
            headers = {'X-API-Key': 'test-token'}
            health = client.get('/api/health', headers=headers)
            self.assertEqual(health.json['model_version'], 'expanded-v2')
            labels = pd.read_csv(ROOT/'scenario_study/labels.csv')
            row = labels[labels.partition.eq('test') & labels.scenario.eq('standard')].iloc[0]
            reply = client.post('/api/predict', data=(ROOT/row.image_path).read_bytes(), content_type='image/jpeg', headers=headers)
            self.assertEqual(reply.status_code, 200, reply.json)
            self.assertEqual(reply.json['model_version'], 'expanded-v2')
            self.assertEqual(len(reply.json['predictions']), 7)
            self.assertTrue(all('feature_recipe' in r for r in reply.json['predictions']))
            blank = io.BytesIO()
            Image.new('RGB', (640, 480), 'white').save(blank, format='PNG')
            rejected = client.post('/api/predict', data=blank.getvalue(), content_type='image/png', headers=headers)
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(rejected.json['predictions'], [])
            history = client.get('/api/history', headers=headers).json
            self.assertEqual(len(history), 2)


if __name__ == '__main__':
    unittest.main()
