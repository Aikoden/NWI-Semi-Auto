import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from laterality import LateralityModel, multiple_knees, preprocess

class LateralityTests(unittest.TestCase):
    def classify_probabilities(self, probabilities):
        model = LateralityModel('/missing/model')
        model.session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name='images')],
            run=lambda *args: [np.array([probabilities], dtype=np.float32)],
        )
        return model.predict(np.zeros((100, 100, 3), np.uint8), np.ones((100, 100), np.uint8))

    def test_low_confidence_does_not_assign_side(self):
        result = self.classify_probabilities([0.55, 0.45])
        self.assertEqual(result['status'], 'low_confidence')
        self.assertIsNone(result['side'])

    def test_invalid_probabilities_do_not_assign_side(self):
        result = self.classify_probabilities([float('nan'), 1])
        self.assertEqual(result['status'], 'invalid_output')
        self.assertIsNone(result['side'])

    def test_left_right_index_mapping(self):
        self.assertEqual(self.classify_probabilities([0.99, 0.01])['side'], 'left')
        self.assertEqual(self.classify_probabilities([0.01, 0.99])['side'], 'right')

    def test_bilateral_components_are_not_given_one_side(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.rectangle(mask, (20, 20), (120, 150), 255, -1)
        cv2.rectangle(mask, (260, 20), (360, 150), 255, -1)
        result = LateralityModel('/missing/model').predict(np.zeros((300, 400, 3), np.uint8), mask)
        self.assertEqual(result['status'], 'multiple_knees')
        self.assertIsNone(result['side'])

    def test_small_lower_artifact_is_not_a_second_knee(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.rectangle(mask, (20, 20), (120, 150), 255, -1)
        cv2.rectangle(mask, (260, 230), (280, 260), 255, -1)
        self.assertFalse(multiple_knees(mask))

    def test_missing_optional_model_is_nonfatal(self):
        self.assertIsNone(LateralityModel('/missing/model').predict(np.zeros((30, 30, 3), np.uint8), np.ones((30, 30), np.uint8))['side'])

    def test_preprocessing_rgb_and_scale(self):
        image = np.zeros((300, 200, 3), np.uint8)
        image[:, :, 2] = 255
        x = preprocess(image)
        self.assertEqual(x.shape, (1, 3, 224, 224))
        np.testing.assert_array_equal(x[0, 0], 1)
        np.testing.assert_array_equal(x[0, 1:], 0)
