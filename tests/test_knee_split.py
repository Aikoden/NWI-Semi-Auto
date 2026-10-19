"""Bilateral images are split between the two femurs and measured knee by knee."""
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from algorithms import find_knee_split

SAMPLES = Path(__file__).resolve().parents[1] / 'samples'


class KneeSplitTests(unittest.TestCase):
    def test_split_falls_in_the_gap_between_two_femurs(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.rectangle(mask, (20, 20), (120, 150), 255, -1)
        cv2.rectangle(mask, (260, 20), (360, 150), 255, -1)
        split = find_knee_split(mask)
        self.assertIsNotNone(split)
        self.assertGreater(split, 120)
        self.assertLess(split, 260)

    def test_split_avoids_femur_pixels_when_bounding_boxes_overlap(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [110, 20], [190, 150], [100, 150]])], 255)
        cv2.fillPoly(mask, [np.array([[290, 20], [380, 20], [300, 150], [210, 150]])], 255)
        split = find_knee_split(mask)
        self.assertIsNotNone(split)
        self.assertEqual(int((mask[:, split] > 0).sum()), 0)

    def test_single_knee_is_not_split(self):
        mask = np.zeros((300, 200), np.uint8)
        cv2.rectangle(mask, (40, 20), (160, 150), 255, -1)
        self.assertIsNone(find_knee_split(mask))

    def test_small_lower_artifact_is_not_a_second_knee(self):
        mask = np.zeros((300, 400), np.uint8)
        cv2.rectangle(mask, (20, 20), (120, 150), 255, -1)
        cv2.rectangle(mask, (260, 230), (280, 260), 255, -1)
        self.assertIsNone(find_knee_split(mask))

    def test_empty_mask_is_not_split(self):
        self.assertIsNone(find_knee_split(None))
        self.assertIsNone(find_knee_split(np.zeros((50, 50), np.uint8)))


@unittest.skipUnless(os.environ.get('NWI_MODEL_PATH') and importlib.util.find_spec('fastapi'),
                     'Set NWI_MODEL_PATH and install fastapi for the web regression')
class WebBilateralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from web import server
        cls.server = server

    def analyse(self, name):
        return self.server.analyse(cv2.imread(str(SAMPLES / name)))

    def test_bilateral_sample_returns_one_measurement_per_knee(self):
        result = self.analyse('Normal-221.jpg')
        knees = result['knees']
        self.assertEqual([k['position'] for k in knees], ['screen_left', 'screen_right'])
        self.assertEqual(knees[0]['region'][0], 0)
        self.assertEqual(knees[0]['region'][1], knees[1]['region'][0])
        self.assertEqual(knees[1]['region'][1], result['width'])
        for knee in knees:
            x0, x1 = knee['region']
            xs = [p[0] for p in knee['contour']]
            self.assertGreaterEqual(min(xs), x0)
            self.assertLessEqual(max(xs), x1)
            for point in (knee['baseline']['P1'], knee['baseline']['P2']):
                self.assertGreater(point[0], x0)
                self.assertLess(point[0], x1)
                # Femoral condyles of this sample sit near the middle of the image height.
                self.assertGreater(point[1], result['height'] * 0.35)
                self.assertLess(point[1], result['height'] * 0.55)
        if self.server.get_laterality_model().session is not None:
            self.assertEqual([k['laterality']['side'] for k in knees], ['right', 'left'])

    def test_single_knee_keeps_the_previous_response_fields(self):
        result = self.analyse('Normal-107.jpg')
        self.assertEqual([k['position'] for k in result['knees']], ['single'])
        for field in ('contour', 'baseline', 'laterality'):
            self.assertEqual(result[field], result['knees'][0][field])

    def test_same_side_for_both_knees_is_not_suggested(self):
        same_side = SimpleNamespace(predict=lambda *args: {'side': 'left', 'confidence': 1.0, 'status': 'suggested'})
        with mock.patch.object(self.server, 'get_laterality_model', return_value=same_side):
            knees = self.analyse('Normal-221.jpg')['knees']
        for knee in knees:
            self.assertIsNone(knee['laterality']['side'])
            self.assertEqual(knee['laterality']['status'], 'inconsistent')


if __name__ == '__main__':
    unittest.main()
