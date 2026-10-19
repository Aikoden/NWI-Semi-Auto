"""Regression: a large hollow tibial outline must not beat solid femur pixels."""
import unittest
import os
from pathlib import Path
import cv2
import numpy as np
from algorithms import AIModel, detect_baseline, full_postprocessing_pipeline


class ComponentSelectionTests(unittest.TestCase):
    def test_solid_femur_wins_over_larger_hollow_outline(self):
        mask = np.zeros((400, 240), dtype=np.uint8)
        cv2.rectangle(mask, (60, 10), (180, 120), 255, -1)
        cv2.rectangle(mask, (30, 150), (210, 390), 255, 2)
        contour, _ = full_postprocessing_pipeline(mask)
        self.assertIsNotNone(contour)
        self.assertLess(contour[:, 0, 1].max(), 140,
                        'Selected hollow tibial outline instead of solid femur')

    def test_holes_in_selected_component_are_still_filled(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (180, 180), 255, -1)
        cv2.circle(mask, (100, 100), 15, 0, -1)
        contour, _ = full_postprocessing_pipeline(mask)
        self.assertGreater(cv2.pointPolygonTest(contour, (100., 100.), False), 0)

    def test_empty_mask_returns_no_contour(self):
        self.assertEqual(full_postprocessing_pipeline(np.zeros((100, 100), np.uint8)),
                         (None, None))


@unittest.skipUnless(os.environ.get('NWI_MODEL_PATH'), 'Set NWI_MODEL_PATH for real-model regression')
class SampleRegressionTests(unittest.TestCase):
    def test_second_sample_selects_femoral_condyles(self):
        image = cv2.imread(str(Path(__file__).resolve().parents[1] / 'samples/Normal-107.jpg'))
        mask = AIModel().predict_mask(image)
        self.assertIsNotNone(mask)
        _, contour = full_postprocessing_pipeline(mask)
        baseline = detect_baseline(contour)
        self.assertIsNotNone(baseline)
        # Broad, image-specific bounds distinguish the visible femoral condyles
        # from the distal tibial shaft picked by the old contour-area ranking.
        for point in (baseline['P1'], baseline['P2']):
            self.assertGreater(point[1], image.shape[0] * 0.20)
            self.assertLess(point[1], image.shape[0] * 0.35)
        self.assertGreater(baseline['P2'][0] - baseline['P1'][0], image.shape[1] * 0.25)


if __name__ == '__main__':
    unittest.main()
