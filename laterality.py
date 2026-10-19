"""Optional left/right suggestion; never changes measurement geometry."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from algorithms import find_knee_split


def preprocess(image_bgr):
    """Match Ultralytics classification: PIL resize, center crop, RGB / 255."""
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    width, height = image.size
    size = 224
    resized = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left = int(round((resized[0] - size) / 2.0))
    top = int(round((resized[1] - size) / 2.0))
    image = image.crop((left, top, left + size, top + size))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def multiple_knees(mask):
    """Conservative screen for separated, similarly sized femoral components."""
    return find_knee_split(mask) is not None


class LateralityModel:
    def __init__(self, path=None):
        self.session = None
        path = path or os.environ.get('NWI_LATERALITY_MODEL_PATH') or str(Path(__file__).parent / 'models/laterality.onnx')
        if not Path(path).is_file():
            return
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
            names = json.loads(session.get_modelmeta().custom_metadata_map.get('class_names', 'null'))
            if names != ['left', 'right'] or session.get_inputs()[0].shape != [1, 3, 224, 224]:
                return
            self.session = session
        except Exception:
            # Missing/invalid optional classifier must not block NWI measurement.
            self.session = None

    def predict(self, image_bgr, femur_mask):
        unknown = {'side': None, 'confidence': None, 'status': 'unavailable'}
        if multiple_knees(femur_mask):
            return {**unknown, 'status': 'multiple_knees'}
        if self.session is None or image_bgr is None or femur_mask is None:
            return unknown
        try:
            probabilities = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: preprocess(image_bgr)})[0]).reshape(-1)
            if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.isclose(probabilities.sum(), 1, atol=1e-3)):
                return {**unknown, 'status': 'invalid_output'}
            index = int(probabilities.argmax())
            confidence = float(probabilities[index])
            # An operational threshold, not a calibrated probability of correctness.
            return {'side': ['left', 'right'][index] if confidence >= 0.9 else None,
                    'confidence': confidence,
                    'status': 'suggested' if confidence >= 0.9 else 'low_confidence'}
        except Exception:
            return unknown
