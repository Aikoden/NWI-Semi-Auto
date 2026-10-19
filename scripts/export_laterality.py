"""Export the user's existing two-class YOLO model, without quantization.

Run in the training environment with torch, ultralytics and onnx installed.
"""
import argparse
import json
from pathlib import Path
import onnx
import torch
from ultralytics import YOLO


class Probabilities(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/laterality.onnx'))
    args = parser.parse_args()
    yolo = YOLO(str(args.weights))
    if yolo.task != 'classify' or yolo.names != {0: 'left', 1: 'right'}:
        raise ValueError('Expected classification labels {0: left, 1: right}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = Probabilities(yolo.model.float().eval()).eval()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(args.output),
                      input_names=['images'], output_names=['probabilities'],
                      opset_version=17, dynamo=False)
    exported = onnx.load(str(args.output))
    for key, value in {
        'class_names': json.dumps(['left', 'right']),
        'preprocessing': 'PIL bilinear shortest-edge resize 224; center crop 224; RGB /255',
    }.items():
        prop = exported.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(exported)
    onnx.save(exported, str(args.output))


if __name__ == '__main__':
    main()
