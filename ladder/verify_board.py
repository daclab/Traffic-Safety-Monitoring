"""Verify files, dependencies and a real inference; no camera required."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo.resolve()
    import cv2
    import numpy as np
    import PIL
    import msgpack
    import pycocotools.coco
    from ai_edge_litert.interpreter import Interpreter
    print('Platform:', platform.machine(), 'Python:', platform.python_version())
    for name in ('ai-edge-litert', 'numpy', 'opencv-python-headless', 'Pillow', 'msgpack', 'pycocotools'):
        print(name, importlib.metadata.version(name))
    if not hasattr(cv2.dnn, 'NMSBoxesBatched'):
        raise RuntimeError('OpenCV lacks native NMSBoxesBatched; install the pinned requirements.')
    manifest = json.loads((root / 'data/coco300/manifest.json').read_text())
    images = manifest['images']
    if len(images) != 300 or len({i['id'] for i in images}) != 300:
        raise RuntimeError('Expected exactly 300 distinct manifest image IDs.')
    missing = [i['file_name'] for i in images if not (root / 'data/coco300/images' / i['file_name']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} JPEGs missing; first: {missing[0]}')
    annotation = json.loads((root / 'ladder/kit/coco300_instances_val2017.json').read_text())
    if not {i['id'] for i in images}.issubset({i['id'] for i in annotation['images']}):
        raise RuntimeError('Annotation file does not cover the manifest IDs.')
    for filename, size, dtype in (
        ('yolo11n_silu_640_float32.tflite', 640, np.float32),
        ('yolo11n_silu_320_float32.tflite', 320, np.float32),
        ('yolo11n_silu_320_w8a8_int8io.tflite', 320, np.int8),
    ):
        model = Interpreter(model_path=str(root / 'models' / filename), num_threads=1)
        model.allocate_tensors()
        inp, out = model.get_input_details()[0], model.get_output_details()[0]
        if inp['shape'].tolist() != [1, size, size, 3] or inp['dtype'] != dtype:
            raise RuntimeError('Unexpected input shape/dtype: ' + filename)
        if out['shape'].tolist() != [1, 84, (size // 8)**2 + (size // 16)**2 + (size // 32)**2]:
            raise RuntimeError('Unexpected output shape: ' + filename)
        model.set_tensor(inp['index'], np.zeros(inp['shape'], dtype=dtype))
        model.invoke()
        if not np.isfinite(model.get_tensor(out['index'])).all():
            raise RuntimeError('Nonfinite model output: ' + filename)
        print('PASS model:', filename)
        del model
    print('PASS: dependencies, 300 image files, annotation IDs and all three models.')
    print('Camera and MCU/LED checks are separate; this verification needs neither.')


if __name__ == '__main__':
    main()
