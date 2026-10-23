"""INT8-ACT: export YOLO11n at 320/480/640 and keep every onnx2tf variant.

Ultralytics' int8 export writes four TFLite files into <stem>_saved_model/,
but only *_int8.tflite (weight-only, dynamic range) was deployed so far. This
script copies all of them under clear names and records their I/O dtypes,
quantization parameters and op/tensor composition.

Run in the yolo-tflite conda environment (Python 3.10, Ultralytics 8.4.45):
    python experiments/int8_activation_w8a8_tflite_local/export_int8_variants.py
    python ... --sizes 320          # one resolution only
    python ... --sizes 320 --calibration-images 1000   # -> artifacts/320_cal1000/
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

import tensorflow as tf
import ultralytics
import yaml
from ultralytics import YOLO


DEFAULT_CALIBRATION_IMAGES = 300
COCO_TRAIN_IMAGES = 118_287
# onnx2tf suffix -> name used in artifacts/ and on the board
VARIANTS = {
    "int8": "weightonly_int8",           # dynamic range: int8 weights, float activations
    "integer_quant": "w8a8_floatio",     # int8 activations, float32 input/output
    "full_integer_quant": "w8a8_int8io",  # int8 activations, int8 input/output
    "float32": "float32",                # unquantized reference
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    ops = collections.Counter(op["op_name"] for op in interpreter._get_ops_details())
    dtypes = collections.Counter(
        t["dtype"].__name__ for t in interpreter.get_tensor_details()
    )
    return {
        "input_shape": inp["shape"].tolist(),
        "input_dtype": inp["dtype"].__name__,
        "input_quantization": [float(v) for v in inp["quantization"]],
        "output_shape": out["shape"].tolist(),
        "output_dtype": out["dtype"].__name__,
        "output_quantization": [float(v) for v in out["quantization"]],
        "tensor_dtypes": dict(dtypes),
        "operator_types": dict(ops),
        "operator_count": sum(ops.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 480, 640])
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="PyTorch checkpoint to export (default: official yolo11n.pt)")
    parser.add_argument("--tag", default="",
                        help="Suffix for the artifact folder, e.g. _ft320 -> artifacts/320_ft320/")
    parser.add_argument("--calibration-images", type=int, default=DEFAULT_CALIBRATION_IMAGES,
                        help="Non-default values write to artifacts/<size>_cal<N>/")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    project_root = experiment_dir.parent
    checkpoint = (args.checkpoint or project_root / "yolo11n.pt").resolve()
    coco_root = project_root / "datasets" / "coco"
    artifacts_root = experiment_dir / "artifacts"
    calibration_yaml = experiment_dir / "coco_train2017_calibration.yaml"

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Expected Python 3.10, found {sys.version.split()[0]}")
    for required in (checkpoint, coco_root / "train2017.txt"):
        if not required.exists():
            raise FileNotFoundError(required)

    # Same calibration setup as baseline_silu_multires_w8a8_tflite_local.
    official_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    data_config = yaml.safe_load(official_yaml.read_text(encoding="utf-8"))
    data_config["path"] = coco_root.as_posix()
    data_config["train"] = "train2017.txt"
    data_config["val"] = "train2017.txt"
    data_config["test"] = None
    calibration_yaml.write_text(
        yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    checkpoint_hash = sha256(checkpoint)

    for size in args.sizes:
        print(f"\n{'=' * 72}\nExporting YOLO11n SiLU at {size} x {size}\n{'=' * 72}")
        exported = Path(
            YOLO(str(checkpoint)).export(
                format="tflite",
                int8=True,
                data=str(calibration_yaml),
                fraction=args.calibration_images / COCO_TRAIN_IMAGES,
                imgsz=size,
                batch=1,
                dynamic=False,
                nms=False,
                device="cpu",
            )
        ).resolve()
        saved_model_dir = exported.parent
        dir_suffix = ("" if args.calibration_images == DEFAULT_CALIBRATION_IMAGES
                      else f"_cal{args.calibration_images}")
        artifact_dir = artifacts_root / f"{size}{dir_suffix}{args.tag}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "experiment_id": f"INT8-ACT-{size}{dir_suffix}{args.tag}",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_hash,
            "ultralytics": ultralytics.__version__,
            "tensorflow": tf.__version__,
            "calibration_source": "COCO train2017",
            "calibration_images": args.calibration_images,
            "input_size": size,
            "variants": {},
        }
        for suffix, name in VARIANTS.items():
            source = saved_model_dir / f"{checkpoint.stem}_{suffix}.tflite"
            if not source.exists():
                raise FileNotFoundError(f"onnx2tf did not produce {source}")
            destination = artifact_dir / f"yolo11n_silu_{size}_{name}.tflite"
            shutil.copy2(source, destination)
            info = inspect(destination)
            if info["input_shape"] != [1, size, size, 3]:
                raise RuntimeError(f"{destination} has input {info['input_shape']}")
            metadata["variants"][name] = {
                "onnx2tf_suffix": suffix,
                "artifact": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                **info,
            }
            print(f"{name:16s} in={info['input_dtype']:8s} out={info['output_dtype']:8s} "
                  f"tensors={info['tensor_dtypes']}  {destination.stat().st_size / 2**20:.2f} MiB")

        (artifact_dir / "conversion_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Artifacts: {artifact_dir}")

    print("\nExports complete. Next: run_pc_accuracy.ps1 for COCO-300 accuracy.")


if __name__ == "__main__":
    main()
