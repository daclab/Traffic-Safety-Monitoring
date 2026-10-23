"""Compare raw model outputs across kernels, thread counts and machines.

Round 3 found the 320 W8A8 (1,000-image calibration) model scoring 17.8% mAP
on the board but 31.9% on the PC with the same file and pipeline code, while
the other W8A8 models agree within ~0.4 points. This localizes the cause:

  xnnpack_t4 vs xnnpack_t4_again   nondeterminism (threading, memory)
  xnnpack_t4 vs xnnpack_t1         thread-partitioning effect
  xnnpack_t4 vs reference          XNNPACK kernels vs TFLite reference kernels
  this machine vs --compare file   board vs PC

PC:     python check_numerics.py --models <...> --save reference/pc_outputs.npz
Board:  python check_numerics.py --models <...> --compare reference/pc_outputs.npz
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import pipeline_core as pc

CONFIDENCE = 0.001

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=Path, nargs="+", required=True)
parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "coco300")
parser.add_argument("--images", type=int, default=20)
parser.add_argument("--save", type=Path, default=None)
parser.add_argument("--compare", type=Path, default=None)
args = parser.parse_args()

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
Interpreter = pc.load_interpreter_class()
OpResolverType = sys.modules[Interpreter.__module__].OpResolverType
items = json.loads((args.data / "manifest.json").read_text())["images"][: args.images]

CONFIGS = {
    "xnnpack_t4": {"num_threads": 4},
    "xnnpack_t4_again": {"num_threads": 4},
    "xnnpack_t1": {"num_threads": 1},
    "reference": {
        "num_threads": 1,
        "experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    },
}


def key(model):
    return f"{model.parent.name}/{model.name}"


def run(model, options):
    interpreter = Interpreter(model_path=str(model), **options)
    interpreter.allocate_tensors()
    io = pc.ModelIO(interpreter)
    prepare = pc.make_preparer("cv2-reduced", io, cv2)
    slot = io.new_slot()
    outputs = []
    for item in items:
        prepare(args.data / "images" / item["file_name"], item["width"], item["height"], slot)
        interpreter.set_tensor(io.input_index, slot)
        interpreter.invoke()
        outputs.append(io.dequantize(interpreter.get_tensor(io.output_index))[0])
    return np.stack(outputs), io


def describe(a, b, io):
    """Differences in output units and in quantization steps; candidate counts."""
    diff = np.abs(a - b)
    step = io.output_scale if io.quantized_output else None
    counts = lambda x: float(np.mean([(x[i, 4:].max(0) > CONFIDENCE).sum() for i in range(len(x))]))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_steps": float(diff.max() / step) if step else None,
        "values_differing_pct": float(100 * (diff > 1e-6).mean()),
        "candidates": (counts(a), counts(b)),
    }


saved = {}
other = dict(np.load(args.compare, allow_pickle=False)) if args.compare else {}
for model in args.models:
    if not model.exists():
        print(f"missing {model}; skipped")
        continue
    results = {}
    for name, options in CONFIGS.items():
        results[name], io = run(model, options)
        saved[f"{key(model)}::{name}"] = results[name]
    print(f"\n=== {key(model)}  output quantization {io.describe()['output_quantization']}")
    print(f"{'comparison':40s} {'max abs':>9s} {'mean abs':>10s} {'max steps':>9s} "
          f"{'% differ':>8s} {'candidates a / b':>18s}")
    pairs = [
        ("xnnpack_t4 vs xnnpack_t4_again", results["xnnpack_t4"], results["xnnpack_t4_again"]),
        ("xnnpack_t4 vs xnnpack_t1", results["xnnpack_t4"], results["xnnpack_t1"]),
        ("xnnpack_t4 vs reference", results["xnnpack_t4"], results["reference"]),
    ]
    for name in ("reference", "xnnpack_t4"):
        remote = other.get(f"{key(model)}::{name}")
        if remote is not None:
            pairs.append((f"{name} vs other machine {name}", results[name], remote))
    for label, a, b in pairs:
        d = describe(a, b, io)
        steps = f"{d['max_steps']:9.1f}" if d["max_steps"] is not None else f"{'-':>9s}"
        print(f"{label:40s} {d['max_abs']:9.4f} {d['mean_abs']:10.6f} {steps} "
              f"{d['values_differing_pct']:8.2f} {d['candidates'][0]:8.1f} / {d['candidates'][1]:7.1f}")

if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, **saved)
    print(f"\nSaved {len(saved)} arrays to {args.save}")
