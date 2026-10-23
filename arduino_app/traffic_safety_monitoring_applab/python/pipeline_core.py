"""Shared preprocessing, quantized I/O, decode and NMS for pipeline v2.

Used by the board runner (run_pipeline_v2.py), the thread sweep, and the PC
accuracy check, so every measurement uses identical pre/postprocessing.

Supported models: YOLO detection TFLite exports with NHWC square input and a
single [1, 84, N] output, in any of the onnx2tf variants:
  float32 I/O  (weight-only *_int8, *_integer_quant, *_float32)
  int8 I/O     (*_full_integer_quant)
"""

import math

import numpy as np


PAD_VALUE = 114
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
COCO80_TO_91_ARRAY = np.asarray(COCO80_TO_91, dtype=np.int32)
BACKENDS = ("pil", "cv2", "cv2-reduced", "opencv-opencl", "opencl-zerocopy")
CPU_BACKENDS = ("pil", "cv2", "cv2-reduced")


def load_interpreter_class():
    """Use the actual LiteRT environment; preserve its original import error."""
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter


class ModelIO:
    """Input/output layout and quantization of one interpreter."""

    def __init__(self, interpreter):
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        shape = inp["shape"].tolist()
        if len(shape) != 4 or shape[0] != 1 or shape[1] != shape[2] or shape[3] != 3:
            raise RuntimeError(f"Expected static NHWC square input, got {shape}")
        out_shape = out["shape"].tolist()
        if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] != 84:
            raise RuntimeError(f"Expected output [1,84,N], got {out_shape}")

        self.size = int(shape[1])
        self.candidates = int(out_shape[2])
        self.input_index = inp["index"]
        self.output_index = out["index"]
        self.input_dtype = np.dtype(inp["dtype"])
        self.output_dtype = np.dtype(out["dtype"])
        self.input_scale, self.input_zero = (float(v) for v in inp["quantization"])
        self.output_scale, self.output_zero = (float(v) for v in out["quantization"])
        self.quantized_input = self.input_dtype != np.float32
        self.quantized_output = self.output_dtype != np.float32

        self.lut = None
        self.xor_fast_path = False
        if self.quantized_input:
            if self.input_scale == 0:
                raise RuntimeError("Quantized input without a quantization scale")
            info = np.iinfo(self.input_dtype)
            levels = np.arange(256, dtype=np.float64) / 255.0
            self.lut = np.clip(
                np.round(levels / self.input_scale + self.input_zero), info.min, info.max
            ).astype(self.input_dtype)
            # scale 1/255, zero point -128: int8 value is pixel - 128 == pixel ^ 0x80.
            expected = (np.arange(256) - 128).astype(np.int8)
            self.xor_fast_path = (
                self.input_dtype == np.int8 and np.array_equal(self.lut, expected)
            )
            self.pad = self.lut[PAD_VALUE]
        else:
            self.pad = np.float32(PAD_VALUE / 255.0)

    def describe(self):
        return {
            "input_size": self.size,
            "candidates": self.candidates,
            "input_dtype": self.input_dtype.name,
            "input_quantization": [self.input_scale, self.input_zero],
            "output_dtype": self.output_dtype.name,
            "output_quantization": [self.output_scale, self.output_zero],
            "int8_input_xor_fast_path": self.xor_fast_path,
        }

    def new_slot(self):
        return np.empty((1, self.size, self.size, 3), dtype=self.input_dtype)

    def write_input(self, slot, rgb, left, top):
        """Letterbox an RGB uint8 image into a preallocated input slot."""
        height, width = rgb.shape[:2]
        canvas = slot[0]
        canvas[:top] = self.pad
        canvas[top + height:] = self.pad
        canvas[top:top + height, :left] = self.pad
        canvas[top:top + height, left + width:] = self.pad
        region = canvas[top:top + height, left:left + width]
        if not self.quantized_input:
            np.multiply(rgb, np.float32(1.0 / 255.0), out=region, casting="unsafe")
        elif self.xor_fast_path:
            np.bitwise_xor(rgb, np.uint8(0x80), out=region.view(np.uint8))
        else:
            region[...] = self.lut[rgb]

    def score_threshold(self, confidence):
        """Threshold usable directly on raw output scores (strict '>')."""
        if not self.quantized_output:
            return confidence
        # (q - zero) * scale > conf  <=>  q > conf / scale + zero  <=>  q > floor(...)
        threshold = math.floor(confidence / self.output_scale + self.output_zero)
        info = np.iinfo(self.output_dtype)
        return None if threshold < info.min else min(threshold, info.max)

    def dequantize(self, values):
        values = values.astype(np.float32)
        if self.quantized_output:
            values -= np.float32(self.output_zero)
            values *= np.float32(self.output_scale)
        return values


# ----------------------------------------------------------------------------
# Preprocessing. Each returns (rgb_resized_uint8, scale, left, top, width, height)
# where width/height are the original image dimensions.
# ----------------------------------------------------------------------------

def letterbox_geometry(width, height, size):
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    return scale, new_width, new_height, (size - new_width) // 2, (size - new_height) // 2


def preprocess_pil(path, size, width=None, height=None):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale, new_width, new_height, left, top = letterbox_geometry(width, height, size)
    resized = np.asarray(image.resize((new_width, new_height), Image.Resampling.BILINEAR))
    return resized, scale, left, top, width, height


def preprocess_cv2(path, size, cv2, width=None, height=None):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode {path}")
    height, width = bgr.shape[:2]
    scale, new_width, new_height, left, top = letterbox_geometry(width, height, size)
    # Resize first, then swap channels on the smaller image.
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), scale, left, top, width, height


def reduced_decode_factor(width, height, new_width, new_height):
    """Largest libjpeg DCT scaling (1/2, 1/4, 1/8) that still downsamples."""
    for factor in (8, 4, 2):
        if width // factor >= new_width and height // factor >= new_height:
            return factor
    return 1


def preprocess_cv2_reduced(path, size, cv2, width, height):
    """Decode the JPEG at 1/2, 1/4 or 1/8 scale; needs the original dimensions."""
    scale, new_width, new_height, left, top = letterbox_geometry(width, height, size)
    factor = reduced_decode_factor(width, height, new_width, new_height)
    flags = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }[factor]
    bgr = cv2.imread(str(path), flags)
    if bgr is None:
        raise RuntimeError(f"Could not decode {path}")
    if bgr.shape[1] != new_width or bgr.shape[0] != new_height:
        bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale, left, top, width, height


def preprocess_opencl(path, size, cv2, width=None, height=None):
    """v1 GPU path (UMat colour conversion and resize); padding is done on CPU."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode {path}")
    height, width = bgr.shape[:2]
    scale, new_width, new_height, left, top = letterbox_geometry(width, height, size)
    gpu = cv2.UMat(bgr)
    gpu = cv2.cvtColor(gpu, cv2.COLOR_BGR2RGB)
    gpu = cv2.resize(gpu, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return gpu.get(), scale, left, top, width, height


def make_preprocessor(backend, size, cv2=None):
    if backend == "pil":
        return lambda path, w, h: preprocess_pil(path, size)
    if backend == "cv2":
        return lambda path, w, h: preprocess_cv2(path, size, cv2)
    if backend == "cv2-reduced":
        return lambda path, w, h: preprocess_cv2_reduced(path, size, cv2, w, h)
    if backend == "opencv-opencl":
        return lambda path, w, h: preprocess_opencl(path, size, cv2)
    raise ValueError(backend)


class OpenCLLetterbox:
    """ctypes wrapper for native_opencl/libyolo_opencl_letterbox.so.

    The JPEG is decoded on the CPU (reduced-size DCT decode) and copied once
    into a GPU-mapped buffer; one kernel resizes, pads, swaps channels and
    converts to the model input type; the mapped result is copied once into
    the input slot. No clEnqueueWrite/ReadBuffer transfers.
    """

    def __init__(self, library_path, io, max_width, max_height):
        import ctypes

        if io.quantized_input and io.input_dtype != np.int8:
            raise RuntimeError(f"Zero-copy letterbox supports float32/int8, not {io.input_dtype}")
        self.ctypes = ctypes
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.lb_init.argtypes = [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_int,
                                                          ctypes.c_char_p, ctypes.c_int]
        self.lib.lb_init.restype = ctypes.c_int
        self.lib.lb_map_source.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.lb_map_source.restype = ctypes.c_void_p
        self.lib.lb_run.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p, ctypes.c_char_p,
                                                         ctypes.c_int]
        self.lib.lb_run.restype = ctypes.c_int
        self.lib.lb_close.argtypes = []
        self.lib.lb_close.restype = None
        self.error = ctypes.create_string_buffer(4096)
        self.capacity = max_width * max_height * 3
        mode = 1 if io.quantized_input else 0
        status = self.lib.lb_init(
            max_width, max_height, io.size, mode,
            io.input_scale if mode else 0.0, int(io.input_zero) if mode else 0,
            self.error, len(self.error),
        )
        if status != 0:
            raise RuntimeError(f"OpenCL letterbox init failed ({status}): {self.error.value.decode()}")

    def source_view(self, height, width):
        pointer = self.lib.lb_map_source(self.error, len(self.error))
        if not pointer:
            raise RuntimeError(f"OpenCL map failed: {self.error.value.decode()}")
        buffer = (self.ctypes.c_uint8 * self.capacity).from_address(pointer)
        return np.frombuffer(buffer, dtype=np.uint8, count=height * width * 3).reshape(
            height, width, 3)

    def run(self, src_width, src_height, new_width, new_height, left, top, slot):
        if not slot.flags.c_contiguous:
            raise RuntimeError("Input slot must be C-contiguous")
        status = self.lib.lb_run(src_width, src_height, new_width, new_height, left, top,
                                 slot.ctypes.data, self.error, len(self.error))
        if status != 0:
            raise RuntimeError(f"OpenCL letterbox failed ({status}): {self.error.value.decode()}")

    def close(self):
        self.lib.lb_close()


def make_preparer(backend, io, cv2=None, letterbox=None):
    """Return prepare(path, width, height, slot) -> (scale, left, top, width, height).

    width/height are the manifest dimensions (needed for reduced decoding);
    the returned ones are the original image dimensions.
    """
    if backend == "opencl-zerocopy":
        if letterbox is None:
            raise ValueError("opencl-zerocopy needs an OpenCLLetterbox")

        def prepare(path, width, height, slot):
            scale, new_width, new_height, left, top = letterbox_geometry(width, height, io.size)
            factor = reduced_decode_factor(width, height, new_width, new_height)
            flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                     4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
            bgr = cv2.imread(str(path), flags)
            if bgr is None:
                raise RuntimeError(f"Could not decode {path}")
            src_height, src_width = bgr.shape[:2]
            np.copyto(letterbox.source_view(src_height, src_width), bgr)
            letterbox.run(src_width, src_height, new_width, new_height, left, top, slot)
            return scale, left, top, width, height

        return prepare

    preprocess = make_preprocessor(backend, io.size, cv2)

    def prepare(path, width, height, slot):
        rgb, scale, left, top, width, height = preprocess(path, width, height)
        io.write_input(slot, rgb, left, top)
        return scale, left, top, width, height

    return prepare


# ----------------------------------------------------------------------------
# Decode and NMS
# ----------------------------------------------------------------------------

EMPTY_XYXY = np.empty((0, 4), dtype=np.float32)
EMPTY_INT = np.empty(0, dtype=np.int64)
EMPTY_FLOAT = np.empty(0, dtype=np.float32)


def decode(raw, io, confidence):
    """Filter raw [1,84,N] output; scores/argmax stay in the int8 domain."""
    prediction = raw[0]
    class_rows = prediction[4:]
    best = class_rows.max(axis=0)
    threshold = io.score_threshold(confidence)
    if threshold is None:
        indices = np.arange(best.size)
    else:
        indices = np.flatnonzero(best > threshold)
    if not indices.size:
        return EMPTY_XYXY, EMPTY_INT, EMPTY_FLOAT
    classes = class_rows[:, indices].argmax(axis=0)
    scores = io.dequantize(best[indices])
    cx, cy, w, h = io.dequantize(prediction[:4, indices]) * np.float32(io.size)
    xyxy = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)
    return xyxy, classes, scores


def _greedy_nms(boxes, scores, iou_threshold, max_keep):
    # Stable descending order: ties broken by index, as in OpenCV. v1 used
    # argsort()[::-1], whose tie order is arbitrary; with int8 scores ties are
    # common enough to change about a third of the kept boxes.
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < max_keep:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        others = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[others, 0])
        y1 = np.maximum(boxes[current, 1], boxes[others, 1])
        x2 = np.minimum(boxes[current, 2], boxes[others, 2])
        y2 = np.minimum(boxes[current, 3], boxes[others, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = ((boxes[current, 2] - boxes[current, 0]) *
                        (boxes[current, 3] - boxes[current, 1]))
        area_others = ((boxes[others, 2] - boxes[others, 0]) *
                       (boxes[others, 3] - boxes[others, 1]))
        iou = intersection / (area_current + area_others - intersection + 1e-9)
        order = others[iou <= iou_threshold]
    return keep


def nms_legacy(xyxy, classes, scores, iou_threshold, max_detections):
    """v1 behaviour: Python greedy NMS once per class."""
    kept = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        kept.extend(indices[k] for k in _greedy_nms(
            xyxy[indices], scores[indices], iou_threshold, max_detections))
    kept = np.asarray(kept, dtype=np.int64)
    return kept[np.argsort(-scores[kept], kind="stable")][:max_detections]


def nms_fast(xyxy, classes, scores, iou_threshold, max_detections, cv2):
    """Class-aware NMS in one native call (OpenCV NMSBoxesBatched)."""

    if not hasattr(getattr(cv2, "dnn", None), "NMSBoxesBatched"):
        raise RuntimeError(
            "Fast NMS requires cv2.dnn.NMSBoxesBatched. Install the app's "
            "pinned opencv-python-headless dependency and restart."
        )
    
    if not scores.size:
        return EMPTY_INT
    xywh = np.empty((len(scores), 4), dtype=np.float64)  # float64 selects Rect2d
    xywh[:, :2] = xyxy[:, :2]
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    kept = cv2.dnn.NMSBoxesBatched(
        xywh, scores.astype(np.float32), classes.astype(np.int32), 0.0, iou_threshold
    )
    kept = np.asarray(kept, dtype=np.int64).reshape(-1)
    return kept[np.argsort(-scores[kept], kind="stable")][:max_detections]


def to_original(xyxy, scale, left, top, width, height):
    """Map letterboxed xyxy to clipped original-image COCO xywh."""
    boxes = xyxy.astype(np.float64)
    x1 = np.maximum(0.0, (boxes[:, 0] - left) / scale)
    y1 = np.maximum(0.0, (boxes[:, 1] - top) / scale)
    x2 = np.minimum(float(width), (boxes[:, 2] - left) / scale)
    y2 = np.minimum(float(height), (boxes[:, 3] - top) / scale)
    return np.stack(
        (x1, y1, np.maximum(0.0, x2 - x1), np.maximum(0.0, y2 - y1)), axis=1
    )


def coco_records(image_id, xywh, classes, scores):
    return [
        {
            "image_id": image_id,
            "category_id": int(COCO80_TO_91_ARRAY[c]),
            "bbox": [float(v) for v in box],
            "score": float(s),
        }
        for box, c, s in zip(xywh, classes, scores)
    ]
