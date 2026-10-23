"""Pipeline v2: COCO-300 LiteRT runner for the Arduino UNO Q.

Changes from gpu_pipeline/run_coco300_pipeline.py (v1):

  #0  Measures CPU time per stage (thread_time), whole-process CPU per frame,
      per-thread CPU and affinity, per-core load, clocks and temperatures.
  #1  Accepts weight-only, W8A8 float-I/O and W8A8 int8-I/O models; int8
      output is thresholded and argmax'd in the int8 domain.
  #2  CPU work outside inference: fast class-aware NMS (one native call),
      eval/deploy confidence modes, CPU OpenCV and reduced-size JPEG decode,
      int8 input written as pixel ^ 0x80, preallocated input ring buffer,
      COCO dicts built after the timed run, no GPU decode stage.
  #3  Core partitioning: inference threads pinned to --infer-cpus, the
      preprocessing/postprocessing/monitor threads pinned to --aux-cpus.

Stages (async postprocessing, the default):
  aux core(s):   preprocess image n+1          postprocess image n-1
  infer cores:   XNNPACK inference image n
"""

import argparse
import json
import os
import platform
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

import board_monitor as bm
import pipeline_core as pc


ROOT = Path(__file__).resolve().parents[1]  # repository root
MODES = {"eval": 0.001, "deploy": 0.25}
NMS_IOU = 0.7
MAX_DETECTIONS = 300

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--data", type=Path, default=ROOT / "data" / "coco300",
                    help="Folder with manifest.json and images/")
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--backend", choices=pc.CPU_BACKENDS, default="cv2-reduced",
                    help="opencl-zerocopy: CPU reduced decode + fused GPU letterbox")
parser.add_argument("--letterbox-library", type=Path,
                    default=Path(__file__).resolve().parent / "native_opencl"
                    / "libyolo_opencl_letterbox.so")
parser.add_argument("--nms", choices=("legacy", "fast"), default="fast")
parser.add_argument("--mode", choices=tuple(MODES), default="eval",
                    help="eval: confidence 0.001 for mAP; deploy: confidence 0.25")
parser.add_argument("--confidence", type=float, default=None,
                    help="Override the confidence implied by --mode")
parser.add_argument("--threads", type=int, default=4, help="XNNPACK threads")
parser.add_argument("--queue-depth", type=int, default=2)
parser.add_argument("--async-postprocess", action=argparse.BooleanOptionalAction,
                    default=True)
parser.add_argument("--infer-cpus", default=None, help="e.g. 0-2; unpinned if omitted")
parser.add_argument("--aux-cpus", default=None, help="e.g. 3; unpinned if omitted")
parser.add_argument("--aux-nice", type=int, default=0,
                    help="Nice value for preprocessing/postprocessing threads")
parser.add_argument("--cv-threads", type=int, default=1,
                    help="cv2.setNumThreads; 1 keeps OpenCV off the inference cores")
parser.add_argument("--camera-fps", type=float, default=0.0,
                    help="Live mode: frames arrive at this rate, only the newest is "
                         "processed (older ones dropped); latency from arrival to result")
parser.add_argument("--duration", type=float, default=30.0,
                    help="Live mode: seconds of camera stream")
parser.add_argument("--prefetch", choices=("queue", "jit"), default="queue",
                    help="queue: preprocessing runs ahead as far as the queues allow; "
                         "jit: the next image is prepared to be ready just when the "
                         "model becomes free (no image waits in a queue)")
parser.add_argument("--jit-margin-ms", type=float, default=2.0,
                    help="jit: start preprocessing this much earlier than the predicted "
                         "end of inference (larger = fewer model stalls, slightly more wait)")
parser.add_argument("--sequential", action="store_true",
                    help="Baseline mode: one image at a time on the main thread "
                         "(implies --no-async-postprocess)")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--limit", type=int, default=None, help="Use the first N images")
parser.add_argument("--monitor-interval", type=float, default=1.0)
parser.add_argument("--label", default=None)
args = parser.parse_args()

if args.queue_depth < 1 or args.threads < 1:
    parser.error("--queue-depth and --threads must be at least 1")
if args.sequential:
    if args.camera_fps > 0:
        parser.error("--sequential cannot be combined with --camera-fps")
    args.async_postprocess = False
confidence = args.confidence if args.confidence is not None else MODES[args.mode]
infer_cpus = bm.parse_cpus(args.infer_cpus)
aux_cpus = bm.parse_cpus(args.aux_cpus)

manifest = json.loads((args.data / "manifest.json").read_text())
items = manifest["images"][: args.limit] if args.limit else manifest["images"]
if not args.limit and len(items) != 300:
    raise RuntimeError("Manifest does not contain exactly 300 images")

cv2 = None
gpu_device = None
if args.backend != "pil" or args.nms == "fast":
    import cv2

    cv2.setNumThreads(args.cv_threads)
if args.backend == "opencv-opencl":
    if not cv2.ocl.haveOpenCL():
        raise RuntimeError("OpenCV reports that no OpenCL runtime is available")


def configure_cv2_thread():
    """OpenCL use is per-thread state in OpenCV; set it explicitly everywhere."""
    if cv2 is None:
        return None
    if args.backend == "opencv-opencl":
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            raise RuntimeError("OpenCV did not enable OpenCL; refusing CPU fallback")
        device = cv2.ocl.Device_getDefault()
        name = device.name
        return name() if callable(name) else name
    cv2.ocl.setUseOpenCL(False)
    return None


# Pin the main thread before the interpreter exists so XNNPACK's worker
# threads inherit the inference affinity.
bm.pin_current_thread(infer_cpus)
configure_cv2_thread()
memory_before_model = bm.process_memory_mb()
Interpreter = pc.load_interpreter_class()
interpreter = Interpreter(model_path=str(args.model), num_threads=args.threads)
interpreter.allocate_tensors()
memory_after_model = bm.process_memory_mb()
io = pc.ModelIO(interpreter)
size = io.size
letterbox = None
if args.backend == "opencl-zerocopy":
    if not args.letterbox_library.exists():
        raise FileNotFoundError(f"{args.letterbox_library}; run native_opencl/build_letterbox.sh")
    letterbox = pc.OpenCLLetterbox(
        args.letterbox_library, io,
        max(item["width"] for item in items), max(item["height"] for item in items),
    )
    gpu_device = "native OpenCL (zero-copy fused letterbox)"
prepare = pc.make_preparer(args.backend, io, cv2, letterbox)

warm = io.new_slot()
warm[...] = io.pad
for _ in range(args.warmup):
    interpreter.set_tensor(io.input_index, warm)
    interpreter.invoke()

# Ring of preallocated input slots: producer fills a free slot, main thread
# copies it into the interpreter and returns it. No per-frame allocation.
slot_count = args.queue_depth + 2
slots = [io.new_slot() for _ in range(slot_count)]
free_slots = queue.Queue()
for index in range(slot_count):
    free_slots.put(index)
class LatestFrame:
    """Single-slot mailbox: a new frame replaces an unconsumed one (dropped)."""

    def __init__(self):
        self.condition = threading.Condition()
        self.work = None
        self.closed = False
        self.dropped = 0

    def put(self, work):
        with self.condition:
            if work is None:
                self.closed = True
            else:
                if self.work is not None:
                    free_slots.put(self.work[1])
                    self.dropped += 1
                self.work = work
            self.condition.notify()

    def get(self):
        with self.condition:
            while self.work is None and not self.closed:
                self.condition.wait()
            work, self.work = self.work, None
            return work


live = args.camera_fps > 0
jit = args.prefetch == "jit" and not live and not args.sequential
if live:
    work_queue = LatestFrame()
else:
    work_queue = queue.Queue(maxsize=1 if jit else args.queue_depth)

# Just-in-time prefetch: at most one image is prepared ahead, and its
# preprocessing is timed to finish when the running inference is expected to
# end (moving averages of inference and preprocessing time).
prefetch_permit = threading.Semaphore(1)
jit_timing = {"inference_start_ns": None, "inference_ms": None, "preprocess_ms": None}


def moving_average(previous, value, weight=0.2):
    return value if previous is None else (1 - weight) * previous + weight * value


def wait_until_due():
    start, inference, pre = (jit_timing["inference_start_ns"], jit_timing["inference_ms"],
                             jit_timing["preprocess_ms"])
    if start is None or inference is None or pre is None:
        return
    due_ns = start + (inference - pre - args.jit_margin_ms) * 1e6
    delay = (due_ns - time.perf_counter_ns()) / 1e9
    if delay > 0:
        time.sleep(delay)
post_queue = queue.Queue(maxsize=args.queue_depth) if args.async_postprocess else None
live_stats = {"captured": 0, "missed_by_preprocessing": 0, "no_free_slot": 0}

stop = threading.Event()
errors = []
known_threads = {}
worker_cpu_ms = {}
records = {
    "pre_wall": [], "pre_cpu": [], "infer_wall": [], "infer_main_cpu": [],
    "post_wall": [], "post_cpu": [], "nms_wall": [], "response": [],
    "queue_wait": [], "candidates": [], "completion": [],
}
results = []


def prepare_frame(item, slot_index, arrival_ns):
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    scale, left, top, width, height = prepare(
        args.data / "images" / item["file_name"], item["width"], item["height"],
        slots[slot_index],
    )
    wall_end = time.perf_counter_ns()
    records["pre_cpu"].append((time.thread_time_ns() - cpu_start) / 1e6)
    records["pre_wall"].append((wall_end - wall_start) / 1e6)
    jit_timing["preprocess_ms"] = moving_average(
        jit_timing["preprocess_ms"], records["pre_wall"][-1])
    work_queue.put(
        (item, slot_index, scale, left, top, width, height, arrival_ns, wall_end)
    )


def camera_frames():
    """Yield (item, arrival_ns) at --camera-fps for --duration seconds."""
    period = 1.0 / args.camera_fps
    end = time.perf_counter() + args.duration
    next_arrival = time.perf_counter()
    index = 0
    while not stop.is_set() and next_arrival < end:
        now = time.perf_counter()
        if now < next_arrival:
            time.sleep(next_arrival - now)
        else:
            # Frames that arrived while preprocessing was busy are lost, as
            # with a camera that only keeps its newest frame.
            missed = int((now - next_arrival) // period)
            live_stats["missed_by_preprocessing"] += missed
            next_arrival += missed * period
        arrival_ns = time.perf_counter_ns()
        next_arrival += period
        live_stats["captured"] += 1
        yield items[index % len(items)], arrival_ns
        index += 1


def producer():
    global gpu_device
    known_threads["preprocess"] = threading.get_native_id()
    try:
        bm.pin_current_thread(aux_cpus, args.aux_nice)
        device = configure_cv2_thread()
        if device is not None:
            gpu_device = device
        if live:
            for item, arrival_ns in camera_frames():
                try:
                    slot_index = free_slots.get_nowait()
                except queue.Empty:
                    live_stats["no_free_slot"] += 1
                    continue
                prepare_frame(item, slot_index, arrival_ns)
        else:
            for item in items:
                if jit:
                    # Released when the model starts the previous image.
                    while not prefetch_permit.acquire(timeout=0.5):
                        if stop.is_set():
                            return
                    wait_until_due()
                if stop.is_set():
                    return
                slot_index = free_slots.get()
                prepare_frame(item, slot_index, time.perf_counter_ns())
    except BaseException as exc:
        errors.append(exc)
        traceback.print_exc()
    finally:
        worker_cpu_ms["preprocess"] = time.thread_time_ns() / 1e6
        work_queue.put(None)


def postprocess(work):
    item, raw, scale, left, top, width, height, response_start = work
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    xyxy, classes, scores = pc.decode(raw, io, confidence)
    nms_start = time.perf_counter_ns()
    if args.nms == "fast":
        kept = pc.nms_fast(xyxy, classes, scores, NMS_IOU, MAX_DETECTIONS, cv2)
    else:
        kept = pc.nms_legacy(xyxy, classes, scores, NMS_IOU, MAX_DETECTIONS)
    xywh = pc.to_original(xyxy[kept], scale, left, top, width, height)
    finish = time.perf_counter_ns()
    records["post_cpu"].append((time.thread_time_ns() - cpu_start) / 1e6)
    records["post_wall"].append((finish - wall_start) / 1e6)
    records["nms_wall"].append((finish - nms_start) / 1e6)
    records["response"].append((finish - response_start) / 1e6)
    records["candidates"].append(len(scores))
    records["completion"].append(finish)
    results.append((item["id"], xywh, classes[kept], scores[kept]))


def postprocessor():
    known_threads["postprocess"] = threading.get_native_id()
    try:
        bm.pin_current_thread(aux_cpus, args.aux_nice)
        configure_cv2_thread()
        while (work := post_queue.get()) is not None:
            postprocess(work)
    except BaseException as exc:
        errors.append(exc)
        stop.set()
        traceback.print_exc()
    finally:
        worker_cpu_ms["postprocess"] = time.thread_time_ns() / 1e6


sampler = bm.Sampler(args.monitor_interval, aux_cpus)
sampler.start()
time.sleep(0.05)
known_threads["monitor"] = sampler.native_id
known_threads["main"] = threading.get_native_id()

threads_before = bm.thread_cpu_seconds()
cores_before = bm.proc_stat_cores()
process_cpu_start = time.process_time_ns()
wall_start = time.perf_counter_ns()

def sequential_work():
    """--sequential: decode, preprocess, infer, postprocess one image at a time
    on the main thread; no worker threads, no overlap."""
    for item in items:
        prepare_frame(item, free_slots.get(), time.perf_counter_ns())
        yield work_queue.get()


producer_thread = None
post_thread = None
if args.sequential:
    work_source = sequential_work()
else:
    producer_thread = threading.Thread(target=producer, name="preprocess", daemon=True)
    producer_thread.start()
    work_source = iter(work_queue.get, None)
if args.async_postprocess:
    post_thread = threading.Thread(target=postprocessor, name="postprocess", daemon=True)
    post_thread.start()

completed_inferences = 0
try:
    for work in work_source:
        item, slot_index, scale, left, top, width, height, response_start, ready = work
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        records["queue_wait"].append((start - ready) / 1e6)
        interpreter.set_tensor(io.input_index, slots[slot_index])
        free_slots.put(slot_index)
        jit_timing["inference_start_ns"] = time.perf_counter_ns()
        if jit:
            prefetch_permit.release()
        interpreter.invoke()
        raw = interpreter.get_tensor(io.output_index)
        records["infer_main_cpu"].append((time.thread_time_ns() - cpu_start) / 1e6)
        records["infer_wall"].append((time.perf_counter_ns() - start) / 1e6)
        jit_timing["inference_ms"] = moving_average(
            jit_timing["inference_ms"],
            (time.perf_counter_ns() - jit_timing["inference_start_ns"]) / 1e6)
        post_work = (item, raw, scale, left, top, width, height, response_start)
        if args.async_postprocess:
            while True:
                if errors:
                    raise RuntimeError("Worker failed") from errors[0]
                try:
                    post_queue.put(post_work, timeout=0.5)
                    break
                except queue.Full:
                    continue
        else:
            postprocess(post_work)
        completed_inferences += 1
        if completed_inferences % 50 == 0:
            print(f"Inferred {completed_inferences}/{len(items)}", flush=True)
    if post_thread is not None:
        post_queue.put(None)
        post_thread.join()
finally:
    stop.set()
    if producer_thread is not None:
        producer_thread.join(timeout=5)

wall_ns = time.perf_counter_ns() - wall_start
process_cpu_ms = (time.process_time_ns() - process_cpu_start) / 1e6
cores_after = bm.proc_stat_cores()
threads_after = bm.thread_cpu_seconds()
memory_end = bm.process_memory_mb()
sampler.stop()
if letterbox is not None:
    letterbox.close()

if errors:
    raise RuntimeError("Pipeline worker failed") from errors[0]
if not live and len(results) != len(items):
    raise RuntimeError(f"Processed {len(results)} images, expected {len(items)}")


def stats(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


# Per-thread CPU over the timed run. Unknown threads are grouped by name:
# XNNPACK workers inherit the process name ("other:python"); GPU driver
# threads (Rusticl/Mesa) show up under their own names.
names = {tid: role for role, tid in known_threads.items() if tid}
thread_report = {}
for tid, (comm, cpu_after, allowed) in threads_after.items():
    cpu_before = threads_before.get(tid, (comm, 0.0, allowed))[1]
    role = names.get(tid, f"other:{comm}")
    entry = thread_report.setdefault(role, {"threads": 0, "cpu_ms": 0.0, "cpus_allowed": []})
    entry["threads"] += 1
    entry["cpu_ms"] += (cpu_after - cpu_before) * 1000.0
    if allowed not in entry["cpus_allowed"]:
        entry["cpus_allowed"].append(allowed)
# Worker threads have exited before the snapshot; they report their own totals.
for role, cpu_ms in worker_cpu_ms.items():
    thread_report[role] = {"threads": 1, "cpu_ms": cpu_ms,
                           "cpus_allowed": [args.aux_cpus or "unpinned"]}
frames = len(results)
for entry in thread_report.values():
    entry["cpu_ms_per_frame"] = entry["cpu_ms"] / frames

completion = np.diff(np.asarray(records["completion"], dtype=np.float64)) / 1e6
wall_seconds = wall_ns / 1e9
summary = {
    "label": args.label,
    "runner": "pipeline_v2",
    "images": frames,
    "model": str(args.model),
    "model_io": io.describe(),
    "backend": args.backend,
    "gpu_device": gpu_device,
    "nms": args.nms,
    "mode": args.mode,
    "confidence": confidence,
    "threads": args.threads,
    "queue_depth": args.queue_depth,
    "async_postprocess": args.async_postprocess,
    "infer_cpus": sorted(infer_cpus) if infer_cpus else None,
    "aux_cpus": sorted(aux_cpus) if aux_cpus else None,
    "aux_nice": args.aux_nice,
    "cv_threads": args.cv_threads,
    "warmup": args.warmup,
    "model_size_mb": args.model.stat().st_size / 2**20,
    "memory_mb": {
        "rss_before_model": memory_before_model["rss_mb"],
        "rss_after_model_load": memory_after_model["rss_mb"],
        "rss_end": memory_end["rss_mb"],
        "peak_rss": memory_end["peak_rss_mb"],
    },
    "sequential": args.sequential,
    "prefetch": "jit" if jit else ("n/a" if args.sequential or live else "queue"),
    "jit_margin_ms": args.jit_margin_ms if jit else None,
    "live": {
        "camera_fps": args.camera_fps,
        "duration_s": args.duration,
        **live_stats,
        "dropped_stale": work_queue.dropped,
        "processed": len(results),
    } if live else None,
    "wall_seconds": wall_seconds,
    "throughput_fps": frames / wall_seconds,
    "amortized_ms_per_image": wall_seconds * 1000 / frames,
    "completion_interval_ms": stats(completion.tolist()),
    "response_ms": stats(records["response"]),
    "queue_wait_ms": stats(records["queue_wait"]),
    "preprocess_wall_ms": stats(records["pre_wall"]),
    "preprocess_cpu_ms": stats(records["pre_cpu"]),
    "inference_wall_ms": stats(records["infer_wall"]),
    "inference_main_thread_cpu_ms": stats(records["infer_main_cpu"]),
    "postprocess_wall_ms": stats(records["post_wall"]),
    "postprocess_cpu_ms": stats(records["post_cpu"]),
    "nms_wall_ms": stats(records["nms_wall"]),
    "candidates_after_threshold": stats(records["candidates"]),
    "process_cpu_ms_per_frame": process_cpu_ms / frames,
    "process_cpu_cores_busy": process_cpu_ms / (wall_ns / 1e6),
    "threads_cpu": thread_report,
    "core_busy_percent": bm.core_busy_percent(cores_before, cores_after),
    "cpufreq_policy": bm.cpufreq_policy() if bm.IS_LINUX else None,
    "monitor": sampler.summary(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "opencv": cv2.__version__ if cv2 is not None else None,
}

args.out_dir.mkdir(parents=True, exist_ok=True)
predictions = []
for image_id, xywh, classes, scores in results:
    predictions.extend(pc.coco_records(image_id, xywh, classes, scores))
summary["predictions"] = len(predictions)
(args.out_dir / "predictions.json").write_text(json.dumps(predictions) + "\n")
(args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps({k: summary[k] for k in (
    "label", "throughput_fps", "amortized_ms_per_image", "response_ms",
    "preprocess_cpu_ms", "inference_wall_ms", "postprocess_cpu_ms",
    "process_cpu_ms_per_frame", "core_busy_percent", "predictions")}, indent=2))
print(f"Summary: {args.out_dir / 'summary.json'}")
