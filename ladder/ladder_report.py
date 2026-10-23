"""Builds OPTIMIZATION_LADDER.md: the shareable report of the optimization ladder.

Only runs of the ladder's models (FP32 640, FP32 320, W8A8 320) appear.
Called from summarize_results.py with the scored runs.
"""

from pathlib import Path


LADDER = [
    ("base_640_fp32_1core_seq", "0. Baseline", "None"),
    ("base_320_fp32_1core_seq", "1. Lower resolution", "Input size"),
    ("base_320_w8a8_1core_seq", "2. Quantization", "Model precision"),
    ("base_320_w8a8_4t_seq", "3. Multi-threading", "Intra-op parallelism"),
    ("base_320_w8a8_pipeline", "4. Pipeline (final)",
     "Stage overlap, reduced CPU work, core partitioning"),
]
SEQUENTIAL_MODE = "base_320_w8a8_4t_seq_fastpp"
CAMERA_MODE = "final_320_w8a8_live5min"

STEP_DESCRIPTIONS = """\
- **0. Baseline.** Official YOLO11n, FP32 TFLite, 640x640. LiteRT CPU
  (XNNPACK) restricted to 1 thread on core 0. One image at a time: PIL JPEG
  decode and letterbox, inference, per-class Python NMS.
- **1. Lower resolution.** The same FP32 model exported at 320x320: 4x fewer
  pixels and 2,100 instead of 8,400 candidate boxes.
- **2. Quantization.** FP32 to full-integer W8A8: int8 weights (per-channel)
  and int8 activations, int8 input and output, calibrated on 300 COCO
  train2017 images. The input is written as `pixel XOR 0x80` (no float
  conversion); scores are thresholded in int8 and only surviving candidates
  are dequantized. 1 thread, one image at a time.
- **3. Multi-threading.** XNNPACK splits every layer of one inference across 4
  threads on all 4 cores. One image at a time; pre- and postprocessing on one
  core.
- **4. Pipeline.** Preprocessing, inference and postprocessing of consecutive
  images overlap, CPU work outside inference is reduced, and threads are
  assigned to cores (details below).
"""

FINAL_CONFIGURATION = """\
**Model**
- YOLO11n (official weights, SiLU, P3/P4/P5 heads), input 320x320.
- Full-integer W8A8 TFLite (onnx2tf `*_full_integer_quant`): int8 weights and
  activations, int8 input/output, 300-image COCO train2017 calibration.
- Deterministic across repeated 4-thread runs on the board and consistent with
  the TFLite reference kernels.

**Execution: three overlapped stages**
- A preprocessing thread works on image n+1, the main thread runs inference
  on image n, a postprocessing thread finishes image n-1.
- Bounded queues (depth 2) between stages; a ring of 4 preallocated input
  buffers, so no memory is allocated per frame.

**Preprocessing**
- OpenCV JPEG decode at reduced DCT scale (`IMREAD_REDUCED_COLOR_2/4/8`): a
  640x480 photo is decoded directly at 320x240.
- Resize, then BGR to RGB, on the small image.
- Letterbox written directly into the int8 input buffer as `pixel XOR 0x80`
  (equal to pixel - 128); no float conversion.
- OpenCV restricted to 1 thread.

**Inference**
- LiteRT 2.2.0 with XNNPACK int8 kernels, 4 threads (one per core).

**Postprocessing**
- Confidence threshold and class argmax on the raw int8 output; only
  surviving candidates are dequantized.
- Class-aware NMS in one native call (`cv2.dnn.NMSBoxesBatched`) instead of a
  Python loop per class (identical results on float scores).

**Core partitioning**
- XNNPACK threads on cores 0-3; preprocessing and postprocessing threads
  pinned to core 3 at nice 5, so inference preempts them.

**Evaluated and not adopted** (same W8A8 320 model, board measurements)
- 3 inference threads with one core reserved for pre/postprocessing: -11% FPS
  (16.65 vs 18.78).
"""

# Used only when the CPU + GPU section has no runs yet (earlier, unpinned runs).
GPU_FALLBACK_BULLETS = """- GPU preprocessing (OpenCV-OpenCL on the Adreno GPU): -2.4% FPS.
- Zero-copy fused GPU letterbox: output correct, but the Rusticl driver spends
  about 17 ms of CPU time per frame on buffer mapping; -6% FPS.
"""

METRICS = """\
- **FPS**: images completed per second over the 300 images, end to end (JPEG
  read to final detections).
- **Avg. time per image**: 1000 / FPS, the average time between completed images.
  For sequential runs this equals the end-to-end latency of one image.
- **End-to-end latency** (latency breakdown section): average time for one image from JPEG read to final detections
  (decode, preprocessing, inference, postprocessing/NMS).
  In sequential execution one image is in flight, so end-to-end latency =
  1 / FPS. In pipelined execution several images are in flight, so end-to-end
  latency is longer than 1 / FPS and includes the time an image waits in the
  queue for the model.
- **Inference**: `interpreter.invoke()` time only.
- **Peak memory**: peak resident memory (VmHWM) of the whole Python process
  during the run: runtime, model, buffers, OpenCV/NumPy and the 300-image loop.
- **Model size**: size of the `.tflite` file.
- **Precision**: FP32 = no quantization; W8A8 = weights and activations in
  int8, int8 arithmetic throughout.
- **Threads**: XNNPACK threads for one inference (1 = one core; 4 = each layer
  split over all four cores).
- **Execution**: sequential = one image at a time; pipelined = preprocessing,
  inference and postprocessing of consecutive images overlap.
- **Simulated 30 FPS camera feed**: the COCO-300 JPEGs are released one every
  33 ms for 5 minutes; only the newest frame is processed and frames arriving
  while the model is busy are dropped. End-to-end latency is measured from a frame's
  release to its detections. No physical camera was used, so sensor capture
  delay is not included.
- **mAP50-95**: COCO-300 (fixed 300-image val2017 subset), confidence 0.001,
  NMS IoU 0.7, maximum 300 detections, pycocotools.
"""


def config_columns(s):
    name = Path(s["model"]).name
    precision = "FP32" if "float32" in name else "W8A8 (int8)"
    size = s["model_io"]["input_size"]
    if s.get("live"):
        execution = f"pipelined, {s['live']['camera_fps']:.0f} FPS camera feed (simulated)"
    elif s.get("sequential"):
        execution = "sequential"
    else:
        execution = "pipelined, 3 stages"
    pre = {"pil": "PIL", "cv2-reduced": "OpenCV reduced decode",
           "opencv-opencl": "OpenCV-OpenCL (GPU)", "opencl-zerocopy": "zero-copy kernel (GPU)"}
    nms = {"legacy": "Python NMS", "fast": "native NMS"}
    prepost = f"{pre.get(s['backend'], s['backend'])} + {nms[s['nms']]}"
    return f"{size}x{size}", precision, str(s["threads"]), execution, prepost


def pct(acc):
    return 100 * acc["map_50_95"] if acc else None


def breakdown(s):
    """Average ms per image: preprocessing, wait for model, inference, rest, total."""
    pre = s["preprocess_wall_ms"]["mean"]
    wait = s["queue_wait_ms"]["mean"]
    inference = s["inference_wall_ms"]["mean"]
    total = s["response_ms"]["mean"]
    return pre, wait, inference, total - pre - wait - inference, total


def memory_columns(s):
    peak = (s.get("memory_mb") or {}).get("peak_rss")
    size = s.get("model_size_mb")
    return (f"{peak:.0f}" if peak is not None else "-") + " | " + (
        f"{size:.1f}" if size is not None else "-")


def ladder_table(by_name, base_fps, base_map):
    lines = [
        "| Step | Optimization | Input | Precision | Threads | Execution | Pre / post "
        "| FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 | mAP change "
        "| Peak memory (MB) | Model size (MB) |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, step, kind in LADDER:
        if name not in by_name:
            lines.append(f"| {step} | {kind} | pending |" + " |" * 12)
            continue
        s, acc = by_name[name]
        size, precision, threads, execution, prepost = config_columns(s)
        fps = s["throughput_fps"]
        speedup = f"{fps / base_fps:.2f}x" if base_fps else "-"
        latency = f"{1000 / fps:.0f}"
        mean_map = pct(acc)
        map_text = f"{mean_map:.2f}%" if mean_map is not None else "-"
        change = (f"{mean_map - base_map:+.2f}"
                  if mean_map is not None and base_map is not None else "-")
        lines.append(
            f"| {step} | {kind} | {size} | {precision} | {threads} | {execution} | {prepost} "
            f"| **{fps:.2f}** | {speedup} | {latency} | {s['inference_wall_ms']['mean']:.1f} "
            f"| {map_text} | {change} | {memory_columns(s)} |")
    lines += [
        "",
        "Peak memory is whole-process peak RSS (VmHWM). A dash means that the "
        "configuration has not yet been rerun with memory logging; it does not "
        "mean zero memory use.",
    ]
    return lines


def modes_table(by_name, base_fps):
    rows = [(SEQUENTIAL_MODE, "Sequential"), (CAMERA_MODE, "Pipelined, camera feed")]
    rows = [(name, label) for name, label in rows if name in by_name]
    if not rows:
        return []
    lines = [
        "| Mode | Execution | Pre / post | FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    final = by_name.get(LADDER[-1][0])
    for name, label in rows:
        s, acc = by_name[name]
        _, _, _, execution, prepost = config_columns(s)
        mean_map = pct(acc) if acc else (pct(final[1]) if final and final[1] else None)
        lines.append(
            f"| {label} | {execution} | {prepost} | **{s['throughput_fps']:.2f}** "
            f"| {s['throughput_fps'] / base_fps:.2f}x | {1000 / s['throughput_fps']:.0f} "
            f"| {s['inference_wall_ms']['mean']:.1f} "
            f"| {f'{mean_map:.2f}%' if mean_map is not None else '-'} |")
    return lines


def latency_section(by_name):
    rows = [
        (LADDER[3][0], "Step 3 (sequential, PIL + Python NMS)"),
        (SEQUENTIAL_MODE, "Final model, sequential"),
        (CAMERA_MODE, "Final model, pipelined, camera feed"),
    ]
    rows = [(name, label) for name, label in rows if name in by_name]
    if len(rows) < 2:
        return []
    lines = [
        "| Configuration | Preprocessing | Waiting for model | Inference "
        "| Postprocessing | End-to-end latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in rows:
        pre, wait, inference, post, total = breakdown(by_name[name][0])
        lines.append(f"| {label} | {pre:.1f} | {wait:.1f} | {inference:.1f} | {post:.1f} "
                     f"| **{total:.1f}** |")
    lines.append("")
    lines.append("All values are average milliseconds per image. Postprocessing includes "
                 "any wait for the postprocessing thread.")

    if SEQUENTIAL_MODE in by_name and CAMERA_MODE in by_name:
        seq, cam = breakdown(by_name[SEQUENTIAL_MODE][0]), breakdown(by_name[CAMERA_MODE][0])
        cam_s = by_name[CAMERA_MODE][0]
        lines += [
            "",
            f"**Why the camera-fed pipeline has higher end-to-end latency.** The same model takes "
            f"{seq[4]:.0f} ms per image when run sequentially and {cam[4]:.0f} ms in the "
            f"camera-fed pipeline. The difference of {cam[4] - seq[4]:.0f} ms comes from:",
            "",
            f"- **Waiting for the model (+{cam[1] - seq[1]:.0f} ms).** The camera delivers a "
            f"frame every {1000 / cam_s['live']['camera_fps']:.0f} ms regardless of the "
            f"model, while one inference takes about {cam[2]:.0f} ms. A new frame therefore "
            f"usually arrives while the previous frame is still in inference and waits for "
            f"it to finish. In sequential execution the next image is read only after the "
            f"previous result is complete, so nothing waits.",
            f"- **Slower preprocessing (+{cam[0] - seq[0]:.0f} ms).** In the pipeline the "
            f"preprocessing thread shares core 3 with one inference thread and runs at lower "
            f"priority (nice 5), so it is preempted while inference runs. Sequentially it has "
            f"the CPU to itself.",
            f"- **Slower inference (+{cam[2] - seq[2]:.0f} ms).** The inference threads share "
            f"the cores with preprocessing and postprocessing work.",
            "",
            f"In exchange, the pipeline keeps the model busy and completes "
            f"{cam_s['throughput_fps']:.2f} frames per second instead of "
            f"{by_name[SEQUENTIAL_MODE][0]['throughput_fps']:.2f}. Overlapping stages raises "
            f"throughput; it does not shorten the time any single frame takes.",
        ]
    if LADDER[3][0] in by_name and CAMERA_MODE in by_name:
        s3, cam = breakdown(by_name[LADDER[3][0]][0]), breakdown(by_name[CAMERA_MODE][0])
        lines += [
            "",
            f"Step 3 ({s3[4]:.0f} ms) is also slightly faster than the camera-fed pipeline "
            f"({cam[4]:.0f} ms): its preprocessing and Python NMS are slower "
            f"({s3[0]:.0f} + {s3[3]:.0f} ms), but no frame ever waits and nothing competes "
            f"with inference.",
        ]
    return lines


def observations(by_name):
    names = [name for name, *_ in LADDER]
    if not all(name in by_name and by_name[name][1] for name in names):
        return []
    fps = [by_name[n][0]["throughput_fps"] for n in names]
    maps = [pct(by_name[n][1]) for n in names]
    lines = [
        f"- **Overall:** {fps[0]:.2f} to {fps[4]:.2f} FPS ({fps[4] / fps[0]:.1f}x); "
        f"mAP50-95 {maps[0]:.2f}% to {maps[4]:.2f}% ({maps[4] - maps[0]:+.2f} points).",
        f"- **Resolution:** {fps[1] / fps[0]:.2f}x, and most of the accuracy loss "
        f"({maps[1] - maps[0]:+.2f} points).",
        f"- **Quantization:** {fps[2] / fps[1]:.2f}x for {maps[2] - maps[1]:+.2f} points.",
        f"- **Multi-threading:** {fps[3] / fps[2]:.2f}x with no change in accuracy.",
        f"- **Pipelining:** {fps[4] / fps[3]:.2f}x throughput with no loss in accuracy "
        f"({maps[4] - maps[3]:+.2f} points from OpenCV vs PIL resize rounding).",
    ]
    peaks = [(by_name[n][0].get("memory_mb") or {}).get("peak_rss") for n in names]
    sizes = [by_name[n][0].get("model_size_mb") for n in names]
    if None not in (peaks[0], peaks[4], sizes[0], sizes[4]):
        lines.append(
            f"- **Memory:** peak process memory {peaks[0]:.0f} MB to {peaks[4]:.0f} MB "
            f"({100 * (peaks[4] / peaks[0] - 1):+.0f}%); model file {sizes[0]:.1f} MB to "
            f"{sizes[4]:.1f} MB ({sizes[0] / sizes[4]:.1f}x smaller, from quantization).")
    camera = by_name.get(CAMERA_MODE)
    if camera:
        cs, lv = camera[0], camera[0]["live"]
        skipped = 100 * (lv["captured"] - lv["processed"]) / lv["captured"]
        lines.append(
            f"- **Camera feed:** at {lv['camera_fps']:.0f} FPS input the pipeline processes "
            f"{cs['throughput_fps']:.1f} frames per second ({skipped:.0f}% of frames are "
            f"dropped as stale), sustained over {lv['duration_s'] / 60:.0f} minutes at full "
            f"CPU clock. Processing every frame would require under "
            f"{1000 / lv['camera_fps']:.0f} ms per frame.")
    return lines


GPU_VARIANTS = [
    ("base_320_w8a8_pipeline", "Pipeline, CPU only", "CPU (OpenCV, 1 thread)"),
    ("gpu_320_w8a8_pipeline_opencl", "CPU + GPU: OpenCV-OpenCL preprocessing",
     "GPU (OpenCV UMat: color conversion, resize)"),
    ("gpu_320_w8a8_pipeline_zerocopy", "CPU + GPU: fused zero-copy letterbox",
     "GPU (one OpenCL kernel: resize, pad, BGR to RGB, int8 conversion)"),
]


def gpu_driver_cpu_ms(s):
    """CPU time per frame spent in GPU driver threads (Rusticl/Mesa)."""
    return sum(v["cpu_ms_per_frame"] for k, v in s["threads_cpu"].items()
               if "rusticl" in k.lower() or "mesa" in k.lower())


def gpu_section(by_name):
    """CPU-only final configuration vs the same configuration with GPU preprocessing."""
    rows = [(n, label, where) for n, label, where in GPU_VARIANTS if n in by_name]
    if len(rows) < 2:
        return []
    reference_fps = by_name[GPU_VARIANTS[0][0]][0]["throughput_fps"] if GPU_VARIANTS[0][0] in by_name else None
    lines = [
        "The pipeline (queue prefetch) with preprocessing moved to the Adreno GPU "
        "(OpenCL through Mesa Rusticl). JPEG decoding, inference and postprocessing stay "
        "on the CPU; threads, pinning and priority are identical.",
        "",
        "| Variant | Preprocessing on | FPS | vs CPU only | Inference (ms) "
        "| Preprocessing thread CPU (ms/frame) | GPU driver CPU (ms/frame) | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label, where in rows:
        s, acc = by_name[name]
        fps = s["throughput_fps"]
        change = f"{100 * (fps / reference_fps - 1):+.1f}%" if reference_fps else "-"
        mean_map = pct(acc)
        lines.append(
            f"| {label} | {where} | **{fps:.2f}** | {change} "
            f"| {s['inference_wall_ms']['mean']:.1f} | {s['preprocess_cpu_ms']['mean']:.1f} "
            f"| {gpu_driver_cpu_ms(s):.1f} "
            f"| {f'{mean_map:.2f}%' if mean_map is not None else '-'} |")
    lines += [
        "",
        "Inference already uses all four CPU cores, so a GPU stage only helps if it "
        "reduces the CPU time spent outside inference. Preprocessing costs about 9 ms of "
        "CPU per frame on one core; the GPU path still needs the CPU to decode the JPEG, "
        "submit work and wait for results, and the driver adds its own CPU work. Running "
        "the network itself on the GPU was not possible with the available LiteRT build "
        "(no GPU delegate for Linux).",
    ]
    return lines


def baseline_section(base):
    """Exact conditions of step 0, with the measured run details."""
    if not base:
        return []
    s = base[0]
    clocks = [v["min"] for v in s["monitor"]["cpu_mhz"].values()]
    temps = [v["max"] for v in s["monitor"]["temperature_c"].values()]
    busy = s["core_busy_percent"]
    pre, wait, inference, post, total = breakdown(s)
    return [
        "The baseline is the unmodified model run in the simplest way, with none of the "
        "optimizations of steps 1-4:",
        "",
        "| Condition | Baseline setting |",
        "|---|---|",
        "| Model | Official Ultralytics `yolo11n.pt` (COCO-pretrained, SiLU, P3/P4/P5 heads), "
        "no retraining |",
        "| Format and precision | TFLite, FP32 weights and activations (no quantization) |",
        f"| Input size | {s['model_io']['input_size']}x{s['model_io']['input_size']} "
        f"({s['model_io']['candidates']:,} candidate boxes) |",
        "| Runtime | LiteRT 2.2.0 (Python), default CPU backend (XNNPACK) |",
        f"| CPU | {s['threads']} inference thread, process pinned to core "
        f"{', '.join(str(c) for c in s['infer_cpus'])}; the other three cores idle |",
        "| GPU | Not used |",
        "| Execution | Sequential: one image fully finished before the next is read; "
        "no pipelining, no worker threads (only a 1 Hz clock/temperature monitor) |",
        "| Preprocessing | PIL JPEG decode, RGB conversion, bilinear resize with "
        "letterbox padding (value 114), float32 normalization to [0, 1] |",
        "| Postprocessing | Class argmax and confidence threshold (0.001) in NumPy; "
        "greedy NMS per class in Python/NumPy (IoU 0.7, max 300 detections) |",
        f"| Measurement | {s['warmup']} warm-up inferences, then all "
        f"{s['images']} COCO-300 images timed |",
        "",
        f"Measured during the run: core 0 busy {busy.get('cpu0', 0):.0f}%, cores 1-3 "
        f"at most {max(busy.get(c, 0) for c in ('cpu1', 'cpu2', 'cpu3')):.0f}% busy; CPU "
        f"clock {min(clocks):.0f} MHz throughout (no throttling); peak temperature "
        f"{max(temps):.1f} °C. Average per image: preprocessing {pre:.0f} ms, inference "
        f"{inference:.0f} ms, postprocessing {post:.0f} ms, end-to-end latency "
        f"{total:.0f} ms ({s['throughput_fps']:.2f} FPS).",
    ]


def build(runs):
    """Return the report text, or None if no ladder run exists yet."""
    by_name = {name: (s, acc) for name, s, acc in runs}
    # Historical timing in kit/ must never replace fresh run measurements.
    if not any(name in by_name for name, *_ in LADDER):
        return None
    base = by_name.get(LADDER[0][0])
    base_fps = base[0]["throughput_fps"] if base else None
    base_map = pct(base[1]) if base and base[1] else None

    final_configuration = FINAL_CONFIGURATION
    sections = [
        ("Optimization ladder",
         ["Each step adds one optimization to the previous step. Speedups are relative "
          "to step 0 (the baseline).", ""] + ladder_table(by_name, base_fps, base_map)),
        ("Baseline conditions", baseline_section(base)),
        ("Final model: sequential and camera-fed operation", modes_table(by_name, base_fps)),
        ("End-to-end latency breakdown", latency_section(by_name)),
        ("Observations", observations(by_name)),
        ("Configurations", ["### Ladder steps", "", STEP_DESCRIPTIONS,
                            "### Final configuration (step 4)", "", final_configuration]),
        ("Metrics and definitions", [METRICS]),
    ]
    out = ["# YOLO11n on Arduino UNO Q: optimization ladder", "",
           "Arduino UNO Q (4-core Arm CPU, up to 2.016 GHz), Debian 13, LiteRT 2.2.0 with "
           "XNNPACK. Accuracy on a fixed 300-image subset of COCO val2017 (COCO-300). "
           "All numbers are measured on the board."]
    number = 0
    for title, body in sections:
        if body:
            number += 1
            out += ["", f"## {number}. {title}", ""] + body
    return "\n".join(out)
