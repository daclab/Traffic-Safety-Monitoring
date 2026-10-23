# YOLO11n on Arduino UNO Q: optimization ladder

Arduino UNO Q (4-core Arm CPU, up to 2.016 GHz), Debian 13, LiteRT 2.2.0 with XNNPACK. Accuracy on a fixed 300-image subset of COCO val2017 (COCO-300). All numbers are measured on the board.

## 1. Optimization ladder

Each step adds one optimization to the previous step. Speedups are relative to step 0 (the baseline).

| Step | Optimization | Input | Precision | Threads | Execution | Pre / post | FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 | mAP change | Peak memory (MB) | Model size (MB) |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0. Baseline | None | 640x640 | FP32 | 1 | sequential | PIL + Python NMS | **0.92** | 1.00x | 1090 | 1027.1 | 43.24% | +0.00 | 169 | 10.2 |
| 1. Lower resolution | Input size | 320x320 | FP32 | 1 | sequential | PIL + Python NMS | **3.48** | 3.79x | 287 | 244.4 | 34.29% | -8.96 | 98 | 10.1 |
| 2. Quantization | Model precision | 320x320 | W8A8 (int8) | 1 | sequential | PIL + Python NMS | **5.05** | 5.50x | 198 | 159.5 | 31.34% | -11.91 | 70 | 2.8 |
| 3. Multi-threading | Intra-op parallelism | 320x320 | W8A8 (int8) | 4 | sequential | PIL + Python NMS | **10.83** | 11.80x | 92 | 53.8 | 31.34% | -11.91 | 70 | 2.8 |
| 4. Pipeline (final) | Stage overlap, reduced CPU work, core partitioning | 320x320 | W8A8 (int8) | 4 | pipelined, 3 stages | OpenCV reduced decode + native NMS | **17.80** | 19.40x | 56 | 55.9 | 31.51% | -11.73 | 75 | 2.8 |

Peak memory is whole-process peak RSS (VmHWM). A dash means that the configuration has not yet been rerun with memory logging; it does not mean zero memory use.

## 2. Baseline conditions

The baseline is the unmodified model run in the simplest way, with none of the optimizations of steps 1-4:

| Condition | Baseline setting |
|---|---|
| Model | Official Ultralytics `yolo11n.pt` (COCO-pretrained, SiLU, P3/P4/P5 heads), no retraining |
| Format and precision | TFLite, FP32 weights and activations (no quantization) |
| Input size | 640x640 (8,400 candidate boxes) |
| Runtime | LiteRT 2.2.0 (Python), default CPU backend (XNNPACK) |
| CPU | 1 inference thread, process pinned to core 0; the other three cores idle |
| GPU | Not used |
| Execution | Sequential: one image fully finished before the next is read; no pipelining, no worker threads (only a 1 Hz clock/temperature monitor) |
| Preprocessing | PIL JPEG decode, RGB conversion, bilinear resize with letterbox padding (value 114), float32 normalization to [0, 1] |
| Postprocessing | Class argmax and confidence threshold (0.001) in NumPy; greedy NMS per class in Python/NumPy (IoU 0.7, max 300 detections) |
| Measurement | 3 warm-up inferences, then all 300 COCO-300 images timed |

Measured during the run: core 0 busy 100%, cores 1-3 at most 6% busy; CPU clock 2016 MHz throughout (no throttling); peak temperature 52.9 °C. Average per image: preprocessing 24 ms, inference 1027 ms, postprocessing 39 ms, end-to-end latency 1090 ms (0.92 FPS).

## 3. Final model: sequential and camera-fed operation

| Mode | Execution | Pre / post | FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| Sequential | sequential | OpenCV reduced decode + native NMS | **15.43** | 16.82x | 65 | 52.8 | 31.51% |

## 4. End-to-end latency breakdown

| Configuration | Preprocessing | Waiting for model | Inference | Postprocessing | End-to-end latency |
|---|---:|---:|---:|---:|---:|
| Step 3 (sequential, PIL + Python NMS) | 19.6 | 0.1 | 53.8 | 18.7 | **92.3** |
| Final model, sequential | 9.0 | 0.1 | 52.8 | 2.8 | **64.7** |

All values are average milliseconds per image. Postprocessing includes any wait for the postprocessing thread.

## 5. Observations

- **Overall:** 0.92 to 17.80 FPS (19.4x); mAP50-95 43.24% to 31.51% (-11.73 points).
- **Resolution:** 3.79x, and most of the accuracy loss (-8.96 points).
- **Quantization:** 1.45x for -2.95 points.
- **Multi-threading:** 2.15x with no change in accuracy.
- **Pipelining:** 1.64x throughput with no loss in accuracy (+0.18 points from OpenCV vs PIL resize rounding).
- **Memory:** peak process memory 169 MB to 75 MB (-56%); model file 10.2 MB to 2.8 MB (3.6x smaller, from quantization).

## 6. Configurations

### Ladder steps

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

### Final configuration (step 4)

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


## 7. Metrics and definitions

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

