# YOLO11n on Arduino UNO Q: optimization ladder

Arduino UNO Q (4-core Arm CPU, up to 2.016 GHz), Debian 13, LiteRT 2.2.0 with XNNPACK. Accuracy on a fixed 300-image subset of COCO val2017 (COCO-300). All numbers are measured on the board.

## 1. Optimization ladder

Each step adds one optimization to the previous step. Speedups are relative to step 0 (the baseline).

| Step | Optimization | Input | Precision | Threads | Execution | Pre / post | FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 | mAP change | Peak memory (MB) | Model size (MB) |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0. Baseline | None | 640x640 | FP32 | 1 | sequential | PIL + Python NMS | **0.92** | 1.00x | 1088 | 1031.9 | 43.24% | +0.00 | 171 | 10.2 |
| 1. Lower resolution | Input size | 320x320 | FP32 | 1 | sequential | PIL + Python NMS | **3.59** | 3.90x | 279 | 237.2 | 34.29% | -8.96 | 98 | 10.1 |
| 2. Quantization | Model precision | 320x320 | W8A8 (int8) | 1 | sequential | PIL + Python NMS | **5.15** | 5.60x | 194 | 156.6 | 31.34% | -11.91 | 70 | 2.8 |
| 3. Multi-threading | Intra-op parallelism | 320x320 | W8A8 (int8) | 4 | sequential | PIL + Python NMS | **11.20** | 12.19x | 89 | 51.1 | 31.34% | -11.91 | 70 | 2.8 |
| 4. Pipeline (final) | Stage overlap, reduced CPU work, core partitioning | 320x320 | W8A8 (int8) | 4 | pipelined, 3 stages | OpenCV reduced decode + native NMS | **18.90** | 20.56x | 53 | 52.7 | 31.51% | -11.73 | 76 | 2.8 |

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

Measured during the run: core 0 busy 100%, cores 1-3 at most 0% busy; CPU clock 2016 MHz throughout (no throttling); peak temperature 45.1 °C. Average per image: preprocessing 18 ms, inference 1032 ms, postprocessing 37 ms, end-to-end latency 1088 ms (0.92 FPS).

## 3. Final model: sequential and camera-fed operation

| Mode | Execution | Pre / post | FPS | Speedup | Avg. time per image (ms) | Inference (ms) | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| Sequential | sequential | OpenCV reduced decode + native NMS | **16.12** | 17.53x | 62 | 50.2 | 31.51% |
| Pipelined, camera feed | pipelined, 30 FPS camera feed (simulated) | OpenCV reduced decode + native NMS | **18.23** | 19.83x | 55 | 54.7 | 31.51% |

## 4. End-to-end latency breakdown

| Configuration | Preprocessing | Waiting for model | Inference | Postprocessing | End-to-end latency |
|---|---:|---:|---:|---:|---:|
| Step 3 (sequential, PIL + Python NMS) | 19.6 | 0.1 | 51.1 | 18.3 | **89.1** |
| Final model, sequential | 8.9 | 0.1 | 50.2 | 2.7 | **62.0** |
| Final model, pipelined, camera feed | 18.1 | 18.0 | 54.7 | 4.0 | **94.8** |

All values are average milliseconds per image. Postprocessing includes any wait for the postprocessing thread.

**Why the camera-fed pipeline has higher end-to-end latency.** The same model takes 62 ms per image when run sequentially and 95 ms in the camera-fed pipeline. The difference of 33 ms comes from:

- **Waiting for the model (+18 ms).** The camera delivers a frame every 33 ms regardless of the model, while one inference takes about 55 ms. A new frame therefore usually arrives while the previous frame is still in inference and waits for it to finish. In sequential execution the next image is read only after the previous result is complete, so nothing waits.
- **Slower preprocessing (+9 ms).** In the pipeline the preprocessing thread shares core 3 with one inference thread and runs at lower priority (nice 5), so it is preempted while inference runs. Sequentially it has the CPU to itself.
- **Slower inference (+5 ms).** The inference threads share the cores with preprocessing and postprocessing work.

In exchange, the pipeline keeps the model busy and completes 18.23 frames per second instead of 16.12. Overlapping stages raises throughput; it does not shorten the time any single frame takes.

Step 3 (89 ms) is also slightly faster than the camera-fed pipeline (95 ms): its preprocessing and Python NMS are slower (20 + 18 ms), but no frame ever waits and nothing competes with inference.

## 5. CPU + GPU

The pipeline (queue prefetch) with preprocessing moved to the Adreno GPU (OpenCL through Mesa Rusticl). JPEG decoding, inference and postprocessing stay on the CPU; threads, pinning and priority are identical.

| Variant | Preprocessing on | FPS | vs CPU only | Inference (ms) | Preprocessing thread CPU (ms/frame) | GPU driver CPU (ms/frame) | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Pipeline, CPU only | CPU (OpenCV, 1 thread) | **18.90** | +0.0% | 52.7 | 9.2 | 0.0 | 31.51% |
| CPU + GPU: OpenCV-OpenCL preprocessing | GPU (OpenCV UMat: color conversion, resize) | **18.56** | -1.8% | 53.5 | 10.9 | 2.8 | 31.67% |
| CPU + GPU: fused zero-copy letterbox | GPU (one OpenCL kernel: resize, pad, BGR to RGB, int8 conversion) | **17.71** | -6.3% | 56.2 | 9.1 | 10.8 | 31.48% |

Inference already uses all four CPU cores, so a GPU stage only helps if it reduces the CPU time spent outside inference. Preprocessing costs about 9 ms of CPU per frame on one core; the GPU path still needs the CPU to decode the JPEG, submit work and wait for results, and the driver adds its own CPU work. Running the network itself on the GPU was not possible with the available LiteRT build (no GPU delegate for Linux).

## 6. Observations

- **Overall:** 0.92 to 18.90 FPS (20.6x); mAP50-95 43.24% to 31.51% (-11.73 points).
- **Resolution:** 3.90x, and most of the accuracy loss (-8.96 points).
- **Quantization:** 1.44x for -2.95 points.
- **Multi-threading:** 2.18x with no change in accuracy.
- **Pipelining:** 1.69x throughput with no loss in accuracy (+0.18 points from OpenCV vs PIL resize rounding).
- **Memory:** peak process memory 171 MB to 76 MB (-56%); model file 10.2 MB to 2.8 MB (3.6x smaller, from quantization).
- **Camera feed:** at 30 FPS input the pipeline processes 18.2 frames per second (39% of frames are dropped as stale), sustained over 5 minutes at full CPU clock. Processing every frame would require under 33 ms per frame.

## 7. Configurations

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


## 8. Metrics and definitions

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

Peak memory for all five ladder steps was measured in repeat runs with identical configurations on 2026-10-09.

