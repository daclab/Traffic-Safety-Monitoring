# How YOLO11n got 20x faster on the Arduino UNO Q

A step-by-step explanation of the final detection pipeline: what happens to
each image, where the data goes, which part of the board does which work, and
why each optimization makes it faster. All numbers are measurements on the
board unless marked otherwise.

---

## 1. The short version

| | Baseline (step 0) | Final pipeline (step 4) |
|---|---:|---:|
| Images per second (FPS) | 0.92 | **18.90** (20.6x) |
| Time per image | 1088 ms | 53 ms between finished images |
| Accuracy (mAP50-95, COCO-300) | 43.24% | 31.51% |
| Peak memory | 171 MB | 76 MB |
| Model file | 10.2 MB | 2.8 MB |

The speedup comes from four ideas, applied one after another:

1. **Do less work per image**: a smaller input (320x320 instead of 640x640).
2. **Do the work in a cheaper number format**: 8-bit integers instead of
   32-bit floats.
3. **Use all four CPU cores** for the neural network instead of one.
4. **Never let the cores wait**: prepare the next image and finish the
   previous one while the network runs, and make that extra work small.

The speedups multiply: 3.90 x 1.44 x 2.18 x 1.69 = 20.6.

The price is accuracy: 11.7 mAP points, almost all of it from the smaller
input (step 1). Steps 3 and 4 are free: they change how fast the work is done,
not what is computed.

---

## 2. The hardware and software

**Board: Arduino UNO Q**
- CPU: 4 Arm cores, up to 2.016 GHz (all measurements ran at full clock; no
  thermal throttling was observed)
- GPU: Qualcomm Adreno (OpenCL through the Mesa Rusticl driver); not used in
  the final pipeline (see section 9)
- RAM: 3.6 GiB
- OS: Debian 13, Linux 6.16

**Software used on the board**

| Library | Version | Used for |
|---|---|---|
| Python | 3.13 | Program logic, threads |
| LiteRT (`ai-edge-litert`) | 2.2.0 | Runs the TFLite model |
| XNNPACK | built into LiteRT | Fast CPU kernels (float and int8), multi-threading |
| OpenCV (`opencv-python-headless`) | 5.0.0 | JPEG decoding, resizing, colour conversion, NMS (final pipeline) |
| Pillow (PIL) | 12.3.0 | JPEG decoding and resizing (baseline) |
| NumPy | 2.5.3 | Arrays, letterbox, decoding the model output |
| pycocotools | 2.0.11 | Accuracy scoring (mAP); not part of the pipeline |

**On the PC (preparation only):** Ultralytics 8.4.45 exports `yolo11n.pt` to
TFLite via onnx2tf and TensorFlow 2.19, which also performs the int8
calibration.

**The model: YOLO11n** (the smallest YOLO11 detector, about 2.6 million
parameters; Ultralytics lists about 6.5 billion operations per image at
640x640). It takes one RGB image and outputs a table of candidate boxes:

```text
output shape [1, 84, N]
             84 rows = 4 box values (centre x, centre y, width, height)
                     + 80 class scores (person, bicycle, car, ...)
             N columns = candidate boxes, one per grid cell of three detection scales
                 640x640: 80x80 + 40x40 + 20x20 = 8,400 candidates
                 320x320: 40x40 + 20x20 + 10x10 = 2,100 candidates
```

**The test data:** 300 fixed images from COCO val2017 ("COCO-300"). A typical
image is a 640x480 JPEG of about 160 KB. Accuracy is scored with pycocotools
(confidence 0.001, NMS IoU 0.7, at most 300 detections per image).

---

## 3. What every pipeline must do

Every version, baseline or final, performs the same four stages for each
image:

```text
JPEG file ──► 1. Preprocess ──► 2. Inference ──► 3. Postprocess ──► detections
               decode, resize,    neural network    pick boxes,         (class, box,
               pad, convert       (YOLO11n)         remove duplicates   score)
```

1. **Preprocessing:** decode the JPEG into pixels, shrink it to the model's
   input size without distorting it ("letterbox": scale to fit, fill the rest
   with grey), and convert the pixels into the number format the model expects.
2. **Inference:** run the neural network. This is by far the most work.
3. **Postprocessing:** from thousands of candidate boxes, keep those with a
   reasonable score, remove duplicates of the same object (non-maximum
   suppression, NMS), and convert box coordinates back to the original photo.

The optimizations change **how** and **where** these stages run, not what
they compute.

---

## 4. Worked example: one image through the baseline (step 0)

**Image:** `000000001532.jpg`, 640x480 pixels, about 160 KB.
**Setup:** FP32 model at 640x640, 1 thread on core 0, one image at a time.

```text
Disk: JPEG 160 KB
  │
  │  PIL Image.open + convert("RGB")          ← full JPEG decode
  ▼
RGB pixels 640x480, uint8 (900 KB)
  │
  │  PIL resize (bilinear) to fit 640x640: scale = min(640/480, 640/640) = 1.0
  │  → stays 640x480
  ▼
Letterbox into a 640x640 float32 buffer:
  rows 0-79 grey, rows 80-559 the photo, rows 560-639 grey
  every pixel converted: value / 255 → a number between 0 and 1
  640 x 640 x 3 x 4 bytes = 4.9 MB
  │
  │  interpreter.set_tensor   (copy 4.9 MB into LiteRT)
  │  interpreter.invoke       (YOLO11n, FP32, 1 thread)       ← 1032 ms
  │  interpreter.get_tensor   (copy the output out)
  ▼
Output float32 [1, 84, 8400] = 2.8 MB
  │
  │  NumPy: for each of the 8,400 candidates, best class and its score
  │  keep scores > 0.001 → about 461 candidates
  │  Python loop: greedy NMS separately for each class        ← 33 ms
  │  map boxes back to the photo: x_photo = (x - 0) / 1.0, y_photo = (y - 80) / 1.0
  ▼
About 133 detections for this image set on average
```

**Time per image (average over 300 images):**

| Preprocess | Inference | Postprocess | Total |
|---:|---:|---:|---:|
| 18 ms | **1032 ms** | 37 ms | **1088 ms** (0.92 FPS) |

**Work distribution:** core 0 busy 100%, cores 1-3 idle (0%). GPU unused.

**Why it is slow:** one core performs about 6.5 billion floating-point
operations per image. That works out to roughly 6 billion operations per
second, the practical speed of one core in FP32. Inference is 95% of the time,
so the network is what has to get cheaper.

---

## 5. Worked example: the same image through the final pipeline (step 4)

**Setup:** W8A8 (int8) model at 320x320, inference on 4 cores, three stages
running at the same time on different images.

### 5.1 Data movement for one image

```text
Disk: JPEG 160 KB
  │
  │  [preprocessing thread, core 3, low priority]
  │  OpenCV imread(IMREAD_REDUCED_COLOR_2)
  │    the JPEG decoder (libjpeg) produces a half-size image directly,
  │    skipping 3/4 of the decoding work
  ▼
BGR pixels 320x240, uint8 (225 KB)        ← already the right size:
  │                                          scale = min(320/480, 320/640) = 0.5
  │                                          → 320x240, so no resize is needed
  │  OpenCV cvtColor BGR → RGB (on the small image)
  ▼
Letterbox written straight into a preallocated int8 buffer (one of 4 "slots"):
  rows 0-39 grey, rows 40-279 the photo, rows 280-319 grey
  each pixel converted with one bit operation: pixel XOR 0x80 (= pixel - 128)
  grey 114 becomes -14
  320 x 320 x 3 x 1 byte = 300 KB   (16x less data than the baseline's 4.9 MB)
  │
  │  queue (holds up to 2 prepared images)
  ▼
  [main thread + 3 XNNPACK worker threads, cores 0-3]
  │  set_tensor   (copy 300 KB into LiteRT; the slot is reused right away)
  │  invoke       (YOLO11n, int8, 4 threads)                 ← 53 ms
  │  get_tensor   (copy the output out)
  ▼
Output int8 [1, 84, 2100] = 176 KB
  │
  │  queue (holds up to 2 outputs)
  ▼
  [postprocessing thread, core 3, low priority]
  │  NumPy, still in int8:
  │    best score of each candidate = maximum over the 80 class rows
  │    keep candidates whose int8 score > -116
  │      (the output scale is 0.0055 and zero point -116, so -116 means 0.0)
  │    → about 220 candidates
  │  only those 220 are converted to real numbers:
  │    score = (q + 116) x 0.0055,  e.g. q = -60 → 0.31
  │    box   = (q + 116) x 0.0055 x 320 pixels
  │  OpenCV dnn.NMSBoxesBatched: duplicate removal for all classes in one call
  │  map back to the photo: x_photo = (x - 0) / 0.5, y_photo = (y - 40) / 0.5
  ▼
About 86 detections per image on average
```

### 5.2 How the three stages overlap

The stages are separate threads connected by queues. At any moment they work
on **three different images**:

```text
time →          0 ms        53 ms       106 ms      159 ms
core 3 (low prio):  [prep #2][post #0]  [prep #3][post #1]  [prep #4][post #2]
cores 0-3:          [==== infer #1 ====][==== infer #2 ====][==== infer #3 ====]
results:                                #1 done              #2 done
```

While the network works on image 2, image 3 is being prepared and image 1 is
being finished. When an inference ends, the next input is already waiting, so
the network starts again immediately. A finished image comes out about every
53 ms, which is the inference time: **18.9 images per second**.

### 5.3 Work distribution (measured)

| Thread | Cores | CPU time per image | Wall time per image |
|---|---|---:|---:|
| Inference (main + 3 XNNPACK workers) | 0, 1, 2, 3 | ~161 ms | 53 ms |
| Preprocessing | 3 only, low priority (nice 5) | 9 ms | 18 ms |
| Postprocessing | 3 only, low priority (nice 5) | 3.5 ms | 7 ms |
| Clock/temperature monitor | 3 only | 0.5 ms | — |

- **Inference: 161 ms of CPU work done in 53 ms** because about three cores
  work on it in parallel.
- **Preprocessing: 9 ms of CPU work spread over 18 ms** because it runs at lower
  priority and pauses whenever the inference thread on core 3 needs the core.
  This is intended: the network always comes first.
- **Core load:** 90%, 84%, 86%, 85%. The CPU is almost fully used.
- **GPU:** unused (see section 9).

```text
          core 0       core 1       core 2       core 3
          inference    inference    inference    inference
                                                 + preprocessing  (low priority)
                                                 + postprocessing (low priority)
```

Pre- and postprocessing together need only about 13 ms of CPU per image, a
quarter of one core. Giving them a core of their own wasted it: with 3
inference threads plus one reserved core the pipeline ran 11% slower (16.65 vs
18.78 FPS).

---

## 6. The ladder: one optimization at a time

Each step adds exactly one kind of optimization to the previous one, so its
effect can be measured on its own.

| Step | What changed | FPS | Speedup | Time per image | Inference | mAP50-95 | Peak memory |
|---|---|---:|---:|---:|---:|---:|---:|
| 0. Baseline | — | 0.92 | 1.0x | 1088 ms | 1032 ms | 43.24% | 171 MB |
| 1. Lower resolution | 640 → 320 input | 3.59 | 3.9x | 279 ms | 237 ms | 34.29% | 98 MB |
| 2. Quantization | FP32 → int8 | 5.15 | 5.6x | 194 ms | 157 ms | 31.34% | 70 MB |
| 3. Multi-threading | 1 → 4 cores for inference | 11.20 | 12.2x | 89 ms | 51 ms | 31.34% | 70 MB |
| 4. Pipeline (final) | overlap + cheaper pre/post + core pinning | **18.90** | **20.6x** | 53 ms | 53 ms | 31.51% | 76 MB |

### Step 1: lower resolution (3.9x faster)

**What changed:** the model input is 320x320 instead of 640x640. Same model,
same weights, exported for the smaller size.

**Why it is faster:** a convolutional network does a fixed amount of work per
pixel in every layer. Half the width and half the height means **a quarter of
the pixels**, so roughly a quarter of the arithmetic. The output also shrinks
from 8,400 to 2,100 candidates, so postprocessing gets cheaper as well.

| | Step 0 | Step 1 |
|---|---:|---:|
| Input tensor | 4.9 MB | 1.2 MB |
| Inference | 1032 ms | 237 ms (4.4x faster) |
| Postprocessing | 37 ms | 22 ms |
| Candidates above threshold | 461 | 255 |
| Peak memory | 171 MB | 98 MB |

**Cost:** -8.96 mAP. At 320x320 a small object (a distant person, a cup)
covers only a few pixels, and the network can no longer find it reliably. This
is by far the largest accuracy loss in the ladder.

### Step 2: quantization to int8 (1.44x faster)

**What changed:** weights and intermediate results (activations) are stored and
computed as 8-bit integers instead of 32-bit floating-point numbers ("W8A8").

**How it works:** each tensor gets a scale and a zero point that map its real
values onto the 256 integer levels from -128 to 127:

```text
real value = (integer - zero_point) x scale
```

The scales were found by running 300 COCO training images through the model
during export ("calibration") and recording the range of every tensor. The
input uses scale 1/255 and zero point -128, so a pixel value converts to int8
with one bit operation (`pixel XOR 0x80`); the baseline's division by 255 into
floats disappears.

**Why it is faster:**
- The Arm CPU's vector (SIMD) instructions process **16 int8 values at once**
  where they process 4 float32 values.
- Each value is a quarter of the size, so **4x less data** moves between memory
  and the CPU. On a small board, memory bandwidth is often the real limit.

| | Step 1 (FP32) | Step 2 (int8) |
|---|---:|---:|
| Inference | 237 ms | 157 ms (1.5x faster) |
| Model file | 10.1 MB | 2.8 MB |
| Input tensor | 1.2 MB | 300 KB |
| Peak memory | 98 MB | 70 MB |

**Why only 1.5x and not 4x:** not every operation gains equally (activation
functions, upsampling, the final box decoding), and some time goes into
reformatting data. Convolutions benefit most.

**Cost:** -2.95 mAP. Rounding every intermediate result to 256 levels loses a
little precision, mostly in the exact box positions and in low-confidence
detections.

### Step 3: multi-threading (2.18x faster)

**What changed:** the network runs with 4 threads, one per core.

**How it works:** XNNPACK splits **each layer** into four parts, one per core.
All four compute their part of the layer, wait for each other, then start the
next layer together. This is parallelism inside one inference; still only one
image is processed at a time.

```text
step 2:   core 0 [========== layer 1 ==========][========== layer 2 ==========] ...
step 3:   core 0 [= l1 =][= l2 =] ...
          core 1 [= l1 =][= l2 =] ...
          core 2 [= l1 =][= l2 =] ...
          core 3 [= l1 =][= l2 =] ...
```

| | Step 2 (1 core) | Step 3 (4 cores) |
|---|---:|---:|
| Inference | 157 ms | 51 ms (3.1x faster) |
| Preprocess + postprocess | 37 ms | 38 ms (unchanged) |
| Total per image | 194 ms | 89 ms (2.18x faster) |

**Why inference is only 3.1x faster with 4 cores:** the later layers of the
network work on tiny feature maps (10x10, 20x20), which do not split into four
equal parts, and every layer ends with the cores waiting for the slowest one.

**Why the total is only 2.18x faster:** preprocessing and postprocessing still
run on a single core while the other three wait. They now take 38 of the 89 ms,
**43% of the time**. This is Amdahl's law: speeding up one part leaves the
unchanged parts dominating. Step 4 attacks exactly this.

### Step 4: the pipeline (1.69x faster)

Three changes, described in detail in section 5:

**a) Overlap the stages.** Separate threads prepare the next image and finish
the previous one while the network runs on the current one. Pre- and
postprocessing time is hidden behind inference instead of added to it.

**b) Make pre- and postprocessing cheaper**, so they disturb inference less:

| Task | Steps 0-3 | Step 4 | Why it is cheaper |
|---|---|---|---|
| JPEG decode | PIL, full size (640x480) | OpenCV, decoded directly at half size (320x240) | The JPEG decoder skips 3/4 of its work |
| Resize | Always | Skipped when the decoded size already fits | Nothing to do for 640x480 photos |
| Pixel conversion | Divide by 255 into floats | `XOR 0x80` into int8 | One bit operation, 4x less data |
| Buffers | New arrays per image | 4 preallocated slots, reused | No memory allocation per image |
| NMS | Python loop, once per class | One OpenCV call for all classes | Native code instead of Python |
| Score filtering | On dequantized floats | On raw int8, before dequantizing | Only survivors are converted |

Measured one image at a time, so nothing overlaps: preprocessing 20 → 9 ms,
postprocessing 18 → 3 ms.

**c) Core partitioning.** The preprocessing and postprocessing threads are
pinned to core 3 with lower priority (Linux "nice 5"). Inference keeps all
four cores and always wins when it needs core 3.

| | Step 3 | Step 4 |
|---|---:|---:|
| Time between finished images | 89 ms | 53 ms |
| FPS | 11.20 | 18.90 (1.69x) |
| Core load | partly idle during pre/post | 84-90% on all cores |

---

## 7. Why it works so well: the four principles

1. **Less work.** The pixel count drops 4x (step 1) and the number format
   shrinks from 32 to 8 bits (step 2). Inference goes from 1032 to 157 ms on a
   single core.
2. **More workers.** Four cores share every layer (step 3). Inference goes from
   157 to 51 ms.
3. **No idle workers.** While the network computes, the next image is already
   being prepared (step 4). The network almost never waits, and the throughput
   approaches its limit of one image per inference time.
4. **Cheap side jobs at low priority.** Everything that is not the network is
   made small (9 + 3.5 ms of CPU per image) and runs only when the network
   does not need the core.

An everyday comparison is a kitchen. The baseline is one cook who prepares,
cooks and plates every dish alone, one dish at a time. The final pipeline uses
smaller portions (resolution), faster equipment (int8), four cooks on the main
dish (threads), and a helper who preps the next dish and plates the previous
one while the main dish cooks (pipeline).

---

## 8. Throughput versus latency

The pipeline raises **throughput** (images finished per second). It does not
make a single image finish faster.

| Way of running the final model | FPS | End-to-end time for one image |
|---|---:|---:|
| One image at a time, final pre/postprocessing | 16.12 | **62 ms** (9 + 50 + 3) |
| Pipelined, all 300 images available at once | 18.90 | ~212 ms (images wait in the queue) |
| Pipelined, simulated 30 FPS camera (newest frame only) | 18.23 | 95 ms |

- **Run one image at a time** when the fastest response for each image matters.
- **Run pipelined** when the most images per second matters.
- With a camera, frames arrive every 33 ms while one inference takes about
  53 ms, so a new frame usually waits a little for the previous inference to
  finish. The pipeline drops stale frames so this waiting cannot build up. (The
  camera run was simulated with the COCO images; no physical camera was used.)

---

## 9. What was tried and not used

| Idea | Result | Why |
|---|---|---|
| GPU preprocessing (OpenCV with OpenCL on the Adreno GPU) | -2% FPS | The CPU still decodes the JPEG and drives the GPU, and the GPU driver adds its own CPU work (about 3 ms per image). That CPU time competes with inference. |
| Custom "zero-copy" GPU letterbox kernel | -6% FPS | Output correct, but the driver spends about 11 ms of CPU per image mapping memory buffers. |
| Running the network on the GPU | Not possible | The available LiteRT build has no GPU support on Linux. |
| 3 inference threads + 1 core reserved for pre/post | -11% FPS | The reserved core would sit idle about 75% of the time. |
| Calibrating int8 on 1,000 instead of 300 images | Rejected | Triggered a multi-threading fault in the board's int8 kernels (outputs changed between identical runs). The deployed model was checked to give identical outputs across repeated runs. |
| "Just-in-time" preprocessing (prepare the next image only shortly before it is needed) | Not adopted | Per-image time fell from 212 to 76 ms, but FPS dropped about 6%. |
| Fine-tuning the model at 320x320 (20 epochs, low learning rate) | Not adopted | +0.14 mAP on the full COCO validation set: within noise. |

The common lesson: on this board the four CPU cores are the bottleneck, and any
extra work that needs the CPU (including driving the GPU) slows inference down.
The fastest pipeline is the one that gives the CPU the least extra work.

---

## 10. Accuracy notes

- **The 300-image subset reads higher than the full COCO validation set.** The
  FP32 model at 320 scores 34.29% on COCO-300 but 28.78% on all 5,000 images
  (PC measurement). Comparisons within the ladder are valid (always the same
  300 images), but absolute numbers should be quoted as COCO-300.
- **Where the accuracy goes:** -8.96 points from the smaller input, -2.95 from
  int8, and nothing from threading or pipelining (+0.18 from slightly different
  resize rounding in OpenCV vs PIL).
- **mAP is deterministic:** repeated runs of the same configuration gave
  identical mAP and identical numbers of detections. FPS varied by less than 2%
  between sessions.

---

## 11. Where everything is (in this repository)

| Item | Location |
|---|---|
| Final model | `models/yolo11n_silu_320_w8a8_int8io.tflite` |
| Baseline model | `models/yolo11n_silu_640_float32.tflite` |
| Step 1 model (FP32 320) | `models/yolo11n_silu_320_float32.tflite` |
| Pipeline code | `ladder/run_pipeline_v2.py`, `ladder/pipeline_core.py`, `ladder/board_monitor.py` |
| Regenerate the whole ladder (on the board) | `bash ladder/run_ladder.sh` |
| Run only the final model (on the board) | `bash ladder/run_final.sh` |
| COCO-300 manifest (images are prepared locally) | `data/coco300/manifest.json` |
| Measured ladder runs | `results/ladder/` |
| Arduino App Lab application | `arduino_app/traffic_safety_monitoring_applab/` |

Full numerical results: `OPTIMIZATION_LADDER.md`.
