# Installing on a new Arduino UNO Q

This guide covers two separate environments: command-line ladder runs on the
board's Linux host, and the camera application inside Arduino App Lab.
Installing packages in the host virtual environment does not install them
inside the App Lab application environment.

The repository was developed for UNO Q with Debian 13, aarch64 Linux, and
Python 3.13.5. Check a new board before installing. No Python upgrade is needed
when it already provides the matching Python version and supported wheels.

## 1. Hardware and software needed

| Item | Ladder evaluation | Live App Lab application |
|---|---|---|
| Arduino UNO Q with its Linux system running | Required | Required |
| USB data cable and suitable board power | Required | Required |
| PC with ADB (Android platform tools), or SSH access | For copying files and opening a terminal | For setup and file transfer |
| Arduino App Lab on the PC | Optional | Required for importing/running the app and uploading its sketch |
| Internet access | Installation only; offline packages can be copied instead | First dependency installation and sketch build, unless already cached |
| USB camera supporting the requested capture mode | Not needed for COCO-300 | Required |
| Powered USB-C hub and suitable power supply | Not needed for stored images | Used by our camera setup |
| Red/green LEDs, two separate series resistors, and jumper wires | Not needed | Required only for external LED indications |
| COCO-300 image files | Required | Not needed for live camera operation |

The final ladder uses CPU inference. CUDA, PyTorch, TensorFlow, Ultralytics,
Qualcomm AI Hub credentials, and an OpenCL/GPU installation are not required
on the board to run the supplied TFLite models. Those training/export tools
belong on the PC. This project's ladder accepts CPU preprocessing backends only.

## 2. Confirm the board and connection

On the **Windows PC**, install Android platform tools if `adb` is unavailable,
then open a new PowerShell terminal and check:

```powershell
adb devices
```

Use the serial shown for your new board; `2250967616` belongs to the original
experimental board and is not a universal serial. Replace `BOARD_SERIAL`
in PC commands below.

```powershell
adb -s BOARD_SERIAL shell
```

In the **board terminal**:

```bash
uname -m
python3 --version
cat /proc/device-tree/model
nproc
```

Expected platform: `aarch64`, Arduino UNO Q/Imola, four CPU cores. The bundled
pycocotools wheel targets aarch64 Linux and CPython 3.12+ through the stable
ABI; it is suitable for the experiment's Python 3.13 platform.

USB is sufficient for terminal access and file transfer. Wi-Fi is needed only
if using network services, downloads, or browser access over the network.
If package installation fails, verify network access and certificates; keep
TLS and package-signature checks enabled.

## 3. Copy the project and prepare evaluation images

In **PC PowerShell**, from the repository directory:

```powershell
cd D:\DevProjects\unoq-yolo11n-ladder
python data\prepare_coco300.py
powershell -ExecutionPolicy Bypass -File .\setup_board.ps1 -Serial BOARD_SERIAL
```

This command transfers scripts, models, dataset and the app to explicit paths,
verifies SHA-256 checksums, and installs/checks host Python dependencies. Stop
running benchmarks/apps first. For installed dependencies use `-SkipInstall`;
to reuse an existing PC subset use `-Coco300Source "C:\path\to\coco300"`.
`-ValidateOnly` checks local files without contacting a board.

Preparing the images requires the COCO val2017 images at
`datasets/coco/images/val2017/` and annotations at
`datasets/coco/annotations/instances_val2017.json` on the PC. The preparation
script uses the deterministic 300-image subset and writes
`data/coco300/images/` plus `manifest.json`. Git does not include those JPEGs.
A Git clone alone therefore does not supply the evaluation images.

Verify on the **board** that copying produced the expected layout:

```bash
cd ~/unoq-yolo11n-ladder
ls models/*.tflite
ls data/coco300/manifest.json
find data/coco300/images -maxdepth 1 -type f -name '*.jpg' | wc -l
```

The last command should report 300. The repository includes the three ladder
models and the subset annotations used for board-side scoring.

## 4. Install Linux host prerequisites

On the **board**, with working internet and permission to use sudo:

```bash
sudo apt update
sudo apt install python3-venv python3-pip
```

Debian normally supplies `python3`. If it is missing, install it as well.
No compiler is needed when compatible binary wheels are available. `git` is
optional if cloning on the board; ADB copying does not require it. Camera
capture uses V4L2 directly, so `v4l2-ctl` is optional rather than a runtime
dependency. `fcntl`, `socket`, `threading`, and `http.server` are Python
standard-library modules, not packages to install through pip. `fcntl` is
Linux-specific, so the application tests do not run unchanged on Windows.

Create the host virtual environment:

```bash
python3 -m venv ~/venvs/yolo-litert
source ~/venvs/yolo-litert/bin/activate
cd ~/unoq-yolo11n-ladder
python -m pip install -r ladder/requirements-board.txt
python -m pip install 'msgpack>=1.0,<2'
python -m pip install --no-index ladder/kit_wheels/pycocotools-*.whl
python -m pip check
```

| Python package | Purpose | Required by |
|---|---|---|
| `ai-edge-litert==2.2.0` | TFLite interpreter with CPU inference | Ladder and live application |
| `numpy==2.5.3` | Tensors, quantization, filtering, geometry | Ladder and live application |
| `opencv-python-headless==5.0.0.93` | JPEG decoding, resizing, color conversion, native NMS, camera processing | Final ladder and live application |
| `pillow==12.3.0` | PIL preprocessing | Baseline ladder and baseline application profile |
| `pycocotools` | COCO mAP scoring | Ladder accuracy reports, not live monitoring |
| `msgpack>=1.0,<2` | Arduino Router messages for external LEDs | Application with external LEDs |

These pins reproduce the repository's declared board environment. Install only
one OpenCV distribution in an environment: the headless build is sufficient
because the application serves its preview in a browser. Avoid mixing pip
OpenCV distributions that all install the same `cv2` module.

## 5. Verify dependencies before a long run

On the **board**, with the environment activated:

```bash
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.machine())"
python -c "from ai_edge_litert.interpreter import Interpreter; import cv2, numpy, PIL, msgpack, pycocotools; print('Imports OK'); print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__); print('Native batched NMS:',hasattr(cv2.dnn,'NMSBoxesBatched'))"
```

`Native batched NMS` should be `True` for the final fast-NMS configuration.
The optimized app reports a dependency error if this API is absent, rather
than silently falling back to slower legacy NMS.

Test the final model's repeated-inference numerical behavior:

```bash
python ladder/check_numerics.py --models models/yolo11n_silu_320_w8a8_int8io.tflite
```

The comparison of identical four-thread runs should have zero difference.
After dependencies, data, and models are installed, stored-image inference
can run without internet:

```bash
bash ladder/run_final.sh
bash ladder/run_ladder.sh
```

Run one benchmark at a time. The first command runs only the final model;
the second runs the full ladder, taking approximately 12 minutes on the
experimental board. Numerical results and performance should be checked on
the new board. Fresh reports use actual measurements from each run's
`summary.json`; `ladder/kit/ladder_published.json` is historical reference only.

## 6. Install and run the App Lab application

### Separate application dependencies

The imported application's `python/requirements.txt` explicitly lists
LiteRT, NumPy, OpenCV headless, Pillow, and msgpack. App Lab installs these
inside its application environment. The host `~/venvs/yolo-litert` environment
is used only when running from the board terminal.

Before importing, on the **PC**:

```powershell
cd D:\DevProjects\unoq-yolo11n-ladder
Compress-Archive -Path arduino_app\traffic_safety_monitoring_applab -DestinationPath Traffic_Safety_Monitoring_AppLab.zip -Force
```

In App Lab, connect to UNO Q, import that zip, and open the application. Connect
the camera through the powered hub, then press Run. Dependency installation
and MCU sketch compilation/upload require working package access on the first
run. The application includes its model, web assets, and MCU sketch; the
COCO dataset is unnecessary for live operation.

### MCU library and LED sketch

The Linux CPU runs YOLO; the separate MCU controls the LEDs. The supplied
`sketch/sketch.ino` includes `Arduino_RouterBridge.h` and exposes
`zone_set_leds` through Arduino Router. App Lab must have the UNO Q board
platform (`arduino:zephyr:unoq`) and the RouterBridge library available through
its platform/build environment. The supplied `sketch.yaml` selects the board
platform but does not separately pin the RouterBridge library. If the header
is missing during compilation, resolve it in App Lab's board/library build
environment; installing a Python package will not fix the sketch build.

Running the app uploads this sketch, replacing the current MCU sketch.
Wire red to D7 through its own series resistor and green to D8 through a
separate resistor; connect cathodes to GND. The project's documented setup
uses 1 kΩ resistors. The MCU watchdog turns the outputs off if commands stop.

The Python side needs msgpack and access to the Arduino Router socket.
It uses App Lab's `APP_SOCKET` when provided, otherwise
`/var/run/arduino-router.sock`. No separate Python GPIO library is needed for
the external LED mode. Camera access and the Router socket must be exposed by
the App Lab runtime.

### Camera, dashboard, and first-run check

The app requests 640×480 MJPEG at 20 FPS, exposure 30 ms, and gain 20. The
camera must support those controls or report its actual behavior. Stop other
camera applications before starting this one. Camera/video device access
requires appropriate permissions; a permission error is separate from a
missing Python dependency.

For a terminal-based check on the **board**:

```bash
source ~/venvs/yolo-litert/bin/activate
cd ~/unoq-yolo11n-ladder/arduino_app/traffic_safety_monitoring_applab
python python/main.py --check
```

This loads the model, runs inference, and reads five camera frames. When using
App Lab, inspect its own logs because the terminal check uses the host environment.
The dashboard runs on port 8081. To reach it over Wi-Fi/Ethernet, the PC and
board need a network path that permits that connection. Alternatively, forward
the port over USB from **PC PowerShell**:

```powershell
adb -s BOARD_SERIAL forward tcp:8081 tcp:8081
```

Then open `http://localhost:8081` while the app is running. Define the safety
zone, save calibration, and verify clear/entry/exit behavior and both LEDs.
Calibration must be repeated if the camera position changes. Off LEDs indicate
startup, calibration, stale results, or shutdown rather than a clear zone.

## 7. If the board has no internet

Download compatible wheels on an internet-connected PC, then copy them over
USB. This example targets the original board's CPython 3.13 aarch64 Linux;
change the target if the new board differs.

On **PC PowerShell**, from the repository root:

```powershell
New-Item -ItemType Directory -Force offline_wheels | Out-Null
python -m pip download --only-binary=:all: --platform manylinux_2_27_aarch64 --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 --python-version 313 --implementation cp --abi cp313 --abi abi3 --abi none --dest offline_wheels -r ladder\requirements-board.txt 'msgpack>=1.0,<2'
adb -s BOARD_SERIAL push offline_wheels /home/arduino/unoq-yolo11n-ladder/
```

If pip cannot resolve a compatible binary wheel, stop and inspect the exact
package/version and target tags; do not download Windows wheels for the board.
On the **board**:

```bash
source ~/venvs/yolo-litert/bin/activate
cd ~/unoq-yolo11n-ladder
python -m pip install --no-index --find-links offline_wheels -r ladder/requirements-board.txt 'msgpack>=1.0,<2'
python -m pip install --no-index ladder/kit_wheels/pycocotools-*.whl
python -m pip check
```

This handles Python dependencies in the host virtual environment only. It does
not install missing Debian packages, App Lab container dependencies, the board
platform, or MCU libraries. An offline App Lab deployment requires those
application/build dependencies to be available in its own environment too.

## 8. Final checklist

- Correct board architecture and Python version checked.
- All three ladder models present; the app's optimized model is bundled.
- COCO-300 JPEGs copied for ladder evaluation.
- Host virtual environment activated and `pip check` passes.
- LiteRT, OpenCV, NumPy, Pillow, and pycocotools import successfully.
- Native batched NMS API available for comparable fast-path measurements.
- Repeated four-thread model outputs checked before benchmarking.
- App Lab dependencies installed separately from the host environment.
- Camera recognized and requested/negotiated settings checked.
- UNO Q sketch builds, uploads, and exposes the LED RPC.
- Dashboard reachable and zone calibrated before judging alerts.

See [the main README](../README.md) for operation and
[the application README](../arduino_app/traffic_safety_monitoring_applab/README.md)
for camera and LED troubleshooting.
