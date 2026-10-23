# YOLO11n on Arduino UNO Q

A CPU optimization ladder for YOLO11n and an Arduino App Lab application
for live traffic safety-zone monitoring.

SEC 2026 Student Innovation Challenge.

Team: Md Nafis Tahmid Akhand; Md Mahfuzul Haque Gazi; Komala Subramanyam Cherukuri.

## Results

Board measurements on the fixed COCO-300 validation subset:

| Configuration | FPS | mAP50-95 | Peak RAM |
|---|---:|---:|---:|
| FP32 640, one thread | 0.92 | 43.24% | 171 MiB |
| FP32 320, one thread | 3.59 | 34.29% | 98 MiB |
| W8A8 320, one thread | 5.15 | 31.34% | 70 MiB |
| W8A8 320, four threads | 11.20 | 31.34% | 70 MiB |
| Final CPU pipeline | **18.90** | **31.51%** | **76 MiB** |

## Set up

You need UNO Q, a USB data cable, ADB on the PC, and aarch64 Linux with
Python 3.13 and `python3-venv` on the board. A camera is needed for live detection.

Run commands in **PC PowerShell**, from this repository. Replace
`BOARD_SERIAL` with the board shown by `adb devices`.

1. Place COCO val2017 images in `datasets/coco/images/val2017/` and annotations
   in `datasets/coco/annotations/instances_val2017.json`, then prepare the subset:

   ```powershell
   python data\prepare_coco300.py
   ```

2. Stop running apps/benchmarks and deploy:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_board.ps1 -Serial BOARD_SERIAL
   ```

Setup copies scripts, models, the dataset, and the application to
`/home/arduino/unoq-yolo11n-ladder/`, verifies checksums, installs dependencies,
and tests the models. Internet is needed for installation.

Use `-SkipInstall` for an already configured board, or
`-Coco300Source "C:\path\to\coco300"` to reuse an existing subset containing
`manifest.json` and `images/`. Dataset JPEGs are not included in Git.

[Installation details and offline setup](docs/NEW_BOARD_SETUP.md)

## Run the ladder

From **PC PowerShell**, keep the terminal open until completion (about 12 minutes):

```powershell
adb -s BOARD_SERIAL shell "cd /home/arduino/unoq-yolo11n-ladder && bash -o pipefail -c 'bash ladder/run_ladder.sh 2>&1 | tee ladder.log'"
```

For only the final configuration:

```powershell
adb -s BOARD_SERIAL shell "cd /home/arduino/unoq-yolo11n-ladder && bash ladder/run_final.sh"
```

Pull the completed ladder report and measurements:

```powershell
$LatestRun = (adb -s BOARD_SERIAL shell "readlink -f /home/arduino/unoq-yolo11n-ladder/reports/ladder_latest").Trim()
if ($LatestRun -notmatch '^/home/arduino/unoq-yolo11n-ladder/reports/ladder_[0-9_]+$') { throw 'No completed run found.' }
adb -s BOARD_SERIAL pull "$LatestRun/." ".\results\board_latest"
```

Reports contain actual measured FPS, latency, memory, and COCO mAP.

## Run the dashboard

From **PC PowerShell**, after setup:

```powershell
adb -s BOARD_SERIAL forward tcp:8081 tcp:8081
adb -s BOARD_SERIAL shell "cd /home/arduino/unoq-yolo11n-ladder/arduino_app/traffic_safety_monitoring_applab && /home/arduino/venvs/yolo-litert/bin/python python/main.py --led off"
```

Open **http://localhost:8081**. Connect a camera and define the safety zone.
Use **Model selection → Apply model** to switch between baseline and optimized.
Without a camera, the dashboard remains available and shows a camera error.
Stop with Ctrl+C. External LEDs require the supplied MCU sketch.

For **Arduino App Lab**, create the import ZIP:

```powershell
Compress-Archive -Path arduino_app\traffic_safety_monitoring_applab -DestinationPath Traffic_Safety_Monitoring_AppLab.zip -Force
```

Import it in App Lab and press **Run**. First-run installation needs internet;
App Lab installs its own dependencies and uploads the LED sketch. Stop the
terminal application before starting the App Lab application.

[App Lab instructions, wiring, camera settings, and troubleshooting](arduino_app/traffic_safety_monitoring_applab/README.md)

## Licenses

YOLO11n and its exports derive from Ultralytics YOLO (AGPL-3.0).
COCO annotations use CC BY 4.0; COCO images are not included.
