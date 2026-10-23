# Traffic Safety Monitoring

SEC 2026 Student Innovation Challenge

Team: Md Nafis Tahmid Akhand; Md Mahfuzul Haque Gazi; Komala Subramanyam Cherukuri.

## How to run the prototype

1. Connect the powered USB-C hub to UNO Q, its PD charger, and the USB camera. Connect the laptop and board to the same network.
2. LED wiring: D7 → 1 kΩ resistor → red LED anode; D8 → a separate 1 kΩ resistor → green LED anode. Both cathodes go to GND.
3. In Arduino App Lab, connect to your UNO Q, choose Import app, and select Traffic_Safety_Monitoring_AppLab.zip. Open **Traffic Safety Monitoring**.
4. Press **Run**. Keep internet available on the first run: App Lab installs python/requirements.txt and builds/uploads sketch/sketch.ino. This replaces the MCU sketch with the D7/D8 LED controller. Wait for the dashboard address in the logs.
5. On the laptop open http://ArdnQ.local:8081/ or http://YOUR_UNO_Q_IP:8081/. You can also use localhost:8081 if you already have an SSH port-forward to the board.
6. Click **Define safety zone**, click the corners of your desired danger zone in order, then **Save calibration**. New imports require calibration. Recalibrate if the camera moves.
7. Choose **Watch for → People**, then **Apply settings**. Only the selected classes get boxes/IDs and affect zone alerts. Bicycle and People & bicycles are available; the latter does not prove a person is riding a bicycle.
8. Verify clear zone → green; selected object enters → red; leaves → green. Entry/clear confirmation uses 2/3 processed frames. During startup, calibration, stale inference, or shutdown both LEDs turn off; off is not a clear-zone indication.
9. Export frame metrics/events from the webpage if needed, then press **Stop** in App Lab before unplugging the board.

## Camera settings

config.json requests 640×480 MJPEG at **20 FPS**, manual exposure **30 ms**, gain **20**, and disables dynamic frame-rate exposure. Settings are applied directly through V4L2 on every start; no v4l2-ctl installation is needed inside the container.

The camera must support the requested rate: its negotiated rate and measured capture FPS are logged. Requested FPS is not a guarantee. If the image is dark, improve lighting or increase gain; you can try exposure_ms: 40 at an actual 20 FPS. Exposure must remain shorter than the frame interval. Stop the app before changing config.json and Run again.

Capture FPS, detection FPS, and browser FPS are different. Preview is limited by completed detections. Read-to-decision latency excludes exposure/driver buffering and subsequent LED/network/browser delay.

## Models and fair comparisons

The dashboard's Model selection dropdown switches between optimized W8A8 320
and baseline FP32 640. Click Apply model: the server stays available while
the old workers stop, the camera is released, and the selected model loads
and warms up. The active filename, input precision, and thread count are
reported by the running backend. Both models are bundled. A successful choice
is saved in `data/model_selection.json` and restored on the next app launch.
Camera settings and saved zone calibration are retained when camera identity
and dimensions match. Each switch creates a separate metrics session.

The real optimized model is included: models/yolo11n_silu_320_w8a8_int8io.tflite. It uses four LiteRT/XNNPACK CPU threads, pipelined preprocessing/postprocessing, reduced JPEG decoding when available, and fast NMS.

The app bundle includes both models. The FP32 640 baseline matches the
repository's top-level `models/yolo11n_silu_640_float32.tflite`.

If you want to compare our optimized model with the baseline, in a terminal on the UNO Q locate the helper:

```bash
find "$HOME/ArduinoApps" -name select_model.py
```

Use the returned path (choose this app if more than one import exists):

```bash
python3 /actual/app/path/select_model.py baseline
```

For the location of the baseline, append --baseline-source /actual/model.tflite. Press Run in App Lab. The baseline automatically uses one inference thread, CPU 0, sequential PIL preprocessing → inference → legacy NMS/postprocessing. Common camera, browser, and LED service threads remain background services.

The dashboard dropdown switches models immediately. The command-line helper
updates both config.json and the saved dashboard selection while the app is
stopped. Both modes retain the same camera settings and calibration. The precision
badge, thread count, and logs reflect the active model.

## Troubleshooting

- Missing LiteRT: this app declares **ai-edge-litert==2.2.0** and imports **ai_edge_litert**. Inspect dependency-install logs; the host yolo-litert virtual environment is not used by App Lab. Do not install edge_ai_litert or TensorFlow as a substitute.
- RouterBridge compile error: inspect the App Lab sketch dependency build; sketch/sketch.yaml declares Arduino_RouterBridge. Current cores may integrate/manage this library automatically. Keep the generated dependency file in the import. The app must finish uploading its sketch before zone_set_leds is available.
- No camera: stop other camera apps and inspect App Lab logs. USB capture nodes are detected by driver/capability, avoiding Qualcomm encoder nodes even when /dev/video numbers change.
- LED unavailable: inspect the dashboard notice. Python uses App Lab's APP_SOCKET when provided, otherwise /var/run/arduino-router.sock. A missing RPC method means the matching MCU sketch has not started successfully.
- Port already in use: stop the earlier command-line app or another app using 8081.

## Verification and scope

Host tests exercise the real bundled INT8 model with synthetic camera frames, HTTP endpoints, geometry/debounce, RPC packets, and baseline scheduling with a mocked FP32 interpreter.
