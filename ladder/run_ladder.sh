#!/usr/bin/env bash
# Regenerate the YOLO11n optimization ladder on the Arduino UNO Q.
#
#   bash run_ladder.sh                  # steps 0-4 + final model sequential (~12 min)
#   bash run_ladder.sh --with-camera    # + 5-minute simulated 30 FPS camera run
#
# Each invocation writes a fresh report folder:
#   <repo>/reports/ladder_<date>_<time>/OPTIMIZATION_LADDER.md
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT=${OUT:-$REPO/reports/ladder_$(date +%Y%m%d_%H%M%S)}
MODELS=${MODELS:-$REPO/models}
DATA=${DATA:-$REPO/data/coco300}
ANNOTATIONS=$HERE/kit/coco300_instances_val2017.json
COOLDOWN=${COOLDOWN:-20}

WITH_CAMERA=0
for arg in "$@"; do
    case "$arg" in
        --with-camera) WITH_CAMERA=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

VENV=${VENV:-$HOME/venvs/yolo-litert}
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

# Offline-safe dependency: pycocotools for mAP (bundled wheel, needs only numpy).
if ! python -c "import pycocotools" 2>/dev/null; then
    echo "Installing pycocotools from the bundled wheel"
    python -m pip install --no-index "$HERE"/kit_wheels/pycocotools-*.whl
fi

FP32_640=$MODELS/yolo11n_silu_640_float32.tflite
FP32_320=$MODELS/yolo11n_silu_320_float32.tflite
W8A8_320=$MODELS/yolo11n_silu_320_w8a8_int8io.tflite
for f in "$FP32_640" "$FP32_320" "$W8A8_320" "$DATA/manifest.json" "$ANNOTATIONS"; do
    [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p "$OUT"

run() {
    local name="$1"; shift
    echo
    echo "=== $name"
    python "$HERE/run_pipeline_v2.py" --data "$DATA" --out-dir "$OUT/$name" --label "$name" "$@" \
        2>&1 | tee "$OUT/$name.log"
    sleep "$COOLDOWN"
}

SEQ="--sequential --backend pil --nms legacy"
FINAL="--backend cv2-reduced --threads 4 --infer-cpus 0-3 --aux-cpus 3 --aux-nice 5"

# Ladder
run base_640_fp32_1core_seq   --model "$FP32_640" $SEQ --threads 1 --infer-cpus 0   # 0. baseline (~6 min)
run base_320_fp32_1core_seq   --model "$FP32_320" $SEQ --threads 1 --infer-cpus 0   # 1. resolution
run base_320_w8a8_1core_seq   --model "$W8A8_320" $SEQ --threads 1 --infer-cpus 0   # 2. quantization
run base_320_w8a8_4t_seq      --model "$W8A8_320" $SEQ --threads 4                  # 3. multi-threading
run base_320_w8a8_pipeline    --model "$W8A8_320" $FINAL                            # 4. pipeline (final)

# Final model, sequential (end-to-end latency without queueing)
run base_320_w8a8_4t_seq_fastpp --model "$W8A8_320" --sequential --backend cv2-reduced --nms fast --threads 4

if [[ $WITH_CAMERA == 1 ]]; then
    run final_320_w8a8_live5min --model "$W8A8_320" $FINAL --mode deploy --camera-fps 30 --duration 300
fi

python "$HERE/summarize_results.py" --results "$OUT" \
    --annotations "$ANNOTATIONS" --manifest "$DATA/manifest.json" \
    --ladder-out "$OUT/OPTIMIZATION_LADDER.md" > "$OUT/summary.log" 2>&1
ln -sfn "$OUT" "$REPO/reports/ladder_latest"
echo
echo "Report: $OUT/OPTIMIZATION_LADDER.md (also $REPO/reports/ladder_latest/)"
