#!/usr/bin/env bash
# Run the final YOLO11n configuration (step 4 of the optimization ladder).
#
#   bash run_final.sh                 # COCO-300 evaluation (confidence 0.001, mAP)
#   bash run_final.sh --mode deploy   # deployment confidence 0.25
#
# Extra arguments are passed to run_pipeline_v2.py. Results go to
# reports/final_<date>_<time>/ (summary.json, predictions.json).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT=${OUT:-$REPO/reports/final_$(date +%Y%m%d_%H%M%S)}
DATA=${DATA:-$REPO/data/coco300}
MODELS=${MODELS:-$REPO/models}

VENV=${VENV:-$HOME/venvs/yolo-litert}
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
python "$HERE/run_pipeline_v2.py" \
    --model "$MODELS/yolo11n_silu_320_w8a8_int8io.tflite" \
    --data "$DATA" \
    --out-dir "$OUT" --label final \
    --backend cv2-reduced --threads 4 --infer-cpus 0-3 --aux-cpus 3 --aux-nice 5 \
    "$@"
echo "Results: $OUT"
