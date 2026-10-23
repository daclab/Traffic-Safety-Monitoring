"""Run on the UNO Q host, with App Lab stopped: python3 select_model.py baseline|optimized."""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['baseline', 'optimized'])
p.add_argument('--baseline-source', type=Path, default=Path.home()/'unoq_yolo11n/models/int8_act/640/yolo11n_silu_640_float32.tflite')
a = p.parse_args()
root = Path(__file__).resolve().parent
name = 'yolo11n_silu_640_float32.tflite' if a.mode == 'baseline' else 'yolo11n_silu_320_w8a8_int8io.tflite'
target = root/'models'/name
if not target.is_file() and a.mode == 'baseline':
    source = a.baseline_source.expanduser()
    if not source.is_file():
        p.error(f'Baseline model is missing: {source}. Supply --baseline-source /actual/path.')
    shutil.copy2(source, target)
if not target.is_file() or target.stat().st_size == 0:
    p.error(f'Model is missing or empty: {target}')
config = root/'config.json'
cfg = json.loads(config.read_text())
cfg.update(model='models/'+name, profile='auto')
temp = config.with_suffix('.json.tmp')
temp.write_text(json.dumps(cfg, indent=2)+'\n')
temp.replace(config)
selection = root/'data/model_selection.json'
selection.parent.mkdir(exist_ok=True)
selection_temp = selection.with_suffix('.json.tmp')
selection_temp.write_text(json.dumps({'profile': a.mode}, indent=2)+'\n')
selection_temp.replace(selection)
print(f'Selected {a.mode}: {name}. Press Run in Arduino App Lab.')
