"""Print the exact runtime, model tensor details, and a real camera-read result."""
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path


def check(root,cfg):
    print('Python:',sys.executable,flush=True)
    print('Platform:',sys.version.split()[0],platform.machine(),flush=True)
    import cv2
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    import pipeline_core as pc
    from camera import Camera
    print('LiteRT:',importlib.metadata.version('ai-edge-litert'),'OpenCV:',cv2.__version__,'NumPy:',np.__version__,flush=True)
    path=Path(cfg['model']).expanduser()
    if not path.is_absolute():path=Path(root)/path
    if not path.is_file():raise FileNotFoundError(f'Model not found: {path}')
    print('Model:',path,flush=True)
    cv2.setNumThreads(1)
    m=Interpreter(model_path=str(path),num_threads=cfg['threads']);m.allocate_tensors();io=pc.ModelIO(m)
    print('Tensors:',json.dumps(io.describe()),flush=True)
    x=io.new_slot();x.fill(io.pad);m.set_tensor(io.input_index,x)
    m.invoke();m.invoke()
    output=m.get_tensor(io.output_index)
    boxes,classes,scores=pc.decode(output,io,cfg['confidence'])
    pc.nms_fast(boxes,classes,scores,cfg['iou'],cfg['max_detections'],cv2)
    print('PASS: LiteRT model invocation and native NMS.',flush=True)
    print('Opening camera using device =',cfg['device'],flush=True)
    camera=Camera(cfg)
    try:
        print('Camera:',json.dumps(camera.info),flush=True)
        for _ in range(5):frame=camera.read()
        print('PASS: five camera reads; payload =',len(frame) if isinstance(frame,bytes) else frame.shape,flush=True)
    finally:camera.close()
    print('\nChecks passed. Start the zone dashboard: bash run.sh',flush=True)
