from pathlib import Path

def apply_profile(cfg):
    mode=cfg.get('profile','auto')
    if mode=='auto':
        name=Path(cfg['model']).name
        if name=='yolo11n_silu_640_float32.tflite':mode='baseline'
        elif name=='yolo11n_silu_320_w8a8_int8io.tflite':mode='optimized'
        else:raise ValueError('Unknown model name: choose --profile baseline or --profile optimized explicitly.')
    cfg['profile']=mode
    cfg['threads']=1 if mode=='baseline' else 4
    cfg['preprocessor']='pil' if mode=='baseline' else 'cv2-reduced'
    cfg['nms']='legacy' if mode=='baseline' else 'fast'
    cfg['sequential']=mode=='baseline'
    return cfg
