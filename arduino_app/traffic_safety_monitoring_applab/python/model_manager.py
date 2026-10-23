"""Keep the HTTP service alive while replacing the complete inference engine."""
import threading
from pathlib import Path

from engine import Engine, atomic_json


MODELS = {
    'optimized': 'yolo11n_silu_320_w8a8_int8io.tflite',
    'baseline': 'yolo11n_silu_640_float32.tflite',
}


class ModelManager:
    def __init__(self, root, cfg):
        self.root = Path(root)
        self.cfg = cfg.copy()
        saved = self.root / 'data/model_selection.json'
        if saved.is_file():
            import json
            choice = json.loads(saved.read_text()).get('profile')
            if choice in MODELS and (self.root / 'models' / MODELS[choice]).is_file():
                self.cfg.update(profile=choice, model='models/' + MODELS[choice])
        self.lock = threading.RLock()
        self.current = Engine(self.root, self.cfg)
        self.switching = False
        self.switch_error = None
        self.pending = None
        self.worker = None
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.current, name)

    @property
    def error(self):
        # Keep the dashboard available so users can retry after a camera/model error.
        return None

    def start(self):
        self.current.start()

    def snapshot(self):
        with self.lock:
            result = self.current.snapshot()
            result['model_choices'] = [
                {'profile': key, 'name': name,
                 'available': (self.root / 'models' / name).is_file()}
                for key, name in MODELS.items()
            ]
            result['switching_model'] = self.pending if self.switching else None
            result['model_switch_error'] = self.switch_error
            if self.switching:
                result.update(status='SWITCHING', detections=[], inside_count=0,
                              detection_fps=0, average={})
            return result

    def select_model(self, profile):
        if profile not in MODELS:
            raise ValueError('Choose baseline or optimized.')
        path = self.root / 'models' / MODELS[profile]
        if not path.is_file():
            raise ValueError('Model file missing: ' + str(path))
        with self.lock:
            if self.closed or self.switching:
                raise ValueError('A model switch is already in progress or the app is stopping.')
            self.switching = True
            self.pending = profile
            self.switch_error = None
            self.current.led.set('SWITCHING')
            self.worker = threading.Thread(target=self._switch, args=(profile,),
                                           name='model-switch', daemon=True)
            self.worker.start()
        return {'accepted': True, 'profile': profile}

    def _switch(self, profile):
        try:
            old = self.current
            cfg = old.cfg.copy()
            old.close()
            cfg.update(profile=profile, model='models/' + MODELS[profile])
            replacement = Engine(self.root, cfg)
            with self.lock:
                self.current = replacement
            replacement.start()
            # Engine startup includes allocation, warmup and opening the camera.
            import time
            deadline = time.monotonic() + 60
            while replacement.phase == 'STARTING' and not replacement.error:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Model/camera startup timed out.')
                time.sleep(.1)
            if replacement.error:
                raise RuntimeError(replacement.error)
            atomic_json(self.root / 'data/model_selection.json', {'profile': profile})
        except Exception as exc:
            self.switch_error = str(exc)
            with self.current.lock:
                self.current.error = self.switch_error
                self.current.phase = 'ERROR'
            self.current.led.set('ERROR')
        finally:
            with self.lock:
                self.switching = False
                self.pending = None

    def save_zone(self, *args):
        with self.lock:
            if self.switching:
                raise ValueError('Wait for model switching to finish.')
            return self.current.save_zone(*args)

    def apply_settings(self, *args):
        with self.lock:
            if self.switching:
                raise ValueError('Wait for model switching to finish.')
            return self.current.apply_settings(*args)

    def close(self):
        with self.lock:
            self.closed = True
            worker = self.worker
        if worker:
            worker.join()
        self.current.close()
