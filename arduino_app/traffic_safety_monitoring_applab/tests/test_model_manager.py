"""Portable lifecycle tests; no camera, Linux modules, or model runtime required."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import threading


class FakeEngine:
    instances = []
    fail_start = False

    def __init__(self, root, cfg):
        self.cfg = cfg.copy()
        self.phase = 'STARTING'
        self.error = None
        self.lock = threading.RLock()
        self.closed = False
        self.led = types.SimpleNamespace(set=lambda state: None)
        self.instances.append(self)

    def start(self):
        if self.fail_start:
            self.error = 'Camera not connected'
            self.phase = 'ERROR'
        else:
            self.phase = 'RUNNING'

    def snapshot(self):
        return {'status': self.phase, 'model': {'name': self.cfg['model']}}

    def close(self):
        self.closed = True


class ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        FakeEngine.instances = []
        FakeEngine.fail_start = False
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'models').mkdir()
        (self.root / 'data').mkdir()
        def atomic(path, value):
            path.write_text(json.dumps(value))
        fake = types.SimpleNamespace(Engine=FakeEngine, atomic_json=atomic)
        path = Path(__file__).resolve().parents[1] / 'python/model_manager.py'
        spec = importlib.util.spec_from_file_location('manager_under_test', path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'engine': fake}):
            spec.loader.exec_module(self.module)
        for name in self.module.MODELS.values():
            (self.root / 'models' / name).write_bytes(b'test model marker')
        self.manager = self.module.ModelManager(self.root, {
            'profile': 'optimized', 'model': 'models/' + self.module.MODELS['optimized'],
            'capture_width': 640, 'capture_height': 480,
        })
        self.addCleanup(self.manager.close)

    def test_switch_stops_previous_engine_and_persists_choice(self):
        old = self.manager.current
        self.manager.select_model('baseline')
        self.manager.worker.join(3)
        self.assertFalse(self.manager.worker.is_alive())
        self.assertTrue(old.closed)
        self.assertEqual(self.manager.current.cfg['profile'], 'baseline')
        self.assertEqual(self.manager.current.cfg['capture_width'], 640)
        self.assertEqual(self.manager.current.phase, 'RUNNING')
        self.assertFalse(self.manager.switching)
        saved = json.loads((self.root / 'data/model_selection.json').read_text())
        self.assertEqual(saved['profile'], 'baseline')
        restored = self.module.ModelManager(self.root, {'model': 'unused'})
        self.assertEqual(restored.current.cfg['profile'], 'baseline')
        restored.close()

    def test_invalid_and_missing_model_rejected_without_stopping_engine(self):
        with self.assertRaises(ValueError):
            self.manager.select_model('../unknown')
        (self.root / 'models' / self.module.MODELS['baseline']).unlink()
        with self.assertRaises(ValueError):
            self.manager.select_model('baseline')
        self.assertFalse(self.manager.current.closed)

    def test_camera_failure_is_visible_and_does_not_persist_choice(self):
        FakeEngine.fail_start = True
        self.manager.select_model('baseline')
        self.manager.worker.join(3)
        status = self.manager.snapshot()
        self.assertIn('Camera not connected', status['model_switch_error'])
        self.assertEqual(status['status'], 'ERROR')
        self.assertFalse((self.root / 'data/model_selection.json').exists())
        FakeEngine.fail_start = False
        self.manager.select_model('optimized')
        self.manager.worker.join(3)
        self.assertEqual(self.manager.current.phase, 'RUNNING')


if __name__ == '__main__':
    unittest.main()
