import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from engine import Engine
from profiles import apply_profile
from test_app import FakeCamera
import pipeline_core as pc

class ProfileTests(unittest.TestCase):
    def test_automatic_selection(self):
        a=apply_profile({'model':'yolo11n_silu_640_float32.tflite'})
        self.assertEqual((a['threads'],a['sequential'],a['preprocessor'],a['nms']),(1,True,'pil','legacy'))
        b=apply_profile({'model':'yolo11n_silu_320_w8a8_int8io.tflite'})
        self.assertEqual((b['threads'],b['sequential'],b['nms']),(4,False,'fast'))

    def test_baseline_serial_stages_and_legacy_nms(self):
        seen=[];requested=[]
        class Interpreter:
            def __init__(self,**kw):requested.append(kw['num_threads'])
            def allocate_tensors(self):pass
            def get_input_details(self):return [dict(shape=np.array([1,640,640,3]),dtype=np.float32,index=0,quantization=(0,0))]
            def get_output_details(self):return [dict(shape=np.array([1,84,8400]),dtype=np.float32,index=1,quantization=(0,0))]
            def set_tensor(self,index,x):
                self_input=x
                assert self_input.shape==(1,640,640,3) and self_input.dtype==np.float32
                assert np.min(x)>=0 and np.max(x)<=1
            def invoke(self):seen.append(('invoke',threading.get_ident()));time.sleep(.005)
            def get_tensor(self,index):return np.zeros((1,84,8400),np.float32)
        def legacy(*args):seen.append(('post',threading.get_ident()));return pc.nms_legacy_original(*args)
        cfg=json.loads((ROOT/'config.json').read_text());cfg.update(profile='baseline',led='off')
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'yolo11n_silu_640_float32.tflite';model.write_bytes(b'test fake interpreter only');cfg['model']=str(model)
            pc.nms_legacy_original=pc.nms_legacy
            with patch('engine.Camera',FakeCamera),patch('engine.pc.load_interpreter_class',return_value=Interpreter),patch('engine.os.sched_getaffinity',return_value={0,1,2,3}),patch('engine.os.sched_setaffinity') as affinity,patch('engine.pc.nms_legacy',side_effect=legacy),patch('engine.pc.nms_fast',side_effect=AssertionError('Baseline used fast NMS')):
                e=Engine(d,cfg)
                try:
                    e.start();deadline=time.monotonic()+5
                    while e.completed<3 and not e.error and time.monotonic()<deadline:time.sleep(.02)
                    self.assertIsNone(e.error);self.assertGreaterEqual(e.completed,3)
                    names=[t.name for t in e.threads]
                    self.assertNotIn('preprocess',names);self.assertNotIn('postprocess',names)
                    self.assertEqual(requested,[1]);self.assertEqual(len({tid for _,tid in seen}),1)
                    self.assertIn(('post',seen[0][1]),seen)
                    self.assertIn(unittest.mock.call(0,{0}),affinity.call_args_list)
                    # Beyond five warmups, each invoke is followed by its own postprocess.
                    live=[name for name,_ in seen][5:]
                    self.assertEqual(live[:6],['invoke','post']*3)
                finally:e.close()
            del pc.nms_legacy_original
