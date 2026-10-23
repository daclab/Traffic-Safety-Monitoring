"""Run on Linux with requirements installed: python -m unittest discover -s tests -v."""
import json
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import pipeline_core as pc
from camera import jpeg_size
from engine import Engine,Latest
from main import serve
from zone_logic import Debounce,Tracker,Zone,validate_zone


class GeometryTests(unittest.TestCase):
    def test_polygons_and_entry(self):
        pts=validate_zone([[.2,.2],[.8,.2],[.8,.8],[.2,.8]])
        z=Zone(pts,640,480)
        self.assertTrue(z.contains([.3,.3,.4,.6]))
        self.assertFalse(z.contains([.3,.3,.4,.9]))
        self.assertTrue(z.contains([.3,.3,.4,.9],'overlap',.1))
        for bad in ([[0,0],[1,1],[0,1],[1,0]],[[0,0],[1,0],[1,0],[0,1]],[[0,0],[float('nan'),1],[1,0]]):
            with self.assertRaises(ValueError):validate_zone(bad)
    def test_debounce_and_short_lived_ids(self):
        d=Debounce(2,3)
        self.assertEqual([d.update(x) for x in (1,1,0,0,0)],['CHECKING','DANGER','DANGER','DANGER','CLEAR'])
        t=Tracker();a=t.update([{'class_id':0,'box':[.1,.1,.3,.5]}],1)[0]['track_id']
        b=t.update([{'class_id':0,'box':[.11,.1,.31,.5]}],1.1)[0]['track_id']
        c=t.update([{'class_id':0,'box':[.11,.1,.31,.5]}],3)[0]['track_id']
        self.assertEqual(a,b);self.assertNotEqual(b,c)
    def test_latest_reclaims_displaced_slots(self):
        released=[];m=Latest(released.append)
        m.put(1);m.put(2);self.assertEqual(released,[1]);self.assertEqual(m.get(),2)
        m.put(3);m.close();m.put(4);self.assertEqual(released,[1,3,4])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter=pc.load_interpreter_class()(model_path=str(ROOT/'models/yolo11n_silu_320_w8a8_int8io.tflite'),num_threads=4)
        cls.interpreter.allocate_tensors();cls.io=pc.ModelIO(cls.interpreter)
    def test_real_int8_model_and_xor(self):
        io=self.io;self.assertEqual(io.size,320);self.assertEqual(io.candidates,2100);self.assertTrue(io.xor_fast_path)
        slot=io.new_slot();rgb=np.arange(320*240*3,dtype=np.uint8).reshape(240,320,3);io.write_input(slot,rgb,0,40)
        np.testing.assert_array_equal(slot[0,40:280],(rgb.astype(np.int16)-128).astype(np.int8))
        self.assertTrue(np.all(slot[0,:40]==io.pad))
        self.interpreter.set_tensor(io.input_index,slot);self.interpreter.invoke()
        raw=self.interpreter.get_tensor(io.output_index)
        self.assertEqual(raw.shape,(1,84,2100));self.assertEqual(raw.dtype,np.int8)
        boxes,classes,scores=pc.decode(raw,io,.25);keep=pc.nms_fast(boxes,classes,scores,.7,100,cv2)
        self.assertTrue(np.all(np.isfinite(boxes[keep])))
    def test_int8_scores_and_native_nms(self):
        io=self.io;raw=np.full((1,84,2100),int(io.output_zero),np.int8)
        values=np.array([.5,.5,.3,.3,.8]);q=np.rint(values/io.output_scale+io.output_zero).astype(np.int8)
        for i in (0,1):raw[0,:4,i]=q[:4];raw[0,4,i]=q[4]
        boxes,classes,scores=pc.decode(raw,io,.25)
        self.assertEqual(len(scores),2);self.assertEqual(len(pc.nms_fast(boxes,classes,scores,.7,100,cv2)),1)
        np.testing.assert_allclose(scores,io.dequantize(q[4]),rtol=1e-5)


class FakeCamera:
    def __init__(self,cfg):
        self.raw=True;self.info={'signature':'test-camera','width':640,'height':480,'name':'Synthetic test camera','device':'test','path_mode':'Raw MJPEG / reduced decode'}
        img=np.full((480,640,3),70,np.uint8)
        cv2.putText(img,'SYNTHETIC TEST FRAME',(75,220),cv2.FONT_HERSHEY_SIMPLEX,.9,(240,240,240),2)
        self.jpeg=cv2.imencode('.jpg',img)[1].tobytes();self.next=time.monotonic()
    def read(self):
        self.next+=1/30;time.sleep(max(0,self.next-time.monotonic()));return self.jpeg
    def close(self):pass


class PipelineTests(unittest.TestCase):
    def test_actual_model_through_live_pipeline_and_http(self):
        cfg=json.loads((ROOT/'config.json').read_text());cfg['model']=str(ROOT/cfg['model']);cfg['led']='off'
        with tempfile.TemporaryDirectory() as d,patch('engine.Camera',FakeCamera):
            e=Engine(Path(d),cfg);server=serve(e,'127.0.0.1',0);base=f'http://127.0.0.1:{server.server_port}'
            try:
                e.start();deadline=time.monotonic()+15
                while not e.frame and not e.error and time.monotonic()<deadline:time.sleep(.05)
                self.assertIsNone(e.error);self.assertIsNotNone(e.frame)
                with urllib.request.urlopen(base+'/api/frame') as r:
                    meta=json.loads(r.headers['X-Frame-Meta']);self.assertEqual(jpeg_size(r.read()),(640,480));self.assertGreater(meta['seq'],0)
                with urllib.request.urlopen(base+'/api/frame?after=999999999&run='+e.run.name) as r:
                    self.assertEqual(r.status,204)
                with urllib.request.urlopen(base+'/api/frame?after=999999999&run=previous-session') as r:
                    self.assertEqual(r.status,200)
                req=urllib.request.Request(base+'/api/zone',data=json.dumps({'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'camera_signature':'test-camera'}).encode(),headers={'Content-Type':'application/json'})
                urllib.request.urlopen(req).close()
                target=e.completed+5;deadline=time.monotonic()+10
                while e.completed<target and not e.error and time.monotonic()<deadline:time.sleep(.05)
                metrics=json.loads(urllib.request.urlopen(base+'/api/status').read())
                self.assertIn(metrics['status'],('CLEAR','DANGER'));self.assertGreater(metrics['detection_fps'],0)
                self.assertTrue((Path(d)/'data/zone.json').exists())
                self.assertGreater(metrics['average']['read_to_decision_ms'],metrics['average']['invoke_ms'])
                print('Synthetic input integration passed; these are NOT UNO Q performance measurements.')
            finally:server.shutdown();server.server_close();e.close()
            self.assertTrue((e.run/'summary.json').exists())


if __name__=='__main__':unittest.main()
