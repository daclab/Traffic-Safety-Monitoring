"""Live pipeline using the user's unmodified pipeline_core model math."""
import csv
import io as bytes_io
import hashlib
import json
import math
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import pipeline_core as pc
from camera import Camera, jpeg_size
from profiles import apply_profile
from zone_logic import LABELS, Debounce, Tracker, Zone, selected_classes, validate_zone


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)


class Latest:
    """One pending work item; replacing it disposes its owned input slot."""
    def __init__(self,dispose=None):
        self.condition=threading.Condition();self.item=None;self.closed=False
        self.dropped=0;self.dispose=dispose
    def put(self,item):
        with self.condition:
            if self.closed:
                if self.dispose:self.dispose(item)
                return
            if self.item is not None:
                self.dropped+=1
                if self.dispose:self.dispose(self.item)
            self.item=item;self.condition.notify()
    def get(self,timeout=.25):
        with self.condition:
            if self.item is None and not self.closed:self.condition.wait(timeout)
            item,self.item=self.item,None
            return item
    def close(self):
        if getattr(self,'_closed',False):return
        with self.condition:
            self.closed=True
            if self.item is not None and self.dispose:self.dispose(self.item)
            self.item=None;self.condition.notify_all()


class Led:
    def __init__(self,mode):
        self.lock=threading.Lock();self.paths={};self.last=None;self.message='Disabled in config'
        if mode=='off':return
        for root,names in [(Path('/dev/leds/builtin'),{'red':'led1_r','green':'led1_g','blue':'led1_b'}),
                           (Path('/sys/class/leds'),{'red':'red:user','green':'green:user','blue':'blue:user'})]:
            paths={k:root/v/'brightness' for k,v in names.items()}
            if all(p.exists() and os.access(p,os.W_OK) for p in paths.values()):
                self.paths=paths;self.message='Built-in RGB LED 1';break
        if not self.paths:
            self.message='LED unavailable: check host Linux LED permissions'
            return
        # Do not change system LED2. Disable LED1 triggers where writable.
        for p in self.paths.values():
            t=p.parent/'trigger'
            try:
                if t.exists() and os.access(t,os.W_OK):t.write_text('none')
            except OSError:pass
        self.set('CALIBRATE')

    def set(self,status):
        with self.lock:
            if not self.paths or self.last==status:return
            try:
                for p in self.paths.values():p.write_text('0')
                color={'DANGER':'red','CLEAR':'green'}.get(status)
                if color:
                    p=self.paths[color];maximum=p.parent/'max_brightness'
                    p.write_text(maximum.read_text().strip() if maximum.exists() else '1')
                self.last=status
            except OSError as e:
                self.message='LED write failed: '+str(e);self.paths={}


class Engine:
    def __init__(self,root,cfg):
        cfg=apply_profile(cfg.copy())
        self.available_cpus=set(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else set()
        self.post_state=(Tracker(),-1,None,None,'STARTING')
        self.root=Path(root);self.cfg=cfg.copy();self.data=self.root/'data';self.data.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.stop=threading.Event();self.started=time.monotonic()
        self.raw=Latest();self.ready=None;self.preview=Latest();self.post=queue.Queue(2)
        self.threads=[];self.camera=None
        if cfg['led']=='external':
            from external_leds import ExternalLed
            self.led=ExternalLed(cfg.get('router_socket','/var/run/arduino-router.sock'),cfg['stale_seconds'])
        else:self.led=Led(cfg['led'])
        self.points=[];self.zone_signature=None;self.generation=0;self.error=None;self.warning=''
        self.phase='STARTING';self.camera_info={};self.model_info={}
        self.last_decision=0.;self.result={};self.frame=None
        self.capture_times=deque(maxlen=300);self.decision_times=deque(maxlen=300);self.preview_times=deque(maxlen=300)
        self.latencies=deque(maxlen=240);self.events=deque(maxlen=40)
        self.captured=0;self.completed=0;self.preview_count=0
        self.run=self.data/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');self.run.mkdir(parents=True)
        self.log=(self.run/'frames.csv').open('w',newline='')
        self.writer=csv.writer(self.log)
        self.writer.writerow(['frame_id','read_complete_s','decision_s','preprocess_ms','input_wait_ms','input_copy_ms','invoke_ms','output_copy_ms','post_wait_ms','postprocess_ms','read_to_decision_ms','status','inside_count'])
        self.event_log=(self.run/'events.csv').open('w',newline='');self.event_writer=csv.writer(self.event_log)
        self.event_writer.writerow(['time_utc','status','inside_count','frame_id'])
        self.last_flush=time.monotonic()
        try:
            p=self.data/'settings.json'
            if p.exists():self.apply_settings(json.loads(p.read_text()),persist=False)
            p=self.data/'zone.json'
            if p.exists():
                z=json.loads(p.read_text());self.points=validate_zone(z['points']);self.zone_signature=z.get('camera_signature')
        except (ValueError,KeyError,TypeError) as e:self.warning='Saved calibration/settings ignored: '+str(e);self.points=[]

    def spawn(self,target,name):
        def guarded():
            try:target()
            except Exception as e:
                traceback.print_exc()
                with self.lock:self.error=f'{name}: {e}';self.phase='ERROR'
                self.stop.set();self.led.set('ERROR')
        t=threading.Thread(target=guarded,name=name,daemon=True);self.threads.append(t);t.start()

    def start(self):
        self.spawn(self.inference,'inference')
        self.spawn(self.watchdog,'watchdog')

    def auxiliary(self):
        cv2.ocl.setUseOpenCL(False)
        try:
            allowed=self.available_cpus or os.sched_getaffinity(0)
            os.sched_setaffinity(0,{max(allowed)})
            os.setpriority(os.PRIO_PROCESS,threading.get_native_id(),self.cfg['aux_nice'])
        except (AttributeError,OSError):pass

    def inference(self):
        cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
        try:
            available=sorted(self.available_cpus)
            if self.cfg['sequential']:
                if 0 not in available:raise RuntimeError('Baseline requires CPU 0, which is outside the allowed CPU affinity.')
                os.sched_setaffinity(0,{0})
            else:os.sched_setaffinity(0,set(available[:4]))
        except (AttributeError,OSError) as e:
            if self.cfg['sequential']:raise RuntimeError('Could not pin the baseline worker to CPU 0') from e
        path=Path(self.cfg['model']).expanduser()
        if not path.is_absolute():path=self.root/path
        if not path.is_file():raise FileNotFoundError(f'Model not found: {path}. Use the bundled models/ file or pass --model with an existing file.')
        interpreter=pc.load_interpreter_class()(model_path=str(path),num_threads=self.cfg['threads'])
        interpreter.allocate_tensors();io=pc.ModelIO(interpreter)
        if len(interpreter.get_input_details())!=1 or len(interpreter.get_output_details())!=1:
            raise ValueError('This app expects the supplied single-input, single-output YOLO11n export.')
        if self.cfg['sequential'] and (io.size!=640 or io.input_dtype!=np.float32):
            raise ValueError('Baseline profile requires the 640 FP32 model.')
        with self.lock:self.model_info={'profile':self.cfg['profile'],'sequential':self.cfg['sequential'],'preprocessor':self.cfg['preprocessor'],'nms':self.cfg['nms'],'infer_cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,**io.describe(),'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backend':'CPU · XNNPACK','threads':self.cfg['threads']}
        warm=io.new_slot();warm.fill(io.pad)
        for _ in range(5):interpreter.set_tensor(io.input_index,warm);interpreter.invoke()
        self.camera=Camera(self.cfg)
        with self.lock:
            self.camera_info=self.camera.info
            if self.points and self.zone_signature!=self.camera_info['signature']:
                self.points=[];self.warning='Camera identity or resolution changed. Recalibrate the zone.'
            self.phase='RUNNING'
        self.io=io;self.slots=[io.new_slot() for _ in range(4)]
        self.free=queue.Queue()
        for i in range(4):self.free.put(i)
        self.ready=Latest(lambda item:self.free.put(item['slot']))
        atomic_json(self.run/'config.json',{'config':self.cfg,'model':self.model_info,'camera':self.camera_info})
        print('Model:',json.dumps(self.model_info),flush=True)
        print('Camera:',json.dumps(self.camera_info),flush=True)
        print('Profile:', self.cfg['profile'], '| threads:',self.cfg['threads'],'| preprocessing:',self.cfg['preprocessor'],'| NMS:',self.cfg['nms'],flush=True)
        if abs(self.camera_info.get('reported_fps',self.cfg['capture_fps'])-self.cfg['capture_fps'])>.5:
            print('WARNING: camera did not report the requested capture FPS. Check supported frame intervals.',flush=True)
        self.spawn(self.capture,'capture')
        if self.cfg['sequential']:
            self.spawn(self.encode,'preview')
            self.sequential_loop(interpreter)
            return
        self.spawn(self.preprocess,'preprocess')
        self.spawn(self.postprocess,'postprocess');self.spawn(self.encode,'preview')
        while not self.stop.is_set():
            work=self.ready.get()
            if work is None:continue
            t0=time.monotonic()
            try:interpreter.set_tensor(io.input_index,self.slots[work['slot']])
            finally:self.free.put(work['slot'])
            t1=time.monotonic();interpreter.invoke();t2=time.monotonic()
            raw=interpreter.get_tensor(io.output_index);t3=time.monotonic()
            work.update(raw=raw,input_wait_ms=(t0-work['prepared'])*1000,
                input_copy_ms=(t1-t0)*1000,invoke_ms=(t2-t1)*1000,output_copy_ms=(t3-t2)*1000,inferred=t3)
            while not self.stop.is_set():
                try:self.post.put(work,timeout=.2);break
                except queue.Full:pass

    def capture(self):
        self.auxiliary()
        try:
            while not self.stop.is_set():
                payload=self.camera.read();now=time.monotonic()
                with self.lock:
                    self.captured+=1;seq=self.captured;self.capture_times.append(now)
                self.raw.put({'seq':seq,'captured':now,'payload':payload,'compressed':self.camera.raw})
        finally:self.camera.close()

    def preprocess(self):
        self.auxiliary();io=self.io;w=self.camera_info['width'];h=self.camera_info['height']
        scale,nw,nh,left,top=pc.letterbox_geometry(w,h,io.size)
        factor=pc.reduced_decode_factor(w,h,nw,nh)
        flags={1:cv2.IMREAD_COLOR,2:cv2.IMREAD_REDUCED_COLOR_2,4:cv2.IMREAD_REDUCED_COLOR_4,8:cv2.IMREAD_REDUCED_COLOR_8}[factor]
        while not self.stop.is_set():
            work=self.raw.get()
            if work is None:continue
            try:slot=self.free.get(timeout=.25)
            except queue.Empty:continue
            submitted=False
            try:
                t0=time.monotonic()
                if work['compressed']:
                    if jpeg_size(work['payload'])!=(w,h):raise RuntimeError('Camera dimensions changed. Restart and recalibrate.')
                    bgr=cv2.imdecode(np.frombuffer(work['payload'],np.uint8),flags)
                else:bgr=work['payload']
                if bgr is None:raise RuntimeError('Camera JPEG could not be decoded.')
                if (bgr.shape[1],bgr.shape[0])!=(nw,nh):bgr=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);io.write_input(self.slots[slot],rgb,left,top)
                now=time.monotonic()
                work.update(slot=slot,prepared=now,preprocess_ms=(now-t0)*1000,geometry=(scale,left,top,w,h))
                self.ready.put(work);submitted=True
            finally:
                if not submitted:self.free.put(slot)

    def postprocess(self):
        self.auxiliary()
        while not self.stop.is_set():
            try:work=self.post.get(timeout=.25)
            except queue.Empty:continue
            self.process_one(work)

    def process_one(self,work):
        tracker,generation,zone,debounce,previous=self.post_state
        t0=time.monotonic()
        with self.lock:cfg=self.cfg.copy();points=[p[:] for p in self.points];gen=self.generation
        if gen!=generation or (self.last_decision and t0-self.last_decision>cfg['stale_seconds']):
            generation=gen;zone=Zone(points,*work['geometry'][-2:]) if points else None
            debounce=Debounce(cfg['enter_frames'],cfg['clear_frames'])
        boxes,classes,scores=pc.decode(work.pop('raw'),self.io,cfg['confidence'])
        kept=(pc.nms_legacy(boxes,classes,scores,cfg['iou'],cfg['max_detections']) if cfg['sequential']
              else pc.nms_fast(boxes,classes,scores,cfg['iou'],cfg['max_detections'],cv2))
        mapped=pc.to_original(boxes[kept],*work['geometry'])
        w,h=work['geometry'][-2:];selected=selected_classes(cfg['classes']);dets=[]
        for (x,y,bw,bh),c,s in zip(mapped,classes[kept],scores[kept]):
            box=[float(np.clip(x/w,0,1)),float(np.clip(y/h,0,1)),float(np.clip((x+bw)/w,0,1)),float(np.clip((y+bh)/h,0,1))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            eligible=int(c) in selected
            inside=bool(eligible and zone and zone.contains(box,cfg['zone_mode'],cfg['overlap_fraction']))
            dets.append({'box':box,'class_id':int(c),'label':LABELS[int(c)],'score':round(float(s),3),'inside':inside,'selected':eligible})
        dets=tracker.update(dets,t0);inside=sum(d['inside'] for d in dets)
        status=debounce.update(inside>0) if zone else 'CALIBRATE'
        now=time.monotonic();latency=(now-work['captured'])*1000
        timing={k:work[k] for k in ('preprocess_ms','input_wait_ms','input_copy_ms','invoke_ms','output_copy_ms')}
        timing.update(post_wait_ms=(t0-work['inferred'])*1000,postprocess_ms=(now-t0)*1000,
                      capture_wait_ms=(work['prepared']-work['preprocess_ms']/1000-work['captured'])*1000,read_to_decision_ms=latency)
        with self.lock:
            # Recalibration during an inference cannot publish a result for the old zone.
            if gen!=self.generation:return
            self.last_decision=now;self.completed+=1;self.decision_times.append(now);self.latencies.append(timing)
            self.result={'seq':work['seq'],'status':status,'inside_count':inside,'detections':dets,
                'zone':points,'zone_mode':cfg['zone_mode'],'timing':{k:round(v,2) for k,v in timing.items()},'generation':gen}
            self.led.set(status)
            if status!=previous:
                event={'time':datetime.now(timezone.utc).isoformat(timespec='seconds'),'status':status,'inside_count':inside,'frame_id':work['seq']}
                self.events.appendleft(event);self.event_writer.writerow(event.values());previous=status
            self.writer.writerow([work['seq'],work['captured']-self.started,now-self.started]+[timing[k] for k in ('preprocess_ms','input_wait_ms','input_copy_ms','invoke_ms','output_copy_ms','post_wait_ms','postprocess_ms','read_to_decision_ms')]+[status,inside])
            if now-self.last_flush>=1:self.log.flush();self.event_log.flush();self.last_flush=now
            meta=self.result.copy()
        self.preview.put({'payload':work['payload'],'compressed':work['compressed'],'meta':meta})
        self.post_state=(tracker,generation,zone,debounce,previous)

    def sequential_loop(self,interpreter):
        # Camera transport/preview are common infrastructure. Model stages do
        # not overlap and all execute on this CPU-0, one-thread worker.
        from PIL import Image
        io=self.io;slot=io.new_slot()
        w,h=self.camera_info['width'],self.camera_info['height']
        scale,nw,nh,left,top=pc.letterbox_geometry(w,h,io.size)
        while not self.stop.is_set():
            work=self.raw.get()
            if work is None:continue
            t0=time.monotonic()
            if work['compressed']:
                with Image.open(bytes_io.BytesIO(work['payload'])) as source:
                    rgb=source.convert('RGB')
            else:
                rgb=Image.fromarray(work['payload'][:,:,::-1].copy())
            if rgb.size!=(w,h):raise RuntimeError('Camera dimensions changed: restart and recalibrate.')
            rgb=np.asarray(rgb.resize((nw,nh),Image.Resampling.BILINEAR),dtype=np.uint8)
            io.write_input(slot,rgb,left,top)
            t1=time.monotonic();interpreter.set_tensor(io.input_index,slot)
            t2=time.monotonic();interpreter.invoke();t3=time.monotonic()
            raw=interpreter.get_tensor(io.output_index);t4=time.monotonic()
            work.update(raw=raw,prepared=t1,geometry=(scale,left,top,w,h),
                preprocess_ms=(t1-t0)*1000,input_wait_ms=0.,input_copy_ms=(t2-t1)*1000,
                invoke_ms=(t3-t2)*1000,output_copy_ms=(t4-t3)*1000,inferred=t4)
            self.process_one(work)

    def encode(self):
        self.auxiliary();deadline=time.monotonic()
        while not self.stop.is_set():
            work=self.preview.get()
            if work is None:continue
            now=time.monotonic()
            if now<deadline:continue
            with self.lock:fps=self.cfg['preview_fps'];quality=self.cfg['jpeg_quality']
            while deadline<=now:deadline+=1/fps
            if work['compressed']:jpeg=work['payload']  # No decode/re-encode for browser; canvas draws boxes.
            else:
                ok,buf=cv2.imencode('.jpg',work['payload'],[cv2.IMWRITE_JPEG_QUALITY,quality])
                if not ok:continue
                jpeg=buf.tobytes()
            with self.lock:
                if work['meta']['generation']!=self.generation:continue
                self.frame=(jpeg,work['meta']);self.preview_count+=1;self.preview_times.append(time.monotonic())

    def watchdog(self):
        while not self.stop.wait(.25):
            with self.lock:
                if self.last_decision and time.monotonic()-self.last_decision>self.cfg['stale_seconds']:
                    self.led.set('STALE')

    @staticmethod
    def rate(times,now):
        recent=[t for t in times if now-t<5]
        return round((len(recent)-1)/(recent[-1]-recent[0]),2) if len(recent)>1 and now-recent[-1]<2 else 0.

    def snapshot(self):
        with self.lock:
            now=time.monotonic();result=self.result.copy()
            pending=('CHECKING' if self.points else 'CALIBRATE') if self.phase=='RUNNING' else self.phase
            status=result.get('status',pending)
            if self.error:status='ERROR'
            elif self.last_decision and now-self.last_decision>self.cfg['stale_seconds']:status='STALE'
            samples=list(self.latencies)
            avg={k:round(float(np.mean([m[k] for m in samples])),2) for k in samples[0]} if samples else {}
            p95=round(float(np.percentile([m['read_to_decision_ms'] for m in samples],95)),2) if samples else None
            try:rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/2**20
            except (OSError,ValueError):rss=None
            return {**result,'status':status,'error':self.error,'warning':self.warning,'camera':self.camera_info,
                'model':self.model_info,'settings':{k:self.cfg[k] for k in ('confidence','classes','zone_mode','preview_fps','enter_frames','clear_frames')},
                'zone':self.points,'camera_signature':self.camera_info.get('signature'),
                'camera_fps':self.rate(self.capture_times,now),'detection_fps':self.rate(self.decision_times,now),
                'preview_fps':self.rate(self.preview_times,now),'average':avg,'latency_p95_ms':p95,
                'last_result_age_ms':round((now-self.last_decision)*1000,1) if self.last_decision else None,
                'completed':self.completed,'captured':self.captured,'preview_count':self.preview_count,
                'capture_dropped':self.raw.dropped,'preprocess_dropped':self.ready.dropped if self.ready else 0,
                'uptime_s':round(now-self.started,1),'rss_mib':round(rss,1) if rss else None,
                'led_available':bool(self.led.paths),'led_message':self.led.message,'events':list(self.events),
                'run_id':self.run.name,'preview_target_fps':self.cfg['preview_fps']}

    def save_zone(self,points,signature):
        pts=validate_zone(points)
        with self.lock:
            if not self.camera_info or signature!=self.camera_info['signature']:raise ValueError('Camera changed or is not ready. Refresh the dashboard before calibrating.')
            atomic_json(self.data/'zone.json',{'points':pts,'camera_signature':signature,'coordinate_system':'normalized camera image'})
            self.points=pts;self.zone_signature=signature;self.generation+=1;self.warning=''
            self.result={};self.frame=None;self.led.set('CHECKING')
        return pts

    def apply_settings(self,settings,persist=True):
        allowed={'confidence','classes','zone_mode','preview_fps'}
        if not isinstance(settings,dict) or set(settings)-allowed:raise ValueError('Unsupported settings.')
        new={}
        for k,v in settings.items():
            if k=='classes':selected_classes(v);new[k]=v
            elif k=='zone_mode':
                if v not in ('foot','overlap'):raise ValueError('Unknown zone mode.')
                new[k]=v
            else:
                v=float(v)
                if not math.isfinite(v) or not ((.05<=v<=.95) if k=='confidence' else (1<=v<=30)):
                    raise ValueError('Confidence must be 0.05–0.95; preview target must be 1–30 FPS.')
                new[k]=v
        with self.lock:
            if persist:atomic_json(self.data/'settings.json',{k:new.get(k,self.cfg[k]) for k in allowed})
            self.cfg.update(new);self.generation+=1;self.result={};self.frame=None;self.led.set('CHECKING')

    def close(self):
        self.stop.set();self.raw.close();self.preview.close()
        if self.ready:self.ready.close()
        for t in self.threads:t.join(timeout=15)
        alive=[t.name for t in self.threads if t.is_alive()]
        if alive:
            self.led.set('ERROR')
            raise RuntimeError('Workers did not stop; restart app: '+', '.join(alive))
        self.led.set('STOPPED')
        if hasattr(self.led,'close'):self.led.close()
        with self.lock:
            self.log.flush();self.event_log.flush()
            atomic_json(self.run/'summary.json',self.snapshot())
            self.log.close();self.event_log.close()
        self._closed=True
