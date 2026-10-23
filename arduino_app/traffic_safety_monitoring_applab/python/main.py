"""Command-line UNO Q application; run through ../run_zone.py or ../run.sh."""
import argparse
import os
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from engine import Engine
from zone_logic import selected_classes

ROOT=Path(__file__).resolve().parent.parent


def serve(engine,host,port):
    assets=ROOT/'assets'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data=b'',kind='application/json',code=200,headers=None):
            self.send_response(code)
            self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            for k,v in (headers or {}).items():self.send_header(k,v)
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        def json(self,value,code=200):self.reply(json.dumps(value,allow_nan=False).encode(),code=code)
        def do_GET(self):
            self.connection.settimeout(4);url=urlsplit(self.path)
            static={'/':('index.html','text/html; charset=utf-8'),'/app.css':('app.css','text/css'),'/app.js':('app.js','application/javascript')}
            if url.path in static:
                name,kind=static[url.path];return self.reply((assets/name).read_bytes(),kind)
            if url.path=='/api/status':return self.json(engine.snapshot())
            if url.path=='/api/frame':
                if getattr(engine,'switching',False):return self.reply(code=204)
                try:after=int(parse_qs(url.query).get('after',['0'])[0])
                except ValueError:return self.json({'error':'Invalid frame ID'},400)
                with engine.lock:frame=engine.frame
                same_run=parse_qs(url.query).get('run',[''])[0]==engine.run.name
                if frame is None or (same_run and frame[1]['seq']<=after):return self.reply(code=204)
                jpeg,meta=frame
                meta={**meta,'run_id':engine.run.name}
                # One response pairs a JPEG with the detections from that EXACT image.
                return self.reply(jpeg,'image/jpeg',headers={'X-Frame-Meta':json.dumps(meta,separators=(',',':'))})
            if url.path in ('/api/events.csv','/api/frames.csv'):
                if getattr(engine,'switching',False):return self.json({'error':'Wait for model switching to finish.'},409)
                with engine.lock:
                    handle=engine.event_log if url.path.endswith('events.csv') else engine.log
                    handle.flush();data=Path(handle.name).read_bytes()
                return self.reply(data,'text/csv',headers={'Content-Disposition':'attachment; filename="'+Path(handle.name).name+'"'})
            if url.path=='/api/report.json':
                return self.reply(json.dumps(engine.snapshot(),indent=2).encode(),headers={'Content-Disposition':'attachment; filename="traffic_safety_session.json"'})
            self.json({'error':'Not found'},404)
        def do_POST(self):
            self.connection.settimeout(4)
            try:
                # Local LAN control; reject cross-origin browser writes.
                origin=self.headers.get('Origin')
                if origin and urlsplit(origin).netloc!=self.headers.get('Host'):return self.json({'error':'Origin mismatch'},403)
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON request required.')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192:raise ValueError('Invalid request size.')
                body=json.loads(self.rfile.read(n))
                if not isinstance(body,dict):raise ValueError('Expected an object.')
                if self.path=='/api/model':
                    return self.json(engine.select_model(body.get('profile')),202)
                if self.path=='/api/zone':
                    points=engine.save_zone(body['points'],body['camera_signature']);return self.json({'points':points})
                if self.path=='/api/settings':
                    engine.apply_settings(body);return self.json({'ok':True})
                self.json({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError,AttributeError,UnicodeDecodeError) as e:self.json({'error':str(e)},400)
            except OSError as e:self.json({'error':'Cannot save configuration: '+str(e)},500)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        allow_reuse_address=True
    server=Server((host,port),Handler)
    threading.Thread(target=server.serve_forever,name='http',daemon=True).start()
    return server


def main():
    p=argparse.ArgumentParser(description='Traffic Safety Monitoring for Arduino UNO Q')
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--model');p.add_argument('--device');p.add_argument('--port',type=int)
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--backend',choices=('cpu',),default='cpu')
    p.add_argument('--threads',type=int)
    p.add_argument('--profile',choices=('auto','optimized','baseline'),default='auto')
    p.add_argument('--capture-fps',type=float)
    p.add_argument('--exposure-ms',type=float)
    p.add_argument('--gain',type=int)
    p.add_argument('--classes',help='person (default), traffic, all, or comma-separated COCO labels')
    p.add_argument('--router-socket',default=os.environ.get('APP_SOCKET','/var/run/arduino-router.sock'))
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='Check the model/runtime and five camera reads, then exit')
    mode.add_argument('--camera-only',action='store_true',help='Show a simple camera preview without loading a model')
    mode.add_argument('--led-test',action='store_true',help='Test external red, green and off; no camera/model is opened')
    p.add_argument('--raw-mjpeg',action=argparse.BooleanOptionalAction,default=None,help='Request raw MJPEG; optimized uses reduced decode, baseline uses PIL full decode')
    p.add_argument('--preview-fps',type=float)
    p.add_argument('--led',choices=('external','auto','off'),help='external=D7/D8 with MCU sketch; auto=built-in RGB LED; off=no LED')
    p.add_argument('--seconds',type=float,default=0,help='Zone mode duration; 0 runs until Ctrl+C')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    for k in ('model','device','port','threads','classes','router_socket'):
        if getattr(a,k) is not None:cfg[k]=getattr(a,k)
    if a.raw_mjpeg is not None:cfg['prefer_raw_mjpeg']=a.raw_mjpeg
    if a.preview_fps is not None:cfg['preview_fps']=a.preview_fps
    if a.led is not None:cfg['led']=a.led
    for k in ('profile','capture_fps','exposure_ms','gain'):
        if getattr(a,k) is not None:cfg[k]=getattr(a,k)
    from profiles import apply_profile
    apply_profile(cfg)
    if a.threads is not None and a.threads!=cfg['threads']:
        p.error('This profile fixes threads to '+str(cfg['threads']))
    if cfg.get('exposure_ms') is not None and not 0.1<=cfg['exposure_ms']<1000/cfg['capture_fps']:
        p.error('Exposure must be positive and shorter than the requested frame interval.')
    if cfg.get('gain') is not None and not 0<=cfg['gain']<=100:p.error('Gain must be 0–100 for this ELP camera.')
    if a.seconds<0:p.error('--seconds must be nonnegative')
    for k in ('capture_width','capture_height','threads','enter_frames','clear_frames','max_detections'):
        if type(cfg[k]) is not int or cfg[k]<1:raise ValueError(f'{k} must be a positive integer.')
    if not 1<=cfg['port']<=65535 or not 1<=cfg['capture_fps']<=120 or not 1<=cfg['preview_fps']<=30:
        raise ValueError('Invalid port or frame rate.')
    if not 0<cfg['confidence']<1 or not 0<cfg['iou']<1 or not 0<cfg['overlap_fraction']<=1:
        raise ValueError('Invalid detection threshold.')
    if cfg['zone_mode'] not in ('foot','overlap') or cfg['led'] not in ('auto','off','external'):raise ValueError('Invalid zone/LED mode.')
    if cfg['stale_seconds']<=0 or not 0<=cfg['aux_nice']<=19:raise ValueError('Invalid stale timeout / nice value.')
    selected_classes(cfg['classes'])
    if a.led_test:
        from external_leds import test_leds
        test_leds(cfg['router_socket'])
        return 0
    if a.check:
        from check_runtime import check
        check(ROOT,cfg)
        return 0
    if a.camera_only:
        from camera_preview import run
        return run(cfg,a.host)
    from model_manager import ModelManager
    engine=ModelManager(ROOT,cfg);server=None;done=threading.Event()
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda *_:done.set())
    try:
        server=serve(engine,a.host,cfg['port'])
        print(f"Traffic Safety Monitoring dashboard: http://<UNO-Q-IP>:{cfg['port']}/",flush=True)
        print(f"Open http://ArdnQ.local:{cfg['port']}/ on your laptop, or use the board Wi-Fi IP.",flush=True)
        print('Starting CPU pipeline. Press Ctrl+C to stop and release the camera.',flush=True)
        engine.start()
        started=time.monotonic();next_report=started+5
        while not done.wait(.5):
            if engine.error:
                print('STOPPED:',engine.error,flush=True)
                return 1
            now=time.monotonic()
            if a.seconds and now-started>=a.seconds:break
            if now>=next_report:
                next_report=now+5;m=engine.snapshot();avg=m['average']
                print(f"{m['status']} | camera {m['camera_fps']:.1f} FPS | detection {m['detection_fps']:.1f} FPS | invoke {avg.get('invoke_ms',0):.1f} ms | read-to-decision {avg.get('read_to_decision_ms',0):.1f} ms | {m['completed']} completed",flush=True)
    finally:
        if server:server.shutdown();server.server_close()
        engine.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
