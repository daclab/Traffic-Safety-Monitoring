"""Small camera-only web page, without model loading or zone logic."""
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import cv2
from camera import Camera

PAGE=b'''<!doctype html><html><meta name="viewport" content="width=device-width,initial-scale=1"><title>UNO Q camera check</title><style>body{background:#102b36;color:#e5f4f1;font:16px system-ui;max-width:1000px;margin:30px auto;padding:15px}img{display:block;width:100%;background:#09191f;border-radius:12px}pre{white-space:pre-wrap;color:#8fccc1}</style><h1>UNO Q camera check</h1><p>Camera-only mode. No YOLO model or zone decisions.</p><img id="feed" alt="Waiting for the first camera frame"><pre id="status"></pre><script>
let seq=0,previous='';async function frame(){const start=performance.now();try{const r=await fetch('/frame?after='+seq);if(r.ok&&r.status!==204){seq=Number(r.headers.get('X-Frame-ID'));const next=URL.createObjectURL(await r.blob());feed.src=next;if(previous)URL.revokeObjectURL(previous);previous=next;}}catch(e){}setTimeout(frame,Math.max(5,100-(performance.now()-start)));}async function info(){try{statusBox.textContent=JSON.stringify(await(await fetch('/status')).json(),null,2);}catch(e){statusBox.textContent='Disconnected';}setTimeout(info,1000);}const feed=document.getElementById('feed'),statusBox=document.getElementById('status');frame();info();</script></html>'''


def run(cfg,host):
    from urllib.parse import parse_qs,urlsplit
    cv2.setNumThreads(1);camera=Camera(cfg);stop=threading.Event();lock=threading.Lock()
    state={'seq':0,'jpeg':None,'error':None,'camera':camera.info,'reads':0,'read_fps':0.}
    print('Camera:',json.dumps(camera.info),flush=True)
    def capture():
        start=time.monotonic();next_preview=start
        try:
            while not stop.is_set():
                frame=camera.read();now=time.monotonic()
                with lock:state['reads']+=1;state['read_fps']=round(state['reads']/max(.001,now-start),1)
                if now<next_preview:continue
                while next_preview<=now:next_preview+=.1
                if isinstance(frame,bytes):jpeg=frame
                else:
                    ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
                    if not ok:continue
                    jpeg=buf.tobytes()
                with lock:state['seq']+=1;state['jpeg']=jpeg
        except Exception as e:
            with lock:state['error']=f'{type(e).__name__}: {e}'
            print('Camera error:',e,flush=True)
        finally:camera.close()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            url=urlsplit(self.path);headers={};code=200
            if url.path=='/':data=PAGE;kind='text/html'
            elif url.path=='/status':
                with lock:data=json.dumps({k:v for k,v in state.items() if k!='jpeg'}).encode()
                kind='application/json'
            elif url.path=='/frame':
                try:after=int(parse_qs(url.query).get('after',['0'])[0])
                except ValueError:after=0
                with lock:data=state['jpeg'];seq=state['seq']
                kind='image/jpeg';headers={'X-Frame-ID':str(seq)}
                if data is None or seq<=after:data=b'';code=204
            else:data=b'Not found';kind='text/plain';code=404
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            for k,v in headers.items():self.send_header(k,v)
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
    server=None;thread=None
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda *_:stop.set())
    try:
        server=ThreadingHTTPServer((host,cfg['port']),Handler);server.daemon_threads=True
        threading.Thread(target=server.serve_forever,daemon=True).start()
        thread=threading.Thread(target=capture,daemon=True);thread.start()
        print(f"Camera preview: http://ArdnQ.local:{cfg['port']}/ (or use the board Wi-Fi IP)",flush=True)
        print('Press Ctrl+C before starting the zone app.',flush=True)
        while not stop.wait(.5):
            if state['error']:break
    finally:
        stop.set()
        if server:server.shutdown();server.server_close()
        if thread:thread.join(timeout=2)
        else:camera.close()
    return 1 if state['error'] else 0
