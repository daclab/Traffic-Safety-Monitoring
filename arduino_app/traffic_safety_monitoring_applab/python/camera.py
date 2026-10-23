"""V4L2 USB camera selection and compressed-frame capture."""
import fcntl
import hashlib
import os
import struct
from pathlib import Path

import cv2
import numpy as np


def capabilities(path):
    buf=bytearray(104)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    try: fcntl.ioctl(fd,0x80685600,buf,True)  # VIDIOC_QUERYCAP
    finally: os.close(fd)
    driver=bytes(buf[:16]).split(b'\0')[0].decode(errors='replace')
    card=bytes(buf[16:48]).split(b'\0')[0].decode(errors='replace')
    bus=bytes(buf[48:80]).split(b'\0')[0].decode(errors='replace')
    caps,device_caps=struct.unpack_from('II',buf,84)
    actual=device_caps if caps & 0x80000000 else caps
    return {'driver':driver,'name':card,'bus':bus,'capture':bool(actual & 1)}


def select_device(request):
    if request != 'auto':
        path=str(Path(request).expanduser());info=capabilities(path)
        if not info['capture']:raise RuntimeError(f'{path} is not a video capture node.')
        return path,info
    paths=sorted(Path('/dev/v4l/by-id').glob('*video-index0'))+sorted(Path('/dev').glob('video*'))
    found={}
    for p in paths:
        real=str(p.resolve())
        if real in found:continue
        try:info=capabilities(str(p))
        except OSError:continue
        if info['capture'] and info['driver']=='uvcvideo':found[real]=(str(p),info)
    if not found:raise RuntimeError('No accessible USB video capture device. Stop other camera apps; check v4l2-ctl --list-devices and /dev/video* permissions.')
    if len(found)>1:raise RuntimeError('Multiple USB cameras found. Set device in config.json: '+', '.join(v[0] for v in found.values()))
    return next(iter(found.values()))


def jpeg_size(data):
    """Read JPEG SOF dimensions without decoding the image."""
    i=2
    while i+4<=len(data):
        if data[i]!=255:i+=1;continue
        while i<len(data) and data[i]==255:i+=1
        if i>=len(data):break
        marker=data[i];i+=1
        if marker in (0xD8,0xD9) or 0xD0<=marker<=0xD7:continue
        if marker==0xDA:break
        length=int.from_bytes(data[i:i+2],'big')
        if length<2 or i+length>len(data):break
        if marker in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
            return int.from_bytes(data[i+5:i+7],'big'),int.from_bytes(data[i+3:i+5],'big')
        i+=length
    raise RuntimeError('Cannot determine camera JPEG dimensions.')


class Camera:
    def __init__(self,cfg):
        self.path,info=select_device(cfg['device']);self.cap=None
        self.raw=False;self.first=None
        try:
            self.cap=self.open(cfg,cfg['prefer_raw_mjpeg'])
            ok,frame=self.cap.read()
            data=frame.tobytes() if ok and frame is not None and frame.ndim<=2 else b''
            if cfg['prefer_raw_mjpeg'] and data.startswith(b'\xff\xd8'):
                self.raw=True;self.first=data;w,h=jpeg_size(data)
            elif not cfg['prefer_raw_mjpeg'] and ok and frame is not None and frame.ndim==3:
                h,w=frame.shape[:2];self.first=frame
            else:
                self.cap.release();self.cap=self.open(cfg,False)
                ok,frame=self.cap.read()
                if not ok or frame.ndim!=3:raise RuntimeError('Camera did not provide decoded BGR frames. Try a lower capture resolution in config.json.')
                h,w=frame.shape[:2];self.first=frame
            fourcc=int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.info={**info,'device':self.path,'width':w,'height':h,
                'requested_fps':cfg['capture_fps'],'reported_fps':self.cap.get(cv2.CAP_PROP_FPS),
                'format':''.join(chr((fourcc>>8*i)&255) for i in range(4)),
                'path_mode':'Raw MJPEG / reduced decode' if self.raw else 'Decoded BGR / OpenCV fallback'}
            identity=f"{info['driver']}|{info['name']}|{info['bus']}|{w}x{h}"
            self.info['signature']=hashlib.sha256(identity.encode()).hexdigest()[:20]
        except BaseException:
            if self.cap is not None:self.cap.release()
            raise

    def open(self,cfg,raw):
        cap=cv2.VideoCapture(self.path,cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError(f'Cannot open {self.path}. Stop the old zone app and other camera apps.')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg['capture_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg['capture_height'])
        cap.set(cv2.CAP_PROP_FPS,cfg['capture_fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if cfg.get('exposure_ms') is not None:
            controls=[(0x009a0903,0),(0x009a0901,1),
                      (0x009a0902,round(cfg['exposure_ms']*10))]
            if cfg.get('gain') is not None:controls.append((0x00980913,cfg['gain']))
            try:
                fd=os.open(self.path,os.O_RDWR|os.O_NONBLOCK)
                try:
                    for control,value in controls:
                        # VIDIOC_S_CTRL; v4l2_control is uint32 id, int32 value.
                        fcntl.ioctl(fd,0xC008561C,bytearray(struct.pack('Ii',control,value)),True)
                finally:os.close(fd)
            except OSError as e:
                cap.release()
                raise RuntimeError('Could not set ELP manual exposure: '+str(e)) from e
        if raw:
            cap.set(cv2.CAP_PROP_CONVERT_RGB,0)
            cap.set(cv2.CAP_PROP_FORMAT,-1)
        return cap

    def read(self):
        if self.first is not None:
            x,self.first=self.first,None;return x
        ok,x=self.cap.read()
        if not ok:raise RuntimeError('USB camera stopped delivering frames. Check the camera, hub, and power; then restart the app.')
        if self.raw:
            x=x.tobytes()
            if not x.startswith(b'\xff\xd8'):raise RuntimeError('Camera stopped sending MJPEG. Set prefer_raw_mjpeg to false.')
        return x

    def close(self):
        if self.cap is not None:self.cap.release()
