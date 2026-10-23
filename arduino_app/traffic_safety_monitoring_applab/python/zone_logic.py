"""Geometry, short-lived IoU identities, and occupancy hysteresis."""
import math

import cv2
import numpy as np

# Multiword labels must retain their COCO positions.
LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
 'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
 'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
 'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
 'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
 'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
 'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
 'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush']


def validate_zone(points):
    if not isinstance(points, list) or not 3 <= len(points) <= 12:
        raise ValueError('Choose 3–12 corners around the zone, in order.')
    if any(not isinstance(p, list) or len(p) != 2 for p in points):
        raise ValueError('Each corner must contain x and y.')
    pts = [[float(x), float(y)] for x, y in points]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for p in pts for v in p):
        raise ValueError('Corners must lie inside the image.')
    if len({tuple(p) for p in pts}) != len(pts):
        raise ValueError('Duplicate corners: do not click the first corner again.')
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):
        return abs(cross(a,b,c)) < 1e-9 and all(min(a[k],b[k])-1e-9 <= c[k] <= max(a[k],b[k])+1e-9 for k in (0,1))
    def intersects(a,b,c,d):
        return (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)))
    n=len(pts)
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1): continue
            if intersects(pts[i],pts[(i+1)%n],pts[j],pts[(j+1)%n]):
                raise ValueError('Zone edges cross. Select corners around the perimeter.')
    if abs(cv2.contourArea(np.asarray(pts,np.float32))) < 0.001:
        raise ValueError('Zone is too small.')
    return pts


def selected_classes(value):
    if value == 'all': return set(range(80))
    if value == 'traffic': return {0,1,2,3,5,7}
    names = [s.strip().lower() for s in value.split(',')]
    if not names or any(s not in LABELS for s in names):
        raise ValueError('Classes must be person, traffic, all, or comma-separated COCO labels.')
    return {LABELS.index(s) for s in names}


class Zone:
    def __init__(self, points, width, height):
        self.poly=np.asarray(points,np.float32)
        self.width,self.height=width,height
        self.mask=None

    def contains(self, box, mode='foot', fraction=.1):
        x1,y1,x2,y2=box
        if mode=='foot':
            return cv2.pointPolygonTest(self.poly, ((x1+x2)/2,y2),False)>=0
        if self.mask is None:
            self.mask=np.zeros((self.height,self.width),np.uint8)
            cv2.fillPoly(self.mask,[np.rint(self.poly*[self.width,self.height]).astype(np.int32)],1)
        a,b,c,d=[int(v) for v in (x1*self.width,y1*self.height,math.ceil(x2*self.width),math.ceil(y2*self.height))]
        area=max(0,c-a)*max(0,d-b)
        return area>0 and int(self.mask[b:d,a:c].sum())>=area*fraction


class Debounce:
    def __init__(self, enter=2, clear=3):
        self.enter,self.clear=enter,clear
        self.status='CHECKING'; self.hits=0; self.empty=0

    def update(self, occupied):
        self.hits=self.hits+1 if occupied else 0
        self.empty=0 if occupied else self.empty+1
        if self.hits>=self.enter: self.status='DANGER'
        elif self.empty>=self.clear: self.status='CLEAR'
        return self.status


def box_iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/(area(a)+area(b)-inter+1e-9)


class Tracker:
    """Greedy IoU association for display IDs; zone logic uses current detections only."""
    def __init__(self, threshold=.3, max_age=.8):
        self.threshold=threshold;self.max_age=max_age;self.tracks={};self.next_id=1

    def update(self, detections, now):
        self.tracks={k:v for k,v in self.tracks.items() if now-v['time']<=self.max_age}
        pairs=[]
        for i,d in enumerate(detections):
            for k,t in self.tracks.items():
                if t['class_id']==d['class_id']:
                    score=box_iou(d['box'],t['box'])
                    if score>=self.threshold:pairs.append((score,i,k))
        used_d=set();used_t=set()
        for _,i,k in sorted(pairs,reverse=True):
            if i not in used_d and k not in used_t:
                detections[i]['track_id']=k;used_d.add(i);used_t.add(k)
        for i,d in enumerate(detections):
            if i not in used_d:d['track_id']=self.next_id;self.next_id+=1
            self.tracks[d['track_id']]={**d,'time':now}
        return detections
