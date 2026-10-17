# app/detectors.py
import cv2, os
import numpy as np

# YOLO (optional)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# InsightFace SCRFD (face)
from insightface.app import FaceAnalysis

_face_app = None
def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l")
        try:
            _face_app.prepare(ctx_id=0)   # GPU
        except Exception:
            _face_app.prepare(ctx_id=-1)  # CPU
    return _face_app

def detect_faces(rgb, score_thresh=0.4):
    faces = get_face_app().get(rgb)
    boxes = []
    for f in faces:
        if f.det_score >= score_thresh:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            boxes.append((x1, y1, x2, y2))
    return boxes

def detect_plates_yolo(rgb, weights_path=None, conf=0.25):
    if weights_path is None or YOLO is None or not os.path.exists(weights_path):
        return []
    model = YOLO(weights_path)
    res = model.predict(source=rgb, imgsz=640, conf=conf, verbose=False)
    boxes = []
    for r in res:
        if r.boxes is None: continue
        for b in r.boxes.xyxy.cpu().numpy().tolist():
            x1,y1,x2,y2 = map(int, b[:4])
            boxes.append((x1,y1,x2,y2))
    return boxes

def detect_plates_haar(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_russian_plate_number.xml"
    cas = cv2.CascadeClassifier(cascade_path)
    if cas.empty():
        return []
    dets = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 18))
    return [(x,y,x+w,y+h) for (x,y,w,h) in dets]
