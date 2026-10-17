# app/utils.py
import cv2, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def boxes_to_mask(rgb, boxes, dilate_px=8):
    h, w, _ = rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1,y1,x2,y2) in boxes:
        x1,y1,x2,y2 = int(max(0,x1)), int(max(0,y1)), int(min(w-1,x2)), int(min(h-1,y2))
        mask[y1:y2, x1:x2] = 255
    if dilate_px>0:
        kernel = np.ones((dilate_px,dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)
    return mask

def overlay_mask(rgb, mask):
    over = rgb.copy()
    red = np.zeros_like(rgb); red[:,:,0]=255
    over[mask>0] = (over[mask>0]*0.3 + red[mask>0]*0.7).astype(np.uint8)
    return over

def save_boxes_json(path, face_boxes, plate_boxes):
    meta = {
        "face_boxes":[tuple(map(int,b)) for b in face_boxes],
        "plate_boxes":[tuple(map(int,b)) for b in plate_boxes]
    }
    with open(path,"w") as f: json.dump(meta,f)

def random_plate_text():
    import random, string
    L, D = string.ascii_uppercase, "0123456789"
    return f"{random.choice(L)}{random.choice(L)} {random.choice(D)}{random.choice(D)}{random.choice(D)} {random.choice(L)}{random.choice(L)}"

def draw_fake_plate_text(img: Image.Image, plate_boxes):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for (x1,y1,x2,y2) in plate_boxes:
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        draw.rectangle([x1,y1,x2,y2], fill=(230,230,230))
        text = random_plate_text()
        try:
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = draw.textlength(text, font=font), font.size
        tx = x1 + max(0,(x2-x1 - tw)//2)
        ty = y1 + max(0,(y2-y1 - th)//2)
        draw.text((tx,ty), text, fill=(0,0,0), font=font)
    return img
