import cv2
import numpy as np
import torch


def preprocess_digit(roi, size=96):
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
     return None, None
    scale = min((size-20)/w, (size-20)/h)
    nw, nh = int(w*scale), int(h*scale)
    roi = cv2.resize(roi, (nw, nh), cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    x, y = (size-nw)//2, (size-nh)//2
    canvas[y:y+nh, x:x+nw] = roi
    t = torch.from_numpy(canvas/255.).float().unsqueeze(0).unsqueeze(0)
    return canvas, t