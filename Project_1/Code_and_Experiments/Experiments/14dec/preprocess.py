import cv2
import numpy as np
from PIL import Image

def preprocess_image(img, invert=False, threshold=True):
    img = img.convert("L")
    img = np.array(img)

    if invert:
        img = cv2.bitwise_not(img)

    if threshold:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = img.astype("float32")

    return img
