from PIL import Image
import numpy as np
import cv2
from ocr_multi import multi_digit_ocr

def batch_ocr(model, files):
    outputs = []

    for f in files:
        img = Image.open(f).convert("L")
        img_np = np.array(img)

        text, conf = multi_digit_ocr(model, img_np)

        outputs.append({
            "filename": f.name,
            "result": text,
            "confidence": round(conf * 100, 2)
        })

    return outputs
