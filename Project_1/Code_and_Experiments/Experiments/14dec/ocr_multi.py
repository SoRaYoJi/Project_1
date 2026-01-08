import cv2
import numpy as np
from preprocess import preprocess_for_model
from inference import predict_single

def sort_boxes(boxes, y_thresh=30):
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []

    for box in boxes:
        placed = False
        for line in lines:
            if abs(box[1] - line[0][1]) < y_thresh:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b[0])

    return lines


def multi_digit_ocr(model, img_gray):
    _, bin_img = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]

    lines = sort_boxes(boxes)

    results = []
    confidences = []

    for line in lines:
        line_text = ""
        for (x, y, w, h) in line:
            roi = bin_img[y:y+h, x:x+w]
            img96, tensor = preprocess_for_model(roi)
            digit, conf = predict_single(model, tensor)
            line_text += str(digit)
            confidences.append(conf)
        results.append(line_text)

    return "\n".join(results), float(np.mean(confidences))
