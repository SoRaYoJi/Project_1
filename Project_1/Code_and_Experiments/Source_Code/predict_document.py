# predict_document.py

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.thai_mobilenet_ocr import load_thai_ocr_model  # noqa: E402
from model.charset import INDEX2CHAR  # noqa: E402
from model.utils import get_eval_transforms  # noqa: E402


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "thai_ocr_mobilenet.pt")


def sort_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    เรียงกล่องตามลำดับบน->ล่าง และซ้าย->ขวา
    boxes: [(x, y, w, h), ...]
    """
    # sort by y แล้วค่อย sort x ภายในแถวเดียวกัน
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] // 40, b[0]))  # 40 = ความสูงประมาณ 1 บรรทัด ปรับได้
    return boxes_sorted


def segment_characters(image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    ใช้ OpenCV หา bounding boxes ของตัวอักษรโดยประมาณ
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    # Threshold ให้เป็นขาวดำ
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ขยายตัวอักษรเล็กน้อยให้ติดกัน
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # หา contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    h_img, w_img = img.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # กรองกล่องเล็กๆ ที่เป็น noise ทิ้ง
        if w * h < 20:
            continue
        if h < 10 or w < 5:
            continue
        boxes.append((x, y, w, h))

    boxes = sort_boxes(boxes)
    return img, boxes


def recognize_document(image_path: str) -> str:
    model, device = load_thai_ocr_model(MODEL_PATH)
    transform = get_eval_transforms()

    img, boxes = segment_characters(image_path)
    print(f"found {len(boxes)} candidate boxes")

    chars: List[str] = []

    for (x, y, w, h) in boxes:
        char_img = img[y:y+h, x:x+w]

        # เพิ่ม margin นิดหน่อย
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        char_img = img[y1:y2, x1:x2]

        # แปลงเป็น PIL-like ด้วย resize จาก transform
        from PIL import Image
        pil_img = Image.fromarray(char_img)

        tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,1,128,128)
        with torch.no_grad():
            out = model(tensor)
            _, pred = torch.max(out, 1)
        idx = int(pred.item())
        ch = INDEX2CHAR.get(idx, "?")
        chars.append(ch)

    text = "".join(chars)
    return text


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path ของภาพเอกสารที่ต้องการอ่าน")
    args = parser.parse_args()

    text = recognize_document(args.image)
    print("\n=== OCR RESULT ===")
    print(text)


if __name__ == "__main__":
    main()
