from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
CODE_AND_EXPERIMENTS_DIR = PROJECT_ROOT / "Code_and_Experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code_and_Experiments.app.ocr_thai_api import call_thai_ocr_api

for env_path in (PROJECT_ROOT / ".env", CODE_AND_EXPERIMENTS_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

GEMINI_KEY = os.getenv("THAI_OCR_API_KEY") or os.getenv("GEMINI_API_KEY")
IMAGE_PATH = Path(os.getenv("THAI_OCR_INPUT_IMAGE", CURRENT_DIR / "t1.png")).expanduser()
OUTPUT_TEXT = Path(os.getenv("THAI_OCR_OUTPUT_TEXT", CURRENT_DIR / "final_output.txt")).expanduser()
SAVE_NUM_DIR = Path(os.getenv("THAI_OCR_SAVE_NUM_DIR", CURRENT_DIR / "numtest")).expanduser()
SAVE_NUM_DIR.mkdir(parents=True, exist_ok=True)


def clean_json_string(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text


def imread_unicode(image_path: Path):
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if file_bytes.size == 0:
        return None
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def imwrite_unicode(image_path: Path, image) -> bool:
    suffix = image_path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        return False
    encoded.tofile(str(image_path))
    return True


def extract_digits_with_opencv(
    image_cv,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    expand_pad: int = 8,
):
    height, width = image_cv.shape[:2]
    y1, y2 = max(0, ymin - expand_pad), min(height, ymax + expand_pad)
    x1, x2 = max(0, xmin - expand_pad), min(width, xmax + expand_pad)
    crop_area = image_cv[y1:y2, x1:x2]

    if crop_area.size == 0:
        return []

    gray = cv2.cvtColor(crop_area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if ch > 10 and cw > 3:
            valid_contours.append((cx, cy, cw, ch))

    valid_contours.sort(key=lambda item: item[0])

    exact_crops = []
    for cx, cy, cw, ch in valid_contours:
        padding = 3
        cy1, cy2 = max(0, cy - padding), min(crop_area.shape[0], cy + ch + padding)
        cx1, cx2 = max(0, cx - padding), min(crop_area.shape[1], cx + cw + padding)
        exact_crops.append(crop_area[cy1:cy2, cx1:cx2])

    return exact_crops


def run_ocr_pipeline() -> None:
    if not GEMINI_KEY:
        print("Error: ไม่พบ API Key กรุณาตั้งค่า THAI_OCR_API_KEY หรือ GEMINI_API_KEY ในไฟล์ .env")
        return

    if not IMAGE_PATH.exists():
        print(f"Error: ไม่พบไฟล์ภาพอินพุตที่ {IMAGE_PATH}")
        return

    print("เริ่มสแกนเอกสารด้วย Gemini API...")
    img_cv = imread_unicode(IMAGE_PATH)
    if img_cv is None:
        print(f"Error: โหลดภาพไม่สำเร็จจาก {IMAGE_PATH}")
        return

    raw_image = Image.open(IMAGE_PATH).convert("RGB")

    try:
        api_result = call_thai_ocr_api(raw_image)
        print(f"API อ่านได้ {len(api_result.detections)} จุด โดยใช้ model {api_result.model_name}")
        global_num_count = 0

        for detection in api_result.detections:
            ymin = detection.box.ymin
            xmin = detection.box.xmin
            ymax = detection.box.ymax
            xmax = detection.box.xmax
            text = detection.raw_text
            if re.search(r"[๐-๙]", text):
                precise_crops = extract_digits_with_opencv(
                    img_cv,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                )
                for digit_img in precise_crops:
                    global_num_count += 1
                    imwrite_unicode(SAVE_NUM_DIR / f"num_{global_num_count}.png", digit_img)

        OUTPUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_TEXT.write_text(api_result.text, encoding="utf-8")

        print("-" * 50)
        print("ทำงานเสร็จสมบูรณ์")
        print(f"บันทึกรูปเลขไทยที่ตัดได้ {global_num_count} รูป ไว้ที่ {SAVE_NUM_DIR}")
        print(f"บันทึกข้อความไว้ที่ {OUTPUT_TEXT}")
        print("-" * 50)
    except Exception as exc:
        print(f"Error ประมวลผลผลลัพธ์ API: {exc}")


if __name__ == "__main__":
    run_ocr_pipeline()
