# ocr_combined.py
# อ่านเอกสารทั้งหน้า:
# - ตัวอักษรไทย: ใช้ Tesseract (lang='tha')
# - ตัวเลขไทย: ใช้โมเดล digit_cnn ของเรา (ทับผล Tesseract เฉพาะกล่องที่เป็นเลขไทยเดี่ยว)
# แสดง 2 หน้าต่าง:
#   1) เอกสารจริง + กรอบ + text เล็กๆ บนกรอบ
#   2) เอกสารพื้นขาว เรียงเป็นบรรทัด อ่านง่าย

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import pytesseract
from pytesseract import Output

# -------------------------------
# ตั้งค่า path โปรเจกต์
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.utils import load_model, get_eval_transforms  # type: ignore

try:
    from model.utils import THAI_DIGITS  # type: ignore
except Exception:
    THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "digit_cnn.pt")

# ฟอนต์ไทยสำหรับวาดตัวอักษรบนภาพ
FONT_PATH = os.path.join(PROJECT_ROOT, "Sarabun-ThinItalic.ttf")

# -------------------------------
# ตั้งค่า path ไปที่ Tesseract.exe (ปรับให้ตรงเครื่องคุณ)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------
# โหลดโมเดลตัวเลขไทยของเรา
# -------------------------------
def load_digit_model():
    model, device = load_model(MODEL_PATH)
    model.eval()
    transform = get_eval_transforms()
    return model, device, transform


# -------------------------------
# ใช้โมเดลเราอ่าน "เลขไทย 1 ตัว" จากภาพเล็ก ๆ
# -------------------------------
def predict_thai_digit(model, device, transform, crop_img: np.ndarray) -> str:
    """
    crop_img: ภาพ gray หรือ BGR ของตัวเลข 1 ตัว (numpy)
    คืนค่า: เลขไทย 1 ตัว (string) หรือ "?" ถ้าไม่มั่นใจ
    """
    if len(crop_img.shape) == 3:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    pil_img = Image.fromarray(crop_img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    idx = int(pred.item())
    conf_val = float(conf.item())
    digit_char = THAI_DIGITS[idx]

    if conf_val < 0.6:
        return "?"
    return digit_char


# -------------------------------
# วาดผล OCR ลงบนภาพ 2 แบบ
# -------------------------------
def show_visualizations(
    img_bgr: np.ndarray,
    boxes_and_texts: List[Tuple[int, int, int, int, str]],
    lines_text: List[str],
) -> None:
    """
    boxes_and_texts: list ของ (x, y, w, h, text) หลังรวมผล Tesseract + โมเดลเลขไทยแล้ว
    lines_text: ข้อความแต่ละบรรทัด (string) ที่จัดเรียงแล้ว
    """
    h_img, w_img = img_bgr.shape[:2]

    # --- เตรียมฟอนต์ไทย ---
    if os.path.isfile(FONT_PATH):
        try:
            font_box = ImageFont.truetype(FONT_PATH, 12)   # สำหรับเขียนบนกรอบ (เล็ก)
            font_line = ImageFont.truetype(FONT_PATH, 20)  # สำหรับหน้าต่าง reconstruct
        except OSError:
            font_box = font_line = ImageFont.load_default()
    else:
        font_box = font_line = ImageFont.load_default()

    # --- หน้าต่างที่ 1: เอกสารจริง + กรอบ + ข้อความเล็ก ๆ ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_1 = Image.fromarray(img_rgb)
    draw_1 = ImageDraw.Draw(pil_1)

    for (x, y, w, h, text) in boxes_and_texts:
        if not text.strip():
            continue
        # กรอบสี่เหลี่ยม (สีเขียว)
        draw_1.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=1)
        # ข้อความวาดด้านบนกรอบ (สีแดง ขนาดเล็ก)
        text_pos = (x, max(0, y - 14))
        draw_1.text(text_pos, text, fill=(255, 0, 0), font=font_box)

    vis1_bgr = cv2.cvtColor(np.array(pil_1), cv2.COLOR_RGB2BGR)

    # --- หน้าต่างที่ 2: พื้นขาว + ข้อความเรียงเป็นบรรทัด ---
    margin_left = 60
    margin_top = 60
    line_spacing = 30  # ระยะห่างแต่ละบรรทัด

    pil_2 = Image.new("RGB", (w_img, h_img), color=(255, 255, 255))
    draw_2 = ImageDraw.Draw(pil_2)

    y_cursor = margin_top
    for line in lines_text:
        if not line.strip():
            y_cursor += line_spacing // 2
            continue
        draw_2.text((margin_left, y_cursor), line, fill=(0, 0, 0), font=font_line)
        y_cursor += line_spacing

    vis2_bgr = cv2.cvtColor(np.array(pil_2), cv2.COLOR_RGB2BGR)

    # --- แสดงสองหน้าต่าง ---
    cv2.imshow("Scanned document + OCR boxes", vis1_bgr)
    cv2.imshow("Reconstructed OCR document", vis2_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
# OCR ทั้งหน้าเอกสาร + ผสมผลจากโมเดลเรา
# -------------------------------
def ocr_document_with_digit_model(image_path: str) -> str:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        img_rgb,
        lang="tha",
        output_type=Output.DICT
    )

    digit_model, device, transform = load_digit_model()

    n_boxes = len(data["text"])
    current_line = 0
    line_tokens: List[str] = []
    lines: List[str] = []

    boxes_and_texts: List[Tuple[int, int, int, int, str]] = []

    for i in range(n_boxes):
        raw_text = data["text"][i]
        if not raw_text.strip():
            continue

        line_num = data["line_num"][i]
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        # เปลี่ยนบรรทัด
        if line_num != current_line:
            if line_tokens:
                # สำหรับภาษาไทยใช้ join แบบติดกันจะอ่านง่ายกว่า
                lines.append("".join(line_tokens))
                line_tokens = []
            current_line = line_num

        stripped = raw_text.strip()
        is_single_thai_digit = len(stripped) == 1 and stripped in THAI_DIGITS
        final_text = stripped

        if is_single_thai_digit:
            crop = img_bgr[y: y + h, x: x + w]
            digit_char = predict_thai_digit(digit_model, device, transform, crop)
            if digit_char != "?":
                final_text = digit_char

        line_tokens.append(final_text)
        boxes_and_texts.append((x, y, w, h, final_text))

    # บรรทัดสุดท้าย
    if line_tokens:
        lines.append("".join(line_tokens))

    full_text = "\n".join(lines)

    # แสดง visualization แบบใหม่
    show_visualizations(img_bgr, boxes_and_texts, lines)

    return full_text

def save_text_file(image_path: str, text: str) -> str:
    """
    บันทึกผล OCR เป็นไฟล์ .txt
    ชื่อไฟล์: <ชื่อภาพ>_ocr.txt
    """
    base = os.path.splitext(image_path)[0]
    out_path = f"{base}_ocr.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nบันทึกผล OCR ไว้ที่: {out_path}")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OCR เอกสารไทย + ใช้โมเดลเลขไทย พร้อม visualization 2 หน้าต่าง"
    )
    parser.add_argument("image", help="path ภาพเอกสาร เช่น doc.png")
    args = parser.parse_args()

    text = ocr_document_with_digit_model(args.image)

    # บันทึกไฟล์ .txt โดยอัตโนมัติ
    save_text_file(args.image, text)

    print("\n===== OCR RESULT (Tesseract + digit model) =====")
    print(text)



if __name__ == "__main__":
    main()
