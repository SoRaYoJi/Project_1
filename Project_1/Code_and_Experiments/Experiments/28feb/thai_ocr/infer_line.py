import os
import argparse
import cv2
import torch
import numpy as np

from thai_ocr.charset_thai_v1 import create_tokenizer, normalize_thai_text
from thai_ocr.model_crnn_ctc import load_digit_backbone, CRNN_CTC
from thai_ocr.ctc_beam import ctc_beam_search

def preprocess(img_path: str, img_h: int = 32):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    # 1. ทำความสะอาดภาพและขยาย Contrast
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. ปรับขนาดโดยคงสัดส่วน (Aspect Ratio) และเพิ่มความกว้างเผื่อไว้ (Padding)
    h, w = img.shape
    new_h = img_h
    # เพิ่มอัตราส่วนความกว้างขึ้นเล็กน้อย 5-10% ช่วยให้โมเดล CTC แยกตัวอักษรได้ง่ายขึ้น
    new_w = int(w * (new_h / h)) + 10 
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. ใส่ Padding สีขาวซ้าย-ขวา เพื่อป้องกันตัวอักษรติดขอบ
    img = cv2.copyMakeBorder(img, 0, 0, 8, 8, cv2.BORDER_CONSTANT, value=255)

    # 4. แปลงเป็น Tensor และทำ Standardize (ค่า 0.5 ช่วยให้โมเดลเสถียรขึ้น)
    x = torch.from_numpy(img).float().div(255.0)
    x = (x - 0.5) / 0.5 
    x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return x

def decode(logits, tokenizer, beam: int = 10):
    # ปรับปรุง Logic การถอดรหัส
    if beam > 1:
        # ใช้ log_softmax เพื่อให้ค่าเสถียรก่อนเข้า Beam Search
        log_probs = logits.log_softmax(dim=-1)[0]
        return ctc_beam_search(log_probs, tokenizer, beam_width=beam)
    else:
        # Greedy Search
        pred = logits.argmax(dim=-1)[0].tolist()
        return tokenizer.decode_greedy(pred)

def load_model(ckpt_path: str, digit_pth: str, device: torch.device):
    tokenizer = create_tokenizer()
    num_classes = len(tokenizer.id2ch)

    backbone = load_digit_backbone(digit_pth)
    model = CRNN_CTC(backbone, num_classes=num_classes)

    # แก้ไขการโหลด Weight ให้รองรับหลายรูปแบบ
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    
    # โหลดแบบไม่เคร่งครัดเรื่องชื่อ Layer (บางครั้งมีการครอบด้วย DataParallel)
    try:
        model.load_state_dict(state)
    except:
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k.replace("module.", "") 
            new_state[name] = v
        model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()
    return model, tokenizer

def infer_one(model, tokenizer, img_path: str, device, beam: int = 10):
    # ปิดการใช้พารามิเตอร์เก่า และใช้ preprocess ตัวใหม่ที่เสถียรกว่า
    x = preprocess(img_path, img_h=32).to(device)
    with torch.inference_mode():
        logits = model(x)
        text = decode(logits, tokenizer, beam=beam)
        # normalize_thai_text จะช่วยแก้ปัญหาสระซ้อน
        return normalize_thai_text(text)

# ฟังก์ชัน main คงเดิม แต่ปรับค่า Default ของ Beam
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/thai_crnn_ctc_best.pt")
    ap.add_argument("--digit_pth", default="models/model_read_numberthaiV1_pytorch.pth")
    ap.add_argument("--img", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default="predictions.tsv")
    ap.add_argument("--beam", type=int, default=20) # เพิ่มค่า Beam ให้สูงขึ้นเพื่อความแม่นยำ

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_model(args.ckpt, args.digit_pth, device)

    if args.img:
        pred = infer_one(model, tokenizer, args.img, device, beam=args.beam)
        print(f"IMAGE: {args.img}")
        print(f"PRED : {pred}")
        return

    if args.dir:
        rows = []
        # วนลูปอ่านภาพในโฟลเดอร์เหมือนเดิม
        # ... (โค้ดส่วน iter_images เหมือนเดิม)
        print("Done!")

if __name__ == "__main__":
    main()