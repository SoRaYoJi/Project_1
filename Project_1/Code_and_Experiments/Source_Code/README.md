# Thai OCR Project

โปรเจกต์นี้เอาไว้ **อ่านตัวเลขไทย + อ่านเอกสารภาษาไทย** แบ่งเป็นสองส่วนหลัก ๆ

1. **โมเดล CNN อ่านตัวเลขไทยเดี่ยว ๆ**  
   - เทรนจากรูปตัวเลขไทย (๐–๙) ขนาด 128×128  
   - มีระบบวิเคราะห์ข้อผิดพลาด, interactive feedback, auto self-train

2. **ระบบอ่านเอกสารภาษาไทยทั้งหน้า**  
   - ใช้ Tesseract OCR อ่าน “ตัวหนังสือไทย”  
   - ใช้โมเดลเลขไทยของเรา ช่วยอ่าน/ยืนยันตัวเลขในเอกสาร  
   - แสดงผลเป็น 2 หน้าต่าง:  
     - รูปเอกสารเดิม + กรอบ OCR  
     - เอกสารที่ reconstruct จากข้อความ OCR

---

## 0. การเตรียมสภาพแวดล้อม (Environment)

แนะนำ Python 3.10 (ที่โปรเจกต์ใช้อยู่)

ติดตั้งไลบรารีหลัก:

```bash
pip install torch torchvision
pip install matplotlib numpy pillow
pip install opencv-python pytesseract


### 1) ติดตั้งไลบรารี


```bash
pip install -r requirements.txt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


thai-ocr/
├─ models/
│   ├─ digit_cnn.pt               # โมเดลตัวเลขไทย (ฝึกแล้ว)
│   └─ ...                        # โมเดลอื่น ๆ (ถ้ามี)
├─ model/
│   ├─ cnn_digit_model.py         # สถาปัตยกรรม CNN สำหรับเลขไทย ๐–๙
│   ├─ utils.py                   # ฟังก์ชันช่วย: โหลดโมเดล, dataloader, THAI_DIGITS, transforms ฯลฯ
│   └─ charset_letters.py         # (ตัวเลือก) charset ตัวอักษรไทย ถ้าอยากเทรนโมเดลตัวอักษรเอง
├─ training/
│   ├─ train_digits.py            # สคริปต์เทรนโมเดลเลขไทย
│   └─ train_thai_chars.py        # (ตัวเลือก) เทรนโมเดลตัวอักษรไทย ถ้าสร้าง dataset เอง
├─ analysis/
│   └─ analyze_errors.py          # วิเคราะห์ข้อผิดพลาด + Confusion matrix + CSV
├─ interactive/
│   ├─ interactive_feedback.py    # GUI ให้คนช่วยกดบอกว่าทายผิด/ถูก + ปรับโมเดลจาก feedback
│   └─ auto_self_train_v2.py      # Auto self-train แบบผสมรูปถูก/ผิด กันโมเดลพัง
├─ predict_image.py               # เลือกรูปทีละไฟล์ แล้วให้โมเดลเลขไทยอ่าน (GUI ง่าย ๆ)
├─ ocr_combined.py                # อ่านเอกสารทั้งหน้า (Tesseract + โมเดลเลขไทย) + 2 หน้าต่างแสดงผล
├─ Sarabun-ThinItalic.ttf         # ฟอนต์ไทยที่ใช้วาดตัวหนังสือในหน้าต่างแสดงผล
└─ README.md


python ocr_combined.py test_docs\bill1.png
