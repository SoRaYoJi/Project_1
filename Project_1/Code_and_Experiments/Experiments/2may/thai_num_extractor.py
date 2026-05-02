import cv2
import json
import numpy as np
from PIL import Image
from google import genai
import re
import os
from dotenv import load_dotenv

# ==========================================
# ส่วนที่ 1: ตั้งค่าเริ่มต้นและกำหนดโฟลเดอร์ทำงาน
# ==========================================
# โหลด Environment Variables (เช่น API Key) จากไฟล์ .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# กำหนดเส้นทาง (Paths) ที่ใช้ในโปรเจกต์
BASE_DIR = "/home/ekkamol/thai-ocr-project/250369/"
IMAGE_PATH = os.path.join(BASE_DIR, "t1.png")          # ไฟล์รูปภาพต้นฉบับ
OUTPUT_TEXT = os.path.join(BASE_DIR, "final_output.txt") # ไฟล์ Text ผลลัพธ์สุดท้าย
SAVE_NUM_DIR = os.path.join(BASE_DIR, "numtest")         # โฟลเดอร์สำหรับเก็บภาพตัวเลขไทยที่ตัดแล้ว

# ตรวจสอบว่ามีโฟลเดอร์สำหรับเก็บรูปหรือยัง ถ้ายังไม่มีให้สร้างขึ้นมาใหม่
if not os.path.exists(SAVE_NUM_DIR):
    os.makedirs(SAVE_NUM_DIR)

# ==========================================
# ส่วนที่ 2: ฟังก์ชันช่วยเหลือ (Helper Functions)
# ==========================================

def clean_json_string(text):
    """
    ฟังก์ชันทำความสะอาดข้อความที่ได้จาก Gemini 
    เนื่องจากบางครั้ง AI จะส่งแท็ก Markdown (เช่น ```json ... ```) มาด้วย 
    ฟังก์ชันนี้จะใช้ Regex ดึงมาเฉพาะโครงสร้าง Array ของ JSON เท่านั้น
    """
    match = re.search(r'\[.*\]', text, re.DOTALL)
    return match.group(0) if match else text

def extract_digits_with_opencv(image_cv, ymin, xmin, ymax, xmax, expand_pad=8):
    """
    ฟังก์ชันสำหรับตัดภาพตัวเลขย่อยๆ ด้วย OpenCV 
    รับพิกัดเริ่มต้นจาก Gemini แล้ววิ่งหา "ขอบหมึกสีดำ" เพื่อตัดภาพให้พอดีเป๊ะ
    """
    h, w = image_cv.shape[:2]
    
    # 1. ขยายพื้นที่จากพิกัดของ Gemini (Padding) เผื่อ AI ตีกรอบมาขาดหรือชิดเกินไป
    y1, y2 = max(0, ymin - expand_pad), min(h, ymax + expand_pad)
    x1, x2 = max(0, xmin - expand_pad), min(w, xmax + expand_pad)
    crop_area = image_cv[y1:y2, x1:x2]
    
    # ป้องกันกรณีเกิด Error ภาพว่างเปล่า
    if crop_area.size == 0: return []

    # 2. แปลงภาพเป็นโหมดขาวดำ (Grayscale)
    gray = cv2.cvtColor(crop_area, cv2.COLOR_BGR2GRAY)
    
    # 3. แปลงภาพให้เป็น ขาว-ดำ สนิท (Binarization) ด้วยวิธี Otsu's thresholding
    # หมึกดำจะกลายเป็นสีขาว ส่วนกระดาษขาวจะกลายเป็นสีดำ เพื่อง่ายต่อการหาขอบเขต
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. ค้นหาเส้นขอบรอบนอก (Contours) ของตัวอักษร
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        # กรอง Noise ทิ้ง (จุดไข่ปลา หรือฝุ่น) โดยกำหนดขนาดขั้นต่ำของความกว้างและความสูง
        if ch > 10 and cw > 3:
            valid_contours.append((cx, cy, cw, ch))
            
    # 5. เรียงลำดับตัวอักษรที่พบจาก ซ้าย-ไป-ขวา (อิงตามแกน X)
    valid_contours.sort(key=lambda b: b[0])
    
    exact_crops = []
    # 6. วนลูปตัดภาพทีละตัวอักษร 
    for (cx, cy, cw, ch) in valid_contours:
        p = 3 # เพิ่มขอบขาวรอบตัวอักษร (Padding) 3 พิกเซล ให้ภาพออกมาดูสมบูรณ์
        cy1, cy2 = max(0, cy - p), min(crop_area.shape[0], cy + ch + p)
        cx1, cx2 = max(0, cx - p), min(crop_area.shape[1], cx + cw + p)
        exact_crops.append(crop_area[cy1:cy2, cx1:cx2])
        
    return exact_crops

# ==========================================
# ส่วนที่ 3: ระบบทำงานหลัก (Main Pipeline)
# ==========================================
def run_ocr_pipeline():
    # ตรวจสอบการเข้าถึง API Key ก่อนเริ่มทำงาน
    if not GEMINI_KEY:
        return print("Error: ไม่พบ API Key โปรดตรวจสอบไฟล์ .env")

    print("เริ่มสแกนหน้ากระดาษด้วย Gemini API...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    # โหลดภาพผ่าน OpenCV สำหรับใช้ตัดภาพ
    img_cv = cv2.imread(IMAGE_PATH)
    if img_cv is None: 
        return print("Error: โหลดภาพไม่สำเร็จ โปรดตรวจสอบพาธรูปภาพ")
    h, w, _ = img_cv.shape
    
    # โหลดภาพผ่าน PIL สำหรับส่งให้ Gemini วิเคราะห์
    raw_image = Image.open(IMAGE_PATH)
    
    # ---------------------------------------------------------
    # Prompt (คำสั่ง AI) - จุดสำคัญในการควบคุมผลลัพธ์
    # 1. EXHAUSTIVE EXTRACTION: บังคับให้อ่านทุกคำ ห้ามข้ามเด็ดขาด (แก้ปัญหาอ่านไม่ครบ)
    # 2. SEPARATE bounding boxes: บังคับให้แยกกล่อง "ตัวเลข" ออกจาก "ข้อความ"
    # ---------------------------------------------------------
    prompt = f"""
    You are a highly accurate Thai OCR system.
    
    YOUR MISSION:
    1. EXHAUSTIVE EXTRACTION: Read EVERY SINGLE word and line from the image. Do not skip any text, paragraph, or sentence. Scan thoroughly from top to bottom.
    2. CRITICAL RULE: Put text and Thai numerals (๐-๙) in SEPARATE bounding boxes!
       For example, for "วันที่ ๒๕", you MUST return two separate JSON objects: one for "วันที่" and one for "๒๕".
    
    IMAGE INFO: Width={w}px, Height={h}px.

    OUTPUT FORMAT: Return ONLY a complete JSON array. Do not truncate the response.
    [
      {{"box_2d": [ymin, xmin, ymax, xmax], "text": "พ.ศ."}},
      {{"box_2d": [ymin, xmin, ymax, xmax], "text": "๒๕๖๗"}}
    ]
    """

    # ส่งข้อมูลให้ AI ประมวลผล
    # หมายเหตุ: temperature=0.1 เพื่อให้ AI โฟกัสกับการดึงข้อความตามจริง ลดการสุ่มหรือการคิดไปเอง
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[prompt, raw_image],
            config={"temperature": 0.1}
        )
    except:
        # ระบบ Fallback หากโมเดลตัวแรกมีปัญหา
        response = client.models.generate_content(
            model="gemini-flash", 
            contents=[prompt, raw_image],
            config={"temperature": 0.1}
        )

    # ==========================================
    # ส่วนที่ 4: การประมวลผลข้อความและการจัดหน้า (Processing & Formatting)
    # ==========================================
    try:
        # แปลงข้อความ Text จาก AI ให้กลายเป็น List Dictionary ของ Python
        detections = json.loads(clean_json_string(response.text))
        print(f"AI อ่านข้อมูลได้ทั้งหมด {len(detections)} จุด กำลังจัดบรรทัด...")

        # จัดเรียงลำดับการอ่าน: บนลงล่าง (แกน Y) และ ซ้ายไปขวา (แกน X)
        # Y_BIN_SIZE คือค่าความคลาดเคลื่อนของบรรทัด (ประมาณความสูงของตัวอักษร 1 บรรทัด)
        Y_BIN_SIZE = 25
        detections.sort(key=lambda d: (d.get('box_2d', [0])[0] // Y_BIN_SIZE, d.get('box_2d', [0,0])[1]))

        formatted_text = ""
        last_y_bin = -1
        global_num_count = 0

        # วนลูปอ่านข้อมูลทีละกล่อง
        for det in detections:
            box = det.get('box_2d')
            text = det.get('text', '').strip()
            
            # ข้ามกล่องที่พิกัดไม่ครบ 4 จุด
            if not box or len(box) != 4: continue
            
            ymin, xmin, ymax, xmax = map(int, box)
            current_y_bin = ymin // Y_BIN_SIZE
            
            # --- ตรวจสอบเงื่อนไข: หากในกล่องมี "เลขไทย" ---
            if re.search(r'[๐-๙]', text):
                # 1. เจาะตัดภาพเฉพาะตัวเลข ด้วย OpenCV (แยกทีละตัว)
                precise_crops = extract_digits_with_opencv(img_cv, ymin, xmin, ymax, xmax)
                for digit_img in precise_crops:
                    global_num_count += 1
                    # บันทึกภาพลงโฟลเดอร์เป้าหมาย
                    cv2.imwrite(os.path.join(SAVE_NUM_DIR, f"num_{global_num_count}.png"), digit_img)
                
                # 2. แปลงข้อมูลที่จะเซฟลง Text: แทนที่ตัวเลขไทยทุกตัวด้วยคำว่า [numthai]
                display_text = re.sub(r'[๐-๙]', '[numthai]', text)
            else:
                # หากเป็นข้อความปกติ ให้ใช้ข้อความเดิม
                display_text = text

            # --- ระบบจัดบรรทัดอัตโนมัติ (Auto-Formatting) ---
            if last_y_bin == -1:
                # คำแรกของเอกสาร
                formatted_text += display_text
            elif current_y_bin - last_y_bin >= 2:   
                # แกน Y ห่างกันเกิน 1 บรรทัด -> มองเป็นการขึ้นย่อหน้าใหม่ (Enter 2 ครั้ง)
                formatted_text += f"\n\n{display_text}"
            elif current_y_bin - last_y_bin == 1:   
                # แกน Y ห่างกัน 1 บรรทัด -> ขึ้นบรรทัดถัดไป (Enter 1 ครั้ง)
                formatted_text += f"\n{display_text}"
            else:                                   
                # แกน Y อยู่ในระดับเดียวกัน -> เคาะเว้นวรรค
                formatted_text += f" {display_text}"
            
            # อัปเดตค่าแกน Y ล่าสุด เพื่อใช้เปรียบเทียบกับคำถัดไป
            last_y_bin = max(last_y_bin, current_y_bin)

        # เขียนผลลัพธ์ข้อความที่จัดบรรทัดเรียบร้อยแล้ว ลงไฟล์ Text
        with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
            f.write(formatted_text)

        print("-" * 50)
        print("ทำงานเสร็จสมบูรณ์!")
        print(f"ตัดแยกเลขไทยได้ทั้งหมด: {global_num_count} รูป (อยู่ใน {SAVE_NUM_DIR})")
        print(f"อ่านข้อความและแปลง [numthai] ไว้ที่: {OUTPUT_TEXT}")
        print("-" * 50)

    except Exception as e:
        print(f"Error ประมวลผล: {e}")

if __name__ == "__main__":
    run_ocr_pipeline()