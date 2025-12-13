import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- ตั้งค่าชื่อไฟล์ (ตรวจสอบให้ตรงกับไฟล์ที่มี) ---
MODEL_PATH = 'thai_digit_model_64x64_Thickness_V2.keras'
IMG_HEIGHT = 64
IMG_WIDTH = 64

# ลาเบลผลลัพธ์ (เลข 0-9 ไทย)
CLASS_LABELS = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

def main():
    # 1. ตรวจสอบไฟล์โมเดล
    if not os.path.exists(MODEL_PATH):
        print(f"Error: ไม่พบไฟล์โมเดล '{MODEL_PATH}'")
        print("คำแนะนำ: กรุณาลากไฟล์ .keras มาวางในโฟลเดอร์เดียวกับไฟล์ predict.py")
        return

    # 2. โหลดโมเดล
    try:
        print("กำลังโหลดโมเดล... (อาจใช้เวลาสักครู่)")
        model = load_model(MODEL_PATH)
        print("โหลดโมเดลสำเร็จ!")
    except Exception as e:
        print(f"Error โหลดโมเดลไม่ได้: {e}")
        return

    # 3. ระบุชื่อรูปที่จะทาย (เปลี่ยนชื่อไฟล์ตรงนี้ได้เลย)
    image_filename = 'images\Screenshot 2025-11-13 200524.png' 
    
    if not os.path.exists(image_filename):
        print(f"\n[เตือน] ไม่พบไฟล์รูปภาพชื่อ '{image_filename}'")
        print("วิธีแก้: นำรูปตัวเลขที่ต้องการทายมาวางในโฟลเดอร์นี้ แล้วเปลี่ยนชื่อตัวแปร image_filename ในโค้ดให้ตรงกัน")
        return

    # 4. เตรียมรูปภาพ (Preprocessing)
    try:
        img = cv2.imread(image_filename)
        
        # แปลงเป็นขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold (ปรับภาพให้เป็น ขาว-ดำ ชัดเจน)
        # ถ้าผลออกมาผิดบ่อยๆ ให้ลองสลับจาก THRESH_BINARY เป็น THRESH_BINARY_INV ดูครับ
        _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # ย่อขนาดให้เท่ากับที่โมเดลต้องการ (64x64)
        resized = cv2.resize(binary_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        # ปรับค่าสีจาก 0-255 เป็น 0-1 (Normalize)
        img_array = resized / 255.0

        # เพิ่มมิติให้เป็น Batch (1, 64, 64, 1)
        img_final = np.expand_dims(img_array, axis=-1)
        img_final = np.expand_dims(img_final, axis=0)

        # 5. ส่งเข้าโมเดลเพื่อทำนาย
        prediction = model.predict(img_final)
        
        # 6. แสดงผล
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result_char = CLASS_LABELS[predicted_index]

        print("\n" + "="*30)
        print(f"ผลลัพธ์คือเลข:  {result_char}")
        print(f"ความมั่นใจ:    {confidence:.2f}%")
        print("="*30 + "\n")

    except Exception as e:
        print(f"เกิดข้อผิดพลาดขณะประมวลผลภาพ: {e}")

if __name__ == "__main__":
    main()