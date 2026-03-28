import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# ---------------- ตั้งค่า Path ----------------
BASE_PATH = r"D:\cnumthai"
FONTS_DIR = r"C:\Windows\Fonts" 
THAI_DIGITS = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

def has_thai_digits(font_path):
    """ตรวจสอบว่าไฟล์ฟอนต์นี้มี Glyph ของเลขไทย (๐) หรือไม่"""
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        if cmap and 0x0E50 in cmap:
            return True
    except Exception:
        pass
    return False

def get_all_thai_fonts(fonts_dir):
    """ค้นหาฟอนต์ทั้งหมดในโฟลเดอร์ที่รองรับเลขไทย"""
    print("กำลังสแกนหาฟอนต์ที่รองรับเลขไทยในเครื่อง...")
    thai_fonts = []
    font_files = glob.glob(os.path.join(fonts_dir, "*.ttf")) + glob.glob(os.path.join(fonts_dir, "*.otf"))
    
    for font_path in font_files:
        if has_thai_digits(font_path):
            thai_fonts.append(font_path)
            
    print(f"เจอพอนต์ที่รองรับภาษาไทยทั้งหมด: {len(thai_fonts)} ฟอนต์")
    return thai_fonts

def create_image_from_font(text, font_path, save_path, image_size=(100, 100)):
    """สร้างรูปภาพจากไฟล์ฟอนต์"""
    try:
        img = Image.new('RGB', image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=70)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (image_size[0] - text_w) / 2
        y = (image_size[1] - text_h) / 2 - bbox[1]
        
        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        img.save(save_path)
    except Exception as e:
        pass # ซ่อน Error เพื่อไม่ให้รกหน้าจอเวลาเจอไฟล์ฟอนต์มีปัญหา

def augment_existing_image(image_path, save_dir, filename):
    """ทำรูปเอียงซ้าย เอียงขวา และทำให้เส้นหนาขึ้น จากรูปที่มีอยู่"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    h, w = img.shape
    shear_factor = 0.3 
    
    # 1. เอียงซ้าย (Shear Left)
    M_shear_left = np.float32([[1, shear_factor, -shear_factor * w / 2], [0, 1, 0]])
    img_sheared_left = cv2.warpAffine(img, M_shear_left, (w, h), borderValue=255)
    
    # 2. เอียงขวา (Shear Right) - สังเกตการกลับเครื่องหมาย
    M_shear_right = np.float32([[1, -shear_factor, shear_factor * w / 2], [0, 1, 0]])
    img_sheared_right = cv2.warpAffine(img, M_shear_right, (w, h), borderValue=255)
    
    # 3. เส้นหนาขึ้น (Bold)
    kernel = np.ones((3, 3), np.uint8)
    img_bold = cv2.erode(img, kernel, iterations=1)
    
    # บันทึกรูปภาพทั้ง 3 แบบ
    cv2.imwrite(os.path.join(save_dir, f"sheared_left_{filename}"), img_sheared_left)
    cv2.imwrite(os.path.join(save_dir, f"sheared_right_{filename}"), img_sheared_right)
    cv2.imwrite(os.path.join(save_dir, f"bold_{filename}"), img_bold)

def main():
    thai_fonts = get_all_thai_fonts(FONTS_DIR)
    
    if not thai_fonts:
        print("ไม่พบฟอนต์ที่รองรับภาษาไทยเลย")
        return

    for i in range(10):
        folder_path = os.path.join(BASE_PATH, str(i))
        
        if not os.path.exists(folder_path):
            continue
            
        print(f"กำลังประมวลผลโฟลเดอร์: {folder_path} (เลข {THAI_DIGITS[i]})")
        
        # สร้างรูปจากฟอนต์
        char = THAI_DIGITS[i]
        for font_path in thai_fonts:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            save_name = f"font_{font_name}_{i}.jpg"
            create_image_from_font(char, font_path, os.path.join(folder_path, save_name))
        
        # ทำ Augmentation จากรูปเดิม
        existing_images = glob.glob(os.path.join(folder_path, "*.*"))
        
        for img_path in existing_images:
            filename = os.path.basename(img_path)
            # ข้ามไฟล์ที่ถูกสร้างจากการ Augment และฟอนต์ เพื่อไม่ให้วนลูปซ้ำ
            if filename.startswith("sheared_") or filename.startswith("bold_") or filename.startswith("font_"):
                continue
            augment_existing_image(img_path, folder_path, filename)

    print("เสร็จสิ้นการรวบรวมฟอนต์และเพิ่ม Data (เอียงซ้าย/ขวา/หนา) ครับ!")

if __name__ == "__main__":
    main()