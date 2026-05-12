# Unified Thai OCR App

แอปนี้รวม 2 ระบบเข้าด้วยกัน:

- OCR ตัวเลขไทยจาก `Experiments/14dec`
- OCR ข้อความไทยผ่าน API จาก `Experiments/2may`

รองรับการรันจาก root โปรเจกต์ได้ 2 แบบ:

```bash
streamlit run app/main.py
```

หรือ

```bash
streamlit run Code_and_Experiments/app/main.py
```

## โครงสร้างที่เกี่ยวข้อง

```text
Code_and_Experiments/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ image_utils.py
│  ├─ main.py
│  ├─ ocr_digit.py
│  ├─ ocr_thai_api.py
│  └─ ui_components.py
├─ Experiments/
│  ├─ 14dec/
│  └─ 2may/
├─ Models/
│  ├─ model_read_numberthaiV1_pytorch.pth
│  └─ thai_digit_modelV3.pth
├─ .env.example
└─ requirements.txt
```

## การติดตั้ง

1. สร้าง virtual environment

```bash
python -m venv .venv
```

2. เปิดใช้งาน environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

## การวางโมเดล

แอปจะพยายามหาโมเดลตามลำดับนี้:

1. `Code_and_Experiments/Models/model_read_numberthaiV1_pytorch.pth`
2. `Code_and_Experiments/Experiments/14dec/models/model_read_numberthaiV1_pytorch.pth`
3. `Code_and_Experiments/Models/thai_digit_modelV3.pth`

แนะนำให้วางโมเดลหลักไว้ที่:

```text
Code_and_Experiments/Models/model_read_numberthaiV1_pytorch.pth
```

## การตั้งค่า API

1. คัดลอก `.env.example` เป็น `.env`
2. วางไฟล์ `.env` ที่ root โปรเจกต์ หรือใน `Code_and_Experiments/`
3. ใส่ค่าอย่างน้อย:

```env
THAI_OCR_API_PROVIDER=gemini
THAI_OCR_API_KEY=your_api_key_here
THAI_OCR_API_MODEL=gemini-2.5-flash
```

หมายเหตุ:

- จากการตรวจโค้ดเดิม `2may` ใช้ Google Gemini SDK โดยตรง
- จึงใช้ `API key` แต่ไม่ได้ต้องมี custom endpoint เสมอไป
- ถ้าต้องการเก็บค่าแบบเดิม แอปใหม่รองรับ `GEMINI_API_KEY` ด้วย

## วิธีรัน

```bash
streamlit run Code_and_Experiments/app/main.py
```

## โหมดการทำงาน

### 1. OCR ตัวเลขไทย

- ใช้โมเดล PyTorch จาก `14dec`
- เหมาะกับภาพที่เป็นตัวเลขไทยล้วน

### 2. OCR ตัวอักษรไทยผ่าน API

- ใช้ Gemini OCR จาก logic ฝั่ง `2may`
- เหมาะกับข้อความไทยทั่วไป

### 3. OCR แบบรวม

- ใช้ API หา block ข้อความทั้งหมด
- ถ้า block ใดมีเลขไทย แอปจะ crop ส่วนนั้นแล้วส่งให้โมเดลตัวเลขไทยอ่านอีกครั้ง
- จากนั้นนำผลกลับมาจัดเรียงตามตำแหน่งเดิม

## การทดสอบ

### ทดสอบ OCR ตัวเลขไทย

1. เปิดแอป
2. เลือกโหมด `OCR ตัวเลขไทย`
3. อัปโหลดภาพตัวเลขไทย
4. ตรวจว่ามีผลลัพธ์ข้อความ, ค่า confidence, และภาพ debug box

### ทดสอบ OCR ข้อความไทยผ่าน API

1. ตั้งค่า `.env`
2. เลือกโหมด `OCR ตัวอักษรไทยผ่าน API`
3. อัปโหลดภาพเอกสารภาษาไทย
4. ตรวจว่าระบบแสดงข้อความและกรอบ bounding box

### ทดสอบ OCR แบบรวม

1. ใช้ภาพที่มีทั้งข้อความไทยและเลขไทย
2. เลือกโหมด `OCR แบบรวม`
3. ตรวจว่าเลขไทยในผลลัพธ์มาจากโมเดลตัวเลข ไม่ใช่ใช้ข้อความจาก API ตรง ๆ

## ปัญหาที่พบบ่อย

### ไม่พบโมเดล

- ตรวจว่ามีไฟล์ `model_read_numberthaiV1_pytorch.pth`
- ตรวจ path ใต้ `Code_and_Experiments/Models/`

### API ใช้ไม่ได้

- ตรวจ `THAI_OCR_API_KEY` หรือ `GEMINI_API_KEY`
- ตรวจว่าเครื่องออกอินเทอร์เน็ตได้
- ถ้า `gemini-2.5-flash` ใช้ไม่ได้ ระบบจะลอง fallback ไป `gemini-flash`

### รูปไม่ชัดหรืออ่านเลขผิด

- ลองเปิด `กลับสีภาพสำหรับ OCR ตัวเลข`
- ปรับ `ระดับลดความหนาเส้น`
- ใช้ภาพครอปที่ชัดขึ้นและมี contrast สูง
