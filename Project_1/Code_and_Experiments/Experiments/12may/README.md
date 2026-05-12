# 12may

โฟลเดอร์นี้เป็น snapshot งานวันที่ 12 พฤษภาคม สำหรับรวบรวมไฟล์ที่ใช้งานจริงและไฟล์ที่เกี่ยวข้องของระบบ OCR ภาษาไทยในโปรเจ็กต์นี้ให้อยู่ในที่เดียว เพื่อหยิบไปทำต่อ ทดสอบต่อ หรือส่งต่อให้คนอื่นอ่านงานได้ง่ายขึ้น

## ทำอะไรไปบ้าง

- clone ไฟล์ที่ใช้งานและเกี่ยวข้องมาไว้ในโฟลเดอร์ `12may`
- แยกชุดไฟล์ให้ยังคงโครงสร้างเดิมบางส่วน เพื่อให้ path และ import ใช้งานต่อได้ง่าย
- รวมทั้งระบบ OCR แบบแอปปัจจุบัน, โมเดลเลขไทย, สคริปต์ทดลอง 2may และสคริปต์ legacy ใน `Source_Code`
- เพิ่ม README นี้เพื่อสรุปภาพรวมงาน วิธีใช้งาน สถานะปัจจุบัน จุดที่ติด และโมเดลที่ใช้

## มีอะไรอยู่ในโฟลเดอร์นี้

```text
12may/
├─ README.md
├─ .env.example
├─ requirements.txt
├─ app/
│  └─ main.py
└─ Code_and_Experiments/
   ├─ __init__.py
   ├─ README.md
   ├─ requirements.txt
   ├─ app/
   │  ├─ config.py
   │  ├─ document_export.py
   │  ├─ image_utils.py
   │  ├─ main.py
   │  ├─ ocr_digit.py
   │  ├─ ocr_thai_api.py
   │  ├─ ui_components.py
   │  └─ __init__.py
   ├─ Models/
   │  ├─ model_read_numberthaiV1_pytorch.pth
   │  └─ thai_digit_modelV3.pth
   ├─ Experiments/
   │  └─ 2may/
   │     ├─ thai_num_extractor.py
   │     ├─ t1.png
   │     └─ final_output_manual.txt
   └─ Source_Code/
      ├─ ocr_combined.py
      ├─ predict_document.py
      ├─ requirements.txt
      ├─ README.md
      └─ Sarabun-ThinItalic.ttf
```

## ระบบที่มี

### 1. Unified Thai OCR App

อยู่ที่ `12may/Code_and_Experiments/app/`

ระบบนี้เป็นแอปหลักที่รวมหลายโหมดไว้ในตัวเดียว

- OCR เอกสารทั้งหน้า
- OCR ข้อความไทยผ่าน API
- OCR แบบรวม: ใช้ API อ่านข้อความก่อน แล้วใช้โมเดลเลขไทยช่วยแทนค่าบล็อกที่เป็นเลขไทย
- OCR เลขไทยล้วน
- export ผลลัพธ์เป็น `txt`, `json`, `png`, `docx`, `pdf`

### 2. OCR เลขไทย

อยู่ที่ `12may/Code_and_Experiments/app/ocr_digit.py`

- ใช้โมเดล PyTorch สำหรับอ่านเลขไทย `๐-๙`
- เหมาะกับภาพ crop ที่เป็นเลขไทยโดยตรง หรือบล็อกเลขที่ตัดมาจากเอกสาร

### 3. OCR ข้อความไทย/อักษรไทย

มี 2 แนวทางใน snapshot นี้

- แนวทางปัจจุบัน: ใช้ Gemini API ผ่าน `12may/Code_and_Experiments/app/ocr_thai_api.py`
- แนวทางเก่า: ใช้ Tesseract ผ่าน `12may/Code_and_Experiments/Source_Code/ocr_combined.py`

สรุปคือระบบนี้รองรับทั้งเลขไทยและอักษรไทย โดยอักษรไทยเน้นอ่านผ่าน API ส่วนเลขไทยมีโมเดลของตัวเองช่วยอ่าน

## โมเดลและเครื่องมือที่ใช้

### โมเดลเลขไทย

- `model_read_numberthaiV1_pytorch.pth`
- `thai_digit_modelV3.pth`

ทั้งสองไฟล์อยู่ใน `12may/Code_and_Experiments/Models/`

### เครื่องมือ OCR ข้อความไทย

- Gemini API
  - ค่าเริ่มต้นในไฟล์ตั้งค่าใช้ `gemini-2.5-flash`
  - มี fallback ไป `gemini-flash`
- Tesseract OCR
  - ใช้ในสคริปต์ legacy `ocr_combined.py`

## ความแม่นยำโดยประมาณ

จากสถานะงานปัจจุบันสามารถอธิบายได้ว่า:

- ความแม่นยำโดยรวมที่คาดหวังอยู่ประมาณ `90%`
- เลขไทยในภาพที่ค่อนข้างชัดมีแนวโน้มแม่นกว่าส่วนข้อความยาว
- งานอ่านทั้งหน้าเอกสารยังขึ้นกับคุณภาพภาพ, การจัดวางข้อความ, และความแม่นของ OCR API/Tesseract

หมายเหตุสำคัญ:

- ยังไม่พบ benchmark กลางแบบ automated test ที่สรุปตัวเลข 90% อย่างเป็นทางการใน repo
- ดังนั้นค่าประมาณ 90% ใน README นี้เป็นสถานะสรุปเชิงใช้งานของโปรเจ็กต์ ณ ตอนนี้ ไม่ใช่ผลวัดมาตรฐานจากชุดทดสอบเดียวกันทั้งหมด

## รันยังไง

### วิธีหลัก: รันแอป Unified OCR

1. สร้าง virtual environment

```powershell
python -m venv .venv
```

2. เปิดใช้งาน

```powershell
.venv\Scripts\Activate.ps1
```

3. ติดตั้ง dependencies

```powershell
pip install -r requirements.txt
```

หรือ

```powershell
pip install -r Code_and_Experiments\requirements.txt
```

4. ถ้าจะใช้ OCR ข้อความผ่าน API ให้คัดลอก `.env.example` เป็น `.env` แล้วใส่ key

```env
THAI_OCR_API_PROVIDER=gemini
THAI_OCR_API_KEY=your_api_key_here
THAI_OCR_API_MODEL=gemini-2.5-flash
```

5. รันแอป

```powershell
streamlit run app\main.py
```

หรือ

```powershell
streamlit run Code_and_Experiments\app\main.py
```

### วิธีรอง: รันสคริปต์ legacy

```powershell
python Code_and_Experiments\Source_Code\ocr_combined.py <path_to_image>
```

```powershell
python Code_and_Experiments\Source_Code\predict_document.py <path_to_image>
```

### วิธีทดลองจากชุด 2may

```powershell
python Code_and_Experiments\Experiments\2may\thai_num_extractor.py
```

ไฟล์นี้จะใช้ `.env` และอ่านภาพตัวอย่าง `t1.png` เป็นหลัก

## อะไรใช้ทำอะไร

- `app/main.py`
  - ตัวเปิดแอปจาก root
- `Code_and_Experiments/app/main.py`
  - หน้า Streamlit หลักของระบบ OCR
- `Code_and_Experiments/app/ocr_digit.py`
  - อ่านเลขไทยจากภาพ
- `Code_and_Experiments/app/ocr_thai_api.py`
  - เรียก OCR API และรวมผลกับโมเดลเลขไทย
- `Code_and_Experiments/app/document_export.py`
  - export ผลลัพธ์เป็นหลายฟอร์แมต
- `Code_and_Experiments/Experiments/2may/thai_num_extractor.py`
  - pipeline ทดลองที่ใช้ API อ่านเอกสารแล้วตัดเลขไทยออกมาเก็บ
- `Code_and_Experiments/Source_Code/ocr_combined.py`
  - pipeline เก่าแบบ Tesseract + โมเดลเลขไทย
- `Code_and_Experiments/Source_Code/predict_document.py`
  - ทดลอง OCR เอกสารแบบ segmentation/recognition

## ตอนนี้ไปถึงไหนแล้ว

- มีแอป OCR ที่รวมหลายโหมดไว้ในจุดเดียวแล้ว
- มีโมเดลเลขไทยที่ใช้งานร่วมกับระบบ OCR เอกสารได้
- มีระบบ export ผลลัพธ์หลายแบบ
- มีทั้งสาย legacy และสาย app ใหม่ให้เลือกใช้
- มีการเชื่อม OCR ข้อความไทยกับ OCR เลขไทยเข้าด้วยกันแล้ว

## ติดตรงไหน

- OCR ข้อความไทยส่วนใหญ่พึ่ง API ภายนอก จึงต้องมี key และอินเทอร์เน็ต
- OCR แบบ legacy พึ่ง Tesseract ที่ติดตั้งในเครื่อง และ path ถูกตั้งแบบ Windows ตายตัว
- ยังไม่มี automated test / benchmark กลางที่วัด accuracy แบบสม่ำเสมอทั้งระบบ
- ความแม่นยำของภาพทั้งหน้าอาจลดลงเมื่อภาพเอียง เบลอ แสงไม่สม่ำเสมอ หรือมี layout ซับซ้อน
- `predict_document.py` เป็นสายทดลอง/legacy และอาจต้องมีโมดูล model เพิ่มเติมนอก snapshot นี้หากจะใช้งานจริงต่อ

## ข้อแนะนำถ้าจะทำต่อ

- ทำชุด test image พร้อม expected output แยกไว้ชัดเจน
- วัด accuracy แยกเป็น 3 งาน: เลขไทย, ข้อความไทย, เอกสารทั้งหน้า
- ลด dependency ที่ผูกกับ path เฉพาะเครื่อง
- สรุปให้ชัดว่า pipeline หลักที่จะใช้ต่อคือสาย `app/` หรือสาย `Source_Code/`

