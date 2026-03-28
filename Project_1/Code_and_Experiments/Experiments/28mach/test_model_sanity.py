import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os

# ---------------- ตั้งค่าพื้นฐาน (ต้องตรงเป๊ะ) ----------------
MODEL_PATH = r"D:\cnumthai\thai_digit_modelV3.pth"
DATA_DIR = r"D:\cnumthai" # Path เดียวกับตอนเทรน
IMAGE_SIZE = 64
THAI_DIGITS_MAP = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

# ตรวจสอบ GPU (RTX 5060 Ti ตัวเดิม)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"กำลังโหลดโมเดลรันบน: {device}")

# ---------------- สถาปัตยกรรม CNN (ต้องตรงเป๊ะ) ----------------
class AdvancedThaiDigitCNN(nn.Module):
    def __init__(self):
        super(AdvancedThaiDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------------- โหลดน้ำหนักโมเดล ----------------
model = AdvancedThaiDigitCNN()
try:
    # เพิ่มweights_only=Trueเพื่อความปลอดภัย
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("✓ โหลดโมเดลสำเร็จ!")
except FileNotFoundError:
    print(f"✗ ไม่พบไฟล์โมเดลที่ Path: {MODEL_PATH}")
    exit()

# กระบวนการเตรียมภาพก่อนส่งให้โมเดล
inference_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------- เริ่มการทดสอบแบบรายคลาส ----------------
print("\n--- เริ่มการทดสอบแบบเอารูปตัวจริงมาป้อนโมเดลดู ---")
print("-" * 50)
print(f"{'เลขไทยเป้าหมาย':<15} | {'โมเดลทายผล':<15} | {'ความมั่นใจ (%)':<15}")
print("-" * 50)

# ลองหยิบรูปแรกจากแต่ละโฟลเดอร์ 0-9 มาเทส
for digit_class in range(10):
    class_dir = os.path.join(DATA_DIR, str(digit_class))
    if not os.path.exists(class_dir):
        print(f"✗ ไม่พบโฟลเดอร์สำหรับเลข {digit_class} ที่ {class_dir}")
        continue
    
    # ดึงไฟล์รูปแรกสุดออกมา
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if not files:
        print(f"✗ ไม่พบไฟล์รูปภาพในโฟลเดอร์ {class_dir}")
        continue
    
    first_image_path = os.path.join(class_dir, files[0])
    
    try:
        # เปิดรูปด้วย PIL
        img = Image.open(first_image_path)
        
        # เตรียมภาพ (Pre-processing)
        input_tensor = inference_transforms(img).unsqueeze(0).to(device)
        
        # ทายผล
        with torch.no_grad():
            outputs = model(input_tensor)
            percentages = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(percentages.data, 1)
            
        prediction_index = predicted.item()
        confidence = max_prob.item() * 100
        
        # แปลงเป็นเลขไทยเพื่อการแสดงผล
        thai_target = THAI_DIGITS_MAP[digit_class]
        thai_predict = THAI_DIGITS_MAP[prediction_index]
        
        print(f"{digit_class} ({thai_target}) : รูปที่ 1 | {prediction_index} ({thai_predict}) | {confidence:.2f}%")
        
    except Exception as e:
        print(f"✗ เกิดข้อผิดพลาดในการเทสเลข {digit_class}: {e}")

print("-" * 50)
print("สรุป:")
print("ถ้าผลลัพธ์เป็น 'ทายว่าเป็นเลข 0' หมดทุกบรรทัด แสดงว่าไฟล์โมเดลพังเงียบๆ ครับ")
print("ถ้าผลลัพธ์เป็นเลขอื่นๆ ที่ถูกต้อง แสดงว่าสมอง AI ปกติ แต่มีบั๊กที่ GUI Code ครับ")