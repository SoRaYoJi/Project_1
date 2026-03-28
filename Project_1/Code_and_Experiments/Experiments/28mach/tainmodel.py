import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import time

# ---------------- 1. ตั้งค่าพื้นฐาน ----------------
# อัปเดต Path ให้ตรงกับโฟลเดอร์ข้อมูลปัจจุบัน
DATA_DIR = r"D:\cnumthai"
BATCH_SIZE = 128  
MAX_EPOCHS = 200  
INITIAL_LR = 0.001
IMAGE_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"จัดเต็ม! กำลังเทรนโมเดลบน: {device}")

# ---------------- 2. เตรียมข้อมูล (Data Augmentation) ----------------
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# โหลดข้อมูลจาก D:\cnumthai โฟลเดอร์ 0-9 จะถูกแปลงเป็น Class อัตโนมัติ
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)

# ตรวจสอบจำนวนคลาส
print(f"พบโฟลเดอร์ (Class) ทั้งหมด: {full_dataset.classes}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transforms

# หมายเหตุ: ใน Windows ถ้าตั้ง num_workers เยอะเกินไปอาจค้างได้ ถ้าค้างให้เปลี่ยนเป็น 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- 3. สร้างสถาปัตยกรรม CNN ----------------
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
            nn.Linear(256, 10) # 10 คลาส (เลข 0-9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = AdvancedThaiDigitCNN().to(device)

# ---------------- 4. Loss, Optimizer และ Scheduler ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ---------------- 5. ลูปการเทรนแบบ Early Stopping ----------------
def train_model():
    best_val_loss = float('inf')
    patience = 15  
    epochs_no_improve = 0
    
    print("เริ่มกระบวนการฝึกฝนแบบปล่อยไหล (Early Stopping)...")
    print("-" * 40)
    
    # ปกป้องการรัน DataLoader แบบ Multiprocessing บน Windows
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] - Time: {epoch_time:.0f}s - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_thai_digit_model_max.pth')
            print(f"-> ★ โมเดลฉลาดขึ้น! เซฟน้ำหนักที่ดีที่สุด (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"-> ไม่มีการพัฒนามาแล้ว {epochs_no_improve}/{patience} รอบ")
            
        print("-" * 40)
        
        if epochs_no_improve >= patience:
            print("🛑 สิ้นสุดการเทรน (Early Stopping)! โมเดลถึงจุดอิ่มตัวและมั่นใจที่สุดแล้วครับ")
            break

if __name__ == '__main__':
    # ป้องกัน Error เวลาใช้ num_workers บน Windows
    import multiprocessing
    multiprocessing.freeze_support()
    train_model()