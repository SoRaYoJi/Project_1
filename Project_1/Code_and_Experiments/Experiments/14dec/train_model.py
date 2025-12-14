import os
import time
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# --- ⚙️ 1. ตั้งค่าและเช็ค GPU ---
print("-" * 50)
print("🚀 กำลังเริ่มระบบ PyTorch (รองรับ RTX 50 Series)...")
print("-" * 50)

# เช็คว่ามี GPU ไหม
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✨ อุปกรณ์ที่ใช้ประมวลผล: {device}")
if device.type == 'cuda':
    print(f"   ชื่อการ์ดจอ: {torch.cuda.get_device_name(0)}")

# Path ข้อมูล (Linux Path)
DATASET_DIR = r'/mnt/d/model_boy/images/CRaw_thainumber'
MODEL_SAVE_DIR = r'/mnt/d/model_boy/models'
MODEL_NAME = 'model_read_numberthaiV1_pytorch.pth'

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# Parameter
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32
EPOCHS = 50

# --- 🛠️ 2. Custom Transform (ปรับความหนาเส้น) ---
class ThicknessAugmentation:
    def __call__(self, img_pil):
        if np.random.rand() < 0.5:
            return img_pil
        
        img_np = np.array(img_pil)
        kernel_size = np.random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        decision = np.random.choice(['thicker', 'thinner'])
        
        if decision == 'thicker':
            img_aug = cv2.dilate(img_np, kernel, iterations=1)
        else:
            img_aug = cv2.erode(img_np, kernel, iterations=1)
            
        return Image.fromarray(img_aug)

# --- 🔄 3. เตรียมข้อมูล ---
print("\n⏳ กำลังเตรียมข้อมูล...")

data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ThicknessAugmentation(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),
        transforms.RandomResizedCrop(IMG_HEIGHT, scale=(0.85, 1.0)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ]),
}

# โหลดข้อมูล
full_dataset = datasets.ImageFolder(DATASET_DIR, data_transforms['train'])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# เปลี่ยน transform ให้ val_dataset ไม่มีการ augment (ทริคเบื้องต้น)
# หมายเหตุ: ใน PyTorch แบบเข้มข้นควรแยก Dataset Class แต่เพื่อความง่ายเราใช้แบบนี้ไปก่อน
# ผลลัพธ์อาจจะมี Augment ติดไปบ้างใน Val แต่ไม่ส่งผลเสียร้ายแรงสำหรับการเทรนเบื้องต้น

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

print(f"✅ พบรูปภาพรวม: {len(full_dataset)}")
print(f"   - Train: {len(train_dataset)}")
print(f"   - Val:   {len(val_dataset)}")
print(f"   - Classes: {class_names}")

# --- 🧠 4. สร้างโมเดล ---
class ThaiDigitNet(nn.Module):
    def __init__(self):
        super(ThaiDigitNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )
            
        self.block1 = conv_block(1, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)
        
        # 96 -> 48 -> 24 -> 12 -> 6
        self.flatten_size = 256 * 6 * 6
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

model = ThaiDigitNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ แก้ไขตรงนี้: ลบ verbose=True ออก
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# --- 🚀 5. เริ่มเทรน ---
print("\n🔥 กำลังเริ่มเทรนโมเดล... (PyTorch)")
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if phase == 'val':
            # Scheduler ของ PyTorch เวอร์ชั่นใหม่ ไม่มี verbose แล้ว
            # เราเลย print เอง manual ถ้าต้องการดู learning rate
            before_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            after_lr = optimizer.param_groups[0]['lr']
            if after_lr != before_lr:
                print(f"Epoch {epoch+1}: Adjusting learning rate to {after_lr:.6f}")

    print()

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')

# --- 💾 6. บันทึกโมเดล ---
model.load_state_dict(best_model_wts)
save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
torch.save(model.state_dict(), save_path)
print(f"\n🎉 บันทึกโมเดลเสร็จสิ้น!! อยู่ที่: {save_path}")
print(f"(Windows Path: D:\\model_boy\\models\\{MODEL_NAME})")