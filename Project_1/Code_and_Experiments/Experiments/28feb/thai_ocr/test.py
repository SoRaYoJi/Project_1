import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. ปรับโครงสร้างให้ตรงตาม Error (Sequential ซ้อน Sequential)
class CRNN(nn.Module):
    def __init__(self, nclass):
        super(CRNN, self).__init__()
        
        # ฟังก์ชันช่วยสร้าง Block (Conv + BN + ReLU)
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

        self.backbone = nn.Sequential(
            conv_block(1, 32),    # backbone.0
            conv_block(32, 64),   # backbone.1
            conv_block(64, 128),  # backbone.2
            conv_block(128, 256), # backbone.3
        )
        
        # เพิ่ม MaxPool แยกออกมาเพื่อให้ลำดับ index ใน backbone ไม่เพี้ยน
        self.pools = nn.ModuleList([
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        ])

        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(256 * 2, nclass)

    def forward(self, x):
        # รันผ่านแต่ละ block และ pool
        for i in range(4):
            x = self.backbone[i](x)
            x = self.pools[i](x)
            
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# 2. ถอดรหัส CTC
def ctc_decode(predictions, alphabet):
    # ไม่ต้อง +1 เพราะ nclass จาก checkpoint คือตัวเลขที่ถูกต้องแล้ว
    char_map = {i + 1: char for i, char in enumerate(alphabet)}
    _, max_indices = torch.max(predictions, 2)
    max_indices = max_indices.transpose(1, 0).contiguous().view(-1)
    
    res = []
    for i in range(len(max_indices)):
        idx = max_indices[i].item()
        # CTC Blank มักจะเป็น 0
        if idx != 0 and (not (i > 0 and idx == max_indices[i - 1].item())):
            if idx in char_map:
                res.append(char_map[idx])
    return ''.join(res)

# 3. โหลดและรันผล
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/mnt/d/ptest/models/thai_crnn_ctc_best.pt'
image_path = '/mnt/d/ptest/data/lines/train/000005.png'

checkpoint = torch.load(model_path, map_location=device)
alphabet = checkpoint['charset']

# แก้ไขจำนวน Class ให้ตรงตาม Error (คือ 117)
n_class = 117 

model = CRNN(n_class).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 4. จัดการรูปภาพ
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = nn.functional.log_softmax(preds, 2)
        result = ctc_decode(preds, alphabet)

    print(f"\n--- Result ---")
    print(f"ข้อความ: {result}")
    print(f"--------------\n")
except Exception as e:
    print(f"Error: {e}")