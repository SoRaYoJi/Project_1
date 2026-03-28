import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageGrab
import tkinter as tk
import numpy as np
import os

# ---------------- ตั้งค่าพื้นฐาน ----------------
MODEL_PATH = r"D:\cnumthai\thai_digit_modelV3.pthh"
IMAGE_SIZE = 64
THAI_DIGITS_MAP = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- สถาปัตยกรรม CNN ----------------
class AdvancedThaiDigitCNN(nn.Module):
    def __init__(self):
        super(AdvancedThaiDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# โหลดโมเดล
model = AdvancedThaiDigitCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

inference_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class DrawingBoard:
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบอ่านเลขไทย (Screenshot Mode)")
        
        # สร้างกระดาน
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white", highlightthickness=5, highlightbackground="red")
        self.canvas.pack(pady=20, padx=20)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="ล้างกระดาน", command=self.clear, font=("Tahoma", 12), width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="ทายผล (Predict)", command=self.predict, font=("Tahoma", 12, "bold"), bg="#2ecc71", fg="white", width=15).pack(side=tk.LEFT, padx=10)
        
        self.label_res = tk.Label(root, text="เขียนเลขไทยให้เต็มกรอบแดง", font=("Tahoma", 16))
        self.label_res.pack(pady=20)
        
        self.brush_size = 10

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")

    def predict(self):
        # --- วิธีใหม่: ถ่ายรูปหน้าจอเฉพาะพิกัดของ Canvas ---
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # ถ่ายรูป (เพิ่มลบขอบนิดหน่อยให้คลีน)
        img = ImageGrab.grab().crop((x+5, y+5, x1-5, y1-5))
        
        # หาขอบตัวเลข
        gray = img.convert('L')
        inverted = ImageOps.invert(gray)
        bbox = inverted.getbbox()
        
        if bbox:
            cropped = img.crop(bbox)
            w, h = cropped.size
            max_side = max(w, h)
            # สร้างขอบขาวรอบๆ ให้เหมือนฟอนต์
            padded = Image.new("RGB", (int(max_side*1.5), int(max_side*1.5)), "white")
            padded.paste(cropped, ((padded.width-w)//2, (padded.height-h)//2))
            
            # เซฟไฟล์ไว้ดูว่า AI เห็นอะไร
            padded.save("ai_vision_test.png")
            
            # ทำนายผล
            input_tensor = inference_transforms(padded).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                prob = torch.nn.functional.softmax(out, dim=1)[0]
                val, idx = torch.max(prob, 0)
            
            self.label_res.config(text=f"ทายว่าเป็นเลข: {idx.item()} ({THAI_DIGITS_MAP[idx.item()]})\nมั่นใจ: {val.item()*100:.2f}%")
        else:
            self.label_res.config(text="มองไม่เห็นรอยวาดเลย!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingBoard(root)
    root.mainloop()