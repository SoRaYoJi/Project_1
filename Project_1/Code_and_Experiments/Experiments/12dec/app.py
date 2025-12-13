import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

# ปิด Log สีแดงๆ ของ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- ตั้งค่า Configuration ---
MODEL_PATH = 'thai_digit_model_64x64_Thickness_V2.keras'
IMG_HEIGHT = 64
IMG_WIDTH = 64
CLASS_LABELS = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

class ThaiDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("โปรแกรมทายเลขไทย (AI Prediction) - แก้ไขบั๊ก Display")
        self.root.geometry("700x750")
        self.root.resizable(False, False)

        self.current_image_path = None
        self.model = None
        
        # ตัวแปรเก็บภาพสำหรับแสดงผล (กันภาพหาย)
        self.tk_img_orig = None
        self.tk_img_proc = None

        self.load_trained_model()

        # --- ส่วนหน้าจอ (UI) ---
        tk.Label(root, text="ระบบทายลายมือเลขไทย", font=("Angsana New", 24, "bold")).pack(pady=10)

        # Container วางรูปคู่
        self.images_container = tk.Frame(root, bg="#EEEEEE", bd=2, relief="groove")
        self.images_container.pack(pady=10, padx=10, fill="x")

        # รูปซ้าย (ต้นฉบับ)
        self.frame_left = tk.Frame(self.images_container, bg="#EEEEEE")
        self.frame_left.pack(side=tk.LEFT, expand=True, padx=5, pady=5)
        tk.Label(self.frame_left, text="1. ภาพต้นฉบับ", font=("Arial", 10, "bold"), bg="#EEEEEE").pack()
        self.label_orig = tk.Label(self.frame_left, text="[รอเลือกรูป]", bg="white", width=30, height=15, relief="sunken")
        self.label_orig.pack(pady=5)

        # รูปขวา (AI เห็น)
        self.frame_right = tk.Frame(self.images_container, bg="#EEEEEE")
        self.frame_right.pack(side=tk.LEFT, expand=True, padx=5, pady=5)
        tk.Label(self.frame_right, text="2. ภาพที่ AI เห็น (ขาวดำ)", font=("Arial", 10, "bold"), bg="#EEEEEE").pack()
        self.label_proc = tk.Label(self.frame_right, text="[รอการประมวลผล]", bg="white", width=30, height=15, relief="sunken")
        self.label_proc.pack(pady=5)

        # ปุ่มเลือกไฟล์
        self.btn_select = tk.Button(root, text="📂 เลือกรูปภาพ", font=("Arial", 12), command=self.select_file, bg="#E0E0E0", cursor="hand2")
        self.btn_select.pack(pady=5)

        # Checkbox
        self.invert_var = tk.BooleanVar(value=False)
        self.chk_invert = tk.Checkbutton(root, text="กลับสีภาพ (ติ๊กเมื่อ: ตัวอักษรดำ พื้นหลังขาว)", 
                                         variable=self.invert_var, font=("Arial", 10), cursor="hand2",
                                         command=self.refresh_processed_view)
        self.chk_invert.pack(pady=5)

        # ปุ่มทำนาย
        self.btn_predict = tk.Button(root, text="🔮 ทำนายผล", font=("Arial", 16, "bold"), command=self.predict_image, bg="#4CAF50", fg="white", state="disabled", cursor="hand2")
        self.btn_predict.pack(pady=15, ipadx=30, ipady=5)

        # ผลลัพธ์
        self.result_frame = tk.Frame(root, bd=1, relief="solid", padx=20, pady=10)
        self.result_frame.pack(pady=10, fill="x", padx=20)
        self.lbl_result = tk.Label(self.result_frame, text="ผลลัพธ์: -", font=("Angsana New", 40, "bold"), fg="blue")
        self.lbl_result.pack()
        self.lbl_conf = tk.Label(self.result_frame, text="ความมั่นใจ: -", font=("Arial", 12))
        self.lbl_conf.pack()

    def load_trained_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH, compile=False)
                print("✅ โหลดโมเดลสำเร็จ!")
            except Exception as e:
                messagebox.showerror("Error", f"โหลดโมเดลไม่สำเร็จ:\n{e}")
        else:
            messagebox.showwarning("Warning", f"❌ ไม่พบไฟล์โมเดล '{MODEL_PATH}'")

    def imread_unicode(self, path):
        try:
            stream = open(path, "rb")
            bytes_data = stream.read()
            stream.close()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_decoded
        except Exception as e:
            print(f"Failed to read image: {e}")
            return None

    def select_file(self):
        try:
            file_path = filedialog.askopenfilename(
                parent=self.root,
                title="เลือกรูปภาพ",
                filetypes=[
                    ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
                    ("All Files", "*.*")
                ]
            )

            if file_path:
                print(f"เลือกไฟล์: {file_path}")
                self.current_image_path = file_path
                self.process_and_display_images(file_path)
                
                if self.model:
                    self.btn_predict.config(state="normal", bg="#4CAF50")
                
                self.reset_results()
        except Exception as e:
            messagebox.showerror("Error", f"เปิดหน้าต่างเลือกไฟล์ไม่ได้: {e}")

    def reset_results(self):
        self.lbl_result.config(text="ผลลัพธ์: -", fg="blue")
        self.lbl_conf.config(text="ความมั่นใจ: -")

    def refresh_processed_view(self):
        if self.current_image_path:
            self.process_and_display_images(self.current_image_path, update_original=False)

    def process_and_display_images(self, path, update_original=True):
        try:
            # 1. แสดงภาพต้นฉบับ
            if update_original:
                pil_img_orig = Image.open(path)
                self.tk_img_orig = self.resize_for_display(pil_img_orig)
                # แก้ไขตรงนี้: เปลี่ยนจาก width="" เป็น width=0
                self.label_orig.configure(image=self.tk_img_orig, text="", width=0, height=0)

            # 2. แสดงภาพที่ AI เห็น (Processed)
            cv_img = self.imread_unicode(path)
            if cv_img is None: 
                raise Exception("อ่านไฟล์รูปภาพไม่ได้")

            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            thresh_type = cv2.THRESH_BINARY_INV if self.invert_var.get() else cv2.THRESH_BINARY
            _, binary_img = cv2.threshold(gray, 128, 255, thresh_type)

            pil_img_proc = Image.fromarray(binary_img)
            self.tk_img_proc = self.resize_for_display(pil_img_proc)
            # แก้ไขตรงนี้: เปลี่ยนจาก width="" เป็น width=0
            self.label_proc.configure(image=self.tk_img_proc, text="", width=0, height=0)
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการแสดงภาพ: {e}")
            print(f"Display Error: {e}")

    def resize_for_display(self, pil_image, target_height=250):
        aspect_ratio = pil_image.width / pil_image.height
        target_width = int(target_height * aspect_ratio)
        resized_img = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized_img)

    def predict_image(self):
        if not self.model or not self.current_image_path:
            return

        try:
            # Preprocess
            img = self.imread_unicode(self.current_image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_type = cv2.THRESH_BINARY_INV if self.invert_var.get() else cv2.THRESH_BINARY
            _, binary_img = cv2.threshold(gray, 128, 255, thresh_type)

            # Resize 64x64
            resized = cv2.resize(binary_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_array = resized.astype("float32") / 255.0
            img_final = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)

            # Predict
            prediction = self.model.predict(img_final)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            result_char = CLASS_LABELS[predicted_index]

            # Show Result
            color = "#008000" if confidence > 80 else "#FF8C00"
            self.lbl_result.config(text=f"ผลลัพธ์: {result_char}", fg=color)
            self.lbl_conf.config(text=f"ความมั่นใจ: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการทำนาย:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThaiDigitApp(root)
    root.mainloop()