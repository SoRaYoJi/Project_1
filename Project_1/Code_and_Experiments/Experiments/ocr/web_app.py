# -------------------------
# 🔐 FIREBASE AUTH UI
# -------------------------
import streamlit as st
import requests

FIREBASE_API_KEY = "AIzaSyBFxo7Psx-O13aFSjd8z8qJwf9-lL5jAW4"

LOGIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"

# 👉 URL หน้า Google Login (HTML)
GOOGLE_LOGIN_PAGE = "http://localhost:8080"  # เปลี่ยนเป็น URL จริงตอน deploy


def firebase_auth(url, email, password):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    return requests.post(url, json=payload).json()


st.set_page_config(page_title="ระบบทายเลขไทย AI", layout="wide")

# -------------------------
# AUTH GATE
# -------------------------
params = st.query_params
token = params.get("token")

def verify_token(id_token):
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:lookup"
        f"?key={FIREBASE_API_KEY}"
    )
    return requests.post(url, json={"idToken": id_token}).json()


if "user" not in st.session_state:
    # 👉 กลับจาก Google
    if token:
        result = verify_token(token)
        if "users" in result:
            st.session_state.user = result["users"][0]["email"]
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Google Login ล้มเหลว")
            st.stop()

    # 👉 UI LOGIN ปกติ
    st.title("🔐 Authentication")
    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            res = firebase_auth(LOGIN_URL, email, password)
            if "idToken" in res:
                st.session_state.user = res["email"]
                st.rerun()
            else:
                st.error(res["error"]["message"])

        st.markdown("---")
        st.markdown("### 🔑 เข้าสู่ระบบด้วย Google")
        st.link_button("Login with Google", GOOGLE_LOGIN_PAGE)

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email", key="r_email")
            password = st.text_input("Password", type="password", key="r_pass")
            submit = st.form_submit_button("Register")

        if submit:
            res = firebase_auth(SIGNUP_URL, email, password)
            if "idToken" in res:
                st.success("สมัครสมาชิกสำเร็จ")
            else:
                st.error(res["error"]["message"])

    st.stop()

st.sidebar.success(f"👤 {st.session_state.user}")
if st.sidebar.button("Logout"):
    del st.session_state.user
    st.rerun()


# -------------------------
# 🔽 ORIGINAL OCR CODE (UNCHANGED)
# -------------------------
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas 

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="ทายเลขไทย AI", page_icon="🇹🇭")

st.title("ระบบทายลายมือเลขไทย 🇹🇭")
st.write("ระบบ: PyTorch Model (รองรับ RTX 50 Series)")
st.info("สถานะ: 🟢 พร้อมทำงาน")

# --- 🧠 โครงสร้างโมเดล (ต้องเหมือนตอนเทรนเป๊ะๆ) ---
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

# --- 🛠️ ฟังก์ชันเตรียมภาพ (Preprocess) ---
def preprocess_for_model(roi, target_size=(96, 96)):
    """
    1. Resize โดยคงสัดส่วน
    2. แปะลงพื้นหลังดำ (Canvas = 0)
    3. แปลงเป็น Tensor (0-1)
    
    ROI ที่รับมาควรเป็น ตัวเลขสีขาว (255) พื้นหลังสีดำ (0)
    """
    h, w = roi.shape[:2]
    padding = 10 
    
    # คำนวณ Scale เพื่อคงสัดส่วนและมี Padding 
    scale = min((target_size[0] - padding*2) / w, (target_size[1] - padding*2) / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # ตรวจสอบขนาดไม่ให้เป็น 0
    if new_w == 0 or new_h == 0:
        return np.zeros(target_size, dtype=np.uint8), torch.zeros(1, 1, target_size[0], target_size[1], dtype=torch.float32)

    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # สร้าง Canvas พื้นหลังดำ (ค่า 0)
    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    # วางรูปที่ Resize แล้วลงตรงกลาง Canvas 
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_roi
    
    # แปลงเป็น Tensor และ Rescale เป็น [0, 1]
    img_tensor = canvas.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0)
    
    return canvas, img_tensor

# --- 🛠️ ฟังก์ชันสำหรับเรียงลำดับตัวเลข (OCR Sorting - Improved) ---
def sort_bounding_boxes(boxes, y_threshold=30): 
    """
    จัดเรียง Bounding Boxes ตามบรรทัด (Center Y) ก่อน แล้วค่อยเรียงตามแนวนอน (X)
    """
    processed_boxes = []
    for x, y, w, h in boxes:
        center_y = y + h // 2
        processed_boxes.append((x, y, w, h, center_y))
        
    processed_boxes.sort(key=lambda b: b[4]) 
    
    sorted_boxes = []
    current_line = []
    
    if not processed_boxes:
        return []

    current_line.append(processed_boxes[0])
    
    for i in range(1, len(processed_boxes)):
        box = processed_boxes[i]
        
        # ตรวจสอบว่ากล่องปัจจุบันอยู่บนบรรทัดเดียวกันหรือไม่ โดยใช้ Center Y
        if abs(box[4] - current_line[0][4]) < y_threshold:
            current_line.append(box)
        else:
            # จัดเรียงบรรทัดเก่าตาม X (ซ้ายไปขวา)
            current_line.sort(key=lambda b: b[0])
            clean_line = [b[:4] for b in current_line] 
            sorted_boxes.append(clean_line) 
            
            current_line = [box]

    # จัดเรียงบรรทัดสุดท้ายที่เหลืออยู่
    if current_line:
        current_line.sort(key=lambda b: b[0])
        clean_line = [b[:4] for b in current_line]
        sorted_boxes.append(clean_line)
        
    return sorted_boxes

# --- ฟังก์ชันโหลด AI ---
@st.cache_resource
def load_engine():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ✅ ชี้ไปที่ไฟล์ PyTorch (.pth)
        MODEL_PATH = os.path.join(current_dir, 'models', 'model_read_numberthaiV1_pytorch.pth')
        
        if not os.path.exists(MODEL_PATH):
            return None, f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}"
            
        model = ThaiDigitNet()
        # โหลดโมเดล
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        return model, "OK"
    except Exception as e:
        return None, str(e)

with st.spinner('กำลังปลุก AI...'):
    model, status = load_engine()

if status != "OK":
    st.error(f"โหลด AI ไม่สำเร็จ: {status}")
    st.stop()
else:
    st.success("✅ AI (PyTorch) พร้อมใช้งานแล้ว!")


# =========================================================================
# === 1. ส่วนอัปโหลดรูปภาพ (ต้องอยู่ด้านบนเพื่อกำหนดค่า uploaded_file) ===
# =========================================================================

st.markdown("---")
st.header("🖼️ อัปโหลดรูปภาพเพื่ออ่านหลายตัวเลข")
# ✅ กำหนดค่าตัวแปร uploaded_file ที่นี่ เพื่อป้องกัน NameError
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"]) 


# =========================================================================
# === 2. ส่วนวาดภาพ (Drawable Canvas) และการทำนายผลทันที (CLEANED) ===
# =========================================================================
st.markdown("---")
st.header("✍️ วาดตัวเลขไทยเพื่อทำนายผลทันที")

# สร้าง Drawing Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)", # ไม่ใช้สีเติม
    stroke_width=15, # ความหนาของเส้น
    stroke_color="#000000", # สีเส้น: ดำ (เพื่อให้ผู้ใช้วาดได้สะดวก)
    background_color="#FFFFFF", # พื้นหลัง: ขาว (เพื่อให้ผู้ใช้วาดได้สะดวก)
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# 💡 ทำนายผลเมื่อมีการวาดเกิดขึ้น
if canvas_result.image_data is not None:
    drawn_img_array = canvas_result.image_data.astype(np.uint8)
    gray_drawn_img = cv2.cvtColor(drawn_img_array, cv2.COLOR_RGBA2GRAY)

    # 🛑 บรรทัดที่ถูกแก้ไข: กลับสีจาก ดำบนขาว เป็น ขาวบนดำ 
    # เพื่อให้สอดคล้องกับโมเดล (ตัวเลขสีขาว, พื้นหลังสีดำ)
    gray_drawn_img = cv2.bitwise_not(gray_drawn_img)
    
    # ตัดส่วนที่มีตัวเลขเท่านั้น (ตอนนี้ตัวเลขเป็นสีขาวแล้ว)
    _, thresh_canvas = cv2.threshold(gray_drawn_img, 20, 255, cv2.THRESH_BINARY)
    contours_canvas, _ = cv2.findContours(thresh_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_canvas:
        # หา Bounding Box ที่ใหญ่ที่สุด
        x, y, w, h = cv2.boundingRect(np.concatenate(contours_canvas))
        
        # ขยายกรอบ Bounding Box เล็กน้อย
        pad = 5
        x_min = max(0, x - pad)
        y_min = max(0, y - pad)
        x_max = min(gray_drawn_img.shape[1], x + w + pad)
        y_max = min(gray_drawn_img.shape[0], y + h + pad)
        
        roi_drawn = gray_drawn_img[y_min:y_max, x_min:x_max]
        
        if roi_drawn.size > 0:
            # Preprocess และ Predict
            display_img_canvas, tensor_img_canvas = preprocess_for_model(roi_drawn, target_size=(96, 96))
            
            with torch.no_grad():
                outputs = model(tensor_img_canvas)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_class = probs.topk(1, dim=1)
                
                predicted_index = top_class.item()
                char_out = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙'][predicted_index]
                prob = top_p.item() * 100
                
            st.markdown(f"**🤖 AI ทำนายว่าคือ:**")
            st.markdown(f"<h1 style='text-align: center; color: #E74C3C; font-size:80px;'>{char_out}</h1>", unsafe_allow_html=True)
            st.caption(f"ความมั่นใจ: {prob:.2f}%")
            
            # แสดงภาพที่ถูกกลับสีและเตรียมเข้าโมเดล (พื้นหลังดำ ตัวเลขขาว)
            st.image(display_img_canvas, caption="ภาพที่เข้าโมเดล (ขาวบนดำ)", width=96, clamp=True) 
            
        else:
            st.caption("กรุณาวาดตัวเลขให้ชัดเจนขึ้น")


# =========================================================================
# === 3. ส่วนประมวลผลไฟล์ที่อัปโหลด (ใช้ uploaded_file ที่ถูกกำหนดค่าแล้ว) ===
# =========================================================================

if uploaded_file is not None:
    # ใช้งาน uploaded_file ได้อย่างปลอดภัย
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปต้นฉบับ', width=300)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        # ค่าเริ่มต้นเป็น True เพราะรูปที่อัปโหลดมาส่วนใหญ่มักจะเป็นตัวอักษรดำบนพื้นหลังขาว
        use_invert = st.checkbox("กลับสีภาพ (ใช้เมื่อตัวดำ พื้นขาว)", value=True, key="inv_upload") 
    with col_opt2:
        thinning_level = st.slider("ระดับลดความหนา (Erosion)", 0, 5, 1, help="ยิ่งเลขมาก เส้นยิ่งบาง", key="thin_upload")
    
    if st.button('ทำนายผลเดี๋ยวนี้', type="primary"):
        if model is None:
            st.error("Model not loaded")
        else:
            with st.spinner('กำลังวิเคราะห์...'):
                try:
                    img_array = np.array(image.convert('RGB'))
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                    # ใช้ THRESH_BINARY_INV เมื่อต้องการกลับสี (ดำบนขาว -> ขาวบนดำ)
                    thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if use_invert else cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    _, binary_img = cv2.threshold(gray, 0, 255, thresh_type)

                    # Erodes (ลดความหนาเส้น) ซึ่งใช้ได้ดีกับตัวเลขสีขาวบนพื้นหลังดำ
                    if thinning_level > 0:
                        kernel = np.ones((2, 2), np.uint8)
                        binary_img = cv2.erode(binary_img, kernel, iterations=thinning_level)

                    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        boundingBoxes = [cv2.boundingRect(c) for c in contours]
                        sorted_lines = sort_bounding_boxes(boundingBoxes)
                    else:
                        st.warning("ไม่พบตัวเลขในภาพ")
                        st.stop()

                    full_result = ""
                    confidence_list = []
                    debug_img = img_array.copy()
                    labels = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']

                    st.write("--- ภาพที่ AI เห็น ---")
                    
                    display_boxes_count = 0
                    cols = None
                    
                    for line_index, line_boxes in enumerate(sorted_lines):
                        if line_index > 0:
                            full_result += '\n'
                            
                        for box in line_boxes:
                            x, y, w, h = box

                            if w < 10 or h < 10: continue

                            # การจัดการตัวเลขที่ติดกัน
                            aspect_ratio = w / float(h)
                            if aspect_ratio > 1.2:
                                n_chars = max(1, int(round(w / h)))
                                step_w = w // n_chars
                            else:
                                n_chars = 1
                                step_w = w

                            for k in range(n_chars):
                                curr_x = x + (k * step_w)
                                curr_w = step_w
                                
                                cv2.rectangle(debug_img, (curr_x, y), (curr_x + curr_w, y + h), (0, 255, 0), 2)
                                
                                roi = binary_img[y:y+h, curr_x:curr_x+curr_w]
                                if roi.size == 0: continue

                                display_img, tensor_img = preprocess_for_model(roi, target_size=(96, 96))

                                # Predict
                                with torch.no_grad():
                                    outputs = model(tensor_img)
                                    probs = torch.nn.functional.softmax(outputs, dim=1)
                                    top_p, top_class = probs.topk(1, dim=1)
                                    
                                    predicted_index = top_class.item()
                                    prob = top_p.item() * 100
                                    char_out = labels[predicted_index]

                                full_result += char_out
                                confidence_list.append(prob)

                                # แสดงตัวอย่างภาพที่ AI เห็น
                                if display_boxes_count < 5:
                                    if display_boxes_count == 0:
                                        cols = st.columns(5)
                                    with cols[display_boxes_count % 5]:
                                        st.image(display_img, caption=f"{char_out}", width=60, clamp=True)
                                    display_boxes_count += 1

                    st.write("---")
                    st.image(debug_img, caption='ตำแหน่งที่ AI เจอ', width=300)
                    
                    if full_result:
                        avg_conf = sum(confidence_list) / len(confidence_list)
                        st.success("อ่านข้อความได้ว่า:")
                        st.markdown(f"<h1 style='text-align: center; color: #2E86C1; font-size:40px;'><pre style='font-family:inherit; white-space: pre-wrap;'>{full_result}</pre></h1>", unsafe_allow_html=True)
                        st.caption(f"ความมั่นใจเฉลี่ย: {avg_conf:.2f}%")
                    else:
                        st.warning("หาตัวเลขไม่เจอ")
                        
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")