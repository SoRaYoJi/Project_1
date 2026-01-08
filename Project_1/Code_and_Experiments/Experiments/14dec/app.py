import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas 

from model import ThaiDigitNet
from preprocess import preprocess_digit
from ocr_utils import sort_boxes


# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Thai Digit OCR",
    page_icon="🇹🇭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Thai Digit OCR")
st.caption("Production-ready handwritten Thai digit recognition")


# ===============================
# Load Model (SAFE PATH)
# ===============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir, "models", "model_read_numberthaiV1_pytorch.pth"
    )

    if not os.path.exists(model_path):
        st.error(f"❌ ไม่พบไฟล์โมเดล:\n{model_path}")
        st.stop()

    model = ThaiDigitNet()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model


model = load_model()
LABELS = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']


# ===============================
# Section 1: Draw Digit
# ===============================
st.divider()
st.subheader("✍️ วาดตัวเลขไทย")

canvas = st_canvas(
    height=300,
    width=300,
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    drawing_mode="freedraw",
    key="draw_canvas",
)

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    gray = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        roi = gray[y:y + h, x:x + w]

        display_img, tensor_img = preprocess_digit(roi)

        if tensor_img is not None:
            with torch.no_grad():
                probs = torch.softmax(model(tensor_img), dim=1)
                idx = probs.argmax(dim=1).item()
                conf = probs.max().item() * 100

            st.markdown(
                f"<h1 style='text-align:center;font-size:72px'>{LABELS[idx]}</h1>",
                unsafe_allow_html=True,
            )
            st.caption(f"ความมั่นใจ: {conf:.2f}%")
            st.image(display_img, width=96, caption="Input to model")


# ===============================
# Section 2: Upload Image OCR
# ===============================
st.divider()
st.subheader("🖼️ อัปโหลดรูปภาพ")

uploaded = st.file_uploader(
    "รองรับ JPG / PNG",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Original Image", width=300)

    col1, col2 = st.columns(2)
    with col1:
        invert = st.checkbox("กลับสี (ดำบนขาว)", value=True)
    with col2:
        erosion = st.slider("ลดความหนาเส้น", 0, 5, 1)

    if st.button("🔍 วิเคราะห์", type="primary"):
        with st.spinner("กำลังประมวลผล..."):
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            thresh_type = (
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            ) | cv2.THRESH_OTSU

            _, binary = cv2.threshold(gray, 0, 255, thresh_type)

            if erosion > 0:
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.erode(binary, kernel, iterations=erosion)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                st.warning("ไม่พบตัวเลขในภาพ")
                st.stop()

            boxes = [cv2.boundingRect(c) for c in contours]
            lines = sort_boxes(boxes)

            result = ""
            confidences = []

            for line in lines:
                for (x, y, w, h) in line:
                    if w < 10 or h < 10:
                        continue

                    roi = binary[y:y + h, x:x + w]
                    _, tensor = preprocess_digit(roi)

                    if tensor is None:
                        continue

                    with torch.no_grad():
                        probs = torch.softmax(model(tensor), dim=1)
                        idx = probs.argmax(dim=1).item()
                        conf = probs.max().item() * 100

                    result += LABELS[idx]
                    confidences.append(conf)

                result += "\n"

            st.success("ผลลัพธ์ที่อ่านได้")
            st.markdown(
                f"<pre style='font-size:28px'>{result}</pre>",
                unsafe_allow_html=True,
            )

            if confidences:
                st.caption(
                    f"ความมั่นใจเฉลี่ย: {sum(confidences)/len(confidences):.2f}%"
                )
