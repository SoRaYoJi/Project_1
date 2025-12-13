import os
import streamlit as st

# --- 🔴 ยาแก้ค้าง (ต้องอยู่บรรทัดบนสุด) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="ทายเลขไทย AI", page_icon="🇹🇭")

st.title("🇹🇭 ระบบทายลายมือเลขไทย")
st.info("สถานะ: 🟢 เว็บไซต์โหลดเสร็จแล้ว (กำลังรอโหลดสมอง AI...)")

# --- ฟังก์ชันโหลด AI ---
@st.cache_resource
def load_engine():
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # ชื่อไฟล์ต้องตรงกับใน GitHub เป๊ะๆ
        MODEL_PATH = 'thai_digit_model_64x64_Thickness_V2.keras'
        
        if not os.path.exists(MODEL_PATH):
            return None, "ไม่พบไฟล์โมเดล (กรุณาเช็คชื่อไฟล์ใน GitHub)"
            
        model = load_model(MODEL_PATH, compile=False)
        return model, "OK"
    except Exception as e:
        return None, str(e)

# --- โหลดโมเดล ---
with st.spinner('กำลังปลุก AI... (รอสักครู่)'):
    model, status = load_engine()

if status != "OK":
    st.error(f"โหลด AI ไม่สำเร็จ: {status}")
    st.stop()
else:
    st.success("AI พร้อมใช้งานแล้ว!")

# --- ส่วนอัปโหลดรูป ---
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='รูปต้นฉบับ', width=200)

    use_invert = st.checkbox("กลับสีภาพ (ใช้เมื่อตัวดำ พื้นขาว)", value=False)
    
    if st.button('ทำนายผลเดี๋ยวนี้', type="primary"):
        if model is None:
            st.error("Model not loaded")
        else:
            with st.spinner('กำลังวิเคราะห์...'):
                try:
                    # Preprocess
                    img_array = np.array(image.convert('RGB')) 
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    thresh_type = cv2.THRESH_BINARY_INV if use_invert else cv2.THRESH_BINARY
                    _, binary_img = cv2.threshold(gray, 128, 255, thresh_type)

                    # Resize 64x64
                    resized = cv2.resize(binary_img, (64, 64), interpolation=cv2.INTER_AREA)
                    
                    with col2:
                        st.image(resized, caption='ภาพที่ AI เห็น', width=200, clamp=True)

                    # Normalize & Predict
                    img_norm = resized.astype("float32") / 255.0
                    img_final = np.expand_dims(np.expand_dims(img_norm, axis=-1), axis=0)

                    prediction = model.predict(img_final)
                    predicted_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    
                    labels = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']
                    result_char = labels[predicted_index]

                    # st.balloons()  <-- ลบออกให้แล้วครับ
                    st.markdown(f"# ผลลัพธ์: <span style='color:green; font-size:40px'>{result_char}</span>", unsafe_allow_html=True)
                    st.write(f"ความมั่นใจ: {confidence:.2f}%")
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")