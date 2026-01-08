import os
import streamlit as st
from PIL import Image
from datetime import datetime
import pandas as pd
import torch
from streamlit_drawable_canvas import st_canvas

from model import ThaiDigitNet
from preprocess import preprocess_image
from inference import predict
from ui_components import header, footer

st.set_page_config(
    page_title="Thai Digit OCR – Professional",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "models", "model_read_numberthaiV1_pytorch.pth")
    model = ThaiDigitNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------ SESSION ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ UI ------------------
header()
tabs = st.tabs([
    "🧠 OCR", 
    "🗂 History",
    "🧮 Multi-Digit",
    "📦 Batch OCR",
    "📊 Dashboard",
    "🗂 History"])

# =================================================
# 🧠 TAB 1 : OCR SYSTEM
# =================================================
with tabs[0]:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("✍️ Input")
        mode = st.radio("Input mode", ["Draw", "Upload"], horizontal=True)

        invert = st.checkbox("Invert")
        threshold = st.checkbox("Threshold", value=True)

        img = None
        input_type = None

        if mode == "Draw":
            canvas = st_canvas(
                fill_color="white",
                stroke_width=10,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas"
            )
            if canvas.image_data is not None:
                img = Image.fromarray(canvas.image_data.astype("uint8"))
                input_type = "draw"

        else:
            file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
            if file:
                img = Image.open(file)
                input_type = "upload"

    with col2:
        st.subheader("📊 Result")
        if img is not None:
            proc = preprocess_image(img, invert, threshold)
            pred, conf, top3, probs = predict(model, proc)

            st.metric("Prediction", pred)
            st.metric("Confidence", f"{conf*100:.2f}%")

            chart_df = pd.DataFrame({
                "Digit": list(range(10)),
                "Confidence": probs
            }).set_index("Digit")

            st.bar_chart(chart_df)

            # save history
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_type,
                "prediction": int(pred),
                "confidence": round(float(conf * 100), 2)
            })

# =================================================
# 🗂 TAB 2 : HISTORY SYSTEM
# =================================================
with tabs[1]:
    st.subheader("📜 Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            hist_df,
            use_container_width=True
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                "⬇️ Export CSV",
                hist_df.to_csv(index=False),
                "thai_digit_ocr_history.csv"
            )

        with col_b:
            if st.button("🗑 Clear History"):
                st.session_state.history.clear()
                st.experimental_rerun()
    else:
        st.info("No history yet. Start predicting digits first 👆")

footer()

