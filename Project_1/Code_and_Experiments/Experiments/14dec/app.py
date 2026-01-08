import os, streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import torch

from model import ThaiDigitNet
from preprocess import preprocess_image
from inference import predict
from ui_components import header, footer

st.set_page_config(page_title="Thai Digit OCR", layout="wide")

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "models", "model_read_numberthaiV1_pytorch.pth")
    model = ThaiDigitNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

header()

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("✍️ Input")
    mode = st.radio("Input mode", ["Draw", "Upload"])

    invert = st.checkbox("Invert")
    threshold = st.checkbox("Threshold", value=True)

    img = None
    if mode == "Draw":
        canvas = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw"
        )
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8"))

    else:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if file:
            img = Image.open(file)

with col2:
    st.subheader("📊 Result")
    if img is not None:
        proc = preprocess_image(img, invert, threshold)
        pred, conf, top3, probs = predict(model, proc)

        st.metric("Prediction", pred)
        st.metric("Confidence", f"{conf*100:.2f}%")

        df = pd.DataFrame({
            "Digit": list(range(10)),
            "Confidence": probs
        })
        st.bar_chart(df.set_index("Digit"))

        st.session_state.history.append({
            "prediction": pred,
            "confidence": float(conf)
        })

st.subheader("🗂 History")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
    st.download_button(
        "⬇️ Export CSV",
        hist_df.to_csv(index=False),
        "history.csv"
    )

footer()
