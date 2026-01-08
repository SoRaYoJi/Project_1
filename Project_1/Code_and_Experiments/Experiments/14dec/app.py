import os, cv2, torch, numpy as np, streamlit as st
from PIL import Image
from model import ThaiDigitNet
from preprocess import preprocess_digit
from ocr_utils import sort_boxes
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Thai Digit OCR", layout="centered")
st.title("Thai Digit OCR – Professional Edition")


@st.cache_resource
def load_model():
    m = ThaiDigitNet()
    m.load_state_dict(torch.load('models/model_read_numberthaiV1_pytorch.pth', map_location='cpu'))
    m.eval(); return m


model = load_model()
labels = ['๐','๑','๒','๓','๔','๕','๖','๗','๘','๙']


st.header("✍️ Draw Digit")
canvas = st_canvas(height=300, width=300, stroke_width=15,
        stroke_color="#000", background_color="#fff")
if canvas.image_data is not None:
    img = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img = cv2.bitwise_not(img)
    _, th = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    x,y,w,h = cv2.boundingRect(np.concatenate(cnts))
    roi = img[y:y+h, x:x+w]
    d,t = preprocess_digit(roi)
    with torch.no_grad():
        p = torch.softmax(model(t),1)
        i = p.argmax(1).item()
    st.markdown(f"<h1 style='text-align:center'>{labels[i]}</h1>", unsafe_allow_html=True)
