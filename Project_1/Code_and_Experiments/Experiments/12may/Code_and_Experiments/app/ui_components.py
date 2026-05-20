from __future__ import annotations

import pandas as pd
import streamlit as st


MODE_OPTIONS = {
    "document_page": "เอกสารทั้งหน้า",
    "thai_api": "OCR ตัวอักษรไทยผ่าน API",
    "combined": "OCR แบบรวม",
    "digit": "OCR ตัวเลขไทย",
}

MODE_DESCRIPTIONS = {
    "document_page": "เหมาะกับเอกสารเต็มหน้า หนังสือราชการ เอกสารสแกน หรือภาพที่ต้องการอ่านข้อความทั้งหน้าเป็นหลัก",
    "thai_api": "ใช้ API อ่านข้อความไทยจากภาพทั่วไป โดยไม่พยายามแทนส่วนตัวเลขด้วยโมเดลตัวเลข",
    "combined": "ใช้ API อ่านข้อความก่อน แล้วแทนเฉพาะบล็อกเลขล้วนที่โมเดลตัวเลขอ่านได้อย่างน่าเชื่อถือ",
    "digit": "ใช้โมเดลตัวเลขไทยล้วน เหมาะกับภาพที่มีตัวเลขไทยเด่น ๆ หรือภาพที่ครอปเฉพาะเลข",
}


def render_sidebar() -> tuple[str, bool, int]:
    st.sidebar.header("ตั้งค่าการทำงาน")
    mode = st.sidebar.radio(
        "เลือกโหมด OCR",
        options=list(MODE_OPTIONS.keys()),
        format_func=lambda key: MODE_OPTIONS[key],
        help="เลือกโหมดให้ตรงกับลักษณะภาพ เพื่อให้ได้ผลแม่นยำขึ้น",
    )
    st.sidebar.caption(MODE_DESCRIPTIONS[mode])
    st.sidebar.markdown("---")
    st.sidebar.caption("ตัวเลือกด้านล่างใช้กับโหมด OCR ตัวเลขไทยเป็นหลัก")
    invert = st.sidebar.checkbox(
        "กลับสีภาพสำหรับ OCR ตัวเลข",
        value=True,
        help="เหมาะกับภาพที่ตัวเลขเข้มบนพื้นสว่าง",
    )
    thinning_level = st.sidebar.slider(
        "ระดับลดความหนาเส้น",
        min_value=0,
        max_value=5,
        value=1,
    )
    return mode, invert, thinning_level


def render_header() -> None:
    st.title("Thai OCR Workspace")
    st.caption("รวม OCR ตัวเลขไทยจากโมเดลเดิม 14dec และ OCR ข้อความไทยผ่าน API จาก 2may ไว้ในหน้าเดียว")


def render_detection_table(records: list[dict]) -> None:
    if not records:
        st.info("ไม่มีข้อมูลสำหรับแสดงในตาราง")
        return
    dataframe = pd.DataFrame(records)
    st.dataframe(dataframe, width="stretch")
