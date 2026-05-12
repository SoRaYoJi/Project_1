from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = CURRENT_DIR.parent
PROJECT_ROOT = PACKAGE_PARENT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code_and_Experiments.app.document_export import build_docx_bytes, build_json_bytes, build_pdf_bytes
from Code_and_Experiments.app.image_utils import (
    draw_detection_boxes,
    image_to_download_bytes,
    open_image,
    pil_to_numpy,
    render_positioned_text_image,
    render_text_overlay_image,
)
from Code_and_Experiments.app.ocr_digit import load_digit_model, recognize_digit_text
from Code_and_Experiments.app.ocr_thai_api import ThaiApiOcrResult, call_thai_ocr_api, run_combined_ocr
from Code_and_Experiments.app.ui_components import (
    MODE_DESCRIPTIONS,
    MODE_OPTIONS,
    render_detection_table,
    render_header,
    render_sidebar,
)


st.set_page_config(page_title="Thai OCR Unified App", page_icon="🇹🇭", layout="wide")


@st.cache_resource
def get_digit_model():
    return load_digit_model()


def ensure_state() -> None:
    if "api_usage_totals" not in st.session_state:
        st.session_state.api_usage_totals = {
            "request_count": 0,
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
        }
    if "last_render_payload" not in st.session_state:
        st.session_state.last_render_payload = None


def add_usage(result: ThaiApiOcrResult | None) -> None:
    ensure_state()
    if result is None:
        return
    usage = result.usage
    totals = st.session_state.api_usage_totals
    totals["request_count"] += usage.request_count or 0
    totals["prompt_token_count"] += usage.prompt_token_count or 0
    totals["candidates_token_count"] += usage.candidates_token_count or 0
    totals["thoughts_token_count"] += usage.thoughts_token_count or 0
    totals["total_token_count"] += usage.total_token_count or 0


def render_api_usage(result: ThaiApiOcrResult | None) -> None:
    st.subheader("สถิติ API")
    if result is None:
        st.info("โหมดนี้ไม่ได้ใช้ API")
        return

    totals = st.session_state.api_usage_totals
    usage = result.usage
    col1, col2, col3 = st.columns(3)
    col1.metric("API calls รอบนี้", usage.request_count)
    col2.metric("Prompt tokens รอบนี้", usage.prompt_token_count or 0)
    col3.metric("Total tokens รอบนี้", usage.total_token_count or 0)

    col4, col5, col6 = st.columns(3)
    col4.metric("API calls สะสม", totals["request_count"])
    col5.metric("Prompt tokens สะสม", totals["prompt_token_count"])
    col6.metric("Total tokens สะสม", totals["total_token_count"])

    st.caption(
        f"Model: {result.model_name} | Candidate tokens: {usage.candidates_token_count or 0} | "
        f"Thought tokens: {usage.thoughts_token_count or 0}"
    )


def build_reconstructed_image(image: Image.Image, result: ThaiApiOcrResult) -> Image.Image:
    boxes = [
        type(item.box)(
            ymin=item.box.ymin,
            xmin=item.box.xmin,
            ymax=item.box.ymax,
            xmax=item.box.xmax,
            text=item.resolved_text,
        )
        for item in result.detections
    ]
    return render_positioned_text_image(image.size, boxes, draw_boxes=False)


def build_overlay_image(image: Image.Image, result: ThaiApiOcrResult) -> Image.Image:
    boxes = [
        type(item.box)(
            ymin=item.box.ymin,
            xmin=item.box.xmin,
            ymax=item.box.ymax,
            xmax=item.box.xmax,
            text=item.resolved_text,
        )
        for item in result.detections
    ]
    return render_text_overlay_image(image, boxes, draw_boxes=True)


def save_render_payload(mode: str, image: Image.Image, result: ThaiApiOcrResult, note: str, digit_results=None) -> None:
    st.session_state.last_render_payload = {
        "mode": mode,
        "image": image.copy(),
        "result": result,
        "note": note,
        "digit_results": digit_results or [],
    }


def render_download_section(image: Image.Image, result: ThaiApiOcrResult) -> None:
    overlay_image = build_overlay_image(image, result)
    text_bytes = result.text.encode("utf-8")
    json_bytes = build_json_bytes(result)
    image_bytes = image_to_download_bytes(overlay_image, fmt="PNG")
    docx_bytes = build_docx_bytes(result.text, title="Thai OCR Result")
    pdf_bytes = build_pdf_bytes(result.text, title="Thai OCR Result")

    st.markdown("**ดาวน์โหลดผลลัพธ์**")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row1_col1.download_button("ดาวน์โหลดข้อความ .txt", data=text_bytes, file_name="ocr_result.txt", mime="text/plain", width="stretch")
    row1_col2.download_button("ดาวน์โหลดตำแหน่ง .json", data=json_bytes, file_name="ocr_result.json", mime="application/json", width="stretch")
    row1_col3.download_button("ดาวน์โหลดภาพผลลัพธ์ .png", data=image_bytes, file_name="ocr_overlay_result.png", mime="image/png", width="stretch")

    row2_col1, row2_col2 = st.columns(2)
    row2_col1.download_button(
        "ดาวน์โหลดเอกสาร .docx",
        data=docx_bytes,
        file_name="ocr_result.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        width="stretch",
    )
    row2_col2.download_button(
        "ดาวน์โหลดเอกสาร .pdf",
        data=pdf_bytes,
        file_name="ocr_result.pdf",
        mime="application/pdf",
        width="stretch",
    )


def render_comparison_section(image: Image.Image, result: ThaiApiOcrResult, title: str) -> None:
    positioned = build_reconstructed_image(image, result)
    overlay_image = build_overlay_image(image, result)
    debug_overlay = draw_detection_boxes(
        pil_to_numpy(image),
        [item.box for item in result.detections],
        show_text=False,
        thickness=1,
    )

    st.subheader("เปรียบเทียบผลลัพธ์")
    compare_tab, canvas_tab, debug_tab = st.tabs(["เทียบกับต้นฉบับ", "จัดวางบนหน้าขาว", "ดูกรอบ OCR"])
    with compare_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("ต้นฉบับ")
            st.image(image, width="stretch")
        with col2:
            st.caption("ผลลัพธ์วางทับตามตำแหน่งเดิม")
            st.image(overlay_image, width="stretch")
    with canvas_tab:
        st.caption(title)
        st.image(positioned, width="stretch")
    with debug_tab:
        st.caption("ใช้สำหรับตรวจว่ากรอบ OCR ครอบตรงกับข้อความจริงหรือไม่")
        st.image(debug_overlay, width="stretch")

    render_download_section(image, result)


def render_saved_payload() -> None:
    payload = st.session_state.last_render_payload
    if not payload:
        return

    result: ThaiApiOcrResult = payload["result"]
    image: Image.Image = payload["image"]

    st.caption(payload["note"])
    text_tab, visual_tab, table_tab, api_tab = st.tabs(["ข้อความ", "ภาพเทียบ", "ตารางตรวจจับ", "การใช้งาน API"])
    with text_tab:
        st.text_area("ข้อความที่อ่านได้", value=result.text, height=300)
    with visual_tab:
        render_comparison_section(image, result, "ผลลัพธ์จัดวางตามตำแหน่งเดิม")
    with table_tab:
        render_detection_table(
            [
                {
                    "source": detection.source,
                    "text": detection.resolved_text,
                    "box": [detection.box.ymin, detection.box.xmin, detection.box.ymax, detection.box.xmax],
                }
                for detection in result.detections
            ]
        )
        if payload["mode"] == "combined":
            digit_results = payload.get("digit_results", [])
            if digit_results:
                st.subheader("ผลจากโมเดลตัวเลขไทย")
                for index, digit_result in enumerate(digit_results, start=1):
                    st.markdown(f"**ส่วนตัวเลข #{index}**")
                    st.write(f"ข้อความ: `{digit_result.text}` | ความมั่นใจเฉลี่ย: {digit_result.average_confidence:.2f}%")
                    st.image(digit_result.debug_image, caption=f"Digit region #{index}", width="stretch")
            else:
                st.info("รอบนี้ไม่มีส่วนไหนที่ผ่านเกณฑ์ให้แทนค่าด้วยโมเดลตัวเลขไทย")
    with api_tab:
        render_api_usage(result)


def render_mode_help(mode: str) -> None:
    st.info(MODE_DESCRIPTIONS[mode])
    if mode == "digit":
        st.caption("โหมดนี้ไม่เรียก API และจึงไม่มี token usage")
    elif mode == "document_page":
        st.caption("โหมดนี้เหมาะที่สุดสำหรับเอกสารเต็มหน้า และจะอ่านทั้งหน้าเป็นข้อความต่อเนื่อง")
    elif mode == "thai_api":
        st.caption("โหมดนี้ใช้ API อ่านข้อความไทยตรง ๆ โดยไม่พยายามแทนส่วนตัวเลขด้วยโมเดลเลข")
    else:
        st.caption("โหมดนี้จะใช้ API เป็นแกนหลัก และใช้โมเดลเลขเฉพาะจุดที่เป็นเลขล้วนและน่าเชื่อถือ")


def main() -> None:
    ensure_state()
    render_header()
    mode, invert, thinning_level = render_sidebar()

    uploaded_file = st.file_uploader(
        "อัปโหลดรูปภาพ",
        type=["png", "jpg", "jpeg"],
        help="รองรับไฟล์ .png .jpg .jpeg",
    )
    if uploaded_file is None:
        st.info("อัปโหลดรูปก่อน แล้วกดปุ่มประมวลผล OCR")
        return

    try:
        image = open_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    preview_col, action_col = st.columns([1.15, 1])
    with preview_col:
        st.subheader("ภาพต้นฉบับ")
        st.image(image, caption="ภาพที่อัปโหลด", width="stretch")

    try:
        digit_model, model_path = get_digit_model()
    except Exception as exc:
        digit_model = None
        model_path = None
        if mode in {"digit", "combined"}:
            with action_col:
                st.error(f"โหลดโมเดลตัวเลขไทยไม่สำเร็จ: {exc}")
            return

    with action_col:
        st.subheader("เริ่มประมวลผล")
        st.caption(f"โหมดปัจจุบัน: {MODE_OPTIONS[mode]}")
        render_mode_help(mode)
        if model_path is not None:
            st.caption(f"Digit model: `{model_path}`")

        process_clicked = st.button("ประมวลผล OCR", type="primary", width="stretch")
        if process_clicked:
            if mode == "digit":
                run_digit_mode(image, digit_model, invert, thinning_level)
            elif mode in {"thai_api", "document_page"}:
                run_api_mode(image, mode)
            else:
                run_combined_mode(image, digit_model)
        elif mode != "digit" and st.session_state.last_render_payload is not None:
            render_saved_payload()
        else:
            st.info("กดปุ่มเพื่อเริ่มประมวลผล")


def run_digit_mode(image: Image.Image, digit_model, invert: bool, thinning_level: int) -> None:
    st.session_state.last_render_payload = None
    with st.spinner("กำลังอ่านตัวเลขไทยด้วยโมเดลตัวเลขล้วน..."):
        try:
            result = recognize_digit_text(
                model=digit_model,
                rgb_image=pil_to_numpy(image),
                invert=invert,
                thinning_level=thinning_level,
            )
        except Exception as exc:
            st.error(f"OCR ตัวเลขไทยล้มเหลว: {exc}")
            return

    st.text_area("ข้อความที่อ่านได้", value=result.text, height=140)
    metric_col, hint_col = st.columns([0.4, 0.6])
    with metric_col:
        st.metric("ความมั่นใจเฉลี่ย", f"{result.average_confidence:.2f}%")
    with hint_col:
        st.caption("โหมดนี้ไม่ได้ใช้ API จึงไม่มี token usage")
    st.image(result.debug_image, caption="ตำแหน่งตัวเลขที่ตรวจพบ", width="stretch")
    render_detection_table(
        [
            {
                "digit": prediction.digit,
                "confidence_percent": round(prediction.confidence, 2),
                "box": prediction.box,
            }
            for prediction in result.predictions
        ]
    )
    render_api_usage(None)


def run_api_mode(image: Image.Image, mode: str) -> None:
    with st.spinner("กำลังเรียก Thai OCR API..."):
        try:
            result = call_thai_ocr_api(image)
            add_usage(result)
        except Exception as exc:
            st.error(f"OCR ผ่าน API ล้มเหลว: {exc}")
            return

    if mode == "document_page":
        note = "โหมดเอกสารทั้งหน้าใช้ API อ่านข้อความทั้งหน้าเป็นหลัก พร้อม export เป็น TXT, JSON, PNG, DOCX และ PDF"
    else:
        note = "ผลลัพธ์ OCR ผ่าน API พร้อมมุมมองเปรียบเทียบและดาวน์โหลด"

    save_render_payload(
        mode=mode,
        image=image,
        result=result,
        note=note,
    )
    render_saved_payload()


def run_combined_mode(image: Image.Image, digit_model) -> None:
    with st.spinner("กำลังประมวลผล OCR แบบรวม..."):
        try:
            result, digit_results = run_combined_ocr(image, digit_model)
            add_usage(result)
        except Exception as exc:
            st.error(f"OCR แบบรวมล้มเหลว: {exc}")
            return

    save_render_payload(
        mode="combined",
        image=image,
        result=result,
        note="โหมดรวมใช้ API เป็นแกนหลัก และจะยอมแทนค่าด้วยโมเดลตัวเลขไทยเฉพาะบล็อกเลขล้วนที่ผ่านเกณฑ์ความน่าเชื่อถือ",
        digit_results=digit_results,
    )
    render_saved_payload()


if __name__ == "__main__":
    main()
