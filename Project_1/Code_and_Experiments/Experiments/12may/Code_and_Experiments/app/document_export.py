from __future__ import annotations

import json
import textwrap
from io import BytesIO

from docx import Document
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .image_utils import DetectionBox, get_thai_font_path
from .ocr_thai_api import ThaiApiOcrResult


def build_json_bytes(result: ThaiApiOcrResult) -> bytes:
    return json.dumps(
        [
            {
                "text": item.resolved_text,
                "raw_text": item.raw_text,
                "source": item.source,
                "box_2d": [item.box.ymin, item.box.xmin, item.box.ymax, item.box.xmax],
            }
            for item in result.detections
        ],
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def build_docx_bytes(result_text: str, title: str = "OCR Result") -> bytes:
    document = Document()
    document.core_properties.title = title
    style = document.styles["Normal"]
    style.font.name = "Leelawadee UI"
    style.font.size = Pt(12)

    for line in result_text.splitlines():
        document.add_paragraph(line if line.strip() else "")

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _register_pdf_font() -> str:
    font_name = "Helvetica"
    font_path = get_thai_font_path()
    if font_path is None:
        return font_name

    custom_font_name = "ThaiUnicodeFont"
    registered = pdfmetrics.getRegisteredFontNames()
    if custom_font_name not in registered:
        pdfmetrics.registerFont(TTFont(custom_font_name, str(font_path)))
    return custom_font_name


def build_pdf_bytes(result_text: str, title: str = "OCR Result") -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin_x = 40
    margin_y = 50
    current_y = height - margin_y
    font_name = _register_pdf_font()
    title_size = 14
    body_size = 12

    pdf.setTitle(title)
    pdf.setFont(font_name, title_size)
    pdf.drawString(margin_x, current_y, title)
    current_y -= 28
    pdf.setFont(font_name, body_size)

    for raw_line in result_text.splitlines():
        wrapped_lines = textwrap.wrap(raw_line, width=82) if raw_line.strip() else [""]
        for line in wrapped_lines:
            if current_y <= margin_y:
                pdf.showPage()
                pdf.setFont(font_name, body_size)
                current_y = height - margin_y
            pdf.drawString(margin_x, current_y, line)
            current_y -= 18
        if raw_line.strip():
            current_y -= 2

    pdf.save()
    return buffer.getvalue()


def build_position_json_records(result: ThaiApiOcrResult) -> list[dict]:
    return [
        {
            "text": item.resolved_text,
            "raw_text": item.raw_text,
            "source": item.source,
            "box": _box_to_list(item.box),
        }
        for item in result.detections
    ]


def _box_to_list(box: DetectionBox) -> list[int]:
    return [box.ymin, box.xmin, box.ymax, box.xmax]
