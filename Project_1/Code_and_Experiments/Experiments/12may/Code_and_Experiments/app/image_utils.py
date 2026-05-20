from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DetectionBox:
    ymin: int
    xmin: int
    ymax: int
    xmax: int
    text: str = ""

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def y_bin(self) -> int:
        return self.ymin


def open_image(uploaded_file) -> Image.Image:
    if uploaded_file is None:
        raise ValueError("ไม่พบไฟล์ภาพ")
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        raise ValueError("ไฟล์ภาพไม่ถูกต้องหรือไม่สามารถเปิดได้") from exc


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def crop_with_padding(
    image: np.ndarray,
    box: DetectionBox,
    pad: int = 8,
) -> np.ndarray:
    height, width = image.shape[:2]
    y1 = max(0, box.ymin - pad)
    x1 = max(0, box.xmin - pad)
    y2 = min(height, box.ymax + pad)
    x2 = min(width, box.xmax + pad)
    return image[y1:y2, x1:x2]


def draw_detection_boxes(
    image: np.ndarray,
    detections: Iterable[DetectionBox],
    color: tuple[int, int, int] = (0, 200, 0),
    show_text: bool = False,
    thickness: int = 1,
) -> np.ndarray:
    output = image.copy()
    for det in detections:
        cv2.rectangle(output, (det.xmin, det.ymin), (det.xmax, det.ymax), color, thickness)
        if show_text and det.text:
            cv2.putText(
                output,
                det.text,
                (det.xmin, max(20, det.ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return output


def sort_text_boxes(detections: list[DetectionBox], y_bin_size: int = 25) -> list[DetectionBox]:
    return sorted(detections, key=lambda det: (det.ymin // y_bin_size, det.xmin))


def get_thai_font_candidates() -> list[Path]:
    return [
        Path("C:/Windows/Fonts/LeelawUI.ttf"),
        Path("C:/Windows/Fonts/LeelawUIb.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/THSarabunNew.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]


def get_thai_font_path() -> Path | None:
    for candidate in get_thai_font_candidates():
        if candidate.exists():
            return candidate
    return None


def get_thai_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in get_thai_font_candidates():
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_positioned_text_image(
    image_size: tuple[int, int],
    detections: list[DetectionBox],
    background: str = "white",
    text_color: str = "black",
    draw_boxes: bool = False,
) -> Image.Image:
    canvas = Image.new("RGB", image_size, background)
    drawer = ImageDraw.Draw(canvas)

    for det in detections:
        text_value = det.text.strip()
        if not text_value:
            continue

        max_font = max(12, min(72, int(det.height * 0.95)))
        min_font = 8
        chosen_font = get_thai_font(min_font)

        for font_size in range(max_font, min_font - 1, -1):
            font = get_thai_font(font_size)
            left, top, right, bottom = drawer.textbbox((0, 0), text_value, font=font)
            text_width = right - left
            text_height = bottom - top
            if text_width <= max(4, det.width - 8) and text_height <= max(4, det.height - 6):
                chosen_font = font
                break

        left, top, right, bottom = drawer.textbbox((0, 0), text_value, font=chosen_font)
        text_width = right - left
        text_height = bottom - top
        draw_x = det.xmin + 4
        draw_y = det.ymin + max(1, (det.height - text_height) // 8)
        drawer.text((draw_x, draw_y), text_value, font=chosen_font, fill=text_color)
        if draw_boxes:
            drawer.rectangle((det.xmin, det.ymin, det.xmax, det.ymax), outline="#4caf50", width=1)
    return canvas


def render_text_overlay_image(
    base_image: Image.Image,
    detections: list[DetectionBox],
    text_color: str = "#111111",
    mask_fill: tuple[int, int, int, int] = (255, 255, 255, 210),
    draw_boxes: bool = True,
) -> Image.Image:
    canvas = base_image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    drawer = ImageDraw.Draw(overlay)

    for det in detections:
        drawer.rectangle((det.xmin, det.ymin, det.xmax, det.ymax), fill=mask_fill)
        if draw_boxes:
            drawer.rectangle((det.xmin, det.ymin, det.xmax, det.ymax), outline=(57, 181, 74, 255), width=1)

    merged = Image.alpha_composite(canvas, overlay).convert("RGB")
    merged_drawer = ImageDraw.Draw(merged)
    for det in detections:
        text_value = det.text.strip()
        if not text_value:
            continue

        max_font = max(12, min(72, int(det.height * 0.95)))
        min_font = 8
        chosen_font = get_thai_font(min_font)

        for font_size in range(max_font, min_font - 1, -1):
            font = get_thai_font(font_size)
            left, top, right, bottom = merged_drawer.textbbox((0, 0), text_value, font=font)
            text_width = right - left
            text_height = bottom - top
            if text_width <= max(4, det.width - 8) and text_height <= max(4, det.height - 6):
                chosen_font = font
                break

        left, top, right, bottom = merged_drawer.textbbox((0, 0), text_value, font=chosen_font)
        text_height = bottom - top
        draw_x = det.xmin + 4
        draw_y = det.ymin + max(1, (det.height - text_height) // 8)
        merged_drawer.text((draw_x, draw_y), text_value, font=chosen_font, fill=text_color)

    return merged


def image_to_download_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()
