from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PIL import Image
from google import genai
from pydantic import BaseModel, Field

from .config import get_api_settings
from .image_utils import DetectionBox, crop_with_padding, pil_to_numpy, sort_text_boxes
from .ocr_digit import DigitOcrResult, ThaiDigitNet, recognize_digit_text


THAI_DIGIT_PATTERN = re.compile(r"[๐-๙]")
THAI_DIGIT_ONLY_PATTERN = re.compile(r"^[\s๐-๙\-/():.,]+$")


@dataclass(frozen=True)
class ApiUsage:
    request_count: int
    prompt_token_count: int | None
    candidates_token_count: int | None
    thoughts_token_count: int | None
    total_token_count: int | None


@dataclass(frozen=True)
class ApiDetection:
    box: DetectionBox
    raw_text: str
    resolved_text: str
    source: str


@dataclass(frozen=True)
class ThaiApiOcrResult:
    text: str
    detections: list[ApiDetection]
    raw_response_text: str
    provider: str
    model_name: str
    usage: ApiUsage


class StructuredOcrDetection(BaseModel):
    box_2d: list[int] = Field(description="Bounding box in [ymin, xmin, ymax, xmax] order.")
    text: str = Field(description="Detected Thai text or Thai numerals found in the box.")


def count_thai_digits(text: str) -> int:
    return len(THAI_DIGIT_PATTERN.findall(text))


def is_digit_only_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(THAI_DIGIT_PATTERN.search(stripped)) and bool(THAI_DIGIT_ONLY_PATTERN.fullmatch(stripped))


def should_use_digit_model(raw_text: str, digit_result: DigitOcrResult) -> bool:
    normalized_prediction = digit_result.text.replace("\n", " ").strip()
    if not normalized_prediction:
        return False
    if not is_digit_only_text(raw_text):
        return False
    if not is_digit_only_text(normalized_prediction):
        return False
    if count_thai_digits(raw_text) != count_thai_digits(normalized_prediction):
        return False
    return digit_result.average_confidence >= 85.0


def scale_detection_boxes(detections: list[ApiDetection], image_size: tuple[int, int]) -> list[ApiDetection]:
    if not detections:
        return detections

    image_width, image_height = image_size
    max_x = max(item.box.xmax for item in detections)
    max_y = max(item.box.ymax for item in detections)
    if max_x <= 0 or max_y <= 0:
        return detections

    scale_x = image_width / max_x
    scale_y = image_height / max_y
    should_scale = scale_x > 1.2 or scale_y > 1.2 or scale_x < 0.8 or scale_y < 0.8
    if not should_scale:
        return detections

    scaled: list[ApiDetection] = []
    for item in detections:
        scaled.append(
            ApiDetection(
                box=DetectionBox(
                    ymin=max(0, int(round(item.box.ymin * scale_y))),
                    xmin=max(0, int(round(item.box.xmin * scale_x))),
                    ymax=min(image_height, int(round(item.box.ymax * scale_y))),
                    xmax=min(image_width, int(round(item.box.xmax * scale_x))),
                    text=item.box.text,
                ),
                raw_text=item.raw_text,
                resolved_text=item.resolved_text,
                source=item.source,
            )
        )
    return scaled


def clean_json_string(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text


def build_prompt(width: int, height: int) -> str:
    return f"""
You are a highly accurate Thai OCR system.

YOUR MISSION:
1. Read every visible Thai word, line, and numeral from the image.
2. Keep Thai numerals (๐-๙) separate from surrounding Thai text whenever possible.
3. Return bounding boxes in image coordinates.

IMAGE INFO:
- width: {width}px
- height: {height}px

OUTPUT FORMAT:
Return only a JSON array like:
[
  {{"box_2d": [ymin, xmin, ymax, xmax], "text": "วันที่"}},
  {{"box_2d": [ymin, xmin, ymax, xmax], "text": "๒๕๖๙"}}
]
""".strip()


def _parse_usage(response) -> ApiUsage:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return ApiUsage(
            request_count=1,
            prompt_token_count=None,
            candidates_token_count=None,
            thoughts_token_count=None,
            total_token_count=None,
        )
    return ApiUsage(
        request_count=1,
        prompt_token_count=getattr(metadata, "prompt_token_count", None),
        candidates_token_count=getattr(metadata, "candidates_token_count", None),
        thoughts_token_count=getattr(metadata, "thoughts_token_count", None),
        total_token_count=getattr(metadata, "total_token_count", None),
    )


def _call_gemini(image: Image.Image):
    settings = get_api_settings()
    client = genai.Client(api_key=settings.api_key)
    prompt = build_prompt(*image.size)

    try:
        response = client.models.generate_content(
            model=settings.model_name,
            contents=[prompt, image],
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "response_schema": list[StructuredOcrDetection],
            },
        )
        model_name = settings.model_name
    except Exception as primary_error:
        fallback_model = "gemini-flash"
        if settings.model_name == fallback_model:
            raise RuntimeError(f"เรียก API ไม่สำเร็จ: {primary_error}") from primary_error
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=[prompt, image],
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": list[StructuredOcrDetection],
                },
            )
            model_name = fallback_model
        except Exception as fallback_error:
            raise RuntimeError(
                f"เรียก API ไม่สำเร็จทั้ง model หลักและ fallback: {fallback_error}"
            ) from fallback_error
    return response, model_name


def call_thai_ocr_api(image: Image.Image) -> ThaiApiOcrResult:
    settings = get_api_settings()
    if settings.provider != "gemini":
        raise ValueError(f"ยังไม่รองรับ provider '{settings.provider}'")
    if not settings.api_key:
        raise ValueError(
            "ไม่พบ API key กรุณาตั้งค่า THAI_OCR_API_KEY หรือ GEMINI_API_KEY ในไฟล์ .env"
        )

    response, model_name = _call_gemini(image)
    raw_text = response.text or ""
    payload = response.parsed
    if payload is None:
        try:
            payload = json.loads(clean_json_string(raw_text))
        except Exception as exc:
            raise RuntimeError("API ส่งผลลัพธ์กลับมาไม่เป็น JSON ตามที่คาดไว้") from exc

    detections: list[ApiDetection] = []
    for item in payload:
        if isinstance(item, StructuredOcrDetection):
            box_2d = item.box_2d
            text = item.text.strip()
        else:
            box_2d = item.get("box_2d")
            text = str(item.get("text", "")).strip()
        if not isinstance(box_2d, list) or len(box_2d) != 4 or not text:
            continue
        ymin, xmin, ymax, xmax = [int(value) for value in box_2d]
        detections.append(
            ApiDetection(
                box=DetectionBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, text=text),
                raw_text=text,
                resolved_text=text,
                source="api",
            )
        )

    detections = scale_detection_boxes(detections, image.size)
    detections.sort(key=lambda det: (det.box.ymin // 25, det.box.xmin))
    return ThaiApiOcrResult(
        text=format_detections_text(detections),
        detections=detections,
        raw_response_text=raw_text,
        provider=settings.provider,
        model_name=model_name,
        usage=_parse_usage(response),
    )


def format_detections_text(detections: list[ApiDetection], y_bin_size: int = 25) -> str:
    if not detections:
        return ""

    sorted_items = sorted(detections, key=lambda det: (det.box.ymin // y_bin_size, det.box.xmin))
    chunks: list[str] = []
    last_y_bin: int | None = None
    for item in sorted_items:
        current_y_bin = item.box.ymin // y_bin_size
        if last_y_bin is None:
            chunks.append(item.resolved_text)
        elif current_y_bin - last_y_bin >= 2:
            chunks.append(f"\n\n{item.resolved_text}")
        elif current_y_bin - last_y_bin == 1:
            chunks.append(f"\n{item.resolved_text}")
        else:
            chunks.append(f" {item.resolved_text}")
        last_y_bin = max(last_y_bin or current_y_bin, current_y_bin)
    return "".join(chunks).strip()


def run_combined_ocr(image: Image.Image, digit_model: ThaiDigitNet) -> tuple[ThaiApiOcrResult, list[DigitOcrResult]]:
    api_result = call_thai_ocr_api(image)
    rgb_image = pil_to_numpy(image)
    combined_detections: list[ApiDetection] = []
    digit_results: list[DigitOcrResult] = []

    for detection in sort_text_boxes([item.box for item in api_result.detections]):
        matching = next(item for item in api_result.detections if item.box == detection)
        if THAI_DIGIT_PATTERN.search(matching.raw_text):
            crop = crop_with_padding(rgb_image, matching.box, pad=8)
            try:
                digit_result = recognize_digit_text(
                    model=digit_model,
                    rgb_image=crop,
                    invert=True,
                    thinning_level=1,
                    allow_split_wide_boxes=True,
                )
            except Exception:
                combined_detections.append(
                    ApiDetection(
                        box=matching.box,
                        raw_text=matching.raw_text,
                        resolved_text=matching.raw_text,
                        source="api-fallback",
                    )
                )
            else:
                digit_results.append(digit_result)
                if should_use_digit_model(matching.raw_text, digit_result):
                    combined_detections.append(
                        ApiDetection(
                            box=matching.box,
                            raw_text=matching.raw_text,
                            resolved_text=digit_result.text.replace("\n", " ").strip(),
                            source="digit-model",
                        )
                    )
                else:
                    combined_detections.append(
                        ApiDetection(
                            box=matching.box,
                            raw_text=matching.raw_text,
                            resolved_text=matching.raw_text,
                            source="api-preferred",
                        )
                    )
        else:
            combined_detections.append(matching)

    remapped = sorted(combined_detections, key=lambda det: (det.box.ymin // 25, det.box.xmin))

    return (
        ThaiApiOcrResult(
            text=format_detections_text(remapped),
            detections=remapped,
            raw_response_text=api_result.raw_response_text,
            provider=api_result.provider,
            model_name=api_result.model_name,
            usage=api_result.usage,
        ),
        digit_results,
    )


def run_digit_ocr_with_api(image: Image.Image, digit_model: ThaiDigitNet) -> tuple[ThaiApiOcrResult, list[DigitOcrResult]]:
    api_result = call_thai_ocr_api(image)
    rgb_image = pil_to_numpy(image)
    digit_detections: list[ApiDetection] = []
    digit_results: list[DigitOcrResult] = []

    for item in api_result.detections:
        if not THAI_DIGIT_PATTERN.search(item.raw_text):
            continue

        crop = crop_with_padding(rgb_image, item.box, pad=8)
        try:
            digit_result = recognize_digit_text(
                model=digit_model,
                rgb_image=crop,
                invert=True,
                thinning_level=1,
                allow_split_wide_boxes=True,
            )
        except Exception:
            digit_detections.append(
                ApiDetection(
                    box=item.box,
                    raw_text=item.raw_text,
                    resolved_text=item.raw_text,
                    source="api-fallback",
                )
            )
        else:
            digit_results.append(digit_result)
            digit_detections.append(
                ApiDetection(
                    box=item.box,
                    raw_text=item.raw_text,
                    resolved_text=digit_result.text.replace("\n", " "),
                    source="digit-model",
                )
            )

    if not digit_detections:
        raise ValueError("API ไม่พบบล็อกตัวเลขไทยในภาพ")

    digit_detections = sorted(digit_detections, key=lambda det: (det.box.ymin // 25, det.box.xmin))
    return (
        ThaiApiOcrResult(
            text=format_detections_text(digit_detections),
            detections=digit_detections,
            raw_response_text=api_result.raw_response_text,
            provider=api_result.provider,
            model_name=api_result.model_name,
            usage=api_result.usage,
        ),
        digit_results,
    )
