from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import THAI_DIGIT_LABELS, THAI_DIGIT_MODEL_CANDIDATES


class ThaiDigitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )

        self.block1 = conv_block(1, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


@dataclass(frozen=True)
class DigitPrediction:
    digit: str
    confidence: float
    box: tuple[int, int, int, int]
    processed_image: np.ndarray


@dataclass(frozen=True)
class DigitOcrResult:
    text: str
    average_confidence: float
    predictions: list[DigitPrediction]
    debug_image: np.ndarray
    model_path: Path


def resolve_model_path(explicit_path: str | Path | None = None) -> Path:
    candidates = [Path(explicit_path)] if explicit_path else []
    candidates.extend(THAI_DIGIT_MODEL_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in THAI_DIGIT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "ไม่พบไฟล์โมเดล OCR ตัวเลขไทย\n"
        f"ค้นหาแล้วที่:\n{searched}"
    )


def load_digit_model(model_path: str | Path | None = None) -> tuple[ThaiDigitNet, Path]:
    resolved_path = resolve_model_path(model_path)
    model = ThaiDigitNet()
    state_dict = torch.load(resolved_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_path


def preprocess_for_model(roi: np.ndarray, target_size: tuple[int, int] = (96, 96)) -> tuple[np.ndarray, torch.Tensor]:
    height, width = roi.shape[:2]
    padding = 10
    scale = min((target_size[0] - padding * 2) / width, (target_size[1] - padding * 2) / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width <= 0 or new_height <= 0:
        blank = np.zeros(target_size, dtype=np.uint8)
        tensor = torch.zeros(1, 1, target_size[0], target_size[1], dtype=torch.float32)
        return blank, tensor

    resized = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return canvas, tensor


def predict_single(model: ThaiDigitNet, tensor_img: torch.Tensor) -> tuple[str, float]:
    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.softmax(logits, dim=1)
        top_probability, top_class = probabilities.topk(1, dim=1)
    predicted_index = top_class.item()
    return THAI_DIGIT_LABELS[predicted_index], top_probability.item() * 100.0


def sort_bounding_boxes(boxes: list[tuple[int, int, int, int]], y_threshold: int = 30) -> list[list[tuple[int, int, int, int]]]:
    processed_boxes = []
    for x, y, w, h in boxes:
        center_y = y + h // 2
        processed_boxes.append((x, y, w, h, center_y))

    processed_boxes.sort(key=lambda item: item[4])
    if not processed_boxes:
        return []

    lines: list[list[tuple[int, int, int, int, int]]] = [[processed_boxes[0]]]
    for box in processed_boxes[1:]:
        if abs(box[4] - lines[-1][0][4]) < y_threshold:
            lines[-1].append(box)
        else:
            lines.append([box])

    sorted_lines: list[list[tuple[int, int, int, int]]] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        sorted_lines.append([item[:4] for item in line])
    return sorted_lines


def binarize_digit_image(gray_image: np.ndarray, invert: bool = True, thinning_level: int = 0) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if invert else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray_image, 0, 255, threshold_type)
    if thinning_level > 0:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=thinning_level)
    return binary


def recognize_digit_text(
    model: ThaiDigitNet,
    rgb_image: np.ndarray,
    invert: bool = True,
    thinning_level: int = 1,
    min_size: int = 10,
    contour_area_threshold: int = 50,
    allow_split_wide_boxes: bool = True,
) -> DigitOcrResult:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary = binarize_digit_image(gray, invert=invert, thinning_level=thinning_level)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) > contour_area_threshold
    ]

    if not boxes:
        raise ValueError("ไม่พบตัวเลขในภาพ")

    sorted_lines = sort_bounding_boxes(boxes)
    predictions: list[DigitPrediction] = []
    lines_text: list[str] = []
    debug_image = rgb_image.copy()

    for line_boxes in sorted_lines:
        line_text = ""
        for x, y, w, h in line_boxes:
            if w < min_size or h < min_size:
                continue

            split_count = 1
            split_width = w
            if allow_split_wide_boxes and (w / float(h)) > 1.2:
                split_count = max(1, int(round(w / h)))
                split_width = max(1, w // split_count)

            for index in range(split_count):
                current_x = x + (index * split_width)
                current_w = split_width if index < split_count - 1 else (x + w) - current_x
                roi = binary[y:y + h, current_x:current_x + current_w]
                if roi.size == 0:
                    continue

                display_img, tensor = preprocess_for_model(roi)
                digit, confidence = predict_single(model, tensor)
                line_text += digit
                predictions.append(
                    DigitPrediction(
                        digit=digit,
                        confidence=confidence,
                        box=(current_x, y, current_w, h),
                        processed_image=display_img,
                    )
                )
                cv2.rectangle(debug_image, (current_x, y), (current_x + current_w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    debug_image,
                    digit,
                    (current_x, max(20, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        if line_text:
            lines_text.append(line_text)

    if not predictions:
        raise ValueError("พบ contour แต่ไม่สามารถอ่านตัวเลขได้")

    average_confidence = float(np.mean([item.confidence for item in predictions]))
    model_path = resolve_model_path()
    return DigitOcrResult(
        text="\n".join(lines_text),
        average_confidence=average_confidence,
        predictions=predictions,
        debug_image=debug_image,
        model_path=model_path,
    )
