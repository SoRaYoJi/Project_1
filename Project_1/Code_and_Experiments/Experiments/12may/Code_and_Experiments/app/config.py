from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
CODE_AND_EXPERIMENTS_DIR = BASE_DIR / "Code_and_Experiments"
APP_DIR = CODE_AND_EXPERIMENTS_DIR / "app"
EXPERIMENTS_DIR = CODE_AND_EXPERIMENTS_DIR / "Experiments"
MODELS_DIR = CODE_AND_EXPERIMENTS_DIR / "Models"
EXPERIMENT_14DEC_DIR = EXPERIMENTS_DIR / "14dec"
EXPERIMENT_2MAY_DIR = EXPERIMENTS_DIR / "2may"


def load_environment() -> None:
    env_candidates = [
        BASE_DIR / ".env",
        CODE_AND_EXPERIMENTS_DIR / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


load_environment()


THAI_DIGIT_LABELS = ["๐", "๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙"]
THAI_DIGIT_MODEL_CANDIDATES = [
    MODELS_DIR / "model_read_numberthaiV1_pytorch.pth",
    EXPERIMENT_14DEC_DIR / "models" / "model_read_numberthaiV1_pytorch.pth",
    MODELS_DIR / "thai_digit_modelV3.pth",
]


@dataclass(frozen=True)
class ApiSettings:
    provider: str
    api_key: str | None
    model_name: str
    api_url: str | None


def get_api_settings() -> ApiSettings:
    provider = os.getenv("THAI_OCR_API_PROVIDER", "gemini").strip().lower()
    api_key = (
        os.getenv("THAI_OCR_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or None
    )
    model_name = (
        os.getenv("THAI_OCR_API_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash"
    )
    api_url = os.getenv("THAI_OCR_API_URL") or None
    return ApiSettings(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        api_url=api_url,
    )
