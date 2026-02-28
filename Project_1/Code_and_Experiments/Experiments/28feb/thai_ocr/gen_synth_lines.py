# thai_ocr/gen_synth_lines.py

import os
import io
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from thai_ocr.charset_thai_v1 import normalize_thai_text, build_thai_charset_v1


def load_fonts(font_dir: str):
    exts = {".ttf", ".otf", ".ttc"}
    font_paths = [p for p in Path(font_dir).rglob("*") if p.suffix.lower() in exts]
    if not font_paths:
        raise FileNotFoundError(f"No fonts found in {font_dir}")
    return font_paths


def load_corpus(corpus_path: str):
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for s in f:
            s = normalize_thai_text(s)
            if len(s) >= 1:
                lines.append(s)
    if not lines:
        raise ValueError("Corpus is empty after normalization.")
    return lines


def sample_text(corpus_lines, max_len=64):
    s = random.choice(corpus_lines)
    # ตัดความยาวกันยาวเกินไป
    if len(s) > max_len:
        start = random.randint(0, max(0, len(s) - max_len))
        s = s[start:start + max_len]
        s = s.strip()
    return s if s else random.choice(corpus_lines)


def render_line(text: str, font_path: str, font_size: int, pad=6):
    font = ImageFont.truetype(str(font_path), font_size)

    # วัดขนาดข้อความ
    dummy = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2

    # กันภาพเล็กเกิน
    w = max(w, 32)
    h = max(h, 16)

    # สุ่มพื้นหลังขาว/ครีมเล็กน้อย
    bg = random.randint(235, 255)
    img = Image.new("L", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # สุ่มสีตัวอักษร (ดำ-เทาเข้ม)
    fg = random.randint(0, 40)
    draw.text((pad, pad), text, fill=fg, font=font)

    return img


def aug_image(img: Image.Image):
    # blur
    if random.random() < 0.35:
        r = random.uniform(0.2, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))

    # brightness/contrast
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))

    # noise
    if random.random() < 0.4:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    # JPEG artifacts
    if random.random() < 0.35:
        q = random.randint(30, 85)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")

    return img


def keep_only_charset(text: str, charset_set: set):
    # กรองอักขระนอก charset ทิ้ง (กัน label พัง)
    return "".join([ch for ch in text if ch in charset_set])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font_dir", default="fonts")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out_dir", default="data/lines/train")
    ap.add_argument("--tsv", default="data/lines/train.tsv")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tsv), exist_ok=True)

    fonts = load_fonts(args.font_dir)
    corpus_lines = load_corpus(args.corpus)

    charset = build_thai_charset_v1()
    charset_set = set(charset)

    with open(args.tsv, "w", encoding="utf-8") as f:
        for i in range(args.n):
            text = sample_text(corpus_lines, max_len=args.max_len)
            text = normalize_thai_text(text)
            text = keep_only_charset(text, charset_set)

            if len(text) < args.min_len:
                continue

            font_path = random.choice(fonts)
            font_size = random.randint(18, 44)

            img = render_line(text, font_path, font_size, pad=random.randint(4, 10))
            img = aug_image(img)

            # save
            fname = f"{i:06d}.png"
            out_path = os.path.join(args.out_dir, fname)
            img.save(out_path)

            f.write(f"{out_path}\t{text}\n")

            if (i + 1) % 2000 == 0:
                print(f"Generated: {i+1}/{args.n}")

    print("Done.")
    print("Images:", args.out_dir)
    print("Labels:", args.tsv)


if __name__ == "__main__":
    main()