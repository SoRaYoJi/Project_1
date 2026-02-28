# thai_ocr/gen_synth_lines_mp.py
import os, io, random, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from thai_ocr.charset_thai_v1 import normalize_thai_text, build_thai_charset_v1


FONT_EXTS = {".ttf", ".otf", ".ttc"}

def load_fonts(font_dir: str):
    fps = [p for p in Path(font_dir).rglob("*") if p.suffix.lower() in FONT_EXTS]
    if not fps:
        raise FileNotFoundError(f"No fonts found in {font_dir}")
    return [str(p) for p in fps]

def load_corpus(corpus_path: str):
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for s in f:
            s = normalize_thai_text(s)
            if s:
                lines.append(s)
    if not lines:
        raise ValueError("Corpus empty")
    return lines

def sample_text(corpus_lines, max_len=80):
    s = random.choice(corpus_lines)
    if len(s) > max_len:
        start = random.randint(0, max(0, len(s) - max_len))
        s = s[start:start + max_len].strip()
    return s or random.choice(corpus_lines)

def keep_only_charset(text: str, charset_set: set):
    return "".join([ch for ch in text if ch in charset_set])

def render_line(text: str, font_path: str, font_size: int, pad=6):
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w = max(32, (bbox[2] - bbox[0]) + pad * 2)
    h = max(16, (bbox[3] - bbox[1]) + pad * 2)

    bg = random.randint(235, 255)
    img = Image.new("L", (w, h), bg)
    draw = ImageDraw.Draw(img)
    fg = random.randint(0, 40)
    draw.text((pad, pad), text, fill=fg, font=font)
    return img

def aug_image(img: Image.Image):
    # ลด augmentation ที่แพงลงนิดเพื่อความเร็ว (คุณปรับได้)
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
    if random.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
    if random.random() < 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.25:
        arr = np.array(img).astype(np.float32)
        arr = np.clip(arr + np.random.normal(0, random.uniform(2, 8), arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    return img

def worker(args):
    (idx, out_dir, corpus_lines, fonts, charset_set, max_len, min_len, seed) = args
    random.seed(seed + idx)
    np.random.seed(seed + idx)

    # วนจนได้ข้อความที่ยาวพอ
    for _ in range(20):
        text = sample_text(corpus_lines, max_len=max_len)
        text = normalize_thai_text(text)
        text = keep_only_charset(text, charset_set)
        if len(text) >= min_len:
            break
    if len(text) < min_len:
        return None

    font_path = random.choice(fonts)
    font_size = random.randint(18, 44)
    img = render_line(text, font_path, font_size, pad=random.randint(4, 10))
    img = aug_image(img)

    # save (PNG ช้า) -> แนะนำ JPEG เร็วกว่า
    fname = f"{idx:06d}.jpg"
    out_path = os.path.join(out_dir, fname)
    img.convert("RGB").save(out_path, quality=85)  # เร็วกว่า PNG
    return out_path, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font_dir", default="font")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out_dir", default="data/lines/train")
    ap.add_argument("--tsv", default="data/lines/train.tsv")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--workers", type=int, default=max(2, cpu_count() - 1))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tsv), exist_ok=True)

    fonts = load_fonts(args.font_dir)
    corpus_lines = load_corpus(args.corpus)
    charset_set = set(build_thai_charset_v1())

    print(f"Workers: {args.workers}, Fonts: {len(fonts)}, Corpus lines: {len(corpus_lines)}")

    jobs = [
        (i, args.out_dir, corpus_lines, fonts, charset_set, args.max_len, args.min_len, args.seed)
        for i in range(args.n)
    ]

    results = []
    with Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=200):
            if r is not None:
                results.append(r)
            if len(results) % 2000 == 0 and len(results) > 0:
                print(f"Generated: {len(results)}/{args.n}")

    # เขียน tsv
    results.sort(key=lambda x: x[0])  # เรียงตามชื่อไฟล์
    with open(args.tsv, "w", encoding="utf-8") as f:
        for path, text in results:
            f.write(f"{path}\t{text}\n")

    print("Done.")
    print("Total written:", len(results))
    print("TSV:", args.tsv)

if __name__ == "__main__":
    main()