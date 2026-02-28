# thai_ocr/setup_corpus.py

import random
from pathlib import Path

THAI_SENTENCES = [
    "บริษัท จำกัด (มหาชน)",
    "เลขที่ ๑๒๓/๔๕ ถนนสุขุมวิท",
    "วันที่ ๕ มกราคม ๒๕๖๗",
    "ชื่อ-นามสกุล นายสมชาย ใจดี",
    "รหัสไปรษณีย์ ๑๐๓๓๐",
    "รวมเป็นเงินทั้งสิ้น ๑,๒๕๐ บาท",
    "โทรศัพท์ ๐๒-๑๒๓-๔๕๖๗",
    "ใบเสร็จรับเงิน",
    "สำเนาถูกต้อง",
    "ผู้มีอำนาจลงนาม",
    "ที่อยู่ ๙๙/๙ หมู่ที่ ๓",
    "อำเภอเมือง จังหวัดกรุงเทพมหานคร",
    "ภาษีมูลค่าเพิ่ม ๗%",
    "ยอดรวมสุทธิ ๓,๕๐๐.๐๐ บาท",
]

THAI_WORDS = [
    "บริษัท", "จำกัด", "มหาชน", "ที่อยู่", "โทรศัพท์",
    "เลขที่", "ถนน", "จังหวัด", "อำเภอ", "ตำบล",
    "ชื่อ", "นามสกุล", "ยอดรวม", "สุทธิ",
    "บาท", "วันที่", "เดือน", "ปี",
    "รหัส", "สินค้า", "จำนวน", "ราคา",
    "รวม", "ภาษี", "สำเนา", "ถูกต้อง",
    "ผู้จัดการ", "กรรมการ", "ใบแจ้งหนี้",
]

THAI_DIGITS = list("๐๑๒๓๔๕๖๗๘๙")

def random_number(length=3):
    return "".join(random.choice(THAI_DIGITS) for _ in range(length))

def random_sentence():
    words = random.sample(THAI_WORDS, random.randint(3, 7))
    if random.random() < 0.7:
        words.append(random_number(random.randint(2, 6)))
    return " ".join(words)

def main():
    project_root = Path(__file__).resolve().parents[1]
    corpus_path = project_root / "corpus.txt"

    lines = []

    # ใส่ประโยคตั้งต้น
    lines.extend(THAI_SENTENCES)

    # สร้างสุ่มเพิ่ม
    for _ in range(5000):
        lines.append(random_sentence())

    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")

    print(f"Created corpus at: {corpus_path}")
    print(f"Total lines: {len(lines)}")

if __name__ == "__main__":
    main()