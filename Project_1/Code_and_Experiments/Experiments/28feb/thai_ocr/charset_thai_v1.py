# thai_ocr/charset_thai_v1.py

import re
import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

# =========================================================
# 1) Unicode Normalization
# =========================================================

ZW_CHARS = ["\u200b", "\ufeff"]  # zero-width space, BOM


def normalize_thai_text(text: str) -> str:
    """
    Normalize Thai text to NFC, remove zero-width characters,
    and normalize whitespace.
    """
    if text is None:
        return ""

    for z in ZW_CHARS:
        text = text.replace(z, "")

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# =========================================================
# 2) Charset Definition (Thai OCR v1)
# =========================================================

def build_thai_charset_v1() -> List[str]:
    """
    Thai OCR Charset v1
    Designed for line-level OCR on documents.
    """

    # 1. Space
    space = [" "]

    # 2. Thai consonants (44 letters)
    consonants = list(
        "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    )

    # 3. Thai vowels and signs
    vowels = list("ะาำิีึืุูเแโใไั")

    # 4. Tone marks
    tones = list("่้๊๋")

    # 5. Special marks
    special_marks = list("์ํ")  # karan, nikhahit

    # 6. Repetition / Thai punctuation
    thai_punct = list("ๆฯ")

    # 7. Thai digits
    thai_digits = list("๐๑๒๓๔๕๖๗๘๙")

    # 8. Arabic digits
    arabic_digits = list("0123456789")

    # 9. Common document punctuation
    punctuation = list(".,:;!?()[]{}\\/ -_+=“”\"'`#%@&*")

    charset = []

    for group in [
        space,
        consonants,
        vowels,
        tones,
        special_marks,
        thai_punct,
        thai_digits,
        arabic_digits,
        punctuation,
    ]:
        for ch in group:
            if ch not in charset:
                charset.append(ch)

    return charset


# =========================================================
# 3) CTC Tokenizer
# =========================================================

@dataclass
class CTCTokenizer:
    id2ch: List[str]
    ch2id: Dict[str, int]
    blank_id: int = 0

    def encode(self, text: str) -> List[int]:
        text = normalize_thai_text(text)
        ids = []
        for ch in text:
            if ch in self.ch2id:
                ids.append(self.ch2id[ch])
            # ถ้าเจอ OOV จะ ignore (สามารถเปลี่ยนให้ log ได้)
        return ids

    def decode_greedy(self, ids: List[int]) -> str:
        """
        Basic CTC greedy decoding:
        - remove blank
        - collapse repeats
        """
        result = []
        prev = None

        for idx in ids:
            if idx == self.blank_id:
                prev = idx
                continue

            if idx == prev:
                continue

            result.append(self.id2ch[idx])
            prev = idx

        return "".join(result)


def create_tokenizer() -> CTCTokenizer:
    charset = build_thai_charset_v1()

    # CTC requires blank at index 0
    id2ch = ["<BLANK>"] + charset
    ch2id = {ch: i for i, ch in enumerate(id2ch) if i != 0}

    return CTCTokenizer(id2ch=id2ch, ch2id=ch2id, blank_id=0)


# =========================================================
# 4) Export Charset to JSON
# =========================================================

def export_charset_json(path: str = "charset_thai_v1.json"):
    charset = build_thai_charset_v1()

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"charset": charset}, f, ensure_ascii=False, indent=2)

    print(f"Saved charset to {path}")
    print(f"Total characters (without blank): {len(charset)}")
    print(f"Total classes (with blank for CTC): {len(charset) + 1}")


# =========================================================
# 5) Run directly to generate JSON
# =========================================================

if __name__ == "__main__":
    export_charset_json("charset_thai_v1.json")