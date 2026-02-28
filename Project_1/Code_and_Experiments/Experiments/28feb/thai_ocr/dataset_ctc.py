# thai_ocr/dataset_ctc.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from thai_ocr.charset_thai_v1 import normalize_thai_text

class LineTSVDataset(Dataset):
    """
    TSV: image_path<TAB>label
    """
    def __init__(self, tsv_path: str, tokenizer, img_h: int = 32, cache_images: bool = False):
        self.items = []
        self.tokenizer = tokenizer
        self.img_h = img_h
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                path, text = line.split("\t", 1)
                text = normalize_thai_text(text)
                self.items.append((path, text))

    def __len__(self):
        return len(self.items)

    def _read_gray(self, path: str):
        if self.cache_images and path in self._cache:
            return self._cache[path]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)

        if self.cache_images:
            self._cache[path] = img
        return img

    def _resize_keep_ratio(self, img):
        h, w = img.shape
        new_h = self.img_h
        new_w = max(8, int(w * (new_h / h)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx):
        path, text = self.items[idx]
        img = self._read_gray(path)
        img = self._resize_keep_ratio(img)

        # float32 [0,1]
        x = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)  # [1,H,W]
        y = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # return width for proper x_lens
        w = x.shape[-1]
        return x, y, text, w

def pad_collate(batch):
    xs, ys, texts, ws = zip(*batch)
    max_w = max(ws)

    padded = []
    for x in xs:
        pad_w = max_w - x.shape[-1]
        if pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w), value=1.0)  # white
        padded.append(x)

    x_batch = torch.stack(padded, dim=0)  # [B,1,H,Wmax]
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    y_concat = torch.cat(ys, dim=0)

    # return widths too
    w_batch = torch.tensor(ws, dtype=torch.long)
    return x_batch, y_concat, y_lens, texts, w_batch