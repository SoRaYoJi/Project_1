# thai_ocr/split_tsv.py
import argparse
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", default="data/lines/train.tsv")
    ap.add_argument("--out_train", default="data/lines/train_100k.tsv")
    ap.add_argument("--out_val", default="data/lines/val.tsv")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    lines = Path(args.in_tsv).read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    random.shuffle(lines)

    n_val = max(200, int(len(lines) * args.val_ratio))
    val = lines[:n_val]
    train = lines[n_val:]

    Path(args.out_train).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.out_val).write_text("\n".join(val) + "\n", encoding="utf-8")

    print("Total:", len(lines))
    print("Train:", len(train))
    print("Val:", len(val))
    print("Saved:", args.out_train, args.out_val)

if __name__ == "__main__":
    main()