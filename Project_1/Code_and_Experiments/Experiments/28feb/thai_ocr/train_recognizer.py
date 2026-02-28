# thai_ocr/train_recognizer.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thai_ocr.charset_thai_v1 import create_tokenizer, normalize_thai_text
from thai_ocr.dataset_ctc import LineTSVDataset, pad_collate
from thai_ocr.model_crnn_ctc import load_digit_backbone, CRNN_CTC


def ctc_greedy_decode(logits, tokenizer):
    pred = logits.argmax(dim=-1)
    return [tokenizer.decode_greedy(row.tolist()) for row in pred]


def cer(pred: str, gt: str) -> float:
    pred = normalize_thai_text(pred)
    gt = normalize_thai_text(gt)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = list(range(len(gt) + 1))
    for i, pc in enumerate(pred, 1):
        prev = dp[0]
        dp[0] = i
        for j, gc in enumerate(gt, 1):
            cur = dp[j]
            cost = 0 if pc == gc else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1] / max(1, len(gt))


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = create_tokenizer()
    num_classes = len(tokenizer.id2ch)
    print("num_classes:", num_classes)

    train_tsv = "data/lines/train_100k.tsv"
    val_tsv   = "data/lines/val.tsv"

    digit_pth = "models/model_read_numberthaiV1_pytorch.pth"

    img_h = 32
    batch_size = 256 if device.type == "cuda" else 32
    num_workers = 8 if device.type == "cuda" else 0

    epochs = 100
    freeze_epochs = 5
    lr_head = 3e-4
    lr_full = 1e-4

    train_ds = LineTSVDataset(train_tsv, tokenizer, img_h=img_h)
    val_ds   = LineTSVDataset(val_tsv, tokenizer, img_h=img_h)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=pad_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=pad_collate,
    )

    backbone = load_digit_backbone(digit_pth)
    model = CRNN_CTC(backbone, num_classes=num_classes).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs("models", exist_ok=True)
    best_cer = 1e9

    total_start_time = time.time()

    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        if ep == freeze_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_full)
            print(">>> Unfroze backbone")

        for x, y_concat, y_lens, texts, w_batch in train_loader:
            x = x.to(device, non_blocking=True)
            y_concat = y_concat.to(device, non_blocking=True)
            y_lens = y_lens.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
                T = log_probs.size(0)

                # IMPORTANT: ใช้ T จริง
                x_lens = torch.full(
                    size=(log_probs.size(1),),
                    fill_value=T,
                    dtype=torch.long,
                    device=device,
                )

                loss = ctc_loss(log_probs, y_concat, x_lens, y_lens)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validate ทุก 10 epoch
        if ep % 10 == 0 or ep == 1:
            model.eval()
            cer_sum = 0.0
            n = 0
            with torch.inference_mode():
                for x, y_concat, y_lens, texts, w_batch in val_loader:
                    x = x.to(device)
                    logits = model(x)
                    preds = ctc_greedy_decode(logits, tokenizer)
                    for p, gt in zip(preds, texts):
                        cer_sum += cer(p, gt)
                        n += 1

            val_cer = cer_sum / max(1, n)

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(
                    {"model": model.state_dict(), "charset": tokenizer.id2ch},
                    "models/thai_crnn_ctc_best.pt",
                )

            epoch_time_min = (time.time() - epoch_start) / 60
            total_time_min = (time.time() - total_start_time) / 60

            print(
                f"[Epoch {ep:03d}] "
                f"train_loss={avg_loss:.4f} | "
                f"val_CER={val_cer:.4f} | "
                f"epoch_time={epoch_time_min:.2f} min | "
                f"total_time={total_time_min:.2f} min | "
                f"best={best_cer:.4f}"
            )
        else:
            epoch_time_min = (time.time() - epoch_start) / 60
            print(
                f"[Epoch {ep:03d}] "
                f"train_loss={avg_loss:.4f} | "
                f"epoch_time={epoch_time_min:.2f} min"
            )

    print("Training Finished.")
    print("Best CER:", best_cer)


if __name__ == "__main__":
    main()