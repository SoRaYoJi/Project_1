# thai_ocr/model_crnn_ctc.py

import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """
    Match digit model state_dict keys:
    block1..block4, classifier.*
    """
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = block(1, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.block4 = block(128, 256)

        # digit head (unused in OCR training, but needed to match keys)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


def load_digit_backbone(pth_path: str) -> nn.Module:
    """
    Load your digit model (.pth state_dict) and return backbone blocks 1-4.
    """
    m = DigitCNN()
    sd = torch.load(pth_path, map_location="cpu")
    m.load_state_dict(sd, strict=False)
    return nn.Sequential(m.block1, m.block2, m.block3, m.block4)


class CRNN_CTC(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        f = self.backbone(x)     # [B,256,H',W']
        f = f.mean(dim=2)        # [B,256,W']
        f = f.permute(0, 2, 1)   # [B,T,256]
        y, _ = self.rnn(f)       # [B,T,512]
        logits = self.fc(y)      # [B,T,C]
        return logits