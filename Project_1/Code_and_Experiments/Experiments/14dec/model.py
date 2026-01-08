import torch
import torch.nn as nn


class ThaiDigitNet(nn.Module):
def __init__(self):
super().__init__()
def block(i, o):
return nn.Sequential(
nn.Conv2d(i, o, 3, padding=1, bias=False),
nn.BatchNorm2d(o),
nn.LeakyReLU(0.1),
nn.Conv2d(o, o, 3, padding=1, bias=False),
nn.BatchNorm2d(o),
nn.LeakyReLU(0.1),
nn.MaxPool2d(2),
nn.Dropout(0.25)
)
self.b1 = block(1, 32)
self.b2 = block(32, 64)
self.b3 = block(64, 128)
self.b4 = block(128, 256)
self.fc = nn.Sequential(
nn.Flatten(),
nn.Linear(256*6*6, 512, bias=False),
nn.BatchNorm1d(512),
nn.LeakyReLU(0.1),
nn.Dropout(0.5),
nn.Linear(512, 10)
)
def forward(self, x):
return self.fc(self.b4(self.b3(self.b2(self.b1(x)))))