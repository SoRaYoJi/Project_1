import torch
import numpy as np

def predict(model, img_np):
    x = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    top3 = probs.argsort()[-3:][::-1]
    return probs.argmax(), probs.max(), top3, probs
