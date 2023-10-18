import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    # Apply sigmoid
    pred = torch.sigmoid(pred)

    gold = gold.float()
    pred = pred.clamp(min=1e-7, max=1-1e-7)  # clip values to avoid log(0)
    return -1 * ((1 - smoothing) * gold * torch.log(pred) + smoothing * (1 - gold) * torch.log(1 - pred))