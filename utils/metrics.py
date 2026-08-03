import torch

def accuracy(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()
