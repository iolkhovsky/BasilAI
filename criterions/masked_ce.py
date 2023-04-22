import torch
import torch.nn as nn

__all__ = ["MaskedCrossEntropy"]


class MaskedCrossEntropy(nn.Module):
    def __init__(self, masked_tokens):
        super(MaskedCrossEntropy, self).__init__()
        self._masked_tokens = masked_tokens
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        targets = targets.long()
        loss = self._ce(logits, targets)
        mask = torch.ones_like(targets).float()
        for token in self._masked_tokens:
            mask = mask * (targets != token).float()
        masked_loss = loss * mask
        return torch.mean(masked_loss)
