import torch
import torch.nn as nn

__all__ = ["MaskedCrossEntropy"]


class MaskedCrossEntropy(nn.Module):
    def __init__(self, masked_tokens):
        super(MaskedCrossEntropy, self).__init__()
        self.register_buffer("masked_tokens", torch.Tensor(masked_tokens))
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        targets = targets
        loss = self._ce(logits, targets)
        loss[torch.isin(targets, self.masked_tokens)] = 0
        return torch.mean(loss)
