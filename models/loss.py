import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self, masked_tokens):
        super(MaskedCrossEntropy, self).__init__()
        self._masked_tokens = masked_tokens
        self._ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, scores, targets):
        targets = targets.long()           
        loss = self._ce(
            scores,
            targets,
        )
        mask = torch.ones_like(targets).float()
        for token in self._masked_tokens:
            mask = mask * (targets != token).float()
        masked_loss = loss * mask
        return torch.mean(masked_loss)
