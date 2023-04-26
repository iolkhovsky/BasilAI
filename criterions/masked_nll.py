import torch
import torch.nn as nn

__all__ = ["MaskedNLL"]


class MaskedNLL(nn.Module):
    def __init__(self, masked_tokens):
        super(MaskedNLL, self).__init__()
        self.register_buffer("masked_tokens", torch.Tensor(masked_tokens))
        self._nll = nn.NLLLoss(reduction="none")

    def forward(self, logits, targets):
        targets = targets
        loss = self._nll(torch.log_softmax(logits, dim=-1), targets)
        loss[torch.isin(targets, self.masked_tokens)] = 0
        return torch.mean(loss)
