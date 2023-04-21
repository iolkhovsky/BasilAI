import torch
import torch.nn as nn


class TemperatureSampler(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self._temperature = temperature
        
    def forward(self, logits):
        assert len(logits.shape) == 2, f'logits must have shape (b, c)'
        scaled_logits = logits / self._temperature
        probabilities = nn.functional.softmax(scaled_logits, dim=-1)
        sample = torch.multinomial(probabilities, num_samples=1)
        return sample.squeeze()
