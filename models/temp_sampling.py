import torch
import torch.nn as nn


class TemperatureSampler(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self._temperature = temperature
        
    def forward(self, logits):
        if self._temperature <= 0.0:
            tokens = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        else:
            b, n, c = logits.shape
            reshaped_logits = torch.reshape(logits, [b*n, c])
            scaled_reshaped_logits = reshaped_logits / self._temperature
            reshaped_probabilities = nn.functional.softmax(scaled_reshaped_logits, dim=-1)
            reshaped_tokens = torch.multinomial(reshaped_probabilities, num_samples=1)
            tokens = torch.reshape(reshaped_tokens, [b, n])
        return tokens
