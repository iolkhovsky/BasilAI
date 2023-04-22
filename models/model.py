import torch
import torch.nn as nn

from models import Decoder, Encoder
from tokenizers import BaseSpecToken
from models.temp_sampling import TemperatureSampler


class BasicLstmChatter(nn.Module):
    def __init__(
        self,
        max_length=60,
        num_embeddings=10000,
        layers=2,
        do=0.5,
        hidden_size=256,
        embedding_dim=128,
        temperature: float = 0.0,
        start_token=BaseSpecToken.START,
        stop_token=BaseSpecToken.STOP,
    ):
        super(BasicLstmChatter, self).__init__()
        self._encoder = Encoder(
            num_embeddings=num_embeddings,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            layers=layers,
            do=do,
        )
        self._decoder = Decoder(
            num_embeddings=num_embeddings,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            layers=layers,
            do=do,
        )
        self._temp_sampler = TemperatureSampler(temperature)
        self.register_buffer("start_token", torch.tensor(start_token, dtype=torch.long))
        self.register_buffer("stop_token", torch.tensor(stop_token, dtype=torch.long))
        self._max_length = max_length

    def forward(self, tokens, dec_input=None):
        context = self._encoder(tokens)
        if self.training or dec_input is not None:
            logits, _ = self._decoder(dec_input, context)
            return logits
        else:
            b, n = tokens.shape
            context = self._encoder(tokens)

            res_tokens = []

            in_token = self.start_token.unsqueeze(0).unsqueeze(0).expand(b, 1)
            logits, context = self._decoder(in_token, context)
            _tokens = self._temp_sampler(logits)
            res_tokens.append(_tokens)

            for token_idx in range(1, self._max_length):
                in_token = res_tokens[-1]
                logits, context = self._decoder(in_token, context)
                _tokens = self._temp_sampler(logits)
                res_tokens.append(_tokens)

            res_tokens = torch.concat(res_tokens, dim=1)
            return res_tokens
