import torch
import torch.nn as nn

from models import Decoder, Encoder, AttnDecoder
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
        use_attention = False,
    ):
        super(BasicLstmChatter, self).__init__()
        self._encoder = Encoder(
            num_embeddings=num_embeddings,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            layers=layers,
            do=do,
        )
        decoder_class = Decoder
        if use_attention:
            decoder_class = AttnDecoder

        self._decoder = decoder_class(
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

    def forward(self, tokens, dec_input=None, return_attn_scores=False, mask=None):
        context = self._encoder(tokens)
        if self.training or dec_input is not None:
            logits, _ = self._decoder(dec_input, context, mask=mask)
            return logits
        else:
            b, _ = tokens.shape
            res_tokens, attn_scores = [], []

            in_token = self.start_token.unsqueeze(0).unsqueeze(0).expand(b, 1)
            logits, context = self._decoder(in_token, context, mask=mask)
            _tokens = self._temp_sampler(logits)
            res_tokens.append(_tokens)
            if return_attn_scores:
                attn_scores.append(context['attention'])

            for _ in range(1, self._max_length):
                in_token = res_tokens[-1]
                logits, context = self._decoder(in_token, context, mask=mask)
                _tokens = self._temp_sampler(logits)
                res_tokens.append(_tokens)
                if return_attn_scores:
                    attn_scores.append(context['attention'])

            res_tokens = torch.concat(res_tokens, dim=1)
            if return_attn_scores:
                attn_scores = torch.concat(attn_scores, dim=1)
                return res_tokens, attn_scores
            else:
                return res_tokens
