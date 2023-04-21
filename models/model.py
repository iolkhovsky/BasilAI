import torch
import torch.nn as nn

from models import Decoder, Encoder
from tokenizers import BaseSpecToken


class BasicLstmChatter(nn.Module):
    def __init__(
        self,
        max_length=60,
        num_embeddings=10000,
        layers=2,
        do=0.5,
        hidden_size=256,
        embedding_dim=128,
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
        self.register_buffer("start_token", torch.tensor(start_token, dtype=torch.long))
        self.register_buffer("stop_token", torch.tensor(stop_token, dtype=torch.long))
        self._max_length = max_length

    def forward(self, tokens, dec_input=None):
        context = self._encoder(tokens)
        if self.training or dec_input is not None:
            logits = self._decoder(dec_input, context, apply_softmax=False)
            return logits
        else:
            b, n = tokens.shape
            context = self._encoder(tokens)
            res = []
            res_logits = []
            in_token = self.start_token.unsqueeze(0).unsqueeze(0).expand(b, 1)
            logits, pred_tokens, context = self._decoder.infer_tokens(in_token, context)
            res.append(pred_tokens)
            res_logits.append(logits)

            for token_idx in range(1, self._max_length):
                in_token = res[-1]
                in_token = in_token.unsqueeze(1)
                logits, pred_tokens, context = self._decoder.infer_tokens(
                    in_token, context
                )
                res.append(pred_tokens)
                res_logits.append(logits)

            res_logits = torch.concat(res_logits, dim=1)
            return res_logits
