import torch
import torch.nn as nn

from models import Encoder, Decoder
from datasets import SpecToken


class BasicLstmChatter(nn.Module):
    def __init__(self, max_length=60, start_token=SpecToken.START, stop_token=SpecToken.STOP):
        super(BasicLstmChatter, self).__init__()
        self._encoder = Encoder()
        self._decoder = Decoder()
        self._loss = nn.CrossEntropyLoss()
        self._start_token = torch.tensor(int(start_token)).long()
        self._stop_token = torch.tensor(int(stop_token)).long()
        self._max_length = max_length

    def forward(self, tokens, dec_input=None, dec_target=None):
        context = self._encoder(tokens)
        if self.training:
            scores = self._decoder(dec_input, context)
            b, n, c = scores.shape
            scores_reshaped = torch.reshape(scores, [b * n, -1])
            targets_reshaped = torch.reshape(dec_target, [b * n]).long()           
            loss = self._loss(
                scores_reshaped,
                targets_reshaped,
            )
            return loss
        else:
            b, n = tokens.shape
            assert b == 1, f"Inference is now available only for batch size 1"
            context = self._encoder(tokens)

            def scalar_to_input(s):
                return torch.unsqueeze(torch.unsqueeze(s, 0), 0)
                       
            res = []
            in_token = scalar_to_input(self._start_token).to(tokens.device)
            scores, pred_token, context = self._decoder.infer_token(in_token, context)
            res.append(pred_token)
            
            for token_idx in range(1, self._max_length):
                in_token = res[-1]
                if in_token == self._stop_token.to(tokens.device):
                    break
                in_token = scalar_to_input(in_token)
                scores, pred_token, context = self._decoder.infer_token(in_token, context)
                res.append(pred_token)
                
            return res

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, max_length=60, start_token=SpecToken.START, stop_token=SpecToken.STOP, device='cpu'):
        model = BasicLstmChatter(max_length=10, start_token=SpecToken.START, stop_token=SpecToken.STOP)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
