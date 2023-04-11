import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_embeddings=10000, hidden_size=256, embedding_dim=128, layers=2, do=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=True
        )
        self._dec_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=do,
            num_layers=layers,
        )
        self._dec_dense = nn.Linear(
            in_features=hidden_size,
            out_features=num_embeddings,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens, context):
        dec_in_h = context['hidden']
        dec_in_c = context['context']
        
        embedded = self.embedding(tokens)
        dec_out, _ = self._dec_lstm(embedded, (dec_in_h, dec_in_c))
        
        logits = self._dec_dense(dec_out)
        scores = self.softmax(logits)
        
        return scores

    def infer_token(self, token, context):
        dec_in_h = context['hidden']
        dec_in_c = context['context']
        
        assert tuple(token.shape) == (1, 1)
        
        embedding = self.embedding(token)
        dec_out, (dec_h, dec_c) = self._dec_lstm(embedding, (dec_in_h, dec_in_c))
        
        logits = self._dec_dense(dec_out)
        scores = self.softmax(logits)
        max_score, max_index = torch.max(scores, dim=-1)
        
        return scores, max_index.squeeze(), {
            'hidden': dec_h,
            'context': dec_c,
        }
