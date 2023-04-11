import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings=10000, hidden_size=256, embedding_dim=128, layers=2, do=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=True
        )
        self._enc_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=do,
            num_layers=layers,
        )

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        _, (enc_h, enc_c) = self._enc_lstm(embedded)
        return {
            'hidden': enc_h,
            'context': enc_c,
        }
