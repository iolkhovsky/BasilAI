import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings=10000, hidden_size=256, embedding_dim=128):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=True
        )
        self._enc_lstm_0 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.,
        )
        self._enc_lstm_1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.,
        )

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        enc0_out, (enc0_h, enc0_c) = self._enc_lstm_0(embedded)
        _, (enc1_h, enc1_c) = self._enc_lstm_1(enc0_out)
        return {
            'enc0_h': enc0_h,
            'enc0_c': enc0_c,
            'enc1_h': enc1_h,
            'enc1_c': enc1_c,
        }
