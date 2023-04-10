import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_embeddings=10000, hidden_size=256, embedding_dim=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=True
        )
        self._dec_lstm_0 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.5,
        )
        self._dec_lstm_1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.5,
        )
        self._dec_dense = nn.Linear(
            in_features=hidden_size,
            out_features=num_embeddings,
        )
        self._dense_do = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens, context):
        dec0_in_h = context['enc0_h']
        dec0_in_c = context['enc0_c']
        dec1_in_h = context['enc1_h']
        dec1_in_c = context['enc1_c']
        
        embedded = self.embedding(tokens)
        dec0_out, _ = self._dec_lstm_0(embedded, (dec0_in_h, dec0_in_c))
        dec1_out, _ = self._dec_lstm_1(dec0_out, (dec1_in_h, dec1_in_c))
        
        logits = self._dec_dense(dec1_out)
        logits = self._dense_do(logits)
        scores = self.softmax(logits)
        
        return scores

    def infer_token(self, token, context):
        dec0_in_h = context['enc0_h']
        dec0_in_c = context['enc0_c']
        dec1_in_h = context['enc1_h']
        dec1_in_c = context['enc1_c']
        
        assert tuple(token.shape) == (1, 1)
        
        embedding = self.embedding(token)
        dec0_out, (dec0_h, dec0_c) = self._dec_lstm_0(embedding, (dec0_in_h, dec0_in_c))
        dec1_out, (dec1_h, dec1_c) = self._dec_lstm_1(dec0_out, (dec1_in_h, dec1_in_c))
        
        logits = self._dec_dense(dec1_out)
        scores = self.softmax(logits)
        max_score, max_index = torch.max(scores, dim=-1)
        
        return scores, max_index.squeeze(), {
            'enc0_h': dec0_h,
            'enc0_c': dec0_c,
            'enc1_h': dec1_h,
            'enc1_c': dec1_c,
        }
