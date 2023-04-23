import torch
import torch.nn as nn

from models.attention import AttentionLayer


class Decoder(nn.Module):
    def __init__(
        self, num_embeddings=10000, hidden_size=256, embedding_dim=128, layers=2, do=0.5
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, max_norm=True
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

    def forward(self, tokens, context, *args, **kwargs):
        dec_in_h = context["hidden"]
        dec_in_c = context["context"]

        embeddings = self.embedding(tokens)
        dec_out, (dec_h, dec_c) = self._dec_lstm(embeddings, (dec_in_h, dec_in_c))

        logits = self._dec_dense(dec_out)

        return (
            logits,
            {
                "hidden": dec_h,
                "context": dec_c,
            },
        )


class AttnDecoder(nn.Module):
    def __init__(
        self, num_embeddings=10000, hidden_size=256, embedding_dim=128, layers=2, do=0.5
    ):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, max_norm=True
        )
        self._dec_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=do,
            num_layers=layers,
        )
        self._attention = AttentionLayer(
            q_in_dims=hidden_size,
            k_in_dims=hidden_size,
            v_in_dims=hidden_size,
            qk_proj_dims=hidden_size,
            v_proj_dims=hidden_size,
        )

        self._dec_dense = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_embeddings,
        )

    def forward(self, tokens, context, mask=None, *args, **kwargs):
        dec_in_h = context["hidden"]
        dec_in_c = context["context"]
        encoder_outputs = context["encoder_output"]

        embeddings = self.embedding(tokens)
        dec_out, (dec_h, dec_c) = self._dec_lstm(embeddings, (dec_in_h, dec_in_c))

        attention_context, attention_weights = self._attention(
            query=dec_out,
            key=encoder_outputs,
            value=encoder_outputs,
            mask=mask,
        )

        combined_features = torch.concat([attention_context, dec_out], dim=-1)
        logits = self._dec_dense(combined_features)

        return (
            logits,
            {
                "hidden": dec_h,
                "context": dec_c,
                "encoder_output": encoder_outputs,
                "attention": attention_weights,
            },
        )
