import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, q_in_dims, k_in_dims, v_in_dims, qk_proj_dims, v_proj_dims=None):
        super(AttentionLayer, self).__init__()

        self._qk_proj_dims = qk_proj_dims
        self._v_proj_dims = v_proj_dims if v_proj_dims else qk_proj_dims

        self._query_proj = nn.Linear(q_in_dims, self._qk_proj_dims)
        self._key_proj = nn.Linear(k_in_dims, self._qk_proj_dims)
        self._value_proj = nn.Linear(v_in_dims, self._v_proj_dims)

    def forward(self, query, key, value, mask=None):
        assert len(query.shape) == 3, f'Query must have shape (b, qn, c)'
        assert len(key.shape) == 3, f'Key must have shape (b, kn, c)'
        assert len(value.shape) == 3, f'Value must have shape (b, kn, c)'
        assert key.shape[1] == value.shape[1], f'Key and Value must have the very same length'          

        proj_query = self._query_proj(query)
        proj_key = self._key_proj(key)
        proj_value = self._value_proj(value)

        algn_scores = torch.bmm(
            proj_query, proj_key.transpose(1, 2)
        ) / torch.sqrt(torch.tensor(self._qk_proj_dims))

        if mask is not None:
            assert len(mask.shape) == 2, f'Mask mush have shape (b, kn)'
            assert key.shape[:2] == mask.shape[:2]
            _, qn, _ = query.shape
            mask = mask.unsqueeze(1).repeat(1, qn, 1)
            algn_scores = algn_scores.masked_fill(mask == 0, -1e9)

        attention_weights = nn.functional.softmax(algn_scores, dim=-1)
        if mask is not None:
            attention_weights = attention_weights * mask.float()
        context_vector = torch.bmm(attention_weights, proj_value)

        return context_vector, attention_weights
