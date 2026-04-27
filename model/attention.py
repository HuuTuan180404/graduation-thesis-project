import math
import torch
import numpy as np
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F


class FlashAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), f"d_model {embed_dim} should be divisible by n_heads {n_heads}."

        self.embed_dim = embed_dim
        self.n_heads = n_heads

    def forward(self, q, k, v):
        # x: [B, L, D]
        B, L, D = q.shape
        head_dim = D // self.n_heads

        q = q.view(B, L, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, head_dim).transpose(1, 2)
        # [B, n_heads, L, head_dim]

        out = F.scaled_dot_product_attention(q, k, v)  # [B, n_heads, L, head_dim]

        out = out.permute(0, 2, 1, 3)

        return out.reshape(B, L, -1)  # (B, L, D)


def prob_attn_factory(embed_dim, n_heads, dropout=0.0):

    attention = ProbAttention(
        mask_flag=False,
        factor=5,
        attention_dropout=dropout,
        output_attention=True,
    )

    return AttentionLayer(attention=attention, d_model=embed_dim, n_heads=n_heads)


class WindowAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, window_size=12, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert (
            embed_dim % n_heads == 0
        ), f"d_model {embed_dim} should be divisible by n_heads {n_heads}."

        # self.attn = FlashAttention(embed_dim, n_heads)
        self.attn = prob_attn_factory(embed_dim, n_heads)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        w = self.window_size
        num_win = L // w

        x = x.reshape(B, num_win, w, D)  # [B, num_win, w, D]
        x = x.reshape(-1, w, D)  # [B*num_win, w, D]
        out, _ = self.attn(x, x, x)  # [B*num_win, w, D]
        out = out.reshape(B, num_win * w, D)  # [B, num_win*w, D]

        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model=108, nhead=9, max_seq_len=204, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = (self.d_k) ** -0.5

        # Projections for left hand (query) and right hand (key, value)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE embeddings - critical for temporal understanding
        self.register_buffer("cos", self._compute_cos_sin(max_seq_len, self.d_k)[0])
        self.register_buffer("sin", self._compute_cos_sin(max_seq_len, self.d_k)[1])

    def _compute_cos_sin(self, seq_len, dim):
        # Lower frequency for sign language temporal patterns
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        seq = torch.arange(seq_len).float()
        freqs = torch.outer(seq, theta)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rope(self, x, cos, sin):
        seq_len = x.shape[2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            left_hand: (batch, 204, d_model) - Left hand features
            right_hand: (batch, 204, d_model) - Right hand features
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key_value)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(key_value)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )

        # Apply RoPE - crucial for temporal understanding
        Q = self._apply_rope(Q, self.cos, self.sin)
        K = self._apply_rope(K, self.cos, self.sin)

        # Cross-attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.w_o(attended), attn_weights


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # [24, 204, 2, 12]
        _, S, _, D = values.shape  # [24, 204, 2, 12]
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        factor=5,
        scale=None,
        attention_dropout=0.0,
        output_attention=True,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2
        )

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = int(self.factor * math.ceil(math.log(L_K)))
        u = int(self.factor * math.ceil(math.log(L_Q)))

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)

        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.num_heads = n_heads
        self.mix = mix
        self.batch_first = None
        self._qkv_same_embed_dim = True
        self.attention_scores = None

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        is_causal=None,
    ):
        # queries = queries.permute(1, 0, 2).type(dtype=torch.float32)
        # keys = keys.permute(1, 0, 2).type(dtype=torch.float32)
        # values = values.permute(1, 0, 2).type(dtype=torch.float32)
        # print('queries', queries.shape) = [24, 204, 42]

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, self.attention_scores = self.inner_attention(
            queries, keys, values, attn_mask
        )
        # print(self.attention_scores)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        out = self.out_projection(out)
        # out = out.permute(1, 0, 2).type(dtype=torch.float32)

        # print(f"out from prob_spare attention: {out.shape}")
        return out, self.attention_scores


#    The reference for the code is the following
#    Title: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
#    Author: Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
#    Availability: https://github.com/zhouhaoyi/Informer2020/tree/main?tab=readme-ov-file
