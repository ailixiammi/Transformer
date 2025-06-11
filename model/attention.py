import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分割为多头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_o(attn_output)
        return output, attn_weights