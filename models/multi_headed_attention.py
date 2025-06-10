import torch
import torch.nn as nn
import math

def attention(query, key, value, mask=None, dropout=None):
    """
    缩放点积注意力机制

    参数:
        query (Tensor): 查询向量
        key (Tensor): 键向量
        value (Tensor): 值向量
        mask (Tensor, optional): 掩码，用于遮蔽某些位置
        dropout (nn.Dropout, optional): Dropout 层

    返回:
        Tensor: 注意力加权的值
        Tensor: 注意力权重
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化多头注意力机制

        参数:
            h (int): 注意力头的数量
            d_model (int): 模型的维度
            dropout (float, optional): Dropout 概率
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数

        参数:
            query (Tensor): 查询向量
            key (Tensor): 键向量
            value (Tensor): 值向量
            mask (Tensor, optional): 掩码，用于遮蔽某些位置

        返回:
            Tensor: 注意力加权的值
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)