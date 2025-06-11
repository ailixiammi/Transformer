import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置索引 [0, 1, 2, ..., max_len-1]，形状为 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # 计算分母项，形状为 [d_model // 2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 初始化位置编码矩阵，形状为 [max_len, 1, d_model]
        pe = torch.zeros(max_len, 1, d_model)
        # 计算偶数位置的正弦值
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 计算奇数位置的余弦值
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 将位置编码矩阵注册为缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入 x 的形状为 [batch_size, seq_len, d_model]
        # 转置为 [seq_len, batch_size, d_model]，方便与位置编码相加
        x = x.transpose(0, 1)
        # 将位置编码加到输入 x 上，只取前 seq_len 部分
        x = x + self.pe[:x.size(0)]
        # 应用 Dropout
        x = self.dropout(x)
        # 转置回原来的形状 [batch_size, seq_len, d_model]
        return x.transpose(0, 1)