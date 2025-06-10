import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码：为模型提供位置信息
    """

    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码

        参数:
            d_model (int): 模型的维度
            dropout (float): Dropout 概率
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout 层

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置的编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置的编码
        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)  # 注册为缓冲区

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (Tensor): 输入的嵌入向量。

        返回:
            Tensor: 添加位置编码后的向量。
        """
        x = x + self.pe[:, :x.size(1)]  # 将位置编码添加到嵌入向量中
        return self.dropout(x)  # 应用 Dropout