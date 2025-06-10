import torch.nn as nn
from models.sublayer_connection import SublayerConnection

class EncoderLayer(nn.Module):
    """
    编码器层：包含多头注意力机制和前馈网络
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        初始化编码器层

        参数:
            size (int): 模型的维度
            self_attn (MultiHeadedAttention): 多头注意力机制
            feed_forward (PositionwiseFeedForward): 前馈网络
            dropout (float): Dropout 概率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        """
        前向传播函数

        参数:
            x (Tensor): 输入序列
            mask (Tensor): 掩码，用于遮蔽某些位置

        返回:
            Tensor: 编码后的序列
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 多头注意力机制
        return self.sublayer[1](x, self.feed_forward)  # 前馈网络