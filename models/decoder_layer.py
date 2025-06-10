import torch.nn as nn
from models.sublayer_connection import SublayerConnection

class DecoderLayer(nn.Module):
    """
    解码器层：包含多头注意力机制、编码器-解码器注意力机制和前馈网络
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        初始化解码器层

        参数:
            size (int): 模型的维度
            self_attn (MultiHeadedAttention): 多头注意力机制
            src_attn (MultiHeadedAttention): 编码器-解码器注意力机制
            feed_forward (PositionwiseFeedForward): 前馈网络
            dropout (float): Dropout 概率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播函数

        参数:
            x (Tensor): 输入序列
            memory (Tensor): 编码器的输出
            src_mask (Tensor): 编码器掩码
            tgt_mask (Tensor): 解码器掩码

        返回:
            Tensor: 解码后的序列
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 多头注意力机制
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # 编码器-解码器注意力机制
        return self.sublayer[2](x, self.feed_forward)  # 前馈网络