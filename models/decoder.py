import torch.nn as nn

class Decoder(nn.Module):
    """
    解码器：由多个相同的层组成
    """

    def __init__(self, layer, N):
        """
        初始化解码器

        参数:
            layer (DecoderLayer): 解码器层
            N (int): 层数。
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

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
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)