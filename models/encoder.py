import torch.nn as nn

class Encoder(nn.Module):
    """
    编码器：由多个相同的层组成
    """

    def __init__(self, layer, N):
        """
        初始化编码器

        参数:
            layer (EncoderLayer): 编码器层
            N (int): 层数
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        前向传播函数

        参数:
            x (Tensor): 输入序列
            mask (Tensor): 掩码，用于遮蔽某些位置

        返回:
            Tensor: 编码后的序列
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)